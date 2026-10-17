//! BF16/F32 Gemma4a weights, with F32 activations and the reference's Conformer order.

pub mod attention;
#[cfg(test)]
mod tests;
mod weights;

use self::attention::kernels as attention_kernels;
use self::attention::{HEAD, HEADS, RPE, WIDTH};
use self::weights::{FF, Ffn, Linear, OUT, TEXT, Weights};
use crate::ops::{Ops, config};
use bloomery_gpu::{Gpu, GpuError, Window, WindowMut, elem::ElemKernels, gemm::Gemm32Kernels};
use cuda_core::{CudaStream, DeviceBuffer};
use gguf::Gguf;
use std::sync::Arc;
use vision::arch::gemma4a::{MEL_BINS, Mel};

const MAX_FRAMES: usize = 2999;
const MAX_TOKENS: usize = MAX_FRAMES.div_ceil(4);
const EPS: f32 = 1e-6;

/// One chunk's text-width soft rows, in token-major order.
pub struct Encoded {
    pub tokens: usize,
    pub values: Vec<f32>,
}

/// Loaded audio tower and bounded scratch, independent of the text encoder.
pub struct Encoder {
    stream: Arc<CudaStream>,
    weights: Weights,
    pass: Pass,
    scratch: Scratch,
}

struct Pass {
    ops: Ops,
    elem: ElemKernels,
    gemm: Gemm32Kernels,
    attention: attention::kernels::LoadedModule,
}

struct Scratch {
    mel: DeviceBuffer<f32>,
    x: DeviceBuffer<f32>,
    a: DeviceBuffer<f32>,
    b: DeviceBuffer<f32>,
    c: DeviceBuffer<f32>,
    q: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    position: DeviceBuffer<f32>,
    relative: DeviceBuffer<f32>,
}

impl Encoder {
    /// Open the audio projector using the runtime's stream and device fault owner.
    pub fn load(gpu: &Gpu, file: &Gguf) -> Result<Self, GpuError> {
        let stream = gpu.stream_handle();
        let weights = Weights::load(file, &stream)?;
        let pass = Pass::load(gpu)?;
        let (position, _) = attention::inputs(MAX_TOKENS);
        let scratch = Scratch {
            mel: DeviceBuffer::zeroed(&stream, MAX_FRAMES * MEL_BINS)?,
            x: DeviceBuffer::zeroed(&stream, MAX_TOKENS * WIDTH)?,
            a: DeviceBuffer::zeroed(&stream, MAX_FRAMES.div_ceil(2) * 64 * 128)?,
            b: DeviceBuffer::zeroed(&stream, MAX_FRAMES.div_ceil(2) * 64 * 128)?,
            c: DeviceBuffer::zeroed(&stream, MAX_TOKENS * FF)?,
            q: DeviceBuffer::zeroed(&stream, MAX_TOKENS * WIDTH)?,
            k: DeviceBuffer::zeroed(&stream, MAX_TOKENS * WIDTH)?,
            v: DeviceBuffer::zeroed(&stream, MAX_TOKENS * WIDTH)?,
            position: DeviceBuffer::from_host(&stream, &position)?,
            relative: DeviceBuffer::zeroed(&stream, WIDTH * RPE)?,
        };
        Ok(Self {
            stream,
            weights,
            pass,
            scratch,
        })
    }

    /// Device-resident expanded weights, including absent biases represented by zeros.
    #[must_use]
    pub fn weight_bytes(&self) -> u64 {
        self.weights.bytes
    }

    /// Fixed tower-owned scratch, including positions. The caller owns its runtime fault word.
    #[must_use]
    pub fn scratch_bytes(&self) -> u64 {
        let s = &self.scratch;
        [
            s.mel.len(),
            s.x.len(),
            s.a.len(),
            s.b.len(),
            s.c.len(),
            s.q.len(),
            s.k.len(),
            s.v.len(),
            s.position.len(),
            s.relative.len(),
        ]
        .into_iter()
        .map(|n| 4 * n as u64)
        .sum::<u64>()
    }

    /// Bytes to add to the same named reserve as the vision tower, not its expert tensors.
    #[must_use]
    pub fn reserve_bytes(&self) -> u64 {
        self.weight_bytes() + self.scratch_bytes()
    }

    /// Encode one independently padded chunk. No state carries between chunks.
    pub fn encode(&mut self, mel: &Mel) -> Result<Encoded, GpuError> {
        self.run(mel, None)
    }

    /// Observe F32 stage outputs without changing the tower's arithmetic.
    pub fn encode_tapped(
        &mut self,
        mel: &Mel,
        tap: &mut impl FnMut(&str, usize, &[f32]),
    ) -> Result<Encoded, GpuError> {
        self.run(mel, Some(tap))
    }

    fn run(
        &mut self,
        mel: &Mel,
        mut tap: Option<&mut dyn FnMut(&str, usize, &[f32])>,
    ) -> Result<Encoded, GpuError> {
        if mel.frames == 0
            || mel.frames > MAX_FRAMES
            || mel.data.len() != mel.frames * MEL_BINS
            || mel.data.iter().any(|v| !v.is_finite())
        {
            return Err(GpuError::Shape {
                what: "gemma4a mel",
                detail: format!(
                    "expected 1..={MAX_FRAMES} finite frames, got {} and {} values",
                    mel.frames,
                    mel.data.len()
                ),
            });
        }
        let n = mel.tokens();
        let stream = &self.stream;
        let w = &self.weights;
        let p = &self.pass;
        let s = &mut self.scratch;
        let mut input = vec![0.0; mel.data.len()];
        for t in 0..mel.frames {
            for f in 0..MEL_BINS {
                input[t * MEL_BINS + f] = mel.data[f * mel.frames + t];
            }
        }
        WindowMut::<f32>::of_mut(&mut s.mel, 0, input.len())?.copy_from_host(stream, &input)?;
        let (t1, f1) = (mel.frames.div_ceil(2), 64usize);
        let prep = p.ops.module.prepare_conv2d(config(t1 * f1 * 128))?;
        p.ops.module.conv2d(
            stream,
            &prep,
            &s.mel,
            &w.conv[0],
            &w.bias[0],
            mel.frames as u32,
            128,
            1,
            128,
            t1 as u32,
            64,
            &mut s.a,
        )?;
        let prep = p.ops.module.prepare_channel_norm(config(t1 * f1))?;
        p.ops.module.channel_norm(
            stream,
            &prep,
            &s.a,
            &w.norm[0],
            (t1 * f1) as u32,
            128,
            EPS,
            p.ops.sink,
            &mut s.b,
        )?;
        let prep = p.ops.module.prepare_conv2d(config(n * 32 * 32))?;
        p.ops.module.conv2d(
            stream, &prep, &s.b, &w.conv[1], &w.bias[1], t1 as u32, 64, 128, 32, n as u32, 32,
            &mut s.a,
        )?;
        let prep = p.ops.module.prepare_channel_norm(config(n * 32))?;
        p.ops.module.channel_norm(
            stream,
            &prep,
            &s.a,
            &w.norm[1],
            (n * 32) as u32,
            32,
            EPS,
            p.ops.sink,
            &mut s.b,
        )?;
        p.mm(stream, &w.input, &s.b, n, &mut s.a, &mut s.x)?;
        p.add(stream, &w.input_bias, n * WIDTH, WIDTH, 1.0, &mut s.x)?;
        take(stream, "sscp", n, WIDTH, &s.x, &mut tap)?;
        for (index, b) in w.blocks.iter().enumerate() {
            p.ffn(stream, &b.ffn[0], n, s)?;
            p.elem
                .enqueue_rms_norm(stream, &s.x, &b.pre, EPS, WIDTH, n, &mut s.b)?;
            p.mm(stream, &b.q, &s.b, n, &mut s.a, &mut s.q)?;
            p.mm(stream, &b.k, &s.b, n, &mut s.a, &mut s.k)?;
            p.mm(stream, &b.v, &s.b, n, &mut s.a, &mut s.v)?;
            p.mm(stream, &b.rel, &s.position, RPE, &mut s.a, &mut s.relative)?;
            let prep = p.attention.prepare_local_attention(config(n * HEADS))?;
            p.attention.local_attention(
                stream,
                &prep,
                &s.q,
                &s.k,
                &s.v,
                &s.relative,
                &b.qs,
                &b.ks,
                n as u32,
                (1.0 / (HEAD as f32).sqrt()) / 2.0_f32.ln(),
                (1.0 + 1.0_f32.exp()).ln() / 2.0_f32.ln(),
                p.ops.sink,
                &mut s.b,
            )?;
            p.mm(stream, &b.o, &s.b, n, &mut s.a, &mut s.c)?;
            p.elem
                .enqueue_rms_norm(stream, &s.c, &b.post, EPS, WIDTH, n, &mut s.b)?;
            p.add(stream, &s.b, n * WIDTH, n * WIDTH, 1.0, &mut s.x)?;
            p.elem
                .enqueue_rms_norm(stream, &s.x, &b.conv_pre, EPS, WIDTH, n, &mut s.b)?;
            p.mm(stream, &b.pw1, &s.b, n, &mut s.a, &mut s.c)?;
            let prep = p.ops.module.prepare_glu(config(n * WIDTH))?;
            p.ops.module.glu(
                stream,
                &prep,
                &s.c,
                n as u32,
                WIDTH as u32,
                p.ops.sink,
                &mut s.b,
            )?;
            let prep = p.ops.module.prepare_depthwise(config(n * WIDTH))?;
            p.ops.module.depthwise(
                stream,
                &prep,
                &s.b,
                &b.dw,
                &b.dw_bias,
                n as u32,
                WIDTH as u32,
                5,
                &mut s.c,
            )?;
            p.elem
                .enqueue_rms_norm(stream, &s.c, &b.conv_post, EPS, WIDTH, n, &mut s.b)?;
            p.unary(stream, n * WIDTH, [-f32::MAX, f32::MAX], 1, &mut s.b)?;
            p.mm(stream, &b.pw2, &s.b, n, &mut s.a, &mut s.c)?;
            p.add(stream, &s.c, n * WIDTH, n * WIDTH, 1.0, &mut s.x)?;
            p.ffn(stream, &b.ffn[1], n, s)?;
            p.elem
                .enqueue_rms_norm(stream, &s.x, &b.last, EPS, WIDTH, n, &mut s.b)?;
            std::mem::swap(&mut s.x, &mut s.b);
            take(stream, &format!("blk{index}"), n, WIDTH, &s.x, &mut tap)?;
        }
        p.mm(stream, &w.out, &s.x, n, &mut s.a, &mut s.c)?;
        p.add(stream, &w.out_bias, n * OUT, OUT, 1.0, &mut s.c)?;
        take(stream, "pre_encode", n, OUT, &s.c, &mut tap)?;
        p.elem
            .enqueue_rms_norm(stream, &s.c, &w.soft_norm, EPS, OUT, n, &mut s.b)?;
        p.mm(stream, &w.soft, &s.b, n, &mut s.a, &mut s.c)?;
        stream.synchronize()?;
        if let Some(fault) = bloomery_gpu::fault::read(&p.ops.fault)? {
            return Err(GpuError::fault("gemma4a output", fault));
        }
        let values = Window::<f32>::of(&s.c, 0, n * TEXT)?.to_host_vec(stream)?;
        if values.iter().any(|v| !v.is_finite()) {
            return Err(GpuError::Shape {
                what: "gemma4a output",
                detail: "non-finite soft row".into(),
            });
        }
        Ok(Encoded { tokens: n, values })
    }
}

fn take(
    stream: &CudaStream,
    name: &str,
    n: usize,
    width: usize,
    x: &DeviceBuffer<f32>,
    tap: &mut Option<&mut dyn FnMut(&str, usize, &[f32])>,
) -> Result<(), GpuError> {
    let Some(tap) = tap else {
        return Ok(());
    };
    stream.synchronize()?;
    let data = Window::<f32>::of(x, 0, n * width)?.to_host_vec(stream)?;
    tap(name, width, &data);
    Ok(())
}

impl Pass {
    fn load(gpu: &Gpu) -> Result<Self, GpuError> {
        let ctx = gpu.context();
        // SAFETY: this crate owns the embedded attention bundle and checks all spans.
        let attention = unsafe { bloomery_gpu::shared_module!(attention_kernels, ctx)? };
        Ok(Self {
            ops: Ops::load(gpu)?,
            elem: ElemKernels::load(ctx, gpu.fault_word())?,
            gemm: Gemm32Kernels::load(ctx)?,
            attention,
        })
    }

    fn mm(
        &self,
        stream: &CudaStream,
        w: &Linear,
        x: &DeviceBuffer<f32>,
        n: usize,
        tmp: &mut DeviceBuffer<f32>,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let len = n * w.w.cols();
        let prep = self.ops.module.prepare_pointwise(config(len))?;
        self.ops.module.pointwise(
            stream,
            &prep,
            x,
            len as u32,
            w.clamp[0],
            w.clamp[1],
            u32::from(w.bf16),
            tmp,
        )?;
        self.gemm.enqueue_f32_tile(stream, &w.w, tmp, n, y)?;
        self.unary(stream, n * w.w.rows(), [w.clamp[2], w.clamp[3]], 0, y)
    }

    fn unary(
        &self,
        stream: &CudaStream,
        n: usize,
        clip: [f32; 2],
        op: u32,
        x: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let prep = self.ops.module.prepare_unary(config(n))?;
        self.ops
            .module
            .unary(stream, &prep, n as u32, clip[0], clip[1], op, x)?;
        Ok(())
    }

    fn add(
        &self,
        stream: &CudaStream,
        other: &DeviceBuffer<f32>,
        n: usize,
        stride: usize,
        scale: f32,
        x: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let prep = self.ops.module.prepare_accumulate(config(n))?;
        self.ops
            .module
            .accumulate(stream, &prep, other, n as u32, stride as u32, scale, x)?;
        Ok(())
    }

    fn ffn(&self, stream: &CudaStream, f: &Ffn, n: usize, s: &mut Scratch) -> Result<(), GpuError> {
        self.elem
            .enqueue_rms_norm(stream, &s.x, &f.norm, EPS, WIDTH, n, &mut s.b)?;
        self.mm(stream, &f.up, &s.b, n, &mut s.a, &mut s.c)?;
        self.unary(stream, n * FF, [-f32::MAX, f32::MAX], 1, &mut s.c)?;
        self.mm(stream, &f.down, &s.c, n, &mut s.a, &mut s.b)?;
        self.elem
            .enqueue_rms_norm(stream, &s.b, &f.post, EPS, WIDTH, n, &mut s.c)?;
        self.add(stream, &s.c, n * WIDTH, n * WIDTH, 0.5, &mut s.x)
    }
}
