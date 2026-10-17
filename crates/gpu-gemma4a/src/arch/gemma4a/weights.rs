use super::attention::{HEADS, WIDTH};
use bloomery_gpu::{DeviceTensor, GpuError};
use cuda_core::{CudaStream, DeviceBuffer};
use gguf::{
    Gguf,
    quant::{GgmlType, dequant_row},
};

pub(super) const LAYERS: usize = 12;
pub(super) const FF: usize = 4096;
pub(super) const OUT: usize = 1536;
pub(super) const TEXT: usize = 512;

pub(super) struct Linear {
    pub w: DeviceTensor<f32>,
    pub clamp: [f32; 4],
    pub bf16: bool,
}

pub(super) struct Ffn {
    pub norm: DeviceBuffer<f32>,
    pub up: Linear,
    pub down: Linear,
    pub post: DeviceBuffer<f32>,
}

pub(super) struct Block {
    pub ffn: [Ffn; 2],
    pub pre: DeviceBuffer<f32>,
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    pub o: Linear,
    pub rel: Linear,
    pub qs: DeviceBuffer<f32>,
    pub ks: DeviceBuffer<f32>,
    pub post: DeviceBuffer<f32>,
    pub conv_pre: DeviceBuffer<f32>,
    pub pw1: Linear,
    pub dw: DeviceBuffer<f32>,
    pub dw_bias: DeviceBuffer<f32>,
    pub conv_post: DeviceBuffer<f32>,
    pub pw2: Linear,
    pub last: DeviceBuffer<f32>,
}

pub(super) struct Weights {
    pub conv: [DeviceBuffer<f32>; 2],
    pub bias: [DeviceBuffer<f32>; 2],
    pub norm: [DeviceBuffer<f32>; 2],
    pub input: Linear,
    pub input_bias: DeviceBuffer<f32>,
    pub blocks: Vec<Block>,
    pub out: Linear,
    pub out_bias: DeviceBuffer<f32>,
    pub soft_norm: DeviceBuffer<f32>,
    pub soft: Linear,
    pub bytes: u64,
}

pub(super) fn read(file: &Gguf, name: &str, dims: &[u64]) -> Result<Vec<f32>, GpuError> {
    let refuse = |detail| GpuError::Shape {
        what: "gemma4a weights",
        detail,
    };
    let t = file
        .find(name)
        .ok_or_else(|| refuse(format!("missing {name}")))?;
    if t.dims != dims || !matches!(t.ty, GgmlType::F32 | GgmlType::BF16) {
        return Err(refuse(format!(
            "{name}: expected F32/BF16 {dims:?}, got {:?} {:?}",
            t.ty, t.dims
        )));
    }
    let n: usize = dims.iter().map(|&d| d as usize).product();
    let mut values = vec![0.0; n];
    let raw = file.data(t).map_err(|e| refuse(format!("{name}: {e}")))?;
    dequant_row(t.ty, raw, &mut values).map_err(|e| refuse(format!("{name}: {e}")))?;
    if values.iter().any(|v| !v.is_finite()) {
        return Err(refuse(format!("{name}: non-finite weight")));
    }
    Ok(values)
}

struct Loader<'a> {
    file: &'a Gguf,
    stream: &'a CudaStream,
    bytes: u64,
}

impl Loader<'_> {
    fn vector(&mut self, name: &str, dims: &[u64]) -> Result<DeviceBuffer<f32>, GpuError> {
        let data = read(self.file, name, dims)?;
        self.bytes += 4 * data.len() as u64;
        Ok(DeviceBuffer::from_host(self.stream, &data)?)
    }

    fn optional(
        &mut self,
        name: &str,
        n: usize,
        value: f32,
    ) -> Result<DeviceBuffer<f32>, GpuError> {
        if self.file.find(name).is_some() {
            return self.vector(name, &[n as u64]);
        }
        self.bytes += 4 * n as u64;
        Ok(DeviceBuffer::from_host(self.stream, &vec![value; n])?)
    }

    fn linear(&mut self, name: &str, input: usize, output: usize) -> Result<Linear, GpuError> {
        let stem = name.strip_suffix(".weight").expect("weight name");
        let mut clamp = [-f32::MAX, f32::MAX, -f32::MAX, f32::MAX];
        for (i, suffix) in ["input_min", "input_max", "output_min", "output_max"]
            .into_iter()
            .enumerate()
        {
            let key = format!("{stem}.{suffix}");
            if !stem.ends_with(".attn_k_rel") && self.file.find(&key).is_some() {
                clamp[i] = read(self.file, &key, &[1])?[0];
            }
        }
        if clamp[0] > clamp[1] || clamp[2] > clamp[3] {
            return Err(GpuError::Shape {
                what: "gemma4a clamp",
                detail: format!("{name}: reversed clamp {clamp:?}"),
            });
        }
        let data = read(self.file, name, &[input as u64, output as u64])?;
        self.bytes += 4 * data.len() as u64;
        Ok(Linear {
            w: DeviceTensor::upload(self.stream, &data, output, input)?,
            clamp,
            bf16: self.file.find(name).expect("validated weight").ty == GgmlType::BF16,
        })
    }

    fn ffn(&mut self, b: &str, suffix: &str) -> Result<Ffn, GpuError> {
        Ok(Ffn {
            norm: self.vector(&format!("{b}.ffn_norm{suffix}.weight"), &[WIDTH as u64])?,
            up: self.linear(&format!("{b}.ffn_up{suffix}.weight"), WIDTH, FF)?,
            down: self.linear(&format!("{b}.ffn_down{suffix}.weight"), FF, WIDTH)?,
            post: self.vector(
                &format!("{b}.ffn_post_norm{suffix}.weight"),
                &[WIDTH as u64],
            )?,
        })
    }
}

impl Weights {
    pub fn load(file: &Gguf, stream: &CudaStream) -> Result<Self, GpuError> {
        if file
            .value("clip.audio.projector_type")
            .and_then(|v| v.as_str())
            != Some("gemma4a")
        {
            return Err(GpuError::Shape {
                what: "gemma4a weights",
                detail: "expected clip.audio.projector_type=gemma4a".into(),
            });
        }
        for (key, expected) in [
            ("block_count", LAYERS as u64),
            ("embedding_length", WIDTH as u64),
            ("attention.head_count", HEADS as u64),
            ("feed_forward_length", FF as u64),
            ("num_mel_bins", vision::arch::gemma4a::MEL_BINS as u64),
        ] {
            let name = format!("clip.audio.{key}");
            if file.value(&name).and_then(|v| v.as_u64()) != Some(expected) {
                return Err(GpuError::Shape {
                    what: "gemma4a weights",
                    detail: format!("{name}: expected {expected}"),
                });
            }
        }
        for name in ["a.conv1d.0.weight", "a.conv1d.1.weight"] {
            if file.find(name).is_none_or(|t| t.ty != GgmlType::F32) {
                return Err(GpuError::Shape {
                    what: "gemma4a convolution",
                    detail: format!("{name}: expected F32 weights with F16 im2col arithmetic"),
                });
            }
        }
        let mut l = Loader {
            file,
            stream,
            bytes: 0,
        };
        let conv = [
            l.vector("a.conv1d.0.weight", &[3, 3, 1, 128])?,
            l.vector("a.conv1d.1.weight", &[3, 3, 128, 32])?,
        ];
        let bias = [
            l.optional("a.conv1d.0.bias", 128, 0.0)?,
            l.optional("a.conv1d.1.bias", 32, 0.0)?,
        ];
        let norm = [
            l.vector("a.conv1d.0.norm.weight", &[128])?,
            l.vector("a.conv1d.1.norm.weight", &[32])?,
        ];
        let input = l.linear("a.input_projection.weight", WIDTH, WIDTH)?;
        let input_bias = l.optional("a.input_projection.bias", WIDTH, 0.0)?;
        let mut blocks = Vec::with_capacity(LAYERS);
        for i in 0..LAYERS {
            let b = format!("a.blk.{i}");
            blocks.push(Block {
                ffn: [l.ffn(&b, "")?, l.ffn(&b, "_1")?],
                pre: l.vector(&format!("{b}.attn_pre_norm.weight"), &[WIDTH as u64])?,
                q: l.linear(&format!("{b}.attn_q.weight"), WIDTH, WIDTH)?,
                k: l.linear(&format!("{b}.attn_k.weight"), WIDTH, WIDTH)?,
                v: l.linear(&format!("{b}.attn_v.weight"), WIDTH, WIDTH)?,
                o: l.linear(&format!("{b}.attn_out.weight"), WIDTH, WIDTH)?,
                rel: l.linear(&format!("{b}.attn_k_rel.weight"), WIDTH, WIDTH)?,
                qs: l.vector(&format!("{b}.per_dim_scale.weight"), &[128])?,
                ks: l.optional(&format!("{b}.per_dim_k_scale.weight"), 128, 1.0)?,
                post: l.vector(&format!("{b}.attn_post_norm.weight"), &[WIDTH as u64])?,
                // The GGUF converter swaps these two norm names.
                conv_pre: l.vector(&format!("{b}.conv_norm.weight"), &[WIDTH as u64])?,
                pw1: l.linear(&format!("{b}.conv_pw1.weight"), WIDTH, 2 * WIDTH)?,
                dw: l.vector(&format!("{b}.conv_dw.weight"), &[5, WIDTH as u64])?,
                dw_bias: l.optional(&format!("{b}.conv_dw.bias"), WIDTH, 0.0)?,
                conv_post: l.vector(&format!("{b}.norm_conv.weight"), &[WIDTH as u64])?,
                pw2: l.linear(&format!("{b}.conv_pw2.weight"), WIDTH, WIDTH)?,
                last: l.vector(&format!("{b}.ln2.weight"), &[WIDTH as u64])?,
            });
        }
        let out = l.linear("a.pre_encode.out.weight", WIDTH, OUT)?;
        let out_bias = l.optional("a.pre_encode.out.bias", OUT, 0.0)?;
        let soft_norm = l.optional("mm.a.soft_emb_norm.weight", OUT, 1.0)?;
        let soft = l.linear("mm.a.input_projection.weight", OUT, TEXT)?;
        Ok(Self {
            conv,
            bias,
            norm,
            input,
            input_bias,
            blocks,
            out,
            out_bias,
            soft_norm,
            soft,
            bytes: l.bytes,
        })
    }
}
