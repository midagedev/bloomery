//! Gemma4 vision: F32 patch convolution and learned positions, BF16 ViT
//! linears, sandwich RMS norms, gainless V norm, scale-one non-causal
//! attention, the header's GELU GLU, three-by-three pooling and the input projection.
//! Audio tensors of the shared projector are never uploaded.
//!
//! Q/K rows and gains use [0..16, 32..48, 16..32, 48..64] in each head.
//! This expresses the two NEOX half-head turns through the shared whole-head
//! RoPE entry; V and the attention output keep their original channel order.
//! Residuals stay F32; the shared GEMMs and attention consume BF16 rows.

use crate::{
    attn::{AttnArgs, AttnKernels, QkvLayout},
    chain::{Encoded, NoTaps, TapSink, read_bf16, read_f32, tap, tap_f32},
    gemm_bf16::{Epilogue, FloatGemmArgs, GemmArgs, GemmKernels},
    mean_groups::MeanGroupsKernels,
    norm::{FloatNormArgs, HeadNormArgs, NormKernels},
    pointwise::{self, Op},
    rope2d::{RopeArgs, RopeKernels, RopeTable},
};
use bloomery_gpu::gemm::Gemm32Kernels;
use bloomery_gpu::{DeviceTensor, GpuError, WindowMut};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use gguf::Gguf;
use std::collections::BTreeMap;
use std::sync::Arc;
use vision::arch::gemma4v::{
    Hparams, MERGE,
    card::{self, ScratchLens},
};
use vision::{Patches, arch::TokenLimits};

const WHAT: &str = "gemma4v tower";

struct Linear {
    w: DeviceBuffer<u16>,
    input: (f32, f32),
    output: (f32, f32),
}
struct Weights {
    patch: DeviceTensor<f32>,
    pos: DeviceBuffer<f32>,
    linear: BTreeMap<String, Linear>,
    norm: BTreeMap<String, DeviceBuffer<f32>>,
    std_bias: DeviceBuffer<f32>,
    std_scale: DeviceBuffer<f32>,
    ones: DeviceBuffer<f32>,
    head_ones: DeviceBuffer<f32>,
}

/// A resident Gemma4 vision tower and its bounded buffers, one image at a time.
pub struct Encoder {
    hp: Hparams,
    limits: TokenLimits,
    standardize: bool,
    w: Weights,
    gemm: GemmKernels,
    patch_gemm: Gemm32Kernels,
    norm: NormKernels,
    ops: pointwise::Kernels,
    rope: RopeKernels,
    attn: AttnKernels,
    mean: MeanGroupsKernels,
    s: Scratch,
    weight_bytes: u64,
    scratch_bytes: u64,
    launches: usize,
    stream: Arc<CudaStream>,
}

struct Scratch {
    raw: DeviceBuffer<f32>,
    patch: DeviceBuffer<f32>,
    x: DeviceBuffer<f32>,
    residual: DeviceBuffer<f32>,
    h: DeviceBuffer<u16>,
    h32: DeviceBuffer<f32>,
    work: DeviceBuffer<u16>,
    att: DeviceBuffer<u16>,
    gathered: DeviceBuffer<u16>,
    qkv: DeviceBuffer<u16>,
    gate: DeviceBuffer<u16>,
    up: DeviceBuffer<u16>,
    act: DeviceBuffer<u16>,
    tmp1: DeviceBuffer<u16>,
    tmp2: DeviceBuffer<u16>,
    pooled: DeviceBuffer<u16>,
    scaled: DeviceBuffer<u16>,
    pooled_norm: DeviceBuffer<u16>,
    rows: DeviceBuffer<u16>,
    cs: DeviceBuffer<f32>,
}

impl Scratch {
    fn new(stream: &CudaStream, l: ScratchLens) -> Result<Self, GpuError> {
        let b = |n| DeviceBuffer::<u16>::zeroed(stream, n);
        Ok(Self {
            raw: DeviceBuffer::zeroed(stream, l.raw)?,
            patch: DeviceBuffer::zeroed(stream, l.hidden)?,
            x: DeviceBuffer::zeroed(stream, l.hidden)?,
            residual: DeviceBuffer::zeroed(stream, l.hidden)?,
            h: b(l.hidden)?,
            h32: DeviceBuffer::zeroed(stream, l.hidden)?,
            work: b(l.hidden)?,
            att: b(l.hidden)?,
            gathered: b(l.hidden)?,
            qkv: b(l.qkv)?,
            gate: b(l.ff)?,
            up: b(l.ff)?,
            act: b(l.ff)?,
            tmp1: b(l.ff)?,
            tmp2: b(l.ff)?,
            pooled: b(l.pooled)?,
            scaled: b(l.pooled)?,
            pooled_norm: b(l.pooled)?,
            rows: b(l.rows)?,
            cs: DeviceBuffer::zeroed(stream, 64 * l.patches)?,
        })
    }
    fn bytes(&self) -> u64 {
        4 * (self.raw.len()
            + self.patch.len()
            + self.cs.len()
            + self.x.len()
            + self.residual.len()
            + self.h32.len()) as u64
            + 2 * [
                &self.h,
                &self.work,
                &self.att,
                &self.gathered,
                &self.qkv,
                &self.gate,
                &self.up,
                &self.act,
                &self.tmp1,
                &self.tmp2,
                &self.pooled,
                &self.scaled,
                &self.pooled_norm,
                &self.rows,
            ]
            .iter()
            .map(|b| b.len() as u64)
            .sum::<u64>()
    }
}

fn permute<T: Copy>(v: &[T], k: usize) -> Vec<T> {
    (0..v.len() / k)
        .flat_map(|r| {
            let h = r / 64 * 64;
            let j = r % 64;
            let src = h + [0, 32, 16, 48][j / 16] + j % 16;
            v[src * k..(src + 1) * k].iter().copied()
        })
        .collect()
}

fn clamp(file: &Gguf, name: &str, side: &str) -> Result<(f32, f32), GpuError> {
    let read = |suffix: &str, default| {
        let key = name.replace(".weight", &format!(".{side}_{suffix}"));
        if file.find(&key).is_none() {
            return Ok(default);
        }
        let v = read_f32(file, &key, WHAT)?;
        match v.as_slice() {
            [x] if x.is_finite() => Ok(*x),
            _ => Err(GpuError::shape(
                WHAT,
                format!("{key}: not one finite scalar"),
            )),
        }
    };
    let bounds = (read("min", -f32::MAX)?, read("max", f32::MAX)?);
    if bounds.0 > bounds.1 {
        return Err(GpuError::shape(
            WHAT,
            format!("{name}: {side} clamp is reversed"),
        ));
    }
    Ok(bounds)
}

impl Encoder {
    /// Load the BF16 vision half of `file` and buffers for `limits.max()` pooled tokens.
    pub fn load(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        file: &Gguf,
        limits: TokenLimits,
    ) -> Result<Self, GpuError> {
        let hp = Hparams::read(file).map_err(|e| GpuError::plan(WHAT, e.to_string()))?;
        let bytes = card::of(file, &hp, limits).map_err(|e| GpuError::plan(WHAT, e.to_string()))?;
        let table = vision::arch::gemma4v::tensors::check(file, &hp)
            .map_err(|e| GpuError::plan(WHAT, e.to_string()))?;
        let mut linear = BTreeMap::new();
        let mut norm = BTreeMap::new();
        for t in &table {
            if t.ty == gguf::GgmlType::BF16 {
                let mut v = read_bf16(file, &t.name, WHAT)?;
                if v.iter().any(|&bits| !crate::bf16_f32(bits).is_finite()) {
                    return Err(GpuError::shape(
                        WHAT,
                        format!("{} contains a non-finite weight", t.name),
                    ));
                }
                if t.name.ends_with(".attn_q.weight") || t.name.ends_with(".attn_k.weight") {
                    v = permute(&v, hp.dim);
                }
                linear.insert(
                    t.name.clone(),
                    Linear {
                        w: DeviceBuffer::from_host(stream, &v)?,
                        input: clamp(file, &t.name, "input")?,
                        output: clamp(file, &t.name, "output")?,
                    },
                );
            } else if t.name.starts_with("v.blk.") && t.name.ends_with(".weight") {
                let mut v = read_f32(file, &t.name, WHAT)?;
                if t.name.ends_with(".attn_q_norm.weight")
                    || t.name.ends_with(".attn_k_norm.weight")
                {
                    v = permute(&v, 1);
                }
                if v.iter().any(|x| !x.is_finite()) {
                    return Err(GpuError::shape(
                        WHAT,
                        format!("{} contains a non-finite gain", t.name),
                    ));
                }
                norm.insert(t.name.clone(), DeviceBuffer::from_host(stream, &v)?);
            }
        }
        let vector = |name, default| -> Result<_, GpuError> {
            let v = if file.find(name).is_some() {
                read_f32(file, name, WHAT)?
            } else {
                vec![default; hp.dim]
            };
            if v.iter().any(|x| !x.is_finite()) {
                return Err(GpuError::shape(
                    WHAT,
                    format!("{name} contains a non-finite value"),
                ));
            }
            Ok(DeviceBuffer::from_host(stream, &v)?)
        };
        let patch = read_f32(file, "v.patch_embd.weight", WHAT)?;
        let pos = read_f32(file, "v.position_embd.weight", WHAT)?;
        if patch.iter().chain(&pos).any(|x| !x.is_finite()) {
            return Err(GpuError::shape(WHAT, "non-finite patch or position weight"));
        }
        let w = Weights {
            patch: DeviceTensor::upload(stream, &patch, hp.dim, 3 * hp.patch * hp.patch)?,
            pos: DeviceBuffer::from_host(stream, &pos)?,
            linear,
            norm,
            std_bias: vector("v.std_bias", 0.0)?,
            std_scale: vector("v.std_scale", 1.0)?,
            ones: DeviceBuffer::from_host(stream, &vec![1.0; hp.dim])?,
            head_ones: DeviceBuffer::from_host(stream, &[1.0f32; 64])?,
        };
        let weight_bytes = 4
            * (w.patch.buf().len()
                + w.pos.len()
                + w.std_bias.len()
                + w.std_scale.len()
                + w.ones.len()
                + w.head_ones.len()
                + w.norm.values().map(DeviceBuffer::len).sum::<usize>()) as u64
            + 2 * w.linear.values().map(|l| l.w.len() as u64).sum::<u64>();
        let lens =
            ScratchLens::new(&hp, limits.max()).map_err(|e| GpuError::plan(WHAT, e.to_string()))?;
        let s = Scratch::new(stream, lens)?;
        let gemm = GemmKernels::load(ctx, stream)?;
        let scratch_bytes = s.bytes() + gemm.scratch_bytes();
        if (weight_bytes, scratch_bytes) != (bytes.weights, bytes.scratch) {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "uploaded weights/scratch {weight_bytes}/{scratch_bytes}, reserved {}/{}",
                    bytes.weights, bytes.scratch
                ),
            ));
        }
        let standardize = file.find("v.std_bias").is_some();
        let clamps = w
            .linear
            .iter()
            .map(|(name, w)| {
                let post = name.ends_with(".attn_out.weight") || name.ends_with(".ffn_down.weight");
                usize::from(w.input != (-f32::MAX, f32::MAX))
                    + if w.output == (-f32::MAX, f32::MAX) {
                        0
                    } else if post {
                        1
                    } else {
                        2
                    }
            })
            .sum::<usize>();
        let launches = 7 + 22 * hp.n_layer + usize::from(standardize) + clamps;
        Ok(Self {
            hp,
            limits,
            standardize,
            w,
            s,
            weight_bytes,
            scratch_bytes,
            launches,
            stream: Arc::clone(stream),
            gemm,
            patch_gemm: Gemm32Kernels::load(ctx)?,
            norm: NormKernels::load(ctx)?,
            ops: pointwise::Kernels::load(ctx)?,
            rope: RopeKernels::load(ctx)?,
            attn: AttnKernels::load(ctx)?,
            mean: MeanGroupsKernels::load(ctx)?,
        })
    }

    /// Encode one raster patch grid with no intermediate readback.
    pub fn encode(&mut self, patches: &Patches) -> Result<Encoded<'_>, GpuError> {
        self.encode_tapped(patches, &mut NoTaps)
    }

    /// Encode one grid, handing requested graph-node rows to `taps`.
    pub fn encode_tapped(
        &mut self,
        p: &Patches,
        taps: &mut dyn TapSink,
    ) -> Result<Encoded<'_>, GpuError> {
        let n = p
            .n_vit_h
            .checked_mul(p.n_vit_w)
            .ok_or_else(|| GpuError::shape(WHAT, "patch count overflows"))?;
        let raw = n
            .checked_mul(p.patch_len)
            .ok_or_else(|| GpuError::shape(WHAT, "patch values overflow"))?;
        let hp = &self.hp;
        let d = hp.dim;
        if n == 0
            || n > 9 * self.limits.max()
            || !p.n_vit_w.is_multiple_of(MERGE)
            || !p.n_vit_h.is_multiple_of(MERGE)
            || p.n_vit_w > hp.pos_rows
            || p.n_vit_h > hp.pos_rows
            || p.patch_len != 3 * hp.patch * hp.patch
            || raw != p.bf16.len()
            || p.plan.span_grid() != (p.n_vit_w / MERGE, p.n_vit_h / MERGE)
        {
            return Err(GpuError::shape(
                WHAT,
                "patch geometry, token budget or learned positions do not match",
            ));
        }
        let stream = &self.stream;
        let s = &mut self.s;
        let input = vision::arch::gemma4v::conv_pixels(&p.bf16)
            .map_err(|e| GpuError::shape(WHAT, e.to_string()))?;
        WindowMut::of_mut(&mut s.raw, 0, raw)?.copy_from_host(stream, &input)?;
        self.patch_gemm
            .enqueue_f32_tile(stream, &self.w.patch, &s.raw, n, &mut s.patch)?;
        self.ops.positions(
            stream,
            pointwise::PositionArgs {
                x: &s.patch,
                table: &self.w.pos,
                rows: n,
                width: d,
                grid_w: p.n_vit_w,
                pos_rows: hp.pos_rows,
                out: &mut s.x,
            },
        )?;
        tap_f32(stream, taps, "pos_embd", &s.x, n, d)?;
        let table = RopeTable::neox_halves(p.n_vit_h, p.n_vit_w, 100.0);
        WindowMut::of_mut(&mut s.cs, 0, table.cs.len())?.copy_from_host(stream, &table.cs)?;
        let c = Chain {
            hp,
            norm: &self.norm,
            rope: &self.rope,
            attn: &self.attn,
            mean: &self.mean,
            standardize: self.standardize,
            gemm: &self.gemm,
            ops: &self.ops,
            w: &self.w,
        };
        for l in 0..hp.n_layer {
            c.attention(stream, s, n, l)?;
            c.ffn(stream, s, n, l)?;
            tap_f32(stream, taps, &format!("layer_out-{l}"), &s.x, n, d)?;
        }
        c.finish(stream, s, n, p.n_vit_w, taps)?;
        let rows = n / 9;
        Ok(Encoded {
            rows: DeviceTensor::window_of(&s.rows, 0, rows, hp.out_dim)?,
            launches: self.launches,
        })
    }
}

// Patch input and output buffers are free after embedding and hold the post-norm work.
struct PostScratch<'a> {
    input: &'a mut DeviceBuffer<u16>,
    raw: &'a mut DeviceBuffer<f32>,
    clamped: &'a mut DeviceBuffer<f32>,
    norm16: &'a mut DeviceBuffer<u16>,
    norm32: &'a mut DeviceBuffer<f32>,
}

struct Chain<'a> {
    hp: &'a Hparams,
    norm: &'a NormKernels,
    rope: &'a RopeKernels,
    attn: &'a AttnKernels,
    mean: &'a MeanGroupsKernels,
    standardize: bool,
    gemm: &'a GemmKernels,
    ops: &'a pointwise::Kernels,
    w: &'a Weights,
}
impl Chain<'_> {
    fn attention(
        &self,
        stream: &CudaStream,
        s: &mut Scratch,
        n: usize,
        l: usize,
    ) -> Result<(), GpuError> {
        let hp = self.hp;
        let d = hp.dim;
        let name = |stem| format!("v.blk.{l}.{stem}.weight");
        self.norm.enqueue_f32(
            stream,
            FloatNormArgs {
                x: &s.x,
                gain: self.gain(&name("ln1"))?,
                eps: hp.eps,
                rows: n,
                to_f32: false,
                y16: &mut s.h,
                y32: &mut s.h32,
            },
        )?;
        for (stem, col) in [("attn_q", 0), ("attn_k", d), ("attn_v", 2 * d)] {
            self.linear(
                stream,
                &name(stem),
                &s.h,
                (n, d, d),
                &mut s.work,
                (&mut s.tmp1, &mut s.tmp2),
            )?;
            self.op(
                stream,
                Op::Copy,
                &s.work,
                &s.work,
                (n, d, 3 * d, col),
                &mut s.qkv,
            )?;
        }
        for (stem, col) in [("attn_q_norm", 0), ("attn_k_norm", d)] {
            self.norm.enqueue_heads(
                stream,
                HeadNormArgs {
                    x: &mut s.qkv,
                    gain: self.gain(&name(stem))?,
                    eps: hp.eps,
                    rows: n,
                    row_width: 3 * d,
                    col0: col,
                    n_heads: hp.n_head,
                },
            )?;
        }
        self.norm.enqueue_heads(
            stream,
            HeadNormArgs {
                x: &mut s.qkv,
                gain: &self.w.head_ones,
                eps: hp.eps,
                rows: n,
                row_width: 3 * d,
                col0: 2 * d,
                n_heads: hp.n_head,
            },
        )?;
        self.rope.enqueue(
            stream,
            RopeArgs {
                cs: &s.cs,
                n,
                row_width: 3 * d,
                q0: 0,
                k0: d,
                n_heads: hp.n_head,
                x: &mut s.qkv,
            },
        )?;
        self.attn.enqueue(
            stream,
            AttnArgs {
                qkv: &s.qkv,
                layout: QkvLayout {
                    row_width: 3 * d,
                    q0: 0,
                    k0: d,
                    v0: 2 * d,
                    n_heads: hp.n_head,
                },
                n,
                scale: 1.0,
                out_width: d,
                out: &mut s.att,
            },
        )?;
        self.post(
            stream,
            (&name("attn_out"), &s.att, d),
            &name("attn_post_norm"),
            n,
            PostScratch {
                input: &mut s.tmp1,
                raw: &mut s.raw,
                clamped: &mut s.patch,
                norm16: &mut s.h,
                norm32: &mut s.h32,
            },
        )?;
        self.ops
            .add_f32(stream, &s.x, &s.h32, n * d, &mut s.residual)?;
        Ok(())
    }
    fn ffn(
        &self,
        stream: &CudaStream,
        s: &mut Scratch,
        n: usize,
        l: usize,
    ) -> Result<(), GpuError> {
        let hp = self.hp;
        let (d, f) = (hp.dim, hp.ff);
        let name = |stem| format!("v.blk.{l}.{stem}.weight");
        self.norm.enqueue_f32(
            stream,
            FloatNormArgs {
                x: &s.residual,
                gain: self.gain(&name("ln2"))?,
                eps: hp.eps,
                rows: n,
                to_f32: false,
                y16: &mut s.h,
                y32: &mut s.h32,
            },
        )?;
        self.linear(
            stream,
            &name("ffn_gate"),
            &s.h,
            (n, f, d),
            &mut s.gate,
            (&mut s.tmp1, &mut s.tmp2),
        )?;
        self.linear(
            stream,
            &name("ffn_up"),
            &s.h,
            (n, f, d),
            &mut s.up,
            (&mut s.tmp1, &mut s.tmp2),
        )?;
        let op = if hp.gelu_tanh {
            Op::GeGlu
        } else {
            Op::QuickGeGlu
        };
        self.op(stream, op, &s.gate, &s.up, (n, f, f, 0), &mut s.act)?;
        self.post(
            stream,
            (&name("ffn_down"), &s.act, f),
            &name("ffn_post_norm"),
            n,
            PostScratch {
                input: &mut s.tmp1,
                raw: &mut s.raw,
                clamped: &mut s.patch,
                norm16: &mut s.h,
                norm32: &mut s.h32,
            },
        )?;
        self.ops
            .add_f32(stream, &s.residual, &s.h32, n * d, &mut s.x)?;
        Ok(())
    }
    fn finish(
        &self,
        stream: &CudaStream,
        s: &mut Scratch,
        n: usize,
        grid_w: usize,
        taps: &mut dyn TapSink,
    ) -> Result<(), GpuError> {
        let hp = self.hp;
        let d = hp.dim;
        let rows = n / 9;
        self.ops.pool_f32(
            stream,
            pointwise::FloatPoolArgs {
                x: &s.x,
                rows: n,
                width: d,
                grid_w,
                out: &mut s.gathered,
            },
        )?;
        self.mean
            .enqueue(stream, n, 9, d, &s.gathered, &mut s.pooled)?;

        self.ops.enqueue(
            stream,
            pointwise::Args {
                op: Op::Scale,
                x: &s.pooled,
                b: &s.pooled,
                gain: &self.w.ones,
                bias: &self.w.std_bias,
                rows,
                width: d,
                out_width: d,
                out_col: 0,
                scale: (d as f32).sqrt(),
                clamp: (-f32::MAX, f32::MAX),
                out: &mut s.scaled,
            },
        )?;
        tap(stream, taps, "pooled", &s.scaled, rows, d)?;
        if self.standardize {
            self.ops.enqueue(
                stream,
                pointwise::Args {
                    op: Op::Affine,
                    x: &s.scaled,
                    b: &s.scaled,
                    gain: &self.w.std_scale,
                    bias: &self.w.std_bias,
                    rows,
                    width: d,
                    out_width: d,
                    out_col: 0,
                    scale: 1.0,
                    clamp: (-f32::MAX, f32::MAX),
                    out: &mut s.pooled,
                },
            )?;
            tap(stream, taps, "std_scaled", &s.pooled, rows, d)?;
        }
        let pooled = if self.standardize {
            &s.pooled
        } else {
            &s.scaled
        };
        self.norm.enqueue(
            stream,
            pooled,
            &self.w.ones,
            hp.eps,
            rows,
            &mut s.pooled_norm,
        )?;
        self.linear(
            stream,
            "mm.input_projection.weight",
            &s.pooled_norm,
            (rows, hp.out_dim, d),
            &mut s.rows,
            (&mut s.tmp1, &mut s.tmp2),
        )?;
        tap(stream, taps, "projected", &s.rows, rows, hp.out_dim)?;
        Ok(())
    }

    fn gain(&self, name: &str) -> Result<&DeviceBuffer<f32>, GpuError> {
        self.w
            .norm
            .get(name)
            .ok_or_else(|| GpuError::tensor(WHAT, name, "a norm gain"))
    }
    fn op(
        &self,
        stream: &CudaStream,
        op: Op,
        x: &DeviceBuffer<u16>,
        b: &DeviceBuffer<u16>,
        shape: (usize, usize, usize, usize),
        out: &mut DeviceBuffer<u16>,
    ) -> Result<(), GpuError> {
        let (rows, width, out_width, out_col) = shape;
        self.ops.enqueue(
            stream,
            pointwise::Args {
                op,
                x,
                b,
                gain: &self.w.ones,
                bias: &self.w.std_bias,
                rows,
                width,
                out_width,
                out_col,
                scale: 1.0,
                clamp: (-f32::MAX, f32::MAX),
                out,
            },
        )
    }
    fn post(
        &self,
        stream: &CudaStream,
        (name, x, k): (&str, &DeviceBuffer<u16>, usize),
        gain: &str,
        n: usize,
        s: PostScratch<'_>,
    ) -> Result<(), GpuError> {
        let d = self.hp.dim;
        let w = self
            .w
            .linear
            .get(name)
            .ok_or_else(|| GpuError::tensor(WHAT, name, "a BF16 linear weight"))?;
        let input = if w.input != (-f32::MAX, f32::MAX) {
            self.ops.enqueue(
                stream,
                pointwise::Args {
                    op: Op::Clamp,
                    x,
                    b: x,
                    gain: &self.w.ones,
                    bias: &self.w.std_bias,
                    rows: n,
                    width: k,
                    out_width: k,
                    out_col: 0,
                    scale: 1.0,
                    clamp: w.input,
                    out: s.input,
                },
            )?;
            &*s.input
        } else {
            x
        };
        self.gemm.enqueue_f32(
            stream,
            FloatGemmArgs {
                a: input,
                b: &w.w,
                m: n,
                n: d,
                k,
                c: s.raw,
            },
        )?;
        let input = if w.output != (-f32::MAX, f32::MAX) {
            self.ops
                .clamp_f32(stream, s.raw, w.output, n * d, s.clamped)?;
            &*s.clamped
        } else {
            &*s.raw
        };
        self.norm.enqueue_f32(
            stream,
            FloatNormArgs {
                x: input,
                gain: self.gain(gain)?,
                eps: self.hp.eps,
                rows: n,
                to_f32: true,
                y16: s.norm16,
                y32: s.norm32,
            },
        )
    }

    fn linear(
        &self,
        stream: &CudaStream,
        name: &str,
        x: &DeviceBuffer<u16>,
        shape: (usize, usize, usize),
        y: &mut DeviceBuffer<u16>,
        temp: (&mut DeviceBuffer<u16>, &mut DeviceBuffer<u16>),
    ) -> Result<(), GpuError> {
        let (n, out, k) = shape;
        let w = self
            .w
            .linear
            .get(name)
            .ok_or_else(|| GpuError::tensor(WHAT, name, "a BF16 linear weight"))?;
        let clamp_into = |x: &DeviceBuffer<u16>, width, bounds, out: &mut DeviceBuffer<u16>| {
            self.ops.enqueue(
                stream,
                pointwise::Args {
                    op: Op::Clamp,
                    x,
                    b: x,
                    gain: &self.w.ones,
                    bias: &self.w.std_bias,
                    rows: n,
                    width,
                    out_width: width,
                    out_col: 0,
                    scale: 1.0,
                    clamp: bounds,
                    out,
                },
            )
        };
        let src = if w.input != (-f32::MAX, f32::MAX) {
            clamp_into(x, k, w.input, temp.0)?;
            &*temp.0
        } else {
            x
        };
        self.gemm.enqueue(
            stream,
            GemmArgs {
                a: src,
                b: &w.w,
                bias: None,
                epilogue: Epilogue::None,
                resid: None,
                m: n,
                n: out,
                k,
                c: y,
            },
        )?;
        if w.output != (-f32::MAX, f32::MAX) {
            clamp_into(y, out, w.output, temp.1)?;
            self.op(stream, Op::Copy, temp.1, temp.1, (n, out, out, 0), y)?;
        }
        Ok(())
    }
}

impl crate::Tower for Encoder {
    fn encode(&mut self, p: &Patches) -> Result<Encoded<'_>, GpuError> {
        Encoder::encode(self, p)
    }
    fn out_dim(&self) -> usize {
        self.hp.out_dim
    }
    fn weight_bytes(&self) -> u64 {
        self.weight_bytes
    }
    fn scratch_bytes(&self) -> u64 {
        self.scratch_bytes
    }
}
