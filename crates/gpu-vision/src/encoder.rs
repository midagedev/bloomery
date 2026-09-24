//! The whole encoder over card buffers: the weights of one `deepseek41v` file uploaded once, and
//! [`Encoder::encode`], from an image's bf16 patches to its aligner rows.
//!
//! The chain, op by op, with the reference line of `inference/vision.py` each launch
//! transcribes (a launch is one kernel; the rounding boundaries are the reference's — every op
//! returns bf16):
//!
//! | # | launch | reference | rounds to bf16 |
//! |---|---|---|---|
//! | 1 | GEMM `patches · patch_embdᵀ + b` (588 → 1024) | `PatchEmbed.proj` :40, :43 | once |
//! | | per block `b` of 32 (`Block.forward` :82–84): | | |
//! | 2 | RMSNorm `ln1` | `Block.norm1` :77, `RMSNorm.forward` :30–34 | once |
//! | 3 | GEMM `h · wqkvᵀ + b` (1024 → 3072) | `Attention.wqkv` :51, :56 | once |
//! | 4 | 2D RoPE of q and k in place | `apply_rotary` :18–21, :57–58; table :8–15, :100 | once per value |
//! | 5 | attention, 16 heads of 64, non-causal | `F.scaled_dot_product_attention` :59 | once |
//! | 6 | GEMM `a · woᵀ + b`, residual epilogue | `Attention.wo` :52, :60; `x + attn` :83 | the product, then the sum |
//! | 7 | RMSNorm `ln2` | `Block.norm2` :79 | once |
//! | 8 | GEMM `h · [gate; up]ᵀ` (1024 → 2·2816), no bias | `MLP.w1` :66, :70 | once |
//! | 9 | `silu(gate) · up` | :71 | the SiLU, then the product |
//! | 10 | GEMM `act · downᵀ`, no bias, residual epilogue | `MLP.w2` :67, :71; `x + mlp` :84 | the product, then the sum |
//! | 11 | RMSNorm `post_ln` | `ViT.norm` :96, :103 | once |
//! | 12 | 3×3 unfold, zero pad | `Aligner.forward` :116–118 | (no arithmetic) |
//! | 13 | GEMM `x · mm1ᵀ + b` (9216 → 5120), GELU epilogue | `Aligner.w1` :111, `F.gelu` :119 | the product, then the GELU |
//! | 14 | GEMM `h · mm2ᵀ + b` (5120 → 5120) | `Aligner.w2` :112, :119 | once |
//!
//! Launches per image: `1 + 9·n_layer + 1 + 3` ([`Encoder::launches`]), 293 at 32 blocks. Eager,
//! not a captured graph: the grid of every launch depends on the image's patch count, which
//! changes per request.
//!
//! Weights: every matrix as the file's bf16; the gains and biases, which the file holds as f32
//! (exact promotions of the checkpoint's bf16), as f32 — the kernels add a bias and scale by a
//! gain in f32. `ffn_gate` and `ffn_up` are uploaded as one `[2·ff, dim]` matrix (gate rows
//! first), `MLP.w1`'s own layout, so step 8 is one GEMM.

use crate::aligner::{AlignerKernels, UnfoldArgs, cells};
use crate::attn::{AttnArgs, AttnKernels, QkvLayout};
use crate::gemm_bf16::{Epilogue, GemmArgs, GemmKernels};
use crate::mlp::MlpKernels;
use crate::norm::NormKernels;
use crate::rope2d::{RopeArgs, RopeKernels, RopeTable};
use bloomery_gpu::{DeviceTensor, GpuError};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use gguf::Gguf;
use std::sync::Arc;
use vision::Patches;
use vision::arch::deepseek41v::{Hparams, names, tensors};

/// One block's weights on the card.
struct BlockWeights {
    ln1: DeviceBuffer<f32>,
    qkv_w: DeviceBuffer<u16>,
    qkv_b: DeviceBuffer<f32>,
    o_w: DeviceBuffer<u16>,
    o_b: DeviceBuffer<f32>,
    ln2: DeviceBuffer<f32>,
    /// `ffn_gate` rows, then `ffn_up` rows.
    w13: DeviceBuffer<u16>,
    w2: DeviceBuffer<u16>,
}

/// Every weight of the encoder on the card.
struct Weights {
    patch_w: DeviceBuffer<u16>,
    patch_b: DeviceBuffer<f32>,
    blocks: Vec<BlockWeights>,
    post_ln: DeviceBuffer<f32>,
    mm1_w: DeviceBuffer<u16>,
    mm1_b: DeviceBuffer<f32>,
    mm2_w: DeviceBuffer<u16>,
    mm2_b: DeviceBuffer<f32>,
}

impl Weights {
    /// Bytes on the card: two per bf16, four per f32.
    fn bytes(&self) -> usize {
        let blocks: usize = self
            .blocks
            .iter()
            .map(|b| {
                2 * (b.qkv_w.len() + b.o_w.len() + b.w13.len() + b.w2.len())
                    + 4 * (b.ln1.len() + b.qkv_b.len() + b.o_b.len() + b.ln2.len())
            })
            .sum();
        blocks
            + 2 * (self.patch_w.len() + self.mm1_w.len() + self.mm2_w.len())
            + 4 * (self.patch_b.len() + self.post_ln.len() + self.mm1_b.len() + self.mm2_b.len())
    }
}

/// Where an encode's intermediate tensors go when a caller asks for them (a gate). `wants` is
/// asked before each named point; a `true` synchronizes the stream and hands the tensor over
/// as bf16 bits, `cols` values per row. Names are the oracle's file stems: `embed`, `blk{b}`,
/// `blk{b}.norm1`, `.qkv`, `.qrot`, `.krot`, `.sdpa`, `.attn`, `.resid1` (the stream after the
/// attention residual), `.norm2`, `.w1`, `.act`, `.mlp`, then `vit`, `aligner.x` (the unfolded
/// rows), `aligner.w1`, `aligner.h`. `blk{b}.attn`, `blk{b}.mlp` and `aligner.w1` are the
/// branch outputs before their fused epilogue: asking for one runs its GEMM a second time without
/// the epilogue, into a buffer of its own, which leaves the chain's buffers untouched.
pub trait TapSink {
    fn wants(&self, name: &str) -> bool;
    fn take(&mut self, name: &str, cols: usize, bits: Vec<u16>);
}

/// The sink of a plain encode.
struct NoTaps;

impl TapSink for NoTaps {
    fn wants(&self, _: &str) -> bool {
        false
    }
    fn take(&mut self, _: &str, _: usize, _: Vec<u16>) {}
}

/// An encode's result: the aligner rows (`n_llm_h · n_llm_w` rows of `out_dim` bf16, reading
/// order) and the chain launches it took.
pub struct Encoded {
    pub rows: DeviceTensor<u16>,
    pub launches: usize,
}

/// The loaded encoder: the kernels and the weights of one file.
pub struct Encoder {
    hp: Hparams,
    gemm: GemmKernels,
    rope: RopeKernels,
    attn: AttnKernels,
    norm: NormKernels,
    mlp: MlpKernels,
    aligner: AlignerKernels,
    w: Weights,
    weight_bytes: usize,
}

/// The activations of one encode, sized for its patch count, and its RoPE table.
struct Scratch {
    n_h: usize,
    n_w: usize,
    cs: DeviceBuffer<f32>,
    /// The block input and output, and the aligner's input rows when `h` holds the final norm.
    xa: DeviceBuffer<u16>,
    xb: DeviceBuffer<u16>,
    h: DeviceBuffer<u16>,
    qkv: DeviceBuffer<u16>,
    att: DeviceBuffer<u16>,
    u: DeviceBuffer<u16>,
    act: DeviceBuffer<u16>,
    /// `h` already holds the final norm's output (a forced tail from it): the tail skips the norm.
    vit_given: bool,
}

fn err(what: &'static str, e: impl std::error::Error + Send + Sync + 'static) -> GpuError {
    GpuError::plan(what, e)
}

impl Encoder {
    /// Read the file's hyperparameters and tensor table (both refused by name when they are not
    /// the V4.1 encoder's) and upload every weight. Load-time only.
    pub fn load(
        ctx: &Arc<CudaContext>,
        stream: &CudaStream,
        file: &Gguf,
    ) -> Result<Encoder, GpuError> {
        let what = "Encoder::load";
        let hp = Hparams::read(file).map_err(|e| err(what, e))?;
        tensors::check(
            &hp,
            file.iter_tensors()
                .map(|t| (t.name.as_str(), t.dims.as_slice(), t.ty)),
        )
        .map_err(|e| err(what, e))?;
        let bf16 = |name: String| up_bf16(stream, file, &name);
        let f32s = |name: String| up_f32(stream, file, &name);
        let mut blocks = Vec::with_capacity(hp.n_layer);
        for b in 0..hp.n_layer {
            let mut gate_up = read_bf16(file, &names::ffn_gate(b))?;
            gate_up.extend(read_bf16(file, &names::ffn_up(b))?);
            blocks.push(BlockWeights {
                ln1: f32s(names::ln1(b))?,
                qkv_w: bf16(names::attn_qkv_weight(b))?,
                qkv_b: f32s(names::attn_qkv_bias(b))?,
                o_w: bf16(names::attn_out_weight(b))?,
                o_b: f32s(names::attn_out_bias(b))?,
                ln2: f32s(names::ln2(b))?,
                w13: DeviceBuffer::from_host(stream, &gate_up)?,
                w2: bf16(names::ffn_down(b))?,
            });
        }
        let w = Weights {
            patch_w: bf16(names::patch_embd_weight())?,
            patch_b: f32s(names::patch_embd_bias())?,
            blocks,
            post_ln: f32s(names::post_ln())?,
            mm1_w: bf16(names::mm1_weight())?,
            mm1_b: f32s(names::mm1_bias())?,
            mm2_w: bf16(names::mm2_weight())?,
            mm2_b: f32s(names::mm2_bias())?,
        };
        let bytes = w.bytes();
        stream.synchronize()?;
        Ok(Encoder {
            gemm: GemmKernels::load(ctx, stream)?,
            rope: RopeKernels::load(ctx)?,
            attn: AttnKernels::load(ctx)?,
            norm: NormKernels::load(ctx)?,
            mlp: MlpKernels::load(ctx)?,
            aligner: AlignerKernels::load(ctx)?,
            hp,
            w,
            weight_bytes: bytes,
        })
    }

    /// The file's hyperparameters.
    #[must_use]
    pub fn hparams(&self) -> &Hparams {
        &self.hp
    }

    /// Bytes of weights on the card.
    #[must_use]
    pub fn weight_bytes(&self) -> usize {
        self.weight_bytes
    }

    /// Chain launches of one encode at `n_layer` blocks: the patch GEMM, nine per block, the
    /// final norm, the unfold and the aligner's two GEMMs.
    #[must_use]
    pub const fn launches(n_layer: usize) -> usize {
        1 + 9 * n_layer + 1 + 3
    }

    /// Encode one image's patches. Eager: allocates the activations for this patch count,
    /// enqueues the chain on `stream` and returns without synchronizing.
    pub fn encode(&self, stream: &CudaStream, patches: &Patches) -> Result<Encoded, GpuError> {
        self.encode_tapped(stream, patches, &mut NoTaps)
    }

    /// [`Encoder::encode`] handing the named intermediate tensors to `taps` ([`TapSink`]).
    pub fn encode_tapped(
        &self,
        stream: &CudaStream,
        patches: &Patches,
        taps: &mut dyn TapSink,
    ) -> Result<Encoded, GpuError> {
        let what = "Encoder::encode";
        let hp = &self.hp;
        let (n_h, n_w) = (patches.n_vit_h, patches.n_vit_w);
        let n = n_h * n_w;
        let k_patch = 3 * hp.patch * hp.patch;
        if n == 0 || patches.patch_len != k_patch || patches.bf16.len() != n * k_patch {
            return Err(GpuError::Shape {
                what,
                detail: format!(
                    "{n_h}x{n_w} patches of {} values ({} in all); the encoder takes {k_patch} per patch",
                    patches.patch_len,
                    patches.bf16.len()
                ),
            });
        }
        let x_in = DeviceBuffer::from_host(stream, &patches.bf16)?;
        let mut s = self.scratch(stream, n_h, n_w)?;
        self.gemm.enqueue(
            stream,
            GemmArgs {
                a: &x_in,
                b: &self.w.patch_w,
                bias: Some(&self.w.patch_b),
                epilogue: Epilogue::None,
                resid: None,
                m: n,
                n: hp.dim,
                k: k_patch,
                c: &mut s.xa,
            },
        )?;
        let mut launches = 1;
        tap(stream, taps, "embed", &s.xa, hp.dim)?;
        for b in 0..hp.n_layer {
            launches += self.block(stream, b, &mut s, taps)?;
        }
        let (rows, tail) = self.tail(stream, &mut s, taps)?;
        Ok(Encoded {
            rows,
            launches: launches + tail,
        })
    }

    /// One block alone on the host rows `x` (`n_h · n_w` rows of `dim` bf16): the chain's launches
    /// of block `b` from `x` and the block's output — a teacher-forced step, for a gate that feeds
    /// the reference's input to each block in turn.
    pub fn forced_block(
        &self,
        stream: &CudaStream,
        (n_h, n_w): (usize, usize),
        b: usize,
        x: &[u16],
    ) -> Result<Vec<u16>, GpuError> {
        let mut s = self.forced_scratch(stream, (n_h, n_w), x)?;
        self.block(stream, b, &mut s, &mut NoTaps)?;
        to_host(stream, &s.xa)
    }

    /// The chain after the last block alone on the host rows `x`: the final norm's output and the
    /// aligner rows, teacher-forced as [`Encoder::forced_block`]. With `from_vit`, `x` is taken as
    /// the final norm's output and only the aligner runs.
    pub fn forced_tail(
        &self,
        stream: &CudaStream,
        (n_h, n_w): (usize, usize),
        x: &[u16],
        from_vit: bool,
    ) -> Result<(Vec<u16>, Vec<u16>), GpuError> {
        let mut s = self.forced_scratch(stream, (n_h, n_w), x)?;
        if from_vit {
            s.vit_given = true;
        }
        let (rows, _) = self.tail(stream, &mut s, &mut NoTaps)?;
        Ok((to_host(stream, &s.h)?, to_host(stream, rows.buf())?))
    }

    /// The activations of an `n_h × n_w` image and its RoPE table.
    fn scratch(&self, stream: &CudaStream, n_h: usize, n_w: usize) -> Result<Scratch, GpuError> {
        let hp = &self.hp;
        let (dim, ff, n) = (hp.dim, hp.ff, n_h * n_w);
        let table = RopeTable::new(n_h, n_w, hp.rope_theta);
        Ok(Scratch {
            n_h,
            n_w,
            cs: DeviceBuffer::from_host(stream, &table.cs)?,
            xa: DeviceBuffer::zeroed(stream, n * dim)?,
            xb: DeviceBuffer::zeroed(stream, n * dim)?,
            h: DeviceBuffer::zeroed(stream, n * dim)?,
            qkv: DeviceBuffer::zeroed(stream, n * 3 * dim)?,
            att: DeviceBuffer::zeroed(stream, n * dim)?,
            u: DeviceBuffer::zeroed(stream, n * 2 * ff)?,
            act: DeviceBuffer::zeroed(stream, n * ff)?,
            vit_given: false,
        })
    }

    /// [`Encoder::scratch`] with the host rows `x` as the block input (`xa`) and as the final
    /// norm's output (`h`).
    fn forced_scratch(
        &self,
        stream: &CudaStream,
        (n_h, n_w): (usize, usize),
        x: &[u16],
    ) -> Result<Scratch, GpuError> {
        let dim = self.hp.dim;
        if n_h == 0 || n_w == 0 || x.len() != n_h * n_w * dim {
            return Err(GpuError::Shape {
                what: "Encoder::forced",
                detail: format!("{} values for {n_h}x{n_w} rows of {dim}", x.len()),
            });
        }
        let mut s = self.scratch(stream, n_h, n_w)?;
        s.xa = DeviceBuffer::from_host(stream, x)?;
        s.h = DeviceBuffer::from_host(stream, x)?;
        Ok(s)
    }

    /// Block `b` of the chain on `s.xa`, its output back in `s.xa`; returns the launches.
    fn block(
        &self,
        stream: &CudaStream,
        b: usize,
        s: &mut Scratch,
        taps: &mut dyn TapSink,
    ) -> Result<usize, GpuError> {
        let hp = &self.hp;
        let bw = &self.w.blocks[b];
        let (dim, ff, heads) = (hp.dim, hp.ff, hp.n_head);
        let n = s.n_h * s.n_w;
        let lay = QkvLayout {
            row_width: 3 * dim,
            q0: 0,
            k0: dim,
            v0: 2 * dim,
            n_heads: heads,
        };
        let scale = 1.0 / ((dim / heads) as f32).sqrt();
        let name = |op: &str| format!("blk{b}.{op}");
        self.norm
            .enqueue(stream, &s.xa, &bw.ln1, hp.eps, n, &mut s.h)?;
        tap(stream, taps, &name("norm1"), &s.h, dim)?;
        self.gemm.enqueue(
            stream,
            GemmArgs {
                a: &s.h,
                b: &bw.qkv_w,
                bias: Some(&bw.qkv_b),
                epilogue: Epilogue::None,
                resid: None,
                m: n,
                n: 3 * dim,
                k: dim,
                c: &mut s.qkv,
            },
        )?;
        tap(stream, taps, &name("qkv"), &s.qkv, 3 * dim)?;
        self.rope.enqueue(
            stream,
            RopeArgs {
                cs: &s.cs,
                n,
                row_width: 3 * dim,
                q0: 0,
                k0: dim,
                n_heads: heads,
                x: &mut s.qkv,
            },
        )?;
        for (op, c0) in [("qrot", 0), ("krot", dim)] {
            if taps.wants(&name(op)) {
                let all = to_host(stream, &s.qkv)?;
                let cols: Vec<u16> = all
                    .chunks_exact(3 * dim)
                    .flat_map(|r| r[c0..c0 + dim].iter().copied())
                    .collect();
                taps.take(&name(op), dim, cols);
            }
        }
        self.attn.enqueue(
            stream,
            AttnArgs {
                qkv: &s.qkv,
                layout: lay,
                n,
                scale,
                out_width: dim,
                out: &mut s.att,
            },
        )?;
        tap(stream, taps, &name("sdpa"), &s.att, dim)?;
        self.side(
            stream,
            taps,
            &name("attn"),
            (&s.att, &bw.o_w, Some(&bw.o_b)),
            (n, dim, dim),
        )?;
        self.gemm.enqueue(
            stream,
            GemmArgs {
                a: &s.att,
                b: &bw.o_w,
                bias: Some(&bw.o_b),
                epilogue: Epilogue::Residual,
                resid: Some(&s.xa),
                m: n,
                n: dim,
                k: dim,
                c: &mut s.xb,
            },
        )?;
        tap(stream, taps, &name("resid1"), &s.xb, dim)?;
        self.norm
            .enqueue(stream, &s.xb, &bw.ln2, hp.eps, n, &mut s.h)?;
        tap(stream, taps, &name("norm2"), &s.h, dim)?;
        self.gemm.enqueue(
            stream,
            GemmArgs {
                a: &s.h,
                b: &bw.w13,
                bias: None,
                epilogue: Epilogue::None,
                resid: None,
                m: n,
                n: 2 * ff,
                k: dim,
                c: &mut s.u,
            },
        )?;
        tap(stream, taps, &name("w1"), &s.u, 2 * ff)?;
        self.mlp.enqueue(stream, &s.u, ff, n, &mut s.act)?;
        tap(stream, taps, &name("act"), &s.act, ff)?;
        self.side(
            stream,
            taps,
            &name("mlp"),
            (&s.act, &bw.w2, None),
            (n, dim, ff),
        )?;
        self.gemm.enqueue(
            stream,
            GemmArgs {
                a: &s.act,
                b: &bw.w2,
                bias: None,
                epilogue: Epilogue::Residual,
                resid: Some(&s.xb),
                m: n,
                n: dim,
                k: ff,
                c: &mut s.xa,
            },
        )?;
        tap(stream, taps, &format!("blk{b}"), &s.xa, dim)?;
        Ok(9)
    }

    /// The chain after the last block: the final norm of `s.xa` into `s.h` (skipped when
    /// `s.vit_given`), the unfold and the aligner's two GEMMs; returns the rows and the launches.
    fn tail(
        &self,
        stream: &CudaStream,
        s: &mut Scratch,
        taps: &mut dyn TapSink,
    ) -> Result<(DeviceTensor<u16>, usize), GpuError> {
        let hp = &self.hp;
        let dim = hp.dim;
        let (n_h, n_w) = (s.n_h, s.n_w);
        let n = n_h * n_w;
        let (ch, cw) = cells(n_h, n_w, hp.downsample);
        let n_llm = ch * cw;
        let unfold_w = dim * hp.downsample * hp.downsample;
        let mut launches = 0;
        if !s.vit_given {
            self.norm
                .enqueue(stream, &s.xa, &self.w.post_ln, hp.eps, n, &mut s.h)?;
            launches += 1;
        }
        tap(stream, taps, "vit", &s.h, dim)?;
        let mut un = DeviceBuffer::zeroed(stream, n_llm * unfold_w)?;
        self.aligner.enqueue(
            stream,
            UnfoldArgs {
                x: &s.h,
                n_h,
                n_w,
                dim,
                r: hp.downsample,
                y: &mut un,
            },
        )?;
        tap(stream, taps, "aligner.x", &un, unfold_w)?;
        self.side(
            stream,
            taps,
            "aligner.w1",
            (&un, &self.w.mm1_w, Some(&self.w.mm1_b)),
            (n_llm, hp.out_dim, unfold_w),
        )?;
        let mut hh = DeviceBuffer::zeroed(stream, n_llm * hp.out_dim)?;
        self.gemm.enqueue(
            stream,
            GemmArgs {
                a: &un,
                b: &self.w.mm1_w,
                bias: Some(&self.w.mm1_b),
                epilogue: Epilogue::Gelu,
                resid: None,
                m: n_llm,
                n: hp.out_dim,
                k: unfold_w,
                c: &mut hh,
            },
        )?;
        tap(stream, taps, "aligner.h", &hh, hp.out_dim)?;
        let mut rows = DeviceTensor::zeroed(stream, n_llm, hp.out_dim)?;
        self.gemm.enqueue(
            stream,
            GemmArgs {
                a: &hh,
                b: &self.w.mm2_w,
                bias: Some(&self.w.mm2_b),
                epilogue: Epilogue::None,
                resid: None,
                m: n_llm,
                n: hp.out_dim,
                k: hp.out_dim,
                c: rows.buf_mut(),
            },
        )?;
        launches += 3;
        Ok((rows, launches))
    }

    /// A branch GEMM without its fused epilogue, for a tap only: `a · bᵀ (+ bias)` of shape
    /// `(m, n, k)` into a buffer of its own, handed to `taps` under `name`.
    #[allow(
        clippy::type_complexity,
        reason = "the operand triple and the shape triple of one GEMM, named at the call"
    )]
    fn side(
        &self,
        stream: &CudaStream,
        taps: &mut dyn TapSink,
        name: &str,
        (a, b, bias): (
            &DeviceBuffer<u16>,
            &DeviceBuffer<u16>,
            Option<&DeviceBuffer<f32>>,
        ),
        (m, n, k): (usize, usize, usize),
    ) -> Result<(), GpuError> {
        if !taps.wants(name) {
            return Ok(());
        }
        let mut c = DeviceBuffer::zeroed(stream, m * n)?;
        self.gemm.enqueue(
            stream,
            GemmArgs {
                a,
                b,
                bias,
                epilogue: Epilogue::None,
                resid: None,
                m,
                n,
                k,
                c: &mut c,
            },
        )?;
        tap(stream, taps, name, &c, n)
    }
}

/// Hand `buf` to `taps` under `name` when it asks for it.
fn tap(
    stream: &CudaStream,
    taps: &mut dyn TapSink,
    name: &str,
    buf: &DeviceBuffer<u16>,
    cols: usize,
) -> Result<(), GpuError> {
    if taps.wants(name) {
        let v = to_host(stream, buf)?;
        taps.take(name, cols, v);
    }
    Ok(())
}

fn to_host(stream: &CudaStream, buf: &DeviceBuffer<u16>) -> Result<Vec<u16>, GpuError> {
    stream.synchronize()?;
    Ok(buf.to_host_vec(stream)?)
}

fn tensor_bytes<'a>(file: &'a Gguf, name: &str) -> Result<&'a [u8], GpuError> {
    let t = file.find(name).ok_or_else(|| GpuError::Tensor {
        what: "Encoder::load",
        name: name.to_string(),
        need: "in the encoder file",
    })?;
    Ok(file.data(t)?)
}

fn up_bf16(stream: &CudaStream, file: &Gguf, name: &str) -> Result<DeviceBuffer<u16>, GpuError> {
    Ok(DeviceBuffer::from_host(stream, &read_bf16(file, name)?)?)
}

fn up_f32(stream: &CudaStream, file: &Gguf, name: &str) -> Result<DeviceBuffer<f32>, GpuError> {
    Ok(DeviceBuffer::from_host(stream, &read_f32(file, name)?)?)
}

fn read_bf16(file: &Gguf, name: &str) -> Result<Vec<u16>, GpuError> {
    Ok(tensor_bytes(file, name)?
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| u16::from_le_bytes(*c))
        .collect())
}

fn read_f32(file: &Gguf, name: &str) -> Result<Vec<f32>, GpuError> {
    Ok(tensor_bytes(file, name)?
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect())
}
