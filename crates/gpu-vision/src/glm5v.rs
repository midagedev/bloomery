//! The `glm5v` tower (GLM-5.3-Flash's mmproj: llama.cpp's `glm4v` graph with a clamped SwiGLU) on
//! the card: the weights of one encoder file uploaded once, the activations of the largest image
//! its token limit allows allocated once beside them, and [`Encoder::encode`], from an image's
//! bf16 patches (merge order, each patch repeated over the two temporal frames,
//! [`vision::arch::glm5v::media`]) to the bf16 rows the text model reads in place of the image's
//! tokens.
//!
//! The chain, op by op, with the graph node of `glm4v.cpp` / `clip.cpp` each launch transcribes
//! (a launch is one kernel; every op reads bf16, computes in f32 and rounds its output to bf16,
//! the contract of [`crate`]):
//!
//! | # | launch | graph node | tap |
//! |---|---|---|---|
//! | 1 | GEMM `patches · [w0 \| w1]ᵀ + b` (K = 2·3·p²) | the two patch convolutions, their sum and `patch_bias` | `embed` |
//! | | per block `b` of `n_layer`: | | |
//! | 2 | RMS norm `ln1` | `ln1` | `blk{b}.norm1` |
//! | 3 | GEMM `h · wqkvᵀ + b` | `Qcur`/`Kcur`/`Vcur` | `blk{b}.qkv` |
//! | 4, 5 | per-head RMS norm of q, then k, in place, one gain of 64 | `Qcur_norm`, `Kcur_norm` | `blk{b}.qnorm`, `.knorm` |
//! | 6 | vision RoPE of q and k in place | `Qcur_pos`, `Kcur_pos` | `blk{b}.qrot`, `.krot` |
//! | 7 | attention, heads of 64, non-causal | `build_attn` | `blk{b}.sdpa` |
//! | 8 | GEMM `a · woᵀ + b`, residual epilogue | `attn_out`, `ffn_inp` | `blk{b}.attn` (without the residual), `blk{b}.resid1` |
//! | 9 | RMS norm `ln2` | `ffn_inp_normed` | `blk{b}.norm2` |
//! | 10 | GEMM `h · [gate \| up]ᵀ + [bg \| bu]` (one GEMM of `2·ff` columns) | `ffn_gate_b`, `ffn_up_b` | `blk{b}.gu` |
//! | 11 | clamped SwiGLU | `ffn_swiglu` | `blk{b}.act` |
//! | 12 | GEMM `act · downᵀ + b`, residual epilogue | `ffn_down`, `layer_out` | `blk{b}.mlp` (without the residual), `blk{b}` |
//! | 13 | RMS norm `post_ln` | `vit_out` | `vit_out` |
//! | 14 | GEMM `[n/4, 4·dim] · convᵀ + b`, the merge a re-read of the rows | the 2×2 convolution | `merger.conv` |
//! | 15 | GEMM `· fcᵀ` | `build_mm(fc)` | `merger.fc` |
//! | 16 | LayerNorm with gain and bias | `build_norm(post_norm)` | |
//! | 17 | GELU (erf), in place | `ggml_gelu_erf` | `after_fc_proj` |
//! | 18 | GEMM `· [gate \| up]ᵀ`, no bias | `build_ffn` | `merger.gu` |
//! | 19 | clamped SwiGLU | `ggml_swiglu_clamp` | `merger.act` |
//! | 20 | GEMM `· downᵀ` | `build_ffn` | `after_ffn_proj`, `embd` (the rows) |
//!
//! Launches per image: [`Encoder::launches`]. Eager, as the other towers: the grid of every launch
//! depends on the image's patch count.
//!
//! Where this tower differs from the shared kernels' other users, and how the difference enters:
//! the RoPE is [`crate::rope2d`]'s kernel over a table of llama.cpp's cache loop ([`ggml_rope_table`]
//! at `ROPE_THETA`; the pairing of its `rope_vision` with four sections of 16 is that file's: rows
//! turn the first sixteen pairs, columns the last sixteen);
//! the per-head norm is [`crate::norm::NormKernels::enqueue_heads`]; the SwiGLU's clamp is a value
//! of [`crate::mlp::MlpKernels::enqueue`]; the merger's GELU is [`crate::gelu_tanh::GeluKind`]'s
//! `Erf`; the convolution's K order is [`vision::arch::glm5v::rules`]'s, applied at load.
//!
//! Weights as the card holds them: every matrix bf16 (an f32 or f16 tensor narrows only when
//! every value round-trips through bf16, else the load is refused by name); the two patch kernels
//! as the halves of one `[dim, 2·3·p²]` matrix; a block's gate and up (and the merger MLP's) as
//! one matrix of the gate's rows then the up's; the merger's convolution reordered to the card's
//! K order; every vector f32. The load refuses uploads that differ from
//! [`vision::arch::glm5v::card::weights`].
//!
//! Scratch is allocated once from the token limit ([`vision::arch::glm5v::card::ScratchLens`]); an
//! image allocates nothing, and one past the limit is refused by name.

use crate::attn::{AttnArgs, AttnKernels, QkvLayout};
use crate::chain::{
    Encoded, NoTaps, TapSink, read_bf16_narrowed, read_f32, side, tap, to_host, up_f32,
};
use crate::gelu_tanh::{GeluKind, GeluTanhKernels};
use crate::gemm_bf16::{Epilogue, GemmArgs, GemmKernels};
use crate::layer_norm::LayerNormKernels;
use crate::mlp::MlpKernels;
use crate::norm::{HEAD, HeadNormArgs, NormKernels};
use crate::rope2d::{PAIRS, RopeArgs, RopeKernels, RopeTable, merge_order};
use bloomery_gpu::{DeviceTensor, GpuError, WindowMut};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use gguf::Gguf;
use std::sync::Arc;
use vision::Patches;
use vision::arch::glm5v::card::{self, ScratchLens};
use vision::arch::glm5v::rules::reorder_merger_weight;
use vision::arch::glm5v::{
    Hparams, MERGER_FF, MERGER_NORM_EPS, ROPE_THETA, TEMPORAL_FRAMES, names, tensors,
};

const WHAT: &str = "glm5v tower load";

/// One block's weights on the card.
struct BlockWeights {
    ln1: DeviceBuffer<f32>,
    q_norm: DeviceBuffer<f32>,
    k_norm: DeviceBuffer<f32>,
    qkv_w: DeviceBuffer<u16>,
    qkv_b: DeviceBuffer<f32>,
    o_w: DeviceBuffer<u16>,
    o_b: DeviceBuffer<f32>,
    ln2: DeviceBuffer<f32>,
    /// The gate's rows then the up's: `2·ff` rows, and their biases likewise.
    gu_w: DeviceBuffer<u16>,
    gu_b: DeviceBuffer<f32>,
    down_w: DeviceBuffer<u16>,
    down_b: DeviceBuffer<f32>,
}

/// Every weight of the tower on the card.
struct Weights {
    /// `[dim, 2·3·p²]`: row `o` is the first temporal kernel's row `o`, then the second's.
    patch_w: DeviceBuffer<u16>,
    patch_b: DeviceBuffer<f32>,
    blocks: Vec<BlockWeights>,
    post_ln: DeviceBuffer<f32>,
    /// `[out, merge²·dim]` in the card's K order.
    conv_w: DeviceBuffer<u16>,
    conv_b: DeviceBuffer<f32>,
    fc_w: DeviceBuffer<u16>,
    post_norm: (DeviceBuffer<f32>, DeviceBuffer<f32>),
    /// The merger MLP's gate rows then up rows (`2·MERGER_FF` rows) and its down matrix.
    mgu_w: DeviceBuffer<u16>,
    mdown_w: DeviceBuffer<u16>,
}

impl Weights {
    /// Bytes uploaded: two per bf16, four per f32.
    fn uploaded(&self) -> u64 {
        let blocks: usize = self
            .blocks
            .iter()
            .map(|b| {
                2 * (b.qkv_w.len() + b.o_w.len() + b.gu_w.len() + b.down_w.len())
                    + 4 * (b.ln1.len()
                        + b.q_norm.len()
                        + b.k_norm.len()
                        + b.qkv_b.len()
                        + b.o_b.len()
                        + b.ln2.len()
                        + b.gu_b.len()
                        + b.down_b.len())
            })
            .sum();
        (blocks
            + 2 * (self.patch_w.len()
                + self.conv_w.len()
                + self.fc_w.len()
                + self.mgu_w.len()
                + self.mdown_w.len())
            + 4 * (self.patch_b.len()
                + self.post_ln.len()
                + self.conv_b.len()
                + self.post_norm.0.len()
                + self.post_norm.1.len())) as u64
    }
}

/// What an encode reads and never writes: the hyperparameters, the kernels and the weights.
struct Chain {
    hp: Hparams,
    gemm: GemmKernels,
    norm: NormKernels,
    layer_norm: LayerNormKernels,
    mlp: MlpKernels,
    gelu: GeluTanhKernels,
    rope: RopeKernels,
    attn: AttnKernels,
    w: Weights,
}

/// The activations of the largest image the limit allows, allocated at load. An image of `n`
/// patches uses the first `n` rows of each buffer: its patches and RoPE table are written first,
/// and every launch reads only rows that a copy or an earlier launch of the same image wrote, so
/// nothing a larger image left behind is read.
struct Scratch {
    lens: ScratchLens,
    patches: DeviceBuffer<u16>,
    cs: DeviceBuffer<f32>,
    /// The block input and output, the attention-residual stream, and the normed rows.
    xa: DeviceBuffer<u16>,
    xb: DeviceBuffer<u16>,
    h: DeviceBuffer<u16>,
    qkv: DeviceBuffer<u16>,
    att: DeviceBuffer<u16>,
    /// The MLP's fused gate and up rows, and the clamped SwiGLU's output.
    gu: DeviceBuffer<u16>,
    act: DeviceBuffer<u16>,
    /// The merger's convolution output; the LayerNorm writes it again from `fc`, so it holds the
    /// GELU's rows (`after_fc_proj`) at the end.
    conv: DeviceBuffer<u16>,
    fc: DeviceBuffer<u16>,
    mgu: DeviceBuffer<u16>,
    mact: DeviceBuffer<u16>,
    rows: DeviceBuffer<u16>,
}

/// The loaded tower: the chain of one file, the activations of the largest image its limit
/// allows, its bytes on the card, and the stream every call runs on. A call takes `&mut self`: one
/// image at a time owns the activations, and the rows it returns borrow them until the next call.
pub struct Encoder {
    chain: Chain,
    s: Scratch,
    weight_bytes: u64,
    scratch_bytes: u64,
    stream: Arc<CudaStream>,
}

impl crate::Tower for Encoder {
    fn encode(&mut self, patches: &Patches) -> Result<Encoded<'_>, GpuError> {
        Encoder::encode(self, patches)
    }
    fn out_dim(&self) -> usize {
        self.hparams().out_dim
    }
    fn weight_bytes(&self) -> u64 {
        Encoder::weight_bytes(self)
    }
    fn scratch_bytes(&self) -> u64 {
        Encoder::scratch_bytes(self)
    }
}

fn err(what: &'static str, e: impl std::error::Error + Send + Sync + 'static) -> GpuError {
    GpuError::plan(what, e)
}

/// The RoPE table of the patches `coords` (each `(row, column)` of an `n_h × n_w` grid) as
/// llama.cpp's vision RoPE builds its cache on the CPU in f32 (`ggml_mrope_cache_init`, four
/// sections of sixteen): `theta_scale = ROPE_THETA^(−2/32)`; the angle of pair `k` of an axis is
/// the position times `theta_scale^k`, formed by multiplying the running angle by `theta_scale`
/// once per pair (the first sixteen pairs take the row, the next sixteen the column, each axis
/// starting again from its own position); `cos` and `sin` of that f32 angle, correctly rounded.
/// The pairing is [`RopeTable::in_order`]'s, so the table feeds [`RopeKernels`] unchanged.
fn ggml_rope_table(
    (n_h, n_w): (usize, usize),
    coords: impl IntoIterator<Item = (usize, usize)>,
) -> RopeTable {
    let theta_scale = f64::from(ROPE_THETA).powf(f64::from(-2.0f32 / PAIRS as f32)) as f32;
    let mut cs = Vec::with_capacity(n_h * n_w * 2 * PAIRS);
    for (h, w) in coords {
        for pos in [h, w] {
            let mut angle = pos as f32;
            for _ in 0..PAIRS / 2 {
                cs.push(f64::from(angle).cos() as f32);
                cs.push(f64::from(angle).sin() as f32);
                angle *= theta_scale;
            }
        }
    }
    RopeTable { n_h, n_w, cs }
}

/// One linear layer `c = a · wᵀ (+ bias)`, with the block's residual added when `resid` is given.
fn linear(
    gemm: &GemmKernels,
    stream: &CudaStream,
    (a, w, bias): (
        &DeviceBuffer<u16>,
        &DeviceBuffer<u16>,
        Option<&DeviceBuffer<f32>>,
    ),
    (m, n, k): (usize, usize, usize),
    resid: Option<&DeviceBuffer<u16>>,
    c: &mut DeviceBuffer<u16>,
) -> Result<(), GpuError> {
    gemm.enqueue(
        stream,
        GemmArgs {
            a,
            b: w,
            bias,
            epilogue: if resid.is_some() {
                Epilogue::Residual
            } else {
                Epilogue::None
            },
            resid,
            m,
            n,
            k,
            c,
        },
    )
}

/// Tap the columns `c0 .. c0 + width` of the first `n` rows of `buf`, which has `row_width`
/// columns, when the sink wants them.
fn tap_cols(
    stream: &CudaStream,
    taps: &mut dyn TapSink,
    name: &str,
    buf: &DeviceBuffer<u16>,
    (n, row_width): (usize, usize),
    (c0, width): (usize, usize),
) -> Result<(), GpuError> {
    if taps.wants(name) {
        let all = to_host(stream, buf, n * row_width)?;
        let cols: Vec<u16> = all
            .chunks_exact(row_width)
            .flat_map(|r| r[c0..c0 + width].iter().copied())
            .collect();
        taps.take(name, width, cols);
    }
    Ok(())
}

impl Encoder {
    /// Read the file's hyperparameters and tensor table (both refused by name when they are not
    /// a `glm5v` encoder's), upload every weight — refused by name when the uploads are not
    /// [`card::weights`]' figure — and allocate the activations for images of at most
    /// `max_tokens` merged tokens; every later call runs on `stream`. Load-time only.
    pub fn load(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        file: &Gguf,
        max_tokens: usize,
    ) -> Result<Encoder, GpuError> {
        let hp = Hparams::read(file).map_err(|e| err(WHAT, e))?;
        tensors::check(
            &hp,
            file.iter_tensors()
                .map(|t| (t.name.as_str(), t.dims.as_slice(), t.ty)),
        )
        .map_err(|e| err(WHAT, e))?;
        if hp.dim / hp.n_head != HEAD || max_tokens == 0 {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!(
                    "heads of {} values, a limit of {max_tokens} tokens; the tower runs heads of {HEAD} and at least one token",
                    hp.dim / hp.n_head
                ),
            });
        }
        let narrow = |name: String| read_bf16_narrowed(file, &name, WHAT);
        let f32s = |name: String| up_f32(stream, file, &name, WHAT);
        let bf16 = |name: String| -> Result<DeviceBuffer<u16>, GpuError> {
            Ok(DeviceBuffer::from_host(stream, &narrow(name)?)?)
        };
        // The gate's values then the up's, as one matrix or vector of `2·rows`.
        let fused = |gate: String, up: String| -> Result<Vec<u16>, GpuError> {
            let mut v = narrow(gate)?;
            v.extend(narrow(up)?);
            Ok(v)
        };
        let mut blocks = Vec::with_capacity(hp.n_layer);
        for b in 0..hp.n_layer {
            let mut gu_b = read_f32(file, &names::ffn_gate_bias(b), WHAT)?;
            gu_b.extend(read_f32(file, &names::ffn_up_bias(b), WHAT)?);
            blocks.push(BlockWeights {
                ln1: f32s(names::ln1_weight(b))?,
                q_norm: f32s(names::attn_q_norm_weight(b))?,
                k_norm: f32s(names::attn_k_norm_weight(b))?,
                qkv_w: bf16(names::attn_qkv_weight(b))?,
                qkv_b: f32s(names::attn_qkv_bias(b))?,
                o_w: bf16(names::attn_out_weight(b))?,
                o_b: f32s(names::attn_out_bias(b))?,
                ln2: f32s(names::ln2_weight(b))?,
                gu_w: DeviceBuffer::from_host(
                    stream,
                    &fused(names::ffn_gate_weight(b), names::ffn_up_weight(b))?,
                )?,
                gu_b: DeviceBuffer::from_host(stream, &gu_b)?,
                down_w: bf16(names::ffn_down_weight(b))?,
                down_b: f32s(names::ffn_down_bias(b))?,
            });
        }
        // The two temporal kernels as one matrix: row `o` is the first kernel's, then the
        // second's, so the GEMM over a patch repeated over both frames is the sum of the convs.
        let (first, second) = (
            narrow(names::patch_embd_weight())?,
            narrow(names::patch_embd_weight_second_frame())?,
        );
        let k_half = 3 * hp.patch * hp.patch;
        let patch_w: Vec<u16> = first
            .chunks_exact(k_half)
            .zip(second.chunks_exact(k_half))
            .flat_map(|(a, b)| a.iter().chain(b).copied())
            .collect();
        let conv_w = reorder_merger_weight(
            &narrow(names::patch_merger_weight())?,
            hp.dim,
            hp.merge,
            hp.out_dim,
        );
        let w = Weights {
            patch_w: DeviceBuffer::from_host(stream, &patch_w)?,
            patch_b: f32s(names::patch_embd_bias())?,
            blocks,
            post_ln: f32s(names::post_ln_weight())?,
            conv_w: DeviceBuffer::from_host(stream, &conv_w)?,
            conv_b: f32s(names::patch_merger_bias())?,
            fc_w: bf16(names::fc_weight())?,
            post_norm: (
                f32s(names::post_norm_weight())?,
                f32s(names::post_norm_bias())?,
            ),
            mgu_w: DeviceBuffer::from_host(
                stream,
                &fused(names::mm_gate_weight(), names::mm_up_weight())?,
            )?,
            mdown_w: bf16(names::mm_down_weight())?,
        };
        let (uploaded, figure) = (w.uploaded(), card::weights(&hp));
        if uploaded != figure {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!(
                    "uploaded {uploaded} B of weights; the figure of the file's hyperparameters (card::weights) is {figure} B"
                ),
            });
        }
        let lens = ScratchLens::of(&hp, max_tokens);
        let s = Scratch::new(stream, lens)?;
        stream.synchronize()?;
        Ok(Encoder {
            chain: Chain {
                gemm: GemmKernels::load(ctx, stream)?,
                norm: NormKernels::load(ctx)?,
                layer_norm: LayerNormKernels::load(ctx)?,
                mlp: MlpKernels::load(ctx)?,
                gelu: GeluTanhKernels::load(ctx)?,
                rope: RopeKernels::load(ctx)?,
                attn: AttnKernels::load(ctx)?,
                hp,
                w,
            },
            s,
            weight_bytes: uploaded,
            scratch_bytes: lens.bytes(),
            stream: Arc::clone(stream),
        })
    }

    /// The file's hyperparameters.
    #[must_use]
    pub fn hparams(&self) -> &Hparams {
        &self.chain.hp
    }

    /// Bytes of weights on the card: what the load uploaded, held to [`card::weights`].
    #[must_use]
    pub fn weight_bytes(&self) -> u64 {
        self.weight_bytes
    }

    /// Bytes of activations on the card: the buffers of the largest image the limit allows.
    #[must_use]
    pub fn scratch_bytes(&self) -> u64 {
        self.scratch_bytes
    }

    /// Chain launches of one encode at `n_layer` blocks: the patch GEMM, eleven per block, the
    /// final norm and the merger's seven.
    #[must_use]
    pub const fn launches(n_layer: usize) -> usize {
        1 + 11 * n_layer + 1 + 7
    }

    /// Encode one image's patches. Eager: copies the patches into the tower's buffers, enqueues
    /// the chain on its stream and returns the rows without waiting for it; read them on that
    /// stream.
    pub fn encode(&mut self, patches: &Patches) -> Result<Encoded<'_>, GpuError> {
        self.encode_tapped(patches, &mut NoTaps)
    }

    /// [`Encoder::encode`] handing the named intermediate tensors to `taps` ([`TapSink`]).
    pub fn encode_tapped(
        &mut self,
        patches: &Patches,
        taps: &mut dyn TapSink,
    ) -> Result<Encoded<'_>, GpuError> {
        let what = "glm5v tower encode";
        let (chain, s, stream) = (&self.chain, &mut self.s, &*self.stream);
        let hp = &chain.hp;
        let (n_h, n_w) = (patches.n_vit_h, patches.n_vit_w);
        let n = n_h * n_w;
        let k_patch = TEMPORAL_FRAMES * 3 * hp.patch * hp.patch;
        if n == 0 || patches.patch_len != k_patch || patches.bf16.len() != n * k_patch {
            return Err(GpuError::Shape {
                what,
                detail: format!(
                    "{n_h}x{n_w} patches of {} values ({} in all); the tower takes {k_patch} per patch",
                    patches.patch_len,
                    patches.bf16.len()
                ),
            });
        }
        s.start(stream, hp, (n_h, n_w), what)?;
        WindowMut::<u16>::of_mut(&mut s.patches, 0, n * k_patch)?
            .copy_from_host(stream, &patches.bf16)?;
        linear(
            &chain.gemm,
            stream,
            (&s.patches, &chain.w.patch_w, Some(&chain.w.patch_b)),
            (n, hp.dim, k_patch),
            None,
            &mut s.xa,
        )?;
        tap(stream, taps, "embed", &s.xa, n, hp.dim)?;
        let mut launches = 1;
        for b in 0..hp.n_layer {
            launches += chain.block(stream, b, n, s, taps)?;
        }
        let (n_tok, tail) = chain.tail(stream, n, s, false, taps)?;
        tap(stream, taps, "embd", &s.rows, n_tok, hp.out_dim)?;
        Ok(Encoded {
            rows: DeviceTensor::window_of(&s.rows, 0, n_tok, hp.out_dim)?,
            launches: launches + tail,
        })
    }

    /// The chain after the final norm alone on the host rows `vit` (`n_h · n_w` rows of `dim`
    /// bf16, the `vit_out` tap): the merger and the projector MLP, a teacher-forced step for a
    /// gate that feeds the reference's `vit_out` to them. Returns the `after_fc_proj` rows and the
    /// output rows.
    pub fn forced_tail(
        &mut self,
        (n_h, n_w): (usize, usize),
        vit: &[u16],
    ) -> Result<(Vec<u16>, Vec<u16>), GpuError> {
        let what = "glm5v tower forced step";
        let (chain, s, stream) = (&self.chain, &mut self.s, &*self.stream);
        let hp = &chain.hp;
        let n = n_h * n_w;
        if n_h == 0 || n_w == 0 || vit.len() != n * hp.dim {
            return Err(GpuError::Shape {
                what,
                detail: format!("{} values for {n_h}x{n_w} rows of {}", vit.len(), hp.dim),
            });
        }
        s.start(stream, hp, (n_h, n_w), what)?;
        WindowMut::<u16>::of_mut(&mut s.h, 0, vit.len())?.copy_from_host(stream, vit)?;
        let (n_tok, _) = chain.tail(stream, n, s, true, &mut NoTaps)?;
        Ok((
            to_host(stream, &s.conv, n_tok * hp.out_dim)?,
            to_host(stream, &s.rows, n_tok * hp.out_dim)?,
        ))
    }
}

impl Scratch {
    /// The buffers at `lens`, zero-filled. Load-time only.
    fn new(stream: &CudaStream, lens: ScratchLens) -> Result<Scratch, GpuError> {
        let bf16 = |n: usize| DeviceBuffer::<u16>::zeroed(stream, n);
        Ok(Scratch {
            lens,
            patches: bf16(lens.patches)?,
            cs: DeviceBuffer::zeroed(stream, lens.cs)?,
            xa: bf16(lens.xa)?,
            xb: bf16(lens.xb)?,
            h: bf16(lens.h)?,
            qkv: bf16(lens.qkv)?,
            att: bf16(lens.att)?,
            gu: bf16(lens.gu)?,
            act: bf16(lens.act)?,
            conv: bf16(lens.conv)?,
            fc: bf16(lens.fc)?,
            mgu: bf16(lens.mgu)?,
            mact: bf16(lens.mact)?,
            rows: bf16(lens.rows)?,
        })
    }

    /// Start an `n_h × n_w` image: refuse a grid that is not whole merge groups or is larger
    /// than the buffers, then write its RoPE table.
    fn start(
        &mut self,
        stream: &CudaStream,
        hp: &Hparams,
        (n_h, n_w): (usize, usize),
        what: &'static str,
    ) -> Result<(), GpuError> {
        let n = n_h * n_w;
        if !n_h.is_multiple_of(hp.merge) || !n_w.is_multiple_of(hp.merge) {
            return Err(GpuError::Shape {
                what,
                detail: format!(
                    "a {n_h}x{n_w} grid is not whole {0}x{0} merge groups",
                    hp.merge
                ),
            });
        }
        if n > self.lens.max_patches {
            return Err(GpuError::Shape {
                what,
                detail: format!(
                    "a {n_h}x{n_w} grid ({n} patches, {} tokens) does not fit the tower's buffers, sized for {} tokens",
                    n / (hp.merge * hp.merge),
                    self.lens.max_tokens
                ),
            });
        }
        let table = ggml_rope_table((n_h, n_w), merge_order(n_h, n_w, hp.merge));
        WindowMut::<f32>::of_mut(&mut self.cs, 0, table.cs.len())?
            .copy_from_host(stream, &table.cs)?;
        Ok(())
    }
}

impl Chain {
    /// Block `b` of the chain on the first `n` rows of `s.xa`, its output back in `s.xa`; returns
    /// the launches.
    fn block(
        &self,
        stream: &CudaStream,
        b: usize,
        n: usize,
        s: &mut Scratch,
        taps: &mut dyn TapSink,
    ) -> Result<usize, GpuError> {
        let hp = &self.hp;
        let bw = &self.w.blocks[b];
        let (dim, heads, ff) = (hp.dim, hp.n_head, hp.ff);
        let lay = QkvLayout {
            row_width: 3 * dim,
            q0: 0,
            k0: dim,
            v0: 2 * dim,
            n_heads: heads,
        };
        let name = |op: &str| format!("blk{b}.{op}");
        self.norm
            .enqueue(stream, &s.xa, &bw.ln1, hp.eps, n, &mut s.h)?;
        tap(stream, taps, &name("norm1"), &s.h, n, dim)?;
        linear(
            &self.gemm,
            stream,
            (&s.h, &bw.qkv_w, Some(&bw.qkv_b)),
            (n, 3 * dim, dim),
            None,
            &mut s.qkv,
        )?;
        tap(stream, taps, &name("qkv"), &s.qkv, n, 3 * dim)?;
        for (op, c0, gain) in [("qnorm", lay.q0, &bw.q_norm), ("knorm", lay.k0, &bw.k_norm)] {
            self.norm.enqueue_heads(
                stream,
                HeadNormArgs {
                    x: &mut s.qkv,
                    gain,
                    eps: hp.eps,
                    rows: n,
                    row_width: lay.row_width,
                    col0: c0,
                    n_heads: heads,
                },
            )?;
            tap_cols(stream, taps, &name(op), &s.qkv, (n, 3 * dim), (c0, dim))?;
        }
        self.rope.enqueue(
            stream,
            RopeArgs {
                cs: &s.cs,
                n,
                row_width: lay.row_width,
                q0: lay.q0,
                k0: lay.k0,
                n_heads: heads,
                x: &mut s.qkv,
            },
        )?;
        for (op, c0) in [("qrot", lay.q0), ("krot", lay.k0)] {
            tap_cols(stream, taps, &name(op), &s.qkv, (n, 3 * dim), (c0, dim))?;
        }
        self.attn.enqueue(
            stream,
            AttnArgs {
                qkv: &s.qkv,
                layout: lay,
                n,
                scale: 1.0 / (HEAD as f32).sqrt(),
                out_width: dim,
                out: &mut s.att,
            },
        )?;
        tap(stream, taps, &name("sdpa"), &s.att, n, dim)?;
        side(
            &self.gemm,
            stream,
            taps,
            &name("attn"),
            (&s.att, &bw.o_w, Some(&bw.o_b)),
            (n, dim, dim),
            &mut s.xb,
        )?;
        linear(
            &self.gemm,
            stream,
            (&s.att, &bw.o_w, Some(&bw.o_b)),
            (n, dim, dim),
            Some(&s.xa),
            &mut s.xb,
        )?;
        tap(stream, taps, &name("resid1"), &s.xb, n, dim)?;
        self.norm
            .enqueue(stream, &s.xb, &bw.ln2, hp.eps, n, &mut s.h)?;
        tap(stream, taps, &name("norm2"), &s.h, n, dim)?;
        linear(
            &self.gemm,
            stream,
            (&s.h, &bw.gu_w, Some(&bw.gu_b)),
            (n, 2 * ff, dim),
            None,
            &mut s.gu,
        )?;
        tap(stream, taps, &name("gu"), &s.gu, n, 2 * ff)?;
        self.mlp
            .enqueue(stream, &s.gu, ff, n, Some(hp.swiglu_clamp), &mut s.act)?;
        tap(stream, taps, &name("act"), &s.act, n, ff)?;
        side(
            &self.gemm,
            stream,
            taps,
            &name("mlp"),
            (&s.act, &bw.down_w, Some(&bw.down_b)),
            (n, dim, ff),
            &mut s.xa,
        )?;
        linear(
            &self.gemm,
            stream,
            (&s.act, &bw.down_w, Some(&bw.down_b)),
            (n, dim, ff),
            Some(&s.xb),
            &mut s.xa,
        )?;
        tap(stream, taps, &format!("blk{b}"), &s.xa, n, dim)?;
        Ok(11)
    }

    /// The chain after the last block on `n` patches: the final norm of `s.xa` into `s.h`
    /// (skipped when `vit_given`: `s.h` holds it already), the merger — `s.h` re-read as rows of
    /// `merge²·dim` values — and the projector MLP into `s.rows`; returns the merged tokens and
    /// the launches.
    fn tail(
        &self,
        stream: &CudaStream,
        n: usize,
        s: &mut Scratch,
        vit_given: bool,
        taps: &mut dyn TapSink,
    ) -> Result<(usize, usize), GpuError> {
        let hp = &self.hp;
        let (dim, out) = (hp.dim, hp.out_dim);
        let merged = dim * hp.merge * hp.merge;
        let n_tok = n / (hp.merge * hp.merge);
        let mut launches = 7;
        if !vit_given {
            self.norm
                .enqueue(stream, &s.xa, &self.w.post_ln, hp.eps, n, &mut s.h)?;
            launches += 1;
        }
        tap(stream, taps, "vit_out", &s.h, n, dim)?;
        linear(
            &self.gemm,
            stream,
            (&s.h, &self.w.conv_w, Some(&self.w.conv_b)),
            (n_tok, out, merged),
            None,
            &mut s.conv,
        )?;
        tap(stream, taps, "merger.conv", &s.conv, n_tok, out)?;
        linear(
            &self.gemm,
            stream,
            (&s.conv, &self.w.fc_w, None),
            (n_tok, out, out),
            None,
            &mut s.fc,
        )?;
        tap(stream, taps, "merger.fc", &s.fc, n_tok, out)?;
        self.layer_norm.enqueue(
            stream,
            &s.fc,
            (&self.w.post_norm.0, &self.w.post_norm.1),
            MERGER_NORM_EPS,
            n_tok,
            &mut s.conv,
        )?;
        self.gelu
            .enqueue_kind(stream, GeluKind::Erf, n_tok * out, &mut s.conv)?;
        tap(stream, taps, "after_fc_proj", &s.conv, n_tok, out)?;
        linear(
            &self.gemm,
            stream,
            (&s.conv, &self.w.mgu_w, None),
            (n_tok, 2 * MERGER_FF, out),
            None,
            &mut s.mgu,
        )?;
        tap(stream, taps, "merger.gu", &s.mgu, n_tok, 2 * MERGER_FF)?;
        self.mlp.enqueue(
            stream,
            &s.mgu,
            MERGER_FF,
            n_tok,
            Some(hp.swiglu_clamp),
            &mut s.mact,
        )?;
        tap(stream, taps, "merger.act", &s.mact, n_tok, MERGER_FF)?;
        linear(
            &self.gemm,
            stream,
            (&s.mact, &self.w.mdown_w, None),
            (n_tok, out, MERGER_FF),
            None,
            &mut s.rows,
        )?;
        tap(stream, taps, "after_ffn_proj", &s.rows, n_tok, out)?;
        Ok((n_tok, launches))
    }
}
