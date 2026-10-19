//! The RADIO tower (`nemotron_omni`: Nemotron 3.5 Bolt) on the card: the weights of one mmproj
//! file uploaded once, the activations of the largest image its token limits allow allocated once
//! beside them, and [`Encoder::encode`], from an image's bf16 patches (merge order, one frame a
//! patch, [`vision::arch::radio::media`]) to the bf16 rows the text model reads in place of the
//! image's tokens.
//!
//! The chain, op by op, with the graph node of llama.cpp's `clip_graph_nemotron_omni` each launch
//! transcribes (a launch is one kernel; every op reads bf16, computes in f32 and rounds its
//! output to bf16, the contract of [`crate`]):
//!
//! | # | launch | graph node | tap |
//! |---|--------|------------|-----|
//! | 1 | position rows: the 128×128 table resized to the grid, half-pixel bilinear | `resize_position_embeddings` | `pos` |
//! | 2 | GEMM `patches · patch_embdᵀ`, the position rows as the residual operand (K = 3·16²) | `inp_pos` | `embed.patch` (without the residual), `embed` |
//! | | the ten prefix rows copied in after the patches | `class_embedding` | |
//! | | per block `b` of 32, over the patches and the prefix: | | |
//! | 3 | LayerNorm `ln1` | `layer_inp_normed` | `blk{b}.norm1` |
//! | 4 | GEMM `h · [wq; wk; wv]ᵀ + b` (1280 → 3840) | `Qcur`/`Kcur`/`Vcur` | `blk{b}.qkv` |
//! | 5 | attention, 16 heads of 80, non-causal, no rotary embedding | `build_attn` | `blk{b}.sdpa` |
//! | 6 | GEMM `a · woᵀ + b`, residual epilogue | `attn_out`, `ffn_inp` | `blk{b}.attn` (without the residual), `blk{b}.resid1` |
//! | 7 | LayerNorm `ln2` | `ffn_inp_normed` | `blk{b}.norm2` |
//! | 8 | GEMM `h · upᵀ + b` (1280 → 5120), erf GELU epilogue | | `blk{b}.up` (without the GELU), `blk{b}.act` |
//! | 9 | GEMM `act · downᵀ + b` (5120 → 1280), residual epilogue | `ffn_out`, `layer_out` | `blk{b}.mlp` (without the residual), `blk{b}` |
//! | | the prefix rows dropped (there is no final norm); the pixel shuffle is the re-read of the patches' rows as rows of 4·1280 | `pixel_shuffle` | `vit` |
//! | 10 | RMSNorm over the merged rows (gain `mm.model.mlp.0`, ε 1e-5) | | `merger.norm` |
//! | 11 | GEMM `h · mm1ᵀ` (5120 → 20480), no bias | | `merger.pre` |
//! | 12 | ReLU², in place | | `merger.h` |
//! | 13 | GEMM `h · mm3ᵀ` (20480 → out), no bias | `embd` | |
//!
//! Launches per image: `2 + 7·n_layer + 4` ([`Encoder::launches`]), 230 at 32 blocks. Eager, as
//! the other towers: the grid of every launch depends on the image's patch count.
//!
//! Row order. The host cuts the patches in merge order (the groups of 2×2 patches consecutive), so
//! the pixel shuffle is a view; attention has no rotary embedding, so the order of its keys moves
//! no value but the accumulation order. The stream is the `n` patches' rows followed by the ten
//! prefix rows, where llama.cpp has the prefix first and the patches in raster order: a tap holds
//! `n + 10` rows in this order and a gate maps them to the reference's. The merged token `t` is
//! the group `t` of the merge order, its four patches at `(y, x) = (2gy + dy, 2gx + dx)` as the
//! channel blocks `dx + 2·dy` — the order of `build_patch_merge_permute`.
//!
//! Weights as the card holds them: every matrix bf16 (an f32 tensor narrows only when every value
//! round-trips through bf16, else the load is refused by name); the three projections of a block
//! as one `[3·dim, dim]` matrix with their biases one vector; the position table and every vector
//! as f32; the prefix rows on the host. The video patch kernel is not loaded. The load refuses
//! uploads that differ from [`weight_bytes_of`]'s figure, the host's own ([`card::weights`]).
//!
//! Scratch is allocated once from the token limit ([`card::ScratchLens`]); an image allocates
//! nothing, and one past the limit is refused by name.

use crate::attn::{AttnArgs, QkvLayout};
use crate::attn_h80::{Attn80Kernels, HEAD_DIM_80};
use crate::chain::{
    Encoded, NoTaps, TapSink, read_bf16_narrowed, read_f32, side, tap, to_host, up_f32,
};
use crate::gemm_bf16::{Epilogue, GemmArgs, GemmKernels};
use crate::layer_norm::LayerNormKernels;
use crate::norm::NormKernels;
use crate::pos_half::{MERGE, PosHalfKernels};
use crate::relu_sq::ReluSqKernels;
use bloomery_gpu::{DeviceTensor, GpuError, WindowMut};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use gguf::Gguf;
use std::mem::size_of;
use std::sync::Arc;
use vision::Patches;
use vision::arch::TokenLimits;
use vision::arch::radio::{
    Hparams, POS_SIDE, PREFIX_ROWS, PROJECTOR_HIDDEN, card, hparams, names, tensors,
};

const WHAT: &str = "RADIO tower load";

/// The RMS norm epsilon of the projector: a constant of llama.cpp's graph (`build_norm(…, 1e-5)`),
/// not a key of the file.
const MLP_NORM_EPS: f32 = 1e-5;

const _: () = assert!(hparams::TILE_N == crate::gemm_bf16::TILE_N as u64);

/// Bytes of weights the card holds for a file with these hyperparameters: the host's figure
/// ([`card::weights`]), which the load holds the uploads to.
#[must_use]
pub fn weight_bytes_of(hp: &Hparams) -> u64 {
    card::weights(hp)
}

/// Bytes of activations the card holds for images of at most `limits`' token count.
#[must_use]
pub fn scratch_bytes_of(hp: &Hparams, limits: TokenLimits) -> u64 {
    card::ScratchLens::of(hp, limits).bytes()
}

/// One block's weights on the card.
struct BlockWeights {
    ln1: (DeviceBuffer<f32>, DeviceBuffer<f32>),
    /// `[3·dim, dim]`: the query rows, then the key rows, then the value rows.
    qkv_w: DeviceBuffer<u16>,
    qkv_b: DeviceBuffer<f32>,
    o_w: DeviceBuffer<u16>,
    o_b: DeviceBuffer<f32>,
    ln2: (DeviceBuffer<f32>, DeviceBuffer<f32>),
    up_w: DeviceBuffer<u16>,
    up_b: DeviceBuffer<f32>,
    down_w: DeviceBuffer<u16>,
    down_b: DeviceBuffer<f32>,
}

/// Every weight of the tower on the card.
struct Weights {
    /// `[dim, 3·p²]`: the patch kernel as a linear weight, no bias.
    patch_w: DeviceBuffer<u16>,
    pos: DeviceBuffer<f32>,
    blocks: Vec<BlockWeights>,
    mlp_norm: DeviceBuffer<f32>,
    mlp_up: DeviceBuffer<u16>,
    mlp_down: DeviceBuffer<u16>,
}

impl Weights {
    /// Bytes uploaded: two per bf16, four per f32.
    fn uploaded(&self) -> u64 {
        let pair = |p: &(DeviceBuffer<f32>, DeviceBuffer<f32>)| p.0.len() + p.1.len();
        let blocks: usize = self
            .blocks
            .iter()
            .map(|b| {
                2 * (b.qkv_w.len() + b.o_w.len() + b.up_w.len() + b.down_w.len())
                    + 4 * (pair(&b.ln1)
                        + pair(&b.ln2)
                        + b.qkv_b.len()
                        + b.o_b.len()
                        + b.up_b.len()
                        + b.down_b.len())
            })
            .sum();
        (blocks
            + 2 * (self.patch_w.len() + self.mlp_up.len() + self.mlp_down.len())
            + 4 * (self.pos.len() + self.mlp_norm.len())) as u64
    }
}

/// What an encode reads and never writes: the hyperparameters, the kernels and the weights.
struct Chain {
    hp: Hparams,
    gemm: GemmKernels,
    norm: LayerNormKernels,
    rms: NormKernels,
    pos: PosHalfKernels,
    attn: Attn80Kernels,
    relu: ReluSqKernels,
    w: Weights,
    /// `v.class_embd` as bf16, [`PREFIX_ROWS`] rows of `dim`, copied behind each image's patches.
    prefix: Vec<u16>,
}

/// The activations of the largest image the limits allow, allocated at load. An image of `n`
/// patches uses the first rows of each buffer: its patches and position rows are written first,
/// and every launch reads only rows that a copy or an earlier launch of the same image wrote, so
/// nothing a larger image left behind is read.
struct Scratch {
    lens: card::ScratchLens,
    patches: DeviceBuffer<u16>,
    pos: DeviceBuffer<u16>,
    /// The block input and output, the attention-residual stream, and the normed rows.
    xa: DeviceBuffer<u16>,
    xb: DeviceBuffer<u16>,
    h: DeviceBuffer<u16>,
    qkv: DeviceBuffer<u16>,
    att: DeviceBuffer<u16>,
    /// The up projection, and the GELU of it.
    up: DeviceBuffer<u16>,
    /// The projector's hidden rows.
    hid: DeviceBuffer<u16>,
    rows: DeviceBuffer<u16>,
}

/// The loaded tower: the chain of one file, the activations of the largest image its limits
/// allow, its bytes on the card, and the stream every call runs on. A call takes `&mut self`: one
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

impl Encoder {
    /// Read the file's hyperparameters and tensor table (both refused by name when they are not
    /// a `nemotron_omni` encoder's), upload every weight — refused by name when the uploads are
    /// not [`weight_bytes_of`]'s figure — and allocate the activations for images of at most
    /// `limits`' token count; every later call runs on `stream`. Load-time only.
    pub fn load(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        file: &Gguf,
        limits: TokenLimits,
    ) -> Result<Encoder, GpuError> {
        let hp = Hparams::read(file).map_err(|e| err(WHAT, e))?;
        tensors::check(
            &hp,
            file.iter_tensors()
                .map(|t| (t.name.as_str(), t.dims.as_slice(), t.ty)),
        )
        .map_err(|e| err(WHAT, e))?;
        if hp.dim / hp.n_head != HEAD_DIM_80
            || hp.merge != MERGE
            || hp.out_dim % crate::gemm_bf16::TILE_N != 0
            || hp.ff % crate::gemm_bf16::TILE_N != 0
        {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!(
                    "heads of {} values, merge {}, ff {}, out {}; the tower runs heads of {HEAD_DIM_80}, merge {MERGE} and widths that are multiples of {}",
                    hp.dim / hp.n_head,
                    hp.merge,
                    hp.ff,
                    hp.out_dim,
                    crate::gemm_bf16::TILE_N
                ),
            });
        }
        let narrow = |name: String| read_bf16_narrowed(file, &name, WHAT);
        let f32s = |name: String| up_f32(stream, file, &name, WHAT);
        let bf16 = |name: String| -> Result<DeviceBuffer<u16>, GpuError> {
            Ok(DeviceBuffer::from_host(stream, &narrow(name)?)?)
        };
        let mut blocks = Vec::with_capacity(hp.n_layer);
        for b in 0..hp.n_layer {
            let mut qkv_w = narrow(names::attn_q_weight(b))?;
            qkv_w.extend(narrow(names::attn_k_weight(b))?);
            qkv_w.extend(narrow(names::attn_v_weight(b))?);
            let mut qkv_b = read_f32(file, &names::attn_q_bias(b), WHAT)?;
            qkv_b.extend(read_f32(file, &names::attn_k_bias(b), WHAT)?);
            qkv_b.extend(read_f32(file, &names::attn_v_bias(b), WHAT)?);
            blocks.push(BlockWeights {
                ln1: (f32s(names::ln1_weight(b))?, f32s(names::ln1_bias(b))?),
                qkv_w: DeviceBuffer::from_host(stream, &qkv_w)?,
                qkv_b: DeviceBuffer::from_host(stream, &qkv_b)?,
                o_w: bf16(names::attn_out_weight(b))?,
                o_b: f32s(names::attn_out_bias(b))?,
                ln2: (f32s(names::ln2_weight(b))?, f32s(names::ln2_bias(b))?),
                up_w: bf16(names::ffn_up_weight(b))?,
                up_b: f32s(names::ffn_up_bias(b))?,
                down_w: bf16(names::ffn_down_weight(b))?,
                down_b: f32s(names::ffn_down_bias(b))?,
            });
        }
        let w = Weights {
            patch_w: bf16(names::patch_embd_weight())?,
            pos: f32s(names::position_embd())?,
            blocks,
            mlp_norm: f32s(names::mlp_norm())?,
            mlp_up: bf16(names::mlp_up())?,
            mlp_down: bf16(names::mlp_down())?,
        };
        let (uploaded, figure) = (w.uploaded(), weight_bytes_of(&hp));
        if uploaded != figure {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!(
                    "uploaded {uploaded} B of weights; the figure of the file's hyperparameters (weight_bytes_of) is {figure} B"
                ),
            });
        }
        let lens = card::ScratchLens::of(&hp, limits);
        let s = Scratch::new(stream, lens)?;
        stream.synchronize()?;
        Ok(Encoder {
            chain: Chain {
                gemm: GemmKernels::load(ctx, stream)?,
                norm: LayerNormKernels::load(ctx)?,
                rms: NormKernels::load(ctx)?,
                pos: PosHalfKernels::load(ctx)?,
                attn: Attn80Kernels::load(ctx)?,
                relu: ReluSqKernels::load(ctx)?,
                prefix: narrow(names::class_embd())?,
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

    /// Bytes of weights on the card: what the load uploaded, held to [`weight_bytes_of`].
    #[must_use]
    pub fn weight_bytes(&self) -> u64 {
        self.weight_bytes
    }

    /// Bytes of activations on the card: the buffers of the largest image the limits allow.
    #[must_use]
    pub fn scratch_bytes(&self) -> u64 {
        self.scratch_bytes
    }

    /// Chain launches of one encode at `n_layer` blocks: the position rows, the patch GEMM,
    /// seven per block and the projector's four.
    #[must_use]
    pub const fn launches(n_layer: usize) -> usize {
        2 + 7 * n_layer + 4
    }

    /// Encode one image's patches. Eager: copies the patches into the tower's buffers, enqueues
    /// the chain on its stream and returns the rows without waiting for it; read them on that
    /// stream.
    pub fn encode(&mut self, patches: &Patches) -> Result<Encoded<'_>, GpuError> {
        self.encode_tapped(patches, &mut NoTaps)
    }

    /// [`Encoder::encode`] handing the named intermediate tensors to `taps` ([`TapSink`]). A
    /// block's taps hold `n + PREFIX_ROWS` rows (module doc); `pos`, `embed.patch`, `embed` and
    /// `vit` hold the `n` patches' rows.
    pub fn encode_tapped(
        &mut self,
        patches: &Patches,
        taps: &mut dyn TapSink,
    ) -> Result<Encoded<'_>, GpuError> {
        let what = "RADIO tower encode";
        let (chain, s, stream) = (&self.chain, &mut self.s, &*self.stream);
        let hp = &chain.hp;
        let (n_h, n_w) = (patches.n_vit_h, patches.n_vit_w);
        let n = n_h * n_w;
        let k_patch = 3 * hp.patch * hp.patch;
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
        s.start(hp, (n_h, n_w), what)?;
        WindowMut::<u16>::of_mut(&mut s.patches, 0, n * k_patch)?
            .copy_from_host(stream, &patches.bf16)?;
        chain.pos.enqueue(
            stream,
            &chain.w.pos,
            (POS_SIDE, hp.dim),
            (n_h, n_w),
            &mut s.pos,
        )?;
        tap(stream, taps, "pos", &s.pos, n, hp.dim)?;
        side(
            &chain.gemm,
            stream,
            taps,
            "embed.patch",
            (&s.patches, &chain.w.patch_w, None),
            (n, hp.dim, k_patch),
            &mut s.xb,
        )?;
        chain.gemm.enqueue(
            stream,
            GemmArgs {
                a: &s.patches,
                b: &chain.w.patch_w,
                bias: None,
                epilogue: Epilogue::Residual,
                resid: Some(&s.pos),
                m: n,
                n: hp.dim,
                k: k_patch,
                c: &mut s.xa,
            },
        )?;
        let mut launches = 2;
        tap(stream, taps, "embed", &s.xa, n, hp.dim)?;
        // the window's offset is in bytes
        let behind_patches = n * hp.dim * size_of::<u16>();
        WindowMut::<u16>::of_mut(&mut s.xa, behind_patches, PREFIX_ROWS * hp.dim)?
            .copy_from_host(stream, &chain.prefix)?;
        let rows = n + PREFIX_ROWS;
        for b in 0..hp.n_layer {
            launches += chain.block(stream, b, rows, s, taps)?;
        }
        let (n_tok, tail) = chain.tail(stream, n, s, taps)?;
        Ok(Encoded {
            rows: DeviceTensor::window_of(&s.rows, 0, n_tok, hp.out_dim)?,
            launches: launches + tail,
        })
    }

    /// One block alone on the host rows `x` (`n_h · n_w + PREFIX_ROWS` rows of `dim` bf16, the
    /// module doc's order): the chain's launches of block `b` from `x` and the block's output — a
    /// teacher-forced step, for a gate that feeds the reference's input to a block.
    pub fn forced_block(
        &mut self,
        (n_h, n_w): (usize, usize),
        b: usize,
        x: &[u16],
    ) -> Result<Vec<u16>, GpuError> {
        self.forced_input((n_h, n_w), x)?;
        let (chain, s, stream) = (&self.chain, &mut self.s, &*self.stream);
        let rows = n_h * n_w + PREFIX_ROWS;
        chain.block(stream, b, rows, s, &mut NoTaps)?;
        to_host(stream, &s.xa, rows * chain.hp.dim)
    }

    /// The chain after the last block alone on the host rows `x` (the stream after block
    /// `n_layer − 1`, the module doc's order): the projector's norm and rows, teacher-forced as
    /// [`Encoder::forced_block`]. Returns the normed merged rows and the output rows.
    pub fn forced_tail(
        &mut self,
        (n_h, n_w): (usize, usize),
        x: &[u16],
    ) -> Result<(Vec<u16>, Vec<u16>), GpuError> {
        self.forced_input((n_h, n_w), x)?;
        let (chain, s, stream) = (&self.chain, &mut self.s, &*self.stream);
        let n = n_h * n_w;
        let (n_tok, _) = chain.tail(stream, n, s, &mut NoTaps)?;
        Ok((
            to_host(stream, &s.h, n * chain.hp.dim)?,
            to_host(stream, &s.rows, n_tok * chain.hp.out_dim)?,
        ))
    }

    /// Start an `n_h × n_w` image whose host rows `x` are the stream `xa`.
    fn forced_input(&mut self, (n_h, n_w): (usize, usize), x: &[u16]) -> Result<(), GpuError> {
        let what = "RADIO tower forced step";
        let (hp, s, stream) = (&self.chain.hp, &mut self.s, &*self.stream);
        let rows = n_h * n_w + PREFIX_ROWS;
        if n_h == 0 || n_w == 0 || x.len() != rows * hp.dim {
            return Err(GpuError::Shape {
                what,
                detail: format!(
                    "{} values for {n_h}x{n_w} patches and {PREFIX_ROWS} prefix rows of {}",
                    x.len(),
                    hp.dim
                ),
            });
        }
        s.start(hp, (n_h, n_w), what)?;
        WindowMut::<u16>::of_mut(&mut s.xa, 0, x.len())?.copy_from_host(stream, x)?;
        Ok(())
    }
}

impl Scratch {
    /// The buffers at `lens`, zero-filled. Load-time only.
    fn new(stream: &CudaStream, lens: card::ScratchLens) -> Result<Scratch, GpuError> {
        let bf16 = |n: usize| DeviceBuffer::<u16>::zeroed(stream, n);
        Ok(Scratch {
            lens,
            patches: bf16(lens.patches)?,
            pos: bf16(lens.pos)?,
            xa: bf16(lens.stream)?,
            xb: bf16(lens.stream)?,
            h: bf16(lens.stream)?,
            qkv: bf16(lens.qkv)?,
            att: bf16(lens.att)?,
            up: bf16(lens.up)?,
            hid: bf16(lens.hidden)?,
            rows: bf16(lens.rows)?,
        })
    }

    /// Start an `n_h × n_w` image: refuse a grid that is not whole merge groups or is larger than
    /// the buffers.
    fn start(
        &self,
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
        Ok(())
    }
}

impl Chain {
    /// Block `b` of the chain on the first `rows` rows of `s.xa`, its output back in `s.xa`;
    /// returns the launches.
    fn block(
        &self,
        stream: &CudaStream,
        b: usize,
        rows: usize,
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
        let scale = 1.0 / ((dim / heads) as f32).sqrt();
        let name = |op: &str| format!("blk{b}.{op}");
        self.norm.enqueue(
            stream,
            &s.xa,
            (&bw.ln1.0, &bw.ln1.1),
            hp.eps,
            rows,
            &mut s.h,
        )?;
        tap(stream, taps, &name("norm1"), &s.h, rows, dim)?;
        self.gemm.enqueue(
            stream,
            GemmArgs {
                a: &s.h,
                b: &bw.qkv_w,
                bias: Some(&bw.qkv_b),
                epilogue: Epilogue::None,
                resid: None,
                m: rows,
                n: 3 * dim,
                k: dim,
                c: &mut s.qkv,
            },
        )?;
        tap(stream, taps, &name("qkv"), &s.qkv, rows, 3 * dim)?;
        self.attn.enqueue(
            stream,
            AttnArgs {
                qkv: &s.qkv,
                layout: lay,
                n: rows,
                scale,
                out_width: dim,
                out: &mut s.att,
            },
        )?;
        tap(stream, taps, &name("sdpa"), &s.att, rows, dim)?;
        side(
            &self.gemm,
            stream,
            taps,
            &name("attn"),
            (&s.att, &bw.o_w, Some(&bw.o_b)),
            (rows, dim, dim),
            &mut s.xb,
        )?;
        self.gemm.enqueue(
            stream,
            GemmArgs {
                a: &s.att,
                b: &bw.o_w,
                bias: Some(&bw.o_b),
                epilogue: Epilogue::Residual,
                resid: Some(&s.xa),
                m: rows,
                n: dim,
                k: dim,
                c: &mut s.xb,
            },
        )?;
        tap(stream, taps, &name("resid1"), &s.xb, rows, dim)?;
        self.norm.enqueue(
            stream,
            &s.xb,
            (&bw.ln2.0, &bw.ln2.1),
            hp.eps,
            rows,
            &mut s.h,
        )?;
        tap(stream, taps, &name("norm2"), &s.h, rows, dim)?;
        side(
            &self.gemm,
            stream,
            taps,
            &name("up"),
            (&s.h, &bw.up_w, Some(&bw.up_b)),
            (rows, ff, dim),
            &mut s.up,
        )?;
        self.gemm.enqueue(
            stream,
            GemmArgs {
                a: &s.h,
                b: &bw.up_w,
                bias: Some(&bw.up_b),
                epilogue: Epilogue::Gelu,
                resid: None,
                m: rows,
                n: ff,
                k: dim,
                c: &mut s.up,
            },
        )?;
        tap(stream, taps, &name("act"), &s.up, rows, ff)?;
        side(
            &self.gemm,
            stream,
            taps,
            &name("mlp"),
            (&s.up, &bw.down_w, Some(&bw.down_b)),
            (rows, dim, ff),
            &mut s.xa,
        )?;
        self.gemm.enqueue(
            stream,
            GemmArgs {
                a: &s.up,
                b: &bw.down_w,
                bias: Some(&bw.down_b),
                epilogue: Epilogue::Residual,
                resid: Some(&s.xb),
                m: rows,
                n: dim,
                k: ff,
                c: &mut s.xa,
            },
        )?;
        tap(stream, taps, &format!("blk{b}"), &s.xa, rows, dim)?;
        Ok(7)
    }

    /// The chain after the last block on `n` patches: the projector's norm of the patches' rows
    /// of `s.xa`, re-read as rows of `merge²·dim` values, into `s.h`, and its two GEMMs into
    /// `s.rows`; returns the merged tokens and the launches.
    fn tail(
        &self,
        stream: &CudaStream,
        n: usize,
        s: &mut Scratch,
        taps: &mut dyn TapSink,
    ) -> Result<(usize, usize), GpuError> {
        let hp = &self.hp;
        let merged = hp.dim * hp.merge * hp.merge;
        let n_tok = n / (hp.merge * hp.merge);
        tap(stream, taps, "vit", &s.xa, n, hp.dim)?;
        self.rms.enqueue(
            stream,
            &s.xa,
            &self.w.mlp_norm,
            MLP_NORM_EPS,
            n_tok,
            &mut s.h,
        )?;
        tap(stream, taps, "merger.norm", &s.h, n_tok, merged)?;
        self.gemm.enqueue(
            stream,
            GemmArgs {
                a: &s.h,
                b: &self.w.mlp_up,
                bias: None,
                epilogue: Epilogue::None,
                resid: None,
                m: n_tok,
                n: PROJECTOR_HIDDEN,
                k: merged,
                c: &mut s.hid,
            },
        )?;
        tap(stream, taps, "merger.pre", &s.hid, n_tok, PROJECTOR_HIDDEN)?;
        self.relu
            .enqueue(stream, n_tok * PROJECTOR_HIDDEN, &mut s.hid)?;
        tap(stream, taps, "merger.h", &s.hid, n_tok, PROJECTOR_HIDDEN)?;
        self.gemm.enqueue(
            stream,
            GemmArgs {
                a: &s.hid,
                b: &self.w.mlp_down,
                bias: None,
                epilogue: Epilogue::None,
                resid: None,
                m: n_tok,
                n: hp.out_dim,
                k: PROJECTOR_HIDDEN,
                c: &mut s.rows,
            },
        )?;
        Ok((n_tok, 4))
    }
}
