//! The Qwen3-VL tower (`qwen3vl_merger`: Clef-Flash, Qwen3.6-35B-A3B, Qwen3.8-Flash-Next) on the
//! card: the weights of one mmproj file uploaded once, the activations of the largest image its
//! token limits allow allocated once beside them, and [`Encoder::encode`], from an image's bf16
//! patches (merge order, each patch repeated over the two temporal frames,
//! [`vision::arch::qwen3vl::media`]) to the bf16 rows the text model reads in place of the
//! image's tokens.
//!
//! The chain, op by op, with the graph node of llama.cpp's `clip_graph_qwen3vl` each launch
//! transcribes (a launch is one kernel; every op reads bf16, computes in f32 and rounds its
//! output to bf16, the contract of [`crate`]):
//!
//! | # | launch | graph node | tap |
//! |---|---|---|---|
//! | 1 | position rows: the 48×48 table resized to the grid | `resize_position_embeddings` | `pos` |
//! | 2 | GEMM `patches · patch_embdᵀ + b`, the position rows as the residual operand (K = 2·768) | `inp_pos_emb` | `embed.patch` (without the residual), `embed` |
//! | | per block `b` of 27: | | |
//! | 3 | LayerNorm `ln1` | `ln1` | `blk{b}.norm1` |
//! | 4 | GEMM `h · wqkvᵀ + b` (1152 → 3456) | `Qcur`/`Kcur`/`Vcur` | `blk{b}.qkv` |
//! | 5 | vision RoPE of q and k in place, heads of 72 | `Qcur_rope` | `blk{b}.qrot`, `.krot` |
//! | 6 | attention, 16 heads of 72, non-causal | `build_attn` | `blk{b}.sdpa` |
//! | 7 | GEMM `a · woᵀ + b`, residual epilogue | `attn_out`, `ffn_inp` | `blk{b}.attn` (without the residual), `blk{b}.resid1` |
//! | 8 | LayerNorm `ln2` | `ffn_inp_normed` | `blk{b}.norm2` |
//! | 9 | GEMM `h · upᵀ + b` (1152 → 4352, the 4304 rows padded) | | `blk{b}.up` |
//! | 10 | GELU (tanh), in place | | `blk{b}.act` |
//! | 11 | GEMM `act · downᵀ + b` (4352 → 1152), residual epilogue | `ffn_out`, `layer_out` | `blk{b}.mlp` (without the residual), `blk{b}` |
//! | 12 | LayerNorm `post_ln` | `norm_b-27` | `vit` |
//! | 13 | GEMM `[n/4, 4608] · mm0ᵀ + b`, the merge a re-read of the rows | | |
//! | 14 | GELU (tanh), in place | | `merger.h` |
//! | 15 | GEMM `h · mm2ᵀ + b` (4608 → out) | `embd` | |
//!
//! Launches per image: `2 + 9·n_layer + 4` ([`Encoder::launches`]), 249 at 27 blocks. Eager, as
//! V4.1's: the grid of every launch depends on the image's patch count.
//!
//! Weights as the card holds them: every matrix bf16 (an f32 or f16 tensor narrows only when every
//! value round-trips through bf16, else the load is refused by name); the two temporal patch
//! kernels as the halves of one `[dim, 2·3·p²]` matrix; `ffn_up`'s 4304 rows and its bias padded
//! with zeros to 4352 and each `ffn_down` row padded the same way, so the GEMMs' column tiles
//! fit — a padded column is `gelu(0) = 0` and adds nothing to the down projection; the position
//! table and every vector as f32. The load refuses uploads that differ from
//! [`weight_bytes_of`]'s figure.
//!
//! Scratch is allocated once from the token limit ([`scratch_bytes_of`]); an image allocates
//! nothing, and one past the limit is refused by name.

use crate::attn::{Attn72Kernels, AttnArgs, QkvLayout};
use crate::chain::{
    Encoded, NoTaps, TapSink, read_bf16_narrowed, read_f32, side, tap, to_host, up_f32,
};
use crate::gelu_tanh::GeluTanhKernels;
use crate::gemm_bf16::{Epilogue, GemmArgs, GemmKernels};
use crate::layer_norm::LayerNormKernels;
use crate::pos_bilinear::{MERGE, PosBilinearKernels};
use crate::rope2d::{HEAD_DIM_72, PAIRS_72, Rope72Kernels, RopeArgs, RopeTable72, merge_order};
use bloomery_gpu::{DeviceTensor, GpuError, WindowMut};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use gguf::Gguf;
use std::sync::Arc;
use vision::Patches;
use vision::arch::qwen3vl::{Hparams, TEMPORAL_FRAMES, TokenLimits, names, tensors};

const WHAT: &str = "Qwen3-VL tower load";

/// The learned position table's side: `v.position_embd.weight` has `POS_SIDE²` rows.
const POS_SIDE: usize = 48;
/// The RoPE base of this tower's vision RoPE: a constant of llama.cpp's graph
/// (`ggml_rope_multi(…, 10000, …)`), not a key of the file.
const ROPE_BASE: f32 = 10_000.0;

/// The columns `ffn_up` has after its zero padding: a multiple of the GEMM's column tile.
fn ff_padded(hp: &Hparams) -> usize {
    hp.ff.next_multiple_of(crate::gemm_bf16::TILE_N)
}

/// Bytes of weights the card holds for a file with these hyperparameters: bf16 matrices
/// (the ffn padded), the f32 position table and the f32 vectors.
#[must_use]
pub fn weight_bytes_of(hp: &Hparams) -> u64 {
    let d = |n: usize| n as u64;
    let (dim, ff, out) = (d(hp.dim), d(ff_padded(hp)), d(hp.out_dim));
    let merged = dim * d(hp.merge * hp.merge);
    let k_patch = d(TEMPORAL_FRAMES * 3 * hp.patch * hp.patch);
    let block_mats = 3 * dim * dim + dim * dim + 2 * ff * dim;
    let block_vecs = 4 * dim + 3 * dim + dim + ff + dim;
    2 * (d(hp.n_layer) * block_mats + dim * k_patch + merged * merged + merged * out)
        + 4 * (d(hp.n_layer) * block_vecs
            + dim
            + (POS_SIDE * POS_SIDE) as u64 * dim
            + 2 * dim
            + merged
            + out)
}

/// The activation buffers' lengths for an image of at most `max_tokens` merged tokens, in values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Lens {
    patches: usize,
    pos: usize,
    cs: usize,
    stream: usize,
    qkv: usize,
    att: usize,
    up: usize,
    hidden: usize,
    rows: usize,
}

impl Lens {
    fn of(hp: &Hparams, max_tokens: usize) -> Lens {
        let n = max_tokens * hp.merge * hp.merge;
        let merged = hp.dim * hp.merge * hp.merge;
        Lens {
            patches: n * TEMPORAL_FRAMES * 3 * hp.patch * hp.patch,
            pos: n * hp.dim,
            cs: n * 2 * PAIRS_72,
            stream: n * hp.dim,
            qkv: n * 3 * hp.dim,
            att: n * hp.dim,
            up: n * ff_padded(hp),
            hidden: max_tokens * merged,
            rows: max_tokens * hp.out_dim,
        }
    }

    /// Bytes of every buffer: the bf16 ones (patches, position rows, the three streams, qkv,
    /// attention, up, the merger's hidden rows, the output rows) and the f32 RoPE table.
    fn bytes(&self) -> u64 {
        let bf16 = self.patches
            + self.pos
            + 3 * self.stream
            + self.qkv
            + self.att
            + self.up
            + self.hidden
            + self.rows;
        (2 * bf16 + 4 * self.cs) as u64
    }
}

/// Bytes of activations the card holds for an image of at most `max_tokens` merged tokens.
#[must_use]
pub fn scratch_bytes_of(hp: &Hparams, max_tokens: usize) -> u64 {
    Lens::of(hp, max_tokens).bytes()
}

/// One block's weights on the card.
struct BlockWeights {
    ln1: (DeviceBuffer<f32>, DeviceBuffer<f32>),
    qkv_w: DeviceBuffer<u16>,
    qkv_b: DeviceBuffer<f32>,
    o_w: DeviceBuffer<u16>,
    o_b: DeviceBuffer<f32>,
    ln2: (DeviceBuffer<f32>, DeviceBuffer<f32>),
    /// `ffn_up`'s rows, zero-padded to [`ff_padded`].
    up_w: DeviceBuffer<u16>,
    up_b: DeviceBuffer<f32>,
    /// `ffn_down`'s rows, each zero-padded to [`ff_padded`] columns.
    down_w: DeviceBuffer<u16>,
    down_b: DeviceBuffer<f32>,
}

/// Every weight of the tower on the card.
struct Weights {
    /// `[dim, 2·3·p²]`: row `o` is the first temporal kernel's row `o`, then the second's.
    patch_w: DeviceBuffer<u16>,
    patch_b: DeviceBuffer<f32>,
    pos: DeviceBuffer<f32>,
    blocks: Vec<BlockWeights>,
    post_ln: (DeviceBuffer<f32>, DeviceBuffer<f32>),
    mm0_w: DeviceBuffer<u16>,
    mm0_b: DeviceBuffer<f32>,
    mm2_w: DeviceBuffer<u16>,
    mm2_b: DeviceBuffer<f32>,
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
            + 2 * (self.patch_w.len() + self.mm0_w.len() + self.mm2_w.len())
            + 4 * (self.patch_b.len()
                + self.pos.len()
                + pair(&self.post_ln)
                + self.mm0_b.len()
                + self.mm2_b.len())) as u64
    }
}

/// What an encode reads and never writes: the hyperparameters, the kernels and the weights.
struct Chain {
    hp: Hparams,
    gemm: GemmKernels,
    norm: LayerNormKernels,
    gelu: GeluTanhKernels,
    pos: PosBilinearKernels,
    rope: Rope72Kernels,
    attn: Attn72Kernels,
    w: Weights,
}

/// The activations of the largest image the limits allow, allocated at load. An image of `n`
/// patches uses the first `n` rows of each buffer: its patches, position rows and RoPE table are
/// written first, and every launch reads only rows that a copy or an earlier launch of the same
/// image wrote, so nothing a larger image left behind is read.
struct Scratch {
    lens: Lens,
    patches: DeviceBuffer<u16>,
    pos: DeviceBuffer<u16>,
    cs: DeviceBuffer<f32>,
    /// The block input and output, the attention-residual stream, and the normed rows.
    xa: DeviceBuffer<u16>,
    xb: DeviceBuffer<u16>,
    h: DeviceBuffer<u16>,
    qkv: DeviceBuffer<u16>,
    att: DeviceBuffer<u16>,
    /// The up projection, and the GELU of it in place.
    up: DeviceBuffer<u16>,
    /// The merger's hidden rows.
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
    /// a `qwen3vl_merger` encoder's), upload every weight — refused by name when the uploads are
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
        if hp.dim / hp.n_head != HEAD_DIM_72
            || hp.merge != MERGE
            || hp.out_dim % crate::gemm_bf16::TILE_N != 0
        {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!(
                    "heads of {} values, merge {}, out {}; the tower runs heads of {HEAD_DIM_72}, merge {MERGE} and an output width that is a multiple of {}",
                    hp.dim / hp.n_head,
                    hp.merge,
                    hp.out_dim,
                    crate::gemm_bf16::TILE_N
                ),
            });
        }
        let ff = ff_padded(&hp);
        let narrow = |name: String| read_bf16_narrowed(file, &name, WHAT);
        let f32s = |name: String| up_f32(stream, file, &name, WHAT);
        let bf16 = |name: String| -> Result<DeviceBuffer<u16>, GpuError> {
            Ok(DeviceBuffer::from_host(stream, &narrow(name)?)?)
        };
        let mut blocks = Vec::with_capacity(hp.n_layer);
        for b in 0..hp.n_layer {
            let mut up_w = narrow(names::ffn_up_weight(b))?;
            up_w.resize(ff * hp.dim, 0);
            let mut up_b = read_f32(file, &names::ffn_up_bias(b), WHAT)?;
            up_b.resize(ff, 0.0);
            let down_w: Vec<u16> = narrow(names::ffn_down_weight(b))?
                .chunks_exact(hp.ff)
                .flat_map(|row| {
                    row.iter()
                        .copied()
                        .chain(std::iter::repeat_n(0, ff - hp.ff))
                })
                .collect();
            blocks.push(BlockWeights {
                ln1: (f32s(names::ln1_weight(b))?, f32s(names::ln1_bias(b))?),
                qkv_w: bf16(names::attn_qkv_weight(b))?,
                qkv_b: f32s(names::attn_qkv_bias(b))?,
                o_w: bf16(names::attn_out_weight(b))?,
                o_b: f32s(names::attn_out_bias(b))?,
                ln2: (f32s(names::ln2_weight(b))?, f32s(names::ln2_bias(b))?),
                up_w: DeviceBuffer::from_host(stream, &up_w)?,
                up_b: DeviceBuffer::from_host(stream, &up_b)?,
                down_w: DeviceBuffer::from_host(stream, &down_w)?,
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
        let w = Weights {
            patch_w: DeviceBuffer::from_host(stream, &patch_w)?,
            patch_b: f32s(names::patch_embd_bias())?,
            pos: f32s(names::position_embd())?,
            blocks,
            post_ln: (f32s(names::post_ln_weight())?, f32s(names::post_ln_bias())?),
            mm0_w: bf16(names::mm0_weight())?,
            mm0_b: f32s(names::mm0_bias())?,
            mm2_w: bf16(names::mm2_weight())?,
            mm2_b: f32s(names::mm2_bias())?,
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
        let lens = Lens::of(&hp, limits.max());
        let s = Scratch::new(stream, lens)?;
        stream.synchronize()?;
        Ok(Encoder {
            chain: Chain {
                gemm: GemmKernels::load(ctx, stream)?,
                norm: LayerNormKernels::load(ctx)?,
                gelu: GeluTanhKernels::load(ctx)?,
                pos: PosBilinearKernels::load(ctx)?,
                rope: Rope72Kernels::load(ctx)?,
                attn: Attn72Kernels::load(ctx)?,
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

    /// Chain launches of one encode at `n_layer` blocks: the position rows, the patch GEMM, nine
    /// per block, the final norm and the merger's three.
    #[must_use]
    pub const fn launches(n_layer: usize) -> usize {
        2 + 9 * n_layer + 4
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
        let what = "Qwen3-VL tower encode";
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
            (&s.patches, &chain.w.patch_w, Some(&chain.w.patch_b)),
            (n, hp.dim, k_patch),
            &mut s.xb,
        )?;
        chain.gemm.enqueue(
            stream,
            GemmArgs {
                a: &s.patches,
                b: &chain.w.patch_w,
                bias: Some(&chain.w.patch_b),
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
        for b in 0..hp.n_layer {
            launches += chain.block(stream, b, n, s, taps)?;
        }
        let (n_tok, tail) = chain.tail(stream, (n_h, n_w), s, false, taps)?;
        Ok(Encoded {
            rows: DeviceTensor::window_of(&s.rows, 0, n_tok, hp.out_dim)?,
            launches: launches + tail,
        })
    }

    /// One block alone on the host rows `x` (`n_h · n_w` rows of `dim` bf16): the chain's launches
    /// of block `b` from `x` and the block's output — a teacher-forced step, for a gate that feeds
    /// the reference's input to a block.
    pub fn forced_block(
        &mut self,
        (n_h, n_w): (usize, usize),
        b: usize,
        x: &[u16],
    ) -> Result<Vec<u16>, GpuError> {
        self.forced_input((n_h, n_w), x)?;
        let (chain, s, stream) = (&self.chain, &mut self.s, &*self.stream);
        let n = n_h * n_w;
        chain.block(stream, b, n, s, &mut NoTaps)?;
        to_host(stream, &s.xa, n * chain.hp.dim)
    }

    /// The chain after the last block alone on the host rows `x`: the final norm's output and the
    /// merger's rows, teacher-forced as [`Encoder::forced_block`]. With `from_vit`, `x` is taken
    /// as the final norm's output and only the merger runs.
    pub fn forced_tail(
        &mut self,
        (n_h, n_w): (usize, usize),
        x: &[u16],
        from_vit: bool,
    ) -> Result<(Vec<u16>, Vec<u16>), GpuError> {
        self.forced_input((n_h, n_w), x)?;
        let (chain, s, stream) = (&self.chain, &mut self.s, &*self.stream);
        let (n_tok, _) = chain.tail(stream, (n_h, n_w), s, from_vit, &mut NoTaps)?;
        Ok((
            to_host(stream, &s.h, n_h * n_w * chain.hp.dim)?,
            to_host(stream, &s.rows, n_tok * chain.hp.out_dim)?,
        ))
    }

    /// Start an `n_h × n_w` image whose host rows `x` are the block input (`xa`) and the final
    /// norm's output (`h`).
    fn forced_input(&mut self, (n_h, n_w): (usize, usize), x: &[u16]) -> Result<(), GpuError> {
        let what = "Qwen3-VL tower forced step";
        let (hp, s, stream) = (&self.chain.hp, &mut self.s, &*self.stream);
        if n_h == 0 || n_w == 0 || x.len() != n_h * n_w * hp.dim {
            return Err(GpuError::Shape {
                what,
                detail: format!("{} values for {n_h}x{n_w} rows of {}", x.len(), hp.dim),
            });
        }
        s.start(stream, hp, (n_h, n_w), what)?;
        for buf in [&mut s.xa, &mut s.h] {
            WindowMut::<u16>::of_mut(buf, 0, x.len())?.copy_from_host(stream, x)?;
        }
        Ok(())
    }
}

impl Scratch {
    /// The buffers at `lens`, zero-filled. Load-time only.
    fn new(stream: &CudaStream, lens: Lens) -> Result<Scratch, GpuError> {
        let bf16 = |n: usize| DeviceBuffer::<u16>::zeroed(stream, n);
        Ok(Scratch {
            lens,
            patches: bf16(lens.patches)?,
            pos: bf16(lens.pos)?,
            cs: DeviceBuffer::zeroed(stream, lens.cs)?,
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
        if n * hp.dim > self.lens.stream {
            return Err(GpuError::Shape {
                what,
                detail: format!(
                    "a {n_h}x{n_w} grid ({n} patches, {} tokens) does not fit the tower's buffers, sized for {} tokens",
                    n / (hp.merge * hp.merge),
                    self.lens.stream / hp.dim / (hp.merge * hp.merge)
                ),
            });
        }
        let table = RopeTable72::in_order(n_h, n_w, merge_order(n_h, n_w, hp.merge), ROPE_BASE);
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
        let (dim, heads, ff) = (hp.dim, hp.n_head, ff_padded(hp));
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
            .enqueue(stream, &s.xa, (&bw.ln1.0, &bw.ln1.1), hp.eps, n, &mut s.h)?;
        tap(stream, taps, &name("norm1"), &s.h, n, dim)?;
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
        tap(stream, taps, &name("qkv"), &s.qkv, n, 3 * dim)?;
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
                let all = to_host(stream, &s.qkv, n * 3 * dim)?;
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
        tap(stream, taps, &name("resid1"), &s.xb, n, dim)?;
        self.norm
            .enqueue(stream, &s.xb, (&bw.ln2.0, &bw.ln2.1), hp.eps, n, &mut s.h)?;
        tap(stream, taps, &name("norm2"), &s.h, n, dim)?;
        self.gemm.enqueue(
            stream,
            GemmArgs {
                a: &s.h,
                b: &bw.up_w,
                bias: Some(&bw.up_b),
                epilogue: Epilogue::None,
                resid: None,
                m: n,
                n: ff,
                k: dim,
                c: &mut s.up,
            },
        )?;
        tap(stream, taps, &name("up"), &s.up, n, ff)?;
        self.gelu.enqueue(stream, n * ff, &mut s.up)?;
        tap(stream, taps, &name("act"), &s.up, n, ff)?;
        side(
            &self.gemm,
            stream,
            taps,
            &name("mlp"),
            (&s.up, &bw.down_w, Some(&bw.down_b)),
            (n, dim, ff),
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
                m: n,
                n: dim,
                k: ff,
                c: &mut s.xa,
            },
        )?;
        tap(stream, taps, &format!("blk{b}"), &s.xa, n, dim)?;
        Ok(9)
    }

    /// The chain after the last block on an `n_h × n_w` grid: the final norm of `s.xa` into `s.h`
    /// (skipped when `vit_given`: `s.h` holds it already) and the merger into `s.rows`, whose
    /// first-GEMM input is `s.h` re-read as rows of `merge²·dim` values; returns the merged
    /// tokens and the launches.
    fn tail(
        &self,
        stream: &CudaStream,
        (n_h, n_w): (usize, usize),
        s: &mut Scratch,
        vit_given: bool,
        taps: &mut dyn TapSink,
    ) -> Result<(usize, usize), GpuError> {
        let hp = &self.hp;
        let dim = hp.dim;
        let n = n_h * n_w;
        let merged = dim * hp.merge * hp.merge;
        let n_tok = n / (hp.merge * hp.merge);
        let mut launches = 0;
        if !vit_given {
            self.norm.enqueue(
                stream,
                &s.xa,
                (&self.w.post_ln.0, &self.w.post_ln.1),
                hp.eps,
                n,
                &mut s.h,
            )?;
            launches += 1;
        }
        tap(stream, taps, "vit", &s.h, n, dim)?;
        self.gemm.enqueue(
            stream,
            GemmArgs {
                a: &s.h,
                b: &self.w.mm0_w,
                bias: Some(&self.w.mm0_b),
                epilogue: Epilogue::None,
                resid: None,
                m: n_tok,
                n: merged,
                k: merged,
                c: &mut s.hid,
            },
        )?;
        tap(stream, taps, "merger.pre", &s.hid, n_tok, merged)?;
        self.gelu.enqueue(stream, n_tok * merged, &mut s.hid)?;
        tap(stream, taps, "merger.h", &s.hid, n_tok, merged)?;
        self.gemm.enqueue(
            stream,
            GemmArgs {
                a: &s.hid,
                b: &self.w.mm2_w,
                bias: Some(&self.w.mm2_b),
                epilogue: Epilogue::None,
                resid: None,
                m: n_tok,
                n: hp.out_dim,
                k: merged,
                c: &mut s.rows,
            },
        )?;
        launches += 3;
        Ok((n_tok, launches))
    }
}
