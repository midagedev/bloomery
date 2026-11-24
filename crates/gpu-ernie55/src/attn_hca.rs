//! ERNIE 5.5's HCA attention sub-layer on the card: the attention norm, the
//! query and latent paths, the compressor, the attention over the window and
//! the compressed rows, and ERNIE's tail — the VHA head mix, the grouped
//! output projection, the sigmoid output gate and `attn_output`. Every entry
//! is `bloomery-gpu`'s. The attention over window rows ⧺ compressed rows, the
//! compressor and the tail rope are the ones DeepSeek V4.1 runs
//! ([`bloomery_gpu::hca`], [`bloomery_gpu::compress`],
//! [`bloomery_gpu::rope_tail`]); the head mix is [`bloomery_gpu::vha`]; the
//! gate is [`bloomery_gpu::linear::out_gate`]'s; each projection runs by its
//! file's type ([`bloomery_gpu::site`]).
//!
//! One [`HcaLayer`] is one layer's tensors, read by every sequence; one
//! [`HcaState`] is one sequence's caches for every HCA layer, the words of the
//! run the host set, and the scratch. A run is `m` tokens at positions
//! `b .. b + m`.
//!
//! The step ([`HcaLayer::enqueue_step`]), one token at `pos`:
//!
//! 1. `attn_norm` of the sub-layer's input `x` (the hyper-connection fold), or
//!    that norm as the trunk's pre already wrote it ([`HcaIn`]);
//! 2. `attn_q_a`, its norm (`attn_q_a_norm`), then `attn_q_b` and the gate
//!    projection `attn_gate` of the normed query latent;
//! 3. the weightless RMS norm of every query head, then the rope of its tail;
//! 4. `attn_kv`, its norm (`attn_kv_a_norm`), the tail rope and the f16 row
//!    into the window ring's slot `pos % window`
//!    ([`RopeKernels::enqueue_kv_norm_rope_append`]);
//! 5. the compressor: `attn_compressor_kv` and `attn_compressor_gate` of the
//!    normed input, the pooling scores plus the position table's row
//!    `ape[pos % ratio]`, then the pool ([`CompressKernels::enqueue_pool`]):
//!    the block `pos % ratio == ratio − 1` completes is pooled, normed
//!    (`attn_compressor_norm`), roped at its first position and written f16
//!    into the compressed rows; the compressor's ring keeps the last
//!    projection of each residue;
//! 6. the attention over `min(pos + 1, window)` window keys and
//!    `(pos + 1) / ratio` compressed rows, a head's sink in the denominator
//!    ([`AttnKernels::enqueue`]);
//! 7. the VHA head mix `o[h] ← Σ_g (I + V Uᵀ)[h, g] · o[g]`, then the inverse
//!    rope of the output's tail;
//! 8. the grouped output projection `attn_output_a` (eight groups of eight
//!    heads, block diagonal), `· sigmoid(attn_gate)` on the grouped rows, and
//!    `attn_output_b`.
//!
//! What the file forces here, each named where it is used: the weightless
//! per-head norm of the query after `attn_q_b` (ERNIE's `qk_norm_type`), the
//! one rope of every rotation — the compressed layers' YaRN table at
//! `attention.compress_rope_freq_base`, the window rows', the query's, the
//! pooled rows' and the inverse's ([`HcaDims::yarn`]) — the compressor's
//! position table, the head mix, the gate from the normed query latent, and
//! the grouped output.
//!
//! The prompt form ([`HcaLayer::enqueue_prompt`]) is the step at every
//! position, in pieces: a piece is [`HcaDims::piece`] tokens starting at a
//! multiple of it, whose own rows are staged beside the ring
//! ([`AttnKernels::enqueue_staged`]) and committed to it after the attention
//! ([`AttnKernels::enqueue_commit`]). The staged read takes staging row `j` as
//! position `b + j`, and the append puts position `p` at row `p % rows`; they
//! agree exactly when the piece starts at a multiple of the staging's rows, so
//! the tokens before the first such multiple, at most `piece − 1`, run one at
//! a time through the step. Each token's row is the step's at its position:
//! the staged attention, the m-column gemvs and the row-wise norms are the
//! one-token launches bit for bit.
//!
//! Numeric contract: the projections are their file type's gemv, the norms
//! `elem::rms_norm`'s f32 tree (the weightless one over a gain of ones), the
//! rotations [`bloomery_gpu::rope_table::rope_pair_rn`] on the table
//! [`RopeSpec::yarn`] builds (its magnitude term `attn_factor · mscale` is 1 to
//! an ulp where the reference cancels it exactly), the window rows and the
//! pooled rows f16, the logits and the online softmax f32 over the f16 query
//! and key rows, the head mix [`bloomery_gpu::vha`]'s fma chain, the gate
//! `attn · sigmoid(g)` rounded once.
//!
//! The words a launch reads — the positions, the visible counts, the
//! compressor's step layout and the rope rows — are the host's, written by
//! [`HcaState::set_run`] before the run's launches: on the stream, outside a
//! captured graph, as the step's other host inputs are. A launch of a run the
//! words do not hold is refused by name. The step allocates nothing and
//! synchronizes nothing; the prompt form neither.
//!
//! Not this module's: restoring the window ring after a rollback. A ring of
//! `window` rows holds each position's row until `window` positions later, so
//! a rollback of `d` tokens loses the `d − 1` rows the rejected positions
//! overwrote; a draft that rolls back needs a per-position shadow to restore
//! from, as V4.1's step has. The compressed rows need none: a position's
//! visible blocks are a prefix of the rows, and the compressor's ring holds
//! one block's residues, which the rejected positions of the same block do not
//! collide with.

use bloomery_gpu::compress::{CompGeom, CompressKernels, PoolArgs, StepInts};
use bloomery_gpu::gemm::Gemm32Kernels;
use bloomery_gpu::hca::{AttnArgs, AttnKernels, CommitArgs, Staged, segments};
use bloomery_gpu::linear::HEAD as GATE_HEAD;
use bloomery_gpu::linear::out_gate::{OutGateArgs, OutGateKernels};
use bloomery_gpu::model::StepKernels;
use bloomery_gpu::q8f32::{E4m3GemvArgs, E4m3Weight, GemvOut};
use bloomery_gpu::rope_table::{Direction, RopeSpec, RopeTable};
use bloomery_gpu::rope_tail::{KvAppendArgs, RopeKernels, TailShape};
use bloomery_gpu::site::{self, SiteTy};
use bloomery_gpu::vha::VhaKernels;
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{
    DeviceBufferRaw, DeviceTensor, Gpu, GpuError, Pitched, Window, WindowMut, capturing, dtod_2d,
};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DeviceCopy};
use gguf::Split;
use gguf::quant::GgmlType;
use model::arch::ernie55::hparams::{Hca, Hparams, Rope};
use model::arch::ernie55::names;
use model::arch::models::OutGate;
use std::sync::Arc;

const WHAT: &str = "ernie55 attn_hca";

/// A value of a window ring row, a compressed row and a latent K/V row.
const LATENT: usize = bloomery_gpu::hca::LATENT;
const _: () = assert!(LATENT == bloomery_gpu::compress::WIDTH);
/// Bytes of an f32 on the card: the windows and the copies are byte-offset.
const F32: usize = size_of::<f32>();

/// The description's facts the HCA shapes are built from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HcaKeys {
    /// `embedding_length`.
    pub hidden: usize,
    /// `attention.head_count`.
    pub heads: usize,
    /// The HCA layer's attention keys.
    pub hca: Hca,
    /// `rope.dimension_count`: the values at the tail of a head that turn.
    pub rope: usize,
    /// `attention.q_lora_rank`: the query latent's values.
    pub q_lora: usize,
    /// `attention.layer_norm_rms_epsilon`.
    pub eps: f32,
    /// The rope keys.
    pub yarn: Rope,
}

/// The HCA layers' shapes: one for every layer, from the description's keys
/// and the context the caches are sized for.
#[derive(Clone, Debug, PartialEq)]
pub struct HcaDims {
    pub hidden: usize,
    pub heads: usize,
    /// A head's values, and a latent row's.
    pub head: usize,
    /// The values at the tail of a head that turn.
    pub rope: usize,
    /// Positions pooled into one compressed row.
    pub ratio: usize,
    /// The window ring's rows.
    pub window: usize,
    pub o_groups: usize,
    pub o_lora: usize,
    pub q_lora: usize,
    pub eps: f32,
    /// The logits' scale, `1 / √head`.
    pub scale: f32,
    /// The positions the caches hold: a position at or past it is refused.
    pub ctx: usize,
    /// Compressed rows the context holds, `⌈ctx / ratio⌉`.
    pub comp_rows: usize,
    /// Tokens one staged piece of the prompt form holds: the staging's rows.
    pub piece: usize,
    /// The rope of every rotation: the compressed layers' YaRN table.
    pub yarn: RopeSpec,
}

impl HcaDims {
    /// Tokens one staged piece of the prompt form holds unless a plan says
    /// another: a multiple of the gemv arm's columns under a pooling block,
    /// so a piece completes at most one block.
    pub const PIECE: usize = 32;
    /// Tokens one projection launch of the prompt form takes: the gemv arm's
    /// columns.
    pub const COLS: usize = 8;
    /// The weightless per-head norm's tail bound, `2 · RMS_THREADS` of the
    /// latent append.
    const ROPE_MAX: usize = 512;

    /// [`HcaDims::new`] of `hp`'s keys.
    pub fn of(hp: &Hparams, ctx: usize, piece: usize) -> Result<HcaDims, GpuError> {
        HcaDims::new(
            HcaKeys {
                hidden: hp.n_embd,
                heads: hp.heads,
                hca: hp.hca,
                rope: hp.mla.pe,
                q_lora: hp.mla.q_lora,
                eps: hp.rms_eps,
                yarn: hp.rope,
            },
            ctx,
            piece,
        )
    }

    /// The shapes of `keys` over a context of `ctx` positions, with prompt
    /// pieces of `piece` tokens. Refused by name: a head the kernels are not
    /// 512 wide, a head count the attention block or the head mix does not
    /// take, a tail the compressor and the append do not turn, a pooling
    /// ratio below 2, an empty window or context, a width the FP8 block does
    /// not divide, a grouped row the output gate's head does not divide, and
    /// a count the kernels' u32 arithmetic cannot address.
    pub fn new(keys: HcaKeys, ctx: usize, piece: usize) -> Result<HcaDims, GpuError> {
        let refuse = |detail: &str| GpuError::shape(WHAT, detail.to_string());
        let HcaKeys {
            hidden,
            heads,
            hca:
                Hca {
                    head,
                    ratio,
                    window,
                    o_groups,
                    o_lora,
                },
            rope,
            q_lora,
            eps,
            yarn,
        } = keys;
        let mma_rows = bloomery_gpu::flash::MMA_ROWS;
        if head != LATENT {
            return Err(refuse(&format!(
                "attention.hca.head_length is {head}; the attention, the compressor and the \
                 latent append are {LATENT} wide"
            )));
        }
        if heads == 0 || !heads.is_multiple_of(mma_rows) {
            return Err(refuse(&format!(
                "attention.head_count is {heads}; an attention block is {mma_rows} heads of one \
                 token"
            )));
        }
        let threads = bloomery_gpu::vha::THREADS as usize;
        if !heads.is_multiple_of(32) || heads > threads || !threads.is_multiple_of(heads) {
            return Err(refuse(&format!(
                "attention.head_count is {heads}; the head mix takes a multiple of 32 that \
                 divides {threads}"
            )));
        }
        if rope < 4 || !rope.is_multiple_of(4) || rope > head || rope > Self::ROPE_MAX {
            return Err(refuse(&format!(
                "rope.dimension_count is {rope}; the tail is a multiple of 4 from 4 to the \
                 lesser of the head's {head} and {}",
                Self::ROPE_MAX
            )));
        }
        if ratio < 2 {
            return Err(refuse(&format!(
                "the pooling ratio is {ratio}; the compressor pools at 2 or above"
            )));
        }
        if window == 0 {
            return Err(refuse("the sliding window holds no position"));
        }
        let group_k = heads * head / o_groups.max(1);
        let grouped = o_groups * o_lora;
        for (what, n) in [
            ("embedding_length", hidden),
            ("attention.q_lora_rank", q_lora),
            ("the grouped output projection's input", group_k),
            ("the grouped output projection's output", grouped),
        ] {
            if n == 0 || !n.is_multiple_of(128) {
                return Err(refuse(&format!(
                    "{what} is {n}; the FP8 projections take a width the 128-value block divides"
                )));
            }
        }
        if !grouped.is_multiple_of(GATE_HEAD) {
            return Err(refuse(&format!(
                "the grouped output is {grouped} values; the output gate takes heads of \
                 {GATE_HEAD}"
            )));
        }
        if piece == 0 || !piece.is_multiple_of(Self::COLS) {
            return Err(refuse(&format!(
                "a prompt piece is a positive multiple of {} tokens",
                Self::COLS
            )));
        }
        let Ok(ctx_u32) = u32::try_from(ctx) else {
            return Err(refuse("the context does not fit u32"));
        };
        let ctx = ctx_u32 as usize;
        if ctx == 0 {
            return Err(refuse("the context holds no position"));
        }
        let comp_rows = ctx.div_ceil(ratio);
        let addressable = [comp_rows * LATENT, window, heads * head, (ctx + 1) / ratio]
            .iter()
            .all(|&n| bloomery_gpu::launch_u32(WHAT, "a count past u32", n).is_ok());
        if !addressable {
            return Err(refuse("a count the kernels' u32 arithmetic cannot address"));
        }
        let Ok(original_ctx) = i32::try_from(yarn.original_ctx) else {
            return Err(refuse("the rope's original context does not fit i32"));
        };
        Ok(HcaDims {
            hidden,
            heads,
            head,
            rope,
            ratio,
            window,
            o_groups,
            o_lora,
            q_lora,
            eps,
            scale: 1.0 / (head as f32).sqrt(),
            ctx,
            comp_rows,
            piece,
            yarn: RopeSpec::yarn(
                yarn.compress_base,
                yarn.factor,
                original_ctx,
                yarn.beta_fast,
                yarn.beta_slow,
                rope,
            ),
        })
    }

    /// The grouped output's values: `o_groups` groups of `o_lora`.
    #[must_use]
    pub fn grouped(&self) -> usize {
        self.o_groups * self.o_lora
    }

    /// The values a group of the grouped output projection reads: its heads'
    /// values.
    #[must_use]
    pub fn group_k(&self) -> usize {
        self.heads * self.head / self.o_groups
    }

    /// Every query head's values: a token's query row.
    #[must_use]
    pub fn q_width(&self) -> usize {
        self.heads * self.head
    }

    /// Segments of one attention launch over the ring and every compressed
    /// row ([`segments`]).
    #[must_use]
    pub fn segs(&self) -> usize {
        segments(self.window, self.comp_rows)
    }

    /// Blocks a run of `tokens` tokens completes at most.
    #[must_use]
    pub fn max_groups(&self, tokens: usize) -> usize {
        tokens.div_ceil(self.ratio)
    }

    /// The compressor's geometry of a run of `tokens` tokens, its step buffer
    /// laid out for `capacity` tokens' blocks.
    #[must_use]
    pub fn geom(&self, tokens: usize, capacity: usize) -> CompGeom {
        CompGeom {
            ratio: self.ratio,
            max_groups: self.max_groups(capacity),
            tokens,
            rows: self.comp_rows,
        }
    }

    /// The bytes one layer's caches take for one sequence ([`HcaCache`]): the
    /// ring, the compressed rows and the compressor's two rings — the stores
    /// alone, what the plan's compressed hold counts. The append's shadow row
    /// and the pool's pre-rope rows are not here: nothing reads them, so one
    /// pair serves every layer ([`HcaDims::scratch_bytes`]).
    #[must_use]
    pub fn cache_bytes(&self) -> usize {
        let lat = |n: usize| n * LATENT;
        2 * (lat(self.window) + lat(self.comp_rows)) + F32 * 2 * lat(self.ratio)
    }

    /// The bytes of the working buffers one sequence's HCA layers share, for
    /// runs of up to [`HcaDims::piece`] tokens ([`HcaState::new`]'s lengths):
    /// the run's intermediates, the staging, the words the launches read, the
    /// flash partials and the two write-only sinks (the append's shadow row
    /// and the pool's pre-rope rows). The layers run one after another on one
    /// stream, so one set serves all of them.
    #[must_use]
    pub fn scratch_bytes(&self) -> usize {
        let (p, lat) = (self.piece, self.head);
        let groups = self.max_groups(p);
        let f32s = p * (self.hidden + 2 * self.q_lora + 3 * self.q_width() + 3 * self.grouped())
            + p * 6 * lat
            + Self::COLS * (self.group_k() + self.o_lora)
            + p * self.heads * self.segs() * (LATENT + 2)
            + lat
            + 2 * p * self.rope
            + groups * self.rope
            + groups * LATENT;
        let u32s = p + 2 * p + self.geom(p, p).words();
        F32 * f32s + size_of::<u32>() * u32s + 2 * (p + 1) * LATENT
    }
}

/// The loaded entries the HCA layers launch.
pub struct HcaKernels {
    pub attn: AttnKernels,
    pub compress: CompressKernels,
    pub rope: RopeKernels,
    pub vha: VhaKernels,
    pub out_gate: OutGateKernels,
    /// The F32 tile, read by a projection of more than one column on a file
    /// that holds its weights as F32.
    pub g32: Gemm32Kernels,
    /// The pair gather that takes the run's position-table rows by the words'
    /// indices.
    pub step: StepKernels,
}

impl HcaKernels {
    /// Load the entries into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<HcaKernels, GpuError> {
        Ok(HcaKernels {
            attn: AttnKernels::load(ctx)?,
            compress: CompressKernels::load(ctx)?,
            rope: RopeKernels::load(ctx)?,
            vha: VhaKernels::load(ctx)?,
            out_gate: OutGateKernels::load(ctx)?,
            g32: Gemm32Kernels::load(ctx)?,
            step: StepKernels::load(ctx)?,
        })
    }
}

/// The types a dense projection runs by name on the gemv arm: the families
/// whose launch reads the f32 rows (`bloomery_gpu::site`, `BY_NAME`).
const PROJECTED: [SiteTy; 5] = [
    SiteTy::Q8_0,
    SiteTy::F32,
    SiteTy::Mxfp8,
    SiteTy::Fp8B128,
    SiteTy::Bf16,
];

/// The types the grouped output projection takes: a weight in the E4M3
/// planes, a group's block a window of its rows.
const GROUPED: [SiteTy; 2] = [SiteTy::Mxfp8, SiteTy::Fp8B128];

/// The head mix's factors of one layer: `U` and `V`, each `heads` rows of
/// `rank` values, one row a head.
pub struct Vha {
    pub rank: usize,
    pub u: Vec<f32>,
    pub v: Vec<f32>,
}

/// The F32 rows of tensor `name` of `file` as the load put them on the card
/// (`w`, read back on `stream`): `rows` rows of `cols` values. The card's copy
/// is the one every load holds, a tensor a checkpoint folder's step makes
/// included, which no file run holds. A header entry of another type or shape,
/// and a resident tensor of another length, are refused by name.
fn f32_rows(
    stream: &CudaStream,
    (file, w): (&Split, &Weights),
    name: &str,
    cols: usize,
    rows: usize,
) -> Result<Vec<f32>, GpuError> {
    let Some((_, t)) = file.find(name) else {
        return Err(GpuError::tensor(WHAT, name, "in the file"));
    };
    if t.ty != GgmlType::F32 || t.dims != [cols as u64, rows as u64] {
        return Err(GpuError::tensor(
            WHAT,
            name,
            "F32 of the shape the description gives this tensor",
        ));
    }
    let values = w.f32_buf(WHAT, name)?.to_host_vec(stream)?;
    if values.len() != rows * cols {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "{name}: {} resident values, not {rows} rows of {cols}",
                values.len()
            ),
        ));
    }
    Ok(values)
}

/// The head mix's factors of layer `l` of `file`, for [`HcaLayer::load`]:
/// `attn_vha_u.weight` and `attn_vha_v.weight`, `[rank, heads]` each, read
/// back from the card copies `w` holds.
pub fn vha_rows(
    stream: &CudaStream,
    (file, w): (&Split, &Weights),
    l: usize,
    (heads, rank): (usize, usize),
) -> Result<Vha, GpuError> {
    Ok(Vha {
        rank,
        u: f32_rows(stream, (file, w), &names::attn_vha_u(l), rank, heads)?,
        v: f32_rows(stream, (file, w), &names::attn_vha_v(l), rank, heads)?,
    })
}

/// The rows a run reads.
#[derive(Clone, Copy)]
pub enum HcaIn<'a> {
    /// The sub-layer's input, the hyper-connection fold: the run norms it by
    /// `attn_norm` first.
    Fold(&'a DeviceBuffer<f32>),
    /// That fold already normed by `attn_norm` under the run's own rule
    /// (`elem::rms_norm`, a block a row, the description's epsilon): the
    /// trunk's pre leaves it, so the run reads it as its normed input and
    /// launches no norm.
    Normed(&'a DeviceBuffer<f32>),
}

impl<'a> HcaIn<'a> {
    /// The rows, whichever form they hold.
    fn rows(self) -> &'a DeviceBuffer<f32> {
        match self {
            HcaIn::Fold(x) | HcaIn::Normed(x) => x,
        }
    }

    /// The same form over `rows`: a window of these rows.
    fn with<'b>(self, rows: &'b DeviceBuffer<f32>) -> HcaIn<'b> {
        match self {
            HcaIn::Fold(_) => HcaIn::Fold(rows),
            HcaIn::Normed(_) => HcaIn::Normed(rows),
        }
    }
}

/// One layer's tensors by the description's names: the resident weights of
/// `Weights`, and the head mix folded at load ([`bloomery_gpu::vha::fold`]).
pub struct HcaLayer {
    /// The layer's index: its cache's, and its fault sink's.
    pub layer: usize,
    pub norm: String,
    pub q_a: String,
    pub q_a_norm: String,
    pub q_b: String,
    pub kv: String,
    pub kv_norm: String,
    pub comp_kv: String,
    pub comp_gate: String,
    pub comp_norm: String,
    /// The pooling's position table, `[ratio, head]`: score bias row `pos %
    /// ratio` of a pooling group.
    pub comp_ape: String,
    pub gate: String,
    pub output_a: String,
    pub output_b: String,
    pub sinks: String,
    /// The head mix, folded and transposed: `heads²` f32 on the card.
    pub mix: DeviceBuffer<f32>,
}

impl HcaLayer {
    /// Layer `l`'s tensors of `w` at `d`'s shapes, the head mix folded from
    /// `vha` ([`vha_rows`] reads a file's). Every vector is F32 of its length
    /// and every projection a resident weight of its shape in a type its
    /// launch reads ([`PROJECTED`]); the grouped output projection takes the
    /// E4M3 planes ([`GROUPED`]). A missing tensor, another type or another
    /// shape is refused by name. Load-time only.
    pub fn load(
        stream: &CudaStream,
        w: &Weights,
        d: &HcaDims,
        l: usize,
        vha: &Vha,
    ) -> Result<HcaLayer, GpuError> {
        let (e, q, hd, grouped, group_k) = (d.hidden, d.q_lora, d.head, d.grouped(), d.group_k());
        let vector = |name: String, len: usize| site::vector(w, WHAT, name, len);
        let site = |name: String, shape: (usize, usize), allowed: &[SiteTy]| {
            site::resident_site(w, WHAT, &name, shape, allowed).map(|_| name)
        };
        let mix = bloomery_gpu::vha::fold(&vha.u, &vha.v, d.heads, vha.rank, WHAT)?;
        Ok(HcaLayer {
            layer: l,
            norm: vector(names::attn_norm(l), e)?,
            q_a: site(names::attn_q_a(l), (q, e), &PROJECTED)?,
            q_a_norm: vector(names::attn_q_a_norm(l), q)?,
            q_b: site(names::attn_q_b(l), (d.q_width(), q), &PROJECTED)?,
            kv: site(names::attn_kv(l), (hd, e), &PROJECTED)?,
            kv_norm: vector(names::attn_kv_a_norm(l), hd)?,
            comp_kv: site(names::attn_compressor_kv(l), (hd, e), &PROJECTED)?,
            comp_gate: site(names::attn_compressor_gate(l), (hd, e), &PROJECTED)?,
            comp_norm: vector(names::attn_compressor_norm(l), hd)?,
            comp_ape: vector(names::attn_compressor_ape(l), d.ratio * hd)?,
            gate: site(names::attn_gate(l), (grouped, q), &PROJECTED)?,
            output_a: site(names::attn_output_a(l), (grouped, group_k), &GROUPED)?,
            output_b: site(names::attn_output_b(l), (e, grouped), &PROJECTED)?,
            sinks: vector(names::attn_sinks(l), d.heads)?,
            mix: DeviceBuffer::from_host(stream, &mix)?,
        })
    }
}

/// One layer's caches for one sequence: the window ring, the compressed rows
/// and the compressor's state.
struct HcaCache {
    /// The raw window ring: `window` latent rows in f16, position `p` at slot
    /// `p % window`.
    ring: DeviceTensor<u16>,
    /// The compressed rows: `comp_rows` latent rows in f16.
    rows: DeviceTensor<u16>,
    /// The compressor's ring: the `ratio` latest pooling projections, slot
    /// `p % ratio`.
    ring_kv: DeviceBuffer<f32>,
    /// Their scores.
    ring_score: DeviceBuffer<f32>,
}

impl HcaCache {
    fn new(stream: &CudaStream, d: &HcaDims) -> Result<HcaCache, GpuError> {
        let lat = |n: usize| n * LATENT;
        Ok(HcaCache {
            ring: DeviceTensor::zeroed(stream, d.window, LATENT)?,
            rows: DeviceTensor::zeroed(stream, d.comp_rows, LATENT)?,
            ring_kv: DeviceBuffer::zeroed(stream, lat(d.ratio))?,
            ring_score: DeviceBuffer::zeroed(stream, lat(d.ratio))?,
        })
    }

    /// Its bytes on the card.
    fn bytes(&self) -> usize {
        let t = |r: &DeviceTensor<u16>| r.rows() * r.cols() * size_of::<u16>();
        t(&self.ring) + t(&self.rows) + F32 * (self.ring_kv.len() + self.ring_score.len())
    }
}

/// The gather's destination indices of `n` values: `0 .. n`, refused by name
/// past `u32`.
fn ape_dst(n: usize) -> Result<Vec<u32>, GpuError> {
    (0..n)
        .map(|i| {
            u32::try_from(i)
                .map_err(|_| GpuError::shape(WHAT, format!("a position-table index {i} past u32")))
        })
        .collect()
}

/// The words the launches of a run read.
struct Words {
    /// The run's positions; word 0 is its first, the staged read's base.
    pos: DeviceBuffer<u32>,
    /// Per token: its window keys, then the compressed rows it sees.
    vis: DeviceBuffer<u32>,
    /// The compressor's step layout ([`CompGeom::pack`]).
    step: DeviceBuffer<u32>,
    /// The run's forward and inverse rope rows, `rope` a token.
    cs_fwd: DeviceBuffer<f32>,
    cs_back: DeviceBuffer<f32>,
    /// The rope rows of the blocks the run completes, at their first
    /// positions.
    cs_comp: DeviceBuffer<f32>,
    /// Per token value: the index of its position-table value,
    /// `(pos % ratio) · head + j`, the gather's source.
    ape_src: DeviceBuffer<u32>,
    /// The gather's destination: token `t`'s value `j` at `t · head + j`,
    /// written at load.
    ape_dst: DeviceBuffer<u32>,
}

/// The scratch one run's launches share, sized for a piece: the layers of a
/// sequence run their pieces one after another on one stream.
struct Scratch {
    /// The normed input, `piece × hidden`.
    normed: DeviceBuffer<f32>,
    q_a: DeviceBuffer<f32>,
    /// The normed query latent.
    q_lat: DeviceBuffer<f32>,
    /// The query rows of every head.
    q: DeviceBuffer<f32>,
    /// Their weightless per-head norm, then their roped rows.
    qh: DeviceBuffer<f32>,
    /// The gate projection's rows.
    gate: DeviceBuffer<f32>,
    kv: DeviceBuffer<f32>,
    comp_kv: DeviceBuffer<f32>,
    comp_gate: DeviceBuffer<f32>,
    /// The position table's rows of the run, and the pooling scores.
    ape: DeviceBuffer<f32>,
    score: DeviceBuffer<f32>,
    /// The latent rows after their norm and rope, in f32.
    kv_row: DeviceBuffer<f32>,
    /// The attention's output rows: the head mix's, then the inverse rope's.
    y: DeviceBuffer<f32>,
    /// A column chunk's gathered input of one group, `COLS × group_k`.
    xg: DeviceBuffer<f32>,
    /// The group's outputs, `COLS × o_lora`.
    og: DeviceBuffer<f32>,
    /// The grouped output, and its gated rows: `piece × grouped`.
    oa: DeviceBuffer<f32>,
    oa_g: DeviceBuffer<f32>,
    part_v: DeviceBuffer<f32>,
    part_ms: DeviceBuffer<f32>,
    /// The weightless norms' gain: `head` ones.
    ones: DeviceBuffer<f32>,
    /// The latent append's shadow, one row: the kernel writes the shadow row
    /// of a position below its rows, so position 0's alone, and nothing reads
    /// it (the module doc's rollback paragraph).
    shadow: DeviceTensor<u16>,
    /// The blocks a run completes, pooled and before their rope, in f32:
    /// written by the pool, read by nothing.
    pre: DeviceBuffer<f32>,
}

/// One sequence's HCA state: a cache per HCA layer, the words of the run the
/// host set, the staging of a prompt piece, the scratch, and every position's
/// rope rows on the host.
pub struct HcaState {
    d: HcaDims,
    /// A cache per layer of the file: `None` where `hca` was false.
    caches: Vec<Option<HcaCache>>,
    words: Words,
    /// A prompt piece's staged latent rows: `piece` rows in f16.
    staging: DeviceTensor<u16>,
    sc: Scratch,
    /// The run the words hold: its first position and its tokens.
    run: Option<(u32, usize)>,
    /// Every position's forward rope row on the host: `ctx × rope`.
    fwd: Vec<f32>,
    /// Every position's inverse rope row.
    back: Vec<f32>,
}

impl HcaState {
    /// The state of layers `0..layers` — a layer `hca` names holds a cache, a
    /// latent one none — for `d`'s shapes, zeroed. Builds both rope tables'
    /// rows of every position on the host. Load-time only.
    pub fn new(
        stream: &CudaStream,
        d: &HcaDims,
        layers: usize,
        hca: impl Fn(usize) -> bool,
    ) -> Result<HcaState, GpuError> {
        let mut caches = Vec::with_capacity(layers);
        for l in 0..layers {
            caches.push(hca(l).then(|| HcaCache::new(stream, d)).transpose()?);
        }
        let (p, rope, groups, lat) = (d.piece, d.rope, d.max_groups(d.piece), d.head);
        let step_words = d.geom(p, p).words();
        let sc = Scratch {
            normed: DeviceBuffer::zeroed(stream, p * d.hidden)?,
            q_a: DeviceBuffer::zeroed(stream, p * d.q_lora)?,
            q_lat: DeviceBuffer::zeroed(stream, p * d.q_lora)?,
            q: DeviceBuffer::zeroed(stream, p * d.q_width())?,
            qh: DeviceBuffer::zeroed(stream, p * d.q_width())?,
            gate: DeviceBuffer::zeroed(stream, p * d.grouped())?,
            kv: DeviceBuffer::zeroed(stream, p * lat)?,
            comp_kv: DeviceBuffer::zeroed(stream, p * lat)?,
            comp_gate: DeviceBuffer::zeroed(stream, p * lat)?,
            ape: DeviceBuffer::zeroed(stream, p * lat)?,
            score: DeviceBuffer::zeroed(stream, p * lat)?,
            kv_row: DeviceBuffer::zeroed(stream, p * lat)?,
            y: DeviceBuffer::zeroed(stream, p * d.q_width())?,
            xg: DeviceBuffer::zeroed(stream, HcaDims::COLS * d.group_k())?,
            og: DeviceBuffer::zeroed(stream, HcaDims::COLS * d.o_lora)?,
            oa: DeviceBuffer::zeroed(stream, p * d.grouped())?,
            oa_g: DeviceBuffer::zeroed(stream, p * d.grouped())?,
            part_v: DeviceBuffer::zeroed(stream, p * d.heads * d.segs() * LATENT)?,
            part_ms: DeviceBuffer::zeroed(stream, p * d.heads * d.segs() * 2)?,
            ones: DeviceBuffer::from_host(stream, &vec![1.0f32; d.head])?,
            shadow: DeviceTensor::zeroed(stream, 1, LATENT)?,
            pre: DeviceBuffer::zeroed(stream, groups * LATENT)?,
        };
        let words = Words {
            pos: DeviceBuffer::zeroed(stream, p)?,
            vis: DeviceBuffer::zeroed(stream, 2 * p)?,
            step: DeviceBuffer::zeroed(stream, step_words)?,
            cs_fwd: DeviceBuffer::zeroed(stream, p * rope)?,
            cs_back: DeviceBuffer::zeroed(stream, p * rope)?,
            cs_comp: DeviceBuffer::zeroed(stream, groups * rope)?,
            ape_src: DeviceBuffer::zeroed(stream, p * lat)?,
            ape_dst: DeviceBuffer::from_host(stream, &ape_dst(p * lat)?)?,
        };
        let table = RopeTable::new(&d.yarn).map_err(|e| GpuError::plan(WHAT, e.to_string()))?;
        let rows = |dir: Direction| {
            let mut rows = Vec::with_capacity(d.ctx * rope);
            for pos in 0..d.ctx as u32 {
                table.push(pos, dir, &mut rows);
            }
            rows
        };
        Ok(HcaState {
            d: d.clone(),
            caches,
            words,
            staging: DeviceTensor::zeroed(stream, p, LATENT)?,
            sc,
            run: None,
            fwd: rows(Direction::Forward),
            back: rows(Direction::Back),
        })
    }

    /// The shapes the state was built for.
    #[must_use]
    pub fn dims(&self) -> &HcaDims {
        &self.d
    }

    /// The run the words hold, once [`HcaState::set_run`] set one.
    #[must_use]
    pub fn run(&self) -> Option<(u32, usize)> {
        self.run
    }

    /// Every cache back to the load's zeros and no run's words held: the
    /// next run writes the bits a fresh state writes. Enqueued on the stream,
    /// outside a captured graph.
    pub fn reset(&mut self, stream: &CudaStream) -> Result<(), GpuError> {
        for c in self.caches.iter_mut().flatten() {
            c.ring.buf_mut().zero_raw(stream)?;
            c.rows.buf_mut().zero_raw(stream)?;
            c.ring_kv.zero_raw(stream)?;
            c.ring_score.zero_raw(stream)?;
        }
        self.run = None;
        Ok(())
    }

    /// The FNV-1a 64 of every cache's bits, read back (synchronizes `stream`).
    pub fn digest(&self, stream: &CudaStream) -> Result<u64, GpuError> {
        let mut h = crate::body::DIGEST_BASIS;
        for c in self.caches.iter().flatten() {
            for tensor in [&c.ring, &c.rows] {
                let v = tensor.buf().to_host_vec(stream)?;
                h = crate::body::fold(h, v.into_iter().map(u32::from));
            }
            for buf in [&c.ring_kv, &c.ring_score] {
                let v = buf.to_host_vec(stream)?;
                h = crate::body::fold(h, v.into_iter().map(f32::to_bits));
            }
        }
        Ok(h)
    }

    /// The bytes the state holds on the card: its caches, staging, words and
    /// scratch.
    #[must_use]
    pub fn bytes(&self) -> usize {
        let f = |b: &DeviceBuffer<f32>| b.len() * F32;
        let u = |b: &DeviceBuffer<u32>| b.len() * size_of::<u32>();
        let (w, s) = (&self.words, &self.sc);
        self.caches
            .iter()
            .flatten()
            .map(HcaCache::bytes)
            .sum::<usize>()
            + self.staging.rows() * self.staging.cols() * size_of::<u16>()
            + u(&w.pos)
            + u(&w.vis)
            + u(&w.step)
            + f(&w.cs_fwd)
            + f(&w.cs_back)
            + f(&w.cs_comp)
            + f(&s.normed)
            + f(&s.q_a)
            + f(&s.q_lat)
            + f(&s.q)
            + f(&s.qh)
            + f(&s.gate)
            + f(&s.kv)
            + f(&s.comp_kv)
            + f(&s.comp_gate)
            + f(&s.ape)
            + f(&s.score)
            + f(&s.kv_row)
            + f(&s.y)
            + f(&s.xg)
            + f(&s.og)
            + f(&s.oa)
            + f(&s.oa_g)
            + f(&s.part_v)
            + f(&s.part_ms)
            + f(&s.ones)
            + f(&s.pre)
            + s.shadow.rows() * s.shadow.cols() * size_of::<u16>()
    }

    /// Write the words of the run `b .. b + m` — the positions, the visible
    /// counts, the compressor's step layout, the rope rows and the position
    /// table's indices — and remember
    /// it: every HCA layer's launches of that run read them. A run the words
    /// already hold is left as it is, so the layers of one prompt piece share
    /// one write. On the stream, outside a captured graph. A run of no token
    /// or more than a piece, one ending past the context, or one the
    /// compressor's layout refuses is refused by name, the words unchanged.
    pub fn set_run(&mut self, stream: &CudaStream, b: u32, m: usize) -> Result<(), GpuError> {
        if self.run == Some((b, m)) {
            return Ok(());
        }
        let refuse = |detail: String| GpuError::shape(WHAT, detail);
        let d = &self.d;
        if m == 0 || m > d.piece || u64::from(b) + m as u64 > d.ctx as u64 {
            return Err(refuse(format!(
                "a run of {m} tokens from position {b}: 1 ..= {} tokens ending at or before the \
                 context's {}",
                d.piece, d.ctx
            )));
        }
        let (rope, r) = (d.rope, d.ratio);
        let mut pos_h = Vec::with_capacity(m);
        let mut vis_h = Vec::with_capacity(2 * m);
        for t in 0..m {
            let pos = b + t as u32;
            pos_h.push(pos);
            vis_h.push((t + b as usize + 1).min(d.window) as u32);
            vis_h.push(((t + b as usize + 1) / r) as u32);
        }
        // The compressor's step: the blocks the run completes pool the ring's
        // slots of the positions before `b` (the first block's alone), then
        // the run's tokens; the ring then keeps the run's last residues.
        let (p, j0) = (b as usize % r, b as usize / r);
        let groups = (m + p) / r;
        let mut read = Vec::with_capacity(groups * r);
        for g in 0..groups {
            for i in 0..r {
                read.push(if g == 0 && i < p {
                    i
                } else {
                    r + g * r + i - p
                } as u32);
            }
        }
        let kept = m.min(r);
        let mut persist: Vec<(usize, usize)> = (0..kept)
            .map(|i| ((b as usize + m - kept + i) % r, m - kept + i))
            .collect();
        persist.sort_unstable();
        let (src, dst): (Vec<u32>, Vec<u32>) =
            persist.iter().map(|&(s, t)| (t as u32, s as u32)).unzip();
        let write_row: Vec<u64> = (0..groups).map(|g| (j0 + g) as u64).collect();
        let step = StepInts {
            write_row: &write_row,
            read: &read,
            persist_src: &src,
            persist_dst: &dst,
        };
        let geom = d.geom(m, d.piece);
        let mut step_h = vec![0u32; geom.words()];
        geom.pack(&step, &mut step_h).map_err(|e| match e {
            GpuError::Shape { detail, .. } => {
                refuse(format!("the compressor's step of the run: {detail}"))
            }
            e => e,
        })?;
        let mut comp_h = Vec::with_capacity(groups * rope);
        for g in 0..groups {
            let at = (j0 + g) * r * rope;
            comp_h.extend_from_slice(&self.fwd[at..at + rope]);
        }
        let hd = d.head;
        let mut ape_h = Vec::with_capacity(m * hd);
        for t in 0..m {
            let row = (b as usize + t) % r * hd;
            for j in 0..hd {
                ape_h.push(
                    u32::try_from(row + j).map_err(|_| {
                        refuse(format!("a position-table index {} past u32", row + j))
                    })?,
                );
            }
        }
        // The words are sized for a piece; a run of `m` tokens writes the
        // front `m` tokens' words of each and leaves the rest, which no
        // launch of the run reads. The step layout is the piece's whole.
        let w = &mut self.words;
        write_front(stream, &mut w.pos, &pos_h)?;
        write_front(stream, &mut w.vis, &vis_h)?;
        w.step.copy_from_host(stream, &step_h)?;
        let (at, to) = (b as usize * rope, (b as usize + m) * rope);
        write_front(stream, &mut w.cs_fwd, &self.fwd[at..to])?;
        write_front(stream, &mut w.cs_back, &self.back[at..to])?;
        if groups > 0 {
            write_front(stream, &mut w.cs_comp, &comp_h)?;
        }
        write_front(stream, &mut w.ape_src, &ape_h)?;
        self.run = Some((b, m));
        Ok(())
    }
}

/// `host` written to the front `host.len()` values of `buf`. The words are
/// sized for a piece; a run of fewer tokens writes only its own.
fn write_front<T: DeviceCopy>(
    stream: &CudaStream,
    buf: &mut DeviceBuffer<T>,
    host: &[T],
) -> Result<(), GpuError> {
    let mut front = WindowMut::<T>::of_mut(buf, 0, host.len())?;
    Ok(front.copy_from_host(stream, host)?)
}

/// The cache of layer `l`: refused when the layer holds none (a latent layer,
/// or past the file's layers).
fn cache(caches: &mut [Option<HcaCache>], l: usize) -> Result<&mut HcaCache, GpuError> {
    caches.get_mut(l).and_then(Option::as_mut).ok_or_else(|| {
        GpuError::shape(
            WHAT,
            format!("layer {l} holds no HCA cache: a latent layer, or past the file's layers"),
        )
    })
}

/// Every head of `m` tokens, its last `d.rope` values turned.
fn tail(d: &HcaDims, m: usize) -> TailShape {
    TailShape {
        width: d.head,
        n_dims: d.rope,
        n_vec: d.heads,
        m,
    }
}

impl HcaLayer {
    /// Refused unless the state's words hold the run `b .. b + m`, on an
    /// eager launch. A capture records the launches alone, and each replay
    /// reads the words the host set before it (the body's refresh), so a
    /// capture is not refused over the words it finds: the open captures the
    /// step before any run.
    fn words_hold(
        st: &HcaState,
        stream: &CudaStream,
        (b, m): (u32, usize),
    ) -> Result<(), GpuError> {
        if capturing(stream)? {
            return Ok(());
        }
        match st.run {
            Some(run) if run == (b, m) => Ok(()),
            run => Err(GpuError::shape(
                WHAT,
                format!(
                    "a launch of the run {b}..{} over words that hold {run:?}: HcaState::set_run \
                     writes them first",
                    b as usize + m
                ),
            )),
        }
    }

    /// `y = W · x` for the resident weight `name` over `n` columns (the gemv
    /// arm).
    #[allow(
        clippy::too_many_arguments,
        reason = "a projection's kernels, weights, stream, name, input, columns and output (rust-quality R8)"
    )]
    fn project(
        gpu: &Gpu,
        k: &HcaKernels,
        w: &Weights,
        stream: &CudaStream,
        name: &str,
        x: &DeviceBuffer<f32>,
        n: usize,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        site::resident_gemv_cols_on(gpu, stream, Some(&k.g32), (w, name), x, n, y)
    }

    /// One run of `m` tokens at `b .. b + m`, the state's words holding it:
    /// the step's launches (the module doc's) over `x`, `m × hidden` of either
    /// form ([`HcaIn`]: the norm of step 1 is launched for a fold only), into
    /// `out`. `staged`: the run's own rows sit in the staging and are
    /// committed to the ring after the attention (a prompt piece starting at
    /// a multiple of the staging's rows); else each token's row goes into its
    /// ring slot before the attention (the step). Asynchronous,
    /// allocation-free, capturable.
    #[allow(
        clippy::too_many_arguments,
        reason = "one run's kernels, weights, state, stream, run, buffers and form (rust-quality R8)"
    )]
    fn enqueue_run(
        &self,
        gpu: &Gpu,
        k: &HcaKernels,
        w: &Weights,
        st: &mut HcaState,
        (stream, run): (&CudaStream, (u32, usize)),
        x: HcaIn<'_>,
        out: &mut DeviceBuffer<f32>,
        staged: bool,
    ) -> Result<(), GpuError> {
        Self::words_hold(st, stream, run)?;
        let m = run.1;
        let d = &st.d;
        let fault = gpu.layer_sink(self.layer)?;
        let (hidden, q, qw, hd, rope, grouped) =
            (d.hidden, d.q_lora, d.q_width(), d.head, d.rope, d.grouped());
        let (rows, fold) = (x.rows(), matches!(x, HcaIn::Fold(_)));
        if rows.len() < m * hidden || out.len() < m * hidden {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "a run of {m} tokens over x of {} and out of {} values; {} each",
                    rows.len(),
                    out.len(),
                    m * hidden
                ),
            ));
        }
        let (sc, words) = (&mut st.sc, &st.words);
        let gain = |name: &str| w.f32_buf(WHAT, name);
        let (elem, lat) = (gpu.elem(), hd * F32);
        let project =
            |stream: &CudaStream,
             name: &str,
             x: &DeviceBuffer<f32>,
             n: usize,
             y: &mut DeviceBuffer<f32>| Self::project(gpu, k, w, stream, name, x, n, y);

        // 1, 2, 4 and 5: the projections, a column chunk at a time.
        for c in (0..m).step_by(HcaDims::COLS) {
            let n = HcaDims::COLS.min(m - c);
            let x_c = Window::<f32>::of(rows, c * hidden * F32, n * hidden)?;
            let normed = if fold {
                {
                    let mut normed =
                        WindowMut::<f32>::of_mut(&mut sc.normed, c * hidden * F32, n * hidden)?;
                    elem.enqueue_rms_norm(
                        stream,
                        &x_c,
                        gain(&self.norm)?,
                        d.eps,
                        hidden,
                        n,
                        &mut normed,
                    )?;
                }
                Window::<f32>::of(&sc.normed, c * hidden * F32, n * hidden)?
            } else {
                x_c
            };
            {
                let mut q_a = WindowMut::<f32>::of_mut(&mut sc.q_a, c * q * F32, n * q)?;
                project(stream, &self.q_a, &normed, n, &mut q_a)?;
            }
            {
                let q_a = Window::<f32>::of(&sc.q_a, c * q * F32, n * q)?;
                let mut q_lat = WindowMut::<f32>::of_mut(&mut sc.q_lat, c * q * F32, n * q)?;
                elem.enqueue_rms_norm(
                    stream,
                    &q_a,
                    gain(&self.q_a_norm)?,
                    d.eps,
                    q,
                    n,
                    &mut q_lat,
                )?;
            }
            let q_lat = Window::<f32>::of(&sc.q_lat, c * q * F32, n * q)?;
            {
                let mut rows = WindowMut::<f32>::of_mut(&mut sc.q, c * qw * F32, n * qw)?;
                project(stream, &self.q_b, &q_lat, n, &mut rows)?;
            }
            {
                let mut rows =
                    WindowMut::<f32>::of_mut(&mut sc.gate, c * grouped * F32, n * grouped)?;
                project(stream, &self.gate, &q_lat, n, &mut rows)?;
            }
            {
                let mut rows = WindowMut::<f32>::of_mut(&mut sc.kv, c * lat, n * hd)?;
                project(stream, &self.kv, &normed, n, &mut rows)?;
            }
            {
                let mut rows = WindowMut::<f32>::of_mut(&mut sc.comp_kv, c * lat, n * hd)?;
                project(stream, &self.comp_kv, &normed, n, &mut rows)?;
            }
            {
                let mut rows = WindowMut::<f32>::of_mut(&mut sc.comp_gate, c * lat, n * hd)?;
                project(stream, &self.comp_gate, &normed, n, &mut rows)?;
            }
        }

        // 3: the weightless norm of every query head, then its tail's rope.
        {
            let rows = Window::<f32>::of(&sc.q, 0, m * qw)?;
            let mut qh = WindowMut::<f32>::of_mut(&mut sc.qh, 0, m * qw)?;
            elem.enqueue_rms_norm(stream, &rows, &sc.ones, d.eps, hd, m * d.heads, &mut qh)?;
            k.rope
                .enqueue_rope_tail(stream, &mut qh, &words.cs_fwd, tail(d, m))?;
        }

        let geom = HcaDims::geom(&st.d, m, d.piece);
        let cache = cache(&mut st.caches, self.layer)?;
        let planes = w.e4m3(WHAT, &self.output_a)?;

        // 4: the latent rows — norm, tail rope, f16 — into the ring (the
        // step), or into the staging (a piece).
        {
            let kv = Window::<f32>::of(&sc.kv, 0, m * hd)?;
            let mut kv_row = WindowMut::<f32>::of_mut(&mut sc.kv_row, 0, m * hd)?;
            let cache_rows = if staged {
                &mut st.staging
            } else {
                &mut cache.ring
            };
            k.rope.enqueue_kv_norm_rope_append(
                stream,
                KvAppendArgs {
                    kv: &kv,
                    gain: gain(&self.kv_norm)?,
                    cs: &words.cs_fwd,
                    pos: &words.pos,
                    eps: d.eps,
                    n_dims: rope,
                    m,
                    out: &mut kv_row,
                    cache: cache_rows,
                    shadow: &mut sc.shadow,
                    fault,
                },
            )?;
        }

        // 5: the pooling scores with the position table's rows, then the pool.
        // The rows are gathered by the words' indices, so a captured step
        // replays at any position.
        k.step.enqueue_gather(
            stream,
            gain(&self.comp_ape)?,
            &words.ape_src,
            &words.ape_dst,
            m * hd,
            &mut sc.ape,
        )?;
        elem.enqueue_add(stream, &sc.comp_gate, &sc.ape, m * hd, &mut sc.score)?;
        k.compress.enqueue_pool(
            stream,
            PoolArgs {
                geom,
                step: &words.step,
                kv: &sc.comp_kv,
                score: &sc.score,
                gain: gain(&self.comp_norm)?,
                cs: &words.cs_comp,
                eps: d.eps,
                n_dims: rope,
                ring_kv: &mut cache.ring_kv,
                ring_score: &mut cache.ring_score,
                pre: &mut sc.pre,
                cache: &mut cache.rows,
                fault,
            },
        )?;

        // 6: the attention over the ring ⧺ the staging, or the ring.
        {
            let q = Window::<f32>::of(&sc.qh, 0, m * qw)?;
            let mut y = WindowMut::<f32>::of_mut(&mut sc.y, 0, m * qw)?;
            let sinks = gain(&self.sinks)?;
            if staged {
                let args = AttnArgs {
                    q: &q,
                    window: &cache.ring,
                    compressed: Some(&cache.rows),
                    selected: None,
                    vis: &words.vis,
                    sinks,
                    scale: d.scale,
                    tokens: m,
                    heads: d.heads,
                    part_v: &mut sc.part_v,
                    part_ms: &mut sc.part_ms,
                    y: &mut y,
                    fault,
                };
                let staged = Staged {
                    rows: &st.staging,
                    base: &words.pos,
                };
                k.attn.enqueue_staged(stream, args, staged)?;
                drop(y);
                k.attn.enqueue_commit(
                    stream,
                    CommitArgs {
                        staged,
                        tokens: m,
                        ring: &mut cache.ring,
                    },
                )?;
            } else {
                let args = AttnArgs {
                    q: &q,
                    window: &cache.ring,
                    compressed: Some(&cache.rows),
                    selected: None,
                    vis: &words.vis,
                    sinks,
                    scale: d.scale,
                    tokens: m,
                    heads: d.heads,
                    part_v: &mut sc.part_v,
                    part_ms: &mut sc.part_ms,
                    y: &mut y,
                    fault,
                };
                k.attn.enqueue(stream, args)?;
            }
        }

        // 7: the head mix, then the inverse rope.
        {
            let mut y = WindowMut::<f32>::of_mut(&mut sc.y, 0, m * qw)?;
            k.vha
                .enqueue_mix(stream, &self.mix, &mut y, d.heads, hd, m, fault)?;
            k.rope
                .enqueue_rope_tail(stream, &mut y, &words.cs_back, tail(d, m))?;
        }

        // 8: the grouped output, a column chunk and a group at a time — each
        // group's input gathered out of the head rows, its block of the
        // weight's rows projected, its outputs scattered into the grouped row.
        let (group_k, g_bytes) = (d.group_k(), d.group_k() * F32);
        let o_bytes = d.o_lora * F32;
        for c in (0..m).step_by(HcaDims::COLS) {
            let n = HcaDims::COLS.min(m - c);
            for g in 0..d.o_groups {
                dtod_2d(
                    stream,
                    (&sc.y, (c * qw + g * group_k) * F32),
                    (&mut sc.xg, 0),
                    Pitched {
                        width: g_bytes,
                        rows: n,
                        dev_pitch: qw * F32,
                        host_pitch: g_bytes,
                    },
                )?;
                let qs = DeviceTensor::window_of::<u32>(
                    planes.qs.buf(),
                    g * d.o_lora * planes.qs.cols() * size_of::<u32>(),
                    d.o_lora,
                    planes.qs.cols(),
                )?;
                let e = DeviceTensor::window_of::<u32>(
                    planes.e.buf(),
                    g * d.o_lora * planes.e.cols() * size_of::<u32>(),
                    d.o_lora,
                    planes.e.cols(),
                )?;
                gpu.q8f32().enqueue_e4m3_gemv(
                    stream,
                    E4m3GemvArgs {
                        w: E4m3Weight {
                            qs: &qs,
                            e: &e,
                            scales: planes.scales,
                        },
                        x: &sc.xg,
                        m: n,
                        out: GemvOut::TokenMajor,
                        y: &mut sc.og,
                    },
                )?;
                dtod_2d(
                    stream,
                    (&sc.og, 0),
                    (&mut sc.oa, (c * grouped + g * d.o_lora) * F32),
                    Pitched {
                        width: o_bytes,
                        rows: n,
                        dev_pitch: o_bytes,
                        host_pitch: grouped * F32,
                    },
                )?;
            }
        }

        // The gate on the grouped rows, then `attn_output_b`.
        {
            let oa = Window::<f32>::of(&sc.oa, 0, m * grouped)?;
            let gate = Window::<f32>::of(&sc.gate, 0, m * grouped)?;
            let mut oa_g = WindowMut::<f32>::of_mut(&mut sc.oa_g, 0, m * grouped)?;
            k.out_gate.enqueue_out_gate(
                stream,
                OutGateArgs {
                    kind: OutGate::Proj,
                    attn: &oa,
                    gate: &gate,
                    n_head: grouped / GATE_HEAD,
                    m,
                    fault,
                    y: &mut oa_g,
                },
            )?;
        }
        for c in (0..m).step_by(HcaDims::COLS) {
            let n = HcaDims::COLS.min(m - c);
            let oa_g = Window::<f32>::of(&sc.oa_g, c * grouped * F32, n * grouped)?;
            let mut rows = WindowMut::<f32>::of_mut(out, c * hidden * F32, n * hidden)?;
            project(stream, &self.output_b, &oa_g, n, &mut rows)?;
        }
        Ok(())
    }

    /// The step at `pos`: the module doc's launches over the token's row `x`
    /// (either form, [`HcaIn`]) into `out`, the row into its ring slot first. An eager step's words
    /// must hold the run (`pos`, 1) ([`HcaState::set_run`]); a capture reads no position, its
    /// replays the words set before each. Asynchronous, allocation-free, capturable.
    #[allow(
        clippy::too_many_arguments,
        reason = "the step's kernels, weights, state, stream, position and buffers (rust-quality R8)"
    )]
    pub fn enqueue_step(
        &self,
        gpu: &Gpu,
        k: &HcaKernels,
        w: &Weights,
        st: &mut HcaState,
        stream: &CudaStream,
        pos: u32,
        x: HcaIn<'_>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        self.enqueue_run(gpu, k, w, st, (stream, (pos, 1)), x, out, false)
    }

    /// The prompt form: `tokens` tokens at `pos0 .. pos0 + tokens` over `x`
    /// into `out`, the step at every position — in staged pieces from the
    /// first multiple of the staging's rows, one token at a time before it.
    /// Writes each run's words on the stream, outside a captured graph.
    /// Asynchronous, allocation-free.
    #[allow(
        clippy::too_many_arguments,
        reason = "the prompt's kernels, weights, state, stream, place, length and buffers (rust-quality R8)"
    )]
    pub fn enqueue_prompt(
        &self,
        gpu: &Gpu,
        k: &HcaKernels,
        w: &Weights,
        st: &mut HcaState,
        stream: &CudaStream,
        pos0: u32,
        tokens: usize,
        x: HcaIn<'_>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let hidden = st.d.hidden;
        let rows = x.rows();
        if tokens == 0 || rows.len() < tokens * hidden || out.len() < tokens * hidden {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "a prompt of {tokens} tokens over x of {} and out of {} values; {} each",
                    rows.len(),
                    out.len(),
                    tokens * hidden
                ),
            ));
        }
        let piece = st.d.piece;
        let (mut b, mut done) = (pos0, 0usize);
        let off = pos0 as usize % piece;
        let head = if off == 0 { 0 } else { tokens.min(piece - off) };
        for i in 0..head {
            st.set_run(stream, b, 1)?;
            let x_i = Window::<f32>::of(rows, (done + i) * hidden * F32, hidden)?;
            let mut out_i = WindowMut::<f32>::of_mut(out, (done + i) * hidden * F32, hidden)?;
            self.enqueue_run(
                gpu,
                k,
                w,
                st,
                (stream, (b, 1)),
                x.with(&x_i),
                &mut out_i,
                false,
            )?;
            b += 1;
        }
        done += head;
        while done < tokens {
            let m = piece.min(tokens - done);
            st.set_run(stream, b, m)?;
            let x_m = Window::<f32>::of(rows, done * hidden * F32, m * hidden)?;
            let mut out_m = WindowMut::<f32>::of_mut(out, done * hidden * F32, m * hidden)?;
            self.enqueue_run(
                gpu,
                k,
                w,
                st,
                (stream, (b, m)),
                x.with(&x_m),
                &mut out_m,
                true,
            )?;
            b += m as u32;
            done += m;
        }
        Ok(())
    }
}
