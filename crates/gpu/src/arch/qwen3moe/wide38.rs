//! Qwen3.8's ubatch walk ([`Gemm38`]): a prompt's positions as units of rows
//! ([`UnitRows38`], one unit of all `m` rows) through
//! [`runtime::sched::walk`]'s `(units, cols, Batch)` — the front, the shared
//! expert in the shadow of the host's service, the serve, the back — through
//! the host tier's batch port at the unit's width, every op over the unit's
//! rows at once:
//! - every Q8_0 projection (the gated-residual mixes' down and up, the PLE
//!   key and value, the delta layer's q·k·v, `z` and output, the selecting
//!   layer's q, k, v and output, the shared expert's gate, up and down) the
//!   32-value GEMM (`crate::gemm`'s `gemm_q8_0p`) over the walk's one-expert
//!   table, filled once per walk, its input quantized per 32 values
//!   (`quantize_gemm32`; the shared expert's down from `swiglu_quant32`);
//! - the F32 matrices (β and α, the indexer's key and query) the wide F32
//!   tile, bit for bit the decode path's gemv for each row and token;
//! - a dense site of a type that reads q8_1 rows (a K-quant: the file's own
//!   type of that site) has no wide GEMM here: it projects by name in chunks
//!   of [`BY_NAME_COLS`] columns ([`project_chunks`]: the quantization, the
//!   K-quant gemv and the token-major copy of each chunk), each column bit
//!   for bit the step's one-column launch, and the weight is read once a
//!   chunk. On a layer with such a site in its shared expert the three
//!   projections run so, the SwiGLU once over every row;
//! - the router the ubatch instance of the 512-expert router, in runs of
//!   [`ROUTE_ROWS`] tokens, bit for bit the fused launch for each token;
//! - the routed experts the slot map puts on the card, on a layer with card
//!   experts, the card route ([`CardRoute38`]): per run of [`ROUTE_ROWS`]
//!   tokens, or of the whole unit on a plan that gave the route that run
//!   (`place::card_route_run`), in the shadow after the front's download,
//!   the compressed ids and the places from the walk's ids, the remapped
//!   route table, the gate and up
//!   GEMMs the kernel table names for the layer's type
//!   (`crate::card_leg::StackW::route`: Q4_K, Q5_K, IQ3_XXS or IQ4_XS over
//!   the run's q8_1 rows; NVFP4's planes over its 32-value rows), the card
//!   slots' SwiGLU (`swiglu_quant32_sel`; with every card gate, up and down
//!   in NVFP4's planes on a native FP4 build the gate, the up, the SwiGLU
//!   and its native FP4 are one launch,
//!   `Gemm32Kernels::enqueue_gate_up_sel32`), the down GEMM of its type (the
//!   file's Q5_1, Q8_0 or IQ4_NL blocks, or NVFP4's planes) and the card sum
//!   into the unit's acc — the host tier
//!   serves the rest of the slots, and the back adds the card sum on a
//!   layer with card experts (`q38_card_shared_add`); on a layer without a
//!   host leg the shared expert runs first and each run's card sum is the
//!   block's output with its gated rows added (`q38_card_sum_shared_add`),
//!   no acc, no host sum and nothing left for the back; when every layer
//!   runs so over NVFP4 downs in the native FP4 form, the down rows are
//!   BF16 (`Down38`);
//! - the selecting layer's rows the prefill flash while `qsa::scored` says a
//!   row selects nothing (every key below its count), the selection and the
//!   prefill flash over each row's list in runs of the arena's selection rows
//!   (`scratch38::select_run`: as many as the plan's selection bytes hold at
//!   the load's cache) from the first row that selects; the pool pass over every row first, so a pool some row of the
//!   unit completes is written before any row selects; both flashes store
//!   each value under its sigmoid gate (the `_gated` entries, the out gate
//!   in the flash's store);
//! - the conv, the delta step (the committed lane, in place), the gated
//!   norm, the PLE gate and conv, the key append, the rope with the cache
//!   append and the gated sum the launches the other walks
//!   run, which take any row count;
//! - every routed expert the host union over the unit's columns.
//!
//! Units. The walk cuts its rows into units ([`unit_rows`]): one unit of
//! every row, or — the split walk ([`SplitWalk`], `BLOOMERY_SPLIT_WALK`) —
//! two of `⌈m/2⌉` and `⌊m/2⌋` rows where the width rule
//! (`runtime::splitwalk::pays` at [`SPLIT_COSTS`]) says the second unit's
//! front under the first unit's host service pays for its union call; a load
//! with an expert tier or an NVMe-paged plan walks one unit, and a split
//! asked of it is refused by name. A part reads and writes a unit's rows
//! through windows of them ([`UnitRows38`]) at the unit's first row: the
//! embedding's ids, rows, positions and key counts, the residual streams,
//! the PLE rows, the feed-forward input, the routed slots and weights, the
//! mixers' and the block's outputs, the shared expert's output, the card
//! sum, the host sums and the mixes' combine weights (which one site leaves
//! for the next, a part apart: [`UnitHc`]). A unit also carries its width,
//! first position, rows that select nothing and one-expert table (the
//! second unit its own, filled for its width). The
//! walk declares [`BatchOrder::Fronts`]: both units' first fronts, then per
//! (layer, unit) the shadow, the serve, the back and the unit's next front,
//! so the second unit's front runs on the card while the host waits for the
//! first unit's download and serves its union. What a part holds
//! only inside its own launches (the mixes' and the mixers' scratch, the
//! activations' quantizations, the router's, the selection's and the card
//! route's runs) is one set the units share: one unit's part runs its
//! launches before another's begin. The exchange with the host is the one
//! reader of whole buffers: the port's key names the rows `at .. u` of the
//! arena's `ffn_x`, routed ids and weights and of its own sets alike, so the
//! front's download takes the arena's buffers, not the unit's windows.
//! Qwen3.8's gated-DeltaNet layers carry a recurrent state and a conv ring
//! per layer and its selecting layers carry pooled keys, so a layer's front
//! reads what the front of the unit before it left in the layer's store: the
//! fronts of a layer run in unit order.
//!
//! The last layer's unread rows. On the walk's last layer, when it runs
//! without a host leg and no route tap records it, a unit runs every row
//! through the mixer's projections and store writes (the key, value and
//! indexer-key appends, the pool; a delta layer's whole mixer) and the rest
//! of the layer (the flash, the gate, the output projection, the
//! feed-forward block) over its rows among the walk's last ones something
//! reads ([`WideParts::kept`]: the head's row, a checkpoint's point row, a
//! tap's rows) alone, the unit cut to them ([`UnitRows38::narrow`]) and its
//! one-expert table filled for them. A cut row keeps its bits (the numeric
//! class below); the other rows' streams stay as the layer before left them.
//!
//! Numeric class. Every op computes a token's values from that token's
//! inputs alone — the GEMM sums each (slot, row) in an order fixed by K, the
//! quantizers work a column at a time, the flash walks a row's keys in fixed
//! tiles, the conv reads a predecessor from the ring or the unit's own rows
//! with the same bits, the host union serves each column as its one-column
//! call does — so a token's bits depend neither on its neighbours' values
//! nor on the ubatch it lands in: a prompt walked as one ubatch or as
//! several leaves the same bits. Against the pass the Q8_0 projections read
//! q8 activations (32 values a scale) where the pass reads the f32 row, and
//! the unselected rows' attention runs the prefill flash, so the two agree
//! to the error of that quantization and are not bit-equal; the card route
//! quantizes the same blocks the pass's card leg quantizes (a q8_1 of 128
//! values a scale for the gate·up's input, a 32-value q8_1 of the SwiGLU),
//! so the two card legs differ only in their sums' order.
//!
//! The walk carries the residual streams as BF16 values (ik and mainline
//! keep them as F32 tensors, [`Streams`]), in the halfwords of the arena's
//! other stream buffer ([`Arena38::res`]): each value a mix's combine or init
//! or the PLE site writes is rounded to nearest even once, and each mix
//! normalizes the values the streams hold. The PLE site reads the combined streams as F32
//! values, and the last layer's combine writes them as F32 values into the
//! final buffer ([`combine`]), which the head's mix and every reader of the
//! target's rows read. The pass keeps F32 streams; against it the walk's
//! streams differ by those roundings too.
//!
//! The walk runs the card route ([`CardRoute38`]) on a layer with card
//! experts, and holds no card buffers on a map without them.
//!
//! On a load with an expert tier (plan (b′)'s 3090, `tier38`) a layer the
//! tier holds experts of downloads the unit's tier places beside its
//! routing, and the batch port's tier leg runs the tier's block route over
//! the tier slots in the host union's shadow, their rows packed by rank and
//! copied into the set's host-mapped rows by the tier's copy engine. The
//! card route there keeps its card slots' down rows and places for the whole
//! unit and sums nothing; the back, once the host has seen the tier's service
//! complete ([`BatchLeg::join_tiered`]), ranks the same places
//! (`q38_tier_rank`) and sums the card's and the tier's slots in slot order
//! (`q38_card_tier_acc`),
//! then the host sums are uploaded and combined as on any card layer. With
//! the union of both cards' experts on one card the same launches read the
//! same values in the same order, so the two loads agree bit for bit. A slot
//! map with an expert on a tier card on a load that hung no tier is refused
//! by name at the walk's entry (`card38`), before anything moves.
//!
//! Under host streaming (`BLOOMERY_XSTREAM=admit|split`, a residency
//! machine, a prompt call fed by ubatches) every walk of the call moves each
//! card layer's residency pool toward the experts the unit routes most, at
//! that layer: in the shadow, the host waits for the front's download,
//! counts the unit's routed ids and the machine's pick
//! (`SwapMachine::call_pick`, the least count admitted [`STREAM_FLOOR`] under
//! `admit`, the stream rule's floor under `split`) sends the pool's coldest
//! residents to the host and copies the hottest host experts over them, in
//! victim-slot order a landing batch a third of them, each batch's event on
//! the copy stream after its last copy; the card route runs a GEMM batch
//! over each batch's slots behind its own event — the slots no copy lands
//! in first, waiting nothing — so the copies and the route's GEMMs pipeline
//! instead of the route waiting for them all, and the serve's union and the
//! card route both run under the moved map. Under `split` the expert stream
//! ([`crate::host::xstream`]) then sends the host experts its rule picks to
//! a half of its ring for the layer's units, its copies batched the same way,
//! and the card route runs a second set of its launches over that half
//! ([`CardRoute38::enqueue`]'s ring pass, a GEMM batch a landing batch)
//! before the card sum, which reads the stack's places and the ring's
//! together; the serve's union leaves the streamed experts out. The route is
//! then the layer's reader of the call ([`PromptStream::read`]), which the
//! next ubatch's pick and the ring's next use of the half copy behind. A
//! split walk picks once a layer, at its first unit, from that unit's
//! counts at twice (the group's width): the second unit shares the pick's
//! landings and the stream's ring and exclusion
//! (`HostTier::xstream_share`), its card route runs over the pick the first
//! unit held for it, and its read is the layer's last; the layer's return is
//! owed at the last unit's back, once every unit's host service ran. The
//! call's placement stays for the decode after it, unless the call returns to
//! its start (a decode history before it): then the call's last unit returns
//! each layer once its host service has run, asked at the next layer's
//! shadow, whose download wait the call's queued return copies fill
//! ([`PromptStream::return_at_shadow`]); a split pick at the last walk's last
//! card layer prices an admit's way back, which no front follows
//! ([`runtime::xsplit::ReturnCost`]), and the call's end issues what is left. The stream's placement never stays. A
//! token's bits then depend on its ubatch's routing too — which of its
//! experts the pick moved or the stream sent sums on the card, not the host
//! — so the numeric class above holds for an unstreamed walk only.

use super::body::ATTN_SCALE_256;
use super::card38::Card38;
use super::image::MediaRows;
use super::plan38::{GDN, GdnPlan, HcSite, Layer38, Mixer38, QsaPlan, geo, head_site};
use super::program38::{BA_PITCH, Ctx38, beta_alpha_rows};
use super::router::gated;
use super::scratch::{Append256, Io, KvPlanes, PrefillFlash, RecStore, cells_view, param_view};
use super::scratch38::{Arena38, Ple38, ROUTE_ROWS, Store38, final_res, no_carry};
use super::tier38::TierSide38;
use super::wide::{RingStacks, with_ring_windows};
use crate::GpuError;
use crate::card_leg::{RouteW, StackW};
use crate::card_slots::{CardAcc, DownRows};
use crate::fault::{FaultSink, LAYER_HEAD};
use crate::flash_gqa::GqaListArgs;
use crate::flash_gqa_prefill::QueryNorm;
use crate::gemm::{
    DownW, DownWin, GateUp, Gemm32Weight, GemmAct, GemmAct32, GemmInput, GemmRoute, MmaForm,
    enqueue_gemm_batch,
};
use crate::hc_gated::{Before, HcMatrix, HcWideScratch, SiteWeights, Streams, WideMixArgs};
use crate::head::Head;
use crate::host::BatchLeg;
use crate::host::handoff::Places;
use crate::host::prompt_stream::PromptStream;
use crate::host::run::HostRun;
use crate::host::xstream::{GROUP_WALK_ORDER, LandBatch, RingLayer};
use crate::kquant::Act;
use crate::linear::conv::{ConvArgs, ConvRows};
use crate::linear::delta::{DeltaArgs, DeltaLanesArgs};
use crate::model::lookup::{f32_gain, f32_tensor};
use crate::ple::{PleConvArgs, PleGateArgs};
use crate::prompt_timing::Mark;
use crate::q38::{
    CardSharedAddArgs, CardSumSharedAddArgs, CardTierAccArgs, EmbedIo, KeyAppendArgs, SLOTS,
    SharedAddArgs, W_PITCH,
};
use crate::qsa::{self, PoolArgs, SelectArgs};
use crate::site::{
    self, BY_NAME_COLS, ByName, Form, Gated, GatedIo, Out, SiteWeight, SwigluRows, WideIn,
    WideKernels,
};
use crate::tensor::{DeviceTensor, Window, WindowMut};
use cuda_core::{CudaEvent, CudaStream, DeviceBuffer};
use gguf::quant::GgmlType;
use runtime::hc_gated::Geometry;
use runtime::sched::{self, At, BatchOrder, LayerProgram, Overlap, PortKind};
use runtime::splitwalk::{self, LoadKind, SplitCosts};

/// What the walk's errors name.
const WHAT: &str = "qwen4exp ubatch walk";

/// The indexer keys a select keeps, as `qsa::scored` takes them.
const KEPT_U32: u32 = geo::KEPT as u32;
const _: () = assert!(KEPT_U32 as usize == geo::KEPT);

/// How a ubatch walk cuts its rows into units (`BLOOMERY_SPLIT_WALK`,
/// [`super::body38::Body38::set_split_walk`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SplitWalk {
    /// One unit of every row: the walk the lever's `off` keeps.
    Off,
    /// Two units of `⌈m/2⌉` and `⌊m/2⌋` rows where the width rule pays
    /// (`runtime::splitwalk::pays` at [`SPLIT_COSTS`]), one unit otherwise.
    Rule,
    /// Two units at every walk of two rows or more, whatever the rule says:
    /// a gate's arm.
    Every,
}

/// The width rule's costs on a card of [`SPLIT_COSTS_SMS`] streaming
/// multiprocessors: the card's time a row of one layer's front on the
/// A6000, and the high edge of the band the prompt-union records fit for one
/// more union call a layer, its cost per listed expert and the second
/// front's enqueue — the edge where a split pays least, so the rule splits
/// only a walk that pays at every fit in the band. A card of another count
/// scales the card's time by its multiprocessors ([`Wide38::new`]; its clock
/// is not read).
pub const SPLIT_COSTS: SplitCosts = SplitCosts {
    card_row_us: 7.07,
    call_us: 2600.0,
    listing_us: 5.7,
    enqueue_us: 920.0,
};

/// The streaming multiprocessors [`SPLIT_COSTS`]' card time is the time of.
const SPLIT_COSTS_SMS: f64 = 84.0;

/// The rows of a selecting layer's unit of `m` rows from position `pos0`
/// that select nothing — every key below their count, the prefill flash's
/// rows — as `qsa::scored` decides: the leading rows it refuses. The
/// predicate is monotone in the count below the cache's rows, so every row
/// past them selects. One answer a walk, the same at every selecting layer.
pub(super) fn dense_rows(pos0: usize, m: usize, ctx: usize) -> Result<usize, GpuError> {
    let ctx = u32::try_from(ctx).map_err(|_| GpuError::shape(WHAT, format!("ctx {ctx}")))?;
    let mut dense = 0;
    while dense < m {
        let count = u32::try_from(pos0 + dense + 1)
            .map_err(|_| GpuError::shape(WHAT, format!("position {}", pos0 + dense)))?;
        if qsa::scored(count, ctx, KEPT_U32) {
            break;
        }
        dense += 1;
    }
    Ok(dense)
}

/// A non-owning window over a buffer, every one of [`UnitRows38`]'s: it frees
/// nothing and dies with the unit, giving back its context handle.
type Win<T> = WindowMut<'static, T>;

/// The rows `r0 .. r0 + m` of a buffer of `stride` words a row, as the offset
/// and the length in words of the window over them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Span {
    off: usize,
    len: usize,
}

impl Span {
    const fn rows(r0: usize, m: usize, stride: usize) -> Span {
        Span {
            off: r0 * stride,
            len: m * stride,
        }
    }
}

/// Where each of a unit's windows lies ([`UnitRows38`]): its rows `r0 .. r0 +
/// m` of each buffer it windows, one field a buffer. The one place a buffer's
/// words a row meet a unit's first row.
#[derive(Clone, Copy, Debug)]
struct UnitSpans {
    /// The input record's ids, the embedding's rows, and each row's position
    /// and live key count.
    tok: Span,
    emb: Span,
    pos: Span,
    n_keys: Span,
    /// Each of the two residual-stream buffers.
    res: Span,
    ple_e: Span,
    ffn_x: Span,
    /// The routed slots and weights, `N_USED + 1` a token.
    ids: Span,
    weights: Span,
    /// The mixer's and the block's outputs, the shared expert's output, the
    /// card sum and the host sums.
    y: Span,
    sh_y: Span,
    acc: Span,
    hsum: Span,
    /// The wide mixes' combine weights, `STREAMS` a token: what a unit's
    /// inject site leaves for its next site's combine, across its parts.
    wgt: Span,
}

impl UnitSpans {
    /// The spans of the unit of `m` rows from row `r0`.
    const fn of(r0: usize, m: usize) -> UnitSpans {
        let (hidden, wide, slots) = (geo::HIDDEN, geo::STREAMS * geo::HIDDEN, geo::N_USED + 1);
        UnitSpans {
            tok: Span::rows(r0, m, 1),
            emb: Span::rows(r0, m, hidden),
            pos: Span::rows(r0, m, 1),
            n_keys: Span::rows(r0, m, 1),
            res: Span::rows(r0, m, wide),
            ple_e: Span::rows(r0, m, hidden),
            ffn_x: Span::rows(r0, m, hidden),
            ids: Span::rows(r0, m, slots),
            weights: Span::rows(r0, m, slots),
            y: Span::rows(r0, m, hidden),
            sh_y: Span::rows(r0, m, hidden),
            acc: Span::rows(r0, m, hidden),
            hsum: Span::rows(r0, m, hidden),
            wgt: Span::rows(r0, m, geo::STREAMS),
        }
    }
}

/// A walk of `m` rows cut into units of `widths` rows, each as its first row
/// and its width, in order. Refused by name when there is no unit, a unit has
/// no row, or the widths do not add up to `m`.
fn unit_rows(m: usize, widths: &[usize]) -> Result<Vec<(usize, usize)>, GpuError> {
    if widths.is_empty() || widths.contains(&0) || widths.iter().sum::<usize>() != m {
        return Err(GpuError::shape(
            WHAT,
            format!("a walk of {m} rows cut into units of {widths:?} rows"),
        ));
    }
    Ok(widths
        .iter()
        .scan(0, |r0, &w| {
            let at = (*r0, w);
            *r0 += w;
            Some(at)
        })
        .collect())
}

/// The leading rows of the unit of `m` rows from row `r0` that select
/// nothing, given the walk's `dense` leading rows that do: the predicate is
/// monotone in a row's key count ([`dense_rows`]), so the rows that select
/// nothing are a prefix of the walk, cut by the unit.
const fn unit_dense(dense: usize, r0: usize, m: usize) -> usize {
    let rest = dense.saturating_sub(r0);
    if rest < m { rest } else { m }
}

/// The rows of the unit of `m` rows from row `r0` at or past the walk's row
/// `from`: its trailing rows among the walk's last ones, a suffix of the
/// unit, as many as lie in it.
const fn unit_kept(from: usize, r0: usize, m: usize) -> usize {
    let first = if from > r0 { from } else { r0 };
    (r0 + m).saturating_sub(first)
}

/// A split under `w` refused by name on a load the width rule does not
/// price (`runtime::splitwalk::refuse`): one with an expert tier or an
/// NVMe-paged plan. [`SplitWalk::Off`] is taken on every load.
fn refuse_split(load: LoadKind, w: SplitWalk) -> Result<(), GpuError> {
    if w == SplitWalk::Off {
        return Ok(());
    }
    splitwalk::refuse(load)
        .map_err(|e| GpuError::shape("BLOOMERY_SPLIT_WALK", format!("{w:?}: {e}")))
}

/// The widths of a walk of `m` rows cut under `w` on a load of kind `load`
/// whose rule costs are `costs`, `host` the most host-resident routed
/// experts a layer holds ([`WideParts::widths`]): `⌈m/2⌉` and `⌊m/2⌋` when
/// the cut splits, else `m`. A split on a load the rule does not price is
/// refused by name ([`refuse_split`]).
fn unit_widths(
    m: usize,
    w: SplitWalk,
    (load, costs): (LoadKind, &SplitCosts),
    host: usize,
) -> Result<Vec<usize>, GpuError> {
    let two = m >= 2
        && match w {
            SplitWalk::Off => false,
            SplitWalk::Every => true,
            SplitWalk::Rule => {
                host > 0
                    && splitwalk::pays(m, costs, host, geo::N_USED)
                        .map_err(|e| GpuError::shape(WHAT, format!("{m} rows: {e}")))?
            }
        };
    if !two {
        return Ok(vec![m]);
    }
    refuse_split(load, w)?;
    Ok(vec![m.div_ceil(2), m / 2])
}

/// The window of `at` over `parent`, refused by name when `parent` ends
/// before it. The window is valid while `parent` lives and stays in place: a
/// unit holds its windows no longer than the walk that cut them (module doc,
/// "Units").
fn rows_of<T>(name: &str, parent: &DeviceBuffer<T>, at: Span) -> Result<Win<T>, GpuError> {
    window_of(name, parent, at)
}

/// [`rows_of`]'s window with `at` counted in `T`s over `parent`'s bytes.
fn window_of<P, T>(name: &str, parent: &DeviceBuffer<P>, at: Span) -> Result<Win<T>, GpuError> {
    let room = parent.len() * size_of::<P>() / size_of::<T>();
    if at.off + at.len > room {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "{name}: a unit's window ends at element {} of {room}",
                at.off + at.len
            ),
        ));
    }
    let ptr = parent.cu_deviceptr() + (at.off * size_of::<T>()) as u64;
    // SAFETY: the window's elements lie inside `parent`'s bytes (checked
    // above), which the walk that holds the window keeps alive and in place.
    Ok(unsafe { WindowMut::from_raw(ptr, at.len, parent.context()) })
}

/// One unit of a walk: its `m` rows from the walk's row `r0`, as windows of
/// the buffers a part reads and writes a token at a time ([`UnitSpans`]) and
/// the scalars a part reads of its unit (module doc, "Units").
struct UnitRows38 {
    r0: usize,
    m: usize,
    /// The position of the unit's first row.
    pos0: usize,
    /// The unit's leading rows that select nothing ([`dense_rows`]).
    dense: usize,
    /// The walk's one-expert table ([`Wide38`]'s `dense`) the unit's width
    /// fills.
    route: usize,
    /// Where the unit's ids start in the input record, the embedding's
    /// `first`.
    first: usize,
    tok: Win<u32>,
    emb: Win<f32>,
    pos: Win<u32>,
    n_keys: Win<u32>,
    /// The unit's streams as BF16 values, and its rows of the final buffer
    /// ([`Arena38::res`]): the PLE site's input, then the last streams.
    streams: Win<u16>,
    fin: Win<f32>,
    ple_e: Win<f32>,
    ffn_x: Win<f32>,
    ids: Win<u32>,
    weights: Win<f32>,
    y: Win<f32>,
    sh_y: Win<f32>,
    /// The card route's sum, on a load whose slot map puts routed experts on
    /// the card.
    acc: Option<Win<f32>>,
    hsum: Win<f32>,
    /// The unit's rows of the wide mixes' combine weights ([`UnitHc`]).
    wgt: Span,
    /// The unit's rows of a selecting layer that select nothing (the prefill
    /// flash's) and those that select, once a selecting layer's front ran.
    split: Option<(usize, usize)>,
}

impl UnitRows38 {
    /// The unit of `m` rows from row `r0` of the walk `p`, its dense table
    /// `route`, over the leg's host sums `hsum`: every window is the unit's
    /// span ([`UnitSpans::of`]) of its buffer, refused by name where the
    /// buffer ends before it, and the unit starts from the walk's stream
    /// buffer. The arena, the walk's own buffers, the input record's ids and
    /// `hsum` stay in place while the walk runs, which the unit lives no
    /// longer than.
    fn new(
        p: &WideParts<'_>,
        hsum: &DeviceBuffer<f32>,
        (r0, m): (usize, usize),
        route: usize,
    ) -> Result<UnitRows38, GpuError> {
        let (s, x) = (&*p.s, &*p.x);
        if route >= x.dense.len() {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "a unit's one-expert table {route} of the walk's {}",
                    x.dense.len()
                ),
            ));
        }
        let at = UnitSpans::of(r0, m);
        // The combine weights' window is made at each mix ([`UnitHc`]); its
        // rows are checked here, before any launch.
        rows_of("the mixes' combine weights", &x.hc.wgt, at.wgt)?;
        let fin = final_res(p.plans);
        Ok(UnitRows38 {
            r0,
            m,
            pos0: p.pos0 + r0,
            dense: unit_dense(p.dense, r0, m),
            route,
            first: p.io.first + r0,
            tok: rows_of("the record's ids", p.io.ids, at.tok)?,
            emb: rows_of("emb", &s.emb, at.emb)?,
            pos: rows_of("pos", &s.pos, at.pos)?,
            n_keys: rows_of("n_keys", &s.n_keys, at.n_keys)?,
            streams: window_of("the streams", &s.res[1 - fin], at.res)?,
            fin: rows_of("the final streams", &s.res[fin], at.res)?,
            ple_e: rows_of("ple.e", &s.ple.e, at.ple_e)?,
            ffn_x: rows_of("ffn_x", &s.ffn_x, at.ffn_x)?,
            ids: rows_of("ids", &x.ids, at.ids)?,
            weights: rows_of("weights", &x.weights, at.weights)?,
            y: rows_of("y", &s.y, at.y)?,
            sh_y: rows_of("sh_y", &s.sh_y, at.sh_y)?,
            acc: x
                .card
                .as_ref()
                .map(|r| rows_of("the card sum", &r.acc, at.acc))
                .transpose()?,
            hsum: rows_of("hsum", hsum, at.hsum)?,
            wgt: at.wgt,
            split: None,
        })
    }

    /// The unit cut to its last `keep` rows: each window over those rows of
    /// its own ([`UnitSpans::of`] from row `m − keep`), the scalars moved with
    /// them. A row keeps its place in every buffer, so a part over the cut
    /// unit reads and writes each of its rows as over the whole unit. Refused
    /// by name past the unit's rows.
    fn narrow(&mut self, keep: usize) -> Result<(), GpuError> {
        let t0 = self.m.checked_sub(keep).ok_or_else(|| {
            GpuError::shape(WHAT, format!("a unit of {} rows cut to {keep}", self.m))
        })?;
        let at = UnitSpans::of(t0, keep);
        self.tok = rows_of("the record's ids", &self.tok, at.tok)?;
        self.emb = rows_of("emb", &self.emb, at.emb)?;
        self.pos = rows_of("pos", &self.pos, at.pos)?;
        self.n_keys = rows_of("n_keys", &self.n_keys, at.n_keys)?;
        self.streams = rows_of("the streams", &self.streams, at.res)?;
        self.fin = rows_of("the final streams", &self.fin, at.res)?;
        self.ple_e = rows_of("ple.e", &self.ple_e, at.ple_e)?;
        self.ffn_x = rows_of("ffn_x", &self.ffn_x, at.ffn_x)?;
        self.ids = rows_of("ids", &self.ids, at.ids)?;
        self.weights = rows_of("weights", &self.weights, at.weights)?;
        self.y = rows_of("y", &self.y, at.y)?;
        self.sh_y = rows_of("sh_y", &self.sh_y, at.sh_y)?;
        if let Some(acc) = self.acc.as_mut() {
            *acc = rows_of("the card sum", acc, at.acc)?;
        }
        self.hsum = rows_of("hsum", &self.hsum, at.hsum)?;
        self.wgt = UnitSpans::of(self.r0 + t0, keep).wgt;
        self.dense = unit_dense(self.dense, t0, keep);
        self.r0 += t0;
        self.pos0 += t0;
        self.first += t0;
        self.m = keep;
        Ok(())
    }
}

/// The wide mixes' scratch with its combine weights on one unit's rows: a
/// mix reads the weights the unit's last inject site left and writes its own
/// there, so two units' weights never meet, whatever parts of the other unit
/// run between a unit's two sites. Dropped, the scratch holds its own
/// weights again.
struct UnitHc<'s> {
    hc: &'s mut HcWideScratch,
    /// The unit's window with the scratch's own buffer swapped into it, the
    /// window's buffer standing in `hc` meanwhile; the drop swaps them back.
    own: Option<Win<f32>>,
}

impl<'s> UnitHc<'s> {
    /// `hc` with the unit's rows `wgt` of its combine weights in place of
    /// the whole buffer; refused by name past the buffer's rows.
    fn on(hc: &'s mut HcWideScratch, wgt: Span) -> Result<UnitHc<'s>, GpuError> {
        let mut own = rows_of("the mixes' combine weights", &hc.wgt, wgt)?;
        std::mem::swap(&mut hc.wgt, &mut *own);
        Ok(UnitHc { hc, own: Some(own) })
    }
}

impl Drop for UnitHc<'_> {
    fn drop(&mut self) {
        if let Some(mut own) = self.own.take() {
            // The scratch takes its own buffer back; the window, its own
            // buffer again, frees nothing as it drops.
            std::mem::swap(&mut self.hc.wgt, &mut *own);
        }
    }
}

/// A gate's route taps of a ubatch walk: each layer's router logits (its
/// `logits()` a token) and routed slots (`N_USED + 1` a token) for up to
/// `rows` tokens, and the positions the last walk wrote them for. Allocated
/// when armed, never on the walk's path otherwise.
pub(super) struct WideTaps {
    pub(super) rows: usize,
    pub(super) logits: Vec<DeviceBuffer<f32>>,
    pub(super) ids: Vec<DeviceBuffer<u32>>,
    /// The router's logits a token.
    pub(super) width: usize,
    /// The last walk's first position and rows, set once it has walked
    /// every layer: `None` while nothing since the arming (or a reset) did.
    pub(super) walked: Option<(usize, usize)>,
}

impl WideTaps {
    /// Zeroed taps of `layers` layers for up to `rows` tokens of `width`
    /// logits. Gate use.
    pub(super) fn new(
        stream: &CudaStream,
        layers: usize,
        rows: usize,
        width: usize,
    ) -> Result<WideTaps, GpuError> {
        Ok(WideTaps {
            rows,
            logits: (0..layers)
                .map(|_| DeviceBuffer::zeroed(stream, rows * width))
                .collect::<Result<Vec<_>, _>>()?,
            ids: (0..layers)
                .map(|_| DeviceBuffer::zeroed(stream, rows * (geo::N_USED + 1)))
                .collect::<Result<Vec<_>, _>>()?,
            width,
            walked: None,
        })
    }

    /// Device bytes.
    pub(super) fn bytes(&self) -> usize {
        self.logits
            .iter()
            .map(DeviceBuffer::num_bytes)
            .sum::<usize>()
            + self.ids.iter().map(DeviceBuffer::num_bytes).sum::<usize>()
    }
}

/// A gate's planted routes ([`super::body38::Body38::plant_ubatch_routes`]):
/// each layer's router logits of the positions `pos0 .. pos0 + rows`,
/// [`geo::EXPERTS`] a token, which the walk's routing launch reads in place
/// of the router's own. Allocated when planted, never on the walk's path
/// otherwise.
pub(super) struct WideForce {
    pos0: usize,
    rows: usize,
    logits: Vec<DeviceBuffer<f32>>,
}

impl WideForce {
    /// `routes[t][l]`'s logits for the positions from `pos0`, uploaded;
    /// refused by name unless every position holds a route for each of
    /// `layers` layers with [`geo::EXPERTS`] logits. Gate use.
    pub(super) fn new(
        stream: &CudaStream,
        layers: usize,
        pos0: usize,
        routes: &[Vec<super::body38::RouteTap>],
    ) -> Result<WideForce, GpuError> {
        let rows = routes.len();
        if rows == 0 {
            return Err(GpuError::shape(WHAT, "planted routes of no position"));
        }
        let mut host = vec![Vec::with_capacity(rows * geo::EXPERTS); layers];
        for (t, by_layer) in routes.iter().enumerate() {
            if by_layer.len() != layers {
                return Err(GpuError::shape(
                    WHAT,
                    format!(
                        "planted routes of position {}: {} layers, the chain has {layers}",
                        pos0 + t,
                        by_layer.len()
                    ),
                ));
            }
            for (l, r) in by_layer.iter().enumerate() {
                if r.logits.len() != geo::EXPERTS {
                    return Err(GpuError::shape(
                        WHAT,
                        format!(
                            "planted route of position {} at layer {l}: {} logits, want {}",
                            pos0 + t,
                            r.logits.len(),
                            geo::EXPERTS
                        ),
                    ));
                }
                host[l].extend_from_slice(&r.logits);
            }
        }
        Ok(WideForce {
            pos0,
            rows,
            logits: host
                .iter()
                .map(|h| DeviceBuffer::from_host(stream, h))
                .collect::<Result<Vec<_>, _>>()?,
        })
    }

    /// Refused by name unless the positions `pos .. pos + n` lie inside the
    /// planted ones.
    pub(super) fn covers(&self, pos: usize, n: usize) -> Result<(), GpuError> {
        if pos < self.pos0 || pos + n > self.pos0 + self.rows {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "a ubatch of positions {pos}..{} outside the planted routes' {}..{}",
                    pos + n,
                    self.pos0,
                    self.pos0 + self.rows
                ),
            ));
        }
        Ok(())
    }
}

/// The ubatch walk's card route: the routed experts the slot map puts on the
/// card, through the grouped GEMMs, in runs of the plan's run (the walk's
/// module doc). Holds its own buffers for one run of `run` tokens and the
/// walk-wide card sum; the stacks are the leg's ([`Card38`]'s), read through
/// [`Card38::stacks`], and
/// the route tables are one per distinct card count (the placement's spread
/// keeps every eligible layer's count within one, so a plan of this program
/// holds at most two).
///
/// Stream order is a contract: the route is enqueued in the walk's shadow,
/// after the front's download of the unit's activations and slots — enqueued
/// before the download, the d2h would queue behind the route's card GEMMs on
/// the one stream and the host tier would start serving late by their whole
/// time, turning the layer's max(host, card) into a sum.
pub(super) struct CardRoute38 {
    /// The tokens one run takes (`place::card_route_run`).
    pub(super) run: usize,
    /// The compressed routed ids, ten a run token: the places launch over
    /// the walk's ids (a pitch of eleven) with an identity map, whose bytes
    /// are the remapped route's input.
    ids: DeviceBuffer<u32>,
    /// The slots' places, ten a run token — ten a token of the whole unit on
    /// a load with an expert tier: the places launch over the same ids with
    /// the slot map's layer row.
    sel: DeviceBuffer<u32>,
    /// The run's normed rows (`[run][HIDDEN]`), the gate and up GEMMs'
    /// input, in the one form the load's card gate and up read ([`RouteX`]).
    x: RouteX,
    /// The gate and up GEMMs' outputs, ten slots a run token of [`geo::FF`]
    /// values, slot-major.
    g: DeviceBuffer<f32>,
    u: DeviceBuffer<f32>,
    /// The card slots' SwiGLU in the 32-value GEMM's activation form, ten
    /// columns a run token, in the native FP4 form alone where one launch
    /// runs the gate and up (`GemmAct32::new_gate_up_output`); a host slot's
    /// column is never written and never read (the route leaves it to the
    /// host tier).
    act: GemmAct32,
    /// The down GEMM's output, ten slots a run token of [`geo::HIDDEN`]
    /// values, slot-major — a token of the whole unit on a load with an
    /// expert tier, whose join reads every run's rows after the host's wait.
    down: Down38,
    /// The card slots' weighted sums, a row a token of the whole walk; a
    /// unit's sums are its window of them ([`UnitRows38`]'s `acc`).
    acc: DeviceBuffer<f32>,
    /// On a load with an expert tier, the unit's tier places, ten a token:
    /// each slot's place on the tier ([`crate::hybrid::HOST`] off it), which
    /// the front downloads beside the routing and the join reads.
    tsel: Option<DeviceBuffer<u32>>,
    /// Beside the tier places, each slot's rank among the unit's tier slots
    /// (`q38_tier_rank`): the row of the tier's packed rows the join reads.
    trank: Option<DeviceBuffer<u32>>,
    /// One route table per distinct card count, in `counts`' order.
    routes: Vec<GemmRoute>,
    counts: Vec<usize>,
    /// The identity map over the experts (`e -> e`): the ids compression's
    /// stand-in for the places rule's map.
    ids_map: DeviceBuffer<u32>,
    /// A landing batch's places, the batched stack pass's (a batch at a
    /// time; `sel` holds the full run's for the card sum).
    bsel: DeviceBuffer<u32>,
    /// The ring pass's buffers once the load armed the expert stream
    /// ([`CardRoute38::arm_ring`]).
    ring: Option<RingRoute38>,
}

/// Where a card route's sums go ([`CardRoute38::enqueue`]), `[m][HIDDEN]`
/// each.
pub(super) enum CardSum<'b> {
    /// A tier layer: the sum is the join's, after the host's wait
    /// ([`CardRoute38::enqueue_tier_acc`]); the route keeps the places and
    /// the down rows of every run for it and sums nothing.
    Join,
    /// The unit's card sum, which the back combines with the host's sum.
    Acc(&'b mut DeviceBuffer<f32>),
    /// A layer the load runs without a host leg ([`Card38::hostless`]): each
    /// run's sum combined with the shared expert's output `sh` into the
    /// block's output `y` in the same launch (`q38_card_sum_shared_add`),
    /// where the back would add the zeros of the host's empty list.
    Block {
        sh: &'b DeviceBuffer<f32>,
        y: &'b mut DeviceBuffer<f32>,
    },
}

/// A card route's down rows ([`CardRoute38`]'s): BF16 bits on a load whose
/// every layer runs without a host leg over NVFP4 downs in the native FP4
/// form, which the down product stores (`enqueue_gemm32_bf16`) and the
/// block's card sum reads; F32 on any other load.
enum Down38 {
    F32(DeviceBuffer<f32>),
    Bf16(DeviceBuffer<u16>),
}

impl Down38 {
    /// Cells `off .. off + len` as a run's window.
    ///
    /// # Safety
    ///
    /// [`cells_view`]'s over the rows.
    unsafe fn window(&self, off: usize, len: usize) -> DownWin {
        match self {
            // SAFETY: the caller's contract.
            Self::F32(d) => DownWin::F32(unsafe { cells_view(d, off, len) }),
            // SAFETY: the caller's contract.
            Self::Bf16(d) => DownWin::Bf16(unsafe { cells_view(d, off, len) }),
        }
    }

    fn rows(&self) -> DownRows<'_> {
        match self {
            Self::F32(d) => DownRows::F32(d),
            Self::Bf16(d) => DownRows::Bf16(d),
        }
    }
}

impl DownWin {
    fn rows(&self) -> DownRows<'_> {
        match self {
            Self::F32(d) => DownRows::F32(d),
            Self::Bf16(d) => DownRows::Bf16(d),
        }
    }
}

/// The ring pass's buffers: the route table over a half of the expert
/// stream's ring, and per run the slots' places in the half (the ring row's)
/// and in the card sum's places (the union row's), ten a run token.
struct RingRoute38 {
    route: GemmRoute,
    sel: DeviceBuffer<u32>,
    union: DeviceBuffer<u32>,
}

/// Layer `l`'s down stack `w` of type `ty` as the route's down GEMM reads it
/// over the SwiGLU's 32-value rows ([`StackW::route`]); a stack no 32-value
/// GEMM reads is refused by name as `what`'s.
pub(super) fn down_weight<'a>(
    what: &'static str,
    l: usize,
    ty: GgmlType,
    w: StackW<'a>,
) -> Result<Gemm32Weight<'a>, GpuError> {
    match w.route(ty) {
        Some(RouteW::Blocks32(w)) => Ok(w),
        Some(RouteW::Grouped(..)) | None => Err(GpuError::shape(
            what,
            format!("layer {l}: a down stack of {ty}, which no route GEMM reads"),
        )),
    }
}

/// The run's normed rows in the gate and up GEMMs' input form, one a load
/// ([`Card38::blocks32_gate_up`]): the K-quant GEMM's activation form (the
/// same q8_1 bytes a `Q8Act` of the same values holds), or the 32-value
/// GEMM's where that GEMM reads the card gate and up, the native FP4 form
/// alone where one native launch runs the gate and up
/// (`GemmAct32::new_native_input`). The plan counts a run of the q8_1 form,
/// the larger, or of the native form (`place::RouteForm`).
enum RouteX {
    Q8_1(Box<GemmAct>),
    Blocks32(Box<GemmAct32>),
}

/// One GEMM batch of a route pass over `map` — the batch's places, a
/// host-mapped row of a landing batch ([`LandBatch`]) or the card's own map
/// copy for the whole of a pass nothing batched — with the engine stream
/// `stream` waiting `wait` first (the batch's landed event; `None` for a
/// batch nothing lands in): the places into `sel`, `geo::N_USED + 1` columns
/// a token, then the batch sequence itself
/// ([`crate::gemm::enqueue_gemm_batch`]) over the same `map` and `sel`.
#[allow(
    clippy::too_many_arguments,
    reason = "the batch's context, wait, ids, map, weights, width, places bound, sink, run rows and buffers (rust-quality R8)"
)]
fn enqueue_places_batch(
    c: &Ctx38<'_>,
    stream: &CudaStream,
    wait: Option<&CudaEvent>,
    (walk_ids, ids): (&DeviceBuffer<u32>, &DeviceBuffer<u32>),
    map: &DeviceBuffer<u32>,
    gate_up: GateUp<'_>,
    down: Gemm32Weight<'_>,
    (n, n_places): (usize, usize),
    sink: FaultSink,
    down_y: &mut DownWin,
    (g, u, act): (
        &mut DeviceBuffer<f32>,
        &mut DeviceBuffer<f32>,
        &mut GemmAct32,
    ),
    (sel, route): (&mut DeviceBuffer<u32>, &mut GemmRoute),
) -> Result<(), GpuError> {
    if let Some(e) = wait {
        stream.wait(e)?;
    }
    c.k.handoff.enqueue_places_cols(
        stream,
        &Places {
            ids: walk_ids,
            map,
            row_off: 0,
            n_expert: map.len(),
        },
        geo::N_USED + 1,
        n,
        sink,
        sel,
    )?;
    enqueue_gemm_batch(
        (&c.k.gemm, &c.k.g32),
        stream,
        ids,
        map,
        gate_up,
        DownW::Blocks32(down),
        (n, n_places),
        (geo::N_USED, geo::FF, geo::HIDDEN),
        sink,
        down_y,
        (g, u, act),
        (sel, route),
    )
}

impl CardRoute38 {
    /// The route's buffers for ubatches of up to `rows` tokens over `card`'s
    /// card experts: a run of `run` tokens (`place::card_route_run`: at most
    /// `rows`, and its slots one route table's), the sums a token of the
    /// whole unit, and one route table per distinct card count
    /// (refused by name past two, naming them); `tiered`, a load with an
    /// expert tier, the places and the down rows a token of the whole unit
    /// and the unit's tier places beside them
    /// (`place::card_tier_join_bytes`). Load-time only.
    pub(super) fn new(
        stream: &CudaStream,
        (rows, run): (usize, usize),
        card: &Card38,
        tiered: bool,
    ) -> Result<CardRoute38, GpuError> {
        let counts = card.card_counts();
        if counts.len() > 2 {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "a card route over {} distinct card counts {:?}: the walk's route tables are \
                     built for two (the placement's spread keeps every count within one)",
                    counts.len(),
                    counts
                ),
            ));
        }
        let slots = run * geo::N_USED;
        let kept = if tiered { rows * geo::N_USED } else { slots };
        let all_nvfp4 = card
            .routed_types()
            .all(|(gate, down)| gate == GgmlType::NVFP4 && down == GgmlType::NVFP4);
        let act = if card.blocks32_gate_up() && all_nvfp4 {
            GemmAct32::new_gate_up_output(stream, slots, geo::FF)?
        } else {
            GemmAct32::new_routed_for(
                stream,
                slots,
                geo::FF,
                card.routed_types().map(|(_, down)| down),
            )?
        };
        let bf16_down = !tiered
            && card.all_hostless()
            && act.has_native_form()
            && card.routed_types().all(|(_, down)| down == GgmlType::NVFP4);
        let fused = act.has_native_form() && card.blocks32_gate_up() && all_nvfp4;
        let product_cells = if fused { 0 } else { slots * geo::FF };
        Ok(CardRoute38 {
            run,
            ids: DeviceBuffer::zeroed(stream, slots)?,
            sel: DeviceBuffer::zeroed(stream, kept)?,
            x: if fused {
                RouteX::Blocks32(Box::new(GemmAct32::new_native_input(
                    stream,
                    run,
                    geo::HIDDEN,
                )?))
            } else if card.blocks32_gate_up() {
                RouteX::Blocks32(Box::new(GemmAct32::new_routed_for(
                    stream,
                    run,
                    geo::HIDDEN,
                    card.routed_types().map(|(gate, _)| gate),
                )?))
            } else {
                RouteX::Q8_1(Box::new(GemmAct::new(stream, run, geo::HIDDEN)?))
            },
            g: DeviceBuffer::zeroed(stream, product_cells)?,
            u: DeviceBuffer::zeroed(stream, product_cells)?,
            act,
            down: if bf16_down {
                Down38::Bf16(DeviceBuffer::zeroed(stream, kept * geo::HIDDEN)?)
            } else {
                Down38::F32(DeviceBuffer::zeroed(stream, kept * geo::HIDDEN)?)
            },
            acc: DeviceBuffer::zeroed(stream, rows * geo::HIDDEN)?,
            tsel: tiered
                .then(|| DeviceBuffer::zeroed(stream, rows * geo::N_USED))
                .transpose()?,
            trank: tiered
                .then(|| DeviceBuffer::zeroed(stream, rows * geo::N_USED))
                .transpose()?,
            routes: counts
                .iter()
                .map(|&n| GemmRoute::new(stream, slots, n))
                .collect::<Result<_, _>>()?,
            counts,
            ids_map: DeviceBuffer::from_host(
                stream,
                &(0..geo::EXPERTS as u32).collect::<Vec<_>>(),
            )?,
            bsel: DeviceBuffer::zeroed(stream, kept)?,
            ring: None,
        })
    }

    /// The ring pass's buffers for a half of `half_slots` slots of the
    /// expert stream's ring: its route table and the run's two places.
    /// Load-time only; a second arming is refused by name.
    pub(super) fn arm_ring(
        &mut self,
        stream: &CudaStream,
        half_slots: usize,
    ) -> Result<(), GpuError> {
        if self.ring.is_some() {
            return Err(GpuError::state(
                WHAT,
                "a card route the ring is not armed on",
            ));
        }
        let slots = self.run * geo::N_USED;
        self.ring = Some(RingRoute38 {
            route: GemmRoute::new(stream, slots, half_slots)?,
            sel: DeviceBuffer::zeroed(stream, slots)?,
            union: DeviceBuffer::zeroed(stream, slots)?,
        });
        Ok(())
    }

    /// Device bytes.
    pub(super) fn bytes(&self) -> usize {
        self.ids.num_bytes()
            + self.sel.num_bytes()
            + match &self.x {
                RouteX::Q8_1(x) => x.bytes(),
                RouteX::Blocks32(x) => x.bytes(),
            }
            + self.g.num_bytes()
            + self.u.num_bytes()
            + self.act.bytes()
            + match &self.down {
                Down38::F32(d) => d.num_bytes(),
                Down38::Bf16(d) => d.num_bytes(),
            }
            + self.acc.num_bytes()
            + self.tsel.as_ref().map_or(0, DeviceBuffer::num_bytes)
            + self.trank.as_ref().map_or(0, DeviceBuffer::num_bytes)
            + self.routes.iter().map(GemmRoute::bytes).sum::<usize>()
            + self.ids_map.num_bytes()
            + self.bsel.num_bytes()
            + self.ring.as_ref().map_or(0, |r| {
                r.route.bytes() + r.sel.num_bytes() + r.union.num_bytes()
            })
    }

    /// The route table's index for the experts' count `n_card`, refused by
    /// name when the route holds none for it.
    fn table_at(&self, n_card: usize) -> Result<usize, GpuError> {
        self.counts
            .iter()
            .position(|&n| n == n_card)
            .ok_or_else(|| {
                GpuError::shape(
                    WHAT,
                    format!(
                        "a card route table of {n_card} experts; the walk holds {:?}",
                        self.counts
                    ),
                )
            })
    }

    /// Enqueue layer `l`'s card route over the unit's `m` tokens (module
    /// doc): per run of at most `run` tokens, the compressed ids and the
    /// full places from the unit's ids (`ids`, eleven slots a token, the
    /// shared expert's last — never read), the q8_1 of the run's normed rows
    /// (`ffn_x`, `[m][HIDDEN]`; their 32-value form for a gate and up the
    /// 32-value GEMM reads), then the stack pass as one GEMM batch a
    /// landing batch of the layer's pick (`land`, [`enqueue_places_batch`]):
    /// the leading batch the slots no copy lands in, which wait nothing,
    /// then the admits' slots in slot order, each batch behind its own
    /// landed event — or, when the pick moved nothing, the whole stack at
    /// once over the slot map's layer row (`slots`) as the pass always ran.
    /// The card sum by the router's weights (`weights`, eleven a token)
    /// for the run reads the full places, into `sum` ([`CardSum`]): the
    /// unit's acc, or on a layer without a host leg the block's output with
    /// the shared expert's gated rows added in the same launch. On a tier
    /// layer ([`CardSum::Join`]) the card sum is the join's
    /// ([`CardRoute38::enqueue_tier_acc`]): the places and the down rows
    /// kept at the run's tokens of the unit.
    ///
    /// With the layer's stream (`ring`, a half of the expert stream's ring)
    /// each run then runs the ring pass before the card sum, one GEMM batch
    /// a landing batch of the stream's copies — each behind its own event,
    /// the route table and the gate and up GEMMs over the batch's map row,
    /// the SwiGLU of the batch's streamed slots and the down GEMM — and the
    /// card sum reads the union places at the stack's experts plus the
    /// streamed ones, so the stack's and the ring's slots sum in slot order
    /// in one launch. Refused by name: a layer without card experts, a
    /// stream on a tier layer, on a layer of NVFP4 stacks (the ring holds
    /// the file's blocks) and on a route the load did not arm
    /// ([`CardRoute38::arm_ring`]), a gate and up or a down no route GEMM
    /// reads ([`StackW::route`]), and a gate and up the 32-value GEMM reads
    /// on a route made with no 32-value rows. Asynchronous, allocation-free,
    /// capturable.
    #[allow(
        clippy::too_many_arguments,
        reason = "the route's ids, weights, normed rows, the card sum, the slot map, the layer, the unit's width, the stream and the pick's batches (rust-quality R8)"
    )]
    pub(super) fn enqueue(
        &mut self,
        c: &Ctx38<'_>,
        card: &Card38,
        l: usize,
        (ffn_x, ids, weights, mut sum): (
            &DeviceBuffer<f32>,
            &DeviceBuffer<u32>,
            &DeviceBuffer<f32>,
            CardSum<'_>,
        ),
        slots: &DeviceTensor<u32>,
        m: usize,
        ring: Option<&RingLayer>,
        land: &[LandBatch],
    ) -> Result<(), GpuError> {
        let st = card.stacks(c.w, l)?;
        let n_card = st.n_card;
        let down = down_weight(WHAT, l, st.down_ty, st.down)?;
        let (gpu, stream, sink) = (c.gpu, c.gpu.stream(), c.gpu.layer_sink(l)?);
        let pitch = geo::N_USED + 1;
        // SAFETY: layer l's row of the slot map's card copy (`EXPERTS` words
        // a row), which stays resident while the window lives (this layer's
        // route launches).
        let map = unsafe { param_view::<u32>(slots.buf(), l * geo::EXPERTS, geo::EXPERTS) };
        let kept = self.tsel.is_some();
        let tiered = matches!(sum, CardSum::Join);
        let words = match &sum {
            CardSum::Join => m * geo::HIDDEN,
            CardSum::Acc(acc) => acc.len(),
            CardSum::Block { sh, y } => sh.len().min(y.len()),
        };
        if (tiered && !kept) || m * geo::HIDDEN > words {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "layer {l}: a card route of {m} tokens{} into sums of {} with kept rows \
                     {kept}",
                    if tiered { " on a tier layer" } else { "" },
                    words / geo::HIDDEN
                ),
            ));
        }
        let gate_up = match (
            st.gate.route(st.gate_up_ty),
            st.up.route(st.gate_up_ty),
            &self.x,
        ) {
            (Some(gate @ RouteW::Grouped(..)), Some(up @ RouteW::Grouped(..)), RouteX::Q8_1(_))
            | (
                Some(gate @ RouteW::Blocks32(_)),
                Some(up @ RouteW::Blocks32(_)),
                RouteX::Blocks32(_),
            ) => [gate, up],
            _ => {
                return Err(GpuError::shape(
                    WHAT,
                    format!(
                        "layer {l}: a card route over a gate and an up of {}, which the route \
                         was not made to read",
                        st.gate_up_ty
                    ),
                ));
            }
        };
        let ring_cols = match (st.gate, st.up, st.down) {
            (StackW::Words(gate), StackW::Words(up), StackW::Words(down)) => {
                Some([gate.cols(), up.cols(), down.cols()])
            }
            _ => None,
        };
        if ring.is_some() && (tiered || self.ring.is_none() || ring_cols.is_none()) {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "layer {l}: a stream of the expert stream's ring on a {}",
                    if tiered {
                        "tier layer (the ring serves a load with no expert tier)"
                    } else if ring_cols.is_none() {
                        "layer of NVFP4 stacks (the ring holds the file's blocks)"
                    } else {
                        "card route the load did not arm for it"
                    }
                ),
            ));
        }
        let ctx = gpu.context();
        let mut run = |windows: Option<(&DeviceBuffer<u32>, &RingStacks<'_>)>| {
            let mut c0 = 0;
            while c0 < m {
                let n = self.run.min(m - c0);
                // The places and the down rows of the run: from the run's
                // first token of the unit when the route keeps them for the
                // join, else from the run buffers' start.
                let at = if kept { c0 } else { 0 };
                // SAFETY: tokens c0 .. c0 + n <= m of the unit's ids and
                // weights (`m · pitch` each) and its `ffn_x` (`m · HIDDEN`);
                // the places and the down rows hold ten slots a token of the
                // unit when kept, else of the run, so at .. at + n lies inside
                // them; every buffer stays in place for this run's launches.
                let (ids_w, w_w, x_w, mut sel_w, mut down_w) = unsafe {
                    (
                        param_view::<u32>(ids, c0 * pitch, n * pitch),
                        cells_view(weights, c0 * pitch, n * pitch),
                        cells_view(ffn_x, c0 * geo::HIDDEN, n * geo::HIDDEN),
                        param_view::<u32>(&self.sel, at * geo::N_USED, n * geo::N_USED),
                        self.down.window(
                            at * geo::N_USED * geo::HIDDEN,
                            n * geo::N_USED * geo::HIDDEN,
                        ),
                    )
                };
                let places = |map: &DeviceBuffer<u32>, out: &mut DeviceBuffer<u32>| {
                    c.k.handoff.enqueue_places_cols(
                        stream,
                        &Places {
                            ids: &ids_w,
                            map,
                            row_off: 0,
                            n_expert: geo::EXPERTS,
                        },
                        pitch,
                        n,
                        sink,
                        out,
                    )
                };
                places(&self.ids_map, &mut self.ids)?;
                let table = self.table_at(n_card)?;
                let gate_up = match (gate_up, &mut self.x) {
                    ([RouteW::Grouped(ty, gate), RouteW::Grouped(_, up)], RouteX::Q8_1(x)) => {
                        gpu.enqueue_quantize_gemm(&x_w, n, x, sink)?;
                        GateUp::Grouped(ty, [gate, up], x)
                    }
                    ([RouteW::Blocks32(gate), RouteW::Blocks32(up)], RouteX::Blocks32(x32)) => {
                        c.k.g32
                            .enqueue_quantize_gemm32(stream, &x_w, n, x32, sink)?;
                        GateUp::Blocks32([gate, up], x32)
                    }
                    _ => {
                        return Err(GpuError::state(
                            WHAT,
                            "a card route checked for its gate and up's form at its entry",
                        ));
                    }
                };
                // The stack pass: one GEMM batch a landing batch of the
                // layer's pick, the leading one (nothing lands in it) first,
                // or the whole stack at once when the pick moved nothing.
                if land.is_empty() {
                    enqueue_places_batch(
                        c,
                        stream,
                        None,
                        (&ids_w, &self.ids),
                        &map,
                        gate_up,
                        down,
                        (n, n_card),
                        sink,
                        &mut down_w,
                        (&mut self.g, &mut self.u, &mut self.act),
                        (&mut sel_w, &mut self.routes[table]),
                    )?;
                } else {
                    // The full places beside the batches: the card sum's
                    // (the join's on a tier layer).
                    places(&map, &mut sel_w)?;
                    for b in land {
                        b.with_map(ctx, geo::EXPERTS, |bmap| {
                            enqueue_places_batch(
                                c,
                                stream,
                                b.event.as_deref(),
                                (&ids_w, &self.ids),
                                bmap,
                                gate_up,
                                down,
                                (n, n_card),
                                sink,
                                &mut down_w,
                                (&mut self.g, &mut self.u, &mut self.act),
                                (&mut self.bsel, &mut self.routes[table]),
                            )
                        })?;
                    }
                }
                // The card sum's places and experts: the stack's, or with a
                // stream the union row's over the stack's and the ring's.
                let mut sum_places = (&*sel_w, n_card);
                if let (Some(r), Some((union_map, w)), Some(rr), GateUp::Grouped(ty, _, x)) =
                    (ring, windows, self.ring.as_mut(), gate_up)
                {
                    places(union_map, &mut rr.union)?;
                    for b in &r.batches {
                        b.with_map(ctx, geo::EXPERTS, |bmap| {
                            enqueue_places_batch(
                                c,
                                stream,
                                b.event.as_deref(),
                                (&ids_w, &self.ids),
                                bmap,
                                GateUp::Grouped(ty, [w.gate, w.up], x),
                                down_weight(WHAT, l, st.down_ty, StackW::Words(w.down))?,
                                (n, r.n),
                                sink,
                                &mut down_w,
                                (&mut self.g, &mut self.u, &mut self.act),
                                (&mut rr.sel, &mut rr.route),
                            )
                        })?;
                    }
                    sum_places = (&rr.union, n_card + r.n);
                }
                match &mut sum {
                    CardSum::Join => {}
                    CardSum::Acc(acc) => {
                        // SAFETY: tokens c0 .. c0 + n <= m of the unit's card
                        // sum (`m · HIDDEN`, checked at entry), which stays in
                        // place for this run's launch.
                        let mut acc_w =
                            unsafe { cells_view(acc, c0 * geo::HIDDEN, n * geo::HIDDEN) };
                        let a = CardAcc {
                            down: down_w.rows().f32(WHAT)?,
                            w: &w_w,
                            sel: sum_places.0,
                            n: geo::HIDDEN,
                            m: n,
                            n_card: sum_places.1,
                            n_used: SLOTS,
                            w_pitch: W_PITCH,
                            fault: sink,
                        };
                        c.k.slots.enqueue_card_acc(stream, &a, &mut acc_w)?;
                    }
                    CardSum::Block { sh, y } => {
                        // SAFETY: tokens c0 .. c0 + n <= m of the unit's
                        // shared expert rows and block output (`m · HIDDEN`
                        // each, checked at entry), which stay in place for
                        // this run's launch.
                        let (sh_w, mut y_w) = unsafe {
                            (
                                cells_view(sh, c0 * geo::HIDDEN, n * geo::HIDDEN),
                                cells_view(y, c0 * geo::HIDDEN, n * geo::HIDDEN),
                            )
                        };
                        c.k.q38.enqueue_card_sum_shared_add(
                            stream,
                            CardSumSharedAddArgs {
                                down: down_w.rows(),
                                w: &w_w,
                                sel: sum_places.0,
                                sh: &sh_w,
                                n: geo::HIDDEN,
                                m: n,
                                n_card: sum_places.1,
                                fault: sink,
                                y: &mut y_w,
                            },
                        )?;
                    }
                }
                c0 += n;
            }
            Ok(())
        };
        match (ring, ring_cols) {
            (Some(r), Some(cols)) => {
                let rows = [geo::FF, geo::FF, geo::HIDDEN];
                with_ring_windows(r, ctx, geo::EXPERTS, rows, cols, |union_map, w| {
                    run(Some((union_map, w)))
                })
            }
            _ => run(None),
        }
    }

    /// The unit's tier places, for the front's places launch and the
    /// tiered download; refused by name on a route made for no tier.
    fn tsel_mut(&mut self) -> Result<&mut DeviceBuffer<u32>, GpuError> {
        self.tsel.as_mut().ok_or(GpuError::state(
            WHAT,
            "a card route made for an expert tier",
        ))
    }

    /// Enqueue tier layer `l`'s join sum over the unit's `m` tokens, after
    /// the host's wait for the tier (`q38_card_tier_acc`): the card's slots
    /// — the kept places and down rows of every run — and the tier's — the
    /// unit's tier places, below `n_tier`, the tier's packed rows `trows` at
    /// their ranks, which `q38_tier_rank` takes first over the same places
    /// the tier ranked — in slot order by the router's `weights` (eleven a
    /// token) into the unit's `acc`, which the back's combine reads as it
    /// reads the card sum. Two launches. A route made for no tier is refused
    /// by name.
    #[allow(
        clippy::too_many_arguments,
        reason = "the walk's context, the layer's card and tier counts, the unit's width, the router's weights, the tier's rows and the unit's card sum (rust-quality R8)"
    )]
    fn enqueue_tier_acc(
        &mut self,
        c: &Ctx38<'_>,
        l: usize,
        (n_card, n_tier): (usize, usize),
        m: usize,
        weights: &DeviceBuffer<f32>,
        trows: &DeviceBuffer<f32>,
        acc: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let (Some(tsel), Some(trank)) = (self.tsel.as_ref(), self.trank.as_mut()) else {
            return Err(GpuError::state(
                WHAT,
                "a card route made for an expert tier",
            ));
        };
        let stream = c.gpu.stream();
        c.k.q38
            .enqueue_tier_rank(stream, tsel, (m * geo::N_USED, n_tier), trank)?;
        c.k.q38.enqueue_card_tier_acc(
            stream,
            CardTierAccArgs {
                down: self.down.rows().f32(WHAT)?,
                trows,
                w: weights,
                sel: &self.sel,
                tsel,
                trank,
                n: geo::HIDDEN,
                m,
                n_card,
                n_tier,
                fault: c.gpu.layer_sink(l)?,
                acc,
            },
        )
    }
}

/// What a ubatch walk reads and writes besides the arena: the one-expert
/// tables, the GEMMs' activations, the wide mixes' scratch, the indexer keys
/// token-major, the walk's routed slots, the last walk's units and its cut
/// of its selecting layers' rows, and how the walks cut their rows into units.
pub(super) struct Wide38 {
    /// The most rows a unit holds.
    pub(super) rows: usize,
    /// The one-expert tables the units fill, unit `i` the `i`-th
    /// ([`UnitRows38`]'s `route`): the first for up to `rows` slots, the
    /// second, on an arena of two rows or more, for the second unit of a
    /// split walk, at most half of them.
    dense: Vec<GemmRoute>,
    /// q8 of the model-width rows (a mix's output, the PLE rows, the shared
    /// expert's input), of the attention-width rows (the output projections'
    /// input), and of the shared expert's SwiGLU, each with room for the
    /// [`WideForms`] its sites read.
    act_hid: GemmAct32,
    act_attn: GemmAct32,
    act_ff: GemmAct32,
    hc: HcWideScratch,
    /// Every token's `N_USED + 1` slots: the routed ids and weights the
    /// download reads, the shared expert's weight the gated sum reads.
    ids: DeviceBuffer<u32>,
    weights: DeviceBuffer<f32>,
    /// The route taps, when a gate armed them.
    pub(super) taps: Option<WideTaps>,
    /// The planted routes, when a gate planted them.
    pub(super) force: Option<WideForce>,
    /// The card route, when the slot map puts routed experts on the card.
    pub(super) card: Option<CardRoute38>,
    /// The last walk's rows of each selecting layer: those that select
    /// nothing (the prefill flash's) and those that select.
    pub(super) split: Option<(usize, usize)>,
    /// The last walk's units, the head reading the last one's rows; their
    /// windows stay valid until the next walk replaces them.
    units: Vec<UnitRows38>,
    /// How the walks cut their rows ([`SplitWalk`]), and what the load keeps
    /// its routed experts on, which the rule prices only when resident.
    split_walk: SplitWalk,
    load: LoadKind,
    /// The width rule's costs on this card ([`SPLIT_COSTS`] at its
    /// multiprocessors).
    split_costs: SplitCosts,
}

/// The [`MmaForm`]s the walk's model-width, attention-width and SwiGLU rows
/// hold room for ([`Wide38::new`]): on a card that reserves them, those a
/// site that reads the rows copies on the dense MMA tile
/// ([`site::reads_form`]). The BF16 form is the model-width rows' alone,
/// whose rounding the ubatch's BF16 sites share; an MXFP8 site's W8A8
/// product reads its rows' MXFP8 form wherever they are. `hc_bf16`: every HC
/// mix (each layer's two and the head's) has a BF16 down on a dense MMA card,
/// so the wide mixes norm their streams into BF16 words alone
/// ([`HcWideScratch::new`]).
pub(super) struct WideForms {
    hid: Vec<MmaForm>,
    attn: Vec<MmaForm>,
    ff: Vec<MmaForm>,
    hc_bf16: bool,
}

impl WideForms {
    /// The forms the sites of `plans` read in `w`, none unless `reserved`,
    /// and whether the HC mixes norm into BF16 words alone on `gpu`'s card.
    pub(super) fn of(
        gpu: &crate::Gpu,
        w: &crate::weights::Weights,
        plans: &[Layer38],
        reserved: bool,
    ) -> Result<WideForms, GpuError> {
        let read = |sites: &[&str], forms: &[MmaForm]| -> Vec<MmaForm> {
            let read = |f: &MmaForm| reserved && site::reads_form(gpu, w, sites, *f);
            forms.iter().copied().filter(read).collect()
        };
        let hid: Vec<&str> = plans.iter().flat_map(Layer38::hidden_sites).collect();
        let attn: Vec<&str> = plans.iter().map(Layer38::attn_site).collect();
        let ff: Vec<&str> = plans.iter().map(|p| &*p.ffn.down_sh).collect();
        let head = head_site();
        let bf16_down =
            |s: &HcSite| matches!(HcMatrix::of(w, WHAT, &s.down), Ok(HcMatrix::Bf16(_)));
        let hc_bf16 = crate::dense_mma(gpu.context())?
            && bf16_down(&head)
            && plans
                .iter()
                .all(|p| bf16_down(&p.attn_hc) && bf16_down(&p.ffn_hc));
        Ok(WideForms {
            hid: read(&hid, &[MmaForm::Bf16, MmaForm::Mxfp8]),
            attn: read(&attn, &[MmaForm::Mxfp8]),
            ff: read(&ff, &[MmaForm::Mxfp8]),
            hc_bf16,
        })
    }
}

/// A [`GemmAct32`] of `cols` columns of `k` values with room for `forms`.
fn act_with(
    stream: &CudaStream,
    (cols, k): (usize, usize),
    forms: &[MmaForm],
) -> Result<GemmAct32, GpuError> {
    forms
        .iter()
        .try_fold(GemmAct32::new(stream, cols, k)?, |act, &f| {
            act.with_form(stream, f)
        })
}

impl Wide38 {
    /// The walk's own buffers for units of up to `rows` rows over `card`'s
    /// card experts, the activations' room for `forms`: the card route's
    /// buffers for runs of `route_run` tokens beside them when a layer has
    /// card experts, made for an expert tier's join when `tiered`; the walks
    /// cut by the width rule on a load with no expert tier, as one unit on
    /// one with it ([`Wide38::paged`] for an NVMe-paged plan). Load-time
    /// only.
    pub(super) fn new(
        stream: &CudaStream,
        rows: usize,
        route_run: usize,
        card: &Card38,
        tiered: bool,
        forms: &WideForms,
    ) -> Result<Wide38, GpuError> {
        let geometry = Geometry::new(geo::STREAMS as u32, geo::RANK as u32, geo::HIDDEN as u32)
            .map_err(|e| GpuError::shape(WHAT, e.to_string()))?;
        let slots = geo::N_USED + 1;
        let card = (card.card_layers() > 0)
            .then(|| CardRoute38::new(stream, (rows, route_run), card, tiered))
            .transpose()?;
        let mut dense = vec![GemmRoute::new(stream, rows, 1)?];
        if rows >= 2 {
            dense.push(GemmRoute::new(stream, rows / 2, 1)?);
        }
        let sms = f64::from(stream.context().multiprocessor_count()?);
        let (split_walk, load) = if tiered {
            (SplitWalk::Off, LoadKind::Tier)
        } else {
            (SplitWalk::Rule, LoadKind::Resident)
        };
        Ok(Wide38 {
            rows,
            dense,
            act_hid: act_with(stream, (rows, geo::HIDDEN), &forms.hid)?,
            act_attn: act_with(stream, (rows, geo::ATTN), &forms.attn)?,
            act_ff: act_with(stream, (rows, geo::FF), &forms.ff)?,
            hc: HcWideScratch::new(stream, geometry, rows, forms.hc_bf16)?,
            ids: DeviceBuffer::zeroed(stream, rows * slots)?,
            weights: DeviceBuffer::zeroed(stream, rows * slots)?,
            taps: None,
            force: None,
            card,
            split: None,
            units: Vec::new(),
            split_walk,
            load,
            split_costs: SplitCosts {
                card_row_us: SPLIT_COSTS.card_row_us * SPLIT_COSTS_SMS / sms,
                ..SPLIT_COSTS
            },
        })
    }

    /// The units the last walk ran; 0 before a walk.
    pub(super) fn walked_units(&self) -> usize {
        self.units.len()
    }

    /// The rows the last walk's last layer ran past its mixer's store
    /// writes, over its units ([`WideParts::kept`]); 0 before a walk.
    pub(super) fn last_rows(&self) -> usize {
        self.units.iter().map(|u| u.m).sum()
    }

    /// The load's plan pages routed experts to the NVMe tier: the walks cut
    /// as one unit, and a split is refused by name
    /// ([`Wide38::set_split_walk`]). Load-time only.
    pub(super) fn paged(&mut self) {
        if self.load == LoadKind::Resident {
            self.load = LoadKind::Paged;
        }
        self.split_walk = SplitWalk::Off;
    }

    /// How the walks cut their rows from the next walk on. A split
    /// ([`SplitWalk::Rule`], [`SplitWalk::Every`]) is refused by name on a
    /// load with an expert tier or an NVMe-paged plan
    /// (`runtime::splitwalk::refuse`), the cut left as it was.
    pub(super) fn set_split_walk(&mut self, w: SplitWalk) -> Result<(), GpuError> {
        refuse_split(self.load, w)?;
        self.split_walk = w;
        Ok(())
    }

    /// How the walks cut their rows.
    pub(super) fn split_walk(&self) -> SplitWalk {
        self.split_walk
    }

    /// Device bytes, the route taps left out.
    pub(super) fn bytes(&self) -> usize {
        self.dense.iter().map(GemmRoute::bytes).sum::<usize>()
            + self.act_hid.bytes()
            + self.act_attn.bytes()
            + self.act_ff.bytes()
            + self.hc.bytes()
            + self.ids.num_bytes()
            + self.weights.num_bytes()
            + self.card.as_ref().map_or(0, CardRoute38::bytes)
    }
}

/// Layer `l`'s plan.
fn plan_of(plans: &[Layer38], l: usize) -> Result<&Layer38, GpuError> {
    plans
        .get(l)
        .ok_or(GpuError::state(WHAT, "a plan for every layer"))
}

/// The parts of the body a ubatch walk writes: [`super::program38::Parts38`]'s
/// over the wide arena and the walk's own buffers, each taking the unit it
/// runs for ([`UnitRows38`]). `m`, `pos0` and `dense` are the walk's whole:
/// its rows, the position of its first and its leading rows that select
/// nothing ([`dense_rows`]) on the host; the walk cuts them into its units.
pub(super) struct WideParts<'a> {
    pub(super) c: Ctx38<'a>,
    pub(super) plans: &'a [Layer38],
    pub(super) stores: &'a mut [Store38],
    pub(super) ple_ring: &'a mut DeviceBuffer<f32>,
    pub(super) s: &'a mut Arena38,
    pub(super) x: &'a mut Wide38,
    pub(super) io: &'a Io<'a>,
    /// The slot map's card copy and the card leg's stacks, read by the card
    /// route on a layer with card experts.
    pub(super) slots: &'a DeviceTensor<u32>,
    pub(super) card: &'a Card38,
    /// The stage card's tier side on a load with an expert tier: each
    /// layer's tier count and the map's tier view.
    pub(super) tier: Option<&'a TierSide38>,
    pub(super) m: usize,
    pub(super) pos0: usize,
    pub(super) dense: usize,
    /// The walk's last rows something reads once it has walked (the head's
    /// row, a checkpoint's point row, a tap's rows), at most `m`: the last
    /// layer runs its rows past the mixer's store writes for those alone
    /// ([`WideParts::kept`]).
    pub(super) read: usize,
    /// The prompt call's host streaming ([`PromptStream`]): whether this
    /// walk picks, and its records.
    pub(super) stream: &'a mut PromptStream,
    /// The media call's staged rows on a media load: each unit's first layer
    /// copies the rows of the spans inside it over the unit's embeddings.
    pub(super) media: Option<&'a mut MediaRows>,
    /// The walk's PLE rows when its plan left them on their read
    /// (`Body38::stage_gemm`): the PLE site's first unit copies them into
    /// the arena's `e` before any unit reads it, the walk's layers before
    /// it enqueued.
    pub(super) ple_rows: Option<&'a mut PleRowsIn<'a>>,
    /// One event a layer, each recorded once the layer's last unit's back is
    /// enqueued: what a checkpoint's copy of the layer's stores waits for
    /// (`Body38`'s end mark).
    pub(super) done: Option<&'a [CudaEvent]>,
    /// Whether the walk's delta layers keep its last chunk out of the state
    /// (`Body38::keep_last`): its last unit's prompt form keeps it.
    pub(super) keep: bool,
}

/// [`WideParts::ple_rows`]: the walk's PLE rows landed in the arena's `e`
/// (`Body38`'s `PleHost::land`, through the arena's planes).
pub(super) type PleRowsIn<'a> = dyn FnMut(&mut Ple38) -> Result<(), GpuError> + 'a;

/// The kernels of the walk's dense sites ([`site::resident_gemm`]).
fn dense_kernels<'a>(c: &Ctx38<'a>) -> WideKernels<'a> {
    WideKernels {
        gemm: &c.k.gemm,
        g32: Some(&c.k.g32),
    }
}

/// The forms of the first `m` columns of the f32 rows `rows` that any of
/// `names` reads into `act`: the q8 for Q8_0 planes, a [`MmaForm`] for a site
/// on the dense MMA tile ([`site::resident_quantize`]).
fn quantize_for(
    c: &Ctx38<'_>,
    names: &[&str],
    rows: (&DeviceBuffer<f32>, usize),
    act: &mut GemmAct32,
    sink: FaultSink,
) -> Result<(), GpuError> {
    site::resident_quantize(c.gpu, &dense_kernels(c), (c.w, names), rows, act, sink)
}

/// [`quantize_for`] unless `act` already holds the one form `names` read of
/// the first `m` columns of `rows` ([`site::only_form`]): the mix that wrote
/// the rows writes that form beside them or marks it no rows' ([`mix`]).
fn quantize_unheld(
    c: &Ctx38<'_>,
    names: &[&str],
    (rows, m): (&DeviceBuffer<f32>, usize),
    act: &mut GemmAct32,
    sink: FaultSink,
) -> Result<(), GpuError> {
    if site::only_form(c.gpu, c.w, names, act).is_some_and(|f| act.holds(f, rows, m)) {
        return Ok(());
    }
    quantize_for(c, names, (rows, m), act, sink)
}

/// `y = W · rows` for the dense weight `name` over the first `m` columns by
/// its resident form, token-major, on the walk's one-expert table `dense`
/// ([`site::resident_gemm`]): Q8_0 over `act`, the q8 of `rows`
/// ([`quantize_for`]). A site that reads q8_1 rows (a K-quant) has no wide
/// GEMM here: it runs by chunks of the by-name unit's columns
/// ([`project_chunks`]).
fn dense_gemm(
    c: &Ctx38<'_>,
    bn: &mut ByName,
    name: &str,
    (rows, act, m): (&DeviceBuffer<f32>, &GemmAct32, usize),
    (dense, sink): (&GemmRoute, FaultSink),
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    if reads_q8_1(c, name)? {
        return project_chunks(c, bn, name, (rows, m), sink, y);
    }
    let x = wide_in(rows, act);
    site::resident_gemm(c.gpu, &dense_kernels(c), (c.w, name), &x, dense, m, y)
}

/// Whether the dense weight `name` reads q8_1 rows (its resident type is a
/// K-quant or NVFP4), the sites the ubatch walk projects by chunks.
fn reads_q8_1(c: &Ctx38<'_>, name: &str) -> Result<bool, GpuError> {
    Ok(site::resident_type(c.w, WHAT, name)?.reads() == Form::Q8x128)
}

/// `y = W · rows` for the dense weight `name` over the first `m` columns of
/// the f32 rows `rows`, token-major, by chunks of [`BY_NAME_COLS`] columns:
/// each chunk the by-name projection of its columns (the quantization, the
/// K-quant gemv and the token-major copy), each column bit for bit the
/// step's one-column launch. The weight is read once a chunk.
fn project_chunks(
    c: &Ctx38<'_>,
    bn: &mut ByName,
    name: &str,
    (rows, m): (&DeviceBuffer<f32>, usize),
    sink: FaultSink,
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    let weight = c.w.resident(WHAT, name)?;
    let (n, k) = (weight.rows(), weight.k());
    let f = size_of::<f32>();
    let mut c0 = 0;
    while c0 < m {
        let cn = BY_NAME_COLS.min(m - c0);
        let xs = Window::<f32>::of(rows, c0 * k * f, cn * k)?;
        let mut ys = WindowMut::<f32>::of_mut(y, c0 * n * f, cn * n)?;
        c.project(bn, name, &xs, cn, sink, Out::whole(&mut ys))?;
        c0 += cn;
    }
    Ok(())
}

/// The f32 rows `rows` and their q8 `act`, the forms a dense site reads.
fn wide_in<'a>(rows: &'a DeviceBuffer<f32>, act: &'a GemmAct32) -> WideIn<'a> {
    WideIn {
        q128: None,
        q32: Some(act),
        f32: Some(rows),
    }
}

/// The wide mix of `site` over `m` columns of `res`, after `before`, into
/// `mixed`, the scratch's combine weights on the unit's rows `wgt`
/// ([`UnitHc`]). When the sites `readers` read one form of `mixed` alone
/// ([`site::only_form`]), the mix writes it into `act` or marks it no rows'
/// ([`quantize_unheld`]).
#[allow(
    clippy::too_many_arguments,
    reason = "one site's streams, rule, width, sink, scratch, table and output (rust-quality R8)"
)]
fn mix(
    c: &Ctx38<'_>,
    site: &HcSite,
    res: Streams<'_>,
    before: Before<'_>,
    m: usize,
    fault: FaultSink,
    (scratch, wgt, dense): (&mut HcWideScratch, Span, &GemmRoute),
    (mixed, readers, act): (&mut DeviceBuffer<f32>, &[&str], &mut GemmAct32),
) -> Result<(), GpuError> {
    let form = site::only_form(c.gpu, c.w, readers, act);
    let hc = UnitHc::on(scratch, wgt)?;
    let w = c.w;
    let inject = match &site.inject {
        Some(n) => Some(f32_tensor(w, n)?),
        None => None,
    };
    c.k.hcw.enqueue_mix(
        c.gpu.stream(),
        WideMixArgs {
            res,
            before,
            w: SiteWeights {
                gamma: f32_gain(w, &site.norm)?,
                down: HcMatrix::of(w, WHAT, &site.down)?,
                up: HcMatrix::of(w, WHAT, &site.up)?,
                inject,
            },
            eps: c.eps,
            m,
            fault,
            scratch: &mut *hc.hc,
            gemm: &c.k.g32,
            q8f32: c.gpu.q8f32(),
            dense,
            mixed,
            out: form.map(|f| (f, act)),
        },
    )
}

/// The combine alone over unit `u`'s columns into its F32 final streams:
/// `fin = streams + wgt · y` per stream, the weights the unit's last inject
/// site left ([`UnitHc`]), a fault raised on `fault`.
fn combine(
    c: &Ctx38<'_>,
    u: &mut UnitRows38,
    scratch: &mut HcWideScratch,
    fault: FaultSink,
) -> Result<(), GpuError> {
    let hc = UnitHc::on(scratch, u.wgt)?;
    let io = (&*u.streams, &mut *u.fin);
    c.k.hcw
        .enqueue_combine(c.gpu.stream(), io, &u.y, u.m, fault, &*hc.hc)
}

/// Unit `u` cut to its last `keep` rows ([`UnitRows38::narrow`]) and its
/// one-expert table filled for them, which every part after the cut reads;
/// nothing when it keeps every row.
fn narrow(c: &Ctx38<'_>, x: &mut Wide38, u: &mut UnitRows38, keep: usize) -> Result<(), GpuError> {
    if keep == u.m {
        return Ok(());
    }
    u.narrow(keep)?;
    if keep == 0 {
        return Ok(());
    }
    let table = x
        .dense
        .get_mut(u.route)
        .ok_or(GpuError::state(WHAT, "a unit's one-expert table"))?;
    c.k.gemm
        .enqueue_route_dense(c.gpu.stream(), keep, table, c.gpu.unlabelled_sink())
}

impl WideParts<'_> {
    /// The rows of unit `u` layer `l` runs past its mixer's store writes
    /// (the delta step, the key, value and indexer-key appends, the pool):
    /// on the walk's last layer, when it runs without a host leg
    /// ([`Card38::hostless`]: no service reads the unit's rows) and no route
    /// tap records it, the unit's rows among the walk's last `read`; every
    /// row otherwise. Nothing past the last layer reads the other rows.
    fn kept(&self, u: &UnitRows38, l: usize) -> usize {
        let cut = l + 1 == self.plans.len() && self.card.hostless(l) && self.x.taps.is_none();
        match cut {
            true => unit_kept(self.m - self.read, u.r0, u.m),
            false => u.m,
        }
    }

    /// Layer `l`'s front of unit `u` up to its feed-forward site
    /// (`Parts38::front`'s order): the embedding in front of layer 0, the
    /// attention site (with the PLE site on its layer), the mixer into the
    /// unit's `y`.
    fn front(&mut self, u: &mut UnitRows38, l: usize) -> Result<(), GpuError> {
        let (plans, gpu, m) = (self.plans, self.c.gpu, u.m);
        let p = plan_of(plans, l)?;
        let sink = gpu.layer_sink(l)?;
        let keep = self.kept(u, l);
        let stream = gpu.stream();
        let c = &self.c;
        let s = &mut *self.s;
        let x = &mut *self.x;
        let (gdn_in, qsa_in);
        let readers: &[&str] = match &p.mixer {
            Mixer38::Gdn(g) => {
                gdn_in = g.input_sites();
                &gdn_in
            }
            Mixer38::Qsa(q) => {
                qsa_in = q.input_sites();
                &qsa_in
            }
        };
        if l == 0 {
            c.k.q38.enqueue_embed_lookup(
                c.gpu,
                "qwen3moe::wide38::front",
                (c.w, &model::arch::qwen35moe::names::token_embd()),
                EmbedIo {
                    ids: &u.tok,
                    pos0: self.io.pos0,
                    first: u.first,
                    width: geo::HIDDEN,
                    y: &mut u.emb,
                    pos: &mut u.pos,
                    n_keys: &mut u.n_keys,
                },
            )?;
            if let Some(media) = self.media.as_deref_mut() {
                media.plan(u.pos0..u.pos0 + u.m);
                media.write(stream, &mut u.emb)?;
            }
            mix(
                c,
                &p.attn_hc,
                Streams::Bf16(&mut u.streams),
                Before::Init { y: &u.emb },
                m,
                sink,
                (&mut x.hc, u.wgt, &x.dense[u.route]),
                (&mut s.mixed, readers, &mut x.act_hid),
            )?;
        } else if let Some(pp) = &p.ple {
            if let Some(rows_in) = self.ple_rows.take() {
                rows_in(&mut s.ple)?;
            }
            combine(c, u, &mut x.hc, gpu.layer_sink(l - 1)?)?;
            let w = c.w;
            let pl = &mut s.ple;
            let e = (&*u.ple_e, m);
            quantize_for(c, &pp.input_sites(), e, &mut x.act_hid, sink)?;
            let dense = &x.dense[u.route];
            let act = (&*u.ple_e, &x.act_hid, m);
            let table = (dense, sink);
            dense_gemm(c, &mut s.by, &pp.key, act, table, &mut pl.key)?;
            dense_gemm(c, &mut s.by, &pp.value, act, table, &mut pl.value)?;
            // The site reads the combined streams as F32 values and writes
            // the walk's BF16 streams.
            let xs = &*u.fin;
            c.k.ple.enqueue_gate(
                stream,
                PleGateArgs {
                    key: &pl.key,
                    value: &pl.value,
                    x: xs,
                    gain_key: f32_gain(w, &pp.norm_key)?,
                    gain_query: f32_gain(w, &pp.norm_query)?,
                    gain_conv: f32_gain(w, &pp.norm_conv)?,
                    eps: c.eps,
                    hc: geo::STREAMS,
                    m,
                    fault: sink,
                    gv: &mut pl.gv,
                    ngv: &mut pl.ngv,
                    gate: &mut pl.gate,
                },
            )?;
            c.k.ple.enqueue_conv(
                stream,
                PleConvArgs {
                    ngv: &pl.ngv,
                    gv: &pl.gv,
                    x: xs,
                    w: f32_gain(w, &pp.conv)?,
                    pos: &u.pos,
                    taps: geo::PLE_TAPS,
                    dilation: geo::PLE_DILATION,
                    hc: geo::STREAMS,
                    m,
                    fault: sink,
                    out: Streams::Bf16(&mut u.streams),
                    ring: &mut *self.ple_ring,
                },
            )?;
            mix(
                c,
                &p.attn_hc,
                Streams::Bf16(&mut u.streams),
                Before::Plain,
                m,
                sink,
                (&mut x.hc, u.wgt, dense),
                (&mut s.mixed, readers, &mut x.act_hid),
            )?;
        } else {
            mix(
                c,
                &p.attn_hc,
                Streams::Bf16(&mut u.streams),
                Before::Combine { y: &u.y },
                m,
                sink,
                (&mut x.hc, u.wgt, &x.dense[u.route]),
                (&mut s.mixed, readers, &mut x.act_hid),
            )?;
        }
        let store = self
            .stores
            .get_mut(l)
            .ok_or(GpuError::state(WHAT, "a store for every layer"))?;
        match (&p.mixer, store) {
            (Mixer38::Gdn(g), Store38::Rec { rec, stamp, .. }) => {
                let lane = self.io.lane.ok_or(GpuError::state(
                    WHAT,
                    "the record's lane word (a chain with delta layers carries one)",
                ))?;
                let chunk = self.keep && u.r0 + u.m == self.m;
                gdn(c, g, (rec, stamp, lane), (s, &mut *x), (u, chunk), sink)?;
                narrow(c, x, u, keep)
            }
            (Mixer38::Qsa(q), Store38::Qsa { kv, ring, pooled }) => {
                let planes = (&mut **kv, ring, &mut **pooled);
                let split = qsa(c, q, planes, (s, x), (u, keep), sink)?;
                u.split = Some(split);
                Ok(())
            }
            _ => Err(GpuError::shape(
                WHAT,
                format!("layer {l}'s plan and store are of two kinds"),
            )),
        }
    }

    /// Layer `l`'s feed-forward site of unit `u`: the mix of the streams, the
    /// mixer's output combined first, into the unit's `ffn_x`.
    fn ffn_mix(&mut self, u: &mut UnitRows38, l: usize) -> Result<(), GpuError> {
        let p = plan_of(self.plans, l)?;
        let sink = self.c.gpu.layer_sink(l)?;
        let x = &mut *self.x;
        mix(
            &self.c,
            &p.ffn_hc,
            Streams::Bf16(&mut u.streams),
            Before::Combine { y: &u.y },
            u.m,
            sink,
            (&mut x.hc, u.wgt, &x.dense[u.route]),
            (&mut u.ffn_x, &p.ffn.shared_inputs(), &mut x.act_hid),
        )
    }

    /// Layer `l`'s router over unit `u`'s `ffn_x`, in runs of [`ROUTE_ROWS`]
    /// tokens, each run's slots copied into the unit's. When a gate armed the
    /// route taps, each run's logits go into layer `l`'s tap first, at the
    /// walk's row of the token; when a gate planted routes, the run is then
    /// routed again over the planted logits of its positions (the shared
    /// expert's gate left the router's); the slots the unit runs go into the
    /// tap last.
    fn route(&mut self, u: &UnitRows38, l: usize) -> Result<(), GpuError> {
        let p = plan_of(self.plans, l)?;
        let sink = self.c.gpu.layer_sink(l)?;
        let stream = self.c.gpu.stream();
        let (s, x, m) = (&mut *self.s, &mut *self.x, u.m);
        let router = gated::RouterWeight::of(self.c.w, WHAT, &p.ffn.router)?;
        let slots = geo::N_USED + 1;
        let width = s.route.dims().logits();
        let mut c0 = 0;
        while c0 < m {
            let n = ROUTE_ROWS.min(m - c0);
            // SAFETY: tokens c0 .. c0 + n of the unit's `ffn_x` (`m` rows,
            // c0 + n <= m); `ffn_x` stays in place while the window lives
            // (this run's router launch).
            let xw = unsafe { cells_view(&u.ffn_x, c0 * geo::HIDDEN, n * geo::HIDDEN) };
            self.c
                .k
                .router
                .enqueue_ubatch(stream, router, &xw, n, sink, &mut s.route)?;
            if let Some(t) = x.taps.as_mut() {
                let lt = t
                    .logits
                    .get_mut(l)
                    .ok_or(GpuError::state(WHAT, "a route tap for every layer"))?;
                // SAFETY: the walk's rows r0 + c0 .. + n <= taps.rows
                // (`plan_gemm` refuses a unit past them, `Gemm38::walk` again)
                // of `width` logits, inside the tap and the route's logits
                // (`n <= ROUTE_ROWS` tokens); both stay in place for the copy.
                let (mut tw, lw) = unsafe {
                    (
                        cells_view(lt, (u.r0 + c0) * width, n * width),
                        cells_view(&s.route.logits, 0, n * width),
                    )
                };
                tw.copy_from_device_async(&lw, stream)?;
            }
            if let Some(f) = &x.force {
                let lf = f
                    .logits
                    .get(l)
                    .ok_or(GpuError::state(WHAT, "a planted route for every layer"))?;
                f.covers(u.pos0 + c0, n)?;
                let at = u.pos0 + c0 - f.pos0;
                for t in 0..n {
                    // SAFETY: the plant holds positions pos0 + c0 .. + n
                    // (checked above), so plant row at + t < f.rows, whose
                    // EXPERTS logits lie inside `lf` (`rows · EXPERTS`); the
                    // route's token t < n <= ROUTE_ROWS holds `width` logits,
                    // the router's EXPERTS rows first and the shared gate's
                    // last. Both stay in place for the copy.
                    let (mut dw, pw) = unsafe {
                        (
                            cells_view(&s.route.logits, t * width, geo::EXPERTS),
                            cells_view(lf, (at + t) * geo::EXPERTS, geo::EXPERTS),
                        )
                    };
                    dw.copy_from_device_async(&pw, stream)?;
                }
                self.c
                    .k
                    .router
                    .enqueue_route(stream, n, sink, &mut s.route)?;
            }
            // SAFETY: tokens c0 .. c0 + n of the unit's slots (`m` rows of
            // `slots`, c0 + n <= m) and the route's first n (<= ROUTE_ROWS)
            // tokens'; every buffer stays in place for the two copies.
            let (mut iw, mut ww, ri, rw) = unsafe {
                (
                    param_view::<u32>(&u.ids, c0 * slots, n * slots),
                    cells_view(&u.weights, c0 * slots, n * slots),
                    param_view::<u32>(&s.route.ids, 0, n * slots),
                    cells_view(&s.route.weights, 0, n * slots),
                )
            };
            iw.copy_from_device_async(&ri, stream)?;
            ww.copy_from_device_async(&rw, stream)?;
            c0 += n;
        }
        if let Some(t) = x.taps.as_mut() {
            let it = t
                .ids
                .get_mut(l)
                .ok_or(GpuError::state(WHAT, "a route tap for every layer"))?;
            // SAFETY: the walk's rows r0 .. r0 + m <= taps.rows of `slots`
            // ids, inside the tap; it stays in place for the copy.
            let mut tw = unsafe { param_view::<u32>(it, u.r0 * slots, m * slots) };
            tw.copy_from_device_async(&u.ids, stream)?;
        }
        Ok(())
    }

    /// Layer `l`'s shared expert over unit `u`'s `ffn_x` into its `sh_y`:
    /// gate and up, the SwiGLU into the form the down reads, down, each at
    /// its resident type ([`site::gated`]).
    fn shared(&mut self, u: &mut UnitRows38, l: usize) -> Result<(), GpuError> {
        let p = plan_of(self.plans, l)?;
        let sink = self.c.gpu.layer_sink(l)?;
        let (c, m) = (&self.c, u.m);
        let (s, x) = (&mut *self.s, &mut *self.x);
        let f = &p.ffn;
        quantize_unheld(c, &f.shared_inputs(), (&*u.ffn_x, m), &mut x.act_hid, sink)?;
        // A block with a site that reads q8_1 rows (a K-quant gate or up) has
        // no wide GEMM: the three projections run by chunks, the SwiGLU once
        // over every row (it is elementwise).
        if reads_q8_1(c, &f.gate_sh)? || reads_q8_1(c, &f.up_sh)? || reads_q8_1(c, &f.down_sh)? {
            let rows = (&*u.ffn_x, m);
            project_chunks(c, &mut s.by, &f.gate_sh, rows, sink, &mut s.sh_g)?;
            project_chunks(c, &mut s.by, &f.up_sh, rows, sink, &mut s.sh_u)?;
            c.gpu.elem().enqueue_swiglu(
                c.gpu.stream(),
                &s.sh_g,
                &s.sh_u,
                geo::FF * m,
                &mut s.sh_h,
            )?;
            return project_chunks(c, &mut s.by, &f.down_sh, (&s.sh_h, m), sink, &mut u.sh_y);
        }
        let site = |name| -> Result<_, GpuError> {
            let ty = site::resident_type(c.w, "qwen38::wide shared", name)?;
            Ok((ty, SiteWeight::Resident(name)))
        };
        let block = Gated {
            w: c.w,
            route: &x.dense[u.route],
            gate: site(&f.gate_sh)?,
            up: site(&f.up_sh)?,
            down: site(&f.down_sh)?,
            ff: geo::FF,
            hidden: geo::HIDDEN,
            rule: Act::SiluMul,
            input: GemmInput::PerSlot,
        };
        // The SwiGLU writes the form the down's type reads: the q8 of its
        // rows, or the f32 rows. Both are held here beside the arena's, not as
        // one `WideActs`: `sh_h` is the gemv arm's row too.
        let io = GatedIo {
            x: &wide_in(&u.ffn_x, &x.act_hid),
            gate: &mut s.sh_g,
            up: &mut s.sh_u,
            rows: SwigluRows::held(None, Some(&mut x.act_ff), Some(&mut s.sh_h)),
            y: &mut u.sh_y,
        };
        site::gated(c.gpu, &dense_kernels(c), &block, (m, m), io, sink)
    }

    /// Layer `l`'s card route over unit `u`'s `ffn_x`, when it has card
    /// experts ([`CardRoute38::enqueue`], the walk's module doc): the places
    /// from the unit's ids, the grouped gate·up, SwiGLU and down over the
    /// run's tokens — a GEMM batch a landing batch of the pick (`land`), each
    /// behind its own landed event — the ring pass over the layer's stream
    /// when `ring` is one, the card sums into the unit's acc — on a layer
    /// without a host leg ([`Card38::hostless`]) with the shared expert's
    /// gated rows into the unit's `y`, the block's output, so the shared
    /// expert runs first. Called from the walk's shadow only — after the
    /// front's download, the stream order the route's buffers exist under.
    fn route_card(
        &mut self,
        u: &mut UnitRows38,
        l: usize,
        ring: Option<&RingLayer>,
        land: &[LandBatch],
    ) -> Result<(), GpuError> {
        if !self.card.has(l) {
            return Ok(());
        }
        let tiered = self.tier_k(l) > 0;
        let c = &self.c;
        let x = &mut *self.x;
        let Some(r) = x.card.as_mut() else {
            return Ok(());
        };
        let sum = if tiered {
            CardSum::Join
        } else if self.card.hostless(l) {
            CardSum::Block {
                sh: &u.sh_y,
                y: &mut u.y,
            }
        } else {
            CardSum::Acc(u.acc.as_mut().ok_or(GpuError::state(
                WHAT,
                "a unit's card sum over a map with card experts",
            ))?)
        };
        r.enqueue(
            c,
            self.card,
            l,
            (&u.ffn_x, &u.ids, &u.weights, sum),
            self.slots,
            u.m,
            ring,
            land,
        )
    }

    /// The tier's experts of layer `l`: 0 off the tier and on a load
    /// without one.
    fn tier_k(&self, l: usize) -> usize {
        self.tier.map_or(0, |t| t.k(l))
    }

    /// Tier layer `l`'s download after its front: the unit's tier places
    /// from its ids and the map's tier view into the card route's, then the
    /// arena's activations, routing and those places into the batch port's
    /// set ([`crate::host::HostTier::enqueue_download_tiered_pitched`]),
    /// which the tier's service reads. The card route's tier places, run
    /// buffers and join rows are indexed from the walk's first row, so a unit
    /// that starts past it is refused by name.
    fn download_tiered(
        &mut self,
        u: &UnitRows38,
        port: &mut BatchLeg<'_, HostRun>,
        key: crate::host::batch::BatchKey,
    ) -> Result<(), GpuError> {
        let l = key.layer;
        if u.r0 != 0 {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "layer {l}: a unit from row {} on a tier layer, whose places and join rows \
                     are the walk's one unit's",
                    u.r0
                ),
            ));
        }
        let tier = self
            .tier
            .ok_or(GpuError::state(WHAT, "the tier side of a tier layer"))?;
        let (c, m) = (&self.c, u.m);
        let stream = c.gpu.stream();
        let (s, x) = (&*self.s, &mut *self.x);
        let r = x
            .card
            .as_mut()
            .ok_or(GpuError::state(WHAT, "the card route of a tier layer"))?;
        let tsel = r.tsel_mut()?;
        c.k.handoff.enqueue_places_cols(
            stream,
            &Places {
                ids: &u.ids,
                map: tier.view(),
                row_off: l * geo::EXPERTS,
                n_expert: geo::EXPERTS,
            },
            geo::N_USED + 1,
            m,
            c.gpu.layer_sink(l)?,
            tsel,
        )?;
        port.hybrid().enqueue_download_tiered_pitched(
            stream,
            [&s.ffn_x, &x.weights],
            &x.ids,
            geo::N_USED + 1,
            &[&*tsel],
            key,
        )
    }

    /// Tier layer `l`'s join after the host's wait for the tier
    /// ([`CardRoute38::enqueue_tier_acc`] over the tier's rows `trows`):
    /// the card sum into unit `u`'s acc, which the back's combine reads.
    fn tier_acc(
        &mut self,
        u: &mut UnitRows38,
        l: usize,
        trows: &DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let n_tier = self.tier_k(l);
        let n_card = self.card.n_card(l);
        let (c, x) = (&self.c, &mut *self.x);
        let r = x
            .card
            .as_mut()
            .ok_or(GpuError::state(WHAT, "the card route of a tier layer"))?;
        let acc = u.acc.as_mut().ok_or(GpuError::state(
            WHAT,
            "a unit's card sum over a map with card experts",
        ))?;
        r.enqueue_tier_acc(c, l, (n_card, n_tier), u.m, &u.weights, trows, acc)
    }

    /// Layer `l`'s block output into unit `u`'s `y` on a layer with a host
    /// leg: the host's routed sums (its `hsum`), on a layer with card experts
    /// the card route's sums too, plus the shared expert's output times its
    /// gate weight.
    fn shared_add(&mut self, u: &mut UnitRows38, l: usize) -> Result<(), GpuError> {
        let sink = self.c.gpu.layer_sink(l)?;
        let (stream, q38) = (self.c.gpu.stream(), &self.c.k.q38);
        let (slot, slots, n, m) = (geo::N_USED, geo::N_USED + 1, geo::HIDDEN, u.m);
        // A layer with card experts has the walk's card route (both read the
        // one loaded map); a layer without them keeps the plain sum — its
        // rows of the acc were never this walk's.
        match (self.card.has(l), u.acc.as_ref()) {
            (true, Some(acc)) => q38.enqueue_card_shared_add(
                stream,
                CardSharedAddArgs {
                    hsum: &u.hsum,
                    acc,
                    sh: &u.sh_y,
                    w: &u.weights,
                    slot,
                    slots,
                    n,
                    m,
                    fault: sink,
                    y: &mut u.y,
                },
            ),
            (false, _) => q38.enqueue_shared_add(
                stream,
                SharedAddArgs {
                    hsum: &u.hsum,
                    sh: &u.sh_y,
                    w: &u.weights,
                    slot,
                    slots,
                    n,
                    m,
                    fault: sink,
                    y: &mut u.y,
                },
            ),
            (true, None) => Err(GpuError::state(
                WHAT,
                "the walk's card route over a map with card experts",
            )),
        }
    }
}

/// A delta layer's mixer over unit `u`'s rows and its store `r`, its lanes'
/// stamps and the lane word, the committed lane in place, the unit's last
/// chunk kept out of the state when `keep` (the walk's last unit of a
/// closing call, [`WideParts::keep`]); the attention
/// site's mix in `s.mixed`, its output projection into the unit's `y`.
fn gdn(
    c: &Ctx38<'_>,
    gp: &GdnPlan,
    (r, stamp, lane): (&mut RecStore, &mut DeviceBuffer<u32>, &DeviceBuffer<u32>),
    (s, x): (&mut Arena38, &mut Wide38),
    (u, keep): (&mut UnitRows38, bool),
    sink: FaultSink,
) -> Result<(), GpuError> {
    let (w, stream, m) = (c.w, c.gpu.stream(), u.m);
    let nv = GDN.n_v;
    let Arena38 {
        mixed,
        gdn: g,
        attn,
        by,
        ..
    } = s;
    quantize_unheld(c, &gp.input_sites(), (mixed, m), &mut x.act_hid, sink)?;
    let dense = &x.dense[u.route];
    let act = (&*mixed, &x.act_hid, m);
    let (k, x_in) = (dense_kernels(c), wide_in(mixed, &x.act_hid));
    let bf16 = if reads_q8_1(c, &gp.qkv)? {
        false
    } else {
        let mut half = WindowMut::<u16>::of_mut(&mut g.x, 0, m * GDN.channels())?;
        site::resident_gemm_bf16(c.gpu, &k, (c.w, &gp.qkv), &x_in, dense, m, &mut half)?
    };
    if !bf16 {
        dense_gemm(c, by, &gp.qkv, act, (dense, sink), &mut g.x)?;
    }
    // The joined stack is `2·nv` rows (`plan38::plans` refuses another shape
    // at load), β's rows first and α's next: one product, each row's β then
    // its α ([`beta_alpha_rows`]).
    site::resident_gemm(c.gpu, &k, (c.w, &gp.beta_alpha), &x_in, dense, m, &mut g.ba)?;
    let (b, a) = beta_alpha_rows(&g.ba, m)?;
    let lin = &c.k.linear;
    let words = Window::<u32>::of(&g.x, 0, m * GDN.channels() / 2)?;
    lin.conv.enqueue_conv_prep(
        stream,
        ConvArgs {
            x: if bf16 {
                ConvRows::Bf16(&words)
            } else {
                ConvRows::F32(&g.x)
            },
            b_raw: &b,
            a_raw: &a,
            ba_pitch: BA_PITCH,
            w: f32_gain(w, &gp.conv)?,
            dt_bias: f32_gain(w, &gp.dt_bias)?,
            ssm_a: f32_gain(w, &gp.ssm_a)?,
            pos: &u.pos,
            shape: GDN,
            eps: c.eps,
            m,
            fault: sink,
            y: &mut g.conv,
            beta: &mut g.beta,
            decay: &mut g.decay,
            ring: &mut r.ring,
        },
    )?;
    drop(words);
    // The conv has read the q·k·v rows: the prompt form's chunk records take
    // their room.
    let len = g.x.len();
    let mut work = WindowMut::<u32>::of_mut(&mut g.x, 0, len)?;
    let carry = r.carry.as_mut().ok_or_else(no_carry)?;
    lin.delta.enqueue_delta_prompt(
        stream,
        DeltaLanesArgs {
            delta: DeltaArgs {
                qkv: &g.conv,
                beta: &g.beta,
                decay: &g.decay,
                lane,
                lane_at: 0,
                lanes: r.lanes,
                shape: GDN,
                m,
                fault: sink,
                o: &mut g.o,
                state: &mut r.state,
            },
            each: false,
            pos: &u.pos,
            stamp,
        },
        carry,
        &mut work,
        keep,
    )?;
    if reads_q8_1(c, &gp.z)? {
        // A K-quant `z` has no tile epilogue to finish the norm in: the
        // projection by chunks, the gated norm over its rows, then the forms
        // the output projection reads.
        project_chunks(c, by, &gp.z, (&*mixed, m), sink, &mut g.z)?;
        lin.norm_gate.enqueue_norm_gate_sigmoid(
            stream,
            crate::linear::norm_gate::NormGateArgs {
                o: &g.o,
                z: &g.z,
                w: f32_gain(w, &gp.ssm_norm)?,
                eps: c.eps,
                n_v: nv,
                m,
                fault: sink,
                y: &mut *attn,
            },
        )?;
        site::resident_quantize(
            c.gpu,
            &k,
            (w, &[&gp.ssm_out]),
            (&*attn, m),
            &mut x.act_attn,
            sink,
        )?;
    } else {
        site::resident_gemm_norm_gate(
            c.gpu,
            &k,
            (w, &gp.z),
            &x_in,
            dense,
            m,
            (
                &lin.norm_gate,
                &mut g.z,
                crate::linear::norm_gate::NormGateTail {
                    o: &g.o,
                    w: f32_gain(w, &gp.ssm_norm)?,
                    eps: c.eps,
                    fault: sink,
                    y: &mut *attn,
                },
            ),
            (&[&gp.ssm_out], &mut x.act_attn),
        )?;
    }
    let act = (&*attn, &x.act_attn, m);
    dense_gemm(c, by, &gp.ssm_out, act, (dense, sink), &mut u.y)
}

/// A selecting attention layer's mixer over unit `u`'s rows and its store
/// (the K/V planes, the raw indexer key ring and the pooled keys), the first
/// `u.dense` of them selecting nothing ([`dense_rows`]), the attention site's
/// mix in `s.mixed`, its output projection into the unit's `y`. Every row goes
/// through the projections, the pool and the appends; the flash, the gate
/// and the output projection run over the last `keep` of them, the unit cut
/// to those first ([`narrow`]). Returns the unit's rows that select nothing
/// and those that select.
fn qsa(
    c: &Ctx38<'_>,
    qp: &QsaPlan,
    (kv, ring, pooled): (
        &mut KvPlanes,
        &mut DeviceBuffer<u16>,
        &mut DeviceBuffer<u16>,
    ),
    (s, x): (&mut Arena38, &mut Wide38),
    (u, keep): (&mut UnitRows38, usize),
    sink: FaultSink,
) -> Result<(usize, usize), GpuError> {
    let (w, stream, ctx) = (c.w, c.gpu.stream(), c.ctx);
    let (m, dense) = (u.m, u.dense);
    let split = (dense, m - dense);
    let Arena38 {
        mixed,
        qsa: q,
        attn,
        by,
        ..
    } = s;
    quantize_unheld(c, &qp.input_sites(), (mixed, m), &mut x.act_hid, sink)?;
    let table = (&x.dense[u.route], sink);
    let act = (&*mixed, &x.act_hid, m);
    dense_gemm(c, by, &qp.q, act, table, &mut q.qg)?;
    dense_gemm(c, by, &qp.k, act, table, &mut q.k)?;
    dense_gemm(c, by, &qp.v, act, table, &mut q.v)?;
    dense_gemm(c, by, &qp.idx_k, act, table, &mut q.kr)?;
    c.k.qsa.enqueue_pool(
        stream,
        PoolArgs {
            kr: &q.kr,
            ring: &*ring,
            gain: f32_gain(w, &qp.idx_k_norm)?,
            table: c.table,
            n_keys: &u.n_keys,
            eps: c.eps,
            ctx,
            m,
            fault: sink,
            pooled: &mut *pooled,
        },
    )?;
    c.k.q38.enqueue_key_append(
        stream,
        KeyAppendArgs {
            kr: &q.kr,
            pos: &u.pos,
            m,
            ctx,
            fault: sink,
            ring: &mut *ring,
        },
    )?;
    dense_gemm(c, by, &qp.idx_q, act, table, &mut q.qi)?;
    // The append over the key heads alone: the flashes norm and turn the
    // query heads they read (`QueryNorm`).
    // SAFETY: the per-sequence embedding supplies distinct cache-valid positions.
    unsafe {
        kv.append_256(
            &c.k.neox,
            stream,
            Append256 {
                qg: &q.qg,
                q: &mut q.q,
                k: &mut q.k,
                v: &q.v,
                gq: f32_gain(w, &qp.q_norm)?,
                gk: f32_gain(w, &qp.k_norm)?,
                table: c.table,
                pos: &u.pos,
                eps: c.eps,
                n_head: 0,
                n_kv: geo::N_KV,
                ctx,
                m,
                fault: sink,
            },
        )
    }?;
    let gq = f32_gain(w, &qp.q_norm)?;
    // The arena's `[q | gate]` rows hold the unit's row `r` at their row
    // `r`; the cut unit's row `r` is the unit's row `t0 + r`, and the gated
    // flash writes it at row `r` of `attn`, the output projection's input.
    let t0 = m
        .checked_sub(keep)
        .ok_or_else(|| GpuError::shape(WHAT, format!("a unit of {m} rows keeping {keep}")))?;
    narrow(c, x, u, keep)?;
    let (m, dense) = (u.m, u.dense);
    if m == 0 {
        return Ok(split);
    }
    if dense > 0 {
        // SAFETY: rows t0 .. t0 + dense <= t0 + m (the unit's rows before the
        // cut) of the `[q | gate]` rows (`rows` rows), and rows 0 .. dense of
        // the cut unit's counts, positions and `attn` (`m` rows); they stay
        // in place for the launch.
        let (gw, nw, pw, mut yw) = unsafe {
            (
                cells_view(&q.qg, t0 * geo::Q_ROWS, dense * geo::Q_ROWS),
                param_view::<u32>(&u.n_keys, 0, dense),
                param_view::<u32>(&u.pos, 0, dense),
                cells_view(attn, 0, dense * geo::ATTN),
            )
        };
        kv.prefill_256_gated(
            &c.k.prefill,
            stream,
            PrefillFlash {
                q: &gw,
                n_keys: &nw,
                scale: ATTN_SCALE_256,
                n_head: geo::N_HEAD,
                n_kv: geo::N_KV,
                ctx,
                t: dense,
                fault: sink,
                y: &mut yw,
            },
            QueryNorm {
                gain: gq,
                table: c.table,
                pos: &pw,
                eps: c.eps,
            },
        )?;
    }
    let (width, run) = (q.sel.width(), q.sel.rows());
    let idx = geo::IDX_HEADS * geo::IDX_DIM;
    let mut r = dense;
    while r < m {
        let n = run.min(m - r);
        let at = t0 + r;
        // SAFETY: rows at .. at + n <= t0 + m (the unit's rows before the
        // cut) of the indexer queries and the `[q | gate]` rows (`rows` rows
        // each), and rows r .. r + n <= m of the cut unit's counts, positions
        // and `attn` (`m` rows); they stay in place for this run's launches.
        let (qi, nw, gw, pw, mut yw) = unsafe {
            (
                cells_view(&q.qi, at * idx, n * idx),
                param_view::<u32>(&u.n_keys, r, n),
                cells_view(&q.qg, at * geo::Q_ROWS, n * geo::Q_ROWS),
                param_view::<u32>(&u.pos, r, n),
                cells_view(attn, r * geo::ATTN, n * geo::ATTN),
            )
        };
        c.k.qsa.enqueue_select(
            stream,
            SelectArgs {
                q: &qi,
                gain: f32_gain(w, &qp.idx_q_norm)?,
                table: c.table,
                n_keys: &nw,
                pooled: &*pooled,
                eps: c.eps,
                ctx,
                kept: geo::KEPT,
                m: n,
                fault: sink,
                scratch: &mut q.sel,
            },
        )?;
        c.k.prefill.enqueue_256_p12_sel_gated(
            stream,
            GqaListArgs {
                q: &gw,
                kv: kv.planes(),
                list: &q.sel.list,
                n_sel: &q.sel.n_sel,
                width,
                scale: ATTN_SCALE_256,
                n_kv: geo::N_KV,
                ctx,
                m: n,
                fault: sink,
                y: &mut yw,
            },
            geo::N_HEAD,
            QueryNorm {
                gain: gq,
                table: c.table,
                pos: &pw,
                eps: c.eps,
            },
        )?;
        r += n;
    }
    quantize_for(c, &[&qp.out], (attn, m), &mut x.act_attn, sink)?;
    let table = (&x.dense[u.route], sink);
    dense_gemm(c, by, &qp.out, (attn, &x.act_attn, m), table, &mut u.y)?;
    Ok(split)
}

/// The no-probe floor: the least count an expert's ubatch routes to it for a
/// pick to admit it on a load whose lane never probed — every `admit` pick,
/// and a `split` pick at a layer the stream holds no constants of — about
/// the least column count whose share of the host union passes one flip's
/// copy on the staging ring (an expert's three stacks), which the layer's
/// card route waits for [derived]; V4.1's `STREAM_FLOOR`, the same value,
/// sits inside that estimate's error. Below it a flip lengthens a layer
/// whose card already waits on its copies more than it shortens the union.
/// A `split` pick at a probed layer admits from the walk's own plan
/// ([`runtime::xsplit::admit_walk`]) instead, counted per pick by the pick
/// record's `fallback`.
pub(super) const STREAM_FLOOR: u32 = 32;

/// A ubatch's walk `(units, cols, Batch)`: every layer through the batch
/// port at each unit's width; the head and the final rows are the caller's
/// ([`Gemm38::head`], [`Gemm38::finals`]).
pub(super) struct Gemm38<'a> {
    pub(super) p: WideParts<'a>,
}

/// The walk's layer program over its units: the schedule names the unit of
/// each part, which runs over that unit's windows. A split walk's first unit
/// picks for the layer and the later units read its pick: `held` keeps the
/// layer's landing batches until the last unit's card route has read them.
struct UnitWalk<'w, 'a> {
    p: &'w mut WideParts<'a>,
    units: Vec<UnitRows38>,
    held: Option<Held>,
}

/// The first unit's pick of a layer, for the later units' card routes: the
/// layer, its landing batches, and whether the layer streams.
struct Held {
    layer: usize,
    land: Vec<LandBatch>,
    streams: bool,
}

/// The walk's unit `at.unit` of `units`, refused by name past them.
fn unit_at(units: &mut [UnitRows38], at: At) -> Result<&mut UnitRows38, GpuError> {
    let n = units.len();
    units
        .get_mut(at.unit)
        .ok_or_else(|| GpuError::shape(WHAT, format!("unit {} of a walk of {n} units", at.unit)))
}

impl<'a> Gemm38<'a> {
    /// Walk every layer through `leg`: the walk's `m` rows cut into units
    /// ([`WideParts::widths`], [`unit_rows`]), the leg keyed by each unit's
    /// rows, each unit's one-expert table filled for its width first, the
    /// units' items in [`BatchOrder::Fronts`]' order. A walk of no row, or
    /// past the arena's rows or the armed route taps', is refused by name
    /// before any launch (`Body38::plan_gemm` refuses the taps' first, before
    /// the call moves anything; this is the windows' own guard). Once every
    /// layer is walked, armed taps record the walk's positions, the units'
    /// cuts of the selecting layers' rows add up to the walk's, and the units
    /// stay for the head and the final rows ([`Gemm38::finals`]).
    pub(super) fn walk(&mut self, leg: &mut BatchLeg<'a, HostRun>) -> Result<(), GpuError> {
        let m = self.p.m;
        let taps = self.p.x.taps.as_ref().map_or(usize::MAX, |t| t.rows);
        if self.p.read > m {
            return Err(GpuError::shape(
                WHAT,
                format!("{} rows read after a walk of {m}", self.p.read),
            ));
        }
        if m == 0 || m > self.p.s.rows || m > self.p.x.rows || m > taps || self.p.dense > m {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "a unit of {m} rows on a {}-row arena ({} route tap rows)",
                    self.p.s.rows,
                    if taps == usize::MAX { 0 } else { taps }
                ),
            ));
        }
        let gpu = self.p.c.gpu;
        self.p.x.split = None;
        let mut units = std::mem::take(&mut self.p.x.units);
        units.clear();
        if let Some(t) = self.p.x.taps.as_mut() {
            t.walked = None;
        }
        let cut = unit_rows(m, &self.p.widths()?)?;
        let rows: Vec<std::ops::Range<usize>> = cut.iter().map(|&(r0, w)| r0..r0 + w).collect();
        leg.set_unit_rows(&rows);
        for (route, at) in cut.into_iter().enumerate() {
            units.push(UnitRows38::new(&self.p, leg.hsum(), at, route)?);
        }
        for u in &units {
            self.p.c.k.gemm.enqueue_route_dense(
                gpu.stream(),
                u.m,
                &mut self.p.x.dense[u.route],
                gpu.unlabelled_sink(),
            )?;
        }
        let o = Overlap {
            units: units.len(),
            cols: units.iter().fold(0, |w, u| w.max(u.m)),
            port: PortKind::Batch,
        };
        let layers = self.p.plans.len();
        leg.begin_walk(units.len())?;
        let mut walker = UnitWalk {
            p: &mut self.p,
            units,
            held: None,
        };
        sched::walk(o, layers, leg, &mut walker)?;
        let UnitWalk { units, held, .. } = walker;
        if let Some(h) = held {
            return Err(GpuError::protocol(
                WHAT,
                format!(
                    "layer {}'s pick held past the walk: a unit's card route did not read it",
                    h.layer
                ),
            ));
        }
        // The last layer's return: no front follows it in the walk, so its
        // copies are the end's.
        self.p.stream.return_owed(leg.hybrid(), gpu.stream())?;
        leg.end_walk()?;
        if let Some(t) = self.p.x.taps.as_mut() {
            t.walked = Some((self.p.pos0, m));
        }
        self.p.x.split = units
            .iter()
            .try_fold((0, 0), |(d, s), u| u.split.map(|(ud, us)| (d + ud, s + us)));
        self.p.x.units = units;
        Ok(())
    }

    /// After the walk of a prompt call whose tap reads every row it ran
    /// ([`super::body38::Prompt38Sink`]), or of a unit a checkpoint keeps the
    /// last row of: the last layer's combine alone over each unit the head
    /// leaves — every unit but the last when `head` runs after it
    /// ([`Self::head`] combines that one), every unit otherwise — at the
    /// unit's own combine weights ([`combine`]), over
    /// the unit's rows the last layer ran ([`WideParts::kept`]), so each row
    /// the walk read holds the target's final streams. One launch a unit that
    /// kept a row. Refused by name before a walk.
    pub(super) fn finals(&mut self, head: bool) -> Result<(), GpuError> {
        let fault = self
            .p
            .c
            .gpu
            .layer_sink(self.p.plans.len().saturating_sub(1))?;
        let Wide38 { units, hc, .. } = &mut *self.p.x;
        if units.is_empty() {
            return Err(GpuError::state(WHAT, "a ubatch walk before its final rows"));
        }
        let left = units.len() - usize::from(head);
        for u in units[..left].iter_mut().filter(|u| u.m > 0) {
            combine(&self.p.c, u, hc, fault)?;
        }
        Ok(())
    }

    /// After the walk: the last layer's combine over the last unit's columns
    /// the last layer ran ([`WideParts::kept`]) into their final streams
    /// ([`combine`]), the head's mix over those streams, its last row into
    /// `head`'s input, and the head. Refused by name before a walk, and after
    /// one that read no row.
    pub(super) fn head(&mut self, head: &mut Head) -> Result<(), GpuError> {
        let (h, gpu) = (geo::HIDDEN, self.p.c.gpu);
        let fault = gpu.fault_sink(LAYER_HEAD);
        let s = &mut *self.p.s;
        let Wide38 {
            units,
            hc,
            dense,
            act_hid,
            ..
        } = &mut *self.p.x;
        let u = units
            .last_mut()
            .ok_or(GpuError::state(WHAT, "a ubatch walk before its head"))?;
        let m = u.m;
        if m == 0 {
            return Err(GpuError::state(
                WHAT,
                "a ubatch walk that reads its last row before its head",
            ));
        }
        combine(&self.p.c, u, hc, fault)?;
        mix(
            &self.p.c,
            &head_site(),
            Streams::F32(&mut u.fin),
            Before::Plain,
            m,
            fault,
            (hc, u.wgt, &dense[u.route]),
            (&mut s.mixed, &[], act_hid),
        )?;
        // Row m − 1 of `mixed` (1 <= m <= rows, the walk refused anything
        // else).
        let row = Window::<f32>::of(&s.mixed, (m - 1) * h * size_of::<f32>(), h)?;
        head.input_mut()
            .copy_from_device_async(&row, gpu.stream())?;
        head.enqueue(gpu, self.p.c.w)
    }
}

impl<'a> WideParts<'a> {
    /// The widths of the walk's units ([`SplitWalk`]): `⌈m/2⌉` and `⌊m/2⌋`
    /// rows when the cut splits a walk of two rows or more — every such walk
    /// under [`SplitWalk::Every`], one the width rule pays for under
    /// [`SplitWalk::Rule`] (`runtime::splitwalk::pays` at the card's
    /// [`SPLIT_COSTS`], over the most host-resident routed experts a layer
    /// holds; a load with none has no union to hide a front under) — else
    /// one unit of all `m`. A split on a load the rule does not price is
    /// refused by name (the setter refused it already).
    fn widths(&self) -> Result<Vec<usize>, GpuError> {
        let x = &*self.x;
        let host = (0..self.plans.len())
            .map(|l| geo::EXPERTS.saturating_sub(self.card.n_card(l)))
            .max()
            .unwrap_or(0);
        unit_widths(self.m, x.split_walk, (x.load, &x.split_costs), host)
    }
}

impl<'a> LayerProgram for UnitWalk<'_, 'a> {
    type Port = BatchLeg<'a, HostRun>;

    /// Every unit's first front ahead of the shadows, then each unit's next
    /// front after its back: a unit's front runs on the card while the host
    /// serves the other's union, and the layer's fronts run in unit order —
    /// the delta layers' recurrent state and conv ring and the selecting
    /// layers' pooled keys advance unit by unit. At one unit, the order of
    /// the walk of one unit.
    const BATCH_ORDER: BatchOrder = GROUP_WALK_ORDER;

    /// A layer the load runs without a host leg ([`Card38::hostless`]) has
    /// none here either: the host run holds no view of it.
    fn host_leg(&self, at: At) -> bool {
        !self.p.card.hostless(at.layer)
    }

    /// The layer up to its router, the mix into the unit's `ffn_x`, then
    /// the download of the unit's activations and routed slots — on a tier
    /// layer with the unit's tier places beside them
    /// ([`WideParts::download_tiered`]); a layer without a host leg
    /// downloads nothing. The download names the unit's rows by the port's
    /// key and takes the arena's buffers whole. A unit the last layer keeps
    /// no row of ([`WideParts::kept`]) runs nothing past its mixer.
    fn front(&mut self, port: &mut BatchLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let l = at.layer;
        let enq = port.part_start();
        port.mark(at, Mark::Front as usize)?;
        let u = unit_at(&mut self.units, at)?;
        let p = &mut *self.p;
        p.front(u, l)?;
        if u.m > 0 {
            p.ffn_mix(u, l)?;
            p.route(u, l)?;
        }
        port.mark(at, Mark::FrontEnd as usize)?;
        let key = port.key(at);
        let stream = p.c.gpu.stream();
        let r = if p.card.hostless(l) {
            Ok(())
        } else if p.tier_k(l) > 0 {
            p.download_tiered(u, port, key)
        } else {
            port.hybrid().enqueue_download_pitched(
                stream,
                [&p.s.ffn_x, &p.x.weights],
                &p.x.ids,
                geo::N_USED + 1,
                key,
            )
        };
        let marked = r.and_then(|()| port.mark(at, Mark::Down as usize));
        port.part_end(at, enq);
        marked
    }

    /// The shared expert over the unit's `ffn_x`, then the card route on a
    /// layer with card experts, whose sum on a layer without a host leg
    /// reads the shared expert's rows ([`WideParts::route_card`]). The route
    /// is enqueued here and nowhere else — after the front's download, the
    /// stream order the route's buffers exist under (its module doc):
    /// enqueued before the download, the d2h would queue behind the route's
    /// card GEMMs and the host tier would start late by their whole time.
    /// Under host streaming the card runs the shared expert while the host
    /// counts and picks; then the layer's pick (and stream), the first
    /// unit's alone ([`PromptStream::pick`]), and the route, its GEMM
    /// batches each behind the landing batch it reads — a later unit's over
    /// the first unit's pick, held for it — then the unit's read
    /// ([`PromptStream::read`]), the last unit's the layer's last read of
    /// its slots in the call. The shadow's first launch is marked for the
    /// walk's timer: the order puts the other unit's parts between a unit's
    /// download and its shadow. A unit with no row left runs nothing.
    fn shadow(&mut self, port: &mut BatchLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let l = at.layer;
        let enq = port.part_start();
        let units = self.units.len();
        let held = &mut self.held;
        let u = unit_at(&mut self.units, at)?;
        let p = &mut *self.p;
        let last = at.unit + 1 == units;
        let walk_end = p.pos0 + p.m;
        let r = port
            .mark(at, Mark::ShadowStart as usize)
            .and_then(|()| {
                if u.m == 0 {
                    return Ok(());
                }
                p.shared(u, l)?;
                if !p.stream.streaming() {
                    return p.route_card(u, l, None, &[]);
                }
                p.stream.return_at_shadow(port, at, walk_end)?;
                let (ring, land) = if at.unit == 0 {
                    p.stream
                        .pick(port, at, (u.m, units), walk_end, |l| p.card.has(l))?
                } else {
                    let h = held.take().filter(|h| h.layer == l).ok_or_else(|| {
                        GpuError::protocol(
                            WHAT,
                            format!(
                                "unit {}'s card route of layer {l} with no pick of the layer's \
                                 first unit held",
                                at.unit
                            ),
                        )
                    })?;
                    let ring = if h.streams {
                        port.hybrid().xstream_ring(l)
                    } else {
                        None
                    };
                    (ring, h.land)
                };
                let streams = ring.is_some();
                p.route_card(u, l, ring.as_ref(), &land)?;
                if !last {
                    *held = Some(Held {
                        layer: l,
                        land,
                        streams,
                    });
                }
                p.stream.read(port, l, last, |l| p.card.has(l))
            })
            .and_then(|()| port.mark(at, Mark::Shadow as usize));
        port.part_end(at, enq);
        r
    }

    /// The gated sum over the host sums the walk's serve uploaded, the card
    /// route's beside them on a layer with card experts; a layer without a
    /// host leg has nothing left, its route wrote the block's output in the
    /// shadow. On a tier layer the join first ([`BatchLeg::join_tiered`]):
    /// once the host has seen the tier's service complete, the card sum over
    /// both cards' slots ([`WideParts::tier_acc`]), then the host sums'
    /// upload.
    fn back(&mut self, port: &mut BatchLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let l = at.layer;
        let enq = port.part_start();
        let units = self.units.len();
        let u = unit_at(&mut self.units, at)?;
        let p = &mut *self.p;
        let summed = if p.card.hostless(l) {
            Ok(())
        } else if p.tier_k(l) > 0 {
            port.join_tiered(at, |_, trows| p.tier_acc(u, l, trows))
                .and_then(|()| p.shared_add(u, l))
        } else {
            p.shared_add(u, l)
        };
        let r = summed.and_then(|()| port.mark(at, Mark::Back as usize));
        // The layer's host service has run for every unit once its last
        // unit's back runs: in the returning call's last walk its return is
        // due, asked at the next layer's shadow.
        if r.is_ok() && at.unit + 1 == units && p.stream.returns_here(p.pos0 + p.m) && p.card.has(l)
        {
            p.stream.owe_return(l);
        }
        let r = r.and_then(|()| match p.done {
            Some(done) if at.unit + 1 == units => match done.get(l) {
                Some(event) => Ok(event.record(p.c.gpu.stream())?),
                None => Err(GpuError::shape(
                    WHAT,
                    format!("layer {l}'s done event of {}", done.len()),
                )),
            },
            _ => Ok(()),
        });
        port.part_end(at, enq);
        r
    }
}

/// Every layer's route taps of the last walk, read back: position `pos0 +
/// t`'s [`super::body38::RouteTap`] of layer `l` at `[t][l]`. Refused by
/// name unless the last walk (since the taps were armed or the model reset)
/// ran exactly the positions `pos0 .. pos0 + m`. Blocking.
pub(super) fn route_taps_host(
    taps: &WideTaps,
    stream: &CudaStream,
    pos0: usize,
    m: usize,
) -> Result<Vec<Vec<super::body38::RouteTap>>, GpuError> {
    match taps.walked {
        Some(w) if w == (pos0, m) => {}
        Some((p, n)) => {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "route taps of positions {pos0}..{}: the last ubatch walk ran {p}..{} and                      the taps hold that walk's alone",
                    pos0 + m,
                    p + n
                ),
            ));
        }
        None => {
            return Err(GpuError::state(
                WHAT,
                "a ubatch walk since the route taps were armed (or the model reset)",
            ));
        }
    }
    if taps.width < geo::EXPERTS {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "route taps of {} logits a token, fewer than the {} experts",
                taps.width,
                geo::EXPERTS
            ),
        ));
    }
    let slots = geo::N_USED + 1;
    let layers = taps
        .logits
        .iter()
        .zip(&taps.ids)
        .map(|(lg, id)| Ok((lg.to_host_vec(stream)?, id.to_host_vec(stream)?)))
        .collect::<Result<Vec<_>, GpuError>>()?;
    Ok((0..m)
        .map(|t| {
            layers
                .iter()
                .map(|(lg, id)| super::body38::RouteTap {
                    logits: lg[t * taps.width..][..geo::EXPERTS].to_vec(),
                    ids: id[t * slots..][..geo::N_USED].to_vec(),
                })
                .collect()
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::{
        LoadKind, SPLIT_COSTS, Span, SplitWalk, UnitSpans, geo, refuse_split, unit_dense,
        unit_kept, unit_rows, unit_widths,
    };

    /// Each of a unit's windows with the words a row of the buffer it cuts:
    /// `Arena38`'s, `Wide38`'s, the card route's, the input record's, the
    /// leg's and the wide mixes' layouts.
    fn fields(u: &UnitSpans) -> [(&'static str, Span, usize); 14] {
        let (hidden, wide, slots) = (geo::HIDDEN, geo::STREAMS * geo::HIDDEN, geo::N_USED + 1);
        [
            ("tok", u.tok, 1),
            ("emb", u.emb, hidden),
            ("pos", u.pos, 1),
            ("n_keys", u.n_keys, 1),
            ("res", u.res, wide),
            ("ple_e", u.ple_e, hidden),
            ("ffn_x", u.ffn_x, hidden),
            ("ids", u.ids, slots),
            ("weights", u.weights, slots),
            ("y", u.y, hidden),
            ("sh_y", u.sh_y, hidden),
            ("acc", u.acc, hidden),
            ("hsum", u.hsum, hidden),
            ("wgt", u.wgt, geo::STREAMS),
        ]
    }

    /// The cut puts the units' rows one after another from row 0 to the
    /// walk's last, each its width, and refuses by name what is not a cut:
    /// no unit, a unit of no row, widths that do not add up.
    #[test]
    fn a_walk_is_cut_into_units_that_follow_each_other() {
        for (m, widths) in [
            (1, vec![1]),
            (4096, vec![4096]),
            (7, vec![4, 3]),
            (4095, vec![2048, 2047]),
            (10, vec![3, 3, 4]),
        ] {
            let cut = unit_rows(m, &widths).expect("a cut");
            let mut next = 0;
            for (&(r0, w), &want) in cut.iter().zip(&widths) {
                assert_eq!((r0, w), (next, want), "{m} rows in {widths:?}");
                next += w;
            }
            assert_eq!(next, m, "{m} rows in {widths:?} end at row {next}");
            assert_eq!(cut.len(), widths.len(), "{m} rows in {widths:?}");
        }
        for (m, widths) in [
            (7, vec![]),
            (7, vec![3, 3]),
            (7, vec![7, 0]),
            (7, vec![4, 4]),
        ] {
            let e = unit_rows(m, &widths).expect_err("not a cut");
            assert!(e.to_string().contains("cut into units"), "{widths:?}: {e}");
        }
    }

    /// Over a toy arena of 7 rows cut in two units, every window of each
    /// unit is exactly its rows of its buffer — from its first row for its
    /// width — so the two units' windows of a buffer follow each other
    /// without a gap or an overlap from word 0 to the buffer's end. A window
    /// left at row 0, or off by a row, or on the whole buffer, names its
    /// buffer.
    #[test]
    fn every_window_of_a_unit_lies_on_its_rows_and_the_units_cover_the_arena() {
        let rows = 7;
        let cut = unit_rows(rows, &[4, 3]).expect("a cut");
        let units: Vec<UnitSpans> = cut.iter().map(|&(r0, m)| UnitSpans::of(r0, m)).collect();
        let mut wrong = Vec::new();
        for (k, (&(r0, m), u)) in cut.iter().zip(&units).enumerate() {
            for (name, got, stride) in fields(u) {
                let want = Span::rows(r0, m, stride);
                if got != want {
                    wrong.push(format!(
                        "unit {k} `{name}`: window {got:?}, its rows {r0}..{} are {want:?}",
                        r0 + m
                    ));
                }
            }
        }
        for (i, (name, first, stride)) in fields(&units[0]).into_iter().enumerate() {
            let second = fields(&units[1])[i].1;
            if first.off != 0 || first.off + first.len != second.off {
                wrong.push(format!(
                    "`{name}`: {first:?} then {second:?} leave a gap or overlap"
                ));
            }
            if second.off + second.len != rows * stride {
                wrong.push(format!(
                    "`{name}`: the units end at word {}, the arena's {rows} rows at {}",
                    second.off + second.len,
                    rows * stride
                ));
            }
        }
        assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    }

    /// A unit's rows that select nothing are the walk's leading ones cut by
    /// the unit: all of the unit's, none, or the walk's remainder; and over
    /// the units they add up to the walk's.
    #[test]
    fn a_units_dense_rows_are_the_walks_leading_ones_cut_by_it() {
        for (m, dense, widths, want) in [
            (10, 6, vec![4, 6], vec![4, 2]),
            (10, 6, vec![3, 3, 4], vec![3, 3, 0]),
            (10, 0, vec![5, 5], vec![0, 0]),
            (10, 10, vec![5, 5], vec![5, 5]),
            (10, 6, vec![10], vec![6]),
        ] {
            let got: Vec<usize> = unit_rows(m, &widths)
                .expect("a cut")
                .into_iter()
                .map(|(r0, w)| unit_dense(dense, r0, w))
                .collect();
            assert_eq!(got, want, "{dense} of {m} rows in {widths:?}");
            assert_eq!(got.iter().sum::<usize>(), dense, "{dense} of {m} rows");
        }
    }

    /// The rows the last layer keeps of each unit are the unit's among the
    /// walk's last `read`: none, its last few, or all of them; over the
    /// units they add up to `read`, and each is a suffix of its unit, so the
    /// last unit keeps the walk's last row whenever a row is read.
    #[test]
    fn a_unit_keeps_its_rows_among_the_walks_last_read() {
        for (m, read, widths, want) in [
            (10, 1, vec![10], vec![1]),
            (10, 0, vec![10], vec![0]),
            (10, 10, vec![10], vec![10]),
            (10, 1, vec![5, 5], vec![0, 1]),
            (10, 7, vec![5, 5], vec![2, 5]),
            (10, 10, vec![5, 5], vec![5, 5]),
            (4095, 1, vec![2048, 2047], vec![0, 1]),
            (1, 1, vec![1], vec![1]),
        ] {
            let got: Vec<usize> = unit_rows(m, &widths)
                .expect("a cut")
                .into_iter()
                .map(|(r0, w)| unit_kept(m - read, r0, w))
                .collect();
            assert_eq!(got, want, "the last {read} of {m} rows in {widths:?}");
            assert_eq!(got.iter().sum::<usize>(), read, "{read} of {m} rows");
        }
    }

    /// A split is refused by name on a load with an expert tier and on an
    /// NVMe-paged plan, under the rule and under every walk; one unit is
    /// taken on every load, and a resident load takes either split.
    #[test]
    fn a_split_is_refused_by_name_on_a_tier_load_and_a_paged_plan() {
        for (load, want) in [
            (LoadKind::Tier, "tier load"),
            (LoadKind::Paged, "NVMe-paged"),
        ] {
            for w in [SplitWalk::Rule, SplitWalk::Every] {
                let e = refuse_split(load, w).expect_err("a split the rule does not price");
                let text = e.to_string();
                assert!(
                    text.contains("BLOOMERY_SPLIT_WALK") && text.contains(want),
                    "{load:?} {w:?}: {text}"
                );
                // The rule splits 4,096 rows over 512 host experts, every
                // walk splits 8.
                let m = if w == SplitWalk::Rule { 4096 } else { 8 };
                let cut = unit_widths(m, w, (load, &SPLIT_COSTS), 512)
                    .expect_err("the cut refuses it too");
                assert!(cut.to_string().contains(want), "{load:?} {w:?}: {cut}");
            }
            refuse_split(load, SplitWalk::Off).expect("one unit on every load");
            assert_eq!(
                unit_widths(8, SplitWalk::Off, (load, &SPLIT_COSTS), 512).expect("one unit"),
                [8]
            );
        }
        for w in [SplitWalk::Off, SplitWalk::Rule, SplitWalk::Every] {
            refuse_split(LoadKind::Resident, w).expect("a resident load takes every cut");
        }
    }

    /// The cut: one unit off, at one row and where the rule does not pay (or
    /// no layer holds a host expert); `⌈m/2⌉` and `⌊m/2⌋` under every walk and
    /// where the rule pays — a full ubatch on the costs' card splits, a
    /// 512-row walk does not.
    #[test]
    fn a_walk_splits_in_halves_where_its_cut_says() {
        let at = |m, w, host| unit_widths(m, w, (LoadKind::Resident, &SPLIT_COSTS), host);
        assert_eq!(at(4096, SplitWalk::Off, 512).unwrap(), [4096]);
        assert_eq!(at(1, SplitWalk::Every, 512).unwrap(), [1]);
        assert_eq!(at(7, SplitWalk::Every, 512).unwrap(), [4, 3]);
        assert_eq!(at(3001, SplitWalk::Every, 0).unwrap(), [1501, 1500]);
        assert_eq!(at(4096, SplitWalk::Rule, 512).unwrap(), [2048, 2048]);
        assert_eq!(at(512, SplitWalk::Rule, 512).unwrap(), [512]);
        assert_eq!(at(4096, SplitWalk::Rule, 0).unwrap(), [4096]);
    }

    /// A unit's window gives back the context handle it takes when it drops.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_a_unit_window_gives_back_its_context_handle() {
        let (ctx, stream) = crate::capsync::fresh_stream(0).expect("CUDA device 0 with a stream");
        let buf = cuda_core::DeviceBuffer::<f32>::zeroed(&stream, 8).expect("a buffer");
        let before = std::sync::Arc::strong_count(&ctx);
        {
            let _rows = super::rows_of("rows", &buf, Span { off: 4, len: 4 }).expect("rows 4 .. 8");
        }
        let after = std::sync::Arc::strong_count(&ctx);
        eprintln!("unit window context handles: before {before} after {after}");
        assert_eq!(after, before, "the window leaves no context handle");
    }
}
