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
//! - the router the ubatch instance of the 512-expert router, in runs of
//!   [`ROUTE_ROWS`] tokens, bit for bit the fused launch for each token;
//! - the routed experts the slot map puts on the card, on a layer with card
//!   experts, the card route ([`CardRoute38`]): per [`ROUTE_ROWS`] run, in
//!   the shadow after the front's download, the compressed ids and the
//!   places from the walk's ids, the remapped route table, the gate and up
//!   GEMMs of the layer's type (Q4_K, Q5_K, IQ3_XXS or IQ4_XS), the card
//!   slots' SwiGLU
//!   (`swiglu_quant32_sel`), the down GEMM of its type (the file's Q5_1,
//!   Q8_0 or IQ4_NL blocks) and the card sum into the unit's acc — the
//!   host tier
//!   serves the rest of the slots, and the back adds the card sum on a
//!   layer with card experts (`q38_card_shared_add`);
//! - the selecting layer's rows the prefill flash while `qsa::scored` says a
//!   row selects nothing (every key below its count), the selection and the
//!   selected flash in runs of [`SELECT_ROWS`] from the first row that
//!   selects; the pool pass over every row first, so a pool some row of the
//!   unit completes is written before any row selects;
//! - the conv, the delta step (the committed lane, in place), the gated
//!   norm, the PLE gate and conv, the key append, the rope with the cache
//!   append, the out gate and the gated sum the launches the other walks
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
//! first position, rows that select nothing, current stream buffer and
//! one-expert table (the second unit its own, filled for its width). The
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
//! then the layer's reader of the call ([`WideParts::stream_read`]), which the
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
//! ([`WideParts::stream_return`]); a split pick there prices an admit's way back
//! when the lane fell behind ([`runtime::xsplit::ReturnCost`]), and the
//! call's end issues what is left. The stream's placement never stays. A
//! token's bits then depend on its ubatch's routing too — which of its
//! experts the pick moved or the stream sent sums on the card, not the host
//! — so the numeric class above holds for an unstreamed walk only.

use super::body::ATTN_SCALE_256;
use super::card38::Card38;
use super::plan38::{GDN, GdnPlan, HcSite, Layer38, Mixer38, QsaPlan, geo, head_site};
use super::program38::{Ctx38, q8};
use super::scratch::{Io, KvPlanes, RecStore, f32_view, param_view};
use super::scratch38::{Arena38, ROUTE_ROWS, SELECT_ROWS, Store38};
use super::tier38::TierSide38;
use crate::GpuError;
use crate::card_slots::CardAcc;
use crate::fault::{FaultSink, LAYER_HEAD};
use crate::flash_gqa::GqaSelArgs;
use crate::flash_gqa_prefill::GqaPrefillArgs;
use crate::gemm::{
    Gemm32Args, Gemm32Weight, GemmAct, GemmAct32, GemmArgs, GemmInput, GemmRoute, GemmWeight,
};
use crate::hc_gated::{Before, HcWideScratch, SiteWeights, WideMixArgs};
use crate::head::Head;
use crate::host::BatchLeg;
use crate::host::handoff::Places;
use crate::host::run::HostRun;
use crate::host::swap::{CallPick, CallReport};
use crate::host::xstream::{GROUP_WALK_ORDER, LandBatch, RingLayer, XLayer, XMode, XReport};
use crate::linear::conv::ConvArgs;
use crate::linear::delta::{DeltaArgs, DeltaLanesArgs};
use crate::linear::norm_gate::NormGateArgs;
use crate::model::lookup::{f32_gain, f32_tensor};
use crate::ple::{PleConvArgs, PleGateArgs};
use crate::prompt_timing::Mark;
use crate::q38::{
    CardSharedAddArgs, CardTierAccArgs, EmbedQ8Args, KeyAppendArgs, OutGateArgs, SLOTS,
    SharedAddArgs, W_PITCH,
};
use crate::qsa::{self, PoolArgs, SelectArgs};
use crate::rope_neox::PartialNeoxArgs;
use crate::tensor::{DeviceTensor, Window};
use cuda_core::{CudaEvent, CudaStream, DeviceBuffer};
use gguf::quant::GgmlType;
use runtime::hc_gated::Geometry;
use runtime::sched::{self, At, BatchOrder, LayerProgram, Overlap, PortKind};
use runtime::splitwalk::{self, LoadKind, SplitCosts};
use runtime::xsplit::ReturnCost;
use std::mem::ManuallyDrop;

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
/// nothing and dies with the unit.
type Win<T> = ManuallyDrop<DeviceBuffer<T>>;

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
    if at.off + at.len > parent.len() {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "{name}: a unit's window ends at word {} of {}",
                at.off + at.len,
                parent.len()
            ),
        ));
    }
    let ptr = parent.cu_deviceptr() + (at.off * size_of::<T>()) as u64;
    // SAFETY: the window's words lie inside `parent` (checked above), which
    // the walk that holds the window keeps alive and in place.
    Ok(unsafe { crate::tensor::window(ptr, at.len, parent.context()) })
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
    /// Which of `res` holds the unit's streams now.
    cur: usize,
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
    res: [Win<f32>; 2],
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
    /// The unit's rows of a selecting layer the prefill flash ran and those
    /// the selection did, once a selecting layer's front ran.
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
        Ok(UnitRows38 {
            r0,
            m,
            pos0: p.pos0 + r0,
            dense: unit_dense(p.dense, r0, m),
            cur: p.cur,
            route,
            first: p.io.first + r0,
            tok: rows_of("the record's ids", p.io.ids, at.tok)?,
            emb: rows_of("emb", &s.emb, at.emb)?,
            pos: rows_of("pos", &s.pos, at.pos)?,
            n_keys: rows_of("n_keys", &s.n_keys, at.n_keys)?,
            res: [
                rows_of("res[0]", &s.res[0], at.res)?,
                rows_of("res[1]", &s.res[1], at.res)?,
            ],
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
}

/// The wide mixes' scratch with its combine weights on one unit's rows: a
/// mix reads the weights the unit's last inject site left and writes its own
/// there, so two units' weights never meet, whatever parts of the other unit
/// run between a unit's two sites. Dropped, the scratch holds its own
/// weights again.
struct UnitHc<'s> {
    hc: &'s mut HcWideScratch,
    own: Option<DeviceBuffer<f32>>,
}

impl<'s> UnitHc<'s> {
    /// `hc` with the unit's rows `wgt` of its combine weights in place of
    /// the whole buffer; refused by name past the buffer's rows.
    fn on(hc: &'s mut HcWideScratch, wgt: Span) -> Result<UnitHc<'s>, GpuError> {
        let w = rows_of("the mixes' combine weights", &hc.wgt, wgt)?;
        let own = std::mem::replace(&mut hc.wgt, ManuallyDrop::into_inner(w));
        Ok(UnitHc { hc, own: Some(own) })
    }
}

impl Drop for UnitHc<'_> {
    fn drop(&mut self) {
        if let Some(own) = self.own.take() {
            // The window frees nothing: its parts are given back, the words
            // stay the scratch's own buffer's.
            drop(std::mem::replace(&mut self.hc.wgt, own).into_raw_parts());
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
/// card, through the grouped GEMMs, in runs of [`ROUTE_ROWS`] tokens (the
/// walk's module doc). Holds its own buffers for one run of `run`
/// (`min(walk rows, ROUTE_ROWS)`) tokens and the walk-wide card sum; the
/// stacks are the leg's ([`Card38`]'s), read through [`Card38::stacks`], and
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
    run: usize,
    /// The compressed routed ids, ten a run token: the places launch over
    /// the walk's ids (a pitch of eleven) with an identity map, whose bytes
    /// are the remapped route's input.
    ids: DeviceBuffer<u32>,
    /// The slots' places, ten a run token — ten a token of the whole unit on
    /// a load with an expert tier: the places launch over the same ids with
    /// the slot map's layer row.
    sel: DeviceBuffer<u32>,
    /// The run's normed rows (`[run][HIDDEN]`) in the K-quant GEMM's
    /// activation form — the same q8_1 bytes a `Q8Act` of the same values
    /// holds, the gate and up GEMMs' input.
    x: GemmAct,
    /// The gate and up GEMMs' outputs, ten slots a run token of [`geo::FF`]
    /// values, slot-major.
    g: DeviceBuffer<f32>,
    u: DeviceBuffer<f32>,
    /// The card slots' SwiGLU in the 32-value GEMM's activation form, ten
    /// columns a run token; a host slot's column is never written and never
    /// read (the route leaves it to the host tier).
    act: GemmAct32,
    /// The down GEMM's output, ten slots a run token of [`geo::HIDDEN`]
    /// values, slot-major — a token of the whole unit on a load with an
    /// expert tier, whose join reads every run's rows after the host's wait.
    down: DeviceBuffer<f32>,
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

/// The ring pass's buffers: the route table over a half of the expert
/// stream's ring, and per run the slots' places in the half (the ring row's)
/// and in the card sum's places (the union row's), ten a run token.
struct RingRoute38 {
    route: GemmRoute,
    sel: DeviceBuffer<u32>,
    union: DeviceBuffer<u32>,
}

/// Layer `l`'s down stack `w` of type `ty` as the 32-value GEMM reads it:
/// the file's Q5_1, Q8_0 or IQ4_NL blocks; any other type refused by name.
fn down_weight(
    l: usize,
    ty: GgmlType,
    w: &DeviceTensor<u32>,
) -> Result<Gemm32Weight<'_>, GpuError> {
    match ty {
        GgmlType::Q5_1 => Ok(Gemm32Weight::Q5_1File(w)),
        GgmlType::Q8_0 => Ok(Gemm32Weight::Q8_0File(w)),
        GgmlType::IQ4_NL => Ok(Gemm32Weight::Iq4NlFile(w)),
        other => Err(GpuError::shape(
            WHAT,
            format!("layer {l}: a down stack of {other}, which no route GEMM reads"),
        )),
    }
}

/// One GEMM batch of a route pass over `map` — the batch's places, a
/// host-mapped row of a landing batch ([`LandBatch`]) or the card's own map
/// copy for the whole of a pass nothing batched — with the engine stream
/// `stream` waiting `wait` first (the batch's landed event; `None` for a
/// batch nothing lands in): the places into `sel`, the route table over the
/// map, the gate and up GEMMs, the SwiGLU over `sel` (every place below
/// `n_places` a column it writes) and the down GEMM into the run's `down_y`
/// rows. Six launches; only the batch's slots are touched, so a pass's
/// batches write the rows the whole pass always wrote, the same kernels on
/// the same values — the bits do not move.
#[allow(
    clippy::too_many_arguments,
    reason = "the batch's context, wait, ids, map, weights, width, places bound, sink, run rows and buffers (rust-quality R8)"
)]
fn enqueue_gemm_batch(
    c: &Ctx38<'_>,
    stream: &CudaStream,
    wait: Option<&CudaEvent>,
    (walk_ids, ids): (&DeviceBuffer<u32>, &DeviceBuffer<u32>),
    map: &DeviceBuffer<u32>,
    (ty, gate, up): (GemmWeight, &DeviceTensor<u32>, &DeviceTensor<u32>),
    down: Gemm32Weight<'_>,
    (n, n_places): (usize, usize),
    sink: FaultSink,
    down_y: &mut DeviceBuffer<f32>,
    (x, g, u, act): (
        &GemmAct,
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
    c.k.g32
        .enqueue_route_remap(stream, ids, map, n * geo::N_USED, route, sink)?;
    for (w, y) in [(gate, &mut *g), (up, &mut *u)] {
        c.k.gemm.enqueue_gemm(
            stream,
            GemmArgs {
                ty,
                w,
                rows_per_expert: geo::FF,
                act: x,
                route,
                input: GemmInput::Shared { top_k: geo::N_USED },
                y,
            },
        )?;
    }
    c.k.g32
        .enqueue_swiglu_quant32_sel(stream, g, u, sel, n_places, n * geo::N_USED, act, sink)?;
    c.k.g32.enqueue_gemm32(
        stream,
        Gemm32Args {
            w: down,
            rows_per_expert: geo::HIDDEN,
            act,
            route,
            input: GemmInput::PerSlot,
            y: down_y,
        },
    )
}

impl CardRoute38 {
    /// The route's buffers for ubatches of up to `rows` tokens over `card`'s
    /// card experts: a run of `min(rows, ROUTE_ROWS)` tokens, the sums a
    /// token of the whole unit, and one route table per distinct card count
    /// (refused by name past two, naming them); `tiered`, a load with an
    /// expert tier, the places and the down rows a token of the whole unit
    /// and the unit's tier places beside them
    /// (`place::card_tier_join_bytes`). Load-time only.
    pub(super) fn new(
        stream: &CudaStream,
        rows: usize,
        card: &Card38,
        tiered: bool,
    ) -> Result<CardRoute38, GpuError> {
        let run = rows.min(ROUTE_ROWS);
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
        Ok(CardRoute38 {
            run,
            ids: DeviceBuffer::zeroed(stream, slots)?,
            sel: DeviceBuffer::zeroed(stream, kept)?,
            x: GemmAct::new(stream, run, geo::HIDDEN)?,
            g: DeviceBuffer::zeroed(stream, slots * geo::FF)?,
            u: DeviceBuffer::zeroed(stream, slots * geo::FF)?,
            act: GemmAct32::new(stream, slots, geo::FF)?,
            down: DeviceBuffer::zeroed(stream, kept * geo::HIDDEN)?,
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
            + self.x.bytes()
            + self.g.num_bytes()
            + self.u.num_bytes()
            + self.act.bytes()
            + self.down.num_bytes()
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
    /// (`ffn_x`, `[m][HIDDEN]`), then the stack pass as one GEMM batch a
    /// landing batch of the layer's pick (`land`, [`enqueue_gemm_batch`]):
    /// the leading batch the slots no copy lands in, which wait nothing,
    /// then the admits' slots in slot order, each batch behind its own
    /// landed event — or, when the pick moved nothing, the whole stack at
    /// once over the slot map's layer row (`slots`) as the pass always ran.
    /// The card sum by the router's weights (`weights`, eleven a token)
    /// into the unit's acc (`acc`, `[m][HIDDEN]`) for the run reads the full
    /// places. On a tier layer (`tiered`) the card sum is the join's
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
    /// stream on a tier layer, and a stream on a route the load did not arm
    /// ([`CardRoute38::arm_ring`]). Asynchronous, allocation-free,
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
        (ffn_x, ids, weights, acc): (
            &DeviceBuffer<f32>,
            &DeviceBuffer<u32>,
            &DeviceBuffer<f32>,
            &mut DeviceBuffer<f32>,
        ),
        slots: &DeviceTensor<u32>,
        (m, tiered): (usize, bool),
        ring: Option<&RingLayer>,
        land: &[LandBatch],
    ) -> Result<(), GpuError> {
        let st = card.stacks(c.w, l)?;
        let (n_card, gate, up) = (st.n_card, st.gate, st.up);
        let gate_up_ty = GemmWeight::from_ggml(st.gate_up_ty)?;
        let down = down_weight(l, st.down_ty, st.down)?;
        let (gpu, stream, sink) = (c.gpu, c.gpu.stream(), c.gpu.layer_sink(l)?);
        let pitch = geo::N_USED + 1;
        // SAFETY: layer l's row of the slot map's card copy (`EXPERTS` words
        // a row), which stays resident while the window lives (this layer's
        // route launches).
        let map = unsafe { param_view::<u32>(slots.buf(), l * geo::EXPERTS, geo::EXPERTS) };
        let kept = self.tsel.is_some();
        if (tiered && !kept) || m * geo::HIDDEN > acc.len() {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "layer {l}: a card route of {m} tokens{} on a card sum of {} with kept \
                     rows {kept}",
                    if tiered { " on a tier layer" } else { "" },
                    acc.len() / geo::HIDDEN
                ),
            ));
        }
        if ring.is_some() && (tiered || self.ring.is_none()) {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "layer {l}: a stream of the expert stream's ring on a {}",
                    if tiered {
                        "tier layer (the ring serves a load with no expert tier)"
                    } else {
                        "card route the load did not arm for it"
                    }
                ),
            ));
        }
        let ctx = gpu.context();
        // SAFETY: the stream's union row is `EXPERTS` words of its mapped
        // host rows, and its half holds `half_slots` experts of this layer's
        // part bytes in each part from the half's first slot — its gate and
        // up rows of the stack's words, its down rows likewise
        // (`XStream::ring_layer`); all stay allocated while the stream lives,
        // past these launches, and the windows are given back below.
        let windows = ring.map(|r| unsafe {
            (
                crate::tensor::window::<u32>(r.union_map, geo::EXPERTS, ctx),
                DeviceTensor::<u32>::window(r.parts[0], r.half_slots * geo::FF, gate.cols(), ctx),
                DeviceTensor::<u32>::window(r.parts[1], r.half_slots * geo::FF, up.cols(), ctx),
                DeviceTensor::<u32>::window(
                    r.parts[2],
                    r.half_slots * geo::HIDDEN,
                    st.down.cols(),
                    ctx,
                ),
            )
        });
        let ran = (|| {
            let mut c0 = 0;
            while c0 < m {
                let n = self.run.min(m - c0);
                // The places and the down rows of the run: from the run's
                // first token of the unit when the route keeps them for the
                // join, else from the run buffers' start.
                let at = if kept { c0 } else { 0 };
                // SAFETY: tokens c0 .. c0 + n <= m of the unit's ids and
                // weights (`m · pitch` each), its `ffn_x` and its card sum
                // (`m · HIDDEN` each); the places and the down rows hold ten
                // slots a token of the unit when kept, else of the run, so at
                // .. at + n lies inside them; every buffer stays in place for
                // this run's launches.
                let (ids_w, w_w, x_w, mut acc_w, mut sel_w, mut down_w) = unsafe {
                    (
                        param_view::<u32>(ids, c0 * pitch, n * pitch),
                        f32_view(weights, c0 * pitch, n * pitch),
                        f32_view(ffn_x, c0 * geo::HIDDEN, n * geo::HIDDEN),
                        f32_view(acc, c0 * geo::HIDDEN, n * geo::HIDDEN),
                        param_view::<u32>(&self.sel, at * geo::N_USED, n * geo::N_USED),
                        f32_view(
                            &self.down,
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
                gpu.enqueue_quantize_gemm(&x_w, n, &mut self.x, sink)?;
                // The stack pass: one GEMM batch a landing batch of the
                // layer's pick, the leading one (nothing lands in it) first,
                // or the whole stack at once when the pick moved nothing.
                if land.is_empty() {
                    enqueue_gemm_batch(
                        c,
                        stream,
                        None,
                        (&ids_w, &self.ids),
                        &map,
                        (gate_up_ty, gate, up),
                        down,
                        (n, n_card),
                        sink,
                        &mut down_w,
                        (&self.x, &mut self.g, &mut self.u, &mut self.act),
                        (&mut sel_w, &mut self.routes[table]),
                    )?;
                } else {
                    // The full places beside the batches: the card sum's
                    // (the join's on a tier layer).
                    places(&map, &mut sel_w)?;
                    for b in land {
                        // SAFETY: the batch's map row is `EXPERTS` words of
                        // its host-mapped row (`LandBatch`), filled before
                        // this launch was enqueued and reused only after the
                        // walk's order has this run complete.
                        let bmap =
                            unsafe { crate::tensor::window::<u32>(b.map, geo::EXPERTS, ctx) };
                        let ran = enqueue_gemm_batch(
                            c,
                            stream,
                            b.event.as_deref(),
                            (&ids_w, &self.ids),
                            &bmap,
                            (gate_up_ty, gate, up),
                            down,
                            (n, n_card),
                            sink,
                            &mut down_w,
                            (&self.x, &mut self.g, &mut self.u, &mut self.act),
                            (&mut self.bsel, &mut self.routes[table]),
                        );
                        drop(std::mem::ManuallyDrop::into_inner(bmap).into_raw_parts());
                        ran?;
                    }
                }
                // The card sum's places and experts: the stack's, or with a
                // stream the union row's over the stack's and the ring's.
                let mut sum_places = (&*sel_w, n_card);
                if let (Some(r), Some((union_map, wg, wu, wd)), Some(rr)) =
                    (ring, windows.as_ref(), self.ring.as_mut())
                {
                    places(union_map, &mut rr.union)?;
                    for b in &r.batches {
                        // SAFETY: as the stack pass's batch rows, the
                        // stream's own.
                        let bmap =
                            unsafe { crate::tensor::window::<u32>(b.map, geo::EXPERTS, ctx) };
                        let ran = enqueue_gemm_batch(
                            c,
                            stream,
                            b.event.as_deref(),
                            (&ids_w, &self.ids),
                            &bmap,
                            (gate_up_ty, wg, wu),
                            down_weight(l, st.down_ty, wd)?,
                            (n, r.n),
                            sink,
                            &mut down_w,
                            (&self.x, &mut self.g, &mut self.u, &mut self.act),
                            (&mut rr.sel, &mut rr.route),
                        );
                        drop(std::mem::ManuallyDrop::into_inner(bmap).into_raw_parts());
                        ran?;
                    }
                    sum_places = (&rr.union, n_card + r.n);
                }
                if !tiered {
                    let a = CardAcc {
                        down: &down_w,
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
                c0 += n;
            }
            Ok(())
        })();
        if let Some((union_map, wg, wu, wd)) = windows {
            drop(std::mem::ManuallyDrop::into_inner(union_map).into_raw_parts());
            for w in [wg, wu, wd] {
                DeviceTensor::release(w);
            }
        }
        ran
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
                down: &self.down,
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
/// token-major, the walk's routed slots, the last walk's last unit and its cut
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
    /// expert's input), of the attention-width rows (the output
    /// projections' input), and of the shared expert's SwiGLU.
    act_hid: GemmAct32,
    act_attn: GemmAct32,
    act_ff: GemmAct32,
    hc: HcWideScratch,
    /// The indexer keys token-major, as the tile writes them, before their
    /// copy into the append's layout.
    kr: DeviceBuffer<f32>,
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
    /// The last walk's rows of each selecting layer: those the prefill flash
    /// ran and those the selection did.
    pub(super) split: Option<(usize, usize)>,
    /// The last walk's last unit, whose rows the head reads; its windows
    /// stay valid until the next walk replaces it.
    last: Option<UnitRows38>,
    /// How the walks cut their rows ([`SplitWalk`]), and what the load keeps
    /// its routed experts on, which the rule prices only when resident.
    split_walk: SplitWalk,
    load: LoadKind,
    /// The width rule's costs on this card ([`SPLIT_COSTS`] at its
    /// multiprocessors).
    split_costs: SplitCosts,
    /// The units the last walk ran; 0 before a walk.
    pub(super) walked_units: usize,
}

impl Wide38 {
    /// The walk's own buffers for units of up to `rows` rows over `card`'s
    /// card experts: the card route's buffers beside them when a layer has
    /// card experts, made for an expert tier's join when `tiered`; the walks
    /// cut by the width rule on a load with no expert tier, as one unit on
    /// one with it ([`Wide38::paged`] for an NVMe-paged plan). Load-time
    /// only.
    pub(super) fn new(
        stream: &CudaStream,
        rows: usize,
        card: &Card38,
        tiered: bool,
    ) -> Result<Wide38, GpuError> {
        let geometry = Geometry::new(geo::STREAMS as u32, geo::RANK as u32, geo::HIDDEN as u32)
            .map_err(|e| GpuError::shape(WHAT, e.to_string()))?;
        let slots = geo::N_USED + 1;
        let card = (card.card_layers() > 0)
            .then(|| CardRoute38::new(stream, rows, card, tiered))
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
            act_hid: GemmAct32::new(stream, rows, geo::HIDDEN)?,
            act_attn: GemmAct32::new(stream, rows, geo::ATTN)?,
            act_ff: GemmAct32::new(stream, rows, geo::FF)?,
            hc: HcWideScratch::new(stream, geometry, rows)?,
            kr: DeviceBuffer::zeroed(stream, rows * geo::IDX_DIM)?,
            ids: DeviceBuffer::zeroed(stream, rows * slots)?,
            weights: DeviceBuffer::zeroed(stream, rows * slots)?,
            taps: None,
            force: None,
            card,
            split: None,
            last: None,
            split_walk,
            load,
            split_costs: SplitCosts {
                card_row_us: SPLIT_COSTS.card_row_us * SPLIT_COSTS_SMS / sms,
                ..SPLIT_COSTS
            },
            walked_units: 0,
        })
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
            + self.kr.num_bytes()
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
/// runs for ([`UnitRows38`]). `m`, `pos0`, `dense` and `cur` are the walk's
/// whole: its rows, the position of its first, its leading rows that select
/// nothing ([`dense_rows`]) on the host, and the stream buffer it starts in;
/// the walk cuts them into its units.
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
    pub(super) cur: usize,
    /// The prompt call's host streaming: whether this walk picks, and its
    /// records.
    pub(super) stream: &'a mut Stream38,
}

/// `y = W · act` for the Q8_0 weight `name` over the first `m` columns of
/// `act`, token-major, through the walk's one-expert table `dense`, which
/// must be filled for exactly those `m` slots (else refused by name: a table
/// of fewer would leave the rest of `y` as another call left it).
fn gemm_q8(
    c: &Ctx38<'_>,
    name: &str,
    (act, m): (&GemmAct32, usize),
    dense: &GemmRoute,
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    if dense.filled() != Some(m) {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "{name}: the one-expert table is filled for {:?} slots, the unit has {m}",
                dense.filled()
            ),
        ));
    }
    let (qs, d) = q8(c.w, name)?;
    c.k.g32.enqueue_gemm32(
        c.gpu.stream(),
        Gemm32Args {
            w: Gemm32Weight::Q8_0Plane { qs, d },
            rows_per_expert: qs.rows(),
            act,
            route: dense,
            input: GemmInput::PerSlot,
            y,
        },
    )
}

/// The wide mix of `site` over `m` columns of `res`, after `before`, into
/// `mixed`, the scratch's combine weights on the unit's rows `wgt`
/// ([`UnitHc`]).
#[allow(
    clippy::too_many_arguments,
    reason = "one site's streams, rule, width, sink, scratch, table and output (rust-quality R8)"
)]
fn mix(
    c: &Ctx38<'_>,
    site: &HcSite,
    res: &mut DeviceBuffer<f32>,
    before: Before<'_>,
    m: usize,
    fault: FaultSink,
    (scratch, wgt, dense): (&mut HcWideScratch, Span, &GemmRoute),
    mixed: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    let hc = UnitHc::on(scratch, wgt)?;
    let w = c.w;
    let (down_qs, down_d) = q8(w, &site.down)?;
    let (up_qs, up_d) = q8(w, &site.up)?;
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
                down_qs,
                down_d,
                up_qs,
                up_d,
                inject,
            },
            eps: c.eps,
            m,
            fault,
            scratch: &mut *hc.hc,
            gemm: &c.k.g32,
            dense,
            mixed,
        },
    )
}

impl WideParts<'_> {
    /// Layer `l`'s front of unit `u` up to its feed-forward site
    /// (`Parts38::front`'s order): the embedding in front of layer 0, the
    /// attention site (with the PLE site on its layer), the mixer into the
    /// unit's `y`.
    fn front(&mut self, u: &mut UnitRows38, l: usize) -> Result<(), GpuError> {
        let (plans, gpu, m) = (self.plans, self.c.gpu, u.m);
        let p = plan_of(plans, l)?;
        let sink = gpu.layer_sink(l)?;
        let stream = gpu.stream();
        let c = &self.c;
        let s = &mut *self.s;
        let x = &mut *self.x;
        if l == 0 {
            let (qs, d) = q8(c.w, &model::arch::qwen35moe::names::token_embd())?;
            c.k.q38.enqueue_embed_rows(
                stream,
                EmbedQ8Args {
                    qs,
                    d,
                    ids: &u.tok,
                    pos0: self.io.pos0,
                    first: u.first,
                    fault: gpu.unlabelled_sink(),
                    y: &mut u.emb,
                    pos: &mut u.pos,
                    n_keys: &mut u.n_keys,
                },
            )?;
            mix(
                c,
                &p.attn_hc,
                &mut u.res[u.cur],
                Before::Init { y: &u.emb },
                m,
                sink,
                (&mut x.hc, u.wgt, &x.dense[u.route]),
                &mut s.mixed,
            )?;
        } else if let Some(pp) = &p.ple {
            let prev = gpu.layer_sink(l - 1)?;
            {
                let hc = UnitHc::on(&mut x.hc, u.wgt)?;
                c.k.hcw
                    .enqueue_combine(stream, &mut u.res[u.cur], &u.y, m, prev, &*hc.hc)?;
            }
            let w = c.w;
            let pl = &mut s.ple;
            c.k.g32
                .enqueue_quantize_gemm32(stream, &u.ple_e, m, &mut x.act_hid, sink)?;
            let dense = &x.dense[u.route];
            gemm_q8(c, &pp.key, (&x.act_hid, m), dense, &mut pl.key)?;
            gemm_q8(c, &pp.value, (&x.act_hid, m), dense, &mut pl.value)?;
            let [res0, res1] = &mut u.res;
            let (xs, out) = if u.cur == 0 {
                (&*res0, res1)
            } else {
                (&*res1, res0)
            };
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
                    out,
                    ring: &mut *self.ple_ring,
                },
            )?;
            u.cur ^= 1;
            mix(
                c,
                &p.attn_hc,
                &mut u.res[u.cur],
                Before::Plain,
                m,
                sink,
                (&mut x.hc, u.wgt, dense),
                &mut s.mixed,
            )?;
        } else {
            mix(
                c,
                &p.attn_hc,
                &mut u.res[u.cur],
                Before::Combine { y: &u.y },
                m,
                sink,
                (&mut x.hc, u.wgt, &x.dense[u.route]),
                &mut s.mixed,
            )?;
        }
        let store = self
            .stores
            .get_mut(l)
            .ok_or(GpuError::state(WHAT, "a store for every layer"))?;
        match (&p.mixer, store) {
            (Mixer38::Gdn(g), Store38::Rec { rec, stamp }) => {
                let lane = self.io.lane.ok_or(GpuError::state(
                    WHAT,
                    "the record's lane word (a chain with delta layers carries one)",
                ))?;
                gdn(c, g, (rec, stamp, lane), (s, x), u, sink)
            }
            (Mixer38::Qsa(q), Store38::Qsa { kv, raw, pooled }) => {
                let split = qsa(c, q, (kv, raw, pooled), (s, x), u, sink)?;
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
            &mut u.res[u.cur],
            Before::Combine { y: &u.y },
            u.m,
            sink,
            (&mut x.hc, u.wgt, &x.dense[u.route]),
            &mut u.ffn_x,
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
        let router = f32_tensor(self.c.w, &p.ffn.router)?;
        let slots = geo::N_USED + 1;
        let width = s.route.dims().logits();
        let mut c0 = 0;
        while c0 < m {
            let n = ROUTE_ROWS.min(m - c0);
            // SAFETY: tokens c0 .. c0 + n of the unit's `ffn_x` (`m` rows,
            // c0 + n <= m); `ffn_x` stays in place while the window lives
            // (this run's router launch).
            let xw = unsafe { f32_view(&u.ffn_x, c0 * geo::HIDDEN, n * geo::HIDDEN) };
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
                        f32_view(lt, (u.r0 + c0) * width, n * width),
                        f32_view(&s.route.logits, 0, n * width),
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
                            f32_view(&s.route.logits, t * width, geo::EXPERTS),
                            f32_view(lf, (at + t) * geo::EXPERTS, geo::EXPERTS),
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
                    f32_view(&u.weights, c0 * slots, n * slots),
                    param_view::<u32>(&s.route.ids, 0, n * slots),
                    f32_view(&s.route.weights, 0, n * slots),
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
    /// gate and up, SwiGLU quantized, down.
    fn shared(&mut self, u: &mut UnitRows38, l: usize) -> Result<(), GpuError> {
        let p = plan_of(self.plans, l)?;
        let sink = self.c.gpu.layer_sink(l)?;
        let (c, m) = (&self.c, u.m);
        let stream = c.gpu.stream();
        let (s, x) = (&mut *self.s, &mut *self.x);
        c.k.g32
            .enqueue_quantize_gemm32(stream, &u.ffn_x, m, &mut x.act_hid, sink)?;
        let dense = &x.dense[u.route];
        gemm_q8(c, &p.ffn.gate_sh, (&x.act_hid, m), dense, &mut s.sh_g)?;
        gemm_q8(c, &p.ffn.up_sh, (&x.act_hid, m), dense, &mut s.sh_u)?;
        c.k.g32
            .enqueue_swiglu_quant32(stream, &s.sh_g, &s.sh_u, m, &mut x.act_ff, sink)?;
        gemm_q8(c, &p.ffn.down_sh, (&x.act_ff, m), dense, &mut u.sh_y)
    }

    /// Layer `l`'s card route over unit `u`'s `ffn_x`, when it has card
    /// experts ([`CardRoute38::enqueue`], the walk's module doc): the places
    /// from the unit's ids, the grouped gate·up, SwiGLU and down over the
    /// run's tokens — a GEMM batch a landing batch of the pick (`land`), each
    /// behind its own landed event — the ring pass over the layer's stream
    /// when `ring` is one, the card sums into the unit's acc. Called from
    /// the walk's shadow only — after the front's download, the stream order
    /// the route's buffers exist under.
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
        let acc = u.acc.as_mut().ok_or(GpuError::state(
            WHAT,
            "a unit's card sum over a map with card experts",
        ))?;
        r.enqueue(
            c,
            self.card,
            l,
            (&u.ffn_x, &u.ids, &u.weights, acc),
            self.slots,
            (u.m, tiered),
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

    /// Layer `l`'s block output into unit `u`'s `y`: the host's routed sums
    /// (its `hsum`), on a layer with card experts the card route's sums too,
    /// plus the shared expert's output times its gate weight.
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
/// stamps and the lane word, the committed lane in place; the attention
/// site's mix in `s.mixed`, its output projection into the unit's `y`.
fn gdn(
    c: &Ctx38<'_>,
    gp: &GdnPlan,
    (r, stamp, lane): (&mut RecStore, &mut DeviceBuffer<u32>, &DeviceBuffer<u32>),
    (s, x): (&mut Arena38, &mut Wide38),
    u: &mut UnitRows38,
    sink: FaultSink,
) -> Result<(), GpuError> {
    let (w, stream, m) = (c.w, c.gpu.stream(), u.m);
    let nv = GDN.n_v;
    let Arena38 {
        mixed,
        gdn: g,
        attn,
        ..
    } = s;
    c.k.g32
        .enqueue_quantize_gemm32(stream, mixed, m, &mut x.act_hid, sink)?;
    let dense = &x.dense[u.route];
    gemm_q8(c, &gp.qkv, (&x.act_hid, m), dense, &mut g.x)?;
    gemm_q8(c, &gp.z, (&x.act_hid, m), dense, &mut g.z)?;
    let ba = f32_tensor(w, &gp.beta_alpha)?;
    let k = ba.cols();
    // SAFETY: the joined stack is `2·nv` rows of `k` (`plan38::plans`
    // refuses another shape at load), β's rows its first `nv · k` values and
    // α's the next; the stack stays resident while the windows live (the two
    // launches), which are given back below.
    let (wb, wa) = unsafe {
        let at = ba.buf().cu_deviceptr();
        (
            DeviceTensor::<f32>::window(at, nv, k, ba.buf().context()),
            DeviceTensor::<f32>::window(
                at + (nv * k * size_of::<f32>()) as u64,
                nv,
                k,
                ba.buf().context(),
            ),
        )
    };
    let r1 = c.k.g32.enqueue_f32_tile(stream, &wb, mixed, m, &mut g.b);
    let r2 = c.k.g32.enqueue_f32_tile(stream, &wa, mixed, m, &mut g.a);
    DeviceTensor::release(wb);
    DeviceTensor::release(wa);
    r1?;
    r2?;
    let lin = &c.k.linear;
    lin.conv.enqueue_conv_prep(
        stream,
        ConvArgs {
            x: &g.x,
            b_raw: &g.b,
            a_raw: &g.a,
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
    lin.delta.enqueue_delta_lanes(
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
    )?;
    lin.norm_gate.enqueue_norm_gate_sigmoid(
        stream,
        NormGateArgs {
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
    c.k.g32
        .enqueue_quantize_gemm32(stream, attn, m, &mut x.act_attn, sink)?;
    gemm_q8(c, &gp.ssm_out, (&x.act_attn, m), dense, &mut u.y)
}

/// A selecting attention layer's mixer over unit `u`'s rows and its store
/// (the K/V planes, the raw and pooled indexer keys), the first `u.dense` of
/// them selecting nothing ([`dense_rows`]), the attention site's mix in
/// `s.mixed`, its output projection into the unit's `y`. Returns the rows
/// the prefill flash ran and those the selection did.
fn qsa(
    c: &Ctx38<'_>,
    qp: &QsaPlan,
    (kv, raw, pooled): (
        &mut KvPlanes,
        &mut DeviceBuffer<u16>,
        &mut DeviceBuffer<u16>,
    ),
    (s, x): (&mut Arena38, &mut Wide38),
    u: &mut UnitRows38,
    sink: FaultSink,
) -> Result<(usize, usize), GpuError> {
    let (w, stream, ctx) = (c.w, c.gpu.stream(), c.ctx);
    let (m, dense) = (u.m, u.dense);
    let Arena38 {
        mixed,
        qsa: q,
        flash,
        attn,
        ..
    } = s;
    c.k.g32
        .enqueue_quantize_gemm32(stream, mixed, m, &mut x.act_hid, sink)?;
    let table = &x.dense[u.route];
    gemm_q8(c, &qp.q, (&x.act_hid, m), table, &mut q.qg)?;
    gemm_q8(c, &qp.k, (&x.act_hid, m), table, &mut q.k)?;
    gemm_q8(c, &qp.v, (&x.act_hid, m), table, &mut q.v)?;
    c.k.g32
        .enqueue_f32_tile(stream, f32_tensor(w, &qp.idx_k)?, mixed, m, &mut x.kr)?;
    // The append reads the keys row-major (`[IDX_DIM][m]`, the gemv's
    // layout): the tile's `[m][IDX_DIM]` turned.
    c.k.proj
        .enqueue_token_major(stream, &x.kr, m, geo::IDX_DIM, &mut q.kr)?;
    c.k.q38.enqueue_key_append(
        stream,
        KeyAppendArgs {
            kr: &q.kr,
            pos: &u.pos,
            m,
            ctx,
            fault: sink,
            raw: &mut *raw,
        },
    )?;
    c.k.qsa.enqueue_pool(
        stream,
        PoolArgs {
            raw: &*raw,
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
    c.k.g32
        .enqueue_f32_tile(stream, f32_tensor(w, &qp.idx_q)?, mixed, m, &mut q.qi)?;
    // The selecting store is f16 (the qwen38 family's stores carry no q8_0
    // form), so the family's shared append runs its f16 arm.
    let (kcache, vcache) = kv.f16_mut("qwen38::wide")?;
    c.k.neox.enqueue_head_norm_neox_append_256(
        stream,
        PartialNeoxArgs {
            qg: &q.qg,
            q: &mut q.q,
            k: &mut q.k,
            v: &q.v,
            gq: f32_gain(w, &qp.q_norm)?,
            gk: f32_gain(w, &qp.k_norm)?,
            table: c.table,
            pos: &u.pos,
            eps: c.eps,
            n_head: geo::N_HEAD,
            n_kv: geo::N_KV,
            ctx,
            m,
            fault: sink,
            cache_k: kcache,
            cache_v: vcache,
        },
    )?;
    let (kcache, vcache) = kv.f16("qwen38::wide")?;
    if dense > 0 {
        // SAFETY: rows 0 .. dense <= m of the queries and the flash's output
        // (`rows` rows each) and of the unit's counts (`m` rows); they stay
        // in place for the launch.
        let (qw, nw, mut yw) = unsafe {
            (
                f32_view(&q.q, 0, dense * geo::ATTN),
                param_view::<u32>(&u.n_keys, 0, dense),
                f32_view(flash, 0, dense * geo::ATTN),
            )
        };
        c.k.prefill.enqueue_256_p4(
            stream,
            GqaPrefillArgs {
                q: &qw,
                kc: kcache,
                vc: vcache,
                n_keys: &nw,
                scale: ATTN_SCALE_256,
                n_head: geo::N_HEAD,
                n_kv: geo::N_KV,
                ctx,
                t: dense,
                fault: sink,
                y: &mut yw,
            },
        )?;
    }
    let width = q.sel.width();
    let idx = geo::IDX_HEADS * geo::IDX_DIM;
    let mut t0 = dense;
    while t0 < m {
        let n = SELECT_ROWS.min(m - t0);
        // SAFETY: rows t0 .. t0 + n <= m of the indexer queries, the queries
        // and the flash's output (`rows` rows each) and of the unit's counts
        // (`m` rows); they stay in place for this run's launches.
        let (qi, nw, qw, mut yw) = unsafe {
            (
                f32_view(&q.qi, t0 * idx, n * idx),
                param_view::<u32>(&u.n_keys, t0, n),
                f32_view(&q.q, t0 * geo::ATTN, n * geo::ATTN),
                f32_view(flash, t0 * geo::ATTN, n * geo::ATTN),
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
        c.k.flash.enqueue_pass_256_p12_sel(
            stream,
            GqaSelArgs {
                q: &qw,
                kc: kcache,
                vc: vcache,
                list: &q.sel.list,
                n_sel: &q.sel.n_sel,
                width,
                scale: ATTN_SCALE_256,
                n_kv: geo::N_KV,
                ctx,
                m: n,
                part_v: &mut q.part_v,
                part_ms: &mut q.part_ms,
                fault: sink,
                y: &mut yw,
            },
            geo::N_HEAD,
        )?;
        t0 += n;
    }
    c.k.q38.enqueue_out_gate(
        stream,
        OutGateArgs {
            attn: &*flash,
            qg: &q.qg,
            n_head: geo::N_HEAD,
            m,
            fault: sink,
            y: &mut *attn,
        },
    )?;
    c.k.g32
        .enqueue_quantize_gemm32(stream, attn, m, &mut x.act_attn, sink)?;
    gemm_q8(c, &qp.out, (&x.act_attn, m), table, &mut u.y)?;
    Ok((dense, m - dense))
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

/// A prompt call's host streaming (`BLOOMERY_XSTREAM`, the residency
/// machine's call mode and the expert stream): what the next prompt calls
/// move, whether one is streaming now, the pick's count buffer, and the
/// call's pick, stream and end records for the binary that prints them.
#[derive(Default)]
pub(super) struct Stream38 {
    /// What the next prompt calls move ([`super::body38::Body38::set_xstream`]).
    pub(super) mode: Option<XMode>,
    /// A call is streaming: every ubatch walk of it picks.
    pub(super) on: bool,
    /// The streaming call runs the expert stream too (`split`).
    pub(super) split: bool,
    /// The card tail rate a split pick's walk prices its unit with [µs per
    /// card pick], set when a streaming call opens: the family's measured
    /// one on the Q4 kind, none on a kind whose route no record has priced.
    pub(super) card_tail_us_per_pick: f64,
    /// The ubatch the walk runs, from 0, the pick records' group.
    pub(super) ubatch: usize,
    /// The streaming call returns to its start ([`WideParts::returns_here`]):
    /// decided at its begin, where the decode history is read.
    pub(super) returning: bool,
    /// The streaming call's end position: the walk of the unit that ends
    /// there is the call's last.
    pub(super) call_end: usize,
    /// The layer of the last unit whose host service ran and whose return the
    /// walk asks for at the next layer's shadow.
    pub(super) owed: Option<usize>,
    /// The call's return runs whole at its end instead of in the walk: the
    /// twin a gate holds the walk's return to (`Body38::set_return_at_end`).
    pub(super) return_at_end: bool,
    counts: Vec<u32>,
    pub(super) picks: Vec<(usize, CallPick)>,
    pub(super) end: Option<CallReport>,
    /// Per streamed layer of the call, its ubatch and what it streamed; the
    /// stream's end.
    pub(super) layers: Vec<(usize, XLayer)>,
    pub(super) xend: Option<XReport>,
}

/// A ubatch's walk `(units, cols, Batch)`: every layer through the batch
/// port at each unit's width; the head is the caller's ([`Gemm38::head`]).
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
    /// cuts of the selecting layers' rows add up to the walk's, and the last
    /// unit stays for the head.
    pub(super) fn walk(&mut self, leg: &mut BatchLeg<'a, HostRun>) -> Result<(), GpuError> {
        let m = self.p.m;
        let taps = self.p.x.taps.as_ref().map_or(usize::MAX, |t| t.rows);
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
        self.p.x.last = None;
        self.p.x.walked_units = 0;
        if let Some(t) = self.p.x.taps.as_mut() {
            t.walked = None;
        }
        let cut = unit_rows(m, &self.p.widths()?)?;
        let rows: Vec<std::ops::Range<usize>> = cut.iter().map(|&(r0, w)| r0..r0 + w).collect();
        leg.set_unit_rows(&rows);
        let mut units = Vec::with_capacity(cut.len());
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
        if let Some(l) = self.p.stream.owed.take() {
            leg.hybrid().call_return(l, gpu.stream())?;
        }
        leg.end_walk()?;
        if let Some(t) = self.p.x.taps.as_mut() {
            t.walked = Some((self.p.pos0, m));
        }
        self.p.x.split = units
            .iter()
            .try_fold((0, 0), |(d, s), u| u.split.map(|(ud, us)| (d + ud, s + us)));
        self.p.x.walked_units = units.len();
        self.p.x.last = units.into_iter().last();
        Ok(())
    }

    /// After the walk: the head's mix over the last unit's columns (the last
    /// layer's combine first), its last row into `head`'s input, and the
    /// head. Refused by name before a walk.
    pub(super) fn head(&mut self, head: &mut Head) -> Result<(), GpuError> {
        let (h, gpu) = (geo::HIDDEN, self.p.c.gpu);
        let fault = gpu.fault_sink(LAYER_HEAD);
        let s = &mut *self.p.s;
        let Wide38 {
            last, hc, dense, ..
        } = &mut *self.p.x;
        let u = last
            .as_mut()
            .ok_or(GpuError::state(WHAT, "a ubatch walk before its head"))?;
        let m = u.m;
        mix(
            &self.p.c,
            &head_site(),
            &mut u.res[u.cur],
            Before::Combine { y: &u.y },
            m,
            fault,
            (hc, u.wgt, &dense[u.route]),
            &mut s.mixed,
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

    /// Under host streaming, on a layer with card experts, the layer's pick
    /// before its card route, for the first unit of `m` rows of a walk of
    /// `units` units: the host waits for the
    /// front's download of the unit's routed ids, counts them, and the
    /// residency machine's pick (`SwapMachine::call_pick`) sends the pool's
    /// coldest residents to the host and copies the unit's hottest host
    /// experts over them — the least count admitted [`STREAM_FLOOR`] on a
    /// load whose lane never probed, the walk's own floor and backlog bound
    /// ([`crate::host::HostTier::call_pick_walk`], every copy at the
    /// beside-union rate) where it did — in victim-slot order, a landing
    /// batch a third of them, each batch's event on the copy stream after
    /// its last copy ([`LandBatch`]) — the host map and the card's copy of
    /// the layer's words moved at once, so the union the walk's serve runs
    /// next, and the card route, run under the moved map, each GEMM batch
    /// waiting its own batch's event and no later copy. Under `split` the
    /// expert stream then takes the layer
    /// ([`crate::host::HostTier::xstream_layer`]): the host experts its rule
    /// sends to the card stream into a half of its ring, its copies batched
    /// the same way, and the serve leaves them out; the layer's ring view
    /// for its card route is returned. Nothing else, and nothing in a walk
    /// narrower than the gate the pick admits from.
    ///
    /// One pick a layer for the walk's units: the first unit's routed ids
    /// count `units` times each (the group's width, `units · m`), so the
    /// pick, its floor and the stream's rule price the group; under `split`
    /// every later unit joins the layer's stream
    /// ([`crate::host::HostTier::xstream_share`]) before any card route
    /// reads it, so its serve leaves out the same experts and the stream
    /// stays open for its card route. A later unit's card route reads the
    /// same landing batches and ring ([`UnitWalk`]'s `held`).
    fn stream_pick(
        &mut self,
        port: &mut BatchLeg<'a, HostRun>,
        at: At,
        (m, units): (usize, usize),
    ) -> Result<(Option<RingLayer>, Vec<LandBatch>), GpuError> {
        let l = at.layer;
        if !self.stream.on || !self.card.has(l) {
            return Ok((None, Vec::new()));
        }
        let scale = u32::try_from(units)
            .map_err(|_| GpuError::shape(WHAT, format!("a walk of {units} units")))?;
        let group = units * m;
        let split = self.stream.split;
        // The walk only where the lane probed: its gate is the least width
        // whose picks could pay a first admit, derived without the tail (the
        // counts it prices from do not exist yet). A load with no probe
        // admits from the stream floor, so short prompts move what they did.
        let k = port.hybrid().xstream().and_then(|x| x.constants(l));
        let walked = split && k.is_some();
        let gate = match k.as_ref() {
            Some(k) if split => runtime::xsplit::walk_gate(k),
            _ => STREAM_FLOOR,
        };
        // A walk of fewer rows than the gate gives no expert a count the
        // pick admits (a call's last ubatch can be one).
        if group < gate as usize {
            return Ok((None, Vec::new()));
        }
        let key = port.key(at);
        let stream = self.c.gpu.stream();
        let ret = self.return_cost(port, l);
        let s = &mut *self.stream;
        let mut pick = match (split, k) {
            (true, Some(k)) => port.hybrid().call_pick_walk_scaled(
                stream,
                key,
                &mut s.counts,
                k,
                s.card_tail_us_per_pick,
                ret,
                scale,
            )?,
            _ => port.hybrid().call_pick_routed_scaled(
                stream,
                key,
                &mut s.counts,
                usize::MAX,
                scale,
            )?,
        };
        // The no-probe floor witness: the pick admitted from the stream
        // floor rather than the walk's measured plan.
        pick.fallback = u32::from(!walked);
        let admitted = pick.admitted;
        let land = std::mem::take(&mut pick.land);
        s.picks.push((s.ubatch, pick));
        if !split {
            return Ok((None, land));
        }
        let x = port
            .hybrid()
            .xstream_layer(stream, key, group, &s.counts, admitted)?;
        s.layers.push((s.ubatch, x));
        for unit in 1..units {
            let other = port.key(At { unit, layer: l });
            port.hybrid().xstream_share(l, other)?;
        }
        Ok((port.hybrid().xstream_ring(l), land))
    }

    /// Whether this walk is the last of a call that returns to its start: it
    /// returns each layer once the layer's host service has run.
    fn returns_here(&self) -> bool {
        let s = &*self.stream;
        s.on && s.returning && self.pos0 + self.m == s.call_end
    }

    /// In the returning call's last walk, at a layer's shadow — its front
    /// already enqueued, so the card never waits on this — the return of the
    /// layer whose host service ran last
    /// ([`crate::host::HostTier::call_return`]), then the call's queued
    /// return copies pumped while the host waits for this layer's download
    /// ([`crate::host::HostTier::call_return_pump`]): the copies fill the
    /// host's idle wait for the card's front, and the layer's pick issues its
    /// own behind a ring's worth of them at most. Nothing else.
    fn stream_return(&mut self, port: &mut BatchLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        if !self.returns_here() {
            return Ok(());
        }
        if let Some(l) = self.stream.owed.take() {
            port.hybrid().call_return(l, self.c.gpu.stream())?;
        }
        let key = port.key(at);
        port.hybrid().call_return_pump(key)?;
        Ok(())
    }

    /// The return cost of an admit at layer `l`'s pick
    /// ([`runtime::xsplit::ReturnCost`]): exposed in the returning call's
    /// last walk while the call owes return copies the walk has not issued
    /// (the lane fell behind the fronts), and at the walk's last card layer,
    /// whose return no front follows; hidden otherwise.
    fn return_cost(&self, port: &mut BatchLeg<'a, HostRun>, l: usize) -> ReturnCost {
        if !self.returns_here() {
            return ReturnCost::Hidden;
        }
        let last = !(l + 1..self.plans.len()).any(|j| self.card.has(j));
        if last || port.hybrid().call_returns_owed() > 0 {
            ReturnCost::Exposed
        } else {
            ReturnCost::Hidden
        }
    }

    /// Under host streaming, on a layer with card experts, a unit's read of
    /// the layer after its card route. The walk's `last` unit's is the
    /// layer's last read of its slots in the call so far
    /// (`SwapMachine::call_reader`): the next ubatch's pick of the layer and
    /// the call's return copies of it wait behind that read — recorded at an
    /// earlier unit, they would land while a later unit's route reads the
    /// slots. Under `split` every unit reads the layer's stream
    /// ([`crate::host::HostTier::xstream_read`]), which closes at the
    /// group's last, the half's next stream copying behind it. Nothing else.
    fn stream_read(
        &mut self,
        port: &mut BatchLeg<'a, HostRun>,
        l: usize,
        last: bool,
    ) -> Result<(), GpuError> {
        if !self.stream.on || !self.card.has(l) {
            return Ok(());
        }
        let stream = self.c.gpu.stream();
        if last {
            port.hybrid().call_reader(l, stream)?;
        }
        if self.stream.split {
            port.hybrid().xstream_read(l, stream)?;
        }
        Ok(())
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

    /// The layer up to its router, the mix into the unit's `ffn_x`, then
    /// the download of the unit's activations and routed slots — on a tier
    /// layer with the unit's tier places beside them
    /// ([`WideParts::download_tiered`]). The download names the unit's rows
    /// by the port's key and takes the arena's buffers whole.
    fn front(&mut self, port: &mut BatchLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let l = at.layer;
        let enq = port.part_start();
        port.mark(at, Mark::Front as usize)?;
        let u = unit_at(&mut self.units, at)?;
        let p = &mut *self.p;
        p.front(u, l)?;
        p.ffn_mix(u, l)?;
        p.route(u, l)?;
        port.mark(at, Mark::FrontEnd as usize)?;
        let key = port.key(at);
        let stream = p.c.gpu.stream();
        let r = if p.tier_k(l) > 0 {
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

    /// The card route on a layer with card experts, then the shared expert
    /// over the unit's `ffn_x`. The route is enqueued here and nowhere
    /// else — after the front's download, the stream order the route's
    /// buffers exist under (its module doc): enqueued before the download,
    /// the d2h would queue behind the route's card GEMMs and the host tier
    /// would start late by their whole time. Under host streaming the shared
    /// expert comes first, so the card runs it while the host counts and
    /// picks; then the layer's pick (and stream), the first unit's alone
    /// ([`WideParts::stream_pick`]), and the route, its GEMM batches each
    /// behind the landing batch it reads — a later unit's over the first
    /// unit's pick, held for it — then the unit's read
    /// ([`WideParts::stream_read`]), the last unit's the layer's last read of
    /// its slots in the call. The shadow's first launch is marked for the
    /// walk's timer: the order puts the other unit's parts between a unit's
    /// download and its shadow.
    fn shadow(&mut self, port: &mut BatchLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let l = at.layer;
        let enq = port.part_start();
        let units = self.units.len();
        let held = &mut self.held;
        let u = unit_at(&mut self.units, at)?;
        let p = &mut *self.p;
        let last = at.unit + 1 == units;
        let r = port
            .mark(at, Mark::ShadowStart as usize)
            .and_then(|()| {
                if !p.stream.on {
                    return p.route_card(u, l, None, &[]).and_then(|()| p.shared(u, l));
                }
                p.shared(u, l)?;
                p.stream_return(port, at)?;
                let (ring, land) = if at.unit == 0 {
                    p.stream_pick(port, at, (u.m, units))?
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
                p.stream_read(port, l, last)
            })
            .and_then(|()| port.mark(at, Mark::Shadow as usize));
        port.part_end(at, enq);
        r
    }

    /// The gated sum over the host sums the walk's serve uploaded, the card
    /// route's beside them on a layer with card experts. On a tier layer the
    /// join first ([`BatchLeg::join_tiered`]): once the host has seen the
    /// tier's service complete, the card sum over both cards' slots
    /// ([`WideParts::tier_acc`]), then the host sums' upload.
    fn back(&mut self, port: &mut BatchLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let l = at.layer;
        let enq = port.part_start();
        let units = self.units.len();
        let u = unit_at(&mut self.units, at)?;
        let p = &mut *self.p;
        let joined = if p.tier_k(l) > 0 {
            port.join_tiered(at, |_, trows| p.tier_acc(u, l, trows))
        } else {
            Ok(())
        };
        let r = joined
            .and_then(|()| p.shared_add(u, l))
            .and_then(|()| port.mark(at, Mark::Back as usize));
        // The layer's host service has run for every unit once its last
        // unit's back runs: in the returning call's last walk its return is
        // due, asked at the next layer's shadow.
        if r.is_ok() && at.unit + 1 == units && p.returns_here() && p.card.has(l) {
            p.stream.owed = Some(l);
        }
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
        unit_rows, unit_widths,
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
}
