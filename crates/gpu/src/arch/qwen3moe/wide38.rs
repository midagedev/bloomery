//! Qwen3.8's ubatch walk ([`Gemm38`]): a prompt's positions as one unit of
//! `m` rows through [`runtime::sched::walk`]'s `(1, m, Batch)` — the front,
//! the shared expert in the shadow of the host's service, the serve, the back
//! — through the host tier's batch port at the ubatch's width, every op over
//! the unit's rows at once:
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
//!   Q8_0 or IQ4_NL blocks) and the card sum into the unit-wide acc — the
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
//! a half of its ring for this unit alone, its copies batched the same way,
//! and the card route runs a second set of its launches over that half
//! ([`CardRoute38::enqueue`]'s ring pass, a GEMM batch a landing batch)
//! before the card sum, which reads the stack's places and the ring's
//! together; the serve's union leaves the streamed experts out. The route is
//! then the layer's reader of the call ([`Gemm38::stream_read`]), which the
//! next ubatch's pick and the ring's next use of the half copy behind. The
//! call's placement stays for the decode after it; the stream's does not. A
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
use crate::host::xstream::{LandBatch, RingLayer, XLayer, XMode, XReport};
use crate::linear::conv::ConvArgs;
use crate::linear::delta::{DeltaArgs, DeltaLanesArgs};
use crate::linear::norm_gate::NormGateArgs;
use crate::model::lookup::{f32_gain, f32_tensor};
use crate::ple::{PleConvArgs, PleGateArgs};
use crate::prompt_timing::Mark;
use crate::q38::{
    CardAccArgs, CardSharedAddArgs, CardTierAccArgs, EmbedQ8Args, KeyAppendArgs, OutGateArgs,
    SharedAddArgs,
};
use crate::qsa::{self, PoolArgs, SelectArgs};
use crate::rope_neox::PartialNeoxArgs;
use crate::tensor::DeviceTensor;
use cuda_core::{CudaEvent, CudaStream, DeviceBuffer};
use gguf::quant::GgmlType;
use runtime::hc_gated::Geometry;
use runtime::sched::{self, At, LayerProgram, Overlap, PortKind};

/// What the walk's errors name.
const WHAT: &str = "qwen4exp ubatch walk";

/// The indexer keys a select keeps, as `qsa::scored` takes them.
const KEPT_U32: u32 = geo::KEPT as u32;
const _: () = assert!(KEPT_U32 as usize == geo::KEPT);

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
/// (`min(walk rows, ROUTE_ROWS)`) tokens and the unit-wide card sum; the
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
    /// The card slots' weighted sums, a row a token of the whole unit.
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
    /// full places from the walk's ids (`ids`, eleven slots a token, the
    /// shared expert's last — never read), the q8_1 of the run's normed rows
    /// (`ffn_x`, `[m][HIDDEN]`), then the stack pass as one GEMM batch a
    /// landing batch of the layer's pick (`land`, [`enqueue_gemm_batch`]):
    /// the leading batch the slots no copy lands in, which wait nothing,
    /// then the admits' slots in slot order, each batch behind its own
    /// landed event — or, when the pick moved nothing, the whole stack at
    /// once over the slot map's layer row (`slots`) as the pass always ran.
    /// The card sum by the router's weights (`weights`, eleven a token)
    /// into the unit-wide acc's rows for the run reads the full places. On a
    /// tier layer (`tiered`) the card sum is the join's
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
        reason = "the route's ids, weights, normed rows, the slot map, the layer, the unit's width, the stream and the pick's batches (rust-quality R8)"
    )]
    pub(super) fn enqueue(
        &mut self,
        c: &Ctx38<'_>,
        card: &Card38,
        l: usize,
        (ffn_x, ids, weights): (&DeviceBuffer<f32>, &DeviceBuffer<u32>, &DeviceBuffer<f32>),
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
        if (tiered && !kept) || m * geo::HIDDEN > self.acc.len() {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "layer {l}: a card route of {m} tokens{} on a route made for {} with kept \
                     rows {kept}",
                    if tiered { " on a tier layer" } else { "" },
                    self.acc.len() / geo::HIDDEN
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
                // SAFETY: tokens c0 .. c0 + n <= m <= rows of the walk's ids
                // and weights (`rows · pitch` each) and the arena's `ffn_x`
                // (`rows · HIDDEN`); the places and the down rows hold ten
                // slots a token of the unit when kept, else of the run, so at
                // .. at + n lies inside them; every buffer stays in place for
                // this run's launches.
                let (ids_w, w_w, x_w, mut acc_w, mut sel_w, mut down_w) = unsafe {
                    (
                        param_view::<u32>(ids, c0 * pitch, n * pitch),
                        f32_view(weights, c0 * pitch, n * pitch),
                        f32_view(ffn_x, c0 * geo::HIDDEN, n * geo::HIDDEN),
                        f32_view(&self.acc, c0 * geo::HIDDEN, n * geo::HIDDEN),
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
                    c.k.q38.enqueue_card_acc(
                        stream,
                        CardAccArgs {
                            down: &down_w,
                            w: &w_w,
                            sel: sum_places.0,
                            n: geo::HIDDEN,
                            m: n,
                            n_card: sum_places.1,
                            fault: sink,
                            acc: &mut acc_w,
                        },
                    )?;
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
    /// token) into the unit-wide acc, which the back's combine reads as it
    /// reads the card sum. Two launches. A route made for no tier is refused
    /// by name.
    #[allow(
        clippy::too_many_arguments,
        reason = "the walk's context, the layer's card and tier counts, the unit's width, the router's weights and the tier's rows (rust-quality R8)"
    )]
    fn enqueue_tier_acc(
        &mut self,
        c: &Ctx38<'_>,
        l: usize,
        (n_card, n_tier): (usize, usize),
        m: usize,
        weights: &DeviceBuffer<f32>,
        trows: &DeviceBuffer<f32>,
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
                acc: &mut self.acc,
            },
        )
    }
}

/// What a ubatch walk reads and writes besides the arena: the one-expert
/// table, the GEMMs' activations, the wide mixes' scratch, the indexer keys
/// token-major, the unit's routed slots, and the last walk's cut of its
/// selecting layers' rows.
pub(super) struct Wide38 {
    /// The most rows a unit holds.
    pub(super) rows: usize,
    dense: GemmRoute,
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
}

impl Wide38 {
    /// The walk's own buffers for units of up to `rows` rows over `card`'s
    /// card experts: the card route's buffers beside them when a layer has
    /// card experts, made for an expert tier's join when `tiered`. Load-time
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
        Ok(Wide38 {
            rows,
            dense: GemmRoute::new(stream, rows, 1)?,
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
        })
    }

    /// Device bytes, the route taps left out.
    pub(super) fn bytes(&self) -> usize {
        self.dense.bytes()
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
/// over the wide arena and the walk's own buffers, the unit's first position
/// and its rows that select nothing ([`dense_rows`]) on the host.
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
/// `mixed`.
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
    (scratch, dense): (&mut HcWideScratch, &GemmRoute),
    mixed: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
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
            scratch,
            gemm: &c.k.g32,
            dense,
            mixed,
        },
    )
}

impl WideParts<'_> {
    /// Layer `l`'s front up to its feed-forward site (`Parts38::front`'s
    /// order): the embedding in front of layer 0, the attention site (with
    /// the PLE site on its layer), the mixer into `y`.
    fn front(&mut self, l: usize) -> Result<(), GpuError> {
        let (plans, gpu, m) = (self.plans, self.c.gpu, self.m);
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
                    ids: self.io.ids,
                    pos0: self.io.pos0,
                    first: self.io.first,
                    fault: gpu.unlabelled_sink(),
                    y: &mut s.emb,
                    pos: &mut s.pos,
                    n_keys: &mut s.n_keys,
                },
            )?;
            mix(
                c,
                &p.attn_hc,
                &mut s.res[self.cur],
                Before::Init { y: &s.emb },
                m,
                sink,
                (&mut x.hc, &x.dense),
                &mut s.mixed,
            )?;
        } else if let Some(pp) = &p.ple {
            let prev = gpu.layer_sink(l - 1)?;
            c.k.hcw
                .enqueue_combine(stream, &mut s.res[self.cur], &s.y, m, prev, &x.hc)?;
            let w = c.w;
            let pl = &mut s.ple;
            c.k.g32
                .enqueue_quantize_gemm32(stream, &pl.e, m, &mut x.act_hid, sink)?;
            gemm_q8(c, &pp.key, (&x.act_hid, m), &x.dense, &mut pl.key)?;
            gemm_q8(c, &pp.value, (&x.act_hid, m), &x.dense, &mut pl.value)?;
            let [r0, r1] = &mut s.res;
            let (xs, out) = if self.cur == 0 {
                (&*r0, r1)
            } else {
                (&*r1, r0)
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
                    pos: &s.pos,
                    taps: geo::PLE_TAPS,
                    dilation: geo::PLE_DILATION,
                    hc: geo::STREAMS,
                    m,
                    fault: sink,
                    out,
                    ring: &mut *self.ple_ring,
                },
            )?;
            self.cur ^= 1;
            mix(
                c,
                &p.attn_hc,
                &mut s.res[self.cur],
                Before::Plain,
                m,
                sink,
                (&mut x.hc, &x.dense),
                &mut s.mixed,
            )?;
        } else {
            mix(
                c,
                &p.attn_hc,
                &mut s.res[self.cur],
                Before::Combine { y: &s.y },
                m,
                sink,
                (&mut x.hc, &x.dense),
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
                gdn(c, g, (rec, stamp, lane), (s, x), m, sink)
            }
            (Mixer38::Qsa(q), Store38::Qsa { kv, raw, pooled }) => {
                let split = qsa(c, q, (kv, raw, pooled), (s, x), (m, self.dense), sink)?;
                x.split = Some(split);
                Ok(())
            }
            _ => Err(GpuError::shape(
                WHAT,
                format!("layer {l}'s plan and store are of two kinds"),
            )),
        }
    }

    /// Layer `l`'s feed-forward site: the mix of the streams, the mixer's
    /// output combined first, into the arena's `ffn_x`.
    fn ffn_mix(&mut self, l: usize) -> Result<(), GpuError> {
        let p = plan_of(self.plans, l)?;
        let sink = self.c.gpu.layer_sink(l)?;
        let (s, x) = (&mut *self.s, &mut *self.x);
        mix(
            &self.c,
            &p.ffn_hc,
            &mut s.res[self.cur],
            Before::Combine { y: &s.y },
            self.m,
            sink,
            (&mut x.hc, &x.dense),
            &mut s.ffn_x,
        )
    }

    /// Layer `l`'s router over the arena's `ffn_x`, in runs of
    /// [`ROUTE_ROWS`] tokens, each run's slots copied into the walk's. When a
    /// gate armed the route taps, each run's logits go into layer `l`'s tap
    /// first; when a gate planted routes, the run is then routed again over
    /// the planted logits of its positions (the shared expert's gate left the
    /// router's); the slots the walk runs go into the tap last.
    fn route(&mut self, l: usize) -> Result<(), GpuError> {
        let p = plan_of(self.plans, l)?;
        let sink = self.c.gpu.layer_sink(l)?;
        let stream = self.c.gpu.stream();
        let (s, x, m) = (&mut *self.s, &mut *self.x, self.m);
        let router = f32_tensor(self.c.w, &p.ffn.router)?;
        let slots = geo::N_USED + 1;
        let width = s.route.dims().logits();
        let mut c0 = 0;
        while c0 < m {
            let n = ROUTE_ROWS.min(m - c0);
            // SAFETY: tokens c0 .. c0 + n of `ffn_x` (`rows · HIDDEN` values,
            // m <= rows); `ffn_x` stays in place while the window lives (this
            // run's router launch).
            let xw = unsafe { f32_view(&s.ffn_x, c0 * geo::HIDDEN, n * geo::HIDDEN) };
            self.c
                .k
                .router
                .enqueue_ubatch(stream, router, &xw, n, sink, &mut s.route)?;
            if let Some(t) = x.taps.as_mut() {
                let lt = t
                    .logits
                    .get_mut(l)
                    .ok_or(GpuError::state(WHAT, "a route tap for every layer"))?;
                // SAFETY: tokens c0 .. c0 + n <= taps.rows (`plan_gemm` refuses
                // a unit past them, `Gemm38::walk` again) of `width` logits,
                // inside the tap and the route's logits (`n <= ROUTE_ROWS`
                // tokens); both stay in place for the copy.
                let (mut tw, lw) = unsafe {
                    (
                        f32_view(lt, c0 * width, n * width),
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
                f.covers(self.pos0 + c0, n)?;
                let at = self.pos0 + c0 - f.pos0;
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
            // SAFETY: tokens c0 .. c0 + n of the walk's slots (`rows · slots`,
            // m <= rows) and the route's first n (<= ROUTE_ROWS) tokens'; every
            // buffer stays in place for the two copies.
            let (mut iw, mut ww, ri, rw) = unsafe {
                (
                    param_view::<u32>(&x.ids, c0 * slots, n * slots),
                    f32_view(&x.weights, c0 * slots, n * slots),
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
            // SAFETY: m <= taps.rows tokens of `slots` ids, inside the tap and
            // the walk's slots; both stay in place for the copy.
            let (mut tw, iw) = unsafe {
                (
                    param_view::<u32>(it, 0, m * slots),
                    param_view::<u32>(&x.ids, 0, m * slots),
                )
            };
            tw.copy_from_device_async(&iw, stream)?;
        }
        Ok(())
    }

    /// Layer `l`'s shared expert over the arena's `ffn_x` into `sh_y`: gate
    /// and up, SwiGLU quantized, down.
    fn shared(&mut self, l: usize) -> Result<(), GpuError> {
        let p = plan_of(self.plans, l)?;
        let sink = self.c.gpu.layer_sink(l)?;
        let (c, m) = (&self.c, self.m);
        let stream = c.gpu.stream();
        let (s, x) = (&mut *self.s, &mut *self.x);
        c.k.g32
            .enqueue_quantize_gemm32(stream, &s.ffn_x, m, &mut x.act_hid, sink)?;
        gemm_q8(c, &p.ffn.gate_sh, (&x.act_hid, m), &x.dense, &mut s.sh_g)?;
        gemm_q8(c, &p.ffn.up_sh, (&x.act_hid, m), &x.dense, &mut s.sh_u)?;
        c.k.g32
            .enqueue_swiglu_quant32(stream, &s.sh_g, &s.sh_u, m, &mut x.act_ff, sink)?;
        gemm_q8(c, &p.ffn.down_sh, (&x.act_ff, m), &x.dense, &mut s.sh_y)
    }

    /// Layer `l`'s card route over the arena's `ffn_x`, when it has card
    /// experts ([`CardRoute38::enqueue`], the walk's module doc): the places
    /// from the walk's ids, the grouped gate·up, SwiGLU and down over the
    /// run's tokens — a GEMM batch a landing batch of the pick (`land`), each
    /// behind its own landed event — the ring pass over the layer's stream
    /// when `ring` is one, the card sums into the unit-wide acc. Called from
    /// the walk's shadow only — after the front's download, the stream order
    /// the route's buffers exist under.
    fn route_card(
        &mut self,
        l: usize,
        ring: Option<&RingLayer>,
        land: &[LandBatch],
    ) -> Result<(), GpuError> {
        if !self.card.has(l) {
            return Ok(());
        }
        let tiered = self.tier_k(l) > 0;
        let (c, m) = (&self.c, self.m);
        let s = &*self.s;
        let x = &mut *self.x;
        let Some(r) = x.card.as_mut() else {
            return Ok(());
        };
        r.enqueue(
            c,
            self.card,
            l,
            (&s.ffn_x, &x.ids, &x.weights),
            self.slots,
            (m, tiered),
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
    /// from the walk's ids and the map's tier view into the card route's,
    /// then the activations, the routing and those places into the batch
    /// port's set ([`crate::host::HostTier::enqueue_download_tiered_pitched`]),
    /// which the tier's service reads.
    fn download_tiered(
        &mut self,
        port: &mut BatchLeg<'_, HostRun>,
        key: crate::host::batch::BatchKey,
    ) -> Result<(), GpuError> {
        let l = key.layer;
        let tier = self
            .tier
            .ok_or(GpuError::state(WHAT, "the tier side of a tier layer"))?;
        let (c, m) = (&self.c, self.m);
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
                ids: &x.ids,
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
    /// the card sum the back's combine reads.
    fn tier_acc(&mut self, l: usize, trows: &DeviceBuffer<f32>) -> Result<(), GpuError> {
        let n_tier = self.tier_k(l);
        let n_card = self.card.n_card(l);
        let (c, m, x) = (&self.c, self.m, &mut *self.x);
        let r = x
            .card
            .as_mut()
            .ok_or(GpuError::state(WHAT, "the card route of a tier layer"))?;
        r.enqueue_tier_acc(c, l, (n_card, n_tier), m, &x.weights, trows)
    }

    /// Layer `l`'s block output into `y`: the host's routed sums `hsum`, on
    /// a layer with card experts the card route's sums too, plus the shared
    /// expert's output times its gate weight.
    fn shared_add(&mut self, l: usize, hsum: &DeviceBuffer<f32>) -> Result<(), GpuError> {
        let sink = self.c.gpu.layer_sink(l)?;
        let s = &mut *self.s;
        let x = &mut *self.x;
        let (stream, q38) = (self.c.gpu.stream(), &self.c.k.q38);
        let (slot, slots, n, m) = (geo::N_USED, geo::N_USED + 1, geo::HIDDEN, self.m);
        // A layer with card experts has the walk's card route (both read the
        // one loaded map); a layer without them keeps the plain sum — its
        // rows of the acc were never this walk's.
        match (self.card.has(l), x.card.as_ref()) {
            (true, Some(r)) => q38.enqueue_card_shared_add(
                stream,
                CardSharedAddArgs {
                    hsum,
                    acc: &r.acc,
                    sh: &s.sh_y,
                    w: &x.weights,
                    slot,
                    slots,
                    n,
                    m,
                    fault: sink,
                    y: &mut s.y,
                },
            ),
            (false, _) => q38.enqueue_shared_add(
                stream,
                SharedAddArgs {
                    hsum,
                    sh: &s.sh_y,
                    w: &x.weights,
                    slot,
                    slots,
                    n,
                    m,
                    fault: sink,
                    y: &mut s.y,
                },
            ),
            (true, None) => Err(GpuError::state(
                WHAT,
                "the walk's card route over a map with card experts",
            )),
        }
    }
}

/// A delta layer's mixer at `m` rows over its store `r`, its lanes' stamps
/// and the lane word, the committed lane in place; the attention site's mix
/// in `s.mixed`, its output projection into `s.y`.
fn gdn(
    c: &Ctx38<'_>,
    gp: &GdnPlan,
    (r, stamp, lane): (&mut RecStore, &mut DeviceBuffer<u32>, &DeviceBuffer<u32>),
    (s, x): (&mut Arena38, &mut Wide38),
    m: usize,
    sink: FaultSink,
) -> Result<(), GpuError> {
    let (w, stream) = (c.w, c.gpu.stream());
    let nv = GDN.n_v;
    let Arena38 {
        mixed,
        gdn: g,
        pos,
        attn,
        y,
        ..
    } = s;
    c.k.g32
        .enqueue_quantize_gemm32(stream, mixed, m, &mut x.act_hid, sink)?;
    gemm_q8(c, &gp.qkv, (&x.act_hid, m), &x.dense, &mut g.x)?;
    gemm_q8(c, &gp.z, (&x.act_hid, m), &x.dense, &mut g.z)?;
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
            pos: &*pos,
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
            pos: &*pos,
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
    gemm_q8(c, &gp.ssm_out, (&x.act_attn, m), &x.dense, y)
}

/// A selecting attention layer's mixer at `m` rows over its store (the K/V
/// planes, the raw and pooled indexer keys), the first `dense` of them
/// selecting nothing ([`dense_rows`]), the attention site's mix in
/// `s.mixed`, its output projection into `s.y`. Returns the rows the
/// prefill flash ran and those the selection did.
fn qsa(
    c: &Ctx38<'_>,
    qp: &QsaPlan,
    (kv, raw, pooled): (
        &mut KvPlanes,
        &mut DeviceBuffer<u16>,
        &mut DeviceBuffer<u16>,
    ),
    (s, x): (&mut Arena38, &mut Wide38),
    (m, dense): (usize, usize),
    sink: FaultSink,
) -> Result<(usize, usize), GpuError> {
    let (w, stream, ctx) = (c.w, c.gpu.stream(), c.ctx);
    let Arena38 {
        mixed,
        qsa: q,
        pos,
        n_keys,
        flash,
        attn,
        y,
        ..
    } = s;
    c.k.g32
        .enqueue_quantize_gemm32(stream, mixed, m, &mut x.act_hid, sink)?;
    gemm_q8(c, &qp.q, (&x.act_hid, m), &x.dense, &mut q.qg)?;
    gemm_q8(c, &qp.k, (&x.act_hid, m), &x.dense, &mut q.k)?;
    gemm_q8(c, &qp.v, (&x.act_hid, m), &x.dense, &mut q.v)?;
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
            pos: &*pos,
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
            n_keys: &*n_keys,
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
            pos: &*pos,
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
        // SAFETY: rows 0 .. dense <= m of the queries, the counts and the
        // flash's output, inside their `rows` rows; they stay in place for
        // the launch.
        let (qw, nw, mut yw) = unsafe {
            (
                f32_view(&q.q, 0, dense * geo::ATTN),
                param_view::<u32>(n_keys, 0, dense),
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
        // SAFETY: rows t0 .. t0 + n <= m of the indexer queries, the counts,
        // the queries and the flash's output, inside their `rows` rows; they
        // stay in place for this run's launches.
        let (qi, nw, qw, mut yw) = unsafe {
            (
                f32_view(&q.qi, t0 * idx, n * idx),
                param_view::<u32>(n_keys, t0, n),
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
    gemm_q8(c, &qp.out, (&x.act_attn, m), &x.dense, y)?;
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
    /// The card tail a split pick's walk balances with, set when a streaming
    /// call opens: the family's measured one on the Q4 kind, none on a kind
    /// whose route no record has priced.
    pub(super) card_tail_us: f64,
    /// The ubatch the walk runs, from 0, the pick records' group.
    pub(super) ubatch: usize,
    counts: Vec<u32>,
    pub(super) picks: Vec<(usize, CallPick)>,
    pub(super) end: Option<CallReport>,
    /// Per streamed layer of the call, its ubatch and what it streamed; the
    /// stream's end.
    pub(super) layers: Vec<(usize, XLayer)>,
    pub(super) xend: Option<XReport>,
}

/// A ubatch's walk `(1, m, Batch)`: every layer through the batch port at
/// the unit's width; the head is the caller's ([`Gemm38::head`]).
pub(super) struct Gemm38<'a> {
    pub(super) p: WideParts<'a>,
}

impl<'a> Gemm38<'a> {
    /// Walk every layer at the unit's `m` columns through `leg`: the
    /// one-expert table filled for them first. A unit of no row, or past the
    /// arena's rows or the armed route taps', is refused by name before any
    /// launch (`Body38::plan_gemm` refuses the taps' first, before the call
    /// moves anything; this is the windows' own guard). Once every layer is
    /// walked, armed taps record the unit's positions.
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
        if let Some(t) = self.p.x.taps.as_mut() {
            t.walked = None;
        }
        self.p.c.k.gemm.enqueue_route_dense(
            gpu.stream(),
            m,
            &mut self.p.x.dense,
            gpu.unlabelled_sink(),
        )?;
        let o = Overlap {
            units: 1,
            cols: m,
            port: PortKind::Batch,
        };
        let layers = self.p.plans.len();
        leg.begin_walk(1)?;
        sched::walk(o, layers, leg, self)?;
        leg.end_walk()?;
        if let Some(t) = self.p.x.taps.as_mut() {
            t.walked = Some((self.p.pos0, m));
        }
        Ok(())
    }

    /// Under host streaming, on a layer with card experts, the layer's pick
    /// before its card route: the host waits for the front's download of the
    /// unit's routed ids, counts them, and the residency machine's pick
    /// (`SwapMachine::call_pick`) sends the pool's coldest residents to the
    /// host and copies the unit's hottest host experts over them — the least
    /// count admitted [`STREAM_FLOOR`] on a load whose lane never probed,
    /// the walk's own floor and backlog bound
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
    /// for its card route is returned. Nothing else, and nothing in a unit
    /// narrower than the gate the pick admits from.
    fn stream_pick(
        &mut self,
        port: &mut BatchLeg<'a, HostRun>,
        at: At,
    ) -> Result<(Option<RingLayer>, Vec<LandBatch>), GpuError> {
        let l = at.layer;
        if !self.p.stream.on || !self.p.card.has(l) {
            return Ok((None, Vec::new()));
        }
        let split = self.p.stream.split;
        // The walk only where the lane probed: its gate is the least width
        // whose picks could pay a first admit. A load with no probe admits
        // from the stream floor, so short prompts move what they did.
        let mut k = port.hybrid().xstream().and_then(|x| x.constants(l));
        let walked = split && k.is_some();
        let gate = match k.as_mut() {
            Some(k) if split => {
                k.card_tail_us = self.p.stream.card_tail_us;
                runtime::xsplit::walk_gate(k)
            }
            _ => STREAM_FLOOR,
        };
        // A unit of fewer rows than the gate gives no expert a count the
        // pick admits (a call's last ubatch can be one).
        if self.p.m < gate as usize {
            return Ok((None, Vec::new()));
        }
        let key = port.key(at);
        let stream = self.p.c.gpu.stream();
        let s = &mut *self.p.stream;
        let mut pick = match (split, k) {
            (true, Some(k)) => port.hybrid().call_pick_walk(
                stream,
                key,
                &mut s.counts,
                k,
                crate::host::swap::RING_SLOTS as u64,
            )?,
            _ => port
                .hybrid()
                .call_pick_routed(stream, key, &mut s.counts, usize::MAX)?,
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
            .xstream_layer(stream, key, self.p.m, &s.counts, admitted)?;
        s.layers.push((s.ubatch, x));
        Ok((port.hybrid().xstream_ring(l), land))
    }

    /// Under host streaming, on a layer with card experts, the layer's last
    /// read of its slots in the call so far, after its card route
    /// (`SwapMachine::call_reader`): the next ubatch's pick of the layer
    /// copies behind it; under `split` the read of its stream's half too
    /// ([`crate::host::HostTier::xstream_read`]), which the half's next
    /// stream copies behind. Nothing else.
    fn stream_read(&mut self, port: &mut BatchLeg<'a, HostRun>, l: usize) -> Result<(), GpuError> {
        if !self.p.stream.on || !self.p.card.has(l) {
            return Ok(());
        }
        let stream = self.p.c.gpu.stream();
        port.hybrid().call_reader(l, stream)?;
        if self.p.stream.split {
            port.hybrid().xstream_read(l, stream)?;
        }
        Ok(())
    }

    /// After the walk: the head's mix over the unit's columns (the last
    /// layer's combine first), its last row into `head`'s input, and the
    /// head.
    pub(super) fn head(&mut self, head: &mut Head) -> Result<(), GpuError> {
        let (m, h, gpu) = (self.p.m, geo::HIDDEN, self.p.c.gpu);
        let fault = gpu.fault_sink(LAYER_HEAD);
        {
            let (s, x) = (&mut *self.p.s, &mut *self.p.x);
            mix(
                &self.p.c,
                &head_site(),
                &mut s.res[self.p.cur],
                Before::Combine { y: &s.y },
                m,
                fault,
                (&mut x.hc, &x.dense),
                &mut s.mixed,
            )?;
        }
        // SAFETY: 1 <= m <= rows (the walk refused anything else), so row m −
        // 1 spans `hidden` values inside `mixed` (`rows · hidden`), which
        // stays in place while the window lives (one copy).
        let row = unsafe { f32_view(&self.p.s.mixed, (m - 1) * h, h) };
        head.input_mut()
            .copy_from_device_async(&row, gpu.stream())?;
        head.enqueue(gpu, self.p.c.w)
    }
}

impl<'a> LayerProgram for Gemm38<'a> {
    type Port = BatchLeg<'a, HostRun>;

    /// The layer up to its router, the mix into the arena's `ffn_x`, then
    /// the download of the unit's activations and routed slots — on a tier
    /// layer with the unit's tier places beside them
    /// ([`WideParts::download_tiered`]).
    fn front(&mut self, port: &mut BatchLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let l = at.layer;
        let enq = port.part_start();
        port.mark(at, Mark::Front as usize)?;
        self.p.front(l)?;
        self.p.ffn_mix(l)?;
        self.p.route(l)?;
        port.mark(at, Mark::FrontEnd as usize)?;
        let key = port.key(at);
        let stream = self.p.c.gpu.stream();
        let r = if self.p.tier_k(l) > 0 {
            self.p.download_tiered(port, key)
        } else {
            port.hybrid().enqueue_download_pitched(
                stream,
                [&self.p.s.ffn_x, &self.p.x.weights],
                &self.p.x.ids,
                geo::N_USED + 1,
                key,
            )
        };
        let marked = r.and_then(|()| port.mark(at, Mark::Down as usize));
        port.part_end(at, enq);
        marked
    }

    /// The card route on a layer with card experts, then the shared expert
    /// over the arena's `ffn_x`. The route is enqueued here and nowhere
    /// else — after the front's download, the stream order the route's
    /// buffers exist under (its module doc): enqueued before the download,
    /// the d2h would queue behind the route's card GEMMs and the host tier
    /// would start late by their whole time. Under host streaming the shared
    /// expert comes first, so the card runs it while the host counts and
    /// picks; then the layer's pick (and stream) and the route, its GEMM
    /// batches each behind the landing batch it reads ([`Gemm38::stream_pick`]),
    /// the route the layer's last read of its slots in the call
    /// ([`Gemm38::stream_read`]).
    fn shadow(&mut self, port: &mut BatchLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let enq = port.part_start();
        let r = if self.p.stream.on {
            self.p
                .shared(at.layer)
                .and_then(|()| self.stream_pick(port, at))
                .and_then(|(ring, land)| self.p.route_card(at.layer, ring.as_ref(), &land))
                .and_then(|()| self.stream_read(port, at.layer))
        } else {
            self.p
                .route_card(at.layer, None, &[])
                .and_then(|()| self.p.shared(at.layer))
        }
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
        let joined = if self.p.tier_k(l) > 0 {
            let p = &mut self.p;
            port.join_tiered(at, |_, trows| p.tier_acc(l, trows))
        } else {
            Ok(())
        };
        let r = joined
            .and_then(|()| self.p.shared_add(l, port.hsum()))
            .and_then(|()| port.mark(at, Mark::Back as usize));
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
