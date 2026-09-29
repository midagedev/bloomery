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
//!   places from the walk's ids, the remapped route table, the Q4_K gate
//!   and up GEMMs, the card slots' SwiGLU (`swiglu_quant32_sel`), the Q5_1
//!   down GEMM and the card sum into the unit-wide acc — the host tier
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
//! experts, and holds no card buffers on a map without them: a slot map with
//! an expert on a tier card is refused by name at its entry (`card38`),
//! before anything moves.

use super::body::ATTN_SCALE_256;
use super::card38::Card38;
use super::plan38::{GDN, GdnPlan, HcSite, Layer38, Mixer38, QsaPlan, geo, head_site};
use super::program38::{Ctx38, q8};
use super::scratch::{Io, KvPlanes, RecStore, f32_view, param_view};
use super::scratch38::{Arena38, ROUTE_ROWS, SELECT_ROWS, Store38};
use crate::GpuError;
use crate::fault::{FaultSink, LAYER_HEAD};
use crate::flash_gqa::GqaSelArgs;
use crate::flash_gqa_prefill::GqaPrefillArgs;
use crate::gemm::{
    Gemm32Args, Gemm32Weight, GemmAct, GemmAct32, GemmArgs, GemmInput, GemmRoute, GemmWeight,
};
use crate::hc_gated::{Before, HcWideScratch, SiteWeights, WideMixArgs};
use crate::head::Head;
use crate::host::handoff::Places;
use crate::host::run::HostRun;
use crate::host::{BatchLeg, LegTimer, ServeNote};
use crate::linear::conv::ConvArgs;
use crate::linear::delta::{DeltaArgs, DeltaLanesArgs};
use crate::linear::norm_gate::NormGateArgs;
use crate::model::lookup::{f32_gain, f32_tensor};
use crate::ple::{PleConvArgs, PleGateArgs};
use crate::q38::{
    CardAccArgs, CardSharedAddArgs, EmbedQ8Args, KeyAppendArgs, OutGateArgs, SharedAddArgs,
};
use crate::qsa::{self, PoolArgs, SelectArgs};
use crate::rope_neox::PartialNeoxArgs;
use crate::tensor::DeviceTensor;
use cuda_core::{CudaContext, CudaEvent, CudaStream, DeviceBuffer};
use runtime::hc_gated::Geometry;
use runtime::sched::{self, At, LayerProgram, Overlap, PortKind};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// What the walk's errors name.
const WHAT: &str = "qwen4exp ubatch walk";

/// The selected flash's pass: the tensor-core one.
const MMA: bool = true;

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
    /// The slots' places, ten a run token: the places launch over the same
    /// ids with the slot map's layer row.
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
    /// values, slot-major.
    down: DeviceBuffer<f32>,
    /// The card slots' weighted sums, a row a token of the whole unit.
    acc: DeviceBuffer<f32>,
    /// One route table per distinct card count, in `counts`' order.
    routes: Vec<GemmRoute>,
    counts: Vec<usize>,
    /// The identity map over the experts (`e -> e`): the ids compression's
    /// stand-in for the places rule's map.
    ids_map: DeviceBuffer<u32>,
}

impl CardRoute38 {
    /// The route's buffers for ubatches of up to `rows` tokens over `card`'s
    /// card experts: a run of `min(rows, ROUTE_ROWS)` tokens, the sums a
    /// token of the whole unit, and one route table per distinct card count
    /// (refused by name past two, naming them). Load-time only.
    pub(super) fn new(
        stream: &CudaStream,
        rows: usize,
        card: &Card38,
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
        Ok(CardRoute38 {
            run,
            ids: DeviceBuffer::zeroed(stream, slots)?,
            sel: DeviceBuffer::zeroed(stream, slots)?,
            x: GemmAct::new(stream, run, geo::HIDDEN)?,
            g: DeviceBuffer::zeroed(stream, slots * geo::FF)?,
            u: DeviceBuffer::zeroed(stream, slots * geo::FF)?,
            act: GemmAct32::new(stream, slots, geo::FF)?,
            down: DeviceBuffer::zeroed(stream, slots * geo::HIDDEN)?,
            acc: DeviceBuffer::zeroed(stream, rows * geo::HIDDEN)?,
            routes: counts
                .iter()
                .map(|&n| GemmRoute::new(stream, slots, n))
                .collect::<Result<_, _>>()?,
            counts,
            ids_map: DeviceBuffer::from_host(
                stream,
                &(0..geo::EXPERTS as u32).collect::<Vec<_>>(),
            )?,
        })
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
            + self.routes.iter().map(GemmRoute::bytes).sum::<usize>()
            + self.ids_map.num_bytes()
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
    /// places from the walk's ids (`ids`, eleven slots a token, the shared
    /// expert's last — never read), the remapped route table over the slot
    /// map's layer row (`slots`), the q8_1 of the run's normed rows
    /// (`ffn_x`, `[m][HIDDEN]`), the Q4_K gate and up GEMMs over the unit's
    /// ten slots a token, the card slots' SwiGLU, the Q5_1 down GEMM and
    /// the card sum by the router's weights (`weights`, eleven a token)
    /// into the unit-wide acc's rows for the run. Nine launches a run.
    /// Refused by name on a layer without card experts. Asynchronous,
    /// allocation-free, capturable.
    #[allow(
        clippy::too_many_arguments,
        reason = "the route's ids, weights, normed rows, the slot map, the layer and the unit's width (rust-quality R8)"
    )]
    pub(super) fn enqueue(
        &mut self,
        c: &Ctx38<'_>,
        card: &Card38,
        l: usize,
        (ffn_x, ids, weights): (&DeviceBuffer<f32>, &DeviceBuffer<u32>, &DeviceBuffer<f32>),
        slots: &DeviceTensor<u32>,
        m: usize,
    ) -> Result<(), GpuError> {
        let st = card.stacks(c.w, l)?;
        let (n_card, gate, up, down) = (st.n_card, st.gate, st.up, st.down);
        let (gpu, stream, sink) = (c.gpu, c.gpu.stream(), c.gpu.layer_sink(l)?);
        let pitch = geo::N_USED + 1;
        // SAFETY: layer l's row of the slot map's card copy (`EXPERTS` words
        // a row), which stays resident while the window lives (this layer's
        // route launches).
        let map = unsafe { param_view::<u32>(slots.buf(), l * geo::EXPERTS, geo::EXPERTS) };
        let mut c0 = 0;
        while c0 < m {
            let n = self.run.min(m - c0);
            // SAFETY: tokens c0 .. c0 + n <= m <= rows of the walk's ids and
            // weights (`rows · pitch` each) and the arena's `ffn_x`
            // (`rows · HIDDEN`); every buffer stays in place for this run's
            // launches.
            let (ids_w, w_w, x_w, mut acc_w) = unsafe {
                (
                    param_view::<u32>(ids, c0 * pitch, n * pitch),
                    f32_view(weights, c0 * pitch, n * pitch),
                    f32_view(ffn_x, c0 * geo::HIDDEN, n * geo::HIDDEN),
                    f32_view(&self.acc, c0 * geo::HIDDEN, n * geo::HIDDEN),
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
            places(&map, &mut self.sel)?;
            let at = self.table_at(n_card)?;
            c.k.g32.enqueue_route_remap(
                stream,
                &self.ids,
                &map,
                n * geo::N_USED,
                &mut self.routes[at],
                sink,
            )?;
            gpu.enqueue_quantize_gemm(&x_w, n, &mut self.x, sink)?;
            for (w, y) in [(gate, &mut self.g), (up, &mut self.u)] {
                c.k.gemm.enqueue_gemm(
                    stream,
                    GemmArgs {
                        ty: GemmWeight::Q4K,
                        w,
                        rows_per_expert: geo::FF,
                        act: &self.x,
                        route: &self.routes[at],
                        input: GemmInput::Shared { top_k: geo::N_USED },
                        y,
                    },
                )?;
            }
            c.k.g32.enqueue_swiglu_quant32_sel(
                stream,
                &self.g,
                &self.u,
                &self.sel,
                n_card,
                n * geo::N_USED,
                &mut self.act,
                sink,
            )?;
            c.k.g32.enqueue_gemm32(
                stream,
                Gemm32Args {
                    w: Gemm32Weight::Q5_1File(down),
                    rows_per_expert: geo::HIDDEN,
                    act: &self.act,
                    route: &self.routes[at],
                    input: GemmInput::PerSlot,
                    y: &mut self.down,
                },
            )?;
            c.k.q38.enqueue_card_acc(
                stream,
                CardAccArgs {
                    down: &self.down,
                    w: &w_w,
                    sel: &self.sel,
                    n: geo::HIDDEN,
                    m: n,
                    n_card,
                    fault: sink,
                    acc: &mut acc_w,
                },
            )?;
            c0 += n;
        }
        Ok(())
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
    /// card experts. Load-time only.
    pub(super) fn new(stream: &CudaStream, rows: usize, card: &Card38) -> Result<Wide38, GpuError> {
        let geometry = Geometry::new(geo::STREAMS as u32, geo::RANK as u32, geo::HIDDEN as u32)
            .map_err(|e| GpuError::shape(WHAT, e.to_string()))?;
        let slots = geo::N_USED + 1;
        let card = (card.card_layers() > 0)
            .then(|| CardRoute38::new(stream, rows, card))
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
    pub(super) m: usize,
    pub(super) pos0: usize,
    pub(super) dense: usize,
    pub(super) cur: usize,
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
    /// run's tokens, the card sums into the unit-wide acc. Called from the
    /// walk's shadow only — after the front's download, the stream order the
    /// route's buffers exist under.
    fn route_card(&mut self, l: usize) -> Result<(), GpuError> {
        if !self.card.has(l) {
            return Ok(());
        }
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
            m,
        )
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
            cache_k: &mut kv.k,
            cache_v: &mut kv.v,
        },
    )?;
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
                kc: &kv.k,
                vc: &kv.v,
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
        c.k.flash.enqueue_pass_256_p4_sel(
            stream,
            GqaSelArgs {
                q: &qw,
                kc: &kv.k,
                vc: &kv.v,
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
            MMA,
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
        leg.begin_walk();
        sched::walk(o, layers, leg, self)?;
        leg.end_walk()?;
        if let Some(t) = self.p.x.taps.as_mut() {
            t.walked = Some((self.p.pos0, m));
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
    /// the download of the unit's activations and routed slots.
    fn front(&mut self, port: &mut BatchLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let l = at.layer;
        let enq = port.part_start();
        port.mark(l, Mark::Front as usize)?;
        self.p.front(l)?;
        self.p.ffn_mix(l)?;
        self.p.route(l)?;
        port.mark(l, Mark::FrontEnd as usize)?;
        let key = port.key(at);
        let stream = self.p.c.gpu.stream();
        let r = port.hybrid().enqueue_download_pitched(
            stream,
            [&self.p.s.ffn_x, &self.p.x.weights],
            &self.p.x.ids,
            geo::N_USED + 1,
            key,
        );
        let marked = r.and_then(|()| port.mark(l, Mark::Down as usize));
        port.part_end(l, enq);
        marked
    }

    /// The card route on a layer with card experts, then the shared expert
    /// over the arena's `ffn_x`. The route is enqueued here and nowhere
    /// else — after the front's download, the stream order the route's
    /// buffers exist under (its module doc): enqueued before the download,
    /// the d2h would queue behind the route's card GEMMs and the host tier
    /// would start late by their whole time.
    fn shadow(&mut self, port: &mut BatchLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let enq = port.part_start();
        let r = self
            .p
            .route_card(at.layer)
            .and_then(|()| self.p.shared(at.layer))
            .and_then(|()| port.mark(at.layer, Mark::Shadow as usize));
        port.part_end(at.layer, enq);
        r
    }

    /// The gated sum over the host sums the walk's serve uploaded, the card
    /// route's beside them on a layer with card experts.
    fn back(&mut self, port: &mut BatchLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let enq = port.part_start();
        let r = self
            .p
            .shared_add(at.layer, port.hsum())
            .and_then(|()| port.mark(at.layer, Mark::Back as usize));
        port.part_end(at.layer, enq);
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

// ------------------------------------------------------------------ timing

/// What the walk's timing refuses names.
const WHAT_TIMING: &str = "qwen4exp ubatch walk timing";

/// A card mark of a layer-batch's stream, in the order the walk enqueues
/// them: the front's first launch, the front's last launch before the route's
/// downloads, the downloads' end (the card route, on a card layer, and the
/// shared expert's launches follow), the shared expert's last launch, the
/// sums' upload, the gated sum.
#[derive(Clone, Copy)]
enum Mark {
    Front = 0,
    FrontEnd = 1,
    Down = 2,
    Shadow = 3,
    Upload = 4,
    Back = 5,
}

/// The marks one layer-batch's card time reads, one a part boundary.
const MARKS: usize = 6;

/// `d` in whole nanoseconds, saturating.
pub(super) fn nanos(d: Duration) -> u64 {
    u64::try_from(d.as_nanos()).unwrap_or(u64::MAX)
}

/// One layer-batch of a ubatch walk's timing: the serve's host time. The card
/// times and the parts' enqueue walls are read once the walk's last mark is
/// waited for.
struct WideLb {
    b: u64,
    layer: usize,
    cols: usize,
    slots: u64,
    wait_ns: u64,
    union_ns: u64,
    serve_ns: u64,
}

/// One layer-batch of a Qwen3.8 ubatch walk under
/// [`GpuModel::set_prompt38_stats`]: its columns and listed host slots, its
/// serve's host time (the wait on the route's copies; the union call's wall,
/// which carries the routing scan and the plan build in front of it; the
/// serve's whole wall, the upload's enqueue in it), the host wall its parts
/// took to enqueue, and its card time by part (the front's launches, the
/// route's downloads, the shared expert under the union, the sums' upload,
/// the gated sum).
///
/// Reached through [`GpuModel::take_prompt38_stats`]; the fields are public
/// for the record a caller renders.
pub struct Prompt38Lb {
    /// The prompt's ubatch the walk ran, from 0.
    pub b: u64,
    pub layer: usize,
    /// The unit's columns.
    pub cols: usize,
    /// Host slots the union listed.
    pub slots: u64,
    pub wait_ns: u64,
    pub union_ns: u64,
    pub serve_ns: u64,
    /// The layer's front, shared expert and gated sum as the host enqueued
    /// them.
    pub enqueue_ns: u64,
    pub front_ms: f64,
    pub down_ms: f64,
    pub shadow_ms: f64,
    pub upload_ms: f64,
    pub back_ms: f64,
}

/// A Qwen3.8 prompt's ubatch walks under [`GpuModel::set_prompt38_stats`]:
/// the walks and layer-batches that ran, the host prologue every walk's plan
/// took (the PLE rows, the record, their copy) and the walks' whole wall, and
/// one row a layer-batch.
///
/// Reached through [`GpuModel::take_prompt38_stats`]; the fields are public
/// for the record a caller renders.
pub struct Prompt38Stats {
    /// Ubatches walked.
    pub ubatches: u64,
    pub prologue_ns: u64,
    pub walk_ns: u64,
    pub rows: Vec<Prompt38Lb>,
}

/// A Qwen3.8 ubatch walk's timing, allocated once a caller arms it
/// ([`GpuModel::set_prompt38_stats`]): one event a mark a layer, the walk in
/// progress's rows, the walks that completed, and the prompt's host walls.
/// Unarmed, nothing records and nothing is read.
pub(super) struct WideTiming {
    marks: Vec<CudaEvent>,
    rows: Vec<WideLb>,
    done: Vec<Prompt38Lb>,
    /// The host wall the layers' parts took to enqueue, a layer each.
    enqueue_ns: Vec<u64>,
    walks: u64,
    prologue_ns: u64,
    walk_ns: u64,
    /// The walk in progress's wall, from its first enqueue.
    walk_t0: Option<Instant>,
}

impl WideTiming {
    /// Timing for `layers` layers over one stream's context. Load-time only.
    pub(super) fn new(ctx: &Arc<CudaContext>, layers: usize) -> Result<WideTiming, GpuError> {
        // Timed events: `new_event(None)` makes a sync-only event, whose
        // `elapsed_ms` the driver refuses.
        let marks = (0..layers * MARKS)
            .map(|_| ctx.new_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT)))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(WideTiming {
            marks,
            rows: Vec::new(),
            done: Vec::new(),
            enqueue_ns: vec![0; layers],
            walks: 0,
            prologue_ns: 0,
            walk_ns: 0,
            walk_t0: None,
        })
    }

    /// A walk's plan's host wall added to the prompt's prologue.
    pub(super) fn add_prologue(&mut self, ns: u64) {
        self.prologue_ns += ns;
    }

    /// The prompt's stats so far, taken — every wall zeroed and the rows
    /// moved — or `None` before a walk completed.
    pub(super) fn take(&mut self) -> Option<Prompt38Stats> {
        if self.walks == 0 {
            return None;
        }
        let stats = Prompt38Stats {
            ubatches: self.walks,
            prologue_ns: std::mem::take(&mut self.prologue_ns),
            walk_ns: std::mem::take(&mut self.walk_ns),
            rows: std::mem::take(&mut self.done),
        };
        self.walks = 0;
        Some(stats)
    }

    /// The walk over: its card marks read into its rows — the last mark
    /// waited for first, so the reads see every event — and its rows filed
    /// under its index, its wall added to the prompt's.
    fn close(&mut self) -> Result<(), GpuError> {
        let wall = self.walk_t0.take().map(|t| nanos(t.elapsed())).unwrap_or(0);
        if let Some(last) = self.marks.last() {
            last.synchronize()?;
        }
        let rows = std::mem::take(&mut self.rows);
        for row in rows {
            let Some(m) = self
                .marks
                .get(row.layer * MARKS..)
                .and_then(|m| m.get(..MARKS))
            else {
                return Err(GpuError::state(
                    WHAT_TIMING,
                    "a card mark for every layer of the walk",
                ));
            };
            let enqueue = *self.enqueue_ns.get(row.layer).unwrap_or(&0);
            let span = |a: Mark, b: Mark| -> Result<f64, GpuError> {
                Ok(f64::from(m[a as usize].elapsed_ms(&m[b as usize])?))
            };
            self.done.push(Prompt38Lb {
                b: row.b,
                layer: row.layer,
                cols: row.cols,
                slots: row.slots,
                wait_ns: row.wait_ns,
                union_ns: row.union_ns,
                serve_ns: row.serve_ns,
                enqueue_ns: enqueue,
                front_ms: span(Mark::Front, Mark::FrontEnd)?,
                down_ms: span(Mark::FrontEnd, Mark::Down)?,
                shadow_ms: span(Mark::Down, Mark::Shadow)?,
                upload_ms: span(Mark::Shadow, Mark::Upload)?,
                back_ms: span(Mark::Upload, Mark::Back)?,
            });
        }
        self.enqueue_ns.fill(0);
        self.walks += 1;
        self.walk_ns += wall;
        Ok(())
    }
}

/// Mark `site` of `layer` on the stream, in [`Mark`]'s layout.
fn record_mark(
    marks: &[CudaEvent],
    stream: &CudaStream,
    layer: usize,
    site: usize,
) -> Result<(), GpuError> {
    Ok(marks[layer * MARKS + site].record(stream)?)
}

/// The ubatch walk's timing behind the batch leg's [`LegTimer`]: the serve's
/// host times and walls as its rows, the parts' card marks and enqueue walls,
/// the walk's own wall.
impl LegTimer for WideTiming {
    /// The sums' upload ([`Mark::Upload`]).
    fn upload_mark(&mut self, stream: &CudaStream, layer: usize) -> Result<(), GpuError> {
        record_mark(&self.marks, stream, layer, Mark::Upload as usize)
    }

    /// The serve's row, in serve order for [`WideTiming::close`].
    fn served(&mut self, note: ServeNote) {
        self.rows.push(WideLb {
            b: self.walks,
            layer: note.layer,
            cols: note.cols,
            slots: note.slots,
            wait_ns: note.times.wait_ns,
            union_ns: note.union_ns,
            serve_ns: note.serve_ns,
        });
    }

    /// A part boundary's mark, in [`Mark`]'s layout.
    fn mark(&mut self, stream: &CudaStream, layer: usize, site: usize) -> Result<(), GpuError> {
        record_mark(&self.marks, stream, layer, site)
    }

    /// A part of layer `layer`'s host enqueue wall.
    fn note_part(&mut self, layer: usize, ns: u64) {
        if let Some(at) = self.enqueue_ns.get_mut(layer) {
            *at += ns;
        }
    }

    /// A walk begins: its rows taken back, its wall started. A walk that
    /// fails leaves no rows: the next one clears them.
    fn begin_walk(&mut self) {
        self.rows.clear();
        self.walk_t0 = Some(Instant::now());
    }

    /// A walk ends: its card marks read and its rows filed. Blocking — the
    /// stream's tail, the last upload and gated sum, is waited for.
    fn end_walk(&mut self) -> Result<(), GpuError> {
        self.close()
    }
}
