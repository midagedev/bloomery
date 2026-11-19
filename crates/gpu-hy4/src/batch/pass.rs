//! The placed prompt pass: consecutive prompt positions of a placed load (some of the routed
//! layers' experts on the host tier) walked layer by layer over the host tier's batch port in units
//! of up to [`PASS_ROWS`] rows, so each hosted layer's host experts are read once for every row of
//! a unit ([`BatchLeg`]: one download, one union serve, one upload a unit) and not once a row, as
//! the decode step does. A walk is the common `sched::walk` of `(units, rows, Batch)` in the
//! fronts order ([`GROUP_WALK_ORDER`]): every unit's front ahead of the serves, one exchange set a
//! unit; what it leaves in the stores is what the prompt batch ([`super::prompt`]) leaves, and
//! each row is the placed step's feed-forward block up to rounding (the experts' sum is
//! `card + host`, the card's slots one grouped GEMM over the unit's rows).
//!
//! A layer the card holds whole runs the whole-card block of [`super::ffn`] over the rows; a
//! hosted routed layer runs it in three parts around the host's serve, in the host leg's order:
//!
//! - **front**: the identity mix's fold and the feed-forward norm, the router over the rows, the
//!   latent down into the rows the host reads and the download of the latent rows, weights and
//!   ids ([`PlacedSide::download_rows`]);
//! - **shadow**, under the host's work: the shared expert; under host streaming the layer's pick
//!   (the common owner's, [`PromptStream::unit_pick`](bloomery_gpu::host::prompt_stream::PromptStream::unit_pick): the first unit picks for the walk, the later
//!   units read its pick); the card's places of every slot ([`PlacedSide::places_rows`]) and,
//!   behind the pick's landings, the card's slots as one grouped GEMM over the layer's card
//!   stacks, whose table is the route over the card's slot ids and leaves the host's slots out
//!   ([`PlacedSide::remap_row`], `enqueue_route_remap`); the streamed experts' slots as the
//!   common ring pass over the stream's ring ([`RingPass::enqueue`]); all summed in slot order
//!   over the slots the card holds, by the stream's union places when the ring ran (the stage's
//!   below `n_card`, `n_card + i` for the `i`-th streamed expert, so a streamed expert's rows join
//!   the sum where a host expert's would);
//! - **back**: the stream's join taken ([`PromptStream::unit_join`](bloomery_gpu::host::prompt_stream::PromptStream::unit_join)), the card's and the host's
//!   sums added (the routed sum is linear in the experts' rows), the latent up, the shared expert
//!   added, the sandwich norm and the identity mix's post.
//!
//! The pass's buffers are the prompt batch's at a walk of `units` units of `rows` rows (the
//! streams a walk wide, the rest a unit wide), each later unit's rows from its front to its back
//! ([`UnitRows`], lent into the batch's buffers for the unit's parts), the host's sums, the layer's
//! map, the slots' places and the card slots' route tables ([`PlacedPass`]); the plan reserves
//! their bytes ([`pass_bytes`]) and the load holds them to it. The stream's ring is the plan's ring
//! term ([`model::arch::hy4::place::stream_ring`]), made at the stream's start.

use bloomery_gpu::card_leg::LegStacks;
use bloomery_gpu::card_slots::CardAcc;
use bloomery_gpu::gemm::{GEMM_MAX_SLOTS, GemmRoute, RouteSet};
use bloomery_gpu::hc_ident::{HC, HeadArgs, MIX};
use bloomery_gpu::head::Head;
use bloomery_gpu::host::census::TierCensus;
use bloomery_gpu::host::costs::ExpertShape;
use bloomery_gpu::host::placed::{PlacedHost, PlacedSide};
use bloomery_gpu::host::prompt_stream::ADMIT_FLOOR;
use bloomery_gpu::host::run::HostRun;
use bloomery_gpu::host::served::prompt_pass;
use bloomery_gpu::host::stream_side::RingShape;
use bloomery_gpu::host::stream_side::{RingPass, RingRows, StreamSide};
use bloomery_gpu::host::swap::{CallPick, CallReport};
use bloomery_gpu::host::swap_source::DEADLINE;
use bloomery_gpu::host::xstream::{
    Costs, GROUP_WALK_ORDER, XCfg, XLayer, XMode, XReport, XStreamBody,
};
use bloomery_gpu::host::{BatchLeg, HybridStats};
use bloomery_gpu::kquant::Act;
use bloomery_gpu::latent::{INDEX_HEAD, POOL};
use bloomery_gpu::media::{CallRows, MediaBody, MediaSpan, check_spans};
use bloomery_gpu::routed_block::{RouteIn, RoutedRows, RouterEntry};
use bloomery_gpu::site::WideForms;
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{Gpu, GpuError, GpuModel};
use cuda_core::{CudaStream, DeviceBuffer};
use model::arch::hy4::hparams::Hparams;
use model::arch::hy4::names::{self, Sub};
use model::arch::place_hybrid::RingForms;
use runtime::sched::{self, At, BatchOrder, LayerProgram, Overlap, PortKind};

use super::{
    Buf, Cfg, Env, ExpertIo, FFN_NORM, FFN_OUT, L_OUT, N_EXPERT, N_USED, Prompt, Q_MAX, Span, WHAT,
    WalkTaps, WideP, attention, chunks, engram, expert_rows, ffn, io, post, pre_norm, project,
    shape, stage, tap, tap_words,
};
use crate::body::{ENGRAM_W, Ffn, LATENT, ROPE, ROW, Shape, Store};

/// The most rows one pass holds: the host union's widest unit.
pub const PASS_ROWS: usize = model::moe::UNION_MAX_COLS;

/// What this file's refusals name.
const PASS: &str = "hy4 placed pass";

/// A pass shape: the rows of a unit, the rows of its attention chunk (the engram and attention
/// sub-layers run `q_rows` rows a launch) and the units a walk holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PassShape {
    pub rows: usize,
    pub q_rows: usize,
    pub units: usize,
}

impl PassShape {
    /// The shape of a pass of one unit of `rows` rows.
    #[must_use]
    pub const fn of(rows: usize) -> PassShape {
        PassShape {
            rows,
            q_rows: if rows < Q_MAX { rows } else { Q_MAX },
            units: 1,
        }
    }

    /// The shape of a walk of `units` units of `rows` rows each.
    #[must_use]
    pub const fn walk(rows: usize, units: usize) -> PassShape {
        PassShape {
            units,
            ..PassShape::of(rows)
        }
    }

    /// The rows a walk of the shape holds: its units' rows.
    #[must_use]
    pub const fn walk_rows(&self) -> usize {
        self.rows * self.units
    }
}

/// The units of the widest walk: the host union's widest group of columns over its widest unit
/// ([`model::moe::UNION_TAIL_MAX_COLS`], [`PASS_ROWS`]), the walk a layer's host experts stream
/// over when the stream's rule sends them to the card.
pub const WALK_UNITS: usize = model::moe::UNION_TAIL_MAX_COLS / PASS_ROWS;

/// The shapes of the full unit's rows: the widest walk's units, [`WALK_UNITS`], halved down to one.
const WALK_STEPS: usize = (usize::BITS - WALK_UNITS.leading_zeros()) as usize;

/// The one-unit shapes of fewer rows: the full unit's rows halved down to the narrowest the card's
/// room may be left with.
const UNIT_STEPS: usize = 3;

/// The shapes a plan may hold, widest first: the widest walk of the host union's widest unit, the
/// walk's units halved down to one unit, then one unit of fewer rows. A narrower pass holds fewer
/// buffers beside the plan and reads each hosted layer's experts once for fewer rows.
pub const SHAPES: [PassShape; WALK_STEPS + UNIT_STEPS] = {
    let mut shapes = [PassShape::of(PASS_ROWS); WALK_STEPS + UNIT_STEPS];
    let mut i = 0;
    while i < WALK_STEPS {
        shapes[i] = PassShape::walk(PASS_ROWS, WALK_UNITS >> i);
        i += 1;
    }
    let mut i = 1;
    while i <= UNIT_STEPS {
        shapes[WALK_STEPS - 1 + i] = PassShape::of(PASS_ROWS >> i);
        i += 1;
    }
    shapes
};

// A walk's units start on an attention chunk, so the stage's chunk bases index them.
const _: () = assert!(PASS_ROWS.is_multiple_of(Q_MAX));
const _: () = assert!(WALK_UNITS >= 1);

// The list runs from the widest walk to one unit of the full rows to the narrowest unit, each
// shape holding fewer rows than the last, so a pick that steps down it holds fewer buffers at
// every step.
const _: () = {
    assert!(SHAPES[0].units == WALK_UNITS && SHAPES[0].rows == PASS_ROWS);
    assert!(SHAPES[WALK_STEPS - 1].units == 1 && SHAPES[WALK_STEPS - 1].rows == PASS_ROWS);
    assert!(SHAPES[SHAPES.len() - 1].rows == PASS_ROWS >> UNIT_STEPS);
    let mut i = 1;
    while i < SHAPES.len() {
        assert!(SHAPES[i].walk_rows() < SHAPES[i - 1].walk_rows());
        i += 1;
    }
};

/// The prompt batch's geometry for walks of `shape`: the streams a walk wide, the rest a unit
/// wide. Refused by name: a unit of no rows or more than [`PASS_ROWS`], no units, units that do
/// not start on an attention chunk, and a walk past the prompt batch's rows.
fn pass_cfg(shape: PassShape) -> Result<Cfg, GpuError> {
    let PassShape {
        rows,
        q_rows,
        units,
    } = shape;
    if !(1..=PASS_ROWS).contains(&rows) {
        return Err(GpuError::shape(
            PASS,
            format!("a pass of {rows} rows a unit: 1 to {PASS_ROWS} rows"),
        ));
    }
    if units == 0 || (units > 1 && !rows.is_multiple_of(q_rows)) {
        return Err(GpuError::shape(
            PASS,
            format!(
                "a walk of {units} units of {rows} rows in attention chunks of {q_rows}: one \
                 unit or more, each starting on a chunk"
            ),
        ));
    }
    if shape.walk_rows() > Cfg::PROMPT.cap {
        return Err(GpuError::shape(
            PASS,
            format!(
                "a walk of {} rows: at most the prompt batch's {}",
                shape.walk_rows(),
                Cfg::PROMPT.cap
            ),
        ));
    }
    Ok(Cfg {
        cap: shape.walk_rows(),
        q_rows,
        ffn_rows: rows,
    })
}

// The unit's slots fit one grouped GEMM.
const _: () = assert!(PASS_ROWS * N_USED <= GEMM_MAX_SLOTS);

/// The device bytes a placed pass of `shape` holds on a load of `ctx` positions, never below
/// what [`PlacedPass::new`] makes: the plan's prompt reserve ([`PlanInputs::with_prompt_reserve`]),
/// which the load holds the made bytes to. Every buffer of the prompt batch at the pass's
/// geometry, counted by the widths it is allocated at, and the later units' rows a walk holds
/// ([`UnitRows`]); the quantized forms the stacks and sites read are counted in both of the forms
/// the kernels have (the file's types pick one, which the plan cannot read), so the bound sits a
/// few megabytes over the made bytes at 512 rows a unit.
///
/// # Errors
/// [`GpuError`] by name for a shape [`pass_cfg`] refuses, and a shape the body is not built for.
///
/// [`PlanInputs::with_prompt_reserve`]: model::arch::place_hybrid::PlanInputs::with_prompt_reserve
pub fn pass_bytes(hp: &Hparams, ctx: usize, shape: PassShape) -> Result<u64, GpuError> {
    let rows = shape.rows;
    let cfg = pass_cfg(shape)?;
    let sh = Shape::of(hp, ctx)?;
    let (n, q) = (sh.hidden, cfg.q_rows);
    let (cap, r) = (cfg.cap, q.max(cfg.ffn_rows));
    let slots = cfg.ffn_rows * N_USED;
    let heads = sh.max_heads;
    let pools = ctx.div_ceil(POOL);
    let (sites, cols) = hp
        .engram
        .as_ref()
        .map_or((0, 1), |e| (e.layers.len(), e.n_cols()));
    let ff = hp.dense_ff.max(hp.experts.shared_width);
    let (wv, wms) = bloomery_gpu::flash::window_partials_lens(q * heads, sh.window);
    let part_v = bloomery_gpu::flash::partials_v_len(q * heads, sh.top_k).max(wv);
    let part_ms = bloomery_gpu::flash::partials_ms_len(q * heads, sh.top_k).max(wms);
    let f32s = 2 * cap * HC * n
        + 2 * cap * ROPE
        + sites * cap * cols * (ENGRAM_W + 1)
        + r * MIX
        + 2 * r * n
        + 2 * q * sh.q_lora
        + 2 * q * heads * sh.head_k
        + q * heads * ROW
        + 3 * q * ROW
        + q * sh.idx_heads * INDEX_HEAD
        + q * INDEX_HEAD
        + q * sh.idx_heads
        + q * sh.idx_heads * INDEX_HEAD
        + q * sh.idx_heads
        + part_v
        + part_ms
        + q * heads * LATENT
        + 3 * q * heads * sh.head_v
        + 2 * r * n
        + q * pools
        + 3 * r * ff
        + 2 * r * sh.latent
        + 3 * r * n
        + cfg.ffn_rows * N_EXPERT
        + slots
        + 2 * slots * sh.expert_ff
        + slots * sh.latent
        + q * cols * ENGRAM_W
        + q * (HC + 1) * n
        + q * HC * n
        + q * HC
        + HC * n;
    let u32s = 2 * cap + cap.div_ceil(q) + q + q * sh.top_k + q + slots;
    let u16s = q * ROW;
    // The rope tail's gather tables at each head count a layer has.
    let mut counts: Vec<usize> = (0..hp.n_layer).map(|l| hp.heads(l)).collect();
    counts.sort_unstable();
    counts.dedup();
    let gather: usize = counts.iter().map(|hd| 2 * q * hd * ROPE).sum();
    // The wide forms at each input width a site reads, in both forms.
    let mut widths = vec![
        n,
        sh.q_lora,
        sh.latent,
        hp.dense_ff,
        hp.experts.shared_width,
    ];
    widths.extend(counts.iter().map(|hd| hd * sh.head_v));
    if let Some(e) = &hp.engram {
        widths.push(e.n_cols() * e.key_length);
    }
    widths.sort_unstable();
    widths.dedup();
    let both = WideForms::q8(true, true);
    let wide: usize = widths.iter().map(|&k| both.bytes(r, k)).sum();
    let experts = both.bytes(cfg.ffn_rows, sh.latent) + both.bytes(slots, sh.expert_ff);
    // A hosted layer's card slots run over a table of its card count, one table a distinct count
    // ([`PlacedPass`]); a plan does not know the placement, so one a routed layer, each no wider
    // than the whole stack's.
    let routed = (0..hp.n_layer).filter(|&l| !hp.dense(l)).count();
    let routes = GemmRoute::bytes_at(r, 1) + (1 + routed) * GemmRoute::bytes_at(slots, N_EXPERT);
    // The host's sums, the layer's map and the slots' places.
    let beside = rows * sh.latent * 4 + N_EXPERT * 4 + rows * N_USED * 4;
    // Each later unit's rows from its front to its back: the norm's output and the mix, the
    // latent rows, the router's weights and ids.
    let later = (shape.units - 1) * 4 * (r * n + r * MIX + r * sh.latent + 2 * slots);
    let total =
        4 * f32s + 4 * u32s + 2 * u16s + 4 * gather + wide + experts + routes + beside + later;
    u64::try_from(total).map_err(|_| GpuError::shape(PASS, "a pass past u64 bytes"))
}

/// The placed pass's buffers: the prompt batch at the pass's geometry, each later unit's rows, the
/// host's sums, the layer's map of its experts' places, the places of a unit's routed slots and the
/// route tables of the hosted layers' card slots. Held for the load's life.
pub struct PlacedPass {
    prompt: Prompt,
    shape: PassShape,
    /// The rows units `1 ..` of a walk hold from a hosted layer's front to its back, lent into the
    /// prompt batch's buffers for each of the unit's parts; unit 0 holds the batch's own.
    later: Vec<UnitRows>,
    /// The form the stream's ring pass reads the hosted layers' card stacks in; `None` when they
    /// hold no card experts or are not of one form a route GEMM entry reads.
    forms: Option<RingForms>,
    /// The host's sums of the unit's rows, in the latent's width, which the batch leg lands them in.
    hsum: DeviceBuffer<f32>,
    /// A hosted layer's row of the slot map: each expert's place in the card's stacks, or `HOST`.
    map: DeviceBuffer<u32>,
    /// The places of the unit's routed slots, from the same map.
    sel: DeviceBuffer<u32>,
    /// The route tables of the hosted layers' card slots, one a distinct card count: a grouped GEMM's
    /// table holds the experts of the one stack it routes over, and a hosted layer's stacks hold
    /// only its card's experts.
    routes: RouteSet,
    bytes: usize,
}

/// A unit's rows a hosted layer's front leaves for its shadow and back while the walk's other units
/// run theirs: the feed-forward norm's output and the identity mix (the shared expert's input and
/// the post's), the latent rows the host and the card's slots read, and the router's weights and
/// ids. Each is a buffer of the prompt batch's own length, swapped in for the unit's parts.
struct UnitRows {
    xn: DeviceBuffer<f32>,
    mix: DeviceBuffer<f32>,
    lat: DeviceBuffer<f32>,
    weights: DeviceBuffer<f32>,
    ids: DeviceBuffer<u32>,
}

impl UnitRows {
    /// Buffers of `buf`'s lengths. Load-time only.
    fn like(stream: &CudaStream, buf: &Buf) -> Result<UnitRows, GpuError> {
        Ok(UnitRows {
            xn: DeviceBuffer::zeroed(stream, buf.xn.len())?,
            mix: DeviceBuffer::zeroed(stream, buf.mix.len())?,
            lat: DeviceBuffer::zeroed(stream, buf.lat.len())?,
            weights: DeviceBuffer::zeroed(stream, buf.weights.len())?,
            ids: DeviceBuffer::zeroed(stream, buf.ids.len())?,
        })
    }

    /// Swap these rows with `buf`'s: before a unit's part its own, after it the batch's back.
    fn swap(&mut self, buf: &mut Buf) {
        std::mem::swap(&mut self.xn, &mut buf.xn);
        std::mem::swap(&mut self.mix, &mut buf.mix);
        std::mem::swap(&mut self.lat, &mut buf.lat);
        std::mem::swap(&mut self.weights, &mut buf.weights);
        std::mem::swap(&mut self.ids, &mut buf.ids);
    }

    fn bytes(&self) -> usize {
        self.xn.num_bytes()
            + self.mix.num_bytes()
            + self.lat.num_bytes()
            + self.weights.num_bytes()
            + self.ids.num_bytes()
    }
}

/// The one form of the card stacks of `side`'s hosted layers that hold card experts, as the ring
/// pass reads them ([`RingForms::of`]); `None` when no layer holds card experts or two differ.
/// Refused by name: a hosted layer's stacks missing from `w`.
fn ring_forms(
    w: &Weights,
    body: &crate::body::Body,
    side: &PlacedSide,
) -> Result<Option<RingForms>, GpuError> {
    let mut seen: Option<Option<RingForms>> = None;
    for l in side.hosted_set() {
        let Ffn::Routed { stacks, .. } = &body.layers[l].ffn else {
            continue;
        };
        if side.n_card(l) == 0 {
            continue;
        }
        let st = LegStacks::of(w, [&stacks[0], &stacks[1], &stacks[2]])?;
        let forms = RingForms::of(st.gate_up_ty(), st.down_ty());
        seen = Some(match seen {
            None => forms,
            Some(s) => s.filter(|_| s == forms),
        });
    }
    Ok(seen.flatten())
}

/// The card counts of `side`'s hosted layers that hold experts on the card.
fn hosted_counts(side: &PlacedSide) -> impl Iterator<Item = usize> + '_ {
    side.hosted_set()
        .into_iter()
        .map(|l| side.n_card(l))
        .filter(|&n| n > 0)
}

impl PlacedPass {
    /// The pass's buffers for walks of `shape` over `body`'s layers and `side`'s hosted layers.
    /// Load-time only.
    pub(crate) fn new(
        gpu: &Gpu,
        w: &Weights,
        body: &crate::body::Body,
        side: &PlacedSide,
        shape: PassShape,
    ) -> Result<PlacedPass, GpuError> {
        let rows = shape.rows;
        let cfg = pass_cfg(shape)?;
        let prompt = Prompt::new(gpu, w, body, cfg)?;
        let stream = gpu.stream();
        let later = (1..shape.units)
            .map(|_| UnitRows::like(stream, &prompt.buf))
            .collect::<Result<Vec<_>, _>>()?;
        let forms = ring_forms(w, body, side)?;
        let hsum = DeviceBuffer::zeroed(stream, rows * body.sh.latent)?;
        let map = DeviceBuffer::zeroed(stream, N_EXPERT)?;
        let sel = DeviceBuffer::zeroed(stream, rows * N_USED)?;
        let slots = cfg.ffn_rows * N_USED;
        let routes = RouteSet::new(stream, slots, hosted_counts(side))?;
        let bytes = prompt.bytes()
            + later.iter().map(UnitRows::bytes).sum::<usize>()
            + hsum.num_bytes()
            + map.num_bytes()
            + sel.num_bytes()
            + routes.bytes();
        Ok(PlacedPass {
            prompt,
            shape,
            later,
            forms,
            hsum,
            map,
            sel,
            routes,
            bytes,
        })
    }

    /// The rows a pass holds: its walk's.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.prompt.cfg.cap
    }

    /// The pass's shape.
    #[must_use]
    pub fn shape(&self) -> PassShape {
        self.shape
    }

    /// Arm the pass's taps: every later walk copies every stage ([`crate::TAP_STAGES`]) of every
    /// layer for its rows ([`Prompt::arm_taps`]).
    pub fn arm_taps(&mut self, gpu: &Gpu) -> Result<(), GpuError> {
        self.prompt.arm_taps(gpu)
    }

    /// The last walk's taps for its first `rows` rows, read back ([`Prompt::taps`]).
    pub fn taps(&self, stream: &CudaStream, rows: usize) -> Result<WalkTaps, GpuError> {
        self.prompt.taps(stream, rows)
    }

    /// The device bytes the pass holds.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

/// The pass's walk `(units, rows, Batch)`: each unit's layer `l` attention (and engram site) over
/// its rows, then its feed-forward block whole, or as the front, shadow and back around the host's
/// serve, in the fronts order. A later unit's parts run over its own rows ([`UnitRows`]).
struct PassWalk<'a> {
    env: Env<'a>,
    stores: &'a mut [Store],
    p: &'a mut Prompt,
    /// Units `1 ..`'s rows, lent into the batch's buffers for their parts.
    later: &'a mut [UnitRows],
    side: &'a PlacedSide,
    map: &'a mut DeviceBuffer<u32>,
    sel: &'a mut DeviceBuffer<u32>,
    /// The hosted layers' card route tables ([`PlacedPass::routes`]).
    routes: &'a mut RouteSet,
    /// The prompt calls' streaming and its ring pass's buffers.
    xs: &'a mut StreamSide,
    /// Each unit's first row of the walk and its rows.
    units: &'a [(usize, usize)],
    /// Each unit's streams' buffer holding its layer input, as the prompt batch's walk keeps it.
    cur: Vec<usize>,
    /// The position past the walk's last row.
    walk_end: usize,
}

impl PassWalk<'_> {
    /// Unit `u`'s first row and rows, refused by name past the walk's units.
    fn unit(&self, u: usize) -> Result<(usize, usize), GpuError> {
        self.units
            .get(u)
            .copied()
            .ok_or_else(|| shape(format!("unit {u} of a walk of {} units", self.units.len())))
    }

    /// The span of unit `u`'s rows over its streams' buffer.
    fn span(&self, u: usize) -> Result<Span, GpuError> {
        let (r0, n) = self.unit(u)?;
        Ok(Span {
            cur: self.cur[u],
            r0,
            index: r0 / self.p.cfg.q_rows,
            n,
        })
    }

    /// `part` of unit `u` over its own rows: a later unit's lent into the batch's buffers for the
    /// part and taken back after it, whatever it returned.
    fn lent<T>(
        &mut self,
        u: usize,
        part: impl FnOnce(&mut Self) -> Result<T, GpuError>,
    ) -> Result<T, GpuError> {
        let Some(k) = u.checked_sub(1) else {
            return part(self);
        };
        if k >= self.later.len() {
            return Err(shape(format!(
                "unit {u} of a pass holding the rows of {} units",
                self.later.len() + 1
            )));
        }
        self.later[k].swap(&mut self.p.buf);
        let r = part(self);
        self.later[k].swap(&mut self.p.buf);
        r
    }

    /// Whether the stream's pick covers layer `l`: a hosted layer whose card holds experts.
    fn card(side: &PlacedSide, l: usize) -> bool {
        side.hosted(l) && side.n_card(l) > 0
    }

    /// Unit `u`'s layer `l` engram site and attention sub-layer over its rows, in the prompt
    /// batch's chunks.
    fn attention_half(&mut self, u: usize, l: usize) -> Result<(), GpuError> {
        let (r0, t) = self.unit(u)?;
        let q = self.p.cfg.q_rows;
        let base = r0 / q;
        if self.env.layers[l].engram.is_some() {
            for (i, rows) in chunks(t, q).enumerate() {
                let sp = Span {
                    cur: self.cur[u],
                    r0: r0 + rows.start,
                    index: base + i,
                    n: rows.len(),
                };
                engram(&self.env, self.p, l, sp)?;
            }
            self.cur[u] ^= 1;
        }
        for (i, rows) in chunks(t, q).enumerate() {
            let sp = Span {
                cur: self.cur[u],
                r0: r0 + rows.start,
                index: base + i,
                n: rows.len(),
            };
            attention(&self.env, self.stores, self.p, l, sp)?;
        }
        self.cur[u] ^= 1;
        Ok(())
    }

    /// A unit's front: a layer the card holds whole entire; a hosted one up to its download.
    fn unit_front(&mut self, port: &mut BatchLeg<'_, HostRun>, at: At) -> Result<(), GpuError> {
        let (l, u) = (at.layer, at.unit);
        self.attention_half(u, l)?;
        if self.side.hosted(l) {
            return self.hosted_front(port, at);
        }
        let (r0, t) = self.unit(u)?;
        let ffn_rows = self.p.cfg.ffn_rows;
        for (i, rows) in chunks(t, ffn_rows).enumerate() {
            let sp = Span {
                cur: self.cur[u],
                r0: r0 + rows.start,
                index: r0 / ffn_rows + i,
                n: rows.len(),
            };
            ffn(&self.env, self.p, l, sp)?;
        }
        self.cur[u] ^= 1;
        Ok(())
    }

    /// A hosted layer's front from the feed-forward norm: the router over the unit's rows, the
    /// latent down into the rows the host reads and the download.
    fn hosted_front(&mut self, port: &mut BatchLeg<'_, HostRun>, at: At) -> Result<(), GpuError> {
        let l = at.layer;
        let env = self.env;
        let Env {
            gpu,
            w,
            sh,
            layers,
            k,
            ..
        } = env;
        let (stream, fault) = (gpu.stream(), gpu.layer_sink(l)?);
        let sp = self.span(at.unit)?;
        let (r0, n) = (sp.r0, sp.n);
        let Ffn::Routed { down_lat, .. } = &layers[l].ffn else {
            return Err(shape(format!(
                "layer {l}: a hosted layer with no routed block"
            )));
        };
        let Prompt {
            gemm,
            dense,
            wide,
            buf,
            taps,
            ..
        } = &mut *self.p;
        gemm.enqueue_route_dense(stream, n, dense, fault)?;
        let wp = WideP {
            gpu,
            w,
            g32: &k.g32,
            gemm,
            dense,
        };
        let Buf {
            hs,
            mix,
            y,
            xn,
            lat,
            probs,
            ids,
            weights,
            ..
        } = buf;
        pre_norm(
            &env,
            (l, Sub::Ffn),
            sp,
            (&hs[sp.cur], mix, y, xn),
            &names::ffn_norm(l),
        )?;
        tap(gpu, taps, (l, FFN_NORM), (&*xn, 0), (r0, n, sh.hidden))?;
        let r = RouteIn {
            w: w.f32_tensor(WHAT, &names::ffn_gate_inp(l))?,
            bias: Some(w.f32_buf(WHAT, &names::exp_probs_b(l))?),
            scale: sh.weights_scale,
            fault,
        };
        k.router.route_rows(
            (stream, gpu, &k.g32),
            xn,
            &r,
            n,
            RoutedRows {
                probs,
                logits: None,
                ids,
                weights,
            },
        )?;
        if let Some(t) = taps.as_mut() {
            tap_words(gpu, &mut t.picks[l], ids, (0, r0 * N_USED, n * N_USED))?;
        }
        project(&wp, wide, l, n, xn, &mut [(down_lat, sh.latent, lat)])?;
        self.side
            .download_rows(port, at, (&*lat, &*weights, &*ids), &[])
    }

    /// A hosted layer's card work under the host's: the shared expert over the unit's rows; the
    /// layer's pick under host streaming ([`PromptStream::unit_pick`](bloomery_gpu::host::prompt_stream::PromptStream::unit_pick)); the card's slots over the
    /// stage stacks behind the pick's landings and the stream's ring, summed in slot order into the
    /// card's routed sum (by the stream's union places when the ring ran); then the unit's read of
    /// the layer ([`PromptStream::unit_read`](bloomery_gpu::host::prompt_stream::PromptStream::unit_read)), which keeps the streamed experts the ring pass joined for the
    /// back.
    fn hosted_shadow(&mut self, port: &mut BatchLeg<'_, HostRun>, at: At) -> Result<(), GpuError> {
        let l = at.layer;
        let env = self.env;
        let Env {
            gpu,
            w,
            sh,
            layers,
            k,
            ..
        } = env;
        let (stream, fault) = (gpu.stream(), gpu.layer_sink(l)?);
        let hid = sh.hidden;
        let (_, n) = self.unit(at.unit)?;
        let (_, m0) = self.unit(0)?;
        let units = self.units.len();
        let Ffn::Routed {
            stacks,
            shared: glu,
            clamp,
            ..
        } = &layers[l].ffn
        else {
            return Err(shape(format!(
                "layer {l}: a hosted layer with no routed block"
            )));
        };
        let Prompt {
            gemm,
            slots,
            dense,
            wide,
            acts,
            buf,
            ..
        } = &mut *self.p;
        gemm.enqueue_route_dense(stream, n, dense, fault)?;
        let wp = WideP {
            gpu,
            w,
            g32: &k.g32,
            gemm,
            dense,
        };
        let Buf {
            xn,
            lat,
            routed,
            fg,
            fu,
            fa,
            shared,
            ids,
            weights,
            gate_rows,
            up_rows,
            down_rows,
            ..
        } = buf;
        project(
            &wp,
            wide,
            l,
            n,
            xn,
            &mut [(&glu.gate, glu.ff, fg), (&glu.up, glu.ff, fu)],
        )?;
        gpu.elem().enqueue_swiglu(stream, fg, fu, n * glu.ff, fa)?;
        project(&wp, wide, l, n, fa, &mut [(&glu.down, hid, shared)])?;
        let side = self.side;
        let card = |k: usize| Self::card(side, k);
        let (ring, land) = self
            .xs
            .stream
            .unit_pick(port, at, (m0, units), self.walk_end, card)?;
        let n_card = side.n_card(l);
        let joined = if n_card > 0 {
            side.places_rows(gpu, l, (&*ids, n), self.sel)?;
            let table = self.routes.of_mut(PASS, l, n_card)?;
            side.remap_row(stream, l, self.map)?;
            // The pick's admits copy into the stage slots the map already names: the one GEMM over
            // the layer's map reads them behind every landing batch's event.
            for event in land.iter().filter_map(|b| b.event.as_deref()) {
                stream.wait(event)?;
            }
            let rule = Act::SwigluGateClamp { limit: *clamp };
            expert_rows(
                &env,
                (l, n),
                (&*gemm, &mut *acts, table),
                (Some(&*self.map), stacks, *clamp),
                ExpertIo {
                    lat: &*lat,
                    ids: &*ids,
                    gate: gate_rows,
                    up: up_rows,
                    down: down_rows,
                },
            )?;
            let joined = match ring {
                Some(r) => {
                    let card_stacks = LegStacks::of(w, [&stacks[0], &stacks[1], &stacks[2]])?;
                    let rows = RingRows {
                        x: &*lat,
                        ids: &*ids,
                        m: n,
                        top_k: N_USED,
                        n_expert: N_EXPERT,
                    };
                    let pass: RingPass<'_> = self.xs.ring_pass(r, slots)?;
                    Some(pass.enqueue(
                        gpu,
                        (Some(&*gemm), &k.g32),
                        (&card_stacks, n_card, sh.expert_ff, sh.latent),
                        rows,
                        (&mut *down_rows, rule),
                        fault,
                    )?)
                }
                None => None,
            };
            let (sel, count) = match joined {
                Some(j) => (self.xs.union_places()?, n_card + j),
                None => (&*self.sel, n_card),
            };
            slots.enqueue_card_acc(
                stream,
                &CardAcc {
                    down: down_rows,
                    w: weights,
                    sel,
                    n: sh.latent,
                    m: n,
                    n_card: count,
                    n_used: N_USED,
                    w_pitch: N_USED,
                    fault,
                },
                routed,
            )?;
            joined
        } else {
            None
        };
        self.xs
            .stream
            .unit_read(port, (at, units), (land, joined), card)
    }

    /// A hosted layer's back from the join on: the stream's join taken (the card's sum already ran
    /// by its union places, [`PromptStream::unit_join`](bloomery_gpu::host::prompt_stream::PromptStream::unit_join)), the card's and the host's sums added, the
    /// latent up, the shared expert added, the sandwich norm, the identity mix's post.
    fn hosted_back(&mut self, port: &BatchLeg<'_, HostRun>, at: At) -> Result<(), GpuError> {
        let l = at.layer;
        self.xs.stream.unit_join(at)?;
        let env = self.env;
        let Env {
            gpu,
            w,
            sh,
            layers,
            k,
            ..
        } = env;
        let (stream, fault) = (gpu.stream(), gpu.layer_sink(l)?);
        let hid = sh.hidden;
        let sp = self.span(at.unit)?;
        let (r0, n) = (sp.r0, sp.n);
        let Ffn::Routed { up_lat, .. } = &layers[l].ffn else {
            return Err(shape(format!(
                "layer {l}: a hosted layer with no routed block"
            )));
        };
        let hsum = port.hsum();
        let Prompt {
            gemm,
            dense,
            wide,
            buf,
            taps,
            ..
        } = &mut *self.p;
        gemm.enqueue_route_dense(stream, n, dense, fault)?;
        let wp = WideP {
            gpu,
            w,
            g32: &k.g32,
            gemm,
            dense,
        };
        let Buf {
            hs,
            mix,
            lat,
            routed,
            lat_up,
            shared,
            f,
            out,
            ..
        } = buf;
        // The card's slots, then the host's: the step's back.
        let sum: &DeviceBuffer<f32> = if self.side.n_card(l) > 0 {
            gpu.elem()
                .enqueue_add(stream, routed, hsum, n * sh.latent, lat)?;
            &*lat
        } else {
            hsum
        };
        project(&wp, wide, l, n, sum, &mut [(up_lat, hid, lat_up)])?;
        let elem = gpu.elem();
        elem.enqueue_add(stream, lat_up, shared, n * hid, f)?;
        elem.enqueue_rms_norm(
            stream,
            f,
            w.f32_buf(WHAT, &names::post_ffw_norm(l))?,
            sh.eps,
            hid,
            n,
            out,
        )?;
        tap(gpu, taps, (l, FFN_OUT), (&*out, 0), (r0, n, hid))?;
        post(&env, hs, sp, (out, mix))?;
        let (_, after) = io(hs, sp.cur);
        tap(gpu, taps, (l, L_OUT), (after, r0), (r0, n, HC * hid))?;
        self.cur[at.unit] ^= 1;
        Ok(())
    }
}

impl<'a> LayerProgram for PassWalk<'a> {
    type Port = BatchLeg<'a, HostRun>;

    const BATCH_ORDER: BatchOrder = GROUP_WALK_ORDER;

    fn host_leg(&self, at: At) -> bool {
        self.side.hosted(at.layer)
    }

    /// A layer the card holds whole entire; a hosted one up to its download.
    fn front(&mut self, port: &mut BatchLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        self.lent(at.unit, |walk| walk.unit_front(port, at))
    }

    fn shadow(&mut self, port: &mut BatchLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        if !self.side.hosted(at.layer) {
            return Ok(());
        }
        self.lent(at.unit, |walk| walk.hosted_shadow(port, at))
    }

    /// The back, then the layer's return, owed in a returning call's last walk
    /// ([`PromptStream::unit_back`](bloomery_gpu::host::prompt_stream::PromptStream::unit_back)).
    fn back(&mut self, port: &mut BatchLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        if !self.side.hosted(at.layer) {
            return Ok(());
        }
        self.lent(at.unit, |walk| walk.hosted_back(port, at))?;
        let card = Self::card(self.side, at.layer);
        self.xs
            .stream
            .unit_back((at, self.units.len()), self.walk_end, card);
        Ok(())
    }
}

/// Walk `toks` (positions `pos0 ..`, at most the pass's rows) through the layers by the host
/// tier's batch port, in units of the pass's rows a unit; the head runs after the last layer when
/// `last`. The positions `media` holds rows for stage those rows ([`stage`]). Under host streaming the walk's end is the stream's ([`PromptStream::end_walk`](bloomery_gpu::host::prompt_stream::PromptStream::end_walk)).
fn walk_pass(
    gpu: &Gpu,
    w: &Weights,
    body: &mut crate::body::Body,
    head: &mut Head,
    (toks, pos0): (&[u32], u32),
    last: bool,
    media: Option<&CallRows<'_>>,
) -> Result<(), GpuError> {
    let crate::body::Body {
        sh,
        layers,
        k,
        stores,
        host,
        rope_rows,
        placed,
        pass,
        taps,
        xs,
        ..
    } = body;
    if taps.is_some() {
        return Err(GpuError::shape(PASS, "taps are armed: a pass does not tap"));
    }
    let PlacedHost { shell, side, .. } = placed.as_mut().ok_or(GpuError::State {
        what: PASS,
        missing: "the placed side of a placed load",
    })?;
    let PlacedPass {
        prompt,
        shape: pass_shape,
        later,
        hsum,
        map,
        sel,
        routes,
        ..
    } = pass.as_deref_mut().ok_or(GpuError::State {
        what: PASS,
        missing: "the pass's buffers (a load opened with its pass)",
    })?;
    let t = toks.len();
    if t == 0 || t > prompt.cfg.cap {
        return Err(shape(format!(
            "a walk of {t} rows in buffers of {}",
            prompt.cfg.cap
        )));
    }
    let rows = pass_shape.rows;
    let units: Vec<(usize, usize)> = chunks(t, rows).map(|r| (r.start, r.len())).collect();
    let cols: Vec<usize> = units.iter().map(|&(_, m)| m).collect();
    let sh = *sh;
    stage(host, prompt, gpu.stream(), &sh, (toks, pos0), media)?;
    let env = Env {
        gpu,
        w,
        sh,
        layers,
        k,
        rope_rows,
    };
    let n_layers = layers.len();
    let mut leg = BatchLeg::new(gpu.stream(), &mut shell.hybrid, hsum, rows);
    leg.set_unit_cols(&cols);
    let mut prog = PassWalk {
        env,
        stores,
        p: &mut *prompt,
        later,
        side: &*side,
        map,
        sel,
        routes,
        xs: &mut *xs,
        units: &units,
        cur: vec![0; units.len()],
        walk_end: pos0 as usize + t,
    };
    let o = Overlap {
        units: units.len(),
        cols: cols.iter().copied().max().unwrap_or(t),
        port: PortKind::Batch,
    };
    sched::walk(o, n_layers, &mut leg, &mut prog)?;
    let cur = prog.cur[units.len() - 1];
    prog.xs.stream.end_walk(leg.hybrid(), gpu.stream())?;
    prog.xs.stream.walk_done();
    if last {
        let n = sh.hidden;
        let x = super::win(&prompt.buf.hs[cur], t - 1, 1, HC * n)?;
        k.hc.enqueue_head(
            gpu.stream(),
            HeadArgs {
                x: &x,
                w: w.f32_buf(WHAT, names::OUTPUT_HC_FN)?,
                scale: w.f32_buf(WHAT, names::OUTPUT_HC_SCALE)?,
                base: w.f32_buf(WHAT, names::OUTPUT_HC_BASE)?,
                eps_norm: sh.eps,
                eps_hc: sh.hc_eps,
                n,
                m: 1,
                y: head.input_mut(),
                fault: gpu.unlabelled_sink(),
            },
        )?;
        head.enqueue(gpu, w)?;
    }
    Ok(())
}

/// The prompt `ids` through `model`'s placed pass from its position, in walks of up to the load's
/// rows: the stores, the engram history and the position stand where `ids.len()` decode steps
/// leave them, and the return is the greedy id after the last row. The call is one residency pass
/// ([`prompt_pass`]: its boundary before the first walk, none inside, 0 rows kept), so every walk
/// reads one slot map, and inside it the call's host streaming ([`PromptStream::begin`](bloomery_gpu::host::prompt_stream::PromptStream::begin) and
/// [`PromptStream::end`](bloomery_gpu::host::prompt_stream::PromptStream::end): a call that continues a sequence holding decode heat returns each layer to
/// the set it started with, a call at position 0 keeps its placement; nothing while the lever is
/// off). Refused by name: no ids, a load made without the pass, a position the stores do not stand
/// at; a call that fails part-way leaves the stores to a reset. The call's own error first: a
/// stream end refused after a failed call is its echo.
pub fn prompt_placed(
    model: &mut GpuModel<crate::body::Body>,
    ids: &[u32],
) -> Result<u32, GpuError> {
    const CALL: &str = "hy4::batch::prompt_placed";
    let token = prompt_pass(model, |model| placed_call(model, ids, None, CALL))?;
    token.ok_or_else(|| GpuError::shape(CALL, "a prompt pass that read no token"))
}

/// The call of [`prompt_placed`] inside the residency pass its caller opened: the walks of `ids`
/// from the model's position, the positions `media` holds rows for staging those rows, and the
/// call's host streaming around them. Refused by name: no ids, a load made without the pass, a
/// position the stores do not stand at.
fn placed_call(
    model: &mut GpuModel<crate::body::Body>,
    ids: &[u32],
    media: Option<&CallRows<'_>>,
    call: &'static str,
) -> Result<Option<u32>, GpuError> {
    if ids.is_empty() {
        return Err(GpuError::shape(call, "a prompt of no id"));
    }
    let rows = model
        .body(call)?
        .pass_rows()
        .ok_or_else(|| GpuError::shape(call, "a load made without the placed pass"))?;
    let n = u32::try_from(ids.len()).map_err(|_| GpuError::shape(call, "ids past u32"))?;
    let start = model.pos();
    let history = {
        let (gpu, _, body) = model.body_parts(call)?;
        body.stream_begin(gpu.stream(), ids.len(), start)?
    };
    let r = model.run_rows(ids.len(), call, |gpu, w, body, head, pos| {
        if pos != body.held {
            return Err(GpuError::shape(
                call,
                format!(
                    "a call at position {pos}; the stores stand at {} (reset after a call that \
                 failed part-way)",
                    body.held
                ),
            ));
        }
        // The cache rooms backed through the call's last position before any walk launches.
        body.fit(pos as usize, ids.len())?;
        // Anything past this point that fails leaves the engram history and the stores part-way.
        body.held = u32::MAX;
        let walks = ids.len().div_ceil(rows);
        for (i, chunk) in ids.chunks(rows).enumerate() {
            let at = pos + (i * rows) as u32;
            walk_pass(gpu, w, body, head, (chunk, at), i + 1 == walks, media)?;
        }
        body.held = pos + n;
        Ok(true)
    });
    let (gpu, _, body) = model.body_parts(call)?;
    let ended = body.stream_end(gpu.stream(), r.is_ok() && !history);
    let v = r?;
    ended?;
    Ok(v)
}

/// The media prompt of a placed load: the walks of [`prompt_placed`] with each span's rows staged
/// at its positions in place of their ids' embeddings, inside the residency pass the session opens
/// around a call (`app::Session::prompt_media`), so the call leaves what a text prompt of the same
/// length does and the greedy token after the last id is the head's.
impl MediaBody for crate::body::Body {
    type Span<'a> = MediaSpan<'a>;

    fn media_span<'s>(s: &'s MediaSpan<'_>) -> &'s MediaSpan<'s> {
        s
    }

    /// A load made without the placed pass steps its prompt one id at a time and has no place
    /// for a span's rows.
    fn media_refusal(m: &GpuModel<Self>, spans: usize) -> Option<String> {
        let stepped = m.body(MEDIA).is_ok_and(|b| b.pass_rows().is_none());
        (spans > 0 && stepped).then(|| {
            format!(
                "{MEDIA}: a load made without the placed pass steps its prompt one id at a time; \
                 a media prompt needs the pass's rows"
            )
        })
    }

    /// Refused by name before anything runs: a call of no id, a span the call does not hold
    /// whole, spans out of order, a load without the pass ([`placed_call`]).
    fn prefill_media(
        m: &mut GpuModel<Self>,
        ids: &[u32],
        spans: &[MediaSpan<'_>],
    ) -> Result<u32, GpuError> {
        let n_embd = m.body(MEDIA)?.sh.hidden;
        if ids.is_empty() {
            return Err(GpuError::shape(MEDIA, "a media prompt of no ids"));
        }
        check_spans(MEDIA, spans, ids.len(), n_embd)?;
        let rows = CallRows::new(m.pos(), n_embd, spans.iter().map(|s| (&s.at, s.rows)))
            .map_err(|r| GpuError::shape(MEDIA, r.to_string()))?;
        placed_call(m, ids, Some(&rows), MEDIA)?
            .ok_or_else(|| GpuError::shape(MEDIA, "a media prompt pass that read no token"))
    }
}

/// The media call's name in its refusals.
const MEDIA: &str = "hy4::batch::prefill_media";

impl crate::body::Body {
    /// The rows the load's placed pass holds; `None` on a load made without it.
    #[must_use]
    pub fn pass_rows(&self) -> Option<usize> {
        self.pass.as_deref().map(PlacedPass::rows)
    }

    /// The load's placed pass, for a gate that arms its taps and reads them back.
    pub fn pass_mut(&mut self) -> Option<&mut PlacedPass> {
        self.pass.as_deref_mut()
    }

    /// The load's placed pass.
    #[must_use]
    pub fn pass(&self) -> Option<&PlacedPass> {
        self.pass.as_deref()
    }

    /// The routed layers the host tier serves a share of; 0 on a whole-card load.
    #[must_use]
    pub fn hosted_layers(&self) -> usize {
        self.placed.as_ref().map_or(0, |p| {
            (0..self.layers.len()).filter(|&l| p.side.hosted(l)).count()
        })
    }

    /// The device bytes the load's placed pass holds; 0 on a load made without it.
    #[must_use]
    pub fn pass_device_bytes(&self) -> usize {
        self.pass.as_deref().map_or(0, PlacedPass::bytes)
    }

    /// The host tier's census since load ([`TierCensus`]: the services' picks and where the host
    /// read them, and the slots a prompt left out for the card's route); `None` on a whole-card
    /// load and on a host that counts no reads.
    #[must_use]
    pub fn host_census(&self) -> Option<TierCensus> {
        self.placed.as_ref()?.shell.hybrid.census(None)
    }

    /// The host tier's service counters since load ([`HybridStats`]: the batch serves and their
    /// columns among them); `None` on a whole-card load.
    #[must_use]
    pub fn host_stats(&self) -> Option<HybridStats> {
        Some(self.placed.as_ref()?.shell.hybrid.stats())
    }

    /// Open the prompt call's streaming over `n` ids from `start` ([`PromptStream::begin`](bloomery_gpu::host::prompt_stream::PromptStream::begin), the
    /// walks being every pass of a placed load): a call whose start holds decode history returns
    /// to it. Whether it does; `false` and nothing more on a whole-card load or while the lever is
    /// off.
    pub(crate) fn stream_begin(
        &mut self,
        stream: &CudaStream,
        n: usize,
        start: u32,
    ) -> Result<bool, GpuError> {
        let layers = self.layers.len();
        let Some(p) = self.placed.as_mut() else {
            return Ok(false);
        };
        // Read once: the call keeps no row, so the history does not move until its end.
        let history = p.shell.hybrid.call_history(start);
        self.xs.stream.begin(
            &mut p.shell.hybrid,
            stream,
            true,
            n,
            start as usize + n,
            history,
            ADMIT_FLOOR,
            0.0,
            layers,
        )?;
        Ok(history)
    }

    /// End the prompt call's streaming ([`PromptStream::end`](bloomery_gpu::host::prompt_stream::PromptStream::end)), its placement `kept` for the
    /// decode after it; nothing on a whole-card load.
    pub(crate) fn stream_end(&mut self, stream: &CudaStream, kept: bool) -> Result<(), GpuError> {
        match self.placed.as_mut() {
            Some(p) => self.xs.stream.end(&mut p.shell.hybrid, stream, kept),
            None => Ok(()),
        }
    }

    /// The stream rule's costs in place of the priced ones ([`PromptStream::set_costs`](bloomery_gpu::host::prompt_stream::PromptStream::set_costs)). Refused
    /// by name: a load with no host tier, before `split` started the stream, and while a call
    /// streams.
    pub fn set_xstream_costs(&mut self, costs: Costs) -> Result<(), GpuError> {
        let p = self.placed.as_mut().ok_or(GpuError::State {
            what: "hy4::Body::set_xstream_costs",
            missing: "a placed load (the host tier the stream serves)",
        })?;
        self.xs.stream.set_costs(&mut p.shell.hybrid, costs)
    }

    /// The last streaming prompt call's streamed layers, each with its walk, and the stream's
    /// end; empty and `None` after a call that did not stream. Taken
    /// ([`PromptStream::take_xstream_records`](bloomery_gpu::host::prompt_stream::PromptStream::take_xstream_records)).
    pub fn take_xstream_records(&mut self) -> (Vec<(usize, XLayer)>, Option<XReport>) {
        self.xs.stream.take_xstream_records()
    }

    /// The last streaming prompt call's picks, each with its walk, and its end; empty and `None`
    /// after a call that did not stream. Taken ([`PromptStream::take_pick_records`](bloomery_gpu::host::prompt_stream::PromptStream::take_pick_records)).
    pub fn take_stream_records(&mut self) -> (Vec<(usize, CallPick)>, Option<CallReport>) {
        self.xs.stream.take_pick_records()
    }

    /// The last streaming call's ring passes: each one's walk, layer and the streamed experts its
    /// unit summed the card's slots by ([`PromptStream::take_join_records`](bloomery_gpu::host::prompt_stream::PromptStream::take_join_records)). Taken: a second read
    /// is empty. A gate's instrument.
    pub fn take_stream_joined(&mut self) -> Vec<(usize, usize, usize)> {
        self.xs.stream.take_join_records()
    }
}

impl XStreamBody for crate::body::Body {
    /// What the next prompt calls move (`BLOOMERY_XSTREAM`): `off` until set here. `Split` starts
    /// the expert stream at its first setting ([`PromptStream::start_split`](bloomery_gpu::host::prompt_stream::PromptStream::start_split): the ring sized from
    /// the card's free bytes past the stream's keep-free and the plan's ring term, the lane's
    /// probe, the costs this device and host price for the latent experts' shape, read once for
    /// each of a walk's units) and arms the ring pass's buffers at a unit's rows; the stream then
    /// stays for the load's life. Refused by name: `admit` or `split` on a whole-card load,
    /// `split` on a load made without the placed pass or whose hosted layers' card stacks are of
    /// no one ring form, and the stream's own refusals (a plan that counts no ring term among
    /// them, [`plan_holds_no_ring`](bloomery_gpu::host::xstream::plan_holds_no_ring)).
    fn set_xstream(&mut self, gpu: &Gpu, mode: XMode) -> Result<(), GpuError> {
        const WHAT_S: &str = "hy4::Body::set_xstream";
        let pass = self
            .pass
            .as_deref()
            .map(|p| (p.shape, p.forms.ok_or(p.shape)));
        let (latent, ff) = (self.sh.latent, self.sh.expert_ff);
        let tier = self.placed.as_mut().map(|p| &mut p.shell.hybrid);
        self.xs.set_xstream(tier, (gpu, WHAT_S), mode, |room| {
            let (shape, forms) = pass.ok_or(GpuError::State {
                what: WHAT_S,
                missing: "a load opened with its placed pass (the walks the stream serves)",
            })?;
            let forms = forms.map_err(|_| GpuError::State {
                what: WHAT_S,
                missing: "hosted layers' card stacks of one form a route GEMM entry reads",
            })?;
            let expert = ExpertShape {
                hidden: latent,
                ff,
                top_k: N_USED,
                units: shape.units,
            };
            let ring = RingShape {
                rows: shape.rows,
                top_k: N_USED,
                n: latent,
                ff,
                forms,
            };
            Ok((XCfg::new(expert, N_EXPERT, DEADLINE, room), ring))
        })
    }

    /// The load line's `xstream=` word ([`PromptStream::word`](bloomery_gpu::host::prompt_stream::PromptStream::word)); the mode alone on a whole-card
    /// load.
    fn xstream_word(&self) -> String {
        match &self.placed {
            Some(p) => self.xs.stream.word(&p.shell.hybrid),
            None => self.xs.stream.mode().word().to_string(),
        }
    }
}
