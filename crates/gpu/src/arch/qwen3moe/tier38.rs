//! Qwen3.8's expert tier ([`crate::host::tier`]): a card beside the stage
//! card that holds some of each routed layer's experts — plan (b′)'s 3090,
//! the next ids after the stage card's prefix — and computes the routed
//! slots the slot map sends it, in the host leg's shadow.
//!
//! The stage card's side is [`TierSide38`]: the slot map's tier view, which
//! the tier handoff (`ds41_ffn_handoff_10_cols_tier`) reads each slot's tier
//! place from, the unit's tier places, which the join
//! (`q38_card_tier_shared_add`, `card38`) reads, and each layer's tier
//! count. A row's tier image carries the normed activation in f32
//! ([`TierAct::F32`]), a column a position: the gate·up `_sel` reads the
//! q8_1 planes of Walk A, which the stage card makes in the shadow after the
//! go, and the tier quantizes the same values with the same kernel. The
//! page's row carries the step port's columns ([`TierSide38::new`]): the
//! step's one, a verify's `Cols(m)` and a pass of several slots' rows.
//!
//! The tier's own layer ([`Tier38`], the architecture's [`TierExperts`]) is
//! the stage card's card leg over the tier's stacks ([`SelRun`]: the q8_1 of
//! the columns, the gate·up `_sel` of the layer's type with SiLU·mul, the
//! 32-value q8_1 of the tier slots' columns, the down `_sel` of its type)
//! into the row's routed rows, slot-major. Each slot's down output is the one
//! the stage card would write for it, so the join over both cards' slots in
//! slot order is the one-card sum over their union.
//!
//! A prompt batch's block ([`Tier38::enqueue_block`], the ubatch walk's
//! tier leg through the batch port) runs the stage card's ubatch card route
//! (`wide38`'s `CardRoute38`) over the tier's stacks ([`BlockRoute38`]): per
//! run of [`ROUTE_ROWS`] tokens the route's expert ids from the tier places
//! (`q38_tier_ids`), the remapped route table over the identity map of the
//! tier's experts, the q8_1 of the run's f32 rows, the gate and up GEMMs, the
//! tier slots' SwiGLU and the down GEMM into the run's own down rows, slot
//! by slot, and the run's tier slots' rows packed into the block's down
//! outputs at their ranks (`q38_tier_rank` once a block over its places,
//! `q38_tier_rows_pack` a run, [`BlockRows::Packed`]); the batch service
//! then copies the packed rows alone to the set's rows, by the copy engine:
//! a block's tier slots are a tenth of its slots, and only theirs cross the
//! bus. The stage card ranks the same places with the same kernel. Each
//! slot's down output is the one the stage card's route writes for it — the
//! GEMMs sum each (slot, row) in an order fixed by K, over the same q8_1
//! bytes of the same f32 rows — so the stage's join over both cards' slots
//! in slot order (`q38_card_tier_acc`) is the one-card sum over their union.

use super::card38::{LegLayer, SelRun, leg_layers};
use super::plan38::geo;
use super::scratch::{cells_view, param_view};
use super::scratch38::ROUTE_ROWS;
use super::swap38::Qwen38Stacks;
use super::wide38::down_weight;
use crate::card_leg::RouteW;
use crate::gemm::{
    Gemm32Args, Gemm32Kernels, GemmAct, GemmAct32, GemmArgs, GemmInput, GemmKernels, GemmRoute,
    SwigluSlots,
};
use crate::host::node::{NodeFamily, NodeOpen};
use crate::host::tier::{
    BlockRows, TierAct, TierBlock, TierCard, TierExperts, TierInput, TierIo, TierOpen, TierSet,
    TierShape,
};
use crate::hybrid::{HOST, SlotMap};
use crate::q38::{Q38Kernels, TierRowsPackArgs};
use crate::weights::Weights;
use crate::{DeviceTensor, Gpu, GpuError};
use cuda_core::DeviceBuffer;
use gguf::Split;
use model::placement::{ModelTensors, Plan};

/// What the tier's errors name.
const WHAT: &str = "qwen4exp tier";

/// The tier the stage card's tier entries serve: they hand one tier its
/// image and join one tier's rows, so a load holds one tier card
/// ([`crate::host::refuse_tier_count`]).
pub(super) const STAGE_TIER: usize = 0;

/// The shapes Qwen3.8's tier is cut for: one row of the normed activation in
/// f32, ten routed slots a column.
const SHAPE: TierShape = TierShape {
    hidden: geo::HIDDEN,
    n_used: geo::N_USED,
    rows: 1,
    act: TierAct::F32,
};

/// Qwen3.8's tier as a tier node opens it ([`crate::host::node::serve`]):
/// the file's tensors as Qwen3.8's placement reads them, and [`Tier38`] over
/// the stacks the node loaded for the open's set.
pub const NODE_FAMILY: NodeFamily = NodeFamily {
    word: "qwen4exp",
    tensors: node_tensors,
    experts: node_experts,
    host: node_host,
};

fn node_tensors(file: &Split) -> Result<ModelTensors, GpuError> {
    model::arch::qwen35moe::place::PlanInputs::describe(file)
        .map(|i| i.model)
        .map_err(|e| GpuError::plan(WHAT, e))
}

fn node_host(
    file: std::sync::Arc<Split>,
    open: &NodeOpen,
) -> Result<crate::host::run::HostRun, GpuError> {
    if open.shape != SHAPE || open.set.n_expert() != geo::EXPERTS {
        return Err(GpuError::shape(
            WHAT,
            "the host tier's shape is not Qwen3.8's",
        ));
    }
    let inputs = model::arch::qwen35moe::place::PlanInputs::describe(&file)
        .map_err(|e| GpuError::plan(WHAT, e))?;
    crate::host::run::HostRun::build(
        file,
        0,
        false,
        crate::host::run::HostWidths {
            embd: geo::HIDDEN,
            ff: geo::FF,
            n_used: geo::N_USED,
        },
        |src| model::arch::qwen35moe::host::layers_of_spec(src, &inputs.spec, 0..open.layers),
    )
}

fn node_experts(
    gpu: &Gpu,
    _: &Split,
    model: &ModelTensors,
    w: &Weights,
    open: &NodeOpen,
) -> Result<Box<dyn TierExperts>, GpuError> {
    open.block
        .0
        .checked_mul(geo::N_USED)
        .and_then(|slots| slots.checked_mul(size_of::<u32>()))
        .ok_or_else(|| {
            GpuError::shape(
                WHAT,
                format!(
                    "a block of {} columns overflows its route bytes",
                    open.block.0
                ),
            )
        })?;
    if open.set.n_expert() != geo::EXPERTS || open.shape != SHAPE {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "a tier of {} experts cut for {:?}; Qwen3.8's is {} experts cut for {SHAPE:?}",
                open.set.n_expert(),
                open.shape,
                geo::EXPERTS
            ),
        ));
    }
    let stacks = super::swap38::stacks_of_set(model, &open.set)?;
    let experts = Tier38::new(
        gpu,
        w,
        &stacks,
        &open.set,
        open.layers,
        open.cols,
        open.block,
    )?;
    Ok(Box::new(experts))
}

/// The stage card's side of the tier layers: the slot map's tier view (a row
/// of places per layer at the card copy's row offsets), the unit's tier
/// places of its routed slots (ten a column, the step port's columns), and
/// per layer the experts the tier holds. Built once at load by a body with a
/// tier card.
pub(super) struct TierSide38 {
    places: DeviceTensor<u32>,
    pub(super) tsel: DeviceBuffer<u32>,
    k: Vec<usize>,
}

impl TierSide38 {
    /// The side for `map`'s tier [`STAGE_TIER`] over its layers, its places
    /// for `cols` columns (the step port's, which the tier card's page
    /// carries too). Load-time only.
    pub(super) fn new(gpu: &Gpu, map: &SlotMap, cols: usize) -> Result<TierSide38, GpuError> {
        let stream = gpu.stream();
        let run = map.layers();
        let k = run
            .clone()
            .map(|l| map.on_tier_of(STAGE_TIER, l))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(TierSide38 {
            places: DeviceTensor::upload(
                stream,
                &map.tier_view(STAGE_TIER)?,
                run.len(),
                map.n_expert(),
            )?,
            tsel: DeviceBuffer::zeroed(stream, cols * geo::N_USED)?,
            k,
        })
    }

    /// The tier's experts of layer `l`: 0 off the tier.
    pub(super) fn k(&self, l: usize) -> usize {
        self.k.get(l).copied().unwrap_or(0)
    }

    /// The layers the tier holds experts of.
    pub(super) fn layers(&self) -> usize {
        self.k.iter().filter(|&&k| k > 0).count()
    }

    /// The card copy of the map's tier view: a row of [`geo::EXPERTS`]
    /// places a layer, each expert's tier slot or [`HOST`].
    pub(super) fn view(&self) -> &DeviceBuffer<u32> {
        self.places.buf()
    }

    /// What a tier layer's handoff reads and writes: the card copy of the
    /// map's tier view and the unit's tier places.
    pub(super) fn handoff_parts(&mut self) -> (&DeviceBuffer<u32>, &mut DeviceBuffer<u32>) {
        (self.places.buf(), &mut self.tsel)
    }

    /// Device bytes the side holds.
    pub(super) fn bytes(&self) -> usize {
        self.places.buf().num_bytes() + self.tsel.num_bytes()
    }
}

/// The Qwen3.8 tier card's computation ([`TierExperts`]): per tier layer the
/// stage card's card-leg launches over the tier's stacks into the row's
/// routed rows (module doc). Its scratch, for up to the step port's columns,
/// is made at load.
pub(super) struct Tier38 {
    run: SelRun,
    /// Per layer of the model, the tier's experts and stacks; `None` off
    /// the tier.
    layers: Vec<Option<LegLayer>>,
    /// The prompt-batch block's route and its buffers.
    block: BlockRoute38,
}

/// The tier's prompt-batch block route (module doc): its modules on the
/// tier card, its buffers for one run of `run` tokens, one route table and
/// one identity map per distinct tier count, and the rest of the plan's
/// reserve for them ([`BlockRoute38::new`]).
struct BlockRoute38 {
    gemm: GemmKernels,
    g32: Gemm32Kernels,
    q38: Q38Kernels,
    /// The most columns a block holds, and the tokens of one run.
    cols: usize,
    run: usize,
    /// The route's expert ids, ten a run token: the tier places with
    /// [`HOST`] as the tier's count.
    ids: DeviceBuffer<u32>,
    /// The run's f32 rows in the K-quant GEMM's activation form.
    x: GemmAct,
    /// The gate and up GEMMs' outputs, ten slots a run token of
    /// [`geo::FF`] values.
    g: DeviceBuffer<f32>,
    u: DeviceBuffer<f32>,
    /// The tier slots' SwiGLU in the 32-value GEMM's activation form.
    act: GemmAct32,
    /// The down GEMM's output, ten slots a run token of [`geo::HIDDEN`]
    /// values, slot-major, which the run's pack reads.
    run_down: DeviceBuffer<f32>,
    /// Each of the block's slots' rank among its tier slots
    /// (`q38_tier_rank`), the row its down output takes packed.
    rank: DeviceBuffer<u32>,
    /// One route table and one map (`0 .. n`, then [`HOST`]) per distinct
    /// tier count `counts[i]`.
    routes: Vec<GemmRoute>,
    maps: Vec<DeviceBuffer<u32>>,
    counts: Vec<usize>,
    /// The plan's reserve for the route past what the tables of these counts
    /// hold: the plan counts two tables at the most experts a layer has,
    /// before it knows the tier's counts, and the tier holds the rest of
    /// that bound here, so its block bytes are the reserve.
    slack: Option<DeviceBuffer<u8>>,
}

impl BlockRoute38 {
    /// The route on `gpu` for blocks of up to `cols` columns over tier
    /// layers of `counts` experts (distinct, ascending): runs of
    /// `min(cols, ROUTE_ROWS)` tokens; `reserve` the plan's bytes for it
    /// (`place::tier_route_scratch_bytes`), of which what the buffers do not
    /// hold is the slack. More than two counts, and buffers past the
    /// reserve, are refused by name. Load-time only.
    fn new(
        gpu: &Gpu,
        cols: usize,
        counts: Vec<usize>,
        reserve: usize,
    ) -> Result<BlockRoute38, GpuError> {
        if counts.is_empty() || counts.len() > 2 {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "a block route over tier counts {counts:?}: the route's tables are built for \
                     one or two (the placement's spread keeps every count within one)"
                ),
            ));
        }
        let stream = gpu.stream();
        let ctx = gpu.context();
        let run = cols.min(ROUTE_ROWS);
        let slots = run * geo::N_USED;
        let map = |n: usize| -> Result<DeviceBuffer<u32>, GpuError> {
            let mut m: Vec<u32> = (0..n as u32).collect();
            m.push(HOST);
            Ok(DeviceBuffer::from_host(stream, &m)?)
        };
        let mut r = BlockRoute38 {
            gemm: GemmKernels::load(ctx)?,
            g32: Gemm32Kernels::load(ctx)?,
            q38: Q38Kernels::load(ctx)?,
            cols,
            run,
            ids: DeviceBuffer::zeroed(stream, slots)?,
            x: GemmAct::new(stream, run, geo::HIDDEN)?,
            g: DeviceBuffer::zeroed(stream, slots * geo::FF)?,
            u: DeviceBuffer::zeroed(stream, slots * geo::FF)?,
            act: GemmAct32::new(stream, slots, geo::FF)?,
            run_down: DeviceBuffer::zeroed(stream, slots * geo::HIDDEN)?,
            rank: DeviceBuffer::zeroed(stream, cols * geo::N_USED)?,
            routes: counts
                .iter()
                .map(|&n| GemmRoute::new(stream, slots, n))
                .collect::<Result<_, _>>()?,
            maps: counts.iter().map(|&n| map(n)).collect::<Result<_, _>>()?,
            counts,
            slack: None,
        };
        let held = r.held();
        let slack = reserve.checked_sub(held).ok_or_else(|| {
            GpuError::shape(
                WHAT,
                format!(
                    "the block route holds {held} B on the tier card, past the plan's reserve of \
                     {reserve} B for it"
                ),
            )
        })?;
        if slack > 0 {
            r.slack = Some(DeviceBuffer::zeroed(stream, slack)?);
        }
        Ok(r)
    }

    /// Device bytes of the buffers, the slack left out.
    fn held(&self) -> usize {
        self.ids.num_bytes()
            + self.x.bytes()
            + self.g.num_bytes()
            + self.u.num_bytes()
            + self.act.bytes()
            + self.run_down.num_bytes()
            + self.rank.num_bytes()
            + self.routes.iter().map(GemmRoute::bytes).sum::<usize>()
            + self.maps.iter().map(DeviceBuffer::num_bytes).sum::<usize>()
    }

    /// Device bytes, the slack in.
    fn bytes(&self) -> usize {
        self.held() + self.slack.as_ref().map_or(0, DeviceBuffer::num_bytes)
    }

    /// Enqueue on `gpu`'s stream layer `l`'s block route over `cl`'s stacks
    /// in `w` (module doc): the ranks of the block's tier slots, then per run
    /// the ids, the route, the q8_1 of the run's rows of `io.x`, the gate and
    /// up GEMMs, the SwiGLU of the tier slots, the down GEMM into the run's
    /// rows and their pack into `io.down` at their ranks. A block of no
    /// column or past the route's columns, buffers short of the block, a
    /// count the route holds no table for, a gate and up the grouped GEMM does
    /// not read over q8_1 rows (NVFP4's planes among them;
    /// `crate::card_leg::StackW::route`) and a down no route GEMM reads are
    /// refused by name. Eager, allocation-free.
    fn enqueue(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        (l, cl): (usize, &LegLayer),
        io: TierBlock<'_>,
    ) -> Result<(), GpuError> {
        let (cols, n, h) = (io.cols, geo::N_USED, geo::HIDDEN);
        if cols == 0 || cols > self.cols {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "layer {l}: a block of {cols} columns; the route takes 1..={}",
                    self.cols
                ),
            ));
        }
        if io.x.len() < cols * h || io.sel.len() < cols * n || io.down.len() < cols * n * h {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "layer {l}: a block of {cols} columns over rows of {}, places of {} and down \
                     outputs of {} values",
                    io.x.len(),
                    io.sel.len(),
                    io.down.len()
                ),
            ));
        }
        let st = cl.stacks(w)?;
        let n_tier = st.n_card;
        let at = self
            .counts
            .iter()
            .position(|&c| c == n_tier)
            .ok_or_else(|| {
                GpuError::shape(
                    WHAT,
                    format!(
                        "layer {l}: a route table of {n_tier} experts; the route holds {:?}",
                        self.counts
                    ),
                )
            })?;
        let (Some(RouteW::Grouped(gate_up_ty, gate)), Some(RouteW::Grouped(_, up))) =
            (st.gate.route(st.gate_up_ty), st.up.route(st.gate_up_ty))
        else {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "layer {l}: a gate and an up of {} on the tier card, whose block route reads \
                     them by the grouped GEMM over q8_1 rows",
                    st.gate_up_ty
                ),
            ));
        };
        let down = down_weight(WHAT, l, st.down_ty, st.down)?;
        let (stream, sink) = (gpu.stream(), gpu.layer_sink(l)?);
        self.q38
            .enqueue_tier_rank(stream, io.sel, (cols * n, n_tier), &mut self.rank)?;
        let mut c0 = 0;
        while c0 < cols {
            let m = self.run.min(cols - c0);
            // SAFETY: tokens c0 .. c0 + m <= cols of the block's rows and
            // places and of the ranks (their lengths checked above and at
            // load); every buffer stays in place for this run's launches.
            let (sel_w, x_w, rank_w) = unsafe {
                (
                    param_view::<u32>(io.sel, c0 * n, m * n),
                    cells_view(io.x, c0 * h, m * h),
                    param_view::<u32>(&self.rank, c0 * n, m * n),
                )
            };
            self.q38
                .enqueue_tier_ids(stream, &sel_w, (m * n, n_tier), &mut self.ids)?;
            self.g32.enqueue_route_remap(
                stream,
                &self.ids,
                &self.maps[at],
                m * n,
                &mut self.routes[at],
                sink,
            )?;
            gpu.enqueue_quantize_gemm(&x_w, m, &mut self.x, sink)?;
            macro_rules! args {
                ($w:expr, $y:expr) => {
                    GemmArgs {
                        ty: gate_up_ty,
                        w: $w,
                        rows_per_expert: geo::FF,
                        act: &self.x,
                        route: &self.routes[at],
                        input: GemmInput::Shared { top_k: n },
                        y: $y,
                    }
                };
            }
            self.gemm.enqueue_gemm(stream, args!(gate, &mut self.g))?;
            self.gemm.enqueue_up_swiglu32(
                &self.g32,
                stream,
                args!(up, &mut self.u),
                &self.g,
                SwigluSlots::Card {
                    sel: &sel_w,
                    n_card: n_tier,
                },
                &mut self.act,
                sink,
            )?;
            self.g32.enqueue_gemm32(
                stream,
                Gemm32Args {
                    w: down,
                    rows_per_expert: h,
                    act: &self.act,
                    route: &self.routes[at],
                    input: GemmInput::PerSlot,
                    y: &mut self.run_down,
                },
            )?;
            self.q38.enqueue_tier_rows_pack(
                stream,
                TierRowsPackArgs {
                    down: &self.run_down,
                    sel: &sel_w,
                    rank: &rank_w,
                    n: h,
                    slots: m * n,
                    n_tier,
                    rows_cap: cols * n,
                    fault: sink,
                    rows: &mut *io.down,
                },
            )?;
            c0 += m;
        }
        Ok(())
    }
}

impl Tier38 {
    /// The tier's computation on `gpu` over `set`, whose stacks `w` holds
    /// (each tier layer's routed gate, up and down with its experts in tier
    /// slot order), each layer's types as `stacks` lists them, for a model of
    /// `layers` layers; its card leg's scratch for `step_cols` columns, the
    /// step port's; its block route for blocks of up to `cols` columns in
    /// `reserve`, the plan's block scratch for the tier. A tier
    /// layer whose stacks are absent, of a type the leg does not run or of
    /// other rows than its experts', and a layer off the tier with resident
    /// stacks, are refused by name. Load-time only.
    pub(super) fn new(
        gpu: &Gpu,
        w: &Weights,
        stacks: &Qwen38Stacks,
        set: &TierSet,
        layers: usize,
        step_cols: usize,
        (cols, reserve): (usize, usize),
    ) -> Result<Tier38, GpuError> {
        let run = set.layers();
        let per = leg_layers(
            w,
            stacks,
            |l| {
                if run.contains(&l) {
                    set.on_tier(l)
                } else {
                    Ok(0)
                }
            },
            layers,
        )?;
        let mut counts: Vec<usize> = per.iter().flatten().map(LegLayer::n).collect();
        counts.sort_unstable();
        counts.dedup();
        Ok(Tier38 {
            run: SelRun::new(gpu, step_cols)?,
            block: BlockRoute38::new(gpu, cols, counts, reserve)?,
            layers: per,
        })
    }
}

impl TierExperts for Tier38 {
    /// The stage card's card-leg launches of the go's columns over the
    /// tier's stacks but the card sum: the staged activation's q8_1, the
    /// gate·up `_sel`, the q8_1 of the tier slots' columns and the down
    /// `_sel` into the row's routed rows.
    fn enqueue_layer(
        &mut self,
        gpu: &Gpu,
        weights: &Weights,
        layer: usize,
        io: TierIo<'_>,
    ) -> Result<(), GpuError> {
        let TierInput::F32(x) = io.act else {
            return Err(GpuError::shape(
                WHAT,
                "a q8_1 tier image: Qwen3.8's tier quantizes the f32 activation itself",
            ));
        };
        let Tier38 { run, layers, .. } = self;
        let cl = layers
            .get(layer)
            .and_then(Option::as_ref)
            .ok_or_else(|| GpuError::shape(WHAT, format!("layer {layer} holds no tier expert")))?;
        run.enqueue(gpu, weights, (layer, cl), x, io.cols, io.sel, io.rows)
    }

    /// The block route over the tier's stacks of `layer` (module doc,
    /// [`BlockRoute38::enqueue`]).
    fn enqueue_block(
        &mut self,
        gpu: &Gpu,
        weights: &Weights,
        layer: usize,
        io: TierBlock<'_>,
    ) -> Result<(), GpuError> {
        let Tier38 { layers, block, .. } = self;
        let cl = layers
            .get(layer)
            .and_then(Option::as_ref)
            .ok_or_else(|| GpuError::shape(WHAT, format!("layer {layer} holds no tier expert")))?;
        block.enqueue(gpu, weights, (layer, cl), io)
    }

    /// The block route's bytes: its buffers and the slack of the plan's
    /// reserve ([`BlockRoute38`]).
    fn block_bytes(&self) -> usize {
        self.block.bytes()
    }

    /// The block packs its tier slots' rows at the front of its down
    /// outputs, by rank, which the batch service copies alone.
    fn block_rows(&self) -> BlockRows {
        BlockRows::Packed
    }
}

/// The tier card `t` of `plan`, tier `tier` of `map`, for the stage card
/// `stage`'s load, opened in this process or on its node through the common
/// node owner ([`NodeOpen::open_tier`], [`NODE_FAMILY`]) from the plan's
/// segments of its card: its set the map's rows of that tier, Qwen3.8's tier
/// computation over them (each layer's types as the set's stacks list them,
/// `layers` layers, the plan's block route's columns and the plan's block
/// scratch `block`), its page rows one of `step_cols` columns, the step
/// port's; `card_dontneed` as for the stage card's segments. The stage
/// card's context is current again on return. Load-time only.
#[allow(
    clippy::too_many_arguments,
    reason = "the stage card, the file, the plan, the tier and its index, the map, the layers, the step port's columns, the block scratch and the page lever (rust-quality R8)"
)]
pub(super) fn open_tier(
    stage: &Gpu,
    file: &std::sync::Arc<Split>,
    plan: &Plan<'_>,
    (t, tier): (&TierOpen, usize),
    map: &SlotMap,
    layers: usize,
    step_cols: usize,
    block: (usize, usize),
    card_dontneed: bool,
) -> Result<TierCard, GpuError> {
    if map.n_expert() != geo::EXPERTS {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "a map of {} experts; the tier is built for {}",
                map.n_expert(),
                geo::EXPERTS
            ),
        ));
    }
    let set = TierSet::of_map(map, tier)?;
    NodeOpen {
        nvme: None,
        family: NODE_FAMILY.word.to_string(),
        name: t.name.clone(),
        device: t.device,
        card: t.card,
        rows: NodeOpen::rows_of(plan, file, t.card)?,
        set,
        shape: SHAPE,
        cols: step_cols,
        layers,
        block,
    }
    .open_tier(stage, file, plan, t.node, &NODE_FAMILY, card_dontneed)
}
