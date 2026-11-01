//! The feed-forward blocks of one layer at one token: `x` (the fold of the
//! streams by the block's own mix) in, the update into `out`.
//!
//! A dense block ([`dense`]): `ffn_norm` RMS, then the common gated block
//! (`bloomery_gpu_deepseek41::block`: the gate·up·SwiGLU with the shared
//! limit and the down projection, launched by the file's weight types). Three
//! launches over q8_0 planes, [`BlockPlan::extra_launches`] more over K-quants,
//! all in the layer's front.
//!
//! A routed block runs around its host leg; the card computes the experts the
//! slot map puts on it ([`CardExperts`]), the host tier the rest:
//! - [`front`]: `ffn_norm` RMS into the boundary's handoff region, the
//!   sigmoid router over 288 experts with the selection bias (top 8, the
//!   weights from the unbiased scores, normalized and scaled), the handoff
//!   into the host-mapped image — each slot's place in the card's stacks, or
//!   the host mark, from the layer's slot-map row — and the go. Three kernels
//!   and the go.
//! - [`shadow`]: while the host computes, the card's slots — the common card
//!   leg (`bloomery_gpu::card_leg`): the norm's q8_1 form, the gate·up `_sel`
//!   with the routed clamped SwiGLU, the q8_1 of those slots' columns and the
//!   down `_sel` — then the shared expert (the common gated block, as the
//!   dense one), and the card slots' weighted sum joined with the shared
//!   expert's output in one launch (`card_sum_join`, [`enqueue_join`]). Two
//!   kernels without card experts over q8_0 planes, [`CARD_LAUNCHES`] more
//!   with, and [`BlockPlan::extra_launches`] more over K-quants.
//! - [`back`]: the wait, then the host's routed sum plus the shadow's:
//!   `hsum + (card + shexp)`, or `hsum + shexp` on a layer without card
//!   experts (`moe + shexp` as ik adds them). One kernel after the wait.
//!
//! A layer the expert tier card holds experts of (a load with a tier,
//! [`crate::tier`]) runs as a tier layer: the front's handoff also writes the
//! tier's image (the normed row in f32 and each slot's tier place,
//! `FfnKernels::enqueue_handoff_tier`) and the go is the host's and the
//! tier's; the shadow runs the card slots' downs but not their sum; the back
//! waits for the host and the tier, then sums the card's and the tier's rows
//! in one weighted sum (`ds41_ffn_card_acc_8_tier`, the card sum's order with
//! the tier's slots in it), adds the shared expert and the host's sum: one
//! launch more than a one-card layer with card experts, whose sum is the
//! shadow's join.
//!
//! The handoff carries the boundary's eight slots through
//! `FfnKernels::enqueue_handoff`, which picks its entry by the slot count and
//! refuses by name a count it has no entry for. Neither post entry of the
//! V4.1 piece is called: the sum is one add, and `hc_post` is the
//! sub-layer's own.
//!
//! ik divides GLM's eight weights by their bare sum; the router divides by
//! the sum plus `1e-20` (`route_core::renorm_divisor`), under half an ulp of
//! every sum from 2^-42 up. The two agree bit for bit unless all eight
//! sigmoid scores sit below that: a named difference.
//!
//! [`BlockPlan::extra_launches`]: bloomery_gpu_deepseek41::block::BlockPlan::extra_launches

use bloomery_gpu::card_leg::{CardLeg, LegBufs, LegRun, LegShape, LegStacks};
use bloomery_gpu::host::leg_tier::{STAGE_TIER, TierSide};
use bloomery_gpu::host::run::HostRun;
use bloomery_gpu::hybrid::{Boundary, Hybrid, SlotMap};
use bloomery_gpu::kquant::{
    Act, EntryPlanes, KquantTileKernels, TileGather, TileScratch, TiledGateUpAct,
};
use bloomery_gpu::routed_block::wait_add;
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{FaultSink, Gpu, GpuError, Q8Act};
use bloomery_gpu_deepseek41::block::{Block, BlockIo, enqueue_block};
use bloomery_gpu_deepseek41::chain::ffn::{
    CardAcc, CardAccTier, CardJoin, DownRows, FfnBatchKernels, Handoff, Places, SharedJoin,
};
use bloomery_gpu_deepseek41::router::glm5next::{N_EXPERT, N_USED};
use bloomery_gpu_deepseek41::span::{span, span_mut};
use cuda_core::{CudaStream, DeviceBuffer};
use model::arch::glm5next::place::{TIER_CHUNK, card_leg_bytes, card_tile_bytes, card_tiles_run};
use model::arch::names::routed_stacks;
use model::placement::accepts::{self, Program};
use model::placement::kernels::{self, DownAct, DownEntry, DownForms, GateUpEntry};
use models::{Ffn, LayerSpec};

use crate::body::{Parts, f32t, f32v};
use crate::tensors::{FfnNames, LayerNames, other_kind};

/// The dense block's launches.
pub(crate) const DENSE_LAUNCHES: usize = 3;

/// A routed block's launches: the front's three kernels and the go, the
/// shadow's two, the wait and the sum.
pub(crate) const MOE_LAUNCHES: usize = 3 + 1 + 2 + 1 + 1;

/// The shadow's launches more on a layer with card experts: the norm's q8_1,
/// the gate·up, the q8_1 of its card columns, the down, and the card sum
/// joined with the shared expert's output.
pub(crate) const CARD_LAUNCHES: usize = 5;

/// A layer's card experts: how many its stacks hold, the routed SwiGLU limit
/// (`swiglu_clamp_exp`) and the stacks' names (gate, up, down), made once at
/// load.
#[derive(Debug)]
struct CardLayer {
    n_card: usize,
    limit: f32,
    names: [String; 3],
}

impl CardLayer {
    /// The stacks' names, as the card leg reads them.
    fn names(&self) -> [&str; 3] {
        self.names.each_ref().map(String::as_str)
    }
}

/// The card leg's shape for a row's one token through routed experts of `ff`
/// rows, `n_embd` wide.
const fn leg_shape(n_embd: usize, ff: usize) -> LegShape {
    LegShape {
        cap: 1,
        slots_per_col: N_USED,
        n: n_embd,
        ff,
    }
}

/// The card's routed experts of the step: the family's kernels, each layer's
/// count and limit from the slot map and the description, and each row's
/// one-token buffers every card layer's shadow writes (a verify's or a pass's
/// row reads its own card sum in its back, after the other rows' shadows). Made at load, after the weights,
/// from the stacks they hold; a layer the map puts experts on holds its three
/// stacks, each of a type the body's card leg arms run ([`card_stacks`]), at
/// `n_card` experts each, and a layer it puts none on holds none — anything else is
/// refused by name. The rows' leg buffers hold the forms the card layers'
/// downs read.
pub(crate) struct CardExperts {
    /// The card leg a step's card slots run on.
    leg: CardLeg,
    /// The grouped entries a prompt batch's card slots run on.
    tiles: KquantTileKernels,
    batch: FfnBatchKernels,
    /// Per layer of the description, `None` without card experts.
    layers: Vec<Option<CardLayer>>,
    keep_host: bool,
    /// The down forms of the card layers whose stacks are not
    /// [`card_tiles_run`]'s: a prompt batch runs those on the card leg, whose
    /// buffers [`CardTiles`] then holds in these forms.
    prompt_leg: DownForms,
    /// Routed expert width: a gate·up slot's rows.
    ff: usize,
    /// Each row's buffers, row 0 the step's: a row of the load's rows in
    /// flight.
    rows: Vec<CardRow>,
    /// The stage card's side of the tier layers, on a load with a tier card.
    tier: Option<TierSide>,
}

/// One row's buffers of the card experts' shadow.
struct CardRow {
    /// The card leg's buffers for one token: the norm's q8_1 form, the
    /// gate·up's rows, the card slots' q8_1 columns in each form the card
    /// layers' downs read and the downs, a slot's `n_embd` rows each.
    leg: LegBufs,
    /// The card slots' weighted sum, and it plus the shared expert's output.
    acc: DeviceBuffer<f32>,
    pre: DeviceBuffer<f32>,
}

impl CardRow {
    fn new(
        stream: &CudaStream,
        n_embd: usize,
        ff: usize,
        downs: DownForms,
    ) -> Result<CardRow, GpuError> {
        Ok(CardRow {
            leg: LegBufs::for_downs(stream, leg_shape(n_embd, ff), downs)?,
            acc: DeviceBuffer::zeroed(stream, n_embd)?,
            pre: DeviceBuffer::zeroed(stream, n_embd)?,
        })
    }

    fn bytes(&self) -> usize {
        self.leg.bytes() + self.acc.num_bytes() + self.pre.num_bytes()
    }
}

impl CardExperts {
    /// The card experts of `layers` (the description) as `map` places them
    /// and `w` holds them, `n_embd` wide with routed experts `ff` wide, a
    /// row's buffers for each of the load's `rows` in flight. Load-time
    /// only.
    pub(crate) fn new(
        gpu: &Gpu,
        w: &Weights,
        layers: &[LayerSpec],
        map: &SlotMap,
        n_embd: usize,
        ff: usize,
        (rows, keep_host): (usize, bool),
    ) -> Result<CardExperts, GpuError> {
        let mut per = Vec::with_capacity(layers.len());
        let (mut downs, mut prompt_leg) = (DownForms::NONE, DownForms::NONE);
        for (l, spec) in layers.iter().enumerate() {
            // A dense block is outside the host run and has no row: no card
            // experts, which the resident-rows check below holds it to. A
            // routed layer the map has no row for is refused by name.
            let n_card = match spec.ffn {
                Ffn::Dense { .. } => 0,
                Ffn::Moe(_) => map.on_card(l)?,
            };
            let names = routed_stacks(l);
            if n_card == 0 {
                if let Some(n) = names.iter().find(|n| w.get(n).is_some()) {
                    return Err(GpuError::Tensor {
                        what: CARD,
                        name: n.clone(),
                        need: "no resident rows: the slot map puts no expert of its layer on the card",
                    });
                }
                per.push(None);
                continue;
            }
            let Ffn::Moe(m) = &spec.ffn else {
                return Err(GpuError::Shape {
                    what: CARD,
                    detail: format!("layer {l}: card experts on a dense block"),
                });
            };
            let layer = CardLayer {
                n_card,
                limit: crate::body::limit_of(m.act)?,
                names,
            };
            // A prompt batch runs a layer the tiles do not on the card leg.
            let st = card_stacks(w, &layer)?;
            st.check(n_card, leg_shape(n_embd, ff))?;
            let form = down_form(&st)?;
            downs = downs.with(form);
            if !card_tiles_run(st.gate_up_ty(), st.down_ty()) {
                prompt_leg = prompt_leg.with(form);
            }
            per.push(Some(layer));
        }
        let stream = gpu.stream();
        Ok(CardExperts {
            leg: CardLeg::load(gpu)?,
            tiles: KquantTileKernels::load(gpu.context(), gpu.fault_word())?,
            batch: FfnBatchKernels::load(gpu.context())?,
            layers: per,
            keep_host,
            prompt_leg,
            ff,
            rows: (0..rows)
                .map(|_| CardRow::new(stream, n_embd, ff, downs))
                .collect::<Result<Vec<_>, _>>()?,
            tier: match map.tiers() {
                0 => None,
                _ => Some(TierSide::new(gpu, map, layers.len(), (rows, N_USED))?),
            },
        })
    }

    /// A tensor-parallel rank's card experts (`crate::tp`): every expert of
    /// every routed layer of `layers` on the card at its own id, `ff` the
    /// rank's part of the routed width, each layer's stacks of a type the
    /// common leg runs ([`LegStacks::of`], the common arms, a Q6_K down
    /// among them) at every expert, and the buffers of the one row a rank
    /// steps, in the forms its layers' downs read. No prompt batch and no tier
    /// card run on a rank. Load-time only.
    pub(crate) fn rank(
        gpu: &Gpu,
        w: &Weights,
        layers: &[LayerSpec],
        n_embd: usize,
        ff: usize,
    ) -> Result<CardExperts, GpuError> {
        let mut downs = DownForms::NONE;
        let mut per = Vec::with_capacity(layers.len());
        for (l, spec) in layers.iter().enumerate() {
            let Ffn::Moe(m) = &spec.ffn else {
                per.push(None);
                continue;
            };
            let layer = CardLayer {
                n_card: N_EXPERT,
                limit: crate::body::limit_of(m.act)?,
                names: routed_stacks(l),
            };
            let st = LegStacks::of(w, layer.names())?;
            st.check(N_EXPERT, leg_shape(n_embd, ff))?;
            downs = downs.with(down_form(&st)?);
            per.push(Some(layer));
        }
        let stream = gpu.stream();
        Ok(CardExperts {
            leg: CardLeg::load(gpu)?,
            tiles: KquantTileKernels::load(gpu.context(), gpu.fault_word())?,
            batch: FfnBatchKernels::load(gpu.context())?,
            layers: per,
            keep_host: false,
            prompt_leg: DownForms::NONE,
            ff,
            rows: vec![CardRow::new(stream, n_embd, ff, downs)?],
            tier: None,
        })
    }

    /// Layer `l` computes experts on the card.
    pub(crate) fn has(&self, l: usize) -> bool {
        self.layer(l).is_some()
    }

    /// A fully resident layer runs without the host leg unless the load keeps it: a residency
    /// machine can move its experts, or the host service writes a route trace.
    pub(crate) fn hostless(&self, l: usize) -> bool {
        !self.keep_host && self.n_card(l) == N_EXPERT
    }

    /// The experts of layer `l` the expert tier holds: 0 without a tier.
    pub(crate) fn tier_k(&self, l: usize) -> usize {
        self.tier.as_ref().map_or(0, |t| t.k(l))
    }

    /// Layer `l` has a card sum: card experts on the stage card or the tier.
    pub(crate) fn sums(&self, l: usize) -> bool {
        self.has(l) || self.tier_k(l) > 0
    }

    /// The stage card's side of the tier layers, on a load with a tier card.
    pub(crate) fn tier(&self) -> Option<&TierSide> {
        self.tier.as_ref()
    }

    /// The down forms of the card layers a prompt batch runs on the card
    /// leg ([`card_tiles_run`]); none when every card layer is a tile layer.
    pub(crate) fn prompt_leg(&self) -> DownForms {
        self.prompt_leg
    }

    /// The stage card's experts of layer `l`: 0 off the card.
    pub(crate) fn n_card(&self, l: usize) -> usize {
        self.layer(l).map_or(0, |c| c.n_card)
    }

    /// The batch kernels the card sums run on.
    pub(crate) fn batch(&self) -> &FfnBatchKernels {
        &self.batch
    }

    fn layer(&self, l: usize) -> Option<&CardLayer> {
        layer_of(&self.layers, l)
    }

    /// Row 0's card sum of the last card layer `ffn::card_sum` or a tier
    /// layer's back wrote.
    pub(crate) fn acc(&self) -> Result<&DeviceBuffer<f32>, GpuError> {
        Ok(&card_row(&self.rows, 0)?.acc)
    }

    /// The routed experts' width: a gate·up slot's rows.
    pub(crate) fn ff(&self) -> usize {
        self.ff
    }

    /// Device bytes of the buffers.
    pub(crate) fn bytes(&self) -> usize {
        self.rows.iter().map(CardRow::bytes).sum::<usize>()
            + self.tier.as_ref().map_or(0, TierSide::bytes)
    }
}

/// What the card experts' errors name.
const CARD: &str = "glm5next CardExperts";

/// Layer `l`'s card experts of `layers`: `None` without any.
fn layer_of(layers: &[Option<CardLayer>], l: usize) -> Option<&CardLayer> {
    layers.get(l).and_then(Option::as_ref)
}

/// Row `row`'s card buffers; a row past the load's is refused by name.
fn card_row(rows: &[CardRow], row: usize) -> Result<&CardRow, GpuError> {
    rows.get(row).ok_or_else(|| past_rows(rows.len(), row))
}

/// Row `row`'s card buffers, to write; a row past the load's is refused by
/// name.
fn card_row_mut(rows: &mut [CardRow], row: usize) -> Result<&mut CardRow, GpuError> {
    let n = rows.len();
    rows.get_mut(row).ok_or_else(|| past_rows(n, row))
}

fn past_rows(n: usize, row: usize) -> GpuError {
    GpuError::Shape {
        what: CARD,
        detail: format!("row {row} of the card experts' {n} rows"),
    }
}

/// Layer `cl`'s three resident routed stacks as the body's card leg reads
/// them ([`LegStacks::of_arms`] over [`accepts::leg_arms`] of the glm5next
/// body, the arms its plan rule reads): the file's blocks, or NVFP4's planes.
/// A stack of a type no arm runs is refused by name.
fn card_stacks<'w>(w: &'w Weights, cl: &'w CardLayer) -> Result<LegStacks<'w>, GpuError> {
    let arms = accepts::leg_arms(Program::Glm5nextBody).ok_or_else(|| GpuError::Shape {
        what: CARD,
        detail: format!("{:?} runs no card leg", Program::Glm5nextBody),
    })?;
    LegStacks::of_arms(
        w,
        cl.names(),
        arms,
        (
            "a resident routed gate or up stack of a type the card leg's gate·up arm runs \
             (`placement::accepts::leg_arms`)",
            "a resident routed down stack of a type the card leg's down arm runs \
             (`placement::accepts::leg_arms`)",
        ),
    )
}

/// The form `st`'s down entry reads ([`kernels::down`]); a down the table
/// names no entry for, which the arms refuse first, is refused by name.
fn down_form(st: &LegStacks<'_>) -> Result<DownAct, GpuError> {
    kernels::down(st.down_ty())
        .map(|(_, act)| act)
        .ok_or_else(|| GpuError::Shape {
            what: CARD,
            detail: format!("a down stack of {} with no entry", st.down_ty()),
        })
}

/// A routed layer's names, borrowed out of their [`FfnNames::Moe`].
pub(crate) struct MoeNames<'a> {
    pub norm: &'a str,
    pub router: &'a str,
    pub bias: &'a str,
    pub sh_gate: &'a str,
    pub sh_up: &'a str,
    pub sh_down: &'a str,
}

/// Layer `l`'s routed block's names; another kind's is refused by name.
pub(crate) fn moe_names<'a>(p: &Parts<'a>, l: usize) -> Result<MoeNames<'a>, GpuError> {
    let names: &'a [LayerNames] = p.names;
    match names.get(l).map(|n| &n.ffn) {
        Some(FfnNames::Moe {
            norm,
            router,
            bias,
            sh_gate,
            sh_up,
            sh_down,
        }) => Ok(MoeNames {
            norm,
            router,
            bias,
            sh_gate,
            sh_up,
            sh_down,
        }),
        _ => Err(other_kind("glm5next routed block", l)),
    }
}

/// The gate, up and down names of a layer's dense block or shared expert.
pub(crate) fn block_names(n: &LayerNames) -> [&str; 3] {
    match &n.ffn {
        FfnNames::Dense { gate, up, down, .. }
        | FfnNames::Moe {
            sh_gate: gate,
            sh_up: up,
            sh_down: down,
            ..
        } => [gate, up, down],
    }
}

/// Enqueue layer `l`'s dense block (module doc).
pub(crate) fn dense(gpu: &Gpu, w: &Weights, p: &mut Parts<'_>, l: usize) -> Result<(), GpuError> {
    let stream = gpu.stream();
    let (d, c) = (*p.d, p.cfg[l]);
    let Some(FfnNames::Dense {
        norm,
        gate,
        up,
        down,
    }) = p.names.get(l).map(|n| &n.ffn)
    else {
        return Err(other_kind("glm5next dense", l));
    };
    let s = &mut *p.s;
    gpu.elem().enqueue_rms_norm(
        stream,
        &s.x,
        f32v(w, norm)?,
        d.rms_eps,
        d.embd,
        1,
        &mut s.xn,
    )?;
    let kernels = p.k.block_kernels();
    let blk = Block {
        plan: c.block,
        names: [gate, up, down],
        ff: c.ff,
        limit: c.limit,
        layer: l,
    };
    let io = BlockIo {
        x: &s.xn,
        h: &mut s.h,
        y: &mut s.out,
    };
    enqueue_block(gpu, w, &kernels, &mut s.block, &blk, io)
}

/// Enqueue layer `l`'s routed block up to its go (module doc).
pub(crate) fn front(
    gpu: &Gpu,
    w: &Weights,
    p: &mut Parts<'_>,
    hybrid: &mut Hybrid<HostRun>,
    l: usize,
) -> Result<(), GpuError> {
    let stream = gpu.stream();
    let (d, c) = (*p.d, p.cfg[l]);
    let fault = gpu.layer_sink(l)?;
    let n = moe_names(p, l)?;
    let s = &mut *p.s;
    gpu.elem().enqueue_rms_norm(
        stream,
        &s.x,
        f32v(w, n.norm)?,
        d.rms_eps,
        d.embd,
        1,
        hybrid.boundary_mut().normed_mut(),
    )?;
    let bias = if c.bias { f32v(w, n.bias)? } else { &s.no_bias };
    p.k.router.enqueue_router(
        stream,
        f32t(w, n.router)?,
        hybrid.boundary().normed(),
        bias,
        d.scale,
        &mut s.rout,
        fault,
    )?;
    let h = Handoff {
        ids: &s.rout.ids,
        weights: &s.rout.weights,
        map: p.slots.buf(),
        row_off: c.row_off,
        n_expert: N_EXPERT,
    };
    if let Some(side) = p.card.tier.as_mut().filter(|t| t.k(l) > 0) {
        let (places, tsel) = side.handoff_parts(p.row)?;
        let targets = hybrid.tier_handoff(p.row, STAGE_TIER)?;
        p.k.ffn
            .enqueue_handoff_tier(stream, &h, places, targets, fault, (&mut s.sel, tsel))?;
        return hybrid.enqueue_tier_go(stream, l, p.row);
    }
    if p.card.hostless(l) {
        return p.card.batch.enqueue_places(
            stream,
            &Places {
                ids: h.ids,
                n: N_USED,
                map: h.map,
                row_off: h.row_off,
                n_expert: N_EXPERT,
            },
            fault,
            &mut s.sel,
        );
    }
    let target = hybrid.boundary_mut().handoff_target_of(p.row)?;
    p.k.ffn
        .enqueue_handoff(stream, &h, target, fault, &mut s.sel)?;
    hybrid.boundary().enqueue_go_of(stream, l, p.row)
}

/// Enqueue layer `l`'s card experts, when it has any, and its shared expert
/// in its host leg's shadow, after its [`front`]. On a layer the expert tier
/// holds experts of, the card sum waits for the tier's rows: it runs in the
/// [`back`].
pub(crate) fn shadow(
    gpu: &Gpu,
    w: &Weights,
    p: &mut Parts<'_>,
    boundary: &Boundary,
    l: usize,
) -> Result<(), GpuError> {
    let c = p.cfg[l];
    let n = p.d.embd;
    let card = p.card.has(l);
    let tiered = p.card.tier_k(l) > 0;
    if card {
        card_slots(gpu, w, p, boundary, l)?;
    }
    let sh = moe_names(p, l)?;
    let s = &mut *p.s;
    let kernels = p.k.block_kernels();
    let blk = Block {
        plan: c.block,
        names: [sh.sh_gate, sh.sh_up, sh.sh_down],
        ff: c.ff,
        limit: c.limit,
        layer: l,
    };
    let io = BlockIo {
        x: boundary.normed(),
        h: &mut s.h,
        y: &mut s.sh_y,
    };
    enqueue_block(gpu, w, &kernels, &mut s.block, &blk, io)?;
    if card && !tiered {
        let n_card = p.card.n_card(l);
        let CardExperts { batch, rows, .. } = &mut *p.card;
        let k = card_row_mut(rows, p.row)?;
        let slots = (k.leg.down(), &s.rout.weights, &s.sel);
        let sink = gpu.layer_sink(l)?;
        enqueue_join(
            batch,
            gpu.stream(),
            slots,
            (n, 1, n_card),
            (&s.sh_y, sink),
            &mut k.pre,
        )?;
    }
    Ok(())
}

/// The card sum of `m` tokens' slots — `down` the card slots' rows,
/// slot-major, `w` and `sel` their weights and places, [`N_USED`] a token —
/// joined with the shared expert's output `sh` into `y` in one launch,
/// `acc + sh` (`card_sum_join`): the shared expert takes no router weight,
/// and the host's sum, where the layer has one, is added after the wait.
pub(crate) fn enqueue_join(
    batch: &FfnBatchKernels,
    stream: &CudaStream,
    (down, w, sel): (&DeviceBuffer<f32>, &DeviceBuffer<f32>, &DeviceBuffer<u32>),
    (n, m, n_card): (usize, usize, usize),
    (sh, fault): (&DeviceBuffer<f32>, FaultSink),
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    let a = CardJoin {
        down: DownRows::F32(down),
        w,
        sel,
        n,
        m,
        n_card,
        n_used: N_USED,
        w_pitch: N_USED,
        shared: SharedJoin { sh, gate: None },
        fault,
    };
    batch.enqueue_card_join(stream, &a, y)
}

/// Layer `l`'s card sum alone into the row's `acc`, for a reader of the
/// sum beside the block's output (`forced_row`): the shadow's join writes the
/// output without it. A tier layer's [`back`] already wrote its sum there.
pub(crate) fn card_sum(gpu: &Gpu, p: &mut Parts<'_>, l: usize) -> Result<(), GpuError> {
    let n_card = p.card.n_card(l);
    let s = &*p.s;
    let CardExperts { batch, rows, .. } = &mut *p.card;
    let k = card_row_mut(rows, p.row)?;
    let a = CardAcc {
        down: k.leg.down(),
        w: &s.rout.weights,
        sel: &s.sel,
        n: k.acc.len(),
        m: 1,
        n_card,
        n_used: N_USED,
        w_pitch: N_USED,
        fault: gpu.layer_sink(l)?,
    };
    batch.enqueue_card_acc(gpu.stream(), &a, &mut k.acc)
}

/// Layer `l`'s card slots' down rows (module doc): the card leg over every
/// slot the handoff placed below the layer's `n_card` on the card, the host's
/// left alone, without the sum: the shadow joins it with the shared expert
/// ([`enqueue_join`]), and a tier layer's [`back`] reads the downs beside the
/// tier's rows.
fn card_slots(
    gpu: &Gpu,
    w: &Weights,
    p: &mut Parts<'_>,
    boundary: &Boundary,
    l: usize,
) -> Result<(), GpuError> {
    const W: &str = "glm5next card_slots";
    let s = &*p.s;
    let CardExperts {
        leg, layers, rows, ..
    } = &mut *p.card;
    let cl = layer_of(layers, l).ok_or(GpuError::State {
        what: W,
        missing: "card experts on the layer",
    })?;
    let k = card_row_mut(rows, p.row)?;
    let stacks = card_stacks(w, cl)?;
    let run = LegRun {
        layer: l,
        stacks: &stacks,
        normed: boundary.normed(),
        m: 1,
        sel: &s.sel,
        n_card: cl.n_card,
        rule: Act::SwigluClamp { limit: cl.limit },
    };
    leg.enqueue(gpu, &mut k.leg, &run, None)
}

/// Enqueue layer `l`'s routed block on a tensor-parallel rank (`crate::tp`),
/// whose card holds every expert at its own id and the rank's part of each
/// one's inner width: `ffn_norm` RMS, the router (as [`front`]'s), each slot's
/// place from the slot map (`ds41_ffn_places`, the handoff's rule), the card
/// leg over the eight slots, the shared expert at the rank's width, and the
/// slots' weighted sum joined with the shared expert's output into `out`
/// ([`enqueue_join`]): the rank's part of the block's output, which the
/// ranks' gather adds. With every
/// expert on the card these are the launches the one-card body's front, shadow
/// and back make but the handoff, the go and the wait, and the sum is theirs
/// with no host part added.
pub(crate) fn rank_moe(
    gpu: &Gpu,
    w: &Weights,
    p: &mut Parts<'_>,
    l: usize,
) -> Result<(), GpuError> {
    const W: &str = "glm5next rank_moe";
    let stream = gpu.stream();
    let (d, c) = (*p.d, p.cfg[l]);
    let fault = gpu.layer_sink(l)?;
    let n = moe_names(p, l)?;
    let s = &mut *p.s;
    gpu.elem().enqueue_rms_norm(
        stream,
        &s.x,
        f32v(w, n.norm)?,
        d.rms_eps,
        d.embd,
        1,
        &mut s.xn,
    )?;
    let bias = if c.bias { f32v(w, n.bias)? } else { &s.no_bias };
    p.k.router.enqueue_router(
        stream,
        f32t(w, n.router)?,
        &s.xn,
        bias,
        d.scale,
        &mut s.rout,
        fault,
    )?;
    let CardExperts {
        leg,
        batch,
        layers,
        rows,
        ..
    } = &mut *p.card;
    let cl = layer_of(layers, l).ok_or(GpuError::State {
        what: W,
        missing: "card experts on the layer",
    })?;
    batch.enqueue_places(
        stream,
        &Places {
            ids: &s.rout.ids,
            n: N_USED,
            map: p.slots.buf(),
            row_off: c.row_off,
            n_expert: N_EXPERT,
        },
        fault,
        &mut s.sel,
    )?;
    let k = card_row_mut(rows, p.row)?;
    let stacks = LegStacks::of(w, cl.names())?;
    let run = LegRun {
        layer: l,
        stacks: &stacks,
        normed: &s.xn,
        m: 1,
        sel: &s.sel,
        n_card: cl.n_card,
        rule: Act::SwigluClamp { limit: cl.limit },
    };
    leg.enqueue(gpu, &mut k.leg, &run, None)?;
    let kernels = p.k.block_kernels();
    let blk = Block {
        plan: c.block,
        names: [n.sh_gate, n.sh_up, n.sh_down],
        ff: c.ff,
        limit: c.limit,
        layer: l,
    };
    let io = BlockIo {
        x: &s.xn,
        h: &mut s.h,
        y: &mut s.sh_y,
    };
    enqueue_block(gpu, w, &kernels, &mut s.block, &blk, io)?;
    let slots = (k.leg.down(), &s.rout.weights, &s.sel);
    let shared = (&s.sh_y, fault);
    enqueue_join(
        batch,
        stream,
        slots,
        (d.embd, 1, cl.n_card),
        shared,
        &mut s.out,
    )
}

/// A prompt batch's card slots ([`card_rows`]): the batch's `t` normed rows
/// and their routing, the places the call writes and the grouped path's
/// scratch, whose downs the shadow's join ([`enqueue_join`]) or a tier
/// layer's back sums.
pub(crate) struct CardRows<'a> {
    pub normed: &'a DeviceBuffer<f32>,
    pub ids: &'a DeviceBuffer<u32>,
    pub t: usize,
    pub sel: &'a mut DeviceBuffer<u32>,
    pub tiles: &'a mut CardTiles,
}

/// The grouped path's scratch for prompt batches of up to `cap` tokens of
/// `n_embd` through routed experts of `ff` rows, [`N_USED`] slots a token:
/// the batch's rows in q8_1 (a column a token), the K-quant family's tile
/// scratch ([`TileScratch`]: the table — the card slots by expert and its
/// tiles — the gate·up rows by entry and their q8_1 form), the entries'
/// gathered Walk A planes, and the downs by slot, slot-major from the
/// batch's first slot (the tier layers' back reads them beside the tier's
/// rows); on a load with a card layer the tiles do not run
/// ([`card_tiles_run`]), the card leg's buffers for [`TIER_CHUNK`] tokens in
/// the forms those layers' downs read. Its bytes are [`card_tile_bytes`]'s
/// plus [`card_leg_bytes`]'s of those forms, which the plan reserves.
pub(crate) struct CardTiles {
    act_x: Q8Act,
    tile: TileScratch,
    ord: EntryPlanes,
    down: DeviceBuffer<f32>,
    leg: Option<LegBufs>,
}

impl CardTiles {
    /// The scratch for batches of up to `cap` tokens, with the card leg's
    /// buffers in the forms `leg` names when it names any, refused by name
    /// when its made bytes are not the plan's formula's. Load-time or
    /// first-prompt only.
    pub(crate) fn new(
        gpu: &Gpu,
        n_embd: usize,
        ff: usize,
        cap: usize,
        leg: DownForms,
    ) -> Result<CardTiles, GpuError> {
        let stream = gpu.stream();
        let slots = cap * N_USED;
        let t = CardTiles {
            act_x: Q8Act::with_tier_cols(stream, cap, n_embd)?,
            tile: TileScratch::new(stream, slots, ff)?,
            ord: EntryPlanes::new(stream, slots, n_embd)?,
            down: DeviceBuffer::zeroed(stream, slots * n_embd)?,
            leg: (leg != DownForms::NONE)
                .then(|| {
                    let shape = LegShape {
                        cap: TIER_CHUNK as usize,
                        slots_per_col: N_USED,
                        n: n_embd,
                        ff,
                    };
                    LegBufs::for_downs(stream, shape, leg)
                })
                .transpose()?,
        };
        let want =
            card_tile_bytes(n_embd, ff, N_USED, cap) + card_leg_bytes(n_embd, ff, N_USED, leg);
        if t.bytes() != want {
            return Err(GpuError::Shape {
                what: CARD,
                detail: format!(
                    "the grouped path's scratch for {cap} tokens made {} B; the plan's formula \
                     reserves {want} B",
                    t.bytes()
                ),
            });
        }
        Ok(t)
    }

    /// Device bytes of the scratch.
    pub(crate) fn bytes(&self) -> usize {
        self.act_x.device_bytes()
            + self.tile.device_bytes()
            + self.ord.device_bytes()
            + self.down.num_bytes()
            + self.leg.as_ref().map_or(0, LegBufs::bytes)
    }

    /// The downs of the last batch, by slot.
    pub(crate) fn down(&self) -> &DeviceBuffer<f32> {
        &self.down
    }
}

/// Layer `l`'s card slots over a prompt batch's `t` tokens into the
/// scratch's downs: every slot's place from the slot map in one launch
/// (`ds41_ffn_places`, the handoff's rule), then the card slots' downs by
/// slot — on a layer [`card_tiles_run`] names, by tile items
/// ([`card_tiles`]: each weight row read once for up to eight of an expert's
/// slots), on any other, on the card leg in runs of its buffers' tokens
/// ([`card_leg_rows`]). Each
/// slot and token is what the one-token launches write for it (the grouped
/// entries are the `_sel`s' bit for bit, the leg runs the `_sel`s), the
/// host's slots left alone. Refused by name on a layer without card experts.
pub(crate) fn card_rows(
    gpu: &Gpu,
    w: &Weights,
    p: &mut Parts<'_>,
    l: usize,
    io: CardRows<'_>,
) -> Result<(), GpuError> {
    const W: &str = "glm5next card_rows";
    let stream = gpu.stream();
    let n = p.d.embd;
    let fault = gpu.layer_sink(l)?;
    let c = p.cfg[l];
    let k = &*p.card;
    let cl = k.layer(l).ok_or(GpuError::State {
        what: W,
        missing: "card experts on the layer",
    })?;
    let st = card_stacks(w, cl)?;
    let t = io.t;
    let slots = t * N_USED;
    k.batch.enqueue_places(
        stream,
        &Places {
            ids: io.ids,
            n: slots,
            map: p.slots.buf(),
            row_off: c.row_off,
            n_expert: N_EXPERT,
        },
        fault,
        &mut *io.sel,
    )?;
    let sel: &DeviceBuffer<u32> = &*io.sel;
    let tl = &mut *io.tiles;
    if card_tiles_run(st.gate_up_ty(), st.down_ty()) {
        card_tiles(gpu, k, cl, &st, l, n, io.normed, sel, t, tl)?;
    } else {
        card_leg_rows(gpu, k, cl, &st, l, n, io.normed, sel, t, tl)?;
    }
    Ok(())
}

/// [`card_rows`]'s card slots on the grouped tile kernels (a layer
/// [`card_tiles_run`] names): the rows' q8_1 form, the table by expert
/// (`grouped_buckets`) and its tiles (`grouped_tiles`,
/// [`TileScratch::enqueue_table`]), the entries' planes gathered
/// (`kq_card_gather`), the gate·up by tiles, the q8_1 of its rows by entry
/// (`q8_1_quantize_ord`) and the down by tiles, which scatters each entry's
/// rows to its slot in `tl`'s downs.
#[allow(clippy::too_many_arguments)]
fn card_tiles(
    gpu: &Gpu,
    k: &CardExperts,
    cl: &CardLayer,
    st: &LegStacks<'_>,
    l: usize,
    n: usize,
    normed: &DeviceBuffer<f32>,
    sel: &DeviceBuffer<u32>,
    t: usize,
    tl: &mut CardTiles,
) -> Result<(), GpuError> {
    let stream = gpu.stream();
    let fault = gpu.layer_sink(l)?;
    let slots = t * N_USED;
    gpu.enqueue_quantize_q8_1_cols(normed, &mut tl.act_x, t, l)?;
    tl.tile.enqueue_table(gpu, sel, slots, cl.n_card, fault)?;
    let g = TileGather {
        x: &tl.act_x,
        cols: t,
        slots_per_col: N_USED,
        table: tl.tile.table(slots, cl.n_card),
    };
    k.tiles
        .enqueue_card_gather(stream, &g, fault, &mut tl.ord)?;
    let [wg, wu, wd] = st.words()?;
    match kernels::gate_up(st.gate_up_ty()) {
        Some(GateUpEntry::Q4k | GateUpEntry::Q5k) => {
            let (table, h) = tl.tile.table_and_h(slots, cl.n_card);
            let a = TiledGateUpAct {
                wg,
                wu,
                act: &tl.ord,
                table,
                rows_per_expert: k.ff,
                rule: Act::SwigluClamp { limit: cl.limit },
            };
            k.tiles
                .enqueue_gate_up_tiles(stream, st.gate_up_ty(), &a, fault, h)?;
        }
        _ => return Err(untiled(&cl.names[0])),
    }
    tl.tile.enqueue_quantize_h(gpu, slots, cl.n_card, fault)?;
    let d = tl.tile.down(wd, n, slots, cl.n_card);
    match kernels::down(st.down_ty()) {
        Some((DownEntry::Q5k, DownAct::Q8Act)) => {
            k.tiles
                .enqueue_gemv_q5k_tiles(stream, &d, fault, &mut tl.down)?
        }
        Some((DownEntry::Q4k, DownAct::Q8Act)) => {
            gpu.q4k_sel()
                .enqueue_gemv_q4k_tiles(stream, &d, fault, &mut tl.down)?
        }
        _ => return Err(untiled(&cl.names[2])),
    }
    Ok(())
}

/// [`card_rows`]'s card slots on the card leg (a layer [`card_tiles_run`]
/// does not name): the batch's `t` columns in runs of the leg buffers' `cap`
/// tokens ([`CardLeg::enqueue_rows`]), each run's downs written to `tl`'s
/// downs by slot at its first slot, where the card sums and the tier's back
/// read them. Refused by name on a load that made no leg buffers.
#[allow(clippy::too_many_arguments)]
fn card_leg_rows(
    gpu: &Gpu,
    k: &CardExperts,
    cl: &CardLayer,
    stacks: &LegStacks<'_>,
    l: usize,
    n: usize,
    normed: &DeviceBuffer<f32>,
    sel: &DeviceBuffer<u32>,
    t: usize,
    tl: &mut CardTiles,
) -> Result<(), GpuError> {
    const W: &str = "glm5next card_leg_rows";
    let CardTiles { leg, down, .. } = tl;
    let bufs = leg.as_mut().ok_or(GpuError::State {
        what: W,
        missing: "the card leg's prompt buffers (a card layer the tiles do not run)",
    })?;
    let chunk = bufs.shape().cap;
    for c0 in (0..t).step_by(chunk) {
        let cn = chunk.min(t - c0);
        let slots = cn * N_USED;
        let x = span(W, normed, c0 * n, cn * n)?;
        let s = span(W, sel, c0 * N_USED, slots)?;
        let mut d = span_mut(W, &mut *down, c0 * N_USED * n, slots * n)?;
        let run = LegRun {
            layer: l,
            stacks,
            normed: &x,
            m: cn,
            sel: &s,
            n_card: cl.n_card,
            rule: Act::SwigluClamp { limit: cl.limit },
        };
        k.leg.enqueue_rows(gpu, bufs, &run, &mut d)?;
    }
    Ok(())
}

/// A stack `name` of a type the card leg runs but the grouped tiles have no
/// entry for, on a layer [`card_tiles_run`] names.
fn untiled(name: &str) -> GpuError {
    GpuError::Tensor {
        what: CARD,
        name: name.to_owned(),
        need: "a type the grouped tiles have an entry for (gate·up Q4_K, Q5_K; down Q5_K, Q4_K)",
    }
}

/// Enqueue layer `l`'s wait and the host's routed sum plus its shadow's —
/// the card sum and the shared expert's output, or the latter alone — into
/// `out`, after its [`shadow`]. On a layer the expert tier holds experts of,
/// the wait is the host's and the tier's ([`Hybrid::enqueue_tier_back`]),
/// then the card sum over the stage card's and the tier's slots in slot
/// order (`ds41_ffn_card_acc_8_tier`) and the shared expert's output added
/// to it: the shadow's two launches of a one-card layer, after the wait.
pub(crate) fn back(
    gpu: &Gpu,
    p: &mut Parts<'_>,
    hybrid: &Hybrid<HostRun>,
    l: usize,
) -> Result<(), GpuError> {
    let n_tier = p.card.tier_k(l);
    if n_tier > 0 {
        return back_tier(gpu, p, hybrid, l, n_tier);
    }
    let n = p.d.embd;
    let s = &mut *p.s;
    if p.card.hostless(l) {
        return s
            .out
            .copy_from_device_async(&card_row(&p.card.rows, p.row)?.pre, gpu.stream())
            .map_err(Into::into);
    }
    let shadowed = if p.card.has(l) {
        &card_row(&p.card.rows, p.row)?.pre
    } else {
        &s.sh_y
    };
    wait_add(gpu, hybrid, (l, p.row), (shadowed, n), &mut s.out)
}

/// [`back`] of a layer whose `n_tier` experts the expert tier holds.
fn back_tier(
    gpu: &Gpu,
    p: &mut Parts<'_>,
    hybrid: &Hybrid<HostRun>,
    l: usize,
    n_tier: usize,
) -> Result<(), GpuError> {
    const W: &str = "glm5next back (tier layer)";
    let stream = gpu.stream();
    let (n, row) = (p.d.embd, p.row);
    let n_card = p.card.n_card(l);
    hybrid.enqueue_tier_back(stream, l, row)?;
    let s = &mut *p.s;
    let CardExperts {
        batch, rows, tier, ..
    } = &mut *p.card;
    let tsel = tier
        .as_ref()
        .ok_or(GpuError::State {
            what: W,
            missing: "the stage card's tier side",
        })?
        .tsel(row)?;
    let k = card_row_mut(rows, row)?;
    let acc = CardAccTier {
        down: k.leg.down(),
        trows: hybrid.tier_rows(row)?,
        w: &s.rout.weights,
        sel: &s.sel,
        tsel,
        n,
        m: 1,
        n_card,
        n_tier,
        n_used: N_USED,
    };
    batch.enqueue_card_acc_tier(stream, &acc, &mut k.acc)?;
    gpu.elem()
        .enqueue_add(stream, &k.acc, &s.sh_y, n, &mut k.pre)?;
    gpu.elem().enqueue_add(
        stream,
        hybrid.boundary().hsum_of(row)?,
        &k.pre,
        n,
        &mut s.out,
    )
}
