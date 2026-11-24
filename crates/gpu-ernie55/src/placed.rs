//! An ERNIE 5.5 load by a placement plan: the stage card holds every layer's dense part, the
//! parts' caches and each routed layer's card share of its experts; the common host tier
//! ([`bloomery_gpu::host::placed`]) serves the rest of each layer's picks over the latent's width
//! (`moe_latent_size`), the routed experts' input. A plan whose card holds every expert makes no
//! host tier and the step runs every layer whole ([`crate::step`]).
//!
//! A layer with a host share runs its feed-forward block in three parts around the host tier's
//! step port, in the host leg's order:
//!
//! - **front**: the attention sub-layer, the feed-forward pre and norm, the router and the latent
//!   down into the boundary's activation ([`RoutedFfn::enqueue_front`]), the handoff of the ids,
//!   the weights and that row with each slot's place in the card's stacks, and the go;
//! - **shadow**, under the host's work: the shared expert and the card's slots over the latent
//!   row at the side's places ([`RoutedFfn::enqueue_shadow`]);
//! - **back**: the wait, the card's and the host's sums added before the latent norm (the routed
//!   sum is linear in the experts' rows), the latent up and the shared expert
//!   ([`RoutedFfn::enqueue_back`]), then the post.
//!
//! [`RoutedFfn::enqueue_front`]: crate::body::RoutedFfn::enqueue_front
//! [`RoutedFfn::enqueue_shadow`]: crate::body::RoutedFfn::enqueue_shadow
//! [`RoutedFfn::enqueue_back`]: crate::body::RoutedFfn::enqueue_back

use std::sync::Arc;

use bloomery_gpu::folder::Model;
use bloomery_gpu::host::StepLeg;
use bloomery_gpu::host::placed::{PlacedHost, PlacedSide, SideShape};
use bloomery_gpu::host::run::{HostRun, HostWidths};
use bloomery_gpu::host::shell::TierShell;
use bloomery_gpu::host::swap::Residency;
use bloomery_gpu::host::swap_source::{ResidencySpec, hosted_stacks};
use bloomery_gpu::hybrid::{Hybrid, SlotMap};
use bloomery_gpu::model::{HostServed, hosted_hybrid_at, refuse_reserve_past, stage_run};
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{Gpu, GpuError, GpuModel};
use bloomery_levers::HostCfg;
use model::arch::ernie55::names::Sub;
use model::arch::ernie55::place::{PROMPT_ROWS, PlanInputs};
use model::placement::{KvBytes, Machine, Plan, PlanLevers};
use runtime::layer::hosted;
use runtime::sched::{self, At, LayerProgram, Overlap, PortKind};

use crate::body::{Body, Cx, Run, tap_picks};
use crate::step;

/// What the placed load's refusals name.
const PLACED: &str = "ernie55 placed load";

/// The step's walk: one row through the host tier's step port.
const STEP_WALK: Overlap = Overlap {
    units: 1,
    cols: 1,
    port: PortKind::Step,
};

/// The rows in flight the host boundary holds: the step's one.
pub(crate) const LOAD_ROWS: usize = 1;

/// The rows a prompt unit of the pass asks the parts for: one HCA compression block.
pub const PASS_ROWS: usize = 128;

impl Body {
    /// The plan of `inputs`' file on `machine` at `ctx` positions under `levers`: the common hybrid
    /// planner's ([`PlanInputs`]), every layer's dense part and the stores on the stage card, each
    /// routed layer's card share of its experts by the common card leg's rule and the rest on the
    /// host. Refused by name: a context past the trained one, a placement that does not hold.
    pub fn plan<'a>(
        inputs: &'a PlanInputs,
        machine: &'a Machine,
        ctx: u64,
        levers: &PlanLevers,
    ) -> Result<Plan<'a>, GpuError> {
        inputs
            .check_ctx(ctx)
            .map_err(|e| GpuError::plan(PLACED, e))?;
        inputs
            .plan(machine, ctx, levers)
            .map_err(|e| GpuError::plan(PLACED, e))
    }

    /// Whether `plan` hosts experts of any routed layer of `inputs`' file: a load of it has a
    /// host tier and steps its prompt.
    #[must_use]
    pub fn plan_hosts(plan: &Plan<'_>, inputs: &PlanInputs) -> bool {
        let n_expert = inputs.hp.experts.n_expert as u64;
        (0..inputs.hp.n_layer)
            .filter(|&l| !inputs.hp.dense(l))
            .any(|l| plan.n_l.get(l).is_none_or(|&n| n < n_expert))
    }

    /// Card `card` of `plan`, which `inputs` made from `file`'s header, resident: every layer's
    /// dense part, each routed layer's card share of its stacks and the parts' caches at the
    /// plan's context, the host tier over the layers the plan hosts experts of; a prompt pass of
    /// `rows` rows a unit when `rows` is `Some` ([`crate::pass`]). Under `residency` the common
    /// machine runs over the card's routed stacks. A folder's card tensors are read through its
    /// part source ([`Model::Folder`]). Refused by name: a plan of another layer count or of more
    /// than one stage card ([`stage_run`]), an expert tier card, a routed run with a gap, a
    /// layer whose card stacks are not its card count's, parts whose caches are not the plan's
    /// bytes ([`crate::rooms::check`]).
    pub fn open_placed(
        file: impl Into<Model>,
        (plan, inputs): (&Plan<'_>, &PlanInputs),
        card: usize,
        host: HostCfg,
        (rows, residency): (Option<usize>, Residency),
    ) -> Result<GpuModel<Body>, GpuError> {
        let hp = &inputs.hp;
        let (file, origin) = file.into().into_origin(residency, plan)?;
        let (layers, ctx) = stage_run(PLACED, plan, card, hp.n_layer)?;
        if let Some(tier) = plan.machine.expert_tiers().next() {
            return Err(GpuError::shape(
                PLACED,
                format!(
                    "card {} is an expert tier card: the body serves its host share from the \
                     host tier alone",
                    tier.name
                ),
            ));
        }
        let run = hosted(&inputs.spec.layers)
            .map_err(|e| GpuError::shape(PLACED, format!("the routed layers: {e}")))?;
        let spec = ResidencySpec::hosted(
            residency,
            hp.experts.n_used,
            LOAD_ROWS,
            Arc::new(hosted_stacks(&inputs.model, PLACED, (run.len(), true))?),
        );
        if let Some(unit) = rows.filter(|&n| n == 0 || n > PROMPT_ROWS) {
            return Err(GpuError::shape(
                PLACED,
                format!(
                    "a prompt pass of {unit} rows a unit: the plan reserves buffers for units of \
                     1..={PROMPT_ROWS} rows"
                ),
            ));
        }
        let planned: u64 = layers.map(|l| inputs.kv.layer_bytes(l, plan.ctx_max)).sum();
        let reserve = inputs
            .prompt_batch_bytes(plan.machine)
            .map_err(|e| GpuError::plan(PLACED, e))?
            .saturating_add(inputs.kv.stage_position_bytes(plan.ctx_max));
        GpuModel::load_placed_from(
            file,
            origin.bytes(),
            plan,
            card,
            host,
            spec,
            (&[], 0),
            |stream, file, layers, w| crate::parts::derive(stream, file, hp, layers, w),
            |gpu, file, w, set, glue| {
                let unit = rows.unwrap_or(1);
                let parts = crate::parts::load(gpu, file, w, hp, (ctx, unit))?;
                crate::rooms::check(&parts, hp, planned)?;
                refuse_reserve_past(
                    PLACED,
                    "prompt pass buffers",
                    reserve,
                    crate::rooms::beside_stores(&parts)?,
                    "the walk's streams and the dense lead's blocks, the mixers' chains, flash partials, \
                     staging, words and score rows, the routed block's rows, card leg and forms, the \
                     head's row",
                )?;
                let mut body = Body::new(gpu, hp, ctx, parts)?;
                let map = SlotMap::of_plan(plan, card, None, run.clone(), hp.experts.n_expert)?;
                let shape = SideShape {
                    n_expert: hp.experts.n_expert,
                    n_used: hp.experts.n_used,
                    width: hp.latent,
                };
                let moe = &body.parts.moe;
                let mut side =
                    PlacedSide::new(gpu, &map, (hp.n_layer, shape), PLACED, |l, on_card| {
                        if hp.dense(l) {
                            return Err(GpuError::shape(
                                PLACED,
                                format!("layer {l}: experts on the card of the dense layer"),
                            ));
                        }
                        moe.check_card_stacks(w, l, on_card)
                    })?;
                let hosted_set = side.hosted_set();
                if !hosted_set.is_empty() && rows.is_some() {
                    return Err(GpuError::shape(
                        PLACED,
                        format!(
                            "a prompt pass over a plan that hosts experts of {} layers: the \
                             routed block's host share over a unit's rows is not built, so the \
                             load steps its prompt (open it with no pass)",
                            hosted_set.len()
                        ),
                    ));
                }
                if !hosted_set.is_empty() {
                    // Ten slots a token: no per-count handoff entry is built for it.
                    side.sink_any_count(gpu.stream())?;
                    let hybrid = hosted_hybrid_at(
                        (gpu, file, plan),
                        (&hosted_set, hp.n_layer, map),
                        (host, set, &glue),
                        (
                            HostWidths {
                                embd: hp.latent,
                                ff: hp.experts.ff,
                                n_used: hp.experts.n_used,
                            },
                            LOAD_ROWS,
                        ),
                        None,
                        |src| hosted_set.iter().map(|&l| moe.host_layer(src, l)).collect(),
                    )?;
                    let mut placed = PlacedHost::new(
                        gpu,
                        TierShell::new(hybrid, glue, hp.n_embd, PLACED),
                        side,
                    )?;
                    placed.prepare_step_union()?;
                    placed.start_residency(gpu)?;
                    body.placed = Some(placed);
                } else if residency != Residency::Off {
                    return Err(GpuError::shape(
                        PLACED,
                        "a residency machine over a plan whose card holds every expert: the \
                         machine moves experts between the host tier and the card",
                    ));
                }
                if rows.is_some() {
                    body.pass = Some(body.parts.rows());
                }
                Ok(body)
            },
        )
    }

    /// The rows a prompt unit of the pass takes; `None` on a load that steps its prompt.
    #[must_use]
    pub fn pass_rows(&self) -> Option<usize> {
        self.pass
    }
}

/// A hosted layer's front from the feed-forward norm on: the router and the latent down into the
/// boundary's activation, the handoff and the go.
fn front(
    cx: Cx<'_>,
    body: &mut Body,
    side: &mut PlacedSide,
    hy: &mut Hybrid<HostRun>,
    l: usize,
) -> Result<(), GpuError> {
    {
        let Body { parts, .. } = &mut *body;
        let x = parts.trunk.normed();
        parts
            .moe
            .enqueue_front(cx, l, x, hy.boundary_mut().normed_mut())?;
    }
    tap_picks(cx.gpu, body, l, 1)?;
    side.go(cx.gpu, body.parts.moe.route(), hy, l)
}

/// A hosted layer's card work under the host's: the shared expert and the card's slots over the
/// latent row the handoff left in the boundary. Whether the card's slots ran.
fn shadow(
    cx: Cx<'_>,
    body: &mut Body,
    side: &PlacedSide,
    hy: &Hybrid<HostRun>,
    l: usize,
) -> Result<bool, GpuError> {
    let Body { parts, .. } = body;
    let x = parts.trunk.normed();
    parts
        .moe
        .enqueue_shadow(cx, l, x, hy.boundary().normed(), side.sel(), side.n_card(l))
}

/// A hosted layer's back from the wait on: the card's and the host's sums through the block's
/// tail into `s.y`, then the post.
fn back(
    (cx, card): (Cx<'_>, bool),
    body: &mut Body,
    hy: &mut Hybrid<HostRun>,
    l: usize,
) -> Result<(), GpuError> {
    {
        let Body { parts, y, .. } = &mut *body;
        hy.boundary().enqueue_back_of(cx.gpu.stream(), 0, l)?;
        let hsum = hy.boundary().hsum_of(0)?;
        parts.moe.enqueue_back(cx, l, (card, Some(hsum)), y)?;
    }
    step::close(cx.gpu, cx.w, body, l, 1)?;
    hy.row_enqueued(l, 0)
}

/// The captured decode step's walk `(1, 1, Step)` through the host tier's step port.
struct StepWalk<'a> {
    cx: Cx<'a>,
    body: &'a mut Body,
    side: &'a mut PlacedSide,
    pos: u32,
    /// Whether the hosted layer's card slots ran, from its shadow to its back.
    card: bool,
}

/// Walk every layer of `body` at `pos` through `placed`'s step port.
pub(crate) fn walk(
    (gpu, w): (&Gpu, &Weights),
    body: &mut Body,
    placed: &mut PlacedHost,
    pos: u32,
) -> Result<(), GpuError> {
    let PlacedHost { shell, side, .. } = placed;
    let mut leg = StepLeg::new(gpu.stream(), &mut shell.hybrid);
    let n = body.sh.n_layer;
    let mut prog = StepWalk {
        cx: Cx { gpu, w },
        body,
        side,
        pos,
        card: false,
    };
    sched::walk(STEP_WALK, n, &mut leg, &mut prog)
}

impl<'a> LayerProgram for StepWalk<'a> {
    type Port = StepLeg<'a, HostRun>;

    fn host_leg(&self, at: At) -> bool {
        self.side.hosted(at.layer)
    }

    /// A layer with no host share whole; a hosted one up to its go.
    fn front(&mut self, leg: &mut StepLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let (l, cx) = (at.layer, self.cx);
        step::attn(cx.gpu, cx.w, self.body, l, Run::Step(self.pos))?;
        if !self.side.hosted(l) {
            step::ffn_whole(cx.gpu, cx.w, self.body, l, 1)?;
            return step::close(cx.gpu, cx.w, self.body, l, 1);
        }
        self.body.parts.trunk.enqueue_pre(cx, l, Sub::Ffn)?;
        front(cx, self.body, self.side, leg.hybrid(), l)
    }

    /// The shared expert and the card's slots of a hosted layer, under the host's work.
    fn shadow(&mut self, leg: &mut StepLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let l = at.layer;
        if !self.side.hosted(l) {
            return Ok(());
        }
        self.card = shadow(self.cx, self.body, self.side, leg.hybrid(), l)?;
        Ok(())
    }

    /// A hosted layer from the wait on, then its post.
    fn back(&mut self, leg: &mut StepLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let l = at.layer;
        if !self.side.hosted(l) {
            return Ok(());
        }
        back((self.cx, self.card), self.body, leg.hybrid(), l)
    }
}
