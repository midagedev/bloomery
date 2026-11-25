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
//! A prompt unit of the pass runs the same three parts around the host tier's batch port
//! ([`walk_pass`], the common schedule's walk `(1, m, Batch)`): the front ends in the download of
//! the unit's latent rows, ids and weights; the shadow is the shared expert, the unit's pick under
//! host streaming (the host experts the unit routes most move into the card's pool,
//! [`crate::stream`]), the places of the unit's slots over the map the pick left and the card's
//! slots at those places, under the tier's one union call, which reads each distinct host expert
//! once for every row that picked it and leaves out the experts the pick moved; the back adds the
//! uploaded sums row by row. The buffers the batch leg needs beside the placed side are
//! [`HostedPass`]'s, counted by the plan's prompt reserve.
//!
//! [`RoutedFfn::enqueue_front`]: crate::body::RoutedFfn::enqueue_front
//! [`RoutedFfn::enqueue_shadow`]: crate::body::RoutedFfn::enqueue_shadow
//! [`RoutedFfn::enqueue_back`]: crate::body::RoutedFfn::enqueue_back

use std::sync::Arc;

use bloomery_gpu::folder::Model;
use bloomery_gpu::host::placed::{PlacedHost, PlacedSide, SideShape};
use bloomery_gpu::host::run::{HostRun, HostWidths};
use bloomery_gpu::host::shell::TierShell;
use bloomery_gpu::host::swap::Residency;
use bloomery_gpu::host::swap_source::{ResidencySpec, hosted_stacks};
use bloomery_gpu::host::{BatchLeg, StepLeg};
use bloomery_gpu::hybrid::{Hybrid, SlotMap};
use bloomery_gpu::model::{HostServed, hosted_hybrid_at, refuse_reserve_past, stage_run};
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{Gpu, GpuError, GpuModel};
use bloomery_levers::HostCfg;
use cuda_core::{CudaStream, DeviceBuffer};
use model::arch::ernie55::names::Sub;
use model::arch::ernie55::place::{PROMPT_ROWS, PlanInputs};
use model::placement::{KvBytes, Machine, Plan, PlanLevers};
use runtime::layer::hosted;
use runtime::sched::{self, At, LayerProgram, Overlap, PortKind};

use crate::body::{Body, Cx, Run, tap_picks};
use crate::step;

/// What the placed load's refusals name.
const PLACED: &str = "ernie55 placed load";

/// What the pass's walk's refusals name.
const PASS: &str = "ernie55 placed pass";

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

/// The hosted prompt pass's buffers beside the placed side: the host's sums of a unit's rows (the
/// latent's width a row, where the batch leg's upload lands them) and the places of a unit's
/// routed slots in the card's stacks, or the host mark. The plan's prompt reserve counts them
/// (`model::arch::ernie55::place::PassTerms::placed`).
pub(crate) struct HostedPass {
    hsum: DeviceBuffer<f32>,
    sel: DeviceBuffer<u32>,
}

impl HostedPass {
    /// The values of the two buffers of units of up to `rows` rows of `latent` values routing
    /// `n_used` slots a row: the host's sums, then the places.
    fn lens_at((rows, latent, n_used): (usize, usize, usize)) -> (usize, usize) {
        (rows * latent, rows * n_used)
    }

    /// The device bytes [`HostedPass::lens_at`] makes, four bytes a value: the plan's reserve's
    /// term for the buffers (`model::arch::ernie55::place::PassTerms::placed`), which a test holds
    /// to it.
    #[cfg(test)]
    pub(crate) fn bytes_at(shape: (usize, usize, usize)) -> usize {
        let (hsum, sel) = Self::lens_at(shape);
        4 * (hsum + sel)
    }

    /// The buffers for units of up to `rows` rows of `latent` values routing `n_used` slots a
    /// row. Load-time only.
    fn new(stream: &CudaStream, shape: (usize, usize, usize)) -> Result<HostedPass, GpuError> {
        let (hsum, sel) = Self::lens_at(shape);
        Ok(HostedPass {
            hsum: DeviceBuffer::zeroed(stream, hsum)?,
            sel: DeviceBuffer::zeroed(stream, sel)?,
        })
    }

    /// The device bytes the buffers hold: the plan's reserve's term for them.
    pub(crate) fn bytes(&self) -> usize {
        self.hsum.num_bytes() + self.sel.num_bytes()
    }
}

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
    /// host tier, which the step serves through its step port and the prompt pass through its
    /// batch port.
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
    /// `rows` rows a unit when `rows` is `Some` ([`crate::pass`]), over the host tier's batch
    /// port where the plan hosts experts. Under `residency` the common machine runs over the
    /// card's routed stacks. A folder's card tensors are read through its part source
    /// ([`Model::Folder`]). Refused by name: a plan of another layer count or of more than one
    /// stage card ([`stage_run`]), an expert tier card, a routed run with a gap, a unit of no
    /// rows or of more than [`PROMPT_ROWS`], a layer whose card stacks are not its card count's,
    /// parts whose caches are not the plan's bytes ([`crate::rooms::check`]) or whose buffers
    /// with the hosted pass's pass the plan's prompt reserve.
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
                let hosted = match rows {
                    Some(rows) if !hosted_set.is_empty() => Some(HostedPass::new(
                        gpu.stream(),
                        (rows, hp.latent, hp.experts.n_used),
                    )?),
                    _ => None,
                };
                refuse_reserve_past(
                    PLACED,
                    "prompt pass buffers",
                    reserve,
                    crate::rooms::beside_stores(&body.parts)?
                        + hosted.as_ref().map_or(0, HostedPass::bytes) as u64,
                    "the walk's streams and the dense lead's blocks, the mixers' chains, flash \
                     partials, staging, words and score rows, the routed block's rows, card leg and \
                     forms, the head's row, the hosted pass's host sums and the slots' places",
                )?;
                if !hosted_set.is_empty() {
                    // Ten slots a token have a handoff entry of their own (`HandoffSlots::Ten`), so this builds no sink;
                    // a count with no entry gets the any-count tier sink.
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
                    if let Some(rows) = rows {
                        // The batch sets and union slabs for the pass's rows, which also serve the
                        // step's one column.
                        placed.prepare_pass_rows(gpu, rows)?;
                    } else {
                        placed.prepare_step_union()?;
                    }
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
                body.hosted = hosted;
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

/// A hosted layer's front from the feed-forward norm on over a unit's `m` rows: the router and
/// the latent down into the buffers' latent rows, the picks tapped and the download of the latent
/// rows, weights and ids ([`PlacedSide::download_rows`]).
fn front_rows(
    cx: Cx<'_>,
    body: &mut Body,
    side: &PlacedSide,
    m: usize,
    port: &mut BatchLeg<'_, HostRun>,
    l: usize,
) -> Result<(), GpuError> {
    {
        let Body { parts, .. } = &mut *body;
        let x = parts.trunk.normed();
        parts.moe.enqueue_front_rows(cx, l, m, x)?;
    }
    tap_picks(cx.gpu, body, l, m)?;
    let moe = &body.parts.moe;
    let (ids, weights) = moe.route();
    side.download_rows(
        port,
        At { unit: 0, layer: l },
        (moe.latent_rows(), weights, ids),
        &[],
    )
}

/// Whether the stream's pick covers layer `l`: a hosted layer whose card holds experts.
fn pick_covers(side: &PlacedSide, l: usize) -> bool {
    side.hosted(l) && side.n_card(l) > 0
}

/// A hosted layer's card work under the host's union, over a unit's `m` rows of a walk ending at
/// position `walk_end`: the shared expert over the rows, which reads no places; the layer's pick
/// under host streaming ([`PromptStream::unit_pick`]), which moves host experts into the card's
/// pool; the places of the unit's slots over the map the pick left ([`PlacedSide::places_rows`]);
/// the card's slots at those places, each behind its landing batch's event; then the unit's read of
/// the layer ([`PromptStream::unit_read`]). The pick sits before the places because the places
/// name the slots it moved. Whether the card's slots ran. Refused by name: a ring the stream gave
/// the layer, which the body never starts.
///
/// [`PromptStream::unit_pick`]: bloomery_gpu::host::prompt_stream::PromptStream::unit_pick
/// [`PromptStream::unit_read`]: bloomery_gpu::host::prompt_stream::PromptStream::unit_read
fn shadow_rows(
    cx: Cx<'_>,
    body: &mut Body,
    (side, port): (&PlacedSide, &mut BatchLeg<'_, HostRun>),
    sel: &mut DeviceBuffer<u32>,
    (at, m, walk_end): (At, usize, usize),
) -> Result<bool, GpuError> {
    let l = at.layer;
    {
        let Body { parts, .. } = &mut *body;
        let x = parts.trunk.normed();
        parts.moe.enqueue_shared_rows(cx, l, m, x)?;
    }
    let covers = |k: usize| pick_covers(side, k);
    let (ring, land) = body
        .xs
        .stream
        .unit_pick(port, at, (m, 1), walk_end, covers)?;
    if ring.is_some() {
        return Err(GpuError::shape(
            PASS,
            format!("layer {l}: the stream gave a ring to a body that starts none"),
        ));
    }
    let n_card = side.n_card(l);
    let ran = if n_card > 0 {
        {
            let (ids, _) = body.parts.moe.route();
            side.places_rows(cx.gpu, l, (ids, m), sel)?;
        }
        // The pick's admits copy into the slots the map now names: the card's slots read them
        // behind every landing batch's event.
        for event in land.iter().filter_map(|b| b.event.as_deref()) {
            cx.gpu.stream().wait(event)?;
        }
        body.parts
            .moe
            .enqueue_card_rows(cx, l, m, (&*sel, n_card))?
    } else {
        false
    };
    body.xs
        .stream
        .unit_read(port, (at, 1), (land, None), covers)?;
    Ok(ran)
}

/// A hosted layer's back from the upload on, over a unit's `m` rows: the card's and the host's
/// sums row by row through the block's tail into `y`, then the post.
fn back_rows(
    (cx, card): (Cx<'_>, bool),
    body: &mut Body,
    hsum: &DeviceBuffer<f32>,
    (l, m): (usize, usize),
) -> Result<(), GpuError> {
    {
        let Body { parts, y, .. } = &mut *body;
        parts
            .moe
            .enqueue_back_rows(cx, l, m, (card, Some(hsum)), y)?;
    }
    step::close(cx.gpu, cx.w, body, l, m)
}

/// The prompt pass's walk `(1, m, Batch)`: one unit of `m` rows through the host tier's batch
/// port.
struct PassWalk<'a> {
    cx: Cx<'a>,
    body: &'a mut Body,
    side: &'a PlacedSide,
    /// The places of the unit's routed slots.
    sel: &'a mut DeviceBuffer<u32>,
    /// The unit's first position and its rows.
    at: (u32, usize),
    /// Whether the hosted layer's card slots ran, from its shadow to its back.
    card: bool,
}

impl PassWalk<'_> {
    /// The walk's last position: the unit's first plus its rows.
    fn walk_end(&self) -> usize {
        self.at.0 as usize + self.at.1
    }
}

/// One unit of the prompt pass over every layer of `body`, its `m` rows (the embedding already in
/// the streams) at positions `at ..`, through `placed`'s batch port. The hosted pass's buffers
/// ([`HostedPass`]) leave `body` for the walk and return whatever it returned.
pub(crate) fn walk_pass(
    (gpu, w): (&Gpu, &Weights),
    body: &mut Body,
    placed: &mut PlacedHost,
    (m, at): (usize, u32),
) -> Result<(), GpuError> {
    let n = body.sh.n_layer;
    let rows = body.pass.ok_or(GpuError::State {
        what: PASS,
        missing: "the load's prompt pass",
    })?;
    let mut hosted = body.hosted.take().ok_or(GpuError::State {
        what: PASS,
        missing: "the hosted pass's buffers (a load opened with the pass over a plan that hosts \
                  experts)",
    })?;
    let PlacedHost { shell, side, .. } = placed;
    let walked = {
        let mut leg = BatchLeg::new(gpu.stream(), &mut shell.hybrid, &mut hosted.hsum, rows);
        let mut prog = PassWalk {
            cx: Cx { gpu, w },
            body,
            side,
            sel: &mut hosted.sel,
            at: (at, m),
            card: false,
        };
        sched::walk(
            Overlap {
                units: 1,
                cols: m,
                port: PortKind::Batch,
            },
            n,
            &mut leg,
            &mut prog,
        )
        .and_then(|()| {
            // The walk's end under host streaming: the last returned layer's return, and a pick
            // still held or a join still waiting refused by name.
            prog.body.xs.stream.end_walk(leg.hybrid(), gpu.stream())?;
            prog.body.xs.stream.walk_done();
            Ok(())
        })
    };
    body.hosted = Some(hosted);
    walked
}

impl<'a> LayerProgram for PassWalk<'a> {
    type Port = BatchLeg<'a, HostRun>;

    fn host_leg(&self, at: At) -> bool {
        self.side.hosted(at.layer)
    }

    /// A layer with no host share whole over the unit's rows; a hosted one up to its download.
    fn front(&mut self, port: &mut BatchLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let (l, (pos0, m)) = (at.layer, self.at);
        step::attn(self.cx.gpu, self.cx.w, self.body, l, Run::Prompt(pos0, m))?;
        if !self.side.hosted(l) {
            step::ffn_whole(self.cx.gpu, self.cx.w, self.body, l, m)?;
            return step::close(self.cx.gpu, self.cx.w, self.body, l, m);
        }
        self.body.parts.trunk.enqueue_pre(self.cx, l, Sub::Ffn)?;
        front_rows(self.cx, self.body, self.side, m, port, l)
    }

    /// The shared expert, the pick and the card's slots of a hosted layer, under the host's
    /// union.
    fn shadow(&mut self, port: &mut BatchLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        if !self.side.hosted(at.layer) {
            return Ok(());
        }
        let m = self.at.1;
        let walk_end = self.walk_end();
        self.card = shadow_rows(
            self.cx,
            self.body,
            (self.side, port),
            &mut *self.sel,
            (at, m, walk_end),
        )?;
        Ok(())
    }

    /// A hosted layer from the upload on, then its post, then the layer's return owed in a
    /// returning call's last walk ([`PromptStream::unit_back`]).
    ///
    /// [`PromptStream::unit_back`]: bloomery_gpu::host::prompt_stream::PromptStream::unit_back
    fn back(&mut self, port: &mut BatchLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        if !self.side.hosted(at.layer) {
            return Ok(());
        }
        let m = self.at.1;
        back_rows((self.cx, self.card), self.body, port.hsum(), (at.layer, m))?;
        let (walk_end, covers) = (self.walk_end(), pick_covers(self.side, at.layer));
        self.body.xs.stream.unit_back((at, 1), walk_end, covers);
        Ok(())
    }
}
