//! The GLM-5.3-Flash session: [`Body`] behind the session's traits, planned
//! once by its placement (the routed experts the card experts read by the
//! expert rule, the rest on the host) and loaded by
//! gpumodel's constructor ([`Body::open_placed`]).
//!
//! A prompt is fed as the configuration says ([`GlmCfg::prefill`]): in
//! batches ([`bloomery_gpu_glm5next::prefill`]) or one step a position
//! ([`bloomery_gpu_glm5next::prompt`]), the two leaving the same bits. Each
//! prompt call takes the KDA layers' checkpoints its marks name, and a cut
//! keeps every fed position, the empty model, or a checkpoint: each KDA
//! layer holds one recurrent state, and its history only in those copies.
//!
//! A session opens at one KDA lane: its loads run one row a pass and pay
//! nothing for a verify. [`open_nextn`] opens the same session with the
//! file's next-token layer beside the target — the body the MTP window drafts
//! on (`MtpBody`), under adaptive expert residency or not — and [`open_pair`]
//! with no layer, for the two-row verify probe; both hold two lanes.
//! [`open_resident`] opens the plain session under adaptive expert residency,
//! at one lane.

use bloomery_gpu::GpuError;
use bloomery_gpu::GpuModel;
use bloomery_gpu::host::swap::Residency;
use bloomery_gpu_glm5next::{Body, PrefillMode};
use bloomery_levers::HostCfg;
use gguf::Split;
use model::arch::glm5next::place::{KdaLanes, NextnInputs, PlanInputs};
use model::placement::{Machine, Plan, PlanLevers};
use runtime::seqstate::Kept;

use crate::{Keep, Loaded, Open, OpenArgs, OpenLog, Prompt, Session, SessionError};

mod mtp;

const WHAT: &str = "glm5next session";

/// How the GLM session opens.
#[derive(Clone, Debug)]
pub struct GlmCfg {
    /// The placement's levers the plan is made under: the card budget.
    pub place: PlanLevers,
    /// The host set's read-in and lock, and the file pages' release.
    pub host: HostCfg,
    /// How a prompt is fed: in batches, or one decode step per id.
    pub prefill: PrefillMode,
    /// Batches a prompt group runs layer by layer
    /// (`BLOOMERY_PREFILL_GROUP`, [`bloomery_gpu_glm5next::set_prefill_group`]).
    pub group: usize,
}

impl Open for Body {
    type Cfg = GlmCfg;
    type Inputs = PlanInputs;

    fn inputs(file: &Split) -> Result<PlanInputs, GpuError> {
        PlanInputs::read(file).map_err(|e| GpuError::plan(WHAT, e))
    }

    fn layer_count(inputs: &PlanInputs) -> usize {
        inputs.model.layers
    }

    /// The plan on `machine` at one KDA lane, refused unless its layers sit
    /// on one card: the chain runs on one. A placement's expert tier card is
    /// hung under the host tier by the load ([`Body::open_placed`]), which
    /// refuses more than the host tier serves.
    fn plan<'a>(
        inputs: &'a PlanInputs,
        machine: &'a Machine,
        ctx: usize,
        cfg: &GlmCfg,
    ) -> Result<Plan<'a>, GpuError> {
        if machine.cards.len() != 1 {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!(
                    "the placement puts the layers on {} cards; the chain runs on one",
                    machine.cards.len()
                ),
            });
        }
        let ctx = u64::try_from(ctx).map_err(|_| GpuError::Shape {
            what: WHAT,
            detail: format!("a context of {ctx} positions passes u64"),
        })?;
        inputs
            .plan(machine, ctx, &cfg.place)
            .map_err(|e| GpuError::plan(WHAT, e))
    }

    fn open(
        file: Split,
        inputs: &PlanInputs,
        plan: &Plan<'_>,
        cfg: &GlmCfg,
    ) -> Result<GpuModel<Body>, GpuError> {
        Body::open_placed(file, plan, inputs, 0, cfg.host)
    }

    /// The configuration's feed and group; the batch feed's buffers made
    /// here, for the group's units.
    fn prepare(m: &mut GpuModel<Body>, cfg: &GlmCfg) -> Result<bool, GpuError> {
        let grew = bloomery_gpu_glm5next::set_prefill_group(m, cfg.group)?;
        Ok(bloomery_gpu_glm5next::set_prefill(m, cfg.prefill)? || grew)
    }
}

/// The GLM session with the file's next-token layer loaded beside the
/// target, as [`Loaded::open`] then [`Loaded::ready`] open the plain one:
/// `file`'s headers and its NextN layer read once, the plan made once on
/// `args`' placement ([`PlanInputs::plan_nextn`]: the target's expert rule
/// within the card less the layer's bytes, two KDA lanes) and its target plan
/// handed to `log` (`false` stops there: `Ok(None)`), then the load by that
/// plan under `residency` ([`Body::open_placed_nextn_with`]: under `mid` the
/// host set also holds each layer's churn pool, which the target plan's host
/// headroom less the layer's host experts must take) in `args`' step mode,
/// handed to `log`, the step captured and the prompt call's buffers made.
/// The placement's expert tier cards serve the target; the draft's walk
/// stays on the stage card and the host. Refused as [`Open::plan`] refuses,
/// as the residency machine refuses at the load, and by name for a file of
/// other than one next-token layer, a plan the layer breaks and a plan that
/// puts one of the layer's experts on a tier.
pub fn open_nextn<M: Fn(usize) -> Machine>(
    file: Split,
    args: OpenArgs<GlmCfg, M>,
    residency: Residency,
    log: &mut impl OpenLog<Body>,
) -> Result<Option<Session<Body>>, SessionError> {
    let inputs = <Body as Open>::inputs(&file)?;
    let nextn = NextnInputs::read(&inputs).map_err(|e| GpuError::plan(WHAT, e))?;
    let (machine, ctx) = one_card(&args, &inputs)?;
    let plan = inputs
        .plan_nextn(&machine, ctx, &args.cfg.place, &nextn)
        .map_err(|e| GpuError::plan(WHAT, e))?;
    log.beside(plan.host_runs().map_err(|e| GpuError::plan(WHAT, e))?.1);
    if !log.plan(args.place, &inputs, &machine, &plan.plan)? {
        return Ok(None);
    }
    let ctx = ctx_of(&plan.plan)?;
    let model =
        Body::open_placed_nextn_with(file, &plan, &inputs, &nextn, 0, args.cfg.host, residency)?;
    ready(model, args.mode, args.cfg, ctx, log)
}

/// The GLM session of [`Loaded::open`] at two KDA lanes, with no next-token
/// layer: the load the two-row verify (`Rows`) runs on, planned by
/// [`PlanInputs::plan_lanes`] at [`KdaLanes::Two`] and loaded by
/// [`Body::open_placed_lanes`], `log` as [`open_nextn`] takes it. Refused as
/// [`Open::plan`] refuses. A plain session is [`Loaded::open`]'s, at one
/// lane.
pub fn open_pair<M: Fn(usize) -> Machine>(
    file: Split,
    args: OpenArgs<GlmCfg, M>,
    log: &mut impl OpenLog<Body>,
) -> Result<Option<Session<Body>>, SessionError> {
    let inputs = <Body as Open>::inputs(&file)?;
    let (machine, ctx) = one_card(&args, &inputs)?;
    let plan = inputs
        .plan_lanes(&machine, ctx, &args.cfg.place, KdaLanes::Two)
        .map_err(|e| GpuError::plan(WHAT, e))?;
    if !log.plan(args.place, &inputs, &machine, &plan)? {
        return Ok(None);
    }
    let ctx = ctx_of(&plan)?;
    let model = Body::open_placed_lanes(
        file,
        &plan,
        &inputs,
        0,
        args.cfg.host,
        Residency::Off,
        KdaLanes::Two,
    )?;
    ready(model, args.mode, args.cfg, ctx, log)
}

/// `args`' machine for `inputs` and its context, refused by name unless the
/// machine is one stage card ([`Open::plan`]'s rule); its expert tier cards
/// are hung under the host tier by the load.
fn one_card<M: Fn(usize) -> Machine>(
    args: &OpenArgs<GlmCfg, M>,
    inputs: &PlanInputs,
) -> Result<(Machine, u64), SessionError> {
    let machine = (args.machine)(<Body as Open>::layer_count(inputs));
    if machine.cards.len() != 1 {
        return Err(GpuError::Shape {
            what: WHAT,
            detail: format!(
                "the placement puts the layers on {} cards; the chain runs on one",
                machine.cards.len()
            ),
        }
        .into());
    }
    let ctx = u64::try_from(args.ctx).map_err(|_| GpuError::Shape {
        what: WHAT,
        detail: format!("a context of {} positions passes u64", args.ctx),
    })?;
    Ok((machine, ctx))
}

/// The plan's context in positions, refused past u32.
fn ctx_of(plan: &Plan<'_>) -> Result<u32, SessionError> {
    u32::try_from(plan.ctx_max).map_err(|_| {
        SessionError::Refused(format!("the plan's ctx_max {} passes u32", plan.ctx_max))
    })
}

/// `model` in step `mode`, handed to `log`, its step captured and the
/// prompt call's buffers made ([`Loaded::ready`]).
fn ready(
    mut model: GpuModel<Body>,
    mode: bloomery_gpu::model::StepMode,
    cfg: GlmCfg,
    ctx: u32,
    log: &mut impl OpenLog<Body>,
) -> Result<Option<Session<Body>>, SessionError> {
    model.set_mode(mode);
    log.load(&model)?;
    Loaded::from_model(model, cfg, ctx).ready(log).map(Some)
}

/// The GLM session under adaptive expert residency, as [`Loaded::open`]
/// then [`Loaded::ready`] open the plain one: `file`'s headers read once,
/// the plan made once on `args`' placement ([`Open::plan`]) and handed to
/// `log` (`false` stops there: `Ok(None)`), then the load by that plan under
/// `residency` ([`Body::open_placed_with`]: the host set also holds each
/// layer's churn pool) in `args`' step mode, handed to `log`, the step
/// captured and the prompt call's buffers made. Refused as [`Open::plan`]
/// refuses, and as the residency machine refuses at the load.
pub fn open_resident<M: Fn(usize) -> Machine>(
    file: Split,
    args: OpenArgs<GlmCfg, M>,
    residency: Residency,
    log: &mut impl OpenLog<Body>,
) -> Result<Option<Session<Body>>, SessionError> {
    let inputs = <Body as Open>::inputs(&file)?;
    let machine = (args.machine)(<Body as Open>::layer_count(&inputs));
    let plan = <Body as Open>::plan(&inputs, &machine, args.ctx, &args.cfg)?;
    if !log.plan(args.place, &inputs, &machine, &plan)? {
        return Ok(None);
    }
    let ctx = u32::try_from(plan.ctx_max).map_err(|_| {
        SessionError::Refused(format!("the plan's ctx_max {} passes u32", plan.ctx_max))
    })?;
    let mut model = Body::open_placed_with(file, &plan, &inputs, 0, args.cfg.host, residency)?;
    model.set_mode(args.mode);
    log.load(&model)?;
    Loaded::from_model(model, args.cfg, ctx)
        .ready(log)
        .map(Some)
}

impl Prompt for Body {
    /// The body's feed ([`bloomery_gpu_glm5next::feed`]): a call past the
    /// stores' positions is refused before anything runs.
    fn prompt(m: &mut GpuModel<Body>, ids: &[u32]) -> Result<u32, GpuError> {
        bloomery_gpu_glm5next::feed(m, ids)
    }
}

impl Keep for Body {
    /// [`Body::keep_point`]: every position, the empty model, or the
    /// nearest checkpoint at or below `n`.
    fn keepable(m: &GpuModel<Body>, n: u32) -> u32 {
        <Body as Keep>::kept(m, n).at
    }

    /// [`Body::kept`].
    fn kept(m: &GpuModel<Body>, n: u32) -> Kept {
        m.body(WHAT)
            .map_or_else(|_| Kept::rule(n, m.pos(), 0), |b| b.kept(n, m.pos()))
    }

    /// Nothing to take back at the model's position; back to empty at 0 —
    /// the fresh-context state, the residency use has built kept, as
    /// [`runtime::Target::reset`] keeps it (only [`Session::residency_reset`] and
    /// [`Session::clear`] take it back to its seed); else the checkpoint at
    /// `n` ([`Body`]'s rollback), any other position refused by name.
    fn cut(m: &mut GpuModel<Body>, n: u32) -> Result<(), GpuError> {
        match n {
            n if n == m.pos() => Ok(()),
            0 => m.reset(),
            n => m.rollback(n),
        }
    }
}
