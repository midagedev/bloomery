//! The GLM-5.3-Flash session: [`Body`] behind the session's traits, planned
//! once by its placement (every routed expert on the host) and loaded by
//! gpumodel's constructor ([`Body::open_placed`]).
//!
//! A prompt is fed one step a position: the step walk is the only walk the
//! body has. Each prompt call takes the KDA layers' checkpoints its marks
//! name ([`bloomery_gpu_glm5next::prompt`]), and a cut keeps every fed
//! position, the empty model, or a checkpoint: each KDA layer holds one
//! recurrent state, and its history only in those copies.

use bloomery_gpu::GpuError;
use bloomery_gpu::GpuModel;
use bloomery_gpu_glm5next::Body;
use bloomery_levers::HostCfg;
use gguf::Split;
use model::arch::glm5next::place::PlanInputs;
use model::placement::{Machine, Plan, PlanLevers};
use runtime::seqstate::Kept;

use crate::{Keep, Open, Prompt};

const WHAT: &str = "glm5next session";

/// How the GLM session opens.
#[derive(Clone, Debug)]
pub struct GlmCfg {
    /// The placement's levers the plan is made under; a hot list is refused
    /// by name (no routed expert is on the card).
    pub place: PlanLevers,
    /// The host set's read-in and lock, and the file pages' release.
    pub host: HostCfg,
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

    /// The plan on `machine`, refused unless its layers sit on one card: the
    /// chain runs on one.
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

    /// No prompt buffers: the prompt is fed by steps.
    fn prepare(_m: &mut GpuModel<Body>, _cfg: &GlmCfg) -> Result<bool, GpuError> {
        Ok(false)
    }
}

impl Prompt for Body {
    /// One step per id, a readback at each checkpoint mark and after the
    /// last ([`bloomery_gpu_glm5next::prompt`]).
    fn prompt(m: &mut GpuModel<Body>, ids: &[u32]) -> Result<u32, GpuError> {
        bloomery_gpu_glm5next::prompt(m, ids)
    }
}

impl Keep for Body {
    /// [`Body::keep_point`]: every fed position, the empty model, or the
    /// nearest checkpoint at or below `n`.
    fn keepable(m: &GpuModel<Body>, n: u32) -> u32 {
        <Body as Keep>::kept(m, n).at
    }

    /// [`Body::kept`].
    fn kept(m: &GpuModel<Body>, n: u32) -> Kept {
        m.body(WHAT)
            .map_or_else(|_| Kept::rule(n, m.pos(), 0), |b| b.kept(n))
    }

    /// Nothing to take back at the model's position; back to empty at 0;
    /// else the checkpoint at `n` ([`Body`]'s rollback), any other position
    /// refused by name.
    fn cut(m: &mut GpuModel<Body>, n: u32) -> Result<(), GpuError> {
        match n {
            n if n == m.pos() => Ok(()),
            0 => m.reset(),
            n => m.rollback(n),
        }
    }
}
