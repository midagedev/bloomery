//! The MiMo-V2.6-Flash session: [`Body`] behind the session's traits, planned
//! once by its placement (every routed expert on the host tier) and loaded by
//! gpumodel's constructor ([`Body::open_placed`]).
//!
//! A prompt is fed one decode step an id ([`GpuModel::step`]): the program has
//! no pass of several rows. The caches are per position, but the body takes no
//! checkpoint and no rollback, so a cut keeps every held position or the
//! empty model.

use bloomery_gpu::{GpuError, GpuModel};
use bloomery_gpu_mimo2::Body;
use bloomery_levers::HostCfg;
use gguf::Split;
use model::arch::mimo2::place::PlanInputs;
use model::placement::{Machine, Plan, PlanLevers};

use crate::{Keep, Open, Prompt};

const WHAT: &str = "mimo2 session";

/// How the MiMo session opens.
#[derive(Clone, Debug)]
pub struct Mimo2Cfg {
    /// The placement's levers the plan is made under: the card budget.
    pub place: PlanLevers,
    /// The host set's read-in and lock, and the file pages' release.
    pub host: HostCfg,
}

impl Open for Body {
    type Cfg = Mimo2Cfg;
    type Inputs = PlanInputs;

    fn inputs(file: &Split) -> Result<PlanInputs, GpuError> {
        PlanInputs::read(file).map_err(|e| GpuError::plan(WHAT, e))
    }

    fn layer_count(inputs: &PlanInputs) -> usize {
        inputs.model.layers
    }

    /// The plan on `machine`, refused unless its layers sit on one card: the
    /// chain runs on one. A placement that hangs an expert tier card is
    /// refused by the plan itself: every routed expert is the host's.
    fn plan<'a>(
        inputs: &'a PlanInputs,
        machine: &'a Machine,
        ctx: usize,
        cfg: &Mimo2Cfg,
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
        cfg: &Mimo2Cfg,
    ) -> Result<GpuModel<Body>, GpuError> {
        Body::open_placed(file, plan, inputs, 0, cfg.host)
    }

    /// The prompt call has no buffers of its own.
    fn prepare(_m: &mut GpuModel<Body>, _cfg: &Mimo2Cfg) -> Result<bool, GpuError> {
        Ok(false)
    }
}

impl Prompt for Body {
    /// One decode step an id, the argmax after the last.
    fn prompt(m: &mut GpuModel<Body>, ids: &[u32]) -> Result<u32, GpuError> {
        m.step(ids)
    }
}

impl Keep for Body {
    /// Every held position, or the empty model: the body holds no
    /// checkpoint and takes no rollback.
    fn keepable(m: &GpuModel<Body>, n: u32) -> u32 {
        if n >= m.pos() { m.pos() } else { 0 }
    }

    /// Nothing to take back at the model's position; back to empty at 0 — a
    /// reset; any other position refused by name.
    fn cut(m: &mut GpuModel<Body>, n: u32) -> Result<(), GpuError> {
        match n {
            n if n == m.pos() => Ok(()),
            0 => m.reset(),
            n => Err(GpuError::Shape {
                what: WHAT,
                detail: format!(
                    "a cut to {n} of {} positions; the body keeps every position or none",
                    m.pos()
                ),
            }),
        }
    }
}
