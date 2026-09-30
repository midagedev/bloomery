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

use bloomery_gpu::GpuError;
use bloomery_gpu::GpuModel;
use bloomery_gpu::hybrid::refuse_expert_tiers;
use bloomery_gpu_glm5next::{Body, PrefillMode};
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
    /// The placement's levers the plan is made under: the card budget.
    pub place: PlanLevers,
    /// The host set's read-in and lock, and the file pages' release.
    pub host: HostCfg,
    /// How a prompt is fed: in batches, or one decode step per id.
    pub prefill: PrefillMode,
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
    /// chain runs on one. A placement with an expert tier card is refused:
    /// the load hangs no tier under its host tier.
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
        refuse_expert_tiers(WHAT, machine)?;
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

    /// The configuration's feed; the batch feed's buffers made here.
    fn prepare(m: &mut GpuModel<Body>, cfg: &GlmCfg) -> Result<bool, GpuError> {
        bloomery_gpu_glm5next::set_prefill(m, cfg.prefill)
    }
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
