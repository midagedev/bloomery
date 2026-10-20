//! The Solar-Open2 session: [`Body`] behind the session's traits, planned
//! once by its placement (the routed experts the card experts read by the
//! expert rule, the rest on the host tier) and loaded by gpumodel's
//! constructor ([`Body::open_placed_slots`]), under adaptive expert residency
//! or not ([`Solar2Cfg::residency`]).
//!
//! A prompt is fed in batches ([`bloomery_gpu_solar2::prefill::prefill`]),
//! one residency pass a call, and leaves the bits the decode steps leave. The
//! 36 KDA layers hold one recurrent state each, written in place by every
//! position, so a cut keeps every held position, the empty model, or a
//! checkpoint ([`Body::kept`]); the 12 GQA layers' K/V planes are cut by
//! position and ask nothing of it. The body serves one sequence.

use std::num::NonZeroUsize;

use bloomery_gpu::host::served::prompt_pass;
use bloomery_gpu::host::swap::Residency;
use bloomery_gpu::{GpuError, GpuModel};
use bloomery_gpu_solar2::Body;
use bloomery_gpu_solar2::prefill::prefill;
use bloomery_levers::HostCfg;
use gguf::Split;
use model::arch::solar2::place::PlanInputs;
use model::placement::{Machine, Plan, PlanLevers};
use runtime::seqstate::Kept;

use super::{cut_checkpointed, one_card_ctx};
use crate::{Keep, Open, Prompt};

const WHAT: &str = "solar2 session";

/// How the Solar session opens.
#[derive(Clone, Debug)]
pub struct Solar2Cfg {
    /// The placement's levers the plan is made under: the card budget.
    pub place: PlanLevers,
    /// The host set's read-in and lock, and the file pages' release.
    pub host: HostCfg,
    /// The adaptive expert residency the load runs ([`Residency::Off`]: the
    /// load's slot map for the model's life).
    pub residency: Residency,
    /// Batches a prompt group runs layer by layer
    /// ([`bloomery_gpu_solar2::set_prefill_group`]).
    pub group: usize,
}

impl Open for Body {
    type Cfg = Solar2Cfg;
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
        cfg: &Solar2Cfg,
    ) -> Result<Plan<'a>, GpuError> {
        let ctx = one_card_ctx(WHAT, machine, ctx)?;
        inputs
            .plan(machine, ctx, &cfg.place)
            .map_err(|e| GpuError::plan(WHAT, e))
    }

    fn open(
        file: Split,
        inputs: &PlanInputs,
        plan: &Plan<'_>,
        cfg: &Solar2Cfg,
    ) -> Result<GpuModel<Body>, GpuError> {
        Body::open_placed_slots(
            file,
            plan,
            inputs,
            0,
            cfg.host,
            cfg.residency,
            NonZeroUsize::MIN,
        )
    }

    /// The configuration's group; the units it needs are made here.
    fn prepare(m: &mut GpuModel<Body>, cfg: &Solar2Cfg) -> Result<bool, GpuError> {
        bloomery_gpu_solar2::set_prefill_group(m, cfg.group)
    }
}

impl Prompt for Body {
    /// The prompt in batches as one residency pass ([`prompt_pass`]): a call
    /// past the stores' positions is refused before anything runs
    /// ([`prefill`]).
    fn prompt(m: &mut GpuModel<Body>, ids: &[u32]) -> Result<u32, GpuError> {
        let token = prompt_pass(m, |m| {
            m.run_rows(ids.len(), WHAT, |gpu, w, body, head, pos| {
                prefill(gpu, w, body, head, ids, pos)
            })
        })?;
        token.ok_or(GpuError::State {
            what: WHAT,
            missing: "the head's token after the prompt",
        })
    }
}

impl Keep for Body {
    /// [`Body::kept`]: every position, the empty model, or the nearest
    /// checkpoint at or below `n`.
    fn keepable(m: &GpuModel<Body>, n: u32) -> u32 {
        <Body as Keep>::kept(m, n).at
    }

    /// [`Body::kept`].
    fn kept(m: &GpuModel<Body>, n: u32) -> Kept {
        m.body(WHAT)
            .map_or_else(|_| Kept::rule(n, m.pos(), 0), |b| b.kept(n, m.pos()))
    }

    /// [`cut_checkpointed`].
    fn cut(m: &mut GpuModel<Body>, n: u32) -> Result<(), GpuError> {
        cut_checkpointed(m, n)
    }
}
