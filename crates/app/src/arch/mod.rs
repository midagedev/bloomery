//! Each model's session pieces: its [`crate::Open`], [`crate::Prompt`] and
//! [`crate::Keep`], and the drafts that run beside it. A binary turns on the
//! feature of the model it opens.
//!
//! A body that runs one chain on one card, fed in prompt batches, is one
//! owner's shape: the families that run it ([`solar2`], [`minimaxm3`]) name
//! only what their architecture forces — the plan inputs, the load, the batch
//! and the cut rule — and [`Batch`] holds the rest: the [`Open`] the placement
//! plans by, the [`Prompt`] the batch runs and the [`Keep`] the cut follows.

use bloomery_gpu::head::Head;
use bloomery_gpu::host::served::prompt_pass;
use bloomery_gpu::host::swap::Residency;
use bloomery_gpu::model::{ChainBody, Rollback};
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{Gpu, GpuError, GpuModel};
use bloomery_levers::HostCfg;
use gguf::Split;
use model::arch::place_hybrid::PlanInputs;
use model::placement::{Machine, Plan, PlanLevers};
use runtime::seqstate::Kept;

use crate::{Keep, Open, Prompt};

#[cfg(feature = "deepseek41")]
pub mod deepseek41;
#[cfg(feature = "glm5next")]
pub mod glm5next;
#[cfg(feature = "mimo2")]
pub mod mimo2;
#[cfg(feature = "minimaxm3")]
pub mod minimaxm3;
/// The qwen3moe family's Qwen3.8 session pieces: the body lives in the gpu
/// crate's own qwen3moe arch, so they need no feature of their own.
pub mod qwen3moe;
#[cfg(feature = "solar2")]
pub mod solar2;

/// How a hybrid session opens: the placement's levers, the host set's load
/// settings, the residency the load runs under and the batches a prompt group
/// runs layer by layer.
#[derive(Clone, Debug)]
pub struct HybridCfg {
    /// The placement's levers the plan is made under: the card budget.
    pub place: PlanLevers,
    /// The host set's read-in and lock, and the file pages' release.
    pub host: HostCfg,
    /// The adaptive expert residency the load runs ([`Residency::Off`]: the
    /// load's slot map for the model's life).
    pub residency: Residency,
    /// The batches a prompt group runs layer by layer.
    pub group: usize,
}

/// The plan inputs of a hybrid body: the family's own hyperparameters under
/// the common plan.
pub trait HybridInputs {
    /// The layers the file holds.
    fn layers(&self) -> usize;

    /// The plan of the file on `machine` at `ctx` positions under `levers`,
    /// refused by name (`what` the session) when the placement does not hold.
    fn plan_at<'a>(
        &'a self,
        machine: &'a Machine,
        ctx: u64,
        levers: &PlanLevers,
        what: &'static str,
    ) -> Result<Plan<'a>, GpuError>;
}

impl<H> HybridInputs for PlanInputs<H> {
    fn layers(&self) -> usize {
        self.model.layers
    }

    fn plan_at<'a>(
        &'a self,
        machine: &'a Machine,
        ctx: u64,
        levers: &PlanLevers,
        what: &'static str,
    ) -> Result<Plan<'a>, GpuError> {
        PlanInputs::plan(self, machine, ctx, levers).map_err(|e| GpuError::plan(what, e))
    }
}

/// A body that runs one chain on one card, fed in prompt batches: what the
/// common [`Open`], [`Prompt`] and [`Keep`] need of a family. The family names
/// its architecture's facts; the impls below hold everything two families
/// already do the same way.
pub trait Batch: ChainBody + Rollback + Sized {
    /// The plan inputs the body is loaded by.
    type Inputs: HybridInputs;

    /// The session's name in its refusals.
    const WHAT: &'static str;

    /// The plan inputs of `file`.
    fn read(file: &Split) -> Result<Self::Inputs, GpuError>;

    /// The body resident by `plan`, which `inputs` made: the layers the plan
    /// puts on the card, the host tier over the routed ones.
    fn open_placed(
        file: Split,
        plan: &Plan<'_>,
        inputs: &Self::Inputs,
        host: HostCfg,
        residency: Residency,
    ) -> Result<GpuModel<Self>, GpuError>;

    /// The prompt batch of `ids` from `at`: the head's row after the last.
    fn batch(
        gpu: &Gpu,
        w: &Weights,
        body: &mut Self,
        head: &mut Head,
        ids: &[u32],
        at: u32,
    ) -> Result<bool, GpuError>;

    /// The batches a prompt group runs layer by layer; whether it made any.
    fn group(m: &mut GpuModel<Self>, g: usize) -> Result<bool, GpuError>;

    /// The positions a cut to `n` keeps: every held position, for a body
    /// whose stores are per position.
    fn keepable(m: &GpuModel<Self>, n: u32) -> u32 {
        crate::keep_positions(m, n)
    }

    /// The rule with its reason: the body's own, for a body that keeps a
    /// prefix only where its state stands.
    fn kept(m: &GpuModel<Self>, n: u32) -> Kept {
        Kept::rule(n, m.pos(), Self::keepable(m, n))
    }

    /// The cut to `n`: back to empty at 0, else the model's rollback.
    fn cut(m: &mut GpuModel<Self>, n: u32) -> Result<(), GpuError> {
        crate::cut_positions(m, n)
    }
}

impl<B: Batch> Open for B {
    type Cfg = HybridCfg;
    type Inputs = B::Inputs;

    fn inputs(file: &Split) -> Result<B::Inputs, GpuError> {
        B::read(file)
    }

    fn layer_count(inputs: &B::Inputs) -> usize {
        inputs.layers()
    }

    /// The plan on `machine`, refused unless its layers sit on one card: the
    /// chain runs on one. A placement that hangs an expert tier card is
    /// refused by the plan itself: every routed expert is the host's.
    fn plan<'a>(
        inputs: &'a B::Inputs,
        machine: &'a Machine,
        ctx: usize,
        cfg: &HybridCfg,
    ) -> Result<Plan<'a>, GpuError> {
        let ctx = one_card_ctx(B::WHAT, machine, ctx)?;
        inputs.plan_at(machine, ctx, &cfg.place, B::WHAT)
    }

    fn open(
        file: Split,
        inputs: &B::Inputs,
        plan: &Plan<'_>,
        cfg: &HybridCfg,
    ) -> Result<GpuModel<B>, GpuError> {
        B::open_placed(file, plan, inputs, cfg.host, cfg.residency)
    }

    /// The configuration's group; the units it needs are made here.
    fn prepare(m: &mut GpuModel<B>, cfg: &HybridCfg) -> Result<bool, GpuError> {
        B::group(m, cfg.group)
    }
}

impl<B: Batch> Prompt for B {
    /// The prompt in batches as one residency pass ([`prompt_pass`]): the
    /// batch runs the ids and the head's token after the last is the argmax.
    fn prompt(m: &mut GpuModel<B>, ids: &[u32]) -> Result<u32, GpuError> {
        let what = B::WHAT;
        let token = prompt_pass(m, |m| {
            m.run_rows(ids.len(), what, |gpu, w, body, head, pos| {
                B::batch(gpu, w, body, head, ids, pos)
            })
        })?;
        token.ok_or(GpuError::State {
            what,
            missing: "the head's token after the prompt",
        })
    }
}

impl<B: Batch> Keep for B {
    /// [`Batch::keepable`].
    fn keepable(m: &GpuModel<B>, n: u32) -> u32 {
        B::keepable(m, n)
    }

    /// [`Batch::kept`].
    fn kept(m: &GpuModel<B>, n: u32) -> Kept {
        B::kept(m, n)
    }

    /// [`Batch::cut`].
    fn cut(m: &mut GpuModel<B>, n: u32) -> Result<(), GpuError> {
        B::cut(m, n)
    }
}

/// `ctx` positions as a plan takes them, for a body whose chain runs on one
/// card: refused by name (`what` the body's session) when `machine` puts the
/// layers on any other number of cards or `ctx` passes u64.
pub fn one_card_ctx(what: &'static str, machine: &Machine, ctx: usize) -> Result<u64, GpuError> {
    if machine.cards.len() != 1 {
        return Err(GpuError::Shape {
            what,
            detail: format!(
                "the placement puts the layers on {} cards; the chain runs on one",
                machine.cards.len()
            ),
        });
    }
    u64::try_from(ctx).map_err(|_| GpuError::Shape {
        what,
        detail: format!("a context of {ctx} positions passes u64"),
    })
}

/// [`crate::Keep::cut`] of a body whose recurrent state has its history only
/// in checkpoints: nothing to take back at the model's position; back to
/// empty at 0 (a reset); else the checkpoint at `n` ([`Rollback`]), any other
/// position refused by name.
pub fn cut_checkpointed<B: Rollback>(m: &mut GpuModel<B>, n: u32) -> Result<(), GpuError> {
    match n {
        n if n == m.pos() => Ok(()),
        0 => m.reset(),
        n => m.rollback(n),
    }
}
