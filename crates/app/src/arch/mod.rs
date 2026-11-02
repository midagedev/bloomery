//! Each model's session pieces: its [`crate::Open`], [`crate::Prompt`] and
//! [`crate::Keep`], and the drafts that run beside it. A binary turns on the
//! feature of the model it opens.
//!
//! A body that runs one chain on one card, fed in prompt batches, is one
//! owner's shape: the families that run it ([`solar2`], [`minimaxm3`]) name
//! only what their architecture forces — the plan inputs, the load, the batch
//! and the cut rule — and [`Batch`] holds the rest: the [`Open`] the placement
//! plans by, the [`Prompt`] the batch runs and the [`Keep`] the cut follows.

use std::num::NonZeroUsize;

use bloomery_gpu::head::Head;
use bloomery_gpu::host::served::prompt_pass;
use bloomery_gpu::host::swap::Residency;
use bloomery_gpu::model::{ChainBody, Rollback, Slots};
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
#[cfg(feature = "flint")]
pub mod flint;
#[cfg(feature = "glm5next")]
pub mod glm5next;
#[cfg(feature = "mimo2")]
pub mod mimo2;
#[cfg(feature = "minimaxm3")]
pub mod minimaxm3;
#[cfg(feature = "mistrallarge4")]
pub mod mistrallarge4;
/// The muse-spark family's session pieces: the body lives in the gpu crate's
/// own musespark arch, so they need no feature of their own.
pub mod musespark;
/// The qwen3moe family's Qwen3.8 session pieces: the body lives in the gpu
/// crate's own qwen3moe arch, so they need no feature of their own.
pub mod qwen3moe;
/// Raon-1-mini's session pieces: the body lives in the gpu crate's own raon1
/// arch, so they need no feature of their own.
pub mod raon1;
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
    /// The resident sequences charged by the plan before any allocation.
    pub slots: NonZeroUsize,
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
        slots: NonZeroUsize,
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
        slots: NonZeroUsize,
        what: &'static str,
    ) -> Result<Plan<'a>, GpuError> {
        self.plan_with_slots(machine, ctx, levers, slots)
            .map_err(|e| GpuError::plan(what, e))
    }
}

/// A body that runs one chain on one card, fed in prompt batches: what the
/// common [`Open`], [`Prompt`] and [`Keep`] need of a family. The family names
/// its architecture's facts; the impls below hold everything two families
/// already do the same way.
pub trait Batch: ChainBody + Rollback + Sized {
    /// The plan inputs the body is loaded by.
    type Inputs: HybridInputs;
    /// The sequence capability the body actually implements.
    type Sequences: SequenceSlots<Self>;

    /// The session's name in its refusals.
    const WHAT: &'static str;

    /// The plan inputs of `file`.
    fn read(file: &Split) -> Result<Self::Inputs, GpuError>;

    /// Price configuration-dependent buffers before placement; unchanged for a fixed schedule.
    fn configure(_inputs: &mut Self::Inputs, _cfg: &HybridCfg) -> Result<(), GpuError> {
        Ok(())
    }

    /// The body resident by `plan`, which `inputs` made: the layers the plan
    /// puts on the card, the host tier over the routed ones.
    fn open_placed(
        file: Split,
        plan: &Plan<'_>,
        inputs: &Self::Inputs,
        host: HostCfg,
        residency: Residency,
        slots: NonZeroUsize,
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

    fn configure_inputs(inputs: &mut B::Inputs, cfg: &HybridCfg) -> Result<(), GpuError> {
        B::configure(inputs, cfg)
    }

    fn layer_count(inputs: &B::Inputs) -> usize {
        inputs.layers()
    }

    /// The plan on `machine`, refused unless its layers sit on one card: the
    /// chain runs on one. A placement that hangs an expert tier card is
    /// refused by the plan itself where the body serves none
    /// (`PlanInputs::tier_reserve`); a body that serves one hangs it under the
    /// host tier at the load, which refuses more than the host tier serves.
    fn plan<'a>(
        inputs: &'a B::Inputs,
        machine: &'a Machine,
        ctx: usize,
        cfg: &HybridCfg,
    ) -> Result<Plan<'a>, GpuError> {
        <B::Sequences as SequenceSlots<B>>::check(cfg.slots, B::WHAT)?;
        let ctx = one_card_ctx(B::WHAT, machine, ctx)?;
        inputs.plan_at(machine, ctx, &cfg.place, cfg.slots, B::WHAT)
    }

    fn open(
        file: Split,
        inputs: &B::Inputs,
        plan: &Plan<'_>,
        cfg: &HybridCfg,
    ) -> Result<GpuModel<B>, GpuError> {
        <B::Sequences as SequenceSlots<B>>::check(cfg.slots, B::WHAT)?;
        B::open_placed(file, plan, inputs, cfg.host, cfg.residency, cfg.slots)
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

/// The sequence operations a common hybrid seat may offer.
pub trait SequenceSlots<B: ChainBody> {
    /// Whether several resident sequences are backed by body-owned stores.
    const RESIDENT: bool;
    /// Refuse an unsupported count before planning or allocation.
    fn check(slots: NonZeroUsize, what: &'static str) -> Result<(), GpuError>;
    /// Park the planned sequences through the session's actual slot operations.
    fn add(s: &mut crate::Session<B>, slots: NonZeroUsize) -> Result<(), crate::SessionError>;
    /// The number of resident sequences the session currently holds.
    fn count(s: &crate::Session<B>) -> usize;
    /// Select the actual sequence every following call acts on.
    fn select(s: &mut crate::Session<B>, slot: usize) -> Result<(), crate::SessionError>;
}

/// A body that holds one sequence; other slots are refused, not simulated.
pub struct SingleSequence;

impl<B: ChainBody> SequenceSlots<B> for SingleSequence {
    const RESIDENT: bool = false;

    fn check(slots: NonZeroUsize, what: &'static str) -> Result<(), GpuError> {
        if slots.get() != 1 {
            return Err(GpuError::Shape {
                what,
                detail: format!(
                    "the body holds one sequence, not {} resident slots",
                    slots.get()
                ),
            });
        }
        Ok(())
    }

    fn add(_s: &mut crate::Session<B>, slots: NonZeroUsize) -> Result<(), crate::SessionError> {
        <Self as SequenceSlots<B>>::check(slots, "single-sequence session").map_err(Into::into)
    }

    fn count(_s: &crate::Session<B>) -> usize {
        1
    }

    fn select(_s: &mut crate::Session<B>, slot: usize) -> Result<(), crate::SessionError> {
        if slot == 0 {
            Ok(())
        } else {
            Err(crate::SessionError::Refused(format!(
                "select_slot {slot}: the body holds one sequence"
            )))
        }
    }
}

/// Resident sequences backed by the body's actual parked stores.
pub struct ResidentSequences;

impl<B: Slots<Seq: 'static>> SequenceSlots<B> for ResidentSequences {
    const RESIDENT: bool = true;

    fn check(_slots: NonZeroUsize, _what: &'static str) -> Result<(), GpuError> {
        Ok(())
    }

    fn add(s: &mut crate::Session<B>, slots: NonZeroUsize) -> Result<(), crate::SessionError> {
        s.add_slots(slots.get())
    }

    fn count(s: &crate::Session<B>) -> usize {
        s.slots()
    }

    fn select(s: &mut crate::Session<B>, slot: usize) -> Result<(), crate::SessionError> {
        s.select_slot(slot)
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
