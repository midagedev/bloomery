//! The MiMo-V2.6-Flash session: [`Body`] behind the session's traits, planned
//! once by its placement (the routed experts the card experts read by the
//! expert rule, the rest on the host tier) and loaded by gpumodel's
//! constructor ([`Body::open_placed_slots`]), under adaptive expert residency
//! or not ([`Mimo2Cfg::residency`]).//!
//! A prompt is fed as the configuration says ([`Mimo2Cfg::prefill`]): in
//! batches ([`bloomery_gpu_mimo2::prefill`]) or one decode step an id
//! ([`GpuModel::step`]), the two leaving the same bits. A full layer's cache
//! is per position. A window layer holds a ring, so a cut keeps its target
//! when that ring still holds the keys the target reads, and otherwise keeps
//! nothing until the marks land. The rollback writes nothing inside that
//! slack. The body parks resident slots ([`bloomery_gpu::model::Slots`]: a
//! slot is its layers' stores), so [`crate::Session::add_slots`] and
//! [`crate::Session::select_slot`] serve it as they serve every body that
//! does.

use std::num::NonZeroUsize;

use bloomery_gpu::host::swap::Residency;
use bloomery_gpu::{GpuError, GpuModel};
use bloomery_gpu_mimo2::{Body, PrefillMode};
use bloomery_levers::HostCfg;
use gguf::Split;
use model::arch::mimo2::place::PlanInputs;
use model::placement::{Machine, Plan, PlanLevers};

use crate::{Keep, Open, Prompt, cut_positions};

const WHAT: &str = "mimo2 session";

/// How the MiMo session opens.
#[derive(Clone, Debug)]
pub struct Mimo2Cfg {
    /// The placement's levers the plan is made under: the card budget.
    pub place: PlanLevers,
    /// The host set's read-in and lock, and the file pages' release.
    pub host: HostCfg,
    /// How a prompt is fed: in batches, or one decode step per id (refused by
    /// name beside a residency machine).
    pub prefill: PrefillMode,
    /// The adaptive expert residency the load runs ([`Residency::Off`]: the
    /// load's slot map for the model's life).
    pub residency: Residency,
    /// Batches a prompt group runs layer by layer
    /// ([`bloomery_gpu_mimo2::set_prefill_group`]).
    pub group: usize,
    /// The resident sequences the plan counts and the load serves
    /// ([`Body::open_placed_slots`]); [`crate::Session::add_slots`] makes
    /// them after the load.
    pub slots: NonZeroUsize,
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

    /// The plan on `machine` counting `cfg.slots` sequences, refused unless
    /// its layers sit on one card: the chain runs on one. A placement that
    /// hangs an expert tier card is refused by the plan itself: every routed
    /// expert is the host's.
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
            .plan_with_slots(machine, ctx, &cfg.place, cfg.slots)
            .map_err(|e| GpuError::plan(WHAT, e))
    }

    fn open(
        file: Split,
        inputs: &PlanInputs,
        plan: &Plan<'_>,
        cfg: &Mimo2Cfg,
    ) -> Result<GpuModel<Body>, GpuError> {
        Body::open_placed_slots(file, plan, inputs, 0, cfg.host, cfg.residency, cfg.slots)
    }

    /// The configuration's feed and group; the batch feed's buffers made
    /// here, for the group's units.
    fn prepare(m: &mut GpuModel<Body>, cfg: &Mimo2Cfg) -> Result<bool, GpuError> {
        let grew = bloomery_gpu_mimo2::set_prefill_group(m, cfg.group)?;
        Ok(bloomery_gpu_mimo2::set_prefill(m, cfg.prefill)? || grew)
    }
}

impl Prompt for Body {
    /// The body's feed ([`bloomery_gpu_mimo2::feed`]): a call past the
    /// stores' positions is refused before anything runs.
    fn prompt(m: &mut GpuModel<Body>, ids: &[u32]) -> Result<u32, GpuError> {
        bloomery_gpu_mimo2::feed(m, ids)
    }
}

impl Keep for Body {
    /// `n` when every live layer still holds the keys the pass at `n` reads:
    /// a full plane always, a ring when the written end minus `n` is within
    /// `rows - window + 1`. Otherwise 0. `n` past the model's position is
    /// that position.
    fn keepable(m: &GpuModel<Body>, n: u32) -> u32 {
        let pos = m.pos();
        let Ok(body) = m.body("mimo2 keep") else {
            return 0;
        };
        if n > pos {
            return pos;
        }
        if body.cut_keeps(pos, n) { n } else { 0 }
    }

    /// Back to empty at 0 — a reset; else the model's rollback. The depth is
    /// the model's position, recorded before the rollback.
    fn cut(m: &mut GpuModel<Body>, n: u32) -> Result<(), GpuError> {
        let pos = m.pos();
        if let Ok((_, _, body)) = m.body_parts("mimo2 cut") {
            body.stand_for_cut(pos);
        }
        cut_positions(m, n)
    }
}
