//! The Muse Spark session: [`Body`] behind the session's traits, planned once
//! by its placement (the routed experts the card's `_sel` launches hold by the
//! expert rule, the rest on the host tier) and loaded by gpumodel's
//! constructor under the configuration's residency
//! ([`GpuModel::open_placed_with`]). The body lives in the gpu crate's own
//! musespark arch, so it needs no feature.
//!
//! A prompt is passes of the load's rows through every layer, the hosted
//! layers' experts through the host tier's batch port ([`GpuModel::prefill`]);
//! the expert stream of its calls (`BLOOMERY_XSTREAM`) is the caller's to set
//! ([`bloomery_gpu::host::xstream::XStreamBody`]). A cut keeps every held
//! position, as every store is a plane of positions the step writes before it
//! reads.

use bloomery_gpu::arch::musespark::Body;
use bloomery_gpu::folder::Model;
use bloomery_gpu::{GpuError, GpuModel};
use gguf::Split;
use model::arch::musespark::place::PlanInputs;
use model::placement::{Machine, Plan};

use super::{HybridCfg, HybridInputs, one_card_ctx};
use crate::{Keep, Open, Prompt, cut_positions, keep_positions};

/// The session's name in its refusals.
const WHAT: &str = "musespark session";

impl Open for Body {
    type Cfg = HybridCfg;
    type Inputs = PlanInputs;

    /// The family's reading with its refusals ([`PlanInputs::read`]): a file
    /// with a feature no program runs is refused by name, every one listed.
    fn inputs(file: &Split) -> Result<PlanInputs, GpuError> {
        PlanInputs::read(file).map_err(|e| GpuError::plan(WHAT, e))
    }

    fn layer_count(inputs: &PlanInputs) -> usize {
        inputs.model.layers
    }

    /// The plan on `machine`, refused unless its layers sit on one card: the
    /// chain runs on one, and a machine that names an expert tier card is
    /// refused by the plan itself, for the body hangs none.
    fn plan<'a>(
        inputs: &'a PlanInputs,
        machine: &'a Machine,
        ctx: usize,
        cfg: &HybridCfg,
    ) -> Result<Plan<'a>, GpuError> {
        <super::SingleSequence as super::SequenceSlots<Body>>::check(cfg.slots, WHAT)?;
        let ctx = one_card_ctx(WHAT, machine, ctx)?;
        inputs.plan_at(machine, ctx, &cfg.place, cfg.slots, WHAT)
    }

    /// The model resident by `plan`, on the plan's one card, under the
    /// configuration's residency ([`Residency::Off`](bloomery_gpu::host::swap::Residency)
    /// keeps the plan's slot map for the model's life).
    fn open(
        file: Model,
        _inputs: &PlanInputs,
        plan: &Plan<'_>,
        cfg: &HybridCfg,
    ) -> Result<GpuModel<Body>, GpuError> {
        <super::SingleSequence as super::SequenceSlots<Body>>::check(cfg.slots, WHAT)?;
        GpuModel::<Body>::open_placed_with(
            crate::split_of(file, WHAT)?,
            plan,
            0,
            cfg.host,
            cfg.residency,
        )
    }

    /// None to make: the load cut the prompt pass's buffers from the card's
    /// room ([`GpuModel::pass_rows`]).
    fn prepare(_m: &mut GpuModel<Body>, _cfg: &HybridCfg) -> Result<bool, GpuError> {
        Ok(false)
    }
}

impl Prompt for Body {
    /// Passes of the load's rows through every layer, one readback after the
    /// last ([`GpuModel::prefill`]).
    fn prompt(m: &mut GpuModel<Body>, ids: &[u32]) -> Result<u32, GpuError> {
        m.prefill(ids)
    }
}

impl Keep for Body {
    /// Every held position up to `n`.
    fn keepable(m: &GpuModel<Body>, n: u32) -> u32 {
        keep_positions(m, n)
    }

    /// Back to empty at 0, else the model's rollback.
    fn cut(m: &mut GpuModel<Body>, n: u32) -> Result<(), GpuError> {
        cut_positions(m, n)
    }
}
