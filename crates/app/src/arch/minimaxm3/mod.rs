//! The MiniMax-M3 session: [`Body`] behind the session's traits, planned once
//! by its placement (the routed experts the card experts read by the expert
//! rule, the rest on the host tier) and loaded by gpumodel's constructor
//! ([`Body::open_placed_slots`]), under adaptive expert residency or not
//! ([`HybridCfg::residency`]).
//!
//! A prompt is fed in batches ([`bloomery_gpu_minimaxm3::prefill::prefill`]),
//! one residency pass a call, and leaves the bits the decode steps leave:
//! every layer's key and value rows at those positions, every sparse layer's
//! index keys, and the last position's logits. Every store is a plane of
//! positions a later position never reads before it writes, so a cut keeps
//! every held position. The body serves one sequence.
//!
//! The session's shape is the common owner's ([`super::Batch`]): what the
//! placement plans, how the prompt runs and how a cut is kept are shared with
//! the other families that run one chain on one card. This module holds what
//! MiniMax-M3's architecture forces: the plan inputs, the load and the batch.

use std::num::NonZeroUsize;

use bloomery_gpu::head::Head;
use bloomery_gpu::host::swap::Residency;
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{Gpu, GpuError, GpuModel};
use bloomery_gpu_minimaxm3::Body;
use bloomery_gpu_minimaxm3::prefill::prefill;
use bloomery_levers::HostCfg;
use gguf::Split;
use model::arch::minimaxm3::place::PlanInputs;
use model::placement::Plan;

use super::Batch;

/// The session's name in its refusals.
const WHAT: &str = "minimaxm3 session";

impl Batch for Body {
    type Inputs = PlanInputs;

    const WHAT: &'static str = self::WHAT;

    /// The family's own reading ([`PlanInputs::describe`]), not the strict
    /// one: the coverage list still names the pre-tokenizer the tokenizer
    /// crate does not run, and the seat's own vocabulary build refuses the
    /// file by name when the server starts.
    fn read(file: &Split) -> Result<PlanInputs, GpuError> {
        PlanInputs::describe(file).map_err(|e| GpuError::plan(WHAT, e))
    }

    fn open_placed(
        file: Split,
        plan: &Plan<'_>,
        inputs: &PlanInputs,
        host: HostCfg,
        residency: Residency,
    ) -> Result<GpuModel<Body>, GpuError> {
        Body::open_placed_slots(file, plan, inputs, 0, host, residency, NonZeroUsize::MIN)
    }

    fn batch(
        gpu: &Gpu,
        w: &Weights,
        body: &mut Body,
        head: &mut Head,
        ids: &[u32],
        at: u32,
    ) -> Result<bool, GpuError> {
        prefill(gpu, w, body, head, ids, at)
    }

    fn group(m: &mut GpuModel<Body>, g: usize) -> Result<bool, GpuError> {
        bloomery_gpu_minimaxm3::prefill::set_prefill_group(m, g)
    }
}
