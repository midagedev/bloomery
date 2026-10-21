//! The Solar-Open2 session: [`Body`] behind the session's traits, planned
//! once by its placement (the routed experts the card experts read by the
//! expert rule, the next ids on the expert tier card a placement names, the
//! rest on the host tier) and loaded by gpumodel's constructor
//! ([`Body::open_placed_slots`]), which hangs the tier card under the host
//! tier, under adaptive expert residency or not ([`HybridCfg::residency`]).
//!
//! A prompt is fed in batches ([`bloomery_gpu_solar2::prefill::prefill`]),
//! one residency pass a call, and leaves the bits the decode steps leave. The
//! 36 KDA layers hold one recurrent state each, written in place by every
//! position, so a cut keeps every held position, the empty model, or a
//! checkpoint ([`Body::kept`]); the 12 GQA layers' K/V planes are cut by
//! position and ask nothing of it. The body serves one sequence.
//!
//! The session's shape is the common owner's ([`super::Batch`]): what the
//! placement plans, how the prompt runs and how a cut is kept are shared with
//! the other families that run one chain on one card. This module holds what
//! Solar's architecture forces: the plan inputs, the load, the batch and the
//! cut rule of a body whose recurrent state keeps its history in checkpoints.

use std::num::NonZeroUsize;

use bloomery_gpu::head::Head;
use bloomery_gpu::host::swap::Residency;
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{Gpu, GpuError, GpuModel};
use bloomery_gpu_solar2::Body;
use bloomery_gpu_solar2::prefill::prefill;
use bloomery_levers::HostCfg;
use gguf::Split;
use model::arch::solar2::place::PlanInputs;
use model::placement::Plan;
use runtime::seqstate::Kept;

use super::{Batch, cut_checkpointed};
use crate::Keep;

/// The session's name in its refusals.
const WHAT: &str = "solar2 session";

impl Batch for Body {
    type Inputs = PlanInputs;

    const WHAT: &'static str = self::WHAT;

    fn read(file: &Split) -> Result<PlanInputs, GpuError> {
        PlanInputs::read(file).map_err(|e| GpuError::plan(WHAT, e))
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
        bloomery_gpu_solar2::set_prefill_group(m, g)
    }

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

    /// [`cut_checkpointed`]: the checkpoints are the only positions a cut
    /// takes back.
    fn cut(m: &mut GpuModel<Body>, n: u32) -> Result<(), GpuError> {
        cut_checkpointed(m, n)
    }
}
