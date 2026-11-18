//! The Hy4-flash session: [`Body`] behind the common hybrid owner's shape ([`super::Batch`]),
//! planned once by the common hybrid planner (every layer's dense part and the stores on the stage
//! card, each routed layer's card share of its experts beside them, the rest on the host tier) and
//! loaded by the body's placed constructor ([`Body::open_placed`], or [`Body::open_placed_pass`]
//! with the placed prompt pass). A load with the pass runs a prompt as passes of its rows
//! ([`prompt_placed`]), the host's experts read once a pass; one without it steps the ids one at a
//! time ([`GpuModel::step`] over the slice). Either way the head's argmax after the last id is the
//! call's. A window layer's ring and the engram history hold the
//! position they stand at alone, so a cut keeps every held position or nothing. Each resident
//! sequence owns its stores and its engram history ([`ResidentSequences`]).

use std::num::NonZeroUsize;

use bloomery_gpu::head::Head;
use bloomery_gpu::host::swap::Residency;
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{Gpu, GpuError, GpuModel};
use bloomery_gpu_hy4::{Body, prompt_placed};
use bloomery_levers::HostCfg;
use gguf::Split;
use model::arch::hy4::place::PlanInputs;
use model::placement::Plan;

use super::{Batch, ResidentSequences};

const WHAT: &str = "hy4 session";

impl Batch for Body {
    type Inputs = PlanInputs;
    type Sequences = ResidentSequences;

    const WHAT: &'static str = self::WHAT;

    /// The file as the plan reads it: every part the body does not run is refused by name.
    fn read(file: &Split) -> Result<PlanInputs, GpuError> {
        PlanInputs::read(file).map_err(|e| GpuError::plan(WHAT, e))
    }

    /// The placed load on stage card 0 ([`Body::open_placed`]). The body keeps the load's slot
    /// map for its life, so a residency other than [`Residency::Off`] is refused by name, never
    /// run as another. The load makes one sequence; the session adds the rest
    /// ([`ResidentSequences`]).
    fn open_placed(
        file: Split,
        plan: &Plan<'_>,
        inputs: &PlanInputs,
        host: HostCfg,
        residency: Residency,
        _slots: NonZeroUsize,
    ) -> Result<GpuModel<Body>, GpuError> {
        super::slot_map_for_life(WHAT, residency)?;
        Body::open_placed(file, plan, inputs, 0, host)
    }

    /// Refused by name: the placed pass is the body's own walk ([`Batch::prompt`]), so nothing
    /// calls a batch.
    fn batch(
        _gpu: &Gpu,
        _w: &Weights,
        _body: &mut Body,
        _head: &mut Head,
        ids: &[u32],
        _at: u32,
    ) -> Result<bool, GpuError> {
        Err(GpuError::shape(
            WHAT,
            format!(
                "a batch of {} ids: the body has no rows pass, its prompt is the placed pass or steps",
                ids.len()
            ),
        ))
    }

    /// No prompt group: the ids are the pass's or stepped.
    fn group(_m: &mut GpuModel<Body>, _g: usize) -> Result<bool, GpuError> {
        Ok(false)
    }

    /// The ids from the model's position: as placed passes when the load has one
    /// ([`prompt_placed`]), else stepped. The body's residency is [`Residency::Off`], so no rule
    /// counts the prompt ids' decode rows.
    fn prompt(m: &mut GpuModel<Body>, ids: &[u32]) -> Result<u32, GpuError> {
        if m.body("hy4::Batch::prompt")?.pass_rows().is_none() {
            return m.step(ids);
        }
        prompt_placed(m, ids)
    }

    /// The window rings and the engram history hold the position they stand at alone: a cut
    /// keeps that position, or nothing.
    fn keepable(m: &GpuModel<Body>, n: u32) -> u32 {
        crate::keep_where_it_stands(m, n)
    }

    /// Back to empty at 0 (a reset), or the position the model stands at (nothing to take back);
    /// any other cut is refused by name, since the body takes no position back.
    fn cut(m: &mut GpuModel<Body>, n: u32) -> Result<(), GpuError> {
        let why = "the window rings and the engram history take no position back";
        crate::cut_where_it_stands(m, n, (WHAT, why))
    }
}
