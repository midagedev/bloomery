//! The Hy4-flash session: [`Body`] behind the common hybrid owner's shape ([`super::Batch`]),
//! planned once by the common hybrid planner (every layer's dense part and the stores on the stage
//! card, each routed layer's card share of its experts beside them, the rest on the host tier) and
//! loaded by the body's placed constructor ([`Body::open_placed_with`], with or without the placed
//! prompt pass) under the session's residency word. A load with the pass runs a prompt as passes
//! of its rows ([`prompt_placed`], one residency pass), the host's experts read once a pass; one
//! without it steps the ids one at a time ([`GpuModel::step`] over the slice). Either way the head's argmax after the last id is the
//! call's. A window layer's ring, a sparse layer's index ring and the engram history hold the
//! position they stand at alone, so a cut keeps every held position, the empty model, or a
//! checkpoint: each prompt call takes one at its start and at its end ([`Body::kept`]), which a
//! request that extends the last prompt returns to. Each resident sequence owns its stores, its
//! engram history and its checkpoints ([`ResidentSequences`]).

use std::num::NonZeroUsize;

use bloomery_gpu::checkpoint::prompt_marked;
use bloomery_gpu::folder::Model;
use bloomery_gpu::head::Head;
use bloomery_gpu::host::swap::Residency;
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{Gpu, GpuError, GpuModel};
use bloomery_gpu_hy4::{Body, prompt_placed};
use bloomery_levers::HostCfg;
use gguf::Split;
use model::arch::hy4::place::PlanInputs;
use model::placement::Plan;
use runtime::seqstate::Kept;

use super::{Batch, ResidentSequences, cut_checkpointed};
use crate::Keep;

const WHAT: &str = "hy4 session";

impl Batch for Body {
    type Inputs = PlanInputs;
    type Sequences = ResidentSequences;

    const WHAT: &'static str = self::WHAT;

    /// The file as the plan reads it: every part the body does not run is refused by name.
    fn read(file: &Split) -> Result<PlanInputs, GpuError> {
        PlanInputs::read(file).map_err(|e| GpuError::plan(WHAT, e))
    }

    /// The placed load on stage card 0 under `residency` ([`Body::open_placed_with`]): a `mid`
    /// word runs the common machine over the card's routed stacks and the slot map the step reads.
    /// The load makes one sequence; the session adds the rest ([`ResidentSequences`]).
    fn open_placed(
        file: Model,
        plan: &Plan<'_>,
        inputs: &PlanInputs,
        host: HostCfg,
        residency: Residency,
        _slots: NonZeroUsize,
    ) -> Result<GpuModel<Body>, GpuError> {
        Body::open_placed_with(
            crate::split_of(file, WHAT)?,
            (plan, inputs),
            0,
            host,
            (None, residency),
        )
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

    /// The ids from the model's position, and the call's checkpoints ([`prompt_marked`]): as
    /// placed passes when the load has one ([`prompt_placed`]: the call one residency pass, 0 rows
    /// kept), else stepped, each step owning its residency pass, so no outer
    /// [`prompt_pass`](bloomery_gpu::host::served::prompt_pass) wraps the ids. An image prompt
    /// takes no checkpoint.
    fn prompt(m: &mut GpuModel<Body>, ids: &[u32]) -> Result<u32, GpuError> {
        prompt_marked(m, ids, |m, seg| {
            if m.body("hy4::Batch::prompt")?.pass_rows().is_none() {
                return m.step(seg);
            }
            prompt_placed(m, seg)
        })
    }

    /// [`Body::kept`]: every position, the empty model, or the nearest checkpoint at or below
    /// `n`.
    fn keepable(m: &GpuModel<Body>, n: u32) -> u32 {
        <Body as Keep>::kept(m, n).at
    }

    /// [`Body::kept`].
    fn kept(m: &GpuModel<Body>, n: u32) -> Kept {
        m.body(WHAT)
            .map_or_else(|_| Kept::rule(n, m.pos(), 0), |b| b.kept(n, m.pos()))
    }

    /// [`cut_checkpointed`]: the checkpoints are the only positions a cut takes back.
    fn cut(m: &mut GpuModel<Body>, n: u32) -> Result<(), GpuError> {
        cut_checkpointed(m, n)
    }
}
