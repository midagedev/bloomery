//! The Kimi K3 session: [`Body`] behind the common hybrid owner's shape ([`super::Batch`]), planned
//! once by its placement (the dense trunk and each routed layer's card experts on the stage card,
//! the next ids on an expert tier card, the rest on the host tier) and loaded by the body's placed
//! constructor ([`GpuModel::open_placed_with`]) under the session's residency word. A load with
//! the placed pass runs a prompt as passes of
//! its rows ([`GpuModel::prompt_placed`]), past a K3.1 file's indexer top-k too (each row's latent
//! layers attend the rows the indexer picks for it); one without the pass steps the ids one at a
//! time ([`GpuModel::step`] over the slice, [`Batch::prompt`]). Either way each id is a decode
//! step's bits and the head's argmax after the last is the call's. A delta layer's state holds the
//! position it stands at alone, so a cut keeps every held position or nothing; the latent layers'
//! caches are per position and ask nothing of it. The body serves one sequence. It lives in the
//! gpu crate's own kimi_k3 arch, so it needs no feature.

use std::num::NonZeroUsize;

use bloomery_gpu::arch::kimi_k3::Body;
use bloomery_gpu::folder::Model;
use bloomery_gpu::head::Head;
use bloomery_gpu::host::swap::Residency;
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{Gpu, GpuError, GpuModel};
use bloomery_levers::HostCfg;
use gguf::Split;
use model::arch::kimi_k3::place::PlanInputs;
use model::placement::Plan;

use super::{Batch, SingleSequence};

const WHAT: &str = "kimi_k3 session";

impl Batch for Body {
    type Inputs = PlanInputs;
    type Sequences = SingleSequence;

    const WHAT: &'static str = self::WHAT;

    /// The file as the plan reads it for a driver that renders no chat: every part the body does
    /// not run is refused by name, the template's tool-call parser excepted (a request that asks
    /// for tool calls is the server's refusal). A K3.1 file runs every context
    /// ([`PlanInputs::check_ctx`]): past its indexer's top-k the latent layers attend the rows the
    /// indexer picks, at a step and at every row of a pass.
    fn read(file: &Split) -> Result<PlanInputs, GpuError> {
        PlanInputs::read_without_chat(file).map_err(|e| GpuError::plan(WHAT, e))
    }

    /// The placed load on stage card 0 under `residency` ([`GpuModel::open_placed_with`]): a `mid`
    /// word runs the common machine over the card's routed stacks and the slot map the step reads.
    fn open_placed(
        file: Model,
        plan: &Plan<'_>,
        inputs: &PlanInputs,
        host: HostCfg,
        residency: Residency,
        _slots: NonZeroUsize,
    ) -> Result<GpuModel<Body>, GpuError> {
        GpuModel::<Body>::open_placed_with(
            crate::split_of(file, WHAT)?,
            (plan, inputs),
            0,
            host,
            (None, residency),
        )
    }

    /// Refused by name: the placed load has no rows pass, so the prompt is [`Batch::prompt`]'s
    /// steps and nothing calls a batch.
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
                "a batch of {} ids: the body has no rows pass on the placed load, its prompt \
                 steps one id at a time",
                ids.len()
            ),
        ))
    }

    /// No prompt group: the ids are stepped.
    fn group(_m: &mut GpuModel<Body>, _g: usize) -> Result<bool, GpuError> {
        Ok(false)
    }

    /// The ids from the model's position: as placed passes when the load has one
    /// ([`GpuModel::prompt_placed`]: the call one residency pass, 0 rows kept), on a K3.1 file
    /// past its indexer's top-k too (every pass row selects as the step at its position does),
    /// else stepped, each step owning its residency pass (its boundary, its kept row, the next
    /// boundary ahead), so no outer [`prompt_pass`](bloomery_gpu::host::served::prompt_pass)
    /// wraps the ids: a host tier's open-call state refuses a boundary inside an open call.
    fn prompt(m: &mut GpuModel<Body>, ids: &[u32]) -> Result<u32, GpuError> {
        if m.body("kimi_k3::Batch::prompt")?.pass_shape().is_none() {
            return m.step(ids);
        }
        m.prompt_placed(ids)
    }

    /// The delta layers' state holds the position it stands at alone: a cut keeps that position,
    /// or nothing.
    fn keepable(m: &GpuModel<Body>, n: u32) -> u32 {
        crate::keep_where_it_stands(m, n)
    }

    /// Back to empty at 0 (a reset), or the position the model stands at (nothing to take back);
    /// any other cut is refused by name, since the body takes no position back.
    fn cut(m: &mut GpuModel<Body>, n: u32) -> Result<(), GpuError> {
        let why = "the delta layers' state takes no position back";
        crate::cut_where_it_stands(m, n, (WHAT, why))
    }
}
