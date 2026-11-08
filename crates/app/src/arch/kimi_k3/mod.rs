//! The Kimi K3 session: [`Body`] behind the common hybrid owner's shape ([`super::Batch`]), planned
//! once by its placement (the dense trunk and each routed layer's card experts on the stage card,
//! the next ids on an expert tier card, the rest on the host tier) and loaded by the body's placed
//! constructor ([`GpuModel::open_placed`]). A load with the placed pass runs a prompt as passes of
//! its rows ([`GpuModel::prompt_placed`]), past a K3.1 file's indexer top-k too (each row's latent
//! layers attend the rows the indexer picks for it); one without the pass steps the ids one at a
//! time ([`GpuModel::step`] over the slice, [`Batch::prompt`]). Either way each id is a decode
//! step's bits and the head's argmax after the last is the call's. A delta layer's state holds the
//! position it stands at alone, so a cut keeps every held position or nothing; the latent layers'
//! caches are per position and ask nothing of it. The body serves one sequence. It lives in the
//! gpu crate's own kimi_k3 arch, so it needs no feature.

use std::num::NonZeroUsize;

use bloomery_gpu::arch::kimi_k3::Body;
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

    /// The placed load on stage card 0 ([`GpuModel::open_placed`]). The body keeps the load's slot
    /// map for its life, so a residency other than [`Residency::Off`] is refused by name, never
    /// run as another.
    fn open_placed(
        file: Split,
        plan: &Plan<'_>,
        inputs: &PlanInputs,
        host: HostCfg,
        residency: Residency,
        _slots: NonZeroUsize,
    ) -> Result<GpuModel<Body>, GpuError> {
        if residency != Residency::Off {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "residency {residency:?}: the body keeps the load's slot map for its life \
                     (BLOOMERY_RESIDENCY off)"
                ),
            ));
        }
        GpuModel::<Body>::open_placed(file, plan, inputs, 0, host)
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
    /// ([`GpuModel::prompt_placed`]), on a K3.1 file past its indexer's top-k too (every pass row
    /// selects as the step at its position does), else stepped. Each pass or step owns its
    /// residency pass (its
    /// boundary, its kept row, the next boundary ahead), and a host tier's open-call state refuses
    /// a boundary inside an open call, so no outer
    /// [`prompt_pass`](bloomery_gpu::host::served::prompt_pass) wraps the ids. The body's
    /// residency is [`Residency::Off`], so no rule counts the prompt ids' decode rows.
    fn prompt(m: &mut GpuModel<Body>, ids: &[u32]) -> Result<u32, GpuError> {
        if m.body("kimi_k3::Batch::prompt")?.pass_shape().is_none() {
            return m.step(ids);
        }
        m.prompt_placed(ids)
    }

    /// The delta layers' state holds the position it stands at alone: a cut keeps that position,
    /// or nothing.
    fn keepable(m: &GpuModel<Body>, n: u32) -> u32 {
        if n >= m.pos() { m.pos() } else { 0 }
    }

    /// Back to empty at 0 (a reset), or the position the model stands at (nothing to take back);
    /// any other cut is refused by name, since the body takes no position back.
    fn cut(m: &mut GpuModel<Body>, n: u32) -> Result<(), GpuError> {
        match n {
            0 => m.reset(),
            n if n >= m.pos() => Ok(()),
            n => Err(GpuError::shape(
                WHAT,
                format!(
                    "a cut to position {n} of {}: the delta layers' state takes no position back",
                    m.pos()
                ),
            )),
        }
    }
}
