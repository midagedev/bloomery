//! The ERNIE 5.5 Flash session: [`Body`] behind the common hybrid owner's shape ([`super::Batch`]),
//! planned once by the common routed planner over the family's [`PlanInputs`] (every layer's dense
//! part and the stores on the stage card, each routed layer's card share of its experts by the
//! common card leg's rule, the rest on the host tier) and loaded by the body's placed constructor
//! ([`Body::open_placed`]) under the session's residency word. `file` is a GGUF split or a
//! checkpoint folder: the body reads either. A plan whose card holds every expert loads the prompt
//! pass and runs a prompt as its units ([`prompt`]); a plan that hosts experts steps the ids one at
//! a time ([`GpuModel::step`]). Either way the head's argmax after the last id is the call's. An HCA
//! layer's window ring holds the position it stands at alone, so a cut keeps every held position
//! or nothing. One resident sequence: the parts hold one sequence's caches.

use std::num::NonZeroUsize;

use bloomery_gpu::folder::Model;
use bloomery_gpu::head::Head;
use bloomery_gpu::host::swap::Residency;
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{Gpu, GpuError, GpuModel};
use bloomery_gpu_ernie55::body::Body;
use bloomery_gpu_ernie55::{PASS_ROWS, prompt};
use bloomery_levers::HostCfg;
use gguf::Split;
use model::arch::ernie55::place::PlanInputs;
use model::placement::Plan;

use super::{Batch, SingleSequence};

const WHAT: &str = "ernie55 session";

impl Batch for Body {
    type Inputs = PlanInputs;
    type Sequences = SingleSequence;

    const WHAT: &'static str = self::WHAT;

    /// The file as the plan reads it: every part the program does not run is refused by name.
    fn read(file: &Split) -> Result<PlanInputs, GpuError> {
        PlanInputs::read(file).map_err(|e| GpuError::plan(WHAT, e))
    }

    /// The placed load on stage card 0, with the prompt pass where the plan's card holds every
    /// expert ([`Body::plan_hosts`]). An MTP draft is the default refusal: the file carries none.
    fn open_placed(
        file: Model,
        plan: &Plan<'_>,
        inputs: &PlanInputs,
        host: HostCfg,
        residency: Residency,
        _slots: NonZeroUsize,
    ) -> Result<GpuModel<Body>, GpuError> {
        let rows = (!Body::plan_hosts(plan, inputs)).then_some(PASS_ROWS);
        Body::open_placed(file, (plan, inputs), 0, host, (rows, residency))
    }

    /// Refused by name: the prompt is [`Batch::prompt`]'s pass units or steps, so nothing calls a
    /// batch.
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
                "a batch of {} ids: the prompt runs as the pass's units or one id at a time",
                ids.len()
            ),
        ))
    }

    /// No prompt group: a prompt runs as the pass's units or stepped.
    fn group(_m: &mut GpuModel<Body>, _g: usize) -> Result<bool, GpuError> {
        Ok(false)
    }

    /// The ids as the pass's units where the load has the pass ([`prompt`]), else stepped.
    fn prompt(m: &mut GpuModel<Body>, ids: &[u32]) -> Result<u32, GpuError> {
        if m.body(WHAT)?.pass_rows().is_none() {
            return m.step(ids);
        }
        prompt(m, ids)
    }

    /// The window rings hold the position they stand at alone: a cut keeps that position, or
    /// nothing.
    fn keepable(m: &GpuModel<Body>, n: u32) -> u32 {
        crate::keep_where_it_stands(m, n)
    }

    /// Back to empty at 0 (a reset), or the position the model stands at; any other cut is
    /// refused by name.
    fn cut(m: &mut GpuModel<Body>, n: u32) -> Result<(), GpuError> {
        let why = "the HCA window rings take no position back";
        crate::cut_where_it_stands(m, n, (WHAT, why))
    }
}
