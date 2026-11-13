//! The seat the exaone-moe gates drive the body through (`gate_exaone_e2e`, `gate_exaone_pp`): the
//! load of K-EXAONE's body (`bloomery_gpu_flint::ExaoneBody`, the composed head128 program over
//! the biased top-eight router instance) on the gate card and the [`Seat`] calls over it
//! ([`crate::lcpp_e2e`] holds the trait and its contract), and the tap ceilings of the fixture
//! GGUF's layers. A gate includes it by `#[path]` beside `shared/lcpp_e2e.rs` and
//! `shared/gate_card.rs`.
//!
//! The prompt call is the body's own ([`bloomery_gpu_flint::prefill::prefill`] through
//! [`GpuModel::run_rows`], a prompt batch of up to
//! `model::arch::flint::place::PROMPT_ROWS` rows a unit); [`Seat::stores`] is refused by name: the
//! body holds its per-layer stores private, and no clause of these gates reads a store digest.

use crate::lcpp_e2e::{Arm, Dims, Seat};
use bloomery_gpu::host::served::prompt_pass;
use bloomery_gpu::model::StepMode;
use bloomery_gpu_flint::{ExaoneBody, ExaoneModel, prefill, set_taps};
use bloomery_gpu_gates::GateError;
use bloomery_gpu_gates::prompts::SIGMA_LOGIT_V41;
use bloomery_gpu_gates::tier::Tier;
use gguf::Split;
use model::arch::flint::place::{self, PlanInputs};
use model::placement::PlanLevers;
use model::placement::workstation::{HostRead, PLAN_ROOM};
use std::path::Path;

/// The layers of the fixture GGUF's trunk: the last is the head's input, held by the logits and
/// logprob clauses, not a layer band (the Kimi K3 e2e gate's rule for its last layer).
pub const TAP_LAST: usize = 5;

/// The two ceilings of a layer output's relative distance between two implementations of the
/// model, over the layers before the last: `TAP_EARLY_CEIL` over the dense lead (the first two
/// layers), `TAP_CEIL` after.
// PIN(2026-11-13): provisional, the Kimi K3 fixture e2e gate's two ceilings (its `LAYER_REL_BAND`
// 5e-2 and 2e-1, measured on a cut of real weights of the same class); re-pinned from this gate's
// first run on the box, which prints every layer's distance, never tuned to pass.
pub const TAP_EARLY_CEIL: f64 = 5.0e-2;
pub const TAP_CEIL: f64 = 2.0e-1;

/// The standard deviation, in logit units, of one id's logit between two implementations of the
/// model: V4.1's, until K-EXAONE's own is measured on the box (the first run prints σ̂ a set, the
/// distance of our logit from the oracle's over its top ids).
pub const SIGMA_LOGIT: f64 = SIGMA_LOGIT_V41;

/// The last-position logits row's relative distance from the oracle's (the Kimi K3 fixture e2e
/// gate's band): the row is closer to the oracle's than the zero row is.
// PIN(2026-11-13): the Kimi K3 fixture e2e gate's `LOGITS_REL_BAND`, 1.0.
pub const LOGITS_REL_BAND: f64 = 1.0;

/// The ceiling of layer `layer`'s output distance: the last trunk layer is held by the logits
/// row's own clauses (the logprob band and the teacher-forced argmax pin), not a layer band.
pub fn tap_ceil(_tier: Tier, layer: usize) -> f64 {
    if layer >= TAP_LAST {
        f64::INFINITY
    } else if layer < 2 {
        TAP_EARLY_CEIL
    } else {
        TAP_CEIL
    }
}

/// The ceilings of [`tap_ceil`] as a gate's line prints them.
pub fn tap_ceils(_tier: Tier) -> String {
    format!(
        "ceilings {TAP_EARLY_CEIL} (layers < 2) and {TAP_CEIL} (to layer {}), the last layer the \
         logits clauses'",
        TAP_LAST - 1
    )
}

/// Open `file` at `ctx` positions by the placement of `arm`: the gate placement
/// (`gate_card::plan_gate`) on the gate card for `a`, plan (b′) (`workstation::plan_bp`) for `bp`,
/// the next-largest card an expert tier under the host tier holding the tier's prompt-batch bytes
/// (`model::arch::flint::place::tier_batch`). The body is opened as the serve seat opens it
/// (`ExaoneBody::open_placed`, the plan `PlanInputs::plan` makes), the taps armed before the step
/// is first captured, so a replay and a prompt batch both write them.
///
/// The including gate declares `shared/gate_card.rs` as `gate_card` beside this module.
pub fn open(file: &Path, ctx: usize, arm: Arm) -> Result<Box<dyn Seat>, GateError> {
    Ok(Box::new(open_loaded(file, ctx, arm)?))
}

/// [`open`]'s seat as the body behind it, for a gate that drives the body between the seat's
/// calls.
pub fn open_loaded(file: &Path, ctx: usize, arm: Arm) -> Result<Loaded, GateError> {
    let levers = bloomery_levers::at_main(&[bloomery_levers::CARD_BUDGET])?;
    let split = Split::open(file).map_err(|e| format!("open {}: {e}", file.display()))?;
    // The file as it is: the gates feed ids, so the features of the file that only the tokenizer
    // and the tool-call parser carry (`PlanInputs::read` lists them as unimplemented) are not the
    // gate's. The plan this gate holds its pins to is the unpaged one at the workstation's plan
    // room: no reading of the box's available bytes enters it.
    let mut inputs = PlanInputs::describe(&split)?;
    inputs.room = (PLAN_ROOM, HostRead::Given);
    let machine = arm.machine(inputs.model.layers, place::tier_batch(&inputs)?)?;
    let place = PlanLevers::from_levers(&levers)?;
    let plan = inputs.plan(&machine, u64::try_from(ctx)?, &place)?;
    let mut m = ExaoneBody::open_placed(split, &plan, &inputs, 0, levers.host())?;
    // Armed under eager steps, then captured with the taps.
    set_taps(&mut m, true)?;
    m.set_mode(StepMode::Graph);
    let dims = Dims {
        layers: m.layers().len(),
        embd: inputs.spec.hidden as usize,
        vocab: inputs.spec.vocab as usize,
    };
    Ok(Loaded { m, dims })
}

/// The body on the gate card behind the seat's calls.
pub struct Loaded {
    m: ExaoneModel,
    dims: Dims,
}

impl Loaded {
    /// The seat's dimensions.
    pub fn dims(&self) -> Dims {
        self.dims
    }

    /// The last call's logits row.
    fn row(&self) -> Result<Vec<f32>, GateError> {
        let mut row = vec![0.0; self.dims.vocab];
        self.m.logits_into(&mut row)?;
        Ok(row)
    }

    /// One batched pass of `ids` from the seat's position ([`prefill`] through
    /// [`GpuModel::run_rows`], one residency pass over it): its last row's
    /// logits and argmax, `None` when the pass ran no head.
    pub fn pass(&mut self, ids: &[u32]) -> Result<Option<(Vec<f32>, u32)>, GateError> {
        let token = prompt_pass(&mut self.m, |m| {
            m.run_rows(ids.len(), "exaone seat", |gpu, w, body, head, at| {
                prefill(gpu, w, body, head, ids, at)
            })
        })?;
        match token {
            Some(t) => Ok(Some((self.row()?, t))),
            None => Ok(None),
        }
    }
}

impl Seat for Loaded {
    fn dims(&self) -> Dims {
        self.dims
    }

    fn reset(&mut self) -> Result<(), GateError> {
        Ok(self.m.reset()?)
    }

    fn pos(&self) -> u32 {
        self.m.pos()
    }

    fn step(&mut self, id: u32) -> Result<Vec<f32>, GateError> {
        self.m.step(&[id])?;
        self.row()
    }

    /// The prompt in one batched call ([`prefill`] through [`GpuModel::run_rows`]), one residency
    /// pass over it (`prompt_pass`).
    fn prompt(&mut self, ids: &[u32]) -> Result<Vec<f32>, GateError> {
        prompt_pass(&mut self.m, |m| {
            m.run_rows(ids.len(), "exaone seat", |gpu, w, body, head, at| {
                prefill(gpu, w, body, head, ids, at)
            })
        })?;
        self.row()
    }

    fn set_group(&mut self, group: usize) -> Result<(), GateError> {
        bloomery_gpu_flint::set_prefill_group(&mut self.m, group)?;
        Ok(())
    }

    fn taps(&mut self) -> Result<Vec<f32>, GateError> {
        let (gpu, _, body) = self.m.body_parts("exaone seat")?;
        Ok(body.taps(gpu)?)
    }

    fn stores(&mut self) -> Result<Vec<(String, u64)>, GateError> {
        Err("this seat names no store digest: the flint body holds its stores private".into())
    }

    fn fault(&mut self) -> Result<(), GateError> {
        if let Some(fault) = self.m.poisoned() {
            return Err(format!("the model is poisoned by {fault:?}").into());
        }
        let (gpu, _, _) = self.m.body_parts("exaone seat")?;
        match gpu.fault()? {
            None => Ok(()),
            Some(fault) => Err(format!("the card's fault word holds {fault:?}").into()),
        }
    }

    /// The head alone (the final norm and the projection) over one row, by the engine's own head:
    /// the minimaxm3 seat's probe.
    fn head_row(&mut self, x: &[f32]) -> Result<Vec<f32>, GateError> {
        self.m.reset()?;
        self.m
            .run_rows(1, "exaone head probe", |gpu, w, _body, head, _pos| {
                head.set_input(gpu, x)?;
                head.enqueue(gpu, w)?;
                Ok(true)
            })?;
        let row = self.row()?;
        self.m.reset()?;
        Ok(row)
    }
}
