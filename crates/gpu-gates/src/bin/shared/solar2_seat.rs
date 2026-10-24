//! The seat the Solar-Open2 gates drive the body through (`gate_solar2_e2e`, `gate_solar2_prefill`): the load of the body on the
//! gate card and the [`Seat`] calls over it ([`crate::lcpp_e2e`] holds the trait and its contract), and the tap ceilings of
//! Solar's layers. A gate includes it by `#[path]` beside `shared/lcpp_e2e.rs` and `shared/gate_card.rs`.
//!
//! The prompt call and the stores are the body's own ([`bloomery_gpu_solar2`]): [`Seat::prompt`] runs `ids` from the seat's
//! position by the prompt batch (up to `bloomery_gpu_solar2::body::T_MAX` rows a unit); [`Seat::stores`] names each KDA layer's
//! state, stamp and conv ring and each GQA layer's K and V planes over the written positions.

use crate::lcpp_e2e::{self, Arm, Dims, Seat, fnv1a};
use bloomery_gpu::model::StepMode;
use bloomery_gpu_gates::GateError;
use bloomery_gpu_gates::tier::Tier;
use bloomery_gpu_solar2::{Body, Solar2Model, set_prefill_group, set_taps};
use gguf::Split;
use model::arch::solar2::place::{self, PlanInputs};
use model::placement::PlanLevers;
use model::placement::workstation::{self, HostRead, PLAN_ROOM};
use refset::arch::solar2::ARCH;
use std::path::{Path, PathBuf};

/// The layers before the real file's stream carries its outlier channels: PAI's own CPU and GPU runs agree to a few percent there.
pub const TAP_EARLY: usize = 4;

/// The two ceilings of a layer output's relative distance between two implementations of the model, in the real file:
/// `TAP_EARLY_CEIL` over the first [`TAP_EARLY`] layers, `TAP_CEIL` after.
///
/// [derived]: PAI's CPU dump against its hybrid GPU run of the same prompt (the oracle round's per-tensor last-row sums of
/// squares and first 8 values) gives the norm distance at most 0.024 and, from the 8 values, an estimate of
/// `‖Δ‖ / ‖ref‖` (the rms-normalised error, if the error is spread evenly) of 0.008-0.022 at layers 0-3 and 0.034-0.158
/// from layer 4 on (median 0.106): the stream grows its outlier channels there and the same roundings weigh more. Ours
/// against PAI is a third implementation of that kind, so each ceiling is about twice its stretch's worst. A beta that is
/// not doubled, or a decay of another form, moves layer 1 by the order of its own contribution, past the early ceiling.
// PIN(2026-10-20): provisional until the first body runs on the box; the first run prints every layer's distance and the
// ceilings are re-pinned there with the full-row distances.
pub const TAP_EARLY_CEIL: f64 = 0.05;
pub const TAP_CEIL: f64 = 0.3;

/// The ceiling of layer `layer`'s output distance in `tier`'s file.
///
/// The fixture has no early stretch: its four layers hold random weights, so nothing damps the q8 re-rounding of each layer's
/// input and it compounds from the first layer on. Ours against PAI reads 0.030 at layer 0 and 0.105-0.136 at layer 3 over the
/// fixture's three prompts with layer outputs; the prompt batch against the steps reads 0.010-0.016 at layer 0 and 0.045-0.163
/// at layer 3 over the prefill gate's prompts. [`TAP_EARLY_CEIL`] is under both from layer 1 on. Every fixture layer takes
/// [`TAP_CEIL`], 1.8 times the larger worst: the margin the real ceilings have over their own stretch's worst.
// PIN(2026-10-21): new for the fixture tier, not a relaxation (the real tier's two ceilings are as they were); derived above.
pub fn tap_ceil(tier: Tier, layer: usize) -> f64 {
    match tier {
        Tier::Real if layer < TAP_EARLY => TAP_EARLY_CEIL,
        Tier::Real | Tier::Fixture => TAP_CEIL,
    }
}

/// The ceilings of [`tap_ceil`] as a gate's line prints them.
pub fn tap_ceils(tier: Tier) -> String {
    match tier {
        Tier::Real => format!("ceilings {TAP_EARLY_CEIL} (layers < {TAP_EARLY}) and {TAP_CEIL}"),
        Tier::Fixture => format!("ceiling {TAP_CEIL} (every layer)"),
    }
}

/// The model file the profile names, checked before anything loads ([`lcpp_e2e::model_path`]): a solar-open2 file.
pub fn model_path(tier_now: Tier, recipe: &str) -> Result<PathBuf, GateError> {
    lcpp_e2e::model_path(tier_now, recipe, ARCH)
}

/// Open `file` at `ctx` positions by the placement of `arm`: the gate placement (`gate_card::plan_gate`) on the gate card for
/// `a`, plan (b′) (`workstation::plan_bp`) for `bp`. The body is opened as the serve seat opens it (`Body::open_placed`, the
/// plan `PlanInputs::plan` makes), the taps armed before the step is first captured, so a replay and a prompt batch both write
/// them.
///
/// The including gate declares `shared/gate_card.rs` as `gate_card` beside this module.
pub fn open(file: &Path, ctx: usize, arm: Arm) -> Result<Box<dyn Seat>, GateError> {
    let levers = bloomery_levers::at_main(&[bloomery_levers::CARD_BUDGET])?;
    crate::gate_card::init()?;
    let split = Split::open(file).map_err(|e| format!("open {}: {e}", file.display()))?;
    // The file as it is: the gates feed ids, so the features of the file that only the tokenizer and the tool-call parser
    // carry (`PlanInputs::read` lists them as unimplemented) are not the gate's. The plan this gate holds its pins to is the
    // unpaged one at the workstation's plan room: no reading of the box's available bytes enters it.
    let mut inputs = PlanInputs::describe(&split)?;
    inputs.room = (PLAN_ROOM, HostRead::Given);
    // Plan (b′): the stage on the largest card, the next-largest an expert tier under the host tier, holding the tier's
    // prompt-batch bytes the file's widths give.
    let machine = match arm {
        Arm::A => crate::gate_card::plan_gate(inputs.model.layers),
        Arm::Bp => workstation::plan_bp(inputs.model.layers, None, place::tier_batch(&inputs)?),
    };
    let place = PlanLevers::from_levers(&levers)?;
    let plan = inputs.plan(&machine, u64::try_from(ctx)?, &place)?;
    let mut m = Body::open_placed(split, &plan, &inputs, 0, levers.host())?;
    // Armed under eager steps, then captured with the taps.
    set_taps(&mut m, true)?;
    m.set_mode(StepMode::Graph);
    let (layers, vocab, embd) = {
        let (_, _, b) = m.body_parts("solar2 seat")?;
        let d = b.dims();
        (b.kinds().len(), d.vocab, d.embd)
    };
    Ok(Box::new(Loaded {
        m,
        dims: Dims {
            layers,
            embd,
            vocab,
        },
    }))
}

/// The body on the gate card behind the seat's calls.
struct Loaded {
    m: Solar2Model,
    dims: Dims,
}

impl Loaded {
    /// The last call's logits row.
    fn row(&self) -> Result<Vec<f32>, GateError> {
        let mut row = vec![0.0; self.dims.vocab];
        self.m.logits_into(&mut row)?;
        Ok(row)
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

    /// The prompt in batches (`app::Prompt for Body`), one residency pass a call.
    fn prompt(&mut self, ids: &[u32]) -> Result<Vec<f32>, GateError> {
        <Body as app::Prompt>::prompt(&mut self.m, ids)?;
        self.row()
    }

    fn set_group(&mut self, group: usize) -> Result<(), GateError> {
        set_prefill_group(&mut self.m, group)?;
        Ok(())
    }

    fn taps(&mut self) -> Result<Vec<f32>, GateError> {
        let (gpu, _, body) = self.m.body_parts("solar2 seat")?;
        Ok(body.taps(gpu)?)
    }

    fn stores(&mut self) -> Result<Vec<(String, u64)>, GateError> {
        let pos = self.m.pos();
        let (gpu, _, body) = self.m.body_parts("solar2 seat")?;
        Ok(body
            .store_bytes_at(gpu, pos)?
            .into_iter()
            .map(|(name, bytes)| (name, fnv1a(&bytes)))
            .collect())
    }

    fn fault(&mut self) -> Result<(), GateError> {
        if let Some(fault) = self.m.poisoned() {
            return Err(format!("the model is poisoned by {fault:?}").into());
        }
        let (gpu, _, _) = self.m.body_parts("solar2 seat")?;
        match gpu.fault()? {
            None => Ok(()),
            Some(fault) => Err(format!("the card's fault word holds {fault:?}").into()),
        }
    }
}
