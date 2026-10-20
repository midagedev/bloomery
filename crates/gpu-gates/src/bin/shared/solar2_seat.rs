//! The seat the Solar-Open2 gates drive the body through (`gate_solar2_e2e`, `gate_solar2_prefill`): the load, a reset, a prompt
//! call and a step, each answering the last position's logits row, and the three reads a gate holds the body to — the layer
//! outputs of the last call, a digest of every store, the card's fault word. A gate includes it by `#[path]`.
//!
//! The contract of each call is the body's, fixed here so the gates and the body ([`bloomery_gpu_solar2`]) are written against
//! one text:
//! - [`Seat::prompt`] runs `ids` from the seat's position by the prompt batch (up to `bloomery_gpu_solar2::body::T_MAX` rows a unit) and answers the last
//!   id's logits row; [`Seat::step`] runs one id by the captured step;
//! - [`Seat::taps`] is `layers * embd` values, layer `l`'s output at the last position of the last call at `[l * embd..]`,
//!   whichever path ran it;
//! - [`Seat::stores`] names every store the written positions fill — each KDA layer's state, stamp and conv ring, each GQA
//!   layer's K and V planes over the written positions — with a 64-bit digest of its bytes ([`fnv1a`]), in layer order;
//! - [`Seat::fault`] is `Ok` while the fault word is clean and the model not poisoned.

use bloomery_gpu::model::StepMode;
use bloomery_gpu_gates::tier::{self, Tier};
use bloomery_gpu_gates::{GateError, ref_model_path};
use bloomery_gpu_solar2::{Body, Solar2Model, set_prefill_group, set_taps};
use gguf::Split;
use model::arch::solar2::place::PlanInputs;
use model::placement::PlanLevers;
use model::placement::workstation::{HostRead, PLAN_ROOM};
use refset::arch::solar2::ARCH;
use std::path::{Path, PathBuf};

/// What the loaded file's shape is: the layers a tap table has, the width of a layer's output and of a logits row.
#[derive(Clone, Copy, Debug)]
pub struct Dims {
    pub layers: usize,
    pub embd: usize,
    pub vocab: usize,
}

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

/// One loaded Solar-Open2 body on the gate card.
pub trait Seat {
    fn dims(&self) -> Dims;

    /// Empty every store; the position is 0.
    fn reset(&mut self) -> Result<(), GateError>;

    /// The position the next id lands at.
    fn pos(&self) -> u32;

    /// Run `id` by the captured step; the logits row after it.
    fn step(&mut self, id: u32) -> Result<Vec<f32>, GateError>;

    /// Run `ids` by the prompt batch; the logits row after the last.
    fn prompt(&mut self, ids: &[u32]) -> Result<Vec<f32>, GateError>;

    /// Set the prompt group to `group` units a walk (`BLOOMERY_PREFILL_GROUP`'s meaning).
    fn set_group(&mut self, group: usize) -> Result<(), GateError>;

    /// Every layer's output at the last position of the last call.
    fn taps(&mut self) -> Result<Vec<f32>, GateError>;

    /// Every store's name and digest over the positions written so far.
    fn stores(&mut self) -> Result<Vec<(String, u64)>, GateError>;

    /// `Ok` while the card's fault word is clean and the model is not poisoned.
    fn fault(&mut self) -> Result<(), GateError>;
}

/// The model file the profile names, checked before anything loads: a solar-open2 file, and under the fixture tier a whole
/// fixture (`tier::Tier::check_file`). `recipe` is the `just` recipe that picks the profile.
pub fn model_path(tier_now: Tier, recipe: &str) -> Result<PathBuf, GateError> {
    let path = ref_model_path()?;
    let file = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
    if file.architecture() != Some(ARCH) {
        return Err(format!(
            "the model file is {:?}, want {ARCH} — run through `just {recipe}`, which picks the {ARCH} profile",
            file.architecture()
        )
        .into());
    }
    let keys = [tier::VERSION_KEY, tier::SUBSET_KEY]
        .into_iter()
        .filter(|k| file.value(k).is_some());
    tier_now.check_file(&path, keys)?;
    Ok(path)
}

/// The placement arm a gate runs: plan (a), the largest card the stage with the experts that fit on it and the rest on the host,
/// or plan (b′), the next-largest card an expert tier under the host tier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arm {
    A,
    Bp,
}

impl Arm {
    /// The arm a `--arm` word names; any other word is refused by name.
    pub fn parse(word: &str) -> Result<Arm, GateError> {
        match word {
            "a" => Ok(Arm::A),
            "bp" => Ok(Arm::Bp),
            _ => Err(format!("--arm is a or bp, got {word:?}").into()),
        }
    }

    #[must_use]
    pub fn word(self) -> &'static str {
        match self {
            Arm::A => "a",
            Arm::Bp => "bp",
        }
    }
}

/// The arm the command line names: `--arm <a|bp>`, `a` without it. An argument it does not know is refused by name.
pub fn arm_arg() -> Result<Arm, GateError> {
    let mut args = std::env::args().skip(1);
    let mut arm = Arm::A;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--arm" => arm = Arm::parse(&args.next().ok_or("--arm wants a or bp")?)?,
            other => {
                return Err(
                    format!("unknown argument {other:?}: the gate takes --arm a|bp").into(),
                );
            }
        }
    }
    Ok(arm)
}

/// FNV-1a over bytes: the digest a store is held to.
#[must_use]
pub fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// Open `file` on the gate card at `ctx` positions by the gate placement (`gate_card::plan_gate`) of `arm`: the body opened as
/// the serve seat opens it (`Body::open_placed`, the plan `PlanInputs::plan` makes), the taps armed before the step is first
/// captured, so a replay and a prompt batch both write them.
///
/// The including gate declares `shared/gate_card.rs` as `gate_card` beside this module.
pub fn open(file: &Path, ctx: usize, arm: Arm) -> Result<Box<dyn Seat>, GateError> {
    if arm == Arm::Bp {
        return Err(
            "solar2 seat: arm bp (plan (b′): the next-largest card an expert tier under the host tier) is refused by name: the \
             Solar body serves no expert tier card yet (`PlaceError::Tier`)"
                .into(),
        );
    }
    let levers = bloomery_levers::at_main(&[bloomery_levers::CARD_BUDGET])?;
    crate::gate_card::init()?;
    let split = Split::open(file).map_err(|e| format!("open {}: {e}", file.display()))?;
    // The file as it is: the gates feed ids, so the features of the file that only the tokenizer and the tool-call parser
    // carry (`PlanInputs::read` lists them as unimplemented) are not the gate's. The plan this gate holds its pins to is the
    // unpaged one at the workstation's plan room: no reading of the box's available bytes enters it.
    let mut inputs = PlanInputs::describe(&split)?;
    inputs.room = (PLAN_ROOM, HostRead::Given);
    let machine = crate::gate_card::plan_gate(inputs.model.layers);
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
