//! [`Generator`] — the load-and-step loop a decode CLI drives: open the file
//! by a placement, load it, capture the step graph before the prompt, feed
//! the prompt one real step per token, then one step per chosen token. It
//! owns the [`GpuModel`] and nothing else: no sampling, no text.
//!
//! Generic over the chain body. The body's own entry (`body::open` of the
//! V4.1 device crate, with the levers the binary parsed) and whatever the
//! caller checks on the loaded body come in as arguments, because this
//! library's source does not name a device
//! crate: named here, it would be linked, device bundle and all, into every
//! gate built with the feature, the ones that launch none of its kernels
//! too (the reason [`crate::ds41_meta::RopeMeta::read`] takes its ropes as
//! functions). The plan line is the caller's for the same reason: the plan's
//! inputs are the architecture's type.

use std::io::Write;
use std::time::Instant;

use bloomery_gpu::model::{ChainBody, StepMode};
use bloomery_gpu::{GpuError, GpuModel};
use gguf::Split;
use model::placement::{Machine, workstation};

use crate::record::{self, Record};
use crate::{GateError, ref_model_path};

/// Which placement the engine loads by.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Place {
    /// The serving plan (`workstation::plan_a`), on the A6000.
    A,
    /// The step gate's plan (`workstation::plan_gate`), on the 3090.
    Gate,
    /// Plan (b′) (`workstation::plan_bp`): plan (a) on the A6000, the 3090
    /// an expert tier under the host tier, holding the DSpark draft's
    /// reserve when a draft is served there.
    Bp,
}

impl Place {
    /// The flag value's placement: `a`, `gate` or `bp`.
    pub fn parse(v: &str) -> Result<Place, GateError> {
        match v {
            "a" => Ok(Place::A),
            "gate" => Ok(Place::Gate),
            "bp" => Ok(Place::Bp),
            other => Err(format!(
                "--place is a, gate or bp (plan (b′): the 3090 as the A6000's expert tier), not \
                 {other}"
            )
            .into()),
        }
    }

    /// The machine the placement plans over, by layer count. `draft_bytes`
    /// is the DSpark draft's resident bytes when the draft is served on the
    /// placement's tier card ([`Place::draft_card`]); only `bp` has one, and
    /// a figure handed to another placement is refused by name.
    pub fn machine(
        self,
        draft_bytes: Option<u64>,
    ) -> Result<impl Fn(usize) -> Machine + Copy, GateError> {
        if draft_bytes.is_some() && self.draft_card().is_none() {
            return Err(format!(
                "--place {}: no tier card holds a draft reserve; the draft's card is outside the plan",
                self.name()
            )
            .into());
        }
        Ok(move |layers| match self {
            Place::A => workstation::plan_a(layers),
            Place::Gate => workstation::plan_gate(layers),
            Place::Bp => workstation::plan_bp(layers, draft_bytes),
        })
    }

    /// The flag value, as the `plan` and `load` lines print it.
    pub fn name(self) -> &'static str {
        match self {
            Place::A => "a",
            Place::Gate => "gate",
            Place::Bp => "bp",
        }
    }

    /// The cards the placement loads, stage cards then the tier, by the
    /// names the engine finds them by: the `load` record's `cards`.
    pub fn cards(self) -> &'static [&'static str] {
        const A: &[&str] = &[workstation::A6000.name];
        const GATE: &[&str] = &[workstation::RTX_3090.name];
        const BP: &[&str] = &[workstation::A6000.name, workstation::RTX_3090.name];
        match self {
            Place::A => A,
            Place::Gate => GATE,
            Place::Bp => BP,
        }
    }

    /// The placement's expert tier card, by the name the engine finds it
    /// by: the 3090 under `bp`, none under `a` and `gate`.
    pub fn tier_card(self) -> Option<&'static str> {
        match self {
            Place::A | Place::Gate => None,
            Place::Bp => Some(workstation::RTX_3090.name),
        }
    }

    /// The card a DSpark draft must sit on under this placement: the tier
    /// card, whose plan reserves the draft's bytes; `None` where the plan
    /// reserves nothing and the draft's card is the caller's choice.
    pub fn draft_card(self) -> Option<&'static str> {
        self.tier_card()
    }

    /// Why a prompt under this placement is fed one decode step per id
    /// whatever `BLOOMERY_PREFILL` says, or `None` where the lever decides:
    /// the tier card has no batch port, and a prompt batch on a model with a
    /// tier is refused by the host tier.
    pub fn steps_only(self) -> Option<&'static str> {
        match self {
            Place::A | Place::Gate => None,
            Place::Bp => Some(
                "the expert tier card has no batch port (tierbatch): a prompt batch on a model \
                 with a tier is refused, so the prompt goes one decode step per id",
            ),
        }
    }
}

/// How [`Generator::open`] loads: the placement, the context the caches are
/// sized for, the step mode, and whether the calling thread pins itself to
/// the dispatcher's cpu slot (`BLOOMERY_PIN_MAIN` is the caller's to read).
#[derive(Debug, Clone, Copy)]
pub struct OpenArgs {
    pub place: Place,
    pub ctx: usize,
    pub mode: StepMode,
    pub pin_main: bool,
}

/// A loaded model standing at a position, and the context it may reach.
pub struct Generator<B: ChainBody> {
    model: GpuModel<B>,
    ctx: usize,
}

impl<B: ChainBody> Generator<B> {
    /// Open `$BLOOMERY_REF_MODEL` by `args.place` through `open`, the body's
    /// loader (the file, the placement's machine and the context), run `check`
    /// on the loaded model (it refuses a body that cannot run this file, and
    /// adds the body's fields to the `load` record it is handed), then write
    /// the `load` record ([`record::LOAD_GENERATOR`]), the host set's records
    /// of a placed load, and in graph mode capture the step before any token
    /// (the `capture` record) — all to `log`.
    pub fn open<O, C>(
        args: OpenArgs,
        open: O,
        check: C,
        log: &mut dyn Write,
    ) -> Result<Generator<B>, GateError>
    where
        O: FnOnce(Split, &dyn Fn(usize) -> Machine, usize) -> Result<GpuModel<B>, GpuError>,
        C: FnOnce(&GpuModel<B>, Record) -> Result<Record, GateError>,
    {
        let pinned = args.pin_main && threads::pool().pin_caller();
        let path = ref_model_path()?;
        let t = Instant::now();
        let file = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let machine = args.place.machine(None)?;
        let mut model = open(file, &machine, args.ctx)?;
        model.set_mode(args.mode);
        let load = Record::new(&record::LOAD_GENERATOR)
            .u("resident_bytes", model.resident_bytes())
            .u("ctx", args.ctx);
        let load = check(&model, load)?
            .w("mode", mode_name(args.mode))
            .w("place", args.place.name())
            .w("pin_main", if args.pin_main { "on" } else { "off" })
            .w("pinned", pinned)
            .f("load_s", t.elapsed().as_secs_f64());
        writeln!(log, "{}", load.line())?;
        if let Some(h) = model.host_residency() {
            for r in record::host_residency(h) {
                writeln!(log, "{}", r.line())?;
            }
        }
        if args.mode == StepMode::Graph {
            // Captured before the prompt, so no token's step pays for it.
            let nodes = model.capture_step()?;
            writeln!(
                log,
                "{}",
                Record::new(&record::CAPTURE).u("graph_nodes", nodes).line()
            )?;
        }
        Ok(Generator {
            model,
            ctx: args.ctx,
        })
    }

    /// Feed `ids` from the current position, one real step per id, and
    /// return the argmax after the last. Refused before any step when the
    /// ids do not fit the context ([`Generator::check_feed`]).
    pub fn prefill(&mut self, ids: &[u32]) -> Result<u32, GateError> {
        self.check_feed(ids.len())?;
        Ok(self.model.step(ids)?)
    }

    /// Whether a feed of `n` ids may start at the current position: at
    /// least one, and all inside the context. The one owner of that check
    /// for every feed of this generator, [`Generator::prefill`] and a
    /// body's own prompt feed alike.
    pub fn check_feed(&self, n: usize) -> Result<(), String> {
        if n == 0 {
            return Err("prefill: no ids to feed".into());
        }
        if self.pos() + n > self.ctx {
            return Err(format!(
                "prefill: position {} + {n} ids exceed the context {}",
                self.pos(),
                self.ctx
            ));
        }
        Ok(())
    }

    /// One step on `tok` at the current position; the argmax after it.
    pub fn step(&mut self, tok: u32) -> Result<u32, GateError> {
        if usize::try_from(self.model.pos())? >= self.ctx {
            return Err(format!("step: the context {} is full", self.ctx).into());
        }
        Ok(self.model.step(&[tok])?)
    }

    /// The head's logits after the last step (`n_vocab` f32). A blocking
    /// read of the whole row, on top of the step's own argmax readback.
    pub fn logits(&self) -> Result<Vec<f32>, GateError> {
        Ok(self.model.logits()?)
    }

    /// The position the next fed token lands in.
    pub fn pos(&self) -> usize {
        // u32 widens into usize on the 64-bit hosts this runs on.
        self.model.pos() as usize
    }

    /// The context the caches were sized for: `pos` never passes it.
    pub fn ctx_max(&self) -> usize {
        self.ctx
    }

    /// The model, for what the loop does not own (the pair pass, probes).
    pub fn model(&self) -> &GpuModel<B> {
        &self.model
    }

    /// See [`Generator::model`].
    pub fn model_mut(&mut self) -> &mut GpuModel<B> {
        &mut self.model
    }
}

/// The step mode as the `load` line prints it.
pub fn mode_name(mode: StepMode) -> &'static str {
    if mode == StepMode::Graph {
        "graph"
    } else {
        "eager"
    }
}
