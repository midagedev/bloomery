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

use bloomery_gpu::host::slots::MAX_TIERS;
use bloomery_gpu::model::{ChainBody, StepMode};
use bloomery_gpu::{GpuError, GpuModel};
use gguf::Split;
use model::placement::Machine;
use model::placement::workstation::{self, CardSpec, TierBatchBytes, TierDraft};

use crate::record::{self, Record};
use crate::{GateError, ref_model_path};

/// Which placement the engine loads by: a stage card that runs every layer
/// and the head, then the expert tier cards beside the host tier, each one
/// of `workstation::CARDS`. The flag word is an alias — `a`, `gate`, `bp` —
/// or the list `<stage>[+<tier>…]` of card names (`workstation::word_cards`);
/// a list of an alias's cards is that alias, name and all, and any other list
/// has the A6000 as its stage card.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Place {
    /// The flag value as the records print it.
    word: &'static str,
    /// The stage card, then the tier cards in tier order, as indices into
    /// `workstation::CARDS`; the first `n` are the placement's.
    cards: [u8; 1 + MAX_TIERS],
    n: u8,
}

/// A placement of `n` cards, indices into `workstation::CARDS`, by `word`.
const fn alias(word: &'static str, list: &[u8]) -> Place {
    let mut cards = [0u8; 1 + MAX_TIERS];
    let mut i = 0;
    while i < list.len() {
        cards[i] = list[i];
        i += 1;
    }
    Place {
        word,
        cards,
        n: list.len() as u8,
    }
}

/// Whether two card names are one, at compile time.
const fn same_name(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

// The aliases' indices are the A6000's and the 3090's in `workstation::CARDS`.
const _: () = assert!(
    same_name(workstation::CARDS[0].name, workstation::A6000.name)
        && same_name(workstation::CARDS[1].name, workstation::RTX_3090.name)
);

#[allow(
    non_upper_case_globals,
    reason = "the aliases keep the names callers match on: Place::A, Place::Gate, Place::Bp"
)]
impl Place {
    /// The serving plan (`workstation::plan_a`), on the A6000.
    pub const A: Place = alias("a", &[0]);
    /// The step gate's plan (`workstation::plan_gate`), on the 3090.
    pub const Gate: Place = alias("gate", &[1]);
    /// Plan (b′) (`workstation::plan_bp`): plan (a) on the A6000, the 3090
    /// an expert tier under the host tier, holding the DSpark draft's
    /// reserve when a draft is served there.
    pub const Bp: Place = alias("bp", &[0, 1]);

    /// The flag value's placement: an alias, or a list word whose cards are
    /// refused by name when there are more tiers than a slot map names
    /// (`MAX_TIERS`), a name no card has, or a card named twice. A list word
    /// that is no alias's cards is refused unless its stage card is the
    /// A6000: no gate runs another stage.
    pub fn parse(v: &str) -> Result<Place, GateError> {
        match v {
            "a" => return Ok(Place::A),
            "gate" => return Ok(Place::Gate),
            "bp" => return Ok(Place::Bp),
            _ => {}
        }
        let specs = workstation::word_cards(v, MAX_TIERS).map_err(|e| {
            format!(
                "--place is a, gate, bp (plan (b′): the 3090 as the A6000's expert tier) or a \
                 card list <stage>[+<tier>…]: {e}"
            )
        })?;
        let mut list = Vec::with_capacity(specs.len());
        for spec in &specs {
            let i = workstation::CARDS
                .iter()
                .position(|c| c == spec)
                .ok_or_else(|| format!("--place {v}: the card {} is not listed", spec.name))?;
            list.push(u8::try_from(i)?);
        }
        if let Some(p) = [Place::A, Place::Gate, Place::Bp]
            .into_iter()
            .find(|p| p.indices() == list.as_slice())
        {
            return Ok(p);
        }
        if specs[0] != workstation::A6000 {
            return Err(format!(
                "--place {v}: the stage card is the A6000; another stage has no gate yet (the \
                 3090 alone is --place gate)"
            )
            .into());
        }
        let names: Vec<String> = specs.iter().map(|s| s.name.to_ascii_lowercase()).collect();
        // A list word that is no alias is spelled once per parse and kept for
        // the process: the records and the open take the name as `'static`.
        let word: &'static str = Box::leak(names.join("+").into_boxed_str());
        let mut cards = [0u8; 1 + MAX_TIERS];
        cards[..list.len()].copy_from_slice(&list);
        Ok(Place {
            word,
            cards,
            n: u8::try_from(list.len())?,
        })
    }

    /// The placement's cards as indices into `workstation::CARDS`.
    fn indices(&self) -> &[u8] {
        &self.cards[..usize::from(self.n)]
    }

    /// The stage card's spec.
    fn stage(self) -> CardSpec {
        workstation::CARDS[usize::from(self.cards[0])]
    }

    /// The tier cards' specs, in tier order.
    fn tier_specs(self) -> Vec<CardSpec> {
        self.indices()[1..]
            .iter()
            .map(|&i| workstation::CARDS[usize::from(i)])
            .collect()
    }

    /// Refuse, before any plan, a placement of more tier cards than `body`
    /// serves — each body declares its count; `tiers` is it.
    pub fn serves(self, body: &str, tiers: usize) -> Result<(), GateError> {
        let k = usize::from(self.n) - 1;
        if k <= tiers {
            return Ok(());
        }
        Err(format!(
            "--place {}: {k} expert tier cards, and the {body} body serves {tiers} at most",
            self.name()
        )
        .into())
    }

    /// The machine the placement plans over, by layer count. `draft_bytes`
    /// is the DSpark draft's resident bytes when the draft is served on the
    /// placement's first tier card ([`Place::draft_card`]); `batch` is the
    /// expert tier's prompt-batch bytes the plan reserves on each tier card
    /// and the host (the model's own figure, from its hyperparameters). A
    /// figure handed to a placement with no tier card is refused by name, a
    /// placement with tier cards without `batch` too, and every refusal of
    /// `workstation::plan_tiers` before the machine is planned.
    pub fn machine(
        self,
        draft_bytes: Option<u64>,
        batch: Option<TierBatchBytes>,
    ) -> Result<impl Fn(usize) -> Machine + Copy, GateError> {
        if draft_bytes.is_some() && self.draft_card().is_none() {
            return Err(format!(
                "--place {}: no tier card holds a draft reserve; the draft's card is outside the plan",
                self.name()
            )
            .into());
        }
        let tiered = self.n > 1;
        let draft = draft_bytes.map(|bytes| TierDraft { on: 0, bytes });
        match batch {
            Some(_) if !tiered => {
                return Err(format!(
                    "--place {}: no tier card serves a prompt batch; the tier's batch reserve is \
                     outside the plan",
                    self.name()
                )
                .into());
            }
            Some(b) => {
                workstation::check_tiers(self.stage(), &self.tier_specs(), draft, b)?;
            }
            None if tiered => {
                return Err(format!(
                    "--place {}: the plan reserves the expert tier's prompt-batch bytes, and none \
                     were given",
                    self.name()
                )
                .into());
            }
            None => {}
        }
        Ok(move |layers| match batch {
            None => workstation::plan_on(self.stage(), layers),
            Some(b) => workstation::plan_tiers(layers, self.stage(), &self.tier_specs(), draft, b)
                .expect("Place::machine checked the placement's cards before planning"),
        })
    }

    /// The flag value, as the `plan` and `load` lines print it.
    pub fn name(self) -> &'static str {
        self.word
    }

    /// The cards the placement loads, the stage card then the tiers, by the
    /// names the engine finds them by; each device the `load` record's
    /// `cards` names must hold its name.
    pub fn cards(self) -> Vec<&'static str> {
        self.indices()
            .iter()
            .map(|&i| workstation::CARDS[usize::from(i)].name)
            .collect()
    }

    /// The placement's expert tier cards in tier order, by the names the
    /// engine finds them by: the 3090 under `bp`, none under `a` and `gate`.
    pub fn tier_cards(self) -> Vec<&'static str> {
        self.cards()[1..].to_vec()
    }

    /// The card a DSpark draft must sit on under this placement: the first
    /// tier card, whose plan reserves the draft's bytes; `None` where the
    /// plan reserves nothing and the draft's card is the caller's choice.
    pub fn draft_card(self) -> Option<&'static str> {
        self.tier_cards().first().copied()
    }
}

/// How [`Generator::open`] loads: the placement and the expert tier's
/// prompt-batch bytes its plan reserves ([`Place::machine`]'s `batch`), the
/// context the caches are sized for, the step mode, and whether the calling
/// thread pins itself to the dispatcher's cpu slot (`BLOOMERY_PIN_MAIN` is
/// the caller's to read).
#[derive(Debug, Clone, Copy)]
pub struct OpenArgs {
    pub place: Place,
    pub tier_batch: Option<TierBatchBytes>,
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
    /// the `load` record ([`record::LOAD_GENERATOR`]), the load's phases when
    /// the load timed them ([`record::LOAD_PHASES`]), the host set's records
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
        let machine = args.place.machine(None, args.tier_batch)?;
        let headers = t.elapsed();
        let mut model = open(file, &machine, args.ctx)?;
        model.set_mode(args.mode);
        let load = Record::new(&record::LOAD_GENERATOR)
            .u("resident_bytes", model.resident_bytes())
            .u("ctx", args.ctx);
        let load = check(&model, load)?;
        let total = t.elapsed();
        let load = load
            .w("mode", mode_name(args.mode))
            .w("place", args.place.name())
            .w("pin_main", if args.pin_main { "on" } else { "off" })
            .w("pinned", pinned)
            .f("load_s", total.as_secs_f64());
        writeln!(log, "{}", load.line())?;
        if let Some(times) = model.load_times() {
            let phases = record::load_phases(
                total.as_secs_f64(),
                headers.as_secs_f64(),
                times.plan.map(|d| d.as_secs_f64()),
                times.context.as_secs_f64(),
                times.upload.as_secs_f64(),
                times.derive.as_secs_f64(),
                times.host_set.as_secs_f64(),
                times.body.as_secs_f64(),
                times.head.as_secs_f64(),
            );
            writeln!(log, "{}", phases.line())?;
        }
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

#[cfg(test)]
mod tests {
    use model::placement::workstation::{self, A6000, RTX_3090, TierBatchBytes};

    use super::Place;

    const BATCH: TierBatchBytes = workstation::tier_batch_bytes(5120, 2304, 6, 512);

    /// Every alias and its list spelling is one placement, printed by the
    /// alias, and plans the machine its plan function makes: `a` and `a6000`
    /// plan (a), `gate` and `3090` the gate plan, `bp` and `a6000+3090` plan
    /// (b′), with and without a draft, for every layer count the plans see.
    #[test]
    fn aliases_and_their_lists_plan_as_before() {
        for (words, want) in [
            (["a", "a6000", "A6000"], Place::A),
            (["gate", "3090", "3090"], Place::Gate),
            (["bp", "a6000+3090", "A6000+3090"], Place::Bp),
        ] {
            for w in words {
                let p = Place::parse(w).unwrap_or_else(|e| panic!("{w}: {e}"));
                assert_eq!(p, want, "{w}");
                assert_eq!(p.name(), want.name(), "{w}");
            }
        }
        assert_eq!(
            (Place::A.name(), Place::Gate.name(), Place::Bp.name()),
            ("a", "gate", "bp")
        );
        assert_eq!(Place::A.cards(), vec![A6000.name]);
        assert_eq!(Place::Gate.cards(), vec![RTX_3090.name]);
        assert_eq!(Place::Bp.cards(), vec![A6000.name, RTX_3090.name]);
        assert_eq!(Place::Bp.tier_cards(), vec![RTX_3090.name]);
        assert!(Place::A.tier_cards().is_empty() && Place::Gate.draft_card().is_none());
        assert_eq!(Place::Bp.draft_card(), Some(RTX_3090.name));
        let a = Place::A.machine(None, None).expect("a");
        let gate = Place::Gate.machine(None, None).expect("gate");
        for layers in [0, 1, 43, 47, 48, 78] {
            assert_eq!(a(layers), workstation::plan_a(layers));
            assert_eq!(gate(layers), workstation::plan_gate(layers));
            for draft in [None, Some(1_843_200_000)] {
                let bp = Place::Bp.machine(draft, Some(BATCH)).expect("bp");
                assert_eq!(bp(layers), workstation::plan_bp(layers, draft, BATCH));
            }
        }
    }

    /// A list word whose stage card is not the A6000, and that is no
    /// alias's cards, is refused by name: no gate runs another stage.
    #[test]
    fn another_stage_is_refused() {
        for w in ["3090+a6000", "3090+A6000"] {
            assert_eq!(
                Place::parse(w).expect_err(w).to_string(),
                format!(
                    "--place {w}: the stage card is the A6000; another stage has no gate yet (the \
                     3090 alone is --place gate)"
                )
            );
        }
    }

    /// The word's refusals, each by name before any plan: more tiers than a
    /// slot map names, a name no card has, a card named twice; then the
    /// placement's: more tier cards than the body serves, and a draft or a
    /// batch beside no tier, or no batch beside one, in today's words.
    #[test]
    fn refusals_are_named_before_any_plan() {
        let refused = |w: &str| Place::parse(w).expect_err(w).to_string();
        let many = ["a6000"; 10].join("+");
        assert!(
            refused(&many).contains("names 9 tier cards, past the 8"),
            "{}",
            refused(&many)
        );
        assert!(
            refused("b").contains("names the card \"b\""),
            "{}",
            refused("b")
        );
        assert!(
            refused("a6000+4090").contains("names the card \"4090\""),
            "{}",
            refused("a6000+4090")
        );
        assert!(
            refused("3090+3090").contains("names the card 3090 twice"),
            "{}",
            refused("3090+3090")
        );
        assert_eq!(
            Place::Bp
                .serves("deepseek41", 0)
                .expect_err("k 1 > 0")
                .to_string(),
            "--place bp: 1 expert tier cards, and the deepseek41 body serves 0 at most"
        );
        assert!(Place::Bp.serves("deepseek41", 1).is_ok() && Place::A.serves("x", 0).is_ok());
        let machine_err =
            |p: Place, d: Option<u64>, b: Option<TierBatchBytes>| match p.machine(d, b) {
                Ok(_) => panic!("{} {d:?} {b:?} was planned", p.name()),
                Err(e) => e.to_string(),
            };
        assert_eq!(
            machine_err(Place::A, Some(1), None),
            "--place a: no tier card holds a draft reserve; the draft's card is outside the plan"
        );
        assert_eq!(
            machine_err(Place::Gate, None, Some(BATCH)),
            "--place gate: no tier card serves a prompt batch; the tier's batch reserve is \
             outside the plan"
        );
        assert_eq!(
            machine_err(Place::Bp, None, None),
            "--place bp: the plan reserves the expert tier's prompt-batch bytes, and none were \
             given"
        );
    }
}
