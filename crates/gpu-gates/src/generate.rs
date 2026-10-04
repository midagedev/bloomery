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
use bloomery_gpu::host::tier::TierCard;
use bloomery_gpu::model::{ChainBody, StepMode};
use bloomery_gpu::{Gpu, GpuError, GpuModel};
use gguf::Split;
use model::placement::Machine;
use model::placement::workstation::{
    self, CardSpec, DeviceId, DeviceInfo, Pick, TierBatchBytes, TierDraft,
};

use crate::record::{self, Record};
use crate::{GateError, ref_model_path};

/// Which placement the engine loads by: a stage card that runs every layer
/// and the head, then the expert tier cards beside the host tier. The flag
/// word is an alias — `a` (the stage on the largest visible card), `bp`
/// (that stage and the next-largest as its tier), `gate` (the card named
/// 3090, alone) — or the list `<stage>[+<tier>…]` of card names and device
/// ordinals (`workstation::word_picks`); a list of an alias's cards on this
/// workstation by name prints as that alias.
///
/// A parsed placement is this workstation's by name, with no census: its
/// cards are `workstation::CARDS` ([`workstation::workstation_spec`]), each
/// opened by its name — the gates' machines. A binary resolves it against
/// this process's devices ([`Place::on_host`]) before it plans: then each
/// card is the spec of the device it picked and opens on that device.
#[derive(Debug, Clone, Copy)]
pub struct Place {
    /// The flag value as the records print it.
    word: &'static str,
    /// How each card is found, the stage then the tiers in tier order; the
    /// first `n` are the placement's.
    picks: [Pick; 1 + MAX_TIERS],
    n: u8,
    /// Each card's spec: this workstation's by name before
    /// [`Place::on`] (`None` for an ordinal, which names no card without a
    /// census), the device's after.
    specs: [Option<CardSpec>; 1 + MAX_TIERS],
    resolved: bool,
}

/// Two placements are one when they print the same word: the word is what
/// the records and the binaries' checks read (`place == Place::Gate`), the
/// same before and after [`Place::on`].
impl PartialEq for Place {
    fn eq(&self, other: &Place) -> bool {
        self.word == other.word
    }
}

impl Eq for Place {}

/// The alias `ALIASES[i]` as a placement, this workstation's by name.
const fn alias(i: usize) -> Place {
    let (word, list) = workstation::ALIASES[i];
    let mut picks = [Pick::Rank(0); 1 + MAX_TIERS];
    let mut specs = [None; 1 + MAX_TIERS];
    let mut k = 0;
    while k < list.len() {
        picks[k] = list[k];
        specs[k] = workstation::workstation_spec(list[k]);
        k += 1;
    }
    Place {
        word,
        picks,
        n: list.len() as u8,
        specs,
        resolved: false,
    }
}

#[allow(
    non_upper_case_globals,
    reason = "the aliases keep the names callers match on: Place::A, Place::Gate, Place::Bp"
)]
impl Place {
    /// The serving plan: the stage on the largest visible card
    /// (`workstation::plan_a` on this workstation, the A6000).
    pub const A: Place = alias(0);
    /// The step gate's plan (`workstation::plan_gate`), on the 3090.
    pub const Gate: Place = alias(1);
    /// Plan (b′): plan (a)'s stage, the next-largest card an expert tier
    /// under the host tier (`workstation::plan_bp` on this workstation, the
    /// 3090 under the A6000), holding the DSpark draft's reserve when a
    /// draft is served there.
    pub const Bp: Place = alias(2);

    /// The flag value's placement: an alias, or a list word whose parts are
    /// refused by name when there are more tiers than a slot map names
    /// (`MAX_TIERS`), a part that is no card name and no ordinal, or a name
    /// or ordinal given twice. Any card may be the stage.
    pub fn parse(v: &str) -> Result<Place, GateError> {
        match v {
            "a" => return Ok(Place::A),
            "gate" => return Ok(Place::Gate),
            "bp" => return Ok(Place::Bp),
            _ => {}
        }
        let list = workstation::word_picks(v, MAX_TIERS).map_err(|e| {
            format!(
                "--place is a, gate, bp (plan (b′): the next-largest card as the largest's \
                 expert tier) or a card list <stage>[+<tier>…] of card names and device \
                 ordinals: {e}"
            )
        })?;
        let mut picks = [Pick::Rank(0); 1 + MAX_TIERS];
        let mut specs = [None; 1 + MAX_TIERS];
        for (i, &p) in list.iter().enumerate() {
            picks[i] = p;
            specs[i] = workstation::workstation_spec(p);
        }
        let named = list.iter().all(|p| matches!(p, Pick::Name(_)));
        let alias = [Place::A, Place::Gate, Place::Bp]
            .into_iter()
            .find(|a| named && a.specs[..usize::from(a.n)] == specs[..list.len()]);
        // A list word that is no alias is spelled once per parse and kept for
        // the process: the records and the open take the name as `'static`.
        let word: &'static str = match alias {
            Some(a) => a.word,
            None => Box::leak(v.to_ascii_lowercase().into_boxed_str()),
        };
        Ok(Place {
            word,
            picks,
            n: u8::try_from(list.len())?,
            specs,
            resolved: false,
        })
    }

    /// The placement on this process's devices ([`crate::gpu_census::census`],
    /// the driver's census with nvidia-smi's holders): [`Place::on`] of the
    /// census.
    pub fn on_host(self) -> Result<Place, GateError> {
        self.on(&crate::gpu_census::census()?)
    }

    /// The placement resolved against `census`: each card the spec of the
    /// device its pick finds ([`workstation::resolve`]), refused by name with
    /// every visible device when a pick finds none, several, or a device an
    /// earlier card is. A list of names spelled as an alias that lands on
    /// other devices than the alias does here prints as its names.
    pub fn on(self, census: &[DeviceInfo]) -> Result<Place, GateError> {
        let resolved = workstation::resolve(self.picks(), census)
            .map_err(|e| format!("--place {}: {e}", self.spelled()))?;
        let mut specs = [None; 1 + MAX_TIERS];
        for (i, s) in resolved.iter().enumerate() {
            specs[i] = Some(*s);
        }
        let devices = |s: &[CardSpec]| s.iter().map(|c| c.device).collect::<Vec<_>>();
        let mut word = self.word;
        if let Some((_, alias)) = workstation::ALIASES.iter().find(|(w, _)| *w == self.word)
            && *alias != self.picks()
            && workstation::resolve(alias, census)
                .ok()
                .is_none_or(|theirs| devices(&theirs) != devices(&resolved))
        {
            let names: Vec<String> = resolved
                .iter()
                .map(|s| s.name.to_ascii_lowercase())
                .collect();
            word = Box::leak(names.join("+").into_boxed_str());
        }
        Ok(Place {
            word,
            specs,
            resolved: true,
            ..self
        })
    }

    /// The word as typed: an alias's own word, or the picks of a list
    /// spelled as an alias (`a6000` for `a`).
    fn spelled(&self) -> String {
        let alias = workstation::ALIASES.iter().find(|(w, _)| *w == self.word);
        match alias {
            Some((_, picks)) if *picks != self.picks() => self
                .picks()
                .iter()
                .map(|p| p.to_string().to_ascii_lowercase())
                .collect::<Vec<_>>()
                .join("+"),
            _ => self.word.to_string(),
        }
    }

    /// How the placement's cards are found.
    fn picks(&self) -> &[Pick] {
        &self.picks[..usize::from(self.n)]
    }

    /// The placement's card specs, the stage first; a card an ordinal names
    /// before [`Place::on`] is refused by name.
    pub fn card_specs(self) -> Result<Vec<CardSpec>, GateError> {
        self.specs[..usize::from(self.n)]
            .iter()
            .zip(self.picks())
            .map(|(s, p)| {
                s.ok_or_else(|| {
                    format!(
                        "--place {}: {p} is a device of this process's census, and the \
                         placement was not resolved against it (Place::on_host)",
                        self.word
                    )
                    .into()
                })
            })
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
        let specs = self.card_specs()?;
        let (stage, tiers) = (specs[0], &specs[1..]);
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
                workstation::check_tiers(stage, tiers, draft, b)?;
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
        let mut tier_specs = [stage; MAX_TIERS];
        tier_specs[..tiers.len()].copy_from_slice(tiers);
        let k = tiers.len();
        Ok(move |layers| match batch {
            None => workstation::plan_on(stage, layers),
            Some(b) => workstation::plan_tiers(layers, stage, &tier_specs[..k], draft, b)
                .expect("Place::machine checked the placement's cards before planning"),
        })
    }

    /// The flag value, as the `plan` and `load` lines print it.
    pub fn name(self) -> &'static str {
        self.word
    }

    /// The cards the placement loads, the stage card then the tiers, by the
    /// names the plan gives them (an ordinal before [`Place::on`] as
    /// `cuda<N>`).
    pub fn cards(self) -> Vec<&'static str> {
        self.specs[..usize::from(self.n)]
            .iter()
            .zip(self.picks())
            .map(|(s, p)| match (s, p) {
                (Some(s), _) => s.name,
                (None, Pick::Ordinal(o)) => ordinal_word(*o),
                (None, _) => "?",
            })
            .collect()
    }

    /// The placement's expert tier cards in tier order, by name: the 3090
    /// under `bp` on this workstation, none under `a` and `gate`.
    pub fn tier_cards(self) -> Vec<&'static str> {
        self.cards()[1..].to_vec()
    }

    /// The card a DSpark draft must sit on under this placement: the first
    /// tier card, whose plan reserves the draft's bytes; `None` where the
    /// plan reserves nothing and the draft's card is the caller's choice.
    pub fn draft_card(self) -> Option<&'static str> {
        self.tier_cards().first().copied()
    }

    /// The device of the draft's card ([`Place::draft_card`]) once resolved.
    pub fn draft_device(self) -> Option<DeviceId> {
        (self.n > 1)
            .then_some(self.specs[1])
            .flatten()
            .and_then(|s| s.device)
    }
}

/// `cuda<o>` as a `'static` word, spelled once per ordinal.
fn ordinal_word(o: u32) -> &'static str {
    static WORDS: std::sync::Mutex<Vec<(u32, &'static str)>> = std::sync::Mutex::new(Vec::new());
    let mut w = WORDS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some((_, s)) = w.iter().find(|(k, _)| *k == o) {
        return s;
    }
    let s: &'static str = Box::leak(format!("cuda{o}").into_boxed_str());
    w.push((o, s));
    s
}

/// `r`, a `load` record ([`record::LOAD_GENERATOR`]) written up to its
/// `unified_addressing` field, with the devices the model runs on (`cards`:
/// the stage card `stage`, then the expert tier cards `tiers`), by the names
/// their drivers report — each space written `_`, since the field is one
/// word — and, when the placement has an expert tier card, the tier's
/// experts and resident bytes. A device that is not the placement's card —
/// of a resolved placement, another device; of a census-free one, a name
/// that does not hold the card's —, tiers that do not match the placement's
/// — another count, or a tier on another card — and more than the one tier
/// the record's fields hold are refused by name. Every body's load record
/// names its cards here.
pub fn with_cards(
    r: Record,
    place: Place,
    stage: &Gpu,
    tiers: &[TierCard],
) -> Result<Record, GateError> {
    let mut gpus = vec![stage];
    gpus.extend(tiers.iter().map(TierCard::gpu));
    let mut devices = Vec::with_capacity(gpus.len());
    let mut ids = Vec::with_capacity(gpus.len());
    for g in &gpus {
        devices.push(g.device_name()?);
        ids.push(g.device_id()?);
    }
    let planned = place.cards();
    let named = devices.len() == planned.len()
        && if place.resolved {
            let specs = place.card_specs()?;
            ids.iter()
                .zip(&specs)
                .all(|(id, s)| s.device.is_some_and(|d| d.uuid == id.uuid))
        } else {
            devices.iter().zip(&planned).all(|(d, p)| d.contains(p))
        };
    if !named {
        return Err(format!(
            "--place {}: the model runs on {devices:?} ({}), the placement's cards are \
             {planned:?}",
            place.name(),
            ids.iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        )
        .into());
    }
    let r = r.csv("cards", devices.iter().map(|d| d.replace(' ', "_")));
    let want = place.tier_cards();
    let got: Vec<&str> = tiers.iter().map(TierCard::name).collect();
    match (tiers, want.as_slice()) {
        ([], []) => Ok(r),
        ([t], [name]) if t.name() == *name => Ok(r
            .u("tier_experts", t.set().experts())
            .u("tier_bytes", t.weights().resident_bytes())),
        ([_, _, ..], _) if got == want => Err(format!(
            "--place {}: the load record holds one expert tier's fields, and the model loaded \
             {} tiers",
            place.name(),
            got.len()
        )
        .into()),
        _ => Err(format!(
            "--place {}: the loaded model's expert tiers are {got:?}, the placement's {want:?}",
            place.name()
        )
        .into()),
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
    use model::placement::Machine;
    use model::placement::workstation::{self, A6000, DeviceInfo, RTX_3090, TierBatchBytes};

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

    /// Any card may be the stage: a list word of this workstation's names
    /// plans its stage on its first card, and a list of ordinals plans
    /// only once resolved.
    #[test]
    fn any_card_is_the_stage() {
        for w in ["3090+a6000", "3090+A6000"] {
            let p = Place::parse(w).unwrap_or_else(|e| panic!("{w}: {e}"));
            assert_eq!(p.name(), "3090+a6000");
            let m = p.machine(None, Some(BATCH)).expect("3090+a6000");
            assert_eq!(
                m(43),
                workstation::plan_tiers(43, RTX_3090, &[A6000], None, BATCH).expect("plan")
            );
        }
        let ordinals = Place::parse("cuda1+0").expect("ordinals");
        assert_eq!(ordinals.cards(), vec!["cuda1", "cuda0"]);
        let unresolved = match ordinals.machine(None, Some(BATCH)) {
            Ok(_) => panic!("an unresolved ordinal was planned"),
            Err(e) => e.to_string(),
        };
        assert!(
            unresolved.contains("cuda1 is a device of this process's census"),
            "{unresolved}"
        );
    }

    /// On this workstation's census, in both of its enumeration orders, the
    /// aliases resolve to the cards they name with no census, each now
    /// naming its device; a list of ordinals plans the cards it numbers.
    /// Names that spell an alias but land elsewhere print as their names.
    #[test]
    fn resolved_on_the_box_plans_as_before() {
        // The plan's shape is what this test holds; the census readings the
        // resolution fills in (the device binding, the free bytes and their
        // holders) ride whatever card state the sitting meets.
        let strip = |mut m: Machine| {
            for c in m.cards.iter_mut().chain(m.tiers.iter_mut()) {
                c.device = None;
                c.free_bytes = None;
                c.held_by = None;
            }
            m
        };
        for order in [["3090", "A6000"], ["A6000", "3090"]] {
            let census = census(&order);
            let a6000 = u32::try_from(order.iter().position(|n| *n == "A6000").expect("A6000"))
                .expect("small");
            let a = Place::A.on(&census).expect("a");
            assert_eq!((a, a.name(), a.cards()), (Place::A, "a", vec![A6000.name]));
            let m = a.machine(None, None).expect("a")(43);
            assert_eq!(m.cards[0].device.map(|d| d.ordinal), Some(a6000));
            assert_eq!(strip(m), workstation::plan_a(43));
            let bp = Place::Bp.on(&census).expect("bp");
            assert_eq!(bp.draft_device().map(|d| d.ordinal), Some(1 - a6000));
            let m = bp.machine(None, Some(BATCH)).expect("bp")(43);
            assert_eq!(strip(m), workstation::plan_bp(43, None, BATCH));
            let gate = Place::Gate.on(&census).expect("gate");
            assert_eq!(
                strip(gate.machine(None, None).expect("gate")(43)),
                workstation::plan_gate(43)
            );
            let w = format!("{a6000}+{}", 1 - a6000);
            let list = Place::parse(&w)
                .expect("ordinals")
                .on(&census)
                .expect("resolved");
            assert_eq!(
                (list.name(), list.cards()),
                (w.as_str(), vec!["A6000", "3090"])
            );
            let m = list.machine(None, Some(BATCH)).expect("list")(43);
            assert_eq!(strip(m), workstation::plan_bp(43, None, BATCH));
        }
        let ada = census(&["NVIDIA RTX 6000 Ada Generation", "A6000"]);
        let named = Place::parse("a6000").expect("a6000");
        assert_eq!(named.name(), "a");
        let on = named.on(&ada).expect("the A6000");
        assert_eq!((on.name(), on.cards()), ("a6000", vec!["A6000"]));
        let e = named
            .on(&census(&["3090"]))
            .expect_err("no A6000")
            .to_string();
        assert!(
            e.starts_with("--place a6000: no visible device is A6000 (visible: cuda0"),
            "{e}"
        );
        let largest = Place::A.on(&ada).expect("a");
        assert_eq!(largest.cards(), vec!["RTX_6000_Ada_Generation"]);
        let two = census(&["3090", "3090"]);
        let e = Place::Gate.on(&two).expect_err("two 3090s").to_string();
        assert!(
            e.starts_with("--place gate: 2 visible devices carry the name 3090"),
            "{e}"
        );
        let bp = Place::Bp.on(&two).expect("two 3090s");
        assert_eq!(bp.cards(), vec!["3090", "3090"]);
        let m = bp.machine(None, Some(BATCH)).expect("stage + tier")(43);
        assert_eq!(m.tiers[0].device.map(|d| d.ordinal), Some(1));
    }

    /// A fake census of devices by short name, each its measured total
    /// (the A6000's and the 3090's) or twice the A6000's.
    fn census(names: &[&str]) -> Vec<DeviceInfo> {
        names
            .iter()
            .enumerate()
            .map(|(i, n)| {
                let (name, total_bytes) = match *n {
                    "3090" => ("NVIDIA GeForce RTX 3090".to_string(), 25_351_356_416),
                    "A6000" => ("NVIDIA RTX A6000".to_string(), 50_952_536_064),
                    other => (other.to_string(), 2 * 50_952_536_064),
                };
                DeviceInfo {
                    ordinal: u32::try_from(i).expect("small"),
                    name,
                    total_bytes,
                    free_bytes: total_bytes / 1024 / 1024 * 1024,
                    uuid: [u8::try_from(i).expect("small") + 1; 16],
                    pci_bus: format!("0000:{:02x}:00.0", 0x41 + i),
                    held_by: None,
                }
            })
            .collect()
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

/// The DSpark draft's card, by device.
impl Place {
    /// The card a DSpark draft loads on under this placement, on `census`:
    /// `set` (`BLOOMERY_DSPARK_CARD`) is one card as a `--place` list word
    /// names one — a CUDA ordinal (`1`, `cuda1`) or a card name exactly one
    /// visible device carries ([`workstation::resolve`]). Under a placement
    /// with a tier card the draft sits on the first tier card, whose plan
    /// reserves its bytes: `set` may name that device and no other. With no
    /// tier, `set`'s card, it may be the stage's own; unset, the visible
    /// device of the fewest usable bytes, ties to the higher ordinal — the
    /// 3090 on this workstation wherever it is in view (beside the A6000
    /// under `a`, the stage itself under `gate`), and on two cards of one
    /// name the one that is not plan (a)'s stage. Each refusal by name.
    pub fn draft_spec(
        self,
        set: Option<&str>,
        census: &[DeviceInfo],
    ) -> Result<CardSpec, GateError> {
        let lever = |v: &str, e: &dyn std::fmt::Display| -> GateError {
            format!("BLOOMERY_DSPARK_CARD={v:?}: {e}").into()
        };
        let named = match set {
            None => None,
            Some(v) => {
                let picks = workstation::word_picks(v, 0).map_err(|e| match e {
                    workstation::WordError::TooManyTiers { .. } => lever(
                        v,
                        &"the lever names one card (a CUDA ordinal or a card name), not a list",
                    ),
                    e => lever(v, &e),
                })?;
                let spec = workstation::resolve(&picks, census).map_err(|e| lever(v, &e))?;
                Some((v, spec[0]))
            }
        };
        let on = |s: CardSpec| s.device.map_or_else(|| "?".to_string(), |d| d.to_string());
        if self.n > 1 {
            let placed = if self.resolved {
                self
            } else {
                self.on(census)?
            };
            let tier = placed.card_specs()?[1];
            return match named {
                None => Ok(tier),
                Some((_, s)) if s.device == tier.device => Ok(tier),
                Some((v, s)) => Err(format!(
                    "BLOOMERY_DSPARK_CARD={v:?} is {} ({}) under --place {}: the draft sits on \
                     the tier card {} ({}), whose plan reserves its bytes; unset the lever or \
                     name that card",
                    on(s),
                    s.name,
                    self.name(),
                    on(tier),
                    tier.name
                )
                .into()),
            };
        }
        if let Some((_, s)) = named {
            return Ok(s);
        }
        let last = census
            .len()
            .checked_sub(1)
            .and_then(|k| u8::try_from(k).ok())
            .ok_or_else(|| {
                format!(
                    "the DSpark draft's card: {} visible devices (visible: {})",
                    census.len(),
                    workstation::visible(census)
                )
            })?;
        let spec = workstation::resolve(&[Pick::Rank(last)], census)
            .map_err(|e| format!("the DSpark draft's card: {e}"))?;
        Ok(spec[0])
    }
}

#[cfg(test)]
mod draft_card_tests {
    use model::placement::workstation::{A6000, DeviceInfo, RTX_3090};

    use super::Place;

    /// A fake census of devices by short name, each its measured total.
    fn census(names: &[&str]) -> Vec<DeviceInfo> {
        names
            .iter()
            .enumerate()
            .map(|(i, n)| {
                let (name, total_bytes) = match *n {
                    "3090" => ("NVIDIA GeForce RTX 3090", 25_351_356_416),
                    "A6000" => ("NVIDIA RTX A6000", 50_952_536_064),
                    other => panic!("no fake device {other}"),
                };
                DeviceInfo {
                    ordinal: u32::try_from(i).expect("small"),
                    name: name.to_string(),
                    total_bytes,
                    free_bytes: total_bytes / 1024 / 1024 * 1024,
                    uuid: [u8::try_from(i).expect("small") + 1; 16],
                    pci_bus: format!("0000:{:02x}:00.0", 0x41 + i),
                    held_by: None,
                }
            })
            .collect()
    }

    /// `BLOOMERY_DSPARK_CARD`'s card on a census ([`Place::draft_spec`]):
    /// unset, the 3090 wherever it is in view (beside the A6000 under `a`,
    /// the stage under `gate`, the tier under `bp`), in both enumeration
    /// orders; set, an ordinal or a name one device carries, the tier's
    /// device alone under `bp`; on two cards of one name, the non-stage one
    /// unset and either by its ordinal.
    #[test]
    fn the_draft_card_is_found_by_device() {
        let at = |p: Place, set: Option<&str>, c: &[DeviceInfo]| {
            p.draft_spec(set, c)
                .map(|s| (s.name, s.device.map(|d| d.ordinal)))
                .map_err(|e| e.to_string())
        };
        for order in [["3090", "A6000"], ["A6000", "3090"]] {
            let c = census(&order);
            let o = |n: &str| {
                Some(u32::try_from(order.iter().position(|m| *m == n).expect(n)).expect("small"))
            };
            for p in [Place::A, Place::Gate, Place::Bp] {
                assert_eq!(
                    at(p, None, &c),
                    Ok((RTX_3090.name, o("3090"))),
                    "{}",
                    p.name()
                );
            }
            assert_eq!(
                at(Place::A, Some("A6000"), &c),
                Ok((A6000.name, o("A6000")))
            );
            assert_eq!(
                at(Place::Gate, Some("a6000"), &c),
                Ok((A6000.name, o("A6000")))
            );
            let ordinal = format!("cuda{}", o("A6000").expect("A6000"));
            assert_eq!(
                at(Place::A, Some(&ordinal), &c),
                Ok((A6000.name, o("A6000")))
            );
            let tier = o("3090").expect("3090").to_string();
            assert_eq!(
                at(Place::Bp, Some(&tier), &c),
                Ok((RTX_3090.name, o("3090")))
            );
            assert_eq!(
                at(Place::Bp.on(&c).expect("bp"), Some("3090"), &c),
                Ok((RTX_3090.name, o("3090")))
            );
            let e = at(Place::Bp, Some("A6000"), &c).expect_err("the stage under bp");
            assert!(
                e.starts_with("BLOOMERY_DSPARK_CARD=\"A6000\" is cuda")
                    && e.contains("the draft sits on the tier card"),
                "{e}"
            );
        }
        let one = census(&["3090"]);
        assert_eq!(at(Place::Gate, None, &one), Ok((RTX_3090.name, Some(0))));
        assert_eq!(at(Place::A, None, &one), Ok((RTX_3090.name, Some(0))));
        let e = at(Place::A, Some("A6000"), &one).expect_err("no A6000");
        assert!(
            e.starts_with("BLOOMERY_DSPARK_CARD=\"A6000\": no visible device is A6000"),
            "{e}"
        );
        let two = census(&["3090", "3090"]);
        assert_eq!(at(Place::A, None, &two), Ok((RTX_3090.name, Some(1))));
        assert_eq!(at(Place::A, Some("0"), &two), Ok((RTX_3090.name, Some(0))));
        assert_eq!(
            at(Place::Bp, Some("cuda1"), &two),
            Ok((RTX_3090.name, Some(1)))
        );
        let e = at(Place::A, Some("3090"), &two).expect_err("two of one name");
        assert!(e.contains("2 visible devices carry the name 3090"), "{e}");
        let e = at(Place::Bp, Some("cuda0"), &two).expect_err("the stage of two");
        assert!(e.contains("the draft sits on the tier card"), "{e}");
        for (v, want) in [
            ("a6000+3090", "the lever names one card"),
            ("4090", "names the card \"4090\""),
            ("cuda7", "no visible device is cuda7"),
        ] {
            let e = at(Place::A, Some(v), &census(&["A6000", "3090"])).expect_err(v);
            assert!(e.contains(want), "{v}: {e}");
        }
        let e = at(Place::A, None, &[]).expect_err("no device");
        assert!(e.contains("0 visible devices"), "{e}");
    }
}
