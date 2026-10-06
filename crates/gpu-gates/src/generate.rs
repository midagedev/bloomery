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
//!
//! The diagnosis flags every generate binary shares ([`Diag`]) live here
//! too, generic over the session (`runtime::Target`): the serve's pick and
//! feed ([`NoEog`], [`ServeFeed`]), the rows behind the first tokens
//! ([`TopRows`]), runs chained by the serve seat's reset ([`repeat_runs`]),
//! and the stage card's copy of the slot map — read back against the host
//! map and the residency ledger ([`Residence`], [`slot_table`]), dumped to a
//! file ([`write_table`], [`dump_table`]) and loaded as a fixed placement
//! ([`place_table`], [`card_table`]). A binary hands in what only its body
//! names: the card's copy of the map, its host tier, its plan.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use bloomery_gpu::host::slots::{HOST, MAX_TIERS, SlotMap};
use bloomery_gpu::host::swap::{CardTable, SwapMachine, TableCheck};
use bloomery_gpu::host::tier::TierCard;
use bloomery_gpu::host::{HostExperts, HostTier};
use bloomery_gpu::model::{ChainBody, StepMode};
use bloomery_gpu::{DeviceTensor, Gpu, GpuError, GpuModel};
use cuda_core::{CudaStream, DeviceBuffer};
use gguf::Split;
use model::placement::workstation::{
    self, CardSpec, DeviceId, DeviceInfo, Pick, TierBatchBytes, TierDraft,
};
use model::placement::{self, Device, ExpertList, Machine, Plan, Role, Row};
use runtime::{Advance, Committed, Out, Target, Want};
use tokenizer::Tokenizer;

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

/// The devices a model runs on, the stage card `stage` then the expert tier
/// cards `tiers`, as a `load` line's `cards` field names them: by the names
/// their drivers report, each space written `_`, since the field is one
/// word. `planned` is the placement `--place <word>`'s cards by the names
/// the plan gives them, and `specs` the devices it resolved to on this
/// process's census (`None` for a census-free placement). Another count of
/// devices than the placement's cards, and a device that is not its card —
/// of a resolved placement, another device; of a census-free one, a name
/// that does not hold the card's — are refused by name. Every body's load
/// line names its cards here.
pub fn card_words(
    word: &str,
    planned: &[&str],
    specs: Option<&[CardSpec]>,
    stage: &Gpu,
    tiers: &[TierCard],
) -> Result<Vec<String>, GateError> {
    let mut gpus = vec![stage];
    gpus.extend(tiers.iter().map(TierCard::gpu));
    let mut devices = Vec::with_capacity(gpus.len());
    let mut ids = Vec::with_capacity(gpus.len());
    for g in &gpus {
        devices.push(g.device_name()?);
        ids.push(g.device_id()?);
    }
    let named = devices.len() == planned.len()
        && match specs {
            Some(specs) => ids
                .iter()
                .zip(specs)
                .all(|(id, s)| s.device.is_some_and(|d| d.uuid == id.uuid)),
            None => devices.iter().zip(planned).all(|(d, p)| d.contains(p)),
        };
    if !named {
        return Err(format!(
            "--place {word}: the model runs on {devices:?} ({}), the placement's cards are \
             {planned:?}",
            ids.iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        )
        .into());
    }
    Ok(devices.iter().map(|d| d.replace(' ', "_")).collect())
}

/// `r`, a `load` record ([`record::LOAD_GENERATOR`]) written up to its
/// `unified_addressing` field, with the devices the model runs on (`cards`,
/// [`card_words`] of the placement's cards) and, when the placement has an
/// expert tier card, the tier's experts and resident bytes. A device that is
/// not the placement's card ([`card_words`]), tiers that do not match the
/// placement's — another count, or a tier on another card — and more than
/// the one tier the record's fields hold are refused by name.
pub fn with_cards(
    r: Record,
    place: Place,
    stage: &Gpu,
    tiers: &[TierCard],
) -> Result<Record, GateError> {
    let specs = if place.resolved {
        Some(place.card_specs()?)
    } else {
        None
    };
    let words = card_words(place.name(), &place.cards(), specs.as_deref(), stage, tiers)?;
    let r = r.csv("cards", words);
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

// ---------------------------------------------------------------- diagnosis

/// The diagnosis flags a generate binary takes beside its own, each acting
/// on the plain runs of the prompt flags:
///
/// - `--top2 K`: a `top2` record for each of the first K generated tokens,
///   after the loop: the logits row behind that token (`i` 0 is the prompt
///   call's last row), read back after its step, by its two largest entries
///   — ids and values, the larger first, ties to the lower id as the argmax
///   takes them — and their margin. A row whose largest entry is not the
///   token the run took, and a NaN in a row, are named errors ([`TopRows`]).
/// - `--rows FILE`: the K rows `--top2` reads, whole, appended to FILE as
///   little-endian f32, run after run (FILE emptied before the first), each
///   run's followed by a `rows` record.
/// - `--repeat R`: the run R times on one load, each after the first from
///   `runtime::Target::reset` — the serve seat's reset on a request that
///   misses its cache, which leaves the adaptive residency where the run
///   before it took it — each opened by an `arm` record (`feed=repeat`)
///   ([`repeat_runs`]).
/// - `--table`: the stage card's copy of the slot map read back after the
///   runs, a `slot table` record: its entries against the host map, the
///   residency ledger (a load that runs the machine) and the map before the
///   first run, and the card slots two ids of one layer both name
///   ([`Residence`], [`slot_table`]).
/// - `--ignore-eos`: each token taken as the serve takes it under a
///   request's `ignore_eos` ([`NoEog`]); `--top2` ranks the rows without the
///   end-of-generation ids, which an `ignore eos` record names, and `--rows`
///   writes the rows as the model made them.
/// - `--serve-feed`: the prompt fed as the serve feeds it ([`ServeFeed`]):
///   every id but the last in one call, then the last id as a pass of the
///   run's pick, whose row is generated token 0's.
/// - `--dump-table FILE` (with `--repeat` 2 or more): the stage card's copy
///   of the map as the second run's token 0 ran on it, read after
///   `Target::reset` (FILE.before) and after that run's token-0 step (FILE),
///   and before the first run (FILE.seed) ([`write_table`]), and a `table
///   dump` record ([`dump_table`]).
/// - `--card-table FILE`: the load placed by that table, residency off:
///   each routed layer's card experts the ids FILE puts on the card, in id
///   order, the rest on the host, every other row the planner's
///   ([`place_table`]); a `card table` record after the load holds the
///   card's copy to FILE's sets ([`card_table`]).
#[derive(Clone, Debug)]
pub struct Diag {
    pub top2: usize,
    pub rows: Option<PathBuf>,
    pub repeat: usize,
    pub table: bool,
    /// The vocabulary's end-of-generation ids under `--ignore-eos`
    /// ([`Diag::finish`] reads them); empty without it.
    pub ignore_eos: Vec<u32>,
    pub serve_feed: bool,
    pub dump_table: Option<PathBuf>,
    pub card_table: Option<PathBuf>,
    /// `--ignore-eos` was given.
    no_eog: bool,
}

impl Default for Diag {
    /// No flag: one run, nothing read back.
    fn default() -> Diag {
        Diag {
            top2: 0,
            rows: None,
            repeat: 1,
            table: false,
            ignore_eos: Vec::new(),
            serve_feed: false,
            dump_table: None,
            card_table: None,
            no_eog: false,
        }
    }
}

impl Diag {
    /// The flags, as a usage line shows them.
    pub const USAGE: &'static str = "[--top2 K [--rows FILE]] [--repeat R] [--table] \
                                     [--ignore-eos] [--serve-feed] [--dump-table FILE] \
                                     [--card-table FILE]";

    /// Take `flag` when it is one of these, with its value from `it` when it
    /// takes one: `true` when taken, `false` for a flag of the binary's own.
    /// A flag given twice takes its last value.
    pub fn take(
        &mut self,
        flag: &str,
        it: &mut impl Iterator<Item = String>,
    ) -> Result<bool, GateError> {
        match flag {
            "--table" => self.table = true,
            "--ignore-eos" => self.no_eog = true,
            "--serve-feed" => self.serve_feed = true,
            "--top2" | "--repeat" | "--rows" | "--dump-table" | "--card-table" => {
                let v = it
                    .next()
                    .ok_or_else(|| format!("{flag} needs a value: {}", Diag::USAGE))?;
                match flag {
                    "--top2" => self.top2 = v.parse()?,
                    "--repeat" => self.repeat = v.parse()?,
                    "--rows" => self.rows = Some(PathBuf::from(v)),
                    "--dump-table" => self.dump_table = Some(PathBuf::from(v)),
                    _ => self.card_table = Some(PathBuf::from(v)),
                }
            }
            _ => return Ok(false),
        }
        Ok(true)
    }

    /// The flags against each other and against the run's `-n` (`n_gen`),
    /// each refusal by name; under `--ignore-eos` the end-of-generation ids
    /// of `model`'s vocabulary, one at least.
    pub fn finish(&mut self, n_gen: usize, model: &Path) -> Result<(), GateError> {
        if self.repeat == 0 {
            return Err("--repeat 0 runs nothing: give 1 or more".into());
        }
        if self.dump_table.is_some() && self.repeat < 2 {
            return Err(
                "--dump-table reads the second run's table: give --repeat 2 or more".into(),
            );
        }
        if self.dump_table.is_some() && self.card_table.is_some() {
            return Err("--dump-table reads a residency load; --card-table loads none".into());
        }
        if self.rows.is_some() && self.top2 == 0 {
            return Err("--rows writes the rows --top2 K reads: give --top2".into());
        }
        if self.top2 > n_gen {
            return Err(format!(
                "--top2 {} asks for more rows than -n {n_gen} tokens",
                self.top2
            )
            .into());
        }
        if self.no_eog {
            let tok = Tokenizer::from_gguf(model)
                .map_err(|e| format!("--ignore-eos: the vocabulary of {}: {e}", model.display()))?;
            self.ignore_eos = tok.eog().to_vec();
            if self.ignore_eos.is_empty() {
                return Err(format!(
                    "--ignore-eos: {} names no end-of-generation id",
                    model.display()
                )
                .into());
            }
        }
        Ok(())
    }

    /// The flags set, by name: a binary refuses them beside a run they do
    /// not act on (a draft, a probe, several slots).
    #[must_use]
    pub fn set(&self) -> Vec<&'static str> {
        [
            (self.top2 > 0, "--top2"),
            (self.rows.is_some(), "--rows"),
            (self.repeat > 1, "--repeat"),
            (self.table, "--table"),
            (self.no_eog, "--ignore-eos"),
            (self.serve_feed, "--serve-feed"),
            (self.dump_table.is_some(), "--dump-table"),
            (self.card_table.is_some(), "--card-table"),
        ]
        .into_iter()
        .filter_map(|(set, name)| set.then_some(name))
        .collect()
    }

    /// `--serve-feed` against a run of `ids` fed ids: the call runs every id
    /// but the last, so two at least.
    pub fn check_feed(&self, ids: usize) -> Result<(), GateError> {
        if self.serve_feed && ids < 2 {
            return Err(format!(
                "--serve-feed calls every id but the last: give 2 or more ids, not {ids}"
            )
            .into());
        }
        Ok(())
    }

    /// Before the first run: `--rows`' file emptied, and under
    /// `--ignore-eos` the `ignore eos` record.
    pub fn begin(&self) -> Result<(), GateError> {
        if let Some(p) = &self.rows {
            std::fs::File::create(p).map_err(|e| format!("--rows {}: {e}", p.display()))?;
        }
        if !self.ignore_eos.is_empty() {
            Record::new(&record::IGNORE_EOS)
                .csv("ids", &self.ignore_eos)
                .print();
        }
        Ok(())
    }
}

/// `--ignore-eos`'s pick, the serve's under `ignore_eos`
/// (`serve::genloop::choose`): the engine's argmax unless it is one of the
/// end-of-generation ids, else the largest logit of the row with those ids
/// at -inf, first of equals. One step a pass, every row read back.
#[derive(Clone, Debug)]
pub struct NoEog {
    ids: Vec<u32>,
}

impl NoEog {
    /// The pick passing over `ids`.
    #[must_use]
    pub fn new(ids: Vec<u32>) -> NoEog {
        NoEog { ids }
    }

    /// The token taken from `out`, a read back of [`Want::Logits`].
    #[must_use]
    pub fn pick(&self, out: &Out<'_>) -> u32 {
        let greedy = out.argmax();
        let Out::Logits { row, .. } = *out else {
            return greedy;
        };
        if !self.ids.contains(&greedy) {
            return greedy;
        }
        let mut best: Option<(usize, f32)> = None;
        for (i, &v) in row.iter().enumerate() {
            let v = match u32::try_from(i) {
                Ok(id) if self.ids.contains(&id) => f32::NEG_INFINITY,
                _ => v,
            };
            if best.is_none_or(|(_, b)| v.total_cmp(&b).is_gt()) {
                best = Some((i, v));
            }
        }
        best.and_then(|(i, _)| u32::try_from(i).ok())
            .unwrap_or(greedy)
    }
}

impl<T: Target> Advance<T> for NoEog {
    fn prompt(&mut self, t: &mut T, ids: &[u32]) -> Result<u32, T::Error> {
        let out = t.prompt(ids, Want::Logits)?;
        Ok(self.pick(&out))
    }

    fn begin(&mut self, _t: &T, _prompt: &[u32], _first: u32) -> Result<(), T::Error> {
        Ok(())
    }

    fn pass(&mut self, t: &mut T, last: u32, out: &mut Vec<u32>) -> Result<Committed, T::Error> {
        let pos = t.pos();
        let o = t.step(last, Want::Logits)?;
        out.push(self.pick(&o));
        Ok(Committed {
            pos,
            kept: 1,
            rows: 1,
            proposed: false,
        })
    }
}

/// `--serve-feed`'s prompt, the serve's (`serve::genloop`:
/// `prefill_marked(ids, 0, n - 1)`, then `next(ids[n - 1])`): every id but
/// the last in one call, then the last id as one of `inner`'s passes, whose
/// kept token is generated token 0. The passes after it are `inner`'s own.
pub struct ServeFeed<'a, A> {
    pub inner: &'a mut A,
}

impl<T, A> Advance<T> for ServeFeed<'_, A>
where
    T: Target,
    T::Error: From<GpuError>,
    A: Advance<T>,
{
    const ROWS: usize = A::ROWS;

    fn prompt(&mut self, t: &mut T, ids: &[u32]) -> Result<u32, T::Error> {
        let refused = |detail: &str| GpuError::Shape {
            what: "ServeFeed",
            detail: detail.to_string(),
        };
        let Some((&last, head)) = ids.split_last() else {
            return Err(refused("an empty prompt: the serve feeds one id or more").into());
        };
        if !head.is_empty() {
            t.prompt(head, Want::Argmax)?;
        }
        let mut out = Vec::with_capacity(A::ROWS);
        self.inner.pass(t, last, &mut out)?;
        out.first()
            .copied()
            .ok_or_else(|| refused("the last id's pass kept no token").into())
    }

    fn begin(&mut self, t: &T, prompt: &[u32], first: u32) -> Result<(), T::Error> {
        self.inner.begin(t, prompt, first)
    }

    fn pass(&mut self, t: &mut T, last: u32, out: &mut Vec<u32>) -> Result<Committed, T::Error> {
        self.inner.pass(t, last, out)
    }
}

/// A logits row's two largest entries, the larger first, ties to the lower
/// id as the argmax takes them.
struct Top2 {
    ids: [usize; 2],
    values: [f32; 2],
}

impl Top2 {
    /// The logits row behind `token`, the token the run took from it,
    /// ranked without the ids in `skip`; a row whose largest such entry is
    /// another, and a NaN, are named errors.
    fn of(row: &[f32], token: u32, skip: &[u32]) -> Result<Top2, GateError> {
        let mut t = Top2 {
            ids: [usize::MAX; 2],
            values: [f32::NEG_INFINITY; 2],
        };
        for (i, &v) in row.iter().enumerate() {
            if v.is_nan() {
                return Err(format!("top2: the logits row is NaN at id {i}").into());
            }
            if u32::try_from(i).is_ok_and(|i| skip.contains(&i)) {
                continue;
            }
            if v > t.values[0] {
                t = Top2 {
                    ids: [i, t.ids[0]],
                    values: [v, t.values[0]],
                };
            } else if v > t.values[1] {
                t.ids[1] = i;
                t.values[1] = v;
            }
        }
        if u32::try_from(t.ids[0]).ok() != Some(token) {
            return Err(format!(
                "top2: the row's largest entry is id {}, and the run took {token} from it",
                t.ids[0]
            )
            .into());
        }
        Ok(t)
    }

    fn print(&self, i: usize) {
        let [a, b] = self.values.map(f64::from);
        Record::new(&record::TOP2)
            .u("i", i)
            .u("top1", self.ids[0])
            .f("top1_logit", a)
            .u("top2", self.ids[1])
            .f("top2_logit", b)
            .f("margin", a - b)
            .print();
    }
}

/// A run's `--top2` rows ([`Diag`]): the first `top2` logits rows the run's
/// tokens came from, each ranked as it is read, and under `--rows` kept
/// whole.
pub struct TopRows {
    top2: usize,
    skip: Vec<u32>,
    tops: Vec<Top2>,
    kept: Option<Vec<f32>>,
}

impl TopRows {
    /// The rows `d` asks a run for.
    #[must_use]
    pub fn new(d: &Diag) -> TopRows {
        TopRows {
            top2: d.top2,
            skip: d.ignore_eos.clone(),
            tops: Vec::with_capacity(d.top2),
            kept: d.rows.as_ref().map(|_| Vec::new()),
        }
    }

    /// Whether the run still wants a row: fewer than `--top2` read.
    #[must_use]
    pub fn wants(&self) -> bool {
        self.tops.len() < self.top2
    }

    /// The logits `row` behind generated token `token`, the next one read.
    pub fn read(&mut self, row: &[f32], token: u32) -> Result<(), GateError> {
        self.tops.push(Top2::of(row, token, &self.skip)?);
        if let Some(k) = self.kept.as_mut() {
            k.extend_from_slice(row);
        }
        Ok(())
    }

    /// After the run's loop: a `top2` record a row and, under `--rows`
    /// (`rows`), the rows appended whole to that file and the `rows` record.
    pub fn finish(&self, rows: Option<&Path>) -> Result<(), GateError> {
        for (i, t) in self.tops.iter().enumerate() {
            t.print(i);
        }
        let (Some(path), Some(kept)) = (rows, &self.kept) else {
            return Ok(());
        };
        let bytes: Vec<u8> = kept.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::OpenOptions::new()
            .append(true)
            .open(path)
            .and_then(|mut f| f.write_all(&bytes))
            .map_err(|e| format!("--rows {}: {e}", path.display()))?;
        Record::new(&record::ROWS)
            .u("rows", self.tops.len())
            .u("n", kept.len() / self.tops.len().max(1))
            .u("bytes", bytes.len())
            .w("path", path.display())
            .print();
        Ok(())
    }
}

/// `--repeat`'s `runs` runs on one load: `each(t, i)` in order, each run
/// after the first from `Target::reset` (the adaptive residency stays where
/// the run before it took it), each opened by its `arm` record (`ids` fed
/// ids, `n_gen` generated tokens). The first failure ends them, naming its
/// run and `bin`.
pub fn repeat_runs<T: Target>(
    t: &mut T,
    runs: usize,
    ids: usize,
    n_gen: usize,
    bin: &str,
    mut each: impl FnMut(&mut T, usize) -> Result<(), GateError>,
) -> Result<(), GateError> {
    for i in 0..runs {
        if i > 0 {
            t.reset()?;
        }
        Record::new(&record::ARM)
            .u("i", i)
            .u("arms", runs)
            .w("feed", "repeat")
            .u("ids", ids)
            .u("n", n_gen)
            .print();
        each(t, i).map_err(|e| format!("{bin}: repeat {i} of {runs}: {e}"))?;
    }
    Ok(())
}

/// What the residency diagnostics read of a loaded body: the stage card's
/// copy of the slot map, which the body holds and hands the machine at its
/// start; the host tier's map and residency machine (`HostTier::slots`,
/// `HostTier::swap`); and the engine stream, which writes the copy. A
/// binary supplies them from its body; the readback is the residency's own
/// ([`CardTable::read`], [`CardTable::check`]).
pub struct Residence<'a> {
    pub card: &'a DeviceBuffer<u32>,
    pub map: &'a SlotMap,
    pub machine: Option<&'a SwapMachine>,
    pub stream: &'a CudaStream,
}

impl<'a> Residence<'a> {
    /// A body's views: `card` its stage card's copy of the map, `tier` its
    /// host tier, on `gpu`'s engine stream.
    #[must_use]
    pub fn of<H: HostExperts>(
        gpu: &'a Gpu,
        card: &'a DeviceTensor<u32>,
        tier: &'a HostTier<H>,
    ) -> Residence<'a> {
        Residence {
            card: card.buf(),
            map: tier.slots(),
            machine: tier.swap(),
            stream: gpu.stream(),
        }
    }

    /// The stage card's copy, read back.
    pub fn table(&self) -> Result<CardTable, GateError> {
        Ok(CardTable::read(
            self.card,
            self.map,
            self.machine,
            self.stream,
        )?)
    }

    /// `t` against the host map and, when the load runs a machine, its
    /// ledger.
    pub fn check(&self, t: &CardTable) -> Result<TableCheck, GateError> {
        Ok(t.check(self.map, self.machine.map(SwapMachine::ledger))?)
    }
}

/// The `slot table` record of `t`, checked as `c`, `vs_load` of its entries
/// off the map before the first run.
pub fn slot_table(t: &CardTable, c: &TableCheck, vs_load: usize) -> Record {
    let show = |v: u32| match v {
        HOST => "h".to_string(),
        v => v.to_string(),
    };
    let first: Vec<String> = c
        .first
        .iter()
        .map(|o| {
            format!(
                "l{}/e{}:card={},map={},ledger={}",
                o.layer,
                o.id,
                show(o.card),
                show(o.map),
                show(o.ledger)
            )
        })
        .collect();
    Record::new(&record::SLOT_TABLE)
        .u("layers", t.layers().len())
        .u("entries", t.entries().len())
        .u("on_card", t.on_card())
        .w("machine", c.vs_ledger.is_some())
        .u("vs_map", c.vs_map)
        .u("vs_ledger", c.vs_ledger.unwrap_or(0))
        .u("vs_load", vs_load)
        .u("doubled", c.doubled)
        .w(
            "first",
            if first.is_empty() {
                "none".to_string()
            } else {
                first.join(";")
            },
        )
}

/// `FILE.before`: the `--dump-table` read after `Target::reset`.
#[must_use]
pub fn before_path(p: &Path) -> PathBuf {
    suffixed(p, ".before")
}

/// `FILE.seed`: the `--dump-table` read before the first run.
#[must_use]
pub fn seed_path(p: &Path) -> PathBuf {
    suffixed(p, ".seed")
}

fn suffixed(p: &Path, suffix: &str) -> PathBuf {
    let mut s = p.as_os_str().to_owned();
    s.push(suffix);
    PathBuf::from(s)
}

/// `t` into `path`: the first layer, the layer count, the expert count,
/// then the entries, little-endian `u32`s.
pub fn write_table(path: &Path, t: &CardTable) -> Result<(), GateError> {
    let layers = t.layers();
    let head = [layers.start, layers.len(), t.n_expert()]
        .map(u32::try_from)
        .into_iter()
        .collect::<Result<Vec<u32>, _>>()?;
    let bytes: Vec<u8> = head
        .iter()
        .chain(t.entries())
        .flat_map(|v| v.to_le_bytes())
        .collect();
    std::fs::write(path, bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(())
}

/// [`write_table`]'s file back; a length that is not its header's is
/// refused by name.
pub fn read_table(path: &Path) -> Result<CardTable, GateError> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let (words, rest) = bytes.as_chunks::<4>();
    let words: Vec<u32> = words.iter().map(|&b| u32::from_le_bytes(b)).collect();
    let [start, len, n, ..] = words[..] else {
        return Err(format!("{}: {} bytes hold no header", path.display(), bytes.len()).into());
    };
    let (start, len, n) = (start as usize, len as usize, n as usize);
    if !rest.is_empty() || words.len() != 3 + len * n {
        return Err(format!(
            "{}: {} bytes; its header names {len} layers of {n} experts",
            path.display(),
            bytes.len()
        )
        .into());
    }
    Ok(CardTable::new(start..start + len, n, words[3..].to_vec())?)
}

/// `--dump-table`: `r`'s card copy into `path`, held to its read after
/// `Target::reset` (FILE.before, written before), and the `table dump`
/// record: the entries, those on the card, those off the host map, and
/// whether the two reads are one table.
pub fn dump_table(r: &Residence<'_>, path: &Path) -> Result<Record, GateError> {
    let t = r.table()?;
    write_table(path, &t)?;
    let before = read_table(&before_path(path))?;
    let c = t.check(r.map, None)?;
    Ok(Record::new(&record::TABLE_DUMP)
        .u("entries", t.entries().len())
        .u("on_card", t.on_card())
        .u("vs_map", c.vs_map)
        .w("same_before", before == t)
        .w("path", path.display()))
}

/// `--card-table`: each routed layer of `plan` holds on its stage card the
/// ids `t` puts on the card, in id order (`placement::routed_row`, the one
/// owner of a stack's split), the rest on the host; the layer's `n_l`
/// follows. A stack the planner left wholly on the host stays there when `t`
/// puts none of its layer on the card. Refused by name: another expert
/// count, a routed layer `t` does not cover, a stack of other than one card
/// segment that `t` puts experts of on the card, and a card format other
/// than the planner's.
pub fn place_table(plan: &mut Plan<'_>, t: &CardTable) -> Result<(), GateError> {
    let model = plan.model;
    let n = t.n_expert();
    let layers = t.layers();
    if u64::try_from(n)? != model.experts {
        return Err(format!(
            "--card-table: {n} experts a layer, the model's {}",
            model.experts
        )
        .into());
    }
    for row in &mut plan.rows {
        let tensor = &model.tensors[row.tensor];
        if tensor.role != Role::RoutedExperts {
            continue;
        }
        let l = tensor
            .layer
            .ok_or_else(|| format!("{}: a routed stack without a layer", tensor.name))?;
        if !layers.contains(&l) {
            return Err(format!("--card-table covers layers {layers:?}, not {l}").into());
        }
        let at = (l - layers.start) * n;
        let ids: Vec<u32> = (0..n)
            .filter(|&e| t.entries()[at + e] != HOST)
            .map(u32::try_from)
            .collect::<Result<_, _>>()?;
        let cards: Vec<&placement::Segment> = row
            .segments
            .iter()
            .filter(|s| matches!(s.device, Device::Card(_)))
            .collect();
        if cards.is_empty() && ids.is_empty() {
            continue;
        }
        let [card] = cards[..] else {
            return Err(format!(
                "{}: {} card segments; --card-table places one card",
                tensor.name,
                cards.len()
            )
            .into());
        };
        let Device::Card(c) = card.device else {
            return Err(format!("{}: a card segment off the card", tensor.name).into());
        };
        let len = ids.len();
        let new = placement::routed_row(
            row.tensor,
            tensor,
            c,
            ExpertList::new(ids, model.experts)?,
            model,
        )?;
        let fmt = new
            .segments
            .iter()
            .find(|s| s.device == Device::Card(c))
            .map(|s| s.format);
        if fmt != Some(card.format) {
            return Err(format!(
                "{}: routed_row's card format {fmt:?}, the planner's {:?}",
                tensor.name, card.format
            )
            .into());
        }
        *row = Row {
            segments: new.segments,
            ..row.clone()
        };
        if let Some(nl) = plan.n_l.get_mut(l) {
            *nl = u64::try_from(len)?;
        }
    }
    Ok(())
}

/// The `card table` record of a `--card-table` load: its card's copy
/// `loaded` against `file`'s table, read from `path`: the experts on one
/// side only, and those on the card in both at another slot.
pub fn card_table(loaded: &CardTable, file: &CardTable, path: &Path) -> Result<Record, GateError> {
    let d = loaded.sets_vs(file)?;
    Ok(Record::new(&record::CARD_TABLE)
        .u("layers", loaded.layers().len())
        .u("on_card", loaded.on_card())
        .u("set_diff", d.set)
        .u("slot_diff", d.slot)
        .w("path", path.display()))
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
