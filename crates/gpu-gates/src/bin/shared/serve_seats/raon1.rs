//! `bloomery-serve --model raon1` — the llama-server-compatible HTTP API on
//! the Raon-1 engine with one sequence: a Raon-1-mini file (RDA and gated
//! latent layers, SiTU experts, Raon_mHC streams), the whole model on one
//! card, opened as `gate_raon1_e2e` opens it (`GpuModel::<raon1::Body>::open`;
//! the step captured in graph mode), behind the session (`app::Session`),
//! and its DFlash2 draft beside it when the draft's file is there.
//!
//!     bloomery-serve --model raon1 [-m PATH | --hf <repo>[:<quant>]]
//!                    [--host 127.0.0.1] [--port 8080] [--ctx C] [--alias NAME]
//!                    [--chat-template-file PATH] [--place W] [--plan]
//!                    [--queue-depth Q] [--api-key KEY] [--api-key-file FNAME]
//!
//! The flags are the nemotron seat's, read the same way: the model's first
//! shard gives the vocabulary, `tokenizer.chat_template` the chat template
//! (`--chat-template-file` replaces it) and `general.name` the alias
//! (`--alias` replaces it; `raon1` for a file that names none).
//!
//! Three facts of the architecture set this seat's shape:
//!
//! - The body holds one sequence: its latent layers' caches of one context
//!   and its RDA layers' states and conv rings. It has no slot rows, so
//!   `--parallel` above 1 is refused by name before any plan.
//! - The prompt runs by the body's prompt rule (`app::Prompt`).
//! - An RDA state cannot step back past what a pass checkpointed. A request
//!   keeps the prefix it shares with what the slot holds only when that
//!   prefix is the whole held sequence; any other shared prefix is kept 0,
//!   the slot reset and the prompt fed again. A rollback to a position the
//!   model does not stand at, a snapshot and a resume are refused by name,
//!   and the seat runs no prompt cache.
//!
//! The draft is Raon-1-mini's DFlash2 draft (`raon1_dflash`: the file
//! `$BLOOMERY_DSPARK_MODEL`, else refset's `raon1::DFLASH_BESIDE` file beside the target),
//! on the target's card, verified by the session's pass of 8 rows behind the
//! width chooser (`BLOOMERY_MTP_WIDTH`), through the serve seats' one owner of
//! a served DSpark draft (`served::ServedSpark`), over the verifying load
//! (`GpuModel::open_verifying`) when a draft is there. `BLOOMERY_DRAFT` picks the requests
//! it runs ([`Drafts`]): `dspark` every greedy one, `off` none, unset the
//! requests a thinking-on prompt opens ([`Think::inside`]). A request that
//! samples runs plain steps; a request the draft does not run costs the draft
//! nothing.
//!
//! `--place W` is the shared placement word (`generate::Place`); unset, the
//! common untiered rule decides on one census reading. A placement the plan
//! refuses (more than one card, a tier card, a card the file does not fit)
//! stops before any load by the plan's own refusal. `--ctx` is the slot's
//! context, chosen by `generate::choose_ctx` as every seat's is.
//!
//! The records on stderr: the `draft rule` line, `place unset`, one `ctx`
//! line, the `plan` record, the `parallel` line, then the `load` record, the
//! draft's `load draft=dflash2` record when one loads and, once the port is
//! bound, the `listening` record (`record::BLOOMERY_SERVE_RAON1`); the width
//! chooser's `mtp width` record at each drafted request's end. `--plan` ends
//! the process after the `parallel` line, nothing loaded.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use app::Session;
use bloomery_gpu::arch::raon1::Body;
use bloomery_gpu::model::{ChainBody, StepMode};
use bloomery_gpu::{Gpu, GpuModel};
use bloomery_gpu_gates::bind::{
    ChatSurface, Pass, Seat, SeatEngine, SeatFlags, Vocab, arm_screen, model_props, note_line,
    parallel_line, placement_of, slots_of,
};
use bloomery_gpu_gates::generate::{Place, choose_ctx};
use bloomery_gpu_gates::need;
use bloomery_gpu_gates::record::{self, Record};
use bloomery_gpu_gates::{GateError, gpu_census, ref_model_path};
use bloomery_levers::Levers;
use gguf::Split;
use model::arch::Arch;
use model::arch::dflash2::Hparams as DraftHparams;
use model::arch::models::ReasoningFormat;
use model::arch::raon1::place::{PlanInputs, VERIFY_LANES};
use model::placement::PlanLevers;
use model::placement::ctx_ask::{DEFAULT_SLOTS, probe_of};
use model::placement::workstation::{DRAFT_RESERVE, DeviceId};
use runtime::width::Mode as WidthMode;
use serve::{CacheNote, Drafted, EngineProps, Saved, ServeError};
use tokenizer::Tokenizer;

use super::served::ServedSpark;

#[path = "../quiet.rs"]
mod quiet;

#[allow(
    dead_code,
    reason = "the serve seat and two gates share the opener; each takes the parts it reads"
)]
#[path = "../raon1_dflash.rs"]
mod raon1_dflash;

use quiet::Quiet;
use raon1_dflash::{Pick, RaonDraft};

const NAME: &str = "bloomery-serve-raon1";

const USAGE: &str = "usage: bloomery-serve --model raon1 [-m PATH | --hf <repo>[:<quant>]] \
                     [--host H] [--port P] [--ctx C] [--alias NAME] [--chat-template-file PATH] \
                     [--place W] [--plan] [--queue-depth Q] [--api-key KEY] \
                     [--api-key-file FNAME]";

/// The levers this seat acts on: the plan's card budget, the start screen,
/// how a structured-output request's fixed runs are fed, the draft word and
/// the width chooser's mode.
pub const ACTS_ON: &[&str] = &[
    bloomery_levers::CARD_BUDGET,
    bloomery_levers::SCREEN,
    bloomery_levers::SCHEMAJUMP,
    bloomery_levers::DRAFT,
    bloomery_levers::MTP_WIDTH,
];

/// Why the seat takes no sequence state: it runs no prompt cache.
const NO_CACHE: &str = "the raon1 seat runs no prompt cache";

/// Why a prefix shorter than the held sequence is not kept.
const NO_STEP_BACK: &str =
    "the RDA state cannot step back: a prefix is kept only where the sequence stands";

/// A verify of a proposal: the token at the target's position, then the
/// draft's 7 ids.
const VERIFY_ROWS: usize = 8;

// The plan counts a lane a verified row, and the body's verifying load holds that many.
const _: () = assert!(
    VERIFY_LANES.get() == VERIFY_ROWS && VERIFY_ROWS == bloomery_gpu::linear::PASS_ROWS,
    "the plan's lanes, the verify's rows and the body's lanes are one count"
);

/// The draft's file name in records and `/props`.
const DRAFT_WORD: &str = "dflash2";

/// The think span's two tags as the vocabulary's special tokens.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Think {
    open: u32,
    close: u32,
}

impl Think {
    /// The tags of `format` among `tok`'s special tokens; `Err` names the tag
    /// the vocabulary does not hold as one.
    fn of(tok: &Tokenizer, format: Option<ReasoningFormat>) -> Result<Think, String> {
        let Some(ReasoningFormat::ThinkSpan { open, close }) = format else {
            return Err("the model's description names no think span".to_owned());
        };
        let id = |text: &str| {
            tok.special_tokens()
                .iter()
                .copied()
                .find(|&id| tok.text(id) == Some(text))
                .ok_or_else(|| format!("the vocabulary holds no special token {text:?}"))
        };
        Ok(Think {
            open: id(open)?,
            close: id(close)?,
        })
    }

    /// Whether a prompt call of `ids` leaves the model inside its think span:
    /// `Some(true)` when the last tag among them opens the span, `Some(false)`
    /// when it closes it, `None` when they hold neither tag.
    ///
    /// This is the rule the unset draft runs by, and why. A drafted pass runs
    /// the draft's block and one verify of 8 rows and keeps τ positions, the
    /// draft's ids the target agrees with and its own; τ plain steps cost τ
    /// of the target's step. With t1 the step, t8 the verify of 8 rows and
    /// t_d the draft's block and append, the pass is faster when
    /// τ > τ* = (t8 + t_d) / t1. On this model at its one-card load the verify
    /// costs under two steps and the draft well under one, so τ* is a little
    /// over 2. The acceptance the vendor's reference measures on
    /// chat-templated prompts sits under τ* on a reply with thinking off
    /// (prose, English and Korean) and well over it with thinking on, whose
    /// reply opens with its reasoning: drafted, a thinking-off request runs
    /// slower and a thinking-on one faster. Raon's template ends a thinking-on
    /// prompt inside an open span (`<think>`) and writes the span closed
    /// (`<think></think>`) when thinking is off, so the last tag of the prompt
    /// call says which. A thinking-off reply whose acceptance is high (code)
    /// is not told apart by its prompt and runs plain steps;
    /// `BLOOMERY_DRAFT=dspark` drafts every request.
    fn inside(self, ids: &[u32]) -> Option<bool> {
        ids.iter().rev().find_map(|&id| {
            if id == self.open {
                Some(true)
            } else if id == self.close {
                Some(false)
            } else {
                None
            }
        })
    }
}

/// The requests the draft runs, as `BLOOMERY_DRAFT` and the files decide.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Drafts {
    /// None: `off`, or unset with no draft to run.
    Never,
    /// `dspark`: every greedy request.
    Every,
    /// Unset, with the draft's file there: each request whose prompt call
    /// leaves the model inside its think span ([`Think::inside`]), up to the
    /// tag that closes it.
    Thinking(Think),
}

impl Drafts {
    /// The `draft rule` record's word.
    fn word(self) -> &'static str {
        match self {
            Drafts::Never => "off",
            Drafts::Every => "every",
            Drafts::Thinking(_) => "thinking",
        }
    }

    /// Whether a drafting request still drafts once `ids` follow: under the
    /// unset rule a request leaves its think span at the close tag and runs
    /// plain steps from there; `dspark` drafts through it.
    fn still(self, ids: &[u32]) -> bool {
        match self {
            Drafts::Never => false,
            Drafts::Every => true,
            Drafts::Thinking(t) => t.inside(ids) != Some(false),
        }
    }
}

/// The draft the engine thread opens, and the requests it runs.
struct OpenDraft {
    path: PathBuf,
    hp: DraftHparams,
    width: WidthMode,
    drafts: Drafts,
    /// What the plan sets aside for the draft on the target's card ([`raon1_dflash::reserve`]).
    reserve: u64,
}

/// What `BLOOMERY_DRAFT` resolves to for a target at `target` (`split` opened), before any
/// plan: the draft to open, if any, with its card reserve, and the `draft rule` record's why.
/// `dspark` with no draft file, `lookup`, `mtp` and a file that is not a DFlash2 draft are
/// refused by name.
fn draft_of(
    levers: &Levers,
    (target, split): (&std::path::Path, &Split),
    think: Result<Think, String>,
) -> Result<(Option<OpenDraft>, String), GateError> {
    let width = WidthMode::of(levers.mtp_width())?;
    let set = refset::arch::draft::dspark_lever().map(PathBuf::from);
    let word = levers.draft();
    let refuse = |w: &str| -> GateError {
        format!(
            "BLOOMERY_DRAFT={w}: the raon1 seat serves its DFlash2 draft (dspark), off or unset"
        )
        .into()
    };
    let every = match word {
        None => false,
        Some("dspark") => true,
        Some("off") => {
            if levers.mtp_width().is_some() {
                return Err(
                    "BLOOMERY_MTP_WIDTH picks the width a drafted window verifies; \
                            BLOOMERY_DRAFT=off serves no draft"
                        .into(),
                );
            }
            return Ok((None, "BLOOMERY_DRAFT=off".to_owned()));
        }
        Some(other) => return Err(refuse(other)),
    };
    let (path, from) = match raon1_dflash::pick(set.as_deref(), target)? {
        Pick::File(path, from) => (path, from),
        Pick::Absent(dirs) => {
            let looked: Vec<String> = dirs.iter().map(|d| d.display().to_string()).collect();
            if every {
                return Err(format!(
                    "BLOOMERY_DRAFT=dspark: no DFlash2 draft at {}",
                    looked.join(", ")
                )
                .into());
            }
            if levers.mtp_width().is_some() {
                return Err(format!(
                    "BLOOMERY_MTP_WIDTH picks the width a drafted window verifies; no draft \
                     file is in {}",
                    looked.join(", ")
                )
                .into());
            }
            return Ok((None, format!("no DFlash2 draft at {}", looked.join(", "))));
        }
    };
    let file = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let hp = DraftHparams::read(&file).map_err(|e| {
        format!(
            "{} ({}): {e}; the raon1 seat serves a DFlash2 draft",
            path.display(),
            from.describe()
        )
    })?;
    let reserve = raon1_dflash::reserve(&file, &hp, split)?;
    drop(file);
    let at = format!("{} ({})", path.display(), from.describe());
    let (drafts, why) = if every {
        (Drafts::Every, format!("BLOOMERY_DRAFT=dspark, {at}"))
    } else {
        match think {
            Ok(t) => (Drafts::Thinking(t), format!("unset, {at}")),
            Err(why) => {
                if levers.mtp_width().is_some() {
                    return Err(format!(
                        "BLOOMERY_MTP_WIDTH picks the width a drafted window verifies; the \
                         unset draft runs no request: {why}"
                    )
                    .into());
                }
                return Ok((None, format!("unset, {at}: {why}")));
            }
        }
    };
    Ok((
        Some(OpenDraft {
            path,
            hp,
            width,
            drafts,
            reserve,
        }),
        why,
    ))
}

/// What the engine thread opens the model with.
struct OpenSeat {
    card: String,
    device: Option<DeviceId>,
    path: PathBuf,
    /// The slot's positions the latent caches are sized for.
    ctx: usize,
    draft: Option<OpenDraft>,
}

/// The model's session, in graph mode with its step captured, and the draft.
struct Raon1 {
    s: Session<Body>,
    ctx: usize,
    draft: Option<ServedSpark<Body>>,
    drafts: Drafts,
    /// Whether the request in flight runs through the draft.
    drafting: bool,
}

impl Raon1 {
    /// The model of the file at `a.path` on the planned card (the verifying
    /// load when a draft is to open), the `load` record on stderr, then the
    /// draft beside it on the same card with its `load draft=dflash2` record
    /// and its verify passes captured.
    fn open(a: OpenSeat) -> Result<Raon1, GateError> {
        let t = Instant::now();
        let file = Split::open(&a.path).map_err(|e| format!("open {}: {e}", a.path.display()))?;
        let opened = t.elapsed();
        let gpu = Gpu::open_card_after_hook(&a.card, a.device)?;
        let mut m = if a.draft.is_some() {
            GpuModel::<Body>::open_verifying(gpu, file, a.ctx)?
        } else {
            GpuModel::<Body>::open(gpu, file, a.ctx)?
        };
        m.set_mode(StepMode::Graph);
        let nodes = m.capture_step()?;
        let total = t.elapsed();
        Record::new(&record::LOAD_RAON1)
            .w("arch", Body::arch().name())
            .u("resident_bytes", m.resident_bytes())
            .u("ctx", m.ctx_max())
            .u("slots", 1)
            .u("layers", m.layers().len())
            .u("graph_nodes", nodes)
            .f("load_s", total.as_secs_f64())
            .eprint();
        record::eprint_load_phases(&m, total, opened);
        let mut s = Session::from_model(m, u32::try_from(a.ctx)?);
        let Some(d) = a.draft else {
            return Ok(Raon1 {
                s,
                ctx: a.ctx,
                draft: None,
                drafts: Drafts::Never,
                drafting: false,
            });
        };
        let t = Instant::now();
        let target =
            Arc::new(Split::open(&a.path).map_err(|e| format!("open {}: {e}", a.path.display()))?);
        // The common loop names its card for the life of the process.
        let card: &'static str = Box::leak(a.card.into_boxed_str());
        let gpu = Gpu::open_card(card, a.device)?;
        let mut spark = raon1_dflash::open(gpu, &d.path, (target, s.model()), a.ctx, card)?;
        let (free, total) = spark.mem_info()?;
        Record::new(&record::LOAD_DRAFT)
            .w("draft", DRAFT_WORD)
            .w("card", spark.card())
            .u("width", VERIFY_ROWS - 1)
            .list("target_layers", &d.hp.target_layers)
            .u("feature_width", d.hp.fc_in())
            .u("resident", spark.resident_bytes())
            .u("working", spark.held().working)
            .u("reserve", d.reserve)
            .u("taken", spark.taken_bytes())
            .u("draft_card_free", free)
            .u("draft_card_total", total)
            .f("load_s", t.elapsed().as_secs_f64())
            .eprint();
        // The draft's load left its own context current; the verify captures
        // read the stream's.
        s.model().gpu().context().bind_to_thread()?;
        let draft = ServedSpark::open::<RaonDraft, VERIFY_ROWS>(
            &mut s,
            spark,
            d.width,
            &mut Quiet,
            Some(d.path),
        )?;
        Ok(Raon1 {
            s,
            ctx: a.ctx,
            draft: Some(draft),
            drafts: d.drafts,
            drafting: false,
        })
    }

    /// The body's keep rule: `n` when it is the whole held sequence, else
    /// nothing.
    fn body_keep(&self, n: usize) -> (usize, Option<String>) {
        if n == self.pos() {
            (n, None)
        } else {
            (0, Some(NO_STEP_BACK.to_owned()))
        }
    }

    /// A call the draft did not take: the draft no longer follows.
    fn left_behind(&mut self) {
        if let Some(d) = &mut self.draft {
            d.moved();
        }
    }
}

impl Seat for Raon1 {
    fn pos(&self) -> usize {
        self.s.model().pos() as usize
    }

    fn ctx_max(&self) -> usize {
        self.ctx
    }

    /// The prompt call: through the draft when the request drafts ([`Drafts`];
    /// a call that holds no think tag keeps the request's choice), else the
    /// body's prompt.
    fn prefill(&mut self, ids: &[u32]) -> Result<u32, GateError> {
        let pos = self.s.model().pos();
        // Under the unset rule the cut keeps the body's prefix and carries no
        // window, so a request drafts only where the draft holds every
        // position before it: a draft started over past position 0 proposes
        // nothing until a window's rows follow.
        self.drafting = match (&self.draft, self.drafts) {
            (Some(_), Drafts::Every) => true,
            (Some(d), Drafts::Thinking(t)) => {
                d.follows(pos) && t.inside(ids).unwrap_or(self.drafting)
            }
            _ => false,
        };
        if let (Some(d), true) = (&mut self.draft, self.drafting) {
            return d.prefill(&mut self.s, ids);
        }
        self.left_behind();
        Ok(<Body as app::Prompt>::prompt(self.s.model_mut(), ids)?)
    }

    /// One plain step; the draft takes in the position when the request
    /// drafts, and the request stops drafting when the id ends it
    /// ([`Drafts::still`]).
    fn step(&mut self, last: u32) -> Result<u32, GateError> {
        let next = self.s.model_mut().step(&[last])?;
        match (&mut self.draft, self.drafting) {
            (Some(d), true) => {
                d.stepped(&mut self.s, last, next)?;
                self.drafting = self.drafts.still(&[next]);
            }
            _ => self.left_behind(),
        }
        Ok(next)
    }

    /// The draft's pass ([`ServedSpark::pass`]) when the request drafts, else
    /// one plain step; the request stops drafting when the pass's ids end it
    /// ([`Drafts::still`]).
    fn pass(&mut self, last: u32, out: &mut Vec<u32>) -> Result<Drafted, GateError> {
        if let (Some(d), true) = (&mut self.draft, self.drafting) {
            let before = out.len();
            let drafted = d.pass(&mut self.s, last, out)?;
            self.drafting = self.drafts.still(&out[before..]);
            return Ok(drafted);
        }
        out.push(self.step(last)?);
        Ok(Drafted::default())
    }

    fn pass_rows(&self) -> usize {
        self.draft.as_ref().map_or(1, ServedSpark::rows)
    }

    fn logits_into(&self, row: &mut [f32]) -> Result<(), GateError> {
        Ok(self.s.model().logits_into(row)?)
    }

    /// The model's reset: the RDA states and conv rings back to zero, the
    /// position to 0; the draft starts over with it.
    fn reset(&mut self) -> Result<(), GateError> {
        self.s.model_mut().reset()?;
        if let Some(d) = &mut self.draft {
            d.reset()?;
        }
        self.drafting = false;
        Ok(())
    }

    /// Never asked for a position other than the one the model stands at
    /// ([`Seat::keep`] grants no other); any other is refused by name.
    fn rollback(&mut self, pos: u32) -> Result<(), GateError> {
        Err(format!(
            "a rollback to position {pos} from {}: {NO_STEP_BACK}",
            self.s.model().pos()
        )
        .into())
    }

    /// `n` when it is the whole held sequence, else nothing: the caller then
    /// resets the slot and feeds the prompt again. Under `BLOOMERY_DRAFT=dspark`
    /// the draft must follow the kept prefix too ([`ServedSpark::keep`]), so
    /// every request drafts; the unset rule keeps the body's prefix and runs
    /// plain a request the draft does not follow.
    fn keep(&self, n: usize) -> (usize, Option<String>) {
        match (&self.draft, self.drafts) {
            (Some(d), Drafts::Every) => d.keep(n, self.s.model().pos(), |m| self.body_keep(m)),
            _ => self.body_keep(n),
        }
    }

    /// Nowhere: no prompt call is cut for a prefix a later request keeps.
    fn splits(&self, _first: usize, _end: usize, _marks: &[usize]) -> Vec<usize> {
        Vec::new()
    }

    /// Never asked: the seat's prompt cache budget is 0.
    fn snapshot(&mut self) -> Result<Arc<dyn Saved>, GateError> {
        Err(format!("a snapshot: {NO_CACHE}").into())
    }

    fn resume(&mut self, _state: &dyn Saved) -> Result<(), GateError> {
        Err(format!("a resume: {NO_CACHE}").into())
    }

    /// `engine.draft` and the draft's bytes, when one is served
    /// ([`ServedSpark::props`]).
    fn props(&self, p: EngineProps) -> EngineProps {
        match &self.draft {
            Some(d) => d.props(p),
            None => p,
        }
    }

    fn note(note: &CacheNote) {
        note_line(NAME, note);
    }
}

/// Loads the model and serves until the listener or the engine fails;
/// `Ok` carries why the server ended.
pub fn run(args: &[String]) -> Result<ServeError, GateError> {
    let levers = bloomery_levers::at_main(ACTS_ON)?;
    record::at_main(NAME, record::BLOOMERY_SERVE_RAON1);
    arm_screen(levers.screen());
    let a = SeatFlags::parse(args, USAGE, |_, _| Ok(false))?;
    let path = ref_model_path()?;
    let split = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let first = split
        .shard(0)
        .ok_or_else(|| format!("{} opened with no shard", path.display()))?;
    let arch = Arch::detect(first).map_err(|e| format!("{}: {e}", path.display()))?;
    if arch != Arch::Raon1 {
        return Err(format!(
            "{} is a {} file; the raon1 seat serves raon1 files",
            path.display(),
            arch.name()
        )
        .into());
    }
    if let Some(n) = a.parallel.filter(|&n| n > 1) {
        return Err(format!(
            "--parallel {n}: the raon1 body holds one sequence and no resident slots; the seat \
             serves --parallel 1"
        )
        .into());
    }
    let chat = ChatSurface::read(
        &path,
        split.value("tokenizer.chat_template"),
        split.value("general.name"),
        a.template_file.path(),
        "raon1",
    )?;
    // The coverage list has no raon1 program yet (`placement::accepts::program_of`), so the plan
    // reads the file as it is; the body's load refuses by name every part it does not run.
    // Every load makes the rows pass's buffers beside the plan's terms; the plan keeps them out of
    // the expert budget.
    let inputs = PlanInputs::describe(&split)?;
    let pass = Body::pass_reserve(&inputs.hp)?;
    let inputs = inputs.with_prompt_reserve(pass);
    let tok = Tokenizer::from_gguf(&path)?;
    let think = Think::of(&tok, inputs.spec.chat.reasoning);
    let (draft, why) = draft_of(&levers, (&path, &split), think)?;
    Record::new(&record::DRAFT_RULE_RAON1)
        .w(
            "drafts",
            draft.as_ref().map_or(Drafts::Never, |d| d.drafts).word(),
        )
        .w("why", &why)
        .eprint();
    let plan_levers = PlanLevers::from_levers(&levers)?;
    let (slots, from) = slots_of(a.parallel, a.ctx.is_some(), DEFAULT_SLOTS)?;
    let census = gpu_census::census()?;
    let chosen = Place::choose_untiered(a.place, &census, "raon1")?;
    chosen.record().eprint();
    let place = chosen.place;
    // A drafted load verifies a block's rows: each delta layer keeps a lane a row, and the card
    // holds the draft beside the target, both counted before the context is sized.
    let inputs = match &draft {
        Some(_) => inputs.with_lanes(VERIFY_LANES),
        None => inputs,
    };
    let mut machine = place.machine_untiered()?(inputs.model.layers);
    if let Some(d) = &draft {
        machine.cards[0]
            .reserves
            .push((DRAFT_RESERVE.to_owned(), d.reserve));
    }
    let asked = choose_ctx(&split, (a.ctx, slots), (None, None), |c| {
        probe_of(inputs.plan(&machine, c, &plan_levers))
            .map_err(|e| format!("--place {}: {e}", place.name()))
    })?;
    eprintln!("{}", asked.line());
    let slot_ctx = usize::try_from(asked.slot_ctx)?;
    let plan = asked.plan;
    record::plan(place.name(), &machine, &plan).eprint();
    need::print(&plan, 0, 0, true)?;
    parallel_line(slots, slot_ctx, from, Pass::One);
    if a.plan_only {
        std::process::exit(0);
    }
    let props = EngineProps {
        model: Some(model_props(&split, &inputs.model)),
        placement: placement_of(NAME, &machine, &plan),
        ..EngineProps::default()
    };
    let card = &machine.cards[0];
    let open = OpenSeat {
        card: card.name.clone(),
        device: card.device,
        path: path.clone(),
        ctx: slot_ctx,
        draft,
    };
    let device = card.name.clone();
    drop(plan);
    drop(split);
    let vocab = Arc::new(Vocab::new(tok)?);
    let engine = SeatEngine::spawn(move || Raon1::open(open), slot_ctx, vocab, device, props, 0)?;
    let server = a.bind(engine, chat, &path, slots, &levers, None)?;
    Record::new(&record::LISTENING_RAON1)
        .w("place", place.name())
        .u("ctx", slot_ctx)
        .u("slots", slots)
        .w("addr", server.local_addr()?)
        .eprint();
    Ok(server.run())
}

#[cfg(test)]
mod tests {
    use super::*;

    const THINK: Think = Think { open: 7, close: 8 };

    /// The unset rule drafts a request whose prompt call ends inside the think
    /// span, up to the close tag in its reply; `dspark` drafts through the tag.
    #[test]
    fn a_thinking_request_drafts_until_its_span_closes() {
        assert_eq!(THINK.inside(&[1, 7]), Some(true));
        assert_eq!(THINK.inside(&[1, 7, 8, 2]), Some(false));
        assert_eq!(THINK.inside(&[8, 7, 2]), Some(true));
        assert_eq!(THINK.inside(&[1, 2]), None);
        let rule = Drafts::Thinking(THINK);
        assert!(rule.still(&[3]));
        assert!(rule.still(&[3, 4, 5]));
        assert!(!rule.still(&[8]));
        assert!(!rule.still(&[3, 8, 4]));
        assert!(rule.still(&[8, 7]));
        assert!(Drafts::Every.still(&[3, 8, 4]));
        assert!(!Drafts::Never.still(&[3]));
    }
}
