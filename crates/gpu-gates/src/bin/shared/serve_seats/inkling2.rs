//! `bloomery-serve --model inkling2` — the llama-server-compatible HTTP API on the Inkling 2
//! engine with one sequence: an Inkling 2 Mini file (a looped trunk of stored layers walked as
//! more visits, 128 routed experts a layer), the whole model on one card, opened as
//! `gate_inkling2_e2e` opens it (`GpuModel::<inkling2::Body>::open`; the step captured in graph
//! mode), its prompt fed by the session's rule (`app::Prompt`).
//!
//!     bloomery-serve --model inkling2 [-m PATH | --hf <repo>[:<quant>]]
//!                    [--host 127.0.0.1] [--port 8080] [--ctx C] [--alias NAME]
//!                    [--chat-template-file PATH] [--place W] [--plan]
//!                    [--queue-depth Q] [--slot-save-path DIR]
//!                    [--api-key KEY] [--api-key-file FNAME]
//!
//! The flags are the nemotron seat's, read the same way: the model's first shard gives the
//! vocabulary, `tokenizer.chat_template` the chat template (`--chat-template-file` replaces it)
//! and `general.name` the alias (`--alias` replaces it; `inkling2` for a file that names none).
//! The template's block protocol is the server's own scan (`serve::inkling`): thinking blocks
//! are the reply's `reasoning_content`, text blocks its `content`, and a block headed by a
//! function's name a call, read only where the request asks for calls.
//!
//! Four facts of the architecture set this seat's shape:
//!
//! - The body holds one sequence: its visits' stores of one context and their convolution
//!   rings. It has no slot rows, so `--parallel` above 1 is refused by name before any plan.
//! - Every store of the body is positional, so it takes positions back as far as its conv ring
//!   holds them: a rollback is at most [`Body::rollback_reach`] positions below the end the
//!   body has written to, and `Rollback for Body` refuses a deeper one by name. A request keeps
//!   the prefix it shares with what the slot holds down to that floor
//!   ([`Body::rollback_floor`]); a shorter shared prefix is kept 0, the slot reset and the
//!   prompt fed again. The draft is the other path that asks a body for a rollback: a verify
//!   writes its rows' end, and the commit of its kept rows takes back at most a draft block's
//!   rows less one, inside the reach. The seat runs no prompt cache and takes no snapshot.
//! - The prompt runs by the body's prompt rule (`app::Prompt`): whatever the session does
//!   with the ids, the seat does the same, and the first id of the answer is a step of the
//!   last prompt id.
//! - The load is the whole model on one card (`GpuModel::open`) when the card leg runs every
//!   routed stack of the file and the card holds all of them at the floor context: no routed
//!   expert sits on the host. The plan is the body's ([`Inputs::plan`], every routed stack of a
//!   type the body's card leg runs on the card beside the dense weights), and a plan that would
//!   leave experts off the card at the asked context is refused by name, a short card at that
//!   context ([`card_experts_floor`]). A file the card leg runs but the card cannot hold whole
//!   even at the floor context (a 12 GB card and the Mini's IQ3_XXS file) takes the mixed load
//!   (`GpuModel::open_placed`): the plan's card share of each layer's experts (the id prefix
//!   `[0, n_l)`, the same count on every layer give or take one) beside the dense weights and
//!   the stores, the host tier serving the rest, a pick's slot on the card or the host as the
//!   handoff places it; the plan line is followed by a `mixed` line with every layer's `n_l`. A
//!   file whose stacks the card leg does not run (the Small's IQ2_XS, IQ3_XXS and IQ4_XS
//!   experts) takes the same load with every routed expert on the host, the prompt stepped one id
//!   at a time. The unset `--ctx` is the largest context the plan holds, up to the file's
//!   trained context: for the Small the card's stores (the 40 windowed layers' rings and the 8
//!   global layers' keys and values) are the bound.
//!
//! `--place W` is the shared placement word (`generate::Place`); unset, the common untiered
//! rule decides on one census reading. A placement the plan refuses (more than one card, a tier
//! card, a card the file does not fit) stops before any load by the plan's own refusal.
//! `--ctx` is the slot's context, chosen by `generate::choose_ctx` as every seat's is.
//!
//! `BLOOMERY_DRAFT` serves a draft behind the width chooser `BLOOMERY_MTP_WIDTH` names
//! ([`ServedDraft`], the serve seats' one owner of it), each verified by the body's pass
//! ([`Rows`]: 2 to 8 rows, a step's bits at every row) and kept by a rollback inside the conv
//! ring's reach: `lookup` is the common n-gram draft, one context for the one sequence, and
//! `dspark` the DSpark draft, in either form: the official checkpoint folder (`config.json`'s
//! `mtp_config` and `mtp.safetensors`) or llama.cpp's `dspark-*.gguf` file, found by
//! `refset::arch::draft`'s discovery (`$BLOOMERY_DSPARK_MODEL` naming either, else the target's
//! own folder when `-m` names one, else the one `dspark-*.gguf` beside a GGUF target; none, or two
//! files, refused by name before any plan), both opened by the one opener, its head reading the target's resident
//! output matrix in place on the target's card. The draft's taps (the rows entering the visits
//! its config names and the final normed row) are attached to the body before the first
//! capture, so the step, the verify and the prompt pass lay them; the plan sets the draft's
//! bytes aside on the card before the context is fitted. A draft asks for the whole-card load:
//! a file that takes the mixed or the hosted one is refused by name under a draft word.
//! Unset, the seat serves the DSpark draft under the width chooser where the draft is
//! found as `dspark` finds it and the whole-card plan holds it ([`UNSET_SERVES_DSPARK`]), and
//! the plain path otherwise; `off` serves the plain path. The `draft rule` record names which
//! and why ([`draft_of`]). A request that samples runs plain steps under a draft. Under a
//! DSpark draft a cut anywhere but where the draft stands keeps nothing and the prompt is fed
//! again through the draft ([`ServedDraft::keep`]).
//!
//! `--embd-model` is taken before this seat parses. The side model is seated before the plan;
//! the stage card holds its bytes. It loads after the chat model, and the server answers its
//! embeddings routes from it.
//!
//! The records on stderr, in order: `place unset`, the `embedding_seat` record when one is
//! named, the `draft rule` record (`record::DRAFT_RULE`), one `ctx` line (plus the stage
//! card's reserve note when it holds the embedding
//! model), the `plan` record, the `parallel` line, then the `load` record
//! (`record::LOAD_INKLING2`), the load's phases, the DSpark draft's `load draft=dspark` record
//! (`record::LOAD_DRAFT`) when one loads, the width chooser's `mtp width` record at each
//! drafted request's end (`record::MTP_WIDTH`) and, once the port is bound, the `listening`
//! record (`record::BLOOMERY_SERVE_INKLING2`). `--plan` ends the process after the `parallel`
//! line, nothing loaded.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use super::folder;
use super::served::ServedDraft;
use app::{Keep, Prompt, Session};
use bloomery_gpu::arch::inkling2::Body;
use bloomery_gpu::folder::Model;
use bloomery_gpu::model::{ChainBody, Rows, StepMode};
use bloomery_gpu::{Gpu, GpuModel};
use bloomery_gpu_gates::beside::{self, Beside};
use bloomery_gpu_gates::bind::{
    ChatSurface, Pass, Seat, SeatEngine, SeatFlags, Vocab, arm_screen, model_props, note_line,
    parallel_line, placement_of, prefill_closing, refuse_plain_width, slots_of,
};
use bloomery_gpu_gates::generate::{Place, SEAT_STEPS, choose_ctx};
use bloomery_gpu_gates::need;
use bloomery_gpu_gates::record::{self, Record};
use bloomery_gpu_gates::side_seat;
use bloomery_gpu_gates::{GateError, gpu_census, ref_model_path};
use bloomery_levers::{DraftWord, HostCfg, Levers};
use gguf::Split;
use model::arch::Arch;
use model::arch::inkling2::hparams::{Hparams, PASSES_OVERRIDE};
use model::arch::inkling2::place::{self, Inputs};
use model::arch::place_hybrid::PlaceError;
use model::placement::accepts::fused_w13_runs;
use model::placement::ctx_ask::{Chosen, DEFAULT_SLOTS, Probe, card_experts_floor, probe_of};
use model::placement::workstation::{DeviceId, draft_reserve};
use model::placement::{Device, Machine, PlacementError, Plan, PlanLevers, Role};
use runtime::Target as _;
use runtime::width::Mode as WidthMode;
use serve::{CacheNote, Drafted, EngineProps, Saved, ServeError};
use tokenizer::Tokenizer;

// The open's and the verify capture's log ([`Quiet`]), by path as the seats of binaries whose
// root holds no `quiet` module read it; a binary whose root holds it too reaches the file twice.
#[path = "../quiet.rs"]
#[allow(
    clippy::duplicate_mod,
    reason = "each binary includes the shared log by path; a seat binary may reach it twice"
)]
mod quiet;

// The DSpark draft's one opener, shared with `gate_inkling2_draft`: its statement, its reserve
// and its load over the target's resident output matrix.
#[cfg(feature = "dspark-hf")]
#[path = "../inkling2_draft.rs"]
#[allow(
    dead_code,
    reason = "the gate and this seat share the opener; each takes the parts it reads"
)]
mod draft_open;

use quiet::Quiet;
use refset::arch::draft::DsparkForm;

const NAME: &str = "bloomery-serve-inkling2";

const USAGE: &str = "usage: bloomery-serve --model inkling2 [-m PATH | --hf <repo>[:<quant>]] \
                     [--host H] [--port P] [--ctx C] [--alias NAME] [--chat-template-file PATH] \
                     [--place W] [--plan] [--queue-depth Q] [--slot-save-path DIR] \
                     [--api-key KEY] [--api-key-file FNAME]";

/// The levers this seat acts on: the plan's card budget, the start screen, how a
/// structured-output request's fixed runs are fed, the draft word ([`draft_asked`]) and the
/// width chooser's mode, which a drafted window reads.
pub const ACTS_ON: &[&str] = &[
    bloomery_levers::CARD_BUDGET,
    bloomery_levers::SCREEN,
    bloomery_levers::SCHEMAJUMP,
    bloomery_levers::DRAFT,
    bloomery_levers::MTP_WIDTH,
    bloomery_levers::INK2_DENSE,
    bloomery_levers::INK2_PASSES,
    bloomery_levers::INK2_LOOPCD,
    bloomery_levers::INK2_LOOPCD_REF,
    bloomery_levers::HOST_POPULATE,
    bloomery_levers::HOST_LOCK,
    bloomery_levers::CARD_DONTNEED,
    bloomery_levers::R8,
];

/// Why the seat takes no sequence state: it runs no prompt cache.
const NO_CACHE: &str = "the inkling2 seat runs no prompt cache";

/// Why a prefix further below the written end than the conv ring holds is not kept.
const PAST_REACH: &str = "the conv ring holds a pass of positions: a prefix is kept only down to the written end \
     less that";

/// A verify of a proposal: the token at the target's position, then the draft's 7 ids.
const VERIFY_ROWS: usize = 8;
const _: () = assert!(VERIFY_ROWS == <Body as Rows>::MAX_ROWS);

/// The draft words this seat serves. `dspark` needs the DSpark draft's reader (`dspark-hf`).
#[cfg(feature = "dspark-hf")]
const DRAFT_WORDS: [DraftWord; 3] = [DraftWord::Lookup, DraftWord::Dspark, DraftWord::Off];
#[cfg(not(feature = "dspark-hf"))]
const DRAFT_WORDS: [DraftWord; 2] = [DraftWord::Lookup, DraftWord::Off];

/// The draft words this seat serves, as a refusal names them.
#[cfg(feature = "dspark-hf")]
const DRAFT_WORDS_DOC: &str = "the lookup draft (lookup) and the DSpark draft of the official checkpoint folder (dspark), off or unset";
#[cfg(not(feature = "dspark-hf"))]
const DRAFT_WORDS_DOC: &str = "the lookup draft (lookup), off or unset";

/// `BLOOMERY_DRAFT` with `BLOOMERY_MTP_WIDTH`'s mode: unset and `off` serve the plain path,
/// `lookup` and `dspark` a draft behind the width chooser; any other word is refused by name.
fn draft_asked(levers: &Levers) -> Result<(Option<DraftWord>, WidthMode), GateError> {
    let width = WidthMode::of(levers.mtp_width())?;
    let word = levers.draft_word(&DRAFT_WORDS).map_err(|e| {
        format!(
            "BLOOMERY_DRAFT={}: the inkling2 seat serves {DRAFT_WORDS_DOC}",
            e.word
        )
    })?;
    Ok((word, width))
}

/// The plan of `inputs` on `machine` at `ctx` positions that holds the whole model on the card:
/// the body's plan ([`Inputs::plan`]) refused where its terms break the card's bound, and where
/// any layer leaves a routed expert off the card (the body's load is the whole model's, so the
/// experts the plan would host are a short card by their bytes).
fn whole_card_plan<'a>(
    inputs: &'a Inputs,
    machine: &'a Machine,
    ctx: u64,
    levers: &PlanLevers,
) -> Result<Plan<'a>, PlaceError> {
    let plan = inputs.plan(machine, ctx, levers)?;
    let broken = plan.violations();
    if !broken.is_empty() {
        return Err(PlaceError::Broken(broken));
    }
    card_experts_floor(&plan, 0, inputs.model.experts).map_err(PlaceError::Placement)?;
    Ok(plan)
}

/// Whether the body's card leg runs every routed stack of `inputs`' file: the whole-card load
/// serves it. A file with a stack of another type (the Small's IQ quants) takes the placed load.
fn card_runs_the_experts(inputs: &Inputs) -> bool {
    inputs
        .model
        .tensors
        .iter()
        .filter(|t| t.role == Role::RoutedExperts)
        .all(|t| fused_w13_runs(&t.name, t.ty))
}

/// The plan of `inputs` on `machine` at `ctx` positions that leaves routed experts on the host:
/// the body's plan ([`Inputs::plan`], the card share of the stacks its leg runs, the rest on the
/// host, the dense weights and the stores on the card) refused where its terms break a bound.
fn hosted_plan<'a>(
    inputs: &'a Inputs,
    machine: &'a Machine,
    ctx: u64,
    levers: &PlanLevers,
) -> Result<Plan<'a>, PlaceError> {
    let plan = inputs.plan(machine, ctx, levers)?;
    let broken = plan.violations();
    if !broken.is_empty() {
        return Err(PlaceError::Broken(broken));
    }
    Ok(plan)
}

/// Which plan shape the seat loads by.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Shape {
    /// The whole model on the card: a file the card leg runs, held whole at the floor context.
    Whole,
    /// The plan's card share of each layer's experts beside the host's: a file the card leg runs
    /// and the card cannot hold whole at the floor context.
    Mixed,
    /// Every routed expert on the host: a file whose stacks the card leg does not run.
    Hosted,
}

impl Shape {
    /// The shape of `inputs`' file on `machine`: [`Shape::Hosted`] for a file the card leg does
    /// not run; for one it runs, [`Shape::Whole`] when the whole-card plan holds at the seat's
    /// floor context ([`SEAT_STEPS`]), else [`Shape::Mixed`] when `mixed` allows it (a GGUF file:
    /// a checkpoint folder's stacks have no host kernel, so its short card stays a refusal).
    fn of(inputs: &Inputs, machine: &Machine, levers: &PlanLevers, mixed: bool) -> Shape {
        if !card_runs_the_experts(inputs) {
            Shape::Hosted
        } else if !mixed || whole_card_plan(inputs, machine, SEAT_STEPS.floor, levers).is_ok() {
            Shape::Whole
        } else {
            Shape::Mixed
        }
    }

    /// Whether the load is the placed one (experts on the host).
    fn placed(self) -> bool {
        self != Shape::Whole
    }
}

/// The plan the seat loads by in `shape`: the whole-card plan, or the plan as it falls (the card
/// share and the host's) refused where its terms break a bound.
fn seat_plan<'a>(
    inputs: &'a Inputs,
    machine: &'a Machine,
    ctx: u64,
    levers: &PlanLevers,
    shape: Shape,
) -> Result<Plan<'a>, PlaceError> {
    match shape {
        Shape::Whole => whole_card_plan(inputs, machine, ctx, levers),
        Shape::Mixed | Shape::Hosted => hosted_plan(inputs, machine, ctx, levers),
    }
}

/// What a placed load is made from beside the plan's card: everything the engine thread needs to
/// plan again (the plan borrows the machine and the inputs) and the host tier's settings.
struct PlacedOpen {
    inputs: Inputs,
    machine: Machine,
    levers: PlanLevers,
    host: HostCfg,
}

/// The weights a seat serves: the official checkpoint folder a directory is, or a GGUF file.
enum Weights {
    /// The folder's located header, tokenizer and part source (boxed: the tokenizer outweighs a
    /// header).
    Folder(Box<folder::Opened>),
    /// A GGUF file's header; the engine thread opens the file again.
    File(Split),
}

/// The model's open on the engine thread ([`Weights::hand_over`]).
type OpenModel = Box<dyn FnOnce() -> Result<Model, GateError> + Send>;

/// The checkpoint folder `path` held to this architecture, its dense matrices written as Q8_0 at
/// load where `q8` asks and its loop run at `passes` passes where one is asked
/// (`model_hf::arch::inkling2::open_passes`, which sets the count in the folder's config before the
/// map reads it); a count other than the folder's own is noted on stderr.
#[cfg(feature = "checkpoint")]
fn open_folder(path: &Path, q8: bool, passes: Option<usize>) -> Result<folder::Opened, GateError> {
    let arch = refset::arch::inkling2::ARCH;
    match passes {
        None => folder::open_with(path, arch, q8),
        Some(p) => {
            if let Ok(trained) = model_hf::arch::inkling2::trained_passes(path) {
                note_passes(p, trained);
            }
            folder::adopt(
                path,
                arch,
                model_hf::arch::inkling2::open_passes(path, q8, passes),
            )
        }
    }
}

/// A checkpoint folder in a build without the `checkpoint` feature, refused by name.
#[cfg(not(feature = "checkpoint"))]
fn open_folder(
    path: &Path,
    _q8: bool,
    _passes: Option<usize>,
) -> Result<folder::Opened, GateError> {
    Err(folder::without_checkpoint(path))
}

/// The stderr line for a pass count that differs from the checkpoint's own: the lever is an
/// experiment, so the change is never silent.
fn note_passes(asked: usize, trained: usize) {
    if asked != trained {
        eprintln!(
            "passes override={asked} trained={trained} lever={}: the checkpoint was trained at \
             {trained} passes; the output at another count is not guaranteed",
            bloomery_levers::INK2_PASSES
        );
    }
}

impl Weights {
    /// The weights at `path`: a directory is the checkpoint folder, its dense matrices written as
    /// Q8_0 at load where `q8` asks (`BLOOMERY_INK2_DENSE`); anything else is the GGUF file. Both
    /// run their loop at `passes` passes where one is asked (`BLOOMERY_INK2_PASSES`).
    fn open(path: &Path, q8: bool, passes: Option<usize>) -> Result<Weights, GateError> {
        if path.is_dir() {
            let f = open_folder(path, q8, passes).map_err(|e| match passes {
                Some(p) => format!("{}={p}: {e}", bloomery_levers::INK2_PASSES).into(),
                None => e,
            })?;
            return Ok(Weights::Folder(Box::new(f)));
        }
        let header = Split::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let header = match passes {
            Some(p) => {
                let trained = header.arch_get_u64("num_loops").ok_or_else(|| {
                    format!(
                        "{}={p}: {} carries no num_loops: it holds no loop to run at another count",
                        bloomery_levers::INK2_PASSES,
                        path.display()
                    )
                })?;
                note_passes(p, usize::try_from(trained)?);
                header.with_overlay(PASSES_OVERRIDE, gguf::Value::U32(u32::try_from(p)?))
            }
            None => header,
        };
        Ok(Weights::File(header))
    }

    /// The header the plan is made from: a split's, a folder's located header.
    fn header(&self) -> &Split {
        match self {
            Weights::Folder(f) => &f.header,
            Weights::File(header) => header,
        }
    }

    /// The tokenizer the vocabulary is built from, and the model's open on the engine thread: a
    /// folder's header and part source go with it, a file is opened again there.
    fn hand_over(self, path: &Path) -> Result<(Tokenizer, OpenModel), GateError> {
        match self {
            Weights::Folder(f) => {
                let folder::Opened {
                    header,
                    tokenizer,
                    source,
                } = *f;
                let open: OpenModel = Box::new(move || folder::model(header, source));
                Ok((tokenizer, open))
            }
            Weights::File(header) => {
                let passes = header.value(PASSES_OVERRIDE).cloned();
                drop(header);
                let tokenizer = Tokenizer::from_gguf(path)?;
                let file = path.to_path_buf();
                let open: OpenModel = Box::new(move || -> Result<Model, GateError> {
                    let split =
                        Split::open(&file).map_err(|e| format!("open {}: {e}", file.display()))?;
                    Ok(match &passes {
                        Some(p) => split.with_overlay(PASSES_OVERRIDE, p.clone()),
                        None => split,
                    }
                    .into())
                });
                Ok((tokenizer, open))
            }
        }
    }
}

/// The largest slot context, a multiple of [`FOLDER_STEP`] up to [`FOLDER_LIMIT`], at which
/// `holds` says the whole-card plan holds, its line on stderr; a context no step holds is refused
/// by name.
fn folder_fit<E>(mut holds: impl FnMut(u64) -> Result<bool, E>) -> Result<usize, GateError>
where
    E: std::fmt::Display,
{
    let mut probe = |c: u64| holds(c).map_err(|e| GateError::from(e.to_string()));
    let (mut lo, mut hi) = (0u64, FOLDER_LIMIT / FOLDER_STEP + 1);
    if probe(FOLDER_LIMIT)? {
        lo = FOLDER_LIMIT / FOLDER_STEP;
    }
    while lo + 1 < hi {
        let mid = (lo + hi) / 2;
        if probe(mid * FOLDER_STEP)? {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    if lo == 0 {
        return Err(format!(
            "no context of {FOLDER_STEP} positions or more holds the folder's whole model on the card"
        )
        .into());
    }
    let ctx = lo * FOLDER_STEP;
    eprintln!("ctx rule=folder-fit ctx={ctx} step={FOLDER_STEP} limit={FOLDER_LIMIT}");
    Ok(usize::try_from(ctx)?)
}

/// The folder fit's context unit and its upper bound (the Mini's trained context).
const FOLDER_STEP: u64 = 1024;
const FOLDER_LIMIT: u64 = 1 << 20;

/// The draft the engine thread opens beside the model, as `BLOOMERY_DRAFT` and the files
/// decided before the plan.
#[derive(Clone)]
struct OpenDraft {
    /// The draft word: `lookup`, or `dspark` with its folder.
    word: DraftWord,
    /// The width chooser's mode.
    width: WidthMode,
    /// The form the DSpark draft was found in, a checkpoint folder or a GGUF file.
    source: Option<DsparkForm>,
    /// The target's visits the draft taps, read from its statement before the plan.
    taps: Vec<usize>,
    /// What the plan sets aside for a DSpark draft beside its context planes, and the bytes a
    /// position of context adds to it, computed from the opened draft's own file and shape in
    /// either form ([`draft_open::Draft::reserve_fixed`], [`draft_open::Draft::reserve_per_pos`]);
    /// 0 for the lookup.
    reserve_fixed: u64,
    reserve_per_pos: u64,
    /// The folder target's dense setting and pass count, read again for the draft's anchor.
    dense: (bool, Option<usize>),
}

/// What the engine thread opens the model with.
struct OpenSeat {
    card: String,
    device: Option<DeviceId>,
    /// The target's path, opened again for the draft's anchor rows.
    path: PathBuf,
    /// The model's open ([`Weights::hand_over`]).
    model: OpenModel,
    /// The slot's positions the visits' stores are sized for.
    ctx: usize,
    /// The draft the load serves, `None` for the plain path.
    draft: Option<OpenDraft>,
    /// LoopCD-Hidden's strength and reference pass, where the strength is set and not 0
    /// (`GpuModel::set_loopcd`, applied before the step is captured).
    loopcd: Option<(f32, u32)>,
    /// The placed load's inputs, for a file whose experts the card leg does not run.
    placed: Option<PlacedOpen>,
}

/// The model's session, in graph mode with its step captured, and the draft beside it.
struct Inkling2 {
    /// Declared first: the DSpark draft reads the target's output matrix in place, so it is
    /// dropped before the session that owns the target.
    draft: ServedDraft<Body>,
    s: Session<Body>,
    ctx: usize,
    /// The buffer the head's padded logits row is read into.
    row: std::cell::RefCell<Vec<f32>>,
}

impl Inkling2 {
    /// The model `a.model` opens on the planned card, the step graph captured, the `load`
    /// record on stderr, then the draft beside it: the lookup's verify captured, the DSpark
    /// draft loaded on the target's card with its `load draft=dspark` record and its verifies
    /// captured.
    fn open(a: OpenSeat) -> Result<Inkling2, GateError> {
        let t = Instant::now();
        let file = (a.model)()?;
        let opened = t.elapsed();
        let mut m = match a.placed {
            None => {
                let gpu = Gpu::open_card_after_hook(&a.card, a.device)?;
                GpuModel::<Body>::open(gpu, file, a.ctx)?
            }
            Some(p) => {
                let (split, _) = file.into_whole();
                let plan = p.inputs.plan(&p.machine, a.ctx as u64, &p.levers)?;
                GpuModel::<Body>::open_placed(split, &plan, &p.inputs, 0, p.host)?
            }
        };
        if let Some((omega, reference)) = a.loopcd {
            m.set_loopcd(omega, reference)?;
        }
        // The draft's taps before the first capture: a capture made before them would not write
        // them.
        let taps = match &a.draft {
            Some(d) if d.word == DraftWord::Dspark => Some(dspark_taps(d)?),
            _ => None,
        };
        if let Some(taps) = taps {
            m.set_draft_taps(&taps)?;
        }
        m.set_mode(StepMode::Graph);
        let nodes = m.capture_step()?;
        let total = t.elapsed();
        Record::new(&record::LOAD_INKLING2)
            .w("arch", Body::arch().name())
            .u("resident_bytes", m.resident_bytes())
            .u("ctx", m.ctx_max())
            .u("slots", 1)
            .u("layers", m.layers().len())
            .u("passes", m.body(NAME)?.passes())
            .u("visits", m.body(NAME)?.visit_table().len())
            .u("graph_nodes", nodes)
            .f("load_s", total.as_secs_f64())
            .eprint();
        record::eprint_load_phases(&m, total, opened);
        let padded = m.n_vocab()?;
        let mut s = Session::from_model(m, u32::try_from(a.ctx)?);
        let draft = match a.draft.as_ref() {
            None => ServedDraft::off(),
            Some(d) if d.word == DraftWord::Lookup => {
                ServedDraft::lookup(&mut s, 1, d.width, &mut Quiet, None)?
            }
            Some(d) => open_dspark(&mut s, (&a.path, &a.card, a.device, a.ctx), d)?,
        };
        Ok(Inkling2 {
            draft,
            s,
            ctx: a.ctx,
            row: std::cell::RefCell::new(vec![0.0; padded]),
        })
    }

    /// The body's keep rule ([`Keep::keepable`]): `n` from the body's floor up to the held
    /// sequence, else nothing.
    fn body_keep(&self, n: usize) -> (usize, Option<String>) {
        let at = u32::try_from(n).unwrap_or(u32::MAX);
        let kept = <Body as Keep>::keepable(self.s.model(), at) as usize;
        (kept, (kept < n).then(|| PAST_REACH.to_owned()))
    }
}

/// The DSpark draft's taps, read from its statement before the plan and handed to the target's
/// first capture.
#[cfg(feature = "dspark-hf")]
fn dspark_taps(d: &OpenDraft) -> Result<Vec<usize>, GateError> {
    dspark_form(d)?;
    Ok(d.taps.clone())
}

/// The DSpark draft beside the session `s`'s target, on the target's card: its anchor's rows
/// from the target at `a.path` (a GGUF file's header, or a checkpoint folder's part source),
/// the common owner loaded ([`draft_open::open`]) and set behind the width chooser
/// ([`ServedDraft::dspark`]), its verifies captured. The `load draft=dspark` record is printed
/// between the two.
#[cfg(feature = "dspark-hf")]
fn open_dspark(
    s: &mut Session<Body>,
    (path, card_name, device, ctx): (&Path, &str, Option<DeviceId>, usize),
    d: &OpenDraft,
) -> Result<ServedDraft<Body>, GateError> {
    let t = Instant::now();
    let form = dspark_form(d)?;
    let (anchor, hp) = anchor_of(path, d.dense)?;
    let draft = draft_open::Draft::open(form, &hp)?;
    // The common loop names its card for the life of the process.
    let card: &'static str = Box::leak(card_name.to_owned().into_boxed_str());
    let gpu = Gpu::open_card(card, device)?;
    let mut spark = draft_open::open(s.model(), gpu, card, anchor, &draft, ctx)?;
    let (free, total) = spark.mem_info()?;
    Record::new(&record::LOAD_DRAFT)
        .w("draft", "dspark")
        .w("card", spark.card())
        .u("width", VERIFY_ROWS - 1)
        .list("target_layers", draft.taps())
        .u("feature_width", (draft.taps().len() * hp.n_embd) as u64)
        .u("resident", spark.resident_bytes() as u64)
        .u("working", spark.held().working)
        .u("reserve", d.reserve_at(ctx as u64))
        .u("taken", spark.taken_bytes() as u64)
        .u("draft_card_free", free as u64)
        .u("draft_card_total", total as u64)
        .f("load_s", t.elapsed().as_secs_f64())
        .eprint();
    // The draft's load left its own context current; the verify captures read the stream's.
    s.model().gpu().context().bind_to_thread()?;
    ServedDraft::dspark::<draft_open::Ink2Dspark, VERIFY_ROWS>(
        s,
        spark,
        d.width,
        &mut Quiet,
        Some(draft.file),
    )
    .map_err(Into::into)
}

/// `BLOOMERY_DRAFT=dspark` in a build without the draft's reader: the word is refused at
/// `check` ([`draft_asked`]); a start that reached the load with it is refused again here.
#[cfg(not(feature = "dspark-hf"))]
fn dspark_taps(_d: &OpenDraft) -> Result<Vec<usize>, GateError> {
    Err(no_dspark_reader())
}

/// As [`dspark_taps`].
#[cfg(not(feature = "dspark-hf"))]
fn open_dspark(
    _s: &mut Session<Body>,
    _open: (&Path, &str, Option<DeviceId>, usize),
    _d: &OpenDraft,
) -> Result<ServedDraft<Body>, GateError> {
    Err(no_dspark_reader())
}

/// The refusal of `dspark` in a build without the DSpark draft's reader.
#[cfg(not(feature = "dspark-hf"))]
fn no_dspark_reader() -> GateError {
    "BLOOMERY_DRAFT=dspark: this build reads no DSpark draft (the dspark-hf feature)".into()
}

/// The form the DSpark draft was found in, as the plan was told it.
#[cfg(feature = "dspark-hf")]
fn dspark_form(d: &OpenDraft) -> Result<&DsparkForm, GateError> {
    Ok(d.source
        .as_ref()
        .ok_or_else(|| "BLOOMERY_DRAFT=dspark: nothing states the draft".to_owned())?)
}

/// The block's anchor rows from the target at `path`: a GGUF file's header, or a checkpoint
/// folder's part source read as the target's load reads it (`dense`, `passes`).
#[cfg(feature = "dspark-hf")]
fn anchor_of(
    path: &Path,
    (q8, passes): (bool, Option<usize>),
) -> Result<(bloomery_gpu::dspark::ctx_dense::AnchorEmbed, Hparams), GateError> {
    if path.is_dir() {
        #[cfg(feature = "checkpoint")]
        {
            let f = open_folder(path, q8, passes)?;
            let hp = Hparams::read(&f.header).map_err(|e| format!("{}: {e}", path.display()))?;
            let anchor = draft_open::anchor_of_folder(&f.source, &hp)?;
            return Ok((anchor, hp));
        }
        #[cfg(not(feature = "checkpoint"))]
        {
            let _ = (q8, passes);
            return Err(folder::without_checkpoint(path));
        }
    }
    let target = Arc::new(Split::open(path).map_err(|e| format!("open {}: {e}", path.display()))?);
    let hp = Hparams::read(&target).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok((draft_open::anchor_of_file(&target, &hp)?, hp))
}

impl Seat for Inkling2 {
    fn pos(&self) -> usize {
        self.s.model().pos() as usize
    }

    fn ctx_max(&self) -> usize {
        self.ctx
    }

    /// The body's prompt rule ([`Prompt::prompt`]) over `ids`, through the draft
    /// ([`ServedDraft::prefill`]: the DSpark draft's own call takes the call's tapped rows, the
    /// lookup the call's ids).
    fn prefill(&mut self, ids: &[u32]) -> Result<u32, GateError> {
        self.draft.prefill(&mut self.s, ids, |s| {
            Ok(<Body as Prompt>::prompt(s.model_mut(), ids)?)
        })
    }

    /// A prompt's last call with its last id the pass's last row ([`prefill_closing`]): the pass
    /// leaves the stores and the head's row where that id's step would, so the call closes and the
    /// row is read after it. The body's prompt rule is this one call, the gate's seat answer too.
    fn prefill_next(&mut self, ids: &[u32], row: Option<&mut [f32]>) -> Result<u32, GateError> {
        let closes = self.s.model().body(NAME)?.pass_rows() > 0;
        prefill_closing(self, ids, row, closes)
    }

    /// One plain step, which the draft takes in ([`ServedDraft::stepped`]: the lookup's context
    /// gains the position, the DSpark draft the step's tapped row).
    fn step(&mut self, last: u32) -> Result<u32, GateError> {
        let next = self.s.model_mut().step(&[last])?;
        self.draft.stepped(&mut self.s, last, next)?;
        Ok(next)
    }

    /// One pass from `last` through the draft ([`ServedDraft::pass`]): the proposal verified by
    /// the body's pass, the rows the target agrees with kept and the rest taken back; the plain
    /// step without one.
    fn pass(&mut self, last: u32, out: &mut Vec<u32>) -> Result<Drafted, GateError> {
        match self.draft.pass(&mut self.s, last, out)? {
            Some(counts) => Ok(counts),
            None => {
                out.push(self.step(last)?);
                Ok(Drafted::default())
            }
        }
    }

    /// The most rows a pass verifies ([`ServedDraft::rows`]): the draft's window.
    fn pass_rows(&self) -> usize {
        self.draft.rows()
    }

    /// The lookup's state is a context a slot ([`ServedDraft::slot_drafts`]); the DSpark draft
    /// serves the one sequence.
    fn slot_drafts(&self) -> bool {
        self.draft.slot_drafts()
    }

    /// The leading columns of the head's row: the head is wider than the vocabulary by its
    /// padded columns, which the server never reads and are cut here. A row wider than the head
    /// is refused by name.
    fn logits_into(&self, row: &mut [f32]) -> Result<(), GateError> {
        let mut padded = self.row.borrow_mut();
        if row.len() > padded.len() {
            return Err(format!(
                "a logits row of {} columns for a head of {}",
                row.len(),
                padded.len()
            )
            .into());
        }
        self.s.model().logits_into(&mut padded)?;
        row.copy_from_slice(&padded[..row.len()]);
        Ok(())
    }

    /// The model's reset: every ring and store back to empty, the position to 0; the draft
    /// starts over with it ([`ServedDraft::reset`]).
    fn reset(&mut self) -> Result<(), GateError> {
        self.s.reset()?;
        self.draft.reset(self.s.selected())
    }

    /// Asked only for a position [`Seat::keep`] granted: the draft no longer follows the
    /// sequence ([`ServedDraft::moved`]), and the session's cut is the body's rollback, refused
    /// by name below its floor (`inkling2::Body::rollback`).
    fn rollback(&mut self, pos: u32) -> Result<(), GateError> {
        self.draft.moved();
        Ok(self.s.cut(pos)?)
    }

    /// The body's rule ([`Inkling2::body_keep`]) and the sentence that kept less; under the
    /// DSpark draft a cut anywhere but where the draft stands keeps nothing, since its prompt
    /// call starts the draft over ([`ServedDraft::keep`]).
    fn keep(&self, n: usize) -> (usize, Option<String>) {
        self.draft
            .keep(n, self.s.model().pos(), |m| self.body_keep(m))
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

    /// `engine.draft` and the draft's resident bytes, when one is served
    /// ([`ServedDraft::props`]).
    fn props(&self, p: EngineProps) -> EngineProps {
        self.draft.props(p)
    }

    /// The draft's width record of the last request printed ([`ServedDraft::close`]).
    fn close(&mut self) -> Result<(), GateError> {
        self.draft.close();
        Ok(())
    }

    fn note(note: &CacheNote) {
        note_line(NAME, note);
    }
}

/// What the unset `BLOOMERY_DRAFT` serves: the DSpark draft under the width chooser where its
/// folder is found and the whole-card plan holds it beside the context, else the plain path
/// ([`draft_of`]). The serving default; false serves the plain path unset, as `off` does.
const UNSET_SERVES_DSPARK: bool = true;

/// What `BLOOMERY_DRAFT` resolved to before the plan ([`draft_of`]): the draft the load opens,
/// `None` for the plain path, and the `draft rule` record's why.
struct DraftRule {
    draft: Option<OpenDraft>,
    why: String,
    /// The draft is the unset word's: a plan the draft's reserve leaves short serves the plain
    /// path instead ([`DraftRule::plain`]), where a set word's is the plan's refusal.
    unset: bool,
}

impl DraftRule {
    /// The plain path for `why`; a set width lever is refused by name, since nothing drafts.
    fn plain(levers: &Levers, why: String) -> Result<DraftRule, GateError> {
        refuse_plain_width(levers, Some(&why))?;
        Ok(DraftRule {
            draft: None,
            why,
            unset: false,
        })
    }

    /// The `draft rule` record on stderr.
    fn eprint(&self) {
        let drafts = self.draft.as_ref().map_or("off", |d| match d.word {
            DraftWord::Lookup => "lookup",
            _ => "dspark",
        });
        Record::new(&record::DRAFT_RULE)
            .w("drafts", drafts)
            .w("why", &self.why)
            .eprint();
    }
}

/// The draft the seat serves, as `BLOOMERY_DRAFT`, the width chooser's mode and the files
/// decide before any plan. `lookup` needs no file; `dspark` reads its folder or GGUF file
/// ([`dspark_source_of`]) and holds the draft's taps against the target's visits
/// ([`visits_of`]); a draft word asks for a whole-card load, each miss refused by name. Unset
/// takes the DSpark draft as `dspark` would ([`UNSET_SERVES_DSPARK`]), and each miss — no
/// folder or file, two candidates, a GGUF draft of another model, a build without the draft's reader, a load the verify does not run on, taps past the
/// visits — is the plain path with its why instead of a refusal.
fn draft_of(
    levers: &Levers,
    (path, hp): (&Path, &Hparams),
    (shape, dense): (Shape, (bool, Option<usize>)),
) -> Result<DraftRule, GateError> {
    let (word, width) = draft_asked(levers)?;
    let word = match word {
        Some(DraftWord::Off) => return DraftRule::plain(levers, "BLOOMERY_DRAFT=off".to_owned()),
        None if !UNSET_SERVES_DSPARK => {
            return DraftRule::plain(levers, "unset: the plain path".to_owned());
        }
        Some(DraftWord::Mtp) => unreachable!("draft_asked refused the word"),
        w => w,
    };
    let unset = word.is_none();
    let named = word.map_or_else(|| "unset".to_owned(), |w| format!("BLOOMERY_DRAFT={w}"));
    if let Some(passes) = levers.ink2_passes() {
        let why = format!(
            "a draft's taps name the visits of the checkpoint's own pass count; \
             BLOOMERY_INK2_PASSES={passes} walks another count"
        );
        if unset {
            return DraftRule::plain(levers, format!("unset: {why}"));
        }
        return Err(format!("{named}: {why}. Unset one of the two").into());
    }
    let load = match shape {
        Shape::Whole => None,
        Shape::Mixed => Some("mixed"),
        Shape::Hosted => Some("hosted"),
    };
    if let Some(load) = load {
        let why = format!(
            "a draft verifies its rows by the body's whole-card pass and reads its taps from the \
             whole-card walk; {} takes the {load} load",
            path.display()
        );
        if unset {
            return DraftRule::plain(levers, format!("unset: {why}"));
        }
        return Err(format!(
            "{named}: {why}. Serve the whole model on one card (a larger --place card, or the \
             Mini's file), or unset the draft"
        )
        .into());
    }
    if word == Some(DraftWord::Lookup) {
        return Ok(DraftRule {
            draft: Some(OpenDraft {
                word: DraftWord::Lookup,
                width,
                source: None,
                taps: Vec::new(),
                reserve_fixed: 0,
                reserve_per_pos: 0,
                dense,
            }),
            why: named,
            unset: false,
        });
    }
    #[cfg(feature = "dspark-hf")]
    {
        // A miss the unset word serves plain with its why; a set word's is the refusal.
        let missed = |why: String, hint: &str| -> Result<DraftRule, GateError> {
            if unset {
                return DraftRule::plain(levers, format!("unset: {why}"));
            }
            Err(format!("{named}: {why}{hint}").into())
        };
        let found = match dspark_source_of(path) {
            Ok(found) => found,
            Err(why) => return missed(why, ""),
        };
        let Some((form, from)) = found else {
            let why = if path.is_dir() {
                format!("{} holds no mtp.safetensors", path.display())
            } else {
                format!(
                    "no dspark-*.gguf beside {} and BLOOMERY_DSPARK_MODEL is unset",
                    path.display()
                )
            };
            return missed(
                format!("no DSpark draft folder or file: {why}"),
                "; set BLOOMERY_DSPARK_MODEL to the official checkpoint folder (config.json and \
                 mtp.safetensors) or to the draft's GGUF file, or set BLOOMERY_DRAFT=off",
            );
        };
        let at = match &form {
            DsparkForm::Folder(p) | DsparkForm::Gguf(p) => p.clone(),
        };
        let draft = match draft_open::Draft::open(&form, hp) {
            Ok(draft) => draft,
            // A folder's refusal stays one for the unset word too; a GGUF's is a plain path with
            // its why (a draft of another model, a type the load does not read, a missing tensor).
            Err(e) if unset && matches!(form, DsparkForm::Gguf(_)) => {
                return DraftRule::plain(levers, format!("unset: {e}"));
            }
            Err(e) => return Err(e),
        };
        let visits = visits_of(hp);
        if draft.taps().iter().any(|&t| t > visits) {
            let why = format!(
                "the draft at {} taps the visits {:?}, and {} walks {visits} visits: the draft is \
                 the Mini's (48 visits)",
                at.display(),
                draft.taps(),
                path.display()
            );
            return missed(why, "");
        }
        Ok(DraftRule {
            why: format!(
                "{named}, {} ({from}, {})",
                draft.file.display(),
                form_name(&form)
            ),
            draft: Some(OpenDraft {
                word: DraftWord::Dspark,
                width,
                taps: draft.taps().to_vec(),
                source: Some(form),
                reserve_fixed: draft.reserve_fixed((place::PASS_ROWS, hp.n_embd)),
                reserve_per_pos: draft.reserve_per_pos(),
                dense,
            }),
            unset,
        })
    }
    #[cfg(not(feature = "dspark-hf"))]
    {
        let _ = (hp, width, dense);
        if unset {
            return DraftRule::plain(
                levers,
                "unset: this build reads no DSpark draft (the dspark-hf feature)".to_owned(),
            );
        }
        Err(no_dspark_reader())
    }
}

/// The DSpark draft and where it was found, through the common discovery
/// ([`refset::arch::draft`]): `$BLOOMERY_DSPARK_MODEL` (a checkpoint folder or a GGUF file, refused
/// by name in neither form), else what sits beside the target ([`dspark_beside`]: its own
/// `mtp.safetensors` for a folder, the one `dspark-*.gguf` for a GGUF file); `None` for neither. Two
/// candidates beside the target, and a path in neither form, are the common function's error text.
#[cfg(feature = "dspark-hf")]
fn dspark_source_of(target: &Path) -> Result<Option<(DsparkForm, &'static str)>, String> {
    use refset::arch::draft::{DraftFrom, dspark_lever, dspark_pick};

    let lever = dspark_lever();
    let set = lever.as_deref().map(Path::new);
    let Some((path, from)) = dspark_pick(set, target).map_err(|e| e.to_string())? else {
        return Ok(None);
    };
    let form = DsparkForm::of(&path).map_err(|e| match set {
        Some(_) => format!("BLOOMERY_DSPARK_MODEL: {e}"),
        None => e.to_string(),
    })?;
    let from = match from {
        DraftFrom::Lever(_) => "BLOOMERY_DSPARK_MODEL",
        DraftFrom::Folder => "the target's folder",
        DraftFrom::Beside | DraftFrom::Family(_) => "beside the target",
    };
    Ok(Some((form, from)))
}

/// The form's name for the `draft rule` record.
#[cfg(feature = "dspark-hf")]
fn form_name(form: &DsparkForm) -> &'static str {
    match form {
        DsparkForm::Folder(_) => "checkpoint folder",
        DsparkForm::Gguf(_) => "GGUF file",
    }
}

/// The visits a file walks: its stored layers, the core's once more a pass past the first.
fn visits_of(hp: &Hparams) -> usize {
    match hp.loop_ {
        Some(lp) if lp.passes > 1 => {
            hp.n_layer + (lp.core_end.saturating_sub(lp.core_start)) * (lp.passes - 1)
        }
        _ => hp.n_layer,
    }
}

/// A probe that carries no plan: the context is chosen on each plan's outcome alone, and the
/// plan at the chosen context is made once more, by the machine that holds the draft's reserve
/// there ([`machine_at`]).
fn unit<P>(probe: Probe<P>) -> Probe<()> {
    match probe {
        Probe::Holds(_) => Probe::Holds(()),
        Probe::Short(short) => Probe::Short(short),
    }
}

/// The machine the seat plans over at `ctx` positions: the placement's, the side model's bytes
/// set aside ([`beside::reserve_on`]) and, under a DSpark draft, the draft's reserve
/// ([`OpenDraft::reserve_at`]). Nothing of the lookup's: its verify runs on the body's own pass.
fn machine_at(
    make_machine: impl Fn(usize) -> Machine,
    (layers, held): (usize, Beside),
    draft: Option<&OpenDraft>,
    ctx: u64,
) -> Result<Machine, PlaceError> {
    let mut machine = make_machine(layers);
    beside::reserve_on(&mut machine, held);
    if let Some(bytes) = draft.map(|d| d.reserve_at(ctx)) {
        machine.reserve(draft_reserve(Device::Card(0), bytes))?;
    }
    Ok(machine)
}

/// The plan of `inputs` on `machine` at `ctx` positions: under a draft the whole-card plan (a
/// draft asks for the whole-card load), else the seat's shape's.
fn plan_on<'a>(
    machine: &'a Machine,
    (inputs, plan_levers): (&'a Inputs, &'a PlanLevers),
    (shape, drafted): (Shape, bool),
    ctx: u64,
) -> Result<Plan<'a>, PlaceError> {
    if drafted {
        whole_card_plan(inputs, machine, ctx, plan_levers)
    } else {
        seat_plan(inputs, machine, ctx, plan_levers, shape)
    }
}

impl OpenDraft {
    /// What the plan sets aside for the draft at `ctx` positions: a DSpark draft's
    /// ([`draft_open::Draft::reserve`]), nothing for the lookup.
    fn reserve_at(&self, ctx: u64) -> u64 {
        self.reserve_fixed + self.reserve_per_pos * ctx
    }
}

/// The levers (unless `read_levers` is false) and flags [`run`] reads first, checked before the
/// model is fetched ([`bloomery_gpu_gates::bind::check_seat`]).
pub fn check(
    args: &[String],
    read_levers: bool,
) -> Result<bloomery_gpu_gates::bind::CheckedSeat<SeatFlags>, GateError> {
    bloomery_gpu_gates::bind::check_seat(read_levers.then_some(ACTS_ON), args, |args, levers| {
        let a = SeatFlags::parse_one_sequence(args, USAGE, "inkling2")?;
        if let Some(levers) = levers {
            draft_asked(levers)?;
        }
        Ok(a)
    })
}

/// Loads the model, and the embedding model `side` names beside it, and serves until the listener
/// or the engine fails; `Ok` carries why the server ended.
pub fn run(args: &[String], side: &side_seat::Flags) -> Result<ServeError, GateError> {
    record::at_main_with_flags(NAME, record::BLOOMERY_SERVE_INKLING2, |flag| {
        bloomery_gpu_gates::llama_launch::serve_option(args, flag)
    });
    let (a, levers) = check(args, true)?.run();

    arm_screen(levers.screen());
    let path = ref_model_path()?;
    // A directory is the official checkpoint folder, opened once here: its located header plans as
    // a split does, and the header and part source go to the engine thread.
    let weights = Weights::open(&path, levers.ink2_dense_q8(), levers.ink2_passes())?;
    let split = weights.header();
    let first = split
        .shard(0)
        .ok_or_else(|| format!("{} opened with no shard", path.display()))?;
    let arch = Arch::detect(first).map_err(|e| format!("{}: {e}", path.display()))?;
    if arch != Arch::Inkling2 {
        return Err(format!(
            "{} is a {} file; the inkling2 seat serves inkling2 files",
            path.display(),
            arch.name()
        )
        .into());
    }
    let chat = ChatSurface::read(
        &path,
        split.value("tokenizer.chat_template"),
        split.value("general.name"),
        a.template_file.path(),
        "inkling2",
    )?;
    let inputs = Inputs::read(split)?;
    let hp = Hparams::read(split).map_err(|e| format!("{}: {e}", path.display()))?;
    let plan_levers = PlanLevers::from_levers(&levers)?;
    let (slots, from) = slots_of(a.parallel, a.ctx.is_some(), DEFAULT_SLOTS)?;
    let census = gpu_census::census()?;
    let chosen = Place::choose_untiered(a.place, &census, "inkling2")?;
    chosen.record().eprint();
    let place = chosen.place;
    // The side model's seat before the plan, which holds its bytes when it sits on the stage card.
    let side = side.read()?;
    let side_seat = side.as_ref().map(|s| s.seat(place)).transpose()?;
    let held = Beside {
        image: None,
        embedding: side_seat.as_ref().and_then(|s| s.reserve),
    };
    let make_machine = place.machine_untiered()?;
    let mut machine = make_machine(inputs.model.layers);
    beside::reserve_on(&mut machine, held);
    let shape = Shape::of(
        &inputs,
        &machine,
        &plan_levers,
        matches!(weights, Weights::File(_)),
    );
    // The draft before the plan: its word decides the load the plan must hold (a draft asks for
    // the whole-card one), and a DSpark draft's bytes are set aside on the card at every context
    // the choice probes.
    let mut rule = draft_of(
        &levers,
        (&path, &hp),
        (shape, (levers.ink2_dense_q8(), levers.ink2_passes())),
    )?;
    let layers = inputs.model.layers;
    // Whether the plan holds at a context, on the machine of that context ([`machine_at`]): under
    // a draft the whole-card plan, else the seat's shape's. The plan itself is made once more at
    // the chosen context, on the machine the seat keeps.
    let plan_at = |draft: Option<&OpenDraft>, c: u64| -> Result<(), PlaceError> {
        let machine = machine_at(make_machine, (layers, held), draft, c)?;
        plan_on(
            &machine,
            (&inputs, &plan_levers),
            (shape, draft.is_some()),
            c,
        )
        .map(|_| ())
    };
    // The context the slots serve under `draft`. A folder's MXFP8 stacks have no host kernel, so
    // a context the card cannot hold whole is the plan's refusal, not a short card: with no
    // context asked, the seat takes the largest it holds.
    let choose = |draft: Option<&OpenDraft>| -> Result<Chosen<()>, GateError> {
        let ctx_ask = match (a.ctx, matches!(weights, Weights::Folder(_))) {
            (None, true) => Some(
                folder_fit(|c| match plan_at(draft, c) {
                    Ok(_) => Ok(true),
                    Err(PlaceError::Placement(PlacementError::NoHostKernel { .. })) => Ok(false),
                    Err(e) => probe_of(Err::<(), _>(e)).map(|_| false),
                })
                .map_err(|e| {
                    format!(
                        "--place {}: the inkling2 seat holds the whole model on the card (there \
                         is no placed load): {e}",
                        place.name()
                    )
                })?,
            ),
            (c, _) => c,
        };
        choose_ctx(split, (ctx_ask, slots), (None, None), |c| {
            probe_of(plan_at(draft, c)).map(unit).map_err(|e| {
                beside::refusal(
                    held,
                    format!(
                        "--place {}: the inkling2 seat {}: {e}",
                        place.name(),
                        match shape {
                            Shape::Whole => "holds the whole model on the card",
                            Shape::Mixed => {
                                "holds the dense trunk, the stores and a share of each layer's \
                                 experts on the card and the rest on the host"
                            }
                            Shape::Hosted => {
                                "holds the dense trunk and the stores on the card and every \
                                 routed expert on the host"
                            }
                        }
                    ),
                )
            })
        })
    };
    // The unset word's draft yields to the plain path where its reserve leaves the card short.
    let asked = match choose(rule.draft.as_ref()) {
        Err(e) if rule.unset => {
            let plain = choose(None)?;
            rule = DraftRule::plain(
                &levers,
                format!(
                    "{}: the plan does not hold the draft beside the context: {e}",
                    rule.why
                ),
            )?;
            plain
        }
        asked => asked?,
    };
    rule.eprint();
    let draft = rule.draft;
    let drafted = draft.is_some();
    eprintln!("{}{}", asked.line(), beside::note(&machine.cards[0]));
    let slot_ctx = usize::try_from(asked.slot_ctx)?;
    // The plan at the chosen context, on the machine the seat keeps for the load's records.
    let machine = machine_at(make_machine, (layers, held), draft.as_ref(), asked.slot_ctx)?;
    let plan = plan_on(
        &machine,
        (&inputs, &plan_levers),
        (shape, drafted),
        asked.slot_ctx,
    )?;
    record::plan(place.name(), &machine, &plan).eprint();
    if shape == Shape::Mixed {
        eprintln!(
            "mixed n_l={:?} of {} experts a layer, host_expert_bytes={}",
            plan.n_l, inputs.hp.n_expert, plan.host.expert_bytes
        );
    }
    need::print(&plan, 0, 0, true)?;
    parallel_line(slots, slot_ctx, from, Pass::One);
    if a.plan_only {
        bloomery_gpu_gates::start::plan_exit();
    }
    let props = EngineProps {
        model: Some(model_props(split, &inputs.model)),
        placement: placement_of(NAME, &machine, &plan),
        ..EngineProps::default()
    };
    let card = &machine.cards[0];
    let host_experts = shape.placed();
    drop(plan);
    let (tok, model) = weights.hand_over(&path)?;
    let open = OpenSeat {
        card: card.name.clone(),
        device: card.device,
        path: path.clone(),
        model,
        ctx: slot_ctx,
        draft,
        loopcd: levers
            .ink2_loopcd()
            .map(|omega| (omega, levers.ink2_loopcd_ref())),
        placed: host_experts.then(|| PlacedOpen {
            inputs,
            machine: machine.clone(),
            levers: plan_levers.clone(),
            host: levers.host(),
        }),
    };
    let device = card.name.clone();
    let vocab = Arc::new(Vocab::new(tok)?);
    let engine = SeatEngine::spawn(
        move || Inkling2::open(open),
        slot_ctx,
        vocab,
        device,
        props,
        0,
    )?;
    let side = side_seat::Side::start(side.zip(side_seat))?;
    let server = a.bind(engine, chat, &path, slots, &levers, side)?;
    Record::new(&record::LISTENING_INKLING2)
        .w("place", place.name())
        .u("ctx", slot_ctx)
        .u("slots", slots)
        .w("addr", server.local_addr()?)
        .eprint();
    Ok(server.run())
}
