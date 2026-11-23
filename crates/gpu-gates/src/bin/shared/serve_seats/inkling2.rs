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
//!   prompt fed again. The draft is the path that asks a body for a rollback, and it is not
//!   wired to this seat: `BLOOMERY_DRAFT` set to any word but `off` is refused by name before
//!   any load. The seat runs no prompt cache and takes no snapshot.
//! - The prompt runs by the body's prompt rule (`app::Prompt`): whatever the session does
//!   with the ids, the seat does the same, and the first id of the answer is a step of the
//!   last prompt id.
//! - The load is the whole model on one card (`GpuModel::open`) when the card leg runs every
//!   routed stack of the file: no routed expert sits on the host. The plan is the body's
//!   ([`Inputs::plan`], every routed stack of a type the body's card leg runs on the card
//!   beside the dense weights), and a plan that would leave experts off the card is refused by
//!   name, a short card at that context ([`card_experts_floor`]). A file whose stacks the card
//!   leg does not run (the Small's IQ2_XS, IQ3_XXS and IQ4_XS experts) takes the placed load
//!   (`GpuModel::open_placed`): the dense trunk and the stores on the card, every routed expert
//!   served by the host tier, the prompt stepped one id at a time. The unset `--ctx` is the
//!   largest context the plan holds, up to the file's trained context: for the Small the
//!   card's stores (the 40 windowed layers' rings and the 8 global layers' keys and values)
//!   are the bound.
//!
//! `--place W` is the shared placement word (`generate::Place`); unset, the common untiered
//! rule decides on one census reading. A placement the plan refuses (more than one card, a tier
//! card, a card the file does not fit) stops before any load by the plan's own refusal.
//! `--ctx` is the slot's context, chosen by `generate::choose_ctx` as every seat's is.
//!
//! `--embd-model` is taken before this seat parses. The side model is seated before the plan;
//! the stage card holds its bytes. It loads after the chat model, and the server answers its
//! embeddings routes from it.
//!
//! The records on stderr, in order: `place unset`, the `embedding_seat` record when one is
//! named, one `ctx` line (plus the stage card's reserve note when it holds the embedding
//! model), the `plan` record, the `parallel` line, then the `load` record
//! (`record::LOAD_INKLING2`), the load's phases and, once the port is bound, the `listening`
//! record (`record::BLOOMERY_SERVE_INKLING2`). `--plan` ends the process after the `parallel`
//! line, nothing loaded.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use super::folder;
use app::{Keep, Prompt};
use bloomery_gpu::arch::inkling2::Body;
use bloomery_gpu::folder::Model;
use bloomery_gpu::model::{ChainBody, StepMode};
use bloomery_gpu::{Gpu, GpuModel};
use bloomery_gpu_gates::beside::{self, Beside};
use bloomery_gpu_gates::bind::{
    ChatSurface, Pass, Seat, SeatEngine, SeatFlags, Vocab, arm_screen, model_props, note_line,
    parallel_line, placement_of, prefill_closing, slots_of,
};
use bloomery_gpu_gates::generate::{Place, choose_ctx};
use bloomery_gpu_gates::need;
use bloomery_gpu_gates::record::{self, Record};
use bloomery_gpu_gates::side_seat;
use bloomery_gpu_gates::{GateError, gpu_census, ref_model_path};
use bloomery_levers::{DraftWord, HostCfg, Levers};
use gguf::Split;
use model::arch::Arch;
use model::arch::inkling2::place::Inputs;
use model::arch::place_hybrid::PlaceError;
use model::placement::accepts::fused_w13_runs;
use model::placement::ctx_ask::{DEFAULT_SLOTS, card_experts_floor, probe_of};
use model::placement::workstation::DeviceId;
use model::placement::{Machine, PlacementError, Plan, PlanLevers, Role};
use serve::{CacheNote, EngineProps, Saved, ServeError};
use tokenizer::Tokenizer;

const NAME: &str = "bloomery-serve-inkling2";

const USAGE: &str = "usage: bloomery-serve --model inkling2 [-m PATH | --hf <repo>[:<quant>]] \
                     [--host H] [--port P] [--ctx C] [--alias NAME] [--chat-template-file PATH] \
                     [--place W] [--plan] [--queue-depth Q] [--slot-save-path DIR] \
                     [--api-key KEY] [--api-key-file FNAME]";

/// The levers this seat acts on: the plan's card budget, the start screen, how a
/// structured-output request's fixed runs are fed, and the draft word, which the seat
/// refuses by name unless it is `off` ([`draft_asked`]).
pub const ACTS_ON: &[&str] = &[
    bloomery_levers::CARD_BUDGET,
    bloomery_levers::SCREEN,
    bloomery_levers::SCHEMAJUMP,
    bloomery_levers::DRAFT,
    bloomery_levers::INK2_DENSE,
    bloomery_levers::INK2_PASSES,
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

/// `BLOOMERY_DRAFT` read for a seat with no draft: unset and `off` pass, any other word is
/// refused by name with the reason the seat has none.
fn draft_asked(levers: &Levers) -> Result<(), GateError> {
    levers.draft_word(&[DraftWord::Off]).map(drop).map_err(|e| {
        format!(
            "BLOOMERY_DRAFT={}: the inkling2 seat drafts nothing: the draft and its verify pass \
             are not wired to it; unset it or set off",
            e.word
        )
        .into()
    })
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

/// The plan of `inputs` on `machine` at `ctx` positions of a file whose routed experts the card
/// does not run: the body's plan ([`Inputs::plan`], every routed stack on the host, the dense
/// weights and the stores on the card) refused where its terms break a bound.
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

/// The plan the seat loads by: the whole-card plan of a file the card leg runs, the hosted plan
/// of one it does not.
fn seat_plan<'a>(
    inputs: &'a Inputs,
    machine: &'a Machine,
    ctx: u64,
    levers: &PlanLevers,
) -> Result<Plan<'a>, PlaceError> {
    if card_runs_the_experts(inputs) {
        whole_card_plan(inputs, machine, ctx, levers)
    } else {
        hosted_plan(inputs, machine, ctx, levers)
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
/// map reads it).
#[cfg(feature = "checkpoint")]
fn open_folder(path: &Path, q8: bool, passes: Option<usize>) -> Result<folder::Opened, GateError> {
    let arch = refset::arch::inkling2::ARCH;
    match passes {
        None => folder::open_with(path, arch, q8),
        Some(_) => folder::adopt(
            path,
            arch,
            model_hf::arch::inkling2::open_passes(path, q8, passes),
        ),
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

impl Weights {
    /// The weights at `path`: a directory is the checkpoint folder, its dense matrices written as
    /// Q8_0 at load where `q8` asks (`BLOOMERY_INK2_DENSE`) and its loop run at `passes` passes
    /// where one is asked (`BLOOMERY_INK2_PASSES`, a variable-loop folder only); anything else is
    /// the GGUF file, which has no pass count to set.
    fn open(path: &Path, q8: bool, passes: Option<usize>) -> Result<Weights, GateError> {
        if path.is_dir() {
            let f = open_folder(path, q8, passes).map_err(|e| match passes {
                Some(p) => format!("{}={p}: {e}", bloomery_levers::INK2_PASSES).into(),
                None => e,
            })?;
            return Ok(Weights::Folder(Box::new(f)));
        }
        if let Some(p) = passes {
            return Err(format!(
                "{}={p}: {} is a GGUF file, which carries its own pass count; only a \
                 variable-loop checkpoint folder runs another",
                bloomery_levers::INK2_PASSES,
                path.display()
            )
            .into());
        }
        Ok(Weights::File(
            Split::open(path).map_err(|e| format!("open {}: {e}", path.display()))?,
        ))
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
                drop(header);
                let tokenizer = Tokenizer::from_gguf(path)?;
                let file = path.to_path_buf();
                let open: OpenModel = Box::new(move || -> Result<Model, GateError> {
                    Ok(Split::open(&file)
                        .map_err(|e| format!("open {}: {e}", file.display()))?
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

/// What the engine thread opens the model with.
struct OpenSeat {
    card: String,
    device: Option<DeviceId>,
    /// The model's open ([`Weights::hand_over`]).
    model: OpenModel,
    /// The slot's positions the visits' stores are sized for.
    ctx: usize,
    /// Whether the load serves a draft: the verify's graphs are captured at load, beside the
    /// step's.
    drafted: bool,
    /// The placed load's inputs, for a file whose experts the card leg does not run.
    placed: Option<PlacedOpen>,
}

/// The model, in graph mode with its step captured.
struct Inkling2 {
    m: GpuModel<Body>,
    ctx: usize,
    /// The buffer the head's padded logits row is read into.
    row: std::cell::RefCell<Vec<f32>>,
}

impl Inkling2 {
    /// The model `a.model` opens on the planned card, the step graph captured, and the `load`
    /// record on stderr.
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
        m.set_mode(StepMode::Graph);
        let nodes = m.capture_step()?;
        if a.drafted {
            m.capture_verify()?;
        }
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
        Ok(Inkling2 {
            m,
            ctx: a.ctx,
            row: std::cell::RefCell::new(vec![0.0; padded]),
        })
    }
}

impl Seat for Inkling2 {
    fn pos(&self) -> usize {
        self.m.pos() as usize
    }

    fn ctx_max(&self) -> usize {
        self.ctx
    }

    /// The body's prompt rule ([`Prompt::prompt`]) over `ids`.
    fn prefill(&mut self, ids: &[u32]) -> Result<u32, GateError> {
        Ok(<Body as Prompt>::prompt(&mut self.m, ids)?)
    }

    /// A prompt's last call with its last id the pass's last row ([`prefill_closing`]): the pass
    /// leaves the stores and the head's row where that id's step would, so the call closes and the
    /// row is read after it. The body's prompt rule is this one call, the gate's seat answer too.
    fn prefill_next(&mut self, ids: &[u32], row: Option<&mut [f32]>) -> Result<u32, GateError> {
        let closes = self.m.body(NAME)?.placed().is_none();
        prefill_closing(self, ids, row, closes)
    }

    fn step(&mut self, last: u32) -> Result<u32, GateError> {
        Ok(self.m.step(&[last])?)
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
        self.m.logits_into(&mut padded)?;
        row.copy_from_slice(&padded[..row.len()]);
        Ok(())
    }

    /// The model's reset: every ring and store back to empty, the position to 0.
    fn reset(&mut self) -> Result<(), GateError> {
        Ok(self.m.reset()?)
    }

    /// Asked only for a position [`Seat::keep`] granted, and refused by the body by name when
    /// it is below the floor (`inkling2::Body::rollback`).
    fn rollback(&mut self, pos: u32) -> Result<(), GateError> {
        Ok(self.m.rollback(pos)?)
    }

    /// `n` from the body's floor up to the held sequence, else nothing: the caller then resets
    /// the slot and feeds the prompt again ([`Keep::keepable`]).
    fn keep(&self, n: usize) -> (usize, Option<String>) {
        let at = u32::try_from(n).unwrap_or(u32::MAX);
        let kept = <Body as Keep>::keepable(&self.m, at) as usize;
        (kept, (kept < n).then(|| PAST_REACH.to_owned()))
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

    fn note(note: &CacheNote) {
        note_line(NAME, note);
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
    let mut machine = place.machine_untiered()?(inputs.model.layers);
    beside::reserve_on(&mut machine, held);
    // A folder's MXFP8 stacks have no host kernel, so a context the card cannot hold whole is the
    // plan's refusal, not a short card: with no context asked, the seat takes the largest it holds.
    let ctx_ask = match (a.ctx, matches!(weights, Weights::Folder(_))) {
        (None, true) => Some(
            folder_fit(
                |c| match whole_card_plan(&inputs, &machine, c, &plan_levers) {
                    Ok(_) => Ok(true),
                    Err(PlaceError::Placement(PlacementError::NoHostKernel { .. })) => Ok(false),
                    Err(e) => probe_of(Err::<Plan<'_>, _>(e)).map(|_| false),
                },
            )
            .map_err(|e| {
                format!(
                    "--place {}: the inkling2 seat holds the whole model on the card (there is no \
                 placed load): {e}",
                    place.name()
                )
            })?,
        ),
        (c, _) => c,
    };
    let asked = choose_ctx(split, (ctx_ask, slots), (None, None), |c| {
        probe_of(seat_plan(&inputs, &machine, c, &plan_levers)).map_err(|e| {
            beside::refusal(
                held,
                format!(
                    "--place {}: the inkling2 seat {}: {e}",
                    place.name(),
                    if card_runs_the_experts(&inputs) {
                        "holds the whole model on the card"
                    } else {
                        "holds the dense trunk and the stores on the card and every routed \
                         expert on the host"
                    }
                ),
            )
        })
    })?;
    eprintln!("{}{}", asked.line(), beside::note(&machine.cards[0]));
    let slot_ctx = usize::try_from(asked.slot_ctx)?;
    let plan = asked.plan;
    record::plan(place.name(), &machine, &plan).eprint();
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
    let host_experts = !card_runs_the_experts(&inputs);
    drop(plan);
    let (tok, model) = weights.hand_over(&path)?;
    let open = OpenSeat {
        card: card.name.clone(),
        device: card.device,
        model,
        ctx: slot_ctx,
        // `check` refused every draft word but off ([`draft_asked`]).
        drafted: matches!(
            levers.draft_word(&[DraftWord::Off]),
            Ok(Some(w)) if w != DraftWord::Off
        ),
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
