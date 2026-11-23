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
//! - The load is the whole model on one card (`GpuModel::open`): no routed expert sits on the
//!   host. The plan is the body's ([`Inputs::plan`], every routed stack of a type the body's
//!   card leg runs on the card beside the dense weights), and a plan that would leave experts
//!   off the card is refused by name, a short card at that context ([`card_experts_floor`]),
//!   since there is no placed load yet. The unset `--ctx` is the largest context the card
//!   holds whole, up to the file's trained context.
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

use std::sync::Arc;
use std::time::Instant;

use app::{Keep, Prompt};
use bloomery_gpu::arch::inkling2::Body;
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
use bloomery_levers::{DraftWord, Levers};
use gguf::Split;
use model::arch::Arch;
use model::arch::inkling2::place::Inputs;
use model::arch::place_hybrid::PlaceError;
use model::placement::ctx_ask::{DEFAULT_SLOTS, card_experts_floor, probe_of};
use model::placement::workstation::DeviceId;
use model::placement::{Machine, Plan, PlanLevers};
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

/// What the engine thread opens the model with.
struct OpenSeat {
    card: String,
    device: Option<DeviceId>,
    path: std::path::PathBuf,
    /// The slot's positions the visits' stores are sized for.
    ctx: usize,
    /// Whether the load serves a draft: the verify's graphs are captured at load, beside the
    /// step's.
    drafted: bool,
}

/// The model, in graph mode with its step captured.
struct Inkling2 {
    m: GpuModel<Body>,
    ctx: usize,
    /// The buffer the head's padded logits row is read into.
    row: std::cell::RefCell<Vec<f32>>,
}

impl Inkling2 {
    /// The model of the file at `a.path` on the planned card, the step graph captured, and the
    /// `load` record on stderr.
    fn open(a: OpenSeat) -> Result<Inkling2, GateError> {
        let t = Instant::now();
        let file = Split::open(&a.path).map_err(|e| format!("open {}: {e}", a.path.display()))?;
        let opened = t.elapsed();
        let gpu = Gpu::open_card_after_hook(&a.card, a.device)?;
        let mut m = GpuModel::<Body>::open(gpu, file, a.ctx)?;
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
        prefill_closing(self, ids, row, true)
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
    let split = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
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
    let inputs = Inputs::read(&split)?;
    let tok = Tokenizer::from_gguf(&path)?;
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
    let asked = choose_ctx(&split, (a.ctx, slots), (None, None), |c| {
        probe_of(whole_card_plan(&inputs, &machine, c, &plan_levers)).map_err(|e| {
            beside::refusal(
                held,
                format!(
                    "--place {}: the inkling2 seat holds the whole model on the card (there is \
                     no placed load): {e}",
                    place.name()
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
        // `check` refused every draft word but off ([`draft_asked`]).
        drafted: matches!(
            levers.draft_word(&[DraftWord::Off]),
            Ok(Some(w)) if w != DraftWord::Off
        ),
    };
    let device = card.name.clone();
    drop(plan);
    drop(split);
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
