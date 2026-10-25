//! `bloomery-serve --model minimaxm3` — the llama-server-compatible HTTP API on
//! the MiniMax-M3 engine: a minimax-m3 file, its layers on one card, the
//! routed experts the plan puts on the card, on an expert tier card when the
//! placement names one, and the rest on the host tier
//! (`PlanInputs::plan_with_slots`), opened in graph mode with the step
//! captured.
//!
//!     bloomery-serve --model minimaxm3 [-m PATH | --hf <repo>[:<quant>]]
//!                    [--host 127.0.0.1] [--port 8080] [--ctx C] [--alias NAME]
//!                    [--chat-template-file PATH] [--place W] [--plan]
//!                    [--parallel N] [--queue-depth Q] [--slot-save-path DIR]
//!                    [--api-key KEY] [--api-key-file FNAME]
//!
//! The server takes `-m`/`--hf` out before this seat parses (its module
//! doc); `--ctx-size` and `-c` are `--ctx` under llama-server's spellings. The
//! model's first shard gives the vocabulary, `tokenizer.chat_template` the chat
//! template (`--chat-template-file` replaces it) and `general.name` the alias
//! (`--alias` replaces it; `minimax-m3` for a file that names none).
//!
//! A model path that is a directory is a MiniMax checkpoint folder, served as stored
//! (`minimaxm3_folder::open`, no prepare step): its located header takes the first shard's place
//! in every rule and plan, its tokenizer is the vocabulary and its `chat_template.jinja` the
//! template, and the engine loads every card tensor from the folder's parts
//! ([`bloomery_gpu_minimaxm3::body::Folder`]) while the host reads its routed experts where the
//! folder stores them. The plan, the `need` record and the `load` record print as a file's do,
//! and so does the residency: the machine stages the folder's routed experts from its parts.
//! Sampling is the sampler crate's chain with no repetition penalty,
//! `temperature <= 0` the engine's argmax. Any other flag is refused as an
//! unknown argument.
//!
//! `--parallel N` serves N resident sequences. Each sequence is its attention
//! planes — every layer's K and V, and a sparse layer's index keys — and the
//! written end, exchanged by a select ([`bloomery_gpu::Slots`]). The step
//! walks one token, so a round of several slots is a select and a step each
//! (`pass=turns`). One pass would be a host walk over the existing launches:
//! the row-wise ones once over every busy row, and each slot's sequence-bound
//! ones over that slot's rows and its own planes. A prompt cache is refused
//! by name, and a request keeps the whole prefix it shares with what the slot
//! holds: every store is a plane of positions, so a cut keeps every held
//! position.
//!
//! The prompt call is fed in batches of up to a unit's columns
//! (`app::Prompt for Body`), the same bits as one decode step an id; the
//! batch's buffers are in the plan's scratch term, which the body holds its
//! made bytes to by name at the load.
//!
//! `--place W` is the shared placement word (`generate::Place`): `a` (the
//! largest visible card), `gate`, or one card of the census by name or
//! ordinal. Unset, the common rule decides (`Place::choose` by
//! [`TIER_RULE`]), on one census reading: two cards in view run `bp` when its
//! plan puts the rule's experts on the tier card, else `a`; a `place unset`
//! record names the word and why. A placement
//! that hangs an expert tier card (`bp`, a two-card list) plans the tier's
//! prompt-batch bytes on it ([`MiniMax::tier_batch`]) and the body hangs the
//! card under the host tier.
//!
//! `--ctx` is the total context the slots share (`--ctx 0` is refused);
//! `--parallel N` gives each slot an Nth of it. The context comes first and
//! the plan fills the machine around it (`generate::choose_ctx`, every
//! seat's): a set value past the file's trained context stops by name, and
//! one the machine cannot hold stops with the largest it would hold. Unset,
//! the one slot asks the trained context or 262,144 positions where that is
//! less, halved while the card cannot hold the stores and resident bytes, or
//! while the host cannot hold the routed experts the card then leaves it: the
//! plan reads the host room (`BLOOMERY_HOST_ROOM` replaces the reading) and
//! the body serves no NVMe expert tier (`PlaceError::NoNvmeTier`), so a
//! shorter context, which keeps more experts on the card, is what that
//! refusal asks for.
//!
//! The records on stderr, in order: `place unset` (word and why), one `ctx`
//! line (`placement::ctx_ask::Chosen::line`), the `residency lever` record
//! (the word and why), the `plan` record, under `mid` the `residency host`
//! record of the churn pool, the `need` record, the `parallel` line
//! (`rule=slots slots=N slot_ctx=C total=N*C from=<word> pass=one` for one
//! slot and `pass=turns` for several); then the `load` record (architecture,
//! resident bytes, a slot's context, the slots, layers, the routed experts
//! the plan keeps on the host and on the card, the step graph's nodes, the
//! prompt group and the load's wall) and, once the port is bound, the
//! `listening` record (`record::BLOOMERY_SERVE_MINIMAXM3`). `--plan` ends the
//! process after the `parallel` line, nothing loaded. An engine error ends
//! the process with the crash block and exit code 70, as every seat's.
//!
//! The levers this seat acts on ([`ACTS_ON`]) are parsed once, at `main`
//! (`bloomery_levers::at_main`), which refuses by name a lever set outside
//! them: the plan's card budget, the host set's load settings,
//! `BLOOMERY_PREFILL_GROUP` (the batches a prompt group runs layer by layer:
//! unset, the row's default), and `BLOOMERY_PIN_MAIN` with
//! `BLOOMERY_HOST_LANES` (the engine thread pinned to the dispatcher's cpu
//! slot, which gives the host legs the pool's CCD map), and the schema jump
//! (`BLOOMERY_SCHEMAJUMP`).
//!
//! `BLOOMERY_RESIDENCY` is the common machine's lever: set, its word as given
//! (refused by name where the plan's card slots leave it no room); unset, the
//! serving word on every placement but `gate`, which the plan then keeps,
//! moves or turns `off` with why (`bloomery_levers::residency_unset`,
//! `residency::at_plan`). The `residency lever` record comes before the plan's
//! own; under `mid`, the `residency host` record of the churn pool comes after
//! the plan. Under `mid` the experts the card holds are the machine's to move
//! by use.
//!
//! `BLOOMERY_XSTREAM` resolves by the rule the Qwen3.8 seat runs
//! (`shared/xstream_rule.rs`): set as given; unset `split` under `--place a`
//! with a residency machine (`admit` when the card has no room for the
//! stream's ring), `admit` under a placement with an expert tier card, `off`
//! everywhere else. A load that streams plans with the stream's room
//! (`PlanInputs::with_stream`, asked before the plan by `plans_ring`); the open resolves the lever last, after the
//! slots, and prints its `xstream=` line (the word, then why) before the
//! `load` record.
//!
//! The model writes its reasoning between the think span's two tags (the pair
//! its description names, `models::ReasoningFormat::ThinkSpan`: the template's
//! `think_begin_token` and `think_end_token`); the server splits the
//! reasoning, runs the reasoning budget and writes a structured-output
//! request's close on them (`Server::with_think_tags`). The template's generation
//! prompt opens no span: the model opens it on a leading tag, or answers with none.
//!
//! The seat sits behind the `deepseek41` feature beside `minimaxm3`: the
//! server surface it binds through (`bind`, the `serve` and `sampler` crates)
//! is scoped to `deepseek41`; it runs no V4.1 code.
//!
//! The run is the tiered seats' ([`super::tiered`]). The open is this
//! module's: the shared hybrid seat holds one sequence and plans one, and
//! Solar stays there. This seat plans `slots` sequences and parks the rest
//! after the step is captured, so the loaded plan is the one the `parallel`
//! line counted.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use app::{Prompt, Session};
use bloomery_gpu::GpuError;
use bloomery_gpu::model::{ChainBody, StepMode};
use bloomery_gpu_gates::GateError;
use bloomery_gpu_gates::beside::{self, Beside};
use bloomery_gpu_gates::bind::{Seat, note_line};
use bloomery_gpu_gates::generate::{Place, TierRule};
use bloomery_gpu_gates::record::{self, Record};
use bloomery_gpu_gates::side_seat;
use bloomery_levers::Levers;
use gguf::Split;
use model::arch::Arch;
use model::arch::models::ReasoningFormat;
use model::arch::place_hybrid::PlanInputs;
use model::placement::PlacementError;
use model::placement::workstation::TierBatchBytes;
use runtime::Target;
use serve::reasoning::ThinkTags;
use serve::{CacheNote, Saved, ServeError};

use super::minimaxm3_folder;
use super::tiered::{self, Open, Read, Tiered};

use super::xstream_rule::{Stage, plans_ring, xstream_rule};

use bloomery_gpu_minimaxm3::Body;
use bloomery_gpu_minimaxm3::prefill::set_prefill_group;

/// The binary's name, its records' owner and the `note` line's.
const NAME: &str = "bloomery-serve-minimaxm3";

const USAGE: &str = "usage: bloomery-serve --model minimaxm3 [-m PATH | --hf <repo>[:<quant>]] \
                     [--host H] [--port P] [--ctx C] [--alias NAME] [--chat-template-file PATH] \
                     [--place W] [--plan] [--parallel N] [--queue-depth Q] [--slot-save-path DIR] \
                     [--api-key KEY] [--api-key-file FNAME]";

/// The levers this seat acts on: the plan's card budget and the host set's
/// load settings, which the open reads (`PlanLevers::from_levers`,
/// `Levers::host`), the residency machine's word, the batches a prompt group
/// runs, the engine thread's pin, the host legs' lane rule, the start screen
/// and the schema jump. The V4.1 host tier's r8 sidecar (`BLOOMERY_R8`) is
/// not among them: this body reads its routed experts from the source.
pub const ACTS_ON: &[&str] = &[
    bloomery_levers::CARD_BUDGET,
    bloomery_levers::RESIDENCY,
    bloomery_levers::PREFILL_GROUP,
    bloomery_levers::HOST_POPULATE,
    bloomery_levers::HOST_LOCK,
    bloomery_levers::CARD_DONTNEED,
    bloomery_levers::PIN_MAIN,
    bloomery_levers::HOST_LANES,
    bloomery_levers::SCREEN,
    bloomery_levers::SCHEMAJUMP,
    bloomery_levers::XSTREAM,
];

/// The family's input to the common unset rule (`Place::choose`): the one
/// expert tier card the body hangs, kept only when the plan puts at least this
/// many experts on it — the tier size whose host time saved covers the tier's
/// enqueue on the step's path (docs/cards/minimaxm31-2card.card).
pub const TIER_RULE: TierRule = TierRule {
    tiers: bloomery_gpu::host::SERVED_TIERS,
    break_even: Some(92),
    basis: "docs/cards/minimaxm31-2card.card",
};

/// The seat: the session and the positions one slot's stores were sized for.
struct MiniMax {
    s: Session<Body>,
    ctx: usize,
}

impl MiniMax {
    /// The session of the model at `a.path` (a file, or the folder `a.folder` the common run
    /// opened) on `a.place`, the step captured, `a.slots` resident sequences parked after it
    /// (each capturing on its first use), the `load` record on stderr, on the calling thread,
    /// pinned to the dispatcher's cpu slot when asked.
    ///
    /// The plan is [`PlanInputs::plan_with_slots`] at `a.ctx` and `a.slots`,
    /// the same call the run's probe made, and the body is opened for that
    /// many sequences. The shared session open plans one sequence, so this
    /// seat does not use it.
    fn open(a: Open<()>) -> Result<MiniMax, GateError> {
        let pinned = a.pin_main && threads::pool().pin_caller();
        let t = Instant::now();
        let file = match a.folder {
            Some(f) => minimaxm3_folder::model(f.header, f.source)?,
            None => minimaxm3_folder::model_at(&a.path)?,
        };
        let inputs = model::arch::minimaxm3::place::PlanInputs::describe(file.header())?;
        let inputs = if a.ring {
            MiniMax::with_ring(inputs)?
        } else {
            inputs
        };
        let machine = beside::with_reserve(
            a.place.machine(None, a.tier_batch)?,
            Beside {
                image: None,
                embedding: a.embedding,
            },
        )(inputs.model.layers);
        let ctx = u64::try_from(a.ctx)
            .map_err(|_| format!("a context of {} positions passes u64", a.ctx))?;
        let plan = inputs
            .plan_with_slots(&machine, ctx, &a.plan, a.slots)
            .map_err(|e| GpuError::plan("minimaxm3 seat", e))?;
        if plan.ctx_max != ctx {
            return Err(format!(
                "the plan serves {} positions; the slot asked for {ctx}",
                plan.ctx_max
            )
            .into());
        }
        let host_experts = plan.host.experts;
        let card_experts = plan
            .cards
            .first()
            .map(|c| c.experts)
            .ok_or("the plan has no card")?;
        let opened = t.elapsed();
        let mut model =
            Body::open_placed_slots(file, &plan, &inputs, 0, a.host, a.residency, a.slots)?;
        model.set_mode(StepMode::Graph);
        let nodes = model.capture_step()?;
        set_prefill_group(&mut model, a.group)?;
        let ctx_u32 = u32::try_from(plan.ctx_max)
            .map_err(|_| format!("the plan's ctx_max {} passes u32", plan.ctx_max))?;
        let mut s = Session::from_model(model, ctx_u32);
        if a.slots.get() > 1 {
            s.add_slots(a.slots.get())?;
        }
        // The expert stream last: a started ring takes the card's free bytes past its keep, so
        // every card buffer the open makes comes before it.
        let (gpu, _, body) = s.model_mut().body_parts(NAME)?;
        let line = xstream_rule(gpu, body, a.xstream, Stage::of(a.place), a.residency)?;
        eprintln!("{line}");
        if let Some((card, rate)) = &a.lane {
            record::lane(card, rate).eprint();
        }
        let total = t.elapsed();
        Record::new(&record::LOAD_MINIMAXM3)
            .w("arch", <Body as ChainBody>::arch().name())
            .u("resident_bytes", s.model().resident_bytes())
            .u("ctx", a.ctx)
            .u("slots", a.slots.get())
            .u("layers", s.model().layers().len())
            .u("host_experts", host_experts)
            .u("card_experts", card_experts)
            .u("graph_nodes", nodes)
            .u("group", a.group)
            .w("pin_main", if a.pin_main { "on" } else { "off" })
            .w("pinned", pinned)
            .f("load_s", total.as_secs_f64())
            .eprint();
        record::eprint_load_phases(s.model(), total, opened);
        Ok(MiniMax { s, ctx: a.ctx })
    }
}

impl Seat for MiniMax {
    fn pos(&self) -> usize {
        self.s.model().pos() as usize
    }

    fn ctx_max(&self) -> usize {
        self.ctx
    }

    /// The resident sequences the seat made at its open (`app::Session::slots`).
    fn slots(&self) -> usize {
        self.s.slots()
    }

    /// The session's slot (`app::Session::select_slot`): the model exchanges
    /// its live sequence with the slot's parked state.
    fn select(&mut self, slot: usize) -> Result<(), GateError> {
        Ok(self.s.select_slot(slot)?)
    }

    /// The prompt in batches (`app::Prompt for Body`), the argmax after the last.
    fn prefill(&mut self, ids: &[u32]) -> Result<u32, GateError> {
        Ok(<Body as Prompt>::prompt(self.s.model_mut(), ids)?)
    }

    fn step(&mut self, last: u32) -> Result<u32, GateError> {
        Ok(self.s.model_mut().step(&[last])?)
    }

    fn logits_into(&self, row: &mut [f32]) -> Result<(), GateError> {
        Ok(self.s.model().logits_into(row)?)
    }

    /// The model's reset ([`bloomery_gpu::GpuModel::reset`]): the live sequence
    /// only. The body never holds a verify's rows, so the session's own reset
    /// adds nothing to it.
    fn reset(&mut self) -> Result<(), GateError> {
        Ok(self.s.model_mut().reset()?)
    }

    /// The session's cut (`app::Keep`), which takes only a position
    /// [`Seat::keep`] granted; any other is refused by name.
    fn rollback(&mut self, pos: u32) -> Result<(), GateError> {
        Ok(self.s.cut(pos)?)
    }

    /// The model's rule (`app::Keep for Body`) and the sentence that kept
    /// less: every held position, the empty model, or a checkpoint.
    fn keep(&self, n: usize) -> (usize, Option<String>) {
        self.s.keep_query(n)
    }

    /// Nowhere: no prompt call is cut for a prefix a later request keeps; the
    /// body takes no checkpoint at a mark yet.
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

/// Why the seat takes no sequence state: it runs no prompt cache.
const NO_CACHE: &str = "the minimaxm3 seat runs no prompt cache";

impl Tiered for MiniMax {
    type H = model::arch::minimaxm3::hparams::Hparams;
    type Flags = ();

    const NAME: &'static str = NAME;
    const USAGE: &'static str = USAGE;
    const ACTS_ON: &'static [&'static str] = ACTS_ON;
    const ALIAS: &'static str = "minimax-m3";
    const FAMILY: &'static str = "minimaxm3";

    const TIER_RULE: TierRule = TIER_RULE;
    const SERVES: &'static str = "minimax-m3";
    const RESIDENT: bool = true;
    /// The body serves slots in turn ([`bloomery_gpu::Slots`], not one pass).
    const ONE_PASS: bool = false;

    fn accepts(arch: Arch) -> bool {
        matches!(arch, Arch::MiniMaxM3)
    }

    /// The ring is kept out of the expert budget where the common rule starts the stream at
    /// `split` ([`plans_ring`]).
    fn ring(levers: &Levers, place: Place) -> bool {
        plans_ring(
            levers.xstream(),
            Stage::of(place),
            levers.residency() == Some("off"),
        )
    }

    /// The inputs of a plan that keeps the stream's ring and its ring pass's buffers out of the
    /// expert budget (`PlanInputs::with_stream`).
    fn with_ring(inputs: PlanInputs<Self::H>) -> Result<PlanInputs<Self::H>, GateError> {
        Ok(inputs.with_stream()?)
    }

    /// A checkpoint folder's open is [`minimaxm3_folder::open`], as the end-to-end gates' seat
    /// opens it; a file is the common run's.
    fn read_model(path: &Path) -> Result<Read, GateError> {
        if !path.is_dir() {
            return tiered::read_file(path);
        }
        let f = minimaxm3_folder::open(path)?;
        Ok(Read {
            split: f.header,
            tokenizer: Some(f.tokenizer),
            source: Some(f.source),
        })
    }

    /// The family's own reading (`PlanInputs::describe`), as its end-to-end
    /// gate's seat reads the file: the coverage list still names the
    /// template's tool-call parser.
    fn read(split: &Split) -> Result<PlanInputs<Self::H>, PlacementError> {
        model::arch::minimaxm3::place::PlanInputs::describe(split)
    }

    fn flags(_flag: &str, _value: &str, _out: &mut ()) -> Result<bool, GateError> {
        Ok(false)
    }

    /// The tier's prompt-batch bytes of the file's widths and down forms over the body's prompt
    /// unit ([`bloomery_gpu_minimaxm3::tier::tier_batch`]), reserved on the tier card and the
    /// host; `None` on a placement with no tier card.
    fn tier_batch(place: Place, inputs: &PlanInputs<Self::H>) -> Option<TierBatchBytes> {
        (!place.tier_cards().is_empty()).then(|| bloomery_gpu_minimaxm3::tier::tier_batch(inputs))
    }

    /// The tags the model's description names for its think span.
    fn think(inputs: &PlanInputs<Self::H>) -> Result<Option<ThinkTags>, GateError> {
        match inputs.spec.chat.reasoning {
            Some(ReasoningFormat::ThinkSpan { open, close }) => Ok(Some(ThinkTags { open, close })),
            None => Err(format!("the {} description names no reasoning tags", Self::FAMILY).into()),
        }
    }

    fn register() {
        record::at_main(NAME, record::BLOOMERY_SERVE_MINIMAXM3);
    }

    fn listening() -> Record {
        Record::new(&record::LISTENING_MINIMAXM3)
    }

    fn open(a: Open<Self::Flags>) -> Result<Self, GateError> {
        MiniMax::open(a)
    }
}

/// Loads the model, and the embedding model `side` names beside it, and serves until the listener
/// or the engine fails; `Ok` carries why the server ended.
pub fn run(args: &[String], side: &side_seat::Flags) -> Result<ServeError, GateError> {
    tiered::run::<MiniMax>(args, side)
}
