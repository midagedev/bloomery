//! `bloomery-serve --model solar2` — the llama-server-compatible HTTP API on
//! the Solar-Open2 engine: a solar-open2 file, its layers on one card, the
//! routed experts the plan puts on the card and the rest on the host tier
//! (`placement::plan_routed` through `PlanInputs::plan`), opened as
//! `gate_solar2_e2e` opens it (`app::Loaded`, in graph mode with the step
//! captured).
//!
//!     bloomery-serve --model solar2 [-m PATH | --hf <repo>[:<quant>]]
//!                    [--host 127.0.0.1] [--port 8080] [--ctx C] [--alias NAME]
//!                    [--chat-template-file PATH] [--place W] [--plan]
//!                    [--queue-depth Q] [--api-key KEY] [--api-key-file FNAME]
//!
//! The server takes `-m`/`--hf` out before this seat parses (its module
//! doc); `--ctx-size` and `-c` are `--ctx` under llama-server's spellings. The
//! model's first shard gives the vocabulary, `tokenizer.chat_template` the chat
//! template (`--chat-template-file` replaces it) and `general.name` the alias
//! (`--alias` replaces it; `solar-open2` for a file that names none).
//! Sampling is the sampler crate's chain with no repetition penalty,
//! `temperature <= 0` the engine's argmax. Any other flag is refused as an
//! unknown argument.
//!
//! Two facts of the architecture set this seat's shape, and nothing else
//! does. The program runs no verify pass of several rows and reads no draft,
//! so the server feeds the prompt less its last id as one prompt call, then
//! steps the last, as `bloomery-serve --model qwen3` does. And the 36 KDA
//! layers each hold one recurrent state written in place by every position,
//! so the body holds one sequence: `--parallel` above 1 is refused by name, a
//! prompt cache is refused by name, and a request keeps the prefix it shares
//! with what the slot holds only where the sequence stands (the model's
//! checkpoints, `app::Keep for Body`, widen that rule once a prompt call takes
//! them); any shorter prefix re-feeds the prompt from an empty model.
//!
//! The prompt call is fed in batches of up to a unit's columns
//! (`app::Prompt for Body`), the same bits as one decode step an id; the
//! batch's buffers come out of the card's free bytes at the load, refused by
//! name when they do not fit.
//!
//! `--place W` is the shared placement word (`generate::Place`): `a` (the
//! largest visible card), `gate`, or one card of the census by name or
//! ordinal. Unset, the common rule every serving seat takes decides
//! (`Place::choose_untiered`: the body serves no tier card), on one census
//! reading: a `place unset` record names the word `a` and why. A placement
//! that hangs an expert tier card (`bp`, a two-card list) is refused before
//! any load by the plan's own refusal (`PlaceError::Tier`), which names the
//! card and its tier.
//!
//! `--ctx` is the context the one slot serves (`--ctx 0` is refused). The
//! context comes first and the plan fills the machine around it
//! (`generate::choose_ctx`, every seat's): a set value past the file's
//! trained context stops by name, and one the machine cannot hold stops with
//! the largest it would hold. Unset, the slot asks 262,144 positions or the
//! trained context where that is less, halved while the card cannot hold the
//! stores and resident bytes, or while the host cannot hold the routed
//! experts the card then leaves it: the plan reads the host room
//! (`BLOOMERY_HOST_ROOM` replaces the reading) and the body serves no NVMe
//! expert tier (`PlaceError::NoNvmeTier`), so a shorter context, which keeps
//! more experts on the card, is what that refusal asks for.
//!
//! The records on stderr, in order: `place unset` (word and why), one `ctx`
//! line (`placement::ctx_ask::Chosen::line`), the `plan` record, the
//! `parallel` line (`rule=slots slots=1 slot_ctx=C total=C from=<word>
//! pass=one`); then the `load` record (architecture, resident bytes, the
//! slot's context, layers, the routed experts the plan keeps on the host and
//! on the card, the step graph's nodes, the prompt group and the load's wall)
//! and, once the port is bound, the `listening` record
//! (`record::BLOOMERY_SERVE_SOLAR2`). `--plan` ends the process after the
//! `parallel` line, nothing loaded. An engine error ends the process with the
//! crash block and exit code 70, as every seat's.
//!
//! The levers this seat acts on ([`ACTS_ON`]) are parsed once, at `main`
//! (`bloomery_levers::at_main`), which refuses by name a lever set outside
//! them: the plan's card budget, the host set's load settings,
//! `BLOOMERY_PREFILL_GROUP` (the batches a prompt group runs layer by layer:
//! unset, the row's default), and `BLOOMERY_PIN_MAIN` with
//! `BLOOMERY_HOST_LANES` (the engine thread pinned to the dispatcher's cpu
//! slot, which gives the host legs the pool's CCD map), and the schema jump
//! (`BLOOMERY_SCHEMAJUMP`). The load runs no adaptive expert residency.
//!
//! The model writes its reasoning between `<|think:start|>` and
//! `<|think:end|>`, the pair its description names
//! (`models::ReasoningFormat::ThinkSpan`); the server splits the reasoning, runs
//! the reasoning budget and writes a structured-output request's close on
//! them, not on `<think>` (`Server::with_think_tags`).
//!
//! The seat sits behind the `deepseek41` feature beside `solar2`: the server
//! surface it binds through (`bind`, the `serve` and `sampler` crates) is
//! scoped to `deepseek41`; it runs no V4.1 code.

use std::sync::Arc;
use std::time::{Duration, Instant};

use app::arch::solar2::Solar2Cfg;
use app::{Loaded, OpenArgs, OpenLog, Session, SessionError};
use bloomery_gpu::GpuModel;
use bloomery_gpu::model::{ChainBody, StepMode};
use bloomery_gpu_gates::GateError;
use bloomery_gpu_gates::beside::{self, Beside};
use bloomery_gpu_gates::bind::{Seat, note_line};
use bloomery_gpu_gates::generate::Lane;
use bloomery_gpu_gates::record::{self, Record};
use bloomery_gpu_gates::side_seat;
use bloomery_gpu_solar2::Body;
use gguf::Split;
use model::arch::Arch;
use model::arch::models::ReasoningFormat;
use model::arch::solar2::place::PlanInputs;
use model::placement::{Machine, PlacementError, Plan};
use runtime::Target as _;
use serve::reasoning::ThinkTags;
use serve::{CacheNote, Saved, ServeError};

use super::tiered::{self, Tiered};

const NAME: &str = "bloomery-serve-solar2";

const USAGE: &str = "usage: bloomery-serve --model solar2 [-m PATH | --hf <repo>[:<quant>]] \
                     [--host H] [--port P] [--ctx C] [--alias NAME] [--chat-template-file PATH] \
                     [--place W] [--plan] [--queue-depth Q] [--api-key KEY] \
                     [--api-key-file FNAME]";

/// The levers this seat acts on: the plan's card budget and the host set's
/// load settings, which the open reads (`PlanLevers::from_levers`,
/// `Levers::host`), the batches a prompt group runs, the engine thread's pin,
/// the host legs' lane rule, the start screen and the schema jump.
pub const ACTS_ON: &[&str] = &[
    bloomery_levers::CARD_BUDGET,
    bloomery_levers::PREFILL_GROUP,
    bloomery_levers::HOST_POPULATE,
    bloomery_levers::HOST_LOCK,
    bloomery_levers::CARD_DONTNEED,
    bloomery_levers::R8,
    bloomery_levers::PIN_MAIN,
    bloomery_levers::HOST_LANES,
    bloomery_levers::SCREEN,
    bloomery_levers::SCHEMAJUMP,
];

/// Why the seat takes no sequence state: it runs no prompt cache.
const NO_CACHE: &str = "the solar2 seat runs no prompt cache";

/// The Solar session on the engine thread and the positions its stores were
/// sized for.
struct Solar {
    s: Session<Body>,
    ctx: usize,
}

/// The open's records: the plan's expert counts are kept until the step is
/// captured, where the model's size and the step graph's nodes complete the
/// `load` record.
struct Log {
    t: Instant,
    /// The wall from `t` to the load, once the plan is made: the phases
    /// record's `open_s`.
    opened: Duration,
    ctx: usize,
    group: usize,
    host_experts: u64,
    card_experts: u64,
    graph_nodes: usize,
    /// `BLOOMERY_PIN_MAIN`'s ask and whether the pin took.
    pin_main: bool,
    pinned: bool,
    /// The lane's rate probed before the plan, and the card it ran on: the
    /// `lane` line beside the `load` record.
    lane: Option<Lane>,
}

impl OpenLog<Body> for Log {
    fn plan(
        &mut self,
        _place: &'static str,
        _inputs: &PlanInputs,
        _machine: &Machine,
        plan: &Plan<'_>,
    ) -> Result<bool, SessionError> {
        self.host_experts = plan.host.experts;
        self.card_experts = plan.cards[0].experts;
        self.opened = self.t.elapsed();
        Ok(true)
    }

    fn load(&mut self, _m: &GpuModel<Body>) -> Result<(), SessionError> {
        Ok(())
    }

    fn capture(&mut self, nodes: usize) -> Result<(), SessionError> {
        self.graph_nodes = nodes;
        Ok(())
    }

    fn prompt_buffers(&mut self, _m: &GpuModel<Body>) -> Result<(), SessionError> {
        Ok(())
    }
}

impl Log {
    /// The `load` record of `m`, the one slot ready.
    fn record(&self, m: &GpuModel<Body>) {
        if let Some((card, rate)) = &self.lane {
            record::lane(card, rate).eprint();
        }
        let total = self.t.elapsed();
        Record::new(&record::LOAD_SOLAR2)
            .w("arch", <Body as ChainBody>::arch().name())
            .u("resident_bytes", m.resident_bytes())
            .u("ctx", self.ctx)
            .u("slots", 1)
            .u("layers", m.layers().len())
            .u("host_experts", self.host_experts)
            .u("card_experts", self.card_experts)
            .u("graph_nodes", self.graph_nodes)
            .u("group", self.group)
            .w("pin_main", if self.pin_main { "on" } else { "off" })
            .w("pinned", self.pinned)
            .f("load_s", total.as_secs_f64())
            .eprint();
        record::eprint_load_phases(m, total, self.opened);
    }
}

impl Solar {
    /// The session of the file at `a.path` on `a.place`, the step captured,
    /// the `load` record on stderr, on the calling thread, pinned to the
    /// dispatcher's cpu slot when asked.
    fn open(a: tiered::Open<()>) -> Result<Solar, GateError> {
        let pinned = a.pin_main && threads::pool().pin_caller();
        let file = Split::open(&a.path).map_err(|e| format!("open {}: {e}", a.path.display()))?;
        let mut log = Log {
            t: Instant::now(),
            opened: Duration::ZERO,
            ctx: a.ctx,
            group: a.group,
            pin_main: a.pin_main,
            pinned,
            host_experts: 0,
            card_experts: 0,
            graph_nodes: 0,
            lane: a.lane,
        };
        let args = OpenArgs {
            place: a.place.name(),
            machine: beside::with_reserve(
                a.place.machine_untiered()?,
                Beside {
                    image: None,
                    embedding: a.embedding,
                },
            ),
            ctx: a.ctx,
            mode: StepMode::Graph,
            cfg: Solar2Cfg {
                place: a.plan,
                host: a.host,
                residency: a.residency,
                group: a.group,
            },
        };
        let s = Loaded::<Body>::open(file, args, &mut log)?
            .ok_or("the open stopped at its plan")?
            .ready(&mut log)?;
        log.record(s.model());
        Ok(Solar { s, ctx: a.ctx })
    }
}

impl Seat for Solar {
    fn pos(&self) -> usize {
        self.s.model().pos() as usize
    }

    fn ctx_max(&self) -> usize {
        self.ctx
    }

    /// The prompt in batches (`app::Prompt for Body`), the argmax after the
    /// last.
    fn prefill(&mut self, ids: &[u32]) -> Result<u32, GateError> {
        Ok(<Body as app::Prompt>::prompt(self.s.model_mut(), ids)?)
    }

    fn step(&mut self, last: u32) -> Result<u32, GateError> {
        Ok(self.s.model_mut().step(&[last])?)
    }

    fn logits_into(&self, row: &mut [f32]) -> Result<(), GateError> {
        Ok(self.s.model().logits_into(row)?)
    }

    /// The model's reset ([`GpuModel::reset`]): the body never holds a
    /// verify's rows, so the session's own reset adds nothing to it.
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

/// The tags the model's description names for its think span, which the server
/// splits the reasoning on ([`serve::Server::with_think_tags`]); the
/// description of every solar-open2 file names them.
fn think_tags(inputs: &PlanInputs) -> Result<ThinkTags, GateError> {
    match inputs.spec.chat.reasoning {
        Some(ReasoningFormat::ThinkSpan { open, close }) => Ok(ThinkTags { open, close }),
        None => Err("the solar2 description names no reasoning tags".into()),
    }
}

impl Tiered for Solar {
    type H = model::arch::solar2::hparams::Hparams;
    type Flags = ();

    const NAME: &'static str = NAME;
    const USAGE: &'static str = USAGE;
    const ACTS_ON: &'static [&'static str] = ACTS_ON;
    const KINDS: &'static [&'static record::Kind] = record::BLOOMERY_SERVE_SOLAR2;
    const LISTENING: &'static record::Kind = &record::LISTENING_SOLAR2;
    const ALIAS: &'static str = "solar-open2";
    const FAMILY: &'static str = "solar2";
    const SERVES: &'static str = "solar-open2";
    const RESIDENT: bool = false;

    fn accepts(arch: Arch) -> bool {
        matches!(arch, Arch::SolarOpen2)
    }

    fn read(split: &Split) -> Result<PlanInputs, PlacementError> {
        PlanInputs::read(split)
    }

    fn flags(_flag: &str, _value: &str, _out: &mut ()) -> Result<bool, GateError> {
        Ok(false)
    }

    fn think(inputs: &PlanInputs) -> Result<Option<ThinkTags>, GateError> {
        think_tags(inputs).map(Some)
    }

    fn open(a: tiered::Open<()>) -> Result<Solar, GateError> {
        Solar::open(a)
    }
}

/// Loads the model, and the embedding model `side` names beside it, and serves until the listener
/// or the engine fails; `Ok` carries why the server ended.
pub fn run(args: &[String], side: &side_seat::Flags) -> Result<ServeError, GateError> {
    tiered::run::<Solar>(args, side)
}
