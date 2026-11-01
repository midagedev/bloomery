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
//!                    [--parallel N] [--slot-ctx S] [--cache-ram MIB] [--queue-depth Q]
//!                    [--slot-save-path DIR] [--api-key KEY] [--api-key-file FNAME]
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
//! ones over that slot's rows and its own planes.
//!
//! A request keeps the longest prefix it shares with what the slot holds (every
//! held position: the planes are rows of positions that a later position never
//! reads before it writes). The prompt cache (llama-server's `--cache-ram`, in
//! MiB; 0 turns it off) holds the selected slot's sequence state when a request
//! of another conversation takes the slot, and puts it back when the
//! conversation returns, so a return feeds only what the state does not hold.
//! A state is the planes' rows alone (`bloomery_gpu::checkpoint::planes`):
//! every layer's K and V rows of the held positions and a sparse layer's index
//! keys, with no recurrent store and no checkpoint, so it restores at any
//! position up to the one it holds, and the first positions the server keeps
//! are the only rows that move back. Its default is the lesser of
//! `bind::CACHE_RAM_CAP` and half of what the host leaves past the plan's need
//! and the residency's churn pool (`bind::CacheRam`); a `cache` line on stderr
//! after the plan prints it with each term. The copy runs on a stream of its
//! own into pinned staging that the budget pins whole at the open, begun when a
//! request ends and waited for only when the state is read back; a pool with no
//! room is the server's `NoRoom`, which makes room or saves on the request
//! path. A state of another model, card or context is refused by name.
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
//! the one slot asks the trained context, or the largest context under it whose
//! plan holds where the card cannot hold the stores and resident bytes. The
//! plan reads the host room (`BLOOMERY_HOST_ROOM` replaces the reading) and
//! pages the routed experts the room cannot hold on the NVMe expert tier,
//! which the body's host run serves; a room under that tier's floor is
//! refused by name.
//!
//! The records on stderr, in order: `place unset` (word and why), one `ctx`
//! line (`placement::ctx_ask::Chosen::line`), the `residency lever` record
//! (the word and why), the `plan` record, under `mid` the `residency host`
//! record of the churn pool, the `need` record, the `parallel` line
//! (`rule=slots slots=N slot_ctx=C total=N*C from=<word> pass=one` for one
//! slot and `pass=turns` for several); then the `load` record (architecture,
//! resident bytes, a slot's context, the slots, layers, the routed experts
//! the plan keeps on the host and on the card, the step graph's nodes, the
//! prompt group and the load's wall), the `nvtier` record (on a load whose plan
//! paged routed experts: the arena's budget, the paged bytes and its counters,
//! again after each prompt call) and, once the port is bound, the `listening`
//! record (`record::BLOOMERY_SERVE_MINIMAXM3`). `--plan` ends the process after
//! the `parallel` line, nothing loaded. An engine error ends the process with
//! the crash block and exit code 70, as every seat's.
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
use bloomery_gpu::checkpoint::planes::{seq_bytes, seq_lane};
use bloomery_gpu::checkpoint::saved::SaveLane;
use bloomery_gpu::host::served::TierBody;
use bloomery_gpu::model::{ChainBody, StepMode};
use bloomery_gpu_gates::GateError;
use bloomery_gpu_gates::bind::{Seat, note_line};
use bloomery_gpu_gates::generate::{Place, TierRule};
use bloomery_gpu_gates::record::{self, Record};
use bloomery_gpu_gates::side_seat;
use bloomery_levers::Levers;
use gguf::Split;
use model::arch::Arch;
use model::arch::models::ReasoningFormat;
use model::arch::place_hybrid::PlanInputs;
use model::placement::workstation::TierBatchBytes;
use model::placement::{PlacementError, Plan};
use runtime::Target;
use serve::reasoning::ThinkTags;
use serve::{CacheNote, Pending, Saved, ServeError};

use super::minimaxm3_folder;
use super::planes;
use super::tiered::{self, CacheFlag, Open, Read, Tiered};

use super::xstream_rule::{Stage, plans_ring, xstream_rule};

use bloomery_gpu_minimaxm3::Body;
use bloomery_gpu_minimaxm3::prefill::set_prefill_group;

/// The binary's name, its records' owner and the `note` line's.
const NAME: &str = "bloomery-serve-minimaxm3";

const USAGE: &str = "usage: bloomery-serve --model minimaxm3 [-m PATH | --hf <repo>[:<quant>]] \
                     [--host H] [--port P] [--ctx C] [--alias NAME] [--chat-template-file PATH] \
                     [--place W] [--plan] [--parallel N] [--slot-ctx S] [--cache-ram MIB] \
                     [--queue-depth Q] [--slot-save-path DIR] [--api-key KEY] [--api-key-file FNAME]";

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

/// What a refusal of another seat's state names ([`planes::resume`]).
const WHOSE: &str = "a minimaxm3 body's";

/// The seat: the session and the positions one slot's stores were sized for.
struct MiniMax {
    s: Session<Body>,
    ctx: usize,
    /// The copy stream and pinned staging saved states are made on
    /// ([`Seat::snapshots_async`]); none with the cache off.
    lane: Option<SaveLane>,
    /// The `tier census` records' span start ([`record::host_tier::CensusSpan`]): each
    /// prompt prints the steps since the last one's, then its own.
    census: record::host_tier::CensusSpan,
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
    fn open(a: Open<CacheFlag>) -> Result<MiniMax, GateError> {
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
        let machine = a.machine;
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
        let lane = match a.cache_ram {
            0 => None,
            ram => Some(seq_lane(s.model_mut(), ram)?),
        };
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
        let hybrid = s.model().body(NAME)?.hybrid();
        record::host_tier::eprint_nvtier(hybrid.nvme_tier());
        let mut census = record::host_tier::CensusSpan::default();
        census.mark("load", record::census_of(hybrid));
        Ok(MiniMax {
            s,
            ctx: a.ctx,
            lane,
            census,
        })
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

    /// The prompt in batches (`app::Prompt for Body`), the argmax after the last;
    /// the `tier census` records of the steps since the last prompt and of
    /// this one, then the `nvtier` record.
    fn prefill(&mut self, ids: &[u32]) -> Result<u32, GateError> {
        self.census.steps(Some(self.s.model().body(NAME)?.hybrid()));
        let next = <Body as Prompt>::prompt(self.s.model_mut(), ids)?;
        self.census
            .prompt(Some(self.s.model().body(NAME)?.hybrid()));
        Ok(next)
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

    fn snapshot_size(&mut self) -> Result<Option<u64>, GateError> {
        planes::snapshot_size(self.s.model_mut())
    }

    /// The selected slot's sequence state ([`planes::snapshot`]), as the
    /// prompt cache holds it.
    fn snapshot(&mut self) -> Result<Arc<dyn Saved>, GateError> {
        planes::snapshot(self.s.model_mut())
    }

    /// Whether the saves copy on the lane: the seat holds one.
    fn snapshots_async(&self) -> bool {
        self.lane.is_some()
    }

    /// The selected slot's state copy begun on the lane
    /// ([`planes::snapshot_begin`]); a pool with no room is the server's
    /// `NoRoom`. Without a lane, the snapshot on the call.
    fn snapshot_begin(&mut self) -> Result<Pending, GateError> {
        match &self.lane {
            Some(lane) => planes::snapshot_begin(self.s.model_mut(), lane),
            None => Ok(Pending::landed(self.snapshot()?)),
        }
    }

    /// The state put back ([`planes::resume`]) on the selected slot after its
    /// reset.
    fn resume(&mut self, state: &dyn Saved) -> Result<(), GateError> {
        planes::resume(self.s.model_mut(), state, WHOSE)
    }

    /// The first `n` positions the state can be cut to put back on the
    /// selected slot after its reset ([`planes::resume_prefix`]): the
    /// sequence's next call runs as it would have after the whole state's
    /// resume and a rollback to the returned position.
    fn resume_prefix(&mut self, state: &dyn Saved, n: usize) -> Result<usize, GateError> {
        planes::resume_prefix(self.s.model_mut(), state, n, WHOSE)
    }

    fn note(note: &CacheNote) {
        note_line(NAME, note);
    }
}

impl Tiered for MiniMax {
    type H = model::arch::minimaxm3::hparams::Hparams;
    type Flags = CacheFlag;

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

    fn flags(flag: &str, value: &str, out: &mut CacheFlag) -> Result<bool, GateError> {
        out.take(flag, value)
    }

    /// [`CacheFlag::budget`] over the plan's store bytes ([`seq_bytes`]): a default under one saved
    /// state (one position) is off. A slot's context is the state's most.
    fn cache_budget(
        flags: &CacheFlag,
        plan: &Plan<'_>,
        pool: u64,
        slot_ctx: usize,
        inputs: &PlanInputs<Self::H>,
    ) -> Result<u64, GateError> {
        flags.budget(plan, pool, slot_ctx, |n| {
            seq_bytes(&inputs.kv, inputs.model.layers, n)
        })
    }

    /// The tier's prompt-batch bytes of the file's widths and down forms over the body's prompt
    /// unit ([`bloomery_gpu_minimaxm3::tier::tier_batch`]), reserved on the tier card and the
    /// host; `None` on a placement with no tier card.
    fn tier_batch(
        place: Place,
        inputs: &PlanInputs<Self::H>,
    ) -> Result<Option<TierBatchBytes>, GateError> {
        Ok((!place.tier_cards().is_empty())
            .then(|| bloomery_gpu_minimaxm3::tier::tier_batch(inputs))
            .transpose()?)
    }

    /// The tags the model's description names for its think span.
    fn think(inputs: &PlanInputs<Self::H>) -> Result<Option<ThinkTags>, GateError> {
        match inputs.spec.chat.reasoning {
            Some(ReasoningFormat::ThinkSpan { open, close }) => Ok(Some(ThinkTags { open, close })),
            None => Err(format!("the {} description names no reasoning tags", Self::FAMILY).into()),
        }
    }

    fn register(args: &[String]) {
        record::at_main_with_flags(NAME, record::BLOOMERY_SERVE_MINIMAXM3, |flag| {
            bloomery_gpu_gates::llama_launch::serve_option(args, flag)
        });
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

/// The seat's levers (unless `read_levers` is false) and flags, checked before the model is fetched
/// ([`tiered::check`]).
pub fn check(args: &[String], read_levers: bool) -> Result<(), GateError> {
    tiered::check::<MiniMax>(args, read_levers).map(drop)
}
