//! `bloomery-serve --model musespark` — the llama-server-compatible HTTP API on
//! the Muse Spark engine: a muse-spark file, its layers on one card, the routed
//! experts the plan puts on the card — and on the expert tier card beside it
//! when two cards are in view — and the rest on the host tier
//! (`PlanInputs::plan_with_slots`), opened in graph mode with the step
//! captured.
//!
//!     bloomery-serve --model musespark [-m PATH | --hf <repo>[:<quant>]]
//!                    [--host 127.0.0.1] [--port 8080] [--ctx C] [--alias NAME]
//!                    [--chat-template-file PATH] [--place W] [--plan]
//!                    [--cache-ram MIB] [--queue-depth Q] [--slot-save-path DIR]
//!                    [--api-key KEY] [--api-key-file FNAME]
//!
//! The server takes `-m`/`--hf` out before this seat parses (its module
//! doc); `--ctx-size` and `-c` are `--ctx` under llama-server's spellings. The
//! model's first shard gives the vocabulary, `tokenizer.chat_template` the chat
//! template (`--chat-template-file` replaces it) and `general.name` the alias
//! (`--alias` replaces it; `muse-spark` for a file that names none). Sampling
//! is the sampler crate's chain with no repetition penalty, `temperature <= 0`
//! the engine's argmax. Any other flag is refused as an unknown argument.
//!
//! Five facts of the architecture set this seat's shape, and nothing else
//! does:
//!
//! - The body holds one sequence: `--parallel` above 1 is refused by name
//!   before any plan, and the plan counts one slot's stores.
//! - Every layer's store is a plane of positions, the 46 sliding layers' too
//!   (the body's head-64 flash has no ring entry), so a cut keeps every held
//!   position (`app::Keep for Body`) and the plan counts planes
//!   (`model::arch::musespark::place::KV_ROWS`).
//! - A saved prompt state is the planes' rows alone: no recurrent store and no
//!   checkpoint, so a state of `n` positions restores at any position up to
//!   `n`. `--cache-ram MIB` is the prompt cache's budget (`bind::CacheRam`:
//!   unset, the default; `0` turns it off): the lesser of `bind::CACHE_RAM_CAP`
//!   and half of what the host leaves past the plan's need and the residency's
//!   churn pool, a `cache` line on stderr printing it with each term. A state
//!   is every layer's K and V rows of the held positions
//!   (`bloomery_gpu::checkpoint::planes`), copied off the request path on a
//!   stream of its own into pinned staging that the budget pins whole at the
//!   open, begun when a request ends and waited for only when the state is
//!   read back; a pool with no room is the server's `NoRoom`, which makes room
//!   or saves on the request path. A request of another conversation takes
//!   the slot and the session's return puts its rows back (the first positions
//!   the server keeps, when it shares only a prefix) instead of feeding the
//!   prompt again. A state of another model, card or context is refused by
//!   name.
//! - The prompt runs as passes of the load's rows through every layer, the
//!   hosted layers' experts through the host tier's batch port
//!   (`app::Prompt for Body`, `GpuModel::prefill`): the `load` record's
//!   `prompt` word is `batch`.
//! - The routed experts the plan placed are the residency machine's to move
//!   (`BLOOMERY_RESIDENCY`, the common word: set as given, else the serving
//!   word on every placement but `gate`, which the plan keeps, moves or turns
//!   `off`; `GpuModel::open_placed_with`), so the prompt calls can stream the
//!   host experts their rule sends to the card.
//!
//! `BLOOMERY_XSTREAM` resolves by the rule `generate_qwen3moe` and the Qwen3.8
//! seat run (`shared/xstream_rule.rs`): set as given; unset `split` under
//! `--place a` with a residency (`admit` when the card has no room for the
//! ring), `admit` under `--place bp` (`split` set runs the ring beside the tier), `off`
//! under `gate` and beside `BLOOMERY_RESIDENCY=off`. Where the rule starts the
//! stream (`plans_ring`: `BLOOMERY_XSTREAM=split`, or unset on plan (a) with the
//! residency not set `off`) the plan keeps the ring's bytes out of the expert
//! budget (`PlanInputs::streaming`), so the ring takes none of the card's
//! margin: a residency the plan then turns `off` leaves those bytes unused, and
//! the `xstream=` line (the word, then why, after the capture lines) says so. A
//! prompt too narrow for the rule's least width streams nothing, and decode
//! runs as under `admit`.
//!
//! The chat surface is the recipient protocol: the generation prompt ends
//! `<|start|>assistant`, the model writes ` to=self<|message|>` reasoning
//! `<|eom|>` and then ` to=user<|message|>` the answer `<|eot|>`, so the
//! reasoning is the `to=self` span and no think tag exists. The seat names no
//! think tags (`Tiered::think` is `None`); the server splits the span by the
//! template's markup (`serve::museatem`). The turn ends at `<|eot|>`, an
//! end-of-generation id by name. The file carries no draft, and the seat
//! serves none.
//!
//! `--place W` is the shared placement word (`generate::Place`): `a` (the
//! largest visible card), `bp` (plan (b′): the A6000 the stage card, the 3090
//! an expert tier holding each layer's next ids), `gate`, or one card of the
//! census by name or ordinal. Unset, the common rule decides (`Place::choose`
//! by [`TIER_RULE`]), on one census reading: two cards in view run `bp` when
//! its plan puts at least the rule's break-even of experts on the tier card,
//! else `a`; a `place unset` record names the word and why. The stage card
//! holds the token embedding, which the body reads there (a placed load's
//! machine, [`Muse::machine`]).
//!
//! `--ctx` is the one slot's context (`--ctx 0` is refused). The context comes
//! first and the plan fills the machine around it (`generate::choose_ctx`,
//! every seat's): a set value past the file's trained context stops by name,
//! and one the machine cannot hold stops with the largest it would hold. Unset,
//! the slot asks the trained context, or the largest context under it whose
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
//! (`rule=slots slots=1 slot_ctx=C total=C from=<word> pass=one`); then the
//! `load` record (architecture, resident bytes, the slot's context, the slots,
//! layers, the routed experts the plan keeps on the host and on the card, the
//! step graph's nodes, how a prompt is fed and the load's wall), the `nvtier`
//! record (on a load whose plan paged routed experts: the arena's budget, the
//! paged bytes and its counters, again after each prompt call), the `xstream=`
//! line and, once the port is bound, the `listening` record
//! (`record::BLOOMERY_SERVE_MUSESPARK`). `--plan` ends the process after the
//! `parallel` line, nothing loaded. An engine error ends the process with the
//! crash block and exit code 70, as every seat's.
//!
//! The levers this seat acts on ([`ACTS_ON`]) are parsed once, at `main`
//! (`bloomery_levers::at_main`), which refuses by name a lever set outside
//! them: the plan's card budget, the host set's load settings,
//! `BLOOMERY_PIN_MAIN` with `BLOOMERY_HOST_LANES` (the engine thread pinned to
//! the dispatcher's cpu slot, which gives the host legs the pool's CCD map),
//! the residency word and the expert stream's (`BLOOMERY_RESIDENCY`,
//! `BLOOMERY_XSTREAM`), the start screen and the schema jump.
//! `BLOOMERY_PREFILL_GROUP` is parsed because the common run reads it for every
//! tiered seat; this body feeds no batches, so it changes nothing here.
//!
//! The seat sits behind the `deepseek41` feature: the server surface it binds
//! through (`bind`, the `serve` and `sampler` crates) is scoped to it; it runs
//! no V4.1 code, and its body is in the gpu crate, so it needs no feature of
//! its own.
//!
//! The run is the tiered seats' ([`super::tiered`]); the open is this module's.

use std::sync::Arc;
use std::time::Instant;

use app::{Prompt, Session};
use bloomery_gpu::arch::musespark::Body;
use bloomery_gpu::checkpoint::planes::{seq_bytes, seq_lane};
use bloomery_gpu::checkpoint::saved::SaveLane;
use bloomery_gpu::host::served::TierBody;
use bloomery_gpu::model::{ChainBody, StepMode};
use bloomery_gpu::{GpuError, GpuModel};
use bloomery_gpu_gates::GateError;
use bloomery_gpu_gates::beside::{self, Beside};
use bloomery_gpu_gates::bind::{Seat, note_line};
use bloomery_gpu_gates::generate::{Place, TierRule};
use bloomery_gpu_gates::record::{self, Record};
use bloomery_gpu_gates::side_seat;
use bloomery_levers::Levers;
use gguf::Split;
use model::arch::Arch;
use model::arch::musespark::hparams::Hparams;
use model::arch::musespark::place::{self, PlanInputs as MuseInputs};
use model::arch::place_hybrid::PlanInputs;
use model::placement::workstation::TierBatchBytes;
use model::placement::{Machine, PlacementError, Plan};
use runtime::Target as _;
use serve::{CacheNote, Pending, Saved, ServeError};

use super::planes;
use super::tiered::{self, CacheFlag, Open, Tiered};

use super::xstream_rule::{Stage, plans_ring, xstream_rule};

/// The binary's name, its records' owner and the `note` line's.
const NAME: &str = "bloomery-serve-musespark";

const USAGE: &str = "usage: bloomery-serve --model musespark [-m PATH | --hf <repo>[:<quant>]] \
                     [--host H] [--port P] [--ctx C] [--alias NAME] [--chat-template-file PATH] \
                     [--place W] [--plan] [--cache-ram MIB] [--queue-depth Q] \
                     [--slot-save-path DIR] [--api-key KEY] [--api-key-file FNAME]";

/// The levers this seat acts on: the plan's card budget and the host set's
/// load settings, which the open reads (`PlanLevers::from_levers`,
/// `Levers::host`), the residency machine's word and the expert stream's, the
/// engine thread's pin, the host legs' lane rule, the start screen and the
/// schema jump; and the prompt group, which the common run reads for every
/// tiered seat and this body does not use. The V4.1 host tier's r8 sidecar is
/// not among them: the body reads its routed experts from the source.
pub const ACTS_ON: &[&str] = &[
    bloomery_levers::CARD_BUDGET,
    bloomery_levers::RESIDENCY,
    bloomery_levers::XSTREAM,
    bloomery_levers::PREFILL_GROUP,
    bloomery_levers::HOST_POPULATE,
    bloomery_levers::HOST_LOCK,
    bloomery_levers::CARD_DONTNEED,
    bloomery_levers::PIN_MAIN,
    bloomery_levers::HOST_LANES,
    bloomery_levers::SCREEN,
    bloomery_levers::SCHEMAJUMP,
];

/// How the body feeds a prompt, the `load` record's `prompt` word: passes
/// through every layer.
const PROMPT: &str = "batch";

/// The family's input to the common unset rule (`Place::choose`): the one
/// expert tier card the body hangs, kept only when the plan puts at least this
/// many experts on it — the tier size whose host time saved covers the tier's
/// fixed cost a token (docs/cards/musespark-2card.card).
pub const TIER_RULE: TierRule = TierRule {
    tiers: bloomery_gpu::host::SERVED_TIERS,
    break_even: Some(845),
    basis: "docs/cards/musespark-2card.card",
};

/// What a refusal of another seat's state names ([`planes::resume`]).
const WHOSE: &str = "a musespark body's";

/// The seat: the session and the positions its stores were sized for.
struct Muse {
    s: Session<Body>,
    ctx: usize,
    /// The copy stream and pinned staging saved states are made on
    /// ([`Seat::snapshots_async`]); none with the cache off.
    lane: Option<SaveLane>,
    /// The `tier census` records' span start ([`record::host_tier::CensusSpan`]): each
    /// prompt prints the steps since the last one's, then its own.
    census: record::host_tier::CensusSpan,
}

impl Muse {
    /// The session of the file at `a.path` on `a.place`, the step captured,
    /// the `load` record on stderr, on the calling thread, pinned to the
    /// dispatcher's cpu slot when asked.
    ///
    /// The plan is [`MuseInputs::plan_with_slots`] at `a.ctx` and `a.slots`
    /// over the inputs the run planned (the stream's ring kept out of the
    /// expert budget when `a.ring`), the same call the run's probe made; the
    /// load runs the residency `a.residency` ([`GpuModel::open_placed_with`]:
    /// the plan's card segment of each layer's stacks resident, the rest on
    /// the host tier), and the expert stream is resolved last, after the
    /// capture, for a started ring takes the card's free bytes past its keep.
    fn open(a: Open<CacheFlag>) -> Result<Muse, GateError> {
        let pinned = a.pin_main && threads::pool().pin_caller();
        let t = Instant::now();
        let file = Split::open(&a.path).map_err(|e| format!("open {}: {e}", a.path.display()))?;
        let inputs = MuseInputs::read(&file)?;
        let inputs = if a.ring {
            Muse::with_ring(inputs)?
        } else {
            inputs
        };
        let mut machine = <Muse as Tiered>::machine(a.place, a.tier_batch, inputs.model.layers)?;
        beside::reserve_on(
            &mut machine,
            Beside {
                image: None,
                embedding: a.embedding,
            },
        );
        let ctx = u64::try_from(a.ctx)
            .map_err(|_| format!("a context of {} positions passes u64", a.ctx))?;
        let plan = inputs
            .plan_with_slots(&machine, ctx, &a.plan, a.slots)
            .map_err(|e| GpuError::plan("musespark seat", e))?;
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
        let mut model = GpuModel::<Body>::open_placed_with(file, &plan, 0, a.host, a.residency)?;
        model.set_mode(StepMode::Graph);
        let nodes = model.capture_step()?;
        let ctx_u32 = u32::try_from(plan.ctx_max)
            .map_err(|_| format!("the plan's ctx_max {} passes u32", plan.ctx_max))?;
        let mut s = Session::from_model(model, ctx_u32);
        // The lane before the expert stream: a started ring takes the card's free bytes past its
        // keep, so every buffer the open makes comes first.
        let lane = match a.cache_ram {
            0 => None,
            ram => Some(seq_lane(s.model_mut(), ram)?),
        };
        let (gpu, _, body) = s.model_mut().body_parts(NAME)?;
        let line = xstream_rule(gpu, body, a.xstream, Stage::of(a.place), a.residency)?;
        eprintln!("{line}");
        if let Some((card, rate)) = &a.lane {
            record::lane(card, rate).eprint();
        }
        let total = t.elapsed();
        Record::new(&record::LOAD_MUSESPARK)
            .w("arch", <Body as ChainBody>::arch().name())
            .u("resident_bytes", s.model().resident_bytes())
            .u("ctx", a.ctx)
            .u("slots", 1)
            .u("layers", s.model().layers().len())
            .u("host_experts", host_experts)
            .u("card_experts", card_experts)
            .u("graph_nodes", nodes)
            .w("prompt", PROMPT)
            .w("pin_main", if a.pin_main { "on" } else { "off" })
            .w("pinned", pinned)
            .f("load_s", total.as_secs_f64())
            .eprint();
        record::eprint_load_phases(s.model(), total, opened);
        let hybrid = s.model().body(NAME)?.placed().map(|p| p.hybrid());
        record::host_tier::eprint_nvtier(hybrid.and_then(|h| h.nvme_tier()));
        let mut census = record::host_tier::CensusSpan::default();
        census.mark("load", hybrid.and_then(record::census_of));
        Ok(Muse {
            s,
            ctx: a.ctx,
            lane,
            census,
        })
    }
}

impl Seat for Muse {
    fn pos(&self) -> usize {
        self.s.model().pos() as usize
    }

    fn ctx_max(&self) -> usize {
        self.ctx
    }

    /// The prompt as passes (`app::Prompt for Body`), the argmax after the last;
    /// the `tier census` records of the steps since the last prompt and of
    /// this one, then the `nvtier` record.
    fn prefill(&mut self, ids: &[u32]) -> Result<u32, GateError> {
        let placed = self.s.model().body(NAME)?.placed();
        self.census.steps(placed.map(|p| p.hybrid()));
        let next = <Body as Prompt>::prompt(self.s.model_mut(), ids)?;
        let placed = self.s.model().body(NAME)?.placed();
        self.census.prompt(placed.map(|p| p.hybrid()));
        Ok(next)
    }

    fn step(&mut self, last: u32) -> Result<u32, GateError> {
        Ok(self.s.model_mut().step(&[last])?)
    }

    fn logits_into(&self, row: &mut [f32]) -> Result<(), GateError> {
        Ok(self.s.model().logits_into(row)?)
    }

    /// The model's reset ([`GpuModel::reset`]): the live sequence only. The
    /// body never holds a verify's rows, so the session's own reset adds
    /// nothing to it.
    fn reset(&mut self) -> Result<(), GateError> {
        Ok(self.s.model_mut().reset()?)
    }

    /// The session's cut (`app::Keep`), which takes only a position
    /// [`Seat::keep`] granted; any other is refused by name.
    fn rollback(&mut self, pos: u32) -> Result<(), GateError> {
        Ok(self.s.cut(pos)?)
    }

    /// The model's rule (`app::Keep for Body`) and the sentence that kept
    /// less: every held position, the plane stores' rule.
    fn keep(&self, n: usize) -> (usize, Option<String>) {
        self.s.keep_query(n)
    }

    /// Nowhere: no prompt call is cut for a prefix a later request keeps.
    fn splits(&self, _first: usize, _end: usize, _marks: &[usize]) -> Vec<usize> {
        Vec::new()
    }

    fn snapshot_size(&mut self) -> Result<Option<u64>, GateError> {
        planes::snapshot_size(self.s.model_mut())
    }

    /// The sequence state ([`planes::snapshot`]), as the prompt cache holds it.
    fn snapshot(&mut self) -> Result<Arc<dyn Saved>, GateError> {
        planes::snapshot(self.s.model_mut())
    }

    /// Whether the saves copy on the lane: the seat holds one.
    fn snapshots_async(&self) -> bool {
        self.lane.is_some()
    }

    /// The state's copy begun on the lane ([`planes::snapshot_begin`]); a pool
    /// with no room is the server's `NoRoom`. Without a lane, the snapshot on
    /// the call.
    fn snapshot_begin(&mut self) -> Result<Pending, GateError> {
        match &self.lane {
            Some(lane) => planes::snapshot_begin(self.s.model_mut(), lane),
            None => Ok(Pending::landed(self.snapshot()?)),
        }
    }

    /// The state put back ([`planes::resume`]) after the model's reset.
    fn resume(&mut self, state: &dyn Saved) -> Result<(), GateError> {
        planes::resume(self.s.model_mut(), state, WHOSE)
    }

    /// The first `n` positions the state can be cut to put back after the
    /// model's reset ([`planes::resume_prefix`]): the sequence's next call
    /// runs as it would have after the whole state's resume and a rollback to
    /// the returned position.
    fn resume_prefix(&mut self, state: &dyn Saved, n: usize) -> Result<usize, GateError> {
        planes::resume_prefix(self.s.model_mut(), state, n, WHOSE)
    }

    fn note(note: &CacheNote) {
        note_line(NAME, note);
    }
}

impl Tiered for Muse {
    type H = Hparams;
    type Flags = CacheFlag;

    const NAME: &'static str = NAME;
    const USAGE: &'static str = USAGE;
    const ACTS_ON: &'static [&'static str] = ACTS_ON;
    const ALIAS: &'static str = "muse-spark";
    const FAMILY: &'static str = "musespark";
    const SERVES: &'static str = "muse-spark";
    /// The body holds one sequence and no resident slots: `--parallel` above 1
    /// is refused by name before any plan.
    const RESIDENT: bool = false;

    fn accepts(arch: Arch) -> bool {
        matches!(arch, Arch::MuseSpark)
    }

    /// The family's own reading, which refuses by name a file with a feature
    /// no program runs (`PlanInputs::read`).
    fn read(split: &Split) -> Result<PlanInputs<Self::H>, PlacementError> {
        MuseInputs::read(split)
    }

    fn flags(flag: &str, value: &str, out: &mut CacheFlag) -> Result<bool, GateError> {
        out.take(flag, value)
    }

    /// [`CacheFlag::budget`] over the plan's store bytes ([`seq_bytes`]): a default under one saved
    /// state (one position) is off.
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

    /// A placed load's machine: the body reads its token embedding from the
    /// Q4_K table on its card (`embed_rows_q4k`), so the plan puts the table on
    /// the stage card ([`Place::machine_placed`] on one card; on a placement
    /// with a tier card, [`Place::machine`] with its stage card held as the
    /// placed load holds it, [`place::stage_placed`]).
    fn machine(
        place: Place,
        tier_batch: Option<TierBatchBytes>,
        layers: usize,
    ) -> Result<Machine, GateError> {
        if place.tier_cards().is_empty() {
            return Ok(place.machine_placed()?(layers));
        }
        let mut machine = place.machine(None, tier_batch)?(layers);
        place::stage_placed(&mut machine)?;
        Ok(machine)
    }

    const TIER_RULE: TierRule = TIER_RULE;

    /// The tier's prompt-batch bytes of the file's widths
    /// ([`place::tier_batch`]), reserved on the tier card and the host; `None`
    /// on a placement with no tier card.
    fn tier_batch(
        place: Place,
        inputs: &PlanInputs<Self::H>,
    ) -> Result<Option<TierBatchBytes>, GateError> {
        Ok((!place.tier_cards().is_empty())
            .then(|| place::tier_batch(&inputs.hp))
            .transpose()?)
    }

    /// The ring is kept out of the expert budget where the common rule starts
    /// the stream at `split` ([`plans_ring`]).
    fn ring(levers: &Levers, place: Place) -> bool {
        plans_ring(
            levers.xstream(),
            Stage::of(place),
            levers.residency() == Some("off"),
        )
    }

    /// The inputs of a plan that keeps the stream's ring and its ring pass's
    /// buffers out of the expert budget ([`PlanInputs::streaming`] by
    /// [`place::stream_bytes`]).
    fn with_ring(inputs: PlanInputs<Self::H>) -> Result<PlanInputs<Self::H>, GateError> {
        let bytes = place::stream_bytes(&inputs.hp, &inputs.model)?;
        Ok(inputs.streaming(bytes))
    }

    fn register(args: &[String]) {
        record::at_main_with_flags(NAME, record::BLOOMERY_SERVE_MUSESPARK, |flag| {
            bloomery_gpu_gates::llama_launch::serve_option(args, flag)
        });
    }

    fn listening() -> Record {
        Record::new(&record::LISTENING_MUSESPARK)
    }

    fn open(a: Open<Self::Flags>) -> Result<Self, GateError> {
        Muse::open(a)
    }
}

/// Loads the model, and the embedding model `side` names beside it, and serves until the listener
/// or the engine fails; `Ok` carries why the server ended.
pub fn run(args: &[String], side: &side_seat::Flags) -> Result<ServeError, GateError> {
    tiered::run::<Muse>(args, side)
}

/// The seat's levers (unless `read_levers` is false) and flags, checked before the model is fetched
/// ([`tiered::check`]).
pub fn check(args: &[String], read_levers: bool) -> Result<(), GateError> {
    tiered::check::<Muse>(args, read_levers).map(drop)
}
