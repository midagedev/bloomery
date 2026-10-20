//! `bloomery-serve --model mimo2` — the llama-server-compatible HTTP API on
//! the MiMo-V2.6-Flash engine: a mimo2 file, its layers on one card, the
//! routed experts the plan puts on the card and the rest on the host tier
//! (`placement::plan_routed` through `PlanInputs::plan_with_slots`), opened as
//! `gate_mimo2_e2e` opens it (`app::Loaded`, in graph mode with the step
//! captured).
//!
//!     bloomery-serve --model mimo2 [-m PATH | --hf <repo>[:<quant>]]
//!                    [--host 127.0.0.1] [--port 8080] [--ctx C] [--alias NAME]
//!                    [--chat-template-file PATH] [--place W] [--plan]
//!                    [--prefill batch|steps] [--parallel N] [--queue-depth Q]
//!                    [--api-key KEY] [--api-key-file FNAME]
//!
//! The server takes `-m`/`--hf` out before this seat parses (its module
//! doc); `--ctx-size` and `-c` are `--ctx` under llama-server's spellings. The
//! model's first shard gives the vocabulary, `tokenizer.chat_template` the chat
//! template (`--chat-template-file` replaces it) and `general.name` the alias
//! (`--alias` replaces it; `mimo-v2.6-flash` for a file that names none).
//! Sampling is the sampler crate's chain with no repetition penalty,
//! `temperature <= 0` the engine's argmax. Any other flag is refused as an
//! unknown argument.
//!
//! Two facts of the architecture set this seat's shape, and nothing else
//! does: the program runs no verify pass of several rows, and no draft is
//! read. So the server feeds the prompt less its last id as one prompt call,
//! then steps the last, as `bloomery-serve --model qwen3` does; a round of
//! several slots is one pass of every busy slot's row (`GpuModel::step_slots`,
//! the body's `SlotRows`, through `serve_seats::rounds::step_rows_one_pass`),
//! the `parallel` line's `pass=one`; a prompt cache is refused by name, and
//! the load refuses an expert tier card.
//!
//! `--parallel N` serves N resident sequences (`app::Session::add_slots`):
//! the model loads one slot's stores and parks the other N − 1 after its step
//! capture, each slot capturing its own step on first use. `--parallel`
//! unset takes one slot; a count is never lowered. The slot count is read
//! before the context, which the slots split: each slot holds ⌊`--ctx` / N⌋
//! positions, and a split that leaves a slot no position is refused by name. The plan counts
//! every slot's stores (`PlanInputs::plan_with_slots`: N times a slot's), so
//! a total the card cannot hold is refused by the plan before any load, and
//! the load serves no more sequences than the plan counted.
//!
//! The prompt call is fed as `--prefill` says (`app::Prompt for Body`): in
//! batches of up to a union's columns (`batch`, the default), or one decode
//! step an id (`steps`, the same-binary arm); the two leave the same bits. A
//! word other than those is refused naming the flag and the word. The batch
//! feed's buffers come out of the card's free bytes at the load, refused by
//! name when they do not fit, and the group lever is not read: a group runs
//! one batch.
//!
//! `--place W` is the shared placement word (`generate::Place`): `a` (the
//! largest visible card), `gate`, or one card of the census by name or
//! ordinal. Unset, the common rule every serving seat takes decides
//! (`Place::choose_untiered`: the body serves no tier card), on one census
//! reading: a `place unset` record names the word `a` and why. A placement
//! that hangs an expert tier card (`bp`, a two-card list) is refused before
//! any load by the plan's own refusal (`PlaceError::Tier`), which names the
//! card and its tier; the tier serves no prompt batch here, so its batch
//! reserve is zero bytes.
//!
//! `--ctx` is the total the slots split (`--ctx 0` is refused); with one slot
//! it is one request's context. The context comes first and the plan fills
//! the machine around it (`generate::choose_ctx`, every seat's): a set share
//! past the file's trained context stops by name, and a share the machine
//! cannot hold stops with the largest it would hold. Unset, a slot asks
//! 262,144 positions or the trained context where that is less, halved while
//! the card cannot hold the stores and resident bytes, or while the host
//! cannot hold the routed experts the card then leaves it: the plan reads the
//! host room (`BLOOMERY_HOST_ROOM` replaces the reading) and the body serves
//! no NVMe expert tier (`PlaceError::NoNvmeTier`), so a shorter context, which
//! keeps more experts on the card, is what that refusal asks for.
//!
//! The records on stderr, in order: `place unset` (word and why), one `ctx`
//! line (`placement::ctx_ask::Chosen::line`: the rule — `set`, `target` or
//! `halved`, with the memory a halving was short of —, a slot's context, the
//! slots, the total and the model's limit), the `plan`
//! record (its `ctx_max` a slot's), the `parallel` line (`rule=slots
//! slots=N slot_ctx=C total=N·C from=<word> pass=one`); then the `load`
//! record (architecture, resident bytes of every slot, a slot's context,
//! slots, layers, the routed experts the plan keeps on the host and on the
//! card, the step graph's nodes, the prompt group and the load's wall) and, once the port is
//! bound, the `listening` record (`record::BLOOMERY_SERVE_MIMO2`). `--plan`
//! ends the process after the `parallel` line, nothing loaded. An engine error ends the process with
//! the crash block and exit code 70, as every seat's.
//!
//! A request keeps the longest prefix it shares with what the slot holds
//! (`app::Keep for Body`): the caches are per-position and the body takes its
//! rollback as given, so every held position is keepable and a resend that
//! diverges at `j` re-feeds from `j` and nothing before it. No prompt call is
//! cut (`Seat::splits` marks nowhere). The seat runs no prompt cache: a save
//! and a resume are refused by name.
//!
//! `BLOOMERY_RESIDENCY` is the common machine's lever: set, its word as given
//! (refused by name beside `--prefill steps`, and where the plan's card slots
//! leave it no room); unset, the serving word on every placement but `gate`,
//! which the plan then keeps, moves or turns `off` with why
//! ([`bloomery_levers::residency_unset`], [`residency::at_plan`]). Its
//! `residency lever` record (and, under `mid`, the `residency host` record of
//! the churn pool) comes before the plan's own. Under `mid` the experts the
//! card holds are the machine's to move by use, in the host leg's shadow.
//!
//! The levers this seat acts on ([`ACTS_ON`]) are parsed once, at `main`
//! (`bloomery_levers::at_main`), which refuses by name a lever set outside
//! them: the plan's card budget, the host set's load settings,
//! `BLOOMERY_PREFILL_GROUP` (the batches a prompt group runs layer by layer:
//! unset, the row's default, 2; the `load` record's `group` is what the body
//! holds), and `BLOOMERY_PIN_MAIN` with `BLOOMERY_HOST_LANES` (the engine
//! thread pinned to the dispatcher's cpu slot, which gives the host legs the
//! pool's CCD map). The plan reserves no card bytes for the prompt batch, so
//! the batch's units, a group of 2's included, come out of the card's margin
//! and are refused by name past its free bytes. The seat
//! sits behind the `deepseek41` feature beside `mimo2`: the server surface it
//! binds through (`bind`, the `serve` and `sampler` crates) is scoped to
//! `deepseek41`; it runs no V4.1 code.

use std::sync::Arc;
use std::time::{Duration, Instant};

use app::arch::mimo2::Mimo2Cfg;
use app::{Loaded, OpenArgs, OpenLog, Session, SessionError};
use bloomery_gpu::host::swap::Residency;
use bloomery_gpu::model::StepMode;
use bloomery_gpu_gates::beside::{self, Beside};
use bloomery_gpu_gates::bind::{Seat, SlotStep, note_line};
use bloomery_gpu_gates::generate::{Lane, Place};
use bloomery_gpu_gates::need;
use bloomery_gpu_gates::record::{self, Record};
use bloomery_gpu_gates::residency38::{churn_bytes, residency_room, residency_set};
use bloomery_gpu_gates::side_seat;
use bloomery_gpu_gates::{GateError, residency};
use bloomery_gpu_mimo2::{Body, Mimo2Model, PrefillMode};
use bloomery_levers::{ChurnHosting, Levers, ResidencyAt, ResidencyPick, ResidencyWhy};
use gguf::Split;
use model::arch::Arch;
use model::arch::mimo2::place::PlanInputs;
use model::placement::{Machine, PlacementError, Plan};
use runtime::Target as _;
use serve::{CacheNote, Saved, ServeError};

use super::rounds;
use super::tiered::{self, Tiered};

const NAME: &str = "bloomery-serve-mimo2";

const USAGE: &str = "usage: bloomery-serve --model mimo2 [-m PATH | --hf <repo>[:<quant>]] \
                     [--host H] [--port P] [--ctx C] [--alias NAME] [--chat-template-file PATH] \
                     [--place W] [--plan] [--prefill batch|steps] [--parallel N] \
                     [--queue-depth Q] [--api-key KEY] [--api-key-file FNAME]";

/// The levers this seat acts on: the plan's card budget and the host set's
/// load settings, which the open reads (`PlanLevers::from_levers`,
/// `Levers::host`), the residency machine's word, the batches a prompt group
/// runs, the engine thread's pin, the host legs' lane rule, the start
/// screen and how a structured-output request's fixed runs are fed.
pub const ACTS_ON: &[&str] = &[
    bloomery_levers::CARD_BUDGET,
    bloomery_levers::RESIDENCY,
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

/// Whether a round of several slots runs as one pass of the busy rows
/// (`GpuModel::step_slots`, the body's `SlotRows`): the `parallel` line's
/// `pass=` word and [`Seat::step_slots`] read this one value.
const ONE_PASS: bool = true;

/// Why the seat takes no sequence state: it runs no prompt cache.
const NO_CACHE: &str = "the mimo2 seat runs no prompt cache";

/// The flags this seat takes beside the common ones ([`SeatFlags`]).
struct Flags {
    /// `--prefill`: how a prompt call is fed.
    prefill: PrefillMode,
}

impl Default for Flags {
    fn default() -> Self {
        Flags {
            prefill: PrefillMode::Batch,
        }
    }
}

/// The MiMo session on the engine thread and the positions each slot's
/// stores were sized for.
struct Mimo {
    s: Session<Body>,
    ctx: usize,
}

/// The open's records: the plan's expert counts are kept until the slots are
/// parked, where the model's size and the step graph's nodes complete the
/// `load` record.
struct Log {
    t: Instant,
    /// The wall from `t` to the load, once the plan is made: the phases
    /// record's `open_s`.
    opened: Duration,
    ctx: usize,
    slots: usize,
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

    fn load(&mut self, _m: &Mimo2Model) -> Result<(), SessionError> {
        Ok(())
    }

    fn capture(&mut self, nodes: usize) -> Result<(), SessionError> {
        self.graph_nodes = nodes;
        Ok(())
    }

    fn prompt_buffers(&mut self, _m: &Mimo2Model) -> Result<(), SessionError> {
        Ok(())
    }
}

impl Log {
    /// The `load` record of `m` with every slot parked.
    fn record(&self, m: &Mimo2Model) -> Result<(), GateError> {
        if let Some((card, rate)) = &self.lane {
            record::lane(card, rate).eprint();
        }
        let total = self.t.elapsed();
        Record::new(&record::LOAD_MIMO2)
            .w("arch", "mimo2")
            .u("resident_bytes", m.resident_bytes())
            .u("scratch_bytes", m.body("load record")?.scratch_bytes())
            .u("ctx", self.ctx)
            .u("slots", self.slots)
            .u("layers", m.layers().len())
            .u("host_experts", self.host_experts)
            .u("card_experts", self.card_experts)
            .u("graph_nodes", self.graph_nodes)
            .u("group", bloomery_gpu_mimo2::prefill_group(m)?)
            .w("pin_main", if self.pin_main { "on" } else { "off" })
            .w("pinned", self.pinned)
            .f("load_s", total.as_secs_f64())
            .eprint();
        record::eprint_load_phases(m, total, self.opened);
        Ok(())
    }
}

impl Mimo {
    /// The session of the file at `a.path` on `a.place`, the step captured,
    /// `a.slots` resident sequences parked after it (each capturing on its
    /// first use), the `load` record on stderr, on the calling thread, pinned
    /// to the dispatcher's cpu slot when asked.
    fn open(a: tiered::Open<Flags>) -> Result<Mimo, GateError> {
        let pinned = a.pin_main && threads::pool().pin_caller();
        let file = Split::open(&a.path).map_err(|e| format!("open {}: {e}", a.path.display()))?;
        let mut log = Log {
            t: Instant::now(),
            opened: Duration::ZERO,
            ctx: a.ctx,
            slots: a.slots.get(),
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
            cfg: Mimo2Cfg {
                place: a.plan,
                host: a.host,
                prefill: a.flags.prefill,
                residency: a.residency,
                group: a.group,
                slots: a.slots,
            },
        };
        let mut s = Loaded::<Body>::open(file, args, &mut log)?
            .ok_or("the open stopped at its plan")?
            .ready(&mut log)?;
        if a.slots.get() > 1 {
            s.add_slots(a.slots.get())?;
        }
        log.record(s.model())?;
        Ok(Mimo { s, ctx: a.ctx })
    }
}

impl Seat for Mimo {
    fn pos(&self) -> usize {
        self.s.model().pos() as usize
    }

    fn ctx_max(&self) -> usize {
        self.ctx
    }

    /// The resident sequences the seat made at its open
    /// (`app::Session::slots`): `--parallel`.
    fn slots(&self) -> usize {
        self.s.slots()
    }

    /// The session's slot (`app::Session::select_slot`): the model exchanges
    /// its live sequence with the slot's parked state — pointer moves.
    fn select(&mut self, slot: usize) -> Result<(), GateError> {
        Ok(self.s.select_slot(slot)?)
    }

    /// The prompt as `--prefill` says (`app::Prompt for Body`), the argmax
    /// after the last.
    fn prefill(&mut self, ids: &[u32]) -> Result<u32, GateError> {
        Ok(<Body as app::Prompt>::prompt(self.s.model_mut(), ids)?)
    }

    fn step(&mut self, last: u32) -> Result<u32, GateError> {
        Ok(self.s.model_mut().step(&[last])?)
    }

    /// One round of several slots' steps ([`rounds::step_round`]): the busy
    /// rows as one pass of the body ([`rounds::step_rows_one_pass`]), laid in
    /// slot order, each row's answer and lent logits row its own.
    fn step_slots(&mut self, rows: &mut [SlotStep]) -> Result<(), String> {
        rounds::step_round(
            self,
            ONE_PASS,
            rows,
            |m, rows| rounds::step_rows_one_pass(&mut m.s, rows),
            |_| Ok(()),
        )
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

    /// Every held position (`app::Keep for Body`), the session's answer.
    fn keep(&self, n: usize) -> (usize, Option<String>) {
        self.s.keep_query(n)
    }

    /// Nowhere: the caches are per-position, so a cut never needs the calls
    /// on either side of it to be whole prompt calls of their own.
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

impl Tiered for Mimo {
    type H = model::arch::mimo2::hparams::Hparams;
    type Flags = Flags;

    const NAME: &'static str = NAME;
    const USAGE: &'static str = USAGE;
    const ACTS_ON: &'static [&'static str] = ACTS_ON;
    const KINDS: &'static [&'static record::Kind] = record::BLOOMERY_SERVE_MIMO2;
    const LISTENING: &'static record::Kind = &record::LISTENING_MIMO2;
    const ALIAS: &'static str = "mimo-v2.6-flash";
    const FAMILY: &'static str = "mimo2";
    const SERVES: &'static str = "mimo2";
    const RESIDENT: bool = true;

    fn accepts(arch: Arch) -> bool {
        matches!(arch, Arch::Mimo2)
    }

    fn read(split: &Split) -> Result<PlanInputs, PlacementError> {
        PlanInputs::read(split)
    }

    fn flags(flag: &str, value: &str, out: &mut Flags) -> Result<bool, GateError> {
        match flag {
            "--prefill" => {
                out.prefill = PrefillMode::from_name(value)
                    .ok_or_else(|| format!("--prefill is batch or steps, not {value}"))?;
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    /// `BLOOMERY_RESIDENCY`: set as given, else the serving word on a serving placement, which
    /// the plan keeps, moves or turns off. The steps feed ends a decode pass per prompt id, which
    /// the machine's rule counts.
    fn residency(
        levers: &Levers,
        flags: &Flags,
        place: Place,
        plan: &Plan<'_>,
    ) -> Result<Option<ResidencyPick>, GateError> {
        let pick = levers.residency_at(ResidencyAt {
            serving_place: place != Place::Gate,
            check_finite: false,
            route_trace: false,
            prefill_steps: flags.prefill == PrefillMode::Steps,
        });
        if pick.why == ResidencyWhy::Set
            && pick.word != "off"
            && flags.prefill == PrefillMode::Steps
        {
            return Err(format!(
                "BLOOMERY_RESIDENCY={} is refused beside --prefill steps (each prompt id would \
                 end a pass the rule counts)",
                pick.word
            )
            .into());
        }
        let pick = residency::at_plan(
            <Body as bloomery_gpu::model::ChainBody>::UNHOSTED_POOL,
            plan,
            0,
            pick,
        )?;
        Ok(Some(pick))
    }

    fn settle(plan: &Plan<'_>, pick: ResidencyPick) -> Result<Residency, GateError> {
        let residency = Residency::parse(pick.word)?;
        if pick.why == ResidencyWhy::Set {
            residency_room(plan, residency, pick.word)?;
        }
        residency_set(
            plan,
            0,
            residency,
            ChurnHosting::Hosted,
            pick.word,
            0,
            Record::eprint,
        )?;
        // The hosted churn pool is host bytes the load holds beside the plan.
        need::print(
            plan,
            churn_bytes(plan, 0, (residency, ChurnHosting::Hosted))?,
            0,
            true,
        )?;
        Ok(residency)
    }

    fn open(a: tiered::Open<Flags>) -> Result<Mimo, GateError> {
        Mimo::open(a)
    }
}

/// Loads the model, and the embedding model `side` names beside it, and serves until the listener
/// or the engine fails; `Ok` carries why the server ended.
pub fn run(args: &[String], side: &side_seat::Flags) -> Result<ServeError, GateError> {
    tiered::run::<Mimo>(args, side)
}
