//! The seat of a body that runs one chain on one card, fed in prompt batches:
//! the one owner of the open, the load record, the seat's calls and the think
//! tags, for the families whose body holds one sequence ([`solar2`](super::solar2),
//! [`minimaxm3`](super::minimaxm3)).
//!
//! A seat gives its family the facts the run needs — the architectures it
//! accepts, its plan inputs, its own flags, the words its refusals name and
//! the records it prints — and this module holds what two families already do
//! the same way: the load on the engine thread, the `load` record, the
//! prompt call, the step, the keep rule and the tags the model's own
//! description names.

use std::marker::PhantomData;
use std::sync::Arc;
use std::time::{Duration, Instant};

use app::arch::HybridCfg;
use app::{Keep, Loaded, OpenArgs, OpenLog, Prompt, Session, SessionError};
use bloomery_gpu::GpuModel;
use bloomery_gpu::model::{ChainBody, StepMode};
use bloomery_gpu_gates::GateError;
use bloomery_gpu_gates::beside::{self, Beside};
use bloomery_gpu_gates::bind::{Seat, note_line};
use bloomery_gpu_gates::generate::Lane;
use bloomery_gpu_gates::record::{self, Record};
use bloomery_gpu_gates::side_seat;
use gguf::Split;
use model::arch::Arch;
use model::arch::models::ReasoningFormat;
use model::arch::place_hybrid::PlanInputs;
use model::placement::{Machine, PlacementError, Plan};
use runtime::Target as _;
use serve::reasoning::ThinkTags;
use serve::{CacheNote, Saved, ServeError};

use super::tiered::{self, Open, Tiered};

/// What a hybrid seat's family gives the common owner: its body, its plan
/// inputs, the words its refusals and its records name, and the flags it
/// takes beside the common ones.
pub trait Family: Sized + 'static {
    /// The body the seat serves: one chain on one card, fed in prompt
    /// batches.
    type Body: ChainBody + app::Open<Inputs = PlanInputs<Self::H>, Cfg = HybridCfg> + Prompt + Keep;

    /// The flags this seat takes beside the common ones.
    type Flags: Default + Send;

    /// The family's hyperparameters: what its plan inputs are read for.
    type H;

    /// The binary's name, its records' owner and the `note` line's.
    const NAME: &'static str;
    /// The usage line the seat's own flags are named in.
    const USAGE: &'static str;
    /// The levers this seat acts on, parsed once at `main`.
    const ACTS_ON: &'static [&'static str];
    /// The alias a file that names none serves under.
    const ALIAS: &'static str;
    /// The seat word a foreign file's refusal names.
    const FAMILY: &'static str;
    /// The files the seat serves.
    const SERVES: &'static str;

    /// Whether `arch` is a file this seat serves.
    fn accepts(arch: Arch) -> bool;

    /// The file's plan inputs, its own reading.
    fn read(split: &Split) -> Result<PlanInputs<Self::H>, PlacementError>;

    /// One of the seat's own flags (`SeatFlags::parse`'s `own`): `Ok(true)`
    /// took it.
    fn flags(flag: &str, value: &str, out: &mut Self::Flags) -> Result<bool, GateError>;

    /// The records the seat prints, registered once at `main`.
    fn register();

    /// The seat's listening record, the one the run fills once the port is
    /// bound.
    fn listening() -> Record;

    /// The seat's load record: the family names its kind, the common owner
    /// fills the fields every seat's load prints.
    fn load_record() -> Record;

    /// The tags the model's description names for its think span, which the
    /// server splits the reasoning on ([`serve::Server::with_think_tags`]); the
    /// description of every file of the family names them.
    fn think_tags(inputs: &PlanInputs<Self::H>) -> Result<ThinkTags, GateError> {
        match inputs.spec.chat.reasoning {
            Some(ReasoningFormat::ThinkSpan { open, close }) => Ok(ThinkTags { open, close }),
            None => Err(format!("the {} description names no reasoning tags", Self::FAMILY).into()),
        }
    }
}

/// The seat of a hybrid family: the session and the positions its stores were
/// sized for.
pub struct Hybrid<F: Family> {
    s: Session<F::Body>,
    ctx: usize,
}

/// The open's records: the plan's expert counts are kept until the step is
/// captured, where the model's size and the step graph's nodes complete the
/// `load` record.
struct Log<F> {
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
    /// The family the log records for.
    _family: PhantomData<F>,
}

impl<F: Family> OpenLog<F::Body> for Log<F> {
    fn plan(
        &mut self,
        _place: &'static str,
        _inputs: &PlanInputs<F::H>,
        _machine: &Machine,
        plan: &Plan<'_>,
    ) -> Result<bool, SessionError> {
        self.host_experts = plan.host.experts;
        self.card_experts = plan.cards[0].experts;
        self.opened = self.t.elapsed();
        Ok(true)
    }

    fn load(&mut self, _m: &GpuModel<F::Body>) -> Result<(), SessionError> {
        Ok(())
    }

    fn capture(&mut self, nodes: usize) -> Result<(), SessionError> {
        self.graph_nodes = nodes;
        Ok(())
    }

    fn prompt_buffers(&mut self, _m: &GpuModel<F::Body>) -> Result<(), SessionError> {
        Ok(())
    }
}

impl<F: Family> Log<F> {
    /// The `load` record of `m`, the one slot ready.
    fn record(&self, m: &GpuModel<F::Body>) {
        if let Some((card, rate)) = &self.lane {
            record::lane(card, rate).eprint();
        }
        let total = self.t.elapsed();
        F::load_record()
            .w("arch", <F::Body as ChainBody>::arch().name())
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

/// The session of the file at `a.path` on `a.place`, the step captured, the
/// `load` record on stderr, on the calling thread, pinned to the dispatcher's
/// cpu slot when asked.
fn open_session<F: Family>(a: Open<F::Flags>) -> Result<Hybrid<F>, GateError> {
    record::require(record::groups::hybrid::GROUP, "hybrid seat");
    let pinned = a.pin_main && threads::pool().pin_caller();
    let file = Split::open(&a.path).map_err(|e| format!("open {}: {e}", a.path.display()))?;
    let mut log = Log::<F> {
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
        _family: PhantomData,
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
        cfg: HybridCfg {
            place: a.plan,
            host: a.host,
            residency: a.residency,
            group: a.group,
        },
    };
    let s = Loaded::<F::Body>::open(file, args, &mut log)?
        .ok_or("the open stopped at its plan")?
        .ready(&mut log)?;
    log.record(s.model());
    Ok(Hybrid { s, ctx: a.ctx })
}

impl<F: Family> Seat for Hybrid<F> {
    fn pos(&self) -> usize {
        self.s.model().pos() as usize
    }

    fn ctx_max(&self) -> usize {
        self.ctx
    }

    /// The prompt in batches (`app::Prompt for Body`), the argmax after the
    /// last.
    fn prefill(&mut self, ids: &[u32]) -> Result<u32, GateError> {
        Ok(<F::Body as Prompt>::prompt(self.s.model_mut(), ids)?)
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
        Err(format!("a snapshot: {}", no_cache::<F>()).into())
    }

    fn resume(&mut self, _state: &dyn Saved) -> Result<(), GateError> {
        Err(format!("a resume: {}", no_cache::<F>()).into())
    }

    fn note(note: &CacheNote) {
        note_line(F::NAME, note);
    }
}

/// Why the seat takes no sequence state: it runs no prompt cache.
fn no_cache<F: Family>() -> String {
    format!("the {} seat runs no prompt cache", F::FAMILY)
}

impl<F: Family> Tiered for Hybrid<F> {
    type H = F::H;
    type Flags = F::Flags;

    const NAME: &'static str = F::NAME;
    const USAGE: &'static str = F::USAGE;
    const ACTS_ON: &'static [&'static str] = F::ACTS_ON;
    const ALIAS: &'static str = F::ALIAS;
    const FAMILY: &'static str = F::FAMILY;
    const SERVES: &'static str = F::SERVES;
    const RESIDENT: bool = false;

    fn accepts(arch: Arch) -> bool {
        F::accepts(arch)
    }

    fn read(split: &Split) -> Result<PlanInputs<F::H>, PlacementError> {
        F::read(split)
    }

    fn flags(flag: &str, value: &str, out: &mut Self::Flags) -> Result<bool, GateError> {
        F::flags(flag, value, out)
    }

    fn think(inputs: &PlanInputs<Self::H>) -> Result<Option<ThinkTags>, GateError> {
        F::think_tags(inputs).map(Some)
    }

    fn register() {
        F::register();
    }

    fn listening() -> Record {
        F::listening()
    }

    fn open(a: Open<Self::Flags>) -> Result<Self, GateError> {
        open_session(a)
    }
}

/// Loads the model, and the embedding model `side` names beside it, and serves
/// until the listener or the engine fails; `Ok` carries why the server ended.
pub fn run<F: Family>(args: &[String], side: &side_seat::Flags) -> Result<ServeError, GateError> {
    tiered::run::<Hybrid<F>>(args, side)
}
