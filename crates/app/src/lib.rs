//! The session: one model on its card behind the runtime's traits.
//!
//! [`Session<B>`] owns a [`GpuModel<B>`] and nothing that repeats what the
//! model holds: the position is the model's. It implements
//! [`runtime::Target`] where the body has a prompt schedule ([`Prompt`]) and
//! a cut rule ([`Keep`]), and [`runtime::Verify`] where it also runs several
//! rows in one pass and takes positions back (gpumodel's [`Rows`] and
//! [`Rollback`]). A model without those capabilities has no `verify`, so a
//! draft on it does not compile.
//!
//! The session opens in two steps, each printing through the caller's
//! [`OpenLog`]: [`Loaded::open`] plans once, hands the plan to the log, and
//! loads by that same plan; [`Loaded::ready`] captures the one-token step and
//! makes the prompt call's buffers. Between the two a card draft attaches its
//! taps to the loaded model, since a capture made before them would not write
//! them. [`Session::with_draft`] then captures the verify passes, one a
//! width its draft may propose.
//!
//! What a model adds is its [`Open`], [`Prompt`] and [`Keep`], under
//! [`arch`]. A body loaded by its own constructor, with no placement plan,
//! becomes a session through [`Session::from_model`].
//!
//! One load serves several prompts: [`Session::clear`] returns the model to
//! the state right after its load, and [`Session::arms`] runs a list of arms
//! with the clear between them, as `llama-bench` keeps a model while the next
//! test's model parameters are the previous one's and clears the context's
//! memory before each repetition. The clear keeps the buffers and zeroes what
//! a later call could read before writing it; it never lifts a fault.

pub mod arch;
// `cuda-core` names the target's context here, and comes with the `dspark` feature.
#[cfg(feature = "dspark")]
pub mod dspark;
pub mod mtp;
pub mod warm;

use bloomery_gpu::best;
use bloomery_gpu::host::PassKind;
use bloomery_gpu::host::served::prompt_pass;
use bloomery_gpu::host::swap::ResetReport;
use bloomery_gpu::media::MediaBody;
use bloomery_gpu::model::{
    ChainBody, Rollback, Rows, SlotRows, Slots, SlotsOut, StepAhead, StepMode,
};
use bloomery_gpu::{Fault, GpuError, GpuModel};
use gguf::Split;
use model::placement::{Machine, Plan};
use runtime::prompt::CallEnd;
use runtime::seqstate::Kept;
use runtime::swaprule::{KeptRows, SwapRule};
use runtime::{Draft, Out, RowLogits, Speculative, Target, Verify, Want, Width, Widths, Window};

/// What a session call failed with. Its text is the failure's own: a card
/// error reads as the card's error.
#[derive(Debug)]
pub enum SessionError {
    /// The card raised its fault word, read at a step's readback or at a
    /// prompt call's group end or call end: the model refuses every call
    /// until [`Target::reset`]. A call on a model a fault poisoned is this
    /// too.
    Fault(GpuError),
    /// Any other failure of the card, or of the model's own checks.
    Gpu(GpuError),
    /// A call the session refuses: out of order, or out of what it holds.
    Refused(String),
    /// An error of the caller's own code run inside a session call (an
    /// instrument's).
    Caller(Box<dyn std::error::Error>),
}

impl From<GpuError> for SessionError {
    fn from(e: GpuError) -> SessionError {
        match e {
            e @ (GpuError::Fault { .. } | GpuError::Poisoned { .. }) => SessionError::Fault(e),
            e => SessionError::Gpu(e),
        }
    }
}

impl std::fmt::Display for SessionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SessionError::Fault(e) | SessionError::Gpu(e) => e.fmt(f),
            SessionError::Refused(s) => f.write_str(s),
            SessionError::Caller(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for SessionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SessionError::Fault(e) | SessionError::Gpu(e) => e.source(),
            SessionError::Refused(_) => None,
            SessionError::Caller(e) => e.source(),
        }
    }
}

/// An arm of [`Session::arms`] that failed: which of how many, and its
/// error. The arms before it ran; the ones after it did not.
#[derive(Debug)]
pub struct ArmFailed<E> {
    /// The arm's index in the list, from 0.
    pub arm: usize,
    pub arms: usize,
    pub error: E,
}

impl<E: std::fmt::Display> std::fmt::Display for ArmFailed<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "arm {} of {}: {}", self.arm, self.arms, self.error)
    }
}

/// Any displayable error, a caller's boxed one included; the text is the
/// arm's and then its error's own.
impl<E: std::fmt::Debug + std::fmt::Display> std::error::Error for ArmFailed<E> {}

/// Whether a model poisoned by `fault` may be cleared: refused by name when
/// it is, since a fault ends the load.
fn clearable(fault: Option<Fault>) -> Result<(), SessionError> {
    match fault {
        Some(fault) => Err(SessionError::Refused(format!(
            "the clear on a model a fault poisoned ({fault}): the fault ends the load"
        ))),
        None => Ok(()),
    }
}

/// [`Session::arms`] over any `s`: `run` of each arm in order, `clear` before
/// every arm after the first, and the first failure ends the list naming its
/// arm.
fn run_arms<S, A, E>(
    s: &mut S,
    arms: &[A],
    mut clear: impl FnMut(&mut S) -> Result<(), SessionError>,
    mut run: impl FnMut(&mut S, usize, &A) -> Result<(), E>,
) -> Result<(), ArmFailed<E>>
where
    E: From<SessionError>,
{
    let failed = |arm, error| ArmFailed {
        arm,
        arms: arms.len(),
        error,
    };
    for (i, arm) in arms.iter().enumerate() {
        if i > 0 {
            clear(s).map_err(|e| failed(i, E::from(e)))?;
        }
        run(s, i, arm).map_err(|e| failed(i, e))?;
    }
    Ok(())
}

/// A body that loads by a placement plan: the file's headers read once, the
/// plan made once from them, and gpumodel's constructor over that plan.
pub trait Open: ChainBody + Sized {
    /// The body's own load levers and schedules, parsed once by the binary.
    type Cfg;
    /// What the plan is made from, read from the file's headers.
    type Inputs;

    /// Read `file`'s headers.
    fn inputs(file: &Split) -> Result<Self::Inputs, GpuError>;

    /// Apply load settings that change planned buffers, before the plan borrows its inputs.
    /// Bodies with no configuration-dependent buffers keep the header-derived inputs.
    fn configure_inputs(_inputs: &mut Self::Inputs, _cfg: &Self::Cfg) -> Result<(), GpuError> {
        Ok(())
    }

    /// The layer count a placement's machine is laid out for.
    fn layer_count(inputs: &Self::Inputs) -> usize;

    /// The plan of `inputs` on `machine` at `ctx` positions under `cfg`.
    fn plan<'a>(
        inputs: &'a Self::Inputs,
        machine: &'a Machine,
        ctx: usize,
        cfg: &Self::Cfg,
    ) -> Result<Plan<'a>, GpuError>;

    /// The model resident by `plan`, which `inputs` made.
    fn open(
        file: Split,
        inputs: &Self::Inputs,
        plan: &Plan<'_>,
        cfg: &Self::Cfg,
    ) -> Result<GpuModel<Self>, GpuError>;

    /// Make the prompt call's buffers `cfg` asks for, before any prompt, so
    /// that a timed prompt allocates nothing; whether it made any.
    fn prepare(m: &mut GpuModel<Self>, cfg: &Self::Cfg) -> Result<bool, GpuError>;
}

/// A body's prompt schedule.
pub trait Prompt: ChainBody {
    /// Run `ids` from the model's position by the schedule the body loaded
    /// with; the argmax after the last. A call that fails stands the model
    /// where the body's own rule leaves it, and is not run again here.
    fn prompt(m: &mut GpuModel<Self>, ids: &[u32]) -> Result<u32, GpuError>;

    /// [`Prompt::prompt`] ending as `end` asks ([`CallEnd`]): the argmax
    /// after the last when the call runs the head, `None` when it runs none.
    /// A seat asks [`CallEnd::Close`] only of a body that folds the last id. The
    /// default is [`Prompt::prompt`], its head run whatever the end; a body
    /// whose prompt call can run without its head overrides it.
    fn prompt_call(
        m: &mut GpuModel<Self>,
        ids: &[u32],
        end: CallEnd,
    ) -> Result<Option<u32>, GpuError> {
        let _ = end;
        Self::prompt(m, ids).map(Some)
    }
}

/// Which cuts a body's caches keep. gpumodel's [`Rollback`] takes positions
/// back; which prefixes survive that is the body's own rule.
pub trait Keep: ChainBody {
    /// The longest prefix of at most `n` positions a cut keeps.
    fn keepable(m: &GpuModel<Self>, n: u32) -> u32;

    /// [`Keep::keepable`] with its reason; a body whose rule states none
    /// says `Why::Rule`.
    fn kept(m: &GpuModel<Self>, n: u32) -> Kept {
        Kept::rule(n, m.pos(), Self::keepable(m, n))
    }

    /// Take back the positions from `n` on; the session asks only for a cut
    /// [`Keep::keepable`] grants.
    fn cut(m: &mut GpuModel<Self>, n: u32) -> Result<(), GpuError>;
}

/// [`Keep::keepable`] of a body whose caches are per-position: every held
/// position up to `n`, as [`Rollback`] takes a position back without device
/// work.
pub fn keep_positions<B: Rollback>(m: &GpuModel<B>, n: u32) -> u32 {
    n.min(m.pos())
}

/// [`Keep::cut`] of the same body: back to empty at 0 — a reset; else the
/// model's rollback.
pub fn cut_positions<B: Rollback>(m: &mut GpuModel<B>, n: u32) -> Result<(), GpuError> {
    match n {
        0 => m.reset(),
        n => m.rollback(n),
    }
}

/// How a session opens: the placement, the context the plan is asked for,
/// the step mode and the body's own configuration.
pub struct OpenArgs<C, M = fn(usize) -> Machine> {
    /// The placement's name, as the plan and load records print it.
    pub place: &'static str,
    /// The placement's machine, by layer count.
    pub machine: M,
    pub ctx: usize,
    pub mode: StepMode,
    pub cfg: C,
}

/// What the caller prints, and checks, at each step of an open. Each is
/// called once, in this order.
pub trait OpenLog<B: Open> {
    /// The plan the model loads by, before anything is loaded; `false` ends
    /// the open there.
    fn plan(
        &mut self,
        place: &'static str,
        inputs: &B::Inputs,
        machine: &Machine,
        plan: &Plan<'_>,
    ) -> Result<bool, SessionError>;

    /// The host bytes the load holds beside the plan's own (a next-token
    /// layer's host experts), told before [`OpenLog::plan`]; a load with
    /// none never calls it.
    fn beside(&mut self, _bytes: u64) {}

    /// The model loaded by that plan, in its step mode, nothing captured.
    fn load(&mut self, m: &GpuModel<B>) -> Result<(), SessionError>;

    /// The one-token step's capture and its node count (graph mode).
    fn capture(&mut self, nodes: usize) -> Result<(), SessionError>;

    /// The prompt call's buffers, made ([`Open::prepare`]).
    fn prompt_buffers(&mut self, m: &GpuModel<B>) -> Result<(), SessionError>;
}

/// What the caller prints when a draft's verify pass is captured
/// ([`Session::with_draft`]).
pub trait RowsLog {
    /// The verify pass's capture of `rows` rows and its node count (graph
    /// mode).
    fn capture_rows(&mut self, rows: usize, nodes: usize) -> Result<(), SessionError>;
}

/// A model loaded and nothing captured: where a draft attaches its taps.
pub struct Loaded<B: Open> {
    model: GpuModel<B>,
    cfg: B::Cfg,
    ctx: u32,
}

impl<B: Open> Loaded<B> {
    /// Read `file`'s headers, plan once on `args`' placement and hand the
    /// plan to `log` (`false` stops here: `Ok(None)`), then load by that plan
    /// in `args`' step mode and hand `log` the model.
    pub fn open<M: Fn(usize) -> Machine>(
        file: Split,
        args: OpenArgs<B::Cfg, M>,
        log: &mut impl OpenLog<B>,
    ) -> Result<Option<Loaded<B>>, SessionError> {
        Loaded::open_edited(file, args, log, |_, _| Ok(()))
    }

    /// [`Loaded::open`] with `edit` applied to the plan before `log` sees
    /// it and the model loads by it: a caller that fixes part of the split
    /// itself (a diagnostic's hand-placed card set) keeps every other row
    /// the planner made.
    pub fn open_edited<M: Fn(usize) -> Machine>(
        file: Split,
        args: OpenArgs<B::Cfg, M>,
        log: &mut impl OpenLog<B>,
        edit: impl FnOnce(&B::Inputs, &mut Plan<'_>) -> Result<(), SessionError>,
    ) -> Result<Option<Loaded<B>>, SessionError> {
        let mut inputs = B::inputs(&file)?;
        B::configure_inputs(&mut inputs, &args.cfg)?;
        let machine = (args.machine)(B::layer_count(&inputs));
        let mut plan = B::plan(&inputs, &machine, args.ctx, &args.cfg)?;
        edit(&inputs, &mut plan)?;
        if !log.plan(args.place, &inputs, &machine, &plan)? {
            return Ok(None);
        }
        let ctx = u32::try_from(plan.ctx_max).map_err(|_| {
            SessionError::Refused(format!("the plan's ctx_max {} passes u32", plan.ctx_max))
        })?;
        let mut model = B::open(file, &inputs, &plan, &args.cfg)?;
        model.set_mode(args.mode);
        log.load(&model)?;
        Ok(Some(Loaded {
            model,
            cfg: args.cfg,
            ctx,
        }))
    }

    /// A model its body's own constructor loaded by a plan the caller made
    /// (a gate's hand-built plan), nothing captured, in `cfg`'s schedule,
    /// whose caches hold `ctx` positions: what [`Loaded::open`] returns.
    pub fn from_model(model: GpuModel<B>, cfg: B::Cfg, ctx: u32) -> Loaded<B> {
        Loaded { model, cfg, ctx }
    }

    /// The model, before any capture.
    pub fn model_mut(&mut self) -> &mut GpuModel<B> {
        &mut self.model
    }

    /// See [`Loaded::model_mut`].
    pub fn model(&self) -> &GpuModel<B> {
        &self.model
    }
}

impl<B: Open> Loaded<B> {
    /// The session: in graph mode the one-token step captured before any
    /// token, so no step pays for it, then the prompt call's buffers.
    pub fn ready(self, log: &mut impl OpenLog<B>) -> Result<Session<B>, SessionError> {
        let Loaded {
            mut model,
            cfg,
            ctx,
        } = self;
        if model.mode() == StepMode::Graph {
            let nodes = model.capture_step()?;
            log.capture(nodes)?;
        }
        if B::prepare(&mut model, &cfg)? {
            log.prompt_buffers(&model)?;
        }
        Ok(Session {
            model,
            ctx,
            slots: 1,
            rows: None,
            slots_waiting: None,
            logits: Vec::new(),
            pass: Vec::new(),
            best: BestRows::default(),
            round: Vec::new(),
            tapped: Vec::new(),
            cleared: None,
        })
    }
}

/// The rows of a verify waiting for their commit, and whether a row read
/// ([`RowLogits::row_logits`]) has fetched the pass's head into the
/// session's copy, and a best read ([`RowLogits::row_best`]) its rows' best.
#[derive(Clone, Copy, Debug)]
struct RowsInFlight {
    first: u32,
    m: usize,
    fetched: bool,
    best_fetched: bool,
}

/// A pass's row read as its best logits ([`GpuModel::pass_best_into`]) and
/// the row a pick reads in its place: the best logits at their ids, every
/// other id at -inf. Each buffer is grown once and reused.
#[derive(Default)]
struct BestRows {
    /// The pass's rows' words as the first best read of the pass fetched them.
    stage: Vec<u32>,
    /// The words of the row read last ([`best::ROW_WORDS`]).
    words: Vec<u32>,
    /// The row a pick reads.
    row: Vec<f32>,
    /// The ids `row` holds a logit at; every other id is -inf.
    ids: Vec<u32>,
}

impl BestRows {
    /// `words` as the row a pick reads, the last row's ids back at -inf first.
    /// Refused by name for a row holding a NaN, as a whole row's read is, and
    /// for an id past the vocabulary.
    fn row(&mut self, n_vocab: usize, r: usize, pos: u32) -> Result<&[f32], SessionError> {
        if let Some(i) = best::row_nan(&self.words) {
            return Err(nan_row(r, i as usize, pos));
        }
        if self.row.len() != n_vocab {
            self.row.clear();
            self.row.resize(n_vocab, f32::NEG_INFINITY);
            self.ids.clear();
        }
        for &id in &self.ids {
            self.row[id as usize] = f32::NEG_INFINITY;
        }
        self.ids.clear();
        for (id, v) in best::row_pairs(&self.words) {
            let at = self.row.get_mut(id as usize).ok_or_else(|| {
                SessionError::Refused(format!("row {r}'s best names id {id} past {n_vocab} ids"))
            })?;
            *at = v;
            self.ids.push(id);
        }
        Ok(&self.row)
    }
}

/// The refusal of row `r`, whose logit `i` is NaN after position `pos`.
fn nan_row(r: usize, i: usize, pos: u32) -> SessionError {
    SessionError::Refused(format!("row {r}'s logit {i} is NaN after position {pos}"))
}

/// One model on its card behind [`Target`] and [`Verify`].
pub struct Session<B: ChainBody> {
    model: GpuModel<B>,
    /// The positions the caches were sized for.
    ctx: u32,
    /// The slots the session's model serves: 1 until
    /// [`Session::add_slots`] grows it. What the clear goes by — a model
    /// whose body parked slots before [`Session::from_model`] took it is
    /// not the session's; grow slots through the session so its clear
    /// knows them.
    slots: usize,
    rows: Option<RowsInFlight>,
    /// The slots of a pass of several slots' verify rows waiting for their
    /// commit ([`Session::verify_slots`]).
    slots_waiting: Option<usize>,
    /// The last logits row read back ([`Want::Logits`],
    /// [`RowLogits::row_logits`]).
    logits: Vec<f32>,
    /// The pass's head as a row read fetched it ([`GpuModel::pass_row_into`]),
    /// grown once and reused.
    pass: Vec<f32>,
    /// The pass's rows as best reads take them ([`RowLogits::row_best`],
    /// [`Session::slots_best`]).
    best: BestRows,
    /// The requested rows' logits of the last round of several slots' steps
    /// ([`Session::step_slots_rounds`]), packed, grown to the widest round
    /// once and reused.
    round: Vec<f32>,
    /// The last tapped rows read back ([`Tapped::taps`]).
    tapped: Vec<f32>,
    /// The residency reset the last clear made, until taken.
    cleared: Option<ResetReport>,
}

impl<B: ChainBody> Session<B> {
    /// A session over `model` as its own constructor loaded and made it
    /// ready, for a body loaded without a placement plan (no [`Open`]),
    /// whose caches hold `ctx` positions.
    pub fn from_model(model: GpuModel<B>, ctx: u32) -> Session<B> {
        Session {
            model,
            ctx,
            slots: 1,
            rows: None,
            slots_waiting: None,
            logits: Vec::new(),
            pass: Vec::new(),
            best: BestRows::default(),
            round: Vec::new(),
            tapped: Vec::new(),
            cleared: None,
        }
    }

    /// Row `r` as a read left it in the session's row: refused by name when
    /// it holds a NaN, as a step's row is.
    fn read_row(&self, r: usize) -> Result<&[f32], SessionError> {
        if let Some(i) = self.logits.iter().position(|v| v.is_nan()) {
            return Err(nan_row(r, i, self.model.pos()));
        }
        Ok(&self.logits)
    }

    /// The clear: the model back to the state it had right after its load
    /// and its captures, so a prompt run after it gives the tokens, logits
    /// and counters that prompt gives in a fresh process ([`GpuModel::reset`]
    /// is the body's half, the verify in flight and the kept logits row the
    /// session's, and a residency machine goes back to its seed,
    /// [`Session::take_cleared`] holding the reset's report). The weights,
    /// the captured chains and the prompt call's
    /// buffers stay: that is the load. Refused on a model a fault poisoned:
    /// a fault ends the load, it is not cleared into the next prompt.
    ///
    /// Over several slots ([`Session::add_slots`]) the clear rewinds the
    /// selected slot's sequence alone and leaves the model-wide state the
    /// parked slots stand on where it is — a residency machine's map stays
    /// where use has taken it, so [`Session::take_cleared`] reports `None`
    /// there; with one slot it is exactly the clear above.
    pub fn clear(&mut self) -> Result<(), SessionError> {
        clearable(self.model.poisoned())?;
        let one = self.slots == 1;
        self.model.reset()?;
        self.rows = None;
        self.slots_waiting = None;
        self.logits.clear();
        self.tapped.clear();
        self.cleared = if one {
            self.model.residency_reset()?
        } else {
            None
        };
        Ok(())
    }

    /// The residency back to its seed ([`GpuModel::residency_reset`]), the
    /// sequence left where it stands: the explicit call a runner makes
    /// before a timed request, which no other call makes
    /// ([`Target::reset`] keeps the residency use has built). `None` on a
    /// model with no residency machine.
    pub fn residency_reset(&mut self) -> Result<Option<ResetReport>, SessionError> {
        self.idle("residency reset")?;
        Ok(self.model.residency_reset()?)
    }

    /// The residency to a new seed ([`GpuModel::residency_reseed`]),
    /// `order[i]` the ranking of the machine's `i`-th layer: what every later
    /// reset, [`Session::clear`]'s included, returns to. `None` on a model
    /// with no residency machine.
    pub fn residency_reseed(
        &mut self,
        order: &[&[u32]],
    ) -> Result<Option<ResetReport>, SessionError> {
        self.idle("residency reseed")?;
        Ok(self.model.residency_reseed(order)?)
    }

    /// The residency machine's rule ([`GpuModel::residency_rule`]); `None` on
    /// a model with no residency machine.
    pub fn residency_rule(&mut self) -> Option<&SwapRule> {
        self.model.residency_rule()
    }

    /// The report of the residency reset the last [`Session::clear`] made,
    /// once: `None` when it made none (no machine) or it was taken.
    pub fn take_cleared(&mut self) -> Option<ResetReport> {
        self.cleared.take()
    }

    /// Several arms after one load: `run` of each of `arms` in order, the
    /// first on the model as it stands and each later one after the
    /// [`Session::clear`]. The first failure ends the list, and names its
    /// arm: a fault is never cleared into the next arm.
    pub fn arms<A, E>(
        &mut self,
        arms: &[A],
        run: impl FnMut(&mut Self, usize, &A) -> Result<(), E>,
    ) -> Result<(), ArmFailed<E>>
    where
        E: From<SessionError>,
    {
        run_arms(self, arms, Session::clear, run)
    }

    /// The model, for what the traits do not carry (records, instruments).
    pub fn model(&self) -> &GpuModel<B> {
        &self.model
    }

    /// The session taken apart: the model as it stands. A caller that holds
    /// one model across several sessions moves it in and out with this.
    pub fn into_model(self) -> GpuModel<B> {
        self.model
    }

    /// See [`Session::model`].
    pub fn model_mut(&mut self) -> &mut GpuModel<B> {
        &mut self.model
    }

    /// One greedy step of `id` for a caller that feeds the answer back
    /// unless it stops there: [`GpuModel::step_ahead`], the answer's step
    /// launched when `ahead` and it lands inside the session's positions.
    pub fn step_ahead(&mut self, id: u32, ahead: bool) -> Result<(u32, StepAhead), SessionError> {
        self.idle("step")?;
        let within = if ahead { self.ctx } else { 0 };
        Ok(self.model.step_ahead(id, within)?)
    }

    /// Refused while a verify's rows, or a pass of several slots' verify
    /// rows, wait for their commit.
    fn idle(&self, what: &str) -> Result<(), SessionError> {
        match (self.rows, self.slots_waiting) {
            (Some(r), _) => Err(SessionError::Refused(format!(
                "{what}: the verify of {} rows from position {} waits for its commit",
                r.m, r.first
            ))),
            (None, Some(n)) => Err(SessionError::Refused(format!(
                "{what}: the pass of {n} slots' verify rows waits for its commit \
                 (Session::commit_slots)"
            ))),
            (None, None) => Ok(()),
        }
    }

    /// `argmax` as `want` asks: with the logits row read back and kept.
    fn read(&mut self, argmax: u32, want: Want) -> Result<Out<'_>, SessionError> {
        match want {
            Want::Argmax => Ok(Out::Argmax(argmax)),
            Want::Logits => {
                self.logits.resize(self.model.n_vocab()?, 0.0);
                self.model.logits_into(&mut self.logits)?;
                Ok(Out::Logits {
                    argmax,
                    row: &self.logits,
                })
            }
        }
    }
}

impl<B: Keep> Session<B> {
    /// The server's keep query ([`Keep::kept`]): the longest prefix of at most
    /// `n` positions a cut keeps, and the rule's sentence when it keeps less
    /// than the session holds of them.
    pub fn keep_query(&self, n: usize) -> (usize, Option<String>) {
        let held = self.model.pos() as usize;
        let k = B::kept(&self.model, u32::try_from(n).unwrap_or(u32::MAX));
        let at = k.at as usize;
        (at, (at < n.min(held)).then(|| k.to_string()))
    }
}

impl<B: MediaBody> Session<B> {
    /// [`Target::prompt`] of `ids` with `media` spliced in: one prompt call ([`prompt_pass`]: one
    /// residency pass) whose span positions take the span's rows in place of their token
    /// embeddings ([`MediaBody::prefill_media`]). A call the body refuses before it enters
    /// ([`MediaBody::media_refusal`]) is refused by name here.
    pub fn prompt_media(
        &mut self,
        ids: &[u32],
        media: &[B::Span<'_>],
        want: Want,
    ) -> Result<Out<'_>, SessionError> {
        self.idle("prompt")?;
        let _busy = self.model().prompt_busy();
        if let Some(why) = B::media_refusal(self.model(), media.len()) {
            return Err(SessionError::Refused(why));
        }
        let argmax = prompt_pass(self.model_mut(), |m| B::prefill_media(m, ids, media))?;
        self.read(argmax, want)
    }
}

impl<B: Slots> Session<B>
where
    B::Seq: 'static,
{
    /// The model serving `n` slots from now on ([`GpuModel::add_slots`]): the
    /// live sequence keeps its state, `n − slots()` new ones allocated and
    /// parked empty. Callable any time — it allocates and parks, touching
    /// nothing the live sequence stands on.
    pub fn add_slots(&mut self, n: usize) -> Result<(), SessionError> {
        self.model.add_slots(n)?;
        self.slots = self.model.slots();
        Ok(())
    }

    /// The slots the model serves: 1 until [`Session::add_slots`] grows it.
    #[must_use]
    pub fn slots(&self) -> usize {
        self.model.slots()
    }

    /// The slot every later call acts on: 0 until a
    /// [`Session::select_slot`] moves it.
    #[must_use]
    pub fn selected(&self) -> usize {
        self.model.selected()
    }

    /// Make `slot` the one every later call acts on
    /// ([`GpuModel::select_slot`]): the model exchanges its live sequence,
    /// position and capture cache with the slot's parked state — pointer
    /// moves, nothing allocated, copied or synchronized. Refused by name
    /// while a verify's rows wait for their commit: the rows were verified
    /// on the sequence that stands live and their commit's rollback lands
    /// there, so a select between would move them onto another slot's
    /// sequence — the commit comes first.
    ///
    /// The kept logits row and taps need no parking of their own: each is a
    /// readback of the call that just ran, handed back inside that call's
    /// borrow, and the next call overwrites it.
    pub fn select_slot(&mut self, slot: usize) -> Result<(), SessionError> {
        self.idle("select_slot")?;
        Ok(self.model.select_slot(slot)?)
    }
}

impl<B: SlotRows> Session<B>
where
    B::Seq: 'static,
{
    /// One pass of several slots' tokens ([`GpuModel::step_slots`]): each
    /// row's greedy next token, each slot's ids bit for bit its tokens
    /// stepped alone. The pass leaves slot 0 selected. Refused by name
    /// while a verify's rows wait for their commit: the pass selects, and
    /// a select between would move them onto another slot's sequence.
    pub fn step_slots(&mut self, rows: &[(usize, &[u32])]) -> Result<SlotsOut, SessionError> {
        self.idle("step_slots")?;
        Ok(self.model.step_slots(rows)?)
    }

    /// One pass of several slots' verify rows ([`GpuModel::verify_slots`]):
    /// each row's greedy next token, each slot standing past its rows and
    /// waiting for [`Session::commit_slots`], which every other call waits
    /// for (refused by name until then). The pass leaves slot 0 selected.
    pub fn verify_slots(&mut self, rows: &[(usize, &[u32])]) -> Result<SlotsOut, SessionError> {
        self.idle("verify_slots")?;
        let out = self.model.verify_slots(rows)?;
        self.slots_waiting = Some(rows.len());
        Ok(out)
    }

    /// Row `r` of the last pass of several slots, in the pass's row order
    /// ([`GpuModel::slots_row_into`]), into the session's row, grown once and
    /// reused: the row a sampled window draws from. `fetched` false on the
    /// first row asked of a pass, whose head a pass laid in one head then
    /// fetches into the session's copy, and true for the pass's later rows.
    /// Refused by name as that refuses, and for a row holding a NaN, as a
    /// step's row is.
    pub fn slots_row(&mut self, r: usize, fetched: bool) -> Result<&[f32], SessionError> {
        let n_vocab = self.model.n_vocab()?;
        self.logits.resize(n_vocab, 0.0);
        self.model
            .slots_row_into(r, &mut self.pass, fetched, &mut self.logits)?;
        self.read_row(r)
    }

    /// Row `r` of the last pass of several slots read as its best logits
    /// ([`GpuModel::slots_best_into`]): the row a sampled window whose pick
    /// needs only a row's best draws from, its [`best::BEST`] best logits in
    /// place and every other id at -inf. `fetched` as [`Session::slots_row`]
    /// takes it, for the pass's best reads. Refused as `slots_row` refuses.
    pub fn slots_best(&mut self, r: usize, fetched: bool) -> Result<&[f32], SessionError> {
        let n_vocab = self.model.n_vocab()?;
        self.best.words.resize(best::ROW_WORDS, 0);
        self.model
            .slots_best_into(r, &mut self.best.stage, fetched, &mut self.best.words)?;
        let pos = self.model.pos();
        self.best.row(n_vocab, r, pos)
    }

    /// Keep each slot's first `kept[i]` rows of the pass waiting for its
    /// commit, in the pass's order ([`GpuModel::commit_slots`]). Refused by
    /// name with no pass waiting; a count the model refuses leaves the pass
    /// waiting.
    pub fn commit_slots(&mut self, kept: &[usize]) -> Result<(), SessionError> {
        if self.slots_waiting.is_none() {
            return Err(SessionError::Refused(format!(
                "a commit of {} slots' rows with no pass of slots waiting",
                kept.len()
            )));
        }
        self.model.commit_slots(kept)?;
        self.slots_waiting = None;
        Ok(())
    }

    /// One round of several slots' drafted windows verified together, the
    /// one owner of a round of any draft's windows (the MTP draft's
    /// [`mtp::pass_slots`], the lookup's): `windows` — each its slot's rows,
    /// the token at its position then its proposal, already proposed — cut
    /// between windows into passes of at most [`SlotRows::MAX_ROWS`] rows and
    /// [`SlotRows::MAX_SLOTS`] sequences, the pass that holds slot 0 last
    /// (`round_passes`), each pass run as one ([`Session::verify_slots`]),
    /// each row bit for bit its step; then each window's ids taken row by
    /// row in position order — the argmax, or a drawing window's draw from
    /// that row's logits ([`Session::slots_row`], or [`Session::slots_best`]
    /// for a draw that reads only the row's best) — up to the first that is
    /// not the proposal's next (runtime's `Speculative::pass_picking`), those
    /// rows kept, `heard` told of them (the window's index, the ids taken and
    /// how many were kept) and the ids appended to its `out`, before the
    /// commit of the pass's kept rows together ([`Session::commit_slots`]).
    /// Returns each window's commit, in the windows' order, and the passes
    /// run. The round leaves slot 0 selected. Refused by name as
    /// `round_passes` refuses the windows' rows, for a pass that reads back
    /// fewer ids than its rows, and as `heard` refuses.
    pub fn verify_rounds(
        &mut self,
        windows: &mut [RoundWindow<'_>],
        mut heard: impl FnMut(usize, &[u32], usize) -> Result<(), SessionError>,
    ) -> Result<(Vec<runtime::Committed>, usize), SessionError> {
        let cuts = round_passes(
            windows.iter().map(|w| (w.slot, w.rows.len())),
            (B::MAX_ROWS, B::MAX_SLOTS),
        )?;
        let mut kept = vec![0; windows.len()];
        let mut taken = Vec::new();
        for cut in &cuts {
            let pass: Vec<(usize, &[u32])> = windows[cut.clone()]
                .iter()
                .map(|w| (w.slot, w.rows))
                .collect();
            let argmax = self.verify_slots(&pass)?.ids;
            let (mut fetched, mut best_fetched) = (false, false);
            let mut at = 0;
            for (i, (w, k)) in windows[cut.clone()]
                .iter_mut()
                .zip(&mut kept[cut.clone()])
                .enumerate()
            {
                let r = w.rows;
                let got = argmax.get(at..at + r.len()).ok_or_else(|| {
                    SessionError::Refused(format!(
                        "a round of slots' windows: a pass of {} rows read back {} ids",
                        at + r.len(),
                        argmax.len()
                    ))
                })?;
                let ids = match w.draw.as_mut() {
                    None => {
                        *k = runtime::accepted_rows(r, got);
                        got
                    }
                    Some(draw) => {
                        taken.clear();
                        for j in 0..r.len() {
                            let id = if draw.best {
                                let row = self.slots_best(at + j, best_fetched)?;
                                best_fetched = true;
                                (draw.take)(row)
                            } else {
                                let row = self.slots_row(at + j, fetched)?;
                                fetched = true;
                                (draw.take)(row)
                            };
                            taken.push(id);
                            if r.get(j + 1) != Some(&id) {
                                break;
                            }
                        }
                        *k = taken.len();
                        &taken[..]
                    }
                };
                heard(cut.start + i, ids, *k)?;
                w.out.extend_from_slice(&ids[..*k]);
                at += r.len();
            }
            self.commit_slots(&kept[cut.clone()])?;
        }
        let done = windows
            .iter()
            .zip(&kept)
            .map(|(w, &k)| runtime::Committed {
                pos: w.pos,
                kept: k,
                rows: w.rows.len(),
                proposed: w.rows.len() > 1,
            })
            .collect();
        Ok((done, cuts.len()))
    }

    /// One round of several slots' steps as passes of [`GpuModel::step_slots`]
    /// — the round a seat serves whose body runs several slots' rows as one
    /// pass: the busy `rows` (one a slot there, the server's rounds being one
    /// step a running slot) cut into passes bounded by [`SlotRows::MAX_ROWS`]
    /// rows and [`SlotRows::MAX_SLOTS`] sequences, run in `round_passes`'
    /// order (the pass that holds slot 0 last), each row's greedy next
    /// token in row order and each slot's
    /// position moved by its own. `logits` names one request per entry of
    /// `rows`; only a requested entry's rows come back, read through
    /// [`GpuModel::slots_row_into`] into the session's packed buffer, lent until
    /// the next call — never [`GpuModel::logits_into`], which a pass of
    /// several rows refuses by name; a NaN among them is a named error, as a
    /// step's row is. The round leaves slot 0 selected, as
    /// [`Session::step_slots`] does. Refused as that refuses the rows, as
    /// `round_passes` refuses them, by name for a pass that answers
    /// another count of ids than its rows, and while a verify's rows wait
    /// for their commit.
    pub fn step_slots_rounds(
        &mut self,
        rows: &[(usize, &[u32])],
        logits: &[bool],
    ) -> Result<SlotsRound<'_>, SessionError> {
        let cuts = round_passes(
            rows.iter().map(|(slot, t)| (*slot, t.len())),
            (B::MAX_ROWS, B::MAX_SLOTS),
        )?;
        if logits.len() != rows.len() {
            return Err(SessionError::Refused(format!(
                "round of {} slots has {} logits requests",
                rows.len(),
                logits.len()
            )));
        }
        let reads = logits.iter().any(|&want| want);
        let n_vocab = if reads { self.model.n_vocab()? } else { 0 };
        let total = rows
            .iter()
            .try_fold(0usize, |n, (_, ids)| n.checked_add(ids.len()))
            .ok_or_else(|| SessionError::Refused("round row count overflow".into()))?;
        // A round that requests no logits builds no offsets and reads no row.
        let mut offsets = Vec::with_capacity(if reads { total } else { 0 });
        let mut wanted = 0usize;
        if reads {
            for ((_, tokens), &want) in rows.iter().zip(logits) {
                for _ in *tokens {
                    offsets.push(want.then_some(wanted));
                    wanted += usize::from(want);
                }
            }
        }
        let floats = wanted
            .checked_mul(n_vocab)
            .ok_or_else(|| SessionError::Refused("round logits size overflow".into()))?;
        if self.round.len() < floats {
            self.round.resize(floats, 0.0);
        }
        let mut ids = vec![0; total];
        for cut in &cuts {
            let held: usize = rows[..cut.start].iter().map(|(_, t)| t.len()).sum();
            let chunk = &rows[cut.clone()];
            let n: usize = chunk.iter().map(|(_, t)| t.len()).sum();
            let out = self.step_slots(chunk)?;
            if out.ids.len() != n {
                return Err(SessionError::Refused(format!(
                    "a pass of {n} rows answered {} ids",
                    out.ids.len()
                )));
            }
            if reads {
                let mut fetched = false;
                let slots = chunk
                    .iter()
                    .flat_map(|(slot, toks)| std::iter::repeat_n(*slot, toks.len()));
                for (r, slot) in slots.enumerate() {
                    round_row(slot, offsets[held + r], n_vocab, &mut self.round, |row| {
                        self.model.slots_row_into(r, &mut self.pass, fetched, row)?;
                        fetched = true;
                        Ok(())
                    })?;
                }
            }
            ids[held..held + n].copy_from_slice(&out.ids);
        }
        Ok(SlotsRound {
            ids,
            passes: cuts.len(),
            logits: (wanted > 0).then(|| RoundLogits {
                all: &self.round[..floats],
                offsets,
                n_vocab,
            }),
        })
    }
}

/// One busy slot's window of a round verified together
/// ([`Session::verify_rounds`]): the slot, the position its rows run from,
/// its rows (the token at that position, then the proposal: one row is a
/// plain step), where its kept ids are appended, and for a request that
/// draws, its draw. `None` takes each row's argmax.
pub struct RoundWindow<'a> {
    pub slot: usize,
    pub pos: u32,
    pub rows: &'a [u32],
    pub out: &'a mut Vec<u32>,
    pub draw: Option<Draw<'a>>,
}

/// A drawing window's draw ([`RoundWindow::draw`]): the id it takes from a
/// row's logits, one call a taken id (`runtime::Sample`'s closure), and
/// whether it reads only the row's best ([`Session::slots_best`]) in place
/// of the whole row ([`Session::slots_row`]).
pub struct Draw<'a> {
    pub take: &'a mut dyn FnMut(&[f32]) -> u32,
    pub best: bool,
}

/// Read and validate a requested row; an unrequested row touches no logits.
fn round_row(
    slot: usize,
    offset: Option<usize>,
    n_vocab: usize,
    round: &mut [f32],
    read: impl FnOnce(&mut [f32]) -> Result<(), GpuError>,
) -> Result<(), SessionError> {
    let Some(offset) = offset else { return Ok(()) };
    let row = &mut round[offset * n_vocab..(offset + 1) * n_vocab];
    read(row)?;
    if let Some(at) = row.iter().position(|v| v.is_nan()) {
        return Err(SessionError::Refused(format!(
            "slot {slot}'s logit {at} is NaN"
        )));
    }
    Ok(())
}

/// The passes a round of several slots runs and their order: the round's
/// entries `rows` (each a slot and its row count, in the order the caller
/// lays them) cut by [`runtime::round_passes`] between entries into passes
/// within the body's `limits` (its rows and sequences a pass), each a range
/// of entries, the pass that holds slot 0 last. A pass of several
/// slots writes the pass arena the live sequence (slot 0's, the homes made
/// canonical) reads its hidden rows from, whether the pass holds slot 0 or
/// not; a body copies a parked slot's rows out to its own buffer (body38's
/// `RowsCopy`), never slot 0's. So slot 0's rows of the round stand for its
/// draft's next walk only when no pass runs after its own. Refused by name
/// as that refuses the limits or an entry.
pub(crate) fn round_passes(
    rows: impl Iterator<Item = (usize, usize)> + Clone,
    limits: (usize, usize),
) -> Result<Vec<std::ops::Range<usize>>, SessionError> {
    let mut cuts = Vec::new();
    runtime::round_passes(rows, limits, &mut cuts)
        .map_err(|e| SessionError::Refused(e.to_string()))?;
    Ok(cuts)
}

/// What [`Session::step_slots_rounds`] ran: each row's greedy next token in
/// row order, the passes the round took (one a cut at the body's
/// row and sequence bounds), and the requested rows' logits in the round's row order, the
/// session's own buffer, lent until its next call.
#[derive(Clone, Debug, PartialEq)]
pub struct SlotsRound<'a> {
    pub ids: Vec<u32>,
    pub passes: usize,
    pub logits: Option<RoundLogits<'a>>,
}

/// A round's requested logits rows, packed in the session's buffer.
#[derive(Clone, Debug, PartialEq)]
pub struct RoundLogits<'a> {
    all: &'a [f32],
    offsets: Vec<Option<usize>>,
    n_vocab: usize,
}

impl<'a> RoundLogits<'a> {
    /// One entry per round row: its requested logits, or `None` for a greedy row.
    pub fn rows(&self) -> impl ExactSizeIterator<Item = Option<&'a [f32]>> + '_ {
        self.offsets
            .iter()
            .map(|offset| offset.map(|i| &self.all[i * self.n_vocab..(i + 1) * self.n_vocab]))
    }
}

impl<B: Prompt> Session<B> {
    /// The body's prompt call of `ids` ending as `end` asks
    /// ([`Prompt::prompt_call`]), under the session's idle check: the argmax
    /// after the last when it runs the head.
    pub fn prompt_call(&mut self, ids: &[u32], end: CallEnd) -> Result<Option<u32>, SessionError> {
        self.idle("prompt")?;
        let _busy = self.model.prompt_busy();
        Ok(B::prompt_call(&mut self.model, ids, end)?)
    }
}

impl<B: Prompt + Keep> Target for Session<B> {
    type Error = SessionError;

    fn pos(&self) -> u32 {
        self.model.pos()
    }

    fn ctx(&self) -> u32 {
        self.ctx
    }

    fn prompt(&mut self, ids: &[u32], want: Want) -> Result<Out<'_>, SessionError> {
        self.idle("prompt")?;
        let _busy = self.model.prompt_busy();
        let argmax = B::prompt(&mut self.model, ids)?;
        self.read(argmax, want)
    }

    fn step(&mut self, id: u32, want: Want) -> Result<Out<'_>, SessionError> {
        self.idle("step")?;
        let argmax = self.model.step(&[id])?;
        self.read(argmax, want)
    }

    fn keepable(&self, n: u32) -> u32 {
        B::keepable(&self.model, n)
    }

    fn kept(&self, n: u32) -> Kept {
        B::kept(&self.model, n)
    }

    fn cut(&mut self, n: u32) -> Result<(), SessionError> {
        self.idle("cut")?;
        let kept = B::kept(&self.model, n);
        if kept.at != n {
            return Err(SessionError::Refused(format!(
                "a cut to position {n} from {}: the caches keep {} ({kept})",
                self.model.pos(),
                kept.at
            )));
        }
        Ok(B::cut(&mut self.model, n)?)
    }

    fn reset(&mut self) -> Result<(), SessionError> {
        self.rows = None;
        self.slots_waiting = None;
        Ok(self.model.reset()?)
    }
}

impl<B: Prompt + Keep + Rows + Rollback> Verify for Session<B> {
    const MAX_ROWS: usize = B::MAX_ROWS;

    fn verify<const M: usize>(&mut self, rows: [u32; M]) -> Result<[u32; M], SessionError> {
        self.idle("verify")?;
        let first = self.model.pos();
        let out = self.model.step_rows::<M>(rows)?;
        self.rows = Some(RowsInFlight {
            first,
            m: M,
            fetched: false,
            best_fetched: false,
        });
        Ok(out)
    }

    fn commit(&mut self, accepted: usize) -> Result<(), SessionError> {
        let r = self.rows.ok_or_else(|| {
            SessionError::Refused(format!("a commit of {accepted} rows with no verify"))
        })?;
        if !(1..=r.m).contains(&accepted) {
            return Err(SessionError::Refused(format!(
                "a commit of {accepted} rows of a verify of {}",
                r.m
            )));
        }
        // Every commit rolls back, a whole accepted verify too: a body that
        // holds the verify pending (Qwen3.8's lane word) settles its state
        // only here, and a rollback to the position a body stands at takes
        // nothing back.
        let back = r.first
            + u32::try_from(accepted)
                .map_err(|_| SessionError::Refused(format!("{accepted} accepted rows pass u32")))?;
        // The rows stay in flight until the rollback and the keep have both
        // taken: a failed one leaves the verify's call open, so the session
        // still refuses every other call by name, and the way out is this
        // commit again (a rollback to the position the body stands at takes
        // nothing back) or `reset`, which abandons the pass at 0 rows kept.
        self.model.rollback(back)?;
        self.model
            .keep_rows(KeptRows::prefix(accepted), PassKind::Pair)?;
        self.rows = None;
        Ok(())
    }
}

impl<B: Prompt + Keep + Rows + Rollback> RowLogits for Session<B> {
    /// Row 0 of the last step or prompt call ([`GpuModel::logits_into`]), or
    /// row `r` of the verify waiting for its commit
    /// ([`GpuModel::pass_row_into`]: a pass laid in one head is fetched at
    /// the first row asked, the rows after it read from the session's copy),
    /// into the session's row, grown once and reused. Refused by name for a
    /// row the call did not run, for row 0 while a pass of several slots
    /// waits, and for a row holding a NaN, as a step's row is.
    fn row_logits(&mut self, r: usize) -> Result<&[f32], SessionError> {
        let n_vocab = self.model.n_vocab()?;
        self.logits.resize(n_vocab, 0.0);
        match &mut self.rows {
            None if r == 0 => {
                self.idle("row_logits")?;
                self.model.logits_into(&mut self.logits)?;
            }
            None => {
                return Err(SessionError::Refused(format!(
                    "row {r} read back with no verify waiting"
                )));
            }
            Some(f) => {
                self.model
                    .pass_row_into(f.m, r, &mut self.pass, f.fetched, &mut self.logits)?;
                f.fetched = true;
            }
        }
        self.read_row(r)
    }

    /// Row `r` of the verify waiting for its commit read as its best logits
    /// ([`GpuModel::pass_best_into`]: the pass's rows' best read at the first
    /// row asked), the row's [`best::BEST`] best logits in place and every
    /// other id at -inf; a step's row 0 is read whole
    /// ([`RowLogits::row_logits`]). Refused as `row_logits` refuses.
    fn row_best(&mut self, r: usize) -> Result<&[f32], SessionError> {
        let Some(f) = &mut self.rows else {
            return self.row_logits(r);
        };
        let n_vocab = self.model.n_vocab()?;
        self.best.words.resize(best::ROW_WORDS, 0);
        self.model.pass_best_into(
            f.m,
            r,
            &mut self.best.stage,
            f.best_fetched,
            &mut self.best.words,
        )?;
        f.best_fetched = true;
        let pos = self.model.pos();
        self.best.row(n_vocab, r, pos)
    }
}

/// The capture of a verify pass of each width, `log` told of each.
struct CaptureRows<'a, B: ChainBody, L> {
    model: &'a mut GpuModel<B>,
    log: &'a mut L,
}

impl<B: Rows, L: RowsLog> Width for CaptureRows<'_, B, L> {
    type Error = SessionError;

    fn run<const R: usize>(&mut self) -> Result<(), SessionError> {
        self.model.capture_rows::<R>()?;
        let nodes = self.model.rows_graph_nodes::<R>()?.len();
        self.log.capture_rows(R, nodes)
    }
}

impl<B: Prompt + Keep + Rows + Rollback> Session<B> {
    /// Drive `draft` on this session: in graph mode a verify pass of every
    /// width the draft may propose, 2 to `M` = [`Draft::WIDTH`] + 1 rows,
    /// captured now with the heads it needs, so no pass pays for them, and
    /// `log` told of each; in eager mode nothing is captured and the first
    /// pass of each width makes its heads. A draft that reads the target's
    /// hidden rows has attached its taps on the [`Loaded`] model already. `M`
    /// past the body's [`Rows::MAX_ROWS`] or a pass's rows, or not the
    /// draft's width + 1, does not compile.
    pub fn with_draft<D: Draft<Self>, const M: usize>(
        &mut self,
        draft: D,
        log: &mut impl RowsLog,
    ) -> Result<Speculative<D, M>, SessionError>
    where
        Window<M>: Widths,
    {
        let () = D::FITS;
        const {
            assert!(
                M == D::WIDTH + 1,
                "a draft's verify runs its WIDTH + 1 rows"
            );
        }
        self.idle("with_draft")?;
        if self.model.mode() == StepMode::Graph {
            Window::<M>::each(&mut CaptureRows {
                model: &mut self.model,
                log,
            })?;
        }
        Ok(Speculative::new(draft))
    }
}

#[cfg(test)]
mod tests {
    use super::{ArmFailed, BestRows, SessionError, clearable, round_passes, run_arms};
    use bloomery_gpu::{Fault, FaultSite, GpuError, best};

    /// A best read's row holds its own list alone: the row read before it is
    /// back at -inf, and a list naming an id past the vocabulary or a row
    /// holding a NaN is refused by name.
    #[test]
    fn a_best_row_holds_its_own_list() {
        let words = |pairs: &[(u32, f32)], nan: u32| {
            let mut w = vec![u32::MAX; best::ROW_WORDS];
            for (k, &(id, v)) in pairs.iter().enumerate() {
                (w[k], w[best::BEST + k]) = (v.to_bits(), id);
            }
            w[2 * best::BEST] = nan;
            w
        };
        let n = f32::NEG_INFINITY;
        let mut b = BestRows {
            words: words(&[(3, 2.0), (1, 1.0)], u32::MAX),
            ..BestRows::default()
        };
        assert_eq!(b.row(5, 0, 7).unwrap(), [n, 1.0, n, 2.0, n]);
        b.words = words(&[(0, 0.5)], u32::MAX);
        assert_eq!(b.row(5, 1, 7).unwrap(), [0.5, n, n, n, n]);
        for (w, why) in [
            (words(&[(5, 1.0)], u32::MAX), "past 5 ids"),
            (words(&[(0, 1.0)], 4), "is NaN"),
        ] {
            b.words = w;
            let e = b.row(5, 2, 7).unwrap_err();
            assert!(
                matches!(&e, SessionError::Refused(m) if m.contains(why)),
                "{e:?}"
            );
        }
    }

    /// The slot each row a draft walks after a round came from, a list per
    /// entry of `entries` (a slot and its rows), with the round cut within
    /// `limits` by `round_passes` and run on a model of body38's pass arena:
    /// every pass lays its rows into the live arena from row 0, slot 0's
    /// draft reads its rows there, and each parked slot's rows are copied
    /// to its own buffer.
    fn walked(entries: &[(usize, usize)], limits: (usize, usize)) -> Vec<Vec<usize>> {
        let cuts = round_passes(entries.iter().copied(), limits).expect("a round's passes");
        let mut live = Vec::new();
        let mut own = vec![Vec::new(); entries.len()];
        for cut in cuts {
            live = entries[cut.clone()]
                .iter()
                .flat_map(|&(slot, n)| std::iter::repeat_n(slot, n))
                .collect();
            for i in cut {
                own[i] = vec![entries[i].0; entries[i].1];
            }
        }
        entries
            .iter()
            .zip(own)
            .map(|(&(slot, n), own)| if slot == 0 { live[..n].to_vec() } else { own })
            .collect()
    }

    /// Each draft walks its own slot's hidden rows after a round of several
    /// passes: nine sampled slots' plain rows at eight a pass, three drafted
    /// windows of four rows, a round that fits one pass or holds no slot 0,
    /// and four slots' plain rows and two-row windows at GLM's four rows and
    /// two sequences a pass. A refused entry comes back as the session's
    /// refusal, naming its slot.
    #[test]
    fn round_passes_keep_each_drafts_hidden_rows() {
        // PIN(2026-10-24): the cut shapes, both pass bounds and every
        // refusal are `runtime::round_passes`'s own test.
        let nine: Vec<(usize, usize)> = (0..9).map(|slot| (slot, 1)).collect();
        let plain: Vec<(usize, usize)> = (0..4).map(|slot| (slot, 1)).collect();
        let pairs: Vec<(usize, usize)> = (0..4).map(|slot| (slot, 2)).collect();
        let three = [(0, 4), (1, 4), (2, 4)];
        let tail = [(1, 4), (2, 4), (3, 4)];
        for (entries, limits) in [
            (&nine[..], (8, 8)),
            (&three[..], (8, 8)),
            (&tail[..], (8, 8)),
            (&[(0, 1), (1, 1)][..], (8, 8)),
            (&plain[..], (4, 2)),
            (&pairs[..], (4, 2)),
        ] {
            for (&(slot, _), rows) in entries.iter().zip(walked(entries, limits)) {
                assert!(
                    rows.iter().all(|&from| from == slot),
                    "slot {slot}'s draft walks rows of slots {rows:?} after {entries:?}"
                );
            }
        }
        let Err(SessionError::Refused(text)) = round_passes([(0, 1), (1, 9)].into_iter(), (8, 8))
        else {
            panic!("a window past the pass was not refused");
        };
        assert!(text.contains("slot 1's"), "{text}");
    }

    #[test]
    fn round_reads_only_requested_logits() {
        let mut row = [f32::NAN; 3];
        super::round_row(0, None, 3, &mut row, |_| panic!("greedy logits read"))
            .expect("greedy rows do not read or validate logits");
        super::round_row(1, Some(0), 3, &mut row, |dst| {
            dst.copy_from_slice(&[1.0, 2.0, 3.0]);
            Ok(())
        })
        .expect("sampled row read");
        let logits = super::RoundLogits {
            all: &row,
            offsets: vec![None, Some(0)],
            n_vocab: 3,
        };
        assert_eq!(logits.rows().collect::<Vec<_>>(), [None, Some(&row[..])]);
        let error = super::round_row(1, Some(0), 3, &mut row, |dst| {
            dst[2] = f32::NAN;
            Ok(())
        })
        .unwrap_err();
        assert!(
            error.to_string().contains("slot 1's logit 2 is NaN"),
            "{error}"
        );
    }

    /// A poisoned model is refused by name, naming its fault; a clean one
    /// may be cleared.
    #[test]
    fn clear_refuses_a_poisoned_model() {
        let fault = Fault::at(3, FaultSite::QuantColumn);
        let text = match clearable(Some(fault)) {
            Err(SessionError::Refused(t)) => t,
            other => panic!("a poisoned model's clear gave {other:?}"),
        };
        assert!(
            text.contains("poisoned") && text.contains(&fault.to_string()),
            "{text}"
        );
        assert!(clearable(None).is_ok());
    }

    /// The clear runs before every arm but the first; the first failure,
    /// the run's or the clear's, ends the list and names its arm, and no
    /// arm after it runs.
    #[test]
    fn arms_clear_between_and_stop_at_the_first_failure() {
        #[derive(Debug, PartialEq)]
        enum Ev {
            Clear,
            Run(usize),
        }
        let mut log = Vec::new();
        let ok = run_arms(
            &mut log,
            &['a', 'b', 'c'],
            |l: &mut Vec<Ev>| {
                l.push(Ev::Clear);
                Ok(())
            },
            |l: &mut Vec<Ev>, i, _| {
                l.push(Ev::Run(i));
                Ok::<(), SessionError>(())
            },
        );
        assert!(ok.is_ok());
        assert_eq!(
            log,
            [Ev::Run(0), Ev::Clear, Ev::Run(1), Ev::Clear, Ev::Run(2)]
        );

        let mut log = Vec::new();
        let run_failed = run_arms(
            &mut log,
            &['a', 'b', 'c'],
            |l: &mut Vec<Ev>| {
                l.push(Ev::Clear);
                Ok(())
            },
            |l: &mut Vec<Ev>, i, _| {
                l.push(Ev::Run(i));
                if i == 1 {
                    Err(SessionError::Refused("arm b".into()))
                } else {
                    Ok(())
                }
            },
        );
        let Err(ArmFailed { arm, arms, error }) = run_failed else {
            panic!("a failing arm ended the list with Ok");
        };
        assert_eq!((arm, arms), (1, 3));
        assert_eq!(error.to_string(), "arm b");
        assert_eq!(log, [Ev::Run(0), Ev::Clear, Ev::Run(1)]);

        let mut log = Vec::new();
        let clear_failed = run_arms(
            &mut log,
            &['a', 'b', 'c'],
            |_: &mut Vec<Ev>| clearable(Some(Fault::at(3, FaultSite::QuantColumn))),
            |l: &mut Vec<Ev>, i, _| {
                l.push(Ev::Run(i));
                Ok::<(), SessionError>(())
            },
        );
        let Err(f) = clear_failed else {
            panic!("a refused clear ended the list with Ok");
        };
        assert_eq!((f.arm, f.arms), (1, 3));
        assert!(
            f.to_string()
                .starts_with("arm 1 of 3: the clear on a model a fault poisoned")
        );
        assert_eq!(log, [Ev::Run(0)]);
    }

    /// A fault reads as one error whichever way the call met it: raised by
    /// its own readback, or found poisoned by an earlier one.
    #[test]
    fn faults_are_one_error() {
        let fault = Fault::at(3, FaultSite::QuantColumn);
        let raised = GpuError::Fault {
            what: "a step",
            fault,
            behind: None,
        };
        let poisoned = GpuError::Poisoned {
            what: "a prompt call",
            fault,
        };
        let other = GpuError::Shape {
            what: "a step",
            detail: "pos 9 + 1 exceeds the cache".into(),
        };
        let text = raised.to_string();
        assert!(
            matches!(SessionError::from(raised), SessionError::Fault(e) if e.to_string() == text)
        );
        assert!(matches!(
            SessionError::from(poisoned),
            SessionError::Fault(_)
        ));
        assert!(matches!(SessionError::from(other), SessionError::Gpu(_)));
    }
}
