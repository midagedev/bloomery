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
pub mod mtp;

use bloomery_gpu::host::PassKind;
use bloomery_gpu::host::swap::ResetReport;
use bloomery_gpu::model::{ChainBody, Rollback, Rows, Slots, StepMode};
use bloomery_gpu::{Fault, GpuError, GpuModel};
use gguf::Split;
use model::placement::{Machine, Plan};
use runtime::seqstate::Kept;
use runtime::{Draft, Out, Speculative, Target, Verify, Want, Width, Widths, Window};

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
        let inputs = B::inputs(&file)?;
        let machine = (args.machine)(B::layer_count(&inputs));
        let plan = B::plan(&inputs, &machine, args.ctx, &args.cfg)?;
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
            logits: Vec::new(),
            tapped: Vec::new(),
            cleared: None,
        })
    }
}

/// The rows of a verify waiting for their commit.
#[derive(Clone, Copy, Debug)]
struct RowsInFlight {
    first: u32,
    m: usize,
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
    /// The last logits row read back ([`Want::Logits`]).
    logits: Vec<f32>,
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
            logits: Vec::new(),
            tapped: Vec::new(),
            cleared: None,
        }
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

    /// Refused while a verify's rows wait for their commit.
    fn idle(&self, what: &str) -> Result<(), SessionError> {
        match self.rows {
            Some(r) => Err(SessionError::Refused(format!(
                "{what}: the verify of {} rows from position {} waits for its commit",
                r.m, r.first
            ))),
            None => Ok(()),
        }
    }

    /// `argmax` as `want` asks: with the logits row read back and kept.
    fn read(&mut self, argmax: u32, want: Want) -> Result<Out<'_>, SessionError> {
        match want {
            Want::Argmax => Ok(Out::Argmax(argmax)),
            Want::Logits => {
                self.logits = self.model.logits()?;
                Ok(Out::Logits {
                    argmax,
                    row: &self.logits,
                })
            }
        }
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
        Ok(self.model.reset()?)
    }
}

impl<B: Prompt + Keep + Rows + Rollback> Verify for Session<B> {
    const MAX_ROWS: usize = B::MAX_ROWS;

    fn verify<const M: usize>(&mut self, rows: [u32; M]) -> Result<[u32; M], SessionError> {
        self.idle("verify")?;
        let first = self.model.pos();
        let out = self.model.step_rows::<M>(rows)?;
        self.rows = Some(RowsInFlight { first, m: M });
        Ok(out)
    }

    fn commit(&mut self, accepted: usize) -> Result<(), SessionError> {
        let r = self.rows.take().ok_or_else(|| {
            SessionError::Refused(format!("a commit of {accepted} rows with no verify"))
        })?;
        if !(1..=r.m).contains(&accepted) {
            self.rows = Some(r);
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
        self.model.rollback(back)?;
        self.model.keep_rows(accepted, PassKind::Pair)?;
        Ok(())
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
    use super::{ArmFailed, SessionError, clearable, run_arms};
    use bloomery_gpu::{Fault, FaultSite, GpuError};

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
