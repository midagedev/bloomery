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
//! them. [`Session::with_draft`] then captures the verify pass.
//!
//! What a model adds is its [`Open`], [`Prompt`] and [`Keep`], under
//! [`arch`].

pub mod arch;

use bloomery_gpu::model::{ChainBody, Rollback, Rows, StepMode};
use bloomery_gpu::{GpuError, GpuModel};
use gguf::Split;
use model::placement::{Machine, Plan};
use runtime::{Draft, Out, Speculative, Target, Verify, Want};

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

    /// Take back the positions from `n` on; the session asks only for a cut
    /// [`Keep::keepable`] grants.
    fn cut(m: &mut GpuModel<Self>, n: u32) -> Result<(), GpuError>;
}

/// How a session opens: the placement, the context the plan is asked for,
/// the step mode and the body's own configuration.
pub struct OpenArgs<C> {
    /// The placement's name, as the plan and load records print it.
    pub place: &'static str,
    /// The placement's machine, by layer count.
    pub machine: fn(usize) -> Machine,
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
    pub fn open(
        file: Split,
        args: OpenArgs<B::Cfg>,
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
            rows: None,
            logits: Vec::new(),
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
    rows: Option<RowsInFlight>,
    /// The last logits row read back ([`Want::Logits`]).
    logits: Vec<f32>,
}

impl<B: ChainBody> Session<B> {
    /// The model, for what the traits do not carry (records, instruments).
    pub fn model(&self) -> &GpuModel<B> {
        &self.model
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

    fn cut(&mut self, n: u32) -> Result<(), SessionError> {
        self.idle("cut")?;
        let kept = B::keepable(&self.model, n);
        if kept != n {
            return Err(SessionError::Refused(format!(
                "a cut to position {n} from {}: the caches keep {kept}",
                self.model.pos()
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
        if accepted < r.m {
            let back = r.first
                + u32::try_from(accepted).map_err(|_| {
                    SessionError::Refused(format!("{accepted} accepted rows pass u32"))
                })?;
            self.model.rollback(back)?;
        }
        Ok(())
    }
}

impl<B: Prompt + Keep + Rows + Rollback> Session<B> {
    /// Drive `draft` on this session: in graph mode its verify pass of `M` =
    /// [`Draft::WIDTH`] + 1 rows captured now, with the heads it needs, so no
    /// pass pays for them, and `log` told; in eager mode nothing is captured
    /// and the first pass makes the heads. A draft that reads the target's
    /// hidden rows has attached its taps on the [`Loaded`] model already. `M`
    /// past the body's [`Rows::MAX_ROWS`], or not the draft's width + 1, does
    /// not compile.
    pub fn with_draft<D: Draft<Self>, const M: usize>(
        &mut self,
        draft: D,
        log: &mut impl RowsLog,
    ) -> Result<Speculative<D, M>, SessionError> {
        let () = D::FITS;
        const {
            assert!(
                M == D::WIDTH + 1,
                "a draft's verify runs its WIDTH + 1 rows"
            );
        }
        self.idle("with_draft")?;
        if self.model.mode() == StepMode::Graph {
            self.model.capture_rows::<M>()?;
            let nodes = self.model.rows_graph_nodes::<M>()?.len();
            log.capture_rows(M, nodes)?;
        }
        Ok(Speculative::new(draft))
    }
}

#[cfg(test)]
mod tests {
    use super::SessionError;
    use bloomery_gpu::{Fault, FaultSite, GpuError};

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
