//! The session's greedy generation: the one loop over the session as the
//! generation loop's target ([`runtime::generate`] over [`Plain`]), stopped
//! by the token count or an end-of-generation id, each picked id handed to a
//! caller's sink as it is picked.
//!
//! This owns no rule the loop's own owners hold: the stop rule and its
//! reasons are [`runtime::Stop`]'s, the prompt's schedule and the chained
//! step are the target's (`target.rs`), and the end-of-generation set is the
//! vocabulary's ([`tokenizer::Tokenizer::eog`]), passed in by the caller. It
//! adds what a caller reading text needs around the loop: the walls of the
//! prompt and the decode, and the sink as a stream.
//!
//! [`greedy_lookup`] is the same loop over the lookup draft
//! ([`runtime::Speculative`] of [`runtime::Lookup`] behind the width chooser
//! [`runtime::width::Choosing`], two rows a verify), and [`greedy_mtp`] over
//! the MTP draft the session opened ([`Session::mtp_draft`]): the ids they
//! pick are [`greedy`]'s, and the passes they ran, the rows each kept, the wall
//! of each and the chooser's account of them are in the result for a caller
//! that prints them.

use std::fmt;
use std::time::{Duration, Instant};

use runtime::width::{Chosen, Clock, GateState, Mode, Wall};
use runtime::{
    Advance, Choosing, Committed, Draft, Lookup, NoTokens, PassSink, Plain, Speculative, Stop,
    StopReason, Tally, Target, WidthError,
};

use super::session::{Session, SessionError, StepMode};

/// A greedy run's refusal or failure.
#[derive(Debug)]
pub enum GenError {
    /// The session refused or failed; its text names what.
    Session(SessionError),
    /// A token budget of no tokens ([`NoTokens`], whose rule this carries
    /// unchanged).
    NoTokens(NoTokens),
    /// The width chooser refused its rule before the run
    /// ([`WidthError`], whose text this carries unchanged).
    Width(WidthError),
    /// The width chooser's count of its passes is not the loop's: the two
    /// are of different runs, and the account would say neither.
    Counts(String),
}

impl fmt::Display for GenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GenError::Session(e) => write!(f, "{e}"),
            GenError::NoTokens(e) => write!(f, "{e}"),
            GenError::Width(e) => write!(f, "{e}"),
            GenError::Counts(detail) => write!(f, "width chooser: {detail}"),
        }
    }
}

impl std::error::Error for GenError {}

impl From<SessionError> for GenError {
    fn from(e: SessionError) -> Self {
        GenError::Session(e)
    }
}

impl From<NoTokens> for GenError {
    fn from(e: NoTokens) -> Self {
        GenError::NoTokens(e)
    }
}

impl From<WidthError> for GenError {
    fn from(e: WidthError) -> Self {
        GenError::Width(e)
    }
}

/// What a greedy run tells its caller.
#[derive(Clone, Debug, PartialEq)]
pub struct Generated {
    /// Every generated token, generated token 0 first, up to the first
    /// end-of-generation id among them.
    pub ids: Vec<u32>,
    /// Why the run ended ([`StopReason`], the stop rule's own word).
    pub stop: StopReason,
    /// The wall of the prompt call: every prompt id through the target's
    /// prompt schedule, read back after the last.
    pub prompt_wall: Duration,
    /// The wall of the loop proper: every pass after the prompt's argmax.
    pub decode_wall: Duration,
    /// The completed passes counted by the runtime after the prompt.
    pub passes: usize,
    /// The wall of each pass of the loop, in order: a first pass that stands
    /// out from the rest is a cost the prompt left behind.
    pub pass_walls: Vec<Duration>,
    /// What each pass of the loop kept, in order, as the runtime counted it:
    /// the rows it ran and the rows it kept, and whether it verified a
    /// proposal. A plain run's passes are each one row.
    pub commits: Vec<Committed>,
    /// The width chooser's account of a drafted run ([`greedy_lookup`],
    /// [`greedy_mtp`]); a plain run has none.
    pub width: Option<WidthAccount>,
}

/// What the width chooser did over a drafted run, read when the run ended.
#[derive(Clone, Debug, PartialEq)]
pub struct WidthAccount {
    /// The mode the run ran under.
    pub mode: Mode,
    /// Under [`Mode::Cost`], what the chooser counted and where its gate
    /// stands; [`Mode::Fixed`] counts nothing and leaves it `None`.
    pub counted: Option<Counted>,
}

/// The chooser's own counts and gate under [`Mode::Cost`].
#[derive(Clone, Debug, PartialEq)]
pub struct Counted {
    /// Every pass by the ids it verified, and the passes it ran with the gate
    /// closed ([`Tally::widths`], [`Tally::closed`]).
    pub tally: Tally,
    /// The gate, each width's cost median and the rates the gate compared at
    /// the run's end.
    pub gate: GateState,
}

impl WidthAccount {
    /// The account of `chooser` over a run whose passes were `commits`, the
    /// chooser's count started over. The chooser's passes and windows are the
    /// loop's own: a count that differs is of another run and is refused by
    /// name.
    fn read<D, C: Clock>(
        chooser: &mut Choosing<D, C>,
        commits: &[Committed],
    ) -> Result<WidthAccount, GenError> {
        let mode = chooser.mode();
        if mode == Mode::Fixed {
            return Ok(WidthAccount {
                mode,
                counted: None,
            });
        }
        let tally = chooser.take_tally();
        let windows = commits.iter().filter(|c| c.proposed).count();
        if tally.passes() != commits.len() as u64 || tally.windows != windows as u64 {
            return Err(GenError::Counts(format!(
                "it counted {} passes and {} windows, the loop ran {} passes and {windows} verified \
                 a proposal",
                tally.passes(),
                tally.windows,
                commits.len()
            )));
        }
        Ok(WidthAccount {
            mode,
            counted: Some(Counted {
                tally,
                gate: chooser.gate_state(),
            }),
        })
    }
}

impl Generated {
    /// The tokens the loop decoded: [`Generated::ids`]'s length.
    #[must_use]
    pub fn tokens(&self) -> usize {
        self.ids.len()
    }

    /// What the run's passes were: the proposals verified and how many of
    /// their drafts the session's argmax kept, the passes that were one step,
    /// and the mean wall of each kind past the first pass of the run (which
    /// carries what the prompt left behind). A verify that follows a step
    /// waits out the token the step ran ahead, so its wall is told apart from
    /// one that follows a verify.
    #[must_use]
    pub fn tally(&self) -> PassTally {
        let mut t = PassTally::default();
        let [mut verify, mut after_verify, mut after_step, mut step] = [(); 4].map(|()| Vec::new());
        for (i, c) in self.commits.iter().enumerate() {
            // The first pass carries what the prompt left behind.
            let wall = self.pass_walls.get(i).copied().filter(|_| i > 0);
            if c.proposed {
                t.proposals += 1;
                t.accepted += c.kept - 1;
                t.partial += usize::from(c.kept < c.rows);
                if let Some(w) = wall {
                    verify.push(w);
                    // `i > 0` holds for every `wall`.
                    if self.commits[i - 1].proposed {
                        after_verify.push(w);
                    } else {
                        after_step.push(w);
                    }
                }
            } else {
                t.steps += 1;
                if let Some(w) = wall {
                    step.push(w);
                }
            }
        }
        let mean = |v: &[Duration]| {
            (!v.is_empty())
                .then(|| v.iter().sum::<Duration>() / u32::try_from(v.len()).unwrap_or(u32::MAX))
        };
        t.verify_wall = mean(&verify);
        t.verify_after_verify_wall = mean(&after_verify);
        t.verify_after_step_wall = mean(&after_step);
        t.step_wall = mean(&step);
        t
    }
}

/// A run's passes by kind ([`Generated::tally`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PassTally {
    /// Passes that verified a proposal.
    pub proposals: usize,
    /// Drafts kept over all proposals: a proposal keeps 0 or 1 of its one
    /// draft.
    pub accepted: usize,
    /// Proposals that kept fewer rows than they ran.
    pub partial: usize,
    /// Passes that were one step: no proposal, or a proposal the chooser's
    /// gate did not verify.
    pub steps: usize,
    /// The mean wall of the verify passes past the run's first pass.
    pub verify_wall: Option<Duration>,
    /// The mean wall of those that follow a verify: no token ahead to wait
    /// out.
    pub verify_after_verify_wall: Option<Duration>,
    /// The mean wall of those that follow a step, which waits out the token
    /// the step ran ahead.
    pub verify_after_step_wall: Option<Duration>,
    /// The mean wall of the step passes past the run's first pass.
    pub step_wall: Option<Duration>,
}

/// Greedy generation over the session: `prompt`'s ids through the target,
/// then the loop feeding each pick back until the count or an
/// end-of-generation id ends it, every generated id passed to `on_id` as it
/// is picked — generated token 0 with it, which the prompt call picks.
///
/// The sink may allocate; the steps it hears of do not (the target's own
/// contract, held by the `t_target` clause).
///
/// # Errors
/// [`GenError::NoTokens`] for a budget of no tokens; [`GenError::Session`]
/// for every refusal and failure of the session, which a caller must not
/// step on again without a reset (a poisoned session refuses by name).
pub fn greedy<F>(
    s: &mut Session,
    prompt: &[u32],
    budget: usize,
    eog: &[u32],
    on_id: F,
) -> Result<Generated, GenError>
where
    F: FnMut(u32),
{
    greedy_with(s, &mut Plain, prompt, budget, eog, on_id)
}

/// [`greedy`] over the lookup draft behind the width chooser
/// ([`runtime::width::Choosing`] by [`Lookup::RULE`], `width` its mode): each
/// pass proposes the one id the context's n-grams name, and the chooser
/// verifies it, with the token at the position, in one pass of two rows or
/// runs one step instead, by the pass walls it reads off the host clock — a
/// pass whose verify costs more than the proposal's kept rows repay is not
/// run. The rows a verify keeps are the ones the session's own argmax agrees
/// with (`runtime::Speculative`); a pass with no proposal is one step. The ids
/// are [`greedy`]'s, bit for bit, whatever the mode, because a verify row is
/// the bits of the step at its position and the chooser changes which rows a
/// pass runs, never a token. [`Generated::width`] holds the chooser's account.
///
/// # Errors
/// As [`greedy`], the session's refusal by name of a verify over a file
/// whose layers the pass does not serve ([`Session::verify_serves`]), and the
/// chooser's: [`GenError::Width`], [`GenError::Counts`].
pub fn greedy_lookup<F>(
    s: &mut Session,
    prompt: &[u32],
    budget: usize,
    eog: &[u32],
    width: Mode,
    on_id: F,
) -> Result<Generated, GenError>
where
    F: FnMut(u32),
{
    let draft =
        <Lookup as Chosen<Session>>::choosing_by(Lookup::new(), width, Lookup::RULE, Wall::new())?;
    greedy_drafted(s, draft, prompt, budget, eog, on_id)
}

/// [`greedy_lookup`] over the MTP draft the session opened ([`Session::open_mtp`])
/// behind the same chooser, by the chooser's default rule: each pass proposes
/// the one id the draft layer predicts from the target's last hidden row, and
/// the chooser verifies it in one pass of two rows or runs one step. The draft
/// is `runtime::mtp::MtpDraft` with the session as its host, the window policy
/// and chooser the CUDA drafts run; the ids are [`greedy`]'s, bit for bit,
/// whatever the draft proposes. A session whose layer reads the target's hidden
/// row as `pre` or `post` sets it first ([`Session::set_mtp_reading`]).
///
/// # Errors
/// As [`greedy_lookup`], and the session's refusal by name when no sidecar is
/// open.
pub fn greedy_mtp<F>(
    s: &mut Session,
    prompt: &[u32],
    budget: usize,
    eog: &[u32],
    width: Mode,
    on_id: F,
) -> Result<Generated, GenError>
where
    F: FnMut(u32),
{
    let draft = s.mtp_draft()?.choosing(width)?;
    greedy_drafted(s, draft, prompt, budget, eog, on_id)
}

/// The loop of a draft behind the chooser, two rows a verify: the ids are
/// [`greedy`]'s, the chooser's account is in the result.
fn greedy_drafted<D, F>(
    s: &mut Session,
    draft: Choosing<D>,
    prompt: &[u32],
    budget: usize,
    eog: &[u32],
    on_id: F,
) -> Result<Generated, GenError>
where
    D: Draft<Session>,
    F: FnMut(u32),
{
    s.verify_serves()?;
    // A pass with no proposal is one step, and the next pass may be a verify,
    // which drops the token a chained step ran ahead: its device time is a
    // whole step, where a host-fed step costs the wake and the encode the
    // chain hides. The chooser reads those host-fed walls as the plain step's
    // cost. A draft that reads the step's residual needs no token ahead of it.
    let chained = s.argmax_mode();
    s.set_argmax_mode(StepMode::HostFed)?;
    let mut spec: Speculative<Choosing<D>, 2> = Speculative::new(draft);
    let out = greedy_with(s, &mut spec, prompt, budget, eog, on_id);
    s.set_argmax_mode(chained)?;
    let mut out = out?;
    out.width = Some(WidthAccount::read(spec.draft_mut(), &out.commits)?);
    Ok(out)
}

/// The loop of every advance over the session — [`greedy`]'s plain step,
/// [`greedy_lookup`]'s chosen lookup, any draft's `runtime::Speculative`:
/// `advance`'s prompt call, then the runtime's generation loop with this
/// module's sink, the ids and the stop cut at `budget` as [`greedy`]'s are.
///
/// # Errors
/// As [`greedy`], and the advance's own.
pub fn greedy_with<A, F>(
    s: &mut Session,
    advance: &mut A,
    prompt: &[u32],
    budget: usize,
    eog: &[u32],
    on_id: F,
) -> Result<Generated, GenError>
where
    A: Advance<Session>,
    F: FnMut(u32),
{
    let mut on_id = on_id;
    let stop = Stop::new(budget, Target::ctx(s))?;
    let t0 = Instant::now();
    let first = advance.prompt(s, prompt)?;
    let prompt_wall = t0.elapsed();
    on_id(first);
    let mut walls = Vec::new();
    let mut commits = Vec::new();
    let mut sink = Sink {
        on_id: &mut on_id,
        walls: &mut walls,
        commits: &mut commits,
        budget,
        sent: 1,
    };
    let t1 = Instant::now();
    let outcome = runtime::generate(s, advance, prompt, first, &stop.with_eog(eog), &mut sink)?;
    let (ids, stop) = within_budget(outcome.tokens, outcome.stop, budget);
    Ok(Generated {
        ids,
        stop,
        prompt_wall,
        decode_wall: t1.elapsed(),
        passes: outcome.passes,
        pass_walls: walls,
        commits,
        width: None,
    })
}

/// The ids and the stop a run reports at `budget`, from the loop's `tokens`
/// and `stop`. A pass of two rows kept at the count's edge may keep one token
/// past it: the ids are cut to the budget, as the plain run's are, and the
/// run ended at the count, as the plain run did, whatever the cut id was. The
/// loop cut its tokens at the first end-of-generation id, so a run over the
/// budget has none before its last token, and the plain run stops at the
/// budget before it would pick that one.
fn within_budget(mut tokens: Vec<u32>, stop: StopReason, budget: usize) -> (Vec<u32>, StopReason) {
    if tokens.len() > budget {
        tokens.truncate(budget);
        return (tokens, StopReason::Length);
    }
    (tokens, stop)
}

/// The caller's sink as the loop's [`PassSink`]: every token a pass keeps,
/// in order.
struct Sink<'a, F: FnMut(u32)> {
    on_id: &'a mut F,
    walls: &'a mut Vec<Duration>,
    commits: &'a mut Vec<Committed>,
    /// The ids the run asked for, and how many the sink has passed on
    /// (generated token 0 included, which the prompt call picked).
    budget: usize,
    sent: usize,
}

impl<F: FnMut(u32)> PassSink<Session> for Sink<'_, F> {
    type Error = SessionError;

    fn begin(&mut self, _t: &Session) -> Result<(), SessionError> {
        Ok(())
    }

    fn pass(
        &mut self,
        _t: &Session,
        c: &Committed,
        tokens: &[u32],
        wall: Duration,
    ) -> Result<(), SessionError> {
        self.walls.push(wall);
        self.commits.push(*c);
        // A pass keeps whole rows; the loop has already cut `tokens` at the
        // first end-of-generation id, so the sink hears no id past the end.
        for &id in tokens {
            if self.sent < self.budget {
                (self.on_id)(id);
                self.sent += 1;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pass(kept: usize, rows: usize, proposed: bool) -> Committed {
        Committed {
            pos: 0,
            kept,
            rows,
            proposed,
        }
    }

    fn generated(commits: Vec<Committed>, ms: &[u64]) -> Generated {
        Generated {
            ids: Vec::new(),
            stop: StopReason::Length,
            prompt_wall: Duration::ZERO,
            decode_wall: Duration::ZERO,
            passes: commits.len(),
            pass_walls: ms.iter().map(|&m| Duration::from_millis(m)).collect(),
            commits,
            width: None,
        }
    }

    fn lookup_chooser(mode: Mode) -> Choosing<Lookup> {
        <Lookup as Chosen<Session>>::choosing_by(Lookup::new(), mode, Lookup::RULE, Wall::new())
            .expect("the lookup's own rule is one the chooser runs by")
    }

    /// The tally counts a proposal's kept draft as `kept - 1`, a partial
    /// commit as fewer rows kept than run, and takes each kind's mean wall
    /// past the first pass, whichever kind the first pass is; a verify's wall
    /// goes to the mean of its predecessor's kind.
    ///
    /// Mutant: a draft counted as `kept`, the first pass left in a mean, or a
    /// verify's wall put under the other predecessor.
    #[test]
    fn the_tally_counts_drafts_kept_and_means_past_the_first_pass() {
        let g = generated(
            vec![
                pass(2, 2, true),
                pass(1, 1, false),
                pass(1, 2, true),
                pass(1, 1, false),
                pass(2, 2, true),
                pass(2, 2, true),
            ],
            &[300, 90, 110, 80, 120, 70],
        );
        assert_eq!(
            g.tally(),
            PassTally {
                proposals: 4,
                accepted: 3,
                partial: 1,
                steps: 2,
                verify_wall: Some(Duration::from_millis(100)),
                verify_after_verify_wall: Some(Duration::from_millis(70)),
                verify_after_step_wall: Some(Duration::from_millis(115)),
                step_wall: Some(Duration::from_millis(85)),
            }
        );
        let first_step = generated(
            vec![pass(1, 1, false), pass(1, 1, false), pass(2, 2, true)],
            &[400, 80, 100],
        );
        assert_eq!(
            first_step.tally(),
            PassTally {
                proposals: 1,
                accepted: 1,
                partial: 0,
                steps: 2,
                verify_wall: Some(Duration::from_millis(100)),
                verify_after_verify_wall: None,
                verify_after_step_wall: Some(Duration::from_millis(100)),
                step_wall: Some(Duration::from_millis(80)),
            }
        );
        let plain = generated(vec![pass(1, 1, false)], &[90]);
        assert_eq!(
            plain.tally(),
            PassTally {
                steps: 1,
                ..PassTally::default()
            }
        );
    }

    /// A run that kept a token past the budget reports the plain run's ids and
    /// stop: the ids cut to the budget and the count as the reason, whether the
    /// cut-off token was an end-of-generation id or not; a run at or under the
    /// budget keeps the loop's own stop, an end-of-generation id at the last id
    /// included.
    ///
    /// Mutant: the loop's stop kept after the cut (`[a, b, eog]` at budget 2
    /// reported as `eog`), or the cut left off.
    #[test]
    fn a_run_past_the_budget_stops_at_the_count_as_the_plain_run_does() {
        const EOG: u32 = 9;
        assert_eq!(
            within_budget(vec![1, 2, EOG], StopReason::Eog, 2),
            (vec![1, 2], StopReason::Length)
        );
        assert_eq!(
            within_budget(vec![1, 2, 3], StopReason::Length, 2),
            (vec![1, 2], StopReason::Length)
        );
        assert_eq!(
            within_budget(vec![1, EOG], StopReason::Eog, 2),
            (vec![1, EOG], StopReason::Eog)
        );
        assert_eq!(
            within_budget(vec![1, 2], StopReason::Length, 2),
            (vec![1, 2], StopReason::Length)
        );
        assert_eq!(
            within_budget(vec![1, 2], StopReason::Ctx, 4),
            (vec![1, 2], StopReason::Ctx)
        );
    }

    /// The account of a cost chooser is of the loop's own passes: a chooser
    /// that counted other passes than the loop ran is refused by name, one
    /// that counted a run's none over a run of none is the empty account, and
    /// a fixed chooser, which counts nothing, has no counts to compare.
    ///
    /// Mutant: the passes compared against nothing, or the fixed mode's zero
    /// count read as a cost chooser's.
    #[test]
    fn the_width_account_is_the_loops_own_passes_or_refused() {
        let verify = pass(2, 2, true);
        let refused = WidthAccount::read(&mut lookup_chooser(Mode::Cost), &[verify]);
        assert!(
            matches!(refused, Err(GenError::Counts(_))),
            "a chooser that counted no pass beside a loop that ran one: {refused:?}"
        );
        let empty = WidthAccount::read(&mut lookup_chooser(Mode::Cost), &[])
            .expect("no pass counted and none run agree");
        let counted = empty.counted.expect("a cost chooser counts");
        assert_eq!((empty.mode, counted.tally.passes()), (Mode::Cost, 0));
        let fixed = WidthAccount::read(&mut lookup_chooser(Mode::Fixed), &[verify])
            .expect("a fixed chooser counts nothing to compare");
        assert_eq!((fixed.mode, fixed.counted), (Mode::Fixed, None));
    }
}
