//! The greedy generation loop over any target: the one loop over a session
//! as the generation loop's target ([`runtime::generate`] over [`Plain`]),
//! stopped by the token count or an end-of-generation id, each picked id
//! handed to a caller's sink as it is picked, generic over the target.
//!
//! This owns no rule the loop's own owners hold: the stop rule and its
//! reasons are [`runtime::Stop`]'s, the prompt's schedule and the chained
//! step are the target's, and the end-of-generation set is the vocabulary's
//! ([`tokenizer::Tokenizer::eog`]), passed in by the caller. It adds what a
//! caller reading text needs around the loop: the walls of the prompt and
//! the decode, and the sink as a stream.

use std::fmt;
use std::time::{Duration, Instant};

use runtime::Tally;
use runtime::width::{Choosing, Clock, GateState, Mode};
use runtime::{
    Advance, Committed, NoTokens, PassSink, Plain, Stop, StopReason, Target, WidthError,
};

/// A greedy run's refusal or failure.
#[derive(Debug)]
pub enum GenError<E> {
    /// The target refused or failed; its text names what.
    Session(E),
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

impl<E: fmt::Display> fmt::Display for GenError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GenError::Session(e) => write!(f, "{e}"),
            GenError::NoTokens(e) => write!(f, "{e}"),
            GenError::Width(e) => write!(f, "{e}"),
            GenError::Counts(detail) => write!(f, "width chooser: {detail}"),
        }
    }
}

impl<E> std::error::Error for GenError<E> where E: fmt::Display + fmt::Debug {}

impl<E> From<NoTokens> for GenError<E> {
    fn from(e: NoTokens) -> Self {
        GenError::NoTokens(e)
    }
}

impl<E> From<WidthError> for GenError<E> {
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
    /// The width chooser's account of a drafted run; a plain run has none.
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
    pub(crate) fn read<D, C: Clock, E>(
        chooser: &mut Choosing<D, C>,
        commits: &[Committed],
    ) -> Result<WidthAccount, GenError<E>> {
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
    /// their drafts the target's argmax kept, the passes that were one step,
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

/// Greedy generation over the target: `prompt`'s ids through the target,
/// then the loop feeding each pick back until the count or an
/// end-of-generation id ends it, every generated id passed to `on_id` as it
/// is picked — generated token 0 with it, which the prompt call picks.
///
/// The sink may allocate; the steps it hears of do not (the target's own
/// contract, held by the `t_target` clause).
///
/// # Errors
/// [`GenError::NoTokens`] for a budget of no tokens; [`GenError::Session`]
/// for every refusal and failure of the target, which a caller must not
/// step on again without a reset (a poisoned session refuses by name).
pub fn greedy<T, F>(
    s: &mut T,
    prompt: &[u32],
    budget: usize,
    eog: &[u32],
    on_id: F,
) -> Result<Generated, GenError<T::Error>>
where
    T: Target,
    F: FnMut(u32),
{
    greedy_with(s, &mut Plain, prompt, budget, eog, on_id)
}

/// The loop of every advance over the target — [`greedy`]'s plain step, a
/// chosen lookup, any draft's `runtime::Speculative`: `advance`'s prompt
/// call, then the runtime's generation loop with this module's sink, the ids
/// and the stop cut at `budget` as [`greedy`]'s are.
///
/// # Errors
/// As [`greedy`], and the advance's own.
pub fn greedy_with<A, F, T>(
    s: &mut T,
    advance: &mut A,
    prompt: &[u32],
    budget: usize,
    eog: &[u32],
    on_id: F,
) -> Result<Generated, GenError<T::Error>>
where
    T: Target,
    A: Advance<T>,
    F: FnMut(u32),
{
    let mut on_id = on_id;
    let stop = Stop::new(budget, Target::ctx(s))?;
    let t0 = Instant::now();
    let first = advance.prompt(s, prompt).map_err(GenError::Session)?;
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
    let outcome = runtime::generate(s, advance, prompt, first, &stop.with_eog(eog), &mut sink)
        .map_err(GenError::Session)?;
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

impl<F: FnMut(u32), T: Target> PassSink<T> for Sink<'_, F> {
    type Error = T::Error;

    fn begin(&mut self, _t: &T) -> Result<(), T::Error> {
        Ok(())
    }

    fn pass(
        &mut self,
        _t: &T,
        c: &Committed,
        tokens: &[u32],
        wall: Duration,
    ) -> Result<(), T::Error> {
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
}
