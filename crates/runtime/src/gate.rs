//! The draft gate: a draft's verify passes run only while they pay.
//!
//! A verify pass keeps `1 + acc` tokens, `acc` the draft ids the target agrees
//! with, and costs `pass` where a plain step costs `step`; it pays while the
//! gain `(1 + acc) · step / pass` ([`gain`]) is above 1. [`Gated`] wraps a
//! [`Draft`] and keeps that gain in view from the passes it sees:
//!
//! - open, it proposes as its draft does, and closes once the gain over the
//!   verified window falls below [`Rule::close_below`];
//! - closed, it proposes nothing to verify, so the pass is the target's plain
//!   step. Every [`Rule::shadow_every`] passes it asks the draft for a
//!   *shadow* proposal and scores it against the target's own next tokens,
//!   and it opens again once the gain over the shadow window passes
//!   [`Rule::open_above`]. A shadow is never a verified acceptance: the two
//!   windows are apart.
//!
//! The two costs are the medians of the last passes of each kind, kept
//! current by probes: while closed a verify pass after every
//! [`Rule::probe_every`] others, while open a plain step as often; a cost with
//! fewer than [`Rule::min_timings`] readings is probed after
//! [`Rule::warm_every`] passes. No decision is taken on a window of fewer than
//! [`Rule::min_window`] passes; the window a switch leads into starts empty,
//! so a state is judged by what it saw itself.
//!
//! One clock reading a pass, at its proposal: the time to the next proposal
//! is the pass's — the proposal, the target's call, the draft's update and
//! whatever the caller does before its next pass. A reading is kept only when
//! exactly one accept or step came between and the target stands where that
//! call left it; a prompt call, a reset or a step outside a pass drops it, and
//! a pending shadow with it.
//!
//! The gate changes which passes run, never a token: every kept token is the
//! target's argmax. While closed the draft still hears every position, through
//! [`Draft::held`].

use std::collections::VecDeque;
use std::fmt;
use std::time::{Duration, Instant};

use crate::stores::PASS_ROWS;
use crate::{Draft, TapNeed, Verify};

/// A clock the gate reads once a pass.
pub trait Clock {
    /// The time since the clock's origin; never less than an earlier reading.
    fn now(&mut self) -> Duration;
}

/// The host's monotonic clock.
#[derive(Clone, Copy, Debug)]
pub struct Wall(Instant);

impl Wall {
    /// A clock whose origin is now.
    #[must_use]
    pub fn new() -> Wall {
        Wall(Instant::now())
    }
}

impl Default for Wall {
    fn default() -> Wall {
        Wall::new()
    }
}

impl Clock for Wall {
    fn now(&mut self) -> Duration {
        self.0.elapsed()
    }
}

/// What the gate does: judge the draft's passes, or let every proposal run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Open while the verified passes pay, closed while they do not.
    Auto,
    /// Always open: the draft's proposals are verified as without a gate,
    /// and the gate only reads its passes.
    Open,
}

/// The gate's numbers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rule {
    /// An open gate closes when the verified window's gain is below this.
    pub close_below: f64,
    /// A closed gate opens when the shadow window's gain is above this; the
    /// band between the two is the hysteresis.
    pub open_above: f64,
    /// Passes an acceptance window holds.
    pub window: usize,
    /// Passes a window holds before the gate decides on it.
    pub min_window: usize,
    /// A closed gate's shadow proposal after this many passes.
    pub shadow_every: u32,
    /// A probe of the other kind of pass after this many passes.
    pub probe_every: u32,
    /// The same while that kind has fewer than `min_timings` readings.
    pub warm_every: u32,
    /// Readings a cost's median is taken over.
    pub timings: usize,
    /// Readings of each cost before the gate decides.
    pub min_timings: usize,
}

impl Rule {
    /// Close as soon as the pass loses, open only once it pays by 3 %;
    /// windows of 64 passes, decisions from 16 on; a shadow every 4th closed
    /// pass; a probe after 32 passes, after 2 while a cost is being learned;
    /// costs as the median of the last 8 readings, from 4 on.
    pub const DEFAULT: Rule = Rule {
        close_below: 1.0,
        open_above: 1.03,
        window: 64,
        min_window: 16,
        shadow_every: 4,
        probe_every: 32,
        warm_every: 2,
        timings: 8,
        min_timings: 4,
    };

    /// Refused by name unless the gate can run by it.
    pub fn check(&self) -> Result<(), GateError> {
        let bounds = self.close_below.is_finite()
            && self.open_above.is_finite()
            && self.close_below > 0.0
            && self.close_below <= self.open_above;
        let refused = if !bounds {
            "the gain bounds must be finite, with 0 < close_below <= open_above"
        } else if self.min_window == 0 {
            "a window of zero passes"
        } else if self.window < self.min_window {
            "a window shorter than its decision minimum"
        } else if self.min_timings == 0 {
            "a cost of zero readings"
        } else if self.timings < self.min_timings {
            "a cost's readings fewer than its decision minimum"
        } else if self.shadow_every == 0 || self.warm_every == 0 {
            "a period of zero passes"
        } else if self.probe_every < self.warm_every {
            "a probe period shorter than the warm-up's"
        } else {
            return Ok(());
        };
        Err(GateError::Rule(refused))
    }
}

/// Input the gate has no answer for.
#[derive(Clone, Debug, PartialEq)]
pub enum GateError {
    /// A [`Rule`] the gate cannot run by.
    Rule(&'static str),
    /// A window of no passes asked for its value.
    EmptyWindow(&'static str),
    /// A time that is not a positive, finite number of milliseconds.
    Time { what: &'static str, ms: f64 },
    /// An acceptance that is not a finite count of ids, 0 or more.
    Acceptance(f64),
    /// A verify that kept no row: row 0 is always kept.
    NoRowKept,
}

impl fmt::Display for GateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GateError::Rule(why) => write!(f, "draft gate: refused rule: {why}"),
            GateError::EmptyWindow(what) => {
                write!(f, "draft gate: the {what} window holds no pass")
            }
            GateError::Time { what, ms } => write!(
                f,
                "draft gate: a {what} time of {ms} ms: a time is positive and finite"
            ),
            GateError::Acceptance(a) => write!(
                f,
                "draft gate: an acceptance of {a}: it is a finite count of ids, 0 or more"
            ),
            GateError::NoRowKept => f.write_str("draft gate: a verify that kept no row"),
        }
    }
}

impl std::error::Error for GateError {}

/// Tokens per unit of time of a verify pass over a plain step's: `(1 +
/// acc) · step_ms / pass_ms`, `acc` the draft ids a verified pass keeps on
/// average. Above 1 the pass pays.
pub fn gain(acc: f64, pass_ms: f64, step_ms: f64) -> Result<f64, GateError> {
    if !(acc.is_finite() && acc >= 0.0) {
        return Err(GateError::Acceptance(acc));
    }
    let pass_ms = positive("pass", pass_ms)?;
    let step_ms = positive("step", step_ms)?;
    Ok((1.0 + acc) * step_ms / pass_ms)
}

fn positive(what: &'static str, ms: f64) -> Result<f64, GateError> {
    if ms.is_finite() && ms > 0.0 {
        Ok(ms)
    } else {
        Err(GateError::Time { what, ms })
    }
}

/// The last passes' kept draft ids, and how many passes it was ever fed.
#[derive(Clone, Debug)]
struct Tally {
    what: &'static str,
    ring: VecDeque<u32>,
    cap: usize,
    sum: u64,
    seen: u64,
}

impl Tally {
    fn new(what: &'static str, cap: usize) -> Tally {
        Tally {
            what,
            ring: VecDeque::with_capacity(cap),
            cap,
            sum: 0,
            seen: 0,
        }
    }

    fn push(&mut self, ids: u32) {
        if self.ring.len() == self.cap
            && let Some(old) = self.ring.pop_front()
        {
            self.sum -= u64::from(old);
        }
        self.ring.push_back(ids);
        self.sum += u64::from(ids);
        self.seen += 1;
    }

    /// Empty the window; the count of passes ever fed stays.
    fn clear(&mut self) {
        self.ring.clear();
        self.sum = 0;
    }

    fn len(&self) -> usize {
        self.ring.len()
    }

    fn mean(&self) -> Result<f64, GateError> {
        if self.ring.is_empty() {
            return Err(GateError::EmptyWindow(self.what));
        }
        Ok(self.sum as f64 / self.ring.len() as f64)
    }
}

/// The last readings of one kind of pass, in milliseconds.
#[derive(Clone, Debug)]
struct Timings {
    what: &'static str,
    ring: VecDeque<f64>,
    cap: usize,
    /// The median's sort, held so a reading allocates nothing.
    sorted: Vec<f64>,
}

impl Timings {
    fn new(what: &'static str, cap: usize) -> Timings {
        Timings {
            what,
            ring: VecDeque::with_capacity(cap),
            cap,
            sorted: Vec::with_capacity(cap),
        }
    }

    fn push(&mut self, ms: f64) -> Result<(), GateError> {
        let ms = positive(self.what, ms)?;
        if self.ring.len() == self.cap {
            self.ring.pop_front();
        }
        self.ring.push_back(ms);
        Ok(())
    }

    fn len(&self) -> usize {
        self.ring.len()
    }

    fn median(&mut self) -> Result<f64, GateError> {
        if self.ring.is_empty() {
            return Err(GateError::EmptyWindow(self.what));
        }
        self.sorted.clear();
        self.sorted.extend(self.ring.iter().copied());
        self.sorted.sort_by(f64::total_cmp);
        let n = self.sorted.len();
        Ok(if n % 2 == 1 {
            self.sorted[n / 2]
        } else {
            f64::midpoint(self.sorted[n / 2 - 1], self.sorted[n / 2])
        })
    }
}

/// What a pass ran, as its proposal left it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// A verify of a proposal: a reading of the pass cost.
    Verify,
    /// A plain step: a reading of the step cost.
    Step,
    /// A plain step after a shadow proposal: no reading.
    Shadow,
}

/// The pass in flight since its proposal.
#[derive(Clone, Copy, Debug)]
struct Flight {
    kind: Kind,
    /// The target's position at the proposal.
    pos: u32,
    at: Duration,
    /// The position the pass's accept or step left the target at; `None`
    /// until one came.
    ends: Option<u32>,
}

/// A shadow proposal scored against the target's next tokens as they come.
#[derive(Clone, Copy, Debug)]
struct Shadow {
    ids: [u32; PASS_ROWS],
    /// Ids proposed; 0 is none pending.
    n: usize,
    /// Leading ids the target's tokens have matched so far.
    matched: usize,
}

impl Shadow {
    const NONE: Shadow = Shadow {
        ids: [0; PASS_ROWS],
        n: 0,
        matched: 0,
    };
}

/// The fields of a `draft gate` record: the gate after its latest pass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GateRecord {
    /// Passes the gate has seen.
    pub pass: u64,
    /// Whether the next pass may verify a proposal.
    pub open: bool,
    /// Draft ids a verified pass kept, over the verified window; `None` while
    /// it is empty.
    pub acc_win: Option<f64>,
    /// The same of the shadow proposals, over the shadow window.
    pub acc_shadow: Option<f64>,
    /// The median of the last verify passes' times, ms.
    pub pass_ms: Option<f64>,
    /// The median of the last plain steps' times, ms.
    pub step_ms: Option<f64>,
    /// The acceptance at which the gate switches from its state: open, the
    /// gain falls below [`Rule::close_below`] under it; closed, it passes
    /// [`Rule::open_above`] over it. `None` until both costs have a reading.
    pub threshold: Option<f64>,
}

/// A draft behind a gate: its proposals verified only while they pay (see
/// the module).
#[derive(Debug)]
pub struct Gated<D, C = Wall> {
    draft: D,
    clock: C,
    rule: Rule,
    mode: Mode,
    open: bool,
    acc_win: Tally,
    acc_shadow: Tally,
    pass_ms: Timings,
    step_ms: Timings,
    since_probe: u32,
    since_shadow: u32,
    passes: u64,
    switches: u64,
    flight: Option<Flight>,
    shadow: Shadow,
    /// The gate right after each switch since the last
    /// [`Gated::take_switched`].
    switched: Vec<GateRecord>,
}

impl<D> Gated<D, Wall> {
    /// `draft` behind a gate of `mode` by `rule`, on the host's clock; open.
    pub fn new(draft: D, mode: Mode, rule: Rule) -> Result<Gated<D, Wall>, GateError> {
        Gated::with_clock(draft, mode, rule, Wall::new())
    }
}

impl<D, C: Clock> Gated<D, C> {
    /// [`Gated::new`] on `clock`.
    pub fn with_clock(
        draft: D,
        mode: Mode,
        rule: Rule,
        clock: C,
    ) -> Result<Gated<D, C>, GateError> {
        rule.check()?;
        Ok(Gated {
            draft,
            clock,
            rule,
            mode,
            open: true,
            acc_win: Tally::new("verified", rule.window),
            acc_shadow: Tally::new("shadow", rule.window),
            pass_ms: Timings::new("pass", rule.timings),
            step_ms: Timings::new("step", rule.timings),
            since_probe: 0,
            since_shadow: 0,
            passes: 0,
            switches: 0,
            flight: None,
            shadow: Shadow::NONE,
            switched: Vec::new(),
        })
    }

    /// The draft.
    pub fn draft(&self) -> &D {
        &self.draft
    }

    /// See [`Gated::draft`].
    pub fn draft_mut(&mut self) -> &mut D {
        &mut self.draft
    }

    /// The gate's mode.
    #[must_use]
    pub fn mode(&self) -> Mode {
        self.mode
    }

    /// Whether the next pass may verify a proposal.
    #[must_use]
    pub fn is_open(&self) -> bool {
        self.open
    }

    /// Times the gate opened or closed.
    #[must_use]
    pub fn switches(&self) -> u64 {
        self.switches
    }

    /// Verified passes the gate has counted, every one it ever saw.
    #[must_use]
    pub fn verified(&self) -> u64 {
        self.acc_win.seen
    }

    /// Shadow proposals the gate has scored, every one it ever saw.
    #[must_use]
    pub fn shadowed(&self) -> u64 {
        self.acc_shadow.seen
    }

    /// The gate right after each switch since the last call, oldest first:
    /// what a caller prints as it goes, the passes running inside its loop.
    pub fn take_switched(&mut self) -> Vec<GateRecord> {
        std::mem::take(&mut self.switched)
    }

    /// The gate after its latest pass, as a `draft gate` record holds it.
    pub fn record(&mut self) -> GateRecord {
        let pass_ms = self.pass_ms.median().ok();
        let step_ms = self.step_ms.median().ok();
        let bound = if self.open {
            self.rule.close_below
        } else {
            self.rule.open_above
        };
        GateRecord {
            pass: self.passes,
            open: self.open,
            acc_win: self.acc_win.mean().ok(),
            acc_shadow: self.acc_shadow.mean().ok(),
            pass_ms,
            step_ms,
            threshold: pass_ms.zip(step_ms).map(|(p, s)| bound * p / s - 1.0),
        }
    }

    /// Forget the pass in flight and a pending shadow: nothing before a new
    /// generation is read as a pass.
    fn drop_flight(&mut self) {
        self.flight = None;
        self.shadow = Shadow::NONE;
    }

    /// Close the pass in flight at `now`, the target at `pos`: its reading
    /// kept when it was heard once and the target stands where it left it.
    fn land(&mut self, now: Duration, pos: u32) -> Result<(), GateError> {
        let Some(f) = self.flight.take() else {
            return Ok(());
        };
        if f.ends != Some(pos) {
            self.shadow = Shadow::NONE;
            return Ok(());
        }
        let ms = match now.checked_sub(f.at) {
            Some(d) => d.as_secs_f64() * 1e3,
            None => -(f.at - now).as_secs_f64() * 1e3,
        };
        match f.kind {
            Kind::Verify => self.pass_ms.push(ms),
            Kind::Step => self.step_ms.push(ms),
            Kind::Shadow => positive("shadow pass", ms).map(|_| ()),
        }
    }

    /// Both costs have their readings.
    fn timed(&self) -> bool {
        self.pass_ms.len() >= self.rule.min_timings && self.step_ms.len() >= self.rule.min_timings
    }

    /// Open or close by the window of the gate's state.
    fn decide(&mut self) -> Result<(), GateError> {
        if self.mode == Mode::Open || !self.timed() {
            return Ok(());
        }
        let window = if self.open {
            &self.acc_win
        } else {
            &self.acc_shadow
        };
        if window.len() < self.rule.min_window {
            return Ok(());
        }
        let g = gain(
            window.mean()?,
            self.pass_ms.median()?,
            self.step_ms.median()?,
        )?;
        if self.open && g < self.rule.close_below {
            self.open = false;
            self.acc_shadow.clear();
            self.since_shadow = 0;
        } else if !self.open && g > self.rule.open_above {
            self.open = true;
            self.acc_win.clear();
            self.shadow = Shadow::NONE;
        } else {
            return Ok(());
        }
        self.since_probe = 0;
        self.switches += 1;
        let now = self.record();
        self.switched.push(now);
        Ok(())
    }

    /// Whether this pass probes the cost `of`: after `probe_every` passes,
    /// after `warm_every` while that cost has too few readings.
    fn probe_due(&self, of: &Timings) -> bool {
        let every = if of.len() < self.rule.min_timings {
            self.rule.warm_every
        } else {
            self.rule.probe_every
        };
        self.since_probe >= every
    }
}

/// A proposal's kind: a verify when it holds ids, else a plain step.
fn proposal(n: usize) -> (Kind, usize) {
    if n == 0 {
        (Kind::Step, 0)
    } else {
        (Kind::Verify, n)
    }
}

/// The draft's proposals, verified while they pay; a closed gate's passes are
/// plain steps the draft hears through [`Draft::held`].
impl<T, D, C> Draft<T> for Gated<D, C>
where
    T: Verify,
    T::Error: From<GateError>,
    D: Draft<T>,
    C: Clock,
{
    const WIDTH: usize = D::WIDTH;
    const TAPS: TapNeed = D::TAPS;

    fn prompt(&mut self, t: &mut T, ids: &[u32]) -> Result<u32, T::Error> {
        self.drop_flight();
        self.draft.prompt(t, ids)
    }

    fn begin(&mut self, t: &T, prompt: &[u32], first: u32) -> Result<(), T::Error> {
        self.drop_flight();
        self.draft.begin(t, prompt, first)
    }

    fn propose(&mut self, t: &mut T, last: u32, out: &mut [u32]) -> Result<usize, T::Error> {
        let now = self.clock.now();
        let pos = t.pos();
        self.land(now, pos)?;
        self.decide()?;
        self.passes += 1;
        let (kind, n) = match (self.mode, self.open) {
            (Mode::Open, _) => proposal(self.draft.propose(t, last, out)?),
            (Mode::Auto, true) => {
                if self.probe_due(&self.step_ms) {
                    self.since_probe = 0;
                    (Kind::Step, 0)
                } else {
                    self.since_probe += 1;
                    proposal(self.draft.propose(t, last, out)?)
                }
            }
            (Mode::Auto, false) => {
                if self.shadow.n > 0 {
                    self.since_probe += 1;
                    (Kind::Step, 0)
                } else if self.probe_due(&self.pass_ms) {
                    self.since_probe = 0;
                    proposal(self.draft.propose(t, last, out)?)
                } else {
                    self.since_probe += 1;
                    self.since_shadow += 1;
                    if self.since_shadow >= self.rule.shadow_every {
                        self.since_shadow = 0;
                        assert!(
                            out.len() <= PASS_ROWS,
                            "draft gate: a proposal of {} ids past a pass's {PASS_ROWS} rows",
                            out.len()
                        );
                        let n = self
                            .draft
                            .propose(t, last, &mut self.shadow.ids[..out.len()])?;
                        self.shadow.n = n;
                        self.shadow.matched = 0;
                        (if n == 0 { Kind::Step } else { Kind::Shadow }, 0)
                    } else {
                        (Kind::Step, 0)
                    }
                }
            }
        };
        self.flight = Some(Flight {
            kind,
            pos,
            at: now,
            ends: None,
        });
        Ok(n)
    }

    fn accept(
        &mut self,
        t: &mut T,
        rows: &[u32],
        out: &[u32],
        accepted: usize,
    ) -> Result<(), T::Error> {
        let kept = accepted.checked_sub(1).ok_or(GateError::NoRowKept)?;
        match &mut self.flight {
            Some(f) if f.kind == Kind::Verify && f.ends.is_none() => {
                f.ends = u32::try_from(accepted).ok().map(|a| f.pos + a);
            }
            _ => self.drop_flight(),
        }
        self.acc_win
            .push(u32::try_from(kept).expect("a verify keeps fewer ids than a pass's rows"));
        self.draft.accept(t, rows, out, accepted)
    }

    fn stepped(&mut self, t: &mut T, last: u32, next: u32) -> Result<(), T::Error> {
        let within = match &mut self.flight {
            Some(f) if f.kind != Kind::Verify && f.ends.is_none() => {
                f.ends = Some(f.pos + 1);
                true
            }
            _ => false,
        };
        if !within {
            self.drop_flight();
        }
        let s = &mut self.shadow;
        if s.n > 0 {
            let hit = s.ids[s.matched] == next;
            if hit {
                s.matched += 1;
            }
            if !hit || s.matched == s.n {
                let matched =
                    u32::try_from(s.matched).expect("a shadow holds a pass's rows at most");
                self.acc_shadow.push(matched);
                self.shadow = Shadow::NONE;
            }
        }
        if self.mode == Mode::Auto && !self.open {
            self.draft.held(t, last, next)
        } else {
            self.draft.stepped(t, last, next)
        }
    }

    fn held(&mut self, t: &mut T, last: u32, next: u32) -> Result<(), T::Error> {
        Draft::<T>::stepped(self, t, last, next)
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::rc::Rc;

    use super::{GateError, Gated, Mode, Rule, Tally, Timings, gain};
    use crate::mock::{Call, Clockwork, Mock, MockError, Quiet};
    use crate::{Advance, Draft, Plain, Speculative, Stop, TapNeed, Target, generate};

    /// The draft's own proposal cost on the mock's clock.
    const DRAFT_US: u64 = 500;
    const STEP_US: u64 = 37_450;

    /// When set, `(positions, a, b)`: the acceptance is `a` and `b` in turn,
    /// each for that many target positions.
    type Phases = Option<(u32, f64, f64)>;

    /// A draft of `W` ids whose proposals the target agrees with at the
    /// acceptance it is set to, spread evenly: draw `k` is right when
    /// `⌊(k + 1)·a⌋ > ⌊k·a⌋`, so every run of `n` draws holds `⌊n·a⌋` or
    /// `⌈n·a⌉` right ones. Like the engine's card draft it refuses to propose
    /// unless it holds every position before the target's, and it queues a
    /// held position until its next call.
    struct Coin<const W: usize> {
        acc: Rc<Cell<f64>>,
        phases: Phases,
        draws: u64,
        clock: Option<Rc<Clockwork>>,
        /// Positions the draft holds.
        at: u32,
        queued: u32,
        held: u64,
        /// Each proposal's leading ids the target agrees with, and the rows
        /// its verify kept when it ran one.
        scored: Vec<(usize, Option<usize>)>,
    }

    impl<const W: usize> Coin<W> {
        fn new(acc: &Rc<Cell<f64>>, clock: Option<&Rc<Clockwork>>) -> Coin<W> {
            Coin {
                acc: Rc::clone(acc),
                phases: None,
                draws: 0,
                clock: clock.map(Rc::clone),
                at: 0,
                queued: 0,
                held: 0,
                scored: Vec::new(),
            }
        }

        fn draw(&mut self, pos: u32) -> bool {
            let a = match self.phases {
                Some((len, a, b)) => {
                    if (pos / len).is_multiple_of(2) {
                        a
                    } else {
                        b
                    }
                }
                None => self.acc.get(),
            };
            let k = self.draws as f64;
            self.draws += 1;
            ((k + 1.0) * a).floor() > (k * a).floor()
        }

        fn flush(&mut self) {
            self.at += self.queued;
            self.queued = 0;
        }
    }

    impl<const W: usize> Draft<Mock> for Coin<W> {
        const WIDTH: usize = W;
        const TAPS: TapNeed = TapNeed::None;

        fn prompt(&mut self, t: &mut Mock, ids: &[u32]) -> Result<u32, MockError> {
            let next = t.prompt(ids, crate::Want::Argmax)?.argmax();
            (self.at, self.queued) = (t.pos(), 0);
            Ok(next)
        }

        fn begin(&mut self, _t: &Mock, _prompt: &[u32], _first: u32) -> Result<(), MockError> {
            Ok(())
        }

        fn propose(
            &mut self,
            t: &mut Mock,
            last: u32,
            out: &mut [u32],
        ) -> Result<usize, MockError> {
            self.flush();
            if self.at != t.pos() {
                return Err(MockError::Mock("the draft missed a position"));
            }
            if let Some(c) = &self.clock {
                c.advance(DRAFT_US);
            }
            let truth = t.greedy_after(last, W);
            let mut agreed = W;
            for (i, (o, &want)) in out.iter_mut().zip(&truth).enumerate() {
                let right = self.draw(t.pos());
                *o = if right { want } else { (want + 1) % 5 };
                if !right && agreed == W {
                    agreed = i;
                }
            }
            self.scored.push((agreed, None));
            Ok(W)
        }

        fn accept(
            &mut self,
            _t: &mut Mock,
            _rows: &[u32],
            _out: &[u32],
            accepted: usize,
        ) -> Result<(), MockError> {
            self.flush();
            self.at += u32::try_from(accepted).unwrap();
            if let Some(last) = self.scored.last_mut() {
                last.1 = Some(accepted);
            }
            Ok(())
        }

        fn stepped(&mut self, _t: &mut Mock, _last: u32, _next: u32) -> Result<(), MockError> {
            self.flush();
            self.at += 1;
            Ok(())
        }

        fn held(&mut self, _t: &mut Mock, _last: u32, _next: u32) -> Result<(), MockError> {
            self.queued += 1;
            self.held += 1;
            Ok(())
        }
    }

    type Pass = Speculative<Gated<Coin<1>, Rc<Clockwork>>, 2>;

    /// A gated draft of acceptance `acc` on the mock, whose steps and
    /// verifies cost `STEP_US` and `verify` on its clock, standing after a
    /// prompt.
    struct Rig {
        t: Mock,
        spec: Pass,
        clock: Rc<Clockwork>,
        acc: Rc<Cell<f64>>,
        last: u32,
    }

    impl Rig {
        fn new(mode: Mode, acc: f64, verify: u64) -> Rig {
            let clock = Clockwork::new(STEP_US, verify);
            let acc = Rc::new(Cell::new(acc));
            let coin = Coin::new(&acc, Some(&clock));
            let gate = Gated::with_clock(coin, mode, Rule::DEFAULT, Rc::clone(&clock)).unwrap();
            let mut spec = Speculative::new(gate);
            let mut t = Mock::new()
                .with_ctx(1_000_000)
                .with_clock(Rc::clone(&clock));
            let prompt = [1, 2, 3, 1, 2];
            let last = spec.prompt(&mut t, &prompt).unwrap();
            spec.begin(&t, &prompt, last).unwrap();
            Rig {
                t,
                spec,
                clock,
                acc,
                last,
            }
        }

        /// `n` passes; `jitter(i)` sets the clock's costs before pass `i`.
        fn run_with(&mut self, n: usize, mut jitter: impl FnMut(usize, &Clockwork)) {
            let mut out = Vec::with_capacity(2);
            for i in 0..n {
                jitter(i, &self.clock);
                out.clear();
                self.spec.pass(&mut self.t, self.last, &mut out).unwrap();
                self.last = *out.last().unwrap();
            }
        }

        fn run(&mut self, n: usize) {
            self.run_with(n, |_, _| {});
        }

        fn gate(&mut self) -> &mut Gated<Coin<1>, Rc<Clockwork>> {
            self.spec.draft_mut()
        }

        fn verifies(&self) -> usize {
            self.t
                .calls()
                .iter()
                .filter(|c| matches!(c, Call::Verify(..)))
                .count()
        }
    }

    /// The gain is kept tokens a pass over the pass's cost in steps.
    #[test]
    fn gain_is_kept_tokens_over_the_cost_ratio() {
        assert_eq!(gain(0.5, 1.0, 1.0), Ok(1.5));
        assert_eq!(gain(1.0, 2.0, 1.0), Ok(1.0));
        assert_eq!(gain(0.0, 1.0, 2.0), Ok(2.0));
        let g = gain(0.51, 60.5, 37.45).unwrap();
        assert!((g - 1.51 * 37.45 / 60.5).abs() < 1e-12 && g < 1.0, "{g}");
    }

    /// A draft whose verify passes lose closes the gate once its window is
    /// read, and a closed gate verifies only its probes.
    #[test]
    fn closes_when_the_verify_pass_loses() {
        let mut r = Rig::new(Mode::Auto, 0.5, 60_500);
        r.run(40);
        assert!(!r.gate().is_open(), "{:?}", r.gate().record());
        let switched = r.gate().take_switched();
        assert!(switched.len() == 1 && !switched[0].open, "{switched:?}");
        let (acc, th) = (switched[0].acc_win.unwrap(), switched[0].threshold.unwrap());
        assert!(acc == 0.5 && th > acc, "{switched:?}");
        let before = r.verifies();
        r.run(990);
        assert_eq!(r.gate().switches(), 1, "{:?}", r.gate().record());
        let probes = r.verifies() - before;
        assert!(
            (29..=31).contains(&probes),
            "{probes} verifies in 990 closed passes"
        );
    }

    /// A gain inside the band between the bounds, its costs jittering, keeps
    /// a closed gate closed.
    #[test]
    fn stays_closed_inside_the_hysteresis() {
        let mut r = Rig::new(Mode::Auto, 1.0, 80_000);
        r.run(60);
        assert!(!r.gate().is_open(), "{:?}", r.gate().record());
        // (1 + 1) · step / pass from 1.007 to 1.023: above 1, below 1.03.
        r.run_with(3000, |i, c| {
            let odd = i % 2 == 1;
            c.step.set(if odd { 37_650 } else { 37_250 });
            c.verify.set(if odd { 73_093 } else { 73_493 });
        });
        let rec = r.gate().record();
        let g = gain(1.0, rec.pass_ms.unwrap(), rec.step_ms.unwrap()).unwrap();
        assert!(g > 1.0 && g < Rule::DEFAULT.open_above, "{g} {rec:?}");
        assert_eq!(r.gate().switches(), 1, "{rec:?}");
    }

    /// A closed gate opens again once its shadow proposals say the verify
    /// pass pays.
    #[test]
    fn reopens_when_the_shadow_says_it_pays() {
        let mut r = Rig::new(Mode::Auto, 0.5, 60_500);
        r.run(200);
        assert!(!r.gate().is_open());
        r.acc.set(1.0);
        r.run(300);
        assert!(r.gate().is_open(), "{:?}", r.gate().record());
        assert_eq!(r.gate().switches(), 2);
        let before = r.verifies();
        r.run(66);
        assert!(r.verifies() - before >= 60, "an open gate verifies");
    }

    /// A shadow proposal is scored in its own window and never counted as a
    /// verified pass: the gate's verified count is the target's verifies.
    #[test]
    fn a_shadow_is_never_a_verified_pass() {
        let mut r = Rig::new(Mode::Auto, 0.5, 60_500);
        r.run(1000);
        assert!(r.gate().shadowed() >= 200, "{}", r.gate().shadowed());
        assert_eq!(r.gate().verified(), r.verifies() as u64);
    }

    /// A draft whose verify passes pay keeps the gate open.
    #[test]
    fn stays_open_while_the_verify_pass_pays() {
        let mut r = Rig::new(Mode::Auto, 0.9, 60_500);
        r.run(1000);
        let rec = r.gate().record();
        assert!(rec.open && r.gate().switches() == 0, "{rec:?}");
        assert!(rec.acc_win.unwrap() > rec.threshold.unwrap(), "{rec:?}");
    }

    /// A closed gate keeps probing the verify pass: when the pass gets
    /// cheaper, a reading taken before it closed does not keep it closed.
    #[test]
    fn the_pass_cost_is_reprobed_while_closed() {
        let mut r = Rig::new(Mode::Auto, 0.5, 74_400);
        r.run(100);
        assert!(!r.gate().is_open());
        r.clock.verify.set(44_440);
        r.run(400);
        assert!(r.gate().is_open(), "{:?}", r.gate().record());
    }

    /// An open gate keeps probing the plain step: when the step gets
    /// cheaper, a reading taken before does not keep it open.
    #[test]
    fn the_step_cost_is_reprobed_while_open() {
        let mut r = Rig::new(Mode::Auto, 0.9, 60_500);
        r.run(100);
        assert!(r.gate().is_open());
        r.clock.step.set(20_000);
        r.run(400);
        assert!(!r.gate().is_open(), "{:?}", r.gate().record());
    }

    /// The gate changes which passes run and never a token: the plain run's
    /// tokens with the gate open, closed, or switching, and the draft hears
    /// every position throughout.
    #[test]
    fn greedy_tokens_are_the_plain_runs() {
        let prompt = [1, 2, 3, 1, 2];
        let stop = Stop::new(1200, 1_000_000).unwrap();
        let mut t = Mock::new().with_ctx(1_000_000);
        let first = Plain.prompt(&mut t, &prompt).unwrap();
        let plain = generate(&mut t, &mut Plain, &prompt, first, &stop, &mut Quiet).unwrap();
        // The mode, the acceptance, its phases, and the switches the run takes.
        let cases: [(Mode, f64, Phases, u64, u64); 3] = [
            (Mode::Open, 0.5, None, 0, 0),
            (Mode::Auto, 0.5, None, 1, 1),
            (Mode::Auto, 0.0, Some((300, 1.0, 0.0)), 3, u64::MAX),
        ];
        for (mode, a, phases, least, most) in cases {
            let clock = Clockwork::new(STEP_US, 60_500);
            let acc = Rc::new(Cell::new(a));
            let mut coin = Coin::<1>::new(&acc, Some(&clock));
            coin.phases = phases;
            let gate = Gated::with_clock(coin, mode, Rule::DEFAULT, Rc::clone(&clock)).unwrap();
            let mut spec = Speculative::<_, 2>::new(gate);
            let mut t = Mock::new().with_ctx(1_000_000).with_clock(clock);
            let first = spec.prompt(&mut t, &prompt).unwrap();
            let out = generate(&mut t, &mut spec, &prompt, first, &stop, &mut Quiet)
                .unwrap_or_else(|e| panic!("{mode:?} {phases:?}: {e}"));
            assert_eq!(out.tokens[..1200], plain.tokens[..], "{mode:?} {phases:?}");
            let g = spec.draft();
            assert!(
                (least..=most).contains(&g.switches()),
                "{mode:?} {phases:?}: {} switches",
                g.switches()
            );
            assert_eq!(g.draft().held > 0, least > 0, "{mode:?} {phases:?}");
        }
    }

    /// A shadow of several ids scores the rows its verify would have kept:
    /// the target's next tokens match its leading ids.
    #[test]
    fn a_shadow_scores_as_its_verify_would() {
        let clock = Clockwork::new(STEP_US, 3 * 60_500);
        let acc = Rc::new(Cell::new(0.7));
        let coin = Coin::<3>::new(&acc, Some(&clock));
        let gate = Gated::with_clock(coin, Mode::Auto, Rule::DEFAULT, Rc::clone(&clock)).unwrap();
        let mut spec = Speculative::<_, 4>::new(gate);
        let mut t = Mock::new().with_ctx(1_000_000).with_clock(clock);
        let prompt = [1, 2, 3, 1, 2];
        let mut last = spec.prompt(&mut t, &prompt).unwrap();
        spec.begin(&t, &prompt, last).unwrap();
        let mut out = Vec::new();
        for _ in 0..800 {
            out.clear();
            spec.pass(&mut t, last, &mut out).unwrap();
            last = *out.last().unwrap();
        }
        let g = spec.draft_mut();
        assert!(!g.is_open() && g.shadowed() >= 64, "{:?}", g.record());
        let scored = &g.draft().scored;
        for &(agreed, kept) in scored {
            if let Some(k) = kept {
                assert_eq!(k, agreed + 1, "a verify keeps row 0 and the agreed ids");
            }
        }
        let shadows: Vec<usize> = scored
            .iter()
            .filter(|s| s.1.is_none())
            .map(|s| s.0)
            .collect();
        let tail = &shadows[shadows.len() - 64..];
        let want = tail.iter().sum::<usize>() as f64 / 64.0;
        assert_eq!(g.record().acc_shadow, Some(want));
        assert!(tail.iter().any(|&n| n >= 2), "no shadow matched two ids");
    }

    /// Input the gate has no answer for is refused by name, never read as a
    /// default: a time of zero or not a number, an acceptance below 0 or not
    /// a number, a window of no passes, a rule of zero-pass windows, and a
    /// clock that does not move.
    #[test]
    fn undefined_input_is_a_named_error() {
        let t0 = GateError::Time {
            what: "step",
            ms: 0.0,
        };
        assert_eq!(gain(0.5, 60.0, 0.0), Err(t0));
        assert!(matches!(
            gain(0.5, f64::NAN, 37.0),
            Err(GateError::Time { what: "pass", .. })
        ));
        assert!(matches!(
            gain(0.5, f64::INFINITY, 37.0),
            Err(GateError::Time { what: "pass", .. })
        ));
        assert!(matches!(
            gain(f64::NAN, 60.0, 37.0),
            Err(GateError::Acceptance(_))
        ));
        assert_eq!(gain(-0.1, 60.0, 37.0), Err(GateError::Acceptance(-0.1)));
        assert_eq!(
            Tally::new("verified", 4).mean(),
            Err(GateError::EmptyWindow("verified"))
        );
        assert_eq!(
            Timings::new("pass", 4).median(),
            Err(GateError::EmptyWindow("pass"))
        );
        assert!(matches!(
            Timings::new("step", 4).push(0.0),
            Err(GateError::Time { what: "step", .. })
        ));
        for (rule, why) in [
            (
                Rule {
                    min_window: 0,
                    ..Rule::DEFAULT
                },
                "a window of zero passes",
            ),
            (
                Rule {
                    window: 8,
                    ..Rule::DEFAULT
                },
                "a window shorter than its decision minimum",
            ),
            (
                Rule {
                    min_timings: 0,
                    ..Rule::DEFAULT
                },
                "a cost of zero readings",
            ),
            (
                Rule {
                    open_above: 0.99,
                    ..Rule::DEFAULT
                },
                "the gain bounds must be finite, with 0 < close_below <= open_above",
            ),
            (
                Rule {
                    close_below: f64::NAN,
                    ..Rule::DEFAULT
                },
                "the gain bounds must be finite, with 0 < close_below <= open_above",
            ),
        ] {
            assert_eq!(rule.check(), Err(GateError::Rule(why)));
            let acc = Rc::new(Cell::new(0.5));
            let e = Gated::new(Coin::<1>::new(&acc, None), Mode::Auto, rule)
                .err()
                .expect("a rule it cannot run by");
            assert_eq!(e, GateError::Rule(why));
        }
        assert_eq!(Rule::DEFAULT.check(), Ok(()));
        // A clock that does not move: the first pass read is refused.
        let mut r = Rig::new(Mode::Auto, 0.5, 0);
        r.clock.step.set(0);
        let mut out = Vec::new();
        let mut last = r.last;
        let e = (0..10)
            .find_map(|_| {
                out.clear();
                match r.spec.pass(&mut r.t, last, &mut out) {
                    Ok(_) => {
                        last = out[out.len() - 1];
                        None
                    }
                    Err(e) => Some(e),
                }
            })
            .expect("a pass of no time is refused");
        assert!(matches!(e, MockError::Gate(GateError::Time { .. })), "{e}");
    }
}
