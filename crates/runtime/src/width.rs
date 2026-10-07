//! The width a drafted window verifies, chosen by measured cost and the
//! draft's own probabilities: [`Choosing`], the [`Draft`] wrapper a drafted
//! pass of every family goes through.
//!
//! A verify of `k + 1` rows keeps `E(k)` positions — row 0, then the draft
//! ids the target agrees with — and costs `cost(k)`, the measured wall of
//! such a pass; `k = 0` is the plain step, `E(0) = 1`. A width's rate is
//! `E(k) / cost(k)`:
//!
//! - `E(k) = 1 + Σ_{j<k} Π_{i≤j} a_i` over an acceptance profile `a`. A
//!   window's own profile is read from the proposal the draft scored:
//!   `a_i = p_i · g_i`, `p_i` the draft's own probability of its id
//!   ([`Draft::propose_p`]; a draft without probabilities reports 1) and
//!   `g_i` a calibration kept online. Each position a pass tested moves its
//!   `g_i` toward what happened, and a window whose ids were all kept pulls
//!   the untested positions' toward the deepest tested one, so a width that
//!   stopped being chosen can grow back. The mean profile is the running
//!   mean of the windows' `p_i` times the current `g_i`, so a calibration
//!   that moved moves it at once.
//! - `cost(k)` is the median of the last [`Rule::timings`] walls of passes
//!   that ran `k`: one clock reading a pass at its proposal, landed at the
//!   next (the proposal, the target's call, the draft's update and whatever
//!   the caller does before its next pass), kept only when exactly one
//!   accept or step came between and the target stands where it left it. A
//!   plain pass whose draft proposed (a shadow, a cut to 0) holds the draft's
//!   walk and is no reading of the plain step.
//!
//! Until every width's cost holds [`Rule::min_timings`] readings the passes
//! rotate the widths, widest first, and nothing is decided. Past that two
//! choices are made, each on its own profile:
//!
//! - the gate, on the mean profile: an open chooser closes as soon as the
//!   plain step's rate is above every drafted width's, a closed one opens
//!   only once a drafted width beats the plain step by [`Rule::margin`],
//!   and an open one's width moves to a rival only past the margin — one
//!   window's profile never moves it;
//! - while open, each window's width, on that window's own profile: the
//!   gate's width, or another its profile rates above it by the margin.
//!
//! After [`Rule::probe_every`] passes ([`Rule::probe_closed`] while closed)
//! the width read longest ago runs once, so no cost goes stale. A proposal
//! shorter than the draft's width (the context's end) moves neither the mean
//! nor the gate.
//!
//! The draft proposes its whole width whenever it proposes, and the pass
//! verifies the front `k` ids. While closed, a plain pass asks nothing of
//! the draft — it hears the position through [`Draft::held`] — but every
//! [`Rule::shadow_every`]th: that proposal runs as a shadow, scored against
//! the target's own next tokens and never verified, keeping the calibration
//! current so the chooser can open again. A proposal no row of which a pass
//! verifies (a shadow, a cut to 0) is [`Draft::unproposed`].
//!
//! A pass of several sequences' windows ([`Choosing::round_ran`]) runs the
//! draft's whole width and is only counted: its wall is the round's, no
//! width's.
//!
//! [`Mode::Fixed`] passes every call to the draft untouched: every proposal
//! verified whole, no clock read, nothing counted.
//!
//! The chooser changes which rows a pass runs, never a token: every kept
//! token is the target's argmax over the rows that ran.

use std::collections::VecDeque;
use std::fmt;
use std::time::{Duration, Instant};

use crate::{Draft, TapNeed, Verify};

/// A clock the chooser reads once a pass.
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

/// What the chooser does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Choose the width each window verifies by measured cost and the
    /// proposal's own probabilities.
    Cost,
    /// Verify every proposal at the draft's own width, the draft's calls
    /// untouched: the same-binary arm.
    Fixed,
}

impl Mode {
    /// `BLOOMERY_MTP_WIDTH`'s word as the levers hold it: unset is
    /// [`Mode::Cost`]; a word that is neither mode is refused by name.
    pub fn of(word: Option<&str>) -> Result<Mode, WidthError> {
        match word {
            None | Some("cost") => Ok(Mode::Cost),
            Some("fixed") => Ok(Mode::Fixed),
            Some(other) => Err(WidthError::Mode(other.to_string())),
        }
    }
}

/// The chooser's numbers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rule {
    /// A rival width takes the incumbent's seat when its rate beats the
    /// incumbent's by this factor, and a drafted width opens a closed
    /// chooser when it beats the plain step's by it; 1 is no hysteresis.
    pub margin: f64,
    /// Readings a width's cost median holds.
    pub timings: usize,
    /// Readings of every width's cost before the chooser compares rates.
    pub min_timings: usize,
    /// The calibration's step: how far a tested position's `g_i` moves
    /// toward what happened.
    pub calibrate: f64,
    /// Passes before the width read longest ago runs once, while open.
    pub probe_every: u32,
    /// The same while closed, when every probe is a drafted width's verify.
    pub probe_closed: u32,
    /// A closed chooser's shadow proposal after this many plain passes.
    pub shadow_every: u32,
}

impl Rule {
    /// A rival must win by 3 %; costs as the median of the last 8 readings,
    /// from 4 on; the calibration moves 5 % of the way a window; a probe
    /// after 64 passes open, 256 closed; a shadow every 16th closed pass.
    /// The periods price what they buy: a closed chooser pays a shadow's
    /// whole chain and a probe's verify where its plain pass is cheaper.
    pub const DEFAULT: Rule = Rule {
        margin: 1.03,
        timings: 8,
        min_timings: 4,
        calibrate: 0.05,
        probe_every: 64,
        probe_closed: 256,
        shadow_every: 16,
    };

    /// Refused by name unless the chooser can run by it.
    pub fn check(&self) -> Result<(), WidthError> {
        let refused = if !(self.margin.is_finite() && self.margin >= 1.0) {
            "the margin must be finite and at least 1"
        } else if self.min_timings == 0 {
            "a cost of zero readings"
        } else if self.timings < self.min_timings {
            "a cost's readings fewer than its decision minimum"
        } else if !(self.calibrate > 0.0 && self.calibrate <= 1.0) {
            "the calibration's step must be inside (0, 1]"
        } else if self.probe_every == 0 || self.probe_closed == 0 || self.shadow_every == 0 {
            "a period of zero passes"
        } else {
            return Ok(());
        };
        Err(WidthError::Rule(refused))
    }
}

/// Input the chooser has no answer for.
#[derive(Clone, Debug, PartialEq)]
pub enum WidthError {
    /// A [`Rule`] the chooser cannot run by.
    Rule(&'static str),
    /// A `BLOOMERY_MTP_WIDTH` word that is no mode.
    Mode(String),
    /// A time that is not a positive, finite number of milliseconds.
    Time { what: &'static str, ms: f64 },
    /// A probability that is not a finite number in `[0, 1]`.
    Probability(f32),
    /// A verify that kept no row: row 0 is always kept.
    NoRowKept,
}

impl fmt::Display for WidthError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WidthError::Rule(why) => write!(f, "draft width: refused rule: {why}"),
            WidthError::Mode(word) => write!(
                f,
                "BLOOMERY_MTP_WIDTH={word}: the width chooser's modes are cost and fixed"
            ),
            WidthError::Time { what, ms } => write!(
                f,
                "draft width: a {what} time of {ms} ms: a time is positive and finite"
            ),
            WidthError::Probability(p) => write!(
                f,
                "draft width: a proposal's probability of {p}: it is finite and in [0, 1]"
            ),
            WidthError::NoRowKept => f.write_str("draft width: a verify that kept no row"),
        }
    }
}

impl std::error::Error for WidthError {}

/// The last readings of one width's pass walls, in milliseconds.
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

    fn push(&mut self, ms: f64) -> Result<(), WidthError> {
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

    /// The median of the readings; `None` with none.
    fn median(&mut self) -> Option<f64> {
        if self.ring.is_empty() {
            return None;
        }
        self.sorted.clear();
        self.sorted.extend(self.ring.iter().copied());
        self.sorted.sort_by(f64::total_cmp);
        let n = self.sorted.len();
        Some(if n % 2 == 1 {
            self.sorted[n / 2]
        } else {
            f64::midpoint(self.sorted[n / 2 - 1], self.sorted[n / 2])
        })
    }
}

fn positive(what: &'static str, ms: f64) -> Result<f64, WidthError> {
    if ms.is_finite() && ms > 0.0 {
        Ok(ms)
    } else {
        Err(WidthError::Time { what, ms })
    }
}

/// Refused by name unless every probability is a finite number in `[0, 1]`.
fn check_p(p: &[f32]) -> Result<(), WidthError> {
    match p
        .iter()
        .find(|p| !(p.is_finite() && (0.0..=1.0).contains(*p)))
    {
        Some(&bad) => Err(WidthError::Probability(bad)),
        None => Ok(()),
    }
}

/// The positions a verify of `k + 1` rows keeps over the acceptance profile
/// `a`: `1 + Σ_{j < k} Π_{i ≤ j} a_i`, row 0 and then the accepted ids.
fn expected(a: &[f64], k: usize) -> f64 {
    let (mut e, mut run) = (1.0, 1.0);
    for &a_i in &a[..k.min(a.len())] {
        run *= a_i;
        e += run;
    }
    e
}

/// What a pass ran, as its proposal left it, for its wall's reading.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// A verify of a proposal: a reading of that width's cost.
    Verify,
    /// A plain step the draft walked nothing for: a reading of the step's.
    Step,
    /// A plain step after the draft's proposal (a shadow, a cut to 0): its
    /// wall holds the draft's walk, no reading.
    Walked,
}

/// The pass in flight since its proposal.
#[derive(Clone, Copy, Debug)]
struct Flight {
    kind: Kind,
    /// The ids the pass verifies, 0 for a plain step.
    width: usize,
    /// The target's position at the proposal.
    pos: u32,
    at: Duration,
    /// The position the pass's accept or step left the target at; `None`
    /// until one came.
    ends: Option<u32>,
}

/// What this pass's proposal left for its accept or step to tell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Turn {
    /// A verify of the front `k` ids of the draft's proposal.
    Verify { k: usize },
    /// A plain step of the chooser's: no width paid, a turn of width 0, a
    /// shadow's pass or a cut to 0.
    Plain,
}

/// A shadow proposal scored against the target's next tokens as they come;
/// `n` 0 is none.
#[derive(Clone, Debug)]
struct Shadow {
    /// The ids the draft proposed and each one's probability, `n` of each.
    ids: Vec<u32>,
    p: Vec<f32>,
    n: usize,
    /// Leading ids the target's tokens have matched so far.
    matched: usize,
}

/// The passes since the last take ([`Choosing::take_tally`]): what a
/// per-request draft record holds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tally {
    /// Passes that verified a proposal.
    pub windows: u64,
    /// The windows by the rows they kept: index `i` the windows that kept
    /// `i + 1`.
    pub kept: Vec<u64>,
    /// Every pass by the ids it verified: index 0 the plain steps.
    pub widths: Vec<u64>,
}

impl Tally {
    fn new(widths: usize) -> Tally {
        Tally {
            windows: 0,
            kept: vec![0; widths],
            widths: vec![0; widths],
        }
    }

    /// `E`: the mean rows a window kept; 0 with no window.
    #[must_use]
    pub fn e(&self) -> f64 {
        if self.windows == 0 {
            return 0.0;
        }
        let rows: u64 = (1..).zip(&self.kept).map(|(r, &n)| r * n).sum();
        rows as f64 / self.windows as f64
    }

    /// Passes counted.
    #[must_use]
    pub fn passes(&self) -> u64 {
        self.widths.iter().sum()
    }
}

/// A draft behind the width chooser (see the module): its proposals
/// verified at the width that pays, or as it proposes them
/// ([`Mode::Fixed`]).
#[derive(Debug)]
pub struct Choosing<D, C = Wall> {
    draft: D,
    clock: C,
    rule: Rule,
    mode: Mode,
    /// The gate: 0 closed; open, the width the mean profile holds (a
    /// window may verify another, [`Choosing::cut`]).
    width: usize,
    /// Passes since the last probe or switch.
    since_probe: u32,
    /// Closed plain passes since the last shadow.
    since_shadow: u32,
    passes: u64,
    /// The calibration `g_i`, one a proposal index.
    gains: Vec<f64>,
    /// The running mean of the draft's `p_i` over its full proposals, and
    /// how many it has seen.
    mean: Vec<f64>,
    seen: u64,
    /// The gate's profile, `mean_i · g_i`, reused.
    gated: Vec<f64>,
    /// The last readings of each width's cost: index `k` the passes that
    /// verified `k` ids, index 0 the plain steps.
    costs: Vec<Timings>,
    /// Each width's median, refreshed as its readings land; `None` while a
    /// cost holds fewer than [`Rule::min_timings`] readings.
    medians: Vec<Option<f64>>,
    /// The pass each width's cost was last read at.
    read_at: Vec<u64>,
    /// The warm-up rotation's widths whose readings are not all in.
    warm: VecDeque<usize>,
    /// The draft's proposal and its probabilities, reused: a pass allocates
    /// nothing.
    ids: Vec<u32>,
    p: Vec<f32>,
    /// The proposal's acceptance profile `a_i`.
    a: Vec<f64>,
    turn: Option<Turn>,
    flight: Option<Flight>,
    shadow: Shadow,
    tally: Tally,
}

/// A draft's entry to the chooser: `draft.choosing(mode)`, and the same by
/// a rule and a clock of its own.
pub trait Chosen<T: Verify>: Draft<T> + Sized {
    /// The draft behind the chooser of `mode` by [`Rule::DEFAULT`], on the
    /// host's clock.
    fn choosing(self, mode: Mode) -> Result<Choosing<Self, Wall>, WidthError>;

    /// [`Chosen::choosing`] by `rule`, on `clock`; refused by name for a
    /// rule the chooser cannot run by.
    fn choosing_by<C: Clock>(
        self,
        mode: Mode,
        rule: Rule,
        clock: C,
    ) -> Result<Choosing<Self, C>, WidthError>;
}

impl<T, D> Chosen<T> for D
where
    T: Verify,
    D: Draft<T>,
{
    fn choosing(self, mode: Mode) -> Result<Choosing<D, Wall>, WidthError> {
        self.choosing_by(mode, Rule::DEFAULT, Wall::new())
    }

    fn choosing_by<C: Clock>(
        self,
        mode: Mode,
        rule: Rule,
        clock: C,
    ) -> Result<Choosing<D, C>, WidthError> {
        rule.check()?;
        let widths = D::WIDTH + 1;
        Ok(Choosing {
            draft: self,
            clock,
            rule,
            mode,
            width: D::WIDTH,
            since_probe: 0,
            since_shadow: 0,
            passes: 0,
            gains: vec![1.0; D::WIDTH],
            mean: vec![0.0; D::WIDTH],
            seen: 0,
            gated: vec![0.0; D::WIDTH],
            costs: (0..widths)
                .map(|k| Timings::new(if k == 0 { "step" } else { "verify" }, rule.timings))
                .collect(),
            medians: vec![None; widths],
            read_at: vec![0; widths],
            warm: (0..widths).rev().collect(),
            ids: vec![0; D::WIDTH],
            p: vec![0.0; D::WIDTH],
            a: vec![0.0; D::WIDTH],
            turn: None,
            flight: None,
            shadow: Shadow {
                ids: vec![0; D::WIDTH],
                p: vec![0.0; D::WIDTH],
                n: 0,
                matched: 0,
            },
            tally: Tally::new(widths),
        })
    }
}

impl<D, C: Clock> Choosing<D, C> {
    /// The draft.
    pub fn draft(&self) -> &D {
        &self.draft
    }

    /// See [`Choosing::draft`].
    pub fn draft_mut(&mut self) -> &mut D {
        &mut self.draft
    }

    /// The chooser's mode.
    #[must_use]
    pub fn mode(&self) -> Mode {
        self.mode
    }

    /// The gate: 0 closed; open, the width the mean profile holds, which a
    /// window verifies unless its own profile rates another by the margin
    /// above it.
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    /// The calibration gains, one a proposal index; every 1 before anything
    /// ran.
    #[must_use]
    pub fn gains(&self) -> &[f64] {
        &self.gains
    }

    /// The passes since the last take, and the count started over.
    pub fn take_tally(&mut self) -> Tally {
        std::mem::replace(&mut self.tally, Tally::new(self.costs.len()))
    }

    /// A pass of several sequences' windows ran this draft's window (its
    /// [`Draft::propose`] and the round's record called on the draft itself,
    /// the chooser bypassed): `rows` ran, `kept` were kept. Counted; no cost
    /// read (the wall is the round's) and nothing in flight survives it.
    pub fn round_ran(&mut self, rows: usize, kept: usize) {
        if self.mode == Mode::Fixed {
            return;
        }
        self.drop_flight();
        self.ran(rows.saturating_sub(1), kept);
    }

    /// Forget the pass in flight and a pending shadow: nothing before a new
    /// generation, or outside the chooser's passes, is read as a pass.
    fn drop_flight(&mut self) {
        self.flight = None;
        self.turn = None;
        self.shadow.n = 0;
    }

    /// Close the pass in flight at `now`, the target at `pos`: its reading
    /// kept when it was heard once and the target stands where it left it.
    fn land(&mut self, now: Duration, pos: u32) -> Result<(), WidthError> {
        let Some(f) = self.flight.take() else {
            return Ok(());
        };
        if f.ends != Some(pos) {
            self.shadow.n = 0;
            return Ok(());
        }
        let ms = match now.checked_sub(f.at) {
            Some(d) => d.as_secs_f64() * 1e3,
            None => -(f.at - now).as_secs_f64() * 1e3,
        };
        if f.kind == Kind::Walked {
            positive("plain pass", ms)?;
            return Ok(());
        }
        let k = f.width;
        self.costs[k].push(ms)?;
        self.read_at[k] = self.passes;
        self.medians[k] = if self.costs[k].len() >= self.rule.min_timings {
            self.costs[k].median()
        } else {
            None
        };
        let warming = !self.warm.is_empty();
        while let Some(&w) = self.warm.front()
            && self.costs[w].len() >= self.rule.min_timings
        {
            self.warm.pop_front();
        }
        if warming && self.warm.is_empty() {
            // The first probe comes a whole period after the warm-up.
            self.since_probe = 0;
        }
        Ok(())
    }

    /// The rate of verifying `k` ids over the profile `a`; `None` while
    /// `k`'s cost is unread.
    fn rate(&self, a: &[f64], k: usize) -> Option<f64> {
        Some(expected(a, k) / self.medians[k]?)
    }

    /// The drafted width `a` rates highest, and its rate, among 1 to
    /// `a.len()`, the wider on a tie; `None` while a cost is unread.
    fn best(&self, a: &[f64]) -> Option<(f64, usize)> {
        let mut best: Option<(f64, usize)> = None;
        for k in 1..=a.len() {
            let r = self.rate(a, k)?;
            if best.is_none_or(|(b, _)| r >= b) {
                best = Some((r, k));
            }
        }
        best
    }

    /// The gate on the mean profile (the module's rule): an open chooser
    /// closes when the plain step's rate is above every drafted width's, a
    /// closed one opens when a drafted width beats the plain step by the
    /// margin, and an open one's width gives way to a rival that beats it by
    /// the margin. Nothing moves while the warm-up's readings are not all
    /// in.
    fn gate(&mut self) {
        if !self.warm.is_empty() {
            return;
        }
        let mut mean = std::mem::take(&mut self.gated);
        for ((a, &p), &g) in mean.iter_mut().zip(&self.mean).zip(&self.gains) {
            *a = (p * g).clamp(0.0, 1.0);
        }
        let held = self.rate(&mean, self.width);
        let decided = self
            .rate(&mean, 0)
            .zip(self.best(&mean))
            .map(|(plain, (best, b))| {
                let margin = self.rule.margin;
                match held.filter(|_| self.width > 0) {
                    None if best > plain * margin => b,
                    None => 0,
                    Some(_) if best < plain => 0,
                    Some(r) if b != self.width && best > r * margin => b,
                    Some(_) => self.width,
                }
            });
        self.gated = mean;
        if let Some(next) = decided
            && next != self.width
        {
            if (next == 0) != (self.width == 0) {
                self.since_probe = 0;
                self.since_shadow = 0;
            }
            self.width = next;
        }
    }

    /// The width this window verifies while open: the gate's, unless the
    /// window's own profile `a[..n]` rates another by the margin above it.
    fn cut(&self, n: usize) -> usize {
        let a = &self.a[..n];
        let w = self.width.min(n);
        match (self.rate(a, w), self.best(a)) {
            (Some(r), Some((best, b))) if b != w && best > r * self.rule.margin => b,
            _ => w,
        }
    }

    /// The proposal's probabilities into the mean: a running mean until the
    /// calibration's step is the smaller, then that step.
    fn learn(&mut self, n: usize) {
        self.seen += 1;
        let step = (1.0 / self.seen as f64).max(self.rule.calibrate);
        for (m, &p) in self.mean[..n].iter_mut().zip(&self.p) {
            *m += step * (f64::from(p) - *m);
        }
    }

    /// The width read longest ago, the incumbent aside.
    fn stalest(&self) -> usize {
        (0..self.costs.len())
            .filter(|&k| k != self.width)
            .min_by_key(|&k| self.read_at[k])
            .unwrap_or(self.width)
    }

    /// `a_i = p_i · g_i` for the proposal's first `n` ids, each in `[0, 1]`.
    fn profile(&mut self, n: usize) {
        for ((a, &p), &g) in self.a[..n].iter_mut().zip(&self.p).zip(&self.gains) {
            *a = (f64::from(p) * g).clamp(0.0, 1.0);
        }
    }

    /// The calibration of a pass that tested `k` ids and kept `j` of them
    /// (the module): each tested position's gain moves toward its hit — a
    /// hit for `i < j`, the miss at `i = j` when `j < k` — and when all `k`
    /// were kept the positions past them move toward the deepest tested
    /// one's.
    fn calibrate(&mut self, k: usize, j: usize) {
        let step = self.rule.calibrate;
        let tested = (j + usize::from(j < k)).min(self.gains.len());
        for i in 0..tested {
            let hit = if i < j { 1.0 } else { 0.0 };
            let a = (f64::from(self.p[i]) * self.gains[i]).clamp(0.0, 1.0);
            self.gains[i] += step * (hit - a);
        }
        if j == k && k > 0 {
            let deepest = self.gains[k - 1];
            for g in &mut self.gains[k..] {
                *g += step * (deepest - *g);
            }
        }
    }

    /// One pass counted: the ids it verified and the rows it kept.
    fn ran(&mut self, width: usize, kept: usize) {
        self.tally.widths[width] += 1;
        if width > 0 {
            self.tally.windows += 1;
            self.tally.kept[kept - 1] += 1;
        }
    }

    /// The pending shadow's next id against `next`, the target's own token:
    /// its match advanced, and at its end the calibration told of it, its
    /// probabilities into the mean and the gate decided on the mean.
    fn score_shadow(&mut self, next: u32) {
        let s = &mut self.shadow;
        if s.n == 0 {
            return;
        }
        if s.ids[s.matched] == next {
            s.matched += 1;
            if s.matched < s.n {
                return;
            }
        }
        let (n, j) = (s.n, s.matched);
        s.n = 0;
        self.p[..n].copy_from_slice(&self.shadow.p[..n]);
        self.calibrate(n, j);
        if n == self.gains.len() {
            self.learn(n);
            self.gate();
        }
    }

    /// This pass a plain step of `kind`.
    fn plain(&mut self, kind: Kind, pos: u32, at: Duration) -> usize {
        self.flight = Some(Flight {
            kind,
            width: 0,
            pos,
            at,
            ends: None,
        });
        self.turn = Some(Turn::Plain);
        0
    }

    /// This pass a verify of the proposal's front `k` ids, written to `out`.
    fn verify(&mut self, out: &mut [u32], k: usize, pos: u32, at: Duration) -> usize {
        out[..k].copy_from_slice(&self.ids[..k]);
        self.flight = Some(Flight {
            kind: Kind::Verify,
            width: k,
            pos,
            at,
            ends: None,
        });
        self.turn = Some(Turn::Verify { k });
        k
    }
}

/// The draft's proposals, verified at the width that pays.
impl<T, D, C> Draft<T> for Choosing<D, C>
where
    T: Verify,
    T::Error: From<WidthError>,
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

    /// One pass's proposal (the module): a turn that does not read the
    /// proposal first — a shadow in flight, the warm-up's rotation, a probe
    /// — then while closed a plain step or a shadow, else the draft's own
    /// proposal with its probabilities, the gate decided on the mean they
    /// move, and the front ids of the window's width ([`Choosing::cut`]) as
    /// the pass's rows.
    fn propose(&mut self, t: &mut T, last: u32, out: &mut [u32]) -> Result<usize, T::Error> {
        if self.mode == Mode::Fixed {
            return self.draft.propose(t, last, out);
        }
        let at = self.clock.now();
        let pos = t.pos();
        self.land(at, pos)?;
        self.passes += 1;
        self.since_probe = self.since_probe.saturating_add(1);
        let room = out.len().min(D::WIDTH);
        if self.shadow.n > 0 {
            return Ok(self.plain(Kind::Step, pos, at));
        }
        let period = if self.width == 0 {
            self.rule.probe_closed
        } else {
            self.rule.probe_every
        };
        let forced = if let Some(&k) = self.warm.front() {
            Some(k)
        } else if self.since_probe >= period {
            self.since_probe = 0;
            Some(self.stalest())
        } else {
            None
        };
        if forced == Some(0) || (forced.is_none() && self.width == 0) {
            let shadow = forced.is_none() && {
                self.since_shadow += 1;
                self.since_shadow >= self.rule.shadow_every
            };
            if !shadow {
                return Ok(self.plain(Kind::Step, pos, at));
            }
            self.since_shadow = 0;
            let n = self
                .draft
                .propose_p(t, last, &mut self.ids[..room], &mut self.p)?;
            if n == 0 {
                return Ok(self.plain(Kind::Step, pos, at));
            }
            check_p(&self.p[..n])?;
            self.shadow.ids[..n].copy_from_slice(&self.ids[..n]);
            self.shadow.p[..n].copy_from_slice(&self.p[..n]);
            (self.shadow.n, self.shadow.matched) = (n, 0);
            self.draft.unproposed();
            return Ok(self.plain(Kind::Walked, pos, at));
        }
        let n = self
            .draft
            .propose_p(t, last, &mut self.ids[..room], &mut self.p)?;
        if n == 0 {
            return Ok(self.plain(Kind::Step, pos, at));
        }
        check_p(&self.p[..n])?;
        self.profile(n);
        if n == D::WIDTH {
            self.learn(n);
        }
        let k = match forced {
            Some(k) => k.min(n),
            None => {
                if n == D::WIDTH {
                    self.gate();
                }
                if self.width == 0 { 0 } else { self.cut(n) }
            }
        };
        if k == 0 {
            self.draft.unproposed();
            return Ok(self.plain(Kind::Walked, pos, at));
        }
        Ok(self.verify(out, k, pos, at))
    }

    /// The verify of `rows` kept its first `accepted` rows: the cost and the
    /// calibration told of the width that ran, then the draft's own accept.
    fn accept(
        &mut self,
        t: &mut T,
        rows: &[u32],
        out: &[u32],
        accepted: usize,
    ) -> Result<(), T::Error> {
        if self.mode == Mode::Fixed {
            return self.draft.accept(t, rows, out, accepted);
        }
        let kept = accepted.checked_sub(1).ok_or(WidthError::NoRowKept)?;
        let k = rows.len() - 1;
        match (self.turn.take(), &mut self.flight) {
            (Some(Turn::Verify { k: chose }), Some(f))
                if chose == k && f.kind == Kind::Verify && f.ends.is_none() =>
            {
                f.ends = u32::try_from(accepted).ok().map(|a| f.pos + a);
                self.calibrate(k, kept);
            }
            _ => self.drop_flight(),
        }
        self.ran(k, accepted);
        self.draft.accept(t, rows, out, accepted)
    }

    /// A step: the chooser's plain pass, counted, a pending shadow scored
    /// against `next`, the draft told it was held back; any other step
    /// (the caller's own, outside a pass) the draft's plain
    /// [`Draft::stepped`], and nothing in flight survives it.
    fn stepped(&mut self, t: &mut T, last: u32, next: u32) -> Result<(), T::Error> {
        if self.mode == Mode::Fixed {
            return self.draft.stepped(t, last, next);
        }
        if self.turn.take() != Some(Turn::Plain) {
            self.drop_flight();
            return self.draft.stepped(t, last, next);
        }
        match &mut self.flight {
            Some(f) if f.kind != Kind::Verify && f.ends.is_none() => f.ends = Some(f.pos + 1),
            _ => self.drop_flight(),
        }
        self.score_shadow(next);
        self.ran(0, 1);
        self.draft.held(t, last, next)
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::rc::Rc;

    use super::{Choosing, Chosen, Mode, Rule, Tally, Wall, WidthError, expected};
    use crate::mock::{Call, Clockwork, Mock, MockError, Quiet};
    use crate::{Advance, Committed, Draft, Plain, Speculative, Stop, TapNeed, Target, generate};

    /// The draft's own walk on the mock's clock, µs, every proposal.
    const DRAFT_US: u64 = 500;
    const STEP_US: u64 = 37_450;
    /// Verifies of 2, 3 and 4 rows at W(3) = 2.15 of the step, the rows
    /// between on a line [derived]: 1.383, 1.767, 2.15.
    const LINEAR: [u64; 3] = [51_800, 66_160, 80_520];

    /// A draft of 3 ids whose id `i` is the target's own next token at the
    /// rate `acc()[i]` and an id the target will not take otherwise, and
    /// whose reported probability of it is `p()[i]`. Each position draws on
    /// a counter of its own, `⌊(k + 1)·a⌋ > ⌊k·a⌋` making its stream hold
    /// exactly its rate over any long run. Like the engine's card draft it
    /// refuses to propose unless it holds every position before the
    /// target's, and it queues a held position until its next call.
    struct Scored {
        acc: Rc<Cell<[f64; 3]>>,
        p: Rc<Cell<[f32; 3]>>,
        draws: [u64; 3],
        clock: Option<Rc<Clockwork>>,
        at: u32,
        queued: u32,
        proposals: u64,
        held: u64,
        unproposed: u64,
        /// Proposals waiting for their accept: one a proposal, dropped by
        /// `unproposed`.
        open: u64,
    }

    impl Scored {
        fn new(acc: &Rc<Cell<[f64; 3]>>, p: &Rc<Cell<[f32; 3]>>) -> Scored {
            Scored {
                acc: Rc::clone(acc),
                p: Rc::clone(p),
                draws: [0; 3],
                clock: None,
                at: 0,
                queued: 0,
                proposals: 0,
                held: 0,
                unproposed: 0,
                open: 0,
            }
        }

        fn right(&mut self, i: usize) -> bool {
            let a = self.acc.get()[i];
            let k = self.draws[i] as f64;
            self.draws[i] += 1;
            ((k + 1.0) * a).floor() > (k * a).floor()
        }

        fn flush(&mut self) {
            self.at += self.queued;
            self.queued = 0;
        }
    }

    impl Draft<Mock> for Scored {
        const WIDTH: usize = 3;
        const TAPS: TapNeed = TapNeed::None;

        fn prompt(&mut self, t: &mut Mock, ids: &[u32]) -> Result<u32, MockError> {
            let next = Plain.prompt(t, ids)?;
            (self.at, self.queued, self.open) = (t.pos(), 0, 0);
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
            let mut p = [0.0f32; 3];
            self.propose_p(t, last, out, &mut p)
        }

        fn propose_p(
            &mut self,
            t: &mut Mock,
            last: u32,
            out: &mut [u32],
            p: &mut [f32],
        ) -> Result<usize, MockError> {
            self.flush();
            assert_eq!(self.at, t.pos(), "the draft missed a position");
            assert_eq!(self.open, 0, "a proposal never verified was not dropped");
            if let Some(c) = &self.clock {
                c.advance(DRAFT_US);
            }
            self.proposals += 1;
            self.open += 1;
            let truth = t.greedy_after(last, 3);
            let n = out.len().min(3);
            for i in 0..n {
                out[i] = if self.right(i) {
                    truth[i]
                } else {
                    (truth[i] + 1) % 5
                };
                p[i] = self.p.get()[i];
            }
            Ok(n)
        }

        fn unproposed(&mut self) {
            self.unproposed += 1;
            self.open -= 1;
        }

        fn accept(
            &mut self,
            _t: &mut Mock,
            _rows: &[u32],
            _out: &[u32],
            accepted: usize,
        ) -> Result<(), MockError> {
            self.flush();
            self.open -= 1;
            self.at += u32::try_from(accepted).unwrap();
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

    type Spec = Speculative<Choosing<Scored, Rc<Clockwork>>, 4>;

    /// A chooser over a [`Scored`] draft on the mock, standing after a
    /// prompt, its steps costing `STEP_US` and its verifies `by_rows`.
    struct Rig {
        t: Mock,
        spec: Spec,
        clock: Rc<Clockwork>,
        acc: Rc<Cell<[f64; 3]>>,
        last: u32,
        /// Every pass's outcome.
        passes: Vec<Committed>,
    }

    impl Rig {
        /// The reported probabilities equal the acceptances: a calibrated
        /// draft.
        fn of(acc: [f64; 3], by_rows: &[u64]) -> Rig {
            let p = [acc[0] as f32, acc[1] as f32, acc[2] as f32];
            Rig::with(acc, p, by_rows, Mode::Cost, Rule::DEFAULT)
        }

        fn with(acc: [f64; 3], p: [f32; 3], by_rows: &[u64], mode: Mode, rule: Rule) -> Rig {
            let clock = Clockwork::new(STEP_US, 0);
            clock.verify_rows(by_rows);
            let acc = Rc::new(Cell::new(acc));
            let p = Rc::new(Cell::new(p));
            let draft = Scored {
                clock: Some(Rc::clone(&clock)),
                ..Scored::new(&acc, &p)
            };
            let choosing = draft.choosing_by(mode, rule, Rc::clone(&clock)).unwrap();
            let mut spec = Speculative::new(choosing);
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
                passes: Vec::new(),
            }
        }

        fn run(&mut self, n: usize) {
            let mut out = Vec::with_capacity(4);
            for _ in 0..n {
                out.clear();
                let c = self.spec.pass(&mut self.t, self.last, &mut out).unwrap();
                self.passes.push(c);
                self.last = *out.last().unwrap();
            }
        }

        fn choosing(&mut self) -> &mut Choosing<Scored, Rc<Clockwork>> {
            self.spec.draft_mut()
        }

        /// The rows of every call the target ran, a step's 1.
        fn rows(&self) -> Vec<usize> {
            self.t
                .calls()
                .iter()
                .filter_map(|c| match *c {
                    Call::Step(_) => Some(1),
                    Call::Verify(_, m) => Some(m),
                    Call::Prompt(..) | Call::Commit(_) => None,
                })
                .collect()
        }
    }

    /// The gate a long run settles on, the passes by width, and the
    /// positions a microsecond of the mock's clock over its last 500 passes.
    fn settles(r: &mut Rig, n: usize) -> (usize, Vec<u64>, f64) {
        r.run(n - 500);
        let (t0, p0) = (r.clock.now.get(), r.t.pos());
        r.run(500);
        let rate = f64::from(r.t.pos() - p0) / (r.clock.now.get() - t0) as f64;
        let tally = r.choosing().take_tally();
        (r.choosing().width(), tally.widths, rate)
    }

    /// The best rate a fixed width gets on the rig: positions a microsecond
    /// of verifying `k` ids every pass, the draft's ids drawn apart.
    fn fixed_rate(acc: [f64; 3], by_rows: [u64; 3], k: usize) -> f64 {
        if k == 0 {
            return 1.0 / STEP_US as f64;
        }
        let (mut e, mut run) = (1.0, 1.0);
        for a in &acc[..k] {
            run *= a;
            e += run;
        }
        e / (by_rows[k - 1] + DRAFT_US) as f64
    }

    /// The chooser settles where the rates say, over the whole range, from
    /// synthetic walls and probabilities: the gate open and its mean on
    /// k = 3 for our prose (E(3) 3.0 at W 2.15, k = 2 within 0.6 % of it), 2
    /// for a profile whose third id loses, 1 for our Korean chat (E(3) 2.3),
    /// closed (k = 0) for a draft no width of which beats the plain step —
    /// and in each the positions a microsecond within 2 % of the best fixed
    /// width's, within 1 % on prose (the warm-up past, the probes' and the
    /// shadows' cost in).
    #[test]
    fn picks_the_width_that_pays() {
        let losing = [71_000, 95_000, 118_000];
        for (acc, by_rows, want, within, what) in [
            ([0.80, 0.8125, 0.846], LINEAR, 3, 0.99, "prose"),
            ([0.95, 0.90, 0.30], LINEAR, 2, 0.98, "a losing third id"),
            ([0.75, 0.55, 0.35], LINEAR, 1, 0.98, "chat"),
            ([0.40, 0.20, 0.10], losing, 0, 0.98, "a losing draft"),
        ] {
            let mut r = Rig::of(acc, &by_rows);
            let (width, widths, rate) = settles(&mut r, 1_000);
            assert_eq!(width, want, "{what}: settled on {width}, passes {widths:?}");
            let best = (0..=3)
                .map(|k| fixed_rate(acc, by_rows, k))
                .fold(0.0, f64::max);
            assert!(
                rate >= best * within,
                "{what}: {:.4} of the best fixed width's rate, passes {widths:?}",
                rate / best
            );
        }
    }

    /// The gate moves on the mean profile alone and keeps its margin: an
    /// open chooser stays open while a drafted width beats the plain step
    /// and closes once none does, a closed one opens only past the margin,
    /// and one low window does not move it; while open, each window
    /// verifies the width its own profile rates highest. Rates from fixed
    /// walls and profiles, no run.
    #[test]
    fn the_gate_keeps_its_margin_and_a_window_its_own_width() {
        let acc = Rc::new(Cell::new([0.8; 3]));
        let p = Rc::new(Cell::new([0.8; 3]));
        let mut c = Scored::new(&acc, &p)
            .choosing_by(Mode::Cost, Rule::DEFAULT, Wall::new())
            .unwrap();
        c.warm.clear();
        c.medians = vec![Some(1.0), Some(1.6), Some(1.85), Some(2.1)];
        let gate = |c: &mut Choosing<Scored>, width: usize, mean: [f64; 3]| {
            c.width = width;
            c.mean.copy_from_slice(&mean);
            c.gate();
            c.width
        };
        // The best drafted width 1 % above the plain step: open stays open,
        // closed stays closed; 1 % below it, open closes.
        let thin = [0.616, 0.0, 0.0];
        assert_eq!(gate(&mut c, 1, thin), 1, "closed while a width still paid");
        assert_eq!(gate(&mut c, 0, thin), 0, "opened inside the margin");
        assert_eq!(
            gate(&mut c, 1, [0.584, 0.0, 0.0]),
            0,
            "stayed open while it lost"
        );
        assert_eq!(
            gate(&mut c, 0, [0.70, 0.0, 0.0]),
            1,
            "did not open past the margin"
        );
        // One low window into a settled mean: the gate stays open.
        let window = |c: &mut Choosing<Scored>, p: [f32; 3]| {
            c.p.copy_from_slice(&p);
            c.profile(3);
            c.learn(3);
            c.gate();
            c.cut(3)
        };
        c.width = 3;
        c.mean.copy_from_slice(&[0.8, 0.81, 0.85]);
        c.seen = 1_000;
        assert_eq!(window(&mut c, [0.9, 0.9, 0.9]), 3, "a sure window cut");
        assert_eq!(
            window(&mut c, [0.5, 0.2, 0.2]),
            1,
            "an unsure window ran wide"
        );
        assert_eq!(window(&mut c, [0.05, 0.05, 0.05]), 1);
        assert_eq!(c.width(), 3, "one low window moved the gate");
    }

    /// A calibration that drifts follows the measured acceptance: the draft
    /// reports p and the target accepts at acc, and the gain of position 0
    /// — the one every window tests — tracks acc/p as it moves.
    #[test]
    fn a_drifting_calibration_follows_the_acceptance() {
        let mut r = Rig::with(
            [0.90, 0.90, 0.90],
            [0.5, 0.5, 0.5],
            &LINEAR,
            Mode::Cost,
            Rule::DEFAULT,
        );
        r.run(300);
        let g0 = r.choosing().gains()[0];
        assert!(
            (1.6..2.0).contains(&g0),
            "the gain missed acc/p = 1.8: {g0}"
        );
        r.acc.set([0.25, 0.25, 0.25]);
        r.run(300);
        let g0 = r.choosing().gains()[0];
        assert!(
            (0.3..0.7).contains(&g0),
            "the gain missed acc/p = 0.5: {g0}"
        );
    }

    /// No decision is taken on fewer than the minimum readings: until every
    /// width's cost holds them the passes rotate the widths, widest first,
    /// whatever the rates say; past that the winner takes over.
    #[test]
    fn no_decision_before_the_minimum_readings() {
        let mut r = Rig::of([0.75, 0.55, 0.35], &LINEAR);
        r.run(16);
        assert_eq!(
            r.rows(),
            [4, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1],
            "the rotation did not run every width in turn"
        );
        assert_eq!(r.choosing().width(), 3, "a decision before the readings");
        r.run(64);
        assert_eq!(r.choosing().width(), 1, "the winner did not take over");
    }

    /// While closed the chooser shadows the draft's proposals: a shadow is
    /// scored against the target's own tokens and never verified, the draft
    /// drops it (`unproposed`), the calibration learns from it, and the
    /// draft hears every plain pass as held.
    #[test]
    fn a_shadow_is_scored_never_verified() {
        // The draft over-reports (p above the acceptance): the calibration
        // settles under 1.
        let mut r = Rig::with(
            [0.40, 0.20, 0.10],
            [0.6, 0.5, 0.4],
            &[71_000, 95_000, 118_000],
            Mode::Cost,
            Rule {
                shadow_every: 4,
                ..Rule::DEFAULT
            },
        );
        r.run(400);
        assert_eq!(r.choosing().width(), 0, "the losing draft did not close");
        let verifies = r.passes.iter().filter(|c| c.proposed).count() as u64;
        let d = r.spec.draft().draft();
        assert!(d.unproposed > 30, "few shadows ran: {}", d.unproposed);
        assert_eq!(
            d.proposals,
            verifies + d.unproposed,
            "a proposal neither verified nor dropped"
        );
        assert!(d.held > 200, "the closed passes were not held: {}", d.held);
        let g = r.choosing().gains()[0];
        assert!(g < 0.9, "the calibration did not follow the shadows: {g}");
    }

    /// Every pass is counted once, whatever ran it — a verify, a plain step,
    /// a shadow's pass, a pass while a shadow resolves — and a window's kept
    /// rows are its verify's.
    #[test]
    fn the_tally_counts_every_pass_once() {
        let mut r = Rig::with(
            [0.85, 0.30, 0.20],
            [0.5, 0.5, 0.5],
            &[71_000, 95_000, 118_000],
            Mode::Cost,
            Rule {
                shadow_every: 1,
                ..Rule::DEFAULT
            },
        );
        r.run(500);
        let t = r.choosing().take_tally();
        assert_eq!(t.passes(), 500, "passes miscounted: {t:?}");
        let mut widths = vec![0u64; 4];
        let mut kept = vec![0u64; 4];
        for c in &r.passes {
            widths[c.rows - 1] += 1;
            if c.proposed {
                kept[c.kept - 1] += 1;
            }
        }
        assert_eq!(t.widths, widths);
        assert_eq!(t.kept, kept);
        assert!(t.widths[0] > 0 && t.windows > 0, "{t:?}");
    }

    /// After the warm-up a probe runs the width read longest ago once every
    /// `probe_every` passes, the incumbent aside.
    #[test]
    fn a_probe_runs_the_stalest_width() {
        let mut r = Rig::with(
            [0.80, 0.8125, 0.846],
            [0.80, 0.8125, 0.846],
            &LINEAR,
            Mode::Cost,
            Rule {
                probe_every: 8,
                ..Rule::DEFAULT
            },
        );
        r.run(16);
        let from = r.rows().len();
        r.run(48);
        let rows = r.rows()[from..].to_vec();
        let off: Vec<usize> = rows.iter().copied().filter(|&m| m != 4).collect();
        // The rotation read 2, 1 and 0 in that order: the probes run them
        // again in it.
        assert_eq!(
            off,
            [3, 2, 1, 3, 2, 1],
            "the probes did not cycle the stale widths: {rows:?}"
        );
    }

    /// The chooser changes which rows a pass runs and never a token: over a
    /// run that drafts wide, closes and opens again, the tokens are the plain
    /// run's.
    #[test]
    fn the_tokens_are_the_plain_runs_whatever_it_cuts() {
        let prompt = [1, 2, 3, 1, 2];
        let stop = Stop::new(1_200, 1_000_000).unwrap();
        let mut t = Mock::new().with_ctx(1_000_000);
        let first = Plain.prompt(&mut t, &prompt).unwrap();
        let plain = generate(&mut t, &mut Plain, &prompt, first, &stop, &mut Quiet).unwrap();

        let mut r = Rig::with(
            [0.85, 0.85, 0.85],
            [0.85, 0.85, 0.85],
            &LINEAR,
            Mode::Cost,
            Rule::DEFAULT,
        );
        let mut ids = vec![r.last];
        let mut seen = Vec::new();
        while ids.len() < 1_200 {
            let at = ids.len();
            let acc = if (400..800).contains(&at) {
                [0.30, 0.25, 0.20]
            } else {
                [0.85, 0.85, 0.85]
            };
            r.acc.set(acc);
            let last = ids[at - 1];
            r.spec.pass(&mut r.t, last, &mut ids).unwrap();
            assert!(ids.len() > at, "a pass that kept nothing after {at} ids");
            seen.push((at, r.choosing().width()));
        }
        assert_eq!(ids[..1_200], plain.tokens[..], "a cut pass changed a token");
        let at = |lo: usize, hi: usize| {
            seen.iter()
                .filter(|(i, _)| (lo..hi).contains(i))
                .map(|&(_, w)| w)
                .collect::<Vec<_>>()
        };
        assert!(at(300, 400).contains(&3), "never drafted wide");
        assert!(at(600, 800).contains(&0), "never closed");
        assert!(at(1_100, 1_200).contains(&3), "never opened again");
    }

    /// Fixed mode passes every call through: every proposal verified whole
    /// whatever the walls say, the tokens the plain run's, the clock never
    /// read (a clock that does not move refuses nothing) and nothing counted.
    #[test]
    fn fixed_mode_verifies_every_proposal_whole() {
        let mut r = Rig::with(
            [0.40, 0.20, 0.10],
            [0.4, 0.2, 0.1],
            &[71_000, 95_000, 118_000],
            Mode::Fixed,
            Rule::DEFAULT,
        );
        r.clock.step.set(0);
        r.clock.verify_rows(&[0, 0, 0]);
        r.run(64);
        assert!(
            r.passes.iter().all(|c| c.proposed && c.rows == 4),
            "a fixed pass ran a cut window: {:?}",
            r.passes.iter().map(|c| c.rows).collect::<Vec<_>>()
        );
        assert_eq!(r.choosing().take_tally().passes(), 0);

        let prompt = [1, 2, 3, 1, 2];
        let stop = Stop::new(64, 1_000_000).unwrap();
        let mut t = Mock::new().with_ctx(1_000_000);
        let first = Plain.prompt(&mut t, &prompt).unwrap();
        let plain = generate(&mut t, &mut Plain, &prompt, first, &stop, &mut Quiet).unwrap();
        let mut t = Mock::new().with_ctx(1_000_000);
        let first = r.spec.prompt(&mut t, &prompt).unwrap();
        let out = generate(&mut t, &mut r.spec, &prompt, first, &stop, &mut Quiet).unwrap();
        assert_eq!(out.tokens[..64], plain.tokens[..64]);
    }

    /// A round's window is counted at the rows it ran, and drops what was in
    /// flight — a pending shadow among it — so the next pass reads nothing
    /// of the round's wall.
    #[test]
    fn a_round_window_is_counted_and_drops_the_flight() {
        let mut r = Rig::with(
            [0.40, 0.20, 0.10],
            [0.6, 0.5, 0.4],
            &[71_000, 95_000, 118_000],
            Mode::Cost,
            Rule {
                shadow_every: 1,
                ..Rule::DEFAULT
            },
        );
        r.run(200);
        assert_eq!(r.choosing().width(), 0);
        for _ in 0..50 {
            if r.choosing().shadow.n > 0 {
                break;
            }
            r.run(1);
        }
        assert!(r.choosing().shadow.n > 0, "no shadow pending");
        let _ = r.choosing().take_tally();
        r.choosing().round_ran(4, 2);
        assert_eq!(r.choosing().shadow.n, 0, "the round left the shadow up");
        assert!(r.choosing().flight.is_none());
        let t = r.choosing().take_tally();
        assert_eq!((t.windows, t.widths[3], t.kept[1]), (1, 1, 1), "{t:?}");
    }

    /// Input the chooser has no answer for is refused by name, never read as
    /// a default: a rule it cannot run by, a mode word it does not take, a
    /// time of zero, a probability that is not one, a verify that kept no
    /// row.
    #[test]
    fn undefined_input_is_a_named_error() {
        assert_eq!(Rule::DEFAULT.check(), Ok(()));
        for (rule, why) in [
            (
                Rule {
                    margin: 0.99,
                    ..Rule::DEFAULT
                },
                "the margin must be finite and at least 1",
            ),
            (
                Rule {
                    margin: f64::NAN,
                    ..Rule::DEFAULT
                },
                "the margin must be finite and at least 1",
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
                    timings: 2,
                    ..Rule::DEFAULT
                },
                "a cost's readings fewer than its decision minimum",
            ),
            (
                Rule {
                    calibrate: 0.0,
                    ..Rule::DEFAULT
                },
                "the calibration's step must be inside (0, 1]",
            ),
            (
                Rule {
                    probe_every: 0,
                    ..Rule::DEFAULT
                },
                "a period of zero passes",
            ),
        ] {
            assert_eq!(rule.check(), Err(WidthError::Rule(why)));
            let acc = Rc::new(Cell::new([0.8; 3]));
            let p = Rc::new(Cell::new([0.8; 3]));
            assert!(
                Scored::new(&acc, &p)
                    .choosing_by(Mode::Cost, rule, Wall::new())
                    .is_err()
            );
        }
        assert_eq!(Mode::of(None), Ok(Mode::Cost));
        assert_eq!(Mode::of(Some("fixed")), Ok(Mode::Fixed));
        assert_eq!(
            Mode::of(Some("wide")),
            Err(WidthError::Mode("wide".to_string()))
        );

        // A clock that does not move: the first reading is refused.
        let mut r = Rig::of([0.8, 0.8, 0.8], &LINEAR);
        r.clock.step.set(0);
        r.clock.verify_rows(&[0, 0, 0]);
        let mut out = Vec::new();
        let e = (0..20)
            .find_map(|_| {
                out.clear();
                match r.spec.pass(&mut r.t, r.last, &mut out) {
                    Ok(_) => {
                        r.last = *out.last().unwrap();
                        None
                    }
                    Err(e) => Some(e),
                }
            })
            .expect("a pass of no time is refused");
        assert!(
            matches!(e, MockError::Width(WidthError::Time { .. })),
            "{e}"
        );

        // A probability outside [0, 1].
        let mut r = Rig::with(
            [0.8; 3],
            [f32::NAN, 0.5, 0.5],
            &LINEAR,
            Mode::Cost,
            Rule::DEFAULT,
        );
        let mut out = Vec::new();
        let e = r.spec.pass(&mut r.t, r.last, &mut out).unwrap_err();
        assert!(
            matches!(e, MockError::Width(WidthError::Probability(p)) if p.is_nan()),
            "{e}"
        );

        // A verify that kept no row.
        let mut r = Rig::of([0.8, 0.8, 0.8], &LINEAR);
        let mut out = [0u32; 3];
        let n = r
            .spec
            .draft_mut()
            .propose(&mut r.t, r.last, &mut out)
            .unwrap();
        assert_eq!(n, 3);
        let rows = [r.last, out[0], out[1], out[2]];
        let e = r
            .spec
            .draft_mut()
            .accept(&mut r.t, &rows, &[0; 4], 0)
            .unwrap_err();
        assert!(matches!(e, MockError::Width(WidthError::NoRowKept)), "{e}");
    }

    /// `E(k)` is row 0 and the accepted ids, and the tally's `E` the mean
    /// rows a window kept.
    #[test]
    fn the_expected_and_tally_arithmetic() {
        let a = [0.8, 0.5, 0.25];
        let e: Vec<f64> = (0..=3).map(|k| expected(&a, k)).collect();
        assert_eq!(e[0], 1.0);
        assert!((e[1] - 1.8).abs() < 1e-12);
        assert!((e[2] - 2.2).abs() < 1e-12);
        assert!((e[3] - 2.3).abs() < 1e-12);
        let t = Tally {
            windows: 3,
            kept: vec![1, 1, 0, 1],
            widths: vec![5, 0, 1, 2],
        };
        assert!((t.e() - 7.0 / 3.0).abs() < 1e-12);
        assert_eq!(t.passes(), 8);
        assert_eq!(Tally::new(4).e(), 0.0);
    }
}
