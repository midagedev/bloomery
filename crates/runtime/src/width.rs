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
//!   `g_i` toward what happened (by a fixed step, or under [`Rule::prior`]
//!   by a running mean's until that is smaller), and a window whose ids were
//!   all kept pulls the untested positions' toward the deepest tested one,
//!   so a width that stopped being chosen can grow back. The mean profile
//!   is the mean of each `p_i` conditional on the ids before it (the
//!   running mean of `Π_{j≤i} p_j` over that of `Π_{j<i} p_j`, over the
//!   windows that reached `i`) times the current `g_i`, so a calibration
//!   that moved moves it at once.
//! - `cost(k)` is the median of the last [`Rule::timings`] walls of passes
//!   that ran `k`: one clock reading a pass at its proposal, landed at the
//!   next (the proposal, the target's call, the draft's update and whatever
//!   the caller does before its next pass), kept only when exactly one
//!   accept or step came between and the target stands where it left it. A
//!   caller that runs other sequences' passes between a sequence's own lands
//!   each pass at its end instead ([`Choosing::finish_pass`]), so no other
//!   call is read as its wall. A plain pass whose draft proposed (a shadow,
//!   a cut to 0) holds the draft's walk and is no reading of the plain step.
//!   A verify a window cut below the ids a chain draft walked is no reading
//!   of its width: its wall less its width's cost, over the ids walked past
//!   it, is a reading of the walk of one id, and a cut's wall is its
//!   width's cost and that walk times the ids walked past it.
//!
//! Until every width's cost holds [`Rule::min_timings`] readings the passes
//! rotate the widths, widest first, and nothing is decided. Past that two
//! choices are made, each on its own profile:
//!
//! - the gate, on the last `1 / calibrate` windows that filled their room
//!   ([`Rule::calibrate`]), each its `p_i` at the positions it reached and
//!   the mean profile's past them: a width's rate is the rows those windows
//!   keep over the walls they cost, a chain draft's windows walking it and
//!   verifying the width each cuts to, another draft's verifying it. An
//!   open chooser closes only once the plain step's rate is above every
//!   drafted width's by [`Rule::margin_close`] and, when it opened after a
//!   close, every width's cost holds [`Rule::dwell`] readings taken since;
//!   a closed one opens only once a drafted width beats the plain step by
//!   [`Rule::margin`], and an open one's width moves to a rival only past
//!   the margin, or, for a chain draft whose windows mostly cut below its
//!   gate, to a narrower one that rates as high. One window never moves
//!   it;
//! - while open, each window's width, on that window's own profile and each
//!   width's own cost: the gate's width, or another its profile rates above
//!   it by the margin. The ids a chain draft walked past a width are spent
//!   whatever the pass verifies, and charged to every width alike they would
//!   rate the wider widths up.
//!
//! After [`Rule::probe_every`] passes the width read longest ago runs once,
//! so no cost goes stale. An open or a close rests on the readings of the
//! side it left as they stood, so it starts the period over and its first
//! probe re-reads that side. A move between open widths keeps the period
//! and every reading's age: moves faster than a period would otherwise hold
//! every probe off, or turn every probe to the two widths the gate moves
//! between, and leave the other costs as they stood. While closed the period backs off: each closed
//! probe doubles it, up to [`Rule::probe_closed`]. A probe whose reading, put
//! in place of its width's median, would move the gate starts that width's
//! readings over from it: the width runs again until it holds
//! [`Rule::min_timings`] readings taken since, and the gate decides on their
//! median. A chain draft's probe past an open gate refreshes the means of
//! the positions past it, because no other pass reaches them; its costs
//! start over only if the refreshed profile would move the gate at either
//! the held median or the new reading.
//! One fast reading cannot
//! move the gate and one slow one cannot hold a width out of it, and one
//! probe after a load step replaces a median read under the old load. A
//! proposal shorter than the room it was handed (the context's end) moves
//! neither the mean nor the gate.
//!
//! A draft proposes into the room the chooser hands it, and the pass
//! verifies the front `k` ids. A draft that makes its whole proposal in one
//! call is handed its whole width. A chain draft ([`Draft::CHAIN`]), whose
//! ids are walks of its own one after another, is handed the width the pass
//! plans to verify (the warm-up's or a probe's width, else the gate's) and
//! stops there: a window's own profile can then cut that width, never widen
//! it, and the mean and the calibration hear of the positions a proposal
//! reached, a position past them keeping what it last heard. A shadow walks
//! the whole width. While closed, a plain pass asks the draft one
//! thing — to walk the rows waiting for its next walk before its step
//! overwrites their arena ([`Draft::before_plain`]) — and hears the position
//! through [`Draft::held`], but every [`Rule::shadow_every`]th: that
//! proposal runs as a shadow, scored against
//! the target's own next tokens and never verified, keeping the calibration
//! current so the chooser can open again. A prompt call on a closed gate
//! drops the windows of the generations before it and runs its first
//! [`Rule::min_timings`] passes as shadows, so the gate decides on that
//! generation's own windows ([`Choosing::prompt_by`]). A proposal no row of
//! which a pass verifies (a shadow, a cut to 0) is [`Draft::unproposed`].
//!
//! A round of several sequences' windows runs the widths [`Batch`] chooses
//! for the whole round, priced by the round's rows and wall. Each window is
//! counted ([`Choosing::round_ran`]) and feeds its slot's conditional
//! acceptance ([`Choosing::batch_profile`]); no width's wall is read from it.
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

    /// The mode's `BLOOMERY_MTP_WIDTH` word.
    #[must_use]
    pub fn word(self) -> &'static str {
        match self {
            Mode::Cost => "cost",
            Mode::Fixed => "fixed",
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
    /// An open chooser closes only when the plain step's rate beats the best
    /// drafted width's by this factor: the close half of the band, the open
    /// half being [`Rule::margin`]; 1 closes the moment the plain step is
    /// ahead.
    ///
    /// Derivation: the gate compares `ρ = E · t̃₀ / t̃_w`, two medians of
    /// [`Rule::timings`] readings and `E` from the calibration. A median of 8
    /// readings of relative spread `c` has spread `m · c`, `m ≈ 0.41` (the
    /// median of 8 of a CV-`c` distribution, simulated for `c ≤ 0.3`;
    /// 0.443 asymptotically), so `σ² = (m c₀)² + (m c_w)² + σ_E²`, `σ_E` the
    /// calibration's own spread. The band is one σ: `margin_close = exp σ`, a
    /// drafted width closed only when the plain step leads by more than the
    /// comparison's noise. The floor of the readings' spread is the spread
    /// across rounds of a forced-id run: ranges of 2 % (the plain step) and
    /// 15 % (a window) over 2.53, the expected range of 6 normal draws in σ,
    /// give `c₀ = 0.8 %`, `c_w = 6.0 %`, and the kept rows' 12 % gives
    /// `σ_E = 4.6 %`: `σ = 5.2 %`, `exp σ = 1.05`.
    /// PIN(2026-10-13): 1.05. Two things the floor leaves out: a paged plan's
    /// pass walls spread by `c ≈ 0.25` within a run (σ 15 %, 1.16), and the
    /// gate looks at nearly every pass, about one independent look in 8, so
    /// no close in 1024 passes takes `z ≈ 3.35` σ (1.19 at the floor); both
    /// are past the 1.073 the warm-up gate test's close ratio allows, and
    /// [`Rule::dwell`] carries the rest.
    pub margin_close: f64,
    /// Readings of every width's cost, taken since the gate last opened, an
    /// open chooser needs before it may close; 0 closes on the rings as they
    /// stand. The gate that opened rested on the rings as they stood then, so
    /// a close on them is the same evidence read again: a median stops
    /// resting on a reading from before the open when most of its ring is
    /// new, `timings / 2 + 1`. While open only the held width reads every
    /// pass; each other width, the plain step among them, reads once a
    /// probe, and the probes take the widths in turn, so the dwell lasts at
    /// least `dwell · probe_every · (widths − 1)` passes. A chooser that
    /// starts open has no such evidence to wait out: the dwell follows a
    /// close that opened again.
    pub dwell: usize,
    /// Readings a width's cost median holds.
    pub timings: usize,
    /// Readings of every width's cost before the chooser compares rates.
    pub min_timings: usize,
    /// The calibration's step: how far a tested position's `g_i` moves
    /// toward what happened. The mean profile moves by it too, and the gate
    /// rates the widths over the last `1 / calibrate` windows, the windows
    /// that step weighs.
    pub calibrate: f64,
    /// The weight of a position's starting gain of 1, in tests: the `n`-th
    /// test of a position moves its `g_i` by `1 / (prior + n)` of the way
    /// while that is above [`Rule::calibrate`], so the gain starts as the
    /// mean of what its tests kept with the start counted `prior` times;
    /// `None` moves it by [`Rule::calibrate`] from the first test. A draft
    /// that reports certainty ([`Draft::propose_p`]'s default) is rated by
    /// its gains alone, and the start of 1 says it keeps every id.
    pub prior: Option<f64>,
    /// Passes before the width read longest ago runs once, while open; and
    /// the first probe after the gate opens or closes, which re-reads the
    /// side it left.
    pub probe_every: u32,
    /// The longest period while closed, when every probe is a drafted
    /// width's verify: each closed probe doubles the period up to it.
    pub probe_closed: u32,
    /// A closed chooser's shadow proposal after this many plain passes.
    pub shadow_every: u32,
}

impl Rule {
    /// A rival must win by 3 %; costs as the median of the last 8 readings,
    /// from 4 on; the calibration moves 5 % of the way a window; a probe
    /// after 64 passes open, closed after 64, 128, then every 256; a shadow
    /// every 16th closed pass; no prior, the fixed step from the first test.
    /// The periods price what they buy: a closed chooser pays a shadow's
    /// whole chain and a probe's verify where its plain pass is cheaper, and
    /// a draft that stays closed pays one verify in 256 passes once its first
    /// two probes are spent.
    pub const DEFAULT: Rule = Rule {
        margin: 1.03,
        margin_close: 1.05,
        dwell: 5,
        timings: 8,
        min_timings: 4,
        calibrate: 0.05,
        prior: None,
        probe_every: 64,
        probe_closed: 256,
        shadow_every: 16,
    };

    /// A close's first probe period: [`Rule::probe_every`], never past
    /// [`Rule::probe_closed`].
    fn first_closed_probe(&self) -> u32 {
        self.probe_every.min(self.probe_closed)
    }

    /// The windows the gate rates the widths over: `1 / calibrate`, at
    /// least one.
    fn recent(&self) -> usize {
        (1.0 / self.calibrate).round().max(1.0) as usize
    }

    /// Refused by name unless the chooser can run by it.
    pub fn check(&self) -> Result<(), WidthError> {
        let refused = if !(self.margin.is_finite() && self.margin >= 1.0) {
            "the margin must be finite and at least 1"
        } else if !(self.margin_close.is_finite() && self.margin_close >= 1.0) {
            "the close margin must be finite and at least 1"
        } else if self.min_timings == 0 {
            "a cost of zero readings"
        } else if self.timings < self.min_timings {
            "a cost's readings fewer than its decision minimum"
        } else if self.dwell > self.timings {
            "a dwell of more readings than a cost holds"
        } else if !(self.calibrate > 0.0 && self.calibrate <= 1.0) {
            "the calibration's step must be inside (0, 1]"
        } else if self.prior.is_some_and(|w| !(w.is_finite() && w >= 0.0)) {
            "the calibration's prior must be finite and not negative"
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
    /// A window that does not fit the body's pass bound.
    Round(crate::RoundError),
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
            WidthError::Round(e) => write!(f, "draft width: {e}"),
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

    /// A reading its caller has checked: a wall is positive, a walk's share
    /// of one at least 0.
    fn push(&mut self, ms: f64) {
        if self.ring.len() == self.cap {
            self.ring.pop_front();
        }
        self.ring.push_back(ms);
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

/// Measured round walls for one active-slot count and body pass bound.
struct BatchCosts {
    slots: usize,
    /// The body's rows and sequences a pass.
    limits: (usize, usize),
    /// Sorted pass row counts and the whole round's wall.
    costs: Vec<(Vec<usize>, Timings)>,
    passes: u32,
    probe: usize,
}

impl BatchCosts {
    fn readings(&self, rows: &[usize]) -> usize {
        self.costs
            .iter()
            .find(|(r, _)| r == rows)
            .map_or(0, |(_, t)| t.len())
    }

    fn cost(&mut self, rows: &[usize]) -> f64 {
        if let Some((_, t)) = self.costs.iter_mut().find(|(r, _)| r == rows) {
            return t.median().expect("an observed batch cost");
        }
        // Unseen packings interpolate wall per pass by mean rows per pass.
        // Scaling back by pass count charges every target traversal.
        let passes = rows.len() as f64;
        let mean_rows = rows.iter().sum::<usize>() as f64 / passes;
        let mut lo: Option<(f64, f64)> = None;
        let mut hi: Option<(f64, f64)> = None;
        for (r, t) in &mut self.costs {
            let n = r.len() as f64;
            let x = r.iter().sum::<usize>() as f64 / n;
            let y = t.median().expect("an observed batch cost") / n;
            if x <= mean_rows && lo.is_none_or(|(a, _)| x > a) {
                lo = Some((x, y));
            }
            if x >= mean_rows && hi.is_none_or(|(b, _)| x < b) {
                hi = Some((x, y));
            }
        }
        let ms = match (lo, hi) {
            (Some((a, x)), Some((b, y))) if b > a => x + (y - x) * ((mean_rows - a) / (b - a)),
            (Some((a, x)), _) => x * (mean_rows / a),
            (_, Some((_, y))) => y,
            _ => unreachable!("the endpoint packings were sampled"),
        };
        ms * passes
    }
}

/// Widths for one shared verify round of decode windows, priced by its
/// actual pass packing. Each active-slot count keeps its own readings and
/// probe progress, so returning to that count resumes its learning.
#[derive(Default)]
pub struct Batch {
    states: Vec<BatchCosts>,
    current: Option<usize>,
    end: usize,
    rows: Vec<usize>,
    cuts: Vec<std::ops::Range<usize>>,
    packed: Vec<usize>,
    caps: Vec<usize>,
    candidate: Vec<usize>,
}

impl Batch {
    /// Cut each slot's maximum width using its conditional acceptance profile.
    /// `allowed` is the body's supported draft-width set, and `limits` its
    /// rows and sequences a pass, the bound the round's passes are cut at; a
    /// slot already at zero has no draft. Fixed mode leaves all widths
    /// untouched.
    pub fn choose<'a>(
        &mut self,
        mode: Mode,
        profiles: impl ExactSizeIterator<Item = &'a [f64]> + Clone,
        widths: &mut [usize],
        allowed: std::ops::RangeInclusive<usize>,
        limits: (usize, usize),
    ) -> Result<(), WidthError> {
        if profiles.len() != widths.len() || widths.is_empty() {
            return Err(WidthError::Rule(
                "batch profiles and widths must name the same nonempty slots",
            ));
        }
        if allowed.is_empty() {
            return Err(WidthError::Rule("batch allowed-width set is empty"));
        }
        for (a, &width) in profiles.clone().zip(widths.iter()) {
            if width > 0 && !allowed.contains(&width) {
                return Err(WidthError::Rule(
                    "batch width outside the body's allowed set",
                ));
            }
            if width > a.len() {
                return Err(WidthError::Rule(
                    "batch width exceeds its acceptance profile",
                ));
            }
            if a.iter().any(|p| !p.is_finite() || !(0.0..=1.0).contains(p)) {
                return Err(WidthError::Rule(
                    "batch acceptance must be finite and in [0, 1]",
                ));
            }
        }
        let slots = widths.len();
        let end = widths
            .iter()
            .try_fold(slots, |n, &w| n.checked_add(w))
            .and_then(|n| n.checked_add(1))
            .ok_or(WidthError::Rule("batch row count overflow"))?;
        self.rows.clear();
        self.rows.extend(widths.iter().map(|&w| w + 1));
        crate::round_passes(
            self.rows.iter().copied().enumerate(),
            limits,
            &mut self.cuts,
        )
        .map_err(WidthError::Round)?;
        if mode == Mode::Fixed {
            return Ok(());
        }
        let current = match self
            .states
            .iter()
            .position(|s| s.slots == slots && s.limits == limits)
        {
            Some(i) => i,
            None => {
                self.states.push(BatchCosts {
                    slots,
                    limits,
                    costs: Vec::new(),
                    passes: 0,
                    probe: 0,
                });
                self.states.len() - 1
            }
        };
        self.current = Some(current);
        self.end = end;
        let max = widths.iter().copied().max().unwrap_or(0);
        let min = (*allowed.start()).min(max);
        // Bracket the row costs; selected rounds supply intermediate readings.
        for cap in [max, min] {
            self.rows.clear();
            self.rows.extend(widths.iter().map(|&w| w.min(cap) + 1));
            self.pack()?;
            if self.states[current].readings(&self.packed) < Rule::DEFAULT.min_timings {
                for w in widths {
                    *w = (*w).min(cap);
                }
                return Ok(());
            }
        }
        let state = &mut self.states[current];
        state.passes += 1;
        if state.passes >= Rule::DEFAULT.probe_every {
            state.passes = 0;
            state.probe = if state.probe < min || state.probe >= max {
                min
            } else {
                state.probe + 1
            };
            for w in widths {
                *w = (*w).min(state.probe);
            }
            return Ok(());
        }
        self.caps.clear();
        self.caps.extend_from_slice(widths);
        self.candidate.clear();
        self.candidate.extend(widths.iter().map(|&w| w.min(min)));
        widths.copy_from_slice(&self.candidate);
        let mut kept: f64 = profiles
            .clone()
            .zip(widths.iter())
            .map(|(a, &w)| expected(a, w))
            .sum();
        self.rows.clear();
        self.rows.extend(widths.iter().map(|&w| w + 1));
        self.pack()?;
        let mut best = kept / self.states[current].cost(&self.packed);
        let start = slots + widths.iter().sum::<usize>();
        // Conditional probabilities make each slot's marginal yield decrease.
        // Taking the largest next yield maximizes tokens for each row budget.
        for _ in start + 1..end {
            let mut next = None;
            for (i, a) in profiles.clone().enumerate() {
                let k = self.candidate[i];
                if k < self.caps[i] {
                    let gain = expected(a, k + 1) - expected(a, k);
                    if next.is_none_or(|(_, g)| gain > g) {
                        next = Some((i, gain));
                    }
                }
            }
            let Some((i, gain)) = next else { break };
            self.candidate[i] += 1;
            self.rows[i] += 1;
            kept += gain;
            self.pack()?;
            let rate = kept / self.states[current].cost(&self.packed);
            if rate > best * Rule::DEFAULT.margin {
                best = rate;
                widths.copy_from_slice(&self.candidate);
            }
        }
        Ok(())
    }

    fn pack(&mut self) -> Result<(), WidthError> {
        let state = &self.states[self.current.expect("a batch was chosen")];
        crate::round_passes(
            self.rows.iter().copied().enumerate(),
            state.limits,
            &mut self.cuts,
        )
        .map_err(WidthError::Round)?;
        self.packed.clear();
        self.packed.extend(
            self.cuts
                .iter()
                .map(|r| self.rows[r.clone()].iter().sum::<usize>()),
        );
        self.packed.sort_unstable();
        Ok(())
    }

    /// Record the whole round's wall, including drafting and commits, in
    /// milliseconds, with each window's actual verified row count in order.
    pub fn observe(
        &mut self,
        rows: impl ExactSizeIterator<Item = usize> + Clone,
        ms: f64,
    ) -> Result<(), WidthError> {
        let current = self
            .current
            .ok_or(WidthError::Rule("batch reading before a choice"))?;
        let state = &self.states[current];
        let total = rows.clone().try_fold(0_usize, |n, r| n.checked_add(r));
        if rows.len() != state.slots || total.is_none_or(|n| n < state.slots || n >= self.end) {
            return Err(WidthError::Rule(
                "batch reading outside the chosen row range",
            ));
        }
        let ms = positive("batch", ms)?;
        self.rows.clear();
        self.rows.extend(rows);
        self.pack()?;
        let state = &mut self.states[current];
        let index = match state.costs.iter().position(|(r, _)| r == &self.packed) {
            Some(i) => i,
            None => {
                state.costs.push((
                    self.packed.clone(),
                    Timings::new("batch", Rule::DEFAULT.timings),
                ));
                state.costs.len() - 1
            }
        };
        state.costs[index].1.push(ms);
        Ok(())
    }
}

/// What a pass ran, as its proposal left it, for its wall's reading.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// A verify of a proposal: a reading of that width's cost.
    Verify,
    /// A plain step: a reading of the step's, its wall holding at most the
    /// draft's walk of the rows that waited for it ([`Draft::before_plain`]).
    Step,
    /// A plain step after the draft's proposal (a shadow, a cut to 0): its
    /// wall holds the draft's whole chain, no reading.
    Walked,
    /// A verify whose window cut it below the ids a chain draft walked: its
    /// wall less its width's cost is a reading of the walks past its width.
    Cut,
}

impl Kind {
    /// Whether the pass verifies a proposal.
    fn verifies(self) -> bool {
        matches!(self, Kind::Verify | Kind::Cut)
    }
}

/// The pass in flight since its proposal.
#[derive(Clone, Copy, Debug)]
struct Flight {
    kind: Kind,
    /// The ids the pass verifies, 0 for a plain step.
    width: usize,
    /// The ids a chain draft walked for a verify; `width` otherwise.
    walked: usize,
    /// The target's position at the proposal.
    pos: u32,
    at: Duration,
    /// The position the pass's accept or step left the target at; `None`
    /// until one came.
    ends: Option<u32>,
    /// A probe: its reading may start its width's readings over.
    probe: bool,
    /// `E` of the width a verify runs over the window's own profile; 0 for a
    /// plain step.
    predicted: f64,
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
#[derive(Clone, Debug, PartialEq)]
pub struct Tally {
    /// Passes that verified a proposal.
    pub windows: u64,
    /// The windows by the rows they kept: index `i` the windows that kept
    /// `i + 1`.
    pub kept: Vec<u64>,
    /// Every pass by the ids it verified: index 0 the plain steps.
    pub widths: Vec<u64>,
    /// The chooser's own passes counted while the gate stood closed, its
    /// probes among them; a round's pass ([`Choosing::round_ran`]) is none.
    pub closed: u64,
    /// The windows the chooser itself proposed and verified: the ones the
    /// next two fields cover (a round's window, [`Choosing::round_ran`], has
    /// no prediction and is in `windows` only).
    pub scored: u64,
    /// The rows those windows kept, summed.
    pub scored_rows: u64,
    /// `E` of the width each of them ran, over the window's own profile at
    /// its proposal, summed.
    pub predicted: f64,
}

impl Tally {
    fn new(widths: usize) -> Tally {
        Tally {
            windows: 0,
            kept: vec![0; widths],
            widths: vec![0; widths],
            closed: 0,
            scored: 0,
            scored_rows: 0,
            predicted: 0.0,
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

/// What the gate stands on now ([`Choosing::gate_state`]): what a draft
/// record prints of the chooser beside its [`Tally`].
#[derive(Clone, Debug, PartialEq)]
pub struct GateState {
    /// The gate: 0 closed; open, the width the recent windows rate highest.
    pub width: usize,
    /// Each width's cost median in milliseconds, index `k` the passes that
    /// verified `k` ids and index 0 the plain step; `None` while a cost holds
    /// fewer than [`Rule::min_timings`] readings.
    pub costs: Vec<Option<f64>>,
    /// The mean acceptance profile, `a_i` one a proposal index: the
    /// conditional mean ([`Stats::mean`]) times the calibration.
    pub a: Vec<f64>,
    /// Each width's gate rate on the recent windows, in positions per
    /// millisecond, index 0 the plain step; `None` while a cost or the
    /// recent windows needed to rate it are unread.
    pub rates: Vec<Option<f64>>,
    /// A chain draft's estimated walk of one id in milliseconds; 0 until
    /// enough cut passes have read it, and for a draft that is not a chain.
    pub walk_ms: f64,
}

/// What the chooser measured of itself ([`Choosing::stats`]): its
/// calibration now, and over the passes since the last take
/// ([`Choosing::take_tally`]) the passes it ran closed and the rows a window
/// was predicted to keep against the rows it kept.
#[derive(Clone, Debug, PartialEq)]
pub struct Stats {
    /// The calibration `g_i`, one a proposal index.
    pub gains: Vec<f64>,
    /// The draft's own `p_i` over the proposals that reached it, before the
    /// calibration, as a mean conditional on the ids before it: the running
    /// mean of `Π_{j≤i} p_j` over the running mean of `Π_{j<i} p_j`, both
    /// over those proposals. A window that stopped short of a position is
    /// rated past it by this mean.
    pub mean: Vec<f64>,
    /// The recent-window gate rates by width ([`GateState::rates`]).
    pub rates: Vec<Option<f64>>,
    /// The estimated per-id chain walk in milliseconds ([`GateState::walk_ms`]).
    pub walk_ms: f64,
    /// The chooser's own passes counted while the gate stood closed
    /// ([`Tally::closed`]).
    pub closed: u64,
    /// The windows the two means below cover ([`Tally::scored`]).
    pub windows: u64,
    /// The mean over those windows of `E` of the width each ran, over the
    /// window's own profile; `None` with no window.
    pub predicted_e: Option<f64>,
    /// The mean rows those windows kept; `None` with no window.
    pub realized_e: Option<f64>,
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
    /// The gate: 0 closed; open, the width the recent windows rate highest
    /// (a window may verify another, [`Choosing::cut`]).
    width: usize,
    /// [`Draft::CHAIN`]: a window's draft walks the width it is handed.
    chain: bool,
    /// Passes since the last probe, open or close.
    since_probe: u32,
    /// The next probe's period while closed: [`Rule::probe_every`] after a
    /// close, doubled by each closed probe up to [`Rule::probe_closed`].
    closed_period: u32,
    /// Closed plain passes since the last shadow.
    since_shadow: u32,
    /// Shadows a request that began on a closed gate still owes before the
    /// gate decides on its windows ([`Rule::min_timings`] of them).
    opening: usize,
    passes: u64,
    /// The calibration `g_i`, one a proposal index.
    gains: Vec<f64>,
    /// Conditional acceptance of round windows, which carry no probabilities.
    batch_gains: Vec<f64>,
    batch_tested: Vec<u64>,
    /// The tests each `g_i` has taken.
    tested: Vec<u64>,
    /// Over the proposals that filled their room, the running means of
    /// `Π_{j<i} p_j` and `Π_{j≤i} p_j` at each position they reached, their
    /// ratio ([`Stats::mean`]), and how many of them reached each position.
    before: Vec<f64>,
    upto: Vec<f64>,
    mean: Vec<f64>,
    seen: Vec<u64>,
    /// The last proposals that filled their room, [`Rule::calibrate`]'s
    /// inverse of them: `recent_len` of them, the next written at
    /// `recent_at`, each its `WIDTH` slots of `p_i` and the positions it
    /// reached, the conditional mean standing for the rest.
    recent: Vec<f32>,
    reached: Vec<usize>,
    recent_len: usize,
    recent_at: usize,
    /// A window's acceptance profile as the gate rates it, reused.
    gated: Vec<f64>,
    /// A chain draft's walk of one id, read off the passes a window cut
    /// below its walk ([`Kind::Cut`]), and its median; 0 until the median
    /// holds [`Rule::min_timings`] readings.
    walk: Timings,
    walk_ms: f64,
    /// The last readings of each width's cost: index `k` the passes that
    /// verified `k` ids, index 0 the plain steps.
    costs: Vec<Timings>,
    /// Each width's median, refreshed as its readings land; `None` while a
    /// cost holds fewer than [`Rule::min_timings`] readings.
    medians: Vec<Option<f64>>,
    /// The pass each width's cost was last read at; 0 for the side the gate
    /// left at its last open or close until a probe re-reads it.
    read_at: Vec<u64>,
    /// Readings of each width's cost since the gate last opened, and whether
    /// that open's dwell ([`Rule::dwell`]) is still running.
    fresh: Vec<usize>,
    dwelling: bool,
    /// The widths whose readings are not all in: the warm-up's rotation,
    /// or a width a probe started over.
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
        let windows = rule.recent();
        let too_many = || {
            WidthError::Rule("the calibration's step keeps more recent windows than can be held")
        };
        let cells = D::WIDTH.checked_mul(windows).ok_or_else(too_many)?;
        let (mut recent, mut reached) = (Vec::new(), Vec::new());
        recent.try_reserve_exact(cells).map_err(|_| too_many())?;
        reached.try_reserve_exact(windows).map_err(|_| too_many())?;
        recent.resize(cells, 0.0);
        reached.resize(windows, 0);
        Ok(Choosing {
            draft: self,
            clock,
            rule,
            mode,
            width: D::WIDTH,
            chain: D::CHAIN,
            since_probe: 0,
            closed_period: rule.first_closed_probe(),
            since_shadow: 0,
            opening: 0,
            passes: 0,
            gains: vec![1.0; D::WIDTH],
            batch_gains: vec![1.0; D::WIDTH],
            batch_tested: vec![0; D::WIDTH],
            tested: vec![0; D::WIDTH],
            before: vec![0.0; D::WIDTH],
            upto: vec![0.0; D::WIDTH],
            mean: vec![0.0; D::WIDTH],
            seen: vec![0; D::WIDTH],
            recent,
            reached,
            recent_len: 0,
            recent_at: 0,
            gated: vec![0.0; D::WIDTH],
            walk: Timings::new("walk", rule.timings),
            walk_ms: 0.0,
            costs: (0..widths)
                .map(|k| Timings::new(if k == 0 { "step" } else { "verify" }, rule.timings))
                .collect(),
            medians: vec![None; widths],
            read_at: vec![0; widths],
            fresh: vec![0; widths],
            dwelling: false,
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

    /// A prompt call of the draft's own, `f` run on the draft with the chooser's flight forgotten
    /// first, as [`Draft::prompt`] forgets it: the prompt starts a generation, so nothing in
    /// flight before it is read as a pass of this one. [`Draft::prompt`] is this call with the
    /// draft's own prompt; a call with another shape of prompt (a media prompt: the draft's
    /// prompt over the same ids with spans spliced in) goes through it, so the two have one owner.
    ///
    /// On a closed gate the generation is judged on its own windows: the earlier generations'
    /// windows are dropped and its first [`Rule::min_timings`] passes run as shadows, the gate
    /// deciding once they are in. A closed gate hears of a generation only through its shadows,
    /// one in [`Rule::shadow_every`] passes, and would otherwise hold the windows of one that paid
    /// less for as long as those take to age out.
    pub fn prompt_by<R>(&mut self, f: impl FnOnce(&mut D) -> R) -> R {
        self.forget_pass();
        if self.mode != Mode::Fixed && self.width == 0 && self.warm.is_empty() {
            (self.recent_len, self.recent_at) = (0, 0);
            self.opening = self.rule.min_timings;
        }
        f(&mut self.draft)
    }

    /// The pass in flight, a pending shadow and the round's acceptance forgotten, for a call of
    /// the caller's own outside the chooser's passes (a row of a pass of several slots, a
    /// closed generation): nothing outside the passes is read as one.
    pub fn forget_pass(&mut self) {
        self.drop_flight();
        self.reset_batch();
    }

    /// The chooser's mode.
    #[must_use]
    pub fn mode(&self) -> Mode {
        self.mode
    }

    /// The gate: 0 closed; open, the width the gate's rule holds on the
    /// recent windows, which a window verifies unless its own profile rates
    /// another by the margin above it.
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

    /// The chooser's calibration and what it predicted against what its
    /// windows kept, over the passes since the last take; `None` under
    /// [`Mode::Fixed`], which counts nothing. A read: the count starts over
    /// with [`Choosing::take_tally`].
    #[must_use]
    pub fn stats(&self) -> Option<Stats> {
        if self.mode == Mode::Fixed {
            return None;
        }
        let t = &self.tally;
        let mean_of = |sum: f64| (t.scored > 0).then(|| sum / t.scored as f64);
        Some(Stats {
            gains: self.gains.clone(),
            mean: self.mean.clone(),
            rates: self.gate_rates(),
            walk_ms: self.walk_ms,
            closed: t.closed,
            windows: t.scored,
            predicted_e: mean_of(t.predicted),
            realized_e: mean_of(t.scored_rows as f64),
        })
    }

    /// The passes since the last take, and the count started over.
    pub fn take_tally(&mut self) -> Tally {
        std::mem::replace(&mut self.tally, Tally::new(self.costs.len()))
    }

    /// The gate, each width's cost median, the mean profile, recent-window
    /// gate rates and the per-id walk cost as they stand now.
    #[must_use]
    pub fn gate_state(&self) -> GateState {
        GateState {
            width: self.width,
            costs: self.medians.clone(),
            a: self
                .mean
                .iter()
                .zip(&self.gains)
                .map(|(&p, &g)| (p * g).clamp(0.0, 1.0))
                .collect(),
            rates: self.gate_rates(),
            walk_ms: self.walk_ms,
        }
    }

    fn gate_rates(&self) -> Vec<Option<f64>> {
        let mut a = vec![0.0; self.mean.len()];
        (0..self.costs.len())
            .map(|k| {
                if k == 0 {
                    self.medians[0].map(|ms| 1.0 / ms)
                } else {
                    self.gate_rate(k, &mut a).map(|(rate, _)| rate)
                }
            })
            .collect()
    }

    /// The conditional acceptance learned from whole-round windows.
    pub fn batch_profile(&self) -> &[f64] {
        &self.batch_gains
    }

    fn reset_batch(&mut self) {
        self.batch_gains.fill(1.0);
        self.batch_tested.fill(0);
    }

    /// A pass of several sequences' windows ran this draft's window (its
    /// [`Draft::propose`] and the round's record called on the draft itself,
    /// the chooser bypassed): `rows` ran, `kept` were kept. Counted, and its
    /// rows feed [`Choosing::batch_profile`]; no cost read (the wall is the
    /// round's, [`Batch::observe`]) and nothing in flight survives it.
    pub fn round_ran(&mut self, rows: usize, kept: usize) {
        if self.mode == Mode::Fixed {
            return;
        }
        self.drop_flight();
        let k = rows.saturating_sub(1);
        let j = kept.saturating_sub(1);
        let tested = j + usize::from(j < k);
        for (i, (a, n)) in self
            .batch_gains
            .iter_mut()
            .zip(&mut self.batch_tested)
            .enumerate()
            .take(tested)
        {
            *n = n.saturating_add(1);
            let hit = if i < j { 1.0 } else { 0.0 };
            let weight = self.rule.calibrate.max(1.0 / *n as f64);
            *a += weight * (hit - *a);
        }
        self.ran(k, kept);
    }

    /// The pass that just ran landed now, the target at `pos`: its wall is
    /// read before the caller runs anything else, so a call between it and
    /// this draft's next pass (another sequence's pass, a prompt call) is
    /// neither read as its wall nor drops it. Nothing under [`Mode::Fixed`].
    pub fn finish_pass(&mut self, pos: u32) -> Result<(), WidthError> {
        if self.mode == Mode::Cost {
            let now = self.clock.now();
            self.land(now, pos)?;
        }
        Ok(())
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
        match f.kind {
            Kind::Walked => {
                positive("plain pass", ms)?;
                return Ok(());
            }
            Kind::Cut => {
                let ms = positive("verify", ms)?;
                if let Some(cost) = self.medians[f.width] {
                    let past = (f.walked - f.width) as f64;
                    self.walk.push(((ms - cost) / past).max(0.0));
                    if self.walk.len() >= self.rule.min_timings {
                        self.walk_ms = self.walk.median().unwrap_or(0.0);
                    }
                }
                return Ok(());
            }
            Kind::Verify | Kind::Step => {}
        }
        let k = f.width;
        let ms = positive(self.costs[k].what, ms)?;
        if f.probe && (self.moves_on(k, ms) || self.decision().is_some_and(|w| w != self.width)) {
            // The median read before gives way to readings from this one on.
            self.start_over(k);
        }
        self.costs[k].push(ms);
        self.read_at[k] = self.passes;
        self.fresh[k] += 1;
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
            // The first probe comes a whole period after the warm-up; a
            // closed gate decides on the readings a probe started over,
            // which no shadow waits for.
            self.since_probe = 0;
            if self.width == 0 {
                self.gate();
            }
        }
        Ok(())
    }

    /// The rate of verifying `k` ids over the profile `a` at `k`'s cost;
    /// `None` while `k`'s cost is unread.
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

    /// The width a window of profile `a` verifies under a gate of `w`: `w`,
    /// unless `a` rates another by the margin above it, each at its own cost
    /// (the module); `None` while a cost is unread.
    fn cut_of(&self, a: &[f64], w: usize) -> Option<usize> {
        let w = w.min(a.len());
        let held = self.rate(a, w)?;
        let (best, b) = self.best(a)?;
        Some(if b != w && best > held * self.rule.margin {
            b
        } else {
            w
        })
    }

    /// The rate a gate of `w` held over the recent windows: the rows they
    /// keep over the walls they cost. A chain draft's windows walk `w` and
    /// verify the width each cuts to ([`Choosing::cut_of`]), each wall the
    /// cut's cost and the walks past it at the walk's median, paid whatever
    /// the cut; another draft's verify `w` at its cost, its proposal in
    /// every width's cost, a window's cut its own. With the rate, the
    /// windows that cut below `w`. `None` while no window is in or a cost is
    /// unread.
    fn gate_rate(&self, w: usize, a: &mut [f64]) -> Option<(f64, usize)> {
        if self.recent_len == 0 {
            return None;
        }
        let (n, walked) = if self.chain { (w, w) } else { (a.len(), 0) };
        let (mut kept, mut ms, mut below) = (0.0, 0.0, 0);
        let windows = self.recent.chunks_exact(a.len()).zip(&self.reached);
        for (p, &reached) in windows.take(self.recent_len) {
            for (i, a) in a.iter_mut().enumerate() {
                let p = if i < reached {
                    f64::from(p[i])
                } else {
                    self.mean[i]
                };
                *a = (p * self.gains[i]).clamp(0.0, 1.0);
            }
            let k = if self.chain {
                self.cut_of(&a[..n], w)?
            } else {
                w
            };
            kept += expected(a, k);
            ms += self.medians[k]? + self.walk_ms * walked.saturating_sub(k) as f64;
            below += usize::from(k < w);
        }
        Some((kept / ms, below))
    }

    /// The gate on the recent windows (the module's rule): an open chooser
    /// closes when the plain step's rate is above every drafted width's, a
    /// closed one opens when a drafted width beats the plain step by the
    /// margin, and an open one's width gives way to a rival that beats it by
    /// the margin. Nothing moves while the warm-up's readings are not all
    /// in. An open or a close starts the probe period over, its first probe
    /// the side it left, and a close starts the closed probes' backoff over;
    /// a move between open widths keeps the period and every reading's age,
    /// or moves faster than a period would hold every probe off.
    fn gate(&mut self) {
        if !self.warm.is_empty() {
            return;
        }
        if let Some(next) = self.decision()
            && next != self.width
        {
            if (next == 0) != (self.width == 0) {
                self.since_probe = 0;
                self.read_at[self.width] = 0;
                self.since_shadow = 0;
                if next == 0 {
                    self.closed_period = self.rule.first_closed_probe();
                } else {
                    self.fresh.fill(0);
                    self.dwelling = true;
                }
            }
            self.width = next;
        }
    }

    /// Whether the gate moves with `ms` in place of width `k`'s median, the
    /// rest as it stands.
    fn moves_on(&mut self, k: usize, ms: f64) -> bool {
        let held = self.medians[k].replace(ms);
        let moves = self.decision().is_some_and(|w| w != self.width);
        self.medians[k] = held;
        moves
    }

    /// Width `k`'s readings started over: it runs until it holds
    /// [`Rule::min_timings`] again, and nothing is decided meanwhile.
    fn start_over(&mut self, k: usize) {
        self.costs[k].ring.clear();
        self.medians[k] = None;
        self.warm.push_back(k);
    }

    /// The width the gate's rule picks on the recent windows and the costs
    /// as they stand, the narrower of two that rate the same; `None` while
    /// no window is in or a cost is unread.
    fn decision(&mut self) -> Option<usize> {
        let mut a = std::mem::take(&mut self.gated);
        self.dwelling = self.dwelling && self.fresh.iter().any(|&n| n < self.rule.dwell);
        let decided = self.decided(&mut a);
        self.gated = a;
        decided
    }

    /// [`Choosing::decision`], `a` the windows' profile reused.
    fn decided(&self, a: &mut [f64]) -> Option<usize> {
        let plain = 1.0 / self.medians[0]?;
        let (mut held, mut cut, mut best) = (None, 0, None);
        for w in 1..=a.len() {
            let (r, below) = self.gate_rate(w, a)?;
            if w == self.width {
                (held, cut) = (Some(r), below);
            }
            if best.is_none_or(|(b, _)| r > b) {
                best = Some((r, w));
            }
        }
        let (best, b) = best?;
        let margin = self.rule.margin;
        Some(match held {
            None if best > plain * margin => b,
            None => 0,
            Some(_) if best * self.rule.margin_close < plain && !self.dwelling => 0,
            Some(r) if b != self.width && best > r * margin => b,
            // A chain draft's gate whose windows mostly cut below it walks its
            // width for nothing: a narrower gate that rates as high takes the
            // seat.
            Some(r) if self.chain && b < self.width && best >= r && 2 * cut > self.recent_len => b,
            Some(_) => self.width,
        })
    }

    /// The width this window verifies while open: the gate's, unless the
    /// window's own profile `a[..n]` rates another by the margin above it.
    fn cut(&self, n: usize) -> usize {
        self.cut_of(&self.a[..n], self.width)
            .unwrap_or(self.width.min(n))
    }

    /// The proposal's probabilities of its first `n` ids into the mean
    /// profile, each position's two running means moving by a running mean's
    /// step until the calibration's is the smaller, then by that; and the
    /// proposal into the recent windows.
    fn learn(&mut self, n: usize) {
        let mut run = 1.0;
        for i in 0..n {
            self.seen[i] += 1;
            let step = (1.0 / self.seen[i] as f64).max(self.rule.calibrate);
            self.before[i] += step * (run - self.before[i]);
            run *= f64::from(self.p[i]);
            self.upto[i] += step * (run - self.upto[i]);
            self.mean[i] = if self.before[i] > 0.0 {
                (self.upto[i] / self.before[i]).clamp(0.0, 1.0)
            } else {
                0.0
            };
        }
        let at = self.recent_at * self.mean.len();
        self.recent[at..at + n].copy_from_slice(&self.p[..n]);
        self.reached[self.recent_at] = n;
        let cap = self.reached.len();
        self.recent_at = (self.recent_at + 1) % cap;
        self.recent_len = (self.recent_len + 1).min(cap);
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
            self.tested[i] += 1;
            let step = self
                .rule
                .prior
                .map_or(step, |w| (1.0 / (w + self.tested[i] as f64)).max(step));
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

    /// One of the chooser's own passes counted ([`Choosing::ran`]), and as
    /// closed while the gate stands closed.
    fn ran_own(&mut self, width: usize, kept: usize) {
        self.ran(width, kept);
        if self.width == 0 {
            self.tally.closed += 1;
        }
    }

    /// The pending shadow's next id against `next`, the target's own token:
    /// its match advanced, and at its end the calibration told of it, its
    /// probabilities into the mean profile and the recent windows, and the
    /// gate decided on the windows.
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
            if self.opening == 0 {
                self.gate();
            }
        }
    }

    /// This pass a plain step of `kind`; `probe` a probe of the plain step.
    fn plain(&mut self, kind: Kind, pos: u32, at: Duration, probe: bool) -> usize {
        self.flight = Some(Flight {
            kind,
            width: 0,
            walked: 0,
            pos,
            at,
            ends: None,
            probe,
            predicted: 0.0,
        });
        self.turn = Some(Turn::Plain);
        0
    }

    /// This pass a verify of the proposal's front `k` ids, written to `out`,
    /// the draft having walked `walked` ids for it (`k` for a draft whose
    /// proposal costs its whole width whatever the pass verifies); `probe` a
    /// probe whose reading may start the width over.
    fn verify(
        &mut self,
        out: &mut [u32],
        (k, walked): (usize, usize),
        pos: u32,
        at: Duration,
        probe: bool,
    ) -> usize {
        out[..k].copy_from_slice(&self.ids[..k]);
        self.flight = Some(Flight {
            kind: if k < walked { Kind::Cut } else { Kind::Verify },
            width: k,
            walked,
            pos,
            at,
            ends: None,
            probe,
            predicted: expected(&self.a, k),
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
    const CHAIN: bool = D::CHAIN;

    fn prompt(&mut self, t: &mut T, ids: &[u32]) -> Result<u32, T::Error> {
        self.prompt_by(|d| d.prompt(t, ids))
    }

    fn begin(&mut self, t: &T, prompt: &[u32], first: u32) -> Result<(), T::Error> {
        self.drop_flight();
        self.reset_batch();
        self.draft.begin(t, prompt, first)
    }

    /// One pass's proposal (the module): a turn that does not read the
    /// proposal first — a shadow in flight, the warm-up's rotation or a
    /// width's readings started over, a probe — then while closed a plain
    /// step (the draft's waiting rows walked first, [`Draft::before_plain`])
    /// or a shadow, else the draft's own proposal with its probabilities
    /// (a chain draft's into the width the pass plans to verify), the gate
    /// decided on the recent windows they join, and the front ids of the
    /// window's width ([`Choosing::cut`]) as the pass's rows.
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
            self.draft.before_plain(t, last)?;
            return Ok(self.plain(Kind::Step, pos, at, false));
        }
        let closed = self.width == 0;
        let period = if closed {
            self.closed_period
        } else {
            self.rule.probe_every
        };
        let (forced, probe) = if let Some(&k) = self.warm.front() {
            (Some(k), false)
        } else if self.since_probe >= period {
            self.since_probe = 0;
            if closed {
                self.closed_period = self
                    .closed_period
                    .saturating_mul(2)
                    .min(self.rule.probe_closed);
            }
            let k = self.stalest();
            if D::CHAIN && !closed && k > self.width {
                // No other pass reaches the positions past the gate's width.
                self.seen[self.width..k].fill(0);
            }
            (Some(k), true)
        } else {
            (None, false)
        };
        if forced == Some(0) || (forced.is_none() && self.width == 0) {
            let shadow = forced.is_none()
                && (self.opening > 0 || {
                    self.since_shadow += 1;
                    self.since_shadow >= self.rule.shadow_every
                });
            if !shadow {
                self.draft.before_plain(t, last)?;
                return Ok(self.plain(Kind::Step, pos, at, probe));
            }
            self.since_shadow = 0;
            self.opening = self.opening.saturating_sub(1);
            let n = self
                .draft
                .propose_p(t, last, &mut self.ids[..room], &mut self.p)?;
            if n == 0 {
                return Ok(self.plain(Kind::Step, pos, at, false));
            }
            check_p(&self.p[..n])?;
            self.shadow.ids[..n].copy_from_slice(&self.ids[..n]);
            self.shadow.p[..n].copy_from_slice(&self.p[..n]);
            (self.shadow.n, self.shadow.matched) = (n, 0);
            self.draft.unproposed();
            return Ok(self.plain(Kind::Walked, pos, at, false));
        }
        let depth = if D::CHAIN {
            forced.unwrap_or(self.width).min(room)
        } else {
            room
        };
        let n = self
            .draft
            .propose_p(t, last, &mut self.ids[..depth], &mut self.p)?;
        if n == 0 {
            return Ok(self.plain(Kind::Step, pos, at, false));
        }
        check_p(&self.p[..n])?;
        self.profile(n);
        let filled = n == depth;
        if filled {
            self.learn(n);
        }
        let k = match forced {
            Some(k) => k.min(n),
            None => {
                if filled {
                    self.gate();
                }
                if self.width == 0 { 0 } else { self.cut(n) }
            }
        };
        if k == 0 {
            self.draft.unproposed();
            return Ok(self.plain(Kind::Walked, pos, at, false));
        }
        let walked = if D::CHAIN { n } else { k };
        Ok(self.verify(out, (k, walked), pos, at, probe))
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
                if chose == k && f.kind.verifies() && f.ends.is_none() =>
            {
                f.ends = u32::try_from(accepted).ok().map(|a| f.pos + a);
                let predicted = f.predicted;
                self.calibrate(k, kept);
                self.tally.scored += 1;
                self.tally.scored_rows += accepted as u64;
                self.tally.predicted += predicted;
            }
            _ => self.drop_flight(),
        }
        self.ran_own(k, accepted);
        self.draft.accept(t, rows, out, accepted)
    }

    /// A step: the chooser's plain pass, counted, a pending shadow scored
    /// against `next`, the draft told it was held back; any other step
    /// (the caller's own, outside a pass) the draft's plain
    /// [`Draft::stepped`], with batch acceptance reset and nothing in flight.
    fn stepped(&mut self, t: &mut T, last: u32, next: u32) -> Result<(), T::Error> {
        if self.mode == Mode::Fixed {
            return self.draft.stepped(t, last, next);
        }
        if self.turn.take() != Some(Turn::Plain) {
            self.drop_flight();
            self.reset_batch();
            return self.draft.stepped(t, last, next);
        }
        match &mut self.flight {
            Some(f) if !f.kind.verifies() && f.ends.is_none() => f.ends = Some(f.pos + 1),
            _ => self.drop_flight(),
        }
        self.score_shadow(next);
        self.ran_own(0, 1);
        self.draft.held(t, last, next)
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::rc::Rc;

    use super::{Batch, Choosing, Chosen, Mode, Rule, Tally, Wall, WidthError, expected};
    use crate::mock::{Call, Clockwork, Mock, MockError, Quiet};
    use crate::{
        Advance, Committed, Draft, Plain, Speculative, Stop, TapNeed, Target, Want, generate,
    };

    /// The draft's own walk on the mock's clock, µs, every proposal; a chain
    /// draft's, every id.
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
    /// target's, and it queues a held position until its next call. `C` is
    /// [`Draft::CHAIN`]: a chain draft's proposal walks one id at a time.
    struct Scored<const C: bool = false> {
        acc: Rc<Cell<[f64; 3]>>,
        p: Rc<Cell<[f32; 3]>>,
        draws: [u64; 3],
        clock: Option<Rc<Clockwork>>,
        at: u32,
        queued: u32,
        proposals: u64,
        /// The ids every proposal held, summed.
        walked: u64,
        held: u64,
        unproposed: u64,
        plains: u64,
        /// Proposals waiting for their accept: one a proposal, dropped by
        /// `unproposed`.
        open: u64,
    }

    impl<const C: bool> Scored<C> {
        fn new(acc: &Rc<Cell<[f64; 3]>>, p: &Rc<Cell<[f32; 3]>>) -> Scored<C> {
            Scored {
                acc: Rc::clone(acc),
                p: Rc::clone(p),
                draws: [0; 3],
                clock: None,
                at: 0,
                queued: 0,
                proposals: 0,
                walked: 0,
                held: 0,
                unproposed: 0,
                plains: 0,
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

    impl<const C: bool> Draft<Mock> for Scored<C> {
        const WIDTH: usize = 3;
        const TAPS: TapNeed = TapNeed::None;
        const CHAIN: bool = C;

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
            let n = out.len().min(3);
            if let Some(c) = &self.clock {
                c.advance(if C { DRAFT_US * n as u64 } else { DRAFT_US });
            }
            self.proposals += 1;
            self.walked += n as u64;
            self.open += 1;
            let truth = t.greedy_after(last, 3);
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

        fn before_plain(&mut self, t: &mut Mock, _last: u32) -> Result<(), MockError> {
            self.flush();
            assert_eq!(
                self.at,
                t.pos(),
                "the waiting rows were not walked before the step"
            );
            self.plains += 1;
            Ok(())
        }
    }

    /// A chooser over a [`Scored`] draft on the mock, standing after a
    /// prompt, its steps costing `STEP_US` and its verifies `by_rows`.
    struct Rig<const C: bool = false> {
        t: Mock,
        spec: Speculative<Choosing<Scored<C>, Rc<Clockwork>>, 4>,
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
            Rig::build(acc, p, by_rows, mode, rule)
        }
    }

    impl<const C: bool> Rig<C> {
        fn build(acc: [f64; 3], p: [f32; 3], by_rows: &[u64], mode: Mode, rule: Rule) -> Rig<C> {
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

        fn choosing(&mut self) -> &mut Choosing<Scored<C>, Rc<Clockwork>> {
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

    /// A draft of one id, the target's own next token at the rate `acc` (one
    /// counter, as [`Scored`]'s), with no probabilities of its own: the
    /// engine's card draft, whose `a` is the calibration gain alone.
    struct Single {
        acc: Rc<Cell<f64>>,
        draws: u64,
    }

    impl Draft<Mock> for Single {
        const WIDTH: usize = 1;
        const TAPS: TapNeed = TapNeed::None;

        fn begin(&mut self, _t: &Mock, _prompt: &[u32], _first: u32) -> Result<(), MockError> {
            Ok(())
        }

        fn propose(
            &mut self,
            t: &mut Mock,
            last: u32,
            out: &mut [u32],
        ) -> Result<usize, MockError> {
            let (a, k) = (self.acc.get(), self.draws as f64);
            self.draws += 1;
            let truth = t.greedy_after(last, 1)[0];
            out[0] = if ((k + 1.0) * a).floor() > (k * a).floor() {
                truth
            } else {
                (truth + 1) % 5
            };
            Ok(1)
        }

        fn accept(
            &mut self,
            _t: &mut Mock,
            _rows: &[u32],
            _out: &[u32],
            _accepted: usize,
        ) -> Result<(), MockError> {
            Ok(())
        }

        fn stepped(&mut self, _t: &mut Mock, _last: u32, _next: u32) -> Result<(), MockError> {
            Ok(())
        }
    }

    /// The plain step and the pair pass (a verify of 2 rows) on the mock's
    /// clock, µs: the pair at 1.6 of the step, inside the engine's DSpark
    /// band on the 3090 [derived from its runs: 1.55–1.75].
    const PLAIN_US: u64 = 40_000;
    const PAIR_US: u64 = 64_000;

    /// A [`Single`] draft behind the chooser of [`Rule::DEFAULT`] on the
    /// mock, standing after a prompt.
    struct PairRig {
        t: Mock,
        spec: Speculative<Choosing<Single, Rc<Clockwork>>, 2>,
        clock: Rc<Clockwork>,
        acc: Rc<Cell<f64>>,
        last: u32,
    }

    impl PairRig {
        fn new(acc: f64) -> PairRig {
            PairRig::with(acc, Rule::DEFAULT)
        }

        fn with(acc: f64, rule: Rule) -> PairRig {
            let clock = Clockwork::new(PLAIN_US, PAIR_US);
            let acc = Rc::new(Cell::new(acc));
            let draft = Single {
                acc: Rc::clone(&acc),
                draws: 0,
            };
            let choosing = draft
                .choosing_by(Mode::Cost, rule, Rc::clone(&clock))
                .unwrap();
            let mut spec = Speculative::new(choosing);
            let mut t = Mock::new()
                .with_ctx(1_000_000)
                .with_clock(Rc::clone(&clock));
            let prompt = [1, 2, 3, 1, 2];
            let last = spec.prompt(&mut t, &prompt).unwrap();
            spec.begin(&t, &prompt, last).unwrap();
            PairRig {
                t,
                spec,
                clock,
                acc,
                last,
            }
        }

        /// One pass: the gate after it, and whether it verified a proposal.
        fn pass(&mut self) -> (usize, bool) {
            let mut out = Vec::with_capacity(2);
            let c = self.spec.pass(&mut self.t, self.last, &mut out).unwrap();
            self.last = *out.last().unwrap();
            (self.spec.draft().width(), c.proposed)
        }
    }

    /// A closed gate re-reads the stale cost it closed on: a warm-up whose
    /// pair passes ran 1.25× slow (c1/c0 2.0 against a nominal 1.6) closes a
    /// draft of one id with no probabilities at the first decision — its
    /// gain 0.864 after the warm-up's 1 kept in 4, so 1.864 / 2.0 of the plain
    /// step's rate — though at 0.9 kept and the nominal cost it pays (1.9 /
    /// 1.6). The first closed probe, `probe_every` passes after the close,
    /// reads 1.6, on which the gate would open: the width runs until it holds
    /// `min_timings` readings from that one on, and the gate opens on their
    /// median. [derived] Open again at the close + `probe_every` +
    /// `min_timings` (pass 9 + 64 + 4 = 77): 68 closed passes, 4 of them
    /// verifies. Before, one reading a probe 256 passes apart joined the
    /// 8-reading median, which the 4th moved to 1.8: about 1030 closed passes.
    #[test]
    fn a_slow_warm_up_does_not_hold_the_gate_closed() {
        let rule = Rule::DEFAULT;
        let warm = u32::try_from(rule.min_timings).unwrap();
        let mut r = PairRig::new(0.25);
        r.clock.verify.set(PAIR_US * 5 / 4);
        for _ in 0..warm {
            assert_eq!(r.pass(), (1, true), "the warm-up's pair passes");
        }
        r.clock.verify.set(PAIR_US);
        r.acc.set(0.9);
        let (mut close, mut reopen) = (None, None);
        for i in warm + 1..=1_500 {
            match (r.pass().0, close) {
                (0, None) => close = Some(i),
                (w, Some(_)) if w > 0 && reopen.is_none() => reopen = Some(i),
                _ => {}
            }
        }
        let close = close.expect("the slow warm-up did not close the gate");
        assert!(
            close <= 2 * warm + 1,
            "closed at pass {close}, after the first decision"
        );
        let bound = close + rule.probe_every + warm;
        assert!(
            reopen.is_some_and(|p| p <= bound),
            "closed at pass {close}, open again at pass {reopen:?}: the bound is pass {bound}"
        );
        let t = r.spec.draft_mut().take_tally();
        assert_eq!(
            t.closed,
            u64::from(reopen.unwrap() - close),
            "the tally's closed passes: {t:?}"
        );
    }

    /// The re-read stays a probe, never a habit: a draft that does not pay
    /// (0.2 kept at c1/c0 1.6, 1.2 / 1.6 of the plain step's rate) closes,
    /// stays closed, and its probes back off to one verify in
    /// `probe_closed` passes. [derived] 5 verifies in the first 1024 passes
    /// after the close (at +64, +192, +448, +704, +960; before, 3 at +256,
    /// +512, +768) and 16 in a window of 4096 passes after those, as before.
    #[test]
    fn a_draft_that_does_not_pay_probes_once_a_long_period() {
        let rule = Rule::DEFAULT;
        let mut r = PairRig::new(0.2);
        let mut close = None;
        let mut verifies = Vec::new();
        for i in 1..=6_500u32 {
            let (w, verified) = r.pass();
            match (w, close) {
                (0, None) => close = Some(i),
                (w, Some(c)) => assert_eq!(w, 0, "closed at pass {c}, open again at pass {i}"),
                _ => {}
            }
            if verified && close.is_some_and(|c| i > c) {
                verifies.push(i - close.unwrap());
            }
        }
        let close = close.expect("the losing draft did not close");
        // The probes at +64, +192, +448, +704 and +960.
        let first_bound = 5;
        let span = 4_096;
        let first = verifies.iter().filter(|&&d| d < 1_024).count();
        let steady = verifies
            .iter()
            .filter(|&&d| (1_024..1_024 + span).contains(&d))
            .count();
        let bound = usize::try_from(span / rule.probe_closed).unwrap();
        assert!(
            first <= first_bound && steady <= bound,
            "closed at pass {close}: {first} verifies in its first 1024 passes (bound \
             {first_bound}), {steady} in the next {span} (bound {bound})"
        );
    }

    /// A draft with no probabilities is rated by its gain alone, and under
    /// [`Rule::prior`] the gain is the mean of what its tests kept: a draft
    /// of one id that keeps 0.18 of them, its pair pass 1.35 of the step,
    /// closes by pass 12 under [`crate::Lookup::RULE`], where the fixed step
    /// reads its gain down from 1 and holds it open past pass 40; one that
    /// keeps 0.7 stays open.
    #[test]
    fn a_draft_with_no_probabilities_closes_on_what_it_kept() {
        let close = |acc: f64, rule: Rule| {
            let mut r = PairRig::with(acc, rule);
            r.clock.verify.set(PLAIN_US * 135 / 100);
            (1..=1_000u32).find(|_| r.pass().0 == 0)
        };
        let lookup = crate::Lookup::RULE;
        let (fast, slow) = (close(0.18, lookup), close(0.18, Rule::DEFAULT));
        assert!(
            fast.is_some_and(|p| p <= 12) && slow.is_some_and(|p| p > 40),
            "closed at pass {fast:?} by the prior, {slow:?} by the fixed step"
        );
        assert_eq!(close(0.7, lookup), None, "a draft that pays closed");
    }

    /// The chooser's stats are what it predicted and what it kept: over a
    /// calibrated draft's windows the mean `E` it predicted for the width
    /// each ran is the mean rows they kept (the acceptances drawn exactly, so
    /// the two agree within the calibration's own spread, 3 %), the realized
    /// mean is the tally's `E`, the calibration and the mean profile are one
    /// a proposal index; a losing draft's closed passes are the tally's; a
    /// take starts the count over (no window, no `E`); and `Mode::Fixed`
    /// counts nothing.
    #[test]
    fn the_stats_hold_what_was_predicted_and_kept() {
        let mut r = Rig::of([0.80, 0.8125, 0.846], &LINEAR);
        r.run(1_000);
        let s = r.choosing().stats().expect("a cost chooser has stats");
        let t = r.choosing().take_tally();
        let (predicted, realized) = (s.predicted_e.unwrap(), s.realized_e.unwrap());
        assert_eq!(s.windows, t.windows, "{s:?}");
        assert!((realized - t.e()).abs() < 1e-12, "{s:?} {t:?}");
        assert!(
            (predicted / realized - 1.0).abs() < 0.03,
            "predicted {predicted} against kept {realized}"
        );
        assert_eq!((s.gains.len(), s.mean.len()), (3, 3), "{s:?}");
        assert!(s.mean.iter().all(|&p| (0.7..0.9).contains(&p)), "{s:?}");
        let s = r.choosing().stats().unwrap();
        assert_eq!((s.windows, s.predicted_e, s.realized_e), (0, None, None));
        let mut r = Rig::of([0.40, 0.20, 0.10], &[71_000, 95_000, 118_000]);
        r.run(400);
        let s = r.choosing().stats().unwrap();
        let t = r.choosing().take_tally();
        assert!(s.closed > 0 && s.closed == t.closed, "{s:?} {t:?}");
        let mut r = Rig::with([0.8; 3], [0.8; 3], &LINEAR, Mode::Fixed, Rule::DEFAULT);
        r.run(50);
        assert_eq!(r.choosing().stats(), None);
    }

    #[test]
    fn gate_diagnostics_include_the_rates_and_walk_cost() {
        let mut r = Rig::<true>::build(
            [0.95, 0.90, 0.30],
            [0.95, 0.90, 0.30],
            &LINEAR,
            Mode::Cost,
            Rule::DEFAULT,
        );
        r.run(200);
        let c = r.choosing();
        c.medians
            .copy_from_slice(&[Some(1.0), Some(2.0), Some(3.0), Some(4.0)]);
        c.gains.fill(1.0);
        (c.recent_len, c.recent_at) = (0, 0);
        for p in [[1.0; 3], [0.0; 3]] {
            c.p.copy_from_slice(&p);
            c.learn(3);
        }
        c.walk_ms = 0.5;
        let state = c.gate_state();
        let stats = c.stats().unwrap();
        assert_eq!(state.rates, stats.rates);
        assert_eq!((state.walk_ms, stats.walk_ms), (0.5, 0.5));
        for (rate, want) in state.rates.iter().zip([1.0, 0.75, 8.0 / 11.0, 5.0 / 7.0]) {
            assert!((rate.unwrap() - want).abs() < 1e-12, "{state:?}");
        }
        c.medians[2] = None;
        assert_eq!(c.gate_state().rates, [Some(1.0), Some(0.75), None, None]);
        c.recent_len = 0;
        assert_eq!(c.gate_state().rates, [Some(1.0), None, None, None]);
    }

    /// A seeded stream of values in `[-1, 1]` (xorshift64).
    struct Noise(u64);

    impl Noise {
        fn next(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 52) as f64 - 1.0
        }
    }

    /// The rig's step and verify walls each scaled by `1 + j·u`, `u` a fresh
    /// draw in `[-1, 1]` per wall: readings of standard deviation `j / √3`
    /// about their wall, and no reading outside it by more than `j`.
    fn jitter<const C: bool>(r: &Rig<C>, by_rows: [u64; 3], j: f64, noise: &mut Noise) {
        let mut wall = |us: u64| (us as f64 * (1.0 + j * noise.next())).round() as u64;
        r.clock.step.set(wall(STEP_US));
        r.clock
            .verify_rows(&[wall(by_rows[0]), wall(by_rows[1]), wall(by_rows[2])]);
    }

    /// Passes of `n` at which the gate stood closed, the rig's walls
    /// jittered by `j` ([`jitter`]) before each pass.
    fn closed_passes(r: &mut Rig, by_rows: [u64; 3], j: f64, seed: u64, n: usize) -> usize {
        let mut noise = Noise(seed);
        let mut closed = 0;
        for _ in 0..n {
            jitter(r, by_rows, j, &mut noise);
            r.run(1);
            closed += usize::from(r.choosing().width() == 0);
        }
        closed
    }

    /// A drafted width ahead of the plain step by 2 % at the rig's walls
    /// (width 1 of acceptance 0.4244: E 1.4244 over the verify of two rows
    /// and the draft's walk, against the step) and the others behind it
    /// [derived]: 1.4244 · 37 450 / 52 300 = 1.0198.
    const THIN: [f64; 3] = [0.4244, 0.10, 0.05];

    /// Noise inside the close band does not close an open draft: the best
    /// width ahead of the plain step by 2 %, under the margin that opens a
    /// closed one, with each wall jittered by the spread the margin is
    /// derived from ([`Rule::margin_close`]), stays open over 1024 passes on
    /// each of 8 seeds. Closing at the first median that puts the plain step
    /// ahead closes it on the first unlucky draw.
    #[test]
    fn noise_inside_the_band_does_not_close_the_draft() {
        for seed in 1..=8u64 {
            let mut r = Rig::of(THIN, &LINEAR);
            let closed = closed_passes(
                &mut r,
                LINEAR,
                JITTER,
                seed.wrapping_mul(0x9E37_79B9_7F4A_7C15),
                1_024,
            );
            assert_eq!(
                closed, 0,
                "seed {seed}: the gate stood closed for {closed} passes"
            );
        }
    }

    /// A chain draft's gate inside the margin of a narrower width holds its
    /// width against noise: a draft that keeps every id and reports so,
    /// width 3's verify slowed to put it 2 % past width 2 [derived: E 4 over
    /// 87 790 against 3 over 67 160, in the rig's µs]. No window cuts below
    /// 3, and with each wall jittered as in the close-band test the gate stays
    /// on 3 over 1024 passes on each of 8 seeds. A gate that gave way to a
    /// narrower width on a tie would move on the first unlucky draw.
    #[test]
    fn a_chain_gate_inside_the_margin_holds_against_noise() {
        let walls = [LINEAR[0], LINEAR[1], 86_290];
        for seed in 1..=8u64 {
            let mut r = Rig::<true>::build([1.0; 3], [1.0; 3], &walls, Mode::Cost, Rule::DEFAULT);
            r.run(64);
            let mut noise = Noise(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let mut off = 0;
            for _ in 0..1_024 {
                jitter(&r, walls, JITTER, &mut noise);
                r.run(1);
                off += usize::from(r.choosing().width() != 3);
            }
            assert_eq!(off, 0, "seed {seed}: the gate stood off 3 for {off} passes");
        }
    }

    /// A drafted width ahead of the plain step by 15 % [derived]: acceptance
    /// 0.606 on width 1, 1.606 · 37 450 / 52 300 = 1.150; the others behind.
    const AHEAD: [f64; 3] = [0.606, 0.10, 0.05];

    /// The readings' spread within a run of the paged plan, as a standard
    /// deviation of the walls about their means, and the jitter that makes
    /// it: uniform `±j` has a standard deviation of `j / √3`.
    const SPREAD: f64 = 0.25;

    /// A draft that pays by 15 % under the spread the run's pass walls show
    /// is closed for 8 % of 1024 passes at most. The gate looks at nearly
    /// every pass and its medians of 8 spread by ten percent, so no rule
    /// keeps it open through a run: the one that closes on the first median
    /// that puts the plain step ahead, or on the close band alone, leaves it
    /// closed for more than that; the dwell, which makes a close that follows
    /// a reopen rest on readings taken since, keeps it under.
    /// PIN(2026-10-13): 8 % of the passes, between the close band alone and
    /// the band with the dwell on these 40 seeds.
    #[test]
    fn a_draft_that_pays_under_the_runs_spread_is_closed_for_8_percent_at_most() {
        let (seeds, passes) = (40u64, 1_024usize);
        let mut closed = 0;
        for seed in 1..=seeds {
            let mut r = Rig::of(AHEAD, &LINEAR);
            let jitter = SPREAD * 3f64.sqrt();
            let noise = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
            closed += closed_passes(&mut r, LINEAR, jitter, noise, passes);
        }
        let total = seeds as usize * passes;
        assert!(
            closed * 100 <= total * 8,
            "closed for {closed} of {total} passes: over 8 %"
        );
    }

    /// The passes between a reopen and the close that follows it, the draft
    /// turning bad at the reopen: the gate closes on a slow warm-up's
    /// readings, opens again on the first probe's, and then the draft's
    /// acceptance falls to 0.05 and its verify to 1.5 of its cost.
    fn shut_after_reopen(rule: Rule) -> u32 {
        let warm = u32::try_from(rule.min_timings).unwrap();
        let mut r = PairRig::with(0.25, rule);
        r.clock.verify.set(PAIR_US * 5 / 4);
        for _ in 0..warm {
            r.pass();
        }
        r.clock.verify.set(PAIR_US);
        r.acc.set(0.9);
        let (mut closed, mut reopen) = (false, None);
        for i in warm + 1..=3_000u32 {
            let w = r.pass().0;
            match (closed, reopen) {
                (false, _) => closed = w == 0,
                (true, None) if w > 0 => {
                    reopen = Some(i);
                    r.acc.set(0.05);
                    r.clock.verify.set(PAIR_US * 3 / 2);
                }
                (true, Some(at)) if w == 0 => return i - at,
                _ => {}
            }
        }
        panic!("the gate did not close, reopen and close again: reopened at {reopen:?}");
    }

    /// A close that follows a reopen waits for the readings since it: on the
    /// band alone the bad draft is closed within a probe period of the
    /// reopen; with the dwell, once the plain step's ring and the pair's hold
    /// [`Rule::dwell`] readings taken since the reopen, and the plain step
    /// reads once a probe while open, so at least `dwell · probe_every`
    /// passes after it. The dwell is a majority of the ring.
    #[test]
    fn a_close_after_a_reopen_waits_for_the_readings_since() {
        let rule = Rule::DEFAULT;
        assert_eq!(rule.dwell, rule.timings / 2 + 1, "a majority of the ring");
        let bare = shut_after_reopen(Rule { dwell: 0, ..rule });
        let held = shut_after_reopen(rule);
        let at_least = u32::try_from(rule.dwell).unwrap() * rule.probe_every;
        assert!(
            bare < rule.probe_every,
            "without the dwell the close came {bare} passes after the reopen"
        );
        assert!(
            held >= at_least,
            "closed {held} passes after the reopen: the dwell is {at_least}"
        );
    }

    /// The jitter of the band test: walls scaled by up to ±`JITTER`, a
    /// standard deviation of `JITTER / √3`.
    const JITTER: f64 = 0.035;

    /// A uniform speed-up does not move the choice: every step and verify
    /// wall halves at one pass (the draft's walk stays), and the gate and its
    /// width stay where the rates put them, for each of the settled cases of
    /// [`picks_the_width_that_pays`]. The rings of the widths the incumbent
    /// does not run hold the old walls until their probes, and the choice
    /// holds meanwhile.
    #[test]
    fn a_uniform_speed_up_keeps_the_choice() {
        let losing = [71_000, 95_000, 118_000];
        for (acc, by_rows, what) in [
            ([0.80, 0.8125, 0.846], LINEAR, "prose"),
            ([0.95, 0.90, 0.30], LINEAR, "a losing third id"),
            ([0.75, 0.55, 0.35], LINEAR, "chat"),
            ([0.40, 0.20, 0.10], losing, "a losing draft"),
        ] {
            let mut r = Rig::of(acc, &by_rows);
            r.run(1_000);
            let before = r.choosing().width();
            r.clock.step.set(STEP_US / 2);
            r.clock.verify_rows(&by_rows.map(|us| us / 2));
            for i in 0..2_000 {
                r.run(1);
                assert_eq!(
                    r.choosing().width(),
                    before,
                    "{what}: the choice left {before} at pass {i} after the speed-up"
                );
            }
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

    /// A chain draft walks the width the pass plans to verify and no more:
    /// on a profile whose third id loses it settles on 2, the ids its
    /// proposals held equal the ids the passes verified, and its positions a
    /// microsecond come within 2 % of the best fixed width's at a walk an id.
    /// Handed its whole width, it would walk a third id every window.
    #[test]
    fn a_chain_draft_walks_the_width_it_verifies() {
        let acc = [0.95, 0.90, 0.30];
        let mut r = Rig::<true>::build(
            acc,
            acc.map(|a| a as f32),
            &LINEAR,
            Mode::Cost,
            Rule::DEFAULT,
        );
        r.run(600);
        let (walked, calls) = (r.choosing().draft().walked, r.rows().len());
        let (t0, p0) = (r.clock.now.get(), r.t.pos());
        r.run(400);
        assert_eq!(r.choosing().width(), 2, "settled elsewhere");
        let verified: usize = r.rows()[calls..].iter().map(|m| m - 1).sum();
        let walked = r.choosing().draft().walked - walked;
        assert_eq!(walked, verified as u64, "walked past the verified width");
        let rate = f64::from(r.t.pos() - p0) / (r.clock.now.get() - t0) as f64;
        let best = (1..=3)
            .map(|k| expected(&acc, k) / (LINEAR[k - 1] + k as u64 * DRAFT_US) as f64)
            .fold(1.0 / STEP_US as f64, f64::max);
        assert!(
            rate >= best * 0.98,
            "{:.4} of the best fixed width's rate",
            rate / best
        );
    }

    /// A wider probe that still loses reads its cost once and leaves the
    /// incumbent free to run on the next pass.
    #[test]
    fn a_losing_chain_probe_does_not_restart_its_cost() {
        let acc = [0.95, 0.90, 0.30];
        let rule = Rule::DEFAULT;
        let mut r = Rig::<true>::build(acc, acc.map(|a| a as f32), &LINEAR, Mode::Cost, rule);
        r.run(600);
        let c = r.choosing();
        assert_eq!(c.width(), 2);
        assert!(c.warm.is_empty());
        c.read_at[3] = 0;
        c.since_probe = rule.probe_every;
        let calls = r.rows().len();
        r.run(2);
        assert_eq!(&r.rows()[calls..], &[4, 3], "the losing probe ran again");
        assert!(r.choosing().warm.is_empty());

        // A refreshed profile can pay at the old median and lose at the
        // new wall: the old median must not open the gate meanwhile.
        let mut r = Rig::<true>::build(acc, acc.map(|a| a as f32), &LINEAR, Mode::Cost, rule);
        r.run(600);
        let c = r.choosing();
        c.read_at[3] = 0;
        c.since_probe = rule.probe_every;
        r.acc.set([1.0; 3]);
        r.choosing().draft().p.set([1.0; 3]);
        r.clock.verify_rows(&[LINEAR[0], LINEAR[1], 160_000]);
        r.run(2);
        let c = r.choosing();
        assert_eq!(c.width(), 2, "the stale median moved the gate");
        assert_eq!(c.warm.front(), Some(&3));
    }

    /// Width 3's verify 20 % slow while `on`, the rig's walls otherwise.
    fn slow_three<const C: bool>(r: &mut Rig<C>, on: bool) {
        let mut walls = LINEAR;
        if on {
            walls[2] = walls[2] * 6 / 5;
        }
        r.clock.verify_rows(&walls);
    }

    /// The third id kept, and reported, at 0.30 while `on`, at 0.95
    /// otherwise.
    fn loses_three<const C: bool>(r: &mut Rig<C>, on: bool) {
        let a = if on { [0.95, 0.90, 0.30] } else { [0.95; 3] };
        r.acc.set(a);
        r.choosing().draft().p.set(a.map(|a| a as f32));
    }

    /// The pass the gate leaves 3 and the pass it is back on it, within
    /// 1 000 passes of a draft that keeps 0.95 of every id, settled on 3:
    /// `bad` turns width 3 bad, and back as soon as the gate leaves it.
    fn left_and_back<const C: bool>(bad: fn(&mut Rig<C>, bool)) -> (usize, Option<usize>) {
        let sure = [0.95; 3];
        let mut r = Rig::<C>::build(
            sure,
            sure.map(|a| a as f32),
            &LINEAR,
            Mode::Cost,
            Rule::DEFAULT,
        );
        r.run(200);
        assert_eq!(r.choosing().width(), 3, "settled elsewhere");
        bad(&mut r, true);
        let mut left = None;
        for i in 1..=1_000 {
            r.run(1);
            match (left, r.choosing().width()) {
                (None, 3) | (Some(_), 0..=2) => {}
                (None, w) => {
                    assert_eq!(w, 2, "the gate left 3 for {w}");
                    bad(&mut r, false);
                    left = Some(i);
                }
                (Some(l), _) => return (l, Some(i)),
            }
        }
        (left.expect("the gate never left 3"), None)
    }

    /// A width the gate left comes back once it pays again. A draft whose
    /// windows the rig's draws keep whole 19 times in 20 runs 3 (E 3.85, 9 %
    /// past 2's rate on the rig's walls [derived]) until width 3 turns bad,
    /// its verify 20 % slow or its third id kept at 0.30, and the gate moves
    /// to 2. Width 3 restored as the gate leaves it, a probe re-reads it,
    /// its readings start over and the gate is back on 3 once they hold
    /// `min_timings`. The move keeps the probe period and the readings'
    /// ages, so the probes take the widths other than 2 oldest first: back
    /// no sooner than `min_timings` after the move and within three periods
    /// and `min_timings` of it. A draft that proposes its whole width hears
    /// its third position every pass, so a third id kept again brings it
    /// back sooner; no pass at 2 of a chain draft reaches it.
    #[test]
    fn a_width_the_gate_left_comes_back_once_it_pays() {
        let rule = Rule::DEFAULT;
        let probe = usize::try_from(rule.probe_every).unwrap();
        let (start, by) = (rule.min_timings, 3 * probe + rule.min_timings);
        let runs = [
            ("slow", start, left_and_back::<false>(slow_three)),
            ("slow, chain", start, left_and_back::<true>(slow_three)),
            ("losing", 1, left_and_back::<false>(loses_three)),
            ("losing, chain", start, left_and_back::<true>(loses_three)),
        ];
        assert!(
            runs.iter().all(|&(_, after, (left, back))| {
                back.is_some_and(|b| (left + after..=left + by).contains(&b))
            }),
            "the passes the gate left 3 and came back, each back no sooner than the passes \
             named after it left and within {by}: {runs:?}"
        );
    }

    /// The first of `passes` passes of a chain draft whose windows come in
    /// 40-pass blocks of `blocks[0]` and `blocks[1]` in turn, each its
    /// acceptance and reported probability, after which width `k`'s median
    /// reads under `ms`; `None` if none.
    fn read_under(
        r: &mut Rig<true>,
        blocks: [[f64; 3]; 2],
        k: usize,
        ms: f64,
        passes: usize,
    ) -> Option<usize> {
        (1..=passes).find(|i| {
            let a = blocks[(i - 1) / 40 % 2];
            r.acc.set(a);
            r.choosing().draft().p.set(a.map(|a| a as f32));
            r.run(1);
            r.choosing().medians[k].is_some_and(|m| m < ms)
        })
    }

    /// Moves between open widths keep the probe period and each width's
    /// reading age, so request profiles that move the gate back and forth
    /// faster than a period still let every cost be re-read: within one
    /// probe of each width but the gate's and one width's `min_timings`.
    /// A chain draft on the rig's walls with a 50 000 step, whose plain step
    /// drops to 25 000 as blocks of windows that keep nothing and windows
    /// that keep everything start to alternate; and one whose warm-up read
    /// a 60 000 step and width 3 at 160 000, restored to the rig's as blocks
    /// keeping every id and keeping none but the first id's two successors
    /// alternate. Restarting the period at each move, the gate's moves
    /// about 20 and 60 passes apart run no probe: neither cost moves in
    /// 4 000 passes [derived by state simulation].
    #[test]
    fn alternating_requests_still_reread_every_cost() {
        let rule = Rule::DEFAULT;
        let probe = usize::try_from(rule.probe_every).unwrap();
        let by = 3 * probe + rule.min_timings;
        let sure = [1.0; 3];
        let mut r = Rig::<true>::build(sure, [1.0; 3], &LINEAR, Mode::Cost, rule);
        r.clock.step.set(50_000);
        r.run(100);
        r.clock.step.set(25_000);
        let plain = read_under(&mut r, [[0.0; 3], sure], 0, 40.0, 4_000);
        let mut r = Rig::<true>::build(
            sure,
            [1.0; 3],
            &[LINEAR[0], LINEAR[1], 160_000],
            Mode::Cost,
            rule,
        );
        r.clock.step.set(60_000);
        r.run(100);
        r.clock.verify_rows(&LINEAR);
        let three = read_under(&mut r, [sure, [0.0, 1.0, 1.0]], 3, 100.0, 4_000);
        assert!(
            [plain, three].iter().all(|p| p.is_some_and(|p| p <= by)),
            "the plain step re-read at pass {plain:?}, width 3 at pass {three:?}: the bound is \
             pass {by}"
        );
    }

    /// A chain draft whose windows come sure and unsure in turn, each sure
    /// at every position or unsure at every one: the mean of their profiles
    /// rates the narrowest width highest, the windows rate the widest, each
    /// unsure one cut to 1. The gate holds 3, at 98 % of that policy's rate
    /// at least, and the cut windows' walls read the walk past their width:
    /// the rig's walk of one id.
    #[test]
    fn mixed_windows_hold_the_width_the_sure_ones_pay() {
        let sure = ([1.0; 3], [0.9f32; 3]);
        let unsure = ([0.0; 3], [0.3f32, 0.2, 0.1]);
        let mut r = Rig::<true>::build(sure.0, sure.1, &LINEAR, Mode::Cost, Rule::DEFAULT);
        let run = |r: &mut Rig<true>, passes: std::ops::Range<usize>| {
            for i in passes {
                let (acc, p) = if i % 2 == 0 { sure } else { unsure };
                r.acc.set(acc);
                r.choosing().draft().p.set(p);
                r.run(1);
            }
        };
        run(&mut r, 0..1_000);
        let (at, pos) = (r.clock.now.get(), r.t.pos());
        run(&mut r, 1_000..3_000);
        let rate = f64::from(r.t.pos() - pos) / (r.clock.now.get() - at) as f64;
        let walk = 3 * DRAFT_US;
        let policy = 5.0 / (LINEAR[2] + walk + LINEAR[0] + walk) as f64;
        assert_eq!(r.choosing().width(), 3, "the gate left the width that pays");
        assert!(rate >= 0.98 * policy, "{rate} under 98 % of {policy}");
        let walk = r.choosing().walk_ms;
        assert!(
            (walk - DRAFT_US as f64 / 1e3).abs() < 1e-6,
            "walk {walk} ms"
        );
    }

    /// Short requests, each started over by `begin` and shorter than a probe
    /// period (128 positions, about 50 passes), hold the gate a chain draft's
    /// windows pay and cut each window to the width its own profile pays at
    /// each width's own cost. Walls in the proportions of a GB10 MTP pass
    /// [derived from its `mtp width` costs], in the rig's µs: the step
    /// 14 000, verifies of 2, 3 and 4 rows 15 500, 19 300 and 20 000, the
    /// rig's walk 500 an id, so widths 1 and 3 cost 16 000 and 21 500 and a
    /// window cut to 1 after a walk of 3 costs 17 000. The windows come sure,
    /// mid, sure, unsure: the sure keep every id, the mid keep ids at 0.6,
    /// 4/7 and 2/9 and say so (E(1) 1.6, E(3) 2.019 [derived]), the unsure
    /// keep none. Width 3 adds 0.419 rows to a mid window for 4 500, under
    /// the rows the run keeps in that time, so the run that pays verifies the
    /// sure at 3 and cuts the rest to 1: 10.6 rows in 77 000. The mid at 3
    /// too keeps 11.019 in 81 500, 98.2 % of it [derived].
    /// The gate stays on 3 from the second request on, and the rate is within
    /// 0.5 % of the run that pays. Pricing the walk past a width into every
    /// width's wall rates the mid at 3 (E(1) / 17 against E(3) / 21.5, under
    /// the margin apart) and runs them all wide.
    #[test]
    fn short_requests_hold_the_gate_and_cut_what_does_not_pay() {
        let walls = [15_500, 19_300, 20_000];
        let sure = ([1.0; 3], [1.0f32; 3]);
        let mid = ([0.6, 4.0 / 7.0, 2.0 / 9.0], [0.6f32, 4.0 / 7.0, 2.0 / 9.0]);
        let unsure = ([0.0; 3], [0.0f32; 3]);
        let mut r = Rig::<true>::build(sure.0, sure.1, &walls, Mode::Cost, Rule::DEFAULT);
        r.clock.step.set(14_000);
        let (mut at, mut pos, mut off, mut i) = (0, 0, 0, 0usize);
        for request in 0..24 {
            if request == 4 {
                (at, pos) = (r.clock.now.get(), r.t.pos());
            }
            r.spec.begin(&r.t, &[1, 2, 3], r.last).unwrap();
            let start = r.t.pos();
            while r.t.pos() - start < 128 {
                let (acc, p) = [sure, mid, sure, unsure][i % 4];
                r.acc.set(acc);
                r.choosing().draft().p.set(p);
                r.run(1);
                off += usize::from(request > 0 && r.choosing().width() != 3);
                i += 1;
            }
        }
        let rate = f64::from(r.t.pos() - pos) / (r.clock.now.get() - at) as f64;
        let pays = 10.6 / 77_000.0;
        assert_eq!(off, 0, "the gate stood off 3 for {off} passes");
        assert!(
            rate >= 0.995 * pays,
            "{:.4} of the rate of the run that pays",
            rate / pays
        );
    }

    /// The gate moves on the recent windows alone and keeps its margin: an
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
        let gate = |c: &mut Choosing<Scored>, width: usize, mean: [f32; 3]| {
            c.width = width;
            (c.recent_len, c.recent_at) = (0, 0);
            c.seen.fill(0);
            c.p.copy_from_slice(&mean);
            c.learn(3);
            c.gate();
            c.width
        };
        // The best drafted width 1 % above the plain step: open stays open,
        // closed stays closed; 1 % below it, open stays open (inside the
        // close band); 6 % below it, past 1 / `margin_close`, open closes.
        // PIN(2026-10-13): the 1 % case closed before the close band; the
        // 6 % mean is E 1.504 / 1.6 = 0.94 of the plain step's rate, under
        // 1 / 1.05 = 0.952.
        let thin = [0.616, 0.0, 0.0];
        assert_eq!(gate(&mut c, 1, thin), 1, "closed while a width still paid");
        assert_eq!(gate(&mut c, 0, thin), 0, "opened inside the margin");
        assert_eq!(
            gate(&mut c, 1, [0.584, 0.0, 0.0]),
            1,
            "closed inside the close band"
        );
        assert_eq!(
            gate(&mut c, 1, [0.504, 0.0, 0.0]),
            0,
            "stayed open past the close band"
        );
        assert_eq!(
            gate(&mut c, 0, [0.70, 0.0, 0.0]),
            1,
            "did not open past the margin"
        );
        // One low window into settled ones: the gate stays open.
        let window = |c: &mut Choosing<Scored>, p: [f32; 3]| {
            c.p.copy_from_slice(&p);
            c.profile(3);
            c.learn(3);
            c.gate();
            c.cut(3)
        };
        c.width = 3;
        for _ in 0..Rule::DEFAULT.recent() {
            c.p.copy_from_slice(&[0.8, 0.81, 0.85]);
            c.learn(3);
        }
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

    /// A request whose prompt call finds the gate closed is judged on its own
    /// windows: a draft that lost the last request closes, the next request's
    /// draft keeps nearly every id, and the gate is open within the passes a
    /// closed gate runs before its first shadow, not after the last request's
    /// windows age out one shadow at a time.
    #[test]
    fn a_request_on_a_closed_gate_opens_on_its_own_windows() {
        let rule = Rule::DEFAULT;
        let mut r = Rig::with(
            [0.20, 0.10, 0.05],
            [0.3, 0.2, 0.1],
            &LINEAR,
            Mode::Cost,
            rule,
        );
        r.run(400);
        assert_eq!(r.choosing().width(), 0, "the losing draft did not close");
        r.acc.set([0.95; 3]);
        r.choosing().draft().p.set([0.95; 3]);
        r.last = r.spec.prompt(&mut r.t, &[r.last]).unwrap();
        let within = usize::try_from(rule.shadow_every).unwrap();
        let opened = (1..=within).find(|_| {
            r.run(1);
            r.choosing().width() > 0
        });
        assert!(
            opened.is_some(),
            "the gate stayed closed {within} passes into a request that pays: {:?}",
            r.choosing().gate_state()
        );
        assert!(
            opened >= Some(rule.min_timings),
            "the gate decided at pass {opened:?}, before the request's {} shadows",
            rule.min_timings
        );
    }

    /// A width-one draft reports certainty throughout: only acceptance changes
    /// between requests, so reopening depends on calibrating the new request.
    /// The kept gains adapt over a recent-window horizon at the shadow cadence.
    #[test]
    fn a_probability_one_request_reopens_on_its_own_windows() {
        let rule = Rule::DEFAULT;
        let mut r = PairRig::with(0.2, rule);
        for _ in 0..400 {
            r.pass();
        }
        assert_eq!(r.spec.draft().width(), 0, "the losing draft did not close");
        let gain = r.spec.draft().gains()[0];
        let draws = r.spec.draft().draft().draws;
        r.acc.set(0.95);
        r.last = r.spec.prompt(&mut r.t, &[r.last]).unwrap();
        assert_eq!(
            r.spec.draft().gains()[0],
            gain,
            "the prompt keeps calibration"
        );
        for _ in 0..rule.min_timings {
            r.pass();
        }
        assert_eq!(
            r.spec.draft().draft().draws - draws,
            u64::try_from(rule.min_timings).unwrap(),
            "the new request must supply every opening shadow"
        );
        assert!(
            r.spec.draft().gains()[0] > gain,
            "the new request's shadows must recalibrate the certainty draft"
        );
        let within = rule.recent() * usize::try_from(rule.shadow_every).unwrap();
        let opened = (1..=within).find(|_| r.pass().0 > 0);
        assert!(
            opened.is_some(),
            "the certainty draft stayed closed {within} passes into a request that pays: {:?}",
            r.spec.draft_mut().gate_state()
        );
        assert_eq!(
            r.spec.draft_mut().stats().unwrap().mean,
            [1.0],
            "the draft reports certainty before and after the prompt"
        );
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

    /// A plain pass the chooser holds back walks the draft's waiting rows
    /// before its step — the draft's own check (`before_plain` seeing the
    /// target at the pass's position) — and nothing else does: a pass that
    /// verifies, and a plain pass whose proposal already ran (a shadow, a
    /// cut to 0), hook nothing. Every width-0 pass is one or the other, so
    /// over a run that closes and shadows the hook ran exactly on the held
    /// passes.
    #[test]
    fn a_held_pass_hooks_the_draft_before_its_step() {
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
        let (plains, unproposed, held) = {
            let d = r.spec.draft().draft();
            (d.plains, d.unproposed, d.held)
        };
        let t = r.choosing().take_tally();
        assert!(plains > 0, "no held pass hooked the draft");
        assert!(unproposed > 0, "no shadow ran");
        assert_eq!(
            t.widths[0],
            plains + unproposed,
            "the hook ran on a pass that verifies, or a plain pass missed it: {t:?}"
        );
        assert_eq!(held, t.widths[0], "a plain pass the draft did not hear of");
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
        assert_eq!(
            t.windows,
            t.widths.iter().skip(1).sum::<u64>(),
            "a window is a pass of a width above the plain step: {t:?}"
        );
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

    /// A round's window is counted at the rows it ran, as no closed pass of
    /// the chooser's (the round ran the draft's whole width), and drops what
    /// was in flight — a pending shadow among it — so the next pass reads
    /// nothing of the round's wall.
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
        let gains = r.choosing().gains().to_vec();
        r.choosing().round_ran(4, 2);
        assert_eq!(r.choosing().gains(), gains);
        assert_eq!(r.choosing().batch_profile(), &[1.0, 0.0, 1.0]);
        assert_eq!(r.choosing().shadow.n, 0, "the round left the shadow up");
        assert!(r.choosing().flight.is_none());
        let t = r.choosing().take_tally();
        assert_eq!(
            (t.windows, t.widths[3], t.kept[1], t.closed),
            (1, 1, 1, 0),
            "{t:?}"
        );
        r.choosing().round_ran(4, 4);
        assert_eq!(r.choosing().gains(), gains);
        assert_eq!(r.choosing().batch_profile(), &[1.0, 0.5, 1.0]);
        r.choosing().round_ran(4, 3);
        assert_eq!(r.choosing().gains(), gains);
        assert_eq!(r.choosing().batch_profile(), &[1.0, 2.0 / 3.0, 0.5]);
        Draft::begin(r.spec.draft_mut(), &r.t, &[], r.last).unwrap();
        assert_eq!(r.choosing().gains(), gains);
        assert_eq!(r.choosing().batch_profile(), &[1.0; 3]);
        assert_eq!(r.choosing().batch_tested, [0; 3]);
    }

    #[test]
    fn batch_acceptance_resets_on_an_external_step() {
        for held in [false, true] {
            let mut r = Rig::with(
                [0.40, 0.20, 0.10],
                [0.6, 0.5, 0.4],
                &[71_000, 95_000, 118_000],
                Mode::Cost,
                Rule::DEFAULT,
            );
            r.run(200);
            let gains = r.choosing().gains().to_vec();
            let tested = r.choosing().tested.clone();
            r.choosing().round_ran(4, 1);
            assert_eq!(r.choosing().batch_profile(), &[0.0, 1.0, 1.0]);
            assert_eq!(r.choosing().batch_tested, [1, 0, 0]);

            // A chooser-owned plain pass keeps this request's observations.
            r.choosing().warm.clear();
            r.choosing().warm.push_back(0);
            r.run(1);
            assert_eq!(r.passes.last().unwrap().rows, 1);
            assert_eq!(r.choosing().batch_profile(), &[0.0, 1.0, 1.0]);
            assert_eq!(r.choosing().batch_tested, [1, 0, 0]);

            // A cached request can start with only its last prompt token.
            let next = r.t.step(r.last, Want::Argmax).unwrap().argmax();
            if held {
                Draft::held(r.spec.draft_mut(), &mut r.t, r.last, next).unwrap();
            } else {
                Draft::stepped(r.spec.draft_mut(), &mut r.t, r.last, next).unwrap();
            }
            assert_eq!(r.choosing().batch_profile(), &[1.0; 3]);
            assert_eq!(r.choosing().batch_tested, [0; 3]);
            assert_eq!(r.choosing().gains(), gains);
            assert_eq!(r.choosing().tested, tested);
            assert!(r.choosing().flight.is_none());
            assert_eq!(r.choosing().shadow.n, 0);

            r.choosing().round_ran(4, 4);
            assert_eq!(r.choosing().batch_profile(), &[1.0; 3]);
            assert_eq!(r.choosing().batch_tested, [1; 3]);
            assert_eq!(r.choosing().gains(), gains);
        }
    }

    /// A prompt call of the draft's own shape ([`Choosing::prompt_by`]) forgets the pass in flight
    /// and a pending shadow as [`Draft::prompt`] does, and returns what the call returns.
    #[test]
    fn a_prompt_by_drops_the_flight() {
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
        for _ in 0..50 {
            if r.choosing().shadow.n > 0 {
                break;
            }
            r.run(1);
        }
        assert!(r.choosing().shadow.n > 0, "no shadow pending");
        assert!(r.choosing().flight.is_some(), "no pass in flight");
        let gains = r.choosing().gains().to_vec();
        r.choosing().batch_gains.fill(0.25);
        r.choosing().batch_tested.fill(4);
        let got = r.choosing().prompt_by(|_| 7_u32);
        assert_eq!(got, 7);
        assert_eq!(r.choosing().shadow.n, 0, "the prompt left the shadow up");
        assert!(
            r.choosing().flight.is_none(),
            "the prompt left a pass in flight"
        );
        assert_eq!(r.choosing().batch_profile(), &[1.0; 3]);
        assert_eq!(r.choosing().batch_tested, [0; 3]);
        r.choosing().round_ran(4, 1);
        assert_eq!(r.choosing().gains(), gains);
        assert_eq!(r.choosing().batch_profile(), &[0.0, 1.0, 1.0]);
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
                    dwell: 9,
                    ..Rule::DEFAULT
                },
                "a dwell of more readings than a cost holds",
            ),
            (
                Rule {
                    margin_close: 0.99,
                    ..Rule::DEFAULT
                },
                "the close margin must be finite and at least 1",
            ),
            (
                Rule {
                    margin_close: f64::NAN,
                    ..Rule::DEFAULT
                },
                "the close margin must be finite and at least 1",
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
                    prior: Some(-1.0),
                    ..Rule::DEFAULT
                },
                "the calibration's prior must be finite and not negative",
            ),
            (
                Rule {
                    prior: Some(f64::INFINITY),
                    ..Rule::DEFAULT
                },
                "the calibration's prior must be finite and not negative",
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
                Scored::<false>::new(&acc, &p)
                    .choosing_by(Mode::Cost, rule, Wall::new())
                    .is_err()
            );
        }
        // A step inside (0, 1] whose recent windows' cells overflow a usize,
        // or whose cells' bytes do, is refused in either mode.
        let acc = Rc::new(Cell::new([0.8; 3]));
        let p = Rc::new(Cell::new([0.8; 3]));
        for calibrate in [f64::MIN_POSITIVE, 1e-18] {
            for mode in [Mode::Cost, Mode::Fixed] {
                let rule = Rule {
                    calibrate,
                    ..Rule::DEFAULT
                };
                assert_eq!(
                    Scored::<false>::new(&acc, &p)
                        .choosing_by(mode, rule, Wall::new())
                        .err(),
                    Some(WidthError::Rule(
                        "the calibration's step keeps more recent windows than can be held"
                    )),
                    "calibrate {calibrate:e}, {mode:?}"
                );
            }
        }
        assert_eq!(Mode::of(None), Ok(Mode::Cost));
        assert_eq!(Mode::of(Some("fixed")), Ok(Mode::Fixed));
        for m in [Mode::Cost, Mode::Fixed] {
            assert_eq!(
                Mode::of(Some(m.word())),
                Ok(m),
                "a word its mode does not read back"
            );
        }
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

    #[test]
    fn batch_costs_keep_different_packings_separate() {
        let mut batch = Batch::default();
        let profiles = [&[0.0; 3][..]; 5];
        for _ in 0..Rule::DEFAULT.min_timings {
            let mut widths = [2, 2, 2, 2, 3];
            batch
                .choose(Mode::Cost, profiles.into_iter(), &mut widths, 0..=3, (8, 8))
                .unwrap();
            batch.observe([3, 3, 3, 3, 4].into_iter(), 30.0).unwrap();
        }
        let mut widths = [3, 3, 2, 2, 1];
        batch
            .choose(Mode::Cost, profiles.into_iter(), &mut widths, 0..=3, (8, 8))
            .unwrap();
        assert_eq!(
            widths,
            [3, 3, 2, 2, 1],
            "the two-pass layout needs its own readings"
        );
        batch.observe([4, 4, 3, 3, 2].into_iter(), 20.0).unwrap();
        assert_eq!(batch.states[0].cost(&[4, 6, 6]), 30.0);
        assert_eq!(batch.states[0].cost(&[8, 8]), 20.0);
        assert_eq!(
            batch.states[0].readings(&[4, 6, 6]),
            Rule::DEFAULT.min_timings
        );
        assert_eq!(batch.states[0].readings(&[8, 8]), 1);
        batch.observe([4, 3, 3, 3, 3].into_iter(), 40.0).unwrap();
        assert_eq!(batch.states[0].cost(&[3, 6, 7]), 40.0);
        assert_eq!(batch.states[0].cost(&[4, 6, 6]), 30.0);
        assert_eq!(batch.states[0].cost(&[8, 8]), 20.0);
        // GLM's pass binds two sequences of its four rows: four plain rows
        // are two passes, as the server cuts them.
        let mut glm = Batch::default();
        let mut widths = [0; 4];
        glm.choose(
            Mode::Cost,
            [&[0.0][..]; 4].into_iter(),
            &mut widths,
            1..=1,
            (4, 2),
        )
        .unwrap();
        glm.observe([1; 4].into_iter(), 10.0).unwrap();
        assert_eq!(glm.states[0].readings(&[2, 2]), 1);
    }

    #[test]
    fn batch_costs_survive_alternating_slot_counts() {
        let mut batch = Batch::default();
        for _ in 0..2 * Rule::DEFAULT.min_timings + 1 {
            for slots in [2, 3] {
                let profiles = vec![&[0.0; 3][..]; slots];
                let mut widths = vec![3; slots];
                batch
                    .choose(Mode::Cost, profiles.into_iter(), &mut widths, 0..=3, (8, 8))
                    .unwrap();
                let rows = slots + widths.iter().sum::<usize>();
                batch
                    .observe(widths.iter().map(|&w| w + 1), rows as f64)
                    .unwrap();
            }
        }
        for slots in [2, 3] {
            let profiles = vec![&[0.0; 3][..]; slots];
            let mut widths = vec![3; slots];
            batch
                .choose(Mode::Cost, profiles.into_iter(), &mut widths, 0..=3, (8, 8))
                .unwrap();
            assert_eq!(
                widths,
                vec![0; slots],
                "learned plain steps for {slots} slots"
            );
        }
    }

    #[test]
    fn batch_observations_require_the_chosen_windows() {
        let mut batch = Batch::default();
        assert!(
            batch
                .observe([1].into_iter(), 1.0)
                .unwrap_err()
                .to_string()
                .contains("before a choice")
        );
        let profiles = [&[0.8; 3][..]; 2];
        let mut widths = [3, 3];
        let error = batch
            .choose(
                Mode::Fixed,
                profiles.into_iter(),
                &mut widths,
                0..=3,
                (3, 3),
            )
            .unwrap_err();
        assert!(error.to_string().contains("at most 3 rows"));
        batch
            .choose(Mode::Cost, profiles.into_iter(), &mut widths, 0..=3, (8, 8))
            .unwrap();
        for rows in [
            vec![4],
            vec![4, 4, 1],
            vec![0, 4],
            vec![5, 4],
            vec![usize::MAX, 1],
        ] {
            assert!(batch.observe(rows.into_iter(), 1.0).is_err());
        }
        for ms in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(batch.observe([4, 4].into_iter(), ms).is_err());
        }
        assert!(batch.states[0].costs.is_empty());
        batch.observe([4, 4].into_iter(), 10.0).unwrap();
        assert_eq!(batch.states[0].cost(&[8]), 10.0);
    }

    #[test]
    fn batch_width_prices_all_slots_and_their_rows() {
        let times = [
            0.0, 10.0, 11.0, 14.0, 30.0, 50.0, 70.0, 90.0, 110.0, 130.0, 150.0, 170.0, 190.0,
            210.0, 230.0, 250.0, 270.0,
        ];
        for (slots, want) in [(1, vec![2]), (2, vec![1, 0]), (4, vec![0; 4])] {
            let mut batch = Batch::default();
            let profiles = vec![&[0.8, 0.8, 0.8][..]; slots];
            let mut widths = vec![3; slots];
            batch
                .choose(
                    Mode::Cost,
                    profiles.iter().copied(),
                    &mut widths,
                    0..=3,
                    (usize::MAX, usize::MAX),
                )
                .unwrap();
            assert_eq!(widths, vec![3; slots]);
            for _ in 0..Rule::DEFAULT.min_timings {
                batch
                    .observe(std::iter::repeat_n(4, slots), times[slots * 4])
                    .unwrap();
            }
            widths.fill(3);
            batch
                .choose(
                    Mode::Cost,
                    profiles.iter().copied(),
                    &mut widths,
                    0..=3,
                    (usize::MAX, usize::MAX),
                )
                .unwrap();
            assert_eq!(widths, vec![0; slots], "sample the opposite endpoint next");
            for (rows, &ms) in times.iter().enumerate().take(slots * 4 + 1).skip(slots) {
                for _ in 0..Rule::DEFAULT.min_timings {
                    batch
                        .observe(
                            (0..slots).map(|i| rows / slots + usize::from(i < rows % slots)),
                            ms,
                        )
                        .unwrap();
                }
            }
            widths.fill(3);
            batch
                .choose(
                    Mode::Cost,
                    profiles.iter().copied(),
                    &mut widths,
                    0..=3,
                    (usize::MAX, usize::MAX),
                )
                .unwrap();
            assert_eq!(widths, want, "batch of {slots}");
            widths.fill(3);
            batch
                .choose(
                    Mode::Fixed,
                    profiles.iter().copied(),
                    &mut widths,
                    0..=3,
                    (usize::MAX, usize::MAX),
                )
                .unwrap();
            assert_eq!(widths, vec![3; slots]);
        }
        // Only the two endpoint row counts read: every width between is
        // priced on the line through them.
        let mut batch = Batch::default();
        let profiles = [&[0.8, 0.8, 0.8][..]; 2];
        let mut widths = [3, 3];
        for (rows, ms) in [(8, 28.0), (2, 10.0)] {
            batch
                .choose(
                    Mode::Cost,
                    profiles.into_iter(),
                    &mut widths,
                    0..=3,
                    (usize::MAX, usize::MAX),
                )
                .unwrap();
            assert_eq!(widths.iter().sum::<usize>() + 2, rows, "endpoint {rows}");
            for _ in 0..Rule::DEFAULT.min_timings {
                batch.observe([rows / 2; 2].into_iter(), ms).unwrap();
            }
            widths.fill(3);
        }
        assert_eq!(
            (batch.states[0].cost(&[3]), batch.states[0].cost(&[5])),
            (13.0, 19.0)
        );
        batch
            .choose(
                Mode::Cost,
                profiles.into_iter(),
                &mut widths,
                0..=3,
                (usize::MAX, usize::MAX),
            )
            .unwrap();
        assert_eq!(widths, [1, 1], "the rate peaks between the endpoints");
        let mut glm = Batch::default();
        let mut widths = [1, 1];
        for _ in 0..=Rule::DEFAULT.min_timings {
            glm.choose(
                Mode::Cost,
                [&[0.0][..], &[0.0][..]].into_iter(),
                &mut widths,
                1..=1,
                (4, 2),
            )
            .unwrap();
            glm.observe([1, 1].into_iter(), 1.0).unwrap();
            glm.observe([2, 2].into_iter(), 100.0).unwrap();
            assert_eq!(widths, [1, 1], "GLM NextN admits only paired rows");
        }
        // PIN(2026-10-23): the prompt-row overflow case went with `choose`'s
        // prompt-row count; a batch round holds decode windows only, one row a
        // slot plus each width, and a width is bounded by its profile.
        let p = &[0.8][..];
        for (profiles, mut widths, allowed, why) in [
            (vec![], vec![1], 0..=3, "same nonempty slots"),
            (vec![], vec![], 0..=3, "same nonempty slots"),
            (
                vec![p],
                vec![1],
                std::ops::RangeInclusive::new(2, 1),
                "set is empty",
            ),
            (vec![p], vec![2], 0..=1, "outside the body's allowed set"),
            (vec![p], vec![2], 0..=3, "exceeds its acceptance profile"),
            (
                vec![&[f64::NAN][..]],
                vec![1],
                0..=3,
                "finite and in [0, 1]",
            ),
            (vec![&[1.1][..]], vec![1], 0..=3, "finite and in [0, 1]"),
        ] {
            let error = Batch::default()
                .choose(
                    Mode::Fixed,
                    profiles.iter().copied(),
                    &mut widths,
                    allowed,
                    (8, 8),
                )
                .unwrap_err();
            assert!(error.to_string().contains(why), "{error}");
        }
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
            closed: 0,
            ..Tally::new(4)
        };
        assert!((t.e() - 7.0 / 3.0).abs() < 1e-12);
        assert_eq!(t.passes(), 8);
        assert_eq!(Tally::new(4).e(), 0.0);
    }
}
