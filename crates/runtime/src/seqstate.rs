//! Checkpoints of a model's recurrent state, as pure rules: where a prompt
//! call takes them ([`marks`]), which prefix a cut keeps ([`Ledger::keep`]),
//! what a cut restores ([`Ledger::cut`]), and which slot a new one fills
//! ([`Ledger::take`]) under a host budget ([`capacity`]).
//!
//! A layer with a recurrent store ([`crate::state::copied`]) holds one state
//! written in place by every position: a cut behind the current position
//! finds that history only in a copy taken at the cut's position. Every other
//! store is cut by position, so a checkpoint is the recurrent layers' stores
//! alone, and the positions it serves are exactly the ones it was taken at.
//!
//! The ledger knows positions, never ids: after a cut to `n` the points past
//! `n` belong to the branch the cut abandoned, and they are dropped (a later
//! branch may reach their positions with other ids). The position itself is
//! the caller's: every call takes the position the model stands at.
//!
//! A cut is planned here and carried out by the caller at its next device
//! call ([`Ledger::pending`]): the restore's copy, or the zeroed stores of an
//! empty model. A point the ledger drops or evicts may still be the source
//! of that pending restore only until the caller applies it, which it does
//! before any take ([`Ledger::take`] refuses while a cut waits).
//!
//! Every outcome is named: [`Kept`] says what a cut keeps and why in a stable
//! code and a sentence with the positions ([`Why::code`], `Display`), and an
//! evicted point stays listed so that a miss it caused reads as an eviction.
//!
//! A verify pass takes positions back without a checkpoint. Its rows run from
//! the position the model stands at, and the commit keeps the first `k` of
//! them, row 0 always. Two rules say what the stores hold after it: [`Lanes`]
//! for a recurrent state written row by row into lanes, and [`Rewind`] for a
//! host history of the last tokens. A pass is begun with the position and its
//! rows and committed with `k`; each refusal is a [`PassError`].

use std::fmt;
use std::num::NonZeroUsize;

use crate::stores::PASS_ROWS;

/// The host bytes one model's checkpoints may hold unless its load names
/// another budget: exllamav3's recurrent cache default.
pub const HOST_BUDGET: u64 = 4 << 30;

/// The positions a prompt call from `from` to `to` (`to` > `from`) takes a
/// checkpoint at, ascending: its start when the model holds any position
/// there, every multiple of `every` inside it (none when `every` is 0), and
/// its end. Multiples of `every` fall on the same positions whichever call
/// reaches them, so a prompt sent again takes the same points.
#[must_use]
pub fn marks(from: u32, to: u32, every: u32) -> Vec<u32> {
    let mut v = Vec::new();
    if to <= from {
        return v;
    }
    if from > 0 {
        v.push(from);
    }
    if let Some(q) = from.checked_div(every) {
        let mut p = (q + 1).saturating_mul(every);
        while p < to {
            v.push(p);
            p = p.saturating_add(every);
        }
    }
    v.push(to);
    v
}

/// How many checkpoints of `bytes` each fit `budget`: refused by name when
/// not one does, or when a checkpoint copies nothing.
pub fn capacity(budget: u64, bytes: u64) -> Result<usize, CheckpointError> {
    if bytes == 0 || budget < bytes {
        return Err(CheckpointError::Budget { budget, bytes });
    }
    usize::try_from(budget / bytes).map_err(|_| CheckpointError::Budget { budget, bytes })
}

/// What a cut to at most `asked` positions keeps of the `held` the model
/// stands at, and why.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Kept {
    pub asked: u32,
    pub held: u32,
    /// The positions the cut keeps: at most `asked` and `held`.
    pub at: u32,
    pub why: Why,
}

/// Why a cut keeps what it keeps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Why {
    /// `asked` reaches every held position: nothing is taken back.
    Current,
    /// A cut to 0: the empty model, which needs no copy.
    Empty,
    /// The checkpoint at `at` is copied back. `lost` is the point an
    /// eviction removed between `at` and `asked`, if one did.
    Checkpoint { lost: Option<u32> },
    /// No checkpoint lies at or below `asked`: the cut keeps nothing. `lost`
    /// is the point an eviction removed at or below `asked`, if one did.
    Missed { lost: Option<u32> },
    /// A body's own rule, which states no reason of its own.
    Rule,
}

impl Why {
    /// The reason as one stable word, for records and metrics.
    #[must_use]
    pub fn code(self) -> &'static str {
        match self {
            Why::Current => "current",
            Why::Empty => "empty",
            Why::Checkpoint { .. } => "checkpoint",
            Why::Missed { lost: None } => "no-checkpoint",
            Why::Missed { lost: Some(_) } => "evicted",
            Why::Rule => "rule",
        }
    }
}

impl Kept {
    /// What a body's own rule keeps, with no reason of its own.
    #[must_use]
    pub fn rule(asked: u32, held: u32, at: u32) -> Kept {
        Kept {
            asked,
            held,
            at,
            why: Why::Rule,
        }
    }
}

impl fmt::Display for Kept {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Kept {
            asked, held, at, ..
        } = *self;
        write!(f, "{}: ", self.why.code())?;
        match self.why {
            Why::Current => write!(f, "the {held} held positions stand (asked {asked})"),
            Why::Empty => write!(f, "a cut to the empty model (asked {asked}, held {held})"),
            Why::Checkpoint { lost } => {
                write!(
                    f,
                    "the recurrent state copied back at {at} for {asked} of {held} held"
                )?;
                match lost {
                    Some(e) => write!(f, "; the checkpoint at {e} was evicted by the host budget"),
                    None => Ok(()),
                }
            }
            Why::Missed { lost: None } => write!(
                f,
                "the recurrent state stands at {held} and no checkpoint lies at or below {asked}"
            ),
            Why::Missed { lost: Some(e) } => write!(
                f,
                "the checkpoint at {e} (at or below {asked}) was evicted by the host budget and \
                 none lies below it; the recurrent state stands at {held}"
            ),
            Why::Rule => write!(f, "the body's rule keeps {at} of {asked} (held {held})"),
        }
    }
}

/// What a cut does to the stores the ledger copies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cut {
    /// Nothing: the cut is at the held position.
    Stay,
    /// Every copied store back to zero.
    Empty,
    /// Every copied store from slot `slot`, which holds position `at`.
    Restore { slot: usize, at: u32 },
}

/// Where a checkpoint at the held position goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Take {
    /// Position 0: the empty model needs no copy.
    Empty,
    /// A checkpoint already stands there.
    Held,
    /// Copy the stores into slot `slot`, which the caller makes first when
    /// `new`; `evicted` is the position whose point gave the slot up.
    Copy {
        slot: usize,
        new: bool,
        evicted: Option<u32>,
    },
}

/// A call the ledger refuses, by name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CheckpointError {
    /// Not one checkpoint of `bytes` fits `budget`, or a checkpoint copies
    /// nothing.
    Budget { budget: u64, bytes: u64 },
    /// A cut past the held position.
    Past { n: u32, held: u32 },
    /// A cut to a position nothing restores; what a cut there keeps instead.
    NotKept(Kept),
    /// A take while a cut waits for its copy.
    Pending { at: u32 },
    /// A take at `at` while a point stands past it: the model was cut
    /// without the ledger.
    Behind { at: u32, point: u32 },
    /// A restore of position `at` from slot `slot`, whose last finished copy
    /// holds `holds` (none: never copied, or a copy that failed).
    Unwritten {
        slot: usize,
        at: u32,
        holds: Option<u32>,
    },
}

impl fmt::Display for CheckpointError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CheckpointError::Budget { budget, bytes } => write!(
                f,
                "a checkpoint of {bytes} bytes and a host budget of {budget}: not one fits"
            ),
            CheckpointError::Past { n, held } => {
                write!(f, "a cut to {n} past the {held} held positions")
            }
            CheckpointError::NotKept(k) => write!(
                f,
                "a cut to {}: no checkpoint restores it; a cut keeps {} ({k})",
                k.asked, k.at
            ),
            CheckpointError::Pending { at } => write!(
                f,
                "a checkpoint at {at} while a cut waits for its copy: apply the cut first"
            ),
            CheckpointError::Behind { at, point } => write!(
                f,
                "a checkpoint at {at} while one stands at {point}: the model was cut without \
                 the ledger"
            ),
            CheckpointError::Unwritten { slot, at, holds } => match holds {
                Some(h) => write!(
                    f,
                    "a restore of position {at} from slot {slot}, whose last finished copy holds \
                     position {h}"
                ),
                None => write!(
                    f,
                    "a restore of position {at} from slot {slot}, which no copy has finished"
                ),
            },
        }
    }
}

impl std::error::Error for CheckpointError {}

/// Whether slot `slot`, whose last finished copy holds `holds`, may be read
/// back as position `at`: only when it holds exactly that. The slot's own
/// record, apart from the ledger's point, so a point that names a slot no
/// copy filled is refused by name instead of restoring its zeros.
pub fn restorable(slot: usize, holds: Option<u32>, at: u32) -> Result<(), CheckpointError> {
    if holds == Some(at) {
        Ok(())
    } else {
        Err(CheckpointError::Unwritten { slot, at, holds })
    }
}

/// What the ledger has done since its model's load.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    /// Checkpoints copied.
    pub taken: u64,
    /// Points given up for a new one under the budget.
    pub evicted: u64,
    /// Cuts that copied a checkpoint back.
    pub restored: u64,
    /// Points a cut or a reset dropped as another branch's.
    pub dropped: u64,
}

/// One checkpoint: its position, its slot, and when it was last used.
#[derive(Clone, Copy, Debug)]
struct Point {
    pos: u32,
    slot: usize,
    used: u64,
}

/// The checkpoints of one model's recurrent state: at most `cap` slots,
/// the least recently used point giving its slot up to a new one.
#[derive(Clone, Debug)]
pub struct Ledger {
    cap: usize,
    /// Ascending by position, every one at or below the held position.
    points: Vec<Point>,
    /// Slots made and holding no point.
    free: Vec<usize>,
    made: usize,
    clock: u64,
    /// Positions whose points an eviction removed, ascending, on the
    /// current branch.
    evicted: Vec<u32>,
    pending: Option<Cut>,
    stats: Stats,
}

impl Ledger {
    /// An empty ledger of `cap` slots, at least one.
    pub fn new(cap: usize) -> Result<Ledger, CheckpointError> {
        if cap == 0 {
            return Err(CheckpointError::Budget {
                budget: 0,
                bytes: 0,
            });
        }
        Ok(Ledger {
            cap,
            points: Vec::new(),
            free: Vec::new(),
            made: 0,
            clock: 0,
            evicted: Vec::new(),
            pending: None,
            stats: Stats::default(),
        })
    }

    #[must_use]
    pub fn capacity(&self) -> usize {
        self.cap
    }

    #[must_use]
    pub fn stats(&self) -> Stats {
        self.stats
    }

    /// The checkpoints' positions, ascending.
    #[must_use]
    pub fn positions(&self) -> Vec<u32> {
        self.points.iter().map(|p| p.pos).collect()
    }

    /// The cut waiting for the caller's copy.
    #[must_use]
    pub fn pending(&self) -> Option<Cut> {
        self.pending
    }

    /// The caller carried out the pending cut.
    pub fn applied(&mut self) {
        self.pending = None;
    }

    /// The longest prefix of at most `n` positions a cut keeps of the
    /// `held` the model stands at: all of them when `n` reaches `held`, the
    /// empty model at 0, else the checkpoint at or below `n` nearest to it.
    #[must_use]
    pub fn keep(&self, n: u32, held: u32) -> Kept {
        let asked = n;
        let n = n.min(held);
        let at = |at, why| Kept {
            asked,
            held,
            at,
            why,
        };
        if n == held {
            return at(held, Why::Current);
        }
        if n == 0 {
            return at(0, Why::Empty);
        }
        let point = self.points.iter().rev().find(|p| p.pos <= n).map(|p| p.pos);
        let lost = self
            .evicted
            .iter()
            .rev()
            .find(|&&e| e <= n && point.is_none_or(|p| e > p))
            .copied();
        match point {
            Some(p) => at(p, Why::Checkpoint { lost }),
            None => at(0, Why::Missed { lost }),
        }
    }

    /// Plan a cut to `n` of the `held` the model stands at: nothing at
    /// `held`, the empty model at 0, the copy back of the checkpoint at `n`;
    /// any other `n` is refused by name. Every point past `n` is dropped,
    /// and the cut waits in [`Ledger::pending`] — a later cut replaces it,
    /// a cut at the held position leaves it.
    pub fn cut(&mut self, n: u32, held: u32) -> Result<Cut, CheckpointError> {
        if n > held {
            return Err(CheckpointError::Past { n, held });
        }
        if n == held {
            return Ok(Cut::Stay);
        }
        let cut = if n == 0 {
            Cut::Empty
        } else {
            let i = self
                .points
                .iter()
                .position(|p| p.pos == n)
                .ok_or_else(|| CheckpointError::NotKept(self.keep(n, held)))?;
            self.clock += 1;
            self.points[i].used = self.clock;
            self.stats.restored += 1;
            Cut::Restore {
                slot: self.points[i].slot,
                at: n,
            }
        };
        self.drop_past(n);
        self.pending = Some(cut);
        Ok(cut)
    }

    /// A checkpoint at `at`, the position the model stands at: nothing at 0
    /// or where one stands; else a free slot, a new one while fewer than the
    /// capacity are made, or the least recently used point's. Refused while
    /// a cut waits, and when a point stands past `at`.
    pub fn take(&mut self, at: u32) -> Result<Take, CheckpointError> {
        if self.pending.is_some() {
            return Err(CheckpointError::Pending { at });
        }
        if let Some(p) = self.points.last()
            && p.pos > at
        {
            return Err(CheckpointError::Behind { at, point: p.pos });
        }
        if at == 0 {
            return Ok(Take::Empty);
        }
        self.clock += 1;
        if let Some(p) = self.points.last_mut()
            && p.pos == at
        {
            p.used = self.clock;
            return Ok(Take::Held);
        }
        let (slot, new, evicted) = if let Some(s) = self.free.pop() {
            (s, false, None)
        } else if self.made < self.cap {
            self.made += 1;
            (self.made - 1, true, None)
        } else {
            let (i, _) = self
                .points
                .iter()
                .enumerate()
                .min_by_key(|(_, p)| p.used)
                .ok_or(CheckpointError::Budget {
                    budget: 0,
                    bytes: 0,
                })?;
            let p = self.points.remove(i);
            if let Err(k) = self.evicted.binary_search(&p.pos) {
                self.evicted.insert(k, p.pos);
            }
            self.stats.evicted += 1;
            (p.slot, false, Some(p.pos))
        };
        self.evicted.retain(|&e| e != at);
        self.points.push(Point {
            pos: at,
            slot,
            used: self.clock,
        });
        self.stats.taken += 1;
        Ok(Take::Copy { slot, new, evicted })
    }

    /// Every point dropped and no cut waiting: the model is empty.
    pub fn clear(&mut self) {
        self.drop_past(0);
        self.evicted.clear();
        self.pending = None;
    }

    /// Drop every point past `n` (another branch's), their slots freed.
    fn drop_past(&mut self, n: u32) {
        let keep = self.points.partition_point(|p| p.pos <= n);
        for p in self.points.drain(keep..) {
            self.free.push(p.slot);
            self.stats.dropped += 1;
        }
        self.evicted.retain(|&e| e <= n);
    }
}

/// A pass a rule refuses, by name. `what` names the rule: `lanes` or
/// `history`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PassError {
    /// A pass of `rows` rows over `lanes` lanes: row `lanes` would write the
    /// lane row 0 wrote, and a pass runs at least one row.
    LaneRows { rows: usize, lanes: usize },
    /// The lane a pass reads, or a commit keeps, was never written: no state
    /// stands in it, not even the zero one.
    LaneUnwritten { lane: usize, want: u32 },
    /// The lane stands at another position than the one asked.
    LaneStale { lane: usize, stands: u32, want: u32 },
    /// A history pass of `rows` rows: it keeps 1 to `PASS_ROWS`.
    ReplayRows { rows: usize },
    /// A commit keeping no row: row 0 is always kept.
    NoRowKept { what: &'static str },
    /// A commit keeping more rows than the pass ran.
    KeptPastRows {
        what: &'static str,
        kept: usize,
        rows: usize,
    },
    /// A commit or a rewind with no pass begun.
    NoPass { what: &'static str },
    /// A pass begun while the last one waits for its commit.
    PassOpen {
        what: &'static str,
        from: u32,
        rows: usize,
    },
    /// A rewind to before the pass's first position: the history kept nothing
    /// older than the pass's start.
    RewindBefore { to: u32, from: u32 },
    /// A rewind past the pass's last row.
    RewindPast { to: u32, end: u32 },
}

impl fmt::Display for PassError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            PassError::LaneRows { rows, lanes } => write!(
                f,
                "lanes: a pass of {rows} rows over {lanes} lanes; a pass writes 1 to {lanes} \
                 rows, one lane each"
            ),
            PassError::LaneUnwritten { lane, want } => write!(
                f,
                "lanes: lane {lane} was never written, and position {want} is asked of it: an \
                 unwritten lane is not a zero state"
            ),
            PassError::LaneStale { lane, stands, want } => write!(
                f,
                "lanes: lane {lane} stands at position {stands}, and position {want} is asked \
                 of it"
            ),
            PassError::ReplayRows { rows } => write!(
                f,
                "history: a pass of {rows} rows; a pass keeps 1 to {PASS_ROWS}"
            ),
            PassError::NoRowKept { what } => {
                write!(f, "{what}: a commit of 0 rows; row 0 is always kept")
            }
            PassError::KeptPastRows { what, kept, rows } => {
                write!(f, "{what}: a commit of {kept} rows of a pass of {rows}")
            }
            PassError::NoPass { what } => write!(f, "{what}: a commit with no pass begun"),
            PassError::PassOpen { what, from, rows } => write!(
                f,
                "{what}: a pass begun while the pass of {rows} rows from {from} waits for its \
                 commit"
            ),
            PassError::RewindBefore { to, from } => write!(
                f,
                "history: a rewind to {to}, before the pass's start {from}: nothing older is kept"
            ),
            PassError::RewindPast { to, end } => {
                write!(f, "history: a rewind to {to}, past the pass's end {end}")
            }
        }
    }
}

impl std::error::Error for PassError {}

/// A pass begun and not yet committed: its first position and its rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Open {
    from: u32,
    rows: usize,
}

impl Open {
    /// Begin a pass of `rows` from `from` in `slot`, refused while one is open.
    fn begin(
        slot: &mut Option<Open>,
        what: &'static str,
        from: u32,
        rows: usize,
    ) -> Result<(), PassError> {
        if let Some(o) = *slot {
            return Err(PassError::PassOpen {
                what,
                from: o.from,
                rows: o.rows,
            });
        }
        *slot = Some(Open { from, rows });
        Ok(())
    }

    /// The open pass, checked to keep `kept` of its rows (1 to its rows); it
    /// stays open when refused.
    fn kept(slot: Option<Open>, what: &'static str, kept: usize) -> Result<Open, PassError> {
        let o = slot.ok_or(PassError::NoPass { what })?;
        if kept == 0 {
            return Err(PassError::NoRowKept { what });
        }
        if kept > o.rows {
            return Err(PassError::KeptPastRows {
                what,
                kept,
                rows: o.rows,
            });
        }
        Ok(o)
    }

    /// The position `rows` rows past the pass's start.
    fn at(self, rows: usize) -> u32 {
        self.from + u32::try_from(rows).expect("a pass's rows are at most PASS_ROWS")
    }
}

/// The lane ledger of a recurrent state kept in `L` lanes. A pass from
/// position `p` reads the current lane `c` and writes row `j`'s state, the
/// state standing at `p + j + 1`, into lane `(c + j) mod L`; a commit of `k`
/// rows makes lane `(c + k − 1) mod L` current. No state is copied: the
/// commit moves the lane word.
///
/// Each lane's stamp is the position its state stands at, or none for a
/// lane never written. A pass reads, and a commit keeps, only a lane whose
/// stamp is the position asked: a lane never written is refused by name,
/// never read as a zero state.
#[derive(Clone, Debug)]
pub struct Lanes {
    stamps: Vec<Option<u32>>,
    current: usize,
    open: Option<Open>,
}

impl Lanes {
    const WHAT: &'static str = "lanes";

    /// `lanes` lanes, none written; lane 0 current.
    #[must_use]
    pub fn new(lanes: NonZeroUsize) -> Lanes {
        Lanes {
            stamps: vec![None; lanes.get()],
            current: 0,
            open: None,
        }
    }

    /// The lane count.
    #[must_use]
    pub fn count(&self) -> usize {
        self.stamps.len()
    }

    /// The current lane: the lane word a pass reads.
    #[must_use]
    pub fn current(&self) -> usize {
        self.current
    }

    /// The position lane `lane`'s state stands at; `None` if never written.
    ///
    /// # Panics
    ///
    /// When `lane` is not one of the ledger's.
    #[must_use]
    pub fn stamp(&self, lane: usize) -> Option<u32> {
        *self
            .stamps
            .get(lane)
            .unwrap_or_else(|| panic!("lane {lane} of a ledger of {} lanes", self.count()))
    }

    /// The current lane now holds the state at `at`, written in place and
    /// not by a pass: the zero state at 0 once the caller has zeroed it, or
    /// a checkpoint copied back. Every other lane holds another branch's
    /// state and is unwritten from here. Refused while a pass is open.
    pub fn restored(&mut self, at: u32) -> Result<(), PassError> {
        self.idle()?;
        self.stamps.fill(None);
        self.stamps[self.current] = Some(at);
        Ok(())
    }

    /// A call that runs positions `from .. to` in place on the current lane,
    /// the prompt call's: refused unless the lane stands at `from`.
    pub fn ran(&mut self, from: u32, to: u32) -> Result<(), PassError> {
        self.idle()?;
        self.stands(self.current, from)?;
        self.stamps[self.current] = Some(to);
        Ok(())
    }

    /// Begin a pass of `rows` rows from `from`: refused unless the current
    /// lane stands at `from` and `rows` is 1 to the lane count. The lane
    /// word the pass reads is [`Lanes::current`]; row `j` writes lane
    /// [`Lanes::lane_of`]`(j)` and stands at `from + j + 1`.
    pub fn begin(&mut self, from: u32, rows: usize) -> Result<(), PassError> {
        if rows == 0 || rows > self.count() {
            return Err(PassError::LaneRows {
                rows,
                lanes: self.count(),
            });
        }
        self.idle()?;
        self.stands(self.current, from)?;
        let o = Open { from, rows };
        self.open = Some(o);
        for j in 0..rows {
            let lane = self.lane_of(j);
            self.stamps[lane] = Some(o.at(j + 1));
        }
        Ok(())
    }

    /// The lane row `j` of a pass from the current lane writes.
    #[must_use]
    pub fn lane_of(&self, j: usize) -> usize {
        (self.current + j) % self.count()
    }

    /// Keep the first `kept` rows of the open pass: lane `(c + kept − 1) mod
    /// L` becomes current, and is returned. Refused with no pass open, for 0
    /// rows or more than the pass ran, and when that lane does not stand at
    /// the pass's start plus `kept`.
    pub fn commit(&mut self, kept: usize) -> Result<usize, PassError> {
        let o = Open::kept(self.open, Self::WHAT, kept)?;
        let lane = self.lane_of(kept - 1);
        self.stands(lane, o.at(kept))?;
        self.open = None;
        self.current = lane;
        Ok(lane)
    }

    /// Refused while a pass waits for its commit.
    fn idle(&self) -> Result<(), PassError> {
        match self.open {
            Some(o) => Err(PassError::PassOpen {
                what: Self::WHAT,
                from: o.from,
                rows: o.rows,
            }),
            None => Ok(()),
        }
    }

    /// Refused unless lane `lane` stands at `want`.
    fn stands(&self, lane: usize, want: u32) -> Result<(), PassError> {
        match self.stamps[lane] {
            None => Err(PassError::LaneUnwritten { lane, want }),
            Some(stands) if stands != want => Err(PassError::LaneStale { lane, stands, want }),
            Some(_) => Ok(()),
        }
    }
}

/// The rewind of a host history of the last tokens (the PLE hash's), which
/// a pass moves one push a row. The history before the pass is kept with the
/// pass's rows; after a commit of `k` rows it is that history with rows `0 ..
/// k` pushed, standing at the pass's start plus `k` — what `k` steps would
/// have left. Nothing older than the pass's start is kept.
#[derive(Clone, Debug)]
pub struct Rewind<H> {
    open: Option<Open>,
    before: Option<H>,
    rows: [u32; PASS_ROWS],
}

impl<H> Default for Rewind<H> {
    fn default() -> Rewind<H> {
        Rewind {
            open: None,
            before: None,
            rows: [0; PASS_ROWS],
        }
    }
}

impl<H> Rewind<H> {
    const WHAT: &'static str = "history";

    /// No pass begun.
    #[must_use]
    pub fn new() -> Rewind<H> {
        Rewind::default()
    }

    /// Begin a pass of `rows` from position `from`, where `before` is the
    /// history standing. Refused unless `rows` holds 1 to `PASS_ROWS` ids.
    pub fn begin(&mut self, from: u32, before: H, rows: &[u32]) -> Result<(), PassError> {
        if rows.is_empty() || rows.len() > PASS_ROWS {
            return Err(PassError::ReplayRows { rows: rows.len() });
        }
        Open::begin(&mut self.open, Self::WHAT, from, rows.len())?;
        self.rows[..rows.len()].copy_from_slice(rows);
        self.before = Some(before);
        Ok(())
    }

    /// The history standing at `to`, the pass's start to its end: the one
    /// before the pass with the rows before `to` pushed by `push`, one call
    /// a row. Ends the pass. Refused with no pass begun and for a `to`
    /// outside the pass; the pass stays open then.
    pub fn rewind(&mut self, to: u32, mut push: impl FnMut(&mut H, u32)) -> Result<H, PassError> {
        let o = self.open.ok_or(PassError::NoPass { what: Self::WHAT })?;
        if to < o.from {
            return Err(PassError::RewindBefore { to, from: o.from });
        }
        let end = o.at(o.rows);
        if to > end {
            return Err(PassError::RewindPast { to, end });
        }
        let mut h = self
            .before
            .take()
            .expect("an open pass holds the history before it");
        self.open = None;
        for &id in &self.rows[..(to - o.from) as usize] {
            push(&mut h, id);
        }
        Ok(h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lanes(n: usize) -> Lanes {
        Lanes::new(NonZeroUsize::new(n).unwrap())
    }

    /// Four lanes, lane 2 current at position 10: lanes 0 and 1 hold the
    /// rejected rows of the pass that got there (8 and 9).
    fn lane_2_at_10() -> Lanes {
        let mut l = lanes(4);
        l.restored(0).unwrap();
        l.ran(0, 7).unwrap();
        l.begin(7, 3).unwrap();
        assert_eq!(l.commit(3), Ok(2));
        l
    }

    /// The lane table from lane 2 at position 10, every T of 1 to 4 and
    /// every k of 1 to T: the stamps the pass leaves (row j at 11 + j in lane
    /// (2 + j) mod 4, the other lanes as they were), the lane the commit
    /// keeps, and the next pass reading it at 10 + k.
    #[test]
    fn lane_table() {
        let n = None;
        // (T, k, stamps after the pass, lane kept)
        let table: [(usize, usize, [Option<u32>; 4], usize); 10] = [
            (1, 1, [Some(8), Some(9), Some(11), n], 2),
            (2, 1, [Some(8), Some(9), Some(11), Some(12)], 2),
            (2, 2, [Some(8), Some(9), Some(11), Some(12)], 3),
            (3, 1, [Some(13), Some(9), Some(11), Some(12)], 2),
            (3, 2, [Some(13), Some(9), Some(11), Some(12)], 3),
            (3, 3, [Some(13), Some(9), Some(11), Some(12)], 0),
            (4, 1, [Some(13), Some(14), Some(11), Some(12)], 2),
            (4, 2, [Some(13), Some(14), Some(11), Some(12)], 3),
            (4, 3, [Some(13), Some(14), Some(11), Some(12)], 0),
            (4, 4, [Some(13), Some(14), Some(11), Some(12)], 1),
        ];
        for (t, k, stamps, lane) in table {
            let mut l = lane_2_at_10();
            assert_eq!(l.current(), 2);
            l.begin(10, t).unwrap();
            let got: Vec<Option<u32>> = (0..4).map(|i| l.stamp(i)).collect();
            assert_eq!(got, stamps, "T {t}");
            assert_eq!(l.commit(k), Ok(lane), "T {t} k {k}");
            assert_eq!(l.current(), lane);
            assert_eq!(l.stamp(lane), Some(10 + k as u32));
            l.begin(10 + k as u32, 1).unwrap();
            assert_eq!(l.commit(1), Ok(lane), "a one-row pass stays on its lane");
        }
    }

    /// The lane count is the caller's: two lanes carry a two-row pass and
    /// refuse a third row; one lane is a state written in place.
    #[test]
    fn lane_count_is_a_parameter() {
        let mut l = lanes(2);
        l.restored(0).unwrap();
        l.begin(0, 2).unwrap();
        assert_eq!(l.commit(2), Ok(1));
        l.begin(2, 2).unwrap();
        assert_eq!(l.commit(1), Ok(1));
        assert_eq!(
            l.begin(3, 3),
            Err(PassError::LaneRows { rows: 3, lanes: 2 })
        );
        let mut one = lanes(1);
        one.restored(0).unwrap();
        for p in 0..5 {
            one.begin(p, 1).unwrap();
            assert_eq!(one.commit(1), Ok(0));
        }
        assert_eq!(one.stamp(0), Some(5));
    }

    /// Every lane refusal by name: a lane never written (a fresh ledger, and
    /// the lanes a restore drops), a lane at another position, a pass of no
    /// rows or of more than the lanes, a commit of none, of too many, or
    /// with no pass, and a pass, a restore or a call begun over an open pass.
    /// A refused commit leaves the pass open.
    #[test]
    fn lane_refusals() {
        let mut l = lanes(4);
        assert_eq!(
            l.begin(0, 1),
            Err(PassError::LaneUnwritten { lane: 0, want: 0 })
        );
        assert_eq!(
            l.ran(0, 5),
            Err(PassError::LaneUnwritten { lane: 0, want: 0 })
        );
        l.restored(0).unwrap();
        assert_eq!(
            l.begin(5, 1),
            Err(PassError::LaneStale {
                lane: 0,
                stands: 0,
                want: 5
            })
        );
        assert_eq!(
            l.begin(0, 0),
            Err(PassError::LaneRows { rows: 0, lanes: 4 })
        );
        assert_eq!(
            l.begin(0, 5),
            Err(PassError::LaneRows { rows: 5, lanes: 4 })
        );
        assert_eq!(l.commit(1), Err(PassError::NoPass { what: "lanes" }));
        l.begin(0, 3).unwrap();
        let open = PassError::PassOpen {
            what: "lanes",
            from: 0,
            rows: 3,
        };
        assert_eq!(l.begin(0, 1), Err(open.clone()));
        assert_eq!(l.restored(0), Err(open.clone()));
        assert_eq!(l.ran(0, 1), Err(open));
        assert_eq!(l.commit(0), Err(PassError::NoRowKept { what: "lanes" }));
        assert_eq!(
            l.commit(4),
            Err(PassError::KeptPastRows {
                what: "lanes",
                kept: 4,
                rows: 3
            })
        );
        assert_eq!(l.commit(2), Ok(1));
        // A restore drops the other branch's lanes: lane 2 held row 2.
        assert_eq!(l.stamp(2), Some(3));
        l.restored(2).unwrap();
        assert_eq!((l.stamp(1), l.stamp(2)), (Some(2), None));
        l.begin(2, 2).unwrap();
        assert_eq!(l.commit(2), Ok(2));
        assert_eq!(
            PassError::LaneUnwritten { lane: 3, want: 7 }.to_string(),
            "lanes: lane 3 was never written, and position 7 is asked of it: an unwritten lane \
             is not a zero state"
        );
    }

    /// The PLE hash's history as `engram::hash::History` keeps it: the last
    /// `n − 1` tokens, the sequence's start read as `eos`, and the next
    /// position; one push a token.
    #[derive(Clone, Debug, PartialEq)]
    struct Tail {
        prev: Vec<u32>,
        next: u64,
    }

    const EOS: u32 = 99;

    impl Tail {
        fn new(n_gram: usize) -> Tail {
            Tail {
                prev: vec![EOS; n_gram - 1],
                next: 0,
            }
        }

        fn push(&mut self, t: u32) {
            self.prev.rotate_left(1);
            let last = self.prev.len() - 1;
            self.prev[last] = t;
            self.next += 1;
        }

        fn of(n_gram: usize, ids: &[u32]) -> Tail {
            let mut h = Tail::new(n_gram);
            for &t in ids {
                h.push(t);
            }
            h
        }
    }

    /// After a pass of T rows from p and a commit of k, the history is the
    /// one k steps would have left — the sequence's tokens through row k − 1,
    /// standing at p + k — for T of 1 to a pass's rows, every k of 0 to T,
    /// from near the start (the eos fill) and from past it.
    #[test]
    fn rewind_is_the_steps() {
        let seq: Vec<u32> = (0..40).map(|i| (i * 7 + 3) % 23).collect();
        for n_gram in [2, 3, 4] {
            for p in [0usize, 1, 9] {
                for t in 1..=PASS_ROWS {
                    let rows = &seq[p..p + t];
                    for k in 0..=t {
                        let before = Tail::of(n_gram, &seq[..p]);
                        let mut r = Rewind::new();
                        r.begin(p as u32, before.clone(), rows).unwrap();
                        let got = r.rewind((p + k) as u32, Tail::push).unwrap();
                        let want = Tail::of(n_gram, &seq[..p + k]);
                        assert_eq!(got, want, "n {n_gram} p {p} T {t} k {k}");
                        assert_eq!(got.next, (p + k) as u64);
                    }
                }
            }
        }
    }

    /// Every rewind refusal by name: a pass of no rows or of more than a
    /// pass holds, a rewind with no pass, to before the pass's start (nothing
    /// older is kept) or past its end, and a pass begun over an open one. A
    /// refused rewind leaves the pass open.
    #[test]
    fn rewind_refusals() {
        let mut r: Rewind<Tail> = Rewind::new();
        let h = Tail::of(3, &[1, 2, 3, 4, 5]);
        assert_eq!(
            r.rewind(5, Tail::push),
            Err(PassError::NoPass { what: "history" })
        );
        assert_eq!(
            r.begin(5, h.clone(), &[]),
            Err(PassError::ReplayRows { rows: 0 })
        );
        let wide = [0u32; PASS_ROWS + 1];
        assert_eq!(
            r.begin(5, h.clone(), &wide),
            Err(PassError::ReplayRows {
                rows: PASS_ROWS + 1
            })
        );
        r.begin(5, h.clone(), &[6, 7, 8]).unwrap();
        assert_eq!(
            r.begin(5, h.clone(), &[6]),
            Err(PassError::PassOpen {
                what: "history",
                from: 5,
                rows: 3
            })
        );
        assert_eq!(
            r.rewind(4, Tail::push),
            Err(PassError::RewindBefore { to: 4, from: 5 })
        );
        assert_eq!(
            r.rewind(9, Tail::push),
            Err(PassError::RewindPast { to: 9, end: 8 })
        );
        assert_eq!(r.rewind(5, Tail::push), Ok(h));
    }

    /// The call's start (when the model holds any position), the multiples
    /// of the spacing strictly inside, and the end.
    #[test]
    fn marks_start_multiples_end() {
        assert_eq!(marks(0, 1500, 512), [512, 1024, 1500]);
        assert_eq!(marks(0, 512, 512), [512]);
        assert_eq!(marks(0, 64, 512), [64]);
        assert_eq!(marks(700, 1100, 512), [700, 1024, 1100]);
        assert_eq!(marks(1024, 1030, 512), [1024, 1030]);
        assert_eq!(marks(100, 900, 0), [100, 900]);
        assert!(marks(9, 9, 512).is_empty());
        assert!(marks(9, 3, 512).is_empty());
    }

    #[test]
    fn capacity_by_budget() {
        assert_eq!(capacity(HOST_BUDGET, 179_372_032), Ok(23));
        assert_eq!(capacity(HOST_BUDGET, 73_728_000), Ok(58));
        assert!(matches!(
            capacity(100, 101),
            Err(CheckpointError::Budget { .. })
        ));
        assert!(matches!(
            capacity(100, 0),
            Err(CheckpointError::Budget { .. })
        ));
        assert!(Ledger::new(0).is_err());
    }

    fn copy(t: Take) -> usize {
        match t {
            Take::Copy { slot, .. } => slot,
            other => panic!("a copy expected, got {other:?}"),
        }
    }

    /// The keep rule: all at or past the held position, 0 empty, else the
    /// nearest checkpoint at or below, else nothing — each with its code.
    #[test]
    fn keep_is_the_nearest_checkpoint_below() {
        let mut l = Ledger::new(4).unwrap();
        copy(l.take(512).unwrap());
        copy(l.take(640).unwrap());
        let k = |n| l.keep(n, 700);
        assert_eq!((k(700).at, k(700).why), (700, Why::Current));
        assert_eq!((k(900).at, k(900).why), (700, Why::Current));
        assert_eq!((k(0).at, k(0).why), (0, Why::Empty));
        assert_eq!(k(600).at, 512);
        assert_eq!(k(640).at, 640);
        assert_eq!(k(699).at, 640);
        assert_eq!(k(600).why, Why::Checkpoint { lost: None });
        assert_eq!((k(511).at, k(511).why), (0, Why::Missed { lost: None }));
        assert_eq!(k(511).why.code(), "no-checkpoint");
        assert_eq!(
            k(600).to_string(),
            "checkpoint: the recurrent state copied back at 512 for 600 of 700 held"
        );
        assert_eq!(
            k(511).to_string(),
            "no-checkpoint: the recurrent state stands at 700 and no checkpoint lies at or below \
             511"
        );
    }

    /// A take where one stands copies nothing; at 0 nothing; slots are made
    /// up to the capacity, then the least recently used point gives its up
    /// and stays listed, so the miss it causes reads as an eviction.
    #[test]
    fn take_fills_then_evicts_the_least_recent() {
        let mut l = Ledger::new(2).unwrap();
        assert_eq!(l.take(0), Ok(Take::Empty));
        assert_eq!(
            l.take(100),
            Ok(Take::Copy {
                slot: 0,
                new: true,
                evicted: None
            })
        );
        assert_eq!(l.take(100), Ok(Take::Held));
        assert_eq!(
            l.take(200),
            Ok(Take::Copy {
                slot: 1,
                new: true,
                evicted: None
            })
        );
        // 100's last use is before 200's: 100 goes.
        assert_eq!(
            l.take(300),
            Ok(Take::Copy {
                slot: 0,
                new: false,
                evicted: Some(100)
            })
        );
        assert_eq!(l.positions(), [200, 300]);
        let k = l.keep(150, 300);
        assert_eq!((k.at, k.why), (0, Why::Missed { lost: Some(100) }));
        assert_eq!(k.why.code(), "evicted");
        assert_eq!(
            l.keep(250, 300).why,
            Why::Checkpoint { lost: None },
            "the eviction at 100 lies below the point that serves 250"
        );
        assert_eq!(l.stats().evicted, 1);
        assert_eq!(l.stats().taken, 3);
    }

    /// A cut frees the slots of the points it drops, and a take fills a
    /// freed slot before it makes a new one.
    #[test]
    fn dropped_slots_are_reused() {
        let mut l = Ledger::new(3).unwrap();
        copy(l.take(100).unwrap());
        copy(l.take(200).unwrap());
        assert_eq!(l.cut(100, 250), Ok(Cut::Restore { slot: 0, at: 100 }));
        l.applied();
        assert_eq!(
            l.take(180),
            Ok(Take::Copy {
                slot: 1,
                new: false,
                evicted: None
            })
        );
        assert_eq!(
            l.take(190),
            Ok(Take::Copy {
                slot: 2,
                new: true,
                evicted: None
            })
        );
    }

    /// A cut drops every point past it — another branch's — and a later
    /// keep never returns one: the stale point below `n` of a longer branch.
    #[test]
    fn cut_drops_the_abandoned_branch() {
        let mut l = Ledger::new(8).unwrap();
        copy(l.take(512).unwrap());
        copy(l.take(640).unwrap());
        assert_eq!(l.cut(512, 643), Ok(Cut::Restore { slot: 0, at: 512 }));
        l.applied();
        assert_eq!(l.positions(), [512]);
        assert_eq!(l.take(512), Ok(Take::Held));
        copy(l.take(712).unwrap());
        let k = l.keep(700, 712);
        assert_eq!(k.at, 512, "640 was the abandoned branch's");
        assert_eq!(l.stats().dropped, 1);
        assert_eq!(l.stats().restored, 1);
    }

    /// Cuts refused by name: past the held position, and to a position no
    /// checkpoint restores (naming what a cut keeps instead); a cut at the
    /// held position is nothing; a cut to 0 is the empty model and drops
    /// every point.
    #[test]
    fn cuts_named() {
        let mut l = Ledger::new(4).unwrap();
        copy(l.take(512).unwrap());
        assert_eq!(
            l.cut(701, 700),
            Err(CheckpointError::Past { n: 701, held: 700 })
        );
        let Err(CheckpointError::NotKept(k)) = l.cut(600, 700) else {
            panic!("a cut between checkpoints was not refused");
        };
        assert_eq!(k.at, 512);
        assert!(
            CheckpointError::NotKept(k)
                .to_string()
                .starts_with("a cut to 600: no checkpoint restores it; a cut keeps 512")
        );
        assert_eq!(l.positions(), [512], "a refused cut drops nothing");
        assert_eq!(l.pending(), None);
        assert_eq!(l.cut(700, 700), Ok(Cut::Stay));
        assert_eq!(l.cut(0, 700), Ok(Cut::Empty));
        assert!(l.positions().is_empty());
    }

    /// The pending cut: a take waits for it, a cut at the held position
    /// keeps it, a deeper cut replaces it, a reset drops it.
    #[test]
    fn pending_cut_rules() {
        let mut l = Ledger::new(4).unwrap();
        copy(l.take(100).unwrap());
        copy(l.take(200).unwrap());
        assert_eq!(l.cut(200, 300), Ok(Cut::Restore { slot: 1, at: 200 }));
        assert_eq!(l.take(200), Err(CheckpointError::Pending { at: 200 }));
        assert_eq!(l.cut(200, 200), Ok(Cut::Stay));
        assert_eq!(l.pending(), Some(Cut::Restore { slot: 1, at: 200 }));
        assert_eq!(l.cut(100, 200), Ok(Cut::Restore { slot: 0, at: 100 }));
        assert_eq!(l.pending(), Some(Cut::Restore { slot: 0, at: 100 }));
        l.clear();
        assert_eq!(l.pending(), None);
        assert!(l.positions().is_empty());
        assert_eq!(l.take(0), Ok(Take::Empty));
    }

    /// A take behind a standing point is refused: the model was cut without
    /// the ledger.
    /// A slot is read back only as the position its last finished copy
    /// holds: never copied, another position, or that position.
    #[test]
    fn restore_reads_only_the_copied_position() {
        assert_eq!(
            restorable(0, None, 512),
            Err(CheckpointError::Unwritten {
                slot: 0,
                at: 512,
                holds: None
            })
        );
        assert!(
            restorable(1, Some(100), 512)
                .unwrap_err()
                .to_string()
                .contains("holds position 100")
        );
        assert_eq!(restorable(1, Some(512), 512), Ok(()));
    }

    #[test]
    fn take_behind_refused() {
        let mut l = Ledger::new(4).unwrap();
        copy(l.take(300).unwrap());
        assert_eq!(
            l.take(200),
            Err(CheckpointError::Behind {
                at: 200,
                point: 300
            })
        );
    }

    // ------------------------------------------------ a recurrent model

    /// A host model with a recurrent state and a conv ring of eleven slots
    /// by position, driven through the ledger as a body drives its stores:
    /// the prefix it keeps must continue exactly as a fresh run of the same
    /// ids does.
    struct Rec {
        state: u64,
        ring: [u32; 11],
        fed: u32,
        slots: Vec<(u64, [u32; 11])>,
        ledger: Ledger,
    }

    impl Rec {
        fn new(cap: usize) -> Rec {
            Rec {
                state: 0,
                ring: [0; 11],
                fed: 0,
                slots: Vec::new(),
                ledger: Ledger::new(cap).unwrap(),
            }
        }

        fn apply(&mut self) {
            match self.ledger.pending() {
                Some(Cut::Restore { slot, .. }) => (self.state, self.ring) = self.slots[slot],
                Some(Cut::Empty) => (self.state, self.ring) = (0, [0; 11]),
                Some(Cut::Stay) | None => {}
            }
            self.ledger.applied();
        }

        /// One position: the state folds the id and the three before it
        /// the ring holds; the output is a function of the state.
        fn step(&mut self, id: u32) -> u64 {
            self.apply();
            let p = self.fed as usize;
            let conv = (1..=3).fold(u64::from(id), |a, d| {
                let back = if p >= d { self.ring[(p - d) % 11] } else { 0 };
                a.wrapping_mul(31).wrapping_add(u64::from(back))
            });
            self.ring[p % 11] = id;
            self.state = self
                .state
                .rotate_left(7)
                .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                ^ conv;
            self.fed += 1;
            self.state ^ (self.state >> 29)
        }

        fn take(&mut self) {
            self.apply();
            if let Take::Copy { slot, new, .. } = self.ledger.take(self.fed).unwrap() {
                if new {
                    assert_eq!(slot, self.slots.len());
                    self.slots.push((0, [0; 11]));
                }
                self.slots[slot] = (self.state, self.ring);
            }
        }

        /// A prompt call with the checkpoints of `marks`.
        fn prompt(&mut self, ids: &[u32], every: u32) -> u64 {
            let from = self.fed;
            let mut out = 0;
            let mut at = from;
            for m in marks(from, from + ids.len() as u32, every) {
                for &id in &ids[(at - from) as usize..(m - from) as usize] {
                    out = self.step(id);
                }
                at = m;
                self.take();
            }
            out
        }

        fn cut(&mut self, n: u32) {
            self.ledger.cut(n, self.fed).unwrap();
            self.fed = n;
        }

        fn reset(&mut self) {
            (self.state, self.ring, self.fed) = (0, [0; 11], 0);
            self.ledger.clear();
        }
    }

    fn fresh(ids: &[u32]) -> u64 {
        let mut r = Rec::new(1);
        let mut out = 0;
        for &id in ids {
            out = r.step(id);
        }
        out
    }

    fn ids(seed: u32, n: usize) -> Vec<u32> {
        (0..n as u32)
            .map(|i| i.wrapping_mul(2_654_435_761).wrapping_add(seed) % 1000)
            .collect()
    }

    /// Chat turns whose next request diverges right after the last prompt
    /// call (the dropped reasoning): each keeps the prompt's end, and the
    /// continuation equals a fresh run of the whole conversation.
    #[test]
    fn rec_turns_keep_the_prompt_end() {
        let mut r = Rec::new(23);
        let mut conv: Vec<u32> = ids(1, 200);
        r.prompt(&conv[..199], 512);
        for t in 0..6u32 {
            // The reply as the model fed it, then the request that renders
            // it otherwise.
            let reply = ids(100 + t, 30);
            for &id in &reply {
                r.step(id);
            }
            let keep = conv.len() as u32 - 1;
            let mut next = conv.clone();
            next.extend(ids(200 + t, 25));
            next.extend(ids(300 + t, 60));
            let k = r.ledger.keep(keep, r.fed);
            assert_eq!(
                (k.at, k.why),
                (keep, Why::Checkpoint { lost: None }),
                "turn {t}"
            );
            r.cut(k.at);
            let out = r.prompt(&next[k.at as usize..next.len() - 1], 512);
            assert_eq!(out, fresh(&next[..next.len() - 1]), "turn {t}");
            conv = next;
        }
    }

    /// The stale branch: a restore below a point the cut abandoned, then a
    /// longer branch — the next keep must not reach the abandoned point.
    #[test]
    fn rec_branch_restores_bit_for_bit() {
        let a = ids(7, 640);
        let d = ids(8, 200);
        let e = ids(9, 64);
        let mut r = Rec::new(23);
        r.prompt(&a, 512);
        r.step(1);
        r.step(2);
        assert_eq!(r.ledger.positions(), [512, 640]);
        let k = r.ledger.keep(600, r.fed);
        assert_eq!(k.at, 512);
        r.cut(512);
        r.prompt(&d, 512);
        assert_eq!(r.ledger.positions(), [512, 712]);
        r.step(3);
        r.step(4);
        // Back to the point taken on the restored branch.
        r.cut(712);
        let got = r.prompt(&e, 512);
        let want: Vec<u32> = [&a[..512], &d, &e].concat();
        assert_eq!(got, fresh(&want));
        // And to the branch's start, below the abandoned 640.
        let k = r.ledger.keep(700, r.fed);
        assert_eq!(k.at, 512);
        r.cut(k.at);
        let got = r.prompt(&e, 512);
        assert_eq!(got, fresh(&[&a[..512], &e[..]].concat()));
        // A reset and a cut to 0 lose every point.
        r.reset();
        assert!(r.ledger.positions().is_empty());
        assert_eq!(r.prompt(&e, 512), fresh(&e));
        r.cut(0);
        assert_eq!(r.prompt(&e, 512), fresh(&e));
    }
}
