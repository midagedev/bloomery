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

use std::fmt;

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
    /// Every copied store from slot `slot`.
    Restore { slot: usize },
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
        }
    }
}

impl std::error::Error for CheckpointError {}

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

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(l.cut(100, 250), Ok(Cut::Restore { slot: 0 }));
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
        assert_eq!(l.cut(512, 643), Ok(Cut::Restore { slot: 0 }));
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
        assert_eq!(l.cut(200, 300), Ok(Cut::Restore { slot: 1 }));
        assert_eq!(l.take(200), Err(CheckpointError::Pending { at: 200 }));
        assert_eq!(l.cut(200, 200), Ok(Cut::Stay));
        assert_eq!(l.pending(), Some(Cut::Restore { slot: 1 }));
        assert_eq!(l.cut(100, 200), Ok(Cut::Restore { slot: 0 }));
        assert_eq!(l.pending(), Some(Cut::Restore { slot: 0 }));
        l.clear();
        assert_eq!(l.pending(), None);
        assert!(l.positions().is_empty());
        assert_eq!(l.take(0), Ok(Take::Empty));
    }

    /// A take behind a standing point is refused: the model was cut without
    /// the ledger.
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
                Some(Cut::Restore { slot }) => (self.state, self.ring) = self.slots[slot],
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
