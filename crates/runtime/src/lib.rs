//! The generation loop and the traits it drives — host only, no card.
//!
//! A [`Target`] runs a model: a prompt call from where it stands, a one-token
//! step, and — where the model can — a pass of several rows ([`Verify`]) that
//! verifies a draft's proposal. How a generation moves on is an [`Advance`]:
//! [`Plain`] steps once a pass, [`Speculative`] runs a [`Draft`]'s proposal
//! through a verify and keeps the rows the target agrees with, and
//! [`width::Choosing`] verifies each proposal at the width that pays.
//! [`generate`] is the one loop over both, until the [`Stop`] rule ends it.
//!
//! A pass takes each row's id by a [`Pick`]: the target's argmax
//! ([`Argmax`], the greedy pass), a draw from the row's logits ([`Sample`]
//! over a [`RowLogits`] target), or the next id of a fixed stream
//! ([`Forced`], driven through [`Picked`]), one call a taken id. Either way a
//! draft changes which passes run and never a token.
//!
//! Beneath a target's call: [`sched`], the order in which a schedule runs a
//! layer program's parts over its units ([`sched::walk`]), [`prompt`], how a
//! prompt call is cut into batches and walked in groups, [`state`], the
//! stores a layer keeps and how later positions read them, [`stores`], their
//! sizes, [`seqstate`], the
//! checkpoints of the recurrent ones and which cuts they serve, and
//! [`layer`], the sub-layer programs a layer's description names.
//! [`words`] is smaller and sits below all of them: the word count of a
//! flat weight stream, the one count the loader's staging and the packings
//! that pin it must agree on.

pub mod combine;
pub mod costs;
pub mod dspark;
pub mod hc_gated;
pub mod layer;
mod lookup;
pub mod msa;
pub mod prompt;
pub mod qsa;
pub mod savemap;
pub mod sched;
pub mod selector;
pub mod seqstate;
mod speculative;
pub mod splitwalk;
pub mod state;
mod stop;
pub mod stores;
pub mod swaprule;
pub mod warmprof;
pub mod width;
pub mod words;
pub mod xsplit;

pub use lookup::Lookup;
pub use speculative::{
    Argmax, Draft, Exhausted, Forced, NotAnId, NotBuilt, Pick, Picked, Picking, Program, RowLogits,
    Sample, SampleBest, Speculative, TapNeed, TapSink, Tapped, TappedPrompt, Width, Widths, Window,
    accepted_rows, ids_from, program,
};
pub use stop::{NoTokens, Stop, StopReason};
pub use width::{Choosing, Chosen, Rule, Tally, WidthError};

use std::time::{Duration, Instant};

/// A round that cannot be cut into passes: its pass bound, or a window
/// that bound cannot hold.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RoundError {
    /// A pass of no sequence, or of more sequences than rows.
    Limits { max_rows: usize, max_slots: usize },
    /// A window of no row or of more rows than a pass holds.
    Window {
        slot: usize,
        rows: usize,
        max_rows: usize,
    },
}

impl std::fmt::Display for RoundError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RoundError::Limits {
                max_rows,
                max_slots,
            } => write!(
                f,
                "a round's passes of {max_rows} rows and {max_slots} slots: both are positive, \
                 with slots <= rows"
            ),
            RoundError::Window {
                slot,
                rows,
                max_rows,
            } => write!(
                f,
                "slot {slot}'s {rows} rows in a round of passes of at most {max_rows} rows"
            ),
        }
    }
}

impl std::error::Error for RoundError {}

/// Cut `(slot, rows)` entries into passes of at most `max_rows` rows and
/// `max_slots` entries (the sequences a pass binds), without splitting a
/// window, and put the pass holding slot 0 last. Reuses `cuts`; bad limits
/// and an empty or oversized window are refused before any passes are
/// written.
pub fn round_passes(
    rows: impl Iterator<Item = (usize, usize)> + Clone,
    (max_rows, max_slots): (usize, usize),
    cuts: &mut Vec<std::ops::Range<usize>>,
) -> Result<(), RoundError> {
    if max_slots == 0 || max_slots > max_rows {
        return Err(RoundError::Limits {
            max_rows,
            max_slots,
        });
    }
    for (slot, n) in rows.clone() {
        if n == 0 || n > max_rows {
            return Err(RoundError::Window {
                slot,
                rows: n,
                max_rows,
            });
        }
    }
    cuts.clear();
    let (mut packed, mut zero) = (0, None);
    for (i, (slot, n)) in rows.enumerate() {
        match cuts.last_mut() {
            Some(cut) if packed <= max_rows - n && cut.len() < max_slots => {
                cut.end = i + 1;
                packed += n;
            }
            _ => {
                cuts.push(i..i + 1);
                packed = n;
            }
        }
        if slot == 0 {
            zero = Some(cuts.len() - 1);
        }
    }
    if let Some(z) = zero {
        let holds_zero = cuts.remove(z);
        cuts.push(holds_zero);
    }
    Ok(())
}

/// What a call of the target reads back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Want {
    /// The argmax alone: greedy never reads the row back.
    Argmax,
    /// The logits row of the last position as well.
    Logits,
}

/// What a call of the target read back, as its [`Want`] asked.
#[derive(Debug)]
pub enum Out<'a> {
    Argmax(u32),
    /// The argmax and the logits row it is the argmax of.
    Logits {
        argmax: u32,
        row: &'a [f32],
    },
}

impl Out<'_> {
    /// The target's greedy next token.
    #[must_use]
    pub fn argmax(&self) -> u32 {
        match *self {
            Out::Argmax(t) | Out::Logits { argmax: t, .. } => t,
        }
    }
}

/// A model standing at a position: the next id it runs lands in cache row
/// [`Target::pos`].
///
/// A call that fails returns its error and nothing here calls again on its
/// own: a target that took a failed call's positions back stands where its
/// error says, and one whose device raised a fault refuses every call until
/// [`Target::reset`].
pub trait Target {
    /// What a failed call says.
    type Error: std::error::Error + 'static;

    /// The cache row the next id lands in: the target's own position, held
    /// nowhere else.
    fn pos(&self) -> u32;

    /// The positions the target's caches hold; [`Target::pos`] never passes it.
    fn ctx(&self) -> u32;

    /// Run `ids` from [`Target::pos`] on by the target's prompt schedule and
    /// read back after the last; the target stands `ids.len()` positions on.
    fn prompt(&mut self, ids: &[u32], want: Want) -> Result<Out<'_>, Self::Error>;

    /// [`Target::prompt`] that leaves each of `marks` keepable: a later
    /// [`Target::cut`] to one keeps it whole. The marks are positions,
    /// ascending, each past [`Target::pos`] and at most the call's end. The
    /// default runs a [`Target::prompt`] a part, cut at each mark before the
    /// end: a target whose calls' ends are its points keeps a mark by ending a
    /// call there. A target that keeps a position inside a call answers here
    /// with one call. A mark out of order or outside the call is the caller's
    /// breach: the default panics by name.
    fn prompt_marked(
        &mut self,
        ids: &[u32],
        marks: &[u32],
        want: Want,
    ) -> Result<Out<'_>, Self::Error> {
        let from = u64::from(self.pos());
        let to = from + ids.len() as u64;
        let mut at = 0;
        let mut prev = from;
        for &u in marks {
            let u = u64::from(u);
            assert!(
                prev < u && u <= to,
                "a prompt of positions {from}..{to} marked at {marks:?}: marks go ascending, \
                 past the call's start and at most its end"
            );
            prev = u;
            let i = usize::try_from(u - from).expect("a mark inside the call is an index of it");
            if i < ids.len() {
                self.prompt(&ids[at..i], Want::Argmax)?;
                at = i;
            }
        }
        self.prompt(&ids[at..], want)
    }

    /// Run `id` at [`Target::pos`], one position, and read back after it.
    fn step(&mut self, id: u32, want: Want) -> Result<Out<'_>, Self::Error>;

    /// The longest prefix of at most `n` positions [`Target::cut`] keeps.
    fn keepable(&self, n: u32) -> u32;

    /// [`Target::keepable`] with its reason. A target whose rule states none
    /// says [`seqstate::Why::Rule`].
    fn kept(&self, n: u32) -> seqstate::Kept {
        seqstate::Kept::rule(n, self.pos(), self.keepable(n))
    }

    /// Take back the positions from `n` on; refused by name unless
    /// [`Target::keepable`] grants `n` whole.
    fn cut(&mut self, n: u32) -> Result<(), Self::Error>;

    /// Empty caches at position 0.
    fn reset(&mut self) -> Result<(), Self::Error>;
}

/// A target that runs several consecutive positions as one pass: what a
/// draft's proposal is verified by.
pub trait Verify: Target {
    /// The most rows one pass runs, 2 or more.
    const MAX_ROWS: usize;

    /// Run `rows[r]` at position `pos + r`, for the target's position `pos`,
    /// as one pass of `M` rows and return each row's argmax; the target stands
    /// `M` positions on until [`Verify::commit`], which must come next. `M` is
    /// 2 to [`Verify::MAX_ROWS`], or the call does not compile.
    fn verify<const M: usize>(&mut self, rows: [u32; M]) -> Result<[u32; M], Self::Error>;

    /// Keep the first `accepted` rows of the last verify, 1 to its `M`; the
    /// positions of the rest are taken back.
    fn commit(&mut self, accepted: usize) -> Result<(), Self::Error>;
}

/// What one pass of an [`Advance`] kept.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Committed {
    /// The target's position before the pass: kept token `r` is the argmax
    /// after position `pos + r`.
    pub pos: u32,
    /// The rows the pass kept on the target, 1 to `rows`.
    pub kept: usize,
    /// The rows the pass ran: 1 for a step, the verify's `M` for a proposal.
    pub rows: usize,
    /// Whether a proposal was verified; a pass with none is one step.
    pub proposed: bool,
}

/// How a generation moves on from the prompt.
pub trait Advance<T: Target> {
    /// The most rows one pass runs: the loop stops at the context when that
    /// many positions do not fit the caches. A one-step advance keeps 1; one
    /// that runs more and does not say so ends its last pass in the target's
    /// refusal at the caches' end.
    const ROWS: usize = 1;

    /// Feed the prompt `ids` through `t` from where it stands, with whatever
    /// this advance takes of it; the argmax after the last id.
    fn prompt(&mut self, t: &mut T, ids: &[u32]) -> Result<u32, T::Error>;

    /// Generation begins after `prompt`, whose argmax `first` is generated
    /// token 0.
    fn begin(&mut self, t: &T, prompt: &[u32], first: u32) -> Result<(), T::Error>;

    /// One pass from `last`, the token at the target's position: the tokens it
    /// keeps are appended to `out`, in position order.
    fn pass(&mut self, t: &mut T, last: u32, out: &mut Vec<u32>) -> Result<Committed, T::Error>;
}

/// One step a pass.
#[derive(Clone, Copy, Debug, Default)]
pub struct Plain;

impl<T: Target> Advance<T> for Plain {
    fn prompt(&mut self, t: &mut T, ids: &[u32]) -> Result<u32, T::Error> {
        Ok(t.prompt(ids, Want::Argmax)?.argmax())
    }

    fn begin(&mut self, _t: &T, _prompt: &[u32], _first: u32) -> Result<(), T::Error> {
        Ok(())
    }

    fn pass(&mut self, t: &mut T, last: u32, out: &mut Vec<u32>) -> Result<Committed, T::Error> {
        self.pass_with(t, last, &mut Argmax, out)
    }
}

/// What [`generate`] tells its caller around the passes.
pub trait PassSink<T: Target> {
    /// What the sink fails with; a target's error converts into it.
    type Error: From<T::Error>;

    /// Before the first pass, after [`Advance::begin`].
    fn begin(&mut self, t: &T) -> Result<(), Self::Error>;

    /// After each pass: what it kept, its `tokens` in position order, and the
    /// wall time around the advance. The tokens end at an end-of-generation
    /// id: fewer than `c.kept` when one came before the pass's last row.
    fn pass(
        &mut self,
        t: &T,
        c: &Committed,
        tokens: &[u32],
        wall: Duration,
    ) -> Result<(), Self::Error>;
}

/// What a generation ended with.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GenOutcome {
    /// Every token the passes kept, generated token 0 first, up to the first
    /// end-of-generation id. A pass keeps whole rows, so the last may carry
    /// the count past the stop's length, and the target may stand past an
    /// end-of-generation id its pass kept before its last row.
    pub tokens: Vec<u32>,
    pub stop: StopReason,
    pub passes: usize,
}

/// The one generation loop, after the prompt: `first` is the prompt call's
/// argmax, generated token 0. Before each pass the [`Stop`] rule reads the
/// tokens so far, the last of them, the target's position and the advance's
/// [`Advance::ROWS`]; after it, the pass's tokens end at the first
/// end-of-generation id among them, which the next check stops at. Each pass
/// is timed around the advance alone, and `sink` hears of it after the wall
/// is taken. An error of the target, the advance or the sink ends the loop at
/// once.
pub fn generate<T, A, S>(
    t: &mut T,
    a: &mut A,
    prompt: &[u32],
    first: u32,
    stop: &Stop,
    sink: &mut S,
) -> Result<GenOutcome, S::Error>
where
    T: Target,
    A: Advance<T>,
    S: PassSink<T>,
{
    a.begin(t, prompt, first)?;
    sink.begin(t)?;
    let mut tokens = Vec::with_capacity(stop.max_tokens());
    tokens.push(first);
    let mut passes = 0;
    let reason = loop {
        let last = tokens[tokens.len() - 1];
        if let Some(r) = stop.check(tokens.len(), last, t.pos(), A::ROWS) {
            break r;
        }
        let from = tokens.len();
        let t0 = Instant::now();
        let c = a.pass(t, last, &mut tokens)?;
        let wall = t0.elapsed();
        passes += 1;
        if let Some(k) = stop.first_eog(&tokens[from..]) {
            tokens.truncate(from + k + 1);
        }
        sink.pass(t, &c, &tokens[from..], wall)?;
    };
    Ok(GenOutcome {
        tokens,
        stop: reason,
        passes,
    })
}

#[cfg(test)]
pub(crate) mod mock;

#[cfg(test)]
mod tests {
    use super::{RoundError, round_passes};

    /// Passes hold whole windows within both bounds, rows and sequences,
    /// the pass that holds slot 0 last and the others in order; bad bounds
    /// and windows are refused before `cuts` changes.
    #[test]
    fn round_packing_keeps_windows_and_the_slot_zero_pass_last() {
        let mut cuts = Vec::new();
        for limits in [(4, 2), (8, 8)] {
            for n in [1, 2, 3, 4, 8, 9] {
                for each in [1, 2, 4] {
                    round_passes((0..n).map(|slot| (slot, each)), limits, &mut cuts).unwrap();
                    let per = limits.1.min(limits.0 / each);
                    assert_eq!(cuts.len(), n.div_ceil(per), "{n} of {each} at {limits:?}");
                    assert!(cuts.last().unwrap().contains(&0));
                    let mut covered: Vec<_> = cuts.iter().flat_map(Clone::clone).collect();
                    covered.sort_unstable();
                    assert_eq!(covered, (0..n).collect::<Vec<_>>());
                    for cut in &cuts {
                        assert!(cut.len() <= limits.1 && cut.len() * each <= limits.0);
                    }
                }
            }
        }
        for (rows, want) in [
            (vec![3, 3, 3, 3, 4], vec![2..4, 4..5, 0..2]),
            (vec![4, 4, 3, 3, 2], vec![2..5, 0..2]),
            (vec![4, 4], std::iter::once(0..2).collect()),
        ] {
            round_passes(rows.iter().copied().enumerate(), (8, 8), &mut cuts).unwrap();
            assert_eq!(cuts, want);
        }
        for (entries, want) in [
            (vec![(0, 1), (1, 2), (2, 1), (3, 2)], vec![2..4, 0..2]),
            (vec![(1, 3), (0, 2), (2, 2)], vec![0..1, 1..3]),
            (vec![(1, 1), (2, 1), (0, 1), (3, 1)], vec![0..2, 2..4]),
            (vec![(1, 1), (2, 1), (3, 1)], vec![0..2, 2..3]),
        ] {
            round_passes(entries.into_iter(), (4, 2), &mut cuts).unwrap();
            assert_eq!(cuts, want);
        }
        let all = (usize::MAX, usize::MAX);
        round_passes([(1, usize::MAX), (2, 1)].into_iter(), all, &mut cuts).unwrap();
        assert_eq!(cuts, [0..1, 1..2]);
        for (n, limits) in [(0, (8, 8)), (9, (8, 8)), (usize::MAX, (4, 2))] {
            let err = round_passes([(3, n)].into_iter(), limits, &mut cuts).unwrap_err();
            assert!(err.to_string().contains(&format!("slot 3's {n} rows")));
            assert_eq!(cuts, [0..1, 1..2], "a refusal preserves the previous plan");
        }
        for (max_rows, max_slots) in [(0, 0), (4, 0), (2, 3)] {
            let limits = (max_rows, max_slots);
            let err = round_passes([(3, 1)].into_iter(), limits, &mut cuts).unwrap_err();
            assert_eq!(
                err,
                RoundError::Limits {
                    max_rows,
                    max_slots
                }
            );
            assert_eq!(cuts, [0..1, 1..2], "a refusal preserves the previous plan");
        }
    }
}
