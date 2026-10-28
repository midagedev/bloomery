//! The geometry of a Gated DeltaNet layer (linear attention): the widths, the conv ring and the
//! head counts a layer's plan and its kernels agree on. The layer's rule and layouts are written
//! with the kernels that run them. The conv ring is [`Ring`], whatever the conv's width: the
//! delta layers and Mamba-2 keep [`DELTA_RING`], four taps, and a short conv keeps its own.

use crate::ChainError;
use runtime::stores;

/// Width of every query, key and value head (d_k = d_v).
pub const HEAD: usize = 128;

/// Taps of the delta layers' and Mamba-2's causal conv.
pub const CONV_TAPS: usize = 4;

/// The conv ring of the delta layers and Mamba-2: [`CONV_TAPS`] taps.
pub const DELTA_RING: Ring = Ring::new(CONV_TAPS);

/// A token's predecessors the delta conv reads: the inputs of its last
/// `CONV_TAPS − 1` positions ([`Ring::reach`] of [`DELTA_RING`]).
pub const CONV_ROWS: usize = DELTA_RING.reach();

/// The widest call a later call may roll back into: the pass of up to eight
/// positions (a verify of up to seven drafted tokens). `runtime::stores` owns
/// the count.
pub const PASS_ROWS: usize = stores::PASS_ROWS;

/// Slots of the delta conv ring ([`Ring::rows`] of [`DELTA_RING`]).
pub const RING_ROWS: usize = DELTA_RING.rows();

/// The conv ring of a layer whose causal depthwise conv has `taps` taps: the inputs of the last
/// positions a call wrote, the input of position `p` in slot `p mod rows`, so a later call finds
/// its tokens' predecessors by position without a copy. The delta layers' conv, Mamba-2's and a
/// short conv's differ in `taps` alone; what a ring does with it is here once.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ring {
    taps: usize,
}

impl Ring {
    /// The ring of a `taps`-tap conv.
    ///
    /// # Panics
    ///
    /// On a conv of no tap, by name.
    #[must_use]
    pub const fn new(taps: usize) -> Ring {
        assert!(taps >= 1, "Ring::new: a conv of at least one tap");
        Ring { taps }
    }

    /// Taps of the conv, the current position's among them.
    #[must_use]
    pub const fn taps(self) -> usize {
        self.taps
    }

    /// A token's predecessors the conv reads: the inputs of its last `taps − 1` positions.
    #[must_use]
    pub const fn reach(self) -> usize {
        self.taps - 1
    }

    /// Slots of the ring, [`stores::conv_ring_rows`]: the reach and a pass ([`PASS_ROWS`]). A
    /// rollback to position `n` inside a call of `m <= PASS_ROWS` positions from `p` reads the
    /// inputs of `n − reach .. n − 1 >= p − reach + 1`, which the ring holds while it keeps
    /// `m + reach − 1` positions; with `m + reach` the call's writes also never land on the slots
    /// of `p − reach .. p − 1`.
    #[must_use]
    pub const fn rows(self) -> usize {
        stores::conv_ring_rows(self.taps)
    }

    /// f32s of a ring over `channels` channels: `rows · channels`.
    #[must_use]
    pub const fn values(self, channels: usize) -> usize {
        self.rows() * channels
    }

    /// The slot of the input `d` positions before position `p` (`1 <= d <= reach`): `None` while
    /// `p < d`, the sequence's start, where the input is zero whatever the ring holds; else
    /// `(p − d) mod rows`. The one indexing of the ring, which the kernels spell out in their own
    /// terms.
    ///
    /// # Panics
    ///
    /// On a `d` of 0 or past the reach, by name.
    #[must_use]
    pub const fn before(self, p: usize, d: usize) -> Option<usize> {
        assert!(
            d >= 1,
            "Ring::before: a predecessor at least 1 position back"
        );
        assert!(
            d <= self.reach(),
            "Ring::before: a predecessor at most taps − 1 positions back"
        );
        if p < d {
            None
        } else {
            Some((p + self.rows() - d) % self.rows())
        }
    }
}

/// Tokens of one chunk of the prompt form (the chunked delta rule): a
/// prompt's chunks start at the multiples of it from the position its rows
/// began at, whatever calls cut the rows. `runtime::stores` owns the count.
pub const CHUNK_ROWS: usize = stores::DELTA_CHUNK_ROWS;

/// Rows of the carry: the tokens of the chunk a call left incomplete.
pub const CARRY_ROWS: usize = stores::DELTA_CARRY_ROWS;

/// Words before the carry's rows: the rows' count, the position after them
/// and two spare words that keep the rows 16-byte aligned.
pub const CARRY_HEAD: usize = stores::DELTA_CARRY_HEAD;

/// The query/key head value head `h` reads when `n_v > n_k`: a property of
/// the file's converter, not of the architecture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KHeadMap {
    /// `h % n_k`: the converter tiled the value heads (Qwen3.5 and later).
    Tiled,
    /// `h / (n_v / n_k)`: grouped value heads (Qwen3-Next).
    Grouped,
}

impl KHeadMap {
    /// The key head of value head `h`.
    #[must_use]
    pub fn k_head(self, h: usize, n_k: usize, n_v: usize) -> usize {
        match self {
            KHeadMap::Tiled => h % n_k,
            KHeadMap::Grouped => h / (n_v / n_k),
        }
    }
}

/// The head counts of one layer. `HEAD`, the conv width and the state layout
/// are fixed; these are the launch arguments.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinearShape {
    /// Query/key heads.
    pub n_k: usize,
    /// Value heads, a multiple of `n_k`.
    pub n_v: usize,
    /// How a value head finds its key head.
    pub map: KHeadMap,
}

impl LinearShape {
    /// Channels of the conv: `(2·n_k + n_v)·HEAD`.
    #[must_use]
    pub fn channels(&self) -> usize {
        (2 * self.n_k + self.n_v) * HEAD
    }

    /// Heads of the conv, each [`HEAD`] channels: `2·n_k + n_v`.
    #[must_use]
    pub fn conv_heads(&self) -> usize {
        2 * self.n_k + self.n_v
    }

    /// f32s of one lane of the recurrent state: `n_v·HEAD·HEAD`.
    #[must_use]
    pub fn state_len(&self) -> usize {
        self.n_v * HEAD * HEAD
    }

    /// f32s of the conv ring: `RING_ROWS·C`.
    #[must_use]
    pub fn ring_len(&self) -> usize {
        DELTA_RING.values(self.channels())
    }

    /// f32s of the carry: [`CARRY_HEAD`] words, then [`CARRY_ROWS`] rows of
    /// the key and value heads, `(n_k + n_v)·HEAD`, then the rows' β and
    /// their decays, `n_v` a row each.
    #[must_use]
    pub fn carry_len(&self) -> usize {
        let bytes = stores::delta_carry_bytes(self.n_v, self.n_k, HEAD);
        usize::try_from(bytes / stores::F32_BYTES).expect("carry count was computed in usize")
    }

    /// f32s of the state of `lanes` lanes and of the conv ring: the pair
    /// a recurrent store (`RecStore::new`) allocates. The ring is not multiplied
    /// by `lanes`.
    #[must_use]
    pub fn lens(&self, lanes: usize) -> [usize; 2] {
        [lanes * self.state_len(), self.ring_len()]
    }

    /// Refuse by name a shape the kernels have no geometry for.
    pub fn check(&self, what: &'static str) -> Result<(), ChainError> {
        if self.n_k == 0 || self.n_v == 0 || !self.n_v.is_multiple_of(self.n_k) {
            return Err(ChainError::shape(
                what,
                format!(
                    "need n_k >= 1 and n_v a positive multiple of n_k, got n_k={} n_v={}",
                    self.n_k, self.n_v
                ),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{HEAD, KHeadMap, LinearShape, PASS_ROWS, Ring};

    /// A ring is the conv's reach and a pass: 8 rows at one tap, 10 at three, the delta layers'
    /// 11 at four (the delta layers'), and 14 at the seven taps of a short conv. Mutant: `Ring::rows` returns `taps + PASS_ROWS`,
    /// or `Ring::reach` returns `taps`.
    #[test]
    fn a_ring_is_the_reach_and_a_pass() {
        let rows: Vec<(usize, usize, usize)> = [1, 3, 4, 7]
            .into_iter()
            .map(|t| (t, Ring::new(t).reach(), Ring::new(t).rows()))
            .collect();
        assert_eq!(rows, [(1, 0, 8), (3, 2, 10), (4, 3, 11), (7, 6, 14)]);
        let shape = LinearShape {
            n_k: 16,
            n_v: 48,
            map: KHeadMap::Tiled,
        };
        assert_eq!(shape.channels(), (2 * 16 + 48) * HEAD);
        assert_eq!(shape.ring_len(), 11 * 10_240);
        assert_eq!(Ring::new(7).values(2048), 14 * 2048);
    }

    /// After a call of `m <= PASS_ROWS` positions from `p`, the slots hold the inputs the next
    /// call reads, whether it continues at `p + m` or a rollback restarts it at any `n` between,
    /// `p` itself included: every predecessor `n − d` is in slot [`Ring::before`] names, or is
    /// before the sequence's start and so zero. The value written at position `q` is `q + 1`.
    /// Mutant: `Ring::rows` one row short (`reach + PASS_ROWS − 1`: the call at `m = PASS_ROWS`
    /// overwrites the slot of `p − reach`), or `before` returns `(p + d) % rows`.
    #[test]
    fn a_ring_holds_every_input_the_next_call_reads() {
        for taps in 1..=8 {
            let ring = Ring::new(taps);
            let rows = ring.rows();
            for p in 0..2 * rows + 3 {
                for m in 1..=PASS_ROWS {
                    let mut slots = vec![0usize; rows];
                    for q in 0..p + m {
                        slots[q % rows] = q + 1;
                    }
                    for n in p..=p + m {
                        for d in 1..=ring.reach() {
                            let want = if n < d { 0 } else { n - d + 1 };
                            let got = ring.before(n, d).map_or(0, |slot| slots[slot]);
                            assert_eq!(
                                got, want,
                                "taps {taps}, call {m} from {p}, next at {n}, d {d}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// The sequence start reads zeros whatever the ring holds: `before` is `None` while the
    /// position is short of the distance. Mutant: `Ring::before` tests `p <= d`.
    #[test]
    fn a_position_short_of_its_distance_has_no_slot() {
        let ring = Ring::new(7);
        assert_eq!(ring.before(2, 3), None);
        assert_eq!(ring.before(3, 3), Some(0));
        assert_eq!(ring.before(20, 6), Some(0));
        assert_eq!(ring.before(21, 6), Some(1));
    }

    /// No conv has no tap.
    #[test]
    #[should_panic(expected = "Ring::new: a conv of at least one tap")]
    fn a_ring_of_no_tap_is_refused_by_name() {
        let _ = Ring::new(0);
    }
}
