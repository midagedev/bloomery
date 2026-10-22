//! The geometry of a Gated DeltaNet layer (linear attention): the widths, the conv ring and the
//! head counts a layer's plan and its kernels agree on. The layer's rule and layouts are written
//! with the kernels that run them.

use crate::ChainError;

/// Width of every query, key and value head (d_k = d_v).
pub const HEAD: usize = 128;

/// Taps of the causal conv.
pub const CONV_TAPS: usize = 4;

/// A token's predecessors the conv reads: the inputs of its last
/// `CONV_TAPS − 1` positions.
pub const CONV_ROWS: usize = CONV_TAPS - 1;

/// The widest call a later call may roll back into: the pass of up to eight
/// positions (a verify of up to seven drafted tokens).
pub const PASS_ROWS: usize = 8;

/// Slots of the conv ring. A rollback to position `n` inside a call of `m <=
/// PASS_ROWS` positions from `p` reads the inputs of `n − 3 .. n − 1 >= p −
/// 2`, which the ring holds while it keeps `m + 2` positions; with `m + 3`
/// the call's writes also never land on the slots of `p − 3 .. p − 1`.
pub const RING_ROWS: usize = CONV_ROWS + PASS_ROWS;

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
        RING_ROWS * self.channels()
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
