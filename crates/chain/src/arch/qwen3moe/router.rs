//! The qwen3moe family's ubatch cap over a router's slots: the planned ubatch and the tokens a
//! ubatch of [`RouterDims`] may hold.

use crate::router::RouterDims;

/// The most tokens one ubatch takes; its slots must also fit the slots one
/// route of the backend's grouped projection holds at the file's `top_k`
/// ([`RouterDims::ubatch`]). Also the default ubatch size: the one that reads
/// each weight the fewest times per prompt and gives each expert's GEMM the
/// most columns. Its owner is the placement's planned ubatch, the size a
/// default machine counts.
pub const UBATCH: usize = model::arch::qwen35moe::place::UBATCH_PLANNED as usize;

impl RouterDims {
    /// The most tokens a ubatch routing may hold ([`ubatch_at`] at this
    /// router's slots a token).
    #[must_use]
    pub fn ubatch(&self, max_slots: usize) -> usize {
        ubatch_at(self.slots(), max_slots)
    }
}

/// The most tokens a ubatch of `slots` slots a token may hold: at most
/// [`UBATCH`], and every slot of them within the `max_slots` one route of the
/// backend's grouped projection holds (a dense FFN's one slot a token too).
#[must_use]
pub fn ubatch_at(slots: usize, max_slots: usize) -> usize {
    UBATCH.min(max_slots / slots)
}

#[cfg(test)]
mod tests {
    use super::{UBATCH, ubatch_at};

    /// A ubatch holds at most [`UBATCH`] tokens and no more slots than one route holds: the
    /// planned ubatch at nine slots a token fills a route of `9 · UBATCH` slots, one slot less cuts
    /// a token, and a dense FFN's one slot a token keeps the cap.
    #[test]
    fn a_ubatch_fits_one_route_and_the_cap() {
        assert_eq!(ubatch_at(9, 9 * UBATCH), UBATCH);
        assert_eq!(ubatch_at(9, 9 * UBATCH - 1), UBATCH - 1);
        assert_eq!(ubatch_at(1, 9 * UBATCH), UBATCH);
    }
}
