//! The load-time arena fit rule, one owner for the families that size a pass
//! arena against the card's free bytes: qwen35's ubatch arena (and its draft
//! taps, which read the same reserve) and the Bolt's prefill pass buffers
//! ask the same question — the largest arena whose bytes and a reserve stay
//! within the room ([`crate::Gpu::alloc_room`]) — so the rule and the reserve
//! live here, and a family keeps only its candidate sizes, its need function
//! and the refusal it words for its own sizes. The bytes of the common
//! activation forms and route tables those need functions add up
//! ([`act_col_bytes`], [`act32_col_bytes`], [`route_bytes`]) live here too,
//! once, beside the rule that reads them.

use crate::gemm::{GEMM_BN, GEMM32_STEP};

/// Card bytes a load leaves free past the arena it sizes by this rule: what
/// the load and the first prompt still allocate after the arena — the
/// output head and the heads of one pass (a vocabulary of logits each),
/// the captured step and passes, and the allocator's rounding of the
/// arena's buffers to its 2 MiB pages — with the rest as margin.
pub const FIT_RESERVE: usize = 256 << 20;

/// The largest of `candidates` whose need, as `need` sizes it, plus
/// `reserve` stays within `free` card bytes: the size a load picks, or
/// `None` when no candidate fits — the refusal a caller raises then names
/// the free bytes, the largest candidate's need and this. Pure: it
/// allocates nothing and reads no card (`free` comes from
/// [`crate::Gpu::alloc_room`]).
pub fn largest_fit(
    candidates: impl Iterator<Item = usize>,
    mut need: impl FnMut(usize) -> usize,
    free: usize,
    reserve: usize,
) -> Option<usize> {
    candidates
        .filter(|&size| need(size) + reserve <= free)
        .max()
}

/// Bytes of one q8_1 activation column of `k` values, a `Q8Act`'s or a
/// `GemmAct`'s: the q3 pairs (u64), the q4 and q6 permutations, the 32-value
/// code sums and the 128-value scales.
pub fn act_col_bytes(k: usize) -> usize {
    let n_sb = k / 256;
    64 * n_sb.div_ceil(2) * 8
        + (256 * n_sb.div_ceil(4) + 128 * n_sb.div_ceil(2)) * 4
        + 10 * n_sb * 4
}

/// Bytes of one `GemmAct32` column of `k` values: per 64-value step sixteen
/// code words, two scales and two sums.
pub fn act32_col_bytes(k: usize) -> usize {
    k.div_ceil(GEMM32_STEP) * (16 + 2 + 2) * 4
}

/// Bytes of a `GemmRoute` table for `slots` slots over `experts` experts.
pub fn route_bytes(slots: usize, experts: usize) -> usize {
    let tiles = slots / GEMM_BN + experts.min(slots);
    let zeros = if experts == 1 { slots } else { 0 };
    (slots + 2 * tiles + 2 + zeros) * 4
}

#[cfg(test)]
mod tests {
    use super::largest_fit;

    /// A need of ten bytes a row, 100 bytes of reserve: a candidate fits when
    /// its need and the reserve are within the free bytes, the boundary
    /// included, and the largest fitting one wins whatever order the
    /// candidates come in.
    #[test]
    fn the_largest_candidate_whose_need_and_reserve_fit_wins() {
        let need = |rows: usize| 10 * rows;
        assert_eq!(
            largest_fit([10, 40, 30, 20].into_iter(), need, 400, 100),
            Some(30)
        );
        assert_eq!(largest_fit([10, 30].into_iter(), need, 399, 100), Some(10));
        assert_eq!(largest_fit([40, 30].into_iter(), need, 399, 100), None);
        assert_eq!(largest_fit(std::iter::empty(), need, 400, 100), None);
    }
}
