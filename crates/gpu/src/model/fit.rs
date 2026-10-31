//! The load-time arena fit rule, one owner for the families that size a pass
//! arena against the card's free bytes: qwen35's ubatch arena (and its draft
//! taps, which read the same reserve) and the Bolt's prefill pass buffers
//! ask the same question — the largest arena whose bytes and a reserve stay
//! within the room ([`crate::Gpu::alloc_room`]) — so the rule and the reserve
//! live here, and a family keeps only its candidate sizes, its need function
//! and the refusal it words for its own sizes. The bytes of the common
//! activation forms and route tables those need functions add up
//! ([`act_col_bytes`], [`act32_col_bytes`], [`route_bytes`]) are exposed here
//! too, with the q8_1 column count delegated to placement's layout owner.

use crate::gemm::GEMM_BN;

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
    usize::try_from(model::placement::kernels::q8_act_bytes(1, k as u64))
        .expect("the q8_1 activation column's bytes fit a usize")
}

/// Bytes of one `GemmAct32` column of `k` values: per 64-value step sixteen
/// code words, two scales and two sums.
pub fn act32_col_bytes(k: usize) -> usize {
    usize::try_from(model::placement::kernels::gemm_act32_bytes(1, k as u64))
        .expect("the q8 blocks of 32's column bytes fit a usize")
}

/// Bytes of a `GemmRoute` table for `slots` slots over `experts` experts.
pub fn route_bytes(slots: usize, experts: usize) -> usize {
    let tiles = slots / GEMM_BN + experts.min(slots);
    let zeros = if experts == 1 { slots } else { 0 };
    (slots + 2 * tiles + 2 + zeros) * 4
}

#[cfg(test)]
mod tests {
    use super::{act_col_bytes, largest_fit};

    #[test]
    fn activation_column_bytes_keep_the_placement_padding() {
        for (k, bytes) in [
            (0, 0),
            (255, 0),
            (256, 2088),
            (511, 2088),
            (512, 2128),
            (768, 3192),
            (1024, 3232),
            (1280, 5320),
            (2048, 6464),
            (2304, 8552),
            (20_480, 64_640),
        ] {
            assert_eq!(act_col_bytes(k), bytes, "k={k}");
        }
        for k in 0..=crate::tensor::Q8ACT_MAX_K {
            assert_eq!(
                act_col_bytes(k) as u64,
                model::placement::kernels::q8_act_bytes(1, k as u64),
                "k={k}",
            );
        }
    }

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
