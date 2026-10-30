//! The load-time row moves this family's indexer tensors need: the file lays
//! each index head out `[rope | nope]` — llama.cpp's graph ropes the head's
//! first `n_rot` values (src/models/mistral-large4.cpp: "the indexer head
//! layout is [rope | nope]") — while the V4.1 indexer kernel the body runs
//! turns each head's last `n_dims` values (gpu-deepseek41/src/indexer.rs: "the
//! rope of each head's last `n_dims` values"). The load moves each head's
//! first `rope` rows to the tail in `indexer.attn_q_b`, `indexer.attn_k` and
//! the k-norm's weight and bias, so the kernel runs unchanged. LayerNorm is
//! equivariant under a row permutation and `(Hq)·(Hk)ᵀ = q·k` for the
//! Hadamard `H`, so the scores are the file's own; only the f16 rounding of
//! the reordered stores moves. This is the one model-only fact of the family
//! (the design's architecture map): everything else runs on common owners.

/// Within each `head`-row block of `rows`, move the first `rope` rows to the
/// tail: the file's `[rope | nope]` index head as the kernel's `[nope |
/// rope]`. `rows.len()` is a whole number of `head`-row blocks and `rope`
/// below `head`; a caller that breaks either panics by name.
pub fn rope_rows_to_tail<T>(rows: &mut [T], head: usize, rope: usize) {
    assert!(head > 0, "an index head of no rows");
    assert!(rope < head, "the rope part {rope} fills the head of {head}");
    assert!(
        rows.len().is_multiple_of(head),
        "{} rows are not a whole number of {head}-row heads",
        rows.len()
    );
    for block in rows.chunks_mut(head) {
        block.rotate_left(rope);
    }
}

#[cfg(test)]
mod tests {
    use super::rope_rows_to_tail;

    /// The first `rope` rows of every head move to its tail, each head's
    /// block by itself; the rope width a proper part of the head. Mutant:
    /// `rotate_right` for `rotate_left`, or the whole-rows assert dropped.
    #[test]
    fn the_rope_rows_move_to_each_heads_tail() {
        // Two heads of four rows, the first two of each the rope part.
        let mut rows = [0, 1, 2, 3, 4, 5, 6, 7];
        rope_rows_to_tail(&mut rows, 4, 2);
        assert_eq!(rows, [2, 3, 0, 1, 6, 7, 4, 5]);
        // A rope part of none is no move.
        let mut rows = [0, 1, 2, 3];
        rope_rows_to_tail(&mut rows, 4, 0);
        assert_eq!(rows, [0, 1, 2, 3]);
        // One head alone: the file's own `indexer.attn_k` shape.
        let mut rows = [0, 1, 2, 3, 4, 5];
        rope_rows_to_tail(&mut rows, 6, 2);
        assert_eq!(rows, [2, 3, 4, 5, 0, 1]);
    }

    /// A rope part that fills the head, and rows that are not whole heads,
    /// are refused by name.
    #[test]
    #[should_panic(expected = "the rope part 4 fills the head of 4")]
    fn a_rope_part_that_fills_the_head_is_named() {
        rope_rows_to_tail(&mut [0; 4], 4, 4);
    }

    /// Rows that are not a whole number of heads are refused by name.
    #[test]
    #[should_panic(expected = "6 rows are not a whole number of 4-row heads")]
    fn rows_between_heads_are_named() {
        rope_rows_to_tail(&mut [0; 6], 4, 2);
    }
}
