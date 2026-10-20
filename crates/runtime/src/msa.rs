//! The block selector of MiniMax-M3's sparse layers (MSA) as the integer and ordering rules the
//! card's kernels (`bloomery_gpu_minimaxm3::attn`) and their gate follow: which blocks a query
//! row sees, which it keeps, how many keys they stand for, and the order at the cut.
//!
//! A row is named by its live count `c` — its position plus one, the keys it sees. Every `block`
//! consecutive positions `[block·b, block·b + block)` form block `b`; the row sees `⌈c / block⌉`
//! blocks, the last one (its own, `L = (c − 1) / block`) partial. Each index head scores a block
//! by the largest score of the block's visible positions, and keeps the `top_k` best blocks; the
//! `local` blocks nearest the row, its own first, are forced in by a bias of [`FORCE`] added to
//! their score, so they rank above any finite one. At or below `top_k` visible blocks the
//! selection is every block: dense attention.
//!
//! The kept blocks are listed ascending, so the row's own block is last and the keys they stand
//! for are `(kept − 1)·block` whole-block positions and the `(c − 1) % block + 1` visible ones of
//! the last. Order at the cut: score descending through [`qsa::order_key`] (which ties `−0.0`
//! with `+0.0` and orders every bit pattern, NaN included), then the lower block.
//!
//! Every index head of a row has the same visible blocks, so the kept count and the key count
//! are the row's, whatever the head.

use std::fmt;

use crate::qsa;

/// The bias added to a forced block's score: ranks it above any finite score. The forced blocks
/// tie with each other at this value, and all of them are kept.
pub const FORCE: f32 = 1e30;

/// A selector's shape: `top_k` blocks kept of `block` positions, the `local` nearest forced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Msa {
    top_k: usize,
    block: usize,
    local: usize,
}

/// A shape [`Msa::new`] refuses: `top_k` or `block` zero, or `local` not within `1..=top_k`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MsaShapeError {
    pub top_k: u32,
    pub block: u32,
    pub local: u32,
}

impl fmt::Display for MsaShapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "block selector: top_k {} block {} local {}; need top_k and block at least 1 and \
             local within 1..=top_k (the row's own block is always kept)",
            self.top_k, self.block, self.local
        )
    }
}

impl std::error::Error for MsaShapeError {}

impl Msa {
    /// The selector keeping `top_k` blocks of `block` positions, `local` of them forced.
    ///
    /// # Errors
    /// [`MsaShapeError`] when `top_k` or `block` is zero or `local` is outside `1..=top_k`.
    pub fn new(top_k: u32, block: u32, local: u32) -> Result<Msa, MsaShapeError> {
        if top_k == 0 || block == 0 || local == 0 || local > top_k {
            return Err(MsaShapeError {
                top_k,
                block,
                local,
            });
        }
        Ok(Msa {
            top_k: top_k as usize,
            block: block as usize,
            local: local as usize,
        })
    }

    /// Blocks a row keeps when it sees more.
    #[must_use]
    pub fn top_k(self) -> usize {
        self.top_k
    }

    /// Positions per block.
    #[must_use]
    pub fn block(self) -> usize {
        self.block
    }

    /// Blocks forced in, the row's own first.
    #[must_use]
    pub fn local(self) -> usize {
        self.local
    }

    /// The blocks a row of live count `count` sees: `⌈count / block⌉`, 0 for no key.
    #[must_use]
    pub fn visible(self, count: usize) -> usize {
        count.div_ceil(self.block)
    }

    /// Whether a row of live count `count` selects: it sees more than `top_k` blocks.
    #[must_use]
    pub fn selects(self, count: usize) -> bool {
        self.visible(count) > self.top_k
    }

    /// The one predicate for "this row is scored": it [`Msa::selects`] and its count is accepted
    /// (`1..=ctx`). The score pass writes a row's block scores only then and the select pass
    /// reads them only then.
    #[must_use]
    pub fn scored(self, count: usize, ctx: usize) -> bool {
        (1..=ctx).contains(&count) && self.selects(count)
    }

    /// Blocks a row of live count `count` keeps: all it sees, at most `top_k`.
    #[must_use]
    pub fn kept(self, count: usize) -> usize {
        self.visible(count).min(self.top_k)
    }

    /// The keys the kept blocks stand for: the whole blocks before the row's own and its visible
    /// positions; `count` when the row does not select, 0 for no key.
    #[must_use]
    pub fn key_count(self, count: usize) -> usize {
        match self.kept(count) {
            0 => 0,
            kept => (kept - 1) * self.block + (count - 1) % self.block + 1,
        }
    }

    /// The longest key count of any row: `top_k` whole blocks.
    #[must_use]
    pub fn width(self) -> usize {
        self.top_k * self.block
    }

    /// The bias block `b` of a row of live count `count` takes: [`FORCE`] for the `local` blocks
    /// nearest the row's own (the own block included), else `0.0`.
    #[must_use]
    pub fn bias(self, count: usize, b: usize) -> f32 {
        let visible = self.visible(count);
        if b < visible && visible - 1 - b < self.local {
            FORCE
        } else {
            0.0
        }
    }

    /// The blocks a row of live count `count` keeps, ascending: every visible block when it does
    /// not [`Msa::selects`], else the `top_k` best of `pooled` (block `b`'s largest score) after
    /// [`Msa::bias`], a tie to the lower block.
    ///
    /// # Panics
    /// When the row selects and `pooled` holds fewer than its visible blocks.
    #[must_use]
    pub fn select(self, count: usize, pooled: &[f32]) -> Vec<u32> {
        let visible = self.visible(count);
        if !self.selects(count) {
            return (0..visible)
                .map(|b| u32::try_from(b).expect("a block index fits u32"))
                .collect();
        }
        assert!(
            pooled.len() >= visible,
            "{} block scores for {visible} visible blocks",
            pooled.len()
        );
        let biased: Vec<f32> = pooled[..visible]
            .iter()
            .enumerate()
            .map(|(b, &s)| s + self.bias(count, b))
            .collect();
        qsa::select_pools(&biased, self.top_k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// MiniMax-M3: `attention.indexer.top_k` 16, `block_size` 128, `local_blocks` 1.
    fn m3() -> Msa {
        Msa::new(16, 128, 1).expect("the model's shape")
    }

    #[test]
    fn a_shape_without_its_own_block_is_refused() {
        for (k, b, l) in [(0, 128, 1), (16, 0, 1), (16, 128, 0), (16, 128, 17)] {
            assert_eq!(
                Msa::new(k, b, l),
                Err(MsaShapeError {
                    top_k: k,
                    block: b,
                    local: l
                })
            );
        }
        assert!(Msa::new(16, 128, 16).is_ok());
    }

    /// Below one block, at one, at 16 and at 17 blocks: up to 2,048 live keys the list is every
    /// block and the keys are every key; the 2,049th key opens a 17th block and drops one.
    #[test]
    fn up_to_sixteen_blocks_the_selection_is_dense() {
        let m = m3();
        assert!(!m.selects(2048) && m.selects(2049));
        for count in [1, 2, 127, 128, 129, 1000, 2047, 2048] {
            assert!(!m.selects(count), "count {count}");
            assert_eq!(m.key_count(count), count, "count {count}");
            assert_eq!(m.kept(count), m.visible(count), "count {count}");
            let all: Vec<u32> = (0..m.visible(count) as u32).collect();
            assert_eq!(m.select(count, &[]), all, "count {count}");
        }
        assert_eq!(m.visible(0), 0);
        assert_eq!(m.key_count(0), 0);
        assert_eq!(m.visible(1), 1);
        assert_eq!(m.visible(128), 1);
        assert_eq!(m.visible(129), 2);
        // The first selecting row keeps 15 whole blocks and one key of the 17th.
        assert_eq!(m.key_count(2049), 15 * 128 + 1);
        // At 32k: 16 blocks, the last one whole.
        assert_eq!(m.key_count(32768), 15 * 128 + 128);
        assert_eq!(m.visible(32768), 256);
        assert_eq!(m.width(), 2048);
        for count in 2049..40_000 {
            assert!(m.key_count(count) <= m.width(), "count {count}");
            assert!(m.key_count(count) > 15 * 128, "count {count}");
        }
    }

    #[test]
    fn scored_is_the_selecting_rows_with_an_accepted_count() {
        let m = m3();
        let ctx = 4096;
        assert!(!m.scored(0, ctx));
        assert!(!m.scored(2048, ctx));
        assert!(m.scored(2049, ctx));
        assert!(m.scored(ctx, ctx));
        assert!(!m.scored(ctx + 1, ctx));
    }

    /// 17 blocks, the 16 best kept: the one dropped is the lowest-scoring block that is not the
    /// row's own, even when the own block scores lowest of all. Mutant: no bias — block 16 (the
    /// own block, score −5) is dropped and the key count's "own block last" no longer holds.
    #[test]
    fn the_own_block_is_kept_whatever_it_scores() {
        let m = m3();
        let count = 2049;
        let mut pooled: Vec<f32> = (0..17).map(|b| 1.0 + b as f32).collect();
        pooled[16] = -5.0;
        pooled[3] = -2.0;
        let kept = m.select(count, &pooled);
        assert_eq!(kept.len(), 16);
        assert!(kept.contains(&16), "{kept:?}");
        assert!(!kept.contains(&3), "{kept:?}");
        assert!(kept.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(*kept.last().unwrap() as usize, (count - 1) / 128);
        assert_eq!(m.bias(count, 16), FORCE);
        assert_eq!(m.bias(count, 15), 0.0);
        assert_eq!(m.bias(count, 17), 0.0);
    }

    /// Tied scores at the cut keep the lower block, `−0.0` ties `+0.0`. Mutant: the higher
    /// block of a tie kept — 4 is kept instead of 2.
    #[test]
    fn a_tie_at_the_cut_keeps_the_lower_block() {
        let m = m3();
        let count = 2049;
        // Blocks 0..16 score 1.0 except two zeros at 2 and 4 (one −0.0): the cut falls between
        // them, one of the zeros is dropped, and it is the higher.
        let mut pooled = vec![1.0f32; 17];
        pooled[2] = 0.0;
        pooled[4] = -0.0;
        pooled[16] = 0.0;
        let kept = m.select(count, &pooled);
        assert_eq!(kept.len(), 16);
        assert!(kept.contains(&2) && !kept.contains(&4), "{kept:?}");
    }

    /// More forced blocks than one: the nearest `local` blocks are all kept.
    #[test]
    fn local_blocks_are_the_nearest_ones() {
        let m = Msa::new(4, 8, 2).expect("a shape");
        let count = 8 * 6 + 3; // blocks 0..=6, own block 6
        let mut pooled = vec![1.0f32; 7];
        pooled[5] = -9.0;
        pooled[6] = -9.0;
        let kept = m.select(count, &pooled);
        assert!(kept.contains(&5) && kept.contains(&6), "{kept:?}");
        assert_eq!(kept.len(), 4);
        assert_eq!(m.key_count(count), 3 * 8 + 3);
        assert_eq!((m.bias(count, 5), m.bias(count, 4)), (FORCE, 0.0));
    }

    #[test]
    #[should_panic(expected = "block scores for")]
    fn a_selecting_row_needs_its_scores() {
        let _ = m3().select(2049, &[0.0; 16]);
    }
}
