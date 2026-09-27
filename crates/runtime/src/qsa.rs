//! The token-pool selector of Qwen3.8's full-attention layers (QSA) and of
//! GLM-5.3-Flash's latent layers (the k-pool indexer), as the integer and
//! ordering rules the card's kernels (`bloomery_gpu::qsa`) and their gates
//! follow: which pools a query row may keep, which tokens a kept pool and the
//! tail stand for, the tie order at the cut, and the one predicate that says
//! whether a row is scored at all.
//!
//! A row is named by its live count `c` — its position plus one, the keys it
//! sees. Every `pool` consecutive tokens `[pool·j, pool·j + pool)` form pool
//! `j`; the row sees the `c / pool` complete pools below it and the tail of
//! `c % pool` tokens after them. It keeps the `top_k / pool` best-scoring
//! complete pools (all of them when there are no more) and always the tail,
//! so its list holds at most `top_k + pool − 1` tokens, ascending. At or
//! below that many live keys the list is every token: the selection equals
//! dense attention there.
//!
//! Order at the cut: score descending, then the [`Tie`] rule — Qwen3.8's
//! selector keeps the lower pools of a tie, GLM's the higher ones. Scores are
//! compared through [`order_key`], which ties `−0.0` with `+0.0` and orders
//! every bit pattern, NaN included (the card raises a fault on a non-finite
//! score and still writes a defined list).

use std::fmt;

/// A selector's shape: `top_k` tokens kept, in pools of `pool` tokens.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Qsa {
    top_k: usize,
    pool: usize,
}

/// A shape [`Qsa::new`] refuses: `pool` zero, or `top_k` not a positive
/// multiple of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QsaShapeError {
    pub top_k: u32,
    pub pool: u32,
}

impl fmt::Display for QsaShapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "token-pool selector: top_k {} is not a positive multiple of pool {}",
            self.top_k, self.pool
        )
    }
}

impl std::error::Error for QsaShapeError {}

impl Qsa {
    /// The selector keeping `top_k` tokens in pools of `pool`.
    ///
    /// # Errors
    /// [`QsaShapeError`] when `pool` is zero or `top_k` is not a positive
    /// multiple of it.
    pub fn new(top_k: u32, pool: u32) -> Result<Qsa, QsaShapeError> {
        if pool == 0 || top_k == 0 || !top_k.is_multiple_of(pool) {
            return Err(QsaShapeError { top_k, pool });
        }
        Ok(Qsa {
            top_k: top_k as usize,
            pool: pool as usize,
        })
    }

    /// Tokens per pool.
    #[must_use]
    pub fn pool(self) -> usize {
        self.pool
    }

    /// Pools a row keeps when it sees more: `top_k / pool`.
    #[must_use]
    pub fn kept(self) -> usize {
        self.top_k / self.pool
    }

    /// The longest list: `top_k` tokens of kept pools and a tail of `pool − 1`.
    #[must_use]
    pub fn width(self) -> usize {
        self.top_k + self.pool - 1
    }

    /// Complete pools a row of live count `count` sees.
    #[must_use]
    pub fn complete(self, count: usize) -> usize {
        count / self.pool
    }

    /// Tokens of its tail: the incomplete pool it sits in.
    #[must_use]
    pub fn tail(self, count: usize) -> usize {
        count % self.pool
    }

    /// Whether the row drops a pool: it sees more complete pools than it keeps.
    #[must_use]
    pub fn selects(self, count: usize) -> bool {
        self.complete(count) > self.kept()
    }

    /// The largest live count whose list is every token: `kept·pool + pool −
    /// 1`, the [`Qsa::width`]. Positions up to this count minus one attend
    /// densely.
    #[must_use]
    pub fn dense_counts(self) -> usize {
        self.kept() * self.pool + self.pool - 1
    }

    /// Whether a row of live count `count` over a cache of `ctx` rows is
    /// scored: the count is one the kernels accept (`1..=ctx`) and the row
    /// [`Qsa::selects`]. The score pass writes the row's scores only then and
    /// the top-k pass reads them only then: both kernels decide through this
    /// one rule, or the top-k ranks scores no pass of this call wrote.
    #[must_use]
    pub fn scored(self, count: usize, ctx: usize) -> bool {
        count >= 1 && count <= ctx && self.selects(count)
    }

    /// The pool a row of live count `count` completes — its own token is the
    /// pool's last — if any. Every pool is completed by exactly one count.
    #[must_use]
    pub fn completes(self, count: usize) -> Option<usize> {
        (count >= self.pool && count.is_multiple_of(self.pool)).then(|| count / self.pool - 1)
    }

    /// The length of the list of a row of live count `count`: its kept pools'
    /// tokens and its tail.
    #[must_use]
    pub fn list_len(self, count: usize) -> usize {
        self.complete(count).min(self.kept()) * self.pool + self.tail(count)
    }
}

/// An unsigned key that orders as the score `v` does: `−0.0` made `+0.0`
/// first, so the two tie; negative values below positive ones; every NaN
/// pattern above `+inf`.
#[must_use]
pub fn order_key(v: f32) -> u32 {
    let b = v.to_bits();
    let b = if b == 0x8000_0000 { 0 } else { b };
    if b & 0x8000_0000 != 0 {
        !b
    } else {
        b | 0x8000_0000
    }
}

/// Which of the pools tied at the cut a row keeps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tie {
    /// The lower pools: Qwen3.8's selector.
    Lower,
    /// The higher pools: ik's CPU `top_k`, a descending sort of `(score,
    /// index)` pairs, which GLM's k-pool selector follows.
    Higher,
}

/// The `k` best of `scores` (pool `j` scored `scores[j]`) by [`order_key`]
/// descending, a tie to the lower pool; all of them when there are at most
/// `k`. Ascending pool indices.
#[must_use]
pub fn select_pools(scores: &[f32], k: usize) -> Vec<u32> {
    select_pools_by(scores, k, Tie::Lower)
}

/// [`select_pools`] with the tie rule `tie`.
#[must_use]
pub fn select_pools_by(scores: &[f32], k: usize, tie: Tie) -> Vec<u32> {
    let mut order: Vec<usize> = (0..scores.len()).collect();
    match tie {
        Tie::Lower => order.sort_by_key(|&j| (std::cmp::Reverse(order_key(scores[j])), j)),
        Tie::Higher => order.sort_by_key(|&j| std::cmp::Reverse((order_key(scores[j]), j))),
    }
    let mut kept: Vec<u32> = order
        .into_iter()
        .take(k)
        .map(|j| u32::try_from(j).expect("a pool index fits u32"))
        .collect();
    kept.sort_unstable();
    kept
}

/// The tokens of a row of live count `count` that keeps `pools` (ascending,
/// each below its complete pools): every token of each kept pool, then the
/// tail. Ascending.
///
/// # Panics
/// When `pools` is not ascending or names a pool the row does not see
/// complete.
#[must_use]
pub fn token_list(q: Qsa, count: usize, pools: &[u32]) -> Vec<u32> {
    let complete = q.complete(count);
    assert!(
        pools.windows(2).all(|w| w[0] < w[1]),
        "kept pools must be ascending and distinct: {pools:?}"
    );
    assert!(
        pools.iter().all(|&j| (j as usize) < complete),
        "a kept pool must be complete below count {count} ({complete} pools): {pools:?}"
    );
    let tok = |t: usize| u32::try_from(t).expect("a token index fits u32");
    let mut out: Vec<u32> = pools
        .iter()
        .flat_map(|&j| (0..q.pool).map(move |i| j as usize * q.pool + i))
        .map(tok)
        .collect();
    out.extend((complete * q.pool..count).map(tok));
    out
}

/// The list of a row of live count `count`: every token when it does not
/// [`Qsa::selects`], else the tokens of the [`select_pools`] of its complete
/// pools' `scores` and its tail.
///
/// # Panics
/// When the row selects and `scores` holds fewer than its complete pools.
#[must_use]
pub fn select(q: Qsa, count: usize, scores: &[f32]) -> Vec<u32> {
    select_by(q, count, scores, Tie::Lower)
}

/// [`select`] with the tie rule `tie`.
///
/// # Panics
/// As [`select`].
#[must_use]
pub fn select_by(q: Qsa, count: usize, scores: &[f32], tie: Tie) -> Vec<u32> {
    if !q.selects(count) {
        return (0..count)
            .map(|t| u32::try_from(t).expect("a token index fits u32"))
            .collect();
    }
    let complete = q.complete(count);
    assert!(
        scores.len() >= complete,
        "{} scores for {complete} complete pools",
        scores.len()
    );
    token_list(
        q,
        count,
        &select_pools_by(&scores[..complete], q.kept(), tie),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Qwen3.8-Flash-Next's full-attention layers: `attention.indexer.top_k`
    /// 2,048, `attention.compress_ratios` 4.
    fn q38() -> Qsa {
        Qsa::new(2048, 4).expect("Qwen3.8's shape")
    }

    #[test]
    fn a_shape_that_is_not_whole_pools_is_refused() {
        assert_eq!(
            Qsa::new(2048, 0),
            Err(QsaShapeError {
                top_k: 2048,
                pool: 0
            })
        );
        assert_eq!(
            Qsa::new(2050, 4),
            Err(QsaShapeError {
                top_k: 2050,
                pool: 4
            })
        );
        assert_eq!(Qsa::new(0, 4), Err(QsaShapeError { top_k: 0, pool: 4 }));
        let q = q38();
        assert_eq!((q.pool(), q.kept(), q.width()), (4, 512, 2051));
    }

    #[test]
    fn positions_to_2050_keep_every_token_and_2051_drops_a_pool() {
        let q = q38();
        assert_eq!(q.dense_counts(), 2051);
        // Count 2051 is position 2,050: 512 complete pools and a tail of 3.
        assert!(!q.selects(2051) && q.selects(2052));
        for count in 1..=2051 {
            assert!(!q.selects(count), "count {count}");
            assert_eq!(q.list_len(count), count, "count {count}");
            assert_eq!(select(q, count, &[]), (0..count as u32).collect::<Vec<_>>());
        }
        // Past it the list is 512 pools and the tail: 2,048 + count % 4.
        for (count, len) in [
            (2052, 2048),
            (2053, 2049),
            (2054, 2050),
            (2055, 2051),
            (2056, 2048),
        ] {
            assert!(q.selects(count));
            assert_eq!(q.list_len(count), len, "count {count}");
        }
        for count in 2052..20_000 {
            assert!(q.list_len(count) <= q.width());
        }
    }

    #[test]
    fn scored_is_the_selecting_rows_with_an_accepted_count() {
        let q = q38();
        let ctx = 4096;
        assert!(!q.scored(0, ctx));
        assert!(!q.scored(2051, ctx));
        assert!(q.scored(2052, ctx));
        assert!(q.scored(ctx, ctx));
        assert!(!q.scored(ctx + 1, ctx));
    }

    #[test]
    fn every_pool_is_completed_by_exactly_one_count() {
        let q = q38();
        let n = 4099;
        let mut seen = vec![0u32; n / 4];
        for count in 0..=n {
            if let Some(j) = q.completes(count) {
                assert_eq!((j + 1) * 4, count);
                seen[j] += 1;
            }
        }
        assert!(seen.iter().all(|&s| s == 1));
        assert_eq!(
            (q.completes(4), q.completes(5), q.completes(8)),
            (Some(0), None, Some(1))
        );
        assert_eq!(q.completes(0), None);
    }

    #[test]
    fn order_key_orders_as_the_scores_and_ties_the_zeros() {
        let v = [
            f32::NEG_INFINITY,
            -3.5,
            -f32::MIN_POSITIVE,
            0.0,
            f32::from_bits(1),
            1.0,
            3.0e38,
            f32::INFINITY,
        ];
        for w in v.windows(2) {
            assert!(order_key(w[0]) < order_key(w[1]), "{} vs {}", w[0], w[1]);
        }
        assert_eq!(order_key(-0.0), order_key(0.0));
        assert!(order_key(f32::NAN) > order_key(f32::INFINITY));
    }

    #[test]
    fn the_cut_takes_the_best_and_the_lower_pool_of_a_tie() {
        // Pools 1, 4 and 6 tie at the cut; two of them fit: 1 and 4.
        let s = [0.5, 2.0, 9.0, 0.25, 2.0, 7.0, 2.0, 0.0];
        assert_eq!(select_pools(&s, 4), vec![1, 2, 4, 5]);
        // relu zeros of either sign tie too; the lowest three win.
        let z = [0.0, -0.0, 0.0, -0.0, 0.0];
        assert_eq!(select_pools(&z, 3), vec![0, 1, 2]);
        // k at or past the pools keeps them all.
        assert_eq!(select_pools(&s, 8), (0..8).collect::<Vec<_>>());
        assert_eq!(select_pools(&s, 20), (0..8).collect::<Vec<_>>());
    }

    #[test]
    fn a_selecting_row_lists_its_kept_pools_then_its_tail() {
        let q = Qsa::new(8, 4).expect("two pools of four");
        // Count 14: pools 0, 1, 2 complete, tail {12, 13}; pool 1 is dropped.
        let list = select(q, 14, &[3.0, 1.0, 3.0]);
        assert_eq!(list, vec![0, 1, 2, 3, 8, 9, 10, 11, 12, 13]);
        // The tie goes to the lower pool: 0 and 1 over 2.
        let list = select(q, 12, &[1.0, 1.0, 1.0]);
        assert_eq!(list, vec![0, 1, 2, 3, 4, 5, 6, 7]);
        for count in 1..200 {
            let scores: Vec<f32> = (0..count / 4).map(|j| ((j * 37) % 11) as f32).collect();
            let list = select(q, count, &scores);
            assert_eq!(list.len(), q.list_len(count));
            assert!(list.windows(2).all(|w| w[0] < w[1]));
            assert!(list.iter().all(|&t| (t as usize) < count));
            let tail = count - count % 4;
            assert!((tail..count).all(|t| list.contains(&(t as u32))));
        }
    }

    #[test]
    fn the_higher_tie_takes_the_best_and_the_higher_pool_of_a_tie() {
        // Pools 1, 4 and 6 tie at the cut; two of them fit: 4 and 6.
        let s = [0.5, 2.0, 9.0, 0.25, 2.0, 7.0, 2.0, 0.0];
        assert_eq!(select_pools_by(&s, 4, Tie::Higher), vec![2, 4, 5, 6]);
        // relu zeros of either sign tie: the highest two win, whatever sign
        // each zero carries.
        let z = [-0.0, 0.0, -0.0, 0.0, -0.0];
        assert_eq!(select_pools_by(&z, 2, Tie::Higher), vec![3, 4]);
        assert_eq!(select_pools_by(&z, 2, Tie::Lower), vec![0, 1]);
        // k at or past the pools keeps them all.
        assert_eq!(
            select_pools_by(&s, 8, Tie::Higher),
            (0..8).collect::<Vec<_>>()
        );
        // A row keeps its higher tied pools and then its tail.
        let q = Qsa::new(8, 4).expect("two pools of four");
        assert_eq!(
            select_by(q, 14, &[1.0, 1.0, 1.0], Tie::Higher),
            vec![4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        );
        // Below the cut the tie rule changes nothing: every token.
        assert_eq!(
            select_by(q, 11, &[], Tie::Higher),
            (0..11).collect::<Vec<_>>()
        );
    }

    /// The cut ik and mainline make (ik `build_qwen4exp.cpp:387,445`, mainline
    /// `qwen4exp.cpp:681-684`): the top `min(count, top_k + pool − 1)` visible
    /// cells, the tail's first (their 1e9 bias), each complete pool's cells at
    /// its score. ggml leaves the order among equal cells open; here it is
    /// cell order.
    fn port_cells(q: Qsa, count: usize, scores: &[f32]) -> Vec<u32> {
        let tail0 = q.complete(count) * q.pool();
        let mut cells: Vec<usize> = (0..count).collect();
        cells.sort_by_key(|&c| {
            let key = if c >= tail0 {
                u32::MAX
            } else {
                order_key(scores[c / q.pool()])
            };
            (std::cmp::Reverse(key), c)
        });
        let mut out: Vec<u32> = cells
            .into_iter()
            .take(count.min(q.width()))
            .map(|c| c as u32)
            .collect();
        out.sort_unstable();
        out
    }

    #[test]
    fn the_ports_take_three_minus_tail_cells_of_the_next_pool_and_we_do_not() {
        let q = q38();
        for count in 1..2200usize {
            let nb = q.complete(count);
            // Distinct scores, so the pool ranked 513th is one pool.
            let scores: Vec<f32> = (0..nb).map(|j| ((j * 7919) % 4099) as f32).collect();
            let ours = select(q, count, &scores);
            let port = port_cells(q, count, &scores);
            if !q.selects(count) || q.tail(count) == q.pool() - 1 {
                assert_eq!(ours, port, "count {count}");
                continue;
            }
            // Past position 2,050 with (p + 1) % 4 != 3: the ports keep every
            // token we keep and 3 − t more, all of the best pool we drop.
            assert!(ours.iter().all(|t| port.contains(t)), "count {count}");
            let extra: Vec<u32> = port.iter().copied().filter(|t| !ours.contains(t)).collect();
            assert_eq!(extra.len(), q.pool() - 1 - q.tail(count), "count {count}");
            let mut order: Vec<usize> = (0..nb).collect();
            order.sort_by_key(|&j| (std::cmp::Reverse(order_key(scores[j])), j));
            let next = order[q.kept()];
            assert!(
                extra.iter().all(|&t| t as usize / q.pool() == next),
                "count {count}: {extra:?} not all of pool {next}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "complete below count")]
    fn a_pool_the_row_does_not_see_complete_is_refused() {
        let _ = token_list(q38(), 7, &[1]);
    }
}
