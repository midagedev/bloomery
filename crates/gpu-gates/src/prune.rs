//! The arithmetic of `lab_prune_ds41` (host only): the cut's expert count, the full-vocabulary KL
//! divergence of one position in f64, and the summary of a run's positions. The lab compares a model
//! whose coldest experts cannot be picked with the same model unpruned, both read from the card as f32
//! logits, so no base file and none of ik's 24-nat floor (`kld.rs`) stands between the two rows.

/// The experts a cut of `pct` percent removes from a layer of `n_expert`: `floor(n_expert * pct / 100 +
/// 1/2)`, in integers. `tools/ref/router-prune.py` prints the same count (`cut_count`), so a replay's
/// table and the lab's mask name the same experts.
#[must_use]
pub fn cut_count(n_expert: usize, pct: usize) -> usize {
    (n_expert * pct * 2 + 100) / 200
}

/// `ln Σ exp(x)` over a row of logits, summed in f64 from the row's maximum.
///
/// # Panics
/// On an empty row: a position's logits are never empty.
#[must_use]
pub fn lse(row: &[f32]) -> f64 {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    assert!(max.is_finite(), "a logits row with no finite maximum");
    let max = f64::from(max);
    max + row
        .iter()
        .map(|&v| (f64::from(v) - max).exp())
        .sum::<f64>()
        .ln()
}

/// KL(P‖Q) = Σ p (ln p − ln q) over the whole vocabulary, P the softmax of `p` (whose `lse` is
/// `p_lse`) and Q the softmax of `q` (whose `lse` is `q_lse`), in f64. Two rows of equal bits give
/// exactly 0.
///
/// # Panics
/// On rows of different lengths.
#[must_use]
pub fn kld(p: &[f32], p_lse: f64, q: &[f32], q_lse: f64) -> f64 {
    assert_eq!(p.len(), q.len(), "KL over two vocabularies of unequal size");
    let mut k = 0.0f64;
    for (&a, &b) in p.iter().zip(q) {
        let lp = f64::from(a) - p_lse;
        let lq = f64::from(b) - q_lse;
        k += lp.exp() * (lp - lq);
    }
    k
}

/// The negative log probability of `id` under the softmax of `row` whose `lse` is `row_lse`.
#[must_use]
pub fn nll(row: &[f32], row_lse: f64, id: u32) -> f64 {
    row_lse - f64::from(row[id as usize])
}

/// A run's per-position values: the mean with its standard error, the nearest-rank percentiles and the
/// maximum. The KL divergence is heavy-tailed, so the median and the 99th percentile stand beside the
/// mean.
#[derive(Default)]
pub struct Stat {
    values: Vec<f64>,
}

impl Stat {
    pub fn push(&mut self, v: f64) {
        self.values.push(v);
    }

    #[must_use]
    pub fn n(&self) -> usize {
        self.values.len()
    }

    /// The mean of the values; 0 for none.
    #[must_use]
    pub fn mean(&self) -> f64 {
        if self.values.is_empty() {
            return 0.0;
        }
        self.values.iter().sum::<f64>() / self.values.len() as f64
    }

    /// The standard error of the mean from the sample variance; 0 for fewer than two values.
    #[must_use]
    pub fn se(&self) -> f64 {
        let n = self.values.len();
        if n < 2 {
            return 0.0;
        }
        let m = self.mean();
        let var = self.values.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / (n - 1) as f64;
        (var / n as f64).sqrt()
    }

    /// The nearest-rank percentile `pct` (1..=100): the value at rank `ceil(pct * n / 100)` of the
    /// sorted values; 0 for none.
    #[must_use]
    pub fn percentile(&self, pct: usize) -> f64 {
        if self.values.is_empty() {
            return 0.0;
        }
        let mut sorted = self.values.clone();
        sorted.sort_by(f64::total_cmp);
        let rank = (pct * sorted.len()).div_ceil(100).clamp(1, sorted.len());
        sorted[rank - 1]
    }

    #[must_use]
    pub fn max(&self) -> f64 {
        self.values.iter().copied().fold(0.0, f64::max)
    }

    /// The mean of the values from index `from` on.
    #[must_use]
    pub fn mean_from(&self, from: usize) -> f64 {
        let tail = self.values.get(from..).unwrap_or_default();
        if tail.is_empty() {
            return 0.0;
        }
        tail.iter().sum::<f64>() / tail.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_cut_counts_are_the_replays() {
        let cuts: Vec<usize> = [25, 40, 50, 60]
            .iter()
            .map(|&x| cut_count(384, x))
            .collect();
        assert_eq!(cuts, [96, 154, 192, 230]);
        assert_eq!(cut_count(8, 25), 2);
        assert_eq!(cut_count(384, 0), 0);
    }

    #[test]
    fn two_rows_give_a_known_kl_and_equal_rows_give_zero() {
        // P = softmax(0, ln 2) = (1/3, 2/3); Q uniform. KL(P‖Q) = 1/3 ln(2/3) + 2/3 ln(4/3).
        let p = [0.0f32, std::f32::consts::LN_2];
        let q = [0.0f32, 0.0];
        let want = (1.0f64 / 3.0) * (2.0f64 / 3.0).ln() + (2.0 / 3.0) * (4.0f64 / 3.0).ln();
        let got = kld(&p, lse(&p), &q, lse(&q));
        assert!((got - want).abs() < 1e-7, "{got} against {want}");
        // a shift of a whole row leaves its softmax alone
        let shifted: Vec<f32> = q.iter().map(|v| v + 100.0).collect();
        let again = kld(&p, lse(&p), &shifted, lse(&shifted));
        assert!((again - got).abs() < 1e-12, "{again} against {got}");
        // equal bits, a long row: exactly zero
        let row: Vec<f32> = (0..1000)
            .map(|i| ((i * 37) % 101) as f32 * 0.13 - 6.0)
            .collect();
        assert_eq!(kld(&row, lse(&row), &row, lse(&row)), 0.0);
    }

    #[test]
    fn nll_is_the_log_of_the_softmax() {
        let row = [0.0f32, std::f32::consts::LN_2];
        let got = nll(&row, lse(&row), 1);
        assert!((got - (3.0f64 / 2.0).ln()).abs() < 1e-7, "{got}");
    }

    #[test]
    fn a_summary_reads_the_mean_its_error_and_the_percentiles() {
        let mut s = Stat::default();
        for v in [1.0, 2.0, 3.0, 4.0, 5.0] {
            s.push(v);
        }
        assert_eq!(s.n(), 5);
        assert!((s.mean() - 3.0).abs() < 1e-12);
        assert!((s.se() - (2.5f64 / 5.0).sqrt()).abs() < 1e-12);
        assert_eq!(s.percentile(50), 3.0);
        assert_eq!(s.percentile(99), 5.0);
        assert_eq!(s.percentile(1), 1.0);
        assert_eq!(s.max(), 5.0);
        assert!((s.mean_from(3) - 4.5).abs() < 1e-12);
        assert_eq!(Stat::default().mean(), 0.0);
        assert_eq!(Stat::default().percentile(50), 0.0);
    }
}
