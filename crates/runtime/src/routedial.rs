//! The router dials' host rules: the resident bonus and the top-k cut of a routed decode step
//! (`BLOOMERY_ROUTE_RESIDENT_BONUS`, `BLOOMERY_ROUTE_TOPK`), in the arithmetic the device runs.
//! A gate's host side holds the device's picks and weights to these bit for bit; a launcher
//! takes its refusals ([`RouteDialError`]) from here.
//!
//! - **Bonus.** The selection value of expert `e` is `score[e] + (bias[e] + λ · r[e])`,
//!   `r[e]` 1 for an expert on the stage card, 0 otherwise. The top `k` are taken over those values,
//!   an equal value going to the larger expert id (the router's own tie rule). The weights of
//!   the picked set come from the unbiased scores ([`cut_weights`] with the full `k`).
//! - **Cut.** The router ranks `k` picks; the first `k'` run. Their weights are
//!   `score_j / divisor(Σ_{j<k'} score_j) · scale`, the sum in f64 in slot order narrowed to
//!   f32, the divisor the router instance's guard (`floor` selects Flint's floor, else the
//!   `1e-20` addend). The slots from `k'` weigh 0.

use std::fmt;

/// A dial a load or a launch refuses, by name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RouteDialError {
    /// `BLOOMERY_ROUTE_TOPK` above the model's own `k`.
    TopkAboveK { topk: usize, k: usize },
    /// A dial beside a feature whose rows would pick without it.
    Beside {
        dial: &'static str,
        other: &'static str,
    },
}

impl fmt::Display for RouteDialError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RouteDialError::TopkAboveK { topk, k } => write!(
                f,
                "BLOOMERY_ROUTE_TOPK={topk}: the model routes {k} experts a token; it takes 1 to {k}"
            ),
            RouteDialError::Beside { dial, other } => write!(
                f,
                "{dial} beside {other}: {other}'s rows pick without the dial, so the two paths \
                 would route one token differently; unset one"
            ),
        }
    }
}

impl std::error::Error for RouteDialError {}

/// The cut `topk` (`None`: the model's own `k`) as a count of the `k` slots a token routes, or the
/// refusal. A cut equal to `k` is `None`: the router's own weights, no launch.
pub fn cut_of(topk: Option<usize>, k: usize) -> Result<Option<usize>, RouteDialError> {
    match topk {
        None => Ok(None),
        Some(t) if t == 0 || t > k => Err(RouteDialError::TopkAboveK { topk: t, k }),
        Some(t) if t == k => Ok(None),
        Some(t) => Ok(Some(t)),
    }
}

/// The router's guard on the weight stage's divisor: Flint's `2^-14` floor, else the `1e-20`
/// addend (`route_core::floor_divisor` and `renorm_divisor`).
#[inline]
fn divisor(sum: f32, floor: bool) -> f32 {
    if floor {
        sum.max(1.0 / 16384.0)
    } else {
        sum + 1e-20
    }
}

/// The weights of a token's `k` picked slots when the first `cut` run: `scores` are the picks'
/// unbiased scores in slot (rank) order; slot `j < cut` weighs
/// `scores[j] / divisor(Σ_{i<cut} scores[i]) · scale`, every later slot 0. With `cut` the whole
/// `scores.len()` these are the router's own weights.
#[must_use]
pub fn cut_weights(scores: &[f32], cut: usize, scale: f32, floor: bool) -> Vec<f32> {
    let cut = cut.min(scores.len());
    let sum: f64 = scores[..cut].iter().map(|&s| f64::from(s)).sum();
    let div = divisor(sum as f32, floor);
    scores
        .iter()
        .enumerate()
        .map(|(j, &s)| if j < cut { s / div * scale } else { 0.0 })
        .collect()
}

/// The top `k` experts by selection value `scores[e] + (bias[e] + lambda · [resident[e]])`, in rank
/// order, an equal value to the larger id; a NaN value never wins.
#[must_use]
pub fn biased_picks(
    scores: &[f32],
    bias: &[f32],
    resident: &[bool],
    lambda: f32,
    k: usize,
) -> Vec<u32> {
    assert!(
        scores.len() == bias.len() && scores.len() == resident.len(),
        "one score, bias and residency per expert"
    );
    let value = |e: usize| {
        scores[e]
            + if resident[e] {
                bias[e] + lambda
            } else {
                bias[e]
            }
    };
    let mut order: Vec<usize> = (0..scores.len()).filter(|&e| !value(e).is_nan()).collect();
    order.sort_by(|&a, &b| {
        value(b)
            .partial_cmp(&value(a))
            .expect("NaN values are filtered out")
            .then(b.cmp(&a))
    });
    order.into_iter().take(k).map(|e| e as u32).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The full cut is the router's own rule; a cut renormalizes over the first `cut` slots and
    /// zeroes the rest; the weights of the run slots sum to `scale` up to the guard.
    #[test]
    fn a_cut_renormalizes_the_first_slots_and_zeroes_the_rest() {
        let s = [0.9f32, 0.6, 0.3, 0.2, 0.1, 0.05];
        let full = cut_weights(&s, 6, 1.5, false);
        let total: f32 = full.iter().sum();
        assert!((total - 1.5).abs() < 1e-5, "{total}");
        let w = cut_weights(&s, 4, 1.5, false);
        assert_eq!(&w[4..], &[0.0, 0.0]);
        let sum4 = f64::from(s[0]) + f64::from(s[1]) + f64::from(s[2]) + f64::from(s[3]);
        assert_eq!(w[0], s[0] / (sum4 as f32 + 1e-20) * 1.5);
        let kept: f32 = w.iter().sum();
        assert!((kept - 1.5).abs() < 1e-5, "{kept}");
        assert!(w[0] > full[0], "fewer slots weigh more each");
        let one = cut_weights(&s, 1, 1.5, false);
        assert_eq!(one[0], s[0] / (f64::from(s[0]) as f32 + 1e-20) * 1.5);
    }

    /// A floor guard keeps an all-underflow cut finite: 0 weights, not 0 / 0.
    #[test]
    fn a_floor_guard_weighs_an_all_zero_cut_zero() {
        let w = cut_weights(&[0.0, 0.0, 0.0], 2, 1.0, true);
        assert_eq!(w, [0.0, 0.0, 0.0]);
        let w = cut_weights(&[0.0, 0.0, 0.0], 2, 1.0, false);
        assert_eq!(w, [0.0, 0.0, 0.0]);
    }

    /// The cut refuses a count past `k` and zero by name; a cut of `k` is no cut.
    #[test]
    fn the_cut_of_a_count() {
        assert_eq!(cut_of(None, 6), Ok(None));
        assert_eq!(cut_of(Some(6), 6), Ok(None));
        assert_eq!(cut_of(Some(4), 6), Ok(Some(4)));
        assert_eq!(
            cut_of(Some(7), 6),
            Err(RouteDialError::TopkAboveK { topk: 7, k: 6 })
        );
        assert!(cut_of(Some(0), 6).is_err());
    }

    /// With λ 0 the picks are the file's; a modest λ moves a resident just under the cut line in;
    /// a λ above the whole spread makes every pick resident while `k` residents exist; ties go to
    /// the larger id.
    #[test]
    fn a_bonus_steers_the_picks_toward_the_card() {
        let n = 16usize;
        let scores: Vec<f32> = (0..n).map(|e| 3.0 - 0.1 * e as f32).collect();
        let bias = vec![0.0f32; n];
        let mut resident = vec![false; n];
        for e in [7usize, 9, 11, 12, 14, 15] {
            resident[e] = true;
        }
        let plain = biased_picks(&scores, &bias, &resident, 0.0, 6);
        assert_eq!(plain, [0, 1, 2, 3, 4, 5]);
        let off_by_zero = biased_picks(&scores, &bias, &vec![false; n], 5.0, 6);
        assert_eq!(off_by_zero, plain, "no resident: no change");
        let mild = biased_picks(&scores, &bias, &resident, 0.6, 6);
        assert!(mild.contains(&7) && mild.len() == 6, "{mild:?}");
        let big = biased_picks(&scores, &bias, &resident, 10.0, 6);
        let mut got = big.clone();
        got.sort_unstable();
        assert_eq!(got, [7, 9, 11, 12, 14, 15]);
        let tie = biased_picks(&[1.0, 1.0, 1.0], &[0.0; 3], &[false; 3], 0.0, 2);
        assert_eq!(tie, [2, 1]);
        let nan = biased_picks(&[f32::NAN, 1.0], &[0.0; 2], &[false; 2], 0.0, 2);
        assert_eq!(nan, [1]);
    }
}
