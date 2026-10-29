//! A router's flips against ik's, the rule the end-to-end gates excuse them
//! by, and the rules a head's tie is excused by.
//!
//! A flip is a token whose chosen experts differ from ik's. Each exchanged
//! pair — `a` ours only, `b` ik's only — carries ik's gap `ik[b] − ik[a]`
//! in the values the pick ranks by, and our error there,
//! `|ours[a] − ik[a]| + |ours[b] − ik[b]|`. A pair whose gap lies within its error is a
//! rounding flip: our ranking differs from ik's by no more than our values'
//! distance from ik's. That rule alone excuses itself once the error is
//! large — a broken chain moves every value and every flip reads as
//! rounding — so a flip is allowed only while each pair's error also sits
//! under a cap the gate pins from what a correct chain reaches.
//!
//! The rules every backend's gate reads — the margin, the top-set check and
//! the head's tie band — are the reference sets' (`refset::flip`), re-exported
//! here under the path the card's gates read them at.

use crate::GateError;
pub use refset::flip::{head_cap, head_gap, head_tie, margin, picks_its_top};

/// A token whose chosen set differs from ik's, with every exchanged pair.
#[derive(Clone, Debug)]
pub struct Flip {
    pub layer: usize,
    pub token: usize,
    /// ik's own margin at the token ([`margin`]).
    pub margin: f64,
    /// `(a, b, gap, err)`: `a` ours only, `b` ik's only, ik's gap and our
    /// two values' error (module doc).
    pub pairs: Vec<(u32, u32, f64, f64)>,
}

impl Flip {
    /// The flip at `(layer, token)` when our ids `ours_ids` (ranked by
    /// `ours`) and ik's ids `ik_ids` (ranked by `ik`, whose margin is
    /// `margin`) are not one set; `None` when they are. An id past a value
    /// row reads NaN, which no rule allows.
    #[must_use]
    pub fn between(
        (layer, token): (usize, usize),
        (ours_ids, ours): (&[u32], &[f32]),
        (ik_ids, ik): (&[i32], &[f32]),
        margin: f64,
    ) -> Option<Flip> {
        let only_ours: Vec<u32> = ours_ids
            .iter()
            .copied()
            .filter(|&e| !ik_ids.contains(&(e as i32)))
            .collect();
        let only_ik: Vec<u32> = ik_ids
            .iter()
            .map(|&e| e as u32)
            .filter(|e| !ours_ids.contains(e))
            .collect();
        if only_ours.is_empty() && only_ik.is_empty() {
            return None;
        }
        let v = |x: &[f32], e: u32| x.get(e as usize).map_or(f64::NAN, |&y| f64::from(y));
        let mut pairs = Vec::new();
        for &a in &only_ours {
            for &b in &only_ik {
                let gap = v(ik, b) - v(ik, a);
                let err = (v(ours, a) - v(ik, a)).abs() + (v(ours, b) - v(ik, b)).abs();
                pairs.push((a, b, gap, err));
            }
        }
        Some(Flip {
            layer,
            token,
            margin,
            pairs,
        })
    }

    /// Allowed only when every pair's gap lies within its error and the
    /// error within `cap`: our ranking then differs from ik's by no more
    /// than our values' distance from ik's, a distance a correct chain stays
    /// under. Past either the pick is wrong, named by [`Flip::line`].
    #[must_use]
    pub fn allowed(&self, cap: f64) -> bool {
        !self.pairs.is_empty()
            && self
                .pairs
                .iter()
                .all(|&(_, _, gap, err)| gap <= err && err <= cap)
    }

    /// The flip's line: its pairs and its verdict under `cap`, `arm` the
    /// run it came from.
    #[must_use]
    pub fn line(&self, arm: &str, cap: f64) -> String {
        let pairs: Vec<String> = self
            .pairs
            .iter()
            .map(|(a, b, gap, err)| format!("{a}<-{b} gap {gap:.3e} err {err:.3e}"))
            .collect();
        let verdict = if self.allowed(cap) {
            "allowed (counted)"
        } else if self
            .pairs
            .iter()
            .any(|&(_, _, _, err)| err > cap || err.is_nan())
        {
            "FAIL: our error past the pinned frontier"
        } else {
            "FAIL: a pair's gap past our error"
        };
        format!(
            "{arm} flip layer={} token={}: ik margin {:.3e}; {} (cap {cap:.2}): {verdict}",
            self.layer,
            self.token,
            self.margin,
            pairs.join(", "),
        )
    }

    /// Each pair's error with our values read from `undone` (`|undone[a] − ik[a]| + |undone[b] −
    /// ik[b]|`, in the pairs' order). An id past a row reads NaN.
    fn errs_in(&self, undone: &[f32], ik: &[f32]) -> Vec<f64> {
        let v = |x: &[f32], e: u32| x.get(e as usize).map_or(f64::NAN, |&y| f64::from(y));
        self.pairs
            .iter()
            .map(|&(a, b, _, _)| (v(undone, a) - v(ik, a)).abs() + (v(undone, b) - v(ik, b)).abs())
            .collect()
    }

    /// PIN(2026-10-29): the rule for a flip past an earlier flip on its path, derived. There our
    /// values differ from ik's by our rounding, which [`Flip::allowed`]'s cap is derived for (a
    /// chain that takes ik's experts), plus the earlier flip's cascade: the expert it exchanged
    /// moves every later router input on its path by that expert's routed output less the one it
    /// replaced, a term of the weights that no rounding model bounds. `undone` is our router's
    /// row at this flip's (layer, token) from the same run with ik's routes taken at every layer,
    /// so its distance from ik's is the rounding alone. A pair is allowed only when its gap lies
    /// within our error and its error in `undone` within `cap`; past either the pick is wrong,
    /// named by [`Flip::line_past`].
    #[must_use]
    pub fn allowed_past(&self, cap: f64, (undone, ik): (&[f32], &[f32])) -> bool {
        !self.pairs.is_empty()
            && self
                .pairs
                .iter()
                .zip(self.errs_in(undone, ik))
                .all(|(&(_, _, gap, err), held)| gap <= err && held <= cap)
    }

    /// The line of a flip past an earlier flip on its path: its pairs, each with its error in
    /// `undone`, and its verdict under `cap` ([`Flip::allowed_past`]).
    #[must_use]
    pub fn line_past(&self, arm: &str, cap: f64, (undone, ik): (&[f32], &[f32])) -> String {
        let held = self.errs_in(undone, ik);
        let pairs: Vec<String> = self
            .pairs
            .iter()
            .zip(&held)
            .map(|((a, b, gap, err), h)| {
                format!("{a}<-{b} gap {gap:.3e} err {err:.3e} undone {h:.3e}")
            })
            .collect();
        let verdict = if self.allowed_past(cap, (undone, ik)) {
            "allowed (counted)"
        } else if held.iter().any(|&h| h > cap || h.is_nan()) {
            "FAIL: our error with the earlier flips undone past the pinned frontier"
        } else {
            "FAIL: a pair's gap past our error"
        };
        format!(
            "{arm} flip layer={} token={}: ik margin {:.3e}; {} (cap {cap:.2}): {verdict}",
            self.layer,
            self.token,
            self.margin,
            pairs.join(", "),
        )
    }
}

/// A head's argmax that differs from ik's, excused as a tie: ours is ik's
/// runner-up, ik's own margin between the two lies within twice our
/// distance from ik's logits at those ids, and our whole logits row lies
/// within `ratio` times the relative distance our head's input (the last
/// layer's streams) has from ik's. The head is a linear map after a norm,
/// so the row carries about its input's distance; a row further off than
/// that is not rounding carried through the head.
#[must_use]
pub fn tie_allowed(
    (top, ik_top, ik_runner_up): (u32, u32, u32),
    (margin, dist): (f64, f64),
    (logits_rel, input_rel): (f64, f64),
    ratio: f64,
) -> bool {
    top != ik_top && top == ik_runner_up && margin <= 2.0 * dist && logits_rel <= ratio * input_rel
}

/// The deviation of a row about its mean: the spread a ranking reads (a common offset moves no
/// rank).
fn spread(v: &[f32]) -> f64 {
    let n = v.len().max(1) as f64;
    let mean = v.iter().map(|&x| f64::from(x)).sum::<f64>() / n;
    (v.iter()
        .map(|&x| (f64::from(x) - mean).powi(2))
        .sum::<f64>()
        / n)
        .sqrt()
}

/// The most error two of a row's values may carry at an excused flip: [`head_cap`]'s bound, each
/// value moving by about `band` times the row's RMS (a dot of an input carrying `band` of relative
/// error), three deviations each.
#[must_use]
pub fn flip_cap(band: f64, row: &[f32]) -> f64 {
    head_cap(band, row)
}

/// The widest gap between two of a row's values the errors can cross: `band` times the row's
/// spread, three deviations each.
#[must_use]
pub fn margin_cap(band: f64, row: &[f32]) -> f64 {
    6.0 * band * spread(row)
}

/// A head row's own margin: its top logit less its runner-up, the quantity a greedy step's near
/// tie is read at. A value that is not finite and a row shorter than two are refused by name —
/// over either the margin is not a number. The top two are [`crate::kld::top2`]'s, whose ties
/// order by id; the margin reads only the values, and a tie reads 0.
pub fn head_margin(row: &[f32]) -> Result<f32, GateError> {
    if row.len() < 2 {
        return Err(format!("{} logits, a top-1 margin needs two", row.len()).into());
    }
    for (i, &v) in row.iter().enumerate() {
        if !v.is_finite() {
            return Err(format!("logit {i} is {v}").into());
        }
    }
    let (_, a, _, b) = crate::kld::top2(row);
    Ok(a - b)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A pair within its error is allowed under the cap and refused past
    /// it; a pair whose gap passes its error is refused either way; the
    /// same set is no flip.
    #[test]
    fn a_flip_is_allowed_only_under_the_cap() {
        let ik = [0.50f32, 0.49, 0.10];
        let ours = [0.48f32, 0.51, 0.10];
        let f =
            Flip::between((3, 1), (&[1], &ours), (&[0], &ik), margin(&ik, &[0])).expect("a flip");
        assert_eq!(f.pairs.len(), 1);
        let (_, _, gap, err) = f.pairs[0];
        assert!(
            (gap - 0.01).abs() < 1e-6 && (err - 0.04).abs() < 1e-6,
            "{f:?}"
        );
        assert!(f.allowed(0.1));
        assert!(!f.allowed(0.03));
        assert!(f.line("free", 0.03).contains("past the pinned frontier"));
        let far = [0.30f32, 0.49, 0.10];
        let g = Flip::between((3, 1), (&[1], &far), (&[0], &ik), 0.0).expect("a flip");
        assert!(g.allowed(1.0), "gap 0.01 within err 0.2: {g:?}");
        let wrong = Flip::between((3, 1), (&[2], &ours), (&[0], &ik), 0.0).expect("a flip");
        assert!(!wrong.allowed(10.0), "gap 0.40 past err 0.02: {wrong:?}");
        assert!(wrong.line("free", 10.0).contains("gap past our error"));
        assert!(Flip::between((3, 1), (&[0], &ours), (&[0], &ik), 0.0).is_none());
    }

    /// Past an earlier flip the cap holds the pair's error in the undone row, not ours: a pair at
    /// err 0.16 (a cascade the cap on our error refuses at 0.1) is allowed over an undone row at
    /// 0.02 and refused over one at 0.61 (a router moved by order one); a pair whose gap passes
    /// its error, and an undone row too short for an id, are refused whatever the cap. Mutant:
    /// the cap read on our error, or the gap test dropped.
    #[test]
    fn a_flip_past_another_is_capped_with_it_undone() {
        let ik = [0.50f32, 0.49, 0.10];
        let ours = [0.40f32, 0.55, 0.10];
        let f =
            Flip::between((2, 0), (&[1], &ours), (&[0], &ik), margin(&ik, &[0])).expect("a flip");
        let (_, _, gap, err) = f.pairs[0];
        assert!(
            (gap - 0.01).abs() < 1e-6 && (err - 0.16).abs() < 1e-6,
            "{f:?}"
        );
        assert!(!f.allowed(0.1), "the cap on our error: {f:?}");
        let rounding = [0.49f32, 0.50, 0.10];
        assert!(f.allowed_past(0.1, (&rounding, &ik)));
        assert!(
            f.line_past("free", 0.1, (&rounding, &ik))
                .contains("undone 2.000e-2")
        );
        let fault = [0.20f32, 0.80, 0.10];
        assert!(!f.allowed_past(0.1, (&fault, &ik)));
        assert!(
            f.line_past("free", 0.1, (&fault, &ik))
                .contains("undone past the pinned frontier")
        );
        let wrong = Flip::between((2, 0), (&[2], &ours), (&[0], &ik), 0.0).expect("a flip");
        assert!(
            !wrong.allowed_past(10.0, (&ik, &ik)),
            "gap 0.40 past err 0.10: {wrong:?}"
        );
        assert!(
            wrong
                .line_past("free", 10.0, (&ik, &ik))
                .contains("gap past our error")
        );
        assert!(
            !f.allowed_past(10.0, (&rounding[..1], &ik)),
            "id 1 past the row reads NaN"
        );
    }

    /// The tie is judged against the head input's own distance, not a band
    /// for other layers: a row at 0.12 of ik's passes over an input at 0.12
    /// and fails over one at 0.05.
    #[test]
    fn a_tie_is_bounded_by_the_head_input() {
        let ids = (7, 5, 7);
        assert!(tie_allowed(ids, (0.1, 0.06), (0.12, 0.12), 1.5));
        assert!(!tie_allowed(ids, (0.1, 0.06), (0.12, 0.05), 1.5));
        assert!(!tie_allowed((7, 5, 9), (0.1, 0.06), (0.01, 0.12), 1.5));
        assert!(!tie_allowed(ids, (0.2, 0.06), (0.01, 0.12), 1.5));
    }

    /// A row of `n` values of a standard normal (the sum of twelve uniforms of a fixed LCG, less
    /// six), scaled by `scale` and moved by `offset`.
    fn row(n: usize, scale: f64, offset: f64) -> Vec<f32> {
        let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut uniform = move || {
            x = x
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (x >> 11) as f64 / (1u64 << 53) as f64
        };
        (0..n)
            .map(|_| {
                let z: f64 = (0..12).map(|_| uniform()).sum::<f64>() - 6.0;
                (offset + scale * z) as f32
            })
            .collect()
    }

    // The five copies the e2e gates carry, as they read when they were lifted: `gate_glm5next_mtp`'s
    // `margin_cap` over its `spread`; `gate_qwen4exp_e2e`'s `flip_cap` (the RMS inline) and
    // `margin_cap` over its `spread`; `gate_qwen4exp_mtp`'s `flip_cap` over its `rms` and
    // `margin_cap` over its `spread`.

    fn copy_spread(v: &[f32]) -> f64 {
        let n = v.len().max(1) as f64;
        let mean = v.iter().map(|&x| f64::from(x)).sum::<f64>() / n;
        (v.iter()
            .map(|&x| (f64::from(x) - mean).powi(2))
            .sum::<f64>()
            / n)
            .sqrt()
    }

    fn copy_margin_cap(band: f64, v: &[f32]) -> f64 {
        6.0 * band * copy_spread(v)
    }

    fn copy_flip_cap_inline(band: f64, logits: &[f32]) -> f64 {
        let rms = (logits.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>()
            / logits.len().max(1) as f64)
            .sqrt();
        6.0 * band * rms
    }

    fn copy_rms(v: &[f32]) -> f64 {
        (v.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>() / v.len().max(1) as f64).sqrt()
    }

    fn copy_flip_cap(band: f64, v: &[f32]) -> f64 {
        6.0 * band * copy_rms(v)
    }

    /// The lifted `flip_cap` and `margin_cap` are the copies they were lifted from, bit for bit
    /// (the expected functions above are the copies' bodies), on a hand row, a row far from zero
    /// (where the spread and the RMS differ), 4,096 normal values and an empty row. Mutant: the
    /// 6, the mean, or the divisor moved.
    #[test]
    fn the_lifted_caps_are_the_copies() {
        let rows = [
            vec![0.5f32, -1.5, 2.0, 0.25],
            vec![100.5f32, 99.5, 101.0, 98.0, 100.0],
            row(4096, 0.7, 0.3),
            Vec::new(),
        ];
        for v in &rows {
            for band in [0.0, 0.0716, 0.1755, 1.0] {
                assert_eq!(
                    flip_cap(band, v).to_bits(),
                    copy_flip_cap_inline(band, v).to_bits(),
                    "flip_cap, the e2e copy"
                );
                assert_eq!(
                    flip_cap(band, v).to_bits(),
                    copy_flip_cap(band, v).to_bits(),
                    "flip_cap, the mtp copy"
                );
                assert_eq!(
                    margin_cap(band, v).to_bits(),
                    copy_margin_cap(band, v).to_bits(),
                    "margin_cap"
                );
            }
        }
        assert_ne!(
            flip_cap(1.0, &rows[1]).to_bits(),
            margin_cap(1.0, &rows[1]).to_bits(),
            "the two caps read the RMS and the spread: a row of one offset tells them apart"
        );
    }

    /// `head_margin` on fixtures: a hand row's top logit less its runner-up (2.0 − 1.25), a tie
    /// holding a ±0 pair at 0, and the fixed generator's 4,096-normal row at its derived bits; a
    /// row with a value that is not finite and a row shorter than two are errors by name, never
    /// a silent NaN, infinity or 0. Mutant: the subtraction swapped (the runner-up less the
    /// top), or a refusal dropped.
    #[test]
    fn head_margin_matches_fixtures_and_names_invalid_rows() {
        for (v, want) in [
            (vec![0.5f32, 2.0, -1.0, 1.25], 0.75f32.to_bits()),
            (vec![-0.0, 0.0, 3.0, 3.0], 0.0f32.to_bits()),
            // Expected bits [derived] for this fixed-generator input.
            (row(4096, 0.7, 0.3), 0x3a9d_9000),
        ] {
            assert_eq!(head_margin(&v).unwrap().to_bits(), want);
        }
        for (v, want) in [
            (vec![1.0, f32::NAN, 2.0], "logit 1 is NaN"),
            (vec![0.0, f32::INFINITY], "logit 1 is inf"),
            (vec![], "0 logits, a top-1 margin needs two"),
            (vec![1.0], "1 logits, a top-1 margin needs two"),
        ] {
            assert_eq!(head_margin(&v).unwrap_err().to_string(), want);
        }
    }
}
