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
}

/// ik's own margin over `values` (one token's row) and its chosen `ids`: the
/// lowest value it kept less the highest it left.
#[must_use]
pub fn margin(values: &[f32], ids: &[i32]) -> f64 {
    let min_in = ids
        .iter()
        .map(|&e| values.get(e as usize).map_or(f64::NAN, |&v| f64::from(v)))
        .fold(f64::INFINITY, f64::min);
    let max_out = (0..values.len())
        .filter(|&e| !ids.contains(&(e as i32)))
        .map(|e| f64::from(values[e]))
        .fold(f64::NEG_INFINITY, f64::max);
    min_in - max_out
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

/// The RMS of a row.
fn rms(v: &[f32]) -> f64 {
    (v.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>() / v.len().max(1) as f64).sqrt()
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

/// The most error two of a row's values may carry at an excused flip: each value moves by about
/// `band` times the row's RMS (a dot of an input carrying `band` of relative error), three
/// deviations each.
#[must_use]
pub fn flip_cap(band: f64, row: &[f32]) -> f64 {
    6.0 * band * rms(row)
}

/// The widest gap between two of a row's values the errors can cross: `band` times the row's
/// spread, three deviations each.
#[must_use]
pub fn margin_cap(band: f64, row: &[f32]) -> f64 {
    6.0 * band * spread(row)
}

/// How far below ik's top logit ik's own row puts our argmax `ours`: `max(ik_row) − ik_row[ours]`,
/// 0 when `ours` is ik's argmax. NaN for an id past the row and for a row with a value that is
/// not finite, which no rule allows.
#[must_use]
pub fn head_gap(ik_row: &[f32], ours: u32) -> f64 {
    if ik_row.iter().any(|v| !v.is_finite()) {
        return f64::NAN;
    }
    let top = ik_row
        .iter()
        .fold(f64::NEG_INFINITY, |m, &v| m.max(f64::from(v)));
    ik_row
        .get(ours as usize)
        .map_or(f64::NAN, |&v| top - f64::from(v))
}

/// The widest [`head_gap`] a head's rounding excuses: `6 · band ·` [`rms`] of ik's row, the
/// row's logits moving by about `band` times their RMS (the head is a linear map after a norm
/// whose input carries `band` of relative error), three deviations each for our logit and
/// ik's.
#[must_use]
pub fn head_cap(band: f64, ik_row: &[f32]) -> f64 {
    flip_cap(band, ik_row)
}

/// A head's argmax excused as a tie band: ours is ik's argmax, or ik's own logit at our argmax
/// lies within [`head_cap`] of ik's top ([`head_gap`]). Unlike [`tie_allowed`] it asks for no
/// runner-up: where ik's row is near flat (random weights), our argmax is ik's third or lower
/// at a rate the runner-up rule cannot hold, and the band is what the error model gives. The
/// bound is inclusive. A NaN band, a row that is not finite or an id past the row is not a tie.
#[must_use]
pub fn head_tie(ik_row: &[f32], ours: u32, band: f64) -> bool {
    head_gap(ik_row, ours) <= head_cap(band, ik_row)
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

    /// The head's tie band on a unit-RMS row (band 0.0716, cap 6 · band · 1 = 0.43): ours is ik's
    /// argmax, or its lag below ik's top is within the cap, bound included. The cases are the
    /// two ends of the band's simulation (ik's row of iid N(0, 1) logits, ours moved by iid
    /// N(0, s²)): at s = 0.05, within the band, our argmax lags by at most 6·s = 0.30 and holds;
    /// at s = 0.15, past it, a lag of 0.60 is a false fail. Mutant: `<` for `<=`, or the cap's
    /// `6` moved.
    #[test]
    fn a_head_tie_is_held_within_the_band_and_fails_past_it() {
        let band = 0.0716;
        let mut ik = row(4096, 1.0, 0.0);
        ik[10] = 4.976;
        ik[11] = 4.976 - 0.30;
        ik[12] = 4.976 - 0.60;
        let cap = head_cap(band, &ik);
        assert!(
            (0.40..0.46).contains(&cap),
            "cap {cap}: the unit-RMS row's 0.43"
        );
        assert_eq!(head_gap(&ik, 10), 0.0);
        assert!((head_gap(&ik, 11) - 0.30).abs() < 1e-6);
        assert!(head_tie(&ik, 10, band), "ik's own argmax");
        assert!(
            head_tie(&ik, 11, band),
            "s = 0.05 <= band: a lag of 6 s = 0.30 holds"
        );
        assert!(
            !head_tie(&ik, 12, band),
            "s = 0.15 > band: a lag of 0.60 is a false fail"
        );

        // The bound itself: a row whose RMS is 1, the lag 2 = 6 · (1/3) · 1.
        let edge = [2.0f32, 0.0, 0.0, 0.0];
        let third = 1.0 / 3.0;
        assert_eq!(head_cap(third, &edge), 2.0);
        assert_eq!(head_gap(&edge, 1), 2.0);
        assert!(head_tie(&edge, 1, third), "the cap is inclusive");
        assert!(
            !head_tie(&edge, 1, third * 0.999),
            "a hair under the cap is not"
        );
    }

    /// What no rule allows is not a tie, whatever the band: an id past the row, a row with a
    /// value that is not finite, an empty row and a NaN band.
    /// Mutant: a NaN read as a pass.
    #[test]
    fn a_head_tie_needs_a_finite_row_and_band() {
        let ik = [1.0f32, 0.5, 0.25];
        assert!(head_tie(&ik, 1, 1.0));
        assert!(!head_tie(&ik, 3, 1.0), "past the row");
        assert!(!head_tie(&[1.0, f32::NAN, 0.0], 0, 1.0), "NaN in the row");
        assert!(
            !head_tie(&[1.0, f32::INFINITY, 0.0], 0, 1.0),
            "inf in the row"
        );
        assert!(!head_tie(&[], 0, 1.0), "an empty row");
        assert!(!head_tie(&ik, 0, f64::NAN), "NaN band");
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
}
