//! A router's flips against ik's, the rule the end-to-end gates excuse them
//! by, and the rule a head's tie is excused by.
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
}
