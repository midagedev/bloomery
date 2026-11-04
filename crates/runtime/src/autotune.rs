//! The load-time pick among the bit-equal ways to run one kernel signature, as a pure function of
//! their timings ([`winner`]), and the cost-store key a pick is kept under ([`key`]). The card
//! side times the ways and launches the pick (`bloomery_gpu::autotune`).
//!
//! The way today's rule takes is the incumbent. Every way runs [`REPEATS`] times, the ways
//! interleaved in each repeat, and a challenger replaces the incumbent only when it is faster on
//! every repeat and its median gain over the incumbent's median exceeds the pair's repeat
//! spread, the larger of the two ways' `(max − min) / median`. Among the challengers that clear
//! it, the least median wins; with none, the incumbent stays. The ways give the same bits, so a
//! pick changes the time of a launch and never its output.

/// The timings each way takes.
pub const REPEATS: usize = 5;

/// One way's per-launch times [µs], one a repeat.
pub type Times = [f64; REPEATS];

/// The median of `t`.
#[must_use]
pub fn median(t: &Times) -> f64 {
    let mut s = *t;
    s.sort_by(f64::total_cmp);
    s[REPEATS / 2]
}

fn spread(t: &Times) -> f64 {
    let (lo, hi) = t.iter().fold((f64::INFINITY, 0.0f64), |(lo, hi), &v| {
        (lo.min(v), hi.max(v))
    });
    (hi - lo) / median(t)
}

/// The index of the way `ways` picks, `ways[0]` the incumbent (module doc); its median [µs].
///
/// # Panics
///
/// By name on no incumbent, or a time that is not finite and positive: a timing the clock
/// could not make, which a median would hide.
#[must_use]
pub fn winner(ways: &[Times]) -> (usize, f64) {
    assert!(
        !ways.is_empty() && ways.iter().flatten().all(|v| v.is_finite() && *v > 0.0),
        "autotune::winner: no incumbent, or a time that is not finite and positive: {ways:?}"
    );
    let rule = &ways[0];
    let base = median(rule);
    let mut best = (0, base);
    for (i, t) in ways.iter().enumerate().skip(1) {
        if !t.iter().zip(rule).all(|(c, r)| c < r) {
            continue;
        }
        let m = median(t);
        if (base - m) / base > spread(rule).max(spread(t)) && m < best.1 {
            best = (i, m);
        }
    }
    best
}

/// The cost-store key of a pick: the card and driver (`device`), the build that holds the
/// kernels, and the kernel's signature.
#[must_use]
pub fn key(device: &str, build: &str, kernel: &str, signature: &str) -> String {
    format!("{device} build={build} autotune kernel={kernel} {signature}")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A challenger faster on every repeat by more than the spread wins; one that loses a
    /// repeat, or wins by less than either way's spread, leaves the incumbent; the faster of two
    /// that clear it wins; a time the clock could not make, in any way, and no incumbent are
    /// refused by name (mutants: the median gain alone, the challenger's spread alone, the
    /// first clearing challenger, an invalid time a median hides or a skip drops).
    #[test]
    fn a_way_replaces_the_rule_only_by_the_margin_on_every_repeat() {
        let rule = [10.0, 10.1, 10.0, 10.2, 10.0];
        assert_eq!(winner(&[rule, [9.0, 9.1, 9.0, 9.0, 9.1]]), (1, 9.0));
        assert_eq!(winner(&[rule]).0, 0);
        assert_eq!(
            winner(&[rule, [9.0, 9.1, 9.0, 10.3, 9.1]]).0,
            0,
            "one lost repeat"
        );
        assert_eq!(
            winner(&[rule, [9.9, 9.9, 9.9, 9.9, 9.9]]).0,
            0,
            "inside the rule's spread"
        );
        let wide = [7.0, 9.95, 9.0, 9.0, 9.0];
        assert_eq!(winner(&[rule, wide]).0, 0, "inside its own spread");
        let (slow, fast) = ([9.0, 9.0, 9.0, 9.0, 9.0], [8.0, 8.0, 8.0, 8.0, 8.0]);
        assert_eq!(winner(&[rule, slow, fast]).0, 2);
        assert_eq!(winner(&[rule, fast, slow]).0, 1);
        let nan = [f64::NAN, 1.0, 1.0, 1.0, 1.0];
        let zero = [0.0; REPEATS];
        for ways in [
            vec![],
            vec![rule, nan],
            vec![nan, fast],
            vec![zero, fast],
            vec![rule, fast, [-1.0; REPEATS]],
        ] {
            let e =
                std::panic::catch_unwind(|| winner(&ways)).expect_err("an invalid time is refused");
            let why = e.downcast_ref::<String>().expect("a formatted panic");
            assert!(why.starts_with("autotune::winner: "), "{why}");
        }
    }
}
