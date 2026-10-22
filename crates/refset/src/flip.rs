//! The rules the end-to-end gates hold a free-running chain to, the half
//! every backend's gate reads: the relative distance a layer output is read
//! at, the router's flip test and the rows a near-tie flip carries, and the
//! head's tie band. Pure functions over ik's rows, so they live with the
//! reference sets — a gate that links no CUDA backend (the Metal session's)
//! is held to the same rule the card's gates are. The pair-exchange flip
//! analysis and the caps it derives are the CUDA gates' own
//! (`gpu-gates`'s `flip`).

/// `‖a − b‖ / ‖base‖` over the zipped values in f64, the denominator's values
/// from `base`.
pub fn rel_base(a: &[f32], b: &[f32], base: impl Fn(usize) -> f64) -> f64 {
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        num += (f64::from(x) - f64::from(y)).powi(2);
        den += base(i).powi(2);
    }
    (num / den.max(f64::MIN_POSITIVE)).sqrt()
}

/// `rel_base` of our last `ik.len()` values against ik's, on ik's own norm.
/// Infinite when ik holds more values than ours, and a NaN reads infinite,
/// so neither passes a band.
pub fn rel_to(ours: &[f32], ik: &[f32]) -> f64 {
    let Some(tail) = ours.len().checked_sub(ik.len()).map(|s| &ours[s..]) else {
        return f64::INFINITY;
    };
    worse(0.0, rel_base(tail, ik, |i| f64::from(ik[i])))
}

/// The larger of two errors, a NaN counting as infinite: `f64::max` would
/// drop it.
pub fn worse(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::INFINITY
    } else {
        a.max(b)
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

/// Whether ids are n_used distinct experts at the top of finite logits.
#[must_use]
pub fn picks_its_top(ids: &[u32], logits: &[f32], n_used: usize) -> bool {
    if ids.len() != n_used
        || logits.iter().any(|v| !v.is_finite())
        || ids
            .iter()
            .enumerate()
            .any(|(i, &e)| e as usize >= logits.len() || ids[..i].contains(&e))
    {
        return false;
    }
    let low = ids
        .iter()
        .map(|&e| logits[e as usize])
        .fold(f32::INFINITY, f32::min);
    logits
        .iter()
        .enumerate()
        .all(|(e, &v)| ids.iter().any(|&id| id as usize == e) || v <= low)
}

/// A token whose chosen set differs from ik's is a near tie when ik's own
/// margin between its eighth pick and the best expert it left lies within
/// twice our logits' largest error at that token.
pub fn router_tie(margin: f32, err: f32) -> bool {
    margin <= 2.0 * err
}

/// The RMS of a row.
pub fn rms(v: &[f32]) -> f64 {
    (v.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>() / v.len().max(1) as f64).sqrt()
}

/// How far below ik's top logit ik's own row puts our argmax `ours`:
/// `max(ik_row) − ik_row[ours]`, 0 when `ours` is ik's argmax. NaN for an id
/// past the row and for a row with a value that is not finite, which no rule
/// allows.
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

/// The widest [`head_gap`] a head's rounding excuses: `6 · band ·` [`rms`] of
/// ik's row, the row's logits moving by about `band` times their RMS (the
/// head is a linear map after a norm whose input carries `band` of relative
/// error), three deviations each for our logit and ik's.
#[must_use]
pub fn head_cap(band: f64, ik_row: &[f32]) -> f64 {
    6.0 * band * rms(ik_row)
}

/// A head's argmax excused as a tie band: ours is ik's argmax, or ik's own
/// logit at our argmax lies within [`head_cap`] of ik's top ([`head_gap`]).
/// The bound is inclusive. A NaN band, a row that is not finite or an id
/// past the row is not a tie.
#[must_use]
pub fn head_tie(ik_row: &[f32], ours: u32, band: f64) -> bool {
    head_gap(ik_row, ours) <= head_cap(band, ik_row)
}

/// A token whose picks differ from ik's at one layer.
pub struct RouteFlip {
    pub layer: usize,
    pub token: usize,
    pub ours: Vec<u32>,
    pub theirs: Vec<i32>,
    pub margin: f32,
    pub err: f32,
    pub tie: bool,
}

/// The rows `[layer][token]` a near-tie flip leaves unheld: the flipped
/// token from its layer's output on, and every later token at the deeper
/// layers (their mixers read the flipped token's state and keys); earlier
/// tokens never see it.
pub fn held_rows(n: usize, t_n: usize, flips: &[RouteFlip]) -> Vec<Vec<bool>> {
    let mut held = vec![vec![true; t_n]; n];
    for f in flips.iter().filter(|f| f.tie) {
        for row in &mut held[f.layer..] {
            row[f.token] = false;
        }
        for row in held.iter_mut().skip(f.layer + 1) {
            for h in &mut row[f.token + 1..] {
                *h = false;
            }
        }
    }
    held
}

#[cfg(test)]
mod tests {
    use super::{head_cap, head_gap, head_tie};

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
}
