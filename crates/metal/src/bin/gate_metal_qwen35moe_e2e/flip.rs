//! The rules the CUDA e2e gate holds a free-running chain to, ported for the
//! Metal session: the relative distance a layer output is read at, the
//! router's flip test and the rows a near-tie flip carries, and the head's
//! tie band. The sources are `crates/gpu-gates/src/flip.rs` (`margin`,
//! `picks_its_top`, `head_gap`, `head_cap`, `head_tie`) and
//! `crates/gpu-gates/src/bin/gate_qwen35moe_e2e.rs` (`rel_base`, `rel_to`,
//! `worse`, `router_tie`, `held_rows`) and `crates/gpu-gates/src/bin/shared/e2e.rs`
//! (`argmax`); the arithmetic is theirs, line for
//! line, so the two backends are held to one rule. Both live in crates that
//! link the CUDA backend, which a Mac cannot build.

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

/// The larger of two errors, a NaN counting as infinite.
pub fn worse(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::INFINITY
    } else {
        a.max(b)
    }
}

/// ik's own margin over `values` (one token's row) and its chosen `ids`: the
/// lowest value it kept less the highest it left.
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
fn rms(v: &[f32]) -> f64 {
    (v.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>() / v.len().max(1) as f64).sqrt()
}

/// How far below ik's top logit ik's own row puts our argmax `ours`; NaN for
/// an id past the row and for a row with a value that is not finite.
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

/// The widest [`head_gap`] a head's rounding excuses: `6 · band ·` RMS of
/// ik's row, three deviations each for our logit and ik's.
pub fn head_cap(band: f64, ik_row: &[f32]) -> f64 {
    6.0 * band * rms(ik_row)
}

/// A head's argmax excused as a tie band: ik's own logit at our argmax lies
/// within [`head_cap`] of ik's top. Inclusive; a NaN is no tie.
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

/// The rows `[layer][token]` a near-tie flip leaves unheld: the flipped token
/// from its layer's output on, and every later token at the deeper layers
/// (their mixers read the flipped token's state and keys).
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

/// The index of the largest value; the first on a tie.
pub fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    u32::try_from(best).expect("a row index fits u32")
}
