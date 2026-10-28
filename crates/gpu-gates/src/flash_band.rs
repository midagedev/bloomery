//! The exact attention the flash gates judge against, and the prefill
//! flash's band: one query head's attention over its first `n` keys in f64
//! on the cache's f16 values, the terms every flash band reads, and the
//! first-order bound of the `flash_gqa_prefill` walk at a head of `head`
//! values and a key tile of `tile` keys. The model is `gate_qwen3moe_flash`'s
//! (its module doc derives each term); each gate keeps the bounds of its own
//! kernels and computes them from these terms. [`scalar_bound`] is the scalar
//! pass's, for a gate whose kernel is that pass at some group.

use crate::rounding::{U, gamma};

/// One query head's exact attention over its first `n` keys, in f64, and
/// the terms the bands read: `s_j = scale·q·k_j`, `M = max s_j`.
pub struct ExactAttn {
    /// The output, `Σ_j p̄_j·v_j`, `head` values.
    pub o: Vec<f64>,
    /// `A_j = |scale|·Σ_d |q_d·k_jd|`.
    pub a: Vec<f64>,
    /// `A_j` with the query's f16 rounding below f16's normal range:
    /// `A_j + |scale|·2^-14·Σ_d |k_jd|`.
    pub a16: Vec<f64>,
    /// `p_j = exp(s_j − M)`.
    pub p: Vec<f64>,
    /// `z = Σ_j p_j`.
    pub z: f64,
    /// `p̄_j = p_j / z`.
    pub pb: Vec<f64>,
    /// `X_j = M − s_j`.
    pub x: Vec<f64>,
    /// `max_j X_j` (0 for no key).
    pub xmax: f64,
}

/// [`ExactAttn`] of the query head `q` (`head` values) over the key and
/// value rows `kh`, `vh` (at least `n` rows of `head` values each) at
/// `scale`. Every sum runs over `j` or `d` ascending.
#[must_use]
pub fn exact_attn(
    q: &[f32],
    kh: &[f32],
    vh: &[f32],
    head: usize,
    n: usize,
    scale: f32,
) -> ExactAttn {
    let sc = f64::from(scale);
    let dot = |j: usize, abs: bool| -> f64 {
        let k = &kh[j * head..(j + 1) * head];
        q.iter()
            .zip(k)
            .map(|(&a, &b)| {
                let p = f64::from(a) * f64::from(b);
                if abs { p.abs() } else { p }
            })
            .sum::<f64>()
    };
    let s: Vec<f64> = (0..n).map(|j| sc * dot(j, false)).collect();
    let a: Vec<f64> = (0..n).map(|j| sc.abs() * dot(j, true)).collect();
    let a16: Vec<f64> = (0..n)
        .map(|j| {
            let k = &kh[j * head..(j + 1) * head];
            a[j] + sc.abs() * 2f64.powi(-14) * k.iter().map(|&b| f64::from(b).abs()).sum::<f64>()
        })
        .collect();
    let m = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let p: Vec<f64> = s.iter().map(|&v| (v - m).exp()).collect();
    let z: f64 = p.iter().sum();
    let pb: Vec<f64> = p.iter().map(|&v| v / z).collect();
    let x: Vec<f64> = s.iter().map(|&v| m - v).collect();
    let xmax = x.iter().copied().fold(0.0f64, f64::max);
    let mut o = vec![0.0f64; head];
    for (j, &w) in pb.iter().enumerate() {
        for (d, od) in o.iter_mut().enumerate() {
            *od += w * f64::from(vh[j * head + d]);
        }
    }
    ExactAttn {
        o,
        a,
        a16,
        p,
        z,
        pb,
        x,
        xmax,
    }
}

/// The scalar flash's bound on each output value of `ex` (`vh` its value
/// rows): the score dot's `γ(head / 4 + 3)` (a partial's `head / 4` fused
/// multiply-adds over four rotating partials, two combine levels, the scale),
/// one rescale per `tile`-key tile and per segment, and the value and weight
/// sums over a segment's `span` keys, the segments and the butterfly. `span`
/// is the segment length the launch cut the `ex.pb.len()` keys by
/// (`seg_span` of them); the pass is `gqa_flash_seg`'s scalar one at that
/// head width, any group.
#[must_use]
pub fn scalar_bound(ex: &ExactAttn, vh: &[f32], head: usize, tile: usize, span: usize) -> Vec<f64> {
    let n = ex.pb.len();
    let segs = n.div_ceil(span);
    let r = (n.div_ceil(tile) + segs) as f64;
    let es = gamma(head / 4 + 3);
    let acc = gamma(span + segs + 8);
    (0..head)
        .map(|d| {
            let (mut t_s, mut pv) = (0.0f64, 0.0f64);
            for j in 0..n {
                let v = f64::from(vh[j * head + d]);
                let dev = (v - ex.o[d]).abs();
                let ee = 4.0 * U + 2.0 * U * ex.x[j];
                t_s += ex.pb[j] * (es * ex.a[j] + ee + 4.0 * U * r + 2.0 * U * ex.xmax) * dev;
                pv += ex.pb[j] * v.abs();
            }
            t_s + acc * (pv + ex.o[d].abs()) + 2.0 * U * ex.o[d].abs()
        })
        .collect()
}

/// The prefill flash's bound on each output value of `ex` (`vh` its value
/// rows): the tensor-core scores (each dim chain, their sum and the query's
/// f16 rounding, `2^-11 + 2·γ(head + 2)` on `a16`), one rescale per
/// `tile`-key tile, each weight rounded to f16 for the value product and
/// normalized by the rounded weights' own sum, the tensor core's value
/// accumulation over `⌈n/16⌉ + 16` roundings twice, and the lanes' weight
/// sums over `⌈n/4⌉` weights and a rescale a tile. `first` is the first
/// key's original index: a group partly masked at either edge still counts.
#[must_use]
pub fn prefill_bound(
    ex: &ExactAttn,
    vh: &[f32],
    head: usize,
    tile: usize,
    first: usize,
) -> Vec<f64> {
    let n = ex.pb.len();
    let es = 2f64.powi(-11) + 2.0 * gamma(head + 2);
    let tiles = (first % tile + n).div_ceil(tile);
    let r = tiles as f64;
    let w16: Vec<f64> = ex
        .pb
        .iter()
        .zip(&ex.p)
        .map(|(&q, &pm)| {
            let rel = 2f64.powi(-11) * q;
            if pm >= 2f64.powi(-14) {
                rel
            } else {
                rel.max(2f64.powi(-25) / ex.z)
            }
        })
        .collect();
    let dbar: f64 = w16.iter().sum();
    let acc_pv = 2.0 * gamma((first % 16 + n).div_ceil(16) + 16);
    let acc_l = gamma((first % 4 + n).div_ceil(4) + tiles + 2);
    (0..head)
        .map(|d| {
            let (mut t_p, mut pv) = (0.0f64, 0.0f64);
            for j in 0..n {
                let v = f64::from(vh[j * head + d]);
                let dev = (v - ex.o[d]).abs();
                let ee = 4.0 * U + 2.0 * U * ex.x[j];
                t_p += (ex.pb[j] * (es * ex.a16[j] + ee + 4.0 * U * r + 2.0 * U * ex.xmax)
                    + w16[j]
                    + ex.pb[j] * dbar)
                    * dev;
                pv += ex.pb[j] * v.abs();
            }
            t_p + acc_pv * pv + acc_l * ex.o[d].abs() + 2.0 * U * ex.o[d].abs()
        })
        .collect()
}

/// Every value of a prefill launch's `y` against its exact value and
/// [`prefill_bound`] at a head of `H` values and a key tile of `TILE` keys:
/// the (row, head) units in `y`'s order over `(n_head, n_kv)`, head `h`
/// reading key head `h / (n_head / n_kv)` of `cache` and row `t` its first
/// `counts[t]` keys. Returns whether every value is within its bound and the
/// largest measured over bound. `first` is the original first-key index
/// when the cache passed here was cut to that key.
#[cfg(feature = "gpu")]
#[must_use]
pub fn judge_prefill<const H: usize, const TILE: usize>(
    q: &[f32],
    counts: &[usize],
    cache: &crate::prefill_gate::HostCache,
    y: &[f32],
    scale: f32,
    (n_head, n_kv): (usize, usize),
    first: usize,
) -> (bool, f64) {
    crate::par_map(counts.len() * n_head, |i| {
        let n = counts[i / n_head];
        let plane = (i % n_head / (n_head / n_kv)) * cache.ctx * H;
        let vh = &cache.vf[plane..plane + n * H];
        let ex = exact_attn(
            &q[i * H..(i + 1) * H],
            &cache.kf[plane..plane + n * H],
            vh,
            H,
            n,
            scale,
        );
        let bound = prefill_bound(&ex, vh, H, TILE, first);
        let (mut ok, mut worst) = (true, 0.0f64);
        for d in 0..H {
            let e = (f64::from(y[i * H + d]) - ex.o[d]).abs();
            ok &= e <= bound[d];
            // A NaN output is out of band and reads as an infinite ratio.
            worst = worst.max(if e.is_nan() {
                f64::INFINITY
            } else {
                e / bound[d]
            });
        }
        (ok, worst)
    })
    .into_iter()
    .fold((true, 0.0f64), |(ok, w), (o, x)| (ok && o, w.max(x)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Zero scores and unit values leave only the value and weight sums: the bound is
    /// `γ(span + segs + 8)` over `1 + |o|`, and `2u|o|`, whatever the head width and tile.
    #[test]
    fn scalar_bound_of_unit_values_is_the_sum_terms() {
        for n in [1usize, 2, 4, 8, 16, 32, 64, 128] {
            let keys = vec![0.0f32; n];
            let values = vec![1.0f32; n];
            let ex = exact_attn(&[0.0], &keys, &values, 1, n, 1.0);
            for span in [64usize, 128] {
                let segs = n.div_ceil(span);
                let want = gamma(span + segs + 8) * 2.0 + 2.0 * U;
                for tile in [32, 64] {
                    let got = scalar_bound(&ex, &values, 1, tile, span)[0];
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "n={n} span={span} tile={tile}"
                    );
                }
            }
        }
    }

    #[test]
    fn prefill_bound_counts_tiles_from_first_key() {
        // Zero scores and unit values make the output and normalized weights
        // exact, leaving only the value and denominator accumulation terms.
        for n in [1usize, 2, 4, 8, 16, 32, 64, 128] {
            let keys = vec![0.0f32; n];
            let values = vec![1.0f32; n];
            let ex = exact_attn(&[0.0], &keys, &values, 1, n, 1.0);
            for tile in [32, 64] {
                let old = 2.0 * gamma(n.div_ceil(16) + 16)
                    + gamma(n.div_ceil(4) + n.div_ceil(tile) + 2)
                    + 2.0 * U;
                for first in [0, tile, 3 * tile] {
                    let bound = prefill_bound(&ex, &values, 1, tile, first);
                    assert_eq!(
                        bound[0].to_bits(),
                        old.to_bits(),
                        "n={n} tile={tile} first={first}"
                    );
                }
            }
        }
        let values = vec![1.0f32; 64];
        let ex = exact_attn(&[0.0], &[0.0; 64], &values, 1, 64, 1.0);
        let shifted = prefill_bound(&ex, &values, 1, 64, 16)[0];
        let two_tiles = 2.0 * gamma(20) + gamma(20) + 2.0 * U;
        assert_eq!(shifted.to_bits(), two_tiles.to_bits());
        assert!(shifted > prefill_bound(&ex, &values, 1, 64, 0)[0]);
        let unaligned = prefill_bound(&ex, &values, 1, 64, 1)[0];
        let groups_at_both_edges = 2.0 * gamma(21) + gamma(21) + 2.0 * U;
        assert_eq!(unaligned.to_bits(), groups_at_both_edges.to_bits());
    }
}
