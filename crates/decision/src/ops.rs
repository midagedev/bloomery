//! The head's f32 arithmetic on the host: the row-dot product, the linear layer (threaded when
//! large), LayerNorm, softmax, the exact GELU, and multi-head attention over a few queries.
//!
//! Sum order: a dot product of length k runs 8 lane sums of k/8 fused multiply-adds each, then adds
//! the lanes pairwise; a linear layer and the memory attention's products are matrixmultiply's
//! sgemm (its own blocked order); a softmax denominator and an attention value sum run blocks of
//! 64 terms, then add the blocks.

/// The multiply-adds that pay for one more thread of a product: below this a product runs on
/// the calling thread.
const THREAD_MACS: usize = 1 << 18;
/// The values that pay for one more thread of a LayerNorm.
const THREAD_NORM: usize = 1 << 18;
/// The block length of the blocked sums.
const SUM_BLOCK: usize = 64;

use crate::pool::{par_chunks_mut, par_map};

#[inline(always)]
fn lanes_sum(acc: [f32; 8]) -> f32 {
    ((acc[0] + acc[4]) + (acc[2] + acc[6])) + ((acc[1] + acc[5]) + (acc[3] + acc[7]))
}

/// `acc += a·b` lane by lane, each lane one fused multiply-add.
#[inline(always)]
fn fma8(acc: &mut [f32; 8], a: &[f32; 8], b: &[f32; 8]) {
    for l in 0..8 {
        acc[l] = a[l].mul_add(b[l], acc[l]);
    }
}

/// `a · b` over their common length (callers pass equal lengths).
#[inline(always)]
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    let (ac, at) = a.as_chunks::<8>();
    let (bc, bt) = b.as_chunks::<8>();
    let mut acc = [0f32; 8];
    for (x, y) in ac.iter().zip(bc) {
        fma8(&mut acc, x, y);
    }
    let mut s = lanes_sum(acc);
    for (x, y) in at.iter().zip(bt) {
        s = x.mul_add(*y, s);
    }
    s
}

/// `C = A·B` for `C` `[m, n]` row-major, `A` `[m, k]` and `B` `[k, n]` read through the strides
/// `(row, column)` of each; matrixmultiply's sgemm kernels on each thread's share.
fn sgemm(
    (m, k, n): (usize, usize, usize),
    a: &[f32],
    (rsa, csa): (usize, usize),
    b: &[f32],
    (rsb, csb): (usize, usize),
    c: &mut [f32],
) {
    if m == 0 || n == 0 {
        return;
    }
    assert!(c.len() >= m * n, "sgemm: C holds {} of {m}x{n}", c.len());
    if k == 0 {
        c[..m * n].fill(0.0);
        return;
    }
    let last_a = (m - 1) * rsa + (k - 1) * csa;
    let last_b = (k - 1) * rsb + (n - 1) * csb;
    assert!(
        last_a < a.len() && last_b < b.len(),
        "sgemm: A or B is shorter than its strides reach"
    );
    let s = |v: usize| isize::try_from(v).expect("a stride fits isize");
    // SAFETY: every element sgemm reads, A[i·rsa + p·csa] for i < m, p < k and B[p·rsb + j·csb]
    // for p < k, j < n, lies at or before the last one asserted in bounds above; it writes
    // C[i·n + j] for i < m, j < n, inside `c` (asserted), and nothing else holds `c`.
    unsafe {
        matrixmultiply::sgemm(
            m,
            k,
            n,
            1.0,
            a.as_ptr(),
            s(rsa),
            s(csa),
            b.as_ptr(),
            s(rsb),
            s(csb),
            0.0,
            c.as_mut_ptr(),
            s(n),
            1,
        );
    }
}

/// [`sgemm`] over threads: by rows of C when it is at least as tall as it is wide (each thread
/// reads all of B), else by columns (each reads all of A), one thread per [`THREAD_MACS`]
/// multiply-adds at most.
fn gemm(
    (m, k, n): (usize, usize, usize),
    a: &[f32],
    sa: (usize, usize),
    b: &[f32],
    sb: (usize, usize),
) -> Vec<f32> {
    let mut out = vec![0f32; m * n];
    let t = threads().min((m * n * k / THREAD_MACS).max(1));
    if t == 1 {
        sgemm((m, k, n), a, sa, b, sb, &mut out);
        return out;
    }
    if m >= n {
        let rows = m.div_ceil(t).next_multiple_of(8);
        par_chunks_mut(&mut out, rows * n, &|p, os| {
            sgemm((os.len() / n, k, n), &a[p * rows * sa.0..], sa, b, sb, os);
        });
        return out;
    }
    let cols = n.div_ceil(t).next_multiple_of(8);
    let parts: Vec<(usize, Vec<f32>)> = par_map(n.div_ceil(cols), &|p| {
        let c0 = p * cols;
        let np = cols.min(n - c0);
        let mut o = vec![0f32; m * np];
        sgemm((m, k, np), a, sa, &b[c0 * sb.1..], sb, &mut o);
        (c0, o)
    });
    for (c0, part) in &parts {
        let np = part.len() / m;
        for i in 0..m {
            out[i * n + c0..i * n + c0 + np].copy_from_slice(&part[i * np..(i + 1) * np]);
        }
    }
    out
}

fn threads() -> usize {
    crate::pool::size()
}

/// `x · wᵀ + bias`: `x` is `[m, k]`, `w` is `[n, k]` (torch's Linear layout), the result `[m, n]`.
#[must_use]
pub(crate) fn linear(x: &[f32], k: usize, w: &[f32], bias: Option<&[f32]>) -> Vec<f32> {
    let (m, n) = (x.len() / k, w.len() / k);
    let mut out = gemm((m, k, n), x, (k, 1), w, (1, k));
    if let Some(b) = bias {
        for row in out.chunks_exact_mut(n) {
            for (o, b) in row.iter_mut().zip(b) {
                *o += b;
            }
        }
    }
    out
}

/// A sum of many terms: blocks of [`SUM_BLOCK`], then the block sums.
#[inline]
pub(crate) fn blocked_sum(v: &[f32]) -> f32 {
    v.chunks(SUM_BLOCK).map(|c| c.iter().sum::<f32>()).sum()
}

/// LayerNorm of every `width`-wide row of `x` in place (eps 1e-5, the biased variance), rows split
/// over threads when there are many.
pub(crate) fn layer_norm(x: &mut [f32], width: usize, w: &[f32], b: &[f32]) {
    let t = threads().min((x.len() / THREAD_NORM).max(1));
    if t == 1 {
        layer_norm_rows(x, width, w, b);
        return;
    }
    let rows = (x.len() / width).div_ceil(t);
    par_chunks_mut(x, rows * width, &|_, part| {
        layer_norm_rows(part, width, w, b)
    });
}

fn layer_norm_rows(x: &mut [f32], width: usize, w: &[f32], b: &[f32]) {
    let inv = 1.0 / width as f32;
    for row in x.chunks_exact_mut(width) {
        let mean = blocked_sum(row) * inv;
        let mut sq = [0f32; 8];
        let (c, t) = row.as_chunks::<8>();
        for v in c {
            for l in 0..8 {
                let d = v[l] - mean;
                sq[l] += d * d;
            }
        }
        let mut var = lanes_sum(sq);
        for v in t {
            var += (v - mean) * (v - mean);
        }
        let rstd = 1.0 / (var * inv + 1e-5).sqrt();
        for ((v, w), b) in row.iter_mut().zip(w).zip(b) {
            *v = (*v - mean) * rstd * w + b;
        }
    }
}

/// LayerNorm into a new buffer, each thread copying and normalizing its own rows.
#[must_use]
pub(crate) fn layer_normed(x: &[f32], width: usize, w: &[f32], b: &[f32]) -> Vec<f32> {
    let t = threads().min((x.len() / THREAD_NORM).max(1));
    if t == 1 {
        let mut y = x.to_vec();
        layer_norm_rows(&mut y, width, w, b);
        return y;
    }
    let mut y = vec![0f32; x.len()];
    let rows = (x.len() / width).div_ceil(t);
    par_chunks_mut(&mut y, rows * width, &|p, dst| {
        dst.copy_from_slice(&x[p * rows * width..p * rows * width + dst.len()]);
        layer_norm_rows(dst, width, w, b);
    });
    y
}

/// Softmax of `v` in place: the max subtracted, the denominator a blocked sum.
pub(crate) fn softmax(v: &mut [f32]) {
    let top = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    for x in v.iter_mut() {
        *x = (*x - top).exp();
    }
    let total = blocked_sum(v);
    for x in v.iter_mut() {
        *x /= total;
    }
}

/// torch's exact GELU, `x/2 · (1 + erf(x/√2))`, evaluated in f64 and rounded once.
#[inline]
pub(crate) fn gelu(x: f32) -> f32 {
    let x = f64::from(x);
    (0.5 * x * (1.0 + erf(x * std::f64::consts::FRAC_1_SQRT_2))) as f32
}

/// `erf` in f64: the Maclaurin series below 2.5, the continued fraction of `erfc` above; both converge
/// well under an f32 ulp. TWIN: `crates/gpu-vision/src/gemm_bf16.rs` `erf_f64` (a device crate this
/// pure crate cannot depend on).
pub(crate) fn erf(x: f64) -> f64 {
    let a = x.abs();
    let v = if a == 0.0 {
        0.0
    } else if a < 2.5 {
        // erf(a) = 2/√π · Σ (-1)^n a^(2n+1) / (n! (2n+1))
        let a2 = a * a;
        let mut term = a;
        let mut sum = a;
        let mut n = 0.0f64;
        loop {
            n += 1.0;
            term *= -a2 / n;
            let add = term / (2.0 * n + 1.0);
            sum += add;
            if add.abs() <= 1e-18 * sum.abs() {
                break;
            }
        }
        sum * std::f64::consts::FRAC_2_SQRT_PI
    } else if a < 6.0 {
        // erfc(a) = exp(-a²)/√π · 1/(a + (1/2)/(a + 1/(a + (3/2)/(a + …)))), from 60 terms down.
        let mut t = a;
        let mut i = 60.0f64;
        while i >= 1.0 {
            t = a + (i / 2.0) / t;
            i -= 1.0;
        }
        1.0 - (-a * a).exp() / (std::f64::consts::PI.sqrt() * t)
    } else {
        1.0
    };
    v.copysign(x)
}

/// `x / max(‖x‖, eps)` (torch's `F.normalize` with eps 1e-12; `cosine_similarity` uses 1e-8).
#[must_use]
pub(crate) fn normalized(x: &[f32], eps: f32) -> Vec<f32> {
    let norm = dot(x, x).sqrt().max(eps);
    x.iter().map(|v| v / norm).collect()
}

/// Multi-head attention's packed weights (torch `nn.MultiheadAttention`).
pub(crate) struct Mha {
    /// `[3w, w]`: the q, k and v projections stacked.
    pub in_w: Vec<f32>,
    pub in_b: Vec<f32>,
    pub out_w: Vec<f32>,
    pub out_b: Vec<f32>,
    pub width: usize,
    pub heads: usize,
}

/// One memory's keys and values: `[n, 2w]`, row t holding k_t then v_t.
pub(crate) struct Kv {
    kv: Vec<f32>,
    n: usize,
}

impl Mha {
    /// The keys and values of `memory` (`[n, w]`).
    #[must_use]
    pub(crate) fn kv(&self, memory: &[f32]) -> Kv {
        let w = self.width;
        Kv {
            kv: linear(memory, w, &self.in_w[w * w..], Some(&self.in_b[w..])),
            n: memory.len() / w,
        }
    }

    /// Attention of the rows of `x` (`[r, w]`) over `memory` (`[n, w]`) through the output
    /// projection, the keys and values never formed: per head h and query q, the scores are
    /// `memory · (W_kₕᵀ qₕ)` (the key bias adds the same `qₕ·b_kₕ` to every score, which the
    /// softmax cancels) and the value is `W_vₕ (Σₜ pₜ memoryₜ) + b_vₕ` (the weights sum to one).
    /// The same attention as [`Mha::attend`] over [`Mha::kv`], in another sum order, at
    /// `r·heads` rows of work per memory row instead of `2w`.
    #[must_use]
    pub(crate) fn attend_memory(&self, x: &[f32], memory: &[f32]) -> Vec<f32> {
        let w = self.width;
        let (r, n, hd) = (x.len() / w, memory.len() / w, w / self.heads);
        let rh = r * self.heads;
        let scale = 1.0 / (hd as f32).sqrt();
        let q = linear(x, w, &self.in_w[..w * w], Some(&self.in_b[..w]));
        let (wk, wv) = (&self.in_w[w * w..2 * w * w], &self.in_w[2 * w * w..]);
        let heads = self.heads;
        let per_head = |f: &(dyn Fn(usize) -> Vec<f32> + Sync)| par_map(heads, f);
        // u, rows (h, i): scale · W_kₕᵀ qᵢₕ.
        let u: Vec<f32> = per_head(&|h| {
            let mut uh = vec![0f32; r * w];
            for (i, row) in uh.chunks_exact_mut(w).enumerate() {
                for d in 0..hd {
                    let qv = q[i * w + h * hd + d] * scale;
                    let wrow = &wk[(h * hd + d) * w..(h * hd + d + 1) * w];
                    for (o, wv) in row.iter_mut().zip(wrow) {
                        *o = qv.mul_add(*wv, *o);
                    }
                }
            }
            uh
        })
        .concat();
        // scores [n, rh], then each column's softmax as a row of p [rh, n].
        let scores = linear(memory, w, &u, None);
        let mut p = vec![0f32; rh * n];
        for (c, row) in p.chunks_exact_mut(n).enumerate() {
            for (t, v) in row.iter_mut().enumerate() {
                *v = scores[t * rh + c];
            }
            softmax(row);
        }
        // mbar [rh, w] = p · memory, rows (h, i).
        let mbar = gemm((rh, n, w), &p, (n, 1), memory, (w, 1));
        let outs = per_head(&|h| {
            let mut o = vec![0f32; r * hd];
            for i in 0..r {
                let m = &mbar[(h * r + i) * w..(h * r + i + 1) * w];
                for d in 0..hd {
                    let j = h * hd + d;
                    o[i * hd + d] = dot(&wv[j * w..(j + 1) * w], m) + self.in_b[2 * w + j];
                }
            }
            o
        });
        let mut joined = vec![0f32; r * w];
        for (h, o) in outs.iter().enumerate() {
            for i in 0..r {
                joined[i * w + h * hd..i * w + (h + 1) * hd]
                    .copy_from_slice(&o[i * hd..(i + 1) * hd]);
            }
        }
        linear(&joined, w, &self.out_w, Some(&self.out_b))
    }

    /// Attention of the rows of `x` (`[r, w]`) over `kv`, through the output projection.
    #[must_use]
    pub(crate) fn attend(&self, x: &[f32], kv: &Kv) -> Vec<f32> {
        let w = self.width;
        let r = x.len() / w;
        let hd = w / self.heads;
        let scale = 1.0 / (hd as f32).sqrt();
        let q = linear(x, w, &self.in_w[..w * w], Some(&self.in_b[..w]));
        let one_head = |h: usize| -> Vec<f32> {
            let mut out = vec![0f32; r * hd];
            let mut p = vec![0f32; kv.n];
            let mut part = vec![0f32; hd];
            for i in 0..r {
                let qh = &q[i * w + h * hd..i * w + (h + 1) * hd];
                for (t, pt) in p.iter_mut().enumerate() {
                    *pt = dot(qh, &kv.kv[t * 2 * w + h * hd..t * 2 * w + (h + 1) * hd]) * scale;
                }
                softmax(&mut p);
                let o = &mut out[i * hd..(i + 1) * hd];
                for (b, block) in p.chunks(SUM_BLOCK).enumerate() {
                    part.fill(0.0);
                    for (j, &pt) in block.iter().enumerate() {
                        let t = b * SUM_BLOCK + j;
                        let v = &kv.kv[t * 2 * w + w + h * hd..t * 2 * w + w + (h + 1) * hd];
                        for (a, v) in part.iter_mut().zip(v) {
                            *a += pt * v;
                        }
                    }
                    for (o, a) in o.iter_mut().zip(&part) {
                        *o += a;
                    }
                }
            }
            out
        };
        let per_head: Vec<Vec<f32>> = if r * kv.n * w < THREAD_MACS / 4 {
            (0..self.heads).map(one_head).collect()
        } else {
            par_map(self.heads, &one_head)
        };
        let mut joined = vec![0f32; r * w];
        for (h, ph) in per_head.iter().enumerate() {
            for i in 0..r {
                joined[i * w + h * hd..i * w + (h + 1) * hd]
                    .copy_from_slice(&ph[i * hd..(i + 1) * hd]);
            }
        }
        linear(&joined, w, &self.out_w, Some(&self.out_b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive(x: &[f32], k: usize, w: &[f32]) -> Vec<f64> {
        let (m, n) = (x.len() / k, w.len() / k);
        let mut o = vec![0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                o[i * n + j] = (0..k)
                    .map(|t| f64::from(x[i * k + t]) * f64::from(w[j * k + t]))
                    .sum();
            }
        }
        o
    }

    fn values(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (s >> 8) as f32 / (1u32 << 24) as f32 - 0.5
            })
            .collect()
    }

    #[test]
    fn linear_matches_f64_on_every_split() {
        // m·n·k past THREAD_MACS on both splits, and under it; k with a tail past the lanes.
        for (m, n, k) in [(5, 7, 13), (300, 70, 1029), (6, 4100, 1027)] {
            let (x, w) = (values(m * k, 1), values(n * k, 2));
            let b = values(n, 3);
            let got = linear(&x, k, &w, Some(&b));
            let want = naive(&x, k, &w);
            for (i, (g, e)) in got.iter().zip(&want).enumerate() {
                let e = e + f64::from(b[i % n]);
                assert!(
                    (f64::from(*g) - e).abs() < 1e-4,
                    "({m},{n},{k}) at {i}: {g} vs {e}"
                );
            }
        }
    }

    #[test]
    fn erf_and_gelu_match_known_values() {
        // erf from Abramowitz & Stegun table 7.1 (to 10 places).
        for (x, want) in [
            (0.5, 0.520_499_877_8),
            (1.0, 0.842_700_792_9),
            (2.0, 0.995_322_265_0),
            (3.0, 0.999_977_909_5),
        ] {
            assert!((erf(x) - want).abs() < 1e-9, "erf({x}) = {}", erf(x));
            assert!((erf(-x) + want).abs() < 1e-9);
        }
        assert_eq!(gelu(0.0), 0.0);
        assert!((gelu(1.0) - 0.841_344_7).abs() < 1e-6);
        assert!((gelu(-1.0) + 0.158_655_25).abs() < 1e-6);
    }

    #[test]
    fn layer_norm_and_softmax() {
        let mut x = vec![1.0f32, 2.0, 3.0, 4.0];
        layer_norm(&mut x, 4, &[1.0; 4], &[0.5; 4]);
        let want = [-1.341_636_f32, -0.447_212, 0.447_212, 1.341_636];
        for (g, w) in x.iter().zip(want) {
            assert!((g - (w + 0.5)).abs() < 1e-5, "{x:?}");
        }
        let mut p = vec![1.0f32, 2.0, 3.0];
        softmax(&mut p);
        assert!((p[2] - 0.665_240_96).abs() < 1e-6 && (p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }
    #[test]
    fn attention_matches_f64() {
        let (w, heads, n, r) = (16, 2, 300, 3);
        let mha = Mha {
            in_w: values(3 * w * w, 4),
            in_b: values(3 * w, 5),
            out_w: values(w * w, 6),
            out_b: values(w, 7),
            width: w,
            heads,
        };
        let (mem, x) = (values(n * w, 8), values(r * w, 9));
        let got = mha.attend(&x, &mha.kv(&mem));
        let by_memory = mha.attend_memory(&x, &mem);
        let proj = |src: &[f32], rows: usize, part: usize| -> Vec<f64> {
            let wpart = &mha.in_w[part * w * w..(part + 1) * w * w];
            let mut o = naive(src, w, wpart);
            for (i, v) in o.iter_mut().enumerate() {
                *v += f64::from(mha.in_b[part * w + i % w]);
            }
            assert_eq!(o.len(), rows * w);
            o
        };
        let (q, k, v) = (proj(&x, r, 0), proj(&mem, n, 1), proj(&mem, n, 2));
        let hd = w / heads;
        let mut joined = vec![0f64; r * w];
        for h in 0..heads {
            for i in 0..r {
                let s: Vec<f64> = (0..n)
                    .map(|t| {
                        (0..hd)
                            .map(|d| q[i * w + h * hd + d] * k[t * w + h * hd + d])
                            .sum::<f64>()
                            / (hd as f64).sqrt()
                    })
                    .collect();
                let top = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let e: Vec<f64> = s.iter().map(|x| (x - top).exp()).collect();
                let z: f64 = e.iter().sum();
                for d in 0..hd {
                    joined[i * w + h * hd + d] =
                        (0..n).map(|t| e[t] / z * v[t * w + h * hd + d]).sum();
                }
            }
        }
        for i in 0..r {
            for j in 0..w {
                let want = f64::from(mha.out_b[j])
                    + (0..w)
                        .map(|t| joined[i * w + t] * f64::from(mha.out_w[j * w + t]))
                        .sum::<f64>();
                for g in [&got, &by_memory] {
                    assert!(
                        (f64::from(g[i * w + j]) - want).abs() < 1e-5,
                        "({i},{j}): {} vs {want}",
                        g[i * w + j]
                    );
                }
            }
        }
    }
}
