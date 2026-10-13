//! The f32 host twins of the two mergers: the op-level rules the tower's teacher-forced clauses
//! hold the card to.

use super::GROUP;
use crate::attn::{HEAD_DIM_72, QkvLayout, attn_72_ref};
use crate::gelu_tanh::gelu_tanh_f32;
use crate::gemm_bf16::gelu_erf;

/// One linear layer on the host: `y = x · Wᵀ + b`, `w` row-major `[out, inp]`.
#[derive(Clone, Debug)]
pub struct HostLinear {
    pub w: Vec<f32>,
    pub b: Vec<f32>,
    pub inp: usize,
    pub out: usize,
}

/// A LayerNorm's gain and bias.
#[derive(Clone, Debug)]
pub struct HostNorm {
    pub g: Vec<f32>,
    pub b: Vec<f32>,
}

/// The vit_merger's weights on the host (bf16 widened, or f32 as the file holds them).
#[derive(Clone, Debug)]
pub struct HostVitMerger {
    pub ln1: HostNorm,
    pub q: HostLinear,
    pub k: HostLinear,
    pub v: HostLinear,
    pub o: HostLinear,
    pub ds_ln: HostNorm,
    pub up: HostLinear,
    pub down: HostLinear,
}

/// The final merger's weights on the host.
#[derive(Clone, Debug)]
pub struct HostFinalMerger {
    pub norm: HostNorm,
    pub up: HostLinear,
    pub down: HostLinear,
}

/// `nn.LayerNorm` over rows of `norm.g.len()` values: mean and variance in f64, the result f32.
#[must_use]
pub fn host_layer_norm(x: &[f32], norm: &HostNorm, eps: f32) -> Vec<f32> {
    let width = norm.g.len();
    x.chunks_exact(width)
        .flat_map(|row| {
            let mean = row.iter().map(|&v| f64::from(v)).sum::<f64>() / width as f64;
            let var = row
                .iter()
                .map(|&v| (f64::from(v) - mean).powi(2))
                .sum::<f64>()
                / width as f64;
            let rstd = 1.0 / (var + f64::from(eps)).sqrt();
            row.iter()
                .zip(norm.g.iter().zip(&norm.b))
                .map(move |(&v, (&g, &b))| (((f64::from(v) - mean) * rstd) as f32) * g + b)
        })
        .collect()
}

/// `y = x · Wᵀ + b` for each row of `x`, accumulated in f64.
#[must_use]
pub fn host_linear(x: &[f32], l: &HostLinear) -> Vec<f32> {
    x.chunks_exact(l.inp)
        .flat_map(|row| {
            (0..l.out).map(move |o| {
                let w = &l.w[o * l.inp..(o + 1) * l.inp];
                let dot: f64 = row
                    .iter()
                    .zip(w)
                    .map(|(&a, &b)| f64::from(a) * f64::from(b))
                    .sum();
                (dot + f64::from(l.b[o])) as f32
            })
        })
        .collect()
}

/// The vit_merger on the host, f32 throughout but for the attention: `x` is `n` rows of `dim`
/// (merge order, `n` a multiple of 4), the output `n / 4` rows of `dim`. Window attention is
/// [`attn_72_ref`] on each group of 4 rows alone (q, k and v rounded to bf16, the reference's
/// input type); the down-sample is the 4 consecutive rows as one, `ds_ln`, the up layer, the tanh
/// GELU, the down layer, plus the mean `((p0 + p1) + p2) + p3` × 0.25 of the group's rows after
/// the attention residual.
#[must_use]
pub fn vit_merger_f32(w: &HostVitMerger, x: &[f32], eps: f32) -> Vec<f32> {
    let dim = w.ln1.g.len();
    let n = x.len() / dim;
    assert!(
        n.is_multiple_of(GROUP) && x.len() == n * dim && dim.is_multiple_of(HEAD_DIM_72),
        "{n} rows of {dim}: the vit_merger takes whole windows of heads of {HEAD_DIM_72}"
    );
    let normed = host_layer_norm(x, &w.ln1, eps);
    let (q, k, v) = (
        host_linear(&normed, &w.q),
        host_linear(&normed, &w.k),
        host_linear(&normed, &w.v),
    );
    let qkv: Vec<u16> = (0..n)
        .flat_map(|r| {
            [&q, &k, &v].into_iter().flat_map(move |m| {
                m[r * dim..(r + 1) * dim]
                    .iter()
                    .map(|&f| crate::f32_bf16(f))
            })
        })
        .collect();
    let lay = QkvLayout {
        row_width: 3 * dim,
        q0: 0,
        k0: dim,
        v0: 2 * dim,
        n_heads: dim / HEAD_DIM_72,
    };
    let scale = 1.0 / (HEAD_DIM_72 as f64).sqrt();
    let mut att = vec![0.0f32; n * dim];
    for g in 0..n / GROUP {
        let rows = &qkv[g * GROUP * 3 * dim..(g + 1) * GROUP * 3 * dim];
        for i in 0..GROUP {
            for h in 0..lay.n_heads {
                let (o, _, _) = attn_72_ref(rows, lay, GROUP, scale, h, i);
                for (d, &od) in o.iter().enumerate() {
                    att[(g * GROUP + i) * dim + h * HEAD_DIM_72 + d] = od as f32;
                }
            }
        }
    }
    let out = host_linear(&att, &w.o);
    let stream: Vec<f32> = x.iter().zip(&out).map(|(&a, &b)| a + b).collect();
    let mean: Vec<f32> = stream
        .chunks_exact(GROUP * dim)
        .flat_map(|g| {
            (0..dim).map(move |d| (((g[d] + g[dim + d]) + g[2 * dim + d]) + g[3 * dim + d]) * 0.25)
        })
        .collect();
    let hidden = host_linear(&host_layer_norm(&stream, &w.ds_ln, eps), &w.up);
    let act: Vec<f32> = hidden.into_iter().map(gelu_tanh_f32).collect();
    host_linear(&act, &w.down)
        .into_iter()
        .zip(mean)
        .map(|(a, b)| a + b)
        .collect()
}

/// The final merger on the host: `x` is `n` rows of `dim` (`n` a multiple of 4, `post_ln`'s
/// output in merge order); four consecutive rows are one of `4·dim`, then LayerNorm, the up layer,
/// the erf GELU and the down layer; the output is `n / 4` rows of `down.out`.
#[must_use]
pub fn final_merger_f32(w: &HostFinalMerger, x: &[f32], eps: f32) -> Vec<f32> {
    let merged = w.norm.g.len();
    assert!(
        x.len().is_multiple_of(merged),
        "{} values are not whole rows of {merged}",
        x.len()
    );
    let hidden = host_linear(&host_layer_norm(x, &w.norm, eps), &w.up);
    let act: Vec<f32> = hidden.into_iter().map(gelu_erf).collect();
    host_linear(&act, &w.down)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Small deterministic weights for the host twins: 2 heads of 72, a 12-wide hidden.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.4
        }
        fn vec(&mut self, n: usize) -> Vec<f32> {
            (0..n).map(|_| self.next()).collect()
        }
        fn linear(&mut self, inp: usize, out: usize) -> HostLinear {
            HostLinear {
                w: self.vec(inp * out),
                b: self.vec(out),
                inp,
                out,
            }
        }
        fn norm(&mut self, n: usize) -> HostNorm {
            HostNorm {
                g: self.vec(n).iter().map(|v| 1.0 + v).collect(),
                b: self.vec(n),
            }
        }
    }

    const DIM: usize = 144;

    fn merger(rng: &mut Lcg) -> HostVitMerger {
        HostVitMerger {
            ln1: rng.norm(DIM),
            q: rng.linear(DIM, DIM),
            k: rng.linear(DIM, DIM),
            v: rng.linear(DIM, DIM),
            o: rng.linear(DIM, DIM),
            ds_ln: rng.norm(4 * DIM),
            up: rng.linear(4 * DIM, 12),
            down: rng.linear(12, DIM),
        }
    }

    /// With a zero attention output and a zero down layer the vit_merger is exactly the mean
    /// `((p0 + p1) + p2) + p3` × 0.25 of each group of four rows, and a group's output does not
    /// depend on the other groups' rows.
    #[test]
    fn vit_merger_twin_is_the_mean_when_the_branches_are_zero() {
        let mut rng = Lcg(7);
        let mut w = merger(&mut rng);
        w.o = HostLinear {
            w: vec![0.0; DIM * DIM],
            b: vec![0.0; DIM],
            inp: DIM,
            out: DIM,
        };
        w.down = HostLinear {
            w: vec![0.0; 12 * DIM],
            b: vec![0.0; DIM],
            inp: 12,
            out: DIM,
        };
        let x = rng.vec(8 * DIM);
        let y = vit_merger_f32(&w, &x, 1e-6);
        assert_eq!(y.len(), 2 * DIM);
        let mean: Vec<f32> = (0..2)
            .flat_map(|g| {
                let x = &x;
                (0..DIM).map(move |d| {
                    let r = |i: usize| x[(4 * g + i) * DIM + d];
                    (((r(0) + r(1)) + r(2)) + r(3)) * 0.25
                })
            })
            .collect();
        assert_eq!(y, mean);
    }

    /// Whole windows: changing the rows of group 1 leaves group 0's output bit for bit, and a
    /// change inside group 0 does move it.
    #[test]
    fn vit_merger_twin_windows_are_independent() {
        let mut rng = Lcg(11);
        let w = merger(&mut rng);
        let x = rng.vec(8 * DIM);
        let y = vit_merger_f32(&w, &x, 1e-6);
        let mut x2 = x.clone();
        x2[5 * DIM + 3] += 0.5;
        let y2 = vit_merger_f32(&w, &x2, 1e-6);
        assert_eq!(&y[..DIM], &y2[..DIM]);
        assert_ne!(&y[DIM..], &y2[DIM..]);
        let mut x3 = x.clone();
        x3[2 * DIM + 3] += 0.5;
        assert_ne!(&y[..DIM], &vit_merger_f32(&w, &x3, 1e-6)[..DIM]);
    }

    /// The final merger reads four consecutive rows as one: with an identity up layer and a down
    /// layer that picks the first `out` features, the output is the erf GELU of the LayerNorm
    /// of the concatenation.
    #[test]
    fn final_merger_twin_reads_four_rows_as_one() {
        let mut rng = Lcg(3);
        let (d, out) = (6, 5);
        let merged = 4 * d;
        let norm = rng.norm(merged);
        let mut up = HostLinear {
            w: vec![0.0; merged * merged],
            b: vec![0.0; merged],
            inp: merged,
            out: merged,
        };
        for i in 0..merged {
            up.w[i * merged + i] = 1.0;
        }
        let mut down = HostLinear {
            w: vec![0.0; out * merged],
            b: vec![0.0; out],
            inp: merged,
            out,
        };
        for i in 0..out {
            down.w[i * merged + i] = 1.0;
        }
        let w = HostFinalMerger {
            norm: norm.clone(),
            up,
            down,
        };
        let x = rng.vec(8 * d);
        let y = final_merger_f32(&w, &x, 1e-6);
        assert_eq!(y.len(), 2 * out);
        let ln = host_layer_norm(&x, &norm, 1e-6);
        for t in 0..2 {
            for c in 0..out {
                assert_eq!(y[t * out + c], gelu_erf(ln[t * merged + c]));
            }
        }
    }
}
