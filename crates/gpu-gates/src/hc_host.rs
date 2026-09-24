//! The host rule for V4.1's HC_PRE (`ggml_compute_forward_hc_pre_f32`) that
//! the hyper-connection gates compare the device kernels with: one token's
//! sigmoid pre/post and Sinkhorn-normalised comb from its mixes. Host-only;
//! the gate binaries bring the device crate and assert that its `HC_STREAMS`
//! and `HC_MIX` equal the ones here.

/// Residual streams (the file's `hyper_connection.count`).
pub const HC_STREAMS: usize = 4;
/// HC_PRE values per token: `pre`, `post`, `comb`.
pub const HC_MIX: usize = 24;

/// Divide every comb entry of a row by `eps` plus the row, the sum in
/// column order.
fn row_norm(m: &mut [f32; 16], eps: f32) {
    for row in m.as_chunks_mut::<4>().0 {
        let s = (((eps + row[0]) + row[1]) + row[2]) + row[3];
        for v in row.iter_mut() {
            *v /= s;
        }
    }
}

/// Divide every comb entry of a column by `eps` plus the column, the sum
/// in row order.
fn col_norm(m: &mut [f32; 16], eps: f32) {
    let mut s = [eps; 4];
    for (c, sc) in s.iter_mut().enumerate() {
        *sc = (((*sc + m[c]) + m[4 + c]) + m[8 + c]) + m[12 + c];
    }
    for (k, v) in m.iter_mut().enumerate() {
        *v /= s[k % 4];
    }
}

/// HC_PRE of one token (`ggml_compute_forward_hc_pre_f32`, the affine as
/// the fused multiply-add ik's build makes of it) with `exp` supplied:
/// [`exp_ik`] is ik's, [`exp_ours`] the kernels'. `mix` and `base` hold
/// [`HC_MIX`] values, `sc` the three scales. Output in the kernel's layout:
/// pre, post, comb.
pub fn hc_pre_f32(
    mix: &[f32],
    sc: [f32; 3],
    base: &[f32],
    eps: f32,
    iters: u32,
    exp: fn(f32) -> f32,
) -> [f32; HC_MIX] {
    let sigmoid = |i: usize, s: f32| 1.0f32 / (1.0 + exp(-mix[i].mul_add(s, base[i])));
    let mut out = [0.0f32; HC_MIX];
    let (head, comb) = out.split_at_mut(2 * HC_STREAMS);
    let (pre, post) = head.split_at_mut(HC_STREAMS);
    for (i, (p, q)) in pre.iter_mut().zip(post.iter_mut()).enumerate() {
        *p = sigmoid(i, sc[0]) + eps;
        *q = 2.0 * sigmoid(HC_STREAMS + i, sc[1]);
    }
    let mut m = [0.0f32; 16];
    for (k, mk) in m.iter_mut().enumerate() {
        *mk = mix[8 + k].mul_add(sc[2], base[8 + k]);
    }
    for row in m.as_chunks_mut::<4>().0 {
        let mut mx = row[0];
        for &x in &row[1..] {
            mx = if mx > x { mx } else { x };
        }
        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            *v = exp(*v - mx);
            sum += *v;
        }
        for v in row.iter_mut() {
            *v = *v / sum + eps;
        }
    }
    col_norm(&mut m, eps);
    for _ in 1..iters {
        row_norm(&mut m, eps);
        col_norm(&mut m, eps);
    }
    comb.copy_from_slice(&m);
    out
}

/// ik's `exp`: glibc's `expf`.
pub fn exp_ik(x: f32) -> f32 {
    x.exp()
}

/// The kernels' `exp`: f64's, rounded to f32.
pub fn exp_ours(x: f32) -> f32 {
    f64::from(x).exp() as f32
}

#[cfg(test)]
mod tests {
    use super::{HC_MIX, exp_ik, exp_ours, hc_pre_f32};

    /// Values in ±2 from a seeded LCG (Numerical Recipes' constants), as the
    /// gates' streams.
    fn lcg(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((s >> 8) as f32 / (1u32 << 24) as f32) * 4.0 - 2.0
            })
            .collect()
    }

    /// FNV-1a over the bits of every output of `rule` on 256 LCG tokens at
    /// three `(eps, iters)` pairs.
    fn digest(rule: impl Fn(&[f32], [f32; 3], &[f32], f32, u32) -> [f32; HC_MIX]) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for (seed, (eps, iters)) in [(1u32, (1e-6f32, 20u32)), (2, (1e-3, 1)), (3, (0.0, 7))] {
            let x = lcg(256 * (2 * HC_MIX + 3), seed);
            for t in x.as_chunks::<{ 2 * HC_MIX + 3 }>().0 {
                let (mix, rest) = t.split_at(HC_MIX);
                let (base, sc) = rest.split_at(HC_MIX);
                for v in rule(mix, [sc[0], sc[1], sc[2]], base, eps, iters) {
                    for b in v.to_bits().to_le_bytes() {
                        h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
                    }
                }
            }
        }
        h
    }

    /// The rule's outputs on a fixed input, with each `exp`, hash to fixed
    /// digests: any change to its arithmetic or its order changes them.
    #[test]
    fn hc_pre_f32_digest() {
        let ours = digest(|m, s, b, e, i| hc_pre_f32(m, s, b, e, i, exp_ours));
        let ik = digest(|m, s, b, e, i| hc_pre_f32(m, s, b, e, i, exp_ik));
        println!("hc_pre_f32 digest exp_ours {ours:016x} exp_ik {ik:016x}");
        assert_eq!(ours, DIGEST_OURS, "exp_ours digest");
        assert_eq!(ik, DIGEST_IK, "exp_ik digest");
    }

    /// `hc_pre_f32_digest`'s digests with `exp_ours` and with `exp_ik`.
    const DIGEST_OURS: u64 = 0x1a3e_ff70_5148_4e5a;
    const DIGEST_IK: u64 = 0x3996_eb01_d160_8742;
}
