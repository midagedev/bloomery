//! The host rule for V4.1's HC_PRE (`ggml_compute_forward_hc_pre_f32`) that
//! the hyper-connection gates compare the device kernels with: one token's
//! sigmoid pre/post and Sinkhorn-normalised comb from its mixes. Also our
//! Q8_0 chain (`hc_pre_q8_0`: RMS + split-K Q8_0 gemv), transcribed lane for
//! lane, and the seeded fixture its gate clause and digest read. Host-only;
//! the gate binaries bring the device crate and assert that its `HC_STREAMS`,
//! `HC_MIX` and `HC_PIECE` equal the ones here.

use crate::rounding::butterfly;
use gguf::quant::half_to_f32;

/// Residual streams (the file's `hyper_connection.count`).
pub const HC_STREAMS: usize = 4;
/// HC_PRE values per token: `pre`, `post`, `comb`.
pub const HC_MIX: usize = 24;
/// Values of K one HC_PRE block owns.
pub const HC_PIECE: usize = 512;

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

/// Values in ±2 from a seeded LCG (Numerical Recipes' constants), as the
/// gates' streams.
pub fn lcg(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((s >> 8) as f32 / (1u32 << 24) as f32) * 4.0 - 2.0
        })
        .collect()
}

/// A seeded Q8_0 `hc_fn` of [`HC_MIX`] rows of `k` values in
/// `bloomery_gpu::q8f32`'s device layout: the code words (`k / 4` a row,
/// every byte value from the LCG, -128 included) and the block scales' f16
/// bits (`k / 32` a row), finite normals of either sign, magnitudes in
/// [2^-14, 2^-8), so the mixes over 16384 unit-RMS values stay of order
/// one to ten and HC_PRE's sigmoids and softmax do not saturate.
pub fn q8_0_fixture(k: usize, seed: u32) -> (Vec<u32>, Vec<u16>) {
    let mut s = seed;
    let mut next = || {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        s
    };
    let qs = (0..HC_MIX * k / 4).map(|_| next()).collect();
    let d = (0..HC_MIX * k / 32)
        .map(|_| {
            let r = next();
            // Sign, a biased exponent in 1..=6, ten mantissa bits.
            let (sign, exp, mant) = ((r >> 31) as u16, 1 + (r >> 8) % 6, (r >> 16) & 0x3ff);
            (sign << 15) | ((exp as u16) << 10) | mant as u16
        })
        .collect();
    (qs, d)
}

/// The sixteen stream values lane `lane` of an `hc_pre_q8_0` piece block
/// reads for token `t` of `x` (`k` values a token) in piece `p`: its four
/// words `128p + lane + 32s`, four values each, in `s` order.
fn q8_0_lane_values(x: &[f32], k: usize, t: usize, p: usize, lane: usize) -> [f32; 16] {
    let xb = t * k + HC_PIECE * p + 4 * lane;
    std::array::from_fn(|i| x[xb + 128 * (i / 4) + i % 4])
}

/// Our Q8_0 chain (`bloomery_gpu_deepseek41::hc`'s `hc_pre_q8_0` rule,
/// steps 1-4) for `t` tokens of raw streams `x` (`k` values a token, `k` a
/// multiple of [`HC_PIECE`]) and the weight in [`q8_0_fixture`]'s layout:
/// the scaled mixes, [`HC_MIX`] per token, the value HC_PRE takes. Lane
/// `l`'s partial of row `r` over piece `p` is `fma(q·d, x, f)` from 0 over
/// its words in `s` order and each word's codes in byte order, the lanes
/// combined by [`butterfly`]; the squares the same over the lane's values;
/// both summed over the pieces in ascending order from piece 0's value.
pub fn hc_chain_q8_0(
    qs: &[u32],
    d: &[u16],
    x: &[f32],
    k: usize,
    t: usize,
    rms_eps: f32,
) -> Vec<f32> {
    let (np, words, blocks) = (k / HC_PIECE, k / 4, k / 32);
    let mut part = vec![0.0f32; np * t * HC_MIX];
    let mut sq = vec![0.0f32; np * t];
    for p in 0..np {
        for tt in 0..t {
            let vals: [[f32; 16]; 32] =
                std::array::from_fn(|lane| q8_0_lane_values(x, k, tt, p, lane));
            sq[p * t + tt] = butterfly(std::array::from_fn(|lane| {
                vals[lane].iter().fold(0.0f32, |acc, &v| v.mul_add(v, acc))
            }));
            for r in 0..HC_MIX {
                part[(p * t + tt) * HC_MIX + r] = butterfly(std::array::from_fn(|lane| {
                    let mut f = 0.0f32;
                    for s in 0..4 {
                        let w = 128 * p + lane + 32 * s;
                        let (q, dd) = (qs[r * words + w], half_to_f32(d[r * blocks + w / 8]));
                        for b in 0..4 {
                            let wt = f32::from((q >> (8 * b)) as u8 as i8) * dd;
                            f = wt.mul_add(vals[lane][4 * s + b], f);
                        }
                    }
                    f
                }));
            }
        }
    }
    let mut mixes = vec![0.0f32; t * HC_MIX];
    for tt in 0..t {
        let mut s = sq[tt];
        for p in 1..np {
            s += sq[p * t + tt];
        }
        let scale = 1.0 / (s / k as f32 + rms_eps).sqrt();
        for r in 0..HC_MIX {
            let mut a = part[tt * HC_MIX + r];
            for p in 1..np {
                a += part[(p * t + tt) * HC_MIX + r];
            }
            mixes[tt * HC_MIX + r] = a * scale;
        }
    }
    mixes
}

#[cfg(test)]
mod tests {
    use super::{HC_MIX, exp_ik, exp_ours, hc_chain_q8_0, hc_pre_f32, lcg, q8_0_fixture};
    use crate::Fnv1a64;

    /// FNV-1a over the bits of every output of `rule` on 256 LCG tokens at
    /// three `(eps, iters)` pairs.
    fn digest(rule: impl Fn(&[f32], [f32; 3], &[f32], f32, u32) -> [f32; HC_MIX]) -> u64 {
        let mut h = Fnv1a64::default();
        for (seed, (eps, iters)) in [(1u32, (1e-6f32, 20u32)), (2, (1e-3, 1)), (3, (0.0, 7))] {
            let x = lcg(256 * (2 * HC_MIX + 3), seed);
            for t in x.as_chunks::<{ 2 * HC_MIX + 3 }>().0 {
                let (mix, rest) = t.split_at(HC_MIX);
                let (base, sc) = rest.split_at(HC_MIX);
                h = h.f32s(&rule(mix, [sc[0], sc[1], sc[2]], base, eps, iters));
            }
        }
        h.value()
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

    /// The Q8_0 chain's mixes on the fixture — one piece for 3 tokens, then
    /// 32 pieces (the GLM-5.3-Flash width) for 2 — hash to a fixed digest:
    /// any change to its arithmetic or its order changes it.
    #[test]
    fn hc_chain_q8_0_digest() {
        let mut h = Fnv1a64::default();
        for (k, t, seed) in [(512usize, 3usize, 7u32), (16_384, 2, 8)] {
            let (qs, d) = q8_0_fixture(k, seed);
            let x = lcg(k * t, seed + 100);
            h = h.f32s(&hc_chain_q8_0(&qs, &d, &x, k, t, 1e-5));
        }
        let h = h.value();
        println!("hc_chain_q8_0 digest {h:016x}");
        assert_eq!(h, DIGEST_Q8_0, "hc_chain_q8_0 digest");
    }

    /// `hc_chain_q8_0_digest`'s digest.
    const DIGEST_Q8_0: u64 = 0xcba5_d4f3_0929_ed4a;
}
