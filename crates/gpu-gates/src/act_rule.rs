//! What each side does to the activation of a dense projection whose weight
//! is a K-quant, by the weight's type: ik's CPU rule (its AVX2 build with
//! `GGML_USE_IQK_MULMAT`, no AVX-512 — `HAVE_FANCY_SIMD` off) and ours (the
//! card's K-quant gemvs). A gate that bounds our output against ik's dump
//! reads here how each side rounded the input; a Q8_0 weight's rule is
//! [`crate::ik_q8_2`] (ik) against f32 (ours). Transcribed from source; calls
//! no engine or device code.
//!
//! | weight | ik's activation | ik's dot | our activation |
//! |---|---|---|---|
//! | Q3_K | q8_K: 256 values, f32 `d` ([`quantize_q8_k`]) | [`dot_q3k`] | q8_1: 128 values, f32 `d` |
//! | Q4_K | q8_2: 32 values, bf16 `d` ([`crate::ik_q8_2::quantize`]) | `mul_mat_qX_K_q8_2_X4_T` | q8_1 |
//! | Q5_K | q8_2 | `mul_mat_qX_K_q8_2_X4_T` | f32 |
//!
//! ik's format per weight type is `gguf::quant::activation_format`'s table
//! (`vec_dot_type`, and `iqk_set_kernels_kquants`'s `expected_type_B` in
//! `iqk_gemm_kquants.cpp`); at one token no weight is repacked, so the
//! weight's own dot runs. Ours: Q3_K and Q4_K read the q8_1 form
//! (`q3k_quantize_q8_1`, [`crate::q8_1_dequant`] its transcription), Q5_K
//! reads f32 (`ds41_q5k_gemv_f32`).
//!
//! The K-quant weights are codes of one sign (Q3_K's `u − 4` goes through
//! the block sums, Q4_K's and Q5_K's minimum likewise), so neither side folds
//! the activation's sign by the weight's: a quantized value stands for `q·d`
//! under every weight, and the rounding error of value `c` is one number,
//! `x_c − q_c·d` ([`Act::err`]).
//!
//! Not in the per-value error: ik's Q4_K/Q5_K dot takes the minimum's
//! term through q8_2's 16-bit block-sum field (`block_q8_2::s`,
//! `ggml-common.h`), which a gate on those weights must transcribe with
//! that dot; and either side's f32 accumulation (ours: per 128-value block
//! one product `(Σ sc·dot) · (d8·d)` added into the lane, then the warp's
//! butterfly; ik: [`dot_q3k`]'s order), which is `γ(n)`-small beside the
//! rounding of the codes.

use gguf::quant::{GgmlType, half_to_f32};

/// Values per ik q8_K block and per K-quant super-block.
pub const QK_K: usize = 256;

/// Values per block of our q8_1 activation form.
pub const Q8_1_BLOCK: usize = 128;

/// Bytes of one Q3_K super-block: `hmask[32]`, `qs[64]`, `scales[12]`, `d`.
pub const Q3K_BYTES: usize = 110;

/// One side's activation format under a weight.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Act {
    /// Read as f32: no rounding.
    F32,
    /// Our q8_1: blocks of [`Q8_1_BLOCK`].
    Q8_1,
    /// ik's q8_K: blocks of [`QK_K`].
    Q8K,
    /// ik's q8_2: blocks of [`crate::ik_q8_2::QK`].
    Q8_2,
}

impl Act {
    /// Values per block (1 for f32).
    #[must_use]
    pub fn block(self) -> usize {
        match self {
            Act::F32 => 1,
            Act::Q8_1 => Q8_1_BLOCK,
            Act::Q8K => QK_K,
            Act::Q8_2 => crate::ik_q8_2::QK,
        }
    }

    /// The values the quantized form of `x` stands for, `q·d` each — exact
    /// in f32. `x` is whole blocks.
    #[must_use]
    pub fn reconstruct(self, x: &[f32]) -> Vec<f32> {
        assert!(
            x.len().is_multiple_of(self.block()),
            "{self:?} blocks are {} values",
            self.block()
        );
        match self {
            Act::F32 => x.to_vec(),
            Act::Q8_1 => crate::q8_1_dequant(x, x.len(), 1),
            Act::Q8K => {
                let a = quantize_q8_k(x);
                a.q.iter()
                    .enumerate()
                    .map(|(c, &q)| f32::from(q) * a.d[c / QK_K])
                    .collect()
            }
            Act::Q8_2 => crate::ik_q8_2::reconstruct(x),
        }
    }

    /// The rounding error of each value of `x`, `x_c − q_c·d`, in f64.
    #[must_use]
    pub fn err(self, x: &[f32]) -> Vec<f64> {
        x.iter()
            .zip(self.reconstruct(x))
            .map(|(&v, r)| f64::from(v) - f64::from(r))
            .collect()
    }

    /// The variance of the rounding of each value of `x` as a spread,
    /// `d²/12` with the block's `d = amax/127` ([`q8_var`]); 0 for f32.
    #[must_use]
    pub fn spread(self, x: &[f32]) -> Vec<f64> {
        match self {
            Act::F32 => vec![0.0; x.len()],
            q => q8_var(x, q.block()),
        }
    }
}

/// Both sides' activation formats under one weight type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rule {
    pub ik: Act,
    pub ours: Act,
}

impl Rule {
    /// The rule for a K-quant weight of type `ty`; `None` for every type
    /// this module does not transcribe (Q8_0's is [`crate::ik_q8_2`]).
    #[must_use]
    pub fn of(ty: GgmlType) -> Option<Rule> {
        match ty {
            GgmlType::Q3_K => Some(Rule {
                ik: Act::Q8K,
                ours: Act::Q8_1,
            }),
            GgmlType::Q4_K => Some(Rule {
                ik: Act::Q8_2,
                ours: Act::Q8_1,
            }),
            GgmlType::Q5_K => Some(Rule {
                ik: Act::Q8_2,
                ours: Act::F32,
            }),
            _ => None,
        }
    }

    /// The variance a projection's input adds to its outputs per value of
    /// `x` — ik's input, the dump's node — before the weight: ik's rounding
    /// exactly (`err²`, its codes are the dump's own), ours as a spread,
    /// since our input differs from ik's by the upstream deviation and a
    /// code that flips there is a first-order move. Taken as independent;
    /// where our block's scale is ik's (the block holding ik's block's
    /// largest value) the two roundings are near equal and cancel instead,
    /// so the sum is conservative there.
    #[must_use]
    pub fn input_var(self, x: &[f32]) -> Vec<f64> {
        self.ik
            .err(x)
            .iter()
            .zip(self.ours.spread(x))
            .map(|(e, s)| e * e + s)
            .collect()
    }
}

/// The variance of a q8 quantizer's rounding of each value of `x`, blocks
/// of `block` values with `d = amax / 127`: `d² / 12`.
#[must_use]
pub fn q8_var(x: &[f32], block: usize) -> Vec<f64> {
    x.chunks(block)
        .flat_map(|b| {
            let amax = b.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
            let d = f64::from(amax) / 127.0;
            std::iter::repeat_n(d * d / 12.0, b.len())
        })
        .collect()
}

/// ik's q8_K blocks of a row: the codes, one f32 scale per 256 values, and
/// the sums of every 16 codes.
#[derive(Clone, Debug, Default)]
pub struct Q8K {
    pub q: Vec<i8>,
    pub d: Vec<f32>,
    pub bsums: Vec<i16>,
}

/// ik's q8_K of `x` on AVX2 (`quantize_row_q8_K` → `iqk_quantize_row_q8_K`,
/// `ggml-quants.c:4024`; `iqk_quantize_row_q8_K_T<0>`, `iqk_quantize.cpp:3824`):
/// per 256 values `d = amax/127` and, separately, `id = 127/amax` (0 when
/// `amax` is 0), each code `x·id` rounded half to even and saturated to i8
/// by the two packs, `bsums[k]` the sum of codes `16k .. 16k + 16`. Not
/// `quantize_row_q8_K_ref`'s rule (`-127/max`, `d = 1/iscale`), whose scale
/// differs in the last bits. `x` is whole blocks.
#[must_use]
pub fn quantize_q8_k(x: &[f32]) -> Q8K {
    assert!(
        x.len().is_multiple_of(QK_K),
        "q8_K blocks are {QK_K} values"
    );
    let mut out = Q8K {
        q: Vec::with_capacity(x.len()),
        d: Vec::with_capacity(x.len() / QK_K),
        bsums: Vec::with_capacity(x.len() / 16),
    };
    for blk in x.as_chunks::<QK_K>().0 {
        let amax = blk.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let id = if amax != 0.0 { 127.0 / amax } else { 0.0 };
        out.d.push(amax / 127.0);
        let at = out.q.len();
        out.q.extend(
            blk.iter()
                .map(|&v| (v * id).round_ties_even().clamp(-128.0, 127.0) as i8),
        );
        out.bsums.extend(
            out.q[at..]
                .as_chunks::<16>()
                .0
                .iter()
                .map(|c| c.iter().map(|&q| i16::from(q)).sum::<i16>()),
        );
    }
    out
}

/// The sixteen sub-block scales of a Q3_K super-block from its twelve
/// scale bytes, the 32 offset removed (`ScaleQ3::make_scales`,
/// `iqk_gemm_kquants.cpp:66`; `dequantize_row_q3_K`'s `aux` shuffle).
#[must_use]
pub fn q3k_scales(b: &[u8; 12]) -> [i32; 16] {
    let word = |i: usize| u32::from_le_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]]);
    let (a0, a1, a2) = (word(0), word(1), word(2));
    let t = [
        (a0 & 0x0f0f_0f0f) | ((a2 << 4) & 0x3030_3030),
        (a1 & 0x0f0f_0f0f) | ((a2 << 2) & 0x3030_3030),
        ((a0 >> 4) & 0x0f0f_0f0f) | (a2 & 0x3030_3030),
        ((a1 >> 4) & 0x0f0f_0f0f) | ((a2 >> 2) & 0x3030_3030),
    ];
    std::array::from_fn(|k| i32::from((t[k / 4] >> (8 * (k % 4))) as u8) - 32)
}

/// ik's dot of one Q3_K row (`row`: whole super-blocks of [`Q3K_BYTES`])
/// with q8_K blocks `sb0 ..` of `a` on AVX2: `mul_mat_qY_K_q8_K_T` with
/// `DequantizerQ3K` (`set_functions`, `iqk_gemm_kquants.cpp:1816`; the
/// AVX2 bodies at `:532` and `:678`). Eight f32 lanes, zero at the row's
/// start; per super-block, first the minimum's term — lane `t` fuses
/// `((−4·d)·d8) · (sc₂ₜ·bsum₂ₜ + sc₂ₜ₊₁·bsum₂ₜ₊₁)` (`process_mins_16`) — then the
/// codes' — lane `t` fuses `(d·d8) · Σ sc·u·q` over values `4t .. 4t + 4` of
/// every 32 (`multiply_add`), `u = q2 | 4·hbit` the unsigned code; every
/// integer exact (`maddubs` pairs stay under 2·7·128). `hsum_float_8` adds
/// lane `t` to `t + 4`, then `(s₀ + s₂) + (s₁ + s₃)`.
#[must_use]
pub fn dot_q3k(row: &[u8], a: &Q8K, sb0: usize) -> f32 {
    assert!(
        row.len().is_multiple_of(Q3K_BYTES),
        "Q3_K rows are whole super-blocks"
    );
    let mut acc = [0.0f32; 8];
    for (i, blk) in row.as_chunks::<Q3K_BYTES>().0.iter().enumerate() {
        let b = sb0 + i;
        let d = half_to_f32(u16::from_le_bytes([blk[108], blk[109]]));
        let sc = q3k_scales(blk[96..108].try_into().expect("twelve scale bytes"));
        let d8 = a.d[b];
        let q = &a.q[b * QK_K..(b + 1) * QK_K];
        let bs = &a.bsums[b * 16..(b + 1) * 16];
        let cm = (-4.0 * d) * d8;
        for (t, l) in acc.iter_mut().enumerate() {
            let prod = sc[2 * t] * i32::from(bs[2 * t]) + sc[2 * t + 1] * i32::from(bs[2 * t + 1]);
            *l = cm.mul_add(prod as f32, *l);
        }
        let mut sumi = [0i32; 8];
        for j in 0..2 {
            for l in 0..4 {
                for m in 0..32 {
                    let q2 = i32::from((blk[32 + 32 * j + m] >> (2 * l)) & 3);
                    let h = i32::from((blk[m] >> (4 * j + l)) & 1);
                    let v = 128 * j + 32 * l + m;
                    sumi[m / 4] += sc[8 * j + 2 * l + m / 16] * (q2 | (h << 2)) * i32::from(q[v]);
                }
            }
        }
        let cd = d * d8;
        for (l, &s) in acc.iter_mut().zip(&sumi) {
            *l = cd.mul_add(s as f32, *l);
        }
    }
    let s: [f32; 4] = std::array::from_fn(|t| acc[t] + acc[t + 4]);
    (s[0] + s[2]) + (s[1] + s[3])
}

#[cfg(test)]
mod tests {
    use super::{Act, QK_K, dot_q3k, q3k_scales, quantize_q8_k};
    use gguf::quant::{GgmlType, dequant_row};

    /// A super-block's scales decode as `dequant_row` reads them: every
    /// byte pattern of a scale word at each position.
    #[test]
    fn q3k_scales_match_dequant() {
        let mut blk = [0u8; 110];
        blk[108..110].copy_from_slice(&0x3c00u16.to_le_bytes());
        for (i, v) in blk[32..96].iter_mut().enumerate() {
            *v = (i as u8).wrapping_mul(37);
        }
        for (i, v) in blk[..32].iter_mut().enumerate() {
            *v = (i as u8).wrapping_mul(91) ^ 0x5a;
        }
        for seed in 0u32..64 {
            for (i, v) in blk[96..108].iter_mut().enumerate() {
                *v = (seed.wrapping_mul(2_654_435_761) >> (i % 24)) as u8 ^ (i as u8 * 13);
            }
            let mut w = [0.0f32; QK_K];
            dequant_row(GgmlType::Q3_K, &blk, &mut w).unwrap();
            let sc = q3k_scales(blk[96..108].try_into().unwrap());
            for (k, chunk) in w.chunks(16).enumerate() {
                let (j, l) = (k / 8, (k % 8) / 2);
                for (m, &x) in chunk.iter().enumerate() {
                    let m = m + 16 * (k % 2);
                    let q2 = i32::from((blk[32 + 32 * j + m] >> (2 * l)) & 3);
                    let h = i32::from((blk[m] >> (4 * j + l)) & 1);
                    assert_eq!(x, (sc[k] * (q2 - 4 * (1 - h))) as f32, "sub-block {k}");
                }
            }
        }
    }

    /// The dot's integer part is the dequantized row against the
    /// reconstructed codes: on a row whose products are exact, the two agree
    /// to f32 accumulation.
    #[test]
    fn dot_q3k_is_the_dequantized_dot() {
        let mut row = vec![0u8; 2 * 110];
        for (i, v) in row.iter_mut().enumerate() {
            *v = (i as u8).wrapping_mul(73) ^ 0x33;
        }
        for sb in 0..2 {
            row[sb * 110 + 108..sb * 110 + 110].copy_from_slice(&0x2e66u16.to_le_bytes());
        }
        let x = crate::activations(2 * QK_K, 1, 7);
        let a = quantize_q8_k(&x);
        let xh = Act::Q8K.reconstruct(&x);
        let mut w = vec![0.0f32; 2 * QK_K];
        dequant_row(GgmlType::Q3_K, &row, &mut w).unwrap();
        let exact: f64 = w
            .iter()
            .zip(&xh)
            .map(|(&a, &b)| f64::from(a) * f64::from(b))
            .sum();
        let abs: f64 = w
            .iter()
            .zip(&xh)
            .map(|(&a, &b)| (f64::from(a) * f64::from(b)).abs())
            .sum();
        let got = f64::from(dot_q3k(&row, &a, 0));
        assert!((got - exact).abs() <= 1e-5 * abs, "{got} vs {exact}");
    }
}
