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

// ---- opg-woaidx ----

/// Host threads for the row walks below.
const ROW_THREADS: usize = 16;

/// A Q3_K weight on the host as both sides' rules read it: the file's rows
/// (for [`dot_q3k`]), the dequantized values (exact in f32), and per value
/// the pair ([`q3k_fields`]) the dot's magnitude is read from.
pub struct Q3kWeight {
    /// Values per row.
    pub k: usize,
    pub rows: usize,
    /// Row-major, `k / QK_K` super-blocks of [`Q3K_BYTES`] per row.
    pub bytes: Vec<u8>,
    /// Row-major.
    pub deq: Vec<f32>,
    /// Row-major, one per value.
    pub fields: Vec<(f32, i8)>,
}

impl Q3kWeight {
    /// `rows` rows of `k` values from `bytes`, the tensor's data. Refused
    /// unless `k` is whole super-blocks, `bytes` is exactly the rows, and the
    /// field walk gives every weight `dequant_row` gives.
    pub fn new(bytes: &[u8], k: usize, rows: usize) -> Result<Q3kWeight, String> {
        if k == 0 || !k.is_multiple_of(QK_K) {
            return Err(format!("Q3_K K={k}: not whole super-blocks"));
        }
        let row_bytes = k / QK_K * Q3K_BYTES;
        if bytes.len() != rows * row_bytes {
            return Err(format!(
                "Q3_K: {} bytes for {rows} rows of {k} values",
                bytes.len()
            ));
        }
        let mut deq = vec![0.0f32; rows * k];
        for (r, out) in deq.chunks_mut(k).enumerate() {
            gguf::quant::dequant_row(
                GgmlType::Q3_K,
                &bytes[r * row_bytes..(r + 1) * row_bytes],
                out,
            )
            .map_err(|e| format!("Q3_K row {r}: {e}"))?;
        }
        let fields = q3k_fields(bytes);
        let exact = deq
            .iter()
            .zip(&fields)
            .all(|(&w, &(s, q))| w.abs().to_bits() == (s * f32::from(q)).abs().to_bits());
        if !exact {
            return Err("Q3_K: the field walk disagrees with dequant_row".to_string());
        }
        Ok(Q3kWeight {
            k,
            rows,
            bytes: bytes.to_vec(),
            deq,
            fields,
        })
    }

    /// Row `r`'s bytes.
    #[must_use]
    pub fn row(&self, r: usize) -> &[u8] {
        let n = self.k / QK_K * Q3K_BYTES;
        &self.bytes[r * n..(r + 1) * n]
    }
}

/// Per value of Q3_K super-blocks, in value order: the sub-block's
/// `|d · sc|` ([`q3k_scales`], rounded once in f32 as `dequantize_row_q3_K`
/// rounds it) and the signed code `q = u − 4 ∈ [−4, 3]`, so the weight is
/// `±|d·sc|·q` and ik's dot adds the `u = q + 4` and `−4` parts apart.
#[must_use]
pub fn q3k_fields(row: &[u8]) -> Vec<(f32, i8)> {
    let mut out = Vec::with_capacity(row.len() / Q3K_BYTES * QK_K);
    for blk in row.as_chunks::<Q3K_BYTES>().0 {
        let d = half_to_f32(u16::from_le_bytes([blk[108], blk[109]]));
        let sc = q3k_scales(blk[96..108].try_into().expect("twelve scale bytes"));
        for v in 0..QK_K {
            let (j, l, m) = (v / 128, (v % 128) / 32, v % 32);
            let q2 = (blk[32 + 32 * j + m] >> (2 * l)) & 3;
            let h = (blk[m] >> (4 * j + l)) & 1;
            let dl = (d * sc[v / 16] as f32).abs();
            out.push((dl, (q2 | (h << 2)) as i8 - 4));
        }
    }
    out
}

/// Our q8_1 activation (`cores::q8_quad`) of `x`, whole blocks of
/// [`Q8_1_BLOCK`]: per block `d = amax/127` (1 for an all-zero block),
/// `q = round(x/d)` half away from zero, clamped to ±127 — each value's
/// `q·d` exact in f64.
#[must_use]
pub fn q8_1_exact(x: &[f32]) -> Vec<f64> {
    x.chunks(Q8_1_BLOCK)
        .flat_map(|b| {
            let amax = b.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
            let d = if amax > 0.0 { amax / 127.0 } else { 1.0 };
            b.iter()
                .map(move |&v| f64::from((v / d).round().clamp(-127.0, 127.0)) * f64::from(d))
        })
        .collect()
}

/// ik's q8_K activation of `x` ([`quantize_q8_k`]) — each value's `q·d`
/// exact in f64.
#[must_use]
pub fn q8k_exact(x: &[f32]) -> Vec<f64> {
    let a = quantize_q8_k(x);
    a.q.iter()
        .enumerate()
        .map(|(c, &q)| f64::from(q) * f64::from(a.d[c / QK_K]))
        .collect()
}

/// One Q3_K row against one input window, both rules exact in f64 and the
/// bands around them.
#[derive(Clone, Copy, Debug, Default)]
pub struct Q3kRow {
    /// The row against our q8_1 values ([`q8_1_exact`]).
    pub ours: f64,
    /// The row against ik's q8_K values ([`q8k_exact`]).
    pub ik: f64,
    /// How far ik's AVX2 dot ([`dot_q3k`]) may sit from `ik`: `γ(2·n_sb + 4)`
    /// of the magnitudes it sums — per super-block the `u` part and the `−4`
    /// part each enter a lane by one fused multiply-add after one rounded
    /// scale product, and eight lanes meet in a three-level sum.
    pub ik_band: f64,
    /// How far our kernel may sit from ik's dot: the two activations'
    /// distances to `x` weighted by `|w|` (each within half its own step of
    /// `x`), `ik_band`, and the gemv's [`crate::KERNEL_BAND`] of the rows'
    /// largest `|ours|` — the kernel against the exact dot of its own q8_1
    /// values, the band `gate_p1` pins.
    pub band: f64,
}

/// [`Q3kRow`] for every row of `w`, row `r` against window
/// `r / rows_per_window` of `x` (`w.k` values each). The kernel band's
/// largest `|ours|` is over all the rows.
#[must_use]
pub fn q3k_rows(w: &Q3kWeight, x: &[f32], rows_per_window: usize) -> Vec<Q3kRow> {
    let k = w.k;
    let xo = q8_1_exact(x);
    let xi = q8k_exact(x);
    let (xo, xi) = (&xo[..], &xi[..]);
    let n_sb = k / QK_K;
    let mut rows = vec![Q3kRow::default(); w.rows];
    let chunk = w.rows.div_ceil(ROW_THREADS).max(1);
    std::thread::scope(|s| {
        for (c, part) in rows.chunks_mut(chunk).enumerate() {
            s.spawn(move || {
                for (i, o) in part.iter_mut().enumerate() {
                    let j = c * chunk + i;
                    let x0 = j / rows_per_window * k;
                    let (wr, fr) = (&w.deq[j * k..(j + 1) * k], &w.fields[j * k..(j + 1) * k]);
                    let (mut so, mut si, mut mag, mut qd) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
                    for (((&wv, &(sc, q)), (&a, &b)), &xv) in wr
                        .iter()
                        .zip(fr)
                        .zip(xo[x0..x0 + k].iter().zip(&xi[x0..x0 + k]))
                        .zip(&x[x0..x0 + k])
                    {
                        let wv = f64::from(wv);
                        so += wv * a;
                        si += wv * b;
                        mag += f64::from(sc) * b.abs() * (f64::from(q + 4) + 4.0);
                        qd += wv.abs() * ((a - f64::from(xv)).abs() + (b - f64::from(xv)).abs());
                    }
                    let ik_band = crate::rounding::gamma(2 * n_sb + 4) * mag;
                    *o = Q3kRow {
                        ours: so,
                        ik: si,
                        ik_band,
                        band: qd + ik_band,
                    };
                }
            });
        }
    });
    let big = rows.iter().fold(0.0f64, |a, r| a.max(r.ours.abs()));
    for r in &mut rows {
        r.band += f64::from(crate::KERNEL_BAND) * big;
    }
    rows
}

/// ik's output of every row of `w`, row `r` against window
/// `r / rows_per_window` of `x` (`w.k` values each): the q8_K blocks of `x`
/// ([`quantize_q8_k`]), then [`dot_q3k`] from the window's first block.
#[must_use]
pub fn ik_q3k_rows(w: &Q3kWeight, x: &[f32], rows_per_window: usize) -> Vec<f32> {
    let a = quantize_q8_k(x);
    let (a, sb) = (&a, w.k / QK_K);
    let mut out = vec![0.0f32; w.rows];
    let chunk = w.rows.div_ceil(ROW_THREADS).max(1);
    std::thread::scope(|s| {
        for (c, part) in out.chunks_mut(chunk).enumerate() {
            s.spawn(move || {
                for (i, o) in part.iter_mut().enumerate() {
                    let r = c * chunk + i;
                    *o = dot_q3k(w.row(r), a, r / rows_per_window * sb);
                }
            });
        }
    });
    out
}

// ---- opg-glueeng ----

/// Per row of [`q3k_rows`] (`k` values a row), how far our kernel's output
/// may sit from ik's dot ([`dot_q3k`]) when both sides project the same
/// input, as a gate that pins our kernel within [`crate::KERNEL_BAND`] of
/// the rows' `ours` (as f32, relative to their largest) proves on that run:
/// - the two exact values' distance `|ours − ik|`, computed, in place of
///   [`Q3kRow::band`]'s worst case of the two roundings;
/// - ik's dot within [`Q3kRow::ik_band`] of `ik`;
/// - our kernel within `KERNEL_BAND·max|fl(ours)|` of `fl(ours)`, and
///   `fl(ours)` within `u·|ours|` of `ours`;
/// - the two f64 sums behind `ours` and `ik`, each `k` products rounded
///   once and added, within `γ₆₄(2k)` of `Σ|w·x̂|`: ik's side is under
///   `mag = ik_band/γ(2·n_sb + 4)` (each `|w| = |d·sc|·|q| ≤ |d·sc|·(q + 8)`),
///   ours under `mag + qd` (`|x̂o| ≤ |x̂i| + |x̂o − x| + |x̂i − x|`, `qd` the
///   band's rounding term).
#[must_use]
pub fn q3k_shared_input_bound(rows: &[Q3kRow], k: usize) -> Vec<f64> {
    use crate::rounding::{U, U64, gamma};
    let big = rows.iter().fold(0.0f64, |a, r| a.max(r.ours.abs()));
    let kernel = f64::from(crate::KERNEL_BAND) * big * (1.0 + U);
    let g_ik = gamma(2 * (k / QK_K) + 4);
    let n64 = 2.0 * k as f64 * U64;
    let g64 = n64 / (1.0 - n64);
    rows.iter()
        .map(|r| {
            let mag = r.ik_band / g_ik;
            let qd = (r.band - r.ik_band - f64::from(crate::KERNEL_BAND) * big).max(0.0);
            (r.ours - r.ik).abs() + r.ik_band + kernel + U * r.ours.abs() + g64 * (2.0 * mag + qd)
        })
        .collect()
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

// ---- opg-ffnmoe ----

/// Bytes of one Q4_K super-block: `d`, `dmin`, `scales[12]`, `qs[128]`.
pub const Q4K_BYTES: usize = 144;

/// Bytes of one Q5_K super-block: `d`, `dmin`, `scales[12]`, `qh[32]`,
/// `qs[128]`.
pub const Q5K_BYTES: usize = 176;

/// ik's q8_2 blocks of a row: the codes, one bf16-valued scale per 32
/// values, and each block's `s` field.
#[derive(Clone, Debug, Default)]
pub struct Q82 {
    pub q: Vec<i8>,
    pub d: Vec<f32>,
    pub s: Vec<i16>,
}

/// ik's q8_2 of `x` on AVX2 (`quantize_row_q8_2_x4` →
/// `quantize_row_q8_1_x4_T<block_q8_2, block_q8_2_x4>`, `iqk_quantize.cpp`):
/// the codes and scales of [`crate::ik_q8_2::quantize`], and `s` the sum of
/// the block's 32 codes as `cvtps_epi32` leaves them, before the packs
/// saturate, stored as an i16 (`block_q8_2::s`, `ggml-common.h`). The x4
/// interleave moves bytes only. `x` is whole blocks.
#[must_use]
pub fn quantize_q8_2(x: &[f32]) -> Q82 {
    use crate::ik_q8_2::QK;
    assert!(x.len().is_multiple_of(QK), "q8_2 blocks are {QK} values");
    let (q, d) = crate::ik_q8_2::quantize(x);
    let s = x
        .as_chunks::<QK>()
        .0
        .iter()
        .zip(&d)
        .map(|(blk, &db)| {
            let id = if db > 0.0 { 1.0 / db } else { 0.0 };
            blk.iter()
                .map(|&v| (v * id).round_ties_even() as i32)
                .sum::<i32>() as i16
        })
        .collect();
    Q82 { q, d, s }
}

/// The eight 6-bit scales and eight 6-bit minimums of a Q4_K or Q5_K
/// super-block from its twelve scale bytes (`make_q4_scales`,
/// `iqk_common.h`; `get_scale_min_k4`).
#[must_use]
pub fn q4k_scales(b: &[u8; 12]) -> ([u8; 8], [u8; 8]) {
    let (mut sc, mut mn) = ([0u8; 8], [0u8; 8]);
    for j in 0..4 {
        sc[j] = b[j] & 63;
        mn[j] = b[j + 4] & 63;
        sc[j + 4] = (b[j + 8] & 0x0f) | ((b[j] >> 6) << 4);
        mn[j + 4] = (b[j + 8] >> 4) | ((b[j + 4] >> 6) << 4);
    }
    (sc, mn)
}

/// ik's dot of one Q4_K row (`row`: whole super-blocks of [`Q4K_BYTES`])
/// with q8_2 blocks `b0 ..` of `a` on AVX2: the walk of `dot_kq`.
#[must_use]
pub fn dot_q4k(row: &[u8], a: &Q82, b0: usize) -> f32 {
    assert!(
        row.len().is_multiple_of(Q4K_BYTES),
        "Q4_K rows are whole super-blocks"
    );
    dot_kq(row.as_chunks::<Q4K_BYTES>().0, a, b0, |blk, sb, m| {
        (blk[16 + 32 * (sb / 2) + m] >> (4 * (sb % 2))) & 0x0f
    })
}

/// ik's dot of one Q5_K row (`row`: whole super-blocks of [`Q5K_BYTES`])
/// with q8_2 blocks `b0 ..` of `a` on AVX2: the Q4_K walk, each code's
/// fifth bit bit `sb` of `qh[m]` (`DequantizerQ5K_AVX2::apply_hbits`).
#[must_use]
pub fn dot_q5k(row: &[u8], a: &Q82, b0: usize) -> f32 {
    assert!(
        row.len().is_multiple_of(Q5K_BYTES),
        "Q5_K rows are whole super-blocks"
    );
    dot_kq(row.as_chunks::<Q5K_BYTES>().0, a, b0, |blk, sb, m| {
        ((blk[48 + 32 * (sb / 2) + m] >> (4 * (sb % 2))) & 0x0f) | (((blk[16 + m] >> sb) & 1) << 4)
    })
}

/// `mul_mat_qX_K_q8_2_X4_T` (`iqk_gemm_kquants.cpp:783`, the AVX2 `#else`
/// body) over super-blocks `blocks` whose code `m` of sub-block `sb` is
/// `code(blk, sb, m)`. Eight f32 lanes, zero at the row's start; per
/// super-block, first the minimum's term — lane `k` fuses
/// `(d8ₖ·sₖ) · (−(dmin·mₖ))` — then per 128 values `j` the codes' — lane
/// `L` fuses `((d·sc₄ⱼ₊ₗ)·d8₄ⱼ₊ₗ) · Σ u·q` over values `16·(L/4) .. + 16` of
/// sub-block `4j + l`, `l = L%4`; every product rounds once, every
/// integer exact (`maddubs` pairs stay under 2·31·128 and the `epi16`
/// lane sums of eight products under 2^15). `hsum_float_8` adds lane `t`
/// to `t + 4`, then `(s₀ + s₂) + (s₁ + s₃)`.
fn dot_kq<const N: usize>(
    blocks: &[[u8; N]],
    a: &Q82,
    b0: usize,
    code: impl Fn(&[u8; N], usize, usize) -> u8,
) -> f32 {
    let mut acc = [0.0f32; 8];
    for (i, blk) in blocks.iter().enumerate() {
        let d = half_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let dmin = half_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
        let (sc, mn) = q4k_scales(blk[4..16].try_into().expect("twelve scale bytes"));
        let b = b0 + 8 * i;
        let d8 = &a.d[b..b + 8];
        for (k, l) in acc.iter_mut().enumerate() {
            let my = d8[k] * f32::from(a.s[b + k]);
            let mins = -(dmin * f32::from(mn[k]));
            *l = my.mul_add(mins, *l);
        }
        let scale: [f32; 8] = std::array::from_fn(|k| d * f32::from(sc[k]));
        for j in 0..2 {
            for (lane, l) in acc.iter_mut().enumerate() {
                let sb = 4 * j + lane % 4;
                let q = &a.q[(b + sb) * 32..(b + sb + 1) * 32];
                let h = 16 * (lane / 4);
                let sumi: i32 = (h..h + 16)
                    .map(|m| i32::from(code(blk, sb, m)) * i32::from(q[m]))
                    .sum();
                let dd = scale[sb] * d8[sb];
                *l = dd.mul_add(sumi as f32, *l);
            }
        }
    }
    let s: [f32; 4] = std::array::from_fn(|t| acc[t] + acc[t + 4]);
    (s[0] + s[2]) + (s[1] + s[3])
}

#[cfg(test)]
mod ffnmoe_tests {
    use super::{Q4K_BYTES, Q5K_BYTES, QK_K, dot_q4k, dot_q5k, quantize_q8_2};
    use gguf::quant::{GgmlType, dequant_row};

    /// The codes and scales are [`crate::ik_q8_2::quantize`]'s, and `s` their
    /// sum where no code saturates.
    #[test]
    fn q8_2_sums_are_the_codes() {
        let x = crate::activations(4 * 32, 1, 11);
        let a = quantize_q8_2(&x);
        let (q, d) = crate::ik_q8_2::quantize(&x);
        assert_eq!((a.q.clone(), a.d.clone()), (q, d));
        for (b, &s) in a.s.iter().enumerate() {
            let sum: i32 = a.q[32 * b..32 * (b + 1)]
                .iter()
                .map(|&c| i32::from(c))
                .sum();
            assert_eq!(i32::from(s), sum, "block {b}");
        }
    }

    /// Each dot is the dequantized row against the reconstructed codes, to
    /// f32 accumulation: the minimum's term through `s` included.
    #[test]
    fn dots_are_the_dequantized_dots() {
        for (ty, bytes) in [(GgmlType::Q4_K, Q4K_BYTES), (GgmlType::Q5_K, Q5K_BYTES)] {
            let mut row = vec![0u8; 2 * bytes];
            for (i, v) in row.iter_mut().enumerate() {
                *v = (i as u8).wrapping_mul(73) ^ 0x33;
            }
            for sb in 0..2 {
                row[sb * bytes..sb * bytes + 2].copy_from_slice(&0x2e66u16.to_le_bytes());
                row[sb * bytes + 2..sb * bytes + 4].copy_from_slice(&0x2a00u16.to_le_bytes());
            }
            let x = crate::activations(2 * QK_K, 1, 5);
            let a = quantize_q8_2(&x);
            let xh = crate::ik_q8_2::reconstruct(&x);
            let mut w = vec![0.0f32; 2 * QK_K];
            dequant_row(ty, &row, &mut w).unwrap();
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
            let got = f64::from(match ty {
                GgmlType::Q4_K => dot_q4k(&row, &a, 0),
                _ => dot_q5k(&row, &a, 0),
            });
            assert!(
                (got - exact).abs() <= 1e-5 * abs,
                "{ty:?}: {got} vs {exact}"
            );
        }
    }
}
