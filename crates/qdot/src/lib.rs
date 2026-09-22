//! qdot — fused quantized row dots for host CPU: Q3_K x Q8_K, Q4_K x Q8_2_X4,
//! Q6_K x Q8_2_X4, Q5_0 x Q8_2_X4, Q5_1 x Q8_2_X4, and Q8_0 x act cell kernels.
//!
//! Dots quantized weight rows directly against quantized activation columns without
//! materializing f32 weights, scaling once per block:
//! - Q3_K x Q8_K: port of `ggml_vec_dot_q3_K_q8_K` (ggml-quants.c:6482).
//! - Q4_K x Q8_2_X4: port of ik's `mul_mat_qX_K_q8_2_X4_T` (iqk_gemm_kquants.cpp:787).
//! - Q6_K x Q8_2_X4: port of ik's `mul_mat_qY_K_q8_2_X4_T` (iqk_gemm_kquants.cpp:938).
//! - Q5_0 x Q8_2_X4: port of ik's `mul_mat_qX_1_q8_2_T<Q5_0_1_Unpacker>` (iqk_gemm_legacy_quants.cpp:507).
//! - Q5_1 x Q8_2_X4: port of ik's `mul_mat_qX_1_q8_2_T<Q5_1_Unpacker>` (iqk_gemm_legacy_quants.cpp:804).
//! - Q8_0 x act cells: fused cell kernel for `q_nope2_absorbed` (model::attn).
//!
//! Super-block geometry (block_q3_K, 110 bytes / 256 values): hmask[32] @+0,
//! qs[64] @+32, scales[12] @+96, f16 d @+108.
//!
//! Each AVX2 kernel has a bit-identical scalar mirror for fallback and verification.

use std::arch::x86_64::*;
use std::fmt;

use gguf::GgmlType;
use gguf::quant::half_to_f32;

/// `sizeof(block_q3_K)` (ggml-common.h): 110 bytes / 256 values.
const Q3K_BLOCK: usize = 110;
/// `sizeof(block_q8_K)`: 296 bytes / 256 values.
const Q8K_STRIDE: usize = 296;
/// `sizeof(block_q8_2_x4)` (ggml-common.h:296): 144 bytes / 128 values.
const Q82X4_STRIDE: usize = 144;
/// `sizeof(block_q5_0)` (ggml-common.h:198): 22 bytes / 32 values.
const Q5F0_BLOCK: usize = 22;
/// `sizeof(block_q5_1)` (ggml-common.h:210): 24 bytes / 32 values.
const Q5F1_BLOCK: usize = 24;
/// `sizeof(block_q8_2)`: 36 bytes / 32 values for tail blocks past x4 groups.
const Q82_BLOCK: usize = 36;

// ----------------------------------------------------------- public errors

/// Error conditions for `dot_row`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QdotError {
    /// The weight type has no fused kernel in this build.
    UnsupportedType(GgmlType),
    /// `k` is not a multiple of the weight type's block size (`gran`).
    UnalignedK { k: usize, gran: usize },
    /// The weight row is shorter than `k` values implies.
    ShortWeightRow { have: usize, need: usize, k: usize },
    /// The activation column is shorter than `k` values implies.
    ShortActivationCol { have: usize, need: usize, k: usize },
}

impl fmt::Display for QdotError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QdotError::UnsupportedType(w) => {
                write!(
                    f,
                    "no fused qdot kernel for weight type {w:?} in this build"
                )
            }
            QdotError::UnalignedK { k, gran } => {
                if *gran == 256 {
                    write!(
                        f,
                        "k = {k}: qdot rows are whole 256-value super-blocks (k % 256 != 0)"
                    )
                } else {
                    write!(
                        f,
                        "k = {k}: qdot rows are whole {gran}-value blocks (k % {gran} != 0)"
                    )
                }
            }
            QdotError::ShortWeightRow { have, need, k } => {
                write!(f, "weight row is {have} bytes, k = {k} needs {need}")
            }
            QdotError::ShortActivationCol { have, need, k } => {
                write!(f, "activation column is {have} bytes, k = {k} needs {need}")
            }
        }
    }
}

impl std::error::Error for QdotError {}

// ------------------------------------------------------------ public API

/// Whether the fused path handles this weight type on this machine: the type
/// is supported and the CPU has the required features.
#[must_use]
pub fn supports(w: GgmlType) -> bool {
    matches!(
        w,
        GgmlType::Q3_K | GgmlType::Q4_K | GgmlType::Q5_0 | GgmlType::Q5_1 | GgmlType::Q6_K
    ) && has_features(w)
}

/// The per-type ISA table matching each fused kernel's `#[target_feature]` requirements.
///
/// Q3_K needs avx2; Q4_K and Q6_K need avx2+fma; Q5_0 and Q5_1 need avx2+fma+f16c.
fn has_features(w: GgmlType) -> bool {
    match w {
        // dot_q3k_q8k_avx2: enable = "avx2"
        GgmlType::Q3_K => std::arch::is_x86_feature_detected!("avx2"),
        // dot_q4k_q82x4_avx2 / dot_q6k_q82x4_avx2: enable = "avx2", "fma"
        GgmlType::Q4_K | GgmlType::Q6_K => {
            std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
        }
        // dot_q5f0_q82x4_avx2 / dot_q5f1_q82x4_avx2:
        // enable = "avx2", "fma", "f16c"
        GgmlType::Q5_0 | GgmlType::Q5_1 => {
            std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
                && std::arch::is_x86_feature_detected!("f16c")
        }
        _ => false,
    }
}

/// The `k` contract: `k` must be a multiple of the weight format's block size
/// (256 for K-quants, 32 for Q5_0 and Q5_1).
pub fn k_granularity(w: GgmlType) -> usize {
    match w {
        GgmlType::Q5_0 | GgmlType::Q5_1 => 32,
        _ => 256,
    }
}

/// Bytes one quantized activation column of `k` values occupies in `w`'s format.
///
/// Panics if `w` has no activation format in this build or `k` breaks the
/// type's [`k_granularity`] contract.
#[must_use]
pub fn col_bytes(w: GgmlType, k: usize) -> usize {
    assert!(
        matches!(
            w,
            GgmlType::Q3_K | GgmlType::Q4_K | GgmlType::Q5_0 | GgmlType::Q5_1 | GgmlType::Q6_K
        ),
        "qdot has no activation format for {w:?} in this build"
    );
    let gran = k_granularity(w);
    if gran == 256 {
        assert!(
            k.is_multiple_of(256),
            "qdot activation columns are whole 256-value super-blocks, k = {k}"
        );
    } else {
        assert!(
            k.is_multiple_of(gran),
            "qdot activation columns are whole {gran}-value blocks ({w:?}), k = {k}"
        );
    }
    match w {
        GgmlType::Q5_0 | GgmlType::Q5_1 => (k / 128) * Q82X4_STRIDE + ((k % 128) / 32) * Q82_BLOCK,
        GgmlType::Q4_K | GgmlType::Q6_K => (k / 128) * Q82X4_STRIDE,
        _ => (k / 256) * Q8K_STRIDE,
    }
}

/// Quantize one activation column into the block format `w` implies.
///
/// Q3_K uses `block_q8_K` (296 B/256 values); Q4_K, Q5_0, Q5_1, and Q6_K use
/// `block_q8_2_x4` (144 B/128 values), with 36-byte `block_q8_2` tails for Q5_0/Q5_1.
/// `out.len()` must equal `col_bytes(w, x.len())`; panics otherwise. Uses the
/// AVX2 encoders when the CPU has AVX2 — byte-identical to the scalar mirrors.
pub fn quantize_col(w: GgmlType, x: &[f32], out: &mut [u8]) {
    quantize_col_check(w, x, out);
    let avx2 = std::arch::is_x86_feature_detected!("avx2");
    match w {
        GgmlType::Q4_K | GgmlType::Q5_0 | GgmlType::Q5_1 | GgmlType::Q6_K => {
            if avx2 {
                // SAFETY: AVX2 detected just above; quantize_col_check pinned
                // out.len() to col_bytes(w, x.len()).
                unsafe { quantize_q82x4_col_avx2(x, out) }
            } else {
                quantize_q82x4_col(x, out);
            }
        }
        _ => {
            let (xblocks, _) = x.as_chunks::<256>();
            let (oblocks, _) = out.as_chunks_mut::<296>();
            if avx2 {
                for (xb, ob) in xblocks.iter().zip(oblocks.iter_mut()) {
                    // SAFETY: AVX2 detected just above; fixed-size block arrays.
                    unsafe { quantize_q8k_block_avx2(xb, ob) }
                }
            } else {
                for (xb, ob) in xblocks.iter().zip(oblocks.iter_mut()) {
                    quantize_q8k_block(xb, ob);
                }
            }
        }
    }
}

/// The scalar mirror behind [`quantize_col`] — the fallback the crate runs
/// when AVX2 is absent, and the oracle its bit-identity gate compares against.
pub fn quantize_col_scalar(w: GgmlType, x: &[f32], out: &mut [u8]) {
    quantize_col_check(w, x, out);
    match w {
        GgmlType::Q4_K | GgmlType::Q5_0 | GgmlType::Q5_1 | GgmlType::Q6_K => {
            quantize_q82x4_col(x, out);
        }
        _ => {
            let (xblocks, _) = x.as_chunks::<256>();
            let (oblocks, _) = out.as_chunks_mut::<296>();
            for (xb, ob) in xblocks.iter().zip(oblocks.iter_mut()) {
                quantize_q8k_block(xb, ob);
            }
        }
    }
}

/// The shape contract every `quantize_col` entry point enforces before encoding.
fn quantize_col_check(w: GgmlType, x: &[f32], out: &[u8]) {
    assert!(
        matches!(
            w,
            GgmlType::Q3_K | GgmlType::Q4_K | GgmlType::Q5_0 | GgmlType::Q5_1 | GgmlType::Q6_K
        ),
        "qdot has no activation format for {w:?} in this build"
    );
    let gran = k_granularity(w);
    assert!(
        x.len().is_multiple_of(gran),
        "qdot activation columns are whole {gran}-value blocks ({w:?}), x.len() = {}",
        x.len()
    );
    assert_eq!(
        out.len(),
        col_bytes(w, x.len()),
        "out must be col_bytes(w, x.len())"
    );
}

/// Quantized row dot product: `dot(weight row, quantized activation column)`.
///
/// Returns Err on shape or alignment mismatch. Uses AVX2 kernels when supported,
/// falling back to scalar mirrors.
pub fn dot_row(w: GgmlType, wrow: &[u8], acol: &[u8], k: usize) -> Result<f32, QdotError> {
    let nb = check_row(w, wrow.len(), acol.len(), k)?;
    let hw = has_features(w);
    match (w, hw) {
        (GgmlType::Q3_K, true) => {
            // SAFETY: AVX2 was just detected; lengths validated by check_row.
            Ok(unsafe { dot_q3k_q8k_avx2(wrow, acol, nb) })
        }
        (GgmlType::Q4_K, true) => {
            // SAFETY: AVX2+FMA were just detected; lengths validated above.
            Ok(unsafe { dot_q4k_q82x4_avx2(wrow, acol, nb) })
        }
        (GgmlType::Q6_K, true) => {
            // SAFETY: AVX2+FMA were just detected; lengths validated above.
            Ok(unsafe { dot_q6k_q82x4_avx2(wrow, acol, nb) })
        }
        (GgmlType::Q5_0, true) => {
            // SAFETY: AVX2+FMA+F16C were just detected; lengths validated
            // above.
            Ok(unsafe { dot_q5f0_q82x4_avx2(wrow, acol, nb) })
        }
        (GgmlType::Q5_1, true) => {
            // SAFETY: AVX2+FMA+F16C were just detected; lengths validated
            // above.
            Ok(unsafe { dot_q5f1_q82x4_avx2(wrow, acol, nb) })
        }
        (w, false) => Ok(dot_row_scalar_ty(w, wrow, acol, nb)),
        _ => Ok(dot_row_scalar_ty(w, wrow, acol, nb)),
    }
}

/// The scalar mirror behind [`dot_row`] — the fallback the crate runs when
/// AVX2 is absent.
pub fn dot_row_scalar(w: GgmlType, wrow: &[u8], acol: &[u8], k: usize) -> Result<f32, QdotError> {
    let nb = check_row(w, wrow.len(), acol.len(), k)?;
    Ok(dot_row_scalar_ty(w, wrow, acol, nb))
}

fn dot_row_scalar_ty(w: GgmlType, wrow: &[u8], acol: &[u8], nb: usize) -> f32 {
    match w {
        GgmlType::Q4_K => dot_q4k_q82x4_emul(wrow, acol, nb),
        GgmlType::Q6_K => dot_q6k_q82x4_emul(wrow, acol, nb),
        GgmlType::Q5_0 => dot_q5f0_q82x4_emul(wrow, acol, nb),
        GgmlType::Q5_1 => dot_q5f1_q82x4_emul(wrow, acol, nb),
        _ => dot_q3k_q8k_scalar(wrow, acol, nb),
    }
}

/// Direct AVX2 kernel call behind [`dot_row`]; panics if required features are missing.
pub fn dot_row_avx2(w: GgmlType, wrow: &[u8], acol: &[u8], k: usize) -> Result<f32, QdotError> {
    let nb = check_row(w, wrow.len(), acol.len(), k)?;
    assert!(
        has_features(w),
        "dot_row_avx2 called on a CPU without the kernel's ISA \
         (avx2; +fma for Q4_K/Q6_K; +f16c for Q5_0/Q5_1)"
    );
    match w {
        GgmlType::Q4_K => {
            // SAFETY: asserted just above; lengths validated by check_row.
            Ok(unsafe { dot_q4k_q82x4_avx2(wrow, acol, nb) })
        }
        GgmlType::Q6_K => {
            // SAFETY: asserted just above; lengths validated by check_row.
            Ok(unsafe { dot_q6k_q82x4_avx2(wrow, acol, nb) })
        }
        GgmlType::Q5_0 => {
            // SAFETY: asserted just above; lengths validated by check_row.
            Ok(unsafe { dot_q5f0_q82x4_avx2(wrow, acol, nb) })
        }
        GgmlType::Q5_1 => {
            // SAFETY: asserted just above; lengths validated by check_row.
            Ok(unsafe { dot_q5f1_q82x4_avx2(wrow, acol, nb) })
        }
        _ => {
            // SAFETY: asserted just above; lengths validated by check_row.
            Ok(unsafe { dot_q3k_q8k_avx2(wrow, acol, nb) })
        }
    }
}

/// Shape validation shared by every `dot_row` entry point.
fn check_row(w: GgmlType, wrow_len: usize, acol_len: usize, k: usize) -> Result<usize, QdotError> {
    if !matches!(
        w,
        GgmlType::Q3_K | GgmlType::Q4_K | GgmlType::Q5_0 | GgmlType::Q5_1 | GgmlType::Q6_K
    ) {
        return Err(QdotError::UnsupportedType(w));
    }
    let gran = k_granularity(w);
    if !k.is_multiple_of(gran) {
        return Err(QdotError::UnalignedK { k, gran });
    }
    let (nb, need_w) = match w {
        GgmlType::Q6_K => (k / 256, (k / 256) * Q6K_BLOCK),
        GgmlType::Q5_0 => (k / 32, (k / 32) * Q5F0_BLOCK),
        GgmlType::Q5_1 => (k / 32, (k / 32) * Q5F1_BLOCK),
        GgmlType::Q4_K => (k / 256, (k / 256) * Q4K_BLOCK),
        _ => (k / 256, (k / 256) * Q3K_BLOCK),
    };
    let need_a = col_bytes(w, k);
    if wrow_len < need_w {
        return Err(QdotError::ShortWeightRow {
            have: wrow_len,
            need: need_w,
            k,
        });
    }
    if acol_len < need_a {
        return Err(QdotError::ShortActivationCol {
            have: acol_len,
            need: need_a,
            k,
        });
    }
    Ok(nb)
}

// ----------------------------------------------------------- q8_K encode

/// ggml's `nearest_int` (ggml-quants.c:1726): round-to-nearest via the
/// 2^23 + 2^22 f32 mantissa trick. Identical to `round_ties_even` on the
/// |v| <= 127 domain `iscale * x` lives in, which is what lets gate 1
/// compare restored values bit for bit against
/// `quantize_row_q8_k_roundtrip` (which rounds with `round_ties_even`).
/// Every activation encoder rounds through this: without SSE4.1 in the
/// baseline target `round_ties_even` is a libm `rintf` call per element.
#[inline]
pub fn nearest_int(fval: f32) -> i32 {
    let val = fval + 12582912.0;
    let i = f32::to_bits(val);
    ((i & 0x007f_ffff) as i32) - 0x0040_0000
}

/// Quantizes one 256-value block into 296-byte `block_q8_K` layout.
fn quantize_q8k_block(x: &[f32], out: &mut [u8]) {
    debug_assert_eq!(x.len(), 256);
    debug_assert_eq!(out.len(), Q8K_STRIDE);
    let mut max = 0.0f32;
    let mut amax = 0.0f32;
    for &v in x {
        let ax = v.abs();
        if ax > amax {
            amax = ax;
            max = v;
        }
    }
    out[4..8].fill(0);
    if amax == 0.0 {
        out.fill(0);
        return;
    }
    let iscale = -127.0f32 / max;
    for (j, &v) in x.iter().enumerate() {
        // Negative side reaches -127 by construction; clamps one-sided at +127.
        out[8 + j] = nearest_int(iscale * v).min(127) as i8 as u8;
    }
    for j in 0..16 {
        let mut sum = 0i32;
        for ii in 0..16 {
            sum += out[8 + 16 * j + ii] as i8 as i32;
        }
        out[264 + 2 * j..264 + 2 * j + 2].copy_from_slice(&(sum as i16).to_le_bytes());
    }
    out[0..4].copy_from_slice(&f32::to_le_bytes(1.0 / iscale));
}

/// `nearest_int`, lane for lane: the scalar's 2^23 + 2^22 mantissa extraction
/// as vector integer ops (`y + 12582912.0`, low 23 mantissa bits, minus the
/// bias). On the encoders' |y| <= 127 band this equals `_mm256_cvtps_epi32`
/// under the default round-to-nearest-even MXCSR; keeping the bit form also
/// reproduces the scalar on the ±inf/NaN products of an overflowing scale,
/// where a saturating convert would differ.
///
/// # Safety
/// AVX2 must be available on the target.
#[inline(always)]
unsafe fn v_nearest_int(y: __m256) -> __m256i {
    unsafe {
        // SAFETY: register-only intrinsics, no memory access.
        let z = _mm256_add_ps(y, _mm256_set1_ps(12582912.0));
        let b = _mm256_castps_si256(z);
        _mm256_sub_epi32(
            _mm256_and_si256(b, _mm256_set1_epi32(0x007f_ffff)),
            _mm256_set1_epi32(0x0040_0000),
        )
    }
}

/// Reduce lanes to the `as i8` wrap range [-128, 127]. In-domain codes never
/// reach the wrap (|iscale * v| and |v * id| stay under 127.5 before
/// rounding); the ±inf/NaN products of an overflowing scale extract to
/// -2^22, whose low byte the scalar cast drops to 0 — the wrap reproduces
/// that, where a saturating pack alone would emit -128.
///
/// # Safety
/// AVX2 must be available on the target.
#[inline(always)]
unsafe fn v_wrap_i8(q: __m256i) -> __m256i {
    unsafe {
        // SAFETY: register-only intrinsics, no memory access.
        let t = _mm256_and_si256(q, _mm256_set1_epi32(0xFF));
        let ge = _mm256_cmpgt_epi32(t, _mm256_set1_epi32(127));
        _mm256_sub_epi32(t, _mm256_and_si256(ge, _mm256_set1_epi32(256)))
    }
}

/// Horizontal f32 max over the eight lanes; exact in any order on the finite
/// domain the encoders reduce.
///
/// # Safety
/// SSE3 must be available on the target.
#[inline(always)]
unsafe fn hmax_ps(v: __m256) -> f32 {
    unsafe {
        // SAFETY: register-only intrinsics, no memory access.
        let lo = _mm256_castps256_ps128(v);
        let hi = _mm256_extractf128_ps(v, 1);
        let m = _mm_max_ps(lo, hi);
        let m = _mm_max_ps(m, _mm_movehl_ps(m, m));
        _mm_cvtss_f32(_mm_max_ss(m, _mm_movehdup_ps(m)))
    }
}

/// AVX2 twin of `quantize_q8k_block`, byte-identical by construction. The max
/// pick reproduces the scalar's first-strictly-greater rule: horizontal max,
/// then the first lane whose |v| equals it. Codes round through
/// [`v_nearest_int`] with `iscale * v` a plain `_mm256_mul_ps` — no FMA in
/// the value path — then clamp and wrap in the scalar's `.min(127) as i8`
/// order. The one-sided +127 clamp is dormant in-domain
/// (|fl(iscale) * v| <= 127·(1+2^-24)² < 127.5); both encoders carry it.
///
/// # Safety
/// CPU must support AVX2; `x`/`out` are the fixed 256-value/296-byte block.
#[target_feature(enable = "avx2")]
unsafe fn quantize_q8k_block_avx2(x: &[f32; 256], out: &mut [u8; 296]) {
    // SAFETY: AVX2 present per contract; every load lands inside the 256-value
    // block (whole 8-lane groups, no over-read) and every store inside the
    // 296-byte block.
    unsafe {
        let sgn = _mm256_set1_ps(-0.0);
        let mut m = _mm256_setzero_ps();
        for i in 0..32 {
            // SAFETY: 8-lane load at 8*i <= 248, inside the block.
            let v = _mm256_loadu_ps(x.as_ptr().add(8 * i));
            m = _mm256_max_ps(m, _mm256_andnot_ps(sgn, v));
        }
        let amax = hmax_ps(m);
        if amax == 0.0 {
            *out = [0u8; Q8K_STRIDE];
            return;
        }
        // First lane whose |v| equals amax: the scalar's strict-`>` pick.
        let amax_v = _mm256_set1_ps(amax);
        let mut first = 256usize;
        for i in 0..32 {
            // SAFETY: 8-lane load at 8*i <= 248, inside the block.
            let v = _mm256_loadu_ps(x.as_ptr().add(8 * i));
            let eq = _mm256_cmp_ps(_mm256_andnot_ps(sgn, v), amax_v, _CMP_EQ_OQ);
            let mask = _mm256_movemask_ps(eq) as u32;
            if mask != 0 {
                first = 8 * i + mask.trailing_zeros() as usize;
                break;
            }
        }
        debug_assert!(first < 256, "amax came off these lanes; one must equal it");
        let max = x[first];
        let iscale = -127.0f32 / max;
        let iscale_v = _mm256_set1_ps(iscale);
        let clamp = _mm256_set1_epi32(127);
        for g in 0..8 {
            let mut qi = [_mm256_setzero_si256(); 4];
            for (j, q) in qi.iter_mut().enumerate() {
                // SAFETY: 8-lane load at 32*g + 8*j <= 248, inside the block.
                let v = _mm256_loadu_ps(x.as_ptr().add(32 * g + 8 * j));
                let y = _mm256_mul_ps(v, iscale_v);
                *q = v_wrap_i8(_mm256_min_epi32(v_nearest_int(y), clamp));
            }
            // In-order i32 -> i8: two packs plus the 32-lane fixup; lanes are
            // already in i8 range, so the saturating packs are identity.
            let p0 = _mm256_packs_epi32(qi[0], qi[1]);
            let p1 = _mm256_packs_epi32(qi[2], qi[3]);
            let c = _mm256_packs_epi16(p0, p1);
            let c = _mm256_permutevar8x32_epi32(c, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
            // SAFETY: 32-byte store at 8 + 32*g <= 232, inside the codes span.
            _mm256_storeu_si256(out.as_mut_ptr().add(8 + 32 * g) as *mut __m256i, c);
            // i32 sums of codes 16j..16j+16 off the wrapped lanes; exact in
            // any order at these magnitudes, then the scalar's `as i16`.
            let slo = hsum_i32(_mm256_add_epi32(qi[0], qi[1]));
            let shi = hsum_i32(_mm256_add_epi32(qi[2], qi[3]));
            let base = 264 + 4 * g;
            out[base..base + 2].copy_from_slice(&(slo as i16).to_le_bytes());
            out[base + 2..base + 4].copy_from_slice(&(shi as i16).to_le_bytes());
        }
        out[0..4].copy_from_slice(&f32::to_le_bytes(1.0 / iscale));
        out[4..8].fill(0);
    }
}

// ------------------------------------------------------------- q3_K dot

/// Scale shuffle table for Q3_K dot.
static K_SHUFFLE_Q3K: [u8; 128] = [
    0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, //
    2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, //
    4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, //
    6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, //
    8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, //
    10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, //
    12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, //
    14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, //
];

/// Horizontal i32 sum of an `__m256i`.
///
/// # Safety
/// AVX2 must be available on the target.
#[inline(always)]
unsafe fn hsum_i32(v: __m256i) -> i32 {
    unsafe {
        // SAFETY: register-only intrinsics, no memory access.
        let lo = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
        let lo = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, 0x4E));
        let lo = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, 0x8D));
        _mm_cvtsi128_si32(lo)
    }
}

/// Vector constants the field dot needs, built once per row.
struct Masks {
    m3: __m256i,
    mone: __m256i,
}

/// One field of one 128-value half: decode weights and dot against Q8_K span.
///
/// # Safety
/// `q8` must point at 32 readable bytes inside column buffer; AVX2 must be present.
#[inline(always)]
unsafe fn field_dot<const SHIFT: i32, const BIT: i32>(
    q8: *const u8,
    hbits: __m256i,
    q3bits: __m256i,
    scales_j: __m256i,
    shuf_f: __m256i,
    masks: &Masks,
    sumi: &mut __m256i,
) {
    // SAFETY: AVX2 present and `q8` points at 32 readable bytes per contract.
    unsafe {
        let q3l = _mm256_and_si256(_mm256_srli_epi16::<SHIFT>(q3bits), masks.m3);
        let q3h = _mm256_slli_epi16::<2>(_mm256_srli_epi16::<BIT>(_mm256_andnot_si256(
            hbits,
            _mm256_slli_epi16::<BIT>(masks.mone),
        )));
        let sc = _mm256_shuffle_epi8(scales_j, shuf_f);
        // SAFETY: unaligned 32-byte load at q8, inside the column buffer.
        let q8f = _mm256_loadu_si256(q8 as *const __m256i);
        let q8s = _mm256_maddubs_epi16(q3h, q8f);
        let p = _mm256_maddubs_epi16(q3l, q8f);
        let p = _mm256_sub_epi16(p, q8s);
        let p = _mm256_madd_epi16(sc, p);
        *sumi = _mm256_add_epi32(*sumi, p);
    }
}

/// AVX2 row dot: Q3_K weights against Q8_K column.
///
/// Partial sum is exact integer (|sumi| <= 4.2e6 < 2^24), folded before f32 scaling.
///
/// # Safety
/// CPU must support AVX2; buffers must hold `nb` super-blocks.
#[target_feature(enable = "avx2")]
unsafe fn dot_q3k_q8k_avx2(wrow: &[u8], acol: &[u8], nb: usize) -> f32 {
    // SAFETY: AVX2 present and both slices hold nb super-blocks per contract.
    unsafe {
        let masks = Masks {
            m3: _mm256_set1_epi8(3),
            mone: _mm256_set1_epi8(1),
        };
        let m32 = _mm_set1_epi8(32);
        // SAFETY: four 32-byte loads inside the 128-byte static table.
        let shuf: [__m256i; 4] = std::array::from_fn(|f| {
            _mm256_loadu_si256(K_SHUFFLE_Q3K.as_ptr().add(32 * f) as *const __m256i)
        });
        let kmask1 = 0x0303_0303u32;
        let kmask2 = 0x0f0f_0f0fu32;

        let mut acc = 0.0f32;
        for sb in 0..nb {
            let base = wrow.as_ptr().add(Q3K_BLOCK * sb);
            // SAFETY: unaligned load inside the super-block.
            let hbits = _mm256_loadu_si256(base as *const __m256i);
            // SAFETY: three unaligned u32 reads inside the super-block.
            let aux = [
                (base.add(96) as *const u32).read_unaligned(),
                (base.add(100) as *const u32).read_unaligned(),
                (base.add(104) as *const u32).read_unaligned(),
            ];
            let s128 = _mm_set_epi32(
                (((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4)) as i32,
                (((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4)) as i32,
                ((aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4)) as i32,
                ((aux[0] & kmask2) | ((aux[2] & kmask1) << 4)) as i32,
            );
            let s128 = _mm_sub_epi8(s128, m32);
            let all_scales = _mm256_cvtepi8_epi16(s128);
            let l = _mm256_extracti128_si256(all_scales, 0);
            let h = _mm256_extracti128_si256(all_scales, 1);
            let scales = [_mm256_set_m128i(l, l), _mm256_set_m128i(h, h)];

            // SAFETY: one unaligned u16 read inside the super-block.
            let d = half_to_f32((base.add(108) as *const u16).read_unaligned());

            let q8base = acol.as_ptr().add(sb * Q8K_STRIDE + 8);
            let mut sumi = _mm256_setzero_si256();
            for (j, scj) in scales.iter().enumerate() {
                // SAFETY: unaligned load inside the super-block.
                let q3bits = _mm256_loadu_si256(base.add(32 + 32 * j) as *const __m256i);
                let scj = *scj;
                if j == 0 {
                    field_dot::<0, 0>(q8base, hbits, q3bits, scj, shuf[0], &masks, &mut sumi);
                    field_dot::<2, 1>(
                        q8base.add(32),
                        hbits,
                        q3bits,
                        scj,
                        shuf[1],
                        &masks,
                        &mut sumi,
                    );
                    field_dot::<4, 2>(
                        q8base.add(64),
                        hbits,
                        q3bits,
                        scj,
                        shuf[2],
                        &masks,
                        &mut sumi,
                    );
                    field_dot::<6, 3>(
                        q8base.add(96),
                        hbits,
                        q3bits,
                        scj,
                        shuf[3],
                        &masks,
                        &mut sumi,
                    );
                } else {
                    field_dot::<0, 4>(
                        q8base.add(128),
                        hbits,
                        q3bits,
                        scj,
                        shuf[0],
                        &masks,
                        &mut sumi,
                    );
                    field_dot::<2, 5>(
                        q8base.add(160),
                        hbits,
                        q3bits,
                        scj,
                        shuf[1],
                        &masks,
                        &mut sumi,
                    );
                    field_dot::<4, 6>(
                        q8base.add(192),
                        hbits,
                        q3bits,
                        scj,
                        shuf[2],
                        &masks,
                        &mut sumi,
                    );
                    field_dot::<6, 7>(
                        q8base.add(224),
                        hbits,
                        q3bits,
                        scj,
                        shuf[3],
                        &masks,
                        &mut sumi,
                    );
                }
            }
            // SAFETY: unaligned f32 read inside the column buffer.
            let dcol = (acol.as_ptr().add(sb * Q8K_STRIDE) as *const f32).read_unaligned();
            acc += dcol * d * hsum_i32(sumi) as f32;
        }
        acc
    }
}

/// Scalar mirror of `dot_q3k_q8k_avx2`, bit-identical by construction.
fn dot_q3k_q8k_scalar(wrow: &[u8], acol: &[u8], nb: usize) -> f32 {
    const KMASK1: u32 = 0x0303_0303;
    const KMASK2: u32 = 0x0f0f_0f0f;
    let mut acc = 0.0f32;
    for sb in 0..nb {
        let blk = &wrow[sb * Q3K_BLOCK..sb * Q3K_BLOCK + Q3K_BLOCK];
        let hmask = &blk[0..32];
        let qs = &blk[32..96];
        let mut aux = [0u32; 3];
        for i in 0..3 {
            aux[i] = u32::from_le_bytes(blk[96 + 4 * i..96 + 4 * i + 4].try_into().unwrap());
        }
        let word = |i: usize| match i {
            0 => (aux[0] & KMASK2) | ((aux[2] & KMASK1) << 4),
            1 => (aux[1] & KMASK2) | (((aux[2] >> 2) & KMASK1) << 4),
            2 => ((aux[0] >> 4) & KMASK2) | (((aux[2] >> 4) & KMASK1) << 4),
            _ => ((aux[1] >> 4) & KMASK2) | (((aux[2] >> 6) & KMASK1) << 4),
        };
        let scale = |idx: usize| (word(idx / 4).to_le_bytes()[idx % 4] as i8 - 32) as i32;

        let d = half_to_f32(u16::from_le_bytes([blk[108], blk[109]]));
        let q8 = &acol[sb * Q8K_STRIDE + 8..sb * Q8K_STRIDE + 8 + 256];
        let dcol = f32::from_le_bytes(
            acol[sb * Q8K_STRIDE..sb * Q8K_STRIDE + 4]
                .try_into()
                .unwrap(),
        );
        let mut sumi = 0i32;
        let mut m = 1u8;
        for half in 0..2 {
            let mut shift = 0u32;
            for field in 0..4 {
                for h16 in 0..2 {
                    let sc = scale(8 * half + 2 * field + h16);
                    for l in 0..16 {
                        let qv = ((qs[32 * half + 16 * h16 + l] >> shift) & 3) as i32;
                        let hv = if hmask[16 * h16 + l] & m != 0 { 0 } else { 4 };
                        let a8 = q8[128 * half + 32 * field + 16 * h16 + l] as i8 as i32;
                        sumi += sc * (qv - hv) * a8;
                    }
                }
                shift += 2;
                m <<= 1;
            }
        }
        acc += dcol * d * sumi as f32;
    }
    acc
}

// --------------------------------------------------------- Q4_K x Q8_2_X4

/// `block_q4_K` (ggml-common.h): d f16 @0, dmin f16 @2, scales u8[12] @4, qs u8[128] @16.
const Q4K_BLOCK: usize = 144;

/// Unpacks 12 packed scale/min bytes into scales (bytes 0..8) and mins (bytes 8..16).
#[inline]
fn make_q4_scales(scales12: &[u8]) -> [u32; 4] {
    let a0 = u32::from_le_bytes([scales12[0], scales12[1], scales12[2], scales12[3]]);
    let a1 = u32::from_le_bytes([scales12[4], scales12[5], scales12[6], scales12[7]]);
    let a2 = u32::from_le_bytes([scales12[8], scales12[9], scales12[10], scales12[11]]);
    [
        a0 & 0x3f3f_3f3f,
        (a2 & 0x0f0f_0f0f) | ((a0 >> 2) & 0x3030_3030),
        a1 & 0x3f3f_3f3f,
        ((a2 >> 4) & 0x0f0f_0f0f) | ((a1 >> 2) & 0x3030_3030),
    ]
}

/// Converts fp32 to bf16 bits with round to nearest even.
#[inline]
fn fp32_to_bf16_bits(s: f32) -> u16 {
    let u = s.to_bits();
    if (u & 0x7fff_ffff) > 0x7f80_0000 {
        ((u >> 16) | 64) as u16
    } else {
        (u.wrapping_add(0x7fff + ((u >> 16) & 1)) >> 16) as u16
    }
}

/// Converts bf16 bits to f32.
#[inline]
fn bf16_bits_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

/// Quantizes activations into Q8_2_X4 column layout.
///
/// Leftover blocks past 128-multiples are stored as standard 36-byte block_q8_2.
fn quantize_q82x4_col(x: &[f32], out: &mut [u8]) {
    let (blocks, _) = x.as_chunks::<32>();
    let nb4 = 4 * (blocks.len() / 4);
    let (groups, _) = out[..nb4 / 4 * Q82X4_STRIDE].as_chunks_mut::<Q82X4_STRIDE>();
    for (gi, group) in groups.iter_mut().enumerate() {
        group.fill(0);
        for ir in 0..4 {
            let xb = &blocks[4 * gi + ir];
            let mut amax = 0.0f32;
            for &v in xb {
                amax = amax.max(v.abs());
            }
            let t = fp32_to_bf16_bits(amax / 127.0);
            let d = bf16_bits_to_f32(t);
            group[2 * ir..2 * ir + 2].copy_from_slice(&t.to_le_bytes());
            let id = if d > 0.0 { 1.0 / d } else { 0.0 };
            let qs = &mut group[16 + 32 * ir..16 + 32 * ir + 32];
            let mut isum = 0i32;
            for (m, &v) in xb.iter().enumerate() {
                let q = nearest_int(v * id) as i8;
                qs[m] = q as u8;
                isum += q as i32;
            }
            group[8 + 2 * ir..8 + 2 * ir + 2].copy_from_slice(&(isum as i16).to_le_bytes());
        }
    }
    for (t, xb) in blocks[nb4..].iter().enumerate() {
        let tb = &mut out[nb4 / 4 * Q82X4_STRIDE + t * Q82_BLOCK..][..Q82_BLOCK];
        let mut amax = 0.0f32;
        for &v in xb {
            amax = amax.max(v.abs());
        }
        let b = fp32_to_bf16_bits(amax / 127.0);
        let d = bf16_bits_to_f32(b);
        tb[0..2].copy_from_slice(&b.to_le_bytes());
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        let qs = &mut tb[4..36];
        let mut isum = 0i32;
        for (m, &v) in xb.iter().enumerate() {
            let q = nearest_int(v * id) as i8;
            qs[m] = q as u8;
            isum += q as i32;
        }
        tb[2..4].copy_from_slice(&(isum as i16).to_le_bytes());
    }
}

/// One 32-value block of the q8_2 family: (codes, bf16 d bits, i16 isum),
/// byte-identical to the scalar loops in `quantize_q82x4_col`. `v * id` is a
/// plain multiply (no FMA); codes carry no clamp — the scalar's bare `as i8`
/// and the wrap-then-pack here agree because bf16's round-down is at most
/// 2^-8 relative, keeping |v * id| < 127.5 before rounding, while the
/// ±inf/NaN products of an overflowing id wrap to 0 through
/// [`v_nearest_int`] + [`v_wrap_i8`].
///
/// # Safety
/// AVX2 must be available on the target; `x` is the fixed 32-value block.
#[inline(always)]
unsafe fn quantize_q82_block_avx2(x: &[f32; 32]) -> ([u8; 32], u16, i16) {
    // SAFETY: AVX2 present per contract; loads stay inside the fixed 32-value
    // block, the one store goes to a local array.
    unsafe {
        let sgn = _mm256_set1_ps(-0.0);
        let mut m = _mm256_setzero_ps();
        for j in 0..4 {
            // SAFETY: 8-lane load at 8*j <= 24, inside the block.
            let v = _mm256_loadu_ps(x.as_ptr().add(8 * j));
            m = _mm256_max_ps(m, _mm256_andnot_ps(sgn, v));
        }
        let amax = hmax_ps(m);
        // The bf16 scale round-trip stays scalar: one conversion per block.
        let t = fp32_to_bf16_bits(amax / 127.0);
        let d = bf16_bits_to_f32(t);
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        let idv = _mm256_set1_ps(id);
        let mut qi = [_mm256_setzero_si256(); 4];
        for (j, q) in qi.iter_mut().enumerate() {
            // SAFETY: 8-lane load at 8*j <= 24, inside the block.
            let v = _mm256_loadu_ps(x.as_ptr().add(8 * j));
            let y = _mm256_mul_ps(v, idv);
            *q = v_wrap_i8(v_nearest_int(y));
        }
        // In-order i32 -> i8: two packs plus the 32-lane fixup; lanes are
        // already in i8 range, so the saturating packs are identity.
        let p0 = _mm256_packs_epi32(qi[0], qi[1]);
        let p1 = _mm256_packs_epi32(qi[2], qi[3]);
        let c = _mm256_packs_epi16(p0, p1);
        let c = _mm256_permutevar8x32_epi32(c, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
        let mut codes = [0u8; 32];
        // SAFETY: 32-byte store into the local 32-byte array.
        _mm256_storeu_si256(codes.as_mut_ptr() as *mut __m256i, c);
        // i32 sum of the 32 wrapped codes; exact in any order at these
        // magnitudes, then the scalar's `as i16`.
        let isum = hsum_i32(_mm256_add_epi32(
            _mm256_add_epi32(qi[0], qi[1]),
            _mm256_add_epi32(qi[2], qi[3]),
        ));
        (codes, t, isum as i16)
    }
}

/// AVX2 twin of `quantize_q82x4_col`, byte-identical by construction: the
/// same group/tail split, offsets and zero fill, block bodies through
/// [`quantize_q82_block_avx2`].
///
/// # Safety
/// CPU must support AVX2; `x.len()` must be a multiple of 32 and
/// `out.len()` equal `col_bytes(w, x.len())` (both enforced by
/// [`quantize_col_check`]).
#[target_feature(enable = "avx2")]
unsafe fn quantize_q82x4_col_avx2(x: &[f32], out: &mut [u8]) {
    // SAFETY: AVX2 present per contract; whole 8-lane loads of each 32-value
    // block (no over-read of x) and writes confined to the sized-out column.
    unsafe {
        let (blocks, _) = x.as_chunks::<32>();
        let nb4 = 4 * (blocks.len() / 4);
        let (groups, _) = out[..nb4 / 4 * Q82X4_STRIDE].as_chunks_mut::<Q82X4_STRIDE>();
        for (gi, group) in groups.iter_mut().enumerate() {
            group.fill(0);
            for ir in 0..4 {
                let (codes, d, isum) = quantize_q82_block_avx2(&blocks[4 * gi + ir]);
                group[2 * ir..2 * ir + 2].copy_from_slice(&d.to_le_bytes());
                group[8 + 2 * ir..8 + 2 * ir + 2].copy_from_slice(&isum.to_le_bytes());
                group[16 + 32 * ir..16 + 32 * ir + 32].copy_from_slice(&codes);
            }
        }
        for (t, xb) in blocks[nb4..].iter().enumerate() {
            let tb = &mut out[nb4 / 4 * Q82X4_STRIDE + t * Q82_BLOCK..][..Q82_BLOCK];
            let (codes, d, isum) = quantize_q82_block_avx2(xb);
            tb[0..2].copy_from_slice(&d.to_le_bytes());
            tb[2..4].copy_from_slice(&isum.to_le_bytes());
            tb[4..36].copy_from_slice(&codes);
        }
    }
}

/// Converts f16 bits to f32.
#[inline]
fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if frac == 0 {
            sign
        } else {
            // Subnormal f16: normalize.
            let mut e = 127 - 15 + 1i32;
            let mut f = frac;
            while f & 0x400 == 0 {
                f <<= 1;
                e -= 1;
            }
            sign | ((e as u32) << 23) | ((f & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (frac << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (frac << 13)
    };
    f32::from_bits(bits)
}

/// Horizontal f32 sum of an `__m256`. Reduction order is load-bearing.
///
/// # Safety
/// AVX and SSE3 must be available on the target.
#[inline(always)]
unsafe fn hsum_float_8(x: __m256) -> f32 {
    // SAFETY: AVX and SSE3 present per contract; register-only, no memory is touched.
    unsafe {
        let mut res = _mm256_extractf128_ps(x, 1);
        res = _mm_add_ps(res, _mm256_castps256_ps128(x));
        res = _mm_add_ps(res, _mm_movehl_ps(res, res));
        res = _mm_add_ss(res, _mm_movehdup_ps(res));
        _mm_cvtss_f32(res)
    }
}

/// AVX2+FMA row dot: Q4_K weights against Q8_2_X4 column.
/// i16: codes 0..15 against activations in [-127, 127], a maddubs pair peaks at
/// 2*15*127 = 3810; saturation is unreachable.
///
/// # Safety
/// Caller must ensure AVX2+FMA are available and buffers match `nb` blocks.
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_q4k_q82x4_avx2(wrow: &[u8], acol: &[u8], nb: usize) -> f32 {
    // SAFETY: AVX2+FMA present and both slices hold nb blocks per contract.
    unsafe {
        let ml = _mm256_set1_epi8(0xF);
        let mut accd = _mm256_setzero_ps();

        for i in 0..nb {
            let wb = &wrow[i * Q4K_BLOCK..(i + 1) * Q4K_BLOCK];

            let d = f16_bits_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
            let dmin = f16_bits_to_f32(u16::from_le_bytes([wb[2], wb[3]]));

            let utmp = make_q4_scales(&wb[4..16]);
            // SAFETY: 8 readable bytes inside the local u32[4].
            let mins_v = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(
                utmp.as_ptr().add(2) as *const __m128i,
            )));
            let mins = _mm256_mul_ps(_mm256_set1_ps(-dmin), mins_v);

            // SAFETY: 8 readable bytes at the head of each validated group.
            let g0 = acol.as_ptr().add((2 * i) * Q82X4_STRIDE);
            let g1 = acol.as_ptr().add((2 * i + 1) * Q82X4_STRIDE);
            let d4_1 = _mm_cvtepu16_epi32(_mm_loadl_epi64(g0 as *const __m128i));
            let d4_2 = _mm_cvtepu16_epi32(_mm_loadl_epi64(g1 as *const __m128i));
            let dy = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_set_m128i(d4_2, d4_1), 16));
            let dy4 = [_mm256_castps256_ps128(dy), _mm256_extractf128_ps(dy, 1)];
            // SAFETY: 8 readable bytes at offset 8 of each group.
            let m4_1 = _mm_cvtepi16_epi32(_mm_loadl_epi64(g0.add(8) as *const __m128i));
            let m4_2 = _mm_cvtepi16_epi32(_mm_loadl_epi64(g1.add(8) as *const __m128i));
            let myi = _mm256_set_m128i(m4_2, m4_1);
            let my = _mm256_mul_ps(dy, _mm256_cvtepi32_ps(myi));

            // SAFETY: 8 readable bytes inside the local u32[4].
            let all_scales = _mm256_mul_ps(
                _mm256_set1_ps(d),
                _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(
                    utmp.as_ptr() as *const __m128i
                ))),
            );
            let lo = _mm256_castps256_ps128(all_scales);
            let hi = _mm256_extractf128_ps(all_scales, 1);
            let scales = [_mm256_set_m128(lo, lo), _mm256_set_m128(hi, hi)];
            let d4d8 = [
                _mm256_mul_ps(scales[0], _mm256_set_m128(dy4[0], dy4[0])),
                _mm256_mul_ps(scales[1], _mm256_set_m128(dy4[1], dy4[1])),
            ];

            let q4 = &wb[16..16 + 128];
            let mut sumi_f = [_mm256_setzero_ps(), _mm256_setzero_ps()];
            for j in 0..2 {
                // SAFETY: loads stay inside the validated 144-byte block.
                let bits0 = _mm256_loadu_si256(q4.as_ptr().add(64 * j) as *const __m256i);
                let values0 = _mm256_and_si256(bits0, ml);
                let values1 = _mm256_and_si256(_mm256_srli_epi16(bits0, 4), ml);
                let bits1 = _mm256_loadu_si256(q4.as_ptr().add(64 * j + 32) as *const __m256i);
                let values2 = _mm256_and_si256(bits1, ml);
                let values3 = _mm256_and_si256(_mm256_srli_epi16(bits1, 4), ml);

                // SAFETY: loads stay inside the validated activation group.
                let qs = acol.as_ptr().add((2 * i + j) * Q82X4_STRIDE + 16);
                let sumi1 = _mm256_maddubs_epi16(values0, _mm256_loadu_si256(qs as *const __m256i));
                let sumi2 =
                    _mm256_maddubs_epi16(values1, _mm256_loadu_si256(qs.add(32) as *const __m256i));
                let sumi3 =
                    _mm256_maddubs_epi16(values2, _mm256_loadu_si256(qs.add(64) as *const __m256i));
                let sumi4 =
                    _mm256_maddubs_epi16(values3, _mm256_loadu_si256(qs.add(96) as *const __m256i));
                let t1 = _mm256_add_epi16(
                    _mm256_unpacklo_epi32(sumi1, sumi2),
                    _mm256_unpackhi_epi32(sumi1, sumi2),
                );
                let t3 = _mm256_add_epi16(
                    _mm256_unpacklo_epi32(sumi3, sumi4),
                    _mm256_unpackhi_epi32(sumi3, sumi4),
                );
                let t =
                    _mm256_add_epi16(_mm256_unpacklo_epi64(t1, t3), _mm256_unpackhi_epi64(t1, t3));
                let sumi = _mm256_madd_epi16(_mm256_set1_epi16(1), t);
                sumi_f[j] = _mm256_cvtepi32_ps(sumi);
            }

            accd = _mm256_fmadd_ps(my, mins, accd);
            accd = _mm256_fmadd_ps(d4d8[0], sumi_f[0], accd);
            accd = _mm256_fmadd_ps(d4d8[1], sumi_f[1], accd);
        }

        hsum_float_8(accd)
    }
}

// Instruction-graph emulator of `dot_q4k_q82x4_avx2` for no-AVX2 fallback.

/// `_mm256_maddubs_epi16` emulator.
#[inline]
fn emul_maddubs(a: [u8; 32], b: [u8; 32]) -> [i16; 16] {
    let mut r = [0i16; 16];
    for p in 0..16 {
        let u0 = a[2 * p] as i32;
        let u1 = a[2 * p + 1] as i32;
        let s0 = b[2 * p] as i8 as i32;
        let s1 = b[2 * p + 1] as i8 as i32;
        r[p] = (u0 * s0 + u1 * s1).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
    }
    r
}

/// `_mm256_add_epi16`, wrapping.
#[inline]
fn emul_add_epi16(a: [i16; 16], b: [i16; 16]) -> [i16; 16] {
    let mut r = [0i16; 16];
    for p in 0..16 {
        r[p] = a[p].wrapping_add(b[p]);
    }
    r
}

/// `_mm256_madd_epi16(set1(1), t)` emulator.
#[inline]
fn emul_madd1(t: [i16; 16]) -> [i32; 8] {
    let mut r = [0i32; 8];
    for p in 0..8 {
        r[p] = t[2 * p] as i32 + t[2 * p + 1] as i32;
    }
    r
}

/// `_mm256_unpacklo_epi32` on i16 lanes.
#[inline]
fn emul_unpacklo_epi32(a: [i16; 16], b: [i16; 16]) -> [i16; 16] {
    let mut r = [0i16; 16];
    for h in 0..2 {
        let o = 8 * h;
        r[o..o + 2].copy_from_slice(&a[o..o + 2]);
        r[o + 2..o + 4].copy_from_slice(&b[o..o + 2]);
        r[o + 4..o + 6].copy_from_slice(&a[o + 2..o + 4]);
        r[o + 6..o + 8].copy_from_slice(&b[o + 2..o + 4]);
    }
    r
}

/// `_mm256_unpackhi_epi32` on i16 lanes.
#[inline]
fn emul_unpackhi_epi32(a: [i16; 16], b: [i16; 16]) -> [i16; 16] {
    let mut r = [0i16; 16];
    for h in 0..2 {
        let o = 8 * h;
        r[o..o + 2].copy_from_slice(&a[o + 4..o + 6]);
        r[o + 2..o + 4].copy_from_slice(&b[o + 4..o + 6]);
        r[o + 4..o + 6].copy_from_slice(&a[o + 6..o + 8]);
        r[o + 6..o + 8].copy_from_slice(&b[o + 6..o + 8]);
    }
    r
}

/// `_mm256_unpacklo_epi64` on i16 lanes.
#[inline]
fn emul_unpacklo_epi64(a: [i16; 16], b: [i16; 16]) -> [i16; 16] {
    let mut r = [0i16; 16];
    for h in 0..2 {
        let o = 8 * h;
        r[o..o + 4].copy_from_slice(&a[o..o + 4]);
        r[o + 4..o + 8].copy_from_slice(&b[o..o + 4]);
    }
    r
}

/// `_mm256_unpackhi_epi64` on i16 lanes.
#[inline]
fn emul_unpackhi_epi64(a: [i16; 16], b: [i16; 16]) -> [i16; 16] {
    let mut r = [0i16; 16];
    for h in 0..2 {
        let o = 8 * h;
        r[o..o + 4].copy_from_slice(&a[o + 4..o + 8]);
        r[o + 4..o + 8].copy_from_slice(&b[o + 4..o + 8]);
    }
    r
}

/// Emulates `dot_q4k_q82x4_avx2` with bit identity.
fn dot_q4k_q82x4_emul(wrow: &[u8], acol: &[u8], nb: usize) -> f32 {
    let mut accd = [0.0f32; 8];

    for i in 0..nb {
        let wb = &wrow[i * Q4K_BLOCK..(i + 1) * Q4K_BLOCK];
        let g0 = &acol[(2 * i) * Q82X4_STRIDE..(2 * i + 1) * Q82X4_STRIDE];
        let g1 = &acol[(2 * i + 1) * Q82X4_STRIDE..(2 * i + 2) * Q82X4_STRIDE];

        let d = f16_bits_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
        let dmin = f16_bits_to_f32(u16::from_le_bytes([wb[2], wb[3]]));

        let utmp = make_q4_scales(&wb[4..16]);
        let sb = [utmp[0].to_le_bytes(), utmp[1].to_le_bytes()];
        let mb = [utmp[2].to_le_bytes(), utmp[3].to_le_bytes()];

        let mut mins = [0.0f32; 8];
        for l in 0..8 {
            let m = mb[l / 4][l % 4];
            mins[l] = -1.0 * (dmin * (m as f32));
        }

        let mut d8 = [0.0f32; 8];
        for l in 0..4 {
            d8[l] = bf16_bits_to_f32(u16::from_le_bytes([g0[2 * l], g0[2 * l + 1]]));
            d8[4 + l] = bf16_bits_to_f32(u16::from_le_bytes([g1[2 * l], g1[2 * l + 1]]));
        }
        for l in 0..4 {
            let s0 = i16::from_le_bytes([g0[8 + 2 * l], g0[9 + 2 * l]]) as f32;
            let s1 = i16::from_le_bytes([g1[8 + 2 * l], g1[9 + 2 * l]]) as f32;
            accd[l] = (d8[l] * s0).mul_add(mins[l], accd[l]);
            accd[4 + l] = (d8[4 + l] * s1).mul_add(mins[4 + l], accd[4 + l]);
        }

        let mut all_scales = [0.0f32; 8];
        for l in 0..8 {
            all_scales[l] = d * (sb[l / 4][l % 4] as f32);
        }

        let q4 = &wb[16..16 + 128];
        for j in 0..2 {
            let mut values = [[0u8; 32]; 4];
            for m in 0..32 {
                let b0 = q4[64 * j + m];
                let b1 = q4[64 * j + 32 + m];
                values[0][m] = b0 & 0xF;
                values[1][m] = b0 >> 4;
                values[2][m] = b1 & 0xF;
                values[3][m] = b1 >> 4;
            }
            let g = &acol[(2 * i + j) * Q82X4_STRIDE..(2 * i + j + 1) * Q82X4_STRIDE];
            let mut qs = [[0u8; 32]; 4];
            for t in 0..4 {
                qs[t].copy_from_slice(&g[16 + 32 * t..16 + 32 * t + 32]);
            }
            let sumi1 = emul_maddubs(values[0], qs[0]);
            let sumi2 = emul_maddubs(values[1], qs[1]);
            let sumi3 = emul_maddubs(values[2], qs[2]);
            let sumi4 = emul_maddubs(values[3], qs[3]);
            let t1 = emul_add_epi16(
                emul_unpacklo_epi32(sumi1, sumi2),
                emul_unpackhi_epi32(sumi1, sumi2),
            );
            let t3 = emul_add_epi16(
                emul_unpacklo_epi32(sumi3, sumi4),
                emul_unpackhi_epi32(sumi3, sumi4),
            );
            let t = emul_add_epi16(emul_unpacklo_epi64(t1, t3), emul_unpackhi_epi64(t1, t3));
            let sumi = emul_madd1(t);

            for l in 0..8 {
                let idx = 4 * j + (l & 3);
                let d4d8 = all_scales[idx] * d8[idx];
                accd[l] = d4d8.mul_add(sumi[l] as f32, accd[l]);
            }
        }
    }

    let t = [
        accd[0] + accd[4],
        accd[1] + accd[5],
        accd[2] + accd[6],
        accd[3] + accd[7],
    ];
    (t[0] + t[2]) + (t[1] + t[3])
}

// --------------------------------------------------------- Q6_K x Q8_2_X4

/// `block_q6_K` (ggml-common.h:386): ql u8[128] @0, qh u8[64] @128, scales s8[16] @192, d f16 @208.
const Q6K_BLOCK: usize = 210;

/// Shuffle table for Q6_K scale interleave.
static K_SHUFFLE_Q6K: [u8; 32] = [
    0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15, //
    0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15,
];

/// AVX2+FMA row dot: Q6_K weights against Q8_2_X4 column. Sign fold (qY form):
/// |code| is maddubs' u8 side, the code's sign is folded into the activation.
/// i16: codes -32..31 against activations in [-127, 127], a maddubs pair peaks
/// at 2*32*127 = 8128; saturation is unreachable.
///
/// # Safety
/// Caller must ensure AVX2+FMA are available and buffers match `nb` blocks.
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_q6k_q82x4_avx2(wrow: &[u8], acol: &[u8], nb: usize) -> f32 {
    // SAFETY: AVX2+FMA present and both slices hold nb blocks per contract.
    unsafe {
        let ml = _mm256_set1_epi8(0xF);
        let mh = _mm256_set1_epi8(0x30);
        let m32n = _mm256_set1_epi8(-32);
        // SAFETY: 32-byte load inside the 32-byte static table.
        let shuff = _mm256_loadu_si256(K_SHUFFLE_Q6K.as_ptr() as *const __m256i);
        let mut accd = _mm256_setzero_ps();

        for i in 0..nb {
            let wb = wrow.as_ptr().add(Q6K_BLOCK * i);

            // SAFETY: one unaligned u16 read inside the super-block.
            let d = f16_bits_to_f32((wb.add(208) as *const u16).read_unaligned());
            let vd = _mm256_set1_ps(d);
            // SAFETY: 16-byte unaligned load inside the super-block.
            let sc16 = _mm256_shuffle_epi8(
                _mm256_cvtepi8_epi16(_mm_loadu_si128(wb.add(192) as *const __m128i)),
                shuff,
            );
            let mut scales = [
                _mm256_mul_ps(
                    vd,
                    _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(sc16))),
                ),
                _mm256_mul_ps(
                    vd,
                    _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(sc16, 1))),
                ),
            ];

            // SAFETY: 8 readable bytes at the head of each validated group.
            let g0 = acol.as_ptr().add((2 * i) * Q82X4_STRIDE);
            let g1 = acol.as_ptr().add((2 * i + 1) * Q82X4_STRIDE);
            let d4_1 = _mm_cvtepu16_epi32(_mm_loadl_epi64(g0 as *const __m128i));
            let d4_2 = _mm_cvtepu16_epi32(_mm_loadl_epi64(g1 as *const __m128i));
            let dy = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_set_m128i(d4_2, d4_1), 16));
            let dyl = _mm256_castps256_ps128(dy);
            let dyh = _mm256_extractf128_ps(dy, 1);
            scales[0] = _mm256_mul_ps(scales[0], _mm256_set_m128(dyl, dyl));
            scales[1] = _mm256_mul_ps(scales[1], _mm256_set_m128(dyh, dyh));

            let mut sumis = [_mm256_setzero_si256(), _mm256_setzero_si256()];
            for j in 0..2 {
                // SAFETY: loads stay inside the validated 210-byte block.
                let lbits1 = _mm256_loadu_si256(wb.add(64 * j) as *const __m256i);
                let lbits2 = _mm256_loadu_si256(wb.add(64 * j + 32) as *const __m256i);
                let hbits = _mm256_loadu_si256(wb.add(128 + 32 * j) as *const __m256i);
                let mut values = [
                    _mm256_or_si256(
                        _mm256_and_si256(lbits1, ml),
                        _mm256_and_si256(_mm256_slli_epi16::<4>(hbits), mh),
                    ),
                    _mm256_or_si256(
                        _mm256_and_si256(lbits2, ml),
                        _mm256_and_si256(_mm256_slli_epi16::<2>(hbits), mh),
                    ),
                    _mm256_or_si256(
                        _mm256_and_si256(_mm256_srli_epi16::<4>(lbits1), ml),
                        _mm256_and_si256(hbits, mh),
                    ),
                    _mm256_or_si256(
                        _mm256_and_si256(_mm256_srli_epi16::<4>(lbits2), ml),
                        _mm256_and_si256(_mm256_srli_epi16::<2>(hbits), mh),
                    ),
                ];
                let mut us = [_mm256_setzero_si256(); 4];
                for k in 0..4 {
                    values[k] = _mm256_add_epi8(values[k], m32n);
                    us[k] = _mm256_sign_epi8(values[k], values[k]);
                }

                // SAFETY: loads stay inside the validated activation group.
                let qs = acol.as_ptr().add((2 * i + j) * Q82X4_STRIDE + 16);
                let sumi1 = _mm256_maddubs_epi16(
                    us[0],
                    _mm256_sign_epi8(_mm256_loadu_si256(qs as *const __m256i), values[0]),
                );
                let sumi2 = _mm256_maddubs_epi16(
                    us[1],
                    _mm256_sign_epi8(_mm256_loadu_si256(qs.add(32) as *const __m256i), values[1]),
                );
                let sumi3 = _mm256_maddubs_epi16(
                    us[2],
                    _mm256_sign_epi8(_mm256_loadu_si256(qs.add(64) as *const __m256i), values[2]),
                );
                let sumi4 = _mm256_maddubs_epi16(
                    us[3],
                    _mm256_sign_epi8(_mm256_loadu_si256(qs.add(96) as *const __m256i), values[3]),
                );
                let t1 = _mm256_add_epi16(
                    _mm256_unpacklo_epi32(sumi1, sumi2),
                    _mm256_unpackhi_epi32(sumi1, sumi2),
                );
                let t3 = _mm256_add_epi16(
                    _mm256_unpacklo_epi32(sumi3, sumi4),
                    _mm256_unpackhi_epi32(sumi3, sumi4),
                );
                let t =
                    _mm256_add_epi16(_mm256_unpacklo_epi64(t1, t3), _mm256_unpackhi_epi64(t1, t3));
                sumis[j] = _mm256_madd_epi16(_mm256_set1_epi16(1), t);
            }

            accd = _mm256_fmadd_ps(scales[0], _mm256_cvtepi32_ps(sumis[0]), accd);
            accd = _mm256_fmadd_ps(scales[1], _mm256_cvtepi32_ps(sumis[1]), accd);
        }

        hsum_float_8(accd)
    }
}

// Instruction-graph emulator of `dot_q6k_q82x4_avx2` for no-AVX2 fallback.

/// `_mm256_sign_epi8` emulator.
#[inline]
fn emul_sign_epi8(a: u8, b: i8) -> u8 {
    if b < 0 {
        (a as i8).wrapping_neg() as u8
    } else if b == 0 {
        0
    } else {
        a
    }
}

/// `_mm256_slli_epi16` emulator.
#[inline]
fn emul_slli_epi16<const S: i32>(a: [u8; 32]) -> [u8; 32] {
    let mut r = [0u8; 32];
    for p in 0..16 {
        let v = u16::from_le_bytes([a[2 * p], a[2 * p + 1]]);
        let s = v.wrapping_shl(S as u32);
        r[2 * p..2 * p + 2].copy_from_slice(&s.to_le_bytes());
    }
    r
}

/// `_mm256_srli_epi16` emulator.
#[inline]
fn emul_srli_epi16<const S: i32>(a: [u8; 32]) -> [u8; 32] {
    let mut r = [0u8; 32];
    for p in 0..16 {
        let v = u16::from_le_bytes([a[2 * p], a[2 * p + 1]]);
        let s = v >> S;
        r[2 * p..2 * p + 2].copy_from_slice(&s.to_le_bytes());
    }
    r
}

/// Emulates `dot_q6k_q82x4_avx2` with bit identity.
fn dot_q6k_q82x4_emul(wrow: &[u8], acol: &[u8], nb: usize) -> f32 {
    let mut accd = [0.0f32; 8];

    for i in 0..nb {
        let wb = &wrow[i * Q6K_BLOCK..(i + 1) * Q6K_BLOCK];
        let g0 = &acol[(2 * i) * Q82X4_STRIDE..(2 * i + 1) * Q82X4_STRIDE];
        let g1 = &acol[(2 * i + 1) * Q82X4_STRIDE..(2 * i + 2) * Q82X4_STRIDE];

        let d = f16_bits_to_f32(u16::from_le_bytes([wb[208], wb[209]]));

        let mut reg = [0u8; 32];
        for l in 0..16 {
            reg[2 * l..2 * l + 2].copy_from_slice(&(wb[192 + l] as i8 as i16).to_le_bytes());
        }
        let mut sc16 = [0u8; 32];
        for q in 0..32 {
            sc16[q] = reg[16 * (q / 16) + K_SHUFFLE_Q6K[q] as usize];
        }
        let mut scales = [[0.0f32; 8]; 2];
        for l in 0..8 {
            let lo = i16::from_le_bytes([sc16[2 * l], sc16[2 * l + 1]]) as f32;
            let hi = i16::from_le_bytes([sc16[2 * l + 16], sc16[2 * l + 17]]) as f32;
            scales[0][l] = d * lo;
            scales[1][l] = d * hi;
        }
        let mut d8 = [0.0f32; 8];
        for l in 0..4 {
            d8[l] = bf16_bits_to_f32(u16::from_le_bytes([g0[2 * l], g0[2 * l + 1]]));
            d8[4 + l] = bf16_bits_to_f32(u16::from_le_bytes([g1[2 * l], g1[2 * l + 1]]));
        }
        for l in 0..8 {
            scales[0][l] *= d8[l & 3];
            scales[1][l] *= d8[4 + (l & 3)];
        }

        for j in 0..2 {
            let lbits1: [u8; 32] = wb[64 * j..64 * j + 32].try_into().unwrap();
            let lbits2: [u8; 32] = wb[64 * j + 32..64 * j + 64].try_into().unwrap();
            let hbits: [u8; 32] = wb[128 + 32 * j..128 + 32 * j + 32].try_into().unwrap();
            let hbits4 = emul_slli_epi16::<4>(hbits);
            let hbits2 = emul_slli_epi16::<2>(hbits);
            let lbits1h = emul_srli_epi16::<4>(lbits1);
            let lbits2h = emul_srli_epi16::<4>(lbits2);
            let hbitsh2 = emul_srli_epi16::<2>(hbits);
            let mut values = [[0u8; 32]; 4];
            for m in 0..32 {
                values[0][m] = (lbits1[m] & 0xF) | (hbits4[m] & 0x30);
                values[1][m] = (lbits2[m] & 0xF) | (hbits2[m] & 0x30);
                values[2][m] = (lbits1h[m] & 0xF) | (hbits[m] & 0x30);
                values[3][m] = (lbits2h[m] & 0xF) | (hbitsh2[m] & 0x30);
            }
            let mut us = [[0u8; 32]; 4];
            for k in 0..4 {
                for m in 0..32 {
                    values[k][m] = (values[k][m] as i8).wrapping_add(-32) as u8;
                    us[k][m] = emul_sign_epi8(values[k][m], values[k][m] as i8);
                }
            }

            let g = &acol[(2 * i + j) * Q82X4_STRIDE..(2 * i + j + 1) * Q82X4_STRIDE];
            let mut qs = [[0u8; 32]; 4];
            for t in 0..4 {
                qs[t].copy_from_slice(&g[16 + 32 * t..16 + 32 * t + 32]);
            }
            let mut signed = [[0u8; 32]; 4];
            for k in 0..4 {
                for m in 0..32 {
                    signed[k][m] = emul_sign_epi8(qs[k][m], values[k][m] as i8);
                }
            }
            let sumi1 = emul_maddubs(us[0], signed[0]);
            let sumi2 = emul_maddubs(us[1], signed[1]);
            let sumi3 = emul_maddubs(us[2], signed[2]);
            let sumi4 = emul_maddubs(us[3], signed[3]);
            let t1 = emul_add_epi16(
                emul_unpacklo_epi32(sumi1, sumi2),
                emul_unpackhi_epi32(sumi1, sumi2),
            );
            let t3 = emul_add_epi16(
                emul_unpacklo_epi32(sumi3, sumi4),
                emul_unpackhi_epi32(sumi3, sumi4),
            );
            let t = emul_add_epi16(emul_unpacklo_epi64(t1, t3), emul_unpackhi_epi64(t1, t3));
            let sumi = emul_madd1(t);

            for l in 0..8 {
                accd[l] = scales[j][l].mul_add(sumi[l] as f32, accd[l]);
            }
        }
    }

    let t = [
        accd[0] + accd[4],
        accd[1] + accd[5],
        accd[2] + accd[6],
        accd[3] + accd[7],
    ];
    (t[0] + t[2]) + (t[1] + t[3])
}

// --------------------------------------------------------- Q5_0 x Q8_2_X4

/// `HBitDequantizer` constants for unpacking high bits.
struct Q5HBit {
    shuffle: __m256i,
    mask: __m256i,
    minus1: __m256i,
}

/// One 32-value weight block's codes as unsigned bytes 0..31.
///
/// # Safety
/// `blk` must hold QS_OFF + 16 readable bytes.
#[inline(always)]
unsafe fn q5x_codes<const QH_OFF: usize, const QS_OFF: usize>(
    blk: &[u8],
    m4: __m256i,
    mh: __m256i,
    hb: &Q5HBit,
) -> __m256i {
    // SAFETY: 16 readable bytes inside the block.
    unsafe {
        let aux128 = _mm_loadu_si128(blk.as_ptr().add(QS_OFF) as *const __m128i);
        let nib = _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16::<4>(aux128), aux128), m4);
        let qh = u32::from_le_bytes([
            blk[QH_OFF],
            blk[QH_OFF + 1],
            blk[QH_OFF + 2],
            blk[QH_OFF + 3],
        ]);
        let bits = _mm256_or_si256(
            // SAFETY: register-only intrinsic on a broadcast register.
            _mm256_shuffle_epi8(_mm256_set1_epi32(qh as i32), hb.shuffle),
            hb.mask,
        );
        let high = _mm256_and_si256(_mm256_cmpeq_epi8(bits, hb.minus1), mh);
        _mm256_or_si256(nib, high)
    }
}

/// AVX2+FMA+F16C row dot: Q5_0 weights against Q8_2_X4 column.
///
/// TWIN of [`dot_q5f1_q82x4_avx2`] — the same template with four
/// differences only: block width (22 vs 24), the `q5x_codes` offsets (qh
/// @2/qs @6 vs @4/@8), the scale gather (four d's through one
/// `_mm_cvtph_ps` vs the (d,m) pair shuffle), and the tail min correction
/// (-16·d_w vs the stored m_w). Every edit outside those four must be made
/// in both.
///
/// # Safety
/// Caller must ensure AVX2+FMA+F16C are available and buffers match `nb` blocks.
#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
unsafe fn dot_q5f0_q82x4_avx2(wrow: &[u8], acol: &[u8], nb: usize) -> f32 {
    // SAFETY: AVX2+FMA+F16C present (checked by dot_row) and both slices hold
    // the validated lengths per contract.
    unsafe {
        let m4 = _mm256_set1_epi8(0xF);
        let mh = _mm256_set1_epi8(0x10);
        let hb = Q5HBit {
            shuffle: _mm256_set_epi64x(
                0x0303_0303_0303_0303,
                0x0202_0202_0202_0202,
                0x0101_0101_0101_0101,
                0x0000_0000_0000_0000,
            ),
            mask: _mm256_set1_epi64x(0x7fbf_dfef_f7fb_fdfe),
            minus1: _mm256_set1_epi64x(-1),
        };
        let m1 = _mm256_set1_epi16(1);
        let min16 = _mm_set1_ps(-16.0f32);
        let mut acc = _mm256_setzero_ps();
        let mut accm = _mm_setzero_ps();

        let nbg = nb / 4;
        for i in 0..nbg {
            let b0 = (4 * i) * Q5F0_BLOCK;
            let b1 = (4 * i + 1) * Q5F0_BLOCK;
            let b2 = (4 * i + 2) * Q5F0_BLOCK;
            let b3 = (4 * i + 3) * Q5F0_BLOCK;
            let qx0 = q5x_codes::<2, 6>(&wrow[b0..b0 + Q5F0_BLOCK], m4, mh, &hb);
            let qx1 = q5x_codes::<2, 6>(&wrow[b1..b1 + Q5F0_BLOCK], m4, mh, &hb);
            let qx2 = q5x_codes::<2, 6>(&wrow[b2..b2 + Q5F0_BLOCK], m4, mh, &hb);
            let qx3 = q5x_codes::<2, 6>(&wrow[b3..b3 + Q5F0_BLOCK], m4, mh, &hb);
            // SAFETY: 8 readable bytes inside the validated row for f16 d scales.
            let mut scales8 = [0u8; 8];
            scales8[0..2].copy_from_slice(&wrow[b0..b0 + 2]);
            scales8[2..4].copy_from_slice(&wrow[b1..b1 + 2]);
            scales8[4..6].copy_from_slice(&wrow[b2..b2 + 2]);
            scales8[6..8].copy_from_slice(&wrow[b3..b3 + 2]);
            let s4 = _mm_cvtph_ps(_mm_loadl_epi64(scales8.as_ptr() as *const __m128i));
            let other = _mm256_set_m128(_mm_mul_ps(s4, min16), s4);

            // SAFETY: 8 readable bytes at the head of the validated group.
            let g = acol.as_ptr().add(i * Q82X4_STRIDE);
            let aux_d = _mm_castsi128_ps(_mm_slli_epi32::<16>(_mm_cvtepu16_epi32(
                _mm_loadl_epi64(g as *const __m128i),
            )));
            // SAFETY: 8 readable bytes at offset 8 of the group.
            let aux_m = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_loadl_epi64(
                g.add(8) as *const __m128i
            )));
            let prep = _mm256_set_m128(_mm_mul_ps(aux_d, aux_m), aux_d);
            let s12 = _mm256_mul_ps(other, prep);
            accm = _mm_add_ps(accm, _mm256_extractf128_ps(s12, 1));
            let lo = _mm256_castps256_ps128(s12);
            let dall = _mm256_set_m128(lo, lo);

            // SAFETY: 32 readable bytes at each offset inside the group.
            let p0 = _mm256_madd_epi16(
                m1,
                _mm256_maddubs_epi16(qx0, _mm256_loadu_si256(g.add(16) as *const __m256i)),
            );
            let p1 = _mm256_madd_epi16(
                m1,
                _mm256_maddubs_epi16(qx1, _mm256_loadu_si256(g.add(48) as *const __m256i)),
            );
            let p2 = _mm256_madd_epi16(
                m1,
                _mm256_maddubs_epi16(qx2, _mm256_loadu_si256(g.add(80) as *const __m256i)),
            );
            let p3 = _mm256_madd_epi16(
                m1,
                _mm256_maddubs_epi16(qx3, _mm256_loadu_si256(g.add(112) as *const __m256i)),
            );
            let p01 =
                _mm256_add_epi32(_mm256_unpacklo_epi32(p0, p1), _mm256_unpackhi_epi32(p0, p1));
            let p23 =
                _mm256_add_epi32(_mm256_unpacklo_epi32(p2, p3), _mm256_unpackhi_epi32(p2, p3));
            let pall = _mm256_add_epi32(
                _mm256_unpacklo_epi64(p01, p23),
                _mm256_unpackhi_epi64(p01, p23),
            );
            acc = _mm256_fmadd_ps(dall, _mm256_cvtepi32_ps(pall), acc);
        }

        let nb4 = 4 * nbg;
        for i in nb4..nb {
            let b = i * Q5F0_BLOCK;
            let wb = &wrow[b..b + Q5F0_BLOCK];
            let dw = f16_bits_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
            let qx0 = q5x_codes::<2, 6>(wb, m4, mh, &hb);
            let tb = nb4 / 4 * Q82X4_STRIDE + (i - nb4) * Q82_BLOCK;
            let da = bf16_bits_to_f32(u16::from_le_bytes([acol[tb], acol[tb + 1]]));
            let ma = i16::from_le_bytes([acol[tb + 2], acol[tb + 3]]) as f32;
            let d = dw * da;
            let corr = (-16.0f32 * dw) * (da * ma) * 0.25f32;
            accm = _mm_add_ps(accm, _mm_set1_ps(corr));
            // SAFETY: 32 readable bytes at offset 4 of the tail block.
            let qs = _mm256_loadu_si256(acol.as_ptr().add(tb + 4) as *const __m256i);
            let p0 = _mm256_madd_epi16(m1, _mm256_maddubs_epi16(qx0, qs));
            acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(p0), acc);
        }

        // Reduction order is load-bearing.
        let sum = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
        let x = _mm_add_ps(sum, accm);
        let x = _mm_add_ps(x, _mm_movehl_ps(x, x));
        _mm_cvtss_f32(_mm_add_ss(x, _mm_movehdup_ps(x)))
    }
}

// Instruction-graph emulator of `dot_q5f0_q82x4_avx2` for no-AVX2 fallback.

/// `_mm256_unpacklo_epi32` on i32 lanes.
#[inline]
fn emul_unpacklo_epi32_i32(a: [i32; 8], b: [i32; 8]) -> [i32; 8] {
    let mut r = [0i32; 8];
    for h in 0..2 {
        let o = 4 * h;
        r[o] = a[o];
        r[o + 1] = b[o];
        r[o + 2] = a[o + 1];
        r[o + 3] = b[o + 1];
    }
    r
}

/// `_mm256_unpackhi_epi32` on i32 lanes.
#[inline]
fn emul_unpackhi_epi32_i32(a: [i32; 8], b: [i32; 8]) -> [i32; 8] {
    let mut r = [0i32; 8];
    for h in 0..2 {
        let o = 4 * h;
        r[o] = a[o + 2];
        r[o + 1] = b[o + 2];
        r[o + 2] = a[o + 3];
        r[o + 3] = b[o + 3];
    }
    r
}

/// `_mm256_unpacklo_epi64` on i32 lanes.
#[inline]
fn emul_unpacklo_epi64_i32(a: [i32; 8], b: [i32; 8]) -> [i32; 8] {
    let mut r = [0i32; 8];
    for h in 0..2 {
        let o = 4 * h;
        r[o..o + 2].copy_from_slice(&a[o..o + 2]);
        r[o + 2..o + 4].copy_from_slice(&b[o..o + 2]);
    }
    r
}

/// `_mm256_unpackhi_epi64` on i32 lanes.
#[inline]
fn emul_unpackhi_epi64_i32(a: [i32; 8], b: [i32; 8]) -> [i32; 8] {
    let mut r = [0i32; 8];
    for h in 0..2 {
        let o = 4 * h;
        r[o..o + 2].copy_from_slice(&a[o + 2..o + 4]);
        r[o + 2..o + 4].copy_from_slice(&b[o + 2..o + 4]);
    }
    r
}

/// `_mm256_add_epi32`, wrapping.
#[inline]
fn emul_add_epi32(a: [i32; 8], b: [i32; 8]) -> [i32; 8] {
    let mut r = [0i32; 8];
    for l in 0..8 {
        r[l] = a[l].wrapping_add(b[l]);
    }
    r
}

/// Scalar decode of one 32-value block's codes.
#[inline]
fn q5x_codes_scalar<const QH_OFF: usize, const QS_OFF: usize>(blk: &[u8]) -> [u8; 32] {
    let qh = u32::from_le_bytes([
        blk[QH_OFF],
        blk[QH_OFF + 1],
        blk[QH_OFF + 2],
        blk[QH_OFF + 3],
    ]);
    let nibbles = &blk[QS_OFF..QS_OFF + 16];
    let mut code = [0u8; 32];
    for (v, c) in code.iter_mut().enumerate() {
        *c = if v < 16 {
            nibbles[v] & 0xF
        } else {
            nibbles[v - 16] >> 4
        };
        if (qh >> v) & 1 != 0 {
            *c |= 0x10;
        }
    }
    code
}

/// Emulates `dot_q5f0_q82x4_avx2` with bit identity.
fn dot_q5f0_q82x4_emul(wrow: &[u8], acol: &[u8], nb: usize) -> f32 {
    let mut acc = [0.0f32; 8];
    let mut accm = [0.0f32; 4];

    let nbg = nb / 4;
    for i in 0..nbg {
        let mut qx = [[0u8; 32]; 4];
        let mut dw = [0.0f32; 4];
        for j in 0..4 {
            let wb = &wrow[(4 * i + j) * Q5F0_BLOCK..(4 * i + j + 1) * Q5F0_BLOCK];
            qx[j] = q5x_codes_scalar::<2, 6>(wb);
            dw[j] = f16_bits_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
        }
        let g = &acol[i * Q82X4_STRIDE..(i + 1) * Q82X4_STRIDE];
        let mut da = [0.0f32; 4];
        let mut ma = [0.0f32; 4];
        for l in 0..4 {
            da[l] = bf16_bits_to_f32(u16::from_le_bytes([g[2 * l], g[2 * l + 1]]));
            ma[l] = i16::from_le_bytes([g[8 + 2 * l], g[9 + 2 * l]]) as f32;
        }
        let mut lo = [0.0f32; 4];
        for l in 0..4 {
            lo[l] = dw[l] * da[l];
            accm[l] += (-16.0f32 * dw[l]) * (da[l] * ma[l]);
        }
        let mut p = [[0i32; 8]; 4];
        for (j, pj) in p.iter_mut().enumerate() {
            let mut qs = [0u8; 32];
            qs.copy_from_slice(&g[16 + 32 * j..16 + 32 * j + 32]);
            *pj = emul_madd1(emul_maddubs(qx[j], qs));
        }
        let p01 = emul_add_epi32(
            emul_unpacklo_epi32_i32(p[0], p[1]),
            emul_unpackhi_epi32_i32(p[0], p[1]),
        );
        let p23 = emul_add_epi32(
            emul_unpacklo_epi32_i32(p[2], p[3]),
            emul_unpackhi_epi32_i32(p[2], p[3]),
        );
        let pall = emul_add_epi32(
            emul_unpacklo_epi64_i32(p01, p23),
            emul_unpackhi_epi64_i32(p01, p23),
        );
        for l in 0..8 {
            acc[l] = lo[l % 4].mul_add(pall[l] as f32, acc[l]);
        }
    }

    let nb4 = 4 * nbg;
    for i in nb4..nb {
        let wb = &wrow[i * Q5F0_BLOCK..(i + 1) * Q5F0_BLOCK];
        let dw = f16_bits_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
        let code = q5x_codes_scalar::<2, 6>(wb);
        let tb = nb4 / 4 * Q82X4_STRIDE + (i - nb4) * Q82_BLOCK;
        let blk = &acol[tb..tb + Q82_BLOCK];
        let da = bf16_bits_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let ma = i16::from_le_bytes([blk[2], blk[3]]) as f32;
        let d = dw * da;
        let corr = (-16.0f32 * dw) * (da * ma) * 0.25f32;
        for a in &mut accm {
            *a += corr;
        }
        let mut qs = [0u8; 32];
        qs.copy_from_slice(&blk[4..36]);
        let p0 = emul_madd1(emul_maddubs(code, qs));
        for l in 0..8 {
            acc[l] = d.mul_add(p0[l] as f32, acc[l]);
        }
    }

    let mut x = [0.0f32; 4];
    for l in 0..4 {
        x[l] = (acc[l] + acc[4 + l]) + accm[l];
    }
    (x[0] + x[2]) + (x[1] + x[3])
}

// --------------------------------------------------------- Q5_1 x Q8_2_X4

/// AVX2+FMA+F16C row dot: Q5_1 weights against Q8_2_X4 column.
///
/// TWIN of [`dot_q5f0_q82x4_avx2`] — the same template with four
/// differences only: block width (24 vs 22), the `q5x_codes` offsets (qh
/// @4/qs @8 vs @2/@6), the scale gather (the (d,m) pair shuffle through one
/// `_mm256_cvtph_ps` vs four d's), and the tail min correction (the stored
/// m_w vs -16·d_w). Every edit outside those four must be made in both.
///
/// # Safety
/// Caller must ensure AVX2+FMA+F16C are available and buffers match `nb` blocks.
#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
unsafe fn dot_q5f1_q82x4_avx2(wrow: &[u8], acol: &[u8], nb: usize) -> f32 {
    // SAFETY: AVX2+FMA+F16C present (checked by dot_row) and both slices
    // hold the validated lengths per contract.
    unsafe {
        let m4 = _mm256_set1_epi8(0xF);
        let mh = _mm256_set1_epi8(0x10);
        let hb = Q5HBit {
            shuffle: _mm256_set_epi64x(
                0x0303_0303_0303_0303,
                0x0202_0202_0202_0202,
                0x0101_0101_0101_0101,
                0x0000_0000_0000_0000,
            ),
            mask: _mm256_set1_epi64x(0x7fbf_dfef_f7fb_fdfe),
            minus1: _mm256_set1_epi64x(-1),
        };
        let m1 = _mm256_set1_epi16(1);
        let dm_shuf = _mm_set_epi16(
            0x0f0e, 0x0b0a, 0x0706, 0x0302, 0x0d0c, 0x0908, 0x0504, 0x0100,
        );
        let mut acc = _mm256_setzero_ps();
        let mut accm = _mm_setzero_ps();

        let nbg = nb / 4;
        for i in 0..nbg {
            let b0 = (4 * i) * Q5F1_BLOCK;
            let b1 = (4 * i + 1) * Q5F1_BLOCK;
            let b2 = (4 * i + 2) * Q5F1_BLOCK;
            let b3 = (4 * i + 3) * Q5F1_BLOCK;
            let qx0 = q5x_codes::<4, 8>(&wrow[b0..b0 + Q5F1_BLOCK], m4, mh, &hb);
            let qx1 = q5x_codes::<4, 8>(&wrow[b1..b1 + Q5F1_BLOCK], m4, mh, &hb);
            let qx2 = q5x_codes::<4, 8>(&wrow[b2..b2 + Q5F1_BLOCK], m4, mh, &hb);
            let qx3 = q5x_codes::<4, 8>(&wrow[b3..b3 + Q5F1_BLOCK], m4, mh, &hb);
            let mut pairs = [0u8; 16];
            pairs[0..4].copy_from_slice(&wrow[b0..b0 + 4]);
            pairs[4..8].copy_from_slice(&wrow[b1..b1 + 4]);
            pairs[8..12].copy_from_slice(&wrow[b2..b2 + 4]);
            pairs[12..16].copy_from_slice(&wrow[b3..b3 + 4]);
            // SAFETY: 16 readable bytes in the local staging buffer.
            let other = _mm256_cvtph_ps(_mm_shuffle_epi8(
                _mm_loadu_si128(pairs.as_ptr() as *const __m128i),
                dm_shuf,
            ));

            // SAFETY: 8 readable bytes at the head of the validated group.
            let g = acol.as_ptr().add(i * Q82X4_STRIDE);
            let aux_d = _mm_castsi128_ps(_mm_slli_epi32::<16>(_mm_cvtepu16_epi32(
                _mm_loadl_epi64(g as *const __m128i),
            )));
            // SAFETY: 8 readable bytes at offset 8 of the group.
            let aux_m = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_loadl_epi64(
                g.add(8) as *const __m128i
            )));
            let prep = _mm256_set_m128(_mm_mul_ps(aux_d, aux_m), aux_d);
            let s12 = _mm256_mul_ps(other, prep);
            accm = _mm_add_ps(accm, _mm256_extractf128_ps(s12, 1));
            let lo = _mm256_castps256_ps128(s12);
            let dall = _mm256_set_m128(lo, lo);

            // SAFETY: 32 readable bytes at each offset inside the group.
            let p0 = _mm256_madd_epi16(
                m1,
                _mm256_maddubs_epi16(qx0, _mm256_loadu_si256(g.add(16) as *const __m256i)),
            );
            let p1 = _mm256_madd_epi16(
                m1,
                _mm256_maddubs_epi16(qx1, _mm256_loadu_si256(g.add(48) as *const __m256i)),
            );
            let p2 = _mm256_madd_epi16(
                m1,
                _mm256_maddubs_epi16(qx2, _mm256_loadu_si256(g.add(80) as *const __m256i)),
            );
            let p3 = _mm256_madd_epi16(
                m1,
                _mm256_maddubs_epi16(qx3, _mm256_loadu_si256(g.add(112) as *const __m256i)),
            );
            let p01 =
                _mm256_add_epi32(_mm256_unpacklo_epi32(p0, p1), _mm256_unpackhi_epi32(p0, p1));
            let p23 =
                _mm256_add_epi32(_mm256_unpacklo_epi32(p2, p3), _mm256_unpackhi_epi32(p2, p3));
            let pall = _mm256_add_epi32(
                _mm256_unpacklo_epi64(p01, p23),
                _mm256_unpackhi_epi64(p01, p23),
            );
            acc = _mm256_fmadd_ps(dall, _mm256_cvtepi32_ps(pall), acc);
        }

        let nb4 = 4 * nbg;
        for i in nb4..nb {
            let b = i * Q5F1_BLOCK;
            let wb = &wrow[b..b + Q5F1_BLOCK];
            let dw = f16_bits_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
            let mw = f16_bits_to_f32(u16::from_le_bytes([wb[2], wb[3]]));
            let qx0 = q5x_codes::<4, 8>(wb, m4, mh, &hb);
            let tb = nb4 / 4 * Q82X4_STRIDE + (i - nb4) * Q82_BLOCK;
            let da = bf16_bits_to_f32(u16::from_le_bytes([acol[tb], acol[tb + 1]]));
            let ma = i16::from_le_bytes([acol[tb + 2], acol[tb + 3]]) as f32;
            let d = dw * da;
            let corr = mw * (da * ma) * 0.25f32;
            accm = _mm_add_ps(accm, _mm_set1_ps(corr));
            // SAFETY: 32 readable bytes at offset 4 of the tail block.
            let qs = _mm256_loadu_si256(acol.as_ptr().add(tb + 4) as *const __m256i);
            let p0 = _mm256_madd_epi16(m1, _mm256_maddubs_epi16(qx0, qs));
            acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(p0), acc);
        }

        // Reduction order is load-bearing.
        let sum = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
        let x = _mm_add_ps(sum, accm);
        let x = _mm_add_ps(x, _mm_movehl_ps(x, x));
        _mm_cvtss_f32(_mm_add_ss(x, _mm_movehdup_ps(x)))
    }
}

/// Emulates `dot_q5f1_q82x4_avx2` with bit identity.
fn dot_q5f1_q82x4_emul(wrow: &[u8], acol: &[u8], nb: usize) -> f32 {
    let mut acc = [0.0f32; 8];
    let mut accm = [0.0f32; 4];

    let nbg = nb / 4;
    for i in 0..nbg {
        let mut qx = [[0u8; 32]; 4];
        let mut dw = [0.0f32; 4];
        let mut mw = [0.0f32; 4];
        for j in 0..4 {
            let wb = &wrow[(4 * i + j) * Q5F1_BLOCK..(4 * i + j + 1) * Q5F1_BLOCK];
            qx[j] = q5x_codes_scalar::<4, 8>(wb);
            dw[j] = f16_bits_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
            mw[j] = f16_bits_to_f32(u16::from_le_bytes([wb[2], wb[3]]));
        }
        let g = &acol[i * Q82X4_STRIDE..(i + 1) * Q82X4_STRIDE];
        let mut da = [0.0f32; 4];
        let mut ma = [0.0f32; 4];
        for l in 0..4 {
            da[l] = bf16_bits_to_f32(u16::from_le_bytes([g[2 * l], g[2 * l + 1]]));
            ma[l] = i16::from_le_bytes([g[8 + 2 * l], g[9 + 2 * l]]) as f32;
        }
        let mut lo = [0.0f32; 4];
        for l in 0..4 {
            lo[l] = dw[l] * da[l];
            accm[l] += mw[l] * (da[l] * ma[l]);
        }
        let mut p = [[0i32; 8]; 4];
        for (j, pj) in p.iter_mut().enumerate() {
            let mut qs = [0u8; 32];
            qs.copy_from_slice(&g[16 + 32 * j..16 + 32 * j + 32]);
            *pj = emul_madd1(emul_maddubs(qx[j], qs));
        }
        let p01 = emul_add_epi32(
            emul_unpacklo_epi32_i32(p[0], p[1]),
            emul_unpackhi_epi32_i32(p[0], p[1]),
        );
        let p23 = emul_add_epi32(
            emul_unpacklo_epi32_i32(p[2], p[3]),
            emul_unpackhi_epi32_i32(p[2], p[3]),
        );
        let pall = emul_add_epi32(
            emul_unpacklo_epi64_i32(p01, p23),
            emul_unpackhi_epi64_i32(p01, p23),
        );
        for l in 0..8 {
            acc[l] = lo[l % 4].mul_add(pall[l] as f32, acc[l]);
        }
    }

    let nb4 = 4 * nbg;
    for i in nb4..nb {
        let wb = &wrow[i * Q5F1_BLOCK..(i + 1) * Q5F1_BLOCK];
        let dw = f16_bits_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
        let mw = f16_bits_to_f32(u16::from_le_bytes([wb[2], wb[3]]));
        let code = q5x_codes_scalar::<4, 8>(wb);
        let tb = nb4 / 4 * Q82X4_STRIDE + (i - nb4) * Q82_BLOCK;
        let blk = &acol[tb..tb + Q82_BLOCK];
        let da = bf16_bits_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let ma = i16::from_le_bytes([blk[2], blk[3]]) as f32;
        let d = dw * da;
        let corr = mw * (da * ma) * 0.25f32;
        for a in &mut accm {
            *a += corr;
        }
        let mut qs = [0u8; 32];
        qs.copy_from_slice(&blk[4..36]);
        let p0 = emul_madd1(emul_maddubs(code, qs));
        for l in 0..8 {
            acc[l] = d.mul_add(p0[l] as f32, acc[l]);
        }
    }

    let mut x = [0.0f32; 4];
    for l in 0..4 {
        x[l] = (acc[l] + acc[4 + l]) + accm[l];
    }
    (x[0] + x[2]) + (x[1] + x[3])
}

// ------------------------------------------------- Q8_0 x act cells
// Sign fold: |w| is maddubs' u8 side, sign(w) is folded into the activation
// bytes, so |w|*(a*sign(w)) = w*a exactly, with no +128 offset to compensate.
// Integer adds are associative, tree shape is free; the f64 epilogue is scalar,
// in the mirror's order — bit identity is algebra, not a port.
// DOMAIN: activation codes must lie in [-127, 127]. -128 under a negative
// weight wraps in `sign_epi8` and flips that term's sign. The producer
// (`model::attn::quantize_act`) clamps at -127; a debug_assert re-checks here.
// Weight codes may be -128 (|w| = 128 is a legal u8 magnitude).
// i16: a maddubs pair peaks at 2*128*127 = 32512 <= 32767; saturation is
// unreachable on any i8 x i8 input in this form. i32: 32 terms <= 516128.

/// One Q8_0 block over 32 values: f16 scale bits then 32 int8 codes (34 bytes).
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Q8Block {
    /// f16 bits of the block scale (convert with `half_to_f32`).
    pub d: u16,
    /// The int8 codes, `[-127, 127]`.
    pub q: [i8; 32],
}

/// One quantized activation block: bf16 scale as f32 and 32 int8 codes.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ActBlock {
    /// The bf16-valued block scale as f32 (already rounded; used as-is).
    pub d: f32,
    /// The int8 codes.
    pub q: [i8; 32],
}

/// Cell-segment kernel for `q_nope2_absorbed` pool split: computes output cells
/// in `[j0, j_end)` for one column.
pub fn q_nope2_cells(
    whead: &[Q8Block],
    acol: &[ActBlock],
    j0: usize,
    j_end: usize,
    out: &mut [f32],
) {
    assert!(j0 <= j_end, "q_nope2_cells: j0 {j0} > j_end {j_end}");
    assert!(
        whead.len() >= j_end * acol.len(),
        "q_nope2_cells: whead has {} blocks; cells up to j_end {j_end} at {} per cell need {}",
        whead.len(),
        acol.len(),
        j_end * acol.len()
    );
    assert!(
        out.len() >= j_end - j0,
        "q_nope2_cells: out holds {} cells, segment is {}",
        out.len(),
        j_end - j0
    );
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        // SAFETY: AVX2+FMA detected just now; asserts bound every index the kernel touches.
        unsafe { q_nope2_cells_avx2_inner(whead, acol, j0, j_end, out) }
    } else {
        q_nope2_cells_scalar(whead, acol, j0, j_end, out);
    }
}

/// Scalar mirror of `q_nope2_cells`, bit-identical by construction.
pub fn q_nope2_cells_scalar(
    whead: &[Q8Block],
    acol: &[ActBlock],
    j0: usize,
    j_end: usize,
    out: &mut [f32],
) {
    let nb = acol.len();
    for j in j0..j_end {
        let mut cell = 0.0f64;
        for (b, ab) in acol.iter().enumerate() {
            let wb = &whead[j * nb + b];
            let mut isum = 0i32;
            for l in 0..32 {
                isum += wb.q[l] as i32 * ab.q[l] as i32;
            }
            cell += (half_to_f32(wb.d) * ab.d) as f64 * f64::from(isum);
        }
        out[j - j0] = cell as f32;
    }
}

/// AVX2 entry point for `q_nope2_cells` for comparison against scalar mirror.
pub fn q_nope2_cells_avx2(
    whead: &[Q8Block],
    acol: &[ActBlock],
    j0: usize,
    j_end: usize,
    out: &mut [f32],
) {
    assert!(
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma"),
        "q_nope2_cells_avx2 called on a CPU without AVX2+FMA"
    );
    assert!(j0 <= j_end, "q_nope2_cells_avx2: j0 {j0} > j_end {j_end}");
    assert!(
        whead.len() >= j_end * acol.len(),
        "q_nope2_cells_avx2: whead has {} blocks; cells up to j_end {j_end} at {} per cell need {}",
        whead.len(),
        acol.len(),
        j_end * acol.len()
    );
    assert!(
        out.len() >= j_end - j0,
        "q_nope2_cells_avx2: out holds {} cells, segment is {}",
        out.len(),
        j_end - j0
    );
    // SAFETY: asserted just above.
    unsafe { q_nope2_cells_avx2_inner(whead, acol, j0, j_end, out) }
}

/// AVX2 inner kernel for `q_nope2_cells`. One fn on purpose: splitting a
/// `target_feature` body into helpers costs throughput.
///
/// # Safety
/// CPU must support AVX2+FMA; caller must ensure slice lengths match segment
/// bounds. Activation codes must be in [-127, 127] (section header, DOMAIN).
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn q_nope2_cells_avx2_inner(
    whead: &[Q8Block],
    acol: &[ActBlock],
    j0: usize,
    j_end: usize,
    out: &mut [f32],
) {
    // SAFETY: AVX2+FMA present and the slice lengths match the segment bounds per contract.
    unsafe {
        let ones = _mm256_set1_epi16(1);
        let nb = acol.len();
        for j in j0..j_end {
            let mut cell = 0.0f64;
            for b in 0..nb {
                let wb = &whead[j * nb + b];
                let ab = &acol[b];
                // SAFETY: 32-byte unaligned loads inside the 32-byte code arrays.
                let wv = _mm256_loadu_si256(wb.q.as_ptr() as *const __m256i);
                let av = _mm256_loadu_si256(ab.q.as_ptr() as *const __m256i);
                let us = _mm256_sign_epi8(wv, wv);
                let sq = _mm256_sign_epi8(av, wv);
                let pairs = _mm256_maddubs_epi16(us, sq);
                let quads = _mm256_madd_epi16(ones, pairs);
                let h1 = _mm256_hadd_epi32(quads, quads);
                let h2 = _mm256_hadd_epi32(h1, h1);
                let isum = _mm256_extract_epi32(h2, 0) + _mm256_extract_epi32(h2, 4);
                #[cfg(debug_assertions)]
                for l in 0..32 {
                    debug_assert!(
                        !(ab.q[l] == -128 && wb.q[l] < 0),
                        "activation code -128 under a negative weight code: \
                         outside the sign-fold kernel's contract"
                    );
                }
                cell += (half_to_f32(wb.d) * ab.d) as f64 * f64::from(isum);
            }
            out[j - j0] = cell as f32;
        }
    }
}

// --------------------------------------------------------- F32 row · column

/// One F32 weight row (little-endian bytes, as the file holds it) dotted with
/// an F32 column, in the reference's float-kernel order (`mul_mat_Qx_Qy_MxN`,
/// iqk_gemm_floats.cpp): one eight-lane accumulator — the first block a plain
/// multiply, every later block `fmadd(y, x, acc)` — then `hsum_float_8` (upper
/// half onto lower, `movehl`, `movehdup`). `None` when the CPU lacks AVX2+FMA or
/// `k` is not a multiple of 8; the caller keeps its scalar loop for that.
pub fn dot_f32(wrow: &[u8], x: &[f32]) -> Option<f32> {
    assert_eq!(
        wrow.len(),
        x.len() * 4,
        "dot_f32: row bytes vs column length"
    );
    #[cfg(target_arch = "x86_64")]
    if !x.is_empty()
        && x.len().is_multiple_of(8)
        && std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("fma")
    {
        // SAFETY: the features were just detected; lengths checked above.
        return Some(unsafe { dot_f32_avx2(wrow, x) });
    }
    None
}

/// # Safety
/// The CPU must support AVX2 and FMA. `x` must be non-empty with a length that
/// is a multiple of eight, and `wrow` must hold `4 * x.len()` readable bytes —
/// the row's f32s, little-endian.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_f32_avx2(wrow: &[u8], x: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let w = wrow.as_ptr() as *const f32;
    let nb = x.len() / 8;
    // SAFETY: `i * 8 + 8 <= x.len()` and `wrow` is four bytes per element; the
    // loads are unaligned, and x86 is little-endian, so the bytes are the f32s.
    unsafe {
        let mut acc = _mm256_mul_ps(_mm256_loadu_ps(x.as_ptr()), _mm256_loadu_ps(w));
        for i in 1..nb {
            let yv = _mm256_loadu_ps(x.as_ptr().add(i * 8));
            let xv = _mm256_loadu_ps(w.add(i * 8));
            acc = _mm256_fmadd_ps(yv, xv, acc);
        }
        let mut s = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
        s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        s = _mm_add_ss(s, _mm_movehdup_ps(s));
        _mm_cvtss_f32(s)
    }
}

// ------------------------------------------------------- sum of squares

/// `Σ (x[i]·x[i]) as f64` — f32 squares accumulated in f64, the reference
/// norm's sum. With AVX2 the f64 accumulation runs in eight lanes (two
/// four-lane registers) instead of left to right: the two orders differ by
/// f64 rounding only (~1e-16 relative), which the caller's narrowing of the
/// mean to f32 absorbs except on an exact rounding tie.
pub fn sum_sq_f64(x: &[f32]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: the feature was just detected.
        return unsafe { sum_sq_f64_avx2(x) };
    }
    x.iter().map(|&v| (v * v) as f64).sum()
}

/// # Safety
/// The CPU must support AVX2. Any `x` is in range — the eight-wide loop stops
/// at `x.len() / 8 * 8` and the tail is a safe scalar loop.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sum_sq_f64_avx2(x: &[f32]) -> f64 {
    use std::arch::x86_64::*;
    let full = x.len() / 8 * 8;
    // SAFETY: `i + 8 <= full <= x.len()`; unaligned loads.
    let mut total = unsafe {
        let mut a0 = _mm256_setzero_pd();
        let mut a1 = _mm256_setzero_pd();
        let mut i = 0;
        while i < full {
            let v = _mm256_loadu_ps(x.as_ptr().add(i));
            let sq = _mm256_mul_ps(v, v);
            a0 = _mm256_add_pd(a0, _mm256_cvtps_pd(_mm256_castps256_ps128(sq)));
            a1 = _mm256_add_pd(a1, _mm256_cvtps_pd(_mm256_extractf128_ps(sq, 1)));
            i += 8;
        }
        let a = _mm256_add_pd(a0, a1);
        let lo = _mm_add_pd(_mm256_castpd256_pd128(a), _mm256_extractf128_pd(a, 1));
        _mm_cvtsd_f64(_mm_add_sd(lo, _mm_unpackhi_pd(lo, lo)))
    };
    for &v in &x[full..] {
        total += (v * v) as f64;
    }
    total
}

// ------------------------------------------------------------ SwiGLU combine

/// `out[i] = silu(gate[i]) * up[i]`, the reference's AVX2 form: `ggml_v_expf` and
/// `ggml_v_silu` ported lane for lane (`x / (1 + exp(-x))`, then `* up`), so the
/// result tracks ik's bits rather than libm's. A tail shorter than eight goes
/// through the same lanes on a padded copy — an element's value never depends on
/// where in the block it sits. Falls back to the scalar libm form without AVX2+FMA.
pub fn swiglu(gate: &[f32], up: &[f32], out: &mut [f32]) {
    assert!(gate.len() == up.len() && gate.len() == out.len());
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        // SAFETY: the features were just detected; the slices are equal length.
        unsafe { swiglu_avx2(gate, up, out) };
        return;
    }
    for (o, (&g, &u)) in out.iter_mut().zip(gate.iter().zip(up)) {
        *o = g / (1.0 + (-g).exp()) * u;
    }
}

/// # Safety
/// The CPU must support AVX2 and FMA, and `gate`, `up` and `out` must all be
/// the same length — the loads and the store share one index.
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn swiglu_avx2(gate: &[f32], up: &[f32], out: &mut [f32]) {
    let n = gate.len();
    let full = n / 8 * 8;
    let mut i = 0;
    while i < full {
        // SAFETY: `i + 8 <= full <= n` for all three slices.
        unsafe {
            let g = _mm256_loadu_ps(gate.as_ptr().add(i));
            let u = _mm256_loadu_ps(up.as_ptr().add(i));
            _mm256_storeu_ps(out.as_mut_ptr().add(i), _mm256_mul_ps(v_silu(g), u));
        }
        i += 8;
    }
    if full < n {
        let (mut gp, mut upad, mut op) = ([0.0f32; 8], [0.0f32; 8], [0.0f32; 8]);
        gp[..n - full].copy_from_slice(&gate[full..]);
        upad[..n - full].copy_from_slice(&up[full..]);
        // SAFETY: the three arrays are eight f32 each.
        unsafe {
            let r = _mm256_mul_ps(
                v_silu(_mm256_loadu_ps(gp.as_ptr())),
                _mm256_loadu_ps(upad.as_ptr()),
            );
            _mm256_storeu_ps(op.as_mut_ptr(), r);
        }
        out[full..].copy_from_slice(&op[..n - full]);
    }
}

/// `ggml_v_silu`: `x / (1 + exp(-x))`.
///
/// # Safety
/// The CPU must support AVX2 and FMA. Register-only: no memory is touched.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn v_silu(x: __m256) -> __m256 {
    let neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
    // SAFETY: same target features as the caller.
    let e = unsafe { v_expf(neg_x) };
    _mm256_div_ps(x, _mm256_add_ps(_mm256_set1_ps(1.0), e))
}

/// `ggml_v_expf` (the Arm optimized-routines polynomial): `exp(x) = 2^n · (1 + j(b))`
/// with `n = round(x / ln2)` and `b = x − n·ln2` in two pieces. Constants are the
/// reference's hex floats as bit patterns. The special-case tail (|n| > 126) is
/// the reference's too: overflow to inf past 192, a two-step scale otherwise.
///
/// # Safety
/// The CPU must support AVX2 and FMA. Register-only: no memory is touched.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn v_expf(x: __m256) -> __m256 {
    let f = |bits: u32| _mm256_set1_ps(f32::from_bits(bits));
    let r = f(0x4B40_0000); // 0x1.8p23
    let z = _mm256_fmadd_ps(x, f(0x3FB8_AA3B), r); // 0x1.715476p+0
    let n = _mm256_sub_ps(z, r);
    let b = _mm256_fnmadd_ps(
        n,
        f(0x35BF_BE8E),                         // 0x1.7f7d1cp-20
        _mm256_fnmadd_ps(n, f(0x3F31_7200), x), // 0x1.62e4p-1
    );
    let e = _mm256_slli_epi32(_mm256_castps_si256(z), 23);
    let k = _mm256_castsi256_ps(_mm256_add_epi32(
        e,
        _mm256_castps_si256(_mm256_set1_ps(1.0)),
    ));
    let absn = _mm256_andnot_ps(_mm256_set1_ps(-0.0), n);
    let c = _mm256_cmp_ps(absn, _mm256_set1_ps(126.0), _CMP_GT_OQ);
    let u = _mm256_mul_ps(b, b);
    let j = _mm256_fmadd_ps(
        _mm256_fmadd_ps(
            _mm256_fmadd_ps(f(0x3C07_2010), b, f(0x3D2B_9F17)), // 0x1.0e4020p-7, 0x1.573e2ep-5
            u,
            _mm256_fmadd_ps(f(0x3E2A_AF33), b, f(0x3EFF_FEDB)), // 0x1.555e66p-3, 0x1.fffdb6p-2
        ),
        u,
        _mm256_mul_ps(f(0x3F7F_FFF6), b), // 0x1.ffffecp-1
    );
    if _mm256_movemask_ps(c) == 0 {
        return _mm256_fmadd_ps(j, k, k);
    }
    let g = _mm256_and_si256(
        _mm256_castps_si256(_mm256_cmp_ps(n, _mm256_setzero_ps(), _CMP_LE_OQ)),
        _mm256_set1_epi32(0x8200_0000u32 as i32),
    );
    let s1 = _mm256_castsi256_ps(_mm256_add_epi32(g, _mm256_set1_epi32(0x7f00_0000)));
    let s2 = _mm256_castsi256_ps(_mm256_sub_epi32(e, g));
    let d = _mm256_cmp_ps(absn, _mm256_set1_ps(192.0), _CMP_GT_OQ);
    _mm256_or_ps(
        _mm256_and_ps(d, _mm256_mul_ps(s1, s1)),
        _mm256_andnot_ps(
            d,
            _mm256_or_ps(
                _mm256_and_ps(c, _mm256_mul_ps(_mm256_fmadd_ps(s2, j, s2), s1)),
                _mm256_andnot_ps(c, _mm256_fmadd_ps(k, j, k)),
            ),
        ),
    )
}
