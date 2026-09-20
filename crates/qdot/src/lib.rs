//! qdot — the fused quantized row dot for the host CPU: Q3_K x Q8_K and
//! Q4_K x Q8_2_X4.
//!
//! `model::ops::matmul_q` today dequantizes every weight row to f32
//! (`gguf::quant::dequant_row`), round-trips the activations through the
//! format the weight type implies (`gguf::quant::quantize_activations`), and
//! dots the two in f32. ik_llama.cpp never materializes f32 weights: it dots
//! the quantized codes in integer lanes and scales once per block. This
//! crate is that kernel, in the shape `matmul_q` calls it: one weight row
//! against one quantized activation column, `k` a runtime parameter (the
//! model uses k in {2048, 512}, both multiples of 256).
//!
//! Nothing here is invented — every decode is a port with its source cited:
//!   * the Q3_K activation quantizer is ggml's `quantize_row_q8_K_ref`
//!     (ggml-quants.c:3974, read in the vendored ik fork) into that fork's
//!     296-byte `block_q8_K`: d f32 @0, sum f32 @4 (an ik-only field the C
//!     ref leaves unwritten and nothing in the q3_K dot reads — zeroed),
//!     s8 qs[256] @8, s16 bsums[16] @264;
//!   * the Q3_K row dot is the `__AVX2__` branch of `ggml_vec_dot_q3_K_q8_K`
//!     (ggml-quants.c:6482), reduced from the q3k-cpu pre-study's M <= 8
//!     column bundle to the one column `matmul_q` dots at a time, with the
//!     super-block count from `k` instead of the pre-study's K = 2048
//!     constant. Both arrived via `crates/q3k-cpu`, which verified them
//!     against ik's own reference outputs at 1e-2 over 24 configs.
//!   * the Q4_K activation quantizer is the `__x86_64__` branch of ik's
//!     `quantize_row_q8_2_x4` (iqk_quantize.cpp:1005): 32-value blocks,
//!     bf16-rounded scale `d` (stored AND used), round-to-nearest-even
//!     quants, and the RAW i16 integer sum in the `s` slot (x86 stores no
//!     bf16 there — the kernel recovers it with a sign-extend, not a
//!     convert). Four blocks interleave into a 144-byte x4 group:
//!     d u16[8] (bf16 scales 0..3, i16 sums 4..7) then qs s8[128].
//!   * the Q4_K row dot is ik's `mul_mat_qX_K_q8_2_X4_T<DequantizerQ4K_AVX2,
//!     1>` (iqk_gemm_kquants.cpp:787) — the pairing the oracle actually
//!     dispatches (iqk_gemm_kquants.cpp:2751/2768), unlike the traits-table
//!     Q4_K x q8_K fallback this crate ported first (commit 34bd457, dump
//!     `q4k-ik-dot.txt` kept in BLOOMERY_DATA). The q8_K pairing was
//!     removed when the x4 one landed: wiring the wrong pairing moved
//!     kqv_out-0 by 6.4e-3 against the oracle (gate 5e-3), and carrying two
//!     Q4_K kernels where the engine can select one invites exactly that
//!     mistake again.
//!
//! Super-block geometry (block_q3_K, 110 bytes / 256 values): hmask[32] @+0,
//! qs[64] @+32, scales[12] @+96, f16 d @+108. Weight element 128c+32f+l
//! reads qs byte 32c+l field f; its high bit is hmask byte l bit 4c+f; its
//! sub-block scale (after the aux[] unpack, minus 32) is scales[8c+2f+l/16].
//! The lane/scale mapping of the AVX2 shuffle chain was re-probed against
//! the intrinsics on the box on 2026-09-20 (distinct scales, one
//! super-block, AVX2 chain == the scalar dequant order) before this port
//! trusted it.
//!
//! A scalar mirror of each AVX2 kernel lives here too, so the crate is
//! testable on machines without AVX2 and the two paths are compared bit for
//! bit by the gate (`tests/qdot.rs`). The Q4_K mirror goes further: it is an
//! instruction-graph emulator (each maddubs/unpack/madd reproduced as array
//! ops in Intel's exact semantics), because the x4 kernel's pack chain
//! assigns lanes in a way hand-derived lane algebra gets wrong — the
//! emulator is right by construction, and it doubles as the no-AVX2 path.

use std::arch::x86_64::*;
use std::fmt;

use gguf::GgmlType;
use gguf::quant::half_to_f32;

/// `sizeof(block_q3_K)` (ggml-common.h): 110 bytes / 256 values.
const Q3K_BLOCK: usize = 110;
/// `sizeof(block_q8_K)` in the vendored ik fork: 296 bytes / 256 values —
/// d f32 @0, sum f32 @4, qs[256] s8 @8, bsums[16] s16 @264.
const Q8K_STRIDE: usize = 296;
/// `sizeof(block_q8_2_x4)` (ggml-common.h:296): 144 bytes / 128 values —
/// d u16[8] (bf16 scales @0..4, raw i16 sums @4..8), qs s8[128] @16. The
/// same number as Q4K_BLOCK is coincidence: one is per 256 WEIGHT values,
/// the other per 128 ACTIVATION values.
const Q82X4_STRIDE: usize = 144;

// ----------------------------------------------------------- public errors

/// Everything `dot_row` refuses to guess about. Length and alignment
/// problems are caller errors, not bugs — the same stance as
/// `gguf::QuantError` — because a partial super-block would compute quietly
/// wrong values, and this API rejects that class instead of panicking.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QdotError {
    /// The weight type has no fused kernel in this build (Q3_K only for now).
    UnsupportedType(GgmlType),
    /// `k` is not a multiple of 256; Q3_K x Q8_K works on whole super-blocks.
    UnalignedK { k: usize },
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
            QdotError::UnalignedK { k } => write!(
                f,
                "k = {k}: qdot rows are whole 256-value super-blocks (k % 256 != 0)"
            ),
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
/// is in this build's table AND the CPU has the ISA the kernel needs.
///
/// The kernel's widest instructions are AVX2 (`vpmaddubsw`/`vpmaddwd` chains
/// — the box is Zen 3: AVX2/FMA3/F16C/BMI2, no AVX-512, no VNNI). FMA is not
/// used (the f32 scaling is plain multiplies, so the scalar mirror can match
/// bit for bit) and the f16 read is an integer conversion, so `avx2` alone
/// is the requirement.
pub fn supports(w: GgmlType) -> bool {
    // Q4_K is wired as of 2026-09-20 (MUL-27): its pairing is q8_2_x4 —
    // ik's dispatch is `mul_mat_qX_K_q8_2_X4_T<DequantizerQ4K_AVX2>`
    // (iqk_gemm_kquants.cpp:2751, 2768) with expected_type_B =
    // GGML_TYPE_Q8_2_X4. The q8_K pairing this crate landed first (commit
    // 34bd457) was the traits-table fallback, not the oracle's path: wiring
    // it moved kqv_out-0 by 6.4e-3 against the oracle (gate 5e-3) — an
    // activation-format delta, not a bug. That kernel is gone; its dump
    // (q4k-ik-dot.txt) stays in BLOOMERY_DATA as the round's record.
    matches!(w, GgmlType::Q3_K | GgmlType::Q4_K) && std::arch::is_x86_feature_detected!("avx2")
}

/// Bytes one quantized activation column of `k` values occupies in `w`'s
/// format: Q3_K pairs q8_K (`k / 256` blocks of 296 bytes), Q4_K pairs
/// q8_2_x4 (`k / 128` groups of 144 bytes).
///
/// Panics if `w` has no activation format in this build or `k % 256 != 0` —
/// both are caller constants at allocation time, and a made-up size for a
/// partial block would allocate a buffer `quantize_col` cannot fill. (The
/// x4 layout itself only needs `k % 128`; the 256 contract is the WEIGHT
/// super-block's, and `dot_row` enforces the same one.)
pub fn col_bytes(w: GgmlType, k: usize) -> usize {
    assert!(
        matches!(w, GgmlType::Q3_K | GgmlType::Q4_K),
        "qdot has no activation format for {w:?} in this build"
    );
    assert!(
        k.is_multiple_of(256),
        "qdot activation columns are whole 256-value super-blocks, k = {k}"
    );
    match w {
        GgmlType::Q4_K => (k / 128) * Q82X4_STRIDE,
        _ => (k / 256) * Q8K_STRIDE,
    }
}

/// Quantize one activation column into the block format `w` implies.
/// `out.len()` must be `col_bytes(w, x.len())`; panics otherwise (see
/// [`col_bytes`] — allocation-time constants, not data-dependent).
///
/// Q3_K: port of `quantize_row_q8_K_ref` (ggml-quants.c:3974) over each
/// 256-value block, with the scale taken from the SIGNED extreme (`iscale =
/// -127/max`) exactly as `gguf::quant::quantize_row_q8_k_roundtrip` does.
/// The round trip and this coder are the same quantization in two
/// representations; the gate checks that bit for bit (`tests/qdot.rs`,
/// gate 1).
///
/// Q4_K: port of the `__x86_64__` branch of ik's `quantize_row_q8_2_x4`
/// (iqk_quantize.cpp:1005) — bf16-rounded `d`, `round_ties_even` quants,
/// raw i16 sums. The gate checks the bytes bit for bit against ik's own
/// coding of the same column (dump `q4k-x4-ik-dot.txt`).
pub fn quantize_col(w: GgmlType, x: &[f32], out: &mut [u8]) {
    assert!(
        matches!(w, GgmlType::Q3_K | GgmlType::Q4_K),
        "qdot has no activation format for {w:?} in this build"
    );
    assert!(
        x.len().is_multiple_of(256),
        "qdot activation columns are whole 256-value super-blocks, x.len() = {}",
        x.len()
    );
    match w {
        GgmlType::Q4_K => {
            assert_eq!(
                out.len(),
                (x.len() / 128) * Q82X4_STRIDE,
                "out must be col_bytes(w, x.len())"
            );
            quantize_q82x4_col(x, out);
        }
        _ => {
            assert_eq!(
                out.len(),
                (x.len() / 256) * Q8K_STRIDE,
                "out must be col_bytes(w, x.len())"
            );
            let (xblocks, _) = x.as_chunks::<256>();
            let (oblocks, _) = out.as_chunks_mut::<296>();
            for (xb, ob) in xblocks.iter().zip(oblocks.iter_mut()) {
                quantize_q8k_block(xb, ob);
            }
        }
    }
}

/// `dot(weight row, quantized activation column)`, both still in quantized
/// form: `wrow` is the raw row bytes straight out of the GGUF, `acol` is
/// [`quantize_col`]'s output for the matching `k` values.
///
/// Errors instead of guessing on shapes (`k % 256 != 0`, an unsupported
/// weight type, or a short buffer is a [`QdotError`], not a panic) — a
/// partial super-block would compute quietly wrong values, and this API
/// refuses that. The dispatch is AVX2 when the CPU has it ([`supports`]),
/// the scalar mirror otherwise; the two agree bit for bit (gate 4).
pub fn dot_row(w: GgmlType, wrow: &[u8], acol: &[u8], k: usize) -> Result<f32, QdotError> {
    let nb = check_row(w, wrow.len(), acol.len(), k)?;
    let avx2 =
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma");
    match (w, avx2) {
        (GgmlType::Q3_K, true) => {
            // SAFETY: AVX2 was just detected; lengths validated by check_row.
            Ok(unsafe { dot_q3k_q8k_avx2(wrow, acol, nb) })
        }
        (GgmlType::Q4_K, true) => {
            // SAFETY: AVX2+FMA were just detected; lengths validated above.
            Ok(unsafe { dot_q4k_q82x4_avx2(wrow, acol, nb) })
        }
        // Unreachable in practice (check_row rejects other types first) but
        // the type system cannot see that; the scalar mirror is the safe
        // placeholder that keeps this total.
        (w, false) => Ok(dot_row_scalar_ty(w, wrow, acol, nb)),
        _ => Ok(dot_row_scalar_ty(w, wrow, acol, nb)),
    }
}

/// The scalar mirror behind [`dot_row`] — the fallback the crate runs when
/// AVX2 is absent. Same validation, same value as the AVX2 kernel, bit for
/// bit: the per-super-block integer sums are exact in i32 (order-free), and
/// the f32 scaling is the same expression in the same order. Public so the
/// gate can compare both paths on one machine (gate 4).
pub fn dot_row_scalar(w: GgmlType, wrow: &[u8], acol: &[u8], k: usize) -> Result<f32, QdotError> {
    let nb = check_row(w, wrow.len(), acol.len(), k)?;
    Ok(dot_row_scalar_ty(w, wrow, acol, nb))
}

fn dot_row_scalar_ty(w: GgmlType, wrow: &[u8], acol: &[u8], nb: usize) -> f32 {
    match w {
        GgmlType::Q4_K => dot_q4k_q82x4_emul(wrow, acol, nb),
        _ => dot_q3k_q8k_scalar(wrow, acol, nb),
    }
}

/// The AVX2 kernel behind [`dot_row`], for the gate's path comparison
/// (gate 4). Panics when the CPU has no AVX2 — on such a machine there is
/// nothing to compare against and the caller (a gate) wants the loud
/// failure, not a quiet fall back.
pub fn dot_row_avx2(w: GgmlType, wrow: &[u8], acol: &[u8], k: usize) -> Result<f32, QdotError> {
    let nb = check_row(w, wrow.len(), acol.len(), k)?;
    assert!(
        std::arch::is_x86_feature_detected!("avx2"),
        "dot_row_avx2 called on a CPU without AVX2"
    );
    match w {
        GgmlType::Q4_K => {
            // SAFETY: asserted just above; lengths validated by check_row.
            Ok(unsafe { dot_q4k_q82x4_avx2(wrow, acol, nb) })
        }
        _ => {
            // SAFETY: asserted just above; lengths validated by check_row.
            Ok(unsafe { dot_q3k_q8k_avx2(wrow, acol, nb) })
        }
    }
}

/// Shape validation shared by every `dot_row` entry point. Returns the
/// weight super-block count on success. The supported-type check comes
/// first so an unsupported type never touches the data.
fn check_row(w: GgmlType, wrow_len: usize, acol_len: usize, k: usize) -> Result<usize, QdotError> {
    if !matches!(w, GgmlType::Q3_K | GgmlType::Q4_K) {
        return Err(QdotError::UnsupportedType(w));
    }
    if !k.is_multiple_of(256) {
        return Err(QdotError::UnalignedK { k });
    }
    let nb = k / 256;
    let need_w = nb
        * match w {
            GgmlType::Q4_K => Q4K_BLOCK,
            _ => Q3K_BLOCK,
        };
    if wrow_len < need_w {
        return Err(QdotError::ShortWeightRow {
            have: wrow_len,
            need: need_w,
            k,
        });
    }
    let need_a = nb
        * match w {
            GgmlType::Q4_K => 2 * Q82X4_STRIDE,
            _ => Q8K_STRIDE,
        };
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
#[inline]
fn nearest_int(fval: f32) -> i32 {
    let val = fval + 12582912.0;
    let i = f32::to_bits(val);
    ((i & 0x007f_ffff) as i32) - 0x0040_0000
}

/// Port of `quantize_row_q8_K_ref` (ggml-quants.c:3974) over one 256-value
/// block, into the vendored ik fork's 296-byte `block_q8_K` layout: d f32
/// @0, sum f32 @4 (ik-only field; the C ref leaves it unwritten and nothing
/// in the q3_K dot reads it, so it is zeroed), qs[256] s8 @8, bsums[16] s16
/// @264. Via `crates/q3k-cpu/src/main.rs` (verified against ik's reference
/// outputs there); rewritten from raw pointers to slices, arithmetic
/// unchanged.
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
    out[4..8].fill(0); // the ik-only `sum` field: zero, never read by the dot
    if amax == 0.0 {
        out.fill(0);
        return;
    }
    // The SIGNED extreme sets the scale (the round trip's convention; the
    // comment on quantize_row_q8_k_roundtrip explains why amax is a trap).
    let iscale = -127.0f32 / max;
    for (j, &v) in x.iter().enumerate() {
        // ggml clamps one-sided at +127: the negative side reaches -127
        // exactly by construction (|iscale * x| <= 127 * (1 + 2^-23), which
        // rounds to -127 at the extreme), so no negative clamp is needed.
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

// ------------------------------------------------------------- q3_K dot

/// `get_scale_shuffle_q3k`'s table (ggml-quants.c:4041), verbatim. Vector f
/// broadcasts scale pair (2f, 2f+1) in the low 16 bytes and (2f+2, 2f+3) in
/// the high 16: the two 16-weight sub-blocks of a 32-value field.
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

/// Horizontal i32 sum of an `__m256i` — the integer counterpart of ggml's
/// `hsum_float_8`, applied before the i32 -> f32 conversion so the
/// reduction stays exact (|sumi| <= ~4.2e6 per super-block; see
/// `dot_q3k_q8k_avx2` for the bound).
///
/// # Safety
/// AVX2 (or at least SSE2) must be available on the target.
#[inline(always)]
unsafe fn hsum_i32(v: __m256i) -> i32 {
    unsafe {
        // SAFETY: register-only intrinsics, no memory access.
        //
        // fold the two 128-bit halves, then lane-pair twice:
        // 0x4E = perm [2,3,0,1], 0x8D = perm [1,0,3,2]
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

/// One field of one 128-value half: decode the shared weight bytes and dot
/// them against the column's matching Q8_K span. SHIFT = 2f (q3bits shift),
/// BIT = 4j+f (hmask bit), `q8` points at the column's super-block quants +
/// 128j + 32f — const args because the shift intrinsics need compile-time
/// immediates; the C kernel unrolls its 2x4 loop the same way.
///
/// # Safety
/// `q8` must point at 32 readable bytes inside the column buffer (the
/// caller derives it from the bounds-checked `acol`); AVX2 must be present.
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
    unsafe {
        // SAFETY: AVX2 present and `q8` points at 32 readable bytes per the
        // fn contract; everything except the annotated load is register-only.
        let q3l = _mm256_and_si256(_mm256_srli_epi16::<SHIFT>(q3bits), masks.m3);
        let q3h = _mm256_slli_epi16::<2>(_mm256_srli_epi16::<BIT>(_mm256_andnot_si256(
            hbits,
            _mm256_slli_epi16::<BIT>(masks.mone),
        )));
        let sc = _mm256_shuffle_epi8(scales_j, shuf_f);
        // SAFETY: unaligned 32-byte load at q8, inside the column buffer.
        let q8f = _mm256_loadu_si256(q8 as *const __m256i);
        // q3l (u8 0..3) and q3h (u8 {0,4}) both sit inside maddubs' s16
        // saturation (2*4*127 = 1016 < 32767); the madd product stays inside
        // s16 too (31*762 = 23622).
        let q8s = _mm256_maddubs_epi16(q3h, q8f);
        let p = _mm256_maddubs_epi16(q3l, q8f);
        let p = _mm256_sub_epi16(p, q8s);
        let p = _mm256_madd_epi16(sc, p);
        *sumi = _mm256_add_epi32(*sumi, p);
    }
}

/// The AVX2 row dot: port of the `__AVX2__` branch of
/// `ggml_vec_dot_q3_K_q8_K` (ggml-quants.c:6482), reduced from the
/// q3k-cpu pre-study's M <= 8 column bundle (`crates/q3k-cpu/src/main.rs`)
/// to the one column `matmul_q` dots at a time, with the super-block count
/// from `k` instead of the pre-study's K = 2048 constant.
///
/// One deliberate difference from the C, kept from the pre-study: the C
/// converts `sumi` with `_mm256_cvtepi32_ps` and accumulates in f32 lanes
/// with an fmadd; this port folds `sumi` to one i32 first and scales in
/// scalar f32 (`dcol * d * sumi as f32`). Every partial sum is an exact
/// integer (per super-block |sumi| <= 256 values * scale 31 * quant spread
/// 7 * 127 ≈ 4.2e6, well below the 2^24 an i32 holds exactly in f32), so
/// the value is the same, and the scalar mirror can match it bit for bit
/// without an FMA Rust may or may not emit.
///
/// # Safety
/// The CPU must support AVX2, and the caller must have validated lengths:
/// `wrow` at nb*110 readable bytes, `acol` at nb*296 readable bytes
/// (`check_q3k` does).
// `target_feature` is not optional here. Without it this fn compiles for the
// baseline SSE2 target and LLVM legalizes the 256-bit intrinsic bodies into
// narrow emulated sequences — measured on the box (MUL-26, 2026-09-20,
// `qdot-rate`): 0.3 GB/s per core plain-release vs 6+ GB/s with the feature
// enabled, a 20x gap that sat inside every engine number since the wiring
// round. The runtime check lives in `dot_row`; this attribute is what makes
// the detected feature reach codegen.
#[target_feature(enable = "avx2")]
unsafe fn dot_q3k_q8k_avx2(wrow: &[u8], acol: &[u8], nb: usize) -> f32 {
    unsafe {
        // SAFETY: AVX2 present and both slices hold nb super-blocks per the
        // fn contract; the pointer arithmetic below stays inside them, with
        // the reasoning repeated at each load.
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
            // hmask[32] @ +0
            // SAFETY: unaligned load inside the super-block.
            let hbits = _mm256_loadu_si256(base as *const __m256i);
            // scales[12] @ +96 decoded with the aux[] shuffle, verbatim.
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
                // C reads (aux[2] >> 0); the identity shift is dropped.
                ((aux[0] & kmask2) | ((aux[2] & kmask1) << 4)) as i32,
            );
            let s128 = _mm_sub_epi8(s128, m32);
            let all_scales = _mm256_cvtepi8_epi16(s128);
            let l = _mm256_extracti128_si256(all_scales, 0);
            let h = _mm256_extracti128_si256(all_scales, 1);
            let scales = [_mm256_set_m128i(l, l), _mm256_set_m128i(h, h)];

            // Super-block scale d: f16 at +108.
            // SAFETY: one unaligned u16 read inside the super-block.
            let d = half_to_f32((base.add(108) as *const u16).read_unaligned());

            let q8base = acol.as_ptr().add(sb * Q8K_STRIDE + 8);
            let mut sumi = _mm256_setzero_si256();
            for (j, scj) in scales.iter().enumerate() {
                // qs[64] @ +32, 32 bytes per half j.
                // SAFETY: unaligned load inside the super-block.
                let q3bits = _mm256_loadu_si256(base.add(32 + 32 * j) as *const __m256i);
                let scj = *scj;
                // j=0: fields 0..3 -> (SHIFT, BIT, q8off) = (2f, f, 32f)
                // j=1: fields 0..3 -> (2f, 4+f, 128+32f)
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
            // d of Q8_K block sb sits at the column block start.
            // SAFETY: unaligned f32 read inside the column buffer.
            let dcol = (acol.as_ptr().add(sb * Q8K_STRIDE) as *const f32).read_unaligned();
            acc += dcol * d * hsum_i32(sumi) as f32;
        }
        acc
    }
}

/// The scalar mirror of the AVX2 kernel — same integer decode, same i32
/// super-block sums (exact, so the element order inside a super-block does
/// not affect the value), same f32 scaling in the same order, so the two
/// paths are bit-identical by construction (gate 4 asserts it). Written
/// against `gguf::quant`'s `dequant_q3_k` decode order, which stage 0
/// verified against ggml's `ggml_dequantize_row_q3_K` — this is the second,
/// independent implementation of that geometry in the repo, and gate 3's
/// f64 reference in the test file is the third.
fn dot_q3k_q8k_scalar(wrow: &[u8], acol: &[u8], nb: usize) -> f32 {
    const KMASK1: u32 = 0x0303_0303;
    const KMASK2: u32 = 0x0f0f_0f0f;
    let mut acc = 0.0f32;
    for sb in 0..nb {
        let blk = &wrow[sb * Q3K_BLOCK..sb * Q3K_BLOCK + Q3K_BLOCK];
        let hmask = &blk[0..32];
        let qs = &blk[32..96];
        // 12 packed scale bytes -> 16 int8 scales via the same aux[] unpack
        // as the AVX2 kernel (word i holds scales 4i..4i+3).
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
        // The unpacked byte is 0..63; the sub-block scale is byte - 32 (the
        // AVX2 kernel's _mm_sub_epi8(s128, m32), and `scales[is] - 32` in
        // gguf::quant's dequant_q3_k — forgetting it was gate 4's first
        // catch).
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
// MUL-27. Port of ik's `mul_mat_qX_K_q8_2_X4_T<DequantizerQ4K_AVX2, 1>`
// (iqk_gemm_kquants.cpp:787) — the pairing the oracle actually dispatches
// (iqk_gemm_kquants.cpp:2751/2768), taken from the non-HAVE_FANCY_SIMD
// branch (Zen 3 has no dpbusd) — plus an instruction-graph emulator of it,
// same regime as the Q3_K round: the mirror matches the kernel bit for bit,
// and the kernel must agree with IK'S OWN x4 KERNEL to within 1 ULP on real
// rows (`tools/ref/q4k_x4_ref.cpp` runs ik's kernel table entry and dumps
// both its activation coding and its row outputs).
//
// The activation side is `quantize_row_q8_2_x4`'s x86 coding (ported as
// `quantize_q82x4_col`): four 32-value blocks per 144-byte group — bf16
// scale d in d[0..4], the RAW i16 integer sum in d[4..8], quants qs[128].
// The kernel turns d back into f32 by a 16-bit shift (bf16 -> f32 bits) and
// the sum by a sign-extend + cvt. Two details the aarch64 codepath hides
// (it stores bf16(d*sum) there): this port is specifically the x86 one the
// box runs.
//
// `#[target_feature]` is not optional (the MUL-26 lesson, measured 21x on
// this box): without it LLVM legalizes the 256-bit intrinsic bodies for the
// SSE2 baseline. FMA is required (`_mm256_fmadd_ps` twice per super-block),
// and the mirror reproduces both with `f32::mul_add` — the same
// correctly-rounded fused operation, so bit identity does not depend on
// what the optimizer emits around it.

/// `block_q4_K` (ggml-common.h): d f16 @0, dmin f16 @2, scales u8[12] @4,
/// qs u8[128] @16 — 256 values, 144 bytes.
const Q4K_BLOCK: usize = 144;

/// `make_q4_scales` (iqk_common.h:175): the 12 packed scale/min bytes into
/// u32[4] — result bytes 0..8 the six-bit scales, bytes 8..16 the six-bit
/// mins. The same decode the q8_K kernel's utmp chain performs (checked
/// equivalent when that kernel landed); written pure here because this
/// kernel's order has no dependency to carry.
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

/// fp32 -> bf16 bits, `ggml_compute_fp32_to_bf16` verbatim (ggml-impl.h:106):
/// NaN is forced quiet; everything else rounds to nearest even by the
/// `+ 0x7fff + lsb` trick, u32 wrapping included.
#[inline]
fn fp32_to_bf16_bits(s: f32) -> u16 {
    let u = s.to_bits();
    if (u & 0x7fff_ffff) > 0x7f80_0000 {
        ((u >> 16) | 64) as u16
    } else {
        (u.wrapping_add(0x7fff + ((u >> 16) & 1)) >> 16) as u16
    }
}

/// bf16 -> f32 the way the kernel does it: bits shifted into the top half
/// (`slli_epi32` + cast) — `ggml_compute_bf16_to_fp32`'s operation. Exact
/// for every normal bf16 (the domain of real scales).
#[inline]
fn bf16_bits_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

/// Port of the `__x86_64__` branch of `quantize_row_q8_2_x4`
/// (iqk_quantize.cpp:1005) for Block = block_q8_2. Per 32-value block:
/// `d = amax / 127` rounded through bf16 — the rounded value is BOTH stored
/// and used — quants are round-to-nearest-even, and the `s` slot takes the
/// RAW i16 integer sum (see the section header). Blocks 4g..4g+4 interleave
/// into one x4 group: d u16[8] (scales then sums), qs s8[128].
///
/// Assumes finite activations (the oracle dumps are): the C's SIMD max/NaN
/// lanes and a NaN's f32->i8 conversion have no scalar meaning worth
/// mirroring, and a NaN here would fail the byte-identity gate loudly
/// anyway.
fn quantize_q82x4_col(x: &[f32], out: &mut [u8]) {
    let (blocks, _) = x.as_chunks::<32>();
    let (groups, _) = out.as_chunks_mut::<Q82X4_STRIDE>();
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
                let q = (v * id).round_ties_even() as i32 as i8;
                qs[m] = q as u8;
                isum += q as i32;
            }
            group[8 + 2 * ir..8 + 2 * ir + 2].copy_from_slice(&(isum as i16).to_le_bytes());
        }
    }
}

/// f16 -> f32 by bit assembly. Exact for every finite input (power-of-two
/// scaling, no rounding), which is the whole domain a scale d/dmin lives
/// in; `avx2` therefore suffices — no F16C load needed. Same values as
/// GGML_CPU_FP16_TO_FP32 for finite inputs.
#[inline]
fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if frac == 0 {
            sign
        } else {
            // Subnormal f16: normalize. Unreachable for real scales but the
            // function must be total to live next to the kernel.
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

/// `hsum_float_8` (ggml-quants.c:59) and the `acc_m` fold at the end of the
/// q4_K kernel: the reduction ORDER is load-bearing — the scalar mirror
/// repeats exactly this sequence, because f32 addition is not associative
/// and bit identity of the two paths is the gate.
#[inline]
unsafe fn hsum_float_8(x: __m256) -> f32 {
    unsafe {
        let mut res = _mm256_extractf128_ps(x, 1);
        res = _mm_add_ps(res, _mm256_castps256_ps128(x));
        res = _mm_add_ps(res, _mm_movehl_ps(res, res));
        res = _mm_add_ss(res, _mm_movehdup_ps(res));
        _mm_cvtss_f32(res)
    }
}

/// Port of `mul_mat_qX_K_q8_2_X4_T<DequantizerQ4K_AVX2, 1>`
/// (iqk_gemm_kquants.cpp:787) reduced to `matmul_q`'s shape: one weight row
/// against one q8_2_x4 activation column. The template's iy loop is gone
/// (nrc_y = 1), its ix loop is the caller's row loop; everything else is
/// the C's arithmetic, non-FANCY branch.
///
/// v2 scheduling (2026-09-20, same round): the verbatim port measured
/// 14.4 GB/s against ik's own 16.3–17.1 on the same shape (qdot-rate +
/// tools/ref/q4k_x4_rate.cpp), so this version re-schedules for ILP while
/// provably keeping the bits: every super-block's integer work is issued
/// BEFORE its three fmadds, super-blocks are unrolled ×2 (a's fmas then
/// b's fmas — the accumulation order on `accd` is unchanged, ascending,
/// mins/j0/j1), the `d8` round trip through memory became register halves
/// (lo128/hi128 of `dy` are exactly d8[0..4] and d8[4..8]), and the two
/// lane multiplies `(-1.0) * (dmin * m)` collapsed into one `(-dmin) * m`
/// (IEEE negation is exact and sign-symmetric through rounding). The
/// emulator and the 1-ULP-ik gate both still hold, which is the proof the
/// re-schedule moved no bit.
///
/// # Safety
/// The CPU must support AVX2+FMA, and the caller must have validated
/// lengths: `wrow` at nb*144 readable bytes, `acol` at nb*288 (`check_row`
/// does).

/// Port of `mul_mat_qX_K_q8_2_X4_T<DequantizerQ4K_AVX2, 1>`
/// (iqk_gemm_kquants.cpp:787) reduced to `matmul_q`'s shape: one weight row
/// against one q8_2_x4 activation column. The template's iy loop is gone
/// (nrc_y = 1), its ix loop is the caller's row loop; everything else is
/// the C's arithmetic, non-FANCY branch.
///
/// Scheduling experiments (2026-09-20, same round), all bit-preserving and
/// all measured on the box (360,448 rows x 1152 B, single core, qdot-rate
/// vs ik's own kernel via tools/ref/q4k_x4_rate.cpp):
///
/// | form | GB/s |
/// |---|---|
/// | verbatim port | 14.4 |
/// | + register d8 halves, one-multiply mins, hoisted integer chains | 14.1 |
/// | + super-block unroll behind a helper (struct, then tuple) | 12.2 / 12.4 |
/// | ik's own kernel (gcc -O2 -mavx2) | 16.3–17.1 |
///
/// The helper-split unrolls LOST ~13% — even tuple-shaped and
/// `#[target_feature]`'d, the split rounds values through memory and the
/// MUL-26 lesson makes the feature attribute on any split-out helper
/// load-bearing (its absence measured 0.6 GB/s, 24x down, before it was
/// caught). What survived here are the wins that did not hurt: dy's halves
/// stay in registers (the C's `d8` store+reload was register-pressure
/// relief gcc needed and this schedule does not), the two lane multiplies
/// `(-1.0) * (dmin * m)` collapsed into one `(-dmin) * m` (identical bits —
/// IEEE negation is exact and sign-symmetric through rounding), and both
/// chunks' integer chains are issued before the super-block's three
/// fmadds. The fmadds stay strictly ordered — mins/j0/j1 per super-block,
/// super-blocks ascending — so the emulator and the 1-ULP-ik gate both
/// hold: nothing here moved a bit.
///
/// The residual 14.1 vs 16.3–17.1 gap is codegen, not structure (same
/// intrinsics, same order; time-per-ROW is flat across this and the Q3_K
/// kernel's shapes — an op/latency wall, not bandwidth). The engine's
/// regime is DRAM-bound at 147.7 GB/s over 32 threads = 4.6 GB/s/thread,
/// three times below either kernel's rate — the runner decides what the
/// gap is worth there, not this bench.
///
/// # Safety
/// The CPU must support AVX2+FMA, and the caller must have validated
/// lengths: `wrow` at nb*144 readable bytes, `acol` at nb*288 (`check_row`
/// does).
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_q4k_q82x4_avx2(wrow: &[u8], acol: &[u8], nb: usize) -> f32 {
    unsafe {
        let ml = _mm256_set1_epi8(0xF);
        let mut accd = _mm256_setzero_ps();

        for i in 0..nb {
            let wb = &wrow[i * Q4K_BLOCK..(i + 1) * Q4K_BLOCK];

            let d = f16_bits_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
            let dmin = f16_bits_to_f32(u16::from_le_bytes([wb[2], wb[3]]));

            // utmp decode; utmp[0..2] = 8 scale bytes, utmp[2..4] = 8 min bytes.
            let utmp = make_q4_scales(&wb[4..16]);
            // mins = (-dmin) * mins_u8 — the C's two multiplies (dmin, then
            // -1.0) collapsed into one; identical bits, IEEE negation being
            // exact and sign-symmetric through rounding.
            // SAFETY: 8 readable bytes inside the local u32[4].
            let mins_v = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(
                utmp.as_ptr().add(2) as *const __m128i,
            )));
            let mins = _mm256_mul_ps(_mm256_set1_ps(-dmin), mins_v);

            // The two x4 groups feeding super-block i: their bf16 scales as
            // f32 (bits << 16). dy's halves ARE the d8 the C stores to
            // memory and reloads per chunk — j=0 reads lanes 0..4, j=1
            // lanes 4..8; keeping them in registers is this port's change.
            // SAFETY: 8 readable bytes at the head of each validated group.
            let g0 = acol.as_ptr().add((2 * i) * Q82X4_STRIDE);
            let g1 = acol.as_ptr().add((2 * i + 1) * Q82X4_STRIDE);
            let d4_1 = _mm_cvtepu16_epi32(_mm_loadl_epi64(g0 as *const __m128i));
            let d4_2 = _mm_cvtepu16_epi32(_mm_loadl_epi64(g1 as *const __m128i));
            let dy = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_set_m128i(d4_2, d4_1), 16));
            let dy4 = [_mm256_castps256_ps128(dy), _mm256_extractf128_ps(dy, 1)];
            // The raw i16 sums, sign-extended, back to f32. `d + 4` in the C
            // is a u16 offset — byte 8, the start of the sums.
            // SAFETY: 8 readable bytes at offset 8 of each group.
            let m4_1 = _mm_cvtepi16_epi32(_mm_loadl_epi64(g0.add(8) as *const __m128i));
            let m4_2 = _mm_cvtepi16_epi32(_mm_loadl_epi64(g1.add(8) as *const __m128i));
            let myi = _mm256_set_m128i(m4_2, m4_1);
            let my = _mm256_mul_ps(dy, _mm256_cvtepi32_ps(myi));

            // all_scales = d * scales_u8 per lane; lo/hi halves broadcast.
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

            // Both chunks' integer chains first (independent work hoisted so
            // it overlaps), then the three fmadds in the C's order — min
            // term, chunk 0, chunk 1. Same bits.
            let q4 = &wb[16..16 + 128];
            let mut sumi_f = [_mm256_setzero_ps(), _mm256_setzero_ps()];
            for j in 0..2 {
                // Q4Bits_AVX2::prepare: nibble-split of qs bytes [64j, 64j+64).
                // SAFETY: loads stay inside the validated 144-byte block.
                let bits0 = _mm256_loadu_si256(q4.as_ptr().add(64 * j) as *const __m256i);
                let values0 = _mm256_and_si256(bits0, ml);
                let values1 = _mm256_and_si256(_mm256_srli_epi16(bits0, 4), ml);
                let bits1 = _mm256_loadu_si256(q4.as_ptr().add(64 * j + 32) as *const __m256i);
                let values2 = _mm256_and_si256(bits1, ml);
                let values3 = _mm256_and_si256(_mm256_srli_epi16(bits1, 4), ml);

                // The x4 group for chunk j: its four 32-byte quant slices.
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

// The instruction-graph emulator of [`dot_q4k_q82x4_avx2`] — safe Rust, no
// ISA requirement, every operation in Intel's exact semantics. It is the
// crate's no-AVX2 fallback for Q4_K AND the bit-identity gate's mirror. The
// x4 kernel's i16 pack chain assigns lanes in a way hand-derived lane
// algebra gets wrong (the q8_K round's per-lane derivation does not carry
// over); emulating each intrinsic on arrays is right by construction and
// costs nothing to check — the same loop structure as the kernel, arrays
// for registers.

/// `_mm256_maddubs_epi16(u8x32, s8x32)`: i16 lane p from adjacent byte
/// pairs, with i16 saturation (unreachable here — max |15*127*2| = 3810,
/// and the saturate is written anyway so the emulator stays honest).
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

/// `_mm256_madd_epi16(set1(1), t)`: i32 lane p = t[2p] + t[2p+1] (i32
/// saturation unreachable at these magnitudes).
#[inline]
fn emul_madd1(t: [i16; 16]) -> [i32; 8] {
    let mut r = [0i32; 8];
    for p in 0..8 {
        r[p] = t[2 * p] as i32 + t[2 * p + 1] as i32;
    }
    r
}

/// `_mm256_unpacklo_epi32` on i16 lanes: per 128-bit half, interleave the
/// low two dwords of a and b.
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

/// `_mm256_unpackhi_epi32` on i16 lanes: per half, the high two dwords.
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

/// `_mm256_unpacklo_epi64` on i16 lanes: per half, low qword of a then b.
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

/// `_mm256_unpackhi_epi64` on i16 lanes: per half, high qword of a then b.
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

/// The emulator: the kernel's graph, one weight row against one q8_2_x4
/// column, bit-identical to [`dot_q4k_q82x4_avx2`] by construction — same
/// operations, same order, `mul_add` where the kernel fuses.
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

        // mins lanes: (-1.0) * (dmin * mins_u8[l]).
        let mut mins = [0.0f32; 8];
        for l in 0..8 {
            let m = mb[l / 4][l % 4];
            mins[l] = -1.0 * (dmin * (m as f32));
        }

        // d8 lanes: bf16 scales, bits << 16; lanes 0..3 group 2i, 4..7 group 2i+1.
        let mut d8 = [0.0f32; 8];
        for l in 0..4 {
            d8[l] = bf16_bits_to_f32(u16::from_le_bytes([g0[2 * l], g0[2 * l + 1]]));
            d8[4 + l] = bf16_bits_to_f32(u16::from_le_bytes([g1[2 * l], g1[2 * l + 1]]));
        }
        // my lanes: d8[l] * (raw i16 sum)[l]; one fused min-term accumulate.
        for l in 0..4 {
            let s0 = i16::from_le_bytes([g0[8 + 2 * l], g0[9 + 2 * l]]) as f32;
            let s1 = i16::from_le_bytes([g1[8 + 2 * l], g1[9 + 2 * l]]) as f32;
            accd[l] = (d8[l] * s0).mul_add(mins[l], accd[l]);
            accd[4 + l] = (d8[4 + l] * s1).mul_add(mins[4 + l], accd[4 + l]);
        }

        // all_scales lanes: d * scales_u8[l]; the j-th broadcast pair takes
        // lanes 4j + (l & 3).
        let mut all_scales = [0.0f32; 8];
        for l in 0..8 {
            all_scales[l] = d * (sb[l / 4][l % 4] as f32);
        }

        let q4 = &wb[16..16 + 128];
        for j in 0..2 {
            // Q4Bits prepare: nibble-split of bytes [64j, 64j+64).
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

            // d4d8 lane l = scales[j][l] * dy4[l] = all_scales[4j+(l&3)] *
            // d8[4j+(l&3)]; one fused accumulate.
            for l in 0..8 {
                let idx = 4 * j + (l & 3);
                let d4d8 = all_scales[idx] * d8[idx];
                accd[l] = d4d8.mul_add(sumi[l] as f32, accd[l]);
            }
        }
    }

    // hsum_float_8 in the kernel's exact order.
    let t = [
        accd[0] + accd[4],
        accd[1] + accd[5],
        accd[2] + accd[6],
        accd[3] + accd[7],
    ];
    (t[0] + t[2]) + (t[1] + t[3])
}
