//! qdot — the fused Q3_K x Q8_K row dot for the host CPU.
//!
//! `model::ops::matmul_q` today dequantizes every weight row to f32
//! (`gguf::quant::dequant_row`), round-trips the activations through Q8_K
//! (`gguf::quant::quantize_row_q8_k_roundtrip`), and dots the two in f32.
//! ik_llama.cpp never materializes f32 weights: it dots the quantized codes
//! in integer lanes and scales once per super-block. This crate is that
//! kernel for one weight type, in the shape `matmul_q` will call it: one
//! weight row against one quantized activation column, `k` a runtime
//! parameter (the model uses k in {2048, 512}, both multiples of 256).
//!
//! Nothing here is invented — every decode is a port with its source cited:
//!   * the activation quantizer is ggml's `quantize_row_q8_K_ref`
//!     (ggml-quants.c:3974, read in the vendored ik fork) into that fork's
//!     296-byte `block_q8_K`: d f32 @0, sum f32 @4 (an ik-only field the C
//!     ref leaves unwritten and nothing in the q3_K dot reads — zeroed),
//!     s8 qs[256] @8, s16 bsums[16] @264;
//!   * the row dot is the `__AVX2__` branch of `ggml_vec_dot_q3_K_q8_K`
//!     (ggml-quants.c:6482), reduced from the q3k-cpu pre-study's M <= 8
//!     column bundle to the one column `matmul_q` dots at a time, with the
//!     super-block count from `k` instead of the pre-study's K = 2048
//!     constant.
//!   * both arrived via `crates/q3k-cpu/src/main.rs`, which verified them
//!     against ik's own reference outputs at 1e-2 over 24 configs
//!     (q3k-cpu's gate, plus its stage-0 geometry check of
//!     `dequantize_row_q3_K` at 1e-7).
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
//! Scope: Q3_K only this round (Q3_K is the bulk of the decode step; the
//! exact share is the lead's profile to quote, not ours). The API takes the
//! weight type so later rounds extend the table without changing call sites.
//! A scalar mirror of the AVX2 kernel lives here too, so the crate is
//! testable on machines without AVX2 and the two paths are compared bit for
//! bit by the gate (`tests/qdot.rs`).

use std::arch::x86_64::*;
use std::fmt;

use gguf::GgmlType;
use gguf::quant::half_to_f32;

/// `sizeof(block_q3_K)` (ggml-common.h): 110 bytes / 256 values.
const Q3K_BLOCK: usize = 110;
/// `sizeof(block_q8_K)` in the vendored ik fork: 296 bytes / 256 values —
/// d f32 @0, sum f32 @4, qs[256] s8 @8, bsums[16] s16 @264.
const Q8K_STRIDE: usize = 296;

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
    // Q4_K is deliberately NOT wired yet (2026-09-20, MUL-27): its kernel is
    // landed (dot_q4k_q8k_* below, gated in tests) but pairs Q8_K activations,
    // and the oracle's own path pairs Q4_K with Q8_2_X4 — ik's dispatch is
    // `mul_mat_qX_K_q8_2_X4_T<DequantizerQ4K_AVX2>` (iqk_gemm_kquants.cpp:2751,
    // 2768) with expected_type_B = GGML_TYPE_Q8_2_X4. Wiring the q8_K pairing
    // moved kqv_out-0 by 6.4e-3 against the oracle (gate 5e-3) — an
    // activation-format delta, not a bug, and opening the band for it would
    // trade oracle fidelity for nothing. The next round ports the x4 kernel;
    // the q8_K one keeps its gates and its role as the decode's proof.
    w == GgmlType::Q3_K && std::arch::is_x86_feature_detected!("avx2")
}

/// Bytes one quantized activation column of `k` values occupies for `w`'s
/// format: `k / 256` blocks of the 296-byte `block_q8_K`.
///
/// Panics if `w` has no activation format in this build or `k % 256 != 0` —
/// both are caller constants at allocation time, and a made-up size for a
/// partial block would allocate a buffer `quantize_col` cannot fill.
pub fn col_bytes(w: GgmlType, k: usize) -> usize {
    assert!(
        matches!(w, GgmlType::Q3_K | GgmlType::Q4_K),
        "qdot has no activation format for {w:?} in this build"
    );
    assert!(
        k.is_multiple_of(256),
        "qdot activation columns are whole 256-value super-blocks, k = {k}"
    );
    (k / 256) * Q8K_STRIDE
}

/// Quantize one activation column into the block format `w` implies.
/// `out.len()` must be `col_bytes(w, x.len())`; panics otherwise (see
/// [`col_bytes`] — allocation-time constants, not data-dependent).
///
/// Port of `quantize_row_q8_K_ref` (ggml-quants.c:3974) over each 256-value
/// block, with the scale taken from the SIGNED extreme (`iscale =
/// -127/max`) exactly as `gguf::quant::quantize_row_q8_k_roundtrip` does.
/// The round trip and this coder are the same quantization in two
/// representations; the gate checks that bit for bit (`tests/qdot.rs`,
/// gate 1).
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
    let nb = check_q3k(w, wrow.len(), acol.len(), k)?;
    let avx2 =
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma");
    match (w, avx2) {
        (GgmlType::Q3_K, true) => {
            // SAFETY: AVX2 was just detected; lengths validated by check_q3k.
            Ok(unsafe { dot_q3k_q8k_avx2(wrow, acol, nb) })
        }
        (GgmlType::Q4_K, true) => {
            // SAFETY: AVX2+FMA were just detected; lengths validated above.
            Ok(unsafe { dot_q4k_q8k_avx2(wrow, acol, nb) })
        }
        // Unreachable in practice (check_q3k rejects other types first) but
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
    let nb = check_q3k(w, wrow.len(), acol.len(), k)?;
    Ok(dot_row_scalar_ty(w, wrow, acol, nb))
}

fn dot_row_scalar_ty(w: GgmlType, wrow: &[u8], acol: &[u8], nb: usize) -> f32 {
    match w {
        GgmlType::Q4_K => dot_q4k_q8k_scalar(wrow, acol, nb),
        _ => dot_q3k_q8k_scalar(wrow, acol, nb),
    }
}

/// The AVX2 kernel behind [`dot_row`], for the gate's path comparison
/// (gate 4). Panics when the CPU has no AVX2 — on such a machine there is
/// nothing to compare against and the caller (a gate) wants the loud
/// failure, not a quiet fall back.
pub fn dot_row_avx2(w: GgmlType, wrow: &[u8], acol: &[u8], k: usize) -> Result<f32, QdotError> {
    let nb = check_q3k(w, wrow.len(), acol.len(), k)?;
    assert!(
        std::arch::is_x86_feature_detected!("avx2"),
        "dot_row_avx2 called on a CPU without AVX2"
    );
    match w {
        GgmlType::Q4_K => {
            // SAFETY: asserted just above; lengths validated by check_q3k.
            Ok(unsafe { dot_q4k_q8k_avx2(wrow, acol, nb) })
        }
        _ => {
            // SAFETY: asserted just above; lengths validated by check_q3k.
            Ok(unsafe { dot_q3k_q8k_avx2(wrow, acol, nb) })
        }
    }
}

/// Shape validation shared by every `dot_row` entry point. Returns the
/// super-block count on success. The supported-type check comes first so an
/// unsupported type never touches the data.
fn check_q3k(w: GgmlType, wrow_len: usize, acol_len: usize, k: usize) -> Result<usize, QdotError> {
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
    let need_a = nb * Q8K_STRIDE;
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

// ------------------------------------------------------------- Q4_K x Q8_K
// MUL-27. Port of the `__AVX2__` branch of ik's `ggml_vec_dot_q4_K_q8_K`
// (ggml-quants.c:7263) plus its scalar mirror, with the same regime as the
// Q3_K round: the mirror matches the kernel bit for bit, and the kernel must
// be MORE accurate than the engine's dequant-then-round-trip path against an
// f64 exact answer (the prediction gate) — or the kernel is wrong.
//
// Pairing is q8_K (ik's own type-traits table, ggml.c:946), so the
// activation side is the SAME 296-byte block_q8_K the Q3_K path already
// quantizes — including the bsums the q4_K dot reads for the `min` term and
// the m-nibble correction. One activation format, two weight kernels.
//
// `#[target_feature]` is not optional (the MUL-26 lesson, measured 21x on
// this box): without it LLVM legalizes the 256-bit intrinsic bodies for the
// SSE2 baseline. FMA is required because the per-block accumulation is
// `_mm256_fmadd_ps`, and the mirror reproduces that with `f32::mul_add` —
// the same correctly-rounded fused operation, so bit identity does not
// depend on what the optimizer emits around it.

/// `block_q4_K` (ggml-common.h): d f16 @0, dmin f16 @2, scales u8[12] @4,
/// qs u8[128] @16 — 256 values, 144 bytes.
const Q4K_BLOCK: usize = 144;

/// `get_scale_shuffle_k4`'s table (ggml-quants.c:4086), verbatim: row `i`
/// is (2i, 2i+1) repeated sixteen times, so `_mm256_shuffle_epi8` puts
/// sub-block 2i's scale in the low i16 of every i32 lane and (2i+1)'s in the
/// high — the madd then applies both scales with no per-lane unpacking.
const K_SHUFFLE_Q4K: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut i = 0;
    while i < 8 {
        let mut j = 0;
        while j < 32 {
            t[i * 32 + j] = if j % 2 == 0 { 2 * i } else { 2 * i + 1 } as u8;
            j += 1;
        }
        i += 1;
    }
    t
};

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

/// Port of `ggml_vec_dot_q4_K_q8_K`'s `__AVX2__` branch (ggml-quants.c:7381
/// region), one weight row against one q8_K column. Value layout notes are
/// in the C: 32 bytes of qs hold 64 values (low nibbles then high), the
/// scales ride the madd, and `bsums` carries the `min` term.
///
/// # Safety
/// The CPU must support AVX2+FMA, and the caller must have validated
/// lengths: `wrow` at nb*144 readable bytes, `acol` at nb*296
/// (`check_q3k` does).
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_q4k_q8k_avx2(wrow: &[u8], acol: &[u8], nb: usize) -> f32 {
    unsafe {
        let m4 = _mm256_set1_epi8(0xF);
        let kmask1 = 0x3f3f_3f3fu32;
        let kmask2 = 0x0f0f_0f0fu32;
        let kmask3 = 0x0303_0303u32;

        let mut acc = _mm256_setzero_ps();
        let mut acc_m = _mm_setzero_ps();

        for i in 0..nb {
            let wb = &wrow[i * Q4K_BLOCK..(i + 1) * Q4K_BLOCK];
            let ab = &acol[i * Q8K_STRIDE..(i + 1) * Q8K_STRIDE];

            // y.d (f32) @0; x.d (f16) @0, x.dmin (f16) @2 — dmin negated, as in the C.
            let yd = f32::from_le_bytes([ab[0], ab[1], ab[2], ab[3]]);
            let d = yd * f16_bits_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
            let dmin = -yd * f16_bits_to_f32(u16::from_le_bytes([wb[2], wb[3]]));

            // The utmp scale decode, verbatim INCLUDING the dependency order
            // (utmp[3] reads the original utmp[2]/utmp[1] before they are
            // overwritten). Result bytes: utmp[0..8] = scales[0..7] (6-bit),
            // utmp[8..16] = mins[0..7] (6-bit).
            let mut utmp = [0u32; 4];
            for (t, w) in utmp.iter_mut().enumerate() {
                *w = u32::from_le_bytes([
                    wb[4 + 4 * t],
                    wb[5 + 4 * t],
                    wb[6 + 4 * t],
                    wb[7 + 4 * t],
                ]);
            }
            utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
            let uaux = utmp[1] & kmask1;
            utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
            utmp[2] = uaux;
            utmp[0] &= kmask1;

            let q4 = &wb[16..16 + 128];
            let q8 = &ab[8..8 + 256];

            // SAFETY: unaligned loads inside the validated 144/296-byte blocks.
            let mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(
                utmp[3] as i32,
                utmp[2] as i32,
                utmp[1] as i32,
                utmp[0] as i32,
            ));

            // q8sums: 16 i16 bsums @264 -> per-32 sums via hadd, madd with
            // the mins half -> 4 i32 lanes; acc_m accumulates the min term.
            // SAFETY: unaligned load inside the validated activation block.
            let q8sums = _mm256_loadu_si256(ab.as_ptr().add(264) as *const __m256i);
            let q8s = _mm_hadd_epi16(
                _mm256_extracti128_si256(q8sums, 0),
                _mm256_extracti128_si256(q8sums, 1),
            );
            let prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
            acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

            let sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
            let scales = _mm256_set_m128i(sc128, sc128);

            let mut sumi = _mm256_setzero_si256();
            for j in 0..4 {
                // SAFETY: shuf reads 32 static table bytes; the q4/q8 loads
                // stay inside the validated blocks (j < 4 => offsets < 128/256).
                let scale_l = _mm256_shuffle_epi8(
                    scales,
                    _mm256_loadu_si256(K_SHUFFLE_Q4K.as_ptr().add(32 * (2 * j)) as *const __m256i),
                );
                let scale_h = _mm256_shuffle_epi8(
                    scales,
                    _mm256_loadu_si256(
                        K_SHUFFLE_Q4K.as_ptr().add(32 * (2 * j + 1)) as *const __m256i
                    ),
                );

                let q4bits = _mm256_loadu_si256(q4.as_ptr().add(32 * j) as *const __m256i);
                let q4l = _mm256_and_si256(q4bits, m4);
                let q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

                let q8l = _mm256_loadu_si256(q8.as_ptr().add(64 * j) as *const __m256i);
                let mut p16l = _mm256_maddubs_epi16(q4l, q8l);
                p16l = _mm256_madd_epi16(scale_l, p16l);

                let q8h = _mm256_loadu_si256(q8.as_ptr().add(64 * j + 32) as *const __m256i);
                let mut p16h = _mm256_maddubs_epi16(q4h, q8h);
                p16h = _mm256_madd_epi16(scale_h, p16h);

                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16l, p16h));
            }

            acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
        }

        // The C's fold, verbatim: acc_m's four lanes first, then one add of
        // the two totals. The mirror repeats this order bit for bit.
        acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
        acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));
        hsum_float_8(acc) + _mm_cvtss_f32(acc_m)
    }
}

/// The scalar mirror of [`dot_q4k_q8k_avx2`]: same value bit for bit. Every
/// integer partial is exact (maddubs pairs max |15*127*2| = 3810 fit i16;
/// the i32 lane sums are exact), so only the f32 combine order matters, and
/// the comments below pin it to the vector's lane and fold structure.
fn dot_q4k_q8k_scalar(wrow: &[u8], acol: &[u8], nb: usize) -> f32 {
    let kmask1 = 0x3f3f_3f3fu32;
    let kmask2 = 0x0f0f_0f0fu32;
    let kmask3 = 0x0303_0303u32;

    let mut acc = [0.0f32; 8];
    let mut acc_m = [0.0f32; 4];

    for i in 0..nb {
        let wb = &wrow[i * Q4K_BLOCK..(i + 1) * Q4K_BLOCK];
        let ab = &acol[i * Q8K_STRIDE..(i + 1) * Q8K_STRIDE];

        let yd = f32::from_le_bytes([ab[0], ab[1], ab[2], ab[3]]);
        let d = yd * f16_bits_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
        let dmin = -yd * f16_bits_to_f32(u16::from_le_bytes([wb[2], wb[3]]));

        let mut utmp = [0u32; 4];
        for (t, w) in utmp.iter_mut().enumerate() {
            *w = u32::from_le_bytes([wb[4 + 4 * t], wb[5 + 4 * t], wb[6 + 4 * t], wb[7 + 4 * t]]);
        }
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        let uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;
        let sb = utmp[0].to_le_bytes(); // scales 0..3
        let sb1 = utmp[1].to_le_bytes(); // scales 4..7
        let mb = utmp[2].to_le_bytes(); // mins 0..3
        let mb1 = utmp[3].to_le_bytes(); // mins 4..7
        let sc = [sb[0], sb[1], sb[2], sb[3], sb1[0], sb1[1], sb1[2], sb1[3]];
        let mins = [mb[0], mb[1], mb[2], mb[3], mb1[0], mb1[1], mb1[2], mb1[3]];

        // bsums -> per-32 sums q8s[k] (i16 pair adds, exact).
        let mut q8s = [0i32; 8];
        for k in 0..8 {
            let b0 = i16::from_le_bytes([ab[264 + 4 * k], ab[265 + 4 * k]]);
            let b1 = i16::from_le_bytes([ab[266 + 4 * k], ab[267 + 4 * k]]);
            q8s[k] = b0 as i32 + b1 as i32;
        }
        // acc_m lanes: madd(mins, q8s) => lane L = mins[2L]*q8s[2L] + mins[2L+1]*q8s[2L+1].
        for lane in 0..4 {
            let prod = mins[2 * lane] as i32 * q8s[2 * lane]
                + mins[2 * lane + 1] as i32 * q8s[2 * lane + 1];
            acc_m[lane] = dmin.mul_add(prod as f32, acc_m[lane]);
        }

        let q4 = &wb[16..16 + 128];
        let q8 = &ab[8..8 + 256];
        // sumi lanes: per j, lane L gets scale(2j)*p16_l[2L] + scale(2j+1)*p16_l[2L+1]
        // plus the high-nibble twin, where p16[k] = q4[k2]*q8[k2] + q4[k2+1]*q8[k2+1].
        let mut sumi = [0i32; 8];
        for j in 0..4 {
            for half in 0..2 {
                // half 0: low nibbles of bytes [32j..32j+32) vs q8 [64j..64j+32)
                // half 1: high nibbles of the same bytes vs q8 [64j+32..64j+64)
                let mut p16 = [0i16; 16];
                for m in 0..16 {
                    // maddubs lane m pairs TWO weight bytes with TWO
                    // activation bytes: nib(q4[2m])*q8[2m] + nib(q4[2m+1])*q8[2m+1].
                    let b0 = q4[32 * j + 2 * m] as i32;
                    let b1 = q4[32 * j + 2 * m + 1] as i32;
                    let n0 = if half == 0 { b0 & 0xF } else { b0 >> 4 };
                    let n1 = if half == 0 { b1 & 0xF } else { b1 >> 4 };
                    let base = 64 * j + 32 * half;
                    let v0 = q8[base + 2 * m] as i8 as i32;
                    let v1 = q8[base + 2 * m + 1] as i8 as i32;
                    p16[m] = (n0 * v0 + n1 * v1) as i16;
                }
                let scale = sc[2 * j + half] as i32;
                for l in 0..8 {
                    sumi[l] += scale as i32 * p16[2 * l] as i32 + scale * p16[2 * l + 1] as i32;
                }
            }
        }
        for l in 0..8 {
            acc[l] = d.mul_add(sumi[l] as f32, acc[l]);
        }
    }

    // hsum_float_8(acc) in the C's exact order, then the acc_m fold.
    let s = [
        acc[4] + acc[0],
        acc[5] + acc[1],
        acc[6] + acc[2],
        acc[7] + acc[3],
    ];
    let u0 = s[0] + s[2];
    let u1 = s[1] + s[3];
    let total = u0 + u1;
    let t0 = acc_m[0] + acc_m[2];
    let t1 = acc_m[1] + acc_m[3];
    let m = t0 + t1;
    total + m
}
