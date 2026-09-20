//! qdot — the fused quantized row dot for the host CPU: Q3_K x Q8_K,
//! Q4_K x Q8_2_X4, Q6_K x Q8_2_X4 (MUL-31), Q5_0
//! x Q8_2_X4 (MUL-32), Q5_1 x Q8_2_X4 (MUL-34), and the Q8_0 x act cell
//! kernel of the q_nope2 absorption (MUL-38) at the file's end.
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
//!   * the Q6_K row dot (MUL-31, 2026-09-20) is ik's
//!     `mul_mat_qY_K_q8_2_X4_T<DequantizerQ6K_AVX2, 1>` — the qY variant:
//!     weight magnitude on the u8 side of maddubs, sign folded into the
//!     activation bytes (`prepare_signed` + `_mm256_sign_epi8`).
//!   * the Q5_0 row dot (MUL-32, 2026-09-20) is ik's
//!     `mul_mat_qX_1_q8_2_T<Q5_0_1_Unpacker, 1>` (iqk_gemm_legacy_quants.cpp:507,
//!     unpacker at 799, dispatched at 2494/2453 with expected_type_B =
//!     Q8_2_X4) — found by reading the dispatch, not the traits table, the
//!     MUL-27 lesson applied from the start. Both tables agree this time
//!     (ggml.c:767 says vec_dot_type = Q8_2_X4 under __AVX2__ +
//!     GGML_USE_IQK_MULMAT), so the activation encoder is the SAME
//!     `quantize_row_q8_2_x4` port the Q4_K round landed — no new coder.
//!     The weight side differs in kind: legacy 32-value blocks
//!     (block_q5_0), codes left UNSIGNED (0..31) with the -16 offset
//!     carried by the min path (-16 * d_w * d_a * raw-sum), which is what
//!     ScaleHelperQ_0_1<16> + ScaleHelperQ8_2 + AccumT<MinusType1> compute.
//!   * the Q5_1 row dot (MUL-34, 2026-09-20) is the SAME template with
//!     `Q5_1_Unpacker` (iqk_gemm_legacy_quants.cpp:804 —
//!     Q_Unpacker<block_q5_1, ScaleHelperQ_1, Q5_1_Dequantizer>, dispatched
//!     at 2343 with the same expected_type_B = Q8_2_X4 at 2329; the entry
//!     condition is ne00 % 32 == 0 at 2327). traits agree again
//!     (ggml.c:777, vec_dot_type = Q8_2_X4 under __AVX2__ + IQK_MULMAT).
//!     The one structural difference from Q5_0 is the scale pair: block_q5_1
//!     carries d f16 AND m f16 (24 bytes, ggml-common.h:210), so the min
//!     term is m_w * (d_a * m_a) — a SECOND stored scale, not -16*d.
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
/// `sizeof(block_q5_0)` (ggml-common.h:198): d f16 @0, qh u8[4] @2 (the
/// 5th bits), qs u8[16] @6 (low/high nibble pairs) — 22 bytes / 32 values.
const Q5F0_BLOCK: usize = 22;
/// `sizeof(block_q5_1)` (ggml-common.h:210): d f16 @0, m f16 @2 (its OWN
/// min scale — not -16*d as in Q5_0), qh u8[4] @4, qs u8[16] @8 — 24 bytes
/// / 32 values.
const Q5F1_BLOCK: usize = 24;
/// `sizeof(block_q8_2)`: d bf16 u16 @0, s raw i16 @2, qs s8[32] @4 — 36
/// bytes. ik's `quantize_row_q8_1_x4_T<block_q8_2, block_q8_2_x4>` stores
/// the blocks of a k % 32 (not % 128) column PAST the last whole x4 group
/// in this plain form (`if (i < nb4) ... else y[i]`, iqk_quantize.cpp:1120),
/// and the kernel's tail path indexes them the same way — `y + i` in
/// block_q8_2 units lands at 144*(nb/4) + 36*(i - nb4) exactly.
const Q82_BLOCK: usize = 36;

// ----------------------------------------------------------- public errors

/// Everything `dot_row` refuses to guess about. Length and alignment
/// problems are caller errors, not bugs — the same stance as
/// `gguf::QuantError` — because a partial super-block would compute quietly
/// wrong values, and this API rejects that class instead of panicking.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QdotError {
    /// The weight type has no fused kernel in this build (the model's five
    /// quant types Q3_K/Q4_K/Q5_0/Q5_1/Q6_K all have one).
    UnsupportedType(GgmlType),
    /// `k` is not a multiple of the weight type's block size; every kernel
    /// works on whole blocks (`gran` carries the modulus — 256 for the
    /// K-quants' super-blocks, 32 for Q5_0/Q5_1's legacy blocks).
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
/// is in this build's table AND the CPU has the ISA the kernel needs —
/// [`has_features`] owns that per-type table (Q3_K: avx2; Q4_K/Q6_K:
/// avx2+fma; Q5_0/Q5_1: avx2+fma+f16c), so it cannot drift against the
/// kernels' `#[target_feature]` sets.
///
/// The kernel's widest instructions are AVX2 (`vpmaddubsw`/`vpmaddwd` chains
/// — the box is Zen 3: AVX2/FMA3/F16C/BMI2, no AVX-512, no VNNI). Q3_K
/// needs avx2 alone: its f32 scaling is plain multiplies (so the scalar
/// mirror can match bit for bit) and its f16 read is an integer conversion.
/// The Q4_K and Q6_K kernels additionally fuse (`_mm256_fmadd_ps`, which the
/// mirrors reproduce with `f32::mul_add`), so FMA is part of their
/// requirement. Q5_0 (MUL-32) and Q5_1 (MUL-34) add F16C for their WEIGHT
/// scales (`_mm_cvtph_ps`, ik's own instruction there — measured 8.0 vs
/// 13.2 GB/s against the branchy scalar conversion); every AVX2 machine
/// carries F16C, `dot_row` detects it anyway, and the scalar mirror serves
/// the type if it is somehow absent.
pub fn supports(w: GgmlType) -> bool {
    // Q4_K is wired as of 2026-09-20 (MUL-27): its pairing is q8_2_x4 —
    // ik's dispatch is `mul_mat_qX_K_q8_2_X4_T<DequantizerQ4K_AVX2>`
    // (iqk_gemm_kquants.cpp:2751, 2768) with expected_type_B =
    // GGML_TYPE_Q8_2_X4. The q8_K pairing this crate landed first (commit
    // 34bd457) was the traits-table fallback, not the oracle's path: wiring
    // it moved kqv_out-0 by 6.4e-3 against the oracle (gate 5e-3) — an
    // activation-format delta, not a bug. That kernel is gone; its dump
    // (q4k-ik-dot.txt) stays in BLOOMERY_DATA as the round's record.
    //
    // Q5_0 is wired as of 2026-09-20 (MUL-32), same lesson applied from
    // the start: ik's dispatch is `mul_mat_qX_1_q8_2_T<Q5_0_1_Unpacker>`
    // (iqk_gemm_legacy_quants.cpp:2494 -> 507) with expected_type_B =
    // GGML_TYPE_Q8_2_X4 — and this time the traits table agrees
    // (ggml.c:767, vec_dot_type = Q8_2_X4 under __AVX2__ + IQK_MULMAT).
    // Q6_K is wired as of 2026-09-20 (MUL-31): same activation format, qY
    // template (iqk_gemm_kquants.cpp:2781).
    //
    // Q5_1 is wired as of 2026-09-20 (MUL-34), the model's LAST unfused
    // quant type: the same x86 dispatch section sends it to
    // `mul_mat_qX_1_q8_2_T<Q5_1_Unpacker>`
    // (iqk_gemm_legacy_quants.cpp:2343-2344 -> 2292-2293) under the same
    // expected_type_B = Q8_2_X4 (:2329) and the same ne00 % 32 == 0 entry
    // (:2327); traits agree (ggml.c:777). One tensor in the model:
    // blk.0.ffn_down, k = 10944 = 342 x 32 — 85 whole x4 groups + 2 tail
    // blocks, the first site whose tail blocks are REAL (Q5_0's k=1408 is
    // a multiple of 128).
    matches!(
        w,
        GgmlType::Q3_K | GgmlType::Q4_K | GgmlType::Q5_0 | GgmlType::Q5_1 | GgmlType::Q6_K
    ) && has_features(w)
}

/// The per-type ISA table, the single owner of what each fused kernel's
/// `#[target_feature]` set needs at runtime — Q3_K: avx2
/// (`dot_q3k_q8k_avx2`); Q4_K/Q6_K: avx2+fma (`dot_q4k_q82x4_avx2`,
/// `dot_q6k_q82x4_avx2`); Q5_0/Q5_1: avx2+fma+f16c (`dot_q5f0_q82x4_avx2`,
/// `dot_q5f1_q82x4_avx2`). [`supports`], `dot_row`'s dispatch and
/// `dot_row_avx2`'s assert all read this one fn: a safe pub fn reaching a
/// `target_feature` fn whose features were not detected is UB, and the table
/// is the one place that can drift against the attributes (2026-09-20
/// hardening round — `supports` used to admit Q4_K/Q6_K on avx2-without-fma,
/// which is exactly that UB).
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

/// The `k` contract of `w`'s fused path: `k` must be a multiple of the
/// weight format's block size. One number per type, owned here so the
/// engine's wiring condition and `col_bytes`/`check_row` cannot disagree:
///
/// | type | granularity | why |
/// |---|---|---|
/// | Q3_K, Q4_K, Q6_K | 256 | the K-quant super-block the dequant consumes |
/// | Q5_0, Q5_1 | 32 | the legacy block (block_q5_0 / block_q5_1) |
///
/// MUL-32 (2026-09-20): the model's every Q5_0 site is ffn_down_exps with
/// k = 1408 = 44 x 32 — NOT a multiple of 256 — so the single 256-value
/// contract the crate carried would never fire on its largest stage-table
/// site (23.1 ms/step, 32.5% of wall). This is the Q5_0 contract being
/// ADDED, not the other types' being relaxed: Q3_K/Q4_K still demand whole
/// 256-value super-blocks, and their kernels' shapes are untouched.
/// MUL-34 (2026-09-20) adds Q5_1 with the same 32-value legacy contract.
pub fn k_granularity(w: GgmlType) -> usize {
    match w {
        GgmlType::Q5_0 | GgmlType::Q5_1 => 32,
        _ => 256,
    }
}

/// Bytes one quantized activation column of `k` values occupies in `w`'s
/// format: Q3_K pairs q8_K (`k / 256` blocks of 296 bytes), Q4_K pairs
/// q8_2_x4 (`k / 128` groups of 144 bytes), Q5_0/Q5_1 pair q8_2_x4 with
/// ik's tail convention (`k / 128` groups of 144 bytes plus `(k % 128) / 32`
/// plain 36-byte blocks — see `Q82_BLOCK`).
///
/// Panics if `w` has no activation format in this build or `k` breaks the
/// type's [`k_granularity`] contract — both are caller constants at
/// allocation time, and a made-up size for a partial block would allocate
/// a buffer `quantize_col` cannot fill. (The x4 layout itself only needs
/// `k % 128`; Q4_K's 256 contract is the WEIGHT super-block's, and
/// `dot_row` enforces the same one.)
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
/// Q4_K, Q5_0 and Q5_1: port of the `__x86_64__` branch of ik's
/// `quantize_row_q8_2_x4` (iqk_quantize.cpp:1005) — bf16-rounded `d`,
/// `round_ties_even` quants, raw i16 sums. Q5_0/Q5_1's looser `k % 32`
/// contract additionally exercises the tail: blocks past the last whole x4
/// group are stored as plain 36-byte block_q8_2 (`y[i]` in the C,
/// iqk_quantize.cpp:1120) — the form ik's own kernel tail path indexes.
/// The gate checks the bytes bit for bit against ik's own coding of the
/// same column (dump `q5f1-ik-dot.txt`; Q5_1's k = 10944 leaves TWO tail
/// blocks, the first dump to exercise the tail at all).
pub fn quantize_col(w: GgmlType, x: &[f32], out: &mut [u8]) {
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
    match w {
        GgmlType::Q4_K | GgmlType::Q5_0 | GgmlType::Q5_1 | GgmlType::Q6_K => {
            assert_eq!(
                out.len(),
                col_bytes(w, x.len()),
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
/// Errors instead of guessing on shapes (`k` breaking the type's
/// [`k_granularity`] contract, an unsupported weight type, or a short
/// buffer is a [`QdotError`], not a panic) — a partial block would compute
/// quietly wrong values, and this API refuses that. The dispatch is AVX2
/// when the CPU has it ([`supports`]), the scalar mirror otherwise; the
/// two agree bit for bit (gate 4).
pub fn dot_row(w: GgmlType, wrow: &[u8], acol: &[u8], k: usize) -> Result<f32, QdotError> {
    let nb = check_row(w, wrow.len(), acol.len(), k)?;
    // The single-owner ISA table: Q3_K avx2; Q4_K/Q6_K +fma (`_mm256_fmadd_ps`);
    // Q5_0/Q5_1 +f16c (the `_mm_cvtph_ps` scale gather, see the kernels'
    // Safety notes). Whatever the table rejects takes the scalar mirror.
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
        GgmlType::Q6_K => dot_q6k_q82x4_emul(wrow, acol, nb),
        GgmlType::Q5_0 => dot_q5f0_q82x4_emul(wrow, acol, nb),
        GgmlType::Q5_1 => dot_q5f1_q82x4_emul(wrow, acol, nb),
        _ => dot_q3k_q8k_scalar(wrow, acol, nb),
    }
}

/// The AVX2 kernel behind [`dot_row`], for the gate's path comparison
/// (gate 4). Panics when the CPU lacks the type's kernel ISA — [`has_features`]'s
/// table: avx2 for Q3_K, +fma for Q4_K/Q6_K, +f16c for Q5_0/Q5_1 — on such a
/// machine there is nothing to compare against and the caller (a gate) wants
/// the loud failure, not a quiet fall back.
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

/// Shape validation shared by every `dot_row` entry point. Returns the
/// weight BLOCK count on success — in `w`'s own units (super-blocks for the
/// K-quants, 32-value blocks for Q5_0); each kernel interprets its `nb`
/// that way. The supported-type check comes first so an unsupported type
/// never touches the data.
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
    // The activation byte length is `col_bytes`'s, not a second arithmetic:
    // per type the hand form was Q3_K (k/256)·296, Q4_K/Q6_K (k/256)·288 =
    // (k/128)·144, Q5_0/Q5_1 the same tail expression `col_bytes` carries —
    // equal for every type, so the one owner (col_bytes) is the only copy
    // (verified equal per type in the 2026-09-20 hardening round).
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
/// Tail blocks (MUL-32): when k is a multiple of 32 but not 128, the C's
/// `if (i < nb4) ... else y[i]` stores the leftover blocks PAST the last
/// whole group as plain 36-byte block_q8_2 — d @0, raw i16 sum @2, qs @4 —
/// at byte offset 144*(k/128) + 36*(i - nb4). Same arithmetic, different
/// container; ik's kernel tail path indexes them exactly there.
///
/// Assumes finite activations (the oracle dumps are): the C's SIMD max/NaN
/// lanes and a NaN's f32->i8 conversion have no scalar meaning worth
/// mirroring, and a NaN here would fail the byte-identity gate loudly
/// anyway.
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
                let q = (v * id).round_ties_even() as i32 as i8;
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
            let q = (v * id).round_ties_even() as i32 as i8;
            qs[m] = q as u8;
            isum += q as i32;
        }
        tb[2..4].copy_from_slice(&(isum as i16).to_le_bytes());
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
///
/// `#[inline(always)]` to match the crate's other intrinsic helpers
/// (`hsum_i32`, `field_dot`, `q5x_codes`): this was already inlined in
/// today's release binary (`nm` verified), and the attribute makes that a
/// guarantee instead of an optimizer decision (2026-09-20 hardening round).
///
/// # Safety
/// AVX must be available (and SSE3 for `_mm_movehdup_ps`); every caller is
/// the body of a `#[target_feature(enable = "avx2", …)]` kernel, which
/// covers both.
#[inline(always)]
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

// --------------------------------------------------------- Q6_K x Q8_2_X4
// MUL-31. Port of ik's `mul_mat_qY_K_q8_2_X4_T<DequantizerQ6K_AVX2, 1>`
// (iqk_gemm_kquants.cpp:938) — the pairing the oracle dispatches
// (iqk_gemm_kquants.cpp:2753/2781), non-HAVE_FANCY_SIMD branch. The qY
// template is NOT Q4_K's qX with a type swapped; two structural
// differences, both ported verbatim:
//
//   * the weight never becomes a signed maddubs operand. `prepare_signed`
//     (iqk_gemm_kquants.cpp:871) turns the 6-bit codes into i8 -32..31,
//     then `us[k] = sign_epi8(values[k], values[k])` keeps only their
//     MAGNITUDE as the u8 side of maddubs, while the SIGN is folded into
//     the ACTIVATION bytes: `sign_epi8(qs, values[k])` negates the q8 lane
//     where the weight is negative and zeroes it where the weight is zero.
//     (us <= 32 and |q| <= 127, so maddubs' i16 saturation stays
//     unreachable at |32*127*2| = 8128.)
//   * the 16 s8 sub-block scales go through make_scales' sign-extension
//     and then k_shuff, a shuffle_epi8 that reorders the eight i16 lanes
//     per 128-bit half to {0,2,4,6,1,3,5,7} — the lane order the i16 pack
//     chain's sums land in (the same reason the Q4_K round refused to
//     hand-derive lane maps; here the shuffle IS the map, ported as bytes).
//
// There is no min term: block_q6_K has no dmin, so accd accumulates only
// the two per-chunk fmadds. The activation side is the same
// `quantize_q82x4_col` coding Q4_K consumes (one encoder, gate 0 shared).
//
// `#[target_feature]` is not optional (MUL-26 lesson, 21x on this box);
// the MUL-27 measurement adds that helper SPLITS lose ~13% on this loop
// shape, so the kernel is one monolithic fn.

/// `block_q6_K` (ggml-common.h:386): ql u8[128] @0, qh u8[64] @128,
/// scales s8[16] @192, d f16 @208 — 256 values, 210 bytes. Weight element
/// 128c+l reads ql byte 64c+l field 0/2 (low nibble) or 64c+32+l field 1/3
/// (high nibble), with two more bits from qh byte 32c+l — five-bit codes
/// split across two arrays.
const Q6K_BLOCK: usize = 210;

/// `mul_mat_qY_K_q8_2_X4_T`'s k_shuff (iqk_gemm_kquants.cpp:949), verbatim:
/// within each 16-byte half, output byte pairs (2p, 2p+1) take input scale
/// {0,2,4,6,1,3,5,7}[p] — the {b0 b2 b4 b6 b1 b3 b5 b7} order the pack
/// chain's i32 lanes come out in (the template's own lane comment).
static K_SHUFFLE_Q6K: [u8; 32] = [
    0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15, //
    0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15,
];

/// Port of `mul_mat_qY_K_q8_2_X4_T<DequantizerQ6K_AVX2, 1>`
/// (iqk_gemm_kquants.cpp:938) reduced to `matmul_q`'s shape: one weight row
/// against one q8_2_x4 activation column. The template's iy loop is gone
/// (nrc_y = 1), its ix loop is the caller's row loop. Scheduling follows
/// MUL-27's surviving form: both chunks' integer chains are issued before
/// the super-block's two fmadds, and the fmadds stay strictly ordered
/// (chunk 0, chunk 1; super-blocks ascending) — the emulator and the
/// 1-ULP-ik gate hold the bits.
///
/// # Safety
/// The CPU must support AVX2+FMA, and the caller must have validated
/// lengths: `wrow` at nb*210 readable bytes, `acol` at nb*288 (`check_row`
/// does).
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_q6k_q82x4_avx2(wrow: &[u8], acol: &[u8], nb: usize) -> f32 {
    unsafe {
        let ml = _mm256_set1_epi8(0xF);
        let mh = _mm256_set1_epi8(0x30);
        let m32n = _mm256_set1_epi8(-32);
        // SAFETY: 32-byte load inside the 32-byte static table.
        let shuff = _mm256_loadu_si256(K_SHUFFLE_Q6K.as_ptr() as *const __m256i);
        let mut accd = _mm256_setzero_ps();

        for i in 0..nb {
            let wb = wrow.as_ptr().add(Q6K_BLOCK * i);

            // Super-block scale d: f16 @ +208.
            // SAFETY: one unaligned u16 read inside the super-block.
            let d = f16_bits_to_f32((wb.add(208) as *const u16).read_unaligned());
            let vd = _mm256_set1_ps(d);
            // make_scales + k_shuff: 16 s8 scales sign-extended to i16,
            // then the {0,2,4,6,1,3,5,7} interleave; each 128-bit half of
            // the shuffled register becomes one f32 scale vector.
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

            // The two x4 groups feeding super-block i: their bf16 scales as
            // f32 (bits << 16). dy's halves ARE the group scales — lanes
            // 0..3 group 2i, 4..7 group 2i+1 — broadcast onto the matching
            // scale vector (the C's nrc_y == 1 branch).
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

            // Both chunks' integer chains first, then the two fmadds in the
            // C's order. Same bits.
            let mut sumis = [_mm256_setzero_si256(), _mm256_setzero_si256()];
            for j in 0..2 {
                // DequantizerQ6K_AVX2::prepare: ql bytes [64j, 64j+64) and
                // qh bytes [32j, 32j+32) -> four 32-value planes of 5-bit
                // codes (low nibble of ql / two bits of qh).
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
                // prepare_signed: codes - 32 -> i8 weights; us = |weight|,
                // the u8 side of maddubs.
                let mut us = [_mm256_setzero_si256(); 4];
                for k in 0..4 {
                    values[k] = _mm256_add_epi8(values[k], m32n);
                    us[k] = _mm256_sign_epi8(values[k], values[k]);
                }

                // The x4 group for chunk j: its four 32-byte quant slices,
                // each with the weight lane's sign folded in — qY's
                // signature move.
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

// The instruction-graph emulator of [`dot_q6k_q82x4_avx2`] — same regime as
// the Q4_K one: safe Rust, no ISA requirement, every operation in Intel's
// exact semantics, the same loop structure. Two families of instructions
// the Q4_K graph did not have are emulated here: the i16 shifts of the
// code-assembly step (as real 16-bit lane ops on byte arrays, no per-byte
// shortcut derived) and `_mm256_sign_epi8`.

/// `_mm256_sign_epi8(a, b)`: lane = b < 0 ? -a : (b == 0 ? 0 : a), the i8
/// negation wrapping at -128 (unreachable for the magnitudes here, written
/// anyway so the emulator stays honest).
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

/// `_mm256_slli_epi16` on a byte array: 16-bit lane left shift, wrapping
/// (bits leaving a lane's top are gone; zeros enter the bottom).
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

/// `_mm256_srli_epi16` on a byte array: 16-bit lane LOGICAL right shift.
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

/// The emulator: the kernel's graph, one weight row against one q8_2_x4
/// column, bit-identical to [`dot_q6k_q82x4_avx2`] by construction.
fn dot_q6k_q82x4_emul(wrow: &[u8], acol: &[u8], nb: usize) -> f32 {
    let mut accd = [0.0f32; 8];

    for i in 0..nb {
        let wb = &wrow[i * Q6K_BLOCK..(i + 1) * Q6K_BLOCK];
        let g0 = &acol[(2 * i) * Q82X4_STRIDE..(2 * i + 1) * Q82X4_STRIDE];
        let g1 = &acol[(2 * i + 1) * Q82X4_STRIDE..(2 * i + 2) * Q82X4_STRIDE];

        let d = f16_bits_to_f32(u16::from_le_bytes([wb[208], wb[209]]));

        // make_scales: 16 s8 -> i16 into a little-endian 32-byte register,
        // then shuffle_epi8 with k_shuff (per 128-bit lane, table byte's
        // high bit would zero — K_SHUFFLE_Q6K entries are all < 16).
        let mut reg = [0u8; 32];
        for l in 0..16 {
            reg[2 * l..2 * l + 2].copy_from_slice(&(wb[192 + l] as i8 as i16).to_le_bytes());
        }
        let mut sc16 = [0u8; 32];
        for q in 0..32 {
            sc16[q] = reg[16 * (q / 16) + K_SHUFFLE_Q6K[q] as usize];
        }
        // scales[j][l] = (d * sc16_lane) * dy_lane, two multiplies in the
        // kernel's order; lanes 0..7 of the low half then the high half.
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
            // prepare: the four code planes, as the kernel builds them.
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
            // prepare_signed: codes - 32 (wrapping i8), us = |weight|.
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
            // The sign fold: qs bytes take the weight lane's sign.
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

            // One fused accumulate per chunk, in the kernel's order.
            for l in 0..8 {
                accd[l] = scales[j][l].mul_add(sumi[l] as f32, accd[l]);
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

// --------------------------------------------------------- Q5_0 x Q8_2_X4
// MUL-32 (2026-09-20). Port of ik's `mul_mat_qX_1_q8_2_T<Q5_0_1_Unpacker,
// 1>` (iqk_gemm_legacy_quants.cpp:507; unpacker 799, dispatch 2494/2453,
// expected_type_B = Q8_2_X4 at 2482) — the pairing the oracle dispatches at
// decode (Ny = 1; the nrc_y >= 32 repack path `is_dequant_better` selects,
// Q8_0_R8 via iqk_convert_qX_q80_r8, is a prefill shape this engine's
// decode never runs). The activation side is the SAME q8_2_x4 coding the
// Q4_K round ported (`quantize_q82x4_col`, now with the tail blocks);
// nothing new was written there, and gate 0 re-checks the bytes against
// ik's own coder (`tools/ref/q5f0_ref.cpp` dumps both).
//
// The arithmetic shape differs from Q4_K's kernel in kind:
//   * the weight codes stay UNSIGNED 0..31 (Q5_1_Dequantizer<block_q5_0>:
//     nibble | (qh bit << 4), iqk_gemm_legacy_quants.cpp:676) — maddubs
//     takes them as the u8 operand directly, no sign tricks;
//   * the -16 offset of the q5_0 decode rides the MIN path: per block,
//     sum((code-16)*qa) = sum(code*qa) - 16*sum(qa), and sum(qa) is the
//     raw i16 the encoder stored. ScaleHelperQ_0_1<16>::prepare4 makes the
//     weight-side pair (d_w, -16.0*d_w); ScaleHelperQ8_2 + convert_scales
//     make the activation-side (d_a, d_a*m_a); AccumT<MinusType1>:
//     accm[l] += (-16*d_w[l]) * (d_a[l]*m_a[l]) per group, dall = d_w*d_a
//     broadcast, acc = fmadd(dall, i32->f32(pall), acc);
//   * the integer combine is an epi32 unpack chain on the madd outputs
//     (Sum4<..., UnsignedDot, can_pack=false>, iqk_gemm_legacy_quants.cpp:56)
//     — i32 lanes, not the i16 chain the Q4_K x4 kernel uses;
//   * the final reduction is MinusType1::result: sum = lo128(acc)+hi128(acc),
//     then hsum_float_4(sum + accm) (iqk_common.h:225) — the reduction
//     ORDER is load-bearing for the mirror's bit identity, as always.
//
// `#[target_feature]` is not optional (the MUL-26 lesson): without it LLVM
// legalizes the 256-bit intrinsic bodies for the SSE2 baseline. FMA is
// required (one fmadd per group/tail block); the mirror reproduces it with
// `f32::mul_add`. F16C is required for the weight scales (`_mm_cvtph_ps`).
//
// Scheduling measurements (2026-09-20, same round, box, single core
// taskset -c 0, 360,448 rows x 968 B, qdot-rate vs ik's own kernel via
// tools/ref/q5f0_rate.cpp, both runs within 0.1 GB/s):
//
// | form | GB/s |
// |---|---|
// | scalar `f16_bits_to_f32` x4 per group (branchy subnormal/NaN paths) | 8.0 |
// | + named locals for qx/dw (no [__m256i; 4] indexing) | 8.0 |
// | + one `_mm_cvtph_ps` for the four scales (this form) | 13.2 |
// | ik's own kernel (gcc -O2 -mavx2, same shape) | 14.5 |
//
// The scale conversion was the whole first gap — 44 branchy scalar
// conversions per 968-byte row is op pressure the hardware instruction
// removes in one. The residual 13.2 vs 14.5 is codegen (same intrinsics,
// same order), the same regime as the Q4_K round's 14.1 vs 16.3-17.1, and
// the engine's DRAM-bound regime (4.6 GB/s/thread) sits three times below
// either number.

/// `HBitDequantizer`'s three constants (iqk_gemm_legacy_quants.cpp:660):
/// the shuffle broadcasts qh byte j/8 to byte j of 32, the mask clears
/// exactly bit j%8 of byte j, and minus1 is the all-0xFF cmpeq reference —
/// so `cmpeq((shuffle | mask), 0xFF)` tests qh bit j, weight value j's 5th
/// bit. Shared by the Q5_0 and Q5_1 kernels (ik's HBitDequantizer is
/// shared by both dequantizers); built once per row (the C constructs its
/// unpacker's dequantizer once per call, constants included).
struct Q5HBit {
    shuffle: __m256i,
    mask: __m256i,
    minus1: __m256i,
}

/// One 32-value weight block's codes as UNSIGNED bytes 0..31:
/// `Dequantizer4bit::dequant(qs)` (nibble split — low nibbles are values
/// 0..15, high nibbles 16..31) OR'd with `Q5_1_Dequantizer`'s high bits
/// (0x10 where the qh bit is set). QH_OFF/QS_OFF are the block's qh/qs
/// offsets: 2/6 for block_q5_0, 4/8 for block_q5_1.
///
/// # Safety
/// `blk` must hold QS_OFF + 16 readable bytes (the caller slices a
/// validated row).
#[inline(always)]
unsafe fn q5x_codes<const QH_OFF: usize, const QS_OFF: usize>(
    blk: &[u8],
    m4: __m256i,
    mh: __m256i,
    hb: &Q5HBit,
) -> __m256i {
    unsafe {
        // SAFETY: 16 readable bytes inside the block.
        let aux128 = _mm_loadu_si128(blk.as_ptr().add(QS_OFF) as *const __m128i);
        let nib = _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16::<4>(aux128), aux128), m4);
        // qh u32 -> 32 one-bit bytes.
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

/// Port of `mul_mat_qX_1_q8_2_T<Q5_0_1_Unpacker, 1>` reduced to
/// `matmul_q`'s shape: one weight row against one q8_2_x4 activation
/// column, nb = the 32-value weight block count. The template's nrc_x loop
/// is the caller's row loop and nrc_y = 1; everything else is the C's
/// arithmetic, non-FANCY branch, both the nb%4==0 group path and the tail
/// path (`AccumT<MinusType1, 1, is_multiple_of_4>`, the 0.25-spread min
/// correction included).
///
/// TWIN of [`dot_q5f1_q82x4_avx2`] — the same template with four
/// differences only: block width (22 vs 24), the `q5x_codes` offsets (qh
/// @2/qs @6 vs @4/@8), the scale gather (four d's through one
/// `_mm_cvtph_ps` vs the (d,m) pair shuffle), and the tail min correction
/// (-16·d_w vs the stored m_w). Every edit outside those four must be made
/// in both. Do NOT merge them — measured: helper splits cost 10-13% on
/// these loops (MUL-27).
///
/// # Safety
/// The CPU must support AVX2+FMA+F16C, and the caller must have validated
/// lengths: `wrow` at nb*22 readable bytes, `acol` at
/// (k/128)*144 + ((k%128)/32)*36 readable bytes (`check_row` does).
/// F16C is this kernel's one addition to the crate's ISA surface: the
/// weight scales convert through `_mm_cvtph_ps` (ik's own instruction
/// there), and every AVX2 machine carries F16C (Ivy Bridge predates
/// Haswell's AVX2) — `dot_row` still detects it rather than assuming.
#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
unsafe fn dot_q5f0_q82x4_avx2(wrow: &[u8], acol: &[u8], nb: usize) -> f32 {
    // SAFETY: AVX2+FMA present (checked by dot_row) and both slices hold
    // the validated lengths — nb*22 weight bytes and the matching activation
    // column — per the fn contract; the pointer arithmetic below stays
    // inside them, with the reasoning repeated at each load.
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
            // Weight side: four blocks' codes plus their f16 scales
            // gathered into one __m128 (the C's scales8[4] + cvtph_ps; the
            // f16->f32 conversion is exact, so building it lane-wise from
            // the same values is the same register). Named locals, NOT an
            // array: an indexed [__m256i; 4] lives in memory and the
            // register file walks the stack for every use — the MUL-27
            // lesson. The scale gather is ONE `_mm_cvtph_ps`, exactly the
            // C's scales8[4] + cvtph_ps: measured 8.0 vs ik's 14.4 GB/s
            // when the four conversions went through the branchy scalar
            // `f16_bits_to_f32` instead (its subnormal loop and NaN branch
            // are per-lane work the hardware instruction does in one op).
            let b0 = (4 * i) * Q5F0_BLOCK;
            let b1 = (4 * i + 1) * Q5F0_BLOCK;
            let b2 = (4 * i + 2) * Q5F0_BLOCK;
            let b3 = (4 * i + 3) * Q5F0_BLOCK;
            let qx0 = q5x_codes::<2, 6>(&wrow[b0..b0 + Q5F0_BLOCK], m4, mh, &hb);
            let qx1 = q5x_codes::<2, 6>(&wrow[b1..b1 + Q5F0_BLOCK], m4, mh, &hb);
            let qx2 = q5x_codes::<2, 6>(&wrow[b2..b2 + Q5F0_BLOCK], m4, mh, &hb);
            let qx3 = q5x_codes::<2, 6>(&wrow[b3..b3 + Q5F0_BLOCK], m4, mh, &hb);
            // SAFETY: 8 readable bytes inside the validated row: the four
            // blocks' f16 d values, gathered exactly as the C gathers them.
            let mut scales8 = [0u8; 8];
            scales8[0..2].copy_from_slice(&wrow[b0..b0 + 2]);
            scales8[2..4].copy_from_slice(&wrow[b1..b1 + 2]);
            scales8[4..6].copy_from_slice(&wrow[b2..b2 + 2]);
            scales8[6..8].copy_from_slice(&wrow[b3..b3 + 2]);
            let s4 = _mm_cvtph_ps(_mm_loadl_epi64(scales8.as_ptr() as *const __m128i));
            // other = (lo: d_w, hi: -16.0 * d_w) — ScaleHelperQ_0_1<16>.
            let other = _mm256_set_m128(_mm_mul_ps(s4, min16), s4);

            // Activation group i: convert_scales (iqk_legacy:132) —
            // bf16 scales as bits<<16, i16 sums sign-extended to f32.
            // SAFETY: 8 readable bytes at the head of the validated group.
            let g = acol.as_ptr().add(i * Q82X4_STRIDE);
            let aux_d = _mm_castsi128_ps(_mm_slli_epi32::<16>(_mm_cvtepu16_epi32(
                _mm_loadl_epi64(g as *const __m128i),
            )));
            // SAFETY: 8 readable bytes at offset 8 of the group.
            let aux_m = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_loadl_epi64(
                g.add(8) as *const __m128i
            )));
            // prep = (lo: d_a, hi: d_a * m_a).
            let prep = _mm256_set_m128(_mm_mul_ps(aux_d, aux_m), aux_d);
            // s12 = other * prep — lo: d_w*d_a, hi: (-16*d_w)*(d_a*m_a).
            let s12 = _mm256_mul_ps(other, prep);
            // MinusType1: accm += hi; dall = lo broadcast to both halves.
            accm = _mm_add_ps(accm, _mm256_extractf128_ps(s12, 1));
            let lo = _mm256_castps256_ps128(s12);
            let dall = _mm256_set_m128(lo, lo);

            // Sum4<UnsignedDot, can_pack=false>: four maddubs->madd, then
            // the epi32 unpack chain. The four integer chains issue first
            // (independent work), the group's one fmadd last — the C's
            // order, kept for the mirror's bit identity.
            // SAFETY: 32 readable bytes at each offset inside the group
            // (144-byte groups validated by check_row).
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

        // Tail blocks (nb % 4 != 0): AccumT's prepare1 path. The min
        // correction is spread as 0.25 per lane (MinusType1's scalar
        // compute), so the final horizontal sum over 4 lanes recovers the
        // whole amount — replicate the exact arithmetic, not a tidier one.
        let nb4 = 4 * nbg;
        for i in nb4..nb {
            let b = i * Q5F0_BLOCK;
            let wb = &wrow[b..b + Q5F0_BLOCK];
            let dw = f16_bits_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
            let qx0 = q5x_codes::<2, 6>(wb, m4, mh, &hb);
            let tb = nb4 / 4 * Q82X4_STRIDE + (i - nb4) * Q82_BLOCK;
            let da = bf16_bits_to_f32(u16::from_le_bytes([acol[tb], acol[tb + 1]]));
            let ma = i16::from_le_bytes([acol[tb + 2], acol[tb + 3]]) as f32;
            // s12 = (dw*da, (-16.0*dw) * (da*ma)); MinusType1 keeps
            // dm.second*0.25f per lane and returns dm.first.
            let d = dw * da;
            let corr = (-16.0f32 * dw) * (da * ma) * 0.25f32;
            accm = _mm_add_ps(accm, _mm_set1_ps(corr));
            // SAFETY: 32 readable bytes at offset 4 of the tail block.
            let qs = _mm256_loadu_si256(acol.as_ptr().add(tb + 4) as *const __m256i);
            let p0 = _mm256_madd_epi16(m1, _mm256_maddubs_epi16(qx0, qs));
            acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(p0), acc);
        }

        // MinusType1::result: sum = lo128(acc)+hi128(acc), hsum_float_4 of
        // sum + accm (iqk_common.h:225 — movehl then movehdup).
        let sum = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
        let x = _mm_add_ps(sum, accm);
        let x = _mm_add_ps(x, _mm_movehl_ps(x, x));
        _mm_cvtss_f32(_mm_add_ss(x, _mm_movehdup_ps(x)))
    }
}

// The instruction-graph emulator of [`dot_q5f0_q82x4_avx2`] — safe Rust, no
// ISA requirement, every operation in Intel's exact semantics, same regime
// as the Q4_K round: the mirror matches the kernel bit for bit by
// construction, and the kernel must agree with IK'S OWN Q5_0 kernel to
// within 1 ULP on real rows (gate B, dump `q5f0-ik-dot.txt`). The epi32
// unpack chain here is on I32 lanes — distinct from the i16-lane emulators
// the Q4_K round wrote for its pack chain — so it gets its own helpers.

/// `_mm256_unpacklo_epi32` on i32 lanes: per 128-bit half, interleave the
/// low two dwords of a and b.
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

/// `_mm256_unpackhi_epi32` on i32 lanes: per half, the high two dwords.
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

/// `_mm256_unpacklo_epi64` on i32 lanes: per half, low qword of a then b.
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

/// `_mm256_unpackhi_epi64` on i32 lanes: per half, high qword of a then b.
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

/// The scalar decode of one 32-value block's codes — the HBit graph's net
/// effect (the broadcast/mask/cmpeq dance IS a bit test of qh bit v) plus
/// the nibble split, byte v = value v's code 0..31. Offsets as
/// [`q5x_codes`]: 2/6 for block_q5_0, 4/8 for block_q5_1.
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
        // Values 0..15 read the low nibble, 16..31 the high nibble of qs
        // byte v%16 — the Dequantizer4bit split.
        *c = if v < 16 {
            nibbles[v] & 0xF
        } else {
            nibbles[v - 16] >> 4
        };
        // The 5th bit: qh bit v (the HBit broadcast's net effect).
        if (qh >> v) & 1 != 0 {
            *c |= 0x10;
        }
    }
    code
}

/// The emulator: the kernel's graph, one weight row against one q8_2_x4
/// column, bit-identical to [`dot_q5f0_q82x4_avx2`] by construction — same
/// operations, same order, `mul_add` where the kernel fuses.
fn dot_q5f0_q82x4_emul(wrow: &[u8], acol: &[u8], nb: usize) -> f32 {
    let mut acc = [0.0f32; 8];
    let mut accm = [0.0f32; 4];

    let nbg = nb / 4;
    for i in 0..nbg {
        // qx[j]: codes of weight block 4i+j; dw[j]: its f16 scale.
        let mut qx = [[0u8; 32]; 4];
        let mut dw = [0.0f32; 4];
        for j in 0..4 {
            let wb = &wrow[(4 * i + j) * Q5F0_BLOCK..(4 * i + j + 1) * Q5F0_BLOCK];
            qx[j] = q5x_codes_scalar::<2, 6>(wb);
            dw[j] = f16_bits_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
        }
        let g = &acol[i * Q82X4_STRIDE..(i + 1) * Q82X4_STRIDE];
        // convert_scales: da[l] bf16->f32 (bits<<16), ma[l] i16->f32.
        let mut da = [0.0f32; 4];
        let mut ma = [0.0f32; 4];
        for l in 0..4 {
            da[l] = bf16_bits_to_f32(u16::from_le_bytes([g[2 * l], g[2 * l + 1]]));
            ma[l] = i16::from_le_bytes([g[8 + 2 * l], g[9 + 2 * l]]) as f32;
        }
        // s12: lo[j] = dw[j]*da[j], hi[j] = (-16*dw[j])*(da[j]*ma[j]).
        // MinusType1: accm += hi; dall lanes l and 4+l use lo[l%4].
        let mut lo = [0.0f32; 4];
        for l in 0..4 {
            lo[l] = dw[l] * da[l];
            accm[l] += (-16.0f32 * dw[l]) * (da[l] * ma[l]);
        }
        // Sum4: p_j = emul_madd1(emul_maddubs(qx[j], qs_j)); the epi32
        // unpack chain; pall lanes land per (block, byte-half).
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
        // acc = fmadd(dall, pall, acc) over all 8 lanes.
        for l in 0..8 {
            acc[l] = lo[l % 4].mul_add(pall[l] as f32, acc[l]);
        }
    }

    // Tail blocks — the 0.25-spread min correction, 4 lanes each.
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

    // MinusType1::result + hsum_float_4: sum[l] = acc[l]+acc[4+l];
    // x[l] = sum[l]+accm[l]; (x0 + x2) + (x1 + x3).
    let mut x = [0.0f32; 4];
    for l in 0..4 {
        x[l] = (acc[l] + acc[4 + l]) + accm[l];
    }
    (x[0] + x[2]) + (x[1] + x[3])
}

// --------------------------------------------------------- Q5_1 x Q8_2_X4
// MUL-34 (2026-09-20). Port of `mul_mat_qX_1_q8_2_T<Q5_1_Unpacker, 1>` —
// the SAME template the Q5_0 round ported, with a different unpacker
// (iqk_gemm_legacy_quants.cpp:804: Q_Unpacker<block_q5_1, ScaleHelperQ_1,
// Q5_1_Dequantizer<block_q5_1>>, Sum4T = Sum4TypeQ82:369-370). Dispatch:
// MulMat::prepare (iqk_mul_mat.cpp:927) -> iqk_set_kernels_legacy_quants
// (:2325, entry ne00%32==0 at :2327) -> expected_typeB = Q8_2_X4 (:2329)
// -> case Q5_1 (:2343-2344) -> set_functions (:2292-2293). The traits
// table agrees (ggml.c:777, vec_dot_type = Q8_2_X4 under __AVX2__ +
// IQK_MULMAT), so the activation side is the same `quantize_q82x4_col`
// coding again — no new encoder, gate 0 shared.
//
// What differs from Q5_0 is exactly one thing: the scale pair. block_q5_1
// (24 bytes: d f16 @0, m f16 @2, qh @4, qs @8) stores its OWN minimum
// scale m, where block_q5_0 folded the -16 offset into -16*d:
//   * ScaleHelperQ_1::prepare4 (:238) gathers the four (d,m) f16 pairs of
//     a group — one u32 read per block grabs both — and shuffles them into
//     [d0 d1 d2 d3 m0 m1 m2 m3] before a single `_mm256_cvtph_ps`
//     (other = lo: d_w, hi: m_w);
//   * s12 = other * prep (ScaleHelperQ8_2/convert_scales:132): lo =
//     d_w*d_a (dall), hi = m_w*(d_a*m_a) (accm += hi per group,
//     MinusType1:275);
//   * the tail's prepare1 path spreads the min term as 0.25 per lane:
//     corr = m_w*(d_a*m_a)*0.25;
//   * qh/qs sit 2 bytes later in the block than Q5_0's (qh @4, qs @8) —
//     the one change `q5x_codes`'s offset parameters carry.
// Everything else — the unsigned 0..31 codes as maddubs' u8 operand, the
// epi32 unpack chain (Sum4<UnsignedDot, can_pack=false>, :56), the
// fmadd per group, MinusType1::result's reduction — is the Q5_0 port's
// arithmetic, byte for byte.
//
// The model's single Q5_1 site (blk.0.ffn_down, k = 10944 = 342 blocks)
// leaves TWO tail blocks past the 85 whole x4 groups: this is the first
// kernel whose tail path is LIVE in the engine, both in the activation
// column (two 36-byte block_q8_2) and in the weight loop.
//
// `#[target_feature]` is not optional (the MUL-26 lesson); FMA is required
// (one fmadd per group/tail block) and F16C for the scale conversion
// (`_mm256_cvtph_ps`, ONE per group where Q5_0 needed one per group too —
// this one converts eight f16s: four d, four m). The mirror reproduces the
// fmadd with `f32::mul_add`.
//
// Rate measurements (2026-09-20, same round, box, single core taskset -c 0,
// 360,448 rows x 8208 B = the ffn_down shape, qdot-rate vs ik's own kernel
// via tools/ref/q5f1_rate.cpp, two runs each within 0.3 GB/s):
//
// | form | GB/s |
// |---|---|
// | this port (scale gather as ONE shuffle + ONE cvtph_ps from the start) | 15.0–15.1 |
// | ik's own kernel (gcc -O2 -mavx2, same shape) | 15.2–15.5 |
//
// 98–99% of ik — the best ratio of the x4 rounds (Q4_K 14.1 vs 16.3–17.1,
// Q5_0 13.2 vs 14.5, both this box). No scheduling experiment was needed:
// MUL-32's lesson (the scale conversion is where the first 40% went) was
// applied at write time — ScaleHelperQ_1's gather is one shuffle_epi8 +
// one _mm256_cvtph_ps per group in ik itself, and this port copies that
// shape directly, so there was no branchy first form to fix. The residual
// gap is the same codegen regime as the other rounds' tails.

/// Port of `mul_mat_qX_1_q8_2_T<Q5_1_Unpacker, 1>` reduced to
/// `matmul_q`'s shape: one weight row against one q8_2_x4 activation
/// column, nb = the 32-value weight block count. Group path (nb/4 whole
/// groups) and tail path (the remainder blocks) both live — the model's
/// only Q5_1 site exercises the tail (k = 10944: 85 groups + 2 blocks).
///
/// TWIN of [`dot_q5f0_q82x4_avx2`] — the same template with four
/// differences only: block width (24 vs 22), the `q5x_codes` offsets (qh
/// @4/qs @8 vs @2/@6), the scale gather (the (d,m) pair shuffle through one
/// `_mm256_cvtph_ps` vs four d's), and the tail min correction (the stored
/// m_w vs -16·d_w). Every edit outside those four must be made in both. Do
/// NOT merge them — measured: helper splits cost 10-13% on these loops
/// (MUL-27).
///
/// # Safety
/// The CPU must support AVX2+FMA+F16C, and the caller must have validated
/// lengths: `wrow` at nb*24 readable bytes, `acol` at
/// (k/128)*144 + ((k%128)/32)*36 readable bytes (`check_row` does). F16C
/// is the same one-instruction scale conversion Q5_0's kernel needs —
/// every AVX2 machine carries it, `dot_row` still detects it.
#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
unsafe fn dot_q5f1_q82x4_avx2(wrow: &[u8], acol: &[u8], nb: usize) -> f32 {
    // SAFETY: AVX2+FMA+F16C present (checked by dot_row) and both slices
    // hold the validated lengths — nb*24 weight bytes and the matching
    // activation column — per the fn contract; the pointer arithmetic
    // below stays inside them, with the reasoning repeated at each load.
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
        // ScaleHelperQ_1's shuffle (:242): the gathered u32[4] is four
        // (d,m) f16 pairs; this permute rewrites it as [d0 d1 d2 d3 m0 m1
        // m2 m3] so one cvtph_ps builds both scale halves. _mm_set_epi16
        // args run high word first — the C's constant, verbatim.
        let dm_shuf = _mm_set_epi16(
            0x0f0e, 0x0b0a, 0x0706, 0x0302, 0x0d0c, 0x0908, 0x0504, 0x0100,
        );
        let mut acc = _mm256_setzero_ps();
        let mut accm = _mm_setzero_ps();

        let nbg = nb / 4;
        for i in 0..nbg {
            // Weight side: four blocks' codes plus their (d,m) pairs. The
            // pairs gather is ScaleHelperQ_1's four u32 memcpys — block j's
            // first 4 bytes ARE (d_j, m_j), and the blocks sit 24 bytes
            // apart, so the gather is four small copies into a 16-byte
            // staging buffer (a single 16-byte row load would be wrong:
            // qh/qs bytes would land in the scales).
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
            // SAFETY: 16 readable bytes in the local staging buffer; the
            // f16->f32 conversion is exact, and it is ONE instruction —
            // the Q5_0 round's 8.0 vs 13.2 GB/s lesson applied here too.
            let other = _mm256_cvtph_ps(_mm_shuffle_epi8(
                _mm_loadu_si128(pairs.as_ptr() as *const __m128i),
                dm_shuf,
            ));

            // Activation group i: convert_scales (iqk_legacy:132) —
            // bf16 scales as bits<<16, i16 sums sign-extended to f32.
            // SAFETY: 8 readable bytes at the head of the validated group.
            let g = acol.as_ptr().add(i * Q82X4_STRIDE);
            let aux_d = _mm_castsi128_ps(_mm_slli_epi32::<16>(_mm_cvtepu16_epi32(
                _mm_loadl_epi64(g as *const __m128i),
            )));
            // SAFETY: 8 readable bytes at offset 8 of the group.
            let aux_m = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_loadl_epi64(
                g.add(8) as *const __m128i
            )));
            // prep = (lo: d_a, hi: d_a * m_a).
            let prep = _mm256_set_m128(_mm_mul_ps(aux_d, aux_m), aux_d);
            // s12 = other * prep — lo: d_w*d_a, hi: m_w*(d_a*m_a).
            let s12 = _mm256_mul_ps(other, prep);
            // MinusType1: accm += hi; dall = lo broadcast to both halves.
            accm = _mm_add_ps(accm, _mm256_extractf128_ps(s12, 1));
            let lo = _mm256_castps256_ps128(s12);
            let dall = _mm256_set_m128(lo, lo);

            // Sum4<UnsignedDot, can_pack=false>: four maddubs->madd, then
            // the epi32 unpack chain, integer chains first and the group's
            // one fmadd last — the C's order, kept for the mirror.
            // SAFETY: 32 readable bytes at each offset inside the group
            // (144-byte groups validated by check_row).
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

        // Tail blocks (nb % 4 != 0): AccumT's prepare1 path. The min term
        // is m_w*(d_a*m_a), spread as 0.25 per lane exactly as Q5_0's tail
        // spread its -16 term — replicate the arithmetic, not a tidier one.
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
            // s12 = (dw*da, mw*(da*ma)); MinusType1 keeps dm.second*0.25f
            // per lane and returns dm.first.
            let d = dw * da;
            let corr = mw * (da * ma) * 0.25f32;
            accm = _mm_add_ps(accm, _mm_set1_ps(corr));
            // SAFETY: 32 readable bytes at offset 4 of the tail block.
            let qs = _mm256_loadu_si256(acol.as_ptr().add(tb + 4) as *const __m256i);
            let p0 = _mm256_madd_epi16(m1, _mm256_maddubs_epi16(qx0, qs));
            acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(p0), acc);
        }

        // MinusType1::result: sum = lo128(acc)+hi128(acc), hsum_float_4 of
        // sum + accm (iqk_common.h:225 — movehl then movehdup).
        let sum = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
        let x = _mm_add_ps(sum, accm);
        let x = _mm_add_ps(x, _mm_movehl_ps(x, x));
        _mm_cvtss_f32(_mm_add_ss(x, _mm_movehdup_ps(x)))
    }
}

/// The emulator: the kernel's graph, one weight row against one q8_2_x4
/// column, bit-identical to [`dot_q5f1_q82x4_avx2`] by construction — same
/// operations, same order, `mul_add` where the kernel fuses. The scale
/// gather is the shuffle+cvtph graph's net effect (f16->f32 exact), so the
/// emulator reads d_j/m_j straight from the row. Shares the Q5_0 emulator's
/// epi32/i16 helper set; only the block geometry and the min term differ.
fn dot_q5f1_q82x4_emul(wrow: &[u8], acol: &[u8], nb: usize) -> f32 {
    let mut acc = [0.0f32; 8];
    let mut accm = [0.0f32; 4];

    let nbg = nb / 4;
    for i in 0..nbg {
        // qx[j]: codes of weight block 4i+j; dw[j]/mw[j]: its f16 scales.
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
        // convert_scales: da[l] bf16->f32 (bits<<16), ma[l] i16->f32.
        let mut da = [0.0f32; 4];
        let mut ma = [0.0f32; 4];
        for l in 0..4 {
            da[l] = bf16_bits_to_f32(u16::from_le_bytes([g[2 * l], g[2 * l + 1]]));
            ma[l] = i16::from_le_bytes([g[8 + 2 * l], g[9 + 2 * l]]) as f32;
        }
        // s12: lo[j] = dw[j]*da[j], hi[j] = mw[j]*(da[j]*ma[j]).
        // MinusType1: accm += hi; dall lanes l and 4+l use lo[l%4].
        let mut lo = [0.0f32; 4];
        for l in 0..4 {
            lo[l] = dw[l] * da[l];
            accm[l] += mw[l] * (da[l] * ma[l]);
        }
        // Sum4: p_j = emul_madd1(emul_maddubs(qx[j], qs_j)); the epi32
        // unpack chain; pall lanes land per (block, byte-half).
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
        // acc = fmadd(dall, pall, acc) over all 8 lanes.
        for l in 0..8 {
            acc[l] = lo[l % 4].mul_add(pall[l] as f32, acc[l]);
        }
    }

    // Tail blocks — the 0.25-spread min term, 4 lanes each.
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

    // MinusType1::result + hsum_float_4: sum[l] = acc[l]+acc[4+l];
    // x[l] = sum[l]+accm[l]; (x0 + x2) + (x1 + x3).
    let mut x = [0.0f32; 4];
    for l in 0..4 {
        x[l] = (acc[l] + acc[4 + l]) + accm[l];
    }
    (x[0] + x[2]) + (x[1] + x[3])
}

// ------------------------------------------------- Q8_0 x act cells (MUL-38)
// The q_nope2_absorbed site (model::attn): one output cell is an f64 block
// sum over `nope/32` blocks of `(f16(w_d) * a_d) as f64 * f64::from(isum)`,
// where isum is the i32 dot of a Q8_0 weight block's codes against a
// block_q8_2-convention activation block's codes (model::attn::quantize_act
// — bf16-valued f32 scale + i8 codes, NOT one of the packed formats above).
// The MUL-35 ledger put that dot at 11.17 ms CPU per step against 30.08 MB
// of blocks — a scalar i32 loop at ~2.7 GB/s/core, 16-25% of this crate's
// other kernels. This section is its AVX2 replacement, in the crate's
// three-part shape reduced to two, with the reduction's reason written out:
//
//   * KERNEL — one maddubs chain per 32-value block, the Q6_K qY sign fold
//     (`prepare_signed`'s move, the Q6_K section above): the weight
//     MAGNITUDE becomes maddubs' u8 side (`sign_epi8(w, w)` = |w|) and the
//     weight SIGN is folded into the ACTIVATION bytes (`sign_epi8(a, w)`),
//     so `|w| · (a·sign(w)) = w·a` term by term — exact integer arithmetic,
//     no +128 offset, hence NO compensation sum. The +128 prepare is the
//     form that works when the u8 side has limited range (Q6_K's codes are
//     -32..31, so +32 puts the u8 side at <= 63 and pairs peak at
//     2·63·127 = 16002, inside i16); it does NOT survive this pairing: a
//     Q8_0 block at +128 has u8 lanes 0..255 against s8 lanes -128..127,
//     pairs reach ±255·127·2 = ±64770 — past i16 — while the sign fold's
//     pairs peak at 2·128·127 = 32512 (see the saturation proof below).
//
//   * SCALAR MIRROR — the cell loop verbatim from attn.rs (the mission
//     contract: the current scalar loop stays as the no-AVX2 fallback) and
//     the gate's comparison mirror.
//
//   * no separate EMULATOR. The Q4_K/Q6_K rounds needed an
//     instruction-graph emulator because their f32 lane trees REORDER
//     roundings — a hand-derived lane map can be wrong and stay green. This
//     kernel's only reordering is of INTEGER adds: products w[l]·a[l] lie in
//     [-127·127, 127·127] and their 32-term sum in [-516128, 516128], well
//     inside i32, and integer addition is associative and commutative
//     EXACTLY — every lane tree computes the same i32 as the scalar
//     ascending loop. The f64 epilogue is not SIMD at all (see the kernel),
//     so there is no second lane order to emulate. Bit identity between
//     kernel and mirror is algebra, not a port.
//
// THE BIT-IDENTITY CLAIM (this round's core argument, held by
// `tests/qdot.rs::hw_q_nope2_cells_bit_identical` and, end to end, by the
// model's exact-zero gates):
//
//   1. Per block b, `isum_b = Σ_l w[l]·a[l]` is an exact i32 in any
//      summation order (bounds above) — the maddubs/hadd/madd tree therefore
//      reproduces the scalar isum bit for bit.
//   2. The cell is `Σ_b (half_to_f32(w_d[b]) * a_d[b]) as f64 *
//      f64::from(isum_b)`, blocks ascending, accumulated in f64 with plain
//      (non-fused) mul+add — the same expression in the same order the
//      scalar loop evaluates, so every rounding lands identically.
//   3. Only one input escapes the identity: an activation code of -128
//      under a NEGATIVE weight code. `sign_epi8(a, w)` negates a through
//      8-bit two's complement, and -(-128) wraps to -128, flipping that
//      term's sign. Weight codes of -128 are fine (|w| = 128 is a legal u8
//      magnitude and the fold's products stay |<= 128·127 = 16256| per
//      term). Both producers cannot emit -128 activations:
//      `model::attn::quantize_act` codes are `v/d` with d a bf16 rounding
//      of amax/127 and |v| <= amax, so |code| <= 127/(1 - 2^-9)·(1+2^-24)^3
//      < 127.25, and round-to-nearest-even of anything below 127.5 is at
//      most 127 — the producer's clamp is -127 and ENFORCED there since
//      2026-09-20 (the old -128 clamp was defensive dead code; the bound
//      means it never engaged, so tightening it is bit-identical by
//      derivation and the gates prove it end to end); the same bound with
//      (1+2^-24)^3 alone covers `quantize_q8_0`'s id = 127/amax (its clamp
//      stays -128: a WEIGHT code of -128 is a legal u8 magnitude for the
//      fold, only the activation side is excluded). A debug_assert scans
//      for the excluded pair anyway — a future producer change should fail
//      loudly in debug, not quietly flip signs.
//   4. maddubs i16 saturation is unreachable on ANY i8 x i8 input in the
//      sign-fold form, let alone the legal domain: a positive pair is at
//      most 2·128·127 = 32512 (sq is i8 and cannot hold +128, so a
//      128·128 positive product does not exist), and a negative pair is at
//      least 2·128·(-128) = -32768 — exactly i16::MIN, representable.
//      (Q6_K's section derives its own bound, 8128, from its -32..31
//      codes; this one is the full-i8 statement.)

/// One Q8_0 block over 32 values: f16 scale bits then 32 int8 codes —
/// ggml's `block_q8_0` layout (34 bytes). Home is qdot (MUL-38): the cell
/// kernel consumes these, and `model::attn` re-exports the type so
/// `Derived`'s storage and the gate's `assert_eq!` (field-wise `PartialEq`
/// — equality here is equality of every value it holds) keep compiling
/// unchanged. `repr(C)` pins the field order to the file layout the
/// reference bytes were verified against (2026-09-19, 0 of 69632 differ);
/// `tests/derived.rs` asserts the 34-byte size.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Q8Block {
    /// f16 bits of the block scale (convert with `half_to_f32`).
    pub d: u16,
    /// The int8 codes, `[-127, 127]` by the producer bound (see section
    /// header, point 3).
    pub q: [i8; 32],
}

/// One quantized activation block in the q_nope2 cell contract: the
/// bf16-valued scale held as f32 plus 32 int8 codes — ik's small-M
/// `block_q8_2` CONVENTIONS (scale rounded to bf16, codes from `id = 1/d`)
/// in a Rust-side layout; not the packed x4 form the K-quant kernels eat.
/// Codes are `[-127, 127]` by the same producer bound.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ActBlock {
    /// The bf16-valued block scale as f32 (already rounded; used as-is).
    pub d: f32,
    /// The int8 codes.
    pub q: [i8; 32],
}

/// The cell-segment kernel behind `q_nope2_absorbed`'s pool split: computes
/// output cells `j in [j0, j_end)` of ONE column of the (col, j) cell space
/// — `whead` is the head's weight blocks (`latent * acol.len()` of them,
/// cell j reading `whead[j*nb .. j*nb+nb]`), `acol` this column's
/// quantized activation blocks, `out` receives the `j_end - j0` cells in
/// j order. The f64 block sum, its block order and its expression are the
/// scalar loop's, exactly (section header, points 1-2); dispatch is AVX2
/// when the CPU has it, the scalar mirror otherwise — same bits either way.
///
/// Panics (not `QdotError`) on shape errors: the caller is a pool callback
/// that cannot propagate a `Result`, the bounds are the caller's own cell
/// split, and a panic through the pool is loud (the pool propagates it —
/// `tests/pool.rs`). A wrong length here would otherwise index another
/// cell's blocks and keep producing plausible numbers, the same failure
/// class `check_row` rejects for `dot_row`.
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
        // SAFETY: AVX2+FMA detected just now; the shape asserts above bound
        // every index the kernel touches.
        unsafe { q_nope2_cells_avx2_inner(whead, acol, j0, j_end, out) }
    } else {
        q_nope2_cells_scalar(whead, acol, j0, j_end, out);
    }
}

/// The scalar mirror: the cell loop verbatim from `q_nope2_absorbed`'s
/// pre-MUL-38 body, and the crate's no-AVX2 fallback for this pairing.
/// Bit-identical to the kernel by the integer-exactness argument (section
/// header) — not by emulation; there is no lane order to reproduce.
/// Public for the gate's kernel-vs-mirror compare.
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

/// The AVX2 kernel behind [`q_nope2_cells`], for the gate's path comparison
/// (the mirror of [`dot_row_avx2`]). Panics when the CPU lacks AVX2+FMA —
/// on such a machine there is nothing to compare and the gate wants the
/// loud failure, not a quiet fallback.
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

/// The kernel: one monolithic fn on purpose (the MUL-27 measurement —
/// helper splits lose 10-13% on this loop shape; the MUL-26 lesson —
/// missing `target_feature` is not an error and costs tens of times over).
/// Per 32-value block: sign fold (2 `vpsignb`), one `vpmaddubsw`, one
/// `vpmaddwd`, two `vphaddd`, two extracts — an exact i32 by the section
/// header's associativity argument — then the scalar f64 epilogue in the
/// scalar loop's own order. No FMA instruction appears (the f64 mul+add
/// must stay two roundings to match the mirror; Rust never contracts
/// them); `fma` is enabled anyway so the detection and the attribute agree
/// with the crate's other kernels.
///
/// # Safety
/// The CPU must support AVX2+FMA, and the caller must have validated the
/// segment bounds: `j_end * acol.len()` weight blocks, all of `acol`, and
/// `j_end - j0` out cells ([`q_nope2_cells`] asserts all three).
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn q_nope2_cells_avx2_inner(
    whead: &[Q8Block],
    acol: &[ActBlock],
    j0: usize,
    j_end: usize,
    out: &mut [f32],
) {
    unsafe {
        let ones = _mm256_set1_epi16(1);
        let nb = acol.len();
        for j in j0..j_end {
            let mut cell = 0.0f64;
            for b in 0..nb {
                let wb = &whead[j * nb + b];
                let ab = &acol[b];
                // SAFETY: 32-byte unaligned loads inside the 32-byte code
                // arrays of blocks whose counts the dispatcher asserted.
                let wv = _mm256_loadu_si256(wb.q.as_ptr() as *const __m256i);
                let av = _mm256_loadu_si256(ab.q.as_ptr() as *const __m256i);
                // The qY sign fold: |w| on the u8 side, w's sign folded
                // into a's bytes. w·a = |w|·(a·sign(w)) term by term; the
                // -128 activation corner this identity misses is excluded
                // by the producer bound (section header, point 3).
                let us = _mm256_sign_epi8(wv, wv);
                let sq = _mm256_sign_epi8(av, wv);
                // 16 i16 pair sums, then 8 i32 quads, then the hadd chain —
                // integer adds, associative, the tree shape is free.
                let pairs = _mm256_maddubs_epi16(us, sq);
                let quads = _mm256_madd_epi16(ones, pairs);
                let h1 = _mm256_hadd_epi32(quads, quads);
                let h2 = _mm256_hadd_epi32(h1, h1);
                // Lane 0 holds the low 128-bit half's total, lane 4 the
                // high half's; their sum is the block's exact i32.
                let isum = _mm256_extract_epi32(h2, 0) + _mm256_extract_epi32(h2, 4);
                // Debug-only guard on the one input pair outside the sign
                // fold's identity (section header, point 3); zero cost in
                // release, loud in debug if a producer ever changes.
                #[cfg(debug_assertions)]
                for l in 0..32 {
                    debug_assert!(
                        !(ab.q[l] == -128 && wb.q[l] < 0),
                        "activation code -128 under a negative weight code: \
                         outside the sign-fold kernel's contract"
                    );
                }
                // The scalar loop's epilogue, character for character: one
                // f32 multiply (one rounding), the exact widening to f64,
                // one f64 multiply and one f64 add per block, blocks
                // ascending. This line IS the bit-identity contract's step 2.
                cell += (half_to_f32(wb.d) * ab.d) as f64 * f64::from(isum);
            }
            out[j - j0] = cell as f32;
        }
    }
}
