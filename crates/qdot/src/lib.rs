//! qdot — fused quantized row dots for host CPU: Q3_K x Q8_K, Q4_K/Q5_K x Q8_2_X4,
//! Q6_K x Q8_2_X4, Q5_0 x Q8_2_X4, Q5_1 x Q8_2_X4, Q8_0 x Q8_2_X4, Q2_K x Q8_K, IQ2_XS x Q8_K,
//! IQ2_XXS x Q8_K, IQ1_S x Q8_K, IQ3_XXS x Q8_K, IQ3_S x Q8_K, MXFP4 x Q8_2_X4, NVFP4 x Q8_2_X4, IQ4_NL x Q8_2_X4,
//! IQ4_XS x Q8_K, Q2_0 x Q8_0, and Q8_0 x act cell kernels.
//!
//! Dots quantized weight rows directly against quantized activation columns without
//! materializing f32 weights, scaling once per block:
//! - Q3_K x Q8_K: port of `ggml_vec_dot_q3_K_q8_K` (ggml-quants.c:6482).
//! - Q4_K x Q8_2_X4: port of ik's `mul_mat_qX_K_q8_2_X4_T` (iqk_gemm_kquants.cpp:783).
//! - Q6_K x Q8_2_X4: port of ik's `mul_mat_qY_K_q8_2_X4_T` (iqk_gemm_kquants.cpp:938).
//! - Q5_K x Q8_2_X4: the Q4_K template with ik's `DequantizerQ5K_AVX2` (iqk_gemm_kquants.cpp:763).
//! - Q5_0 x Q8_2_X4: port of ik's `mul_mat_qX_1_q8_2_T<Q5_0_1_Unpacker>` (iqk_gemm_legacy_quants.cpp:507).
//! - Q5_1 x Q8_2_X4: port of ik's `mul_mat_qX_1_q8_2_T<Q5_1_Unpacker>` (iqk_gemm_legacy_quants.cpp:804).
//! - Q2_K x Q8_K: port of ik's `mul_mat_qY_K_q8_K_T<DequantizerQ2K, 1>`
//!   (iqk_gemm_kquants.cpp:678, :511), the body its AVX2 (non-AVX512) build runs.
//! - IQ2_XS x Q8_K: port of ik's `mul_mat_qX_K_q8_K_IQ_N<DequantizerIQ2XS, 1>`
//!   (iqk_gemm_iquants.cpp:788, :236), the body its AVX2 (non-AVX512) build runs.
//! - IQ2_XXS x Q8_K: port of ik's `mul_mat_qX_K_q8_K_IQ_N<DequantizerIQ2XXS, nrc_y>`
//!   (iqk_gemm_iquants.cpp:788, :148), the body its AVX2 (non-AVX512) build runs at every
//!   column count below the 32 where it converts to Q8_K_R8 instead (iqk_mul_mat.cpp:249):
//!   the signs on the weight side, the grid values offset by +43 onto maddubs' unsigned
//!   side and the offset folded back out through the column's bsums.
//! - IQ1_S x Q8_K: port of ik's `mul_mat_iq1_s_q8_K<nrc_y>` (iqk_gemm_1bit.cpp:793), the
//!   body its AVX2 (non-AVX512) build runs at every column count below the 32 where it
//!   converts the weights to Q8_K_R8 instead (iqk_mul_mat.cpp:315): the grid values offset
//!   by +1 onto maddubs' unsigned side, the `8·(2s + 1)` scale inside the i16 dot pairs,
//!   and the `±1` delta folded back out through the column's bsums.
//! - IQ3_XXS x Q8_K: port of ik's `mul_mat_qX_K_q8_K_IQ_N<DequantizerIQ3XXS, 1>`
//!   (iqk_gemm_iquants.cpp:787, :494), the body its AVX2 (non-AVX512) build runs.
//! - IQ3_S x Q8_K: port of ik's `mul_mat_qX_K_q8_K_IQ_N<DequantizerIQ3S, 1>`
//!   (iqk_gemm_iquants.cpp:787, :624), the body its AVX2 (non-AVX512) build runs.
//! - MXFP4 x Q8_2_X4: port of ik's `mul_mat_qX_1_q8_2_T<MXFP4_Unpacker>` (iqk_gemm_legacy_quants.cpp:779).
//! - NVFP4 x Q8_2_X4: ours (ik has no NVFP4; mainline's `ggml_vec_dot_nvfp4_q8_0` reads Q8_0 columns). Each 16
//!   values' UE4M3 scale multiplies that group's integer code sum in f32, so the sums take the signed form
//!   (`maddubs(|w|, sign(q8, w))`, the IQ4_NL kernel's) and no offset term exists. One kernel reads mainline's
//!   block ([`GgmlType::NVFP4`]) and bloomery's rows ([`GgmlType::NVFP4_ROWS`]) as 9-byte groups ([`dot_row`]) or as
//!   two planes ([`dot_nvfp4_planes_row`]). Rows of a plane matrix against many columns go through
//!   [`dot_nvfp4_planes_rows_cols`]: a tile of up to [`TILE_COLS`] columns that dots the offset codes (`w + 12`, as
//!   MXFP4 does) and takes each column's 16-value code sums, built once for the call, off in integers, so its sums
//!   and float steps are the one-column kernel's. The dot is the row's own: the tensor scale is the caller's factor.
//! - IQ4_NL x Q8_2_X4: port of ik's `mul_mat_qX_0_q8_0_T<IQ4_NL_UnpackerS, _, block_q8_2>`
//!   (iqk_gemm_legacy_quants.cpp:407, :2596), the body its AVX2 (non-AVX512) build runs.
//! - IQ4_XS x Q8_K: port of ik's `mul_mat_qX_K_q8_K_T<DequantizerIQ4XS>`
//!   with its signed table and `multiply_add_avx2`, the body its AVX2 build runs.
//! - Q8_0 x Q8_2_X4: port of ik's `mul_mat_qX_0_q8_0_T<Q8_0_Unpacker, 1, block_q8_2>`
//!   (iqk_gemm_legacy_quants.cpp:404, :753), the body its AVX2 (no AVX512-VNNI) build runs.
//! - Q2_0 x Q8_0: port of mainline's `ggml_vec_dot_q2_0_q8_0_generic` (llama.cpp `f2cc7282c`
//!   ggml-cpu/quants.c:177), the loop its x86 build runs (arch-fallback.h:90), with its float
//!   steps fused as a native GCC or Clang build contracts them; ik has no Q2_0.
//! - Q8_0 x act cells: fused cell kernel for `q_nope2_absorbed` (model::arch::deepseek2::attn).
//! - Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q5_1, Q8_0, IQ2_XS, IQ2_XXS, IQ1_S, IQ3_XXS, IQ3_S, IQ4_XS, IQ4_NL and MXFP4 tiles: one weight row against up
//!   to [`TILE_COLS`] columns, each block unpacked once, every column bit-identical to its
//!   one-column kernel ([`dot_row_cols`]); the Q5_1 and Q8_0 tiles are ik's `AccumT::compute` at `nrc_y` = the
//!   column count (iqk_gemm_legacy_quants.cpp:302); the Q6_K tile is `mul_mat_qY_K_q8_2_X4_T` at `nrc_y` = the
//!   column count (iqk_gemm_kquants.cpp:938).
//! - IQ2_XXS wide tile: one weight row against up to [`WIDE_COLS`] columns
//!   ([`dot_row_cols_wide`]), each block unpacked once for every column of the call
//!   and the columns dotted eight at a time with their sums in registers, every
//!   column bit-identical to its one-column kernel; the IQ2_XXS tile is its one
//!   group.
//! - Q3_K row-lane tile: [`Q3K_R8_ROWS`] rows repacked into one lane-interleaved layout
//!   ([`repack_q3k_r8`], undone by [`unpack_q3k_r8`]) against up to [`TILE_COLS`] columns,
//!   every (row, column) value bit-identical to [`dot_row`] ([`dot_q3k_r8_cols`]).
//! - The card-rule kernels ([`card_q8_1`], [`card_q3k_dot_row_cols`],
//!   [`card_q4k_dot_row_cols`], [`card_swiglu_clamp`]): V4.1's routed expert
//!   as the card engine computes it — q8_1 activations per 128 values, the
//!   warp-lane walks and the xor butterfly — so an expert holds the same
//!   bits whichever tier runs it. Ours, not ik's: the host tier's other
//!   columns keep the kernels above.
//!
//! - [`generic`]: a type with no hand kernel above runs its block decoder ([`generic::HostDecode`])
//!   through one kernel per activation form (Q8_K, Q8_2_X4); the hand kernels are matched first.
//!   IQ2_S x Q8_K and IQ1_M x Q8_K are listed there.
//!
//! Super-block geometry (block_q3_K, 110 bytes / 256 values): hmask[32] @+0,
//! qs[64] @+32, scales[12] @+96, f16 d @+108.
//!
//! Each AVX2 kernel has a bit-identical scalar mirror for fallback and verification.
//! Q4_K's one-column kernel, its tile and its row lanes also have AVX-512 VNNI
//! bodies, picked at run time ([`q4k_avx512`]) and bit-identical to the AVX2 ones.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::fmt;
use std::ops::Range;

use gguf::GgmlType;
use gguf::iq_tables::{
    IQ1S_GRID, IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID, KSIGNS_IQ2XS, KVALUES_IQ4NL,
};
use gguf::quant::{
    KVALUES_MXFP4, bf16_bits_to_f32, e8m0_to_f32_half, fp32_to_bf16_bits, half_to_f32, q82_inverse,
};
#[cfg(target_arch = "x86_64")]
use gguf::quant::{f32_to_f16_bits, non_finite_activation, q80_inverse};

pub mod generic;

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
/// `sizeof(block_q8_0)` (ggml-common.h): f16 d @0, qs i8[32] @2 — 34 bytes / 32 values.
const Q8F0_BLOCK: usize = 34;
/// `sizeof(block_q2_0)` (llama.cpp `f2cc7282c` ggml-common.h:187-192): f16 d @0, qs[16] @2,
/// 18 bytes / 64 values.
const Q2F0_BLOCK: usize = 18;
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
    /// A [`dot_row_cols`] call's column count is outside `1..=TILE_COLS`, or
    /// its output slice does not hold one value per column.
    TileShape { cols: usize, outs: usize },
    /// A [`dot_row_cols_wide`] call's column count is outside `1..=WIDE_COLS`,
    /// or its output slice does not hold one value per column.
    WideShape { cols: usize, outs: usize },
    /// A row-lane repack's row count is not a multiple of [`Q3K_R8_ROWS`].
    RowGroup { rows: usize },
    /// A row-lane buffer is not exactly the bytes its rows and `k` make;
    /// `buf` names which: `"repack source"` or `"repack destination"`
    /// ([`repack_q3k_r8`]'s `rows` or `out`), `"unpack source"` or
    /// `"unpack destination"` ([`unpack_q3k_r8`]'s `groups` or `out`),
    /// `"tile group"` (the group handed to [`dot_q3k_r8_cols`]), or `"lane
    /// rows"`, `"lane next rows"` or `"lane group"` ([`pack_lanes`]' `rows`,
    /// `next` or `packed`, the group [`dot_lanes_cols`] reads).
    RowGroupBytes {
        buf: &'static str,
        have: usize,
        need: usize,
    },
    /// The weight type has no row-lane kernel on this machine
    /// ([`has_lanes`]: Q4_K, Q5_K and Q6_K).
    NoLanes(GgmlType),
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
            QdotError::TileShape { cols, outs } => {
                write!(
                    f,
                    "tile call with {cols} activation columns and {outs} outputs: \
                     a call takes 1..={TILE_COLS} columns and one output per column"
                )
            }
            QdotError::WideShape { cols, outs } => {
                write!(
                    f,
                    "wide call with {cols} activation columns and {outs} outputs: \
                     a call takes 1..={WIDE_COLS} columns and one output per column"
                )
            }
            QdotError::RowGroup { rows } => {
                write!(
                    f,
                    "{rows} rows: the row-lane layout takes rows in groups of {Q3K_R8_ROWS}"
                )
            }
            QdotError::RowGroupBytes { buf, have, need } => {
                write!(
                    f,
                    "row-lane {buf} is {have} bytes, its rows and k make {need}"
                )
            }
            QdotError::NoLanes(w) => {
                write!(
                    f,
                    "no row-lane qdot kernel for weight type {w:?} on this machine \
                     (Q4_K, Q5_K and Q6_K with AVX2+FMA+F16C)"
                )
            }
        }
    }
}

impl std::error::Error for QdotError {}

// ------------------------------------------------------------ public API

/// The weight types this crate has a kernel and an activation format for — the
/// one list every entry point checks against: the hand kernels, then the types
/// [`generic::ENTRIES`] decodes.
fn has_kernel(w: GgmlType) -> bool {
    matches!(
        w,
        GgmlType::Q2_K
            | GgmlType::Q3_K
            | GgmlType::Q4_K
            | GgmlType::Q5_K
            | GgmlType::Q5_0
            | GgmlType::Q5_1
            | GgmlType::Q8_0
            | GgmlType::Q6_K
            | GgmlType::IQ2_XS
            | GgmlType::IQ2_XXS
            | GgmlType::IQ1_S
            | GgmlType::IQ3_XXS
            | GgmlType::IQ3_S
            | GgmlType::MXFP4
            | GgmlType::NVFP4
            | GgmlType::NVFP4_ROWS
            | GgmlType::IQ4_NL
            | GgmlType::IQ4_XS
            | GgmlType::Q2_0
    ) || generic::entry(w).is_some()
}

/// Whether the fused path handles this weight type on this machine: the type
/// is supported and the CPU has the required features.
#[cfg(target_arch = "x86_64")]
#[must_use]
pub fn supports(w: GgmlType) -> bool {
    has_kernel(w) && has_features(w)
}

/// Off x86_64 every type with a kernel takes the fused path through its
/// scalar mirror: the AVX2 kernels' bits, not their speed.
#[cfg(not(target_arch = "x86_64"))]
#[must_use]
pub fn supports(w: GgmlType) -> bool {
    has_kernel(w)
}

/// Whether a `k`-wide row of `w` takes the fused path: [`supports`] and the
/// [`k_granularity`] contract. The matmul dispatch's decision; anything else
/// dequantizes the row to f32 first.
#[must_use]
pub fn fuses(w: GgmlType, k: usize) -> bool {
    supports(w) && k.is_multiple_of(k_granularity(w))
}

/// The per-type ISA table matching each fused kernel's `#[target_feature]` requirements.
///
/// Q3_K needs avx2+f16c; MXFP4, NVFP4, NVFP4 rows and the generic decoders need avx2+fma; every other type avx2+fma+f16c.
#[cfg(target_arch = "x86_64")]
fn has_features(w: GgmlType) -> bool {
    // Per arm, so a call checks only the features its kernel needs: `dot_row`
    // asks once per row.
    let avx2 = || std::arch::is_x86_feature_detected!("avx2");
    let fma = || std::arch::is_x86_feature_detected!("fma");
    let f16c = || std::arch::is_x86_feature_detected!("f16c");
    match w {
        // dot_q3k_q8k_avx2: enable = "avx2", "f16c"
        GgmlType::Q3_K => avx2() && f16c(),
        // dot_mxfp4_q82x4_avx2, dot_nvfp4_q82x4_avx2: enable = "avx2", "fma"
        GgmlType::MXFP4 | GgmlType::NVFP4 | GgmlType::NVFP4_ROWS => avx2() && fma(),
        // dot_q4k/q5k/q6k_q82x4_avx2, dot_q5f0/q5f1/q8f0_q82x4_avx2, dot_q2k_q8k_avx2,
        // dot_iq2xs_q8k_avx2, dot_iq2xxs_q8k_avx2, dot_iq1s_q8k_avx2, dot_iq3xxs_q8k_avx2,
        // dot_iq3s_q8k_avx2, dot_iq4nl_q82x4_avx2, dot_iq4xs_q8k_avx2, dot_q2f0_q8f0_avx2:
        // enable = "avx2", "fma", "f16c"
        GgmlType::Q2_K
        | GgmlType::IQ2_XS
        | GgmlType::IQ2_XXS
        | GgmlType::IQ1_S
        | GgmlType::Q4_K
        | GgmlType::Q5_K
        | GgmlType::Q6_K
        | GgmlType::Q5_0
        | GgmlType::Q5_1
        | GgmlType::Q8_0
        | GgmlType::IQ3_XXS
        | GgmlType::IQ3_S
        | GgmlType::IQ4_NL
        | GgmlType::IQ4_XS
        | GgmlType::Q2_0 => avx2() && fma() && f16c(),
        // generic::dot_avx2: enable = "avx2", "fma"
        other => generic::entry(other).is_some() && avx2() && fma(),
    }
}

/// Off x86_64 no AVX2 kernel is compiled, so no type has its ISA.
#[cfg(not(target_arch = "x86_64"))]
fn has_features(_: GgmlType) -> bool {
    false
}

/// Whether Q4_K's one-column kernel, its tile and its row lanes run their
/// AVX-512 VNNI bodies on this machine: the CPU has AVX-512 F, BW and VNNI
/// beside the AVX2 kernels' ISA ([`has_features`]), and
/// `BLOOMERY_QDOT_AVX512` is unset or `1`; `0` keeps the AVX2 bodies, the
/// same-binary arm. Both bodies write the same bits. Read once a process; a
/// value other than `0` or `1` is refused by name on every architecture,
/// before the CPU is asked. Off x86_64 no AVX-512 body is compiled and the
/// answer is false. The host tier's load asks it for every Q4_K stack
/// (`model::ops::host_stack_runs`), so a refused value stops the load.
#[must_use]
pub fn q4k_avx512() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let wanted = match std::env::var_os("BLOOMERY_QDOT_AVX512") {
            None => true,
            Some(v) if v == "1" => true,
            Some(v) if v == "0" => false,
            Some(v) => panic!("BLOOMERY_QDOT_AVX512={v:?}: takes 0 or 1"),
        };
        #[cfg(target_arch = "x86_64")]
        let isa = has_features(GgmlType::Q4_K)
            && std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512vnni");
        #[cfg(not(target_arch = "x86_64"))]
        let isa = false;
        wanted && isa
    })
}

/// The `k` contract: `k` must be a multiple of the weight format's block size
/// (256 for K-quants, IQ2_XS, IQ2_XXS, IQ1_S, IQ3_XXS, IQ3_S and IQ4_XS, 64 for Q2_0 and NVFP4, 32 for Q5_0, Q5_1, Q8_0,
/// MXFP4, NVFP4 rows and IQ4_NL), a generic decoder's block values. A type with no kernel here gets the K-quant contract: its
/// callers outside this crate ask the size contract of weight types that never reach a kernel.
pub fn k_granularity(w: GgmlType) -> usize {
    match w {
        GgmlType::Q5_0
        | GgmlType::Q5_1
        | GgmlType::Q8_0
        | GgmlType::MXFP4
        | GgmlType::NVFP4_ROWS
        | GgmlType::IQ4_NL => 32,
        GgmlType::Q2_0 | GgmlType::NVFP4 => 64,
        other => generic::entry(other).map_or(256, |e| e.values),
    }
}

/// Bytes one quantized activation column of `k` values occupies in `w`'s format.
///
/// Panics if `w` has no activation format in this build or `k` breaks the
/// type's [`k_granularity`] contract.
#[must_use]
pub fn col_bytes(w: GgmlType, k: usize) -> usize {
    assert!(
        has_kernel(w),
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
        GgmlType::Q5_0
        | GgmlType::Q5_1
        | GgmlType::Q8_0
        | GgmlType::MXFP4
        | GgmlType::NVFP4
        | GgmlType::NVFP4_ROWS
        | GgmlType::IQ4_NL => (k / 128) * Q82X4_STRIDE + ((k % 128) / 32) * Q82_BLOCK,
        GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q6_K => (k / 128) * Q82X4_STRIDE,
        GgmlType::Q2_0 => (k / 32) * Q8F0_BLOCK,
        GgmlType::Q2_K
        | GgmlType::Q3_K
        | GgmlType::IQ2_XS
        | GgmlType::IQ2_XXS
        | GgmlType::IQ1_S
        | GgmlType::IQ3_XXS
        | GgmlType::IQ3_S
        | GgmlType::IQ4_XS => (k / 256) * Q8K_STRIDE,
        // has_kernel above refused every other type but the generic ones.
        other => generic::entry(other)
            .expect("has_kernel named a type with neither a hand kernel nor an entry")
            .col_bytes(k),
    }
}

/// Quantize one activation column into the block format `w` implies.
///
/// Q2_K, Q3_K, IQ2_XS, IQ2_XXS, IQ1_S, IQ3_XXS, IQ3_S and IQ4_XS use `block_q8_K` (296 B/256 values); Q4_K, Q5_K, Q5_0, Q5_1,
/// Q8_0, Q6_K, MXFP4, NVFP4, NVFP4 rows and IQ4_NL use `block_q8_2_x4` (144 B/128 values), with 36-byte
/// `block_q8_2` tails for Q5_0/Q5_1/Q8_0/MXFP4/NVFP4/NVFP4 rows/IQ4_NL; Q2_0 uses `block_q8_0` (34 B/32 values).
/// `out.len()` must equal `col_bytes(w, x.len())`; panics otherwise, and by
/// name on a non-finite activation value (undefined input is refused, never
/// encoded). Uses the AVX2 encoders when the CPU has AVX2 — byte-identical to
/// the scalar mirrors.
#[cfg(target_arch = "x86_64")]
pub fn quantize_col(w: GgmlType, x: &[f32], out: &mut [u8]) {
    quantize_col_check(w, x, out);
    let avx2 = std::arch::is_x86_feature_detected!("avx2");
    match w {
        GgmlType::Q4_K
        | GgmlType::Q5_K
        | GgmlType::Q5_0
        | GgmlType::Q5_1
        | GgmlType::Q8_0
        | GgmlType::Q6_K
        | GgmlType::MXFP4
        | GgmlType::NVFP4
        | GgmlType::NVFP4_ROWS
        | GgmlType::IQ4_NL => {
            if avx2 {
                // SAFETY: AVX2 detected just above; quantize_col_check pinned
                // out.len() to col_bytes(w, x.len()).
                unsafe { quantize_q82x4_col_avx2(x, out) }
            } else {
                quantize_q82x4_col(x, out);
            }
        }
        GgmlType::Q2_K
        | GgmlType::Q3_K
        | GgmlType::IQ2_XS
        | GgmlType::IQ2_XXS
        | GgmlType::IQ1_S
        | GgmlType::IQ3_XXS
        | GgmlType::IQ3_S
        | GgmlType::IQ4_XS => {
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
        GgmlType::Q2_0 => {
            if avx2 {
                // SAFETY: AVX2 detected just above; quantize_col_check pinned
                // out.len() to col_bytes(w, x.len()).
                unsafe { quantize_q8f0_col_avx2(x, out) }
            } else {
                quantize_q8f0_col(x, out);
            }
        }
        // quantize_col_check refused every type without a kernel or an entry.
        other => quantize_generic(other, x, out, avx2),
    }
}

/// Off x86_64 no AVX2 encoder is compiled: [`quantize_col_scalar`].
#[cfg(not(target_arch = "x86_64"))]
pub fn quantize_col(w: GgmlType, x: &[f32], out: &mut [u8]) {
    quantize_col_scalar(w, x, out);
}

/// The scalar mirror behind [`quantize_col`] — the fallback the crate runs
/// when AVX2 is absent, and the oracle its bit-identity gate compares against.
pub fn quantize_col_scalar(w: GgmlType, x: &[f32], out: &mut [u8]) {
    quantize_col_check(w, x, out);
    match w {
        GgmlType::Q4_K
        | GgmlType::Q5_K
        | GgmlType::Q5_0
        | GgmlType::Q5_1
        | GgmlType::Q8_0
        | GgmlType::Q6_K
        | GgmlType::MXFP4
        | GgmlType::NVFP4
        | GgmlType::NVFP4_ROWS
        | GgmlType::IQ4_NL => {
            quantize_q82x4_col(x, out);
        }
        GgmlType::Q2_K
        | GgmlType::Q3_K
        | GgmlType::IQ2_XS
        | GgmlType::IQ2_XXS
        | GgmlType::IQ1_S
        | GgmlType::IQ3_XXS
        | GgmlType::IQ3_S
        | GgmlType::IQ4_XS => {
            let (xblocks, _) = x.as_chunks::<256>();
            let (oblocks, _) = out.as_chunks_mut::<296>();
            for (xb, ob) in xblocks.iter().zip(oblocks.iter_mut()) {
                quantize_q8k_block(xb, ob);
            }
        }
        GgmlType::Q2_0 => quantize_q8f0_col(x, out),
        // quantize_col_check refused every type without a kernel or an entry.
        other => quantize_generic(other, x, out, false),
    }
}

/// The activation column of a type [`generic::ENTRIES`] decodes, in its entry's form.
fn quantize_generic(w: GgmlType, x: &[f32], out: &mut [u8], avx2: bool) {
    generic::entry(w)
        .expect("quantize_col_check named a type with neither a hand kernel nor an entry")
        .quantize(x, out, avx2);
}

/// The shape contract every `quantize_col` entry point enforces before encoding.
fn quantize_col_check(w: GgmlType, x: &[f32], out: &[u8]) {
    assert!(
        has_kernel(w),
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
/// Returns Err on shape or alignment mismatch. Uses AVX2 kernels when supported
/// (Q4_K's AVX-512 VNNI body where [`q4k_avx512`] says so), falling back to
/// scalar mirrors.
#[cfg(target_arch = "x86_64")]
pub fn dot_row(w: GgmlType, wrow: &[u8], acol: &[u8], k: usize) -> Result<f32, QdotError> {
    let nb = check_row(w, wrow.len(), acol.len(), k)?;
    let hw = has_features(w);
    match (w, hw) {
        (GgmlType::Q3_K, true) => {
            // SAFETY: AVX2 was just detected; lengths validated by check_row.
            Ok(unsafe { dot_q3k_q8k_avx2(wrow, acol, nb) })
        }
        (GgmlType::Q4_K, true) if q4k_avx512() => {
            // SAFETY: q4k_avx512 saw AVX-512 F+BW+VNNI and AVX2+FMA+F16C;
            // check_row sized the row and the column for nb super-blocks.
            Ok(unsafe { dot_q4k_q82x4_tile_avx512::<1>(wrow, &[acol.as_ptr()], nb) }[0])
        }
        (GgmlType::Q4_K, true) => {
            // SAFETY: AVX2+FMA were just detected; lengths validated above.
            Ok(unsafe { dot_q4k_q82x4_avx2(wrow, acol, nb) })
        }
        (GgmlType::Q5_K, true) => {
            // SAFETY: AVX2+FMA were just detected; lengths validated above.
            Ok(unsafe { dot_q5k_q82x4_avx2(wrow, acol, nb) })
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
        (GgmlType::Q8_0, true) => {
            // SAFETY: AVX2+FMA+F16C were just detected; lengths validated
            // above.
            Ok(unsafe { dot_q8f0_q82x4_avx2(wrow, acol, nb) })
        }
        (GgmlType::Q2_K, true) => {
            // SAFETY: AVX2+FMA+F16C were just detected; lengths validated above.
            Ok(unsafe { dot_q2k_q8k_avx2(wrow, acol, nb) })
        }
        (GgmlType::IQ2_XS, true) => {
            // SAFETY: AVX2+FMA+F16C were just detected; lengths validated above.
            Ok(unsafe { dot_iq2xs_q8k_avx2(wrow, acol, nb) })
        }
        (GgmlType::IQ2_XXS, true) => {
            // SAFETY: AVX2+FMA+F16C were just detected; lengths validated above.
            Ok(unsafe { dot_iq2xxs_q8k_avx2(wrow, acol, nb) })
        }
        (GgmlType::IQ1_S, true) => {
            // SAFETY: AVX2+FMA+F16C were just detected; lengths validated above.
            Ok(unsafe { dot_iq1s_q8k_avx2(wrow, acol, nb) })
        }
        (GgmlType::IQ3_XXS, true) => {
            // SAFETY: AVX2+FMA+F16C were just detected; lengths validated above.
            Ok(unsafe { dot_iq3xxs_q8k_avx2(wrow, acol, nb) })
        }
        (GgmlType::IQ3_S, true) => {
            // SAFETY: AVX2+FMA+F16C were just detected; lengths validated above.
            Ok(unsafe { dot_iq3s_q8k_avx2(wrow, acol, nb) })
        }
        (GgmlType::MXFP4, true) => {
            // SAFETY: AVX2+FMA were just detected; lengths validated above.
            Ok(unsafe { dot_mxfp4_q82x4_avx2(wrow, acol, nb) })
        }
        (GgmlType::NVFP4, true) => {
            // SAFETY: AVX2+FMA were just detected; check_row sized the row and the column for nb units.
            Ok(
                unsafe {
                    dot_nvfp4_q82x4_avx2::<NV_BLOCKS>(wrow.as_ptr(), wrow.as_ptr(), acol, nb)
                },
            )
        }
        (GgmlType::NVFP4_ROWS, true) => {
            // SAFETY: AVX2+FMA were just detected; check_row sized the row and the column for nb units.
            Ok(unsafe { dot_nvfp4_q82x4_avx2::<NV_ROWS>(wrow.as_ptr(), wrow.as_ptr(), acol, nb) })
        }
        (GgmlType::IQ4_NL, true) => {
            // SAFETY: AVX2+FMA+F16C were just detected; lengths validated above.
            Ok(unsafe { dot_iq4nl_q82x4_avx2(wrow, acol, nb) })
        }
        (GgmlType::IQ4_XS, true) => {
            // SAFETY: AVX2+FMA+F16C were just detected; lengths validated above.
            Ok(unsafe { dot_iq4xs_q8k_avx2(wrow, acol, nb) })
        }
        (GgmlType::Q2_0, true) => {
            // SAFETY: AVX2+FMA+F16C were just detected; lengths validated above.
            Ok(unsafe { dot_q2f0_q8f0_avx2(wrow, acol, nb) })
        }
        (GgmlType::Q4_K, false) => {
            // A CPU without the AVX2 kernels' ISA still has q4k_avx512 read
            // the lever, so a refused value stops here too.
            let _ = q4k_avx512();
            Ok(dot_row_scalar_ty(w, wrow, acol, nb))
        }
        _ => match generic::entry(w) {
            // SAFETY: has_features saw AVX2+FMA; lengths validated above.
            Some(e) if hw => Ok(unsafe { e.dot_avx2(wrow, acol, nb) }),
            _ => Ok(dot_row_scalar_ty(w, wrow, acol, nb)),
        },
    }
}

/// Off x86_64 no AVX2 kernel is compiled: every type runs its scalar mirror
/// ([`dot_row_scalar`]). Q4_K asks [`q4k_avx512`] first, so a refused
/// `BLOOMERY_QDOT_AVX512` value stops here as it does on x86_64.
#[cfg(not(target_arch = "x86_64"))]
pub fn dot_row(w: GgmlType, wrow: &[u8], acol: &[u8], k: usize) -> Result<f32, QdotError> {
    if w == GgmlType::Q4_K {
        let _ = q4k_avx512();
    }
    dot_row_scalar(w, wrow, acol, k)
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
        GgmlType::Q5_K => dot_q5k_q82x4_emul(wrow, acol, nb),
        GgmlType::Q6_K => dot_q6k_q82x4_emul(wrow, acol, nb),
        GgmlType::Q5_0 => dot_q5f0_q82x4_emul(wrow, acol, nb),
        GgmlType::Q5_1 => dot_q5f1_q82x4_emul(wrow, acol, nb),
        GgmlType::Q8_0 => dot_q8f0_q82x4_emul(wrow, acol, nb),
        GgmlType::Q2_K => dot_q2k_q8k_emul(wrow, acol, nb),
        GgmlType::Q3_K => dot_q3k_q8k_scalar(wrow, acol, nb),
        GgmlType::IQ2_XS => dot_iq2xs_q8k_emul(wrow, acol, nb),
        GgmlType::IQ2_XXS => dot_iq2xxs_q8k_emul(wrow, acol, nb),
        GgmlType::IQ1_S => dot_iq1s_q8k_emul(wrow, acol, nb),
        GgmlType::IQ3_XXS => dot_iq3xxs_q8k_emul(wrow, acol, nb),
        GgmlType::IQ3_S => dot_iq3s_q8k_emul(wrow, acol, nb),
        GgmlType::MXFP4 => dot_mxfp4_q82x4_emul(wrow, acol, nb),
        GgmlType::NVFP4 => dot_nvfp4_q82x4_emul::<NV_BLOCKS>(wrow, wrow, acol, nb),
        GgmlType::NVFP4_ROWS => dot_nvfp4_q82x4_emul::<NV_ROWS>(wrow, wrow, acol, nb),
        GgmlType::IQ4_NL => dot_iq4nl_q82x4_emul(wrow, acol, nb),
        GgmlType::IQ4_XS => dot_iq4xs_q8k_emul(wrow, acol, nb),
        GgmlType::Q2_0 => dot_q2f0_q8f0_scalar(wrow, acol, nb),
        // Every caller passed check_row, which refused a type with neither a kernel nor an entry.
        other => generic::entry(other)
            .expect("check_row named a type with neither a hand kernel nor an entry")
            .dot_scalar(wrow, acol, nb),
    }
}

/// Calls the AVX2 kernel of `w` directly. [`dot_row`] runs the same kernel,
/// except for Q4_K when [`q4k_avx512`] is true: then it runs the AVX-512 VNNI
/// body. Panics if required features are missing.
#[cfg(target_arch = "x86_64")]
pub fn dot_row_avx2(w: GgmlType, wrow: &[u8], acol: &[u8], k: usize) -> Result<f32, QdotError> {
    let nb = check_row(w, wrow.len(), acol.len(), k)?;
    assert!(
        has_features(w),
        "dot_row_avx2 called on a CPU without the kernel's ISA \
         (avx2+f16c for Q3_K, avx2+fma for MXFP4, NVFP4 and NVFP4 rows, avx2+fma+f16c for the rest)"
    );
    match w {
        GgmlType::Q4_K => {
            // SAFETY: asserted just above; lengths validated by check_row.
            Ok(unsafe { dot_q4k_q82x4_avx2(wrow, acol, nb) })
        }
        GgmlType::Q5_K => {
            // SAFETY: asserted just above; lengths validated by check_row.
            Ok(unsafe { dot_q5k_q82x4_avx2(wrow, acol, nb) })
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
        GgmlType::Q8_0 => {
            // SAFETY: asserted just above; lengths validated by check_row.
            Ok(unsafe { dot_q8f0_q82x4_avx2(wrow, acol, nb) })
        }
        GgmlType::Q2_K => {
            // SAFETY: asserted just above; lengths validated by check_row.
            Ok(unsafe { dot_q2k_q8k_avx2(wrow, acol, nb) })
        }
        GgmlType::IQ2_XS => {
            // SAFETY: asserted just above; lengths validated by check_row.
            Ok(unsafe { dot_iq2xs_q8k_avx2(wrow, acol, nb) })
        }
        GgmlType::IQ2_XXS => {
            // SAFETY: asserted just above; lengths validated by check_row.
            Ok(unsafe { dot_iq2xxs_q8k_avx2(wrow, acol, nb) })
        }
        GgmlType::IQ1_S => {
            // SAFETY: asserted just above; lengths validated by check_row.
            Ok(unsafe { dot_iq1s_q8k_avx2(wrow, acol, nb) })
        }
        GgmlType::IQ3_XXS => {
            // SAFETY: asserted just above; lengths validated by check_row.
            Ok(unsafe { dot_iq3xxs_q8k_avx2(wrow, acol, nb) })
        }
        GgmlType::IQ3_S => {
            // SAFETY: asserted just above; lengths validated by check_row.
            Ok(unsafe { dot_iq3s_q8k_avx2(wrow, acol, nb) })
        }
        GgmlType::MXFP4 => {
            // SAFETY: asserted just above; lengths validated by check_row.
            Ok(unsafe { dot_mxfp4_q82x4_avx2(wrow, acol, nb) })
        }
        GgmlType::NVFP4 => {
            // SAFETY: asserted just above; lengths validated by check_row.
            Ok(
                unsafe {
                    dot_nvfp4_q82x4_avx2::<NV_BLOCKS>(wrow.as_ptr(), wrow.as_ptr(), acol, nb)
                },
            )
        }
        GgmlType::NVFP4_ROWS => {
            // SAFETY: asserted just above; lengths validated by check_row.
            Ok(unsafe { dot_nvfp4_q82x4_avx2::<NV_ROWS>(wrow.as_ptr(), wrow.as_ptr(), acol, nb) })
        }
        GgmlType::IQ4_NL => {
            // SAFETY: asserted just above; lengths validated by check_row.
            Ok(unsafe { dot_iq4nl_q82x4_avx2(wrow, acol, nb) })
        }
        GgmlType::IQ4_XS => {
            // SAFETY: asserted just above; lengths validated by check_row.
            Ok(unsafe { dot_iq4xs_q8k_avx2(wrow, acol, nb) })
        }
        GgmlType::Q3_K => {
            // SAFETY: asserted just above; lengths validated by check_row.
            Ok(unsafe { dot_q3k_q8k_avx2(wrow, acol, nb) })
        }
        GgmlType::Q2_0 => {
            // SAFETY: asserted just above; lengths validated by check_row.
            Ok(unsafe { dot_q2f0_q8f0_avx2(wrow, acol, nb) })
        }
        // check_row refused a type with neither a kernel nor an entry.
        other => {
            let e = generic::entry(other)
                .expect("check_row named a type with neither a hand kernel nor an entry");
            // SAFETY: asserted just above; lengths validated by check_row.
            Ok(unsafe { e.dot_avx2(wrow, acol, nb) })
        }
    }
}

/// The most activation columns one [`dot_row_cols`] call takes: the tile
/// width of the multi-column kernels.
pub const TILE_COLS: usize = 8;

/// The weight types with a multi-column tile kernel, each tied to the kernel
/// it runs.
#[derive(Clone, Copy)]
enum TileKind {
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q51,
    Q80,
    Iq2Xs,
    Iq2Xxs,
    Iq1s,
    Iq3Xxs,
    Iq3S,
    Iq4Xs,
    Iq4Nl,
    Mxfp4,
}

/// The tile kernel `w` takes on this machine: a type with one and the ISA its
/// kernel needs (the one-column kernel's, [`has_features`]).
fn tile_kind(w: GgmlType) -> Option<TileKind> {
    let kind = match w {
        GgmlType::Q2_K => TileKind::Q2K,
        GgmlType::Q3_K => TileKind::Q3K,
        GgmlType::Q4_K => TileKind::Q4K,
        GgmlType::Q5_K => TileKind::Q5K,
        GgmlType::Q6_K => TileKind::Q6K,
        GgmlType::Q5_1 => TileKind::Q51,
        GgmlType::Q8_0 => TileKind::Q80,
        GgmlType::IQ2_XS => TileKind::Iq2Xs,
        GgmlType::IQ2_XXS => TileKind::Iq2Xxs,
        GgmlType::IQ1_S => TileKind::Iq1s,
        GgmlType::IQ3_XXS => TileKind::Iq3Xxs,
        GgmlType::IQ3_S => TileKind::Iq3S,
        GgmlType::IQ4_XS => TileKind::Iq4Xs,
        GgmlType::IQ4_NL => TileKind::Iq4Nl,
        GgmlType::MXFP4 => TileKind::Mxfp4,
        _ => return None,
    };
    has_features(w).then_some(kind)
}

/// Whether [`dot_row_cols`] runs a tile kernel for `w` on this machine
/// (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q5_1, Q8_0, IQ2_XS, IQ2_XXS, IQ1_S, IQ3_XXS, IQ3_S, IQ4_XS, IQ4_NL and MXFP4 with their ISA); otherwise it dots each column
/// through [`dot_row`].
#[must_use]
pub fn has_tile(w: GgmlType) -> bool {
    tile_kind(w).is_some()
}

/// One weight row against up to [`TILE_COLS`] quantized activation columns:
/// `out[j]` is `dot_row(w, wrow, acols[j], k)`, bit for bit.
///
/// With a tile kernel ([`has_tile`]) and two or more columns, each block of
/// the row is unpacked once and dotted with every column; each column keeps
/// the one-column kernel's integer sums and its float accumulation order,
/// so only the loop over columns moves inside the block loop. The Q3_K tile
/// reads a q8_K block's 16 code sums (bytes 264..296) for its −4 fold, so a
/// column must come from [`quantize_col`], which writes them. One column, or
/// a type or CPU without a tile kernel, dots each column through
/// [`dot_row`] — the scalar mirror on a CPU without AVX2. A column count
/// outside `1..=TILE_COLS` or an `out` of another length is
/// [`QdotError::TileShape`]; every column is checked as [`dot_row`] checks it.
pub fn dot_row_cols(
    w: GgmlType,
    wrow: &[u8],
    acols: &[&[u8]],
    k: usize,
    out: &mut [f32],
) -> Result<(), QdotError> {
    let c = acols.len();
    if c == 0 || c > TILE_COLS || out.len() != c {
        return Err(QdotError::TileShape {
            cols: c,
            outs: out.len(),
        });
    }
    let mut nb = 0;
    for a in acols {
        nb = check_row(w, wrow.len(), a.len(), k)?;
    }
    let kind = match tile_kind(w) {
        Some(kind) if c > 1 => kind,
        _ => {
            for (o, a) in out.iter_mut().zip(acols) {
                *o = dot_row(w, wrow, a, k)?;
            }
            return Ok(());
        }
    };
    // The column count picks `C` for either tile.
    macro_rules! by_cols {
        ($tile:ident $(, $kind:expr)?) => {
            match c {
                2 => $tile::<2>($($kind,)? wrow, acols, nb, out),
                3 => $tile::<3>($($kind,)? wrow, acols, nb, out),
                4 => $tile::<4>($($kind,)? wrow, acols, nb, out),
                5 => $tile::<5>($($kind,)? wrow, acols, nb, out),
                6 => $tile::<6>($($kind,)? wrow, acols, nb, out),
                7 => $tile::<7>($($kind,)? wrow, acols, nb, out),
                _ => $tile::<8>($($kind,)? wrow, acols, nb, out),
            }
        };
    }
    if matches!(kind, TileKind::Q4K) && q4k_avx512() {
        by_cols!(tile_q4k_avx512);
        return Ok(());
    }
    by_cols!(tile, kind);
    Ok(())
}

/// The `C`-column tile of `kind` over validated inputs: `acols` holds `C`
/// columns of at least the activation bytes `nb` blocks need, `wrow` the
/// row's `nb` blocks, `out` `C` values.
#[cfg(target_arch = "x86_64")]
fn tile<const C: usize>(kind: TileKind, wrow: &[u8], acols: &[&[u8]], nb: usize, out: &mut [f32]) {
    let cols: [*const u8; C] = std::array::from_fn(|j| acols[j].as_ptr());
    let v = match kind {
        // SAFETY: tile_kind saw AVX2+F16C; check_row sized the row and every
        // column for nb super-blocks.
        TileKind::Q3K => unsafe { dot_q3k_q8k_tile_avx2::<C>(wrow, &cols, nb) },
        // SAFETY: tile_kind saw AVX2+FMA+F16C; check_row sized the row and
        // every column for nb blocks.
        TileKind::Q4K => unsafe { dot_q45k_q82x4_tile_avx2::<C, false>(wrow, &cols, nb) },
        // SAFETY: as the Q4_K arm.
        TileKind::Q5K => unsafe { dot_q45k_q82x4_tile_avx2::<C, true>(wrow, &cols, nb) },
        // SAFETY: as the Q4_K arm.
        TileKind::Q6K => unsafe { dot_q6k_q82x4_tile_avx2::<C>(wrow, &cols, nb) },
        // SAFETY: tile_kind saw AVX2+FMA+F16C; check_row sized the row for nb
        // blocks and every column for nb blocks' x4 groups and q8_2 tails.
        TileKind::Q51 => unsafe { dot_q5f1_q82x4_tile_avx2::<C>(wrow, &cols, nb) },
        // SAFETY: as the Q5_1 arm.
        TileKind::Q80 => unsafe { dot_q8f0_q82x4_tile_avx2::<C>(wrow, &cols, nb) },
        // SAFETY: tile_kind saw AVX2+FMA+F16C; check_row sized the row for nb
        // blocks and every column for nb Q8_K blocks.
        TileKind::Iq3Xxs => unsafe { dot_iq3xxs_q8k_tile_avx2::<C>(wrow, &cols, nb) },
        // SAFETY: as the IQ3_XXS arm.
        TileKind::Q2K => unsafe { dot_q2k_q8k_tile_avx2::<C>(wrow, &cols, nb) },
        // SAFETY: as the IQ3_XXS arm.
        TileKind::Iq2Xs => unsafe { dot_iq2xs_q8k_tile_avx2::<C>(wrow, &cols, nb) },
        // SAFETY: as the IQ3_XXS arm.
        TileKind::Iq2Xxs => unsafe { dot_iq2xxs_q8k_tile_avx2::<C>(wrow, &cols, nb) },
        // SAFETY: as the IQ3_XXS arm.
        TileKind::Iq1s => unsafe { dot_iq1s_q8k_tile_avx2::<C>(wrow, &cols, nb) },
        // SAFETY: as the IQ3_XXS arm.
        TileKind::Iq3S => unsafe { dot_iq3s_q8k_tile_avx2::<C>(wrow, &cols, nb) },
        // SAFETY: as the IQ3_XXS arm.
        TileKind::Iq4Xs => unsafe { dot_iq4xs_q8k_tile_avx2::<C>(wrow, &cols, nb) },
        // SAFETY: as the Q5_1 arm.
        TileKind::Iq4Nl => unsafe { dot_iq4nl_q82x4_tile_avx2::<C>(wrow, &cols, nb) },
        // SAFETY: tile_kind saw AVX2+FMA; check_row sized the row for nb blocks
        // and every column for nb blocks' x4 groups and q8_2 tails.
        TileKind::Mxfp4 => unsafe { dot_mxfp4_q82x4_tile_avx2::<C>(wrow, &cols, nb) },
    };
    out.copy_from_slice(&v);
}

/// [`tile`] for Q4_K on its AVX-512 VNNI body ([`q4k_avx512`]).
#[cfg(target_arch = "x86_64")]
fn tile_q4k_avx512<const C: usize>(wrow: &[u8], acols: &[&[u8]], nb: usize, out: &mut [f32]) {
    let cols: [*const u8; C] = std::array::from_fn(|j| acols[j].as_ptr());
    // SAFETY: q4k_avx512 saw AVX-512 F+BW+VNNI and AVX2+FMA+F16C; check_row
    // sized the row and every column for nb super-blocks.
    let v = unsafe { dot_q4k_q82x4_tile_avx512::<C>(wrow, &cols, nb) };
    out.copy_from_slice(&v);
}

/// Off x86_64 [`q4k_avx512`] is false, so no call reaches it.
#[cfg(not(target_arch = "x86_64"))]
fn tile_q4k_avx512<const C: usize>(_: &[u8], _: &[&[u8]], _: usize, _: &mut [f32]) {
    unreachable!("no qdot tile kernel off x86_64")
}

/// Off x86_64 [`tile_kind`] is `None` for every type, so no call reaches a tile.
#[cfg(not(target_arch = "x86_64"))]
fn tile<const C: usize>(_: TileKind, _: &[u8], _: &[&[u8]], _: usize, _: &mut [f32]) {
    unreachable!("no qdot tile kernel off x86_64")
}

/// The most activation columns one [`dot_row_cols_wide`] call takes.
pub const WIDE_COLS: usize = 32;

/// Columns one group of the wide kernel dots at once: their sums, one a
/// column, stay in registers beside a sub-block's values and scale vectors.
#[cfg(target_arch = "x86_64")]
const WIDE_GROUP: usize = 8;

/// Whether [`dot_row_cols_wide`] runs a wide kernel for `w` on this machine
/// (IQ2_XXS with its one-column kernel's ISA); otherwise it runs the columns
/// through [`dot_row_cols`] in runs of [`TILE_COLS`].
#[must_use]
pub fn has_wide(w: GgmlType) -> bool {
    matches!(w, GgmlType::IQ2_XXS) && has_features(w)
}

/// One weight row against up to [`WIDE_COLS`] quantized activation columns:
/// `out[j]` is `dot_row(w, wrow, acols[j], k)`, bit for bit.
///
/// With a wide kernel ([`has_wide`]) each block of the row is unpacked once
/// for every column of the call, where [`dot_row_cols`] unpacks it once a run
/// of [`TILE_COLS`]; each column keeps the one-column kernel's integer lanes
/// and its float accumulation order. Without one, the columns run through
/// [`dot_row_cols`] in runs of [`TILE_COLS`]. A column count outside
/// `1..=WIDE_COLS` or an `out` of another length is
/// [`QdotError::WideShape`]; every column is checked as [`dot_row`] checks it.
pub fn dot_row_cols_wide(
    w: GgmlType,
    wrow: &[u8],
    acols: &[&[u8]],
    k: usize,
    out: &mut [f32],
) -> Result<(), QdotError> {
    let c = acols.len();
    if c == 0 || c > WIDE_COLS || out.len() != c {
        return Err(QdotError::WideShape {
            cols: c,
            outs: out.len(),
        });
    }
    let mut nb = 0;
    for a in acols {
        nb = check_row(w, wrow.len(), a.len(), k)?;
    }
    if !has_wide(w) {
        for (a, o) in acols.chunks(TILE_COLS).zip(out.chunks_mut(TILE_COLS)) {
            dot_row_cols(w, wrow, a, k, o)?;
        }
        return Ok(());
    }
    wide_iq2xxs(wrow, acols, nb, out);
    Ok(())
}

/// The IQ2_XXS wide kernel over validated inputs: `acols` holds
/// `1..=WIDE_COLS` columns of at least the Q8_K bytes `nb` blocks need,
/// `wrow` the row's `nb` blocks, `out` one value a column.
#[cfg(target_arch = "x86_64")]
fn wide_iq2xxs(wrow: &[u8], acols: &[&[u8]], nb: usize, out: &mut [f32]) {
    let mut cols = [std::ptr::null::<u8>(); WIDE_COLS];
    for (p, a) in cols.iter_mut().zip(acols) {
        *p = a.as_ptr();
    }
    // SAFETY: has_wide saw AVX2+FMA+F16C; check_row sized the row for nb blocks
    // and every column for nb Q8_K blocks; dot_row_cols_wide's shape check holds
    // the columns to 1..=WIDE_COLS and `out` to one value a column.
    unsafe { dot_iq2xxs_q8k_wide_avx2(wrow, &cols[..acols.len()], nb, out) };
}

/// Off x86_64 [`has_wide`] is false for every type, so no call reaches it.
#[cfg(not(target_arch = "x86_64"))]
fn wide_iq2xxs(_: &[u8], _: &[&[u8]], _: usize, _: &mut [f32]) {
    unreachable!("no qdot wide kernel off x86_64")
}

/// Rows one group of the row-lane Q3_K layout interleaves ([`repack_q3k_r8`]).
pub const Q3K_R8_ROWS: usize = 8;

/// The version of [`repack_q3k_r8`]'s byte order that a file of row-lane
/// groups stores: a reader refuses any other, and a change to the order is a
/// new version.
pub const Q3K_R8_LAYOUT: u32 = 1;

/// Q3_K rows into the row-lane layout [`dot_q3k_r8_cols`] reads: `n_rows`
/// rows of `k` values (`rows`, row after row) into `out`, which is exactly as
/// long — the layout holds the same bits, permuted.
///
/// Rows go in groups of [`Q3K_R8_ROWS`], each group's super-blocks in order,
/// 880 bytes per group super-block:
/// - bytes 0..16: row `r`'s f16 `d` at `2r`;
/// - bytes 16..112: the 128 six-bit scales (`s + 32`) as four 32-byte vectors
///   `X_q` — byte `b` of `X_q` is the scale of sub-block `2p + b % 2` of row
///   `(b % 16) / 2`, `p = 2q + b / 16` — stored as the low nibbles
///   `X_0 | X_1 << 4`, then `X_2 | X_3 << 4`, then the high two bits
///   `X_q >> 4` at bits `2q`;
/// - bytes 112..880: eight sub-block pairs `p` of 96 bytes. Code vector `W_f`
///   (`f` in 0..8: sub-block `2p + f / 4`, its values `4 (f % 4)` onward)
///   holds at byte `b` the code `u = value + 4` (0..7) of row `b / 4`, value
///   `b % 4`; the pair stores `A = W0 | W1 << 3 | (W2 & 3) << 6`,
///   `B = W3 | W4 << 3 | (W5 & 3) << 6` and
///   `C = W6 | W7 << 3 | (W2 >> 2) << 6 | (W5 >> 2) << 7`.
///
/// A `k` off the 256-value grid, a row count off the group grid
/// ([`QdotError::RowGroup`]; rows are not padded) and a `rows` or `out` of
/// another length ([`QdotError::RowGroupBytes`]) are named errors, and none
/// writes to `out`. `k = 0` is an empty product: `Ok` with both buffers
/// empty, nothing written.
pub fn repack_q3k_r8(
    rows: &[u8],
    n_rows: usize,
    k: usize,
    out: &mut [u8],
) -> Result<(), QdotError> {
    if !k.is_multiple_of(256) {
        return Err(QdotError::UnalignedK { k, gran: 256 });
    }
    if !n_rows.is_multiple_of(Q3K_R8_ROWS) {
        return Err(QdotError::RowGroup { rows: n_rows });
    }
    let nb = k / 256;
    let row_bytes = nb * Q3K_BLOCK;
    let need = n_rows * row_bytes;
    for (buf, have) in [
        ("repack source", rows.len()),
        ("repack destination", out.len()),
    ] {
        if have != need {
            return Err(QdotError::RowGroupBytes { buf, have, need });
        }
    }
    // k = 0: both buffers are empty (need = 0), and `chunks_exact` takes no
    // zero-byte chunk.
    if k == 0 {
        return Ok(());
    }
    let group_bytes = Q3K_R8_ROWS * row_bytes;
    for (src, dst) in rows
        .chunks_exact(group_bytes)
        .zip(out.chunks_exact_mut(group_bytes))
    {
        for (sb, blk) in dst.as_chunks_mut::<Q3K_R8_BLOCK>().0.iter_mut().enumerate() {
            let blocks: [&[u8]; Q3K_R8_ROWS] = std::array::from_fn(|r| {
                let at = r * row_bytes + sb * Q3K_BLOCK;
                &src[at..at + Q3K_BLOCK]
            });
            r8_pack_block(&blocks, blk);
        }
    }
    Ok(())
}

/// [`repack_q3k_r8`]'s exact inverse: `n_rows` rows of `k` values in the
/// row-lane layout (`groups`) back to Q3_K rows, row after row, in `out`,
/// which is exactly as long. Both layouts use every bit once (the layout's
/// doc lists where each goes), so the pair is a bijection on each 880-byte
/// group super-block: any bytes unpack, and `repack` of the result gives
/// them back.
///
/// The error contract is `repack_q3k_r8`'s: a `k` off the 256-value grid, a
/// row count off the group grid ([`QdotError::RowGroup`]) and a `groups` or
/// `out` of another length ([`QdotError::RowGroupBytes`]) are named errors,
/// and none writes to `out`. `k = 0` is `Ok` with both buffers empty.
pub fn unpack_q3k_r8(
    groups: &[u8],
    n_rows: usize,
    k: usize,
    out: &mut [u8],
) -> Result<(), QdotError> {
    if !k.is_multiple_of(256) {
        return Err(QdotError::UnalignedK { k, gran: 256 });
    }
    if !n_rows.is_multiple_of(Q3K_R8_ROWS) {
        return Err(QdotError::RowGroup { rows: n_rows });
    }
    let nb = k / 256;
    let row_bytes = nb * Q3K_BLOCK;
    let need = n_rows * row_bytes;
    for (buf, have) in [
        ("unpack source", groups.len()),
        ("unpack destination", out.len()),
    ] {
        if have != need {
            return Err(QdotError::RowGroupBytes { buf, have, need });
        }
    }
    // k = 0: both buffers are empty (need = 0), and `chunks_exact` takes no
    // zero-byte chunk.
    if k == 0 {
        return Ok(());
    }
    let group_bytes = Q3K_R8_ROWS * row_bytes;
    for (src, dst) in groups
        .chunks_exact(group_bytes)
        .zip(out.chunks_exact_mut(group_bytes))
    {
        for (sb, blk) in src.as_chunks::<Q3K_R8_BLOCK>().0.iter().enumerate() {
            for r in 0..Q3K_R8_ROWS {
                let at = r * row_bytes + sb * Q3K_BLOCK;
                r8_unpack_row(blk, r, &mut dst[at..at + Q3K_BLOCK]);
            }
        }
    }
    Ok(())
}

/// NVFP4 rows' code bytes (a checkpoint folder's code plane, [`GgmlType::NVFP4_ROWS`]: value `2i`
/// of a 16-value group the low nibble of the group's byte `i`, value `2i + 1` its high nibble) in
/// the order an NVFP4 block holds them (ggml's `block_nvfp4`, the card's code plane: the group's
/// byte `j` holds value `j` low and value `j + 8` high), into `dst`, exactly as long: group `g`
/// stays at bytes `8g..8g + 8`, its nibbles moved and none changed. AVX2 where the machine has
/// it, else the scalar rule ([`nvfp4_codes_to_blocks_scalar`]), bit for bit the same.
///
/// A `src` of no whole groups ([`QdotError::UnalignedK`], `k` its values) and a `dst` of another
/// length ([`QdotError::RowGroupBytes`]) are named errors, and neither writes to `dst`.
pub fn nvfp4_codes_to_blocks(src: &[u8], dst: &mut [u8]) -> Result<(), QdotError> {
    nvfp4_codes_check(src, dst)?;
    if !nvfp4_codes_fast(src, dst) {
        nvfp4_codes_groups(src, dst);
    }
    Ok(())
}

/// [`nvfp4_codes_to_blocks`]' scalar rule, its mirror for fallback and verification.
pub fn nvfp4_codes_to_blocks_scalar(src: &[u8], dst: &mut [u8]) -> Result<(), QdotError> {
    nvfp4_codes_check(src, dst)?;
    nvfp4_codes_groups(src, dst);
    Ok(())
}

/// [`nvfp4_codes_to_blocks`]' refusals.
fn nvfp4_codes_check(src: &[u8], dst: &[u8]) -> Result<(), QdotError> {
    if !src.len().is_multiple_of(8) {
        return Err(QdotError::UnalignedK {
            k: 2 * src.len(),
            gran: 16,
        });
    }
    if dst.len() != src.len() {
        return Err(QdotError::RowGroupBytes {
            buf: "nvfp4 codes destination",
            have: dst.len(),
            need: src.len(),
        });
    }
    Ok(())
}

/// The scalar rule over whole groups of equal-length `src` and `dst`: byte `j` of a group's
/// output is value `j` (nibble `j % 2` of the group's byte `j / 2`) low and value `j + 8` (nibble
/// `j % 2` of its byte `4 + j / 2`) high.
fn nvfp4_codes_groups(src: &[u8], dst: &mut [u8]) {
    for (g, o) in src
        .as_chunks::<8>()
        .0
        .iter()
        .zip(dst.as_chunks_mut::<8>().0)
    {
        *o = gguf::quant::nvfp4_group_to_block(g);
    }
}

/// [`nvfp4_codes_avx2`] when the machine has AVX2: whether it ran.
#[cfg(target_arch = "x86_64")]
fn nvfp4_codes_fast(src: &[u8], dst: &mut [u8]) -> bool {
    if !std::arch::is_x86_feature_detected!("avx2") {
        return false;
    }
    // SAFETY: AVX2 is present (checked above); the caller checked the lengths equal and whole
    // groups.
    unsafe { nvfp4_codes_avx2(src, dst) };
    true
}

/// Off x86_64 no AVX2 body is compiled: the scalar rule runs.
#[cfg(not(target_arch = "x86_64"))]
fn nvfp4_codes_fast(_: &[u8], _: &mut [u8]) -> bool {
    false
}

/// [`nvfp4_codes_groups`] four groups (32 bytes) a step, the tail by the scalar rule. Per 128-bit
/// lane (two groups): the bytes' low and high nibbles interleaved give each group's 16 values one a
/// byte in value order, and value `j` low with value `j + 8` high is the vector or'd with itself
/// eight bytes down, shifted up a nibble.
///
/// # Safety
/// AVX2 must be available; `src` and `dst` are equally long, a multiple of 8.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn nvfp4_codes_avx2(src: &[u8], dst: &mut [u8]) {
    let n = src.len() / 32 * 32;
    // SAFETY: AVX2 present per contract; every load and store covers bytes [at, at + 32) with
    // at + 32 <= n <= len of both slices, unaligned.
    unsafe {
        let lo = _mm256_set1_epi8(0x0f);
        let mut at = 0;
        while at < n {
            let x = _mm256_loadu_si256(src.as_ptr().add(at) as *const __m256i);
            let l = _mm256_and_si256(x, lo);
            let h = _mm256_and_si256(_mm256_srli_epi16::<4>(x), lo);
            // Each lane's first group's values (unpacklo) and its second's (unpackhi).
            let v0 = _mm256_unpacklo_epi8(l, h);
            let v1 = _mm256_unpackhi_epi8(l, h);
            // Every value is below 16, so the 16-bit shift moves no bit across a byte.
            let s0 = _mm256_or_si256(v0, _mm256_slli_epi16::<4>(_mm256_bsrli_epi128::<8>(v0)));
            let s1 = _mm256_or_si256(v1, _mm256_slli_epi16::<4>(_mm256_bsrli_epi128::<8>(v1)));
            // A lane's two groups back in order: the first's eight bytes, then the second's.
            let out = _mm256_unpacklo_epi64(s0, s1);
            _mm256_storeu_si256(dst.as_mut_ptr().add(at) as *mut __m256i, out);
            at += 32;
        }
    }
    nvfp4_codes_groups(&src[n..], &mut dst[n..]);
}

/// One Q3_K 8-row group of [`repack_q3k_r8`]'s layout against up to
/// [`TILE_COLS`] Q8_K columns: `out[c][r]` is `dot_row(Q3_K, row r,
/// acols[c], k)` bit for bit.
///
/// `group` is one group's `k / 256` super-blocks, exactly. Every (row,
/// column) keeps the one-column kernel's integer sum per super-block and its
/// float step, so only the order in which the values are formed moves. As
/// the Q3_K tile of [`dot_row_cols`], the kernel reads a q8_K block's 16 code
/// sums (bytes 264..296) for its −4 fold, so a column must come from
/// [`quantize_col`]. A CPU without AVX2+F16C runs the scalar mirror
/// ([`dot_q3k_r8_cols_scalar`]). A column count outside `1..=TILE_COLS` or
/// an `out` of another length ([`QdotError::TileShape`]), a `k` off the
/// 256-value grid, a group of another length ([`QdotError::RowGroupBytes`])
/// and a short column are named errors.
pub fn dot_q3k_r8_cols(
    group: &[u8],
    acols: &[&[u8]],
    k: usize,
    out: &mut [[f32; Q3K_R8_ROWS]],
) -> Result<(), QdotError> {
    let nb = check_r8(group.len(), acols, k, out.len())?;
    if !has_features(GgmlType::Q3_K) {
        r8_scalar(group, acols, nb, out);
        return Ok(());
    }
    match acols.len() {
        1 => r8_tile::<1>(group, acols, nb, out),
        2 => r8_tile::<2>(group, acols, nb, out),
        3 => r8_tile::<3>(group, acols, nb, out),
        4 => r8_tile::<4>(group, acols, nb, out),
        5 => r8_tile::<5>(group, acols, nb, out),
        6 => r8_tile::<6>(group, acols, nb, out),
        7 => r8_tile::<7>(group, acols, nb, out),
        _ => r8_tile::<8>(group, acols, nb, out),
    }
    Ok(())
}

/// The scalar mirror behind [`dot_q3k_r8_cols`], read straight off the
/// repacked layout: the fallback without AVX2+F16C, and the gate's second
/// path.
pub fn dot_q3k_r8_cols_scalar(
    group: &[u8],
    acols: &[&[u8]],
    k: usize,
    out: &mut [[f32; Q3K_R8_ROWS]],
) -> Result<(), QdotError> {
    let nb = check_r8(group.len(), acols, k, out.len())?;
    r8_scalar(group, acols, nb, out);
    Ok(())
}

/// The row-lane tile's shape contract; the super-block count on success.
fn check_r8(group_len: usize, acols: &[&[u8]], k: usize, outs: usize) -> Result<usize, QdotError> {
    let c = acols.len();
    if c == 0 || c > TILE_COLS || outs != c {
        return Err(QdotError::TileShape { cols: c, outs });
    }
    if !k.is_multiple_of(256) {
        return Err(QdotError::UnalignedK { k, gran: 256 });
    }
    let nb = k / 256;
    let need = nb * Q3K_R8_BLOCK;
    if group_len != need {
        return Err(QdotError::RowGroupBytes {
            buf: "tile group",
            have: group_len,
            need,
        });
    }
    let need_a = nb * Q8K_STRIDE;
    if let Some(a) = acols.iter().find(|a| a.len() < need_a) {
        return Err(QdotError::ShortActivationCol {
            have: a.len(),
            need: need_a,
            k,
        });
    }
    Ok(nb)
}

/// The `C`-column row-lane tile over validated inputs: `group` holds `nb`
/// group super-blocks, `acols` `C` columns of at least `nb` Q8_K blocks, `out`
/// `C` values.
#[cfg(target_arch = "x86_64")]
fn r8_tile<const C: usize>(
    group: &[u8],
    acols: &[&[u8]],
    nb: usize,
    out: &mut [[f32; Q3K_R8_ROWS]],
) {
    let cols: [*const u8; C] = std::array::from_fn(|j| acols[j].as_ptr());
    // SAFETY: the caller saw AVX2+F16C (has_features(Q3_K)); check_r8 sized
    // the group for nb super-blocks and every column for nb Q8_K blocks.
    let v = unsafe { dot_q3k_r8_tile_avx2::<C>(group, &cols, nb) };
    out.copy_from_slice(&v);
}

/// Off x86_64 [`dot_q3k_r8_cols`] takes the scalar mirror for every call.
#[cfg(not(target_arch = "x86_64"))]
fn r8_tile<const C: usize>(_: &[u8], _: &[&[u8]], _: usize, _: &mut [[f32; Q3K_R8_ROWS]]) {
    unreachable!("no qdot row-lane Q3_K kernel off x86_64")
}

/// Rows one row-lane group holds ([`pack_lanes`]).
pub const LANE_ROWS: usize = 8;

/// Columns one row-lane kernel call runs at once: its accumulators, one a
/// column, stay in registers beside a sub-block's four code vectors.
/// [`dot_lanes_cols`] takes up to [`TILE_COLS`] and runs them in calls of
/// this many.
const LANE_COLS: usize = 4;

/// Bytes of one packed super-block of a row-lane group: 64 code vectors of
/// 32 bytes, then sixteen vectors of eight f32 scale products — Q4_K and
/// Q5_K's eight scales and eight mins, Q6_K's sixteen scales.
const LANE_SB_BYTES: usize = LANE_SC + 16 * 32;
/// Where a packed super-block's scale vectors start.
const LANE_SC: usize = 64 * 32;
/// Where a packed Q4_K or Q5_K super-block's min vectors start.
#[cfg(target_arch = "x86_64")]
const LANE_MN: usize = LANE_SC + 8 * 32;

/// Whether `w` runs row lanes on this machine ([`pack_lanes`],
/// [`dot_lanes_cols`]): Q4_K, Q5_K and Q6_K with their one-column kernel's
/// ISA.
#[must_use]
pub fn has_lanes(w: GgmlType) -> bool {
    matches!(w, GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q6_K) && has_features(w)
}

/// Bytes [`pack_lanes`] writes for one group of `k`-value rows: `k / 256`
/// packed super-blocks.
#[must_use]
pub fn lane_pack_bytes(k: usize) -> usize {
    k / 256 * LANE_SB_BYTES
}

/// The shape checks every row-lane entry point opens with: `w` has lanes and
/// `k` is whole super-blocks; returns the super-block count.
fn check_lanes(w: GgmlType, k: usize) -> Result<usize, QdotError> {
    if !has_lanes(w) {
        return Err(QdotError::NoLanes(w));
    }
    if !k.is_multiple_of(256) {
        return Err(QdotError::UnalignedK { k, gran: 256 });
    }
    Ok(k / 256)
}

/// [`LANE_ROWS`] rows of `k` values of `w` (`rows`, row after row, each its
/// type's row bytes) unpacked into one row-lane group in `packed`, which is
/// exactly [`lane_pack_bytes`] long — the layout [`dot_lanes_cols`] reads.
///
/// `next` is empty, or the next group's rows, `rows`' length: then each
/// super-block's pass also prefetches the next `1 / (k / 256)` of `next`'s
/// bytes toward L2, so the next group is in flight by the time this one is
/// packed and lands while its columns are dotted. A prefetch moves no value:
/// `packed` is the same bytes either way.
///
/// Per super-block `i`, 2,560 bytes at `2,560 i`. Q4_K and Q5_K: code vector
/// `(s, g)` at `32 (8 s + g)` holds at byte `b` row `b / 4`'s code (0..15;
/// Q5_K with its high bit, 0..31) of sub-block `s`, value `4 g + b % 4`; then
/// at `2,048 + 32 s` lane `r` is row `r`'s `d · sc_s`, and at `2,304 + 32 s`
/// row `r`'s `−dmin · m_s`. Q6_K: code vector `(s, g)` at `32 (4 s + g)` holds
/// at byte `b` row `b / 4`'s six-bit code (0..63, its `− 32` left to the dot)
/// of sixteen-value sub-block `s`, value `4 g + b % 4`; then at `2,048 + 32 s`
/// lane `r` is row `r`'s `d · scales_s`. Every scale product is the f32 product
/// the one-column kernel forms, one rounding each.
///
/// A type without row lanes ([`QdotError::NoLanes`]), a `k` off the
/// 256-value grid, a `rows` or `packed` of another length and a non-empty
/// `next` of another length than `rows` ([`QdotError::RowGroupBytes`]) are
/// named errors, and none writes to `packed`. `k = 0` is `Ok` with every
/// buffer empty.
pub fn pack_lanes(
    w: GgmlType,
    rows: &[u8],
    next: &[u8],
    k: usize,
    packed: &mut [u8],
) -> Result<(), QdotError> {
    let nb = check_lanes(w, k)?;
    let block = match w {
        GgmlType::Q4_K => Q4K_BLOCK,
        GgmlType::Q5_K => Q5K_BLOCK,
        _ => Q6K_BLOCK,
    };
    let group = LANE_ROWS * nb * block;
    for (buf, have, need) in [
        ("lane rows", rows.len(), group),
        (
            "lane next rows",
            next.len(),
            if next.is_empty() { 0 } else { group },
        ),
        ("lane group", packed.len(), lane_pack_bytes(k)),
    ] {
        if have != need {
            return Err(QdotError::RowGroupBytes { buf, have, need });
        }
    }
    if next.is_empty() {
        pack_group::<false>(w, rows, next, nb, packed);
    } else {
        pack_group::<true>(w, rows, next, nb, packed);
    }
    Ok(())
}

/// [`pack_lanes`] over checked buffers: `w` a row-lane type, `rows`
/// [`LANE_ROWS`] rows of `nb` blocks of it, `next` (with `AHEAD`) as many
/// bytes, `packed` `nb` packed super-blocks. One function an arm, so each
/// arm's three kernels are compiled alike: each the one call of its body.
#[cfg(target_arch = "x86_64")]
#[inline(never)]
fn pack_group<const AHEAD: bool>(
    w: GgmlType,
    rows: &[u8],
    next: &[u8],
    nb: usize,
    packed: &mut [u8],
) {
    // SAFETY: has_lanes saw AVX2+FMA+F16C; the buffers' sizes are the
    // caller's checks.
    unsafe {
        match w {
            GgmlType::Q4_K => pack_lanes_avx2::<false, AHEAD>(rows, next, nb, packed),
            GgmlType::Q5_K => pack_lanes_avx2::<true, AHEAD>(rows, next, nb, packed),
            _ => pack_lanes_q6k_avx2::<AHEAD>(rows, next, nb, packed),
        }
    }
}

/// Off x86_64 [`has_lanes`] is false for every type, so [`pack_lanes`] refuses first.
#[cfg(not(target_arch = "x86_64"))]
fn pack_group<const AHEAD: bool>(_: GgmlType, _: &[u8], _: &[u8], _: usize, _: &mut [u8]) {
    unreachable!("no qdot row-lane pack off x86_64")
}

/// One row-lane group ([`pack_lanes`]' `packed`) against up to [`TILE_COLS`]
/// columns of `k` values in `w`'s activation encoding: `out[c][r]` is
/// `dot_row(w, row r, acols[c], k)` bit for bit.
///
/// Every (row, column) keeps the one-column kernel's eight float lanes and
/// their order — Q4_K and Q5_K: lane `t + 4 h` takes, super-block by
/// super-block, the min term, then sub-block `t`'s and then sub-block
/// `4 + t`'s half `h` (16 values); Q6_K: lane `t + 4 h` takes sub-block
/// `2 t + h` and then `8 + 2 t + h` — the same integer sum, the same f32
/// products and the same fused adds in the same order, and the lanes are
/// summed as the one-column kernel sums them. Only where the integer sums are
/// formed moves: a ymm lane is a row, so a column's activation bytes are read
/// once for the group's eight rows.
///
/// A type without row lanes ([`QdotError::NoLanes`]), a column count outside
/// `1..=TILE_COLS` or an `out` of another length ([`QdotError::TileShape`]),
/// a `k` off the 256-value grid, a `packed` of another length
/// ([`QdotError::RowGroupBytes`]) and a short column are named errors.
pub fn dot_lanes_cols(
    w: GgmlType,
    packed: &[u8],
    acols: &[&[u8]],
    k: usize,
    out: &mut [[f32; LANE_ROWS]],
) -> Result<(), QdotError> {
    if !has_lanes(w) {
        return Err(QdotError::NoLanes(w));
    }
    let c = acols.len();
    if c == 0 || c > TILE_COLS || out.len() != c {
        return Err(QdotError::TileShape {
            cols: c,
            outs: out.len(),
        });
    }
    let nb = check_lanes(w, k)?;
    let need = lane_pack_bytes(k);
    if packed.len() != need {
        return Err(QdotError::RowGroupBytes {
            buf: "lane group",
            have: packed.len(),
            need,
        });
    }
    let need_a = col_bytes(w, k);
    if let Some(a) = acols.iter().find(|a| a.len() < need_a) {
        return Err(QdotError::ShortActivationCol {
            have: a.len(),
            need: need_a,
            k,
        });
    }
    let kind = match w {
        GgmlType::Q6_K => LaneKind::Q6K,
        GgmlType::Q4_K if q4k_avx512() => LaneKind::Q4KAvx512,
        _ => LaneKind::Q45K,
    };
    for (cols, out) in acols.chunks(LANE_COLS).zip(out.chunks_mut(LANE_COLS)) {
        match cols.len() {
            1 => lanes::<1>(kind, packed, cols, nb, out),
            2 => lanes::<2>(kind, packed, cols, nb, out),
            3 => lanes::<3>(kind, packed, cols, nb, out),
            _ => lanes::<4>(kind, packed, cols, nb, out),
        }
    }
    Ok(())
}

/// The row-lane kernel [`dot_lanes_cols`] runs for a packed group.
#[derive(Clone, Copy)]
enum LaneKind {
    Q45K,
    Q4KAvx512,
    Q6K,
}

/// The `C`-column row-lane kernel over validated inputs: `packed` holds `nb`
/// packed super-blocks of the type `kind` runs, `acols` `C` columns of at
/// least `nb` super-blocks' x4 groups, `out` `C` values.
#[cfg(target_arch = "x86_64")]
fn lanes<const C: usize>(
    kind: LaneKind,
    packed: &[u8],
    acols: &[&[u8]],
    nb: usize,
    out: &mut [[f32; LANE_ROWS]],
) {
    let cols: [*const u8; C] = std::array::from_fn(|j| acols[j].as_ptr());
    // SAFETY: the caller saw AVX2+FMA+F16C (has_lanes), and q4k_avx512 saw
    // AVX-512 F+BW+VNNI for LaneKind::Q4KAvx512; the caller sized `packed`
    // for nb packed super-blocks of the type and every column for nb
    // super-blocks.
    let v = unsafe {
        match kind {
            LaneKind::Q6K => dot_lanes_q6k_avx2::<C>(packed, &cols, nb),
            LaneKind::Q4KAvx512 => dot_lanes_avx512::<C>(packed, &cols, nb),
            LaneKind::Q45K => dot_lanes_avx2::<C>(packed, &cols, nb),
        }
    };
    out.copy_from_slice(&v);
}

/// Off x86_64 [`has_lanes`] is false for every type, so [`dot_lanes_cols`] refuses first.
#[cfg(not(target_arch = "x86_64"))]
fn lanes<const C: usize>(_: LaneKind, _: &[u8], _: &[&[u8]], _: usize, _: &mut [[f32; LANE_ROWS]]) {
    unreachable!("no qdot row-lane kernel off x86_64")
}

/// Shape validation shared by every `dot_row` entry point.
fn check_row(w: GgmlType, wrow_len: usize, acol_len: usize, k: usize) -> Result<usize, QdotError> {
    if !has_kernel(w) {
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
        GgmlType::Q8_0 => (k / 32, (k / 32) * Q8F0_BLOCK),
        GgmlType::Q4_K => (k / 256, (k / 256) * Q4K_BLOCK),
        GgmlType::Q5_K => (k / 256, (k / 256) * Q5K_BLOCK),
        GgmlType::Q2_K => (k / 256, (k / 256) * Q2K_BLOCK),
        GgmlType::Q3_K => (k / 256, (k / 256) * Q3K_BLOCK),
        GgmlType::IQ2_XS => (k / 256, (k / 256) * IQ2XS_BLOCK),
        GgmlType::IQ2_XXS => (k / 256, (k / 256) * IQ2XXS_BLOCK),
        GgmlType::IQ1_S => (k / 256, (k / 256) * IQ1S_BLOCK),
        GgmlType::IQ3_XXS => (k / 256, (k / 256) * IQ3XXS_BLOCK),
        GgmlType::IQ3_S => (k / 256, (k / 256) * IQ3S_BLOCK),
        GgmlType::MXFP4 => (k / 32, (k / 32) * MXFP4_BLOCK),
        GgmlType::NVFP4 => (k / 32, (k / 64) * NVFP4_BLOCK),
        GgmlType::NVFP4_ROWS => (k / 32, (k / 16) * NVFP4_GROUP),
        GgmlType::IQ4_NL => (k / 32, (k / 32) * IQ4NL_BLOCK),
        GgmlType::IQ4_XS => (k / 256, (k / 256) * IQ4XS_BLOCK),
        GgmlType::Q2_0 => (k / 64, (k / 64) * Q2F0_BLOCK),
        // The has_kernel check above refused every other type but the generic ones.
        other => {
            let e = generic::entry(other)
                .expect("has_kernel named a type with neither a hand kernel nor an entry");
            (k / e.values, (k / e.values) * e.bytes)
        }
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

pub use gguf::quant::nearest_int;

/// Quantizes one 256-value block into 296-byte `block_q8_K` layout. Refuses a
/// block with a non-finite value ([`gguf::quant::non_finite_activation`]).
fn quantize_q8k_block(x: &[f32], out: &mut [u8]) {
    debug_assert_eq!(x.len(), 256);
    debug_assert_eq!(out.len(), Q8K_STRIDE);
    let max = gguf::quant::activation_max(x, "qdot");
    out[4..8].fill(0);
    if max == 0.0 {
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
#[cfg(target_arch = "x86_64")]
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

/// Reduce lanes to the `as i8` wrap range [-128, 127] — the q8_K encoder's.
/// In-domain codes never reach the wrap (|iscale * v| stays under 127.5
/// before rounding); the ±inf/NaN products of an overflowing scale extract to
/// -2^22, whose low byte the scalar cast drops to 0 — the wrap reproduces
/// that, where a saturating pack alone would emit -128.
///
/// # Safety
/// AVX2 must be available on the target.
#[cfg(target_arch = "x86_64")]
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
/// AVX (`_mm256_castps256_ps128`, `_mm256_extractf128_ps`) and SSE3
/// (`_mm_movehdup_ps`) must be available on the target; being
/// `#[inline(always)]`, it takes them from its caller's features.
#[cfg(target_arch = "x86_64")]
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
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn quantize_q8k_block_avx2(x: &[f32; 256], out: &mut [u8; 296]) {
    // SAFETY: AVX2 present per contract; every load lands inside the 256-value
    // block (whole 8-lane groups, no over-read) and every store inside the
    // 296-byte block.
    unsafe {
        let sgn = _mm256_set1_ps(-0.0);
        let mut m = _mm256_setzero_ps();
        let mut unord = _mm256_setzero_ps();
        for i in 0..32 {
            // SAFETY: 8-lane load at 8*i <= 248, inside the block.
            let v = _mm256_loadu_ps(x.as_ptr().add(8 * i));
            // `max_ps` returns its second operand when either is NaN, so the
            // accumulator goes second and `m` never holds a NaN; the NaN
            // itself is collected in `unord` and refused below.
            m = _mm256_max_ps(_mm256_andnot_ps(sgn, v), m);
            unord = _mm256_or_ps(unord, _mm256_cmp_ps(v, v, _CMP_UNORD_Q));
        }
        let amax = hmax_ps(m);
        if _mm256_movemask_ps(unord) != 0 || !amax.is_finite() {
            non_finite_activation(x, "qdot");
        }
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
#[cfg(target_arch = "x86_64")]
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
#[cfg(target_arch = "x86_64")]
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
#[cfg(target_arch = "x86_64")]
struct Masks {
    m3: __m256i,
    mone: __m256i,
}

/// One field of one 128-value half: decode weights and dot against Q8_K span.
///
/// # Safety
/// `q8` must point at 32 readable bytes inside column buffer; AVX2 must be present.
#[cfg(target_arch = "x86_64")]
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
/// CPU must support AVX2+F16C; buffers must hold `nb` super-blocks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "f16c")]
unsafe fn dot_q3k_q8k_avx2(wrow: &[u8], acol: &[u8], nb: usize) -> f32 {
    // SAFETY: AVX2+F16C present and both slices hold nb super-blocks per contract.
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
            let d = f16c_to_f32((base.add(108) as *const u16).read_unaligned());

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

/// A Q3_K super-block's 16 scales as stored: six bits, `s + 32`. With
/// [`q3k_codes`], the one scalar unpack of a Q3_K block: the scalar mirror
/// and the row-lane repack read a block through these two.
fn q3k_scales6(blk: &[u8]) -> [u8; 16] {
    const KMASK1: u32 = 0x0303_0303;
    const KMASK2: u32 = 0x0f0f_0f0f;
    let aux: [u32; 3] =
        std::array::from_fn(|i| u32::from_le_bytes([0, 1, 2, 3].map(|b| blk[96 + 4 * i + b])));
    let words = [
        (aux[0] & KMASK2) | ((aux[2] & KMASK1) << 4),
        (aux[1] & KMASK2) | (((aux[2] >> 2) & KMASK1) << 4),
        ((aux[0] >> 4) & KMASK2) | (((aux[2] >> 4) & KMASK1) << 4),
        ((aux[1] >> 4) & KMASK2) | (((aux[2] >> 6) & KMASK1) << 4),
    ];
    std::array::from_fn(|j| words[j / 4].to_le_bytes()[j % 4])
}

/// A Q3_K super-block's 256 codes `u = value + 4` in 0..7: the two low bits
/// from `qs`, bit 2 the high bit of `hmask` (set = no −4).
fn q3k_codes(blk: &[u8]) -> [u8; 256] {
    std::array::from_fn(|i| {
        let (half, field, cell) = (i / 128, (i / 32) % 4, i % 32);
        let low = (blk[32 + 32 * half + cell] >> (2 * field)) & 3;
        let high = (blk[cell] >> (4 * half + field)) & 1;
        low | (high << 2)
    })
}

/// The inverse of [`q3k_scales6`]: sixteen six-bit scales (`s + 32`, each
/// below 64) into a Q3_K block's scale bytes 96..108 — the low nibbles of
/// scales `k` and `8 + k` in byte `96 + k`, of `4 + k` and `12 + k` in
/// `100 + k`, and the four scales' high pairs in `104 + k` (k in 0..4).
fn q3k_put_scales6(s: &[u8; 16], blk: &mut [u8]) {
    for k in 0..4 {
        blk[96 + k] = (s[k] & 0xF) | ((s[8 + k] & 0xF) << 4);
        blk[100 + k] = (s[4 + k] & 0xF) | ((s[12 + k] & 0xF) << 4);
        blk[104 + k] =
            (s[k] >> 4) | ((s[4 + k] >> 4) << 2) | ((s[8 + k] >> 4) << 4) | ((s[12 + k] >> 4) << 6);
    }
}

/// The inverse of [`q3k_codes`]: 256 codes `u` (each below 8) into a Q3_K
/// block's `hmask` (bytes 0..32) and `qs` (bytes 32..96). Value
/// `128 h + 32 f + c` keeps its low two bits at bits `2f` of `qs` byte
/// `32 h + c` and its third bit at bit `4h + f` of `hmask` byte `c`.
fn q3k_put_codes(u: &[u8; 256], blk: &mut [u8]) {
    for c in 0..32 {
        let mut high = 0u8;
        for h in 0..2 {
            let mut low = 0u8;
            for f in 0..4 {
                let v = u[128 * h + 32 * f + c];
                low |= (v & 3) << (2 * f);
                high |= (v >> 2) << (4 * h + f);
            }
            blk[32 + 32 * h + c] = low;
        }
        blk[c] = high;
    }
}

/// Scalar mirror of `dot_q3k_q8k_avx2`, bit-identical by construction: per
/// super-block the integer `Σ s_(i/16) · (u_i − 4) · q_i` over its 256 values
/// (exact in i32, so its order is free), then the kernel's float step.
fn dot_q3k_q8k_scalar(wrow: &[u8], acol: &[u8], nb: usize) -> f32 {
    let mut acc = 0.0f32;
    for sb in 0..nb {
        let blk = &wrow[sb * Q3K_BLOCK..sb * Q3K_BLOCK + Q3K_BLOCK];
        let scales = q3k_scales6(blk);
        let codes = q3k_codes(blk);
        let d = half_to_f32(u16::from_le_bytes([blk[108], blk[109]]));
        let q8 = &acol[sb * Q8K_STRIDE + 8..sb * Q8K_STRIDE + 8 + 256];
        let dcol = f32::from_le_bytes(
            acol[sb * Q8K_STRIDE..sb * Q8K_STRIDE + 4]
                .try_into()
                .unwrap(),
        );
        let mut sumi = 0i32;
        for (i, (&u, &a8)) in codes.iter().zip(q8).enumerate() {
            let sc = i32::from(scales[i / 16]) - 32;
            sumi += sc * (i32::from(u) - 4) * i32::from(a8 as i8);
        }
        acc += dcol * d * sumi as f32;
    }
    acc
}

/// [`field_dot`] for `C` columns, the high bit folded into the code: the
/// field is unpacked once as `u = q3l | h << 2` in 0..7 — the stored value
/// plus 4 (the high bit set is the value without its −4) — and its scales
/// shuffled once, then each column takes one load, one maddubs, one madd
/// and one add into its own `sumi`. The −4 is not this field's: the tile
/// opens each column's `sumi` with it.
///
/// # Safety
/// Each `q8[c] + off` must point at 32 readable bytes inside column `c`'s
/// buffer; AVX2 must be present.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
#[allow(
    clippy::too_many_arguments,
    reason = "the field_dot operands plus the column table, all registers of one inlined kernel body"
)]
unsafe fn field_dot_cols<const SHIFT: i32, const BIT: i32, const C: usize>(
    q8: &[*const u8; C],
    off: usize,
    hbits: __m256i,
    q3bits: __m256i,
    scales_j: __m256i,
    shuf_f: __m256i,
    masks: &Masks,
    sumi: &mut [__m256i; C],
) {
    // SAFETY: AVX2 present and each `q8[c] + off` points at 32 readable bytes per contract.
    unsafe {
        let q3l = _mm256_and_si256(_mm256_srli_epi16::<SHIFT>(q3bits), masks.m3);
        let h = _mm256_and_si256(_mm256_srli_epi16::<BIT>(hbits), masks.mone);
        let u = _mm256_or_si256(q3l, _mm256_slli_epi16::<2>(h));
        let sc = _mm256_shuffle_epi8(scales_j, shuf_f);
        for (q, s) in q8.iter().zip(sumi.iter_mut()) {
            // SAFETY: unaligned 32-byte load at q + off, inside column c's buffer.
            let q8f = _mm256_loadu_si256(q.add(off) as *const __m256i);
            let p = _mm256_madd_epi16(sc, _mm256_maddubs_epi16(u, q8f));
            *s = _mm256_add_epi32(*s, p);
        }
    }
}

/// AVX2 tile: one Q3_K weight row against `C` Q8_K columns, value `c` equal
/// to `dot_q3k_q8k_avx2(wrow, column c, nb)` bit for bit.
///
/// Per super-block the weight's scales, `d`, high-bit mask and code words are
/// read once, and each field is unpacked once for every column as codes
/// `u = value + 4` ([`field_dot_cols`]). The fold's `−4 · Σ_j s_j · bsum_j`
/// opens each column's integer sum: one madd of `−4 · s` against the 16 code
/// sums of the column's block (bytes 264..296, which `quantize_col` writes),
/// so the value holds for blocks whose sums are those of their codes. The
/// integer equals the one-column body's, `Σ s·(u − 4)·q = Σ s·u·q −
/// 4·Σ s·bsum`, and every step is exact for any i8 codes: maddubs pairs of
/// `u` in 0..7 stay within ±1792 (no saturation), madd pairs within ±524288,
/// i32 partials within ±1.2e7, and the super-block's |sumi| <= 4.2e6 < 2^24.
/// The `C` sums are then reduced together by one 8×8 transpose-add
/// ([`hsum8_i32`]) — an i32 add tree, the sum `hsum_i32` forms — and lane `c`
/// takes the one-column float step `acc + (dcol · d) · sumi`, in super-block
/// order: the same operations in the same order, column by column. Lanes
/// past `C` hold zeros and are never returned.
///
/// # Safety
/// CPU must support AVX2+F16C; `wrow` must hold `nb` super-blocks and each
/// `acols[c]` point at `nb` readable Q8_K blocks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "f16c")]
unsafe fn dot_q3k_q8k_tile_avx2<const C: usize>(
    wrow: &[u8],
    acols: &[*const u8; C],
    nb: usize,
) -> [f32; C] {
    const { assert!(C != 0 && C <= TILE_COLS) };
    // SAFETY: AVX2+F16C present, `wrow` holds nb super-blocks and every column
    // nb Q8_K blocks per contract.
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

        let zero = _mm256_setzero_si256();
        // Lane c is column c's accumulator; lanes past C stay unused.
        let mut acc = _mm256_setzero_ps();
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
            // −4·s_j per 16-code block, lane j against the column's block sum
            // j (both in block order); −4·s is within −124..128.
            let neg4 = _mm256_slli_epi16::<2>(_mm256_sub_epi16(zero, all_scales));

            // SAFETY: one unaligned u16 read inside the super-block.
            let d = f16c_to_f32((base.add(108) as *const u16).read_unaligned());

            // SAFETY: column c's super-block sb starts at sb * Q8K_STRIDE.
            let blk: [*const u8; C] = std::array::from_fn(|c| acols[c].add(sb * Q8K_STRIDE));
            // SAFETY: the codes start 8 bytes into the block.
            let q8: [*const u8; C] = std::array::from_fn(|c| blk[c].add(8));
            // Each column's sum opens with the fold's −4 · Σ s_j · bsum_j.
            let mut sumi = [zero; C];
            for (s, b) in sumi.iter_mut().zip(&blk) {
                // SAFETY: 32-byte load of the 16 i16 code sums at 264..296 of
                // the column's block.
                let bsums = _mm256_loadu_si256(b.add(264) as *const __m256i);
                *s = _mm256_madd_epi16(neg4, bsums);
            }
            for (j, scj) in scales.iter().enumerate() {
                // SAFETY: unaligned load inside the super-block.
                let q3bits = _mm256_loadu_si256(base.add(32 + 32 * j) as *const __m256i);
                let scj = *scj;
                if j == 0 {
                    field_dot_cols::<0, 0, C>(
                        &q8, 0, hbits, q3bits, scj, shuf[0], &masks, &mut sumi,
                    );
                    field_dot_cols::<2, 1, C>(
                        &q8, 32, hbits, q3bits, scj, shuf[1], &masks, &mut sumi,
                    );
                    field_dot_cols::<4, 2, C>(
                        &q8, 64, hbits, q3bits, scj, shuf[2], &masks, &mut sumi,
                    );
                    field_dot_cols::<6, 3, C>(
                        &q8, 96, hbits, q3bits, scj, shuf[3], &masks, &mut sumi,
                    );
                } else {
                    field_dot_cols::<0, 4, C>(
                        &q8, 128, hbits, q3bits, scj, shuf[0], &masks, &mut sumi,
                    );
                    field_dot_cols::<2, 5, C>(
                        &q8, 160, hbits, q3bits, scj, shuf[1], &masks, &mut sumi,
                    );
                    field_dot_cols::<4, 6, C>(
                        &q8, 192, hbits, q3bits, scj, shuf[2], &masks, &mut sumi,
                    );
                    field_dot_cols::<6, 7, C>(
                        &q8, 224, hbits, q3bits, scj, shuf[3], &masks, &mut sumi,
                    );
                }
            }
            // Zero registers past C: those lanes sum to 0.
            let mut s8 = [zero; TILE_COLS];
            s8[..C].copy_from_slice(&sumi);
            let tot = hsum8_i32(&s8);
            let mut dcol = [0.0f32; TILE_COLS];
            for (dc, b) in dcol.iter_mut().zip(&blk) {
                // SAFETY: unaligned f32 read at the head of the column's block.
                *dc = (*b as *const f32).read_unaligned();
            }
            let dcol = _mm256_setr_ps(
                dcol[0], dcol[1], dcol[2], dcol[3], dcol[4], dcol[5], dcol[6], dcol[7],
            );
            let t = _mm256_mul_ps(dcol, _mm256_set1_ps(d));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(t, _mm256_cvtepi32_ps(tot)));
        }
        let mut lanes = [0.0f32; TILE_COLS];
        // SAFETY: 32-byte store into the local 8-lane array.
        _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
        let mut out = [0.0f32; C];
        out.copy_from_slice(&lanes[..C]);
        out
    }
}

/// Eight horizontal i32 sums at once: lane `c` of the result is the sum of
/// `s[c]`'s eight lanes (an 8×8 transpose-add; exact, as integer adds are in
/// any order).
///
/// # Safety
/// AVX2 must be available on the target.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn hsum8_i32(s: &[__m256i; 8]) -> __m256i {
    // SAFETY: register-only intrinsics, no memory access.
    unsafe {
        // Per 128-bit half of a pair (a, b) = (s[i], s[i + 1]):
        // [a0 + a2, b0 + b2, a1 + a3, b1 + b3].
        let p01 = _mm256_add_epi32(
            _mm256_unpacklo_epi32(s[0], s[1]),
            _mm256_unpackhi_epi32(s[0], s[1]),
        );
        let p23 = _mm256_add_epi32(
            _mm256_unpacklo_epi32(s[2], s[3]),
            _mm256_unpackhi_epi32(s[2], s[3]),
        );
        let p45 = _mm256_add_epi32(
            _mm256_unpacklo_epi32(s[4], s[5]),
            _mm256_unpackhi_epi32(s[4], s[5]),
        );
        let p67 = _mm256_add_epi32(
            _mm256_unpacklo_epi32(s[6], s[7]),
            _mm256_unpackhi_epi32(s[6], s[7]),
        );
        // Per 128-bit half, lane i of q0 (q1): the four lanes of s[i]
        // (s[4 + i]) in that half, summed.
        let q0 = _mm256_add_epi32(
            _mm256_unpacklo_epi64(p01, p23),
            _mm256_unpackhi_epi64(p01, p23),
        );
        let q1 = _mm256_add_epi32(
            _mm256_unpacklo_epi64(p45, p67),
            _mm256_unpackhi_epi64(p45, p67),
        );
        _mm256_add_epi32(
            _mm256_permute2x128_si256::<0x20>(q0, q1),
            _mm256_permute2x128_si256::<0x31>(q0, q1),
        )
    }
}

// ------------------------------------------------- Q3_K row-lane tile

/// One super-block of an 8-row group: the eight rows' 110 bytes, permuted.
const Q3K_R8_BLOCK: usize = Q3K_R8_ROWS * Q3K_BLOCK;
/// Where a group super-block's scales and code pairs start, and a pair's
/// bytes ([`repack_q3k_r8`] has the layout).
const R8_SCALES: usize = 16;
const R8_CODES: usize = R8_SCALES + 96;
const R8_PAIR: usize = 96;
// The tile's bounds rest on these: the eight f16 `d` fill bytes 0..16 (one
// 16-byte load, a lane per row), and the eighth code pair ends on the
// super-block's last byte.
const _: () = assert!(R8_SCALES == 2 * Q3K_R8_ROWS);
const _: () = assert!(R8_CODES + 8 * R8_PAIR == Q3K_R8_BLOCK);

/// Scale-pair shuffles of the row-lane tile, per 128-bit half: from a vector
/// whose dword `r` is the i16 pair `[s_r,2p, s_r,2p+1]`, the first 32 bytes
/// make `[s_r,2p, s_r,2p]` and the last 32 `[s_r,2p+1, s_r,2p+1]`.
#[cfg(target_arch = "x86_64")]
static R8_DUP: [u8; 64] = [
    0, 1, 0, 1, 4, 5, 4, 5, 8, 9, 8, 9, 12, 13, 12, 13, //
    0, 1, 0, 1, 4, 5, 4, 5, 8, 9, 8, 9, 12, 13, 12, 13, //
    2, 3, 2, 3, 6, 7, 6, 7, 10, 11, 10, 11, 14, 15, 14, 15, //
    2, 3, 2, 3, 6, 7, 6, 7, 10, 11, 10, 11, 14, 15, 14, 15, //
];

/// One group super-block of [`repack_q3k_r8`]'s layout from the eight rows'
/// 110-byte blocks.
fn r8_pack_block(rows: &[&[u8]; Q3K_R8_ROWS], dst: &mut [u8; Q3K_R8_BLOCK]) {
    let scales = rows.map(q3k_scales6);
    let codes = rows.map(q3k_codes);
    for (r, blk) in rows.iter().enumerate() {
        dst[2 * r..2 * r + 2].copy_from_slice(&blk[108..110]);
    }
    for b in 0..32 {
        let x: [u8; 4] =
            std::array::from_fn(|q| scales[(b % 16) / 2][2 * (2 * q + b / 16) + b % 2]);
        dst[R8_SCALES + b] = (x[0] & 0xF) | ((x[1] & 0xF) << 4);
        dst[R8_SCALES + 32 + b] = (x[2] & 0xF) | ((x[3] & 0xF) << 4);
        dst[R8_SCALES + 64 + b] =
            (x[0] >> 4) | ((x[1] >> 4) << 2) | ((x[2] >> 4) << 4) | ((x[3] >> 4) << 6);
    }
    for p in 0..8 {
        for b in 0..32 {
            let w: [u8; 8] =
                std::array::from_fn(|f| codes[b / 4][16 * (2 * p + f / 4) + 4 * (f % 4) + b % 4]);
            let at = R8_CODES + R8_PAIR * p + b;
            dst[at] = w[0] | (w[1] << 3) | ((w[2] & 3) << 6);
            dst[at + 32] = w[3] | (w[4] << 3) | ((w[5] & 3) << 6);
            dst[at + 64] = w[6] | (w[7] << 3) | ((w[2] >> 2) << 6) | ((w[5] >> 2) << 7);
        }
    }
}

/// Row `r`'s 110-byte Q3_K block out of a group super-block — the inverse of
/// [`r8_pack_block`] for that row: its fields read through [`r8_scale`] and
/// [`r8_code`], written through the inverses of [`q3k_scales6`] and
/// [`q3k_codes`].
fn r8_unpack_row(blk: &[u8; Q3K_R8_BLOCK], r: usize, dst: &mut [u8]) {
    // r8_scale gives s in -32..32; the stored six bits are s + 32.
    let scales: [u8; 16] = std::array::from_fn(|j| (r8_scale(blk, r, j) + 32) as u8);
    let codes: [u8; 256] = std::array::from_fn(|i| r8_code(blk, r, i));
    q3k_put_codes(&codes, dst);
    q3k_put_scales6(&scales, dst);
    dst[108..110].copy_from_slice(&blk[2 * r..2 * r + 2]);
}

/// Scale `s` of sub-block `j` of row `r`, off a group super-block.
fn r8_scale(blk: &[u8], r: usize, j: usize) -> i32 {
    let p = j / 2;
    let (q, b) = (p / 2, 16 * (p % 2) + 2 * r + j % 2);
    let low = (blk[R8_SCALES + 32 * (q / 2) + b] >> (4 * (q % 2))) & 0xF;
    let high = (blk[R8_SCALES + 64 + b] >> (2 * q)) & 3;
    i32::from(low | (high << 4)) - 32
}

/// Code `u` of value `i` of row `r`, off a group super-block.
fn r8_code(blk: &[u8], r: usize, i: usize) -> u8 {
    let (j, t) = (i / 16, i % 16);
    let (p, f) = (j / 2, 4 * (j % 2) + t / 4);
    let at = R8_CODES + R8_PAIR * p + 4 * r + t % 4;
    let (a, b, c) = (blk[at], blk[at + 32], blk[at + 64]);
    match f {
        0 => a & 7,
        1 => (a >> 3) & 7,
        2 => (a >> 6) | (((c >> 6) & 1) << 2),
        3 => b & 7,
        4 => (b >> 3) & 7,
        5 => (b >> 6) | (((c >> 7) & 1) << 2),
        6 => c & 7,
        _ => (c >> 3) & 7,
    }
}

/// Scalar mirror of `dot_q3k_r8_tile_avx2` over validated inputs: each
/// (row, column) the one-column scalar mirror's integer sum and float step,
/// read off the repacked layout.
fn r8_scalar(group: &[u8], acols: &[&[u8]], nb: usize, out: &mut [[f32; Q3K_R8_ROWS]]) {
    for (acol, o) in acols.iter().zip(out.iter_mut()) {
        for (r, v) in o.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for sb in 0..nb {
                let blk = &group[sb * Q3K_R8_BLOCK..(sb + 1) * Q3K_R8_BLOCK];
                let d = half_to_f32(u16::from_le_bytes([blk[2 * r], blk[2 * r + 1]]));
                let col = &acol[sb * Q8K_STRIDE..(sb + 1) * Q8K_STRIDE];
                let dcol = f32::from_le_bytes([col[0], col[1], col[2], col[3]]);
                let mut sumi = 0i32;
                for j in 0..16 {
                    let s = r8_scale(blk, r, j);
                    for l in 0..16 {
                        let u = i32::from(r8_code(blk, r, 16 * j + l));
                        sumi += s * (u - 4) * i32::from(col[8 + 16 * j + l] as i8);
                    }
                }
                acc += dcol * d * sumi as f32;
            }
            *v = acc;
        }
    }
}

/// One sub-block of the row-lane tile for every column: `w` its four code
/// vectors, `s` its scale pairs `[s_rj, s_rj]`, `off` the offset of its first
/// code in a Q8_K block. Per column, the four maddubs against four-byte
/// broadcasts are summed in i16 (each lane 8 products of `u` ≤ 7 and
/// |q| ≤ 128: at most 7168), then one madd with `s` adds `s_rj · Σ u·q` to
/// the lane's i32.
///
/// # Safety
/// Each `blk[c] + off .. + 16` must be readable; AVX2 must be present.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn r8_sub_block<const C: usize>(
    w: &[__m256i; 4],
    s: __m256i,
    blk: &[*const u8; C],
    off: usize,
    sumi: &mut [__m256i; C],
) {
    // SAFETY: AVX2 present and each column's 16 codes at `off` readable per contract.
    unsafe {
        for (acc, b) in sumi.iter_mut().zip(blk) {
            let q = b.add(off) as *const i32;
            // SAFETY: four 4-byte reads inside the sub-block's 16 codes.
            let x0 = _mm256_set1_epi32(q.read_unaligned());
            let x1 = _mm256_set1_epi32(q.add(1).read_unaligned());
            let x2 = _mm256_set1_epi32(q.add(2).read_unaligned());
            let x3 = _mm256_set1_epi32(q.add(3).read_unaligned());
            let t = _mm256_add_epi16(
                _mm256_add_epi16(
                    _mm256_maddubs_epi16(w[0], x0),
                    _mm256_maddubs_epi16(w[1], x1),
                ),
                _mm256_add_epi16(
                    _mm256_maddubs_epi16(w[2], x2),
                    _mm256_maddubs_epi16(w[3], x3),
                ),
            );
            *acc = _mm256_add_epi32(*acc, _mm256_madd_epi16(s, t));
        }
    }
}

/// AVX2 row-lane tile: one 8-row group of [`repack_q3k_r8`]'s layout against
/// `C` Q8_K columns; lane `r` of value `c` equals
/// `dot_q3k_q8k_avx2(row r, column c, nb)` bit for bit.
///
/// A ymm lane is a row. Per sub-block pair the eight code vectors and the
/// scale pairs are unpacked once for every column ([`r8_sub_block`] does the
/// columns). The fold's `−4 · Σ_j s_rj · bsum_j` opens each pair: one madd
/// of `4 · [s_r,2p, s_r,2p+1]` against the column's block sums `2p, 2p+1`
/// (bytes 264..296, which `quantize_col` writes), subtracted. The super-block's
/// integer `Σ_j s_rj Σ (u − 4)·q` is the one-column body's for any i8 codes
/// (every partial within ±2^23, exact), and lane `r` takes the one-column
/// float step `acc + (dcol · d_r) · sumi` in super-block order: the same
/// operations in the same order, (row, column) by (row, column).
///
/// # Safety
/// CPU must support AVX2+F16C; `group` must hold `nb` group super-blocks and
/// each `acols[c]` point at `nb` readable Q8_K blocks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "f16c")]
unsafe fn dot_q3k_r8_tile_avx2<const C: usize>(
    group: &[u8],
    acols: &[*const u8; C],
    nb: usize,
) -> [[f32; Q3K_R8_ROWS]; C] {
    const { assert!(C != 0 && C <= TILE_COLS) };
    // SAFETY: AVX2+F16C present, `group` holds nb group super-blocks and every
    // column nb Q8_K blocks per contract.
    unsafe {
        let m7 = _mm256_set1_epi8(7);
        let m3 = _mm256_set1_epi8(3);
        let m4 = _mm256_set1_epi8(4);
        let m0f = _mm256_set1_epi8(0x0F);
        let m30 = _mm256_set1_epi8(0x30);
        let m32 = _mm256_set1_epi8(32);
        // SAFETY: two 32-byte loads inside the 64-byte static table.
        let dup_lo = _mm256_loadu_si256(R8_DUP.as_ptr() as *const __m256i);
        let dup_hi = _mm256_loadu_si256(R8_DUP.as_ptr().add(32) as *const __m256i);
        let zero = _mm256_setzero_si256();
        let mut acc = [_mm256_setzero_ps(); C];
        for sb in 0..nb {
            // SAFETY: sb < nb, and check_r8 sized `group` to exactly nb group
            // super-blocks: base .. base + Q3K_R8_BLOCK lies inside it.
            let base = group.as_ptr().add(Q3K_R8_BLOCK * sb);
            // SAFETY: three 32-byte loads inside the super-block's scale bytes.
            let l0 = _mm256_loadu_si256(base.add(R8_SCALES) as *const __m256i);
            let l1 = _mm256_loadu_si256(base.add(R8_SCALES + 32) as *const __m256i);
            let hb = _mm256_loadu_si256(base.add(R8_SCALES + 64) as *const __m256i);
            // The 128 scales as signed bytes, pair p at bytes 16p..16p + 16.
            let scales = [
                _mm256_sub_epi8(
                    _mm256_or_si256(
                        _mm256_and_si256(l0, m0f),
                        _mm256_and_si256(_mm256_slli_epi16::<4>(hb), m30),
                    ),
                    m32,
                ),
                _mm256_sub_epi8(
                    _mm256_or_si256(
                        _mm256_and_si256(_mm256_srli_epi16::<4>(l0), m0f),
                        _mm256_and_si256(_mm256_slli_epi16::<2>(hb), m30),
                    ),
                    m32,
                ),
                _mm256_sub_epi8(
                    _mm256_or_si256(_mm256_and_si256(l1, m0f), _mm256_and_si256(hb, m30)),
                    m32,
                ),
                _mm256_sub_epi8(
                    _mm256_or_si256(
                        _mm256_and_si256(_mm256_srli_epi16::<4>(l1), m0f),
                        _mm256_and_si256(_mm256_srli_epi16::<2>(hb), m30),
                    ),
                    m32,
                ),
            ];
            let sc = scales.as_ptr() as *const u8;
            // SAFETY: sb < nb, and check_r8 saw every column hold nb Q8_K
            // blocks: column c's block sb starts at sb * Q8K_STRIDE, inside it.
            let blk: [*const u8; C] = std::array::from_fn(|c| acols[c].add(sb * Q8K_STRIDE));
            let mut sumi = [zero; C];
            for p in 0..8 {
                // SAFETY: 16-byte load inside the local 128-byte scale array.
                let v = _mm256_cvtepi8_epi16(_mm_loadu_si128(sc.add(16 * p) as *const __m128i));
                let v4 = _mm256_slli_epi16::<2>(v);
                for (s, b) in sumi.iter_mut().zip(&blk) {
                    // SAFETY: 4-byte read of block sums 2p, 2p + 1 at 264 + 4p.
                    let bs = _mm256_set1_epi32((b.add(264 + 4 * p) as *const i32).read_unaligned());
                    *s = _mm256_sub_epi32(*s, _mm256_madd_epi16(v4, bs));
                }
                // SAFETY: p < 8, and the eighth pair ends at R8_CODES + 8 *
                // R8_PAIR == Q3K_R8_BLOCK (const-asserted): inside the
                // super-block at base.
                let codes = base.add(R8_CODES + R8_PAIR * p);
                // SAFETY: 32-byte loads inside the pair's 96 bytes.
                let a = _mm256_loadu_si256(codes as *const __m256i);
                let b = _mm256_loadu_si256(codes.add(32) as *const __m256i);
                let c = _mm256_loadu_si256(codes.add(64) as *const __m256i);
                let w = [
                    _mm256_and_si256(a, m7),
                    _mm256_and_si256(_mm256_srli_epi16::<3>(a), m7),
                    _mm256_or_si256(
                        _mm256_and_si256(_mm256_srli_epi16::<6>(a), m3),
                        _mm256_and_si256(_mm256_srli_epi16::<4>(c), m4),
                    ),
                    _mm256_and_si256(b, m7),
                ];
                let s = _mm256_shuffle_epi8(v, dup_lo);
                r8_sub_block::<C>(&w, s, &blk, 8 + 32 * p, &mut sumi);
                // SAFETY: 32-byte loads inside the pair's 96 bytes.
                let b = _mm256_loadu_si256(codes.add(32) as *const __m256i);
                let c = _mm256_loadu_si256(codes.add(64) as *const __m256i);
                let w = [
                    _mm256_and_si256(_mm256_srli_epi16::<3>(b), m7),
                    _mm256_or_si256(
                        _mm256_and_si256(_mm256_srli_epi16::<6>(b), m3),
                        _mm256_and_si256(_mm256_srli_epi16::<5>(c), m4),
                    ),
                    _mm256_and_si256(c, m7),
                    _mm256_and_si256(_mm256_srli_epi16::<3>(c), m7),
                ];
                let s = _mm256_shuffle_epi8(v, dup_hi);
                r8_sub_block::<C>(&w, s, &blk, 8 + 32 * p + 16, &mut sumi);
            }
            // SAFETY: 16-byte load of the eight f16 `d` at the super-block's head.
            let d = _mm256_cvtph_ps(_mm_loadu_si128(base as *const __m128i));
            for ((a, s), b) in acc.iter_mut().zip(&sumi).zip(&blk) {
                // SAFETY: f32 read at the head of the column's block.
                let dcol = _mm256_set1_ps((*b as *const f32).read_unaligned());
                let t = _mm256_mul_ps(dcol, d);
                *a = _mm256_add_ps(*a, _mm256_mul_ps(t, _mm256_cvtepi32_ps(*s)));
            }
        }
        let mut out = [[0.0f32; Q3K_R8_ROWS]; C];
        for (o, a) in out.iter_mut().zip(&acc) {
            // SAFETY: 32-byte store into the column's eight lanes.
            _mm256_storeu_ps(o.as_mut_ptr(), *a);
        }
        out
    }
}

// ------------------------------------- V4.1 card-rule kernels (q8_1, Q3_K, Q4_K)
//
// The card engine's own rule for a routed expert's contribution — q8_1
// activations per 128 values, the warp-lane walks and the xor butterfly —
// as host kernels, so an expert computes the same bits whichever tier runs
// it. Everything here transcribes the device cores; each doc names the
// function it copies, and the float op spellings (a true division, `round`
// half away from zero, which products contract into an FMA) were read from
// the PTX the backend emits for them, not assumed from the Rust source.
// These kernels are ours, not ik's: the host tier's other columns keep
// q8_K / q8_2_x4 and the dots above.

/// The card's q8_1 form of one activation column, in the host's own layout:
/// `k` codes in value order, one scale per 128-value block (`d8`), one
/// signed sum per 32 values (`s8`) and the per-block refusal flags. The
/// codes and scales are the bytes every card quantizer writes
/// (`q8_1_quant_block`, crates/gpu/src/lib.rs); only the plane
/// permutations the card's gemv geometries read are dropped — the dots below
/// take their words from `codes` directly.
#[derive(Clone, Debug)]
pub struct CardQ81 {
    k: usize,
    codes: Vec<i8>,
    d8: Vec<f32>,
    s8: Vec<i32>,
    refused: Vec<bool>,
}

impl CardQ81 {
    /// The zeroed form of a column of `k` values (a positive multiple of
    /// 256), its buffers owned for [`card_q8_1_fill`].
    #[must_use]
    pub fn for_k(k: usize) -> CardQ81 {
        assert!(
            k > 0 && k.is_multiple_of(256),
            "card q8_1 columns are whole 256-value super-blocks, k = {k}"
        );
        let n_sb = k / 256;
        CardQ81 {
            k,
            codes: vec![0i8; k],
            d8: vec![0.0; 2 * n_sb],
            s8: vec![0; 8 * n_sb],
            refused: vec![false; 2 * n_sb],
        }
    }

    /// Values in the column (a positive multiple of 256).
    #[must_use]
    pub fn k(&self) -> usize {
        self.k
    }

    /// The q8_1 codes in value order: `codes()[j]` is value `j`'s code,
    /// zero for every value of a refused block.
    #[must_use]
    pub fn codes(&self) -> &[i8] {
        &self.codes
    }

    /// One scale per 128-value block, `d8()[b]` = `amax / 127` of block `b`
    /// (`1.0` for an all-zero block, NaN for a refused one).
    #[must_use]
    pub fn d8(&self) -> &[f32] {
        &self.d8
    }

    /// One exact signed code sum per 32 values: `s8()[g]` sums values
    /// `32g .. 32g + 31`, zero for a refused block's groups.
    #[must_use]
    pub fn s8(&self) -> &[i32] {
        &self.s8
    }

    /// One flag per 128-value block: whether it held a non-finite value and
    /// was stored as a NaN scale with zero codes and sums (the card's one
    /// q8_1 refusal rule, `q8_1_quant_vals`, crates/gpu/src/lib.rs).
    #[must_use]
    pub fn refused(&self) -> &[bool] {
        &self.refused
    }

    /// Whether any block was refused.
    #[must_use]
    pub fn refused_any(&self) -> bool {
        self.refused.contains(&true)
    }
}

/// The card's q8_1 quantizer of one column of `k` values (a positive
/// multiple of 256), AVX2 where available: per 128-value block `b`,
/// `d = if amax > 0 { amax / 127 } else { 1 }` over the block's |v| (a
/// true division, `div.rn.f32` in the PTX), codes
/// `(v / d).round().clamp(-127, 127)` — `round` half away from zero: the
/// PTX spells it `cvt.rzi(x + copysign(0.5, x))`, `cvt.rzi(x)` where
/// `|x| < 0.5` and `x` itself where `|x| > 2^23`, which is `f32::round` on
/// every finite `x`, not the ties-even `cvt.rni` — and the exact signed
/// code sums per 32 values. No operation flushes subnormals: a block whose
/// scale underflows to zero gives `±inf` quotients (codes `±127`) and `0/0`
/// (code 0), as the card's saturating convert does. A block holding any non-finite
/// value is refused: NaN scale, zero codes, zero sums. The value semantics
/// are `q8_1_quant_vals` + `q8_quad` (crates/gpu/src/lib.rs,
/// crates/gpu/src/cores.rs); the warp's lane partition is gone because a
/// block's bytes do not depend on it.
pub fn card_q8_1(x: &[f32]) -> CardQ81 {
    card_q8_1_check(x);
    let mut out = CardQ81::for_k(x.len());
    card_q8_1_fill(x, &mut out);
    out
}

/// The scalar mirror behind [`card_q8_1`], and the oracle its bit-identity
/// gate compares against.
pub fn card_q8_1_scalar(x: &[f32]) -> CardQ81 {
    card_q8_1_check(x);
    let mut out = CardQ81::for_k(x.len());
    card_q8_1_fill_scalar(x, &mut out);
    out
}

/// [`card_q8_1`] into a form [`CardQ81::for_k`] made for `x.len()` — the
/// per-column scratch a caller refills every call, no allocation inside.
pub fn card_q8_1_fill(x: &[f32], out: &mut CardQ81) {
    card_q8_1_check(x);
    assert_eq!(out.k, x.len(), "card_q8_1_fill: the form's k is x.len()");
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        // SAFETY: the features were just detected; card_q8_1_check pinned
        // x.len() to whole 128-value blocks and the assert the form's shape.
        unsafe { card_q8_1_fill_avx2(x, out) };
        return;
    }
    card_q8_1_fill_scalar(x, out);
}

/// The scalar mirror behind [`card_q8_1_fill`].
pub fn card_q8_1_fill_scalar(x: &[f32], out: &mut CardQ81) {
    card_q8_1_check(x);
    assert_eq!(out.k, x.len(), "card_q8_1_fill: the form's k is x.len()");
    let (blocks, _) = x.as_chunks::<128>();
    for (b, blk) in blocks.iter().enumerate() {
        let bad = blk.iter().any(|v| !v.is_finite());
        let amax = blk.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let d = if bad {
            f32::NAN
        } else if amax > 0.0 {
            amax / 127.0
        } else {
            1.0
        };
        for g in 0..4 {
            let mut sum = 0i32;
            for j in 0..32 {
                let v = blk[32 * g + j];
                let q = if bad {
                    0
                } else {
                    (v / d).round().clamp(-127.0, 127.0) as i32
                };
                out.codes[b * 128 + 32 * g + j] = q as i8;
                sum += q;
            }
            out.s8[b * 4 + g] = sum;
        }
        out.d8[b] = d;
        out.refused[b] = bad;
    }
}

/// The shape contract every [`card_q8_1`] entry point enforces first.
fn card_q8_1_check(x: &[f32]) {
    assert!(
        !x.is_empty() && x.len().is_multiple_of(256),
        "card q8_1 columns are whole 256-value super-blocks, x.len() = {}",
        x.len()
    );
}

/// # Safety
/// The CPU must support AVX2 and FMA, and `x` whole 128-value blocks with
/// `out` the form [`CardQ81::for_k`] made for them.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn card_q8_1_fill_avx2(x: &[f32], out: &mut CardQ81) {
    let (blocks, _) = x.as_chunks::<128>();
    let abs_mask = _mm256_set1_ps(-0.0);
    for (b, blk) in blocks.iter().enumerate() {
        // SAFETY: blk is 128 f32; every load and store below stays inside
        // blk or the form's buffers by these indices.
        unsafe {
            // The block's first four vectors (values 0..32); the loop
            // below walks vectors 4..16 to the block's end.
            let v0 = _mm256_loadu_ps(blk.as_ptr());
            let v1 = _mm256_loadu_ps(blk.as_ptr().add(8));
            let v2 = _mm256_loadu_ps(blk.as_ptr().add(16));
            let v3 = _mm256_loadu_ps(blk.as_ptr().add(24));
            let a0 = _mm256_andnot_ps(abs_mask, v0);
            let a1 = _mm256_andnot_ps(abs_mask, v1);
            let a2 = _mm256_andnot_ps(abs_mask, v2);
            let a3 = _mm256_andnot_ps(abs_mask, v3);
            // Finite test over all sixteen vectors: |v| <= MAX is false
            // for NaN and for inf, and the whole block refuses if any lane
            // fails it (the ballot's value).
            let fmax = _mm256_set1_ps(f32::MAX);
            let mut finite = _mm256_and_ps(
                _mm256_and_ps(
                    _mm256_cmp_ps(a0, fmax, _CMP_LE_OQ),
                    _mm256_cmp_ps(a1, fmax, _CMP_LE_OQ),
                ),
                _mm256_and_ps(
                    _mm256_cmp_ps(a2, fmax, _CMP_LE_OQ),
                    _mm256_cmp_ps(a3, fmax, _CMP_LE_OQ),
                ),
            );
            // amax: the block's max |v| over all sixteen vectors, exact in
            // any order over non-NaN, folded to lane 0.
            let mut m = _mm256_max_ps(_mm256_max_ps(a0, a1), _mm256_max_ps(a2, a3));
            for w in 4..16 {
                let v = _mm256_loadu_ps(blk.as_ptr().add(8 * w));
                let a = _mm256_andnot_ps(abs_mask, v);
                finite = _mm256_and_ps(finite, _mm256_cmp_ps(a, fmax, _CMP_LE_OQ));
                m = _mm256_max_ps(m, a);
            }
            let bad = _mm256_movemask_ps(finite) != 0xff;
            m = _mm256_max_ps(m, _mm256_permute2f128_ps(m, m, 0x01));
            m = _mm256_max_ps(m, _mm256_permute_ps(m, 78));
            m = _mm256_max_ps(m, _mm256_permute_ps(m, 177));
            let amax = _mm256_cvtss_f32(m);
            let d = if bad {
                f32::NAN
            } else if amax > 0.0 {
                amax / 127.0
            } else {
                1.0
            };
            let dv = _mm256_set1_ps(d);
            for g in 0..4 {
                let mut sum = _mm256_setzero_si256();
                for j in 0..4 {
                    // SAFETY: 8 readable values of the block at 32g + 8j.
                    let v = _mm256_loadu_ps(blk.as_ptr().add(32 * g + 8 * j));
                    // codes: (v / d).round() half away from zero, clamped.
                    // roundps has no ties-away mode: round to nearest even,
                    // then step the exact +.5 differences up — |v/d| at a tie
                    // rounds to the even neighbour, half away wants the larger,
                    // and the ax - e difference is exact (Sterbenz), so the
                    // compare sees a tie only at a real one.
                    let q = if bad {
                        _mm256_setzero_ps()
                    } else {
                        let r = _mm256_div_ps(v, dv);
                        let ax = _mm256_andnot_ps(abs_mask, r);
                        let e = _mm256_round_ps(ax, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                        let dif = _mm256_sub_ps(ax, e);
                        let tie = _mm256_cmp_ps(dif, _mm256_set1_ps(0.5), _CMP_EQ_OQ);
                        let away = _mm256_add_ps(e, _mm256_and_ps(tie, _mm256_set1_ps(1.0)));
                        _mm256_xor_ps(away, _mm256_and_ps(abs_mask, r))
                    };
                    // The clamp in floats, as the card's: `max_ps`/`min_ps`
                    // return their second operand on a NaN, so a NaN (`0/0`
                    // of an underflowed scale) passes it and ±inf clamps to
                    // ±127; the convert's NaN then becomes code 0, the card's
                    // saturating `cvt.rzi.s32` value.
                    let qc = _mm256_min_ps(
                        _mm256_set1_ps(127.0),
                        _mm256_max_ps(_mm256_set1_ps(-127.0), q),
                    );
                    let cl = _mm256_and_si256(
                        _mm256_cvtps_epi32(qc),
                        _mm256_castps_si256(_mm256_cmp_ps(qc, qc, _CMP_ORD_Q)),
                    );
                    // SAFETY: 8 writable code bytes of the form's column; the
                    // values are inside ±127, so both packs are saturation-free.
                    let packed = _mm_packs_epi32(
                        _mm256_castsi256_si128(cl),
                        _mm256_extracti128_si256(cl, 1),
                    );
                    let packed = _mm_packs_epi16(packed, packed);
                    _mm_storel_epi64(
                        out.codes.as_mut_ptr().add(b * 128 + 32 * g + 8 * j) as *mut __m128i,
                        packed,
                    );
                    // The sub-vector's exact signed sum, from the clamped
                    // codes; the four add into the group's (exact i32).
                    let s = _mm256_add_epi32(cl, _mm256_permute2f128_si256(cl, cl, 0x01));
                    let s = _mm256_hadd_epi32(s, s);
                    let s = _mm256_hadd_epi32(s, s);
                    sum = _mm256_add_epi32(sum, s);
                }
                out.s8[b * 4 + g] = _mm256_cvtsi256_si32(sum);
            }
            out.d8[b] = d;
            out.refused[b] = bad;
        }
    }
}

/// The warp's xor-butterfly sum of 32 lane values held as four `__m256`
/// (vector `v` = lanes `8v .. 8v+7`), lane 0's value: the tree
/// `warp::reduce_sum_f32` builds at lane 0 (cuda-device `warp.rs`, strides
/// 16, 8, 4, 2, 1 — each step adds the partner's value as it stands after
/// the steps before it). Only lane 0 is reduced: every consumer of a row's
/// dot reads lane 0's store.
///
/// # Safety
/// AVX2 must be available on the target. Register-only.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn card_warp_sum(f: [__m256; 4]) -> f32 {
    // stride 16 (f0+f2, f1+f3) folded with stride 8 (their sum): after
    // both, lane j of `c` holds (oj + oj^16) + (oj^8 + oj^24).
    let c = _mm256_add_ps(_mm256_add_ps(f[0], f[2]), _mm256_add_ps(f[1], f[3]));
    // stride 4 crosses the two 128-bit halves: lanes 0..3 take 4..7.
    let c = _mm256_add_ps(c, _mm256_permute2f128_ps(c, c, 0x01));
    // stride 2 and 1 stay inside each half: src lanes [2,3,0,1] (78 =
    // _MM_SHUFFLE(1,0,3,2)), then [1,0,3,2] (177 = _MM_SHUFFLE(2,3,0,1)).
    let c = _mm256_add_ps(c, _mm256_permute_ps(c, 78));
    let c = _mm256_add_ps(c, _mm256_permute_ps(c, 177));
    _mm256_cvtss_f32(c)
}

/// The little-endian u32 at `p[i .. i + 4]`.
#[inline(always)]
fn le32(p: &[u8], i: usize) -> u32 {
    u32::from_le_bytes(p[i..i + 4].try_into().unwrap())
}

/// The little-endian u16 at `p[i .. i + 2]`.
#[inline(always)]
fn le16(p: &[u8], i: usize) -> u16 {
    u16::from_le_bytes(p[i..i + 2].try_into().unwrap())
}

/// `dp4a(vi, w, 0)` over the four byte lanes: the exact i32 sum of the
/// signed-byte products. Integer arithmetic, so any order gives these bits.
#[inline(always)]
fn card_dp4a(vi: u32, w: u32) -> i32 {
    (0..4).fold(0i32, |a, b| {
        a + (vi.to_le_bytes()[b] as i8 as i32) * (w.to_le_bytes()[b] as i8 as i32)
    })
}

/// The code word of values `4 * v4 .. 4 * v4 + 3` of a column.
#[inline(always)]
fn card_word(codes: &[i8], v4: usize) -> u32 {
    let b = &codes[4 * v4..4 * v4 + 4];
    u32::from_le_bytes([b[0] as u8, b[1] as u8, b[2] as u8, b[3] as u8])
}

/// The card's `q3k_dequant` (crates/gpu/src/cores.rs) field `f` of the
/// lane's qs/hmask words: the signed weight byte `nib - 4*hbit` packed four
/// to a u32.
#[inline(always)]
fn card_q3k_field(vl: u32, vh1: u32, f: u32) -> u32 {
    let sh = 2 * f;
    let hb = match f {
        0 => vh1 << 2,
        1 => (vh1 >> 1) << 2,
        2 => (vh1 >> 2) << 2,
        _ => (vh1 >> 3) << 2,
    };
    ((((vl >> sh) & 0x0303_0303) | 0x8080_8080).wrapping_sub(hb & 0x0404_0404)) ^ 0x8080_8080
}

/// The card's `q3k_sub_scales4` (crates/gpu/src/cores.rs) for one lane:
/// the four sub-block scales `[sc(s0), sc(s0+2), sc(s0+4), sc(s0+6)]`, each
/// a 6-bit field minus the 32 offset, from the lane-parity scales window.
#[inline(always)]
fn card_q3k_sub_scales(a0w: u32, a1w: u32, a2w: u32, s0: usize) -> [i32; 4] {
    let sh = (((s0 & 1) << 3) | ((s0 >> 3) << 2)) as u32;
    let p0 = ((a0w >> sh) & 0x000f_000f) | (((a2w >> sh) & 0x0003_0003) << 4);
    let p1 = ((a1w >> sh) & 0x000f_000f) | (((a2w >> (sh + 2)) & 0x0003_0003) << 4);
    [
        (p0 & 0xff) as i32 - 32,
        (p0 >> 16) as i32 - 32,
        (p1 & 0xff) as i32 - 32,
        (p1 >> 16) as i32 - 32,
    ]
}

/// The card's `q4k_scale_min` (crates/gpu/src/cores.rs): sub-block `s`'s
/// 6-bit scale and min out of the super-block's three scale words.
#[inline(always)]
fn card_q4k_scale_min(s: usize, w1: u32, w2: u32, w3: u32) -> (i32, i32) {
    if s < 4 {
        let sh = 8 * s as u32;
        (((w1 >> sh) & 63) as i32, ((w2 >> sh) & 63) as i32)
    } else {
        let sh = 8 * (s - 4) as u32;
        let qs = (w2 >> sh) & 0xff;
        let qs4 = (w3 >> sh) & 0xff;
        let qsm = (w1 >> sh) & 0xff;
        (
            ((qs4 & 0x0f) as i32) | (((qsm >> 6) as i32) << 4),
            ((qs4 >> 4) as i32) | (((qs >> 6) as i32) << 4),
        )
    }
}

/// The card's `q4k_nibble` (crates/gpu/src/cores.rs): the nibble field's
/// `nib - 8` signed bytes packed four to a u32.
#[inline(always)]
fn card_q4k_nibble(qsw: u32, nib_sh: u32) -> u32 {
    ((((qsw >> nib_sh) & 0x0f0f_0f0f) | 0x8080_8080).wrapping_sub(0x0808_0808)) ^ 0x8080_8080
}

/// The shape contract every card-rule dot entry point enforces, as a named
/// error before any load: at least one column and one output per column
/// ([`QdotError::TileShape`]), every column the first one's `k`
/// ([`QdotError::ShortActivationCol`]), and a row of exactly `blk · k/256`
/// bytes — a whole Q3_K (110) or Q4_K (144) row ([`QdotError::ShortWeightRow`]).
/// Returns the super-block count. Both walks read their windows as unaligned
/// bytes — the device's parity funnels reconstitute the same windows from its
/// aligned words — and each walk's guard skips the tail an odd super-block
/// count leaves, as the card's does.
fn card_dot_check(
    wrow: &[u8],
    blk: usize,
    cols: &[&CardQ81],
    out: &[f32],
) -> Result<usize, QdotError> {
    if cols.is_empty() || out.len() != cols.len() {
        return Err(QdotError::TileShape {
            cols: cols.len(),
            outs: out.len(),
        });
    }
    let k = cols[0].k;
    if let Some(c) = cols.iter().find(|c| c.k != k) {
        return Err(QdotError::ShortActivationCol {
            have: c.k,
            need: k,
            k,
        });
    }
    // CardQ81::for_k holds every form to whole 256-value super-blocks.
    let n_sb = k / 256;
    if wrow.len() != blk * n_sb {
        return Err(QdotError::ShortWeightRow {
            have: wrow.len(),
            need: blk * n_sb,
            k,
        });
    }
    Ok(n_sb)
}

/// The Q3_K walk one card lane runs (crates/gpu/src/cores.rs,
/// `q3k_row_dot_1col_span`): iteration `it` folds the lane's integer chain
/// through `q3k_acc` — `fma(mul(e0, drow), a as f32, f)`, the product rounded
/// first, both operations spelled on the device — into the lane's
/// accumulator. This is the value every host body computes per lane; the
/// lane partition is the gemv's: `w16 = lane & 15`, `half = lane >> 4`,
/// `s0 = 8·(w16>>3) + ((w16&7)>>2)`, `d8_base = w16>>3`, and field `f`'s
/// code word is value word
/// `128·it + 64·half + 32·d8_base + 16·(f>>1) + 8·(f&1) + (lane&7)`.
#[inline(always)]
fn card_q3k_lane_walk(wrow: &[u8], codes: &[i8], d8: &[f32], n_sb: usize, lane: usize) -> f32 {
    let iters = n_sb.div_ceil(2);
    let w16 = lane & 15;
    let half = lane >> 4;
    let s0 = 8 * (w16 >> 3) + ((w16 & 7) >> 2);
    let d8_base = w16 >> 3;
    let mut f = 0.0f32;
    for it in 0..iters {
        let sbp = 2 * it + half;
        // The walk's guard, as the card's; live on an odd tail.
        if sbp >= n_sb {
            continue;
        }
        let base = sbp * 110;
        // `q3k_sb_decode`'s windows, as unaligned byte reads: the device's
        // parity funnels reconstitute exactly these windows from its aligned
        // words, so the host needs no parity of its own.
        let vl = le32(wrow, base + 32 + 4 * w16);
        let hm = le32(wrow, base + 4 * (w16 & 7));
        let vh1 = (!hm) >> (4 * (w16 >> 3)) as u32;
        let (a0w, a1w, a2w) = (
            le32(wrow, base + 96),
            le32(wrow, base + 100),
            le32(wrow, base + 104),
        );
        let sc = card_q3k_sub_scales(a0w, a1w, a2w, s0);
        let drow = half_to_f32(le16(wrow, base + 108));
        let v4 = 128 * it + 64 * half + 32 * d8_base + (lane & 7);
        let mut a = 0i32;
        for (fld, &scf) in sc.iter().enumerate() {
            let w = card_word(codes, v4 + 16 * (fld >> 1) + 8 * (fld & 1));
            a += card_dp4a(card_q3k_field(vl, vh1, fld as u32), w) * scf;
        }
        let e0 = d8[2 * sbp + d8_base];
        f = (e0 * drow).mul_add(a as f32, f);
    }
    f
}

/// The Q4_K walk one card lane runs (crates/gpu/src/cores.rs,
/// `q4k_row_dot_1col`): sub-block value by `q4k_sub_value` —
/// `fma(cda, a as f32, cdb * (b as f32))`, the `cdb` product rounded first —
/// folded by `q4k_acc`'s rounding: the first iteration of each pair of the
/// guard-free prefix adds a separately rounded `e0·x`, every other iteration
/// contracts into `fma(e0, x, f)` (`Q4K_ITER_UNROLL` = 2). Lane partition:
/// `s = lane & 7` (the sub-block), `grp = lane >> 3`; chain word `i` reads
/// value word `256·it + 64·grp + 8·s + i`, and the lane's 32 codes are
/// group `32·it + lane` of `s8`, its block scale `d8[2·sbp + s>>2]`.
#[inline(always)]
fn card_q4k_lane_walk(wrow: &[u8], act: &CardQ81, n_sb: usize, lane: usize) -> f32 {
    let iters = n_sb.div_ceil(4);
    let full = n_sb / 4;
    let s = lane & 7;
    let grp = lane >> 3;
    let (codes, s8, d8) = (act.codes(), act.s8(), act.d8());
    let mut f = 0.0f32;
    for it in 0..iters {
        let sbp = 4 * it + grp;
        // The walk's guard, as the card's; live on an odd tail.
        if sbp >= n_sb {
            continue;
        }
        let sb = sbp * 144;
        let d = half_to_f32(le16(wrow, sb));
        let dmin = half_to_f32(le16(wrow, sb + 2));
        let (w1, w2, w3) = (le32(wrow, sb + 4), le32(wrow, sb + 8), le32(wrow, sb + 12));
        let (sc, mi) = card_q4k_scale_min(s, w1, w2, w3);
        let cda = d * sc as f32;
        let cdb = (8.0 * d).mul_add(sc as f32, -dmin * mi as f32);
        let nib_sh = (s as u32 & 1) * 4;
        // `q4k_sb_decode`'s qs window: words 4 + 8·(s>>1) of the super-block.
        let qsk = sb + 16 + 32 * (s >> 1);
        let mut a = 0i32;
        for i in 0..8 {
            a += card_dp4a(
                card_q4k_nibble(le32(wrow, qsk + 4 * i), nib_sh),
                card_word(codes, 256 * it + 64 * grp + 8 * s + i),
            );
        }
        let b = s8[32 * it + lane];
        let e0 = d8[2 * sbp + (s >> 2)];
        let x = cda.mul_add(a as f32, cdb * b as f32);
        let split = it & 1 == 0 && it + 2 <= full;
        f = if split { f + e0 * x } else { e0.mul_add(x, f) };
    }
    f
}

/// The scalar butterfly over 32 lane values (the mirror of
/// [`card_warp_sum`]): each step adds the partner's current value, so lane
/// 0 ends with the device tree's value.
#[inline(always)]
fn card_warp_sum_scalar(v: &mut [f32; 32]) -> f32 {
    for stride in [16, 8, 4, 2, 1] {
        for l in 0..32 {
            if l & stride == 0 {
                v[l] += v[l | stride];
            }
        }
    }
    v[0]
}

/// One Q3_K weight row (`110 · k/256` bytes) against one card-q8_1 column:
/// the card gemv's value — every lane's walk ([`card_q3k_lane_walk`]) summed
/// by the warp butterfly. AVX2 where available; the scalar mirror is
/// [`card_q3k_dot_row_scalar`].
pub fn card_q3k_dot_row(wrow: &[u8], act: &CardQ81) -> Result<f32, QdotError> {
    let mut out = [0.0f32];
    card_q3k_dot_row_cols(wrow, &[act], &mut out)?;
    Ok(out[0])
}

/// The scalar mirror behind [`card_q3k_dot_row`], and the oracle the AVX2
/// body's bit-identity gate compares against.
pub fn card_q3k_dot_row_scalar(wrow: &[u8], act: &CardQ81) -> Result<f32, QdotError> {
    let mut out = [0.0f32];
    card_q3k_dot_row_cols_scalar(wrow, &[act], &mut out)?;
    Ok(out[0])
}

/// The scalar walks of every lane over one row, one column.
fn card_q3k_dot_row_scalar_into(wrow: &[u8], n_sb: usize, cols: &[&CardQ81], out: &mut [f32]) {
    let mut v = [0.0f32; 32];
    for (c, col) in cols.iter().enumerate() {
        for (lane, slot) in v.iter_mut().enumerate() {
            *slot = card_q3k_lane_walk(wrow, col.codes(), col.d8(), n_sb, lane);
        }
        out[c] = card_warp_sum_scalar(&mut v);
    }
}

/// One Q3_K weight row against any number of card-q8_1 columns, the weight
/// decoded once per [`CARD_TILE_COLS`] of them: `out[j]` is
/// [`card_q3k_dot_row`] of column `j`, bit for bit — each column's lanes run
/// the same walk whatever else the call carries.
pub fn card_q3k_dot_row_cols(
    wrow: &[u8],
    cols: &[&CardQ81],
    out: &mut [f32],
) -> Result<(), QdotError> {
    let n_sb = card_dot_check(wrow, Q3K_BLOCK, cols, out)?;
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        // SAFETY: the features were just detected; card_dot_check pinned the
        // row to n_sb super-blocks and every column to k = 256·n_sb.
        unsafe { card_q3k_dot_row_cols_avx2(wrow, n_sb, cols, out) };
        return Ok(());
    }
    card_q3k_dot_row_scalar_into(wrow, n_sb, cols, out);
    Ok(())
}

/// The card-q8_1 columns one weight decode of the card-rule dots serves (the
/// card's tile width, `q4k_sel::TILE_COLS`); a call of more runs one tile
/// per this many.
pub const CARD_TILE_COLS: usize = 8;

/// The scalar mirror behind [`card_q3k_dot_row_cols`].
pub fn card_q3k_dot_row_cols_scalar(
    wrow: &[u8],
    cols: &[&CardQ81],
    out: &mut [f32],
) -> Result<(), QdotError> {
    let n_sb = card_dot_check(wrow, Q3K_BLOCK, cols, out)?;
    card_q3k_dot_row_scalar_into(wrow, n_sb, cols, out);
    Ok(())
}

/// # Safety
/// The CPU must support AVX2 and FMA; [`card_dot_check`] pinned `wrow` to
/// `n_sb` super-blocks, every column to `k = 256·n_sb` and `out` to one
/// value per column.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
unsafe fn card_q3k_dot_row_cols_avx2(wrow: &[u8], n_sb: usize, cols: &[&CardQ81], out: &mut [f32]) {
    let iters = n_sb.div_ceil(2);
    // SAFETY: AVX2+FMA+F16C present per the dispatchers' detection, and
    // card_dot_check pinned the row and column shapes; every load below
    // stays inside them by the offsets the walk computes.
    unsafe {
        let m3 = _mm256_set1_epi32(0x0303_0303);
        let m4b = _mm256_set1_epi32(0x0404_0404);
        let four8 = _mm256_set1_epi8(4);
        // Columns in tiles of CARD_TILE_COLS: each tile's accumulators live
        // in registers across the walk; a call of fewer columns runs one
        // shorter tile (every column an independent walk).
        for (c0, chunk) in cols.chunks(CARD_TILE_COLS).enumerate() {
            let mut acc = [[_mm256_setzero_ps(); 4]; CARD_TILE_COLS];
            for it in 0..iters {
                #[allow(
                    clippy::needless_range_loop,
                    reason = "`v` names the lane vector: it derives the decode windows and selects \
                              the accumulator slot acc[t][v]"
                )]
                for v in 0..4usize {
                    let half = v >> 1;
                    let b8 = v & 1;
                    let sbp = 2 * it + half;
                    if sbp >= n_sb {
                        continue;
                    }
                    let base = sbp * 110;
                    // SAFETY: the loads below stay inside wrow's checked
                    // 110*n_sb bytes by `base` and these windows.
                    let drow = half_to_f32(le16(wrow, base + 108));
                    // The eight lanes' qs and hmask words are each 32
                    // consecutive bytes: w16 = l + 8*b8 gives qs bytes
                    // 32 + 32*b8 + 4l, hmask bytes 4l.
                    let vl = _mm256_loadu_si256(
                        wrow.as_ptr().add(base + 32 + 32 * b8) as *const __m256i
                    );
                    let hm = _mm256_loadu_si256(wrow.as_ptr().add(base) as *const __m256i);
                    // hm >> (4*(w16>>3)) with w16>>3 = b8, uniform per vector:
                    // the hmask bits themselves, not the card's inverted
                    // `vh1` — the unsigned weight word below adds 4 where
                    // the bit is set.
                    let vh = if b8 == 0 {
                        hm
                    } else {
                        _mm256_srli_epi32(hm, 4)
                    };
                    // The scales window and the two per-lane shifts of
                    // `q3k_sub_scales4` (sh = 8*((l>>2)&1) | 4*b8): the
                    // same unaligned bytes whatever the super-block's
                    // parity.
                    let (a0w, a1w, a2w) = (
                        le32(wrow, base + 96),
                        le32(wrow, base + 100),
                        le32(wrow, base + 104),
                    );
                    // Each lane's four scales as i16 pairs (the scale in both
                    // halves of its i32), the multiplier `madd` takes below.
                    let sh = [4 * b8, 4 * b8 + 8];
                    let mut sc = [_mm256_setzero_si256(); 4];
                    for (h, &s) in sh.iter().enumerate() {
                        let s = s as u32;
                        let p0 = ((a0w >> s) & 0x000f_000f) | (((a2w >> s) & 0x0003_0003) << 4);
                        let p1 =
                            ((a1w >> s) & 0x000f_000f) | (((a2w >> (s + 2)) & 0x0003_0003) << 4);
                        let vals = [
                            (p0 & 0xff) as i32 - 32,
                            (p0 >> 16) as i32 - 32,
                            (p1 & 0xff) as i32 - 32,
                            (p1 >> 16) as i32 - 32,
                        ];
                        for fld in 0..4 {
                            let v16 = vals[fld] as u32 & 0xffff;
                            let bcast = _mm256_set1_epi32((v16 | (v16 << 16)) as i32);
                            // lanes 4h..4h+3 take this shift's scales.
                            sc[fld] = if h == 0 {
                                _mm256_blend_epi32(sc[fld], bcast, 0x0f)
                            } else {
                                _mm256_blend_epi32(sc[fld], bcast, 0xf0)
                            };
                        }
                    }
                    // Field f's weight word `w' = nib | 4*hbit` (unsigned,
                    // 0..7, `hbit` the hmask bit): the card's signed weight
                    // `nib - 4*(1-hbit)` is `w' - 4`. Per byte pair,
                    // `maddubs(w', cw) - maddubs(4, cw)` is that weight's
                    // products as an i16 (both terms under 2·7·127), and `madd`
                    // by the lane's scale pair sums them into the card's
                    // `dp4a(w, cw)·sc` — integers, exact in any order. The
                    // shifts are immediates, so the four fields unroll.
                    let v4base = 128 * it + 64 * half + 32 * b8;
                    let mut a = [_mm256_setzero_si256(); CARD_TILE_COLS];
                    // (field, vl shift, hbit selector, value-word offset in words)
                    let terms: [(u32, u32, u8, usize); 4] =
                        [(0, 0, 0, 0), (1, 2, 1, 8), (2, 4, 2, 16), (3, 6, 3, 24)];
                    for (fld, vsh, hsel, woff) in terms {
                        let nib = match vsh {
                            0 => _mm256_and_si256(vl, m3),
                            2 => _mm256_and_si256(_mm256_srli_epi32(vl, 2), m3),
                            4 => _mm256_and_si256(_mm256_srli_epi32(vl, 4), m3),
                            _ => _mm256_and_si256(_mm256_srli_epi32(vl, 6), m3),
                        };
                        let hb = match hsel {
                            0 => _mm256_slli_epi32(vh, 2),
                            1 => _mm256_slli_epi32(vh, 1),
                            2 => vh,
                            _ => _mm256_srli_epi32(vh, 1),
                        };
                        let w8 = _mm256_or_si256(nib, _mm256_and_si256(hb, m4b));
                        let off = 4 * (v4base + woff);
                        for (t, col) in chunk.iter().enumerate() {
                            // 32 readable code bytes of the column: v4base +
                            // woff + 7 <= 64·sbp + 63 < 64·n_sb, its words.
                            let cw =
                                _mm256_loadu_si256(col.codes().as_ptr().add(off) as *const __m256i);
                            let p = _mm256_sub_epi16(
                                _mm256_maddubs_epi16(w8, cw),
                                _mm256_maddubs_epi16(four8, cw),
                            );
                            a[t] = _mm256_add_epi32(a[t], _mm256_madd_epi16(p, sc[fld as usize]));
                        }
                    }
                    for (t, col) in chunk.iter().enumerate() {
                        let e0 = *col.d8().get_unchecked(2 * sbp + b8);
                        let ed = e0 * drow;
                        acc[t][v] = _mm256_fmadd_ps(
                            _mm256_set1_ps(ed),
                            _mm256_cvtepi32_ps(a[t]),
                            acc[t][v],
                        );
                    }
                }
            }
            for (t, _) in chunk.iter().enumerate() {
                out[c0 * CARD_TILE_COLS + t] = card_warp_sum(acc[t]);
            }
        }
    }
}

/// One Q4_K weight row (`144 · k/256` bytes) against one card-q8_1 column:
/// the card down gemv's value — every lane's walk ([`card_q4k_lane_walk`])
/// summed by the warp butterfly. AVX2 where available; the scalar mirror is
/// [`card_q4k_dot_row_scalar`].
pub fn card_q4k_dot_row(wrow: &[u8], act: &CardQ81) -> Result<f32, QdotError> {
    let mut out = [0.0f32];
    card_q4k_dot_row_cols(wrow, &[act], &mut out)?;
    Ok(out[0])
}

/// The scalar mirror behind [`card_q4k_dot_row`], and the oracle the AVX2
/// body's bit-identity gate compares against.
pub fn card_q4k_dot_row_scalar(wrow: &[u8], act: &CardQ81) -> Result<f32, QdotError> {
    let mut out = [0.0f32];
    card_q4k_dot_row_cols_scalar(wrow, &[act], &mut out)?;
    Ok(out[0])
}

/// One Q4_K weight row against any number of card-q8_1 columns, the weight
/// decoded once per [`CARD_TILE_COLS`] of them: `out[j]` is
/// [`card_q4k_dot_row`] of column `j`, bit for bit.
pub fn card_q4k_dot_row_cols(
    wrow: &[u8],
    cols: &[&CardQ81],
    out: &mut [f32],
) -> Result<(), QdotError> {
    let n_sb = card_dot_check(wrow, Q4K_BLOCK, cols, out)?;
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        // SAFETY: the features were just detected; card_dot_check pinned the
        // row to n_sb super-blocks and every column to k = 256·n_sb.
        unsafe { card_q4k_dot_row_cols_avx2(wrow, n_sb, cols, out) };
        return Ok(());
    }
    card_q4k_dot_row_scalar_into(wrow, n_sb, cols, out);
    Ok(())
}

/// The scalar mirror behind [`card_q4k_dot_row_cols`].
pub fn card_q4k_dot_row_cols_scalar(
    wrow: &[u8],
    cols: &[&CardQ81],
    out: &mut [f32],
) -> Result<(), QdotError> {
    let n_sb = card_dot_check(wrow, Q4K_BLOCK, cols, out)?;
    card_q4k_dot_row_scalar_into(wrow, n_sb, cols, out);
    Ok(())
}

/// The scalar walks of every lane over one Q4_K row.
fn card_q4k_dot_row_scalar_into(wrow: &[u8], n_sb: usize, cols: &[&CardQ81], out: &mut [f32]) {
    let mut v = [0.0f32; 32];
    for (c, col) in cols.iter().enumerate() {
        for (lane, slot) in v.iter_mut().enumerate() {
            *slot = card_q4k_lane_walk(wrow, col, n_sb, lane);
        }
        out[c] = card_warp_sum_scalar(&mut v);
    }
}

/// # Safety
/// The CPU must support AVX2 and FMA; [`card_dot_check`] pinned `wrow` to
/// `n_sb` super-blocks, every column to `k = 256·n_sb` and `out` to one
/// value per column.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
unsafe fn card_q4k_dot_row_cols_avx2(wrow: &[u8], n_sb: usize, cols: &[&CardQ81], out: &mut [f32]) {
    let iters = n_sb.div_ceil(4);
    let full = n_sb / 4;
    // SAFETY: AVX2+FMA+F16C present per the dispatchers' detection, and
    // card_dot_check pinned the row and column shapes; every load below
    // stays inside them by the offsets the walk computes.
    unsafe {
        let ones16 = _mm256_set1_epi16(1);
        let lo_mask = _mm256_set1_epi8(0x0f);
        for (c0, chunk) in cols.chunks(CARD_TILE_COLS).enumerate() {
            let mut acc = [[_mm256_setzero_ps(); 4]; CARD_TILE_COLS];
            for it in 0..iters {
                #[allow(
                    clippy::needless_range_loop,
                    reason = "`v` names the lane vector: it derives the super-block and selects \
                              the accumulator slot acc[t][v]"
                )]
                for v in 0..4usize {
                    // This vector's eight lanes, s = 0..7, share super-block
                    // sbp = 4*it + v.
                    let sbp = 4 * it + v;
                    if sbp >= n_sb {
                        continue;
                    }
                    let sb = sbp * 144;
                    let d = half_to_f32(le16(wrow, sb));
                    let dmin = half_to_f32(le16(wrow, sb + 2));
                    let (w1, w2, w3) =
                        (le32(wrow, sb + 4), le32(wrow, sb + 8), le32(wrow, sb + 12));
                    // The eight lanes' (sc, mi) and the two chain coefficients.
                    let mut scm = [0i32; 8];
                    let mut mim = [0i32; 8];
                    for s in 0..8 {
                        (scm[s], mim[s]) = card_q4k_scale_min(s, w1, w2, w3);
                    }
                    let scv =
                        _mm256_cvtepi32_ps(_mm256_loadu_si256(scm.as_ptr() as *const __m256i));
                    let miv =
                        _mm256_cvtepi32_ps(_mm256_loadu_si256(mim.as_ptr() as *const __m256i));
                    // cda = d*sc (one rounded product); cdb = fma(8d, sc,
                    // -(dmin*mi)): the PTX's spelling of `q4k_coeff`.
                    let cda = _mm256_mul_ps(_mm256_set1_ps(d), scv);
                    let cdb = _mm256_fmadd_ps(
                        _mm256_set1_ps(8.0 * d),
                        scv,
                        _mm256_mul_ps(_mm256_set1_ps(-dmin), miv),
                    );
                    // Lane s's 32 unsigned nibbles: qs bytes 32·(s>>1) ..
                    // +32 of the super-block, the low nibbles for even s, the
                    // high for odd — the window `q4k_sb_decode` reads.
                    let mut nib = [_mm256_setzero_si256(); 8];
                    for j in 0..4 {
                        let q = _mm256_loadu_si256(
                            wrow.as_ptr().add(sb + 16 + 32 * j) as *const __m256i
                        );
                        nib[2 * j] = _mm256_and_si256(q, lo_mask);
                        nib[2 * j + 1] = _mm256_and_si256(_mm256_srli_epi16(q, 4), lo_mask);
                    }
                    for (t, col) in chunk.iter().enumerate() {
                        // Lane s's codes are value words 256·it + 64·v + 8·s
                        // .. +7: 32 consecutive bytes. Its A chain, Σ (nib −
                        // 8)·code, is Σ nib·code − 8·B with B its s8 group
                        // (the exact sum of the same 32 codes): integers, so
                        // any order gives the card's dp4a chain's value.
                        let cb = col.codes().as_ptr().add(4 * (256 * it + 64 * v));
                        let mut p = [_mm256_setzero_si256(); 8];
                        for (s, ps) in p.iter_mut().enumerate() {
                            let cw = _mm256_loadu_si256(cb.add(32 * s) as *const __m256i);
                            *ps = _mm256_madd_epi16(_mm256_maddubs_epi16(nib[s], cw), ones16);
                        }
                        // The eight lanes' sums into i32 lanes 0..7.
                        let h01 = _mm256_hadd_epi32(p[0], p[1]);
                        let h23 = _mm256_hadd_epi32(p[2], p[3]);
                        let h45 = _mm256_hadd_epi32(p[4], p[5]);
                        let h67 = _mm256_hadd_epi32(p[6], p[7]);
                        let q03 = _mm256_hadd_epi32(h01, h23);
                        let q47 = _mm256_hadd_epi32(h45, h67);
                        let sums = _mm256_add_epi32(
                            _mm256_permute2x128_si256(q03, q47, 0x20),
                            _mm256_permute2x128_si256(q03, q47, 0x31),
                        );
                        // B: groups 32·it + 8·v + s, eight consecutive i32.
                        let bi = _mm256_loadu_si256(
                            col.s8().as_ptr().add(32 * it + 8 * v) as *const __m256i
                        );
                        let a = _mm256_sub_epi32(sums, _mm256_slli_epi32(bi, 3));
                        let x = _mm256_fmadd_ps(
                            cda,
                            _mm256_cvtepi32_ps(a),
                            _mm256_mul_ps(cdb, _mm256_cvtepi32_ps(bi)),
                        );
                        // e0 = d8[2·sbp + s>>2]: lanes 0..3 block 2·sbp, 4..7
                        // block 2·sbp + 1.
                        let d8 = col.d8();
                        let e0 = _mm256_set_m128(
                            _mm_set1_ps(*d8.get_unchecked(2 * sbp + 1)),
                            _mm_set1_ps(*d8.get_unchecked(2 * sbp)),
                        );
                        let split = it & 1 == 0 && it + 2 <= full;
                        acc[t][v] = if split {
                            _mm256_add_ps(acc[t][v], _mm256_mul_ps(e0, x))
                        } else {
                            _mm256_fmadd_ps(e0, x, acc[t][v])
                        };
                    }
                }
            }
            for (t, _) in chunk.iter().enumerate() {
                out[c0 * CARD_TILE_COLS + t] = card_warp_sum(acc[t]);
            }
        }
    }
}

/// The device `expf_ik` (crates/gpu/src/linear/mod.rs), one lane, op for
/// op: ik's AVX2 `v_expf` as scalar Rust. Every product that meets an add is
/// a `mul_add` and the escape paths are the reference's, so it equals the
/// vector path [`v_expf`] bit for bit — the pure gate sweeps f32 bit
/// patterns over the whole relevant exponent range and holds them equal.
#[inline]
#[must_use]
pub fn expf_ik_scalar(x: f32) -> f32 {
    const SHIFT: f32 = f32::from_bits(0x4b40_0000); // 0x1.8p23
    const LOG2E: f32 = f32::from_bits(0x3fb8_aa3b); // 0x1.715476p+0
    const LN2_HI: f32 = f32::from_bits(0x3f31_7200); // 0x1.62e4p-1
    const LN2_LO: f32 = f32::from_bits(0x35bf_be8e); // 0x1.7f7d1cp-20
    const C0: f32 = f32::from_bits(0x3f7f_fff6); // 0x1.ffffecp-1
    const C1: f32 = f32::from_bits(0x3eff_fedb); // 0x1.fffdb6p-2
    const C2: f32 = f32::from_bits(0x3e2a_af33); // 0x1.555e66p-3
    const C3: f32 = f32::from_bits(0x3d2b_9f17); // 0x1.573e2ep-5
    const C4: f32 = f32::from_bits(0x3c07_2010); // 0x1.0e4020p-7
    let z = x.mul_add(LOG2E, SHIFT);
    let n = z - SHIFT;
    let b = (-n).mul_add(LN2_LO, (-n).mul_add(LN2_HI, x));
    let e = z.to_bits() << 23;
    let k = f32::from_bits(e.wrapping_add(0x3f80_0000));
    let u = b * b;
    let j = C4
        .mul_add(b, C3)
        .mul_add(u, C2.mul_add(b, C1))
        .mul_add(u, C0 * b);
    // An ordered compare, as the device's: a NaN `n` takes the main path.
    if n.abs() > 126.0 {
        let g: u32 = if n <= 0.0 { 0x8200_0000 } else { 0 };
        let s1 = f32::from_bits(g.wrapping_add(0x7f00_0000));
        let s2 = f32::from_bits(e.wrapping_sub(g));
        return if n.abs() > 192.0 {
            s1 * s1
        } else {
            s2.mul_add(j, s2) * s1
        };
    }
    j.mul_add(k, k)
}

/// The card's clamped SwiGLU for one value (crates/gpu-deepseek41/src/
/// experts.rs, `swiglu_clamp`): `min(silu(g), limit) · clamp(u, -limit,
/// limit)` — the clamp bites on the silu output, and neither applies when
/// `limit <= 1e-6`. The `min`/`max` are the device's comparison chains: a
/// NaN silu passes its clamp, a NaN up becomes `limit`. The silu is
/// `g / (1 + expf_ik(0 - g))` with the scalar mirror of the device
/// exponential.
#[inline]
#[must_use]
pub fn card_swiglu_clamp_1(g: f32, u: f32, limit: f32) -> f32 {
    let mut s = g / (1.0 + expf_ik_scalar(0.0 - g));
    let mut uc = u;
    if limit > 1e-6 {
        s = if limit < s { limit } else { s };
        uc = if u < limit { u } else { limit };
        uc = if -limit < uc { uc } else { -limit };
    }
    uc * s
}

/// The card's clamped SwiGLU over slices, AVX2 where available: `out[i] =
/// card_swiglu_clamp_1(gate[i], up[i], limit)` — the AVX2 lanes are
/// `v_silu` / [`v_expf`], which the pure gate holds equal to
/// [`expf_ik_scalar`] bit for bit, and the clamp's `min_ps`/`max_ps`
/// operand orders spell the same comparison chains as the scalar.
pub fn card_swiglu_clamp(gate: &[f32], up: &[f32], limit: f32, out: &mut [f32]) {
    assert!(
        gate.len() == up.len() && gate.len() == out.len(),
        "card_swiglu_clamp: gate, up and out are one length"
    );
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        // SAFETY: the features were just detected; the slices are equal
        // length, and both bodies spell the same value per lane as
        // `card_swiglu_clamp_1`.
        unsafe {
            if limit > 1e-6 {
                swiglu_clamp_avx2(gate, up, limit, out);
            } else {
                swiglu_avx2(gate, up, out);
            }
        };
        return;
    }
    for ((o, &g), &u) in out.iter_mut().zip(gate).zip(up) {
        *o = card_swiglu_clamp_1(g, u, limit);
    }
}

/// The exponential [`card_swiglu_clamp`]'s silu runs, over a slice: `v_expf`
/// eight lanes at a time where AVX2 is available (a zero-padded last vector),
/// [`expf_ik_scalar`] elsewhere — the one door its gate holds to
/// [`expf_ik_scalar`] bit for bit.
pub fn card_expf(x: &[f32], out: &mut [f32]) {
    assert_eq!(x.len(), out.len(), "card_expf: one output per input");
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        // SAFETY: the features were just detected; the slices are one length.
        unsafe { card_expf_avx2(x, out) };
        return;
    }
    for (o, &v) in out.iter_mut().zip(x) {
        *o = expf_ik_scalar(v);
    }
}

/// # Safety
/// The CPU must support AVX2 and FMA, and `x` and `out` must be one length.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn card_expf_avx2(x: &[f32], out: &mut [f32]) {
    let (body, tail) = x.as_chunks::<8>();
    // SAFETY: vector `i` reads x[8i .. 8i + 8] and writes the same span of
    // `out`, the same length as `x`; the tail goes through 8-lane arrays.
    unsafe {
        for (i, v) in body.iter().enumerate() {
            let e = v_expf(_mm256_loadu_ps(v.as_ptr()));
            _mm256_storeu_ps(out.as_mut_ptr().add(8 * i), e);
        }
        if !tail.is_empty() {
            let (mut xp, mut op) = ([0.0f32; 8], [0.0f32; 8]);
            xp[..tail.len()].copy_from_slice(tail);
            _mm256_storeu_ps(op.as_mut_ptr(), v_expf(_mm256_loadu_ps(xp.as_ptr())));
            out[8 * body.len()..].copy_from_slice(&op[..tail.len()]);
        }
    }
}

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
            let amax = gguf::quant::activation_max(xb, "qdot").abs();
            let t = fp32_to_bf16_bits(amax / 127.0);
            group[2 * ir..2 * ir + 2].copy_from_slice(&t.to_le_bytes());
            let id = q82_inverse(t);
            let qs = &mut group[16 + 32 * ir..16 + 32 * ir + 32];
            let mut isum = 0i32;
            for (m, &v) in xb.iter().enumerate() {
                let q = nearest_int(v * id);
                qs[m] = q.clamp(-128, 127) as i8 as u8;
                isum += q;
            }
            group[8 + 2 * ir..8 + 2 * ir + 2].copy_from_slice(&(isum as i16).to_le_bytes());
        }
    }
    for (t, xb) in blocks[nb4..].iter().enumerate() {
        let tb = &mut out[nb4 / 4 * Q82X4_STRIDE + t * Q82_BLOCK..][..Q82_BLOCK];
        let amax = gguf::quant::activation_max(xb, "qdot").abs();
        let b = fp32_to_bf16_bits(amax / 127.0);
        tb[0..2].copy_from_slice(&b.to_le_bytes());
        let id = q82_inverse(b);
        let qs = &mut tb[4..36];
        let mut isum = 0i32;
        for (m, &v) in xb.iter().enumerate() {
            let q = nearest_int(v * id);
            qs[m] = q.clamp(-128, 127) as i8 as u8;
            isum += q;
        }
        tb[2..4].copy_from_slice(&(isum as i16).to_le_bytes());
    }
}

/// One 32-value block of the q8_2 family: (codes, bf16 d bits, i16 isum),
/// byte-identical to the scalar loops in `quantize_q82x4_col`: [`quantize_q8_block_avx2`]
/// under the bf16 scale and [`q82_inverse`]'s multiplier.
///
/// # Safety
/// AVX2 must be available on the target; `x` is the fixed 32-value block.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn quantize_q82_block_avx2(x: &[f32; 32]) -> ([u8; 32], u16, i16) {
    // SAFETY: AVX2 present per contract.
    let (codes, t, isum) = unsafe {
        quantize_q8_block_avx2(x, |amax| {
            let t = fp32_to_bf16_bits(amax / 127.0);
            (t, q82_inverse(t))
        })
    };
    (codes, t, isum as i16)
}

/// One 32-value q8 block under the scale rule `rule`: `rule(amax)` gives the
/// block's stored scale bits and its code multiplier `id`, and the result is
/// (codes, scale bits, i32 sum of the unsaturated codes). `v * id` is a plain
/// multiply (no FMA); `id` is finite, so every product is finite and
/// [`v_nearest_int`] exact; the codes are the saturating packs of the
/// unsaturated ints, the scalar's clamp. A NaN or an infinity in the block is
/// refused by name ([`non_finite_activation`]).
///
/// # Safety
/// AVX2 must be available on the target; `x` is the fixed 32-value block.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn quantize_q8_block_avx2(
    x: &[f32; 32],
    rule: impl FnOnce(f32) -> (u16, f32),
) -> ([u8; 32], u16, i32) {
    // SAFETY: AVX2 present per contract; loads stay inside the fixed 32-value
    // block, the one store goes to a local array.
    unsafe {
        let sgn = _mm256_set1_ps(-0.0);
        let mut m = _mm256_setzero_ps();
        let mut unord = _mm256_setzero_ps();
        for j in 0..4 {
            // SAFETY: 8-lane load at 8*j <= 24, inside the block.
            let v = _mm256_loadu_ps(x.as_ptr().add(8 * j));
            // Accumulator second so `m` never holds a NaN; the NaN itself is
            // collected in `unord` and refused below.
            m = _mm256_max_ps(_mm256_andnot_ps(sgn, v), m);
            unord = _mm256_or_ps(unord, _mm256_cmp_ps(v, v, _CMP_UNORD_Q));
        }
        let amax = hmax_ps(m);
        if _mm256_movemask_ps(unord) != 0 || !amax.is_finite() {
            non_finite_activation(x, "qdot");
        }
        // The scale rule stays scalar: one conversion per block.
        let (t, id) = rule(amax);
        let idv = _mm256_set1_ps(id);
        let mut qi = [_mm256_setzero_si256(); 4];
        for (j, q) in qi.iter_mut().enumerate() {
            // SAFETY: 8-lane load at 8*j <= 24, inside the block.
            let v = _mm256_loadu_ps(x.as_ptr().add(8 * j));
            let y = _mm256_mul_ps(v, idv);
            *q = v_nearest_int(y);
        }
        // In-order i32 -> i8: two saturating packs plus the 32-lane fixup;
        // the saturation is the scalar's clamp.
        let p0 = _mm256_packs_epi32(qi[0], qi[1]);
        let p1 = _mm256_packs_epi32(qi[2], qi[3]);
        let c = _mm256_packs_epi16(p0, p1);
        let c = _mm256_permutevar8x32_epi32(c, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
        let mut codes = [0u8; 32];
        // SAFETY: 32-byte store into the local 32-byte array.
        _mm256_storeu_si256(codes.as_mut_ptr() as *mut __m256i, c);
        // i32 sum of the 32 unsaturated ints; exact in any order at these
        // magnitudes.
        let isum = hsum_i32(_mm256_add_epi32(
            _mm256_add_epi32(qi[0], qi[1]),
            _mm256_add_epi32(qi[2], qi[3]),
        ));
        (codes, t, isum)
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
#[cfg(target_arch = "x86_64")]
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

/// Quantizes activations into `block_q8_0` blocks (34 bytes / 32 values: f16
/// `d`, 32 i8 codes) by mainline's x86 `quantize_row_q8_0` (llama.cpp
/// `f2cc7282c` ggml-cpu/arch/x86/quants.c:302), the column its
/// `ggml_vec_dot_q2_0_q8_0` reads: `d = f16(amax / 127)` rounded to nearest
/// even (mainline's F16C conversion, [`gguf::quant::f32_to_f16_bits`] here), codes
/// `round_half_even(v * id)` with [`gguf::quant::q80_inverse`]'s `id`.
/// So a block whose `d` rounds to an f16 zero keeps its codes. The reference
/// encoder `quantize_row_q8_0_ref` differs: it codes with `1 / d` of the f32
/// `d = amax / 127`, before its f16 rounding, and rounds half away from zero.
fn quantize_q8f0_col(x: &[f32], out: &mut [u8]) {
    let (blocks, _) = x.as_chunks::<32>();
    let (oblocks, _) = out.as_chunks_mut::<Q8F0_BLOCK>();
    for (xb, ob) in blocks.iter().zip(oblocks) {
        ob.copy_from_slice(&gguf::quant::quantize_q8_0(xb).to_bytes());
    }
}

/// AVX2 twin of `quantize_q8f0_col`, byte-identical by construction: each
/// block through [`quantize_q8_block_avx2`] under the f16 scale and
/// [`q80_inverse`]'s multiplier: mainline's x86 steps, `_mm256_mul_ps`, a
/// round to nearest even ([`v_nearest_int`], the integers mainline's
/// `_mm256_round_ps` gives) and the saturating packs.
///
/// # Safety
/// CPU must support AVX2; `x.len()` must be a multiple of 32 and `out.len()`
/// equal `col_bytes(Q2_0, x.len())` (both enforced by [`quantize_col_check`]).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn quantize_q8f0_col_avx2(x: &[f32], out: &mut [u8]) {
    let (blocks, _) = x.as_chunks::<32>();
    let (oblocks, _) = out.as_chunks_mut::<Q8F0_BLOCK>();
    for (xb, ob) in blocks.iter().zip(oblocks) {
        // SAFETY: AVX2 present per contract; `xb` is a whole 32-value block.
        let (codes, d, _) = unsafe {
            quantize_q8_block_avx2(xb, |amax| {
                (f32_to_f16_bits(amax / 127.0), q80_inverse(amax))
            })
        };
        ob[0..2].copy_from_slice(&d.to_le_bytes());
        ob[2..].copy_from_slice(&codes);
    }
}

/// f16 bits to f32 through F16C (`vcvtph2ps`), the reference's `GGML_FP16_TO_FP32`
/// on this build. The conversion is exact, so it equals [`half_to_f32`] (the scalar
/// mirrors' converter) on every f16 but a NaN's payload.
///
/// # Safety
/// F16C must be available on the target; being `#[inline(always)]`, it takes it
/// from its caller's features.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn f16c_to_f32(h: u16) -> f32 {
    // SAFETY: register-only intrinsics, no memory access.
    unsafe { _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(i32::from(h)))) }
}

/// Two adjacent little-endian f16s (a K-quant block's `d` and `dmin`) to f32
/// through one 32-bit move and one `vcvtph2ps`. [`f16c_to_f32`] on each would
/// insert each half with a `vpinsrw`, which merges into a register LLVM picks
/// and so waits for that register's last write (in the row-lane pack, the
/// previous row's mins).
///
/// # Safety
/// F16C and SSE3 must be available on the target; being `#[inline(always)]`,
/// it takes them from its caller's features.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn f16c_pair_to_f32(h: [u8; 4]) -> (f32, f32) {
    // SAFETY: SSE2 present per contract; register-only, no memory is touched.
    let halves = unsafe { _mm_cvtsi32_si128(i32::from_le_bytes(h)) };
    // SAFETY: F16C present per contract; register-only, no memory is touched.
    let v = unsafe { _mm_cvtph_ps(halves) };
    // SAFETY: SSE3 present per contract; register-only, no memory is touched.
    let high = unsafe { _mm_movehdup_ps(v) };
    // SAFETY: SSE present per contract; register-only, no memory is touched.
    let d = unsafe { _mm_cvtss_f32(v) };
    // SAFETY: SSE present per contract; register-only, no memory is touched.
    let dmin = unsafe { _mm_cvtss_f32(high) };
    (d, dmin)
}

/// Horizontal f32 sum of an `__m256`. Reduction order is load-bearing.
///
/// # Safety
/// AVX and SSE3 must be available on the target.
#[cfg(target_arch = "x86_64")]
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

/// Super-block `i`'s column factors in the Q4_K and Q5_K kernels, read from
/// the heads of its two Q8_2_X4 groups: `dy` holds the eight blocks' bf16
/// scales as f32, and `my` is `dy` times the blocks' i16 sums.
///
/// # Safety
/// `col` must point at a column whose groups `2 * i` and `2 * i + 1` are
/// readable.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn q82x4_dy_my(col: *const u8, i: usize) -> (__m256, __m256) {
    // SAFETY: group 2 * i lies in the column per contract.
    let g0 = unsafe { col.add((2 * i) * Q82X4_STRIDE) };
    // SAFETY: group 2 * i + 1 lies in the column per contract.
    let g1 = unsafe { col.add((2 * i + 1) * Q82X4_STRIDE) };
    // SAFETY: reads group 2 * i's four scales, its bytes 0..8.
    let d4_1 = _mm_cvtepu16_epi32(unsafe { _mm_loadl_epi64(g0 as *const __m128i) });
    // SAFETY: reads group 2 * i + 1's four scales, its bytes 0..8.
    let d4_2 = _mm_cvtepu16_epi32(unsafe { _mm_loadl_epi64(g1 as *const __m128i) });
    let dy = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_set_m128i(d4_2, d4_1), 16));
    // SAFETY: byte 8 of group 2 * i, inside its 16-byte head.
    let s0 = unsafe { g0.add(8) };
    // SAFETY: byte 8 of group 2 * i + 1, inside its 16-byte head.
    let s1 = unsafe { g1.add(8) };
    // SAFETY: reads group 2 * i's four sums, its bytes 8..16.
    let m4_1 = _mm_cvtepi16_epi32(unsafe { _mm_loadl_epi64(s0 as *const __m128i) });
    // SAFETY: reads group 2 * i + 1's four sums, its bytes 8..16.
    let m4_2 = _mm_cvtepi16_epi32(unsafe { _mm_loadl_epi64(s1 as *const __m128i) });
    let my = _mm256_mul_ps(dy, _mm256_cvtepi32_ps(_mm256_set_m128i(m4_2, m4_1)));
    (dy, my)
}

/// A Q4_K or Q5_K block's eight sub-block scales, as f32, from bytes 0..8 of
/// its unpacked scales and mins `utmp` ([`make_q4_scales`]). The kernels
/// multiply them by the block's `d`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
fn q45k_scales_ps(utmp: &[u32; 4]) -> __m256 {
    // SAFETY: reads `utmp`'s bytes 0..8.
    let b = unsafe { _mm_loadl_epi64(utmp.as_ptr() as *const __m128i) };
    _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(b))
}

/// A Q4_K or Q5_K block's eight sub-block mins, as f32, from bytes 8..16 of
/// its unpacked scales and mins `utmp` ([`make_q4_scales`]). The kernels
/// multiply them by the block's `-dmin`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
fn q45k_mins_ps(utmp: &[u32; 4]) -> __m256 {
    // SAFETY: reads `utmp`'s bytes 8..16.
    let b = unsafe { _mm_loadl_epi64(utmp[2..].as_ptr() as *const __m128i) };
    _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(b))
}

/// AVX2+FMA row dot: Q4_K weights against Q8_2_X4 column.
/// i16: codes 0..15 against activations in [-127, 127], a maddubs pair peaks at
/// 2*15*127 = 3810; saturation is unreachable.
///
/// # Safety
/// Caller must ensure AVX2+FMA+F16C are available and buffers match `nb` blocks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
unsafe fn dot_q4k_q82x4_avx2(wrow: &[u8], acol: &[u8], nb: usize) -> f32 {
    // SAFETY: AVX2+FMA+F16C present and both slices hold nb blocks per contract.
    unsafe {
        let ml = _mm256_set1_epi8(0xF);
        let mut accd = _mm256_setzero_ps();

        for i in 0..nb {
            let wb = &wrow[i * Q4K_BLOCK..(i + 1) * Q4K_BLOCK];

            let (d, dmin) = f16c_pair_to_f32([wb[0], wb[1], wb[2], wb[3]]);

            let utmp = make_q4_scales(&wb[4..16]);
            let mins_v = q45k_mins_ps(&utmp);
            let mins = _mm256_mul_ps(_mm256_set1_ps(-dmin), mins_v);

            // SAFETY: the column holds super-block i's two validated groups.
            let (dy, my) = q82x4_dy_my(acol.as_ptr(), i);
            let dy4 = [_mm256_castps256_ps128(dy), _mm256_extractf128_ps(dy, 1)];

            let all_scales = _mm256_mul_ps(_mm256_set1_ps(d), q45k_scales_ps(&utmp));
            let lo = _mm256_castps256_ps128(all_scales);
            let hi = _mm256_extractf128_ps(all_scales, 1);
            let scales = [_mm256_set_m128(lo, lo), _mm256_set_m128(hi, hi)];
            let d4d8 = [
                _mm256_mul_ps(scales[0], _mm256_set_m128(dy4[0], dy4[0])),
                _mm256_mul_ps(scales[1], _mm256_set_m128(dy4[1], dy4[1])),
            ];

            let q4 = &wb[16..16 + 128];
            let mut sumi_f = [_mm256_setzero_ps(), _mm256_setzero_ps()];
            for (j, sumi_f_j) in sumi_f.iter_mut().enumerate() {
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
                *sumi_f_j = _mm256_cvtepi32_ps(sumi);
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

        let d = half_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
        let dmin = half_to_f32(u16::from_le_bytes([wb[2], wb[3]]));

        let utmp = make_q4_scales(&wb[4..16]);
        let sb = [utmp[0].to_le_bytes(), utmp[1].to_le_bytes()];
        let mb = [utmp[2].to_le_bytes(), utmp[3].to_le_bytes()];

        let mut mins = [0.0f32; 8];
        for l in 0..8 {
            let m = mb[l / 4][l % 4];
            mins[l] = -(dmin * (m as f32));
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
/// Caller must ensure AVX2+FMA+F16C are available and buffers match `nb` blocks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
unsafe fn dot_q6k_q82x4_avx2(wrow: &[u8], acol: &[u8], nb: usize) -> f32 {
    // SAFETY: AVX2+FMA+F16C present and both slices hold nb blocks per contract.
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
            let d = f16c_to_f32((wb.add(208) as *const u16).read_unaligned());
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
            for (j, sumi_j) in sumis.iter_mut().enumerate() {
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
                *sumi_j = _mm256_madd_epi16(_mm256_set1_epi16(1), t);
            }

            accd = _mm256_fmadd_ps(scales[0], _mm256_cvtepi32_ps(sumis[0]), accd);
            accd = _mm256_fmadd_ps(scales[1], _mm256_cvtepi32_ps(sumis[1]), accd);
        }

        hsum_float_8(accd)
    }
}

/// AVX2+FMA tile: one Q6_K weight row against `C` Q8_2_X4 columns, value `c`
/// equal to [`dot_q6k_q82x4_avx2`] of column `c` bit for bit.
///
/// The body is ik's `mul_mat_qY_K_q8_2_X4_T<DequantizerQ6K_AVX2, nrc_y = C>`
/// (iqk_gemm_kquants.cpp:938), whose `nrc_y = 1` instance the one-column
/// kernel ports: per block the weight's `d` and interleaved scales once; per
/// column its bf16 block scales; per 128-value half the four code vectors
/// (ql and qh folded, −32) and their absolute values once, then per column
/// the sign-folded maddubs sum and `accd = fma(scales_j · dy4, sumi, accd)`.
/// The one-column kernel multiplies the scales by the column's `dy` before the
/// fma and the tile multiplies them in the same order, so column `c` sees the
/// same products, the same single fma per half and the same horizontal sum;
/// the integer sums are the one-column instructions on the same codes.
///
/// # Safety
/// CPU must support AVX2+FMA+F16C; `wrow` must hold `nb` Q6_K blocks and each
/// `acols[c]` point at the `2 · nb` readable Q8_2_X4 groups of `nb`
/// super-blocks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
unsafe fn dot_q6k_q82x4_tile_avx2<const C: usize>(
    wrow: &[u8],
    acols: &[*const u8; C],
    nb: usize,
) -> [f32; C] {
    // SAFETY: AVX2+FMA+F16C present, `wrow` holds nb blocks and every column
    // 2·nb groups per contract.
    unsafe {
        let ml = _mm256_set1_epi8(0xF);
        let mh = _mm256_set1_epi8(0x30);
        let m32n = _mm256_set1_epi8(-32);
        // SAFETY: 32-byte load inside the 32-byte static table.
        let shuff = _mm256_loadu_si256(K_SHUFFLE_Q6K.as_ptr() as *const __m256i);
        let mut accd = [_mm256_setzero_ps(); C];

        for i in 0..nb {
            let wb = wrow.as_ptr().add(Q6K_BLOCK * i);

            // SAFETY: one unaligned u16 read inside the super-block.
            let d = f16c_to_f32((wb.add(208) as *const u16).read_unaligned());
            let vd = _mm256_set1_ps(d);
            // SAFETY: 16-byte unaligned load inside the super-block.
            let sc16 = _mm256_shuffle_epi8(
                _mm256_cvtepi8_epi16(_mm_loadu_si128(wb.add(192) as *const __m128i)),
                shuff,
            );
            let scales = [
                _mm256_mul_ps(
                    vd,
                    _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(sc16))),
                ),
                _mm256_mul_ps(
                    vd,
                    _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(sc16, 1))),
                ),
            ];

            let mut dy = [_mm256_setzero_ps(); C];
            for (col, dyc) in acols.iter().zip(dy.iter_mut()) {
                // SAFETY: 8 readable bytes at the head of each of the column's two groups.
                let g0 = col.add((2 * i) * Q82X4_STRIDE);
                let g1 = col.add((2 * i + 1) * Q82X4_STRIDE);
                let d4_1 = _mm_cvtepu16_epi32(_mm_loadl_epi64(g0 as *const __m128i));
                let d4_2 = _mm_cvtepu16_epi32(_mm_loadl_epi64(g1 as *const __m128i));
                *dyc = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_set_m128i(d4_2, d4_1), 16));
            }

            for (j, sj) in scales.iter().enumerate() {
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
                for (v, u) in values.iter_mut().zip(us.iter_mut()) {
                    *v = _mm256_add_epi8(*v, m32n);
                    *u = _mm256_sign_epi8(*v, *v);
                }

                for ((col, dyc), acc) in acols.iter().zip(&dy).zip(accd.iter_mut()) {
                    // SAFETY: loads stay inside the column's validated group 2i + j.
                    let qs = col.add((2 * i + j) * Q82X4_STRIDE + 16);
                    let sumi1 = _mm256_maddubs_epi16(
                        us[0],
                        _mm256_sign_epi8(_mm256_loadu_si256(qs as *const __m256i), values[0]),
                    );
                    let sumi2 = _mm256_maddubs_epi16(
                        us[1],
                        _mm256_sign_epi8(
                            _mm256_loadu_si256(qs.add(32) as *const __m256i),
                            values[1],
                        ),
                    );
                    let sumi3 = _mm256_maddubs_epi16(
                        us[2],
                        _mm256_sign_epi8(
                            _mm256_loadu_si256(qs.add(64) as *const __m256i),
                            values[2],
                        ),
                    );
                    let sumi4 = _mm256_maddubs_epi16(
                        us[3],
                        _mm256_sign_epi8(
                            _mm256_loadu_si256(qs.add(96) as *const __m256i),
                            values[3],
                        ),
                    );
                    let t1 = _mm256_add_epi16(
                        _mm256_unpacklo_epi32(sumi1, sumi2),
                        _mm256_unpackhi_epi32(sumi1, sumi2),
                    );
                    let t3 = _mm256_add_epi16(
                        _mm256_unpacklo_epi32(sumi3, sumi4),
                        _mm256_unpackhi_epi32(sumi3, sumi4),
                    );
                    let t = _mm256_add_epi16(
                        _mm256_unpacklo_epi64(t1, t3),
                        _mm256_unpackhi_epi64(t1, t3),
                    );
                    let sumi = _mm256_madd_epi16(_mm256_set1_epi16(1), t);
                    let dy4 = if j == 0 {
                        _mm256_castps256_ps128(*dyc)
                    } else {
                        _mm256_extractf128_ps(*dyc, 1)
                    };
                    let d4d8 = _mm256_mul_ps(*sj, _mm256_set_m128(dy4, dy4));
                    *acc = _mm256_fmadd_ps(d4d8, _mm256_cvtepi32_ps(sumi), *acc);
                }
            }
        }

        let mut out = [0.0f32; C];
        for (o, acc) in out.iter_mut().zip(&accd) {
            *o = hsum_float_8(*acc);
        }
        out
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

        let d = half_to_f32(u16::from_le_bytes([wb[208], wb[209]]));

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

// --------------------------------------------------------- Q5_K x Q8_2_X4

/// `block_q5_K` (ggml-common.h:367): d f16 @0, dmin f16 @2, scales u8[12] @4, qh u8[32] @16,
/// qs u8[128] @48.
const Q5K_BLOCK: usize = 176;

/// AVX2+FMA row dot: Q5_K weights against Q8_2_X4 column.
///
/// TWIN of [`dot_q4k_q82x4_avx2`] — ik runs both through `mul_mat_qX_K_q8_2_X4_T`,
/// and they differ in three places only: block width (176 vs 144), the nibble
/// offset (qs @48 vs @16), and `DequantizerQ5K_AVX2::prepare`'s fold of qh into
/// bit 4 of each code. In half j, value 128j + 32b + l takes bit 4j + b of qh[l]:
/// qh itself at j = 0, qh >> 4 (16-bit lanes) at j = 1, shifted left by 4 - b
/// onto bit 4. Every edit outside those three must be made in both.
/// i16: codes 0..31 against activations in [-127, 127]; after the two
/// `add_epi16` levels a lane holds 8 products, at most 8*31*127 = 31496, so
/// saturation is unreachable.
///
/// # Safety
/// Caller must ensure AVX2+FMA+F16C are available and buffers match `nb` blocks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
unsafe fn dot_q5k_q82x4_avx2(wrow: &[u8], acol: &[u8], nb: usize) -> f32 {
    // SAFETY: AVX2+FMA+F16C present and both slices hold nb blocks per contract.
    unsafe {
        let ml = _mm256_set1_epi8(0xF);
        let mh = _mm256_set1_epi8(0x10);
        let mut accd = _mm256_setzero_ps();

        for i in 0..nb {
            let wb = &wrow[i * Q5K_BLOCK..(i + 1) * Q5K_BLOCK];

            let (d, dmin) = f16c_pair_to_f32([wb[0], wb[1], wb[2], wb[3]]);

            let utmp = make_q4_scales(&wb[4..16]);
            let mins_v = q45k_mins_ps(&utmp);
            let mins = _mm256_mul_ps(_mm256_set1_ps(-dmin), mins_v);

            // SAFETY: the column holds super-block i's two validated groups.
            let (dy, my) = q82x4_dy_my(acol.as_ptr(), i);
            let dy4 = [_mm256_castps256_ps128(dy), _mm256_extractf128_ps(dy, 1)];

            let all_scales = _mm256_mul_ps(_mm256_set1_ps(d), q45k_scales_ps(&utmp));
            let lo = _mm256_castps256_ps128(all_scales);
            let hi = _mm256_extractf128_ps(all_scales, 1);
            let scales = [_mm256_set_m128(lo, lo), _mm256_set_m128(hi, hi)];
            let d4d8 = [
                _mm256_mul_ps(scales[0], _mm256_set_m128(dy4[0], dy4[0])),
                _mm256_mul_ps(scales[1], _mm256_set_m128(dy4[1], dy4[1])),
            ];

            // SAFETY: 32-byte load inside the validated 176-byte block.
            let qh = _mm256_loadu_si256(wb.as_ptr().add(16) as *const __m256i);
            let q5 = &wb[48..48 + 128];
            let mut sumi_f = [_mm256_setzero_ps(), _mm256_setzero_ps()];
            for (j, sf) in sumi_f.iter_mut().enumerate() {
                // SAFETY: loads stay inside the validated 176-byte block.
                let bits0 = _mm256_loadu_si256(q5.as_ptr().add(64 * j) as *const __m256i);
                let bits1 = _mm256_loadu_si256(q5.as_ptr().add(64 * j + 32) as *const __m256i);
                let hbits = if j == 0 {
                    qh
                } else {
                    _mm256_srli_epi16::<4>(qh)
                };
                let values0 = _mm256_or_si256(
                    _mm256_and_si256(bits0, ml),
                    _mm256_and_si256(_mm256_slli_epi16::<4>(hbits), mh),
                );
                let values1 = _mm256_or_si256(
                    _mm256_and_si256(_mm256_srli_epi16::<4>(bits0), ml),
                    _mm256_and_si256(_mm256_slli_epi16::<3>(hbits), mh),
                );
                let values2 = _mm256_or_si256(
                    _mm256_and_si256(bits1, ml),
                    _mm256_and_si256(_mm256_slli_epi16::<2>(hbits), mh),
                );
                let values3 = _mm256_or_si256(
                    _mm256_and_si256(_mm256_srli_epi16::<4>(bits1), ml),
                    _mm256_and_si256(_mm256_slli_epi16::<1>(hbits), mh),
                );

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
                *sf = _mm256_cvtepi32_ps(sumi);
            }

            accd = _mm256_fmadd_ps(my, mins, accd);
            accd = _mm256_fmadd_ps(d4d8[0], sumi_f[0], accd);
            accd = _mm256_fmadd_ps(d4d8[1], sumi_f[1], accd);
        }

        hsum_float_8(accd)
    }
}

/// Emulates `dot_q5k_q82x4_avx2` with bit identity: `dot_q4k_q82x4_emul` with
/// the qh fold's 16-bit lane shifts emulated as the kernel issues them.
fn dot_q5k_q82x4_emul(wrow: &[u8], acol: &[u8], nb: usize) -> f32 {
    let mut accd = [0.0f32; 8];

    for i in 0..nb {
        let wb = &wrow[i * Q5K_BLOCK..(i + 1) * Q5K_BLOCK];
        let g0 = &acol[(2 * i) * Q82X4_STRIDE..(2 * i + 1) * Q82X4_STRIDE];
        let g1 = &acol[(2 * i + 1) * Q82X4_STRIDE..(2 * i + 2) * Q82X4_STRIDE];

        let d = half_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
        let dmin = half_to_f32(u16::from_le_bytes([wb[2], wb[3]]));

        let utmp = make_q4_scales(&wb[4..16]);
        let sb = [utmp[0].to_le_bytes(), utmp[1].to_le_bytes()];
        let mb = [utmp[2].to_le_bytes(), utmp[3].to_le_bytes()];

        // The kernel's (-dmin) * m: negation is exact, so -(dmin * m) is the same bits.
        let mut mins = [0.0f32; 8];
        for l in 0..8 {
            let m = mb[l / 4][l % 4];
            mins[l] = -(dmin * (m as f32));
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

        let qh: [u8; 32] = wb[16..48].try_into().unwrap();
        let q5 = &wb[48..48 + 128];
        for j in 0..2 {
            let hbits = if j == 0 { qh } else { emul_srli_epi16::<4>(qh) };
            let high = [
                emul_slli_epi16::<4>(hbits),
                emul_slli_epi16::<3>(hbits),
                emul_slli_epi16::<2>(hbits),
                emul_slli_epi16::<1>(hbits),
            ];
            let mut values = [[0u8; 32]; 4];
            for m in 0..32 {
                let b0 = q5[64 * j + m];
                let b1 = q5[64 * j + 32 + m];
                values[0][m] = (b0 & 0xF) | (high[0][m] & 0x10);
                values[1][m] = (b0 >> 4) | (high[1][m] & 0x10);
                values[2][m] = (b1 & 0xF) | (high[2][m] & 0x10);
                values[3][m] = (b1 >> 4) | (high[3][m] & 0x10);
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

/// AVX2+FMA tile: one Q4_K (`Q5 = false`) or Q5_K (`Q5 = true`) weight row
/// against `C` Q8_2_X4 columns, value `c` equal to
/// `dot_q4k_q82x4_avx2` / `dot_q5k_q82x4_avx2` of column `c` bit for bit.
///
/// The body is ik's `mul_mat_qX_K_q8_2_X4_T<Dequantizer, nrc_y = C>`
/// (iqk_gemm_kquants.cpp:796), whose `nrc_y = 1` instance the two one-column
/// kernels port: per block the weight's `d`, mins and scales once; per column
/// its bf16 block scales and sums, then `accd = fma(my, mins, accd)`; per
/// 128-value half the nibbles (and, for Q5_K, the qh fold into bit 4) once,
/// then per column the maddubs sum and `accd = fma(scales_j · dy4, sumi,
/// accd)`. Column `c` sees the one-column kernel's three fmas per block in
/// its order and the same horizontal sum at the end; the integer sums are the
/// one-column instructions on the same codes. `Q5` selects the block width,
/// the nibble offset and the fold at compile time.
///
/// # Safety
/// CPU must support AVX2+FMA+F16C; `wrow` must hold `nb` blocks of its type
/// and each `acols[c]` point at the `2 · nb` readable Q8_2_X4 groups of `nb`
/// super-blocks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
unsafe fn dot_q45k_q82x4_tile_avx2<const C: usize, const Q5: bool>(
    wrow: &[u8],
    acols: &[*const u8; C],
    nb: usize,
) -> [f32; C] {
    // SAFETY: AVX2+FMA+F16C present, `wrow` holds nb blocks and every column
    // 2·nb groups per contract.
    unsafe {
        let (block, qs_off) = if Q5 { (Q5K_BLOCK, 48) } else { (Q4K_BLOCK, 16) };
        let ml = _mm256_set1_epi8(0xF);
        let mh = _mm256_set1_epi8(0x10);
        let mut accd = [_mm256_setzero_ps(); C];

        for i in 0..nb {
            let wb = &wrow[i * block..(i + 1) * block];

            let (d, dmin) = f16c_pair_to_f32([wb[0], wb[1], wb[2], wb[3]]);

            let utmp = make_q4_scales(&wb[4..16]);
            let mins_v = q45k_mins_ps(&utmp);
            let mins = _mm256_mul_ps(_mm256_set1_ps(-dmin), mins_v);

            let mut dy = [_mm256_setzero_ps(); C];
            for ((col, dyc), acc) in acols.iter().zip(dy.iter_mut()).zip(accd.iter_mut()) {
                // SAFETY: every column holds super-block i's two validated groups.
                let (dyv, my) = q82x4_dy_my(*col, i);
                *dyc = dyv;
                *acc = _mm256_fmadd_ps(my, mins, *acc);
            }

            let all_scales = _mm256_mul_ps(_mm256_set1_ps(d), q45k_scales_ps(&utmp));
            let lo = _mm256_castps256_ps128(all_scales);
            let hi = _mm256_extractf128_ps(all_scales, 1);
            let scales = [_mm256_set_m128(lo, lo), _mm256_set_m128(hi, hi)];

            let qh = if Q5 {
                // SAFETY: 32-byte load inside the validated 176-byte Q5_K block.
                _mm256_loadu_si256(wb.as_ptr().add(16) as *const __m256i)
            } else {
                _mm256_setzero_si256()
            };
            let q = &wb[qs_off..qs_off + 128];
            for (j, sj) in scales.iter().enumerate() {
                // SAFETY: loads stay inside the validated block.
                let bits0 = _mm256_loadu_si256(q.as_ptr().add(64 * j) as *const __m256i);
                let bits1 = _mm256_loadu_si256(q.as_ptr().add(64 * j + 32) as *const __m256i);
                let mut values = [
                    _mm256_and_si256(bits0, ml),
                    _mm256_and_si256(_mm256_srli_epi16::<4>(bits0), ml),
                    _mm256_and_si256(bits1, ml),
                    _mm256_and_si256(_mm256_srli_epi16::<4>(bits1), ml),
                ];
                if Q5 {
                    let hbits = if j == 0 {
                        qh
                    } else {
                        _mm256_srli_epi16::<4>(qh)
                    };
                    values[0] = _mm256_or_si256(
                        values[0],
                        _mm256_and_si256(_mm256_slli_epi16::<4>(hbits), mh),
                    );
                    values[1] = _mm256_or_si256(
                        values[1],
                        _mm256_and_si256(_mm256_slli_epi16::<3>(hbits), mh),
                    );
                    values[2] = _mm256_or_si256(
                        values[2],
                        _mm256_and_si256(_mm256_slli_epi16::<2>(hbits), mh),
                    );
                    values[3] = _mm256_or_si256(
                        values[3],
                        _mm256_and_si256(_mm256_slli_epi16::<1>(hbits), mh),
                    );
                }

                for ((col, dyc), acc) in acols.iter().zip(&dy).zip(accd.iter_mut()) {
                    // SAFETY: loads stay inside the column's validated group 2i + j.
                    let qs = col.add((2 * i + j) * Q82X4_STRIDE + 16);
                    let sumi1 =
                        _mm256_maddubs_epi16(values[0], _mm256_loadu_si256(qs as *const __m256i));
                    let sumi2 = _mm256_maddubs_epi16(
                        values[1],
                        _mm256_loadu_si256(qs.add(32) as *const __m256i),
                    );
                    let sumi3 = _mm256_maddubs_epi16(
                        values[2],
                        _mm256_loadu_si256(qs.add(64) as *const __m256i),
                    );
                    let sumi4 = _mm256_maddubs_epi16(
                        values[3],
                        _mm256_loadu_si256(qs.add(96) as *const __m256i),
                    );
                    let t1 = _mm256_add_epi16(
                        _mm256_unpacklo_epi32(sumi1, sumi2),
                        _mm256_unpackhi_epi32(sumi1, sumi2),
                    );
                    let t3 = _mm256_add_epi16(
                        _mm256_unpacklo_epi32(sumi3, sumi4),
                        _mm256_unpackhi_epi32(sumi3, sumi4),
                    );
                    let t = _mm256_add_epi16(
                        _mm256_unpacklo_epi64(t1, t3),
                        _mm256_unpackhi_epi64(t1, t3),
                    );
                    let sumi = _mm256_madd_epi16(_mm256_set1_epi16(1), t);
                    let dy4 = if j == 0 {
                        _mm256_castps256_ps128(*dyc)
                    } else {
                        _mm256_extractf128_ps(*dyc, 1)
                    };
                    let d4d8 = _mm256_mul_ps(*sj, _mm256_set_m128(dy4, dy4));
                    *acc = _mm256_fmadd_ps(d4d8, _mm256_cvtepi32_ps(sumi), *acc);
                }
            }
        }

        let mut out = [0.0f32; C];
        for (o, acc) in out.iter_mut().zip(&accd) {
            *o = hsum_float_8(*acc);
        }
        out
    }
}

/// AVX-512 VNNI tile: one Q4_K weight row against `C` Q8_2_X4 columns,
/// value `c` equal to `dot_q4k_q82x4_avx2` of column `c` bit for bit; at
/// `C = 1` it is [`dot_row`]'s Q4_K kernel on a CPU with the ISA
/// ([`q4k_avx512`]).
///
/// The float side is the AVX2 kernels' per block and column: the same
/// `mins`, `my`, `scales` and `dy4` products and the three fmas
/// `fma(my, mins)`, `fma(scales_0 * dy4_0, sumi_0)`, `fma(scales_1 * dy4_1,
/// sumi_1)` in that order, then `hsum_float_8`. Only the integer sums move:
/// zmm `t` holds sub-block `t`'s codes beside sub-block `4 + t`'s and the
/// column's blocks `t` of its two groups, so one `vpdpbusd` a sub-block pair
/// sums four products a dword, and the AVX2 kernels' 4×4 transpose, done per
/// 128-bit lane on i32, leaves in each of `sumi`'s quarters the four
/// sub-blocks' 16-value halves: half 0 then half 1 of sub-blocks 0..4, then
/// of 4..8. Those are `sumi_0` and `sumi_1`, the same integers (codes ≤ 15
/// against |q| ≤ 128: nothing saturates on either side).
///
/// # Safety
/// CPU must support AVX-512 F+BW+VNNI and AVX2+FMA+F16C; `wrow` must hold
/// `nb` Q4_K blocks and each `acols[c]` point at the `2 * nb` readable
/// Q8_2_X4 groups of `nb` super-blocks.
#[cfg(target_arch = "x86_64")]
#[target_feature(
    enable = "avx512f",
    enable = "avx512bw",
    enable = "avx512vnni",
    enable = "avx2",
    enable = "fma",
    enable = "f16c"
)]
unsafe fn dot_q4k_q82x4_tile_avx512<const C: usize>(
    wrow: &[u8],
    acols: &[*const u8; C],
    nb: usize,
) -> [f32; C] {
    let ml = _mm512_set1_epi8(0xF);
    let mut accd = [_mm256_setzero_ps(); C];

    for i in 0..nb {
        let wb = &wrow[i * Q4K_BLOCK..(i + 1) * Q4K_BLOCK];

        // SAFETY: this function enables F16C; the call touches registers only.
        let (d, dmin) = unsafe { f16c_pair_to_f32([wb[0], wb[1], wb[2], wb[3]]) };
        let utmp = make_q4_scales(&wb[4..16]);
        let all_scales = _mm256_mul_ps(_mm256_set1_ps(d), q45k_scales_ps(&utmp));
        let mins = _mm256_mul_ps(_mm256_set1_ps(-dmin), q45k_mins_ps(&utmp));

        let lo = _mm256_castps256_ps128(all_scales);
        let hi = _mm256_extractf128_ps(all_scales, 1);
        let scales = [_mm256_set_m128(lo, lo), _mm256_set_m128(hi, hi)];

        // Code bytes 32 c..32 c + 32 hold sub-block 2 c in their low
        // nibbles and 2 c + 1 in their high ones: w[t] is sub-block t
        // beside sub-block 4 + t.
        // 32-byte loads, as in the AVX2 kernels: a 64-byte load at a block's
        // 144-byte pitch splits a cache line far more often, which costs
        // at DRAM speed.
        let code = |o: usize| {
            // SAFETY: o + 32 <= 128, so byte 16 + o lies in `wb`'s 144.
            let at = unsafe { wb.as_ptr().add(16 + o) };
            // SAFETY: the 32 bytes at `at` lie in the block's code bytes 16..144.
            unsafe { _mm256_loadu_si256(at as *const __m256i) }
        };
        let half =
            |o: usize, p: usize| _mm512_inserti64x4::<1>(_mm512_castsi256_si512(code(o)), code(p));
        let b02 = half(0, 64);
        let b13 = half(32, 96);
        let w = [
            _mm512_and_si512(b02, ml),
            _mm512_and_si512(_mm512_srli_epi16::<4>(b02), ml),
            _mm512_and_si512(b13, ml),
            _mm512_and_si512(_mm512_srli_epi16::<4>(b13), ml),
        ];

        for (&col, acc) in acols.iter().zip(accd.iter_mut()) {
            // SAFETY: this function enables AVX2, and the column holds
            // super-block i's two groups, 2 * i and 2 * i + 1, of its 2 * nb.
            let (dy, my) = unsafe { q82x4_dy_my(col, i) };
            let act = |o: usize| {
                let load = |g: usize| {
                    // SAFETY: g < 2 * nb and o + 32 <= 128, so the offset lies in
                    // group g of the column's 2 * nb.
                    let at = unsafe { col.add(g * Q82X4_STRIDE + 16 + o) };
                    // SAFETY: the 32 bytes at `at` lie in group g's code bytes 16..144.
                    unsafe { _mm256_loadu_si256(at as *const __m256i) }
                };
                _mm512_inserti64x4::<1>(_mm512_castsi256_si512(load(2 * i)), load(2 * i + 1))
            };
            let zero = _mm512_setzero_si512();
            let s0 = _mm512_dpbusd_epi32(zero, w[0], act(0));
            let s1 = _mm512_dpbusd_epi32(zero, w[1], act(32));
            let s2 = _mm512_dpbusd_epi32(zero, w[2], act(64));
            let s3 = _mm512_dpbusd_epi32(zero, w[3], act(96));
            let t1 = _mm512_add_epi32(_mm512_unpacklo_epi32(s0, s1), _mm512_unpackhi_epi32(s0, s1));
            let t3 = _mm512_add_epi32(_mm512_unpacklo_epi32(s2, s3), _mm512_unpackhi_epi32(s2, s3));
            let sumi = _mm512_cvtepi32_ps(_mm512_add_epi32(
                _mm512_unpacklo_epi64(t1, t3),
                _mm512_unpackhi_epi64(t1, t3),
            ));
            let sumi_0 = _mm512_castps512_ps256(sumi);
            let sumi_1 = _mm256_castpd_ps(_mm512_extractf64x4_pd::<1>(_mm512_castps_pd(sumi)));

            let dy4 = [_mm256_castps256_ps128(dy), _mm256_extractf128_ps(dy, 1)];
            let d4d8 = [
                _mm256_mul_ps(scales[0], _mm256_set_m128(dy4[0], dy4[0])),
                _mm256_mul_ps(scales[1], _mm256_set_m128(dy4[1], dy4[1])),
            ];
            *acc = _mm256_fmadd_ps(my, mins, *acc);
            *acc = _mm256_fmadd_ps(d4d8[0], sumi_0, *acc);
            *acc = _mm256_fmadd_ps(d4d8[1], sumi_1, *acc);
        }
    }

    let mut out = [0.0f32; C];
    for (o, acc) in out.iter_mut().zip(&accd) {
        // SAFETY: this function enables AVX and SSE3; the sum touches registers only.
        *o = unsafe { hsum_float_8(*acc) };
    }
    out
}

// --------------------------------------- Q4_K/Q5_K/Q6_K row lanes x Q8_2_X4

/// The 8×8 transpose of 32-bit lanes: lane `r` of output `g` is lane `g` of
/// input `r`.
///
/// # Safety
/// AVX2 must be present.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn transpose8_epi32(v: [__m256i; 8]) -> [__m256i; 8] {
    // SAFETY: register-only AVX2 shuffles.
    unsafe {
        let t0 = _mm256_unpacklo_epi32(v[0], v[1]);
        let t1 = _mm256_unpackhi_epi32(v[0], v[1]);
        let t2 = _mm256_unpacklo_epi32(v[2], v[3]);
        let t3 = _mm256_unpackhi_epi32(v[2], v[3]);
        let t4 = _mm256_unpacklo_epi32(v[4], v[5]);
        let t5 = _mm256_unpackhi_epi32(v[4], v[5]);
        let t6 = _mm256_unpacklo_epi32(v[6], v[7]);
        let t7 = _mm256_unpackhi_epi32(v[6], v[7]);
        let u0 = _mm256_unpacklo_epi64(t0, t2);
        let u1 = _mm256_unpackhi_epi64(t0, t2);
        let u2 = _mm256_unpacklo_epi64(t1, t3);
        let u3 = _mm256_unpackhi_epi64(t1, t3);
        let u4 = _mm256_unpacklo_epi64(t4, t6);
        let u5 = _mm256_unpackhi_epi64(t4, t6);
        let u6 = _mm256_unpacklo_epi64(t5, t7);
        let u7 = _mm256_unpackhi_epi64(t5, t7);
        [
            _mm256_permute2x128_si256::<0x20>(u0, u4),
            _mm256_permute2x128_si256::<0x20>(u1, u5),
            _mm256_permute2x128_si256::<0x20>(u2, u6),
            _mm256_permute2x128_si256::<0x20>(u3, u7),
            _mm256_permute2x128_si256::<0x31>(u0, u4),
            _mm256_permute2x128_si256::<0x31>(u1, u5),
            _mm256_permute2x128_si256::<0x31>(u2, u6),
            _mm256_permute2x128_si256::<0x31>(u3, u7),
        ]
    }
}

/// [`transpose8_epi32`] of f32 lanes; the bits move unchanged.
///
/// # Safety
/// AVX2 must be present.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn transpose8_ps(v: [__m256; 8]) -> [__m256; 8] {
    // SAFETY: register-only casts and AVX2 shuffles.
    unsafe {
        let mut x = [_mm256_setzero_si256(); 8];
        for (x, v) in x.iter_mut().zip(v) {
            *x = _mm256_castps_si256(v);
        }
        let t = transpose8_epi32(x);
        let mut out = [_mm256_setzero_ps(); 8];
        for (o, t) in out.iter_mut().zip(t) {
            *o = _mm256_castsi256_ps(t);
        }
        out
    }
}

/// One 32-byte chunk `c4` of the eight rows' code bytes as packed code
/// vectors: the chunk transposed in dwords, then its low nibbles as sub-block
/// `2 c4` and its high nibbles as sub-block `2 c4 + 1` — with Q5_K's high bit
/// from the transposed `qh` (`hb`: `qh` for chunks 0 and 1, `qh >> 4` per
/// 16-bit lane for 2 and 3), shifted to bit 4 by `LO` and `HI` as the
/// one-column kernel shifts it — stored at code vectors `(2 c4, g)` and
/// `(2 c4 + 1, g)` of the super-block at `dst`. Every operation is per byte,
/// so the transpose, which moves whole dwords, commutes with it.
///
/// # Safety
/// AVX2 must be present; `dst` must be writable for one packed super-block.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn lane_codes<const Q5: bool, const LO: i32, const HI: i32>(
    chunk: [__m256i; 8],
    hb: [__m256i; 8],
    c4: usize,
    dst: *mut u8,
) {
    // SAFETY: AVX2 present; the stores stay inside the super-block at `dst`.
    unsafe {
        let ml = _mm256_set1_epi8(0xF);
        let mh = _mm256_set1_epi8(0x10);
        let t = transpose8_epi32(chunk);
        for g in 0..8 {
            let mut lo = _mm256_and_si256(t[g], ml);
            let mut hi = _mm256_and_si256(_mm256_srli_epi16::<4>(t[g]), ml);
            if Q5 {
                lo = _mm256_or_si256(lo, _mm256_and_si256(_mm256_slli_epi16::<LO>(hb[g]), mh));
                hi = _mm256_or_si256(hi, _mm256_and_si256(_mm256_slli_epi16::<HI>(hb[g]), mh));
            }
            _mm256_storeu_si256(dst.add(32 * (8 * (2 * c4) + g)) as *mut __m256i, lo);
            _mm256_storeu_si256(dst.add(32 * (8 * (2 * c4 + 1) + g)) as *mut __m256i, hi);
        }
    }
}

/// Super-block `i`'s share of the next group's prefetch ([`pack_lanes`]'
/// `next`): the `SHARE` bytes of `next` from `i · SHARE`, a line at a time,
/// toward L2 (`T1`: the group is read only after this one's columns are
/// dotted, which runs the packed group and the columns through L1). Before
/// the first share the kernel takes `next`'s last byte ([`lane_ahead_last`]),
/// whose line the steps of 64 miss when `next` does not start on a line.
///
/// # Safety
/// `next` must hold at least `(i + 1) · SHARE` bytes.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn lane_ahead<const SHARE: usize>(next: &[u8], i: usize) {
    // SAFETY: every offset is below `(i + 1) · SHARE`, inside `next` (the
    // caller's contract); a prefetch reads nothing into the program.
    unsafe {
        let at = next.as_ptr().add(i * SHARE);
        for t in 0..SHARE.div_ceil(64) {
            _mm_prefetch::<_MM_HINT_T1>(at.add(64 * t) as *const i8);
        }
    }
}

/// The line of `next`'s last byte toward L2 ([`lane_ahead`]).
///
/// # Safety
/// `next` must not be empty.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn lane_ahead_last(next: &[u8]) {
    // SAFETY: `next.len() - 1` is inside the non-empty `next` (the caller's
    // contract); a prefetch reads nothing into the program.
    unsafe { _mm_prefetch::<_MM_HINT_T1>(next.as_ptr().add(next.len() - 1) as *const i8) };
}

/// AVX2 pack of [`LANE_ROWS`] Q4_K (or, with `Q5`, Q5_K) rows of `nb` blocks
/// into [`pack_lanes`]' layout; with `AHEAD`, each super-block's pass also
/// prefetches its share of `next` ([`lane_ahead`]).
///
/// # Safety
/// AVX2+FMA+F16C must be present; `rows` must hold [`LANE_ROWS`] rows of `nb`
/// blocks, `next` (with `AHEAD`) as many bytes, and `packed` `nb` packed
/// super-blocks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
unsafe fn pack_lanes_avx2<const Q5: bool, const AHEAD: bool>(
    rows: &[u8],
    next: &[u8],
    nb: usize,
    packed: &mut [u8],
) {
    // SAFETY: AVX2+FMA+F16C present; every read lies inside a row's block and
    // every write inside the super-block's 2,560 bytes (the caller's sizes).
    unsafe {
        let (block, qs_off) = if Q5 { (Q5K_BLOCK, 48) } else { (Q4K_BLOCK, 16) };
        let row_bytes = nb * block;
        if AHEAD {
            lane_ahead_last(next);
        }
        for i in 0..nb {
            if AHEAD {
                if Q5 {
                    lane_ahead::<{ LANE_ROWS * Q5K_BLOCK }>(next, i);
                } else {
                    lane_ahead::<{ LANE_ROWS * Q4K_BLOCK }>(next, i);
                }
            }
            let dst = packed.as_mut_ptr().add(i * LANE_SB_BYTES);
            let wb: [*const u8; LANE_ROWS] =
                std::array::from_fn(|r| rows.as_ptr().add(r * row_bytes + i * block));
            let mut sc = [_mm256_setzero_ps(); LANE_ROWS];
            let mut mn = [_mm256_setzero_ps(); LANE_ROWS];
            for r in 0..LANE_ROWS {
                let b = std::slice::from_raw_parts(wb[r], block);
                let (d, dmin) = f16c_pair_to_f32([b[0], b[1], b[2], b[3]]);
                let utmp = make_q4_scales(&b[4..16]);
                sc[r] = _mm256_mul_ps(_mm256_set1_ps(d), q45k_scales_ps(&utmp));
                mn[r] = _mm256_mul_ps(_mm256_set1_ps(-dmin), q45k_mins_ps(&utmp));
            }
            let sc = transpose8_ps(sc);
            let mn = transpose8_ps(mn);
            for s in 0..8 {
                _mm256_storeu_ps(dst.add(LANE_SC + 32 * s) as *mut f32, sc[s]);
                _mm256_storeu_ps(dst.add(LANE_MN + 32 * s) as *mut f32, mn[s]);
            }
            let mut hb0 = [_mm256_setzero_si256(); 8];
            let mut hb1 = [_mm256_setzero_si256(); 8];
            if Q5 {
                let mut qh = [_mm256_setzero_si256(); 8];
                for (q, &w) in qh.iter_mut().zip(&wb) {
                    // SAFETY: the 32 high-bit bytes at 16 of each Q5_K block.
                    *q = _mm256_loadu_si256(w.add(16) as *const __m256i);
                }
                hb0 = transpose8_epi32(qh);
                for (h1, &h0) in hb1.iter_mut().zip(&hb0) {
                    *h1 = _mm256_srli_epi16::<4>(h0);
                }
            }
            let mut chunk = [[_mm256_setzero_si256(); 8]; 4];
            for (c4, chunk) in chunk.iter_mut().enumerate() {
                for (v, &w) in chunk.iter_mut().zip(&wb) {
                    // SAFETY: a 32-byte load inside the row's 128 code bytes.
                    *v = _mm256_loadu_si256(w.add(qs_off + 32 * c4) as *const __m256i);
                }
            }
            lane_codes::<Q5, 4, 3>(chunk[0], hb0, 0, dst);
            lane_codes::<Q5, 2, 1>(chunk[1], hb0, 1, dst);
            lane_codes::<Q5, 4, 3>(chunk[2], hb1, 2, dst);
            lane_codes::<Q5, 2, 1>(chunk[3], hb1, 3, dst);
        }
    }
}

/// AVX2 pack of [`LANE_ROWS`] Q6_K rows of `nb` blocks into [`pack_lanes`]'
/// layout: each row's sixteen `d · scales_s` as `dot_q6k_q82x4_avx2` forms
/// them, transposed to a vector a sub-block; then per half `j` the rows' low
/// and high bits transposed in dwords and each code formed as the one-column
/// kernel forms it before its `− 32` (`ql` nibble, `qh` pair onto bits 4 and
/// 5) — block `t` of the half holds sub-blocks `8 j + 2 t` (dwords 0..4) and
/// `8 j + 2 t + 1` (dwords 4..8), so its eight transposed vectors are code
/// vectors `4 (8 j + 2 t) ..` in order. Every operation on the bits is per
/// byte within a 16-bit lane, so the transpose, which moves whole dwords,
/// commutes with it.
///
/// With `AHEAD`, each super-block's pass also prefetches its share of `next`
/// ([`lane_ahead`]).
///
/// # Safety
/// AVX2+FMA+F16C must be present; `rows` must hold [`LANE_ROWS`] rows of `nb`
/// Q6_K blocks, `next` (with `AHEAD`) as many bytes, and `packed` `nb` packed
/// super-blocks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
unsafe fn pack_lanes_q6k_avx2<const AHEAD: bool>(
    rows: &[u8],
    next: &[u8],
    nb: usize,
    packed: &mut [u8],
) {
    // SAFETY: AVX2+FMA+F16C present; every read lies inside a row's block and
    // every write inside the super-block's 2,560 bytes (the caller's sizes).
    unsafe {
        let ml = _mm256_set1_epi8(0xF);
        let mh = _mm256_set1_epi8(0x30);
        let row_bytes = nb * Q6K_BLOCK;
        if AHEAD {
            lane_ahead_last(next);
        }
        for i in 0..nb {
            if AHEAD {
                lane_ahead::<{ LANE_ROWS * Q6K_BLOCK }>(next, i);
            }
            let dst = packed.as_mut_ptr().add(i * LANE_SB_BYTES);
            let wb: [*const u8; LANE_ROWS] =
                std::array::from_fn(|r| rows.as_ptr().add(r * row_bytes + i * Q6K_BLOCK));
            let mut lo = [_mm256_setzero_ps(); LANE_ROWS];
            let mut hi = [_mm256_setzero_ps(); LANE_ROWS];
            for r in 0..LANE_ROWS {
                // SAFETY: d at 208 and the 16 scale bytes at 192 of the block.
                let vd =
                    _mm256_set1_ps(f16c_to_f32((wb[r].add(208) as *const u16).read_unaligned()));
                lo[r] = _mm256_mul_ps(
                    vd,
                    _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(
                        wb[r].add(192) as *const __m128i
                    ))),
                );
                hi[r] = _mm256_mul_ps(
                    vd,
                    _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(
                        wb[r].add(200) as *const __m128i
                    ))),
                );
            }
            let lo = transpose8_ps(lo);
            let hi = transpose8_ps(hi);
            for s in 0..8 {
                _mm256_storeu_ps(dst.add(LANE_SC + 32 * s) as *mut f32, lo[s]);
                _mm256_storeu_ps(dst.add(LANE_SC + 32 * (8 + s)) as *mut f32, hi[s]);
            }
            for j in 0..2 {
                let mut l1 = [_mm256_setzero_si256(); 8];
                let mut l2 = [_mm256_setzero_si256(); 8];
                let mut hb = [_mm256_setzero_si256(); 8];
                for r in 0..LANE_ROWS {
                    // SAFETY: 32-byte loads inside the block's ql (0..128) and
                    // qh (128..192).
                    l1[r] = _mm256_loadu_si256(wb[r].add(64 * j) as *const __m256i);
                    l2[r] = _mm256_loadu_si256(wb[r].add(64 * j + 32) as *const __m256i);
                    hb[r] = _mm256_loadu_si256(wb[r].add(128 + 32 * j) as *const __m256i);
                }
                let l1 = transpose8_epi32(l1);
                let l2 = transpose8_epi32(l2);
                let hb = transpose8_epi32(hb);
                let at =
                    |t: usize, g: usize| dst.add(32 * (4 * (8 * j + 2 * t) + g)) as *mut __m256i;
                for g in 0..8 {
                    let v0 = _mm256_or_si256(
                        _mm256_and_si256(l1[g], ml),
                        _mm256_and_si256(_mm256_slli_epi16::<4>(hb[g]), mh),
                    );
                    let v1 = _mm256_or_si256(
                        _mm256_and_si256(l2[g], ml),
                        _mm256_and_si256(_mm256_slli_epi16::<2>(hb[g]), mh),
                    );
                    let v2 = _mm256_or_si256(
                        _mm256_and_si256(_mm256_srli_epi16::<4>(l1[g]), ml),
                        _mm256_and_si256(hb[g], mh),
                    );
                    let v3 = _mm256_or_si256(
                        _mm256_and_si256(_mm256_srli_epi16::<4>(l2[g]), ml),
                        _mm256_and_si256(_mm256_srli_epi16::<2>(hb[g]), mh),
                    );
                    _mm256_storeu_si256(at(0, g), v0);
                    _mm256_storeu_si256(at(1, g), v1);
                    _mm256_storeu_si256(at(2, g), v2);
                    _mm256_storeu_si256(at(3, g), v3);
                }
            }
        }
    }
}

/// Super-blocks one pass of [`dot_lanes_avx2`] (and [`dot_lanes_avx512`])
/// walks before it moves to the next lane: their packed codes (10 KB) and the
/// columns' activation bytes stay in L1 across the eight lanes.
#[cfg(target_arch = "x86_64")]
const LANE_CHUNK: usize = 4;

/// AVX2 row-lane kernel: one packed group ([`pack_lanes`]) against `C`
/// Q8_2_X4 columns; `out[c][r]` equals `dot_q4k_q82x4_avx2` (or the Q5_K
/// kernel) of row `r` and column `c` bit for bit.
///
/// The one-column kernel's accumulator lane `l = t + 4 h` takes, per
/// super-block, `fma(my_l, min_l)`, then `fma(d·sc_s · d8_s, Σ)` for sub-block
/// `s = t` and then `s = 4 + t`, `Σ` the integer dot of the sub-block's half
/// `h`. Here the eight lanes are eight passes over a chunk of super-blocks,
/// each holding lane `l` of every row (a ymm lane is a row) for the `C`
/// columns, so each (row, column) gets the same fused adds in the same order;
/// the column's `d8` and `my` vectors are the one-column kernel's, formed once
/// a chunk. The integer dot of a half is four maddubs of the rows' code
/// dwords against a broadcast activation dword (codes ≤ 31 against |q| ≤ 127:
/// four pairs of an i16 lane sum to at most 8 · 31 · 127 = 31,496, so nothing
/// saturates) and one madd: the same integer. At the end lane `l` of a row
/// is transposed back and summed by `hsum_float_8`.
///
/// # Safety
/// AVX2+FMA+F16C must be present; `packed` must hold `nb` packed
/// super-blocks and every column `nb` super-blocks' Q8_2_X4 groups.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
unsafe fn dot_lanes_avx2<const C: usize>(
    packed: &[u8],
    acols: &[*const u8; C],
    nb: usize,
) -> [[f32; LANE_ROWS]; C] {
    // SAFETY: AVX2+FMA+F16C present; every read lies inside a packed
    // super-block or a column's validated groups (the caller's sizes).
    unsafe {
        let ones = _mm256_set1_epi16(1);
        let mut acc = [[_mm256_setzero_ps(); C]; 8];
        let mut i0 = 0;
        while i0 < nb {
            let n = (nb - i0).min(LANE_CHUNK);
            // Column c's d8 and my vectors of super-block i0 + ii at [c][ii]:
            // the one-column kernel's `dy` and `my` ([`q82x4_dy_my`]).
            let mut dy = [[[0.0f32; 8]; LANE_CHUNK]; C];
            let mut my = [[[0.0f32; 8]; LANE_CHUNK]; C];
            for c in 0..C {
                for ii in 0..n {
                    // SAFETY: every column holds super-block i0 + ii's two validated groups.
                    let (dyv, myv) = q82x4_dy_my(acols[c], i0 + ii);
                    _mm256_storeu_ps(dy[c][ii].as_mut_ptr(), dyv);
                    _mm256_storeu_ps(my[c][ii].as_mut_ptr(), myv);
                }
            }
            for l in 0..8 {
                let (t, h) = (l % 4, l / 4);
                let mut a = acc[l];
                for ii in 0..n {
                    let i = i0 + ii;
                    let base = packed.as_ptr().add(i * LANE_SB_BYTES);
                    let mins = _mm256_loadu_ps(base.add(LANE_MN + 32 * l) as *const f32);
                    for c in 0..C {
                        a[c] = _mm256_fmadd_ps(_mm256_set1_ps(my[c][ii][l]), mins, a[c]);
                    }
                    for j in 0..2 {
                        let s = 4 * j + t;
                        let sc = _mm256_loadu_ps(base.add(LANE_SC + 32 * s) as *const f32);
                        let mut w = [_mm256_setzero_si256(); 4];
                        for (g, w) in w.iter_mut().enumerate() {
                            *w = _mm256_loadu_si256(
                                base.add(32 * (8 * s + 4 * h + g)) as *const __m256i
                            );
                        }
                        for c in 0..C {
                            let q = acols[c].add((2 * i + j) * Q82X4_STRIDE + 16 + 32 * t + 16 * h)
                                as *const i32;
                            let p0 =
                                _mm256_maddubs_epi16(w[0], _mm256_set1_epi32(q.read_unaligned()));
                            let p1 = _mm256_maddubs_epi16(
                                w[1],
                                _mm256_set1_epi32(q.add(1).read_unaligned()),
                            );
                            let p2 = _mm256_maddubs_epi16(
                                w[2],
                                _mm256_set1_epi32(q.add(2).read_unaligned()),
                            );
                            let p3 = _mm256_maddubs_epi16(
                                w[3],
                                _mm256_set1_epi32(q.add(3).read_unaligned()),
                            );
                            let sumi = _mm256_madd_epi16(
                                _mm256_add_epi16(
                                    _mm256_add_epi16(p0, p1),
                                    _mm256_add_epi16(p2, p3),
                                ),
                                ones,
                            );
                            let d4d8 = _mm256_mul_ps(sc, _mm256_set1_ps(dy[c][ii][s]));
                            a[c] = _mm256_fmadd_ps(d4d8, _mm256_cvtepi32_ps(sumi), a[c]);
                        }
                    }
                }
                acc[l] = a;
            }
            i0 += n;
        }
        let mut out = [[0.0f32; LANE_ROWS]; C];
        for (c, o) in out.iter_mut().enumerate() {
            let mut lanes = [_mm256_setzero_ps(); 8];
            for (x, a) in lanes.iter_mut().zip(&acc) {
                *x = a[c];
            }
            let rows = transpose8_ps(lanes);
            for (o, &v) in o.iter_mut().zip(&rows) {
                *o = hsum_float_8(v);
            }
        }
        out
    }
}

/// AVX-512 VNNI row-lane kernel for Q4_K: one packed Q4_K group
/// ([`pack_lanes`]) against `C` Q8_2_X4 columns; `out[c][r]` equals
/// `dot_q4k_q82x4_avx2` of row `r` and column `c` bit for bit.
///
/// [`dot_lanes_avx2`]'s walk with two columns a zmm: its low half holds
/// column `2 p`'s ymm lanes and its high half column `2 p + 1`'s (an odd `C`
/// pairs its last column with itself), each lane a row as there, so every
/// (row, column) gets the same products and the same fused adds in the same
/// order. The integer dot of a half is four `vpdpbusd` of the rows' code
/// dwords against the columns' broadcast activation dwords, the same integer
/// as the four maddubs and the madd (Q4_K codes ≤ 15: nothing saturates).
///
/// # Safety
/// CPU must support AVX-512 F+BW+VNNI and AVX2+FMA+F16C; `packed` must hold
/// `nb` packed Q4_K super-blocks and every column `nb` super-blocks' Q8_2_X4
/// groups.
#[cfg(target_arch = "x86_64")]
#[target_feature(
    enable = "avx512f",
    enable = "avx512bw",
    enable = "avx512vnni",
    enable = "avx2",
    enable = "fma",
    enable = "f16c"
)]
unsafe fn dot_lanes_avx512<const C: usize>(
    packed: &[u8],
    acols: &[*const u8; C],
    nb: usize,
) -> [[f32; LANE_ROWS]; C] {
    let pairs = C.div_ceil(2);
    let pair = |p: usize| (2 * p, (2 * p + 1).min(C - 1));
    let mut acc = [[_mm512_setzero_ps(); C]; 8];
    let mut i0 = 0;
    while i0 < nb {
        let n = (nb - i0).min(LANE_CHUNK);
        // Column c's d8 and my vectors of super-block i0 + ii at [c][ii]:
        // the one-column kernel's `dy` and `my` ([`q82x4_dy_my`]).
        let mut dy = [[[0.0f32; 8]; LANE_CHUNK]; C];
        let mut my = [[[0.0f32; 8]; LANE_CHUNK]; C];
        for c in 0..C {
            for ii in 0..n {
                // SAFETY: this function enables AVX2, and every column holds
                // super-block i0 + ii's two groups (i0 + ii < nb).
                let (dyv, myv) = unsafe { q82x4_dy_my(acols[c], i0 + ii) };
                // SAFETY: the store writes the eight f32 of dy[c][ii].
                unsafe { _mm256_storeu_ps(dy[c][ii].as_mut_ptr(), dyv) };
                // SAFETY: the store writes the eight f32 of my[c][ii].
                unsafe { _mm256_storeu_ps(my[c][ii].as_mut_ptr(), myv) };
            }
        }
        for l in 0..8 {
            let (t, h) = (l % 4, l / 4);
            let mut a = acc[l];
            for ii in 0..n {
                let i = i0 + ii;
                // SAFETY: i < nb, and `packed` holds nb packed super-blocks.
                let sb = unsafe { packed.as_ptr().add(i * LANE_SB_BYTES) };
                let codes = |at: usize| {
                    // SAFETY: at + 32 <= LANE_SC, so `at` lies in packed super-block i.
                    let v = unsafe { sb.add(at) };
                    // SAFETY: the 32 bytes at `v` lie in the super-block's code vectors.
                    unsafe { _mm256_loadu_si256(v as *const __m256i) }
                };
                let factors = |at: usize| {
                    // SAFETY: at + 32 <= LANE_SB_BYTES, so `at` lies in packed super-block i.
                    let v = unsafe { sb.add(at) };
                    // SAFETY: LANE_SC <= at, so the eight f32 at `v` lie in the
                    // super-block's scale or min vectors.
                    unsafe { _mm256_loadu_pd(v as *const f64) }
                };
                let mins = _mm512_castpd_ps(_mm512_broadcast_f64x4(factors(LANE_MN + 32 * l)));
                for (p, a) in a.iter_mut().enumerate().take(pairs) {
                    let (c0, c1) = pair(p);
                    let m = _mm512_mask_mov_ps(
                        _mm512_set1_ps(my[c0][ii][l]),
                        0xFF00,
                        _mm512_set1_ps(my[c1][ii][l]),
                    );
                    *a = _mm512_fmadd_ps(m, mins, *a);
                }
                for j in 0..2 {
                    let s = 4 * j + t;
                    let sc = _mm512_castpd_ps(_mm512_broadcast_f64x4(factors(LANE_SC + 32 * s)));
                    let mut w = [_mm512_setzero_si512(); 4];
                    for (g, w) in w.iter_mut().enumerate() {
                        *w = _mm512_broadcast_i64x4(codes(32 * (8 * s + 4 * h + g)));
                    }
                    for (p, a) in a.iter_mut().enumerate().take(pairs) {
                        let (c0, c1) = pair(p);
                        let at = (2 * i + j) * Q82X4_STRIDE + 16 + 32 * t + 16 * h;
                        let act = |c: usize| {
                            // SAFETY: c < C, and 2 * i + j < 2 * nb, so `at` lies in group
                            // 2 * i + j of column c.
                            let v = unsafe { acols[c].add(at) };
                            // SAFETY: 32 * t + 16 * h + 16 <= 128, so the 16 bytes at `v`
                            // lie in that group's code bytes.
                            unsafe { _mm_loadu_si128(v as *const __m128i) }
                        };
                        // Column c0's four activation dwords in each 128-bit
                        // lane of the low half, c1's in the high half; an
                        // in-lane shuffle spreads dword g over each half.
                        let q = _mm512_inserti64x4::<1>(
                            _mm512_castsi256_si512(_mm256_broadcastsi128_si256(act(c0))),
                            _mm256_broadcastsi128_si256(act(c1)),
                        );
                        let zero = _mm512_setzero_si512();
                        let s01 = _mm512_dpbusd_epi32(
                            _mm512_dpbusd_epi32(zero, w[0], _mm512_shuffle_epi32::<0x00>(q)),
                            w[1],
                            _mm512_shuffle_epi32::<0x55>(q),
                        );
                        let s23 = _mm512_dpbusd_epi32(
                            _mm512_dpbusd_epi32(zero, w[2], _mm512_shuffle_epi32::<0xAA>(q)),
                            w[3],
                            _mm512_shuffle_epi32::<0xFF>(q),
                        );
                        let sumi = _mm512_add_epi32(s01, s23);
                        let d8 = _mm512_mask_mov_ps(
                            _mm512_set1_ps(dy[c0][ii][s]),
                            0xFF00,
                            _mm512_set1_ps(dy[c1][ii][s]),
                        );
                        let d4d8 = _mm512_mul_ps(sc, d8);
                        *a = _mm512_fmadd_ps(d4d8, _mm512_cvtepi32_ps(sumi), *a);
                    }
                }
            }
            acc[l] = a;
        }
        i0 += n;
    }
    let mut out = [[0.0f32; LANE_ROWS]; C];
    for (c, o) in out.iter_mut().enumerate() {
        let mut lanes = [_mm256_setzero_ps(); 8];
        for (x, a) in lanes.iter_mut().zip(&acc) {
            let both = _mm512_castps_pd(a[c / 2]);
            *x = _mm256_castpd_ps(if c % 2 == 0 {
                _mm512_castpd512_pd256(both)
            } else {
                _mm512_extractf64x4_pd::<1>(both)
            });
        }
        // SAFETY: this function enables AVX2; the transpose touches registers only.
        let rows = unsafe { transpose8_ps(lanes) };
        for (o, &v) in o.iter_mut().zip(&rows) {
            // SAFETY: this function enables AVX and SSE3; the sum touches registers only.
            *o = unsafe { hsum_float_8(v) };
        }
    }
    out
}

/// AVX2 Q6_K row-lane kernel: one packed Q6_K group ([`pack_lanes`]) against
/// `C` Q8_2_X4 columns; `out[c][r]` equals `dot_q6k_q82x4_avx2` of row `r` and
/// column `c` bit for bit.
///
/// The one-column kernel's accumulator lane `l = t + 4 h` takes, per
/// super-block, `fma((d·sc_s) · d8, Σ)` for sub-block `s = 2 t + h` of half 0
/// and then `s = 8 + 2 t + h` of half 1 (`K_SHUFFLE_Q6K`'s order; `d8` of block
/// `t` of the half's group), `Σ` the sub-block's exact `Σ (c − 32) q`. Here,
/// as in [`dot_lanes_avx2`], the lanes are passes over a chunk of super-blocks
/// with a row a ymm lane, and a column's `dy` is formed once a chunk. `Σ` is
/// four maddubs of the rows' code dwords (0..63) against a broadcast
/// activation dword, two pairs added in i16 (at most 4 · 63 · 127 = 32,004)
/// and each pair widened by a madd, less `32 Σ q` over the sixteen values,
/// formed once a chunk by the one-column kernel's reduction of a maddubs by
/// ones (exact: at most 16 · 127) — the same integer for |q| ≤ 127,
/// [`quantize_col`]'s range.
///
/// # Safety
/// AVX2+FMA+F16C must be present; `packed` must hold `nb` packed Q6_K
/// super-blocks and every column `nb` super-blocks' Q8_2_X4 groups.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
unsafe fn dot_lanes_q6k_avx2<const C: usize>(
    packed: &[u8],
    acols: &[*const u8; C],
    nb: usize,
) -> [[f32; LANE_ROWS]; C] {
    // SAFETY: AVX2+FMA+F16C present; every read lies inside a packed
    // super-block or a column's validated groups (the caller's sizes).
    unsafe {
        let ones = _mm256_set1_epi16(1);
        let ones8 = _mm256_set1_epi8(1);
        let mut acc = [[_mm256_setzero_ps(); C]; 8];
        let mut i0 = 0;
        while i0 < nb {
            let n = (nb - i0).min(LANE_CHUNK);
            // Column c's dy of super-block i0 + ii at [c][ii], and at
            // [c][ii][8 j + l] the term `32 Σ q` of block l % 4 of group
            // 2 i + j, half l / 4.
            let mut dy = [[[0.0f32; 8]; LANE_CHUNK]; C];
            let mut corr = [[[0i32; 16]; LANE_CHUNK]; C];
            for c in 0..C {
                for ii in 0..n {
                    let i = i0 + ii;
                    let g0 = acols[c].add((2 * i) * Q82X4_STRIDE);
                    let g1 = acols[c].add((2 * i + 1) * Q82X4_STRIDE);
                    let d4_1 = _mm_cvtepu16_epi32(_mm_loadl_epi64(g0 as *const __m128i));
                    let d4_2 = _mm_cvtepu16_epi32(_mm_loadl_epi64(g1 as *const __m128i));
                    let dyv =
                        _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_set_m128i(d4_2, d4_1), 16));
                    _mm256_storeu_ps(dy[c][ii].as_mut_ptr(), dyv);
                    for (j, g) in [g0, g1].into_iter().enumerate() {
                        let q = g.add(16);
                        let s0 =
                            _mm256_maddubs_epi16(ones8, _mm256_loadu_si256(q as *const __m256i));
                        let s1 = _mm256_maddubs_epi16(
                            ones8,
                            _mm256_loadu_si256(q.add(32) as *const __m256i),
                        );
                        let s2 = _mm256_maddubs_epi16(
                            ones8,
                            _mm256_loadu_si256(q.add(64) as *const __m256i),
                        );
                        let s3 = _mm256_maddubs_epi16(
                            ones8,
                            _mm256_loadu_si256(q.add(96) as *const __m256i),
                        );
                        let t1 = _mm256_add_epi16(
                            _mm256_unpacklo_epi32(s0, s1),
                            _mm256_unpackhi_epi32(s0, s1),
                        );
                        let t3 = _mm256_add_epi16(
                            _mm256_unpacklo_epi32(s2, s3),
                            _mm256_unpackhi_epi32(s2, s3),
                        );
                        let t = _mm256_add_epi16(
                            _mm256_unpacklo_epi64(t1, t3),
                            _mm256_unpackhi_epi64(t1, t3),
                        );
                        _mm256_storeu_si256(
                            corr[c][ii][8 * j..].as_mut_ptr() as *mut __m256i,
                            _mm256_slli_epi32::<5>(_mm256_madd_epi16(t, ones)),
                        );
                    }
                }
            }
            for l in 0..8 {
                let (t, h) = (l % 4, l / 4);
                let mut a = acc[l];
                for ii in 0..n {
                    let i = i0 + ii;
                    let base = packed.as_ptr().add(i * LANE_SB_BYTES);
                    for j in 0..2 {
                        let s = 8 * j + 2 * t + h;
                        let sc = _mm256_loadu_ps(base.add(LANE_SC + 32 * s) as *const f32);
                        let mut w = [_mm256_setzero_si256(); 4];
                        for (g, w) in w.iter_mut().enumerate() {
                            *w = _mm256_loadu_si256(base.add(32 * (4 * s + g)) as *const __m256i);
                        }
                        for c in 0..C {
                            let q = acols[c].add((2 * i + j) * Q82X4_STRIDE + 16 + 32 * t + 16 * h)
                                as *const i32;
                            let p0 =
                                _mm256_maddubs_epi16(w[0], _mm256_set1_epi32(q.read_unaligned()));
                            let p1 = _mm256_maddubs_epi16(
                                w[1],
                                _mm256_set1_epi32(q.add(1).read_unaligned()),
                            );
                            let p2 = _mm256_maddubs_epi16(
                                w[2],
                                _mm256_set1_epi32(q.add(2).read_unaligned()),
                            );
                            let p3 = _mm256_maddubs_epi16(
                                w[3],
                                _mm256_set1_epi32(q.add(3).read_unaligned()),
                            );
                            let sumi = _mm256_sub_epi32(
                                _mm256_add_epi32(
                                    _mm256_madd_epi16(_mm256_add_epi16(p0, p1), ones),
                                    _mm256_madd_epi16(_mm256_add_epi16(p2, p3), ones),
                                ),
                                _mm256_set1_epi32(corr[c][ii][8 * j + l]),
                            );
                            let d4d8 = _mm256_mul_ps(sc, _mm256_set1_ps(dy[c][ii][4 * j + t]));
                            a[c] = _mm256_fmadd_ps(d4d8, _mm256_cvtepi32_ps(sumi), a[c]);
                        }
                    }
                }
                acc[l] = a;
            }
            i0 += n;
        }
        let mut out = [[0.0f32; LANE_ROWS]; C];
        for (c, o) in out.iter_mut().enumerate() {
            let mut lanes = [_mm256_setzero_ps(); 8];
            for (x, a) in lanes.iter_mut().zip(&acc) {
                *x = a[c];
            }
            let rows = transpose8_ps(lanes);
            for (o, &v) in o.iter_mut().zip(&rows) {
                *o = hsum_float_8(v);
            }
        }
        out
    }
}

// --------------------------------------------------------- Q5_0 x Q8_2_X4

/// `HBitDequantizer` constants for unpacking high bits.
#[cfg(target_arch = "x86_64")]
struct Q5HBit {
    shuffle: __m256i,
    mask: __m256i,
    minus1: __m256i,
}

/// One 32-value weight block's codes as unsigned bytes 0..31.
///
/// # Safety
/// `blk` must hold QS_OFF + 16 readable bytes.
#[cfg(target_arch = "x86_64")]
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
#[cfg(target_arch = "x86_64")]
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
            let dw = f16c_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
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
            dw[j] = half_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
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
        let dw = half_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
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
/// Its column tile [`dot_q5f1_q82x4_tile_avx2`] runs these steps per column
/// and takes every edit to them too.
///
/// # Safety
/// Caller must ensure AVX2+FMA+F16C are available and buffers match `nb` blocks.
#[cfg(target_arch = "x86_64")]
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
            let dw = f16c_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
            let mw = f16c_to_f32(u16::from_le_bytes([wb[2], wb[3]]));
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
            dw[j] = half_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
            mw[j] = half_to_f32(u16::from_le_bytes([wb[2], wb[3]]));
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
        let dw = half_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
        let mw = half_to_f32(u16::from_le_bytes([wb[2], wb[3]]));
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

/// The Q5_1 tile: one row's blocks against up to [`TILE_COLS`] Q8_2_X4
/// columns, ik's `AccumT<MinusType1<nrc_y>, nrc_y, _>::compute`
/// (iqk_gemm_legacy_quants.cpp:302) at `nrc_y = C`, whose `nrc_y = 1`
/// instance [`dot_q5f1_q82x4_avx2`] ports: per x4 group the four blocks'
/// codes and (d, m) pairs once; per column its bf16 scales and sums, its
/// `accm` add and its one FMA; per tail block the codes and `dw`, `mw` once,
/// then per column the one-column tail step. Each column keeps its own
/// `acc` and `accm` and ends in the one-column kernel's reduction, so column
/// `c` sees the one-column kernel's float operations in its order.
///
/// TWIN of [`dot_q5f1_q82x4_avx2`]: any edit to the float or integer steps of
/// one is made in both.
///
/// # Safety
/// CPU must support AVX2+FMA+F16C; `wrow` must hold `nb` Q5_1 blocks and each
/// `acols[c]` point at `col_bytes(Q5_1, 32 · nb)` readable bytes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
unsafe fn dot_q5f1_q82x4_tile_avx2<const C: usize>(
    wrow: &[u8],
    acols: &[*const u8; C],
    nb: usize,
) -> [f32; C] {
    const { assert!(C != 0 && C <= TILE_COLS) };
    // SAFETY: AVX2+FMA+F16C present, `wrow` holds nb blocks and every column
    // the Q8_2_X4 groups and q8_2 tails of nb blocks per contract.
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
        let mut acc = [_mm256_setzero_ps(); C];
        let mut accm = [_mm_setzero_ps(); C];

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

            for ((col, a), am) in acols.iter().zip(acc.iter_mut()).zip(accm.iter_mut()) {
                // SAFETY: 8 readable bytes at the head of the column's group i.
                let g = col.add(i * Q82X4_STRIDE);
                let aux_d = _mm_castsi128_ps(_mm_slli_epi32::<16>(_mm_cvtepu16_epi32(
                    _mm_loadl_epi64(g as *const __m128i),
                )));
                // SAFETY: 8 readable bytes at offset 8 of the group.
                let aux_m = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_loadl_epi64(
                    g.add(8) as *const __m128i
                )));
                let prep = _mm256_set_m128(_mm_mul_ps(aux_d, aux_m), aux_d);
                let s12 = _mm256_mul_ps(other, prep);
                *am = _mm_add_ps(*am, _mm256_extractf128_ps(s12, 1));
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
                *a = _mm256_fmadd_ps(dall, _mm256_cvtepi32_ps(pall), *a);
            }
        }

        let nb4 = 4 * nbg;
        for i in nb4..nb {
            let b = i * Q5F1_BLOCK;
            let wb = &wrow[b..b + Q5F1_BLOCK];
            let dw = f16c_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
            let mw = f16c_to_f32(u16::from_le_bytes([wb[2], wb[3]]));
            let qx0 = q5x_codes::<4, 8>(wb, m4, mh, &hb);
            let tb = nb4 / 4 * Q82X4_STRIDE + (i - nb4) * Q82_BLOCK;
            for ((col, a), am) in acols.iter().zip(acc.iter_mut()).zip(accm.iter_mut()) {
                // SAFETY: the tail block's 36 bytes are inside the column.
                let t = col.add(tb);
                let da = bf16_bits_to_f32(u16::from_le_bytes([*t, *t.add(1)]));
                let ma = i16::from_le_bytes([*t.add(2), *t.add(3)]) as f32;
                let d = dw * da;
                let corr = mw * (da * ma) * 0.25f32;
                *am = _mm_add_ps(*am, _mm_set1_ps(corr));
                let qs = _mm256_loadu_si256(t.add(4) as *const __m256i);
                let p0 = _mm256_madd_epi16(m1, _mm256_maddubs_epi16(qx0, qs));
                *a = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(p0), *a);
            }
        }

        // Reduction order is load-bearing: the one-column kernel's, per column.
        let mut out = [0.0f32; C];
        for ((o, a), am) in out.iter_mut().zip(&acc).zip(&accm) {
            let sum = _mm_add_ps(_mm256_castps256_ps128(*a), _mm256_extractf128_ps(*a, 1));
            let x = _mm_add_ps(sum, *am);
            let x = _mm_add_ps(x, _mm_movehl_ps(x, x));
            *o = _mm_cvtss_f32(_mm_add_ss(x, _mm_movehdup_ps(x)));
        }
        out
    }
}

// --------------------------------------------------------- Q8_0 x Q8_2_X4

/// AVX2+FMA+F16C row dot: Q8_0 weights against Q8_2_X4 column.
///
/// ik's AVX2 instruction graph: the i8 x i8 products through the sign trick
/// (`maddubs(|w|, a·sign(w))` then `madd` by ones), four blocks' lanes folded
/// by the unpack adds, one FMA per x4 group under `f16(d_w)·bf16(d_a)`, one
/// FMA per tail block on its raw eight lanes, then an eight-lane hsum. No min
/// term: Q8_0 codes are signed, so there is no `accm` for the reduction to add.
/// `maddubs` cannot saturate: `|w| ≤ 128` against `|a| ≤ 128`, a pair at most
/// 2·128·128 = 32,768 in magnitude, reached only at −32,768. The sign trick
/// is ik's rule on its one wrap: an activation code −128 under a negative
/// weight code stays −128 (`_mm256_sign_epi8`), so that product is taken with
/// the wrong sign — reachable only from a q8_2 block whose bf16 scale is
/// subnormal, and the mirror reproduces it. Its column tile
/// [`dot_q8f0_q82x4_tile_avx2`] runs these steps per column and takes every
/// edit to them too.
///
/// # Safety
/// Caller must ensure AVX2+FMA+F16C are available and buffers match `nb` blocks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
unsafe fn dot_q8f0_q82x4_avx2(wrow: &[u8], acol: &[u8], nb: usize) -> f32 {
    // SAFETY: AVX2+FMA+F16C present (checked by dot_row) and both slices
    // hold the validated lengths per contract.
    unsafe {
        let m1 = _mm256_set1_epi16(1);
        let mut acc = _mm256_setzero_ps();

        let nbg = nb / 4;
        for i in 0..nbg {
            let b0 = (4 * i) * Q8F0_BLOCK;
            let b1 = (4 * i + 1) * Q8F0_BLOCK;
            let b2 = (4 * i + 2) * Q8F0_BLOCK;
            let b3 = (4 * i + 3) * Q8F0_BLOCK;
            let w = wrow.as_ptr();
            // SAFETY: 32 readable bytes at offset 2 of each validated block.
            let qx0 = _mm256_loadu_si256(w.add(b0 + 2) as *const __m256i);
            let qx1 = _mm256_loadu_si256(w.add(b1 + 2) as *const __m256i);
            let qx2 = _mm256_loadu_si256(w.add(b2 + 2) as *const __m256i);
            let qx3 = _mm256_loadu_si256(w.add(b3 + 2) as *const __m256i);
            let mut scales8 = [0u8; 8];
            scales8[0..2].copy_from_slice(&wrow[b0..b0 + 2]);
            scales8[2..4].copy_from_slice(&wrow[b1..b1 + 2]);
            scales8[4..6].copy_from_slice(&wrow[b2..b2 + 2]);
            scales8[6..8].copy_from_slice(&wrow[b3..b3 + 2]);
            // SAFETY: 8 readable bytes in the local staging buffer.
            let s4 = _mm_cvtph_ps(_mm_loadl_epi64(scales8.as_ptr() as *const __m128i));

            // SAFETY: 8 readable bytes at the head of the validated group.
            let g = acol.as_ptr().add(i * Q82X4_STRIDE);
            let aux_d = _mm_castsi128_ps(_mm_slli_epi32::<16>(_mm_cvtepu16_epi32(
                _mm_loadl_epi64(g as *const __m128i),
            )));
            let d4 = _mm_mul_ps(s4, aux_d);
            let dall = _mm256_set_m128(d4, d4);

            // SAFETY: 32 readable bytes at each offset inside the group.
            let y0 = _mm256_loadu_si256(g.add(16) as *const __m256i);
            let y1 = _mm256_loadu_si256(g.add(48) as *const __m256i);
            let y2 = _mm256_loadu_si256(g.add(80) as *const __m256i);
            let y3 = _mm256_loadu_si256(g.add(112) as *const __m256i);
            let p0 = _mm256_madd_epi16(
                m1,
                _mm256_maddubs_epi16(_mm256_sign_epi8(qx0, qx0), _mm256_sign_epi8(y0, qx0)),
            );
            let p1 = _mm256_madd_epi16(
                m1,
                _mm256_maddubs_epi16(_mm256_sign_epi8(qx1, qx1), _mm256_sign_epi8(y1, qx1)),
            );
            let p2 = _mm256_madd_epi16(
                m1,
                _mm256_maddubs_epi16(_mm256_sign_epi8(qx2, qx2), _mm256_sign_epi8(y2, qx2)),
            );
            let p3 = _mm256_madd_epi16(
                m1,
                _mm256_maddubs_epi16(_mm256_sign_epi8(qx3, qx3), _mm256_sign_epi8(y3, qx3)),
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
            let b = i * Q8F0_BLOCK;
            let dw = f16c_to_f32(u16::from_le_bytes([wrow[b], wrow[b + 1]]));
            // SAFETY: 32 readable bytes at offset 2 of the validated block.
            let qx0 = _mm256_loadu_si256(wrow.as_ptr().add(b + 2) as *const __m256i);
            let tb = nb4 / 4 * Q82X4_STRIDE + (i - nb4) * Q82_BLOCK;
            let da = bf16_bits_to_f32(u16::from_le_bytes([acol[tb], acol[tb + 1]]));
            let d = dw * da;
            // SAFETY: 32 readable bytes at offset 4 of the tail block.
            let y = _mm256_loadu_si256(acol.as_ptr().add(tb + 4) as *const __m256i);
            let p0 = _mm256_madd_epi16(
                m1,
                _mm256_maddubs_epi16(_mm256_sign_epi8(qx0, qx0), _mm256_sign_epi8(y, qx0)),
            );
            acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(p0), acc);
        }

        // Reduction order is load-bearing: ik's `hsum_float_8`.
        hsum_float_8(acc)
    }
}

/// Emulates `dot_q8f0_q82x4_avx2` with bit identity; both `_mm256_sign_epi8`
/// go through [`emul_sign_epi8`], its −128 wrap included.
fn dot_q8f0_q82x4_emul(wrow: &[u8], acol: &[u8], nb: usize) -> f32 {
    // `madd(1, maddubs(sign(w, w), sign(a, w)))` of one block's 32 codes.
    let block_dot = |w: &[u8], a: &[u8]| -> [i32; 8] {
        let mut ax = [0u8; 32];
        let mut sy = [0u8; 32];
        for m in 0..32 {
            ax[m] = emul_sign_epi8(w[m], w[m] as i8);
            sy[m] = emul_sign_epi8(a[m], w[m] as i8);
        }
        emul_madd1(emul_maddubs(ax, sy))
    };
    let mut acc = [0.0f32; 8];

    let nbg = nb / 4;
    for i in 0..nbg {
        let g = &acol[i * Q82X4_STRIDE..(i + 1) * Q82X4_STRIDE];
        let mut lo = [0.0f32; 4];
        let mut p = [[0i32; 8]; 4];
        for j in 0..4 {
            let wb = &wrow[(4 * i + j) * Q8F0_BLOCK..(4 * i + j + 1) * Q8F0_BLOCK];
            let dw = half_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
            let da = bf16_bits_to_f32(u16::from_le_bytes([g[2 * j], g[2 * j + 1]]));
            lo[j] = dw * da;
            p[j] = block_dot(&wb[2..34], &g[16 + 32 * j..16 + 32 * j + 32]);
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
        for (l, a) in acc.iter_mut().enumerate() {
            *a = lo[l % 4].mul_add(pall[l] as f32, *a);
        }
    }

    let nb4 = 4 * nbg;
    for i in nb4..nb {
        let wb = &wrow[i * Q8F0_BLOCK..(i + 1) * Q8F0_BLOCK];
        let dw = half_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
        let tb = nb4 / 4 * Q82X4_STRIDE + (i - nb4) * Q82_BLOCK;
        let blk = &acol[tb..tb + Q82_BLOCK];
        let da = bf16_bits_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let d = dw * da;
        let p0 = block_dot(&wb[2..34], &blk[4..36]);
        for (a, &p) in acc.iter_mut().zip(&p0) {
            *a = d.mul_add(p as f32, *a);
        }
    }

    let mut x = [0.0f32; 4];
    for (l, xl) in x.iter_mut().enumerate() {
        *xl = acc[l] + acc[4 + l];
    }
    (x[0] + x[2]) + (x[1] + x[3])
}

/// The Q8_0 tile: one row's blocks against up to [`TILE_COLS`] Q8_2_X4
/// columns, ik's `AccumT<MinusType0, nrc_y, _>::compute`
/// (iqk_gemm_legacy_quants.cpp:302) at `nrc_y = C`, whose `nrc_y = 1`
/// instance [`dot_q8f0_q82x4_avx2`] ports: per x4 group the four blocks'
/// codes, their `sign(w, w)` magnitudes and the four f16 scales once; per
/// column its bf16 scales, the `sign(a, w)` products and its one FMA; per
/// tail block the codes and `dw` once, then per column the one-column tail
/// step; each column's `acc` ends in [`hsum_float_8`]. The weight magnitude
/// is an integer function of the weight alone, so taking it once per block
/// changes no product; `sign(a, w)` stays per column, the −128 wrap
/// included.
///
/// TWIN of [`dot_q8f0_q82x4_avx2`]: any edit to the float or integer steps of
/// one is made in both.
///
/// # Safety
/// CPU must support AVX2+FMA+F16C; `wrow` must hold `nb` Q8_0 blocks and each
/// `acols[c]` point at `col_bytes(Q8_0, 32 · nb)` readable bytes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
unsafe fn dot_q8f0_q82x4_tile_avx2<const C: usize>(
    wrow: &[u8],
    acols: &[*const u8; C],
    nb: usize,
) -> [f32; C] {
    const { assert!(C != 0 && C <= TILE_COLS) };
    // SAFETY: AVX2+FMA+F16C present, `wrow` holds nb blocks and every column
    // the Q8_2_X4 groups and q8_2 tails of nb blocks per contract.
    unsafe {
        let m1 = _mm256_set1_epi16(1);
        let mut acc = [_mm256_setzero_ps(); C];

        let nbg = nb / 4;
        for i in 0..nbg {
            let b0 = (4 * i) * Q8F0_BLOCK;
            let b1 = (4 * i + 1) * Q8F0_BLOCK;
            let b2 = (4 * i + 2) * Q8F0_BLOCK;
            let b3 = (4 * i + 3) * Q8F0_BLOCK;
            let w = wrow.as_ptr();
            // SAFETY: 32 readable bytes at offset 2 of each validated block.
            let qx0 = _mm256_loadu_si256(w.add(b0 + 2) as *const __m256i);
            let qx1 = _mm256_loadu_si256(w.add(b1 + 2) as *const __m256i);
            let qx2 = _mm256_loadu_si256(w.add(b2 + 2) as *const __m256i);
            let qx3 = _mm256_loadu_si256(w.add(b3 + 2) as *const __m256i);
            let ax0 = _mm256_sign_epi8(qx0, qx0);
            let ax1 = _mm256_sign_epi8(qx1, qx1);
            let ax2 = _mm256_sign_epi8(qx2, qx2);
            let ax3 = _mm256_sign_epi8(qx3, qx3);
            let mut scales8 = [0u8; 8];
            scales8[0..2].copy_from_slice(&wrow[b0..b0 + 2]);
            scales8[2..4].copy_from_slice(&wrow[b1..b1 + 2]);
            scales8[4..6].copy_from_slice(&wrow[b2..b2 + 2]);
            scales8[6..8].copy_from_slice(&wrow[b3..b3 + 2]);
            // SAFETY: 8 readable bytes in the local staging buffer.
            let s4 = _mm_cvtph_ps(_mm_loadl_epi64(scales8.as_ptr() as *const __m128i));

            for (col, a) in acols.iter().zip(acc.iter_mut()) {
                // SAFETY: 8 readable bytes at the head of the column's group i.
                let g = col.add(i * Q82X4_STRIDE);
                let aux_d = _mm_castsi128_ps(_mm_slli_epi32::<16>(_mm_cvtepu16_epi32(
                    _mm_loadl_epi64(g as *const __m128i),
                )));
                let d4 = _mm_mul_ps(s4, aux_d);
                let dall = _mm256_set_m128(d4, d4);

                // SAFETY: 32 readable bytes at each offset inside the group.
                let y0 = _mm256_loadu_si256(g.add(16) as *const __m256i);
                let y1 = _mm256_loadu_si256(g.add(48) as *const __m256i);
                let y2 = _mm256_loadu_si256(g.add(80) as *const __m256i);
                let y3 = _mm256_loadu_si256(g.add(112) as *const __m256i);
                let p0 =
                    _mm256_madd_epi16(m1, _mm256_maddubs_epi16(ax0, _mm256_sign_epi8(y0, qx0)));
                let p1 =
                    _mm256_madd_epi16(m1, _mm256_maddubs_epi16(ax1, _mm256_sign_epi8(y1, qx1)));
                let p2 =
                    _mm256_madd_epi16(m1, _mm256_maddubs_epi16(ax2, _mm256_sign_epi8(y2, qx2)));
                let p3 =
                    _mm256_madd_epi16(m1, _mm256_maddubs_epi16(ax3, _mm256_sign_epi8(y3, qx3)));
                let p01 =
                    _mm256_add_epi32(_mm256_unpacklo_epi32(p0, p1), _mm256_unpackhi_epi32(p0, p1));
                let p23 =
                    _mm256_add_epi32(_mm256_unpacklo_epi32(p2, p3), _mm256_unpackhi_epi32(p2, p3));
                let pall = _mm256_add_epi32(
                    _mm256_unpacklo_epi64(p01, p23),
                    _mm256_unpackhi_epi64(p01, p23),
                );
                *a = _mm256_fmadd_ps(dall, _mm256_cvtepi32_ps(pall), *a);
            }
        }

        let nb4 = 4 * nbg;
        for i in nb4..nb {
            let b = i * Q8F0_BLOCK;
            let wb = &wrow[b..b + Q8F0_BLOCK];
            let dw = f16c_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
            // SAFETY: 32 readable bytes at offset 2 of the validated block.
            let qx0 = _mm256_loadu_si256(wb.as_ptr().add(2) as *const __m256i);
            let ax0 = _mm256_sign_epi8(qx0, qx0);
            let tb = nb4 / 4 * Q82X4_STRIDE + (i - nb4) * Q82_BLOCK;
            for (col, a) in acols.iter().zip(acc.iter_mut()) {
                // SAFETY: the tail block's 36 bytes are inside the column.
                let t = col.add(tb);
                let da = bf16_bits_to_f32(u16::from_le_bytes([*t, *t.add(1)]));
                let d = dw * da;
                let y = _mm256_loadu_si256(t.add(4) as *const __m256i);
                let p0 = _mm256_madd_epi16(m1, _mm256_maddubs_epi16(ax0, _mm256_sign_epi8(y, qx0)));
                *a = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(p0), *a);
            }
        }

        // Reduction order is load-bearing: ik's `hsum_float_8`, per column.
        let mut out = [0.0f32; C];
        for (o, a) in out.iter_mut().zip(&acc) {
            *o = hsum_float_8(*a);
        }
        out
    }
}

// ------------------------------------------------------------- Q2_0 x Q8_0

/// AVX2 row dot: Q2_0 weights against a Q8_0 column, a port of mainline's
/// `ggml_vec_dot_q2_0_q8_0_generic` (llama.cpp `f2cc7282c`
/// ggml-cpu/quants.c:177), the loop mainline's x86 build runs (it has no x86
/// kernel of its own, arch-fallback.h:90). Value j of a Q2_0 block is bits
/// `2 * (j % 4)` of `qs[j / 4]`, code q standing for `q - 1`. Each of the
/// block's two q8_0 blocks gives the exact integer `Σ (q - 1) * y`, here
/// `Σ q * y - Σ y` through `maddubs` (a pair at most 2 * 3 * 128 in i16), with the
/// codes and the activations in one order of their own inside each 16-value
/// lane, which an integer sum does not see. The float steps are the loop's,
/// in its order, each fused as a native build with FMA contraction on fuses
/// it (GCC 15.2 at `-O3 -march=native` compiles `ggml_vec_dot_q2_0_q8_0` to
/// `vfmadd132ss`, `vfmadd132ss`, `vfmadd231ss`, Clang 21 to three FMAs as
/// well): `sumi = d1 * s + sumi` over the two q8_0 blocks from `sumi = 0`,
/// then `sumf = d0 * sumi + sumf`, block by block. A build that rounds each
/// step apart (contraction off, or a CPU variant without FMA) gives other
/// bits.
///
/// TWIN of [`dot_q2f0_q8f0_scalar`]: an edit to the float steps of one is
/// made in both.
///
/// # Safety
/// Caller must ensure AVX2+FMA+F16C are available, `wrow` holds `nb` Q2_0
/// blocks and `acol` `2 * nb` q8_0 blocks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
unsafe fn dot_q2f0_q8f0_avx2(wrow: &[u8], acol: &[u8], nb: usize) -> f32 {
    // SAFETY: AVX2+FMA+F16C present per contract; every load lies inside a
    // block of the validated row or column.
    unsafe {
        let m3 = _mm256_set1_epi8(3);
        let ones8 = _mm256_set1_epi8(1);
        let ones16 = _mm256_set1_epi16(1);
        // q8_0 block k's eight code bytes are code words 2k and 2k + 1: word
        // 2k + l to every u32 slot of lane l, then slot m shifted by 2m, so
        // byte 4m + b of lane l is the code of value 16l + 4b + m.
        let spread = [
            _mm256_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1),
            _mm256_setr_epi32(2, 2, 2, 2, 3, 3, 3, 3),
        ];
        let shifts = _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6);
        // The activations into that order: byte 4m + b of a lane from 4b + m.
        let order = _mm256_setr_epi8(
            0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 0, 4, 8, 12, 1, 5, 9, 13, 2, 6,
            10, 14, 3, 7, 11, 15,
        );
        // The f16 scale at `p`, lane 0 of a whole 16-byte load: a load, where
        // `f16c_to_f32`'s insert (`vpinsrw`) would make the conversion wait on
        // the register it merges into (LLVM picks `sumf`'s).
        let scale =
            |p: *const u8| _mm_cvtss_f32(_mm_cvtph_ps(_mm_loadu_si128(p as *const __m128i)));
        let blocks = wrow.as_chunks::<Q2F0_BLOCK>().0[..nb].iter();
        let mut sumf = 0.0f32;
        for (wb, ab) in blocks.zip(acol.as_chunks::<{ 2 * Q8F0_BLOCK }>().0) {
            // SAFETY: 16 readable bytes at offset 0 of the 18-byte block.
            let d0 = scale(wb.as_ptr());
            // SAFETY: 16 readable bytes at offset 2 of the 18-byte block; the
            // spreads read only its four words.
            let qs = _mm256_castsi128_si256(_mm_loadu_si128(wb.as_ptr().add(2) as *const __m128i));
            let mut p = [_mm256_setzero_si256(); 2];
            for ((pk, sk), yb) in p.iter_mut().zip(spread).zip(ab.as_chunks::<Q8F0_BLOCK>().0) {
                let spread = _mm256_permutevar8x32_epi32(qs, sk);
                let q = _mm256_and_si256(_mm256_srlv_epi32(spread, shifts), m3);
                // SAFETY: 32 readable bytes at offset 2 of the 34-byte block.
                let y = _mm256_loadu_si256(yb.as_ptr().add(2) as *const __m256i);
                let y = _mm256_shuffle_epi8(y, order);
                let qy = _mm256_maddubs_epi16(q, y);
                *pk =
                    _mm256_madd_epi16(_mm256_sub_epi16(qy, _mm256_maddubs_epi16(ones8, y)), ones16);
            }
            // Lanes 0 and 1 of `s`: each q8_0 block's eight lanes summed.
            let h = _mm256_add_epi32(
                _mm256_unpacklo_epi32(p[0], p[1]),
                _mm256_unpackhi_epi32(p[0], p[1]),
            );
            let s = _mm_add_epi32(_mm256_castsi256_si128(h), _mm256_extracti128_si256::<1>(h));
            let s = _mm_add_epi32(s, _mm_unpackhi_epi64(s, s));
            // SAFETY: 16 readable bytes at offset 0 of each 34-byte block.
            let d1 = [scale(ab.as_ptr()), scale(ab.as_ptr().add(Q8F0_BLOCK))];
            let sumi = d1[0].mul_add(_mm_cvtsi128_si32(s) as f32, 0.0);
            let sumi = d1[1].mul_add(_mm_extract_epi32::<1>(s) as f32, sumi);
            sumf = d0.mul_add(sumi, sumf);
        }
        sumf
    }
}

/// Scalar twin of [`dot_q2f0_q8f0_avx2`], bit for bit: the same exact integer
/// sums and the same fused float steps in the same order.
fn dot_q2f0_q8f0_scalar(wrow: &[u8], acol: &[u8], nb: usize) -> f32 {
    let blocks = wrow.as_chunks::<Q2F0_BLOCK>().0[..nb].iter();
    let mut sumf = 0.0f32;
    for (wb, ab) in blocks.zip(acol.as_chunks::<{ 2 * Q8F0_BLOCK }>().0) {
        let d0 = half_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
        let mut sumi = 0.0f32;
        let (codes, _) = wb[2..].as_chunks::<8>();
        for (qs, yb) in codes.iter().zip(ab.as_chunks::<Q8F0_BLOCK>().0) {
            let d1 = half_to_f32(u16::from_le_bytes([yb[0], yb[1]]));
            let mut s = 0i32;
            for (j, &y) in yb[2..].iter().enumerate() {
                let q = (qs[j / 4] >> (2 * (j % 4))) & 3;
                s += (i32::from(q) - 1) * i32::from(y as i8);
            }
            sumi = d1.mul_add(s as f32, sumi);
        }
        sumf = d0.mul_add(sumi, sumf);
    }
    sumf
}

// ------------------------------------------------------------ IQ3_XXS x Q8_K

/// `block_iq3_xxs` (ggml-common.h): f16 d @0, qs[96] @2. Bytes 0..64 of qs index
/// [`IQ3XXS_GRID`] (four magnitudes each); bytes 64..96 are eight u32 words, one per
/// 32-value sub-block: bits 7l..7l+7 the [`KSIGNS_IQ2XS`] index of its eight-value
/// group l, bits 28..32 the scale s, weight 2s + 1.
const IQ3XXS_BLOCK: usize = 98;

/// ik's `keven_signs` (iqk_common.h): entry i is [`KSIGNS_IQ2XS`]`[i]` spread to one
/// byte per bit, 0xff where the bit is set and 0x01 where it is clear — the
/// `sign_epi8` operand that negates or keeps each of eight grid values.
#[cfg(target_arch = "x86_64")]
static KEVEN_SIGNS: [u64; 128] = keven_signs();

#[cfg(target_arch = "x86_64")]
const fn keven_signs() -> [u64; 128] {
    let mut t = [0u64; 128];
    let mut i = 0;
    while i < 128 {
        let s = KSIGNS_IQ2XS[i];
        let mut b = 0;
        while b < 8 {
            let byte: u64 = if (s >> b) & 1 != 0 { 0xff } else { 0x01 };
            t[i] |= byte << (8 * b);
            b += 1;
        }
        i += 1;
    }
    t
}

/// The four sign vectors of a sign word whose four 7-bit fields (the IQ3_XXS and IQ2_XXS
/// sub-block words) each index [`KEVEN_SIGNS`]: field `l`'s eight values in qword `l`.
///
/// # Safety
/// AVX2 must be available (taken from the caller, being `#[inline(always)]`).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn keven_signs_word(aux: u32) -> __m256i {
    // SAFETY: register-only intrinsics; every index is masked below the table's 128.
    unsafe {
        _mm256_set_epi64x(
            KEVEN_SIGNS[((aux >> 21) & 127) as usize] as i64,
            KEVEN_SIGNS[((aux >> 14) & 127) as usize] as i64,
            KEVEN_SIGNS[((aux >> 7) & 127) as usize] as i64,
            KEVEN_SIGNS[(aux & 127) as usize] as i64,
        )
    }
}

/// ik's `DequantizerIQ3XXS::minv`: the offset that puts the signed grid values
/// (magnitudes 4..62) on maddubs' unsigned side, 2..126.
const IQ3XXS_MIN: i8 = 64;

/// AVX2+FMA+F16C row dot: IQ3_XXS weights against a Q8_K column.
///
/// ik's form: a sub-block's 32 grid values take their signs through `sign_epi8`,
/// then +64; the offset comes back out through the column's 16-value sums,
/// `(-64·d)·dy · Σ s·bsum`, one fmadd per block ahead of the codes' fmadd, with
/// `d = 0.25·f16`. i16: a maddubs pair peaks at 2·126·128 = 32256 on any column
/// bytes, so saturation is unreachable. i32: a lane sums 8 sub-blocks × 31 × 4 ×
/// 126 × 128 < 2^24, so its f32 conversion is exact.
///
/// TWIN of [`dot_iq3xxs_q8k_tile_avx2`], its column tile: any edit to the float or
/// integer steps of one is made in both.
///
/// # Safety
/// Caller must ensure AVX2+FMA+F16C are available and buffers match `nb` blocks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
unsafe fn dot_iq3xxs_q8k_avx2(wrow: &[u8], acol: &[u8], nb: usize) -> f32 {
    // SAFETY: AVX2+FMA+F16C present and both slices hold nb blocks per contract.
    unsafe {
        let min_value = _mm256_set1_epi8(IQ3XXS_MIN);
        // Scales8KBase::shuffle: scale b onto the two 16-value sums of sub-block b.
        let mins_lo = _mm_set_epi32(0x0706_0706, 0x0504_0504, 0x0302_0302, 0x0100_0100);
        let mins_hi = _mm_set_epi32(0x0f0e_0f0e, 0x0d0c_0d0c, 0x0b0a_0b0a, 0x0908_0908);
        let mut accd = _mm256_setzero_ps();

        for i in 0..nb {
            let blk = wrow.as_ptr().add(IQ3XXS_BLOCK * i);
            // SAFETY: one unaligned u16 read inside the block.
            let d = 0.25f32 * f16c_to_f32((blk as *const u16).read_unaligned());
            let qs = blk.add(2);
            // SAFETY: the eight scale/sign words, 32 bytes inside the block.
            let words = _mm256_loadu_si256(qs.add(64) as *const __m256i);
            let sc32 = _mm256_or_si256(
                _mm256_slli_epi32::<1>(_mm256_srli_epi32::<28>(words)),
                _mm256_set1_epi32(1),
            );
            let sc16 = _mm_packs_epi32(
                _mm256_castsi256_si128(sc32),
                _mm256_extracti128_si256::<1>(sc32),
            );
            let mins = _mm256_set_m128i(
                _mm_shuffle_epi8(sc16, mins_hi),
                _mm_shuffle_epi8(sc16, mins_lo),
            );
            let all_scales = _mm256_set_m128i(sc16, sc16);
            let dmin = -d * f32::from(IQ3XXS_MIN);

            let col = acol.as_ptr().add(Q8K_STRIDE * i);
            // SAFETY: the column block's f32 d @0 and its i16 bsums[16] @264.
            let dy = (col as *const f32).read_unaligned();
            let bsums = _mm256_loadu_si256(col.add(264) as *const __m256i);
            accd = _mm256_fmadd_ps(
                _mm256_set1_ps(dmin * dy),
                _mm256_cvtepi32_ps(_mm256_madd_epi16(mins, bsums)),
                accd,
            );

            let mut sumi = _mm256_setzero_si256();
            for j in 0..2 {
                let mut p = [_mm256_setzero_si256(); 4];
                for (k, pk) in p.iter_mut().enumerate() {
                    let b = 4 * j + k;
                    let g = qs.add(8 * b);
                    // SAFETY: eight index bytes inside the block; each indexes the
                    // 256-entry grid.
                    let q = _mm256_set_epi32(
                        IQ3XXS_GRID[usize::from(*g.add(7))] as i32,
                        IQ3XXS_GRID[usize::from(*g.add(6))] as i32,
                        IQ3XXS_GRID[usize::from(*g.add(5))] as i32,
                        IQ3XXS_GRID[usize::from(*g.add(4))] as i32,
                        IQ3XXS_GRID[usize::from(*g.add(3))] as i32,
                        IQ3XXS_GRID[usize::from(*g.add(2))] as i32,
                        IQ3XXS_GRID[usize::from(*g.add(1))] as i32,
                        IQ3XXS_GRID[usize::from(*g.add(0))] as i32,
                    );
                    // SAFETY: sub-block b's word, inside the block.
                    let aux = (qs.add(64 + 4 * b) as *const u32).read_unaligned();
                    let signs = keven_signs_word(aux);
                    let values = _mm256_add_epi8(_mm256_sign_epi8(q, signs), min_value);
                    // set_scales_8: sub-block b's scale on every i16 lane.
                    let sc = _mm256_shuffle_epi8(
                        all_scales,
                        _mm256_set1_epi16(((2 * b) | ((2 * b + 1) << 8)) as i16),
                    );
                    // SAFETY: 32 codes of sub-block b inside the column block.
                    let q8 = _mm256_loadu_si256(col.add(8 + 32 * b) as *const __m256i);
                    *pk = _mm256_madd_epi16(sc, _mm256_maddubs_epi16(values, q8));
                }
                // multiply_add: exact i32 sums, so the order is free; ik's is kept.
                if j == 0 {
                    sumi = _mm256_add_epi32(
                        _mm256_add_epi32(p[0], p[2]),
                        _mm256_add_epi32(p[1], p[3]),
                    );
                } else {
                    sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p[0], p[2]));
                    sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p[1], p[3]));
                }
            }
            accd = _mm256_fmadd_ps(_mm256_set1_ps(d * dy), _mm256_cvtepi32_ps(sumi), accd);
        }

        hsum_float_8(accd)
    }
}

/// One column's block step of the IQ2_XXS, IQ3_XXS, IQ3_S and IQ4_XS kernels (their steps
/// are the same past the weight decode): the bsums fold's fmadd, the eight
/// sub-blocks' `madd(scale, maddubs(values, q8))` summed in ik's order, then
/// the codes' fmadd, onto `accd`. `values` are the block's 32-code vectors
/// (unsigned, offset), `scales` each sub-block's scale on every i16 lane.
///
/// # Safety
/// AVX2+FMA must be available (taken from the caller, being
/// `#[inline(always)]`); `col` must point at a validated Q8_K block.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn q8k_block_step(
    values: &[__m256i; 8],
    scales: &[__m256i; 8],
    mins: __m256i,
    d: f32,
    dmin: f32,
    col: *const u8,
    accd: __m256,
) -> __m256 {
    // SAFETY: the column block's f32 d @0, i16 bsums[16] @264 and the 256
    // codes @8, all inside the validated block.
    unsafe {
        let dy = (col as *const f32).read_unaligned();
        let bsums = _mm256_loadu_si256(col.add(264) as *const __m256i);
        let accd = _mm256_fmadd_ps(
            _mm256_set1_ps(dmin * dy),
            _mm256_cvtepi32_ps(_mm256_madd_epi16(mins, bsums)),
            accd,
        );
        let mut sumi = _mm256_setzero_si256();
        for j in 0..2 {
            let mut p = [_mm256_setzero_si256(); 4];
            for (k, pk) in p.iter_mut().enumerate() {
                let b = 4 * j + k;
                let q8 = _mm256_loadu_si256(col.add(8 + 32 * b) as *const __m256i);
                *pk = _mm256_madd_epi16(scales[b], _mm256_maddubs_epi16(values[b], q8));
            }
            // multiply_add: exact i32 sums, so the order is free; ik's is kept.
            if j == 0 {
                sumi = _mm256_add_epi32(_mm256_add_epi32(p[0], p[2]), _mm256_add_epi32(p[1], p[3]));
            } else {
                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p[0], p[2]));
                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p[1], p[3]));
            }
        }
        _mm256_fmadd_ps(_mm256_set1_ps(d * dy), _mm256_cvtepi32_ps(sumi), accd)
    }
}

/// The IQ3_XXS tile: one row's blocks against up to [`TILE_COLS`] Q8_K
/// columns. Per block the weight side once — the grid values with their signs
/// and the offset, each sub-block's scale vector, the bsums fold's `mins`, `d`
/// — then per column [`q8k_block_step`], the one-column kernel's float and
/// integer steps. Each column keeps its own `accd` and ends in
/// `hsum_float_8`, so column `c` sees the one-column kernel's operations in
/// its order. Nothing in the one-column kernel mixes weight and column state
/// before its two fmadds, so the whole decode splits off.
///
/// TWIN of [`dot_iq3xxs_q8k_avx2`]: any edit to the float or integer steps of
/// one is made in both.
///
/// # Safety
/// CPU must support AVX2+FMA+F16C; `wrow` must hold `nb` IQ3_XXS blocks and
/// each `acols[c]` point at `nb` Q8_K blocks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
unsafe fn dot_iq3xxs_q8k_tile_avx2<const C: usize>(
    wrow: &[u8],
    acols: &[*const u8; C],
    nb: usize,
) -> [f32; C] {
    const { assert!(C != 0 && C <= TILE_COLS) };
    // SAFETY: AVX2+FMA+F16C present, `wrow` holds nb blocks and every column
    // nb Q8_K blocks per contract.
    unsafe {
        let min_value = _mm256_set1_epi8(IQ3XXS_MIN);
        // Scales8KBase::shuffle: scale b onto the two 16-value sums of sub-block b.
        let mins_lo = _mm_set_epi32(0x0706_0706, 0x0504_0504, 0x0302_0302, 0x0100_0100);
        let mins_hi = _mm_set_epi32(0x0f0e_0f0e, 0x0d0c_0d0c, 0x0b0a_0b0a, 0x0908_0908);
        let mut accd = [_mm256_setzero_ps(); C];

        for i in 0..nb {
            let blk = wrow.as_ptr().add(IQ3XXS_BLOCK * i);
            // SAFETY: one unaligned u16 read inside the block.
            let d = 0.25f32 * f16c_to_f32((blk as *const u16).read_unaligned());
            let qs = blk.add(2);
            // SAFETY: the eight scale/sign words, 32 bytes inside the block.
            let words = _mm256_loadu_si256(qs.add(64) as *const __m256i);
            let sc32 = _mm256_or_si256(
                _mm256_slli_epi32::<1>(_mm256_srli_epi32::<28>(words)),
                _mm256_set1_epi32(1),
            );
            let sc16 = _mm_packs_epi32(
                _mm256_castsi256_si128(sc32),
                _mm256_extracti128_si256::<1>(sc32),
            );
            let mins = _mm256_set_m128i(
                _mm_shuffle_epi8(sc16, mins_hi),
                _mm_shuffle_epi8(sc16, mins_lo),
            );
            let all_scales = _mm256_set_m128i(sc16, sc16);
            let dmin = -d * f32::from(IQ3XXS_MIN);

            let mut values = [_mm256_setzero_si256(); 8];
            let mut scales = [_mm256_setzero_si256(); 8];
            for b in 0..8 {
                let g = qs.add(8 * b);
                // SAFETY: eight index bytes inside the block; each indexes the
                // 256-entry grid.
                let q = _mm256_set_epi32(
                    IQ3XXS_GRID[usize::from(*g.add(7))] as i32,
                    IQ3XXS_GRID[usize::from(*g.add(6))] as i32,
                    IQ3XXS_GRID[usize::from(*g.add(5))] as i32,
                    IQ3XXS_GRID[usize::from(*g.add(4))] as i32,
                    IQ3XXS_GRID[usize::from(*g.add(3))] as i32,
                    IQ3XXS_GRID[usize::from(*g.add(2))] as i32,
                    IQ3XXS_GRID[usize::from(*g.add(1))] as i32,
                    IQ3XXS_GRID[usize::from(*g.add(0))] as i32,
                );
                // SAFETY: sub-block b's word, inside the block.
                let aux = (qs.add(64 + 4 * b) as *const u32).read_unaligned();
                let signs = keven_signs_word(aux);
                values[b] = _mm256_add_epi8(_mm256_sign_epi8(q, signs), min_value);
                // set_scales_8: sub-block b's scale on every i16 lane.
                scales[b] = _mm256_shuffle_epi8(
                    all_scales,
                    _mm256_set1_epi16(((2 * b) | ((2 * b + 1) << 8)) as i16),
                );
            }

            for (col, a) in acols.iter().zip(accd.iter_mut()) {
                // SAFETY: the column's block i, inside the validated column.
                *a = q8k_block_step(&values, &scales, mins, d, dmin, col.add(Q8K_STRIDE * i), *a);
            }
        }

        // Reduction order is load-bearing: the one-column kernel's, per column.
        let mut out = [0.0f32; C];
        for (o, a) in out.iter_mut().zip(&accd) {
            *o = hsum_float_8(*a);
        }
        out
    }
}

/// Mirror of `dot_iq3xxs_q8k_avx2`, bit-identical: the integer lanes are exact
/// sums (no saturation, no wrap — the kernel's bounds), so they are computed
/// directly; the two fmadds per block and the final reduction are the kernel's.
fn dot_iq3xxs_q8k_emul(wrow: &[u8], acol: &[u8], nb: usize) -> f32 {
    let mut accd = [0.0f32; 8];
    for i in 0..nb {
        let blk = &wrow[IQ3XXS_BLOCK * i..IQ3XXS_BLOCK * (i + 1)];
        let d = 0.25f32 * half_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let qs = &blk[2..];
        let words: [u32; 8] = std::array::from_fn(|b| {
            u32::from_le_bytes(qs[64 + 4 * b..68 + 4 * b].try_into().unwrap())
        });
        let sc: [i32; 8] = std::array::from_fn(|b| 2 * (words[b] >> 28) as i32 + 1);
        let dmin = -d * f32::from(IQ3XXS_MIN);

        let col = &acol[Q8K_STRIDE * i..Q8K_STRIDE * (i + 1)];
        let dy = f32::from_le_bytes(col[0..4].try_into().unwrap());
        let bsum = |t: usize| i32::from(i16::from_le_bytes([col[264 + 2 * t], col[265 + 2 * t]]));
        for (l, a) in accd.iter_mut().enumerate() {
            let m = sc[l] * bsum(2 * l) + sc[l] * bsum(2 * l + 1);
            *a = (dmin * dy).mul_add(m as f32, *a);
        }

        let mut sumi = [0i32; 8];
        for (b, &s) in sc.iter().enumerate() {
            for n in 0..32 {
                let mag = IQ3XXS_GRID[usize::from(qs[8 * b + n / 4])].to_le_bytes()[n % 4] as i8;
                let signs = KSIGNS_IQ2XS[((words[b] >> (7 * (n / 8))) & 127) as usize];
                let v = if (signs >> (n % 8)) & 1 != 0 {
                    -mag
                } else {
                    mag
                };
                let u = i32::from(v.wrapping_add(IQ3XXS_MIN) as u8);
                let a = i32::from(col[8 + 32 * b + n] as i8);
                sumi[n / 4] += s * u * a;
            }
        }
        for (a, &si) in accd.iter_mut().zip(&sumi) {
            *a = (d * dy).mul_add(si as f32, *a);
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

// ------------------------------------------------------------ IQ3_S x Q8_K

/// `block_iq3_s` (ggml-common.h): f16 d @0, qs[64] @2, qh[8] @66, signs[32] @74,
/// scales[4] @106. Sub-block b (32 values) is eight grid words: word m is
/// [`IQ3S_GRID`] entry `qs[8b + m] | (bit m of qh[b]) << 8` (four magnitudes), and
/// its value n takes bit `n % 8` of `signs[4b + n / 8]`, set meaning negative. The
/// scale is nibble `b & 1` of `scales[b / 2]`, weight 2s + 1.
const IQ3S_BLOCK: usize = 110;

/// ik's `DequantizerIQ3S::minv`: the offset that puts the signed grid values
/// (magnitudes 1..15) on maddubs' unsigned side, 1..31.
const IQ3S_MIN: i8 = 16;

/// ik's `DequantizerIQ3S::make_scales`: the eight sub-block weights `2s + 1` on i16
/// lanes in sub-block order (the four low nibbles interleaved with the four high).
///
/// # Safety
/// AVX2 must be available (taken from the caller, being `#[inline(always)]`);
/// `blk` must point at a whole IQ3_S block.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn iq3s_scales(blk: *const u8) -> __m128i {
    // SAFETY: one unaligned u32 read of the four scale bytes inside the block.
    unsafe {
        let aux = (blk.add(106) as *const u32).read_unaligned();
        let lo = aux & 0x0f0f_0f0f;
        let hi = (aux >> 4) & 0x0f0f_0f0f;
        let both = _mm_set_epi32(0, 0, hi as i32, lo as i32);
        let s8 = _mm_shuffle_epi8(both, _mm_set1_epi64x(0x0703_0602_0501_0400));
        _mm_or_si128(
            _mm_slli_epi16::<1>(_mm_cvtepi8_epi16(s8)),
            _mm_set1_epi16(1),
        )
    }
}

/// One IQ3_S sub-block's 32 values on maddubs' unsigned side: ik's
/// `IndexHelperIQ3S::make2` (the nine-bit indices built in SIMD and stored, then
/// eight scalar grid loads), `SignHelper::sign_4_values` (the sign bytes' bits to
/// ±1 per value through `sign_epi8`) and the offset [`IQ3S_MIN`].
///
/// # Safety
/// AVX2 must be available (taken from the caller, being `#[inline(always)]`);
/// `blk` must point at a whole IQ3_S block and `b` be below 8.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn iq3s_values(blk: *const u8, b: usize) -> __m256i {
    // SAFETY: sub-block b's eight index bytes, its qh byte and its four sign bytes,
    // inside the block.
    unsafe {
        let idx_l = _mm256_cvtepu8_epi32(_mm_loadl_epi64(blk.add(2 + 8 * b) as *const __m128i));
        let qh = _mm256_set1_epi32(i32::from(*blk.add(66 + b)));
        // Lane l shifts bit l of qh up to bit 8.
        let idx_h = _mm256_and_si256(
            _mm256_sllv_epi32(qh, _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8)),
            _mm256_set1_epi32(256),
        );
        let mut ix = [0u32; 8];
        _mm256_storeu_si256(
            ix.as_mut_ptr() as *mut __m256i,
            _mm256_or_si256(idx_h, idx_l),
        );
        // SAFETY: each index is a byte, or a byte and 256: below 512.
        let q = _mm256_set_epi32(
            *IQ3S_GRID.get_unchecked(ix[7] as usize) as i32,
            *IQ3S_GRID.get_unchecked(ix[6] as usize) as i32,
            *IQ3S_GRID.get_unchecked(ix[5] as usize) as i32,
            *IQ3S_GRID.get_unchecked(ix[4] as usize) as i32,
            *IQ3S_GRID.get_unchecked(ix[3] as usize) as i32,
            *IQ3S_GRID.get_unchecked(ix[2] as usize) as i32,
            *IQ3S_GRID.get_unchecked(ix[1] as usize) as i32,
            *IQ3S_GRID.get_unchecked(ix[0] as usize) as i32,
        );
        // Value n of the 32 selects sign byte n / 8, bit n % 8.
        let mask1 = _mm256_set_epi64x(
            0x0303_0303_0303_0303,
            0x0202_0202_0202_0202,
            0x0101_0101_0101_0101,
            0,
        );
        let mask2 = _mm256_set1_epi64x(0x8040_2010_0804_0201u64 as i64);
        let sign_bytes = (blk.add(74 + 4 * b) as *const u32).read_unaligned();
        let bits = _mm256_and_si256(
            _mm256_shuffle_epi8(_mm256_set1_epi32(sign_bytes as i32), mask1),
            mask2,
        );
        let signs = _mm256_or_si256(_mm256_cmpeq_epi8(bits, mask2), _mm256_set1_epi8(1));
        _mm256_add_epi8(_mm256_sign_epi8(q, signs), _mm256_set1_epi8(IQ3S_MIN))
    }
}

/// AVX2+FMA+F16C row dot: IQ3_S weights against a Q8_K column.
///
/// ik's form, as [`dot_iq3xxs_q8k_avx2`]'s: a sub-block's 32 grid values take
/// their signs through `sign_epi8`, then +16; the offset comes back out through the
/// column's 16-value sums, `(-16·d)·dy · Σ s·bsum`, one fmadd per block ahead of the
/// codes' fmadd, with `d` the f16 scale undivided. i16: a maddubs pair peaks at
/// 2·31·128 = 7936 on any column bytes, so saturation is unreachable. i32: a lane
/// sums 8 sub-blocks × 31 × 2 × 7936 < 2^24, so its f32 conversion is exact.
///
/// TWIN of [`dot_iq3s_q8k_tile_avx2`], its column tile, and of
/// [`dot_iq3xxs_q8k_avx2`] past the weight decode: any edit to the float or integer
/// steps of one is made in all three.
///
/// # Safety
/// Caller must ensure AVX2+FMA+F16C are available and buffers match `nb` blocks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
unsafe fn dot_iq3s_q8k_avx2(wrow: &[u8], acol: &[u8], nb: usize) -> f32 {
    // SAFETY: AVX2+FMA+F16C present and both slices hold nb blocks per contract.
    unsafe {
        // Scales8KBase::shuffle: scale b onto the two 16-value sums of sub-block b.
        let mins_lo = _mm_set_epi32(0x0706_0706, 0x0504_0504, 0x0302_0302, 0x0100_0100);
        let mins_hi = _mm_set_epi32(0x0f0e_0f0e, 0x0d0c_0d0c, 0x0b0a_0b0a, 0x0908_0908);
        let mut accd = _mm256_setzero_ps();

        for i in 0..nb {
            let blk = wrow.as_ptr().add(IQ3S_BLOCK * i);
            // SAFETY: one unaligned u16 read inside the block.
            let d = f16c_to_f32((blk as *const u16).read_unaligned());
            let sc16 = iq3s_scales(blk);
            let mins = _mm256_set_m128i(
                _mm_shuffle_epi8(sc16, mins_hi),
                _mm_shuffle_epi8(sc16, mins_lo),
            );
            let all_scales = _mm256_set_m128i(sc16, sc16);
            let dmin = -d * f32::from(IQ3S_MIN);

            let col = acol.as_ptr().add(Q8K_STRIDE * i);
            // SAFETY: the column block's f32 d @0 and its i16 bsums[16] @264.
            let dy = (col as *const f32).read_unaligned();
            let bsums = _mm256_loadu_si256(col.add(264) as *const __m256i);
            accd = _mm256_fmadd_ps(
                _mm256_set1_ps(dmin * dy),
                _mm256_cvtepi32_ps(_mm256_madd_epi16(mins, bsums)),
                accd,
            );

            let mut sumi = _mm256_setzero_si256();
            for j in 0..2 {
                let mut p = [_mm256_setzero_si256(); 4];
                for (k, pk) in p.iter_mut().enumerate() {
                    let b = 4 * j + k;
                    let values = iq3s_values(blk, b);
                    // set_scales_8: sub-block b's scale on every i16 lane.
                    let sc = _mm256_shuffle_epi8(
                        all_scales,
                        _mm256_set1_epi16(((2 * b) | ((2 * b + 1) << 8)) as i16),
                    );
                    // SAFETY: 32 codes of sub-block b inside the column block.
                    let q8 = _mm256_loadu_si256(col.add(8 + 32 * b) as *const __m256i);
                    *pk = _mm256_madd_epi16(sc, _mm256_maddubs_epi16(values, q8));
                }
                // multiply_add: exact i32 sums, so the order is free; ik's is kept.
                if j == 0 {
                    sumi = _mm256_add_epi32(
                        _mm256_add_epi32(p[0], p[2]),
                        _mm256_add_epi32(p[1], p[3]),
                    );
                } else {
                    sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p[0], p[2]));
                    sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p[1], p[3]));
                }
            }
            accd = _mm256_fmadd_ps(_mm256_set1_ps(d * dy), _mm256_cvtepi32_ps(sumi), accd);
        }

        hsum_float_8(accd)
    }
}

/// The IQ3_S tile: one row's blocks against up to [`TILE_COLS`] Q8_K columns. Per
/// block the weight side once — the eight sub-blocks' values with their signs and
/// the offset, each sub-block's scale vector, the bsums fold's `mins`, `d` — then per
/// column [`q8k_block_step`], the one-column kernel's float and integer steps (shared
/// with the IQ3_XXS and IQ4_XS tiles). Each column keeps its own `accd` and ends in
/// `hsum_float_8`, so column `c` sees the one-column kernel's operations in its order.
///
/// TWIN of [`dot_iq3s_q8k_avx2`]: any edit to the float or integer steps of one is
/// made in both.
///
/// # Safety
/// CPU must support AVX2+FMA+F16C; `wrow` must hold `nb` IQ3_S blocks and each
/// `acols[c]` point at `nb` Q8_K blocks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
unsafe fn dot_iq3s_q8k_tile_avx2<const C: usize>(
    wrow: &[u8],
    acols: &[*const u8; C],
    nb: usize,
) -> [f32; C] {
    const { assert!(C != 0 && C <= TILE_COLS) };
    // SAFETY: AVX2+FMA+F16C present, `wrow` holds nb blocks and every column
    // nb Q8_K blocks per contract.
    unsafe {
        // Scales8KBase::shuffle: scale b onto the two 16-value sums of sub-block b.
        let mins_lo = _mm_set_epi32(0x0706_0706, 0x0504_0504, 0x0302_0302, 0x0100_0100);
        let mins_hi = _mm_set_epi32(0x0f0e_0f0e, 0x0d0c_0d0c, 0x0b0a_0b0a, 0x0908_0908);
        let mut accd = [_mm256_setzero_ps(); C];

        for i in 0..nb {
            let blk = wrow.as_ptr().add(IQ3S_BLOCK * i);
            // SAFETY: one unaligned u16 read inside the block.
            let d = f16c_to_f32((blk as *const u16).read_unaligned());
            let sc16 = iq3s_scales(blk);
            let mins = _mm256_set_m128i(
                _mm_shuffle_epi8(sc16, mins_hi),
                _mm_shuffle_epi8(sc16, mins_lo),
            );
            let all_scales = _mm256_set_m128i(sc16, sc16);
            let dmin = -d * f32::from(IQ3S_MIN);

            let mut values = [_mm256_setzero_si256(); 8];
            let mut scales = [_mm256_setzero_si256(); 8];
            for b in 0..8 {
                values[b] = iq3s_values(blk, b);
                // set_scales_8: sub-block b's scale on every i16 lane.
                scales[b] = _mm256_shuffle_epi8(
                    all_scales,
                    _mm256_set1_epi16(((2 * b) | ((2 * b + 1) << 8)) as i16),
                );
            }

            for (col, a) in acols.iter().zip(accd.iter_mut()) {
                // SAFETY: the column's block i, inside the validated column.
                *a = q8k_block_step(&values, &scales, mins, d, dmin, col.add(Q8K_STRIDE * i), *a);
            }
        }

        // Reduction order is load-bearing: the one-column kernel's, per column.
        let mut out = [0.0f32; C];
        for (o, a) in out.iter_mut().zip(&accd) {
            *o = hsum_float_8(*a);
        }
        out
    }
}

/// Mirror of `dot_iq3s_q8k_avx2`, bit-identical: the integer lanes are exact sums
/// (no saturation, no wrap — the kernel's bounds), so they are computed directly; the
/// two fmadds per block and the final reduction are the kernel's.
fn dot_iq3s_q8k_emul(wrow: &[u8], acol: &[u8], nb: usize) -> f32 {
    let mut accd = [0.0f32; 8];
    for i in 0..nb {
        let blk = &wrow[IQ3S_BLOCK * i..IQ3S_BLOCK * (i + 1)];
        let d = half_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let qs = &blk[2..66];
        let qh = &blk[66..74];
        let signs = &blk[74..106];
        let scales = &blk[106..110];
        let sc: [i32; 8] =
            std::array::from_fn(|b| 2 * i32::from((scales[b / 2] >> (4 * (b % 2))) & 0x0f) + 1);
        let dmin = -d * f32::from(IQ3S_MIN);

        let col = &acol[Q8K_STRIDE * i..Q8K_STRIDE * (i + 1)];
        let dy = f32::from_le_bytes(col[0..4].try_into().unwrap());
        let bsum = |t: usize| i32::from(i16::from_le_bytes([col[264 + 2 * t], col[265 + 2 * t]]));
        for (l, a) in accd.iter_mut().enumerate() {
            let m = sc[l] * bsum(2 * l) + sc[l] * bsum(2 * l + 1);
            *a = (dmin * dy).mul_add(m as f32, *a);
        }

        let mut sumi = [0i32; 8];
        for (b, &s) in sc.iter().enumerate() {
            for n in 0..32 {
                let word = n / 4;
                let idx = usize::from(qs[8 * b + word]) | (usize::from((qh[b] >> word) & 1) << 8);
                let mag = IQ3S_GRID[idx].to_le_bytes()[n % 4] as i8;
                let v = if (signs[4 * b + n / 8] >> (n % 8)) & 1 != 0 {
                    -mag
                } else {
                    mag
                };
                let u = i32::from(v.wrapping_add(IQ3S_MIN) as u8);
                let a = i32::from(col[8 + 32 * b + n] as i8);
                sumi[n / 4] += s * u * a;
            }
        }
        for (a, &si) in accd.iter_mut().zip(&sumi) {
            *a = (d * dy).mul_add(si as f32, *a);
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

// ------------------------------------------------------------ Q2_K and IQ2_XS x Q8_K
// Both pair with Q8_K and end in the same per-block steps as IQ3_XXS, IQ3_S and IQ4_XS
// ([`q8k_block_step`]); they differ from those in carrying one scale per 16 values, so a
// sub-block's scale vector holds its two scales on the two 128-bit lanes, and the bsums fold's
// `mins` are 16 lanes of their own.

/// One block's weight side apart from its codes, for the Q8_K dots with a scale per 16 values: the
/// fold's 16 `mins` (the bsums fold's weights), the 16 `scales` (one i16 lane each), `d` and the
/// fold's factor `dmin`.
#[cfg(target_arch = "x86_64")]
struct Q8kFold {
    mins: __m256i,
    scales: __m256i,
    d: f32,
    dmin: f32,
}

/// ik's `set_scales_16` for sub-block `b` of a block's 16 scales (`all`, one i16 each): scale `2b`
/// on the low 128-bit lane and `2b + 1` on the high lane.
///
/// # Safety
/// AVX2 must be available (taken from the caller, being `#[inline(always)]`).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn scale_lanes(all: __m256i, b: usize) -> __m256i {
    // SAFETY: register-only intrinsics.
    unsafe {
        let half = if b < 4 {
            _mm256_castsi256_si128(all)
        } else {
            _mm256_extracti128_si256::<1>(all)
        };
        let k = b % 4;
        let lanes = _mm256_set_m128i(
            _mm_set1_epi16(((4 * k + 2) | ((4 * k + 3) << 8)) as i16),
            _mm_set1_epi16(((4 * k) | ((4 * k + 1) << 8)) as i16),
        );
        _mm256_shuffle_epi8(_mm256_broadcastsi128_si256(half), lanes)
    }
}

/// One column's pass over a row's blocks for the Q8_K dots with a scale per 16 values: per block
/// the fold's fmadd, then the eight sub-blocks' `madd(scale, maddubs(values, q8))` and the codes'
/// fmadd, [`q8k_block_step`]'s float and integer steps with the weight side supplied by `fold(i)`
/// (block `i`'s scales) and `values(i, b)` (its sub-block `b`'s 32 codes on maddubs' unsigned
/// side).
///
/// # Safety
/// AVX2+FMA must be available (taken from the caller, being `#[inline(always)]`); `acol` must
/// point at `nb` Q8_K blocks and the closures read `nb` blocks of the row.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn q8k_dot_row(
    nb: usize,
    acol: *const u8,
    fold: impl Fn(usize) -> Q8kFold,
    values: impl Fn(usize, usize) -> __m256i,
) -> f32 {
    // SAFETY: the column's block i, inside the validated column.
    unsafe {
        let mut accd = _mm256_setzero_ps();
        for i in 0..nb {
            let f = fold(i);
            let col = acol.add(Q8K_STRIDE * i);
            let dy = (col as *const f32).read_unaligned();
            let bsums = _mm256_loadu_si256(col.add(264) as *const __m256i);
            accd = _mm256_fmadd_ps(
                _mm256_set1_ps(f.dmin * dy),
                _mm256_cvtepi32_ps(_mm256_madd_epi16(f.mins, bsums)),
                accd,
            );
            let mut sumi = _mm256_setzero_si256();
            for j in 0..2 {
                let mut p = [_mm256_setzero_si256(); 4];
                for (k, pk) in p.iter_mut().enumerate() {
                    let b = 4 * j + k;
                    let q8 = _mm256_loadu_si256(col.add(8 + 32 * b) as *const __m256i);
                    *pk = _mm256_madd_epi16(
                        scale_lanes(f.scales, b),
                        _mm256_maddubs_epi16(values(i, b), q8),
                    );
                }
                // multiply_add: exact i32 sums, so the order is free; ik's is kept.
                if j == 0 {
                    sumi = _mm256_add_epi32(
                        _mm256_add_epi32(p[0], p[2]),
                        _mm256_add_epi32(p[1], p[3]),
                    );
                } else {
                    sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p[0], p[2]));
                    sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p[1], p[3]));
                }
            }
            accd = _mm256_fmadd_ps(_mm256_set1_ps(f.d * dy), _mm256_cvtepi32_ps(sumi), accd);
        }
        hsum_float_8(accd)
    }
}

/// The tile over the same weight side: per block `fold(i)` and the eight sub-blocks' codes once,
/// then per column [`q8k_block_step`] — the integer and float steps [`q8k_dot_row`] runs, so
/// column `c` sees its operations in its order. Each column keeps its own `accd` and ends in
/// `hsum_float_8`.
///
/// # Safety
/// AVX2+FMA must be available (taken from the caller, being `#[inline(always)]`); each
/// `acols[c]` must point at `nb` Q8_K blocks and the closures read `nb` blocks of the row.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn q8k_dot_tile<const C: usize>(
    nb: usize,
    acols: &[*const u8; C],
    fold: impl Fn(usize) -> Q8kFold,
    values: impl Fn(usize, usize) -> __m256i,
) -> [f32; C] {
    const { assert!(C != 0 && C <= TILE_COLS) };
    // SAFETY: each column's block i, inside the validated column.
    unsafe {
        let mut accd = [_mm256_setzero_ps(); C];
        for i in 0..nb {
            let f = fold(i);
            let codes: [__m256i; 8] = std::array::from_fn(|b| values(i, b));
            let scales: [__m256i; 8] = std::array::from_fn(|b| scale_lanes(f.scales, b));
            for (col, a) in acols.iter().zip(accd.iter_mut()) {
                *a = q8k_block_step(
                    &codes,
                    &scales,
                    f.mins,
                    f.d,
                    f.dmin,
                    col.add(Q8K_STRIDE * i),
                    *a,
                );
            }
        }
        // Reduction order is load-bearing: the one-column kernel's, per column.
        let mut out = [0.0f32; C];
        for (o, a) in out.iter_mut().zip(&accd) {
            *o = hsum_float_8(*a);
        }
        out
    }
}

/// `block_q2_K` (ggml-common.h): scales[16] @0 (one byte per 16 values: scale in the low
/// nibble, min in the high), qs[64] @16, f16 d @80, f16 dmin @82. Sub-block `b = 4j + k` (32
/// values) is bits `2k` of qs bytes `32j .. 32j + 32`; a value is `d·scale·q − dmin·min`.
const Q2K_BLOCK: usize = 84;

/// ik's `DequantizerQ2K::new_block` (iqk_gemm_kquants.cpp:511): the scale nibbles the scale
/// lanes, the min nibbles the fold's `mins`, and the fold's factor is `-dmin`.
///
/// # Safety
/// AVX2 and F16C must be available (taken from the caller, being `#[inline(always)]`); `blk`
/// must point at a whole Q2_K block.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn q2k_fold(blk: *const u8) -> Q8kFold {
    // SAFETY: the 16 scale bytes and the two f16 inside the block.
    unsafe {
        let sm = _mm_loadu_si128(blk as *const __m128i);
        let m4 = _mm_set1_epi8(0x0f);
        Q8kFold {
            mins: _mm256_cvtepi8_epi16(_mm_and_si128(_mm_srli_epi16::<4>(sm), m4)),
            scales: _mm256_cvtepi8_epi16(_mm_and_si128(sm, m4)),
            d: f16c_to_f32((blk.add(80) as *const u16).read_unaligned()),
            dmin: -f16c_to_f32((blk.add(82) as *const u16).read_unaligned()),
        }
    }
}

/// ik's `Q2Bits::prepare` (iqk_common.h:512): sub-block `b`'s two-bit codes, the unsigned side as
/// they are.
///
/// # Safety
/// AVX2 must be available (taken from the caller, being `#[inline(always)]`); `blk` must point at
/// a whole Q2_K block and `b` be below 8.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn q2k_values(blk: *const u8, b: usize) -> __m256i {
    // SAFETY: 32 code bytes inside the block.
    unsafe {
        let q = _mm256_loadu_si256(blk.add(16 + 32 * (b / 4)) as *const __m256i);
        let shifted = _mm256_srl_epi16(q, _mm_cvtsi32_si128(2 * (b % 4) as i32));
        _mm256_and_si256(shifted, _mm256_set1_epi8(0x03))
    }
}

/// AVX2+FMA+F16C row dot: Q2_K weights against a Q8_K column.
///
/// ik's form: the min fold `(-dmin)·dy · Σ min·bsum` first, then the eight sub-blocks'
/// `madd(scale, maddubs(code, q8))`, then `d·dy`. i16: a maddubs pair peaks at 2·3·128 = 768,
/// times a scale of 15 and summed in pairs by madd: far below 2^15 and 2^31, so saturation and
/// wrap are unreachable and the i32 lanes convert to f32 exactly.
///
/// # Safety
/// Caller must ensure AVX2+FMA+F16C are available and buffers match `nb` blocks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
unsafe fn dot_q2k_q8k_avx2(wrow: &[u8], acol: &[u8], nb: usize) -> f32 {
    // SAFETY: AVX2+FMA+F16C present and both slices hold nb blocks per contract.
    unsafe {
        let w = wrow.as_ptr();
        q8k_dot_row(
            nb,
            acol.as_ptr(),
            |i| q2k_fold(w.add(Q2K_BLOCK * i)),
            |i, b| q2k_values(w.add(Q2K_BLOCK * i), b),
        )
    }
}

/// The Q2_K tile: one row's blocks against up to [`TILE_COLS`] Q8_K columns ([`q8k_dot_tile`]).
///
/// # Safety
/// CPU must support AVX2+FMA+F16C; `wrow` must hold `nb` Q2_K blocks and each `acols[c]` point
/// at `nb` Q8_K blocks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
unsafe fn dot_q2k_q8k_tile_avx2<const C: usize>(
    wrow: &[u8],
    acols: &[*const u8; C],
    nb: usize,
) -> [f32; C] {
    // SAFETY: AVX2+FMA+F16C present, `wrow` holds nb blocks and every column nb Q8_K blocks per
    // contract.
    unsafe {
        let w = wrow.as_ptr();
        q8k_dot_tile::<C>(
            nb,
            acols,
            |i| q2k_fold(w.add(Q2K_BLOCK * i)),
            |i, b| q2k_values(w.add(Q2K_BLOCK * i), b),
        )
    }
}

/// Mirror of `dot_q2k_q8k_avx2`, bit-identical: the integer lanes are exact sums (no
/// saturation, no wrap — the kernel's bounds), so they are computed directly; the two fmadds per
/// block and the final reduction are the kernel's.
fn dot_q2k_q8k_emul(wrow: &[u8], acol: &[u8], nb: usize) -> f32 {
    let mut accd = [0.0f32; 8];
    for i in 0..nb {
        let blk = &wrow[Q2K_BLOCK * i..Q2K_BLOCK * (i + 1)];
        let (sm, qs) = (&blk[0..16], &blk[16..80]);
        let d = half_to_f32(u16::from_le_bytes([blk[80], blk[81]]));
        let dmin = -half_to_f32(u16::from_le_bytes([blk[82], blk[83]]));
        let scale = |t: usize| i32::from(sm[t] & 0x0f);
        let min = |t: usize| i32::from(sm[t] >> 4);

        let col = &acol[Q8K_STRIDE * i..Q8K_STRIDE * (i + 1)];
        let dy = f32::from_le_bytes(col[0..4].try_into().unwrap());
        let bsum = |t: usize| i32::from(i16::from_le_bytes([col[264 + 2 * t], col[265 + 2 * t]]));
        for (l, a) in accd.iter_mut().enumerate() {
            let m = min(2 * l) * bsum(2 * l) + min(2 * l + 1) * bsum(2 * l + 1);
            *a = (dmin * dy).mul_add(m as f32, *a);
        }

        let mut sumi = [0i32; 8];
        for b in 0..8 {
            let (j, k) = (b / 4, b % 4);
            for n in 0..32 {
                let u = i32::from((qs[32 * j + n] >> (2 * k)) & 3);
                let a = i32::from(col[8 + 32 * b + n] as i8);
                sumi[n / 4] += scale(2 * b + n / 16) * u * a;
            }
        }
        for (a, &si) in accd.iter_mut().zip(&sumi) {
            *a = (d * dy).mul_add(si as f32, *a);
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

/// `block_iq2_xs` (ggml-common.h): f16 d @0, u16 qs[32] @2, scales[8] @66. Code `4b + t` is
/// eight values of sub-block `b`: its low 9 bits index [`IQ2XS_GRID`] (eight magnitudes, one per
/// byte), its high 7 bits are the [`KSIGNS_IQ2XS`] index of its eight signs, bit `n` set meaning
/// value `n` negative. Scale nibble `s` (byte `s / 2`, low nibble for even `s`) covers values
/// `16s .. 16s + 16`, weight `2·nibble + 1`.
const IQ2XS_BLOCK: usize = 74;

/// ik's `DequantizerIQ2XS::minv`: the offset that puts the signed grid values (magnitudes 8, 25,
/// 43) on maddubs' unsigned side, 0..86.
const IQ2XS_MIN: i8 = 43;

/// ik's `DequantizerIQ2XS::load_scales` (iqk_gemm_iquants.cpp:236): the scale weights `2s + 1` on
/// 16 i16 lanes (the scales and the fold's `mins` both), `d = 0.125·f16` and the fold's factor
/// `-43·d`.
///
/// # Safety
/// AVX2 and F16C must be available (taken from the caller, being `#[inline(always)]`); `blk`
/// must point at a whole IQ2_XS block.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn iq2xs_fold(blk: *const u8) -> Q8kFold {
    // SAFETY: the f16 and the 8 scale bytes inside the block.
    unsafe {
        let d = 0.125f32 * f16c_to_f32((blk as *const u16).read_unaligned());
        let tmp = _mm_loadl_epi64(blk.add(66) as *const __m128i);
        let nibbles = _mm_and_si128(
            _mm_unpacklo_epi8(tmp, _mm_srli_epi16::<4>(tmp)),
            _mm_set1_epi8(0x0f),
        );
        let weights =
            _mm256_cvtepi8_epi16(_mm_or_si128(_mm_slli_epi16::<1>(nibbles), _mm_set1_epi8(1)));
        Q8kFold {
            mins: weights,
            scales: weights,
            d,
            dmin: -d * f32::from(IQ2XS_MIN),
        }
    }
}

/// ik's `DequantizerIQ2XS::make4_signed` (iqk_gemm_iquants.cpp:236) for sub-block `b`: its four
/// grid words with their signs and the offset. The signs come from [`KEVEN_SIGNS`] by the 7-bit
/// index (ik's AVX2 build derives the same bytes by a parity shuffle).
///
/// # Safety
/// AVX2 must be available (taken from the caller, being `#[inline(always)]`); `blk` must point at
/// a whole IQ2_XS block and `b` be below 8.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn iq2xs_values(blk: *const u8, b: usize) -> __m256i {
    // SAFETY: four codes of sub-block b inside the block; each 9-bit index is below the grid's
    // 512 entries and each 7-bit index below the sign table's 128.
    unsafe {
        let codes = blk.add(2 + 8 * b) as *const u16;
        let c = [
            codes.read_unaligned(),
            codes.add(1).read_unaligned(),
            codes.add(2).read_unaligned(),
            codes.add(3).read_unaligned(),
        ];
        let q = _mm256_set_epi64x(
            IQ2XS_GRID[usize::from(c[3] & 511)] as i64,
            IQ2XS_GRID[usize::from(c[2] & 511)] as i64,
            IQ2XS_GRID[usize::from(c[1] & 511)] as i64,
            IQ2XS_GRID[usize::from(c[0] & 511)] as i64,
        );
        let signs = _mm256_set_epi64x(
            KEVEN_SIGNS[usize::from(c[3] >> 9)] as i64,
            KEVEN_SIGNS[usize::from(c[2] >> 9)] as i64,
            KEVEN_SIGNS[usize::from(c[1] >> 9)] as i64,
            KEVEN_SIGNS[usize::from(c[0] >> 9)] as i64,
        );
        _mm256_add_epi8(_mm256_sign_epi8(q, signs), _mm256_set1_epi8(IQ2XS_MIN))
    }
}

/// AVX2+FMA+F16C row dot: IQ2_XS weights against a Q8_K column.
///
/// ik's form: a sub-block's 32 grid values take their signs through `sign_epi8`, then +43; the
/// offset comes back out through the column's 16-value sums, `(-43·d)·dy · Σ w·bsum`, one fmadd
/// per block ahead of the codes' fmadd, with `d = 0.125·f16`. i16: a maddubs pair peaks at
/// 2·86·128 = 22016 on any column bytes, so saturation is unreachable. i32: a lane sums 8
/// sub-blocks × 31 × 4 × 86 × 128 < 2^24, so its f32 conversion is exact.
///
/// # Safety
/// Caller must ensure AVX2+FMA+F16C are available and buffers match `nb` blocks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
unsafe fn dot_iq2xs_q8k_avx2(wrow: &[u8], acol: &[u8], nb: usize) -> f32 {
    // SAFETY: AVX2+FMA+F16C present and both slices hold nb blocks per contract.
    unsafe {
        let w = wrow.as_ptr();
        q8k_dot_row(
            nb,
            acol.as_ptr(),
            |i| iq2xs_fold(w.add(IQ2XS_BLOCK * i)),
            |i, b| iq2xs_values(w.add(IQ2XS_BLOCK * i), b),
        )
    }
}

/// The IQ2_XS tile: one row's blocks against up to [`TILE_COLS`] Q8_K columns ([`q8k_dot_tile`]).
///
/// # Safety
/// CPU must support AVX2+FMA+F16C; `wrow` must hold `nb` IQ2_XS blocks and each `acols[c]`
/// point at `nb` Q8_K blocks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
unsafe fn dot_iq2xs_q8k_tile_avx2<const C: usize>(
    wrow: &[u8],
    acols: &[*const u8; C],
    nb: usize,
) -> [f32; C] {
    // SAFETY: AVX2+FMA+F16C present, `wrow` holds nb blocks and every column nb Q8_K blocks per
    // contract.
    unsafe {
        let w = wrow.as_ptr();
        q8k_dot_tile::<C>(
            nb,
            acols,
            |i| iq2xs_fold(w.add(IQ2XS_BLOCK * i)),
            |i, b| iq2xs_values(w.add(IQ2XS_BLOCK * i), b),
        )
    }
}

/// Mirror of `dot_iq2xs_q8k_avx2`, bit-identical: the integer lanes are exact sums (no
/// saturation, no wrap — the kernel's bounds), so they are computed directly; the two fmadds per
/// block and the final reduction are the kernel's.
fn dot_iq2xs_q8k_emul(wrow: &[u8], acol: &[u8], nb: usize) -> f32 {
    let mut accd = [0.0f32; 8];
    for i in 0..nb {
        let blk = &wrow[IQ2XS_BLOCK * i..IQ2XS_BLOCK * (i + 1)];
        let d = 0.125f32 * half_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let qs = &blk[2..66];
        let scales = &blk[66..74];
        let weight = |t: usize| 2 * i32::from((scales[t / 2] >> (4 * (t % 2))) & 0x0f) + 1;
        let dmin = -d * f32::from(IQ2XS_MIN);

        let col = &acol[Q8K_STRIDE * i..Q8K_STRIDE * (i + 1)];
        let dy = f32::from_le_bytes(col[0..4].try_into().unwrap());
        let bsum = |t: usize| i32::from(i16::from_le_bytes([col[264 + 2 * t], col[265 + 2 * t]]));
        for (l, a) in accd.iter_mut().enumerate() {
            let m = weight(2 * l) * bsum(2 * l) + weight(2 * l + 1) * bsum(2 * l + 1);
            *a = (dmin * dy).mul_add(m as f32, *a);
        }

        let mut sumi = [0i32; 8];
        for b in 0..8 {
            for n in 0..32 {
                let code = usize::from(u16::from_le_bytes([
                    qs[8 * b + 2 * (n / 8)],
                    qs[8 * b + 2 * (n / 8) + 1],
                ]));
                let mag = IQ2XS_GRID[code & 511].to_le_bytes()[n % 8] as i8;
                let v = if (KSIGNS_IQ2XS[code >> 9] >> (n % 8)) & 1 != 0 {
                    -mag
                } else {
                    mag
                };
                let u = i32::from(v.wrapping_add(IQ2XS_MIN) as u8);
                let a = i32::from(col[8 + 32 * b + n] as i8);
                sumi[n / 4] += weight(2 * b + n / 16) * u * a;
            }
        }
        for (a, &si) in accd.iter_mut().zip(&sumi) {
            *a = (d * dy).mul_add(si as f32, *a);
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

// ------------------------------------------------------------ IQ2_XXS x Q8_K

/// `block_iq2_xxs` (ggml-common.h): f16 d @0, qs u16[32] @2. Sub-block `b` (32 values)
/// is the eight bytes at `2 + 8·b`: four grid index bytes — byte `l` indexes
/// [`IQ2XXS_GRID`] (eight magnitudes, one per byte) for group `l`'s eight values — then a
/// word whose bits `7l .. 7l + 7` hold group `l`'s [`KSIGNS_IQ2XS`] index and whose top
/// nibble `s` scales the sub-block by `2s + 1`, each value `d·0.125·(2s + 1)·(±grid[j])`.
const IQ2XXS_BLOCK: usize = 66;

/// ik's `DequantizerIQ2XXS::minv` (iqk_gemm_iquants.cpp:228): the offset that puts the
/// signed grid values (magnitudes 8, 25, 43) on maddubs' unsigned side, 0..86.
const IQ2XXS_MIN: i8 = 43;

/// AVX2+FMA+F16C row dot: IQ2_XXS weights against a Q8_K column.
///
/// ik's form (`mul_mat_qX_K_q8_K_IQ_N<DequantizerIQ2XXS, nrc_y>`, iqk_gemm_iquants.cpp:788
/// and :148, the body its AVX2 build runs at every column count, :1011): a sub-block's 32
/// grid values take their signs on the weight side through `sign_epi8`, then +43
/// ([`iq2xxs_pack`]); the offset comes back out through the column's 16-value sums,
/// `(-43·d)·dy · Σ s·bsum`, one fmadd per block ahead of the codes' fmadd, with
/// `d = 0.125·f16` ([`q8k_block_step`], the IQ3_XXS, IQ3_S and IQ4_XS kernels' steps).
/// i16: a maddubs pair peaks at 2·86·128 = 22016 on any column bytes, so saturation is
/// unreachable. i32: a lane sums 8 sub-blocks × 31 × 4 × 86 × 128 < 2^24, so its f32
/// conversion is exact.
///
/// TWIN of [`iq2xxs_wide_group`], the body of its wide tile ([`dot_iq2xxs_q8k_wide_avx2`])
/// and of its column tile ([`dot_iq2xxs_q8k_tile_avx2`], the wide tile's one group): each
/// column there runs [`q8k_block_step`]'s float steps, so any edit to them is made in both.
///
/// # Safety
/// Caller must ensure AVX2+FMA+F16C are available and buffers match `nb` blocks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
unsafe fn dot_iq2xxs_q8k_avx2(wrow: &[u8], acol: &[u8], nb: usize) -> f32 {
    // SAFETY: AVX2+FMA+F16C present and both slices hold nb blocks per contract.
    unsafe {
        let mut accd = _mm256_setzero_ps();
        for i in 0..nb {
            let p = iq2xxs_pack(wrow.as_ptr().add(IQ2XXS_BLOCK * i));
            let col = acol.as_ptr().add(Q8K_STRIDE * i);
            accd = q8k_block_step(&p.values, &p.scales, p.mins, p.d, p.dmin, col, accd);
        }
        hsum_float_8(accd)
    }
}

/// The IQ2_XXS tile: one row's blocks against up to [`TILE_COLS`] Q8_K columns, the wide
/// tile's one group ([`dot_iq2xxs_q8k_wide_avx2`]): per block the weight side once into
/// the block's pack ([`iq2xxs_pack`]), then [`iq2xxs_wide_group`] over the `C` columns —
/// per sub-block the pack's values and scale against each column's codes into that
/// column's sum, and per column [`q8k_block_step`]'s two fmadds. Each column keeps its own
/// `accd` and ends in `hsum_float_8`, so every value is [`dot_iq2xxs_q8k_avx2`]'s bit for
/// bit (the wide tile's argument).
///
/// # Safety
/// CPU must support AVX2+FMA+F16C; `wrow` must hold `nb` IQ2_XXS blocks and each
/// `acols[c]` point at `nb` Q8_K blocks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
unsafe fn dot_iq2xxs_q8k_tile_avx2<const C: usize>(
    wrow: &[u8],
    acols: &[*const u8; C],
    nb: usize,
) -> [f32; C] {
    const { assert!(C != 0 && C <= TILE_COLS && C <= WIDE_GROUP) };
    // SAFETY: AVX2+FMA+F16C present, `wrow` holds nb blocks and every column nb Q8_K
    // blocks per contract.
    unsafe {
        let mut accd = [_mm256_setzero_ps(); C];

        for i in 0..nb {
            let pack = iq2xxs_pack(wrow.as_ptr().add(IQ2XXS_BLOCK * i));
            iq2xxs_wide_group::<C>(&pack, acols, Q8K_STRIDE * i, &mut accd);
        }

        // Reduction order is load-bearing: the one-column kernel's, per column.
        let mut out = [0.0f32; C];
        for (o, a) in out.iter_mut().zip(&accd) {
            *o = hsum_float_8(*a);
        }
        out
    }
}

/// The IQ2_XXS wide tile: one row's blocks against up to [`WIDE_COLS`] Q8_K
/// columns. Per block the weight side once ([`iq2xxs_pack`]), then the columns in
/// groups of up to [`WIDE_GROUP`] ([`iq2xxs_wide_group`]): per sub-block the pack's
/// values and scale, and per column one `maddubs` against its codes and one `madd`
/// into the column's sum; then per column the fold's fmadd and the codes' fmadd,
/// [`q8k_block_step`]'s float steps in its order. The integer sums are exact, so
/// their order is free, and the closing `hsum_float_8` is the one-column kernel's,
/// so every column's value is [`dot_iq2xxs_q8k_avx2`]'s bit for bit.
///
/// TWIN of [`dot_iq2xxs_q8k_avx2`]: any edit to the float or integer steps of one
/// is made in both.
///
/// # Safety
/// CPU must support AVX2+FMA+F16C; `wrow` must hold `nb` IQ2_XXS blocks, `acols`
/// `1..=WIDE_COLS` columns each pointing at `nb` Q8_K blocks, and `out` one value
/// a column.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
unsafe fn dot_iq2xxs_q8k_wide_avx2(wrow: &[u8], acols: &[*const u8], nb: usize, out: &mut [f32]) {
    let c = acols.len();
    assert!(
        (1..=WIDE_COLS).contains(&c) && out.len() == c,
        "the IQ2_XXS wide tile takes 1..={WIDE_COLS} columns and one output a column: \
         {c} columns, {} outputs",
        out.len()
    );
    // SAFETY: AVX2+FMA+F16C present, `wrow` holds nb blocks and every column nb Q8_K
    // blocks per contract.
    unsafe {
        let mut accd = [_mm256_setzero_ps(); WIDE_COLS];

        for i in 0..nb {
            let pack = iq2xxs_pack(wrow.as_ptr().add(IQ2XXS_BLOCK * i));

            let at = Q8K_STRIDE * i;
            let mut g0 = 0;
            while g0 < c {
                let g = (c - g0).min(WIDE_GROUP);
                let (cols, acc) = (&acols[g0..g0 + g], &mut accd[g0..g0 + g]);
                match g {
                    1 => iq2xxs_wide_group::<1>(&pack, cols, at, acc),
                    2 => iq2xxs_wide_group::<2>(&pack, cols, at, acc),
                    3 => iq2xxs_wide_group::<3>(&pack, cols, at, acc),
                    4 => iq2xxs_wide_group::<4>(&pack, cols, at, acc),
                    5 => iq2xxs_wide_group::<5>(&pack, cols, at, acc),
                    6 => iq2xxs_wide_group::<6>(&pack, cols, at, acc),
                    7 => iq2xxs_wide_group::<7>(&pack, cols, at, acc),
                    _ => iq2xxs_wide_group::<WIDE_GROUP>(&pack, cols, at, acc),
                }
                g0 += g;
            }
        }

        // Reduction order is load-bearing: the one-column kernel's, per column.
        for (o, a) in out.iter_mut().zip(&accd) {
            *o = hsum_float_8(*a);
        }
    }
}

/// An IQ2_XXS block's weight side in ik's form, built once for every column of a call
/// ([`iq2xxs_pack`]).
#[cfg(target_arch = "x86_64")]
struct Iq2xxsPack {
    /// Sub-block `b`'s 32 grid magnitudes with their signs and the offset
    /// [`IQ2XXS_MIN`], maddubs' unsigned side (`make4_signed`).
    values: [__m256i; 8],
    /// Sub-block `b`'s odd scale `2s + 1` on every i16 lane (`set_scales_8`).
    scales: [__m256i; 8],
    /// The bsums fold's weights: scale `b` on the two 16-value sums of sub-block `b`
    /// (`Scales8KBase::shuffle`).
    mins: __m256i,
    /// `0.125·f16` (`load_scales`).
    d: f32,
    /// The fold's factor `-43·d` (`new_block`).
    dmin: f32,
}

/// ik's `DequantizerIQ2XXS` (iqk_gemm_iquants.cpp:148) on one block: `load_scales` reads
/// the eight scale nibbles as one vector (here the high word of each sub-block's eight
/// bytes, gathered by a shuffle and a qword permute), `make4_signed` signs each
/// sub-block's grid words through [`keven_signs_word`] and adds [`IQ2XXS_MIN`].
///
/// # Safety
/// AVX2 and F16C must be available (taken from the caller, being `#[inline(always)]`);
/// `blk` must point at a whole IQ2_XXS block.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn iq2xxs_pack(blk: *const u8) -> Iq2xxsPack {
    // SAFETY: AVX2 and F16C present per contract; every read is inside the block.
    unsafe {
        let d = 0.125f32 * f16c_to_f32((blk as *const u16).read_unaligned());
        // Words 1 and 3 of each 128-bit lane of qs[0..32] and qs[32..64] are the
        // sign/scale words of sub-blocks (0, 1, 4, 5 | 2, 3, 6, 7); the permute puts
        // them in sub-block order.
        let lo = _mm256_castsi256_ps(_mm256_loadu_si256(blk.add(2) as *const __m256i));
        let hi = _mm256_castsi256_ps(_mm256_loadu_si256(blk.add(34) as *const __m256i));
        let words =
            _mm256_permute4x64_epi64::<0b11_01_10_00>(_mm256_castps_si256(_mm256_shuffle_ps::<
                0b11_01_11_01,
            >(lo, hi)));
        let sc32 = _mm256_or_si256(
            _mm256_slli_epi32::<1>(_mm256_srli_epi32::<28>(words)),
            _mm256_set1_epi32(1),
        );
        let sc16 = _mm_packs_epi32(
            _mm256_castsi256_si128(sc32),
            _mm256_extracti128_si256::<1>(sc32),
        );
        // Scales8KBase::shuffle: scale b onto the two 16-value sums of sub-block b.
        let mins = _mm256_set_m128i(
            _mm_shuffle_epi8(
                sc16,
                _mm_set_epi32(0x0f0e_0f0e, 0x0d0c_0d0c, 0x0b0a_0b0a, 0x0908_0908),
            ),
            _mm_shuffle_epi8(
                sc16,
                _mm_set_epi32(0x0706_0706, 0x0504_0504, 0x0302_0302, 0x0100_0100),
            ),
        );
        let all_scales = _mm256_set_m128i(sc16, sc16);
        let min_value = _mm256_set1_epi8(IQ2XXS_MIN);

        let mut values = [_mm256_setzero_si256(); 8];
        let mut scales = [_mm256_setzero_si256(); 8];
        for b in 0..8 {
            let qs = blk.add(2 + 8 * b);
            let aux = (qs.add(4) as *const u32).read_unaligned();
            // Each of the four grid index bytes indexes the 256-entry grid.
            let q2 = _mm256_set_epi64x(
                IQ2XXS_GRID[usize::from(*qs.add(3))] as i64,
                IQ2XXS_GRID[usize::from(*qs.add(2))] as i64,
                IQ2XXS_GRID[usize::from(*qs.add(1))] as i64,
                IQ2XXS_GRID[usize::from(*qs.add(0))] as i64,
            );
            values[b] = _mm256_add_epi8(_mm256_sign_epi8(q2, keven_signs_word(aux)), min_value);
            // set_scales_8: sub-block b's scale on every i16 lane.
            scales[b] = _mm256_shuffle_epi8(
                all_scales,
                _mm256_set1_epi16(((2 * b) | ((2 * b + 1) << 8)) as i16),
            );
        }
        Iq2xxsPack {
            values,
            scales,
            mins,
            d,
            dmin: -d * f32::from(IQ2XXS_MIN),
        }
    }
}

/// One group of [`dot_iq2xxs_q8k_wide_avx2`]'s columns against block `pack`: per
/// sub-block, in order, each column's 32 codes at `at + 8 + 32·b` through `maddubs`
/// against the sub-block's values and `madd` by its scale into the column's sum; then
/// per column [`q8k_block_step`]'s float steps — the fold's fmadd of
/// `madd(mins, bsums)` (the bsums at `at + 264`) against `dmin·dy`, then the codes'
/// fmadd against `d·dy` (`dy` the f32 at `at`) — onto its accumulator.
///
/// # Safety
/// AVX2+FMA must be available (taken from the caller, being `#[inline(always)]`);
/// `cols` must hold `G` columns, each with a Q8_K block at byte `at`, and `acc`
/// `G` accumulators.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn iq2xxs_wide_group<const G: usize>(
    pack: &Iq2xxsPack,
    cols: &[*const u8],
    at: usize,
    acc: &mut [__m256],
) {
    let cols: &[*const u8; G] = cols
        .try_into()
        .expect("a wide group's columns are its width");
    let acc: &mut [__m256; G] = acc
        .try_into()
        .expect("a wide group's accumulators are its width");
    // SAFETY: AVX2+FMA present per contract; every column holds a Q8_K block at `at`.
    unsafe {
        let mut sumi = [_mm256_setzero_si256(); G];
        for (b, (&values, &sc)) in pack.values.iter().zip(&pack.scales).enumerate() {
            for (col, s) in cols.iter().zip(sumi.iter_mut()) {
                let q8 = _mm256_loadu_si256(col.add(at + 8 + 32 * b) as *const __m256i);
                *s = _mm256_add_epi32(*s, _mm256_madd_epi16(sc, _mm256_maddubs_epi16(values, q8)));
            }
        }
        for ((col, s), a) in cols.iter().zip(&sumi).zip(acc.iter_mut()) {
            let dy = (col.add(at) as *const f32).read_unaligned();
            let bsums = _mm256_loadu_si256(col.add(at + 264) as *const __m256i);
            let folded = _mm256_fmadd_ps(
                _mm256_set1_ps(pack.dmin * dy),
                _mm256_cvtepi32_ps(_mm256_madd_epi16(pack.mins, bsums)),
                *a,
            );
            *a = _mm256_fmadd_ps(_mm256_set1_ps(pack.d * dy), _mm256_cvtepi32_ps(*s), folded);
        }
    }
}

/// Mirror of `dot_iq2xxs_q8k_avx2`, bit-identical: the integer lanes are exact sums (no
/// saturation, no wrap — the kernel's bounds), so they are computed directly; the two
/// fmadds per block and the final reduction are the kernel's.
fn dot_iq2xxs_q8k_emul(wrow: &[u8], acol: &[u8], nb: usize) -> f32 {
    let mut accd = [0.0f32; 8];
    for i in 0..nb {
        let blk = &wrow[IQ2XXS_BLOCK * i..IQ2XXS_BLOCK * (i + 1)];
        let d = 0.125f32 * half_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let words: [u32; 8] = std::array::from_fn(|b| {
            u32::from_le_bytes(blk[6 + 8 * b..10 + 8 * b].try_into().unwrap())
        });
        let sc: [i32; 8] = std::array::from_fn(|b| 2 * (words[b] >> 28) as i32 + 1);
        let dmin = -d * f32::from(IQ2XXS_MIN);

        let col = &acol[Q8K_STRIDE * i..Q8K_STRIDE * (i + 1)];
        let dy = f32::from_le_bytes(col[0..4].try_into().unwrap());
        let bsum = |t: usize| i32::from(i16::from_le_bytes([col[264 + 2 * t], col[265 + 2 * t]]));
        for (l, a) in accd.iter_mut().enumerate() {
            let m = sc[l] * bsum(2 * l) + sc[l] * bsum(2 * l + 1);
            *a = (dmin * dy).mul_add(m as f32, *a);
        }

        // A lane takes the values 4·lane .. 4·lane + 4 of every sub-block.
        let mut sumi = [0i32; 8];
        for (b, &s) in sc.iter().enumerate() {
            let idx = &blk[2 + 8 * b..6 + 8 * b];
            for n in 0..32 {
                let mag = IQ2XXS_GRID[usize::from(idx[n / 8])].to_le_bytes()[n % 8] as i8;
                let signs = KSIGNS_IQ2XS[((words[b] >> (7 * (n / 8))) & 127) as usize];
                let v = if (signs >> (n % 8)) & 1 != 0 {
                    -mag
                } else {
                    mag
                };
                let u = i32::from(v.wrapping_add(IQ2XXS_MIN) as u8);
                let a = i32::from(col[8 + 32 * b + n] as i8);
                sumi[n / 4] += s * u * a;
            }
        }
        for (a, &si) in accd.iter_mut().zip(&sumi) {
            *a = (d * dy).mul_add(si as f32, *a);
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

// ------------------------------------------------------------- IQ1_S x Q8_K

/// `sizeof(block_iq1_s)` (ggml-common.h): f16 d @0, qs u8[32] @2, qh u16[8] @34 — 50 bytes /
/// 256 values. Sub-block `b`'s four index bytes are `qs[4b .. 4b + 4]`, its `qh` word's bits
/// `3l .. 3l + 3` the three high index bits of group `l`, bits 12..15 the scale and bit 15 the
/// delta's sign.
const IQ1S_BLOCK: usize = 50;

/// ik's `iq1s_grid_us` (iqk_gemm_1bit.cpp:12): the grid values with +1 a byte, maddubs'
/// unsigned side (bytes 0..=2). The offset is applied a byte at compile time — a `u64` add
/// of `0x0101…` would carry a `0xFF` byte (the value −1) into its neighbour.
static IQ1S_GRID_US: [u64; 2048] = {
    let mut t = [0u64; 2048];
    let mut i = 0;
    while i < 2048 {
        let mut j = 0;
        while j < 8 {
            let byte = ((IQ1S_GRID[i] >> (8 * j)) & 0xff) + 1;
            t[i] |= (byte & 0xff) << (8 * j);
            j += 1;
        }
        i += 1;
    }
    t
};

/// An IQ1_S block's weight side in ik's form, built once for every column of a call
/// ([`iq1s_pack`]).
#[cfg(target_arch = "x86_64")]
struct Iq1sPack {
    /// Sub-block `b`'s 32 grid values + 1 (bytes 0..=2), maddubs' unsigned side —
    /// ik's `iq1s_grid_us`.
    values: [__m256i; 8],
    /// 64-value group `g`'s two sub-block scales `8·(2s + 1)`: four i16 lanes each, in
    /// `2g` then `2g + 1` order a 128-bit lane (ik's `shuffle0`).
    scales: [__m256i; 4],
    /// The bsums fold's weights: sub-block `b`'s `(2s + 1)·(δ − 8)` on the two i16 lanes of
    /// its two 16-value sums.
    deltas: __m256i,
    /// The block's f16 `d`.
    d: f32,
}

/// ik's weight side of one block (`mul_mat_iq1_s_q8_K`, iqk_gemm_1bit.cpp:804-834): the eight
/// `qh` words at once, their bits 12..15 the sub-block scales `2s + 1` and bit 15 the delta
/// sign; each sub-block's four index bytes `qs[4b + l] | (qh >> 3l & 7) << 8` gather a grid
/// entry whose bytes are offset by +1 (`iq1s_grid_us`, iqk_gemm_1bit.cpp:12).
///
/// # Safety
/// AVX2 and F16C must be available (taken from the caller, being `#[inline(always)]`);
/// `blk` must point at a whole IQ1_S block.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn iq1s_pack(blk: *const u8) -> Iq1sPack {
    // SAFETY: AVX2 and F16C present per contract; every read is inside the block.
    unsafe {
        let d = f16c_to_f32((blk as *const u16).read_unaligned());
        let qhb = _mm_loadu_si128(blk.add(34) as *const __m128i);
        let scales128 = _mm_add_epi16(
            _mm_slli_epi16::<1>(_mm_and_si128(_mm_srli_epi16::<12>(qhb), _mm_set1_epi16(7))),
            _mm_set1_epi16(1),
        );
        // δ − 8: -9 when bit 15 of the sub-block's qh is set, -7 otherwise.
        let mask = _mm_cmpeq_epi16(
            _mm_and_si128(qhb, _mm_set1_epi16(-32768)),
            _mm_set1_epi16(-32768),
        );
        let deltas128 = _mm_or_si128(
            _mm_and_si128(mask, _mm_set1_epi16(-9)),
            _mm_andnot_si128(mask, _mm_set1_epi16(-7)),
        );
        let deltas128 = _mm_mullo_epi16(scales128, deltas128);
        let scales128 = _mm_slli_epi16::<3>(scales128);
        let deltas = _mm256_set_m128i(
            _mm_unpackhi_epi16(deltas128, deltas128),
            _mm_unpacklo_epi16(deltas128, deltas128),
        );
        let all_scales = _mm256_broadcastsi128_si256(scales128);
        let mut scales = [_mm256_setzero_si256(); 4];
        let mut shuffle = _mm256_set_epi64x(
            0x0302_0302_0302_0302,
            0x0100_0100_0100_0100,
            0x0302_0302_0302_0302,
            0x0100_0100_0100_0100,
        );
        for s in scales.iter_mut() {
            *s = _mm256_shuffle_epi8(all_scales, shuffle);
            shuffle = _mm256_add_epi8(shuffle, _mm256_set1_epi8(4));
        }
        let mut values = [_mm256_setzero_si256(); 8];
        for (b, vb) in values.iter_mut().enumerate() {
            let qs = blk.add(2 + 4 * b);
            let qh = (blk.add(34 + 2 * b) as *const u16).read_unaligned();
            let grid = |l: usize| {
                let idx = usize::from(*qs.add(l)) | (((qh >> (3 * l)) & 7) as usize) << 8;
                IQ1S_GRID_US[idx]
            };
            // Each of the four index bytes indexes the 2048-entry offset grid.
            *vb = _mm256_set_epi64x(
                grid(3) as i64,
                grid(2) as i64,
                grid(1) as i64,
                grid(0) as i64,
            );
        }
        Iq1sPack {
            values,
            scales,
            deltas,
            d,
        }
    }
}

/// One column's block step of the IQ1_S kernels: per 64-value group the two sub-blocks'
/// `maddubs` partials interleaved by 64-bit halves (ik's `unpacklo/hi_epi64`), the group's
/// two `8·(2s + 1)` scales on the pairs through `madd`, then the deltas' fold against the
/// column's bsums and one fmadd of `d·dy` — ik's integer order, an exact sum. This order is
/// the gate.
///
/// # Safety
/// AVX2+FMA must be available (taken from the caller, being `#[inline(always)]`);
/// `col` must point at a validated Q8_K block.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn iq1s_block_step(pack: &Iq1sPack, col: *const u8, accd: __m256) -> __m256 {
    // SAFETY: the column block's f32 d @0, i16 bsums[16] @264 and the 256 codes @8, all
    // inside the validated block.
    unsafe {
        let mut sumi = _mm256_setzero_si256();
        for (g, &sc) in pack.scales.iter().enumerate() {
            let dot1 = _mm256_maddubs_epi16(
                pack.values[2 * g],
                _mm256_loadu_si256(col.add(8 + 64 * g) as *const __m256i),
            );
            let dot2 = _mm256_maddubs_epi16(
                pack.values[2 * g + 1],
                _mm256_loadu_si256(col.add(40 + 64 * g) as *const __m256i),
            );
            let dot = _mm256_add_epi16(
                _mm256_unpacklo_epi64(dot1, dot2),
                _mm256_unpackhi_epi64(dot1, dot2),
            );
            sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(sc, dot));
        }
        let dy = (col as *const f32).read_unaligned();
        let bsums = _mm256_loadu_si256(col.add(264) as *const __m256i);
        sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(bsums, pack.deltas));
        _mm256_fmadd_ps(_mm256_set1_ps(pack.d * dy), _mm256_cvtepi32_ps(sumi), accd)
    }
}

/// AVX2+FMA+F16C row dot: IQ1_S weights against a Q8_K column.
///
/// ik's form (`mul_mat_iq1_s_q8_K<nrc_y>`, iqk_gemm_1bit.cpp:793, its non-`HAVE_FANCY_SIMD`
/// branches :811, :845-849 and :854-855, the body its AVX2 build runs at every column count
/// below the 32 where it converts the weights to Q8_K_R8 instead, iqk_mul_mat.cpp:315): the
/// grid values offset by +1 onto maddubs' unsigned side, the `8·(2s + 1)` scale inside the
/// i16 dot pairs, and the `±1` delta taken back out through the column's 16-value sums,
/// `(2s + 1)·(δ − 8)·Σ q` of each sub-block, so the whole block is one integer sum and one
/// fmadd of `d·dy`; the row's `0.125` (the code's `d / 8`) closes the hsum.
///
/// i16: a maddubs pair peaks at 2·2·127 = 508 on any column bytes, so saturation is
/// unreachable. i32: a lane sums 4 groups × 2 × 120·508 + 2 × 135·2032 < 2^21, so its f32
/// conversion is exact.
///
/// TWIN of [`dot_iq1s_q8k_tile_avx2`]: each column there runs this kernel's steps, so any
/// edit to them is made in both.
///
/// # Safety
/// Caller must ensure AVX2+FMA+F16C are available and buffers match `nb` blocks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
unsafe fn dot_iq1s_q8k_avx2(wrow: &[u8], acol: &[u8], nb: usize) -> f32 {
    // SAFETY: AVX2+FMA+F16C present and both slices hold nb blocks per contract.
    unsafe {
        let mut accd = _mm256_setzero_ps();
        for i in 0..nb {
            let pack = iq1s_pack(wrow.as_ptr().add(IQ1S_BLOCK * i));
            accd = iq1s_block_step(&pack, acol.as_ptr().add(Q8K_STRIDE * i), accd);
        }
        0.125 * hsum_float_8(accd)
    }
}

/// The IQ1_S tile: one row's blocks against up to [`TILE_COLS`] Q8_K columns, ik's template
/// at `nrc_y` = the column count (its AVX2 build's column counts below its repack
/// threshold): per block the weight side once into the block's pack ([`iq1s_pack`]), then
/// per column [`iq1s_block_step`] onto its own accumulator. Each column keeps the
/// one-column kernel's integer lanes and float order, so every value is
/// [`dot_iq1s_q8k_avx2`]'s bit for bit.
///
/// TWIN of [`dot_iq1s_q8k_avx2`]: any edit to the float or integer steps of one is made in
/// both.
///
/// # Safety
/// CPU must support AVX2+FMA+F16C; `wrow` must hold `nb` IQ1_S blocks and each
/// `acols[c]` point at `nb` Q8_K blocks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
unsafe fn dot_iq1s_q8k_tile_avx2<const C: usize>(
    wrow: &[u8],
    acols: &[*const u8; C],
    nb: usize,
) -> [f32; C] {
    const { assert!(C != 0 && C <= TILE_COLS) };
    // SAFETY: AVX2+FMA+F16C present, `wrow` holds nb blocks and every column nb Q8_K
    // blocks per contract.
    unsafe {
        let mut accd = [_mm256_setzero_ps(); C];
        for i in 0..nb {
            let pack = iq1s_pack(wrow.as_ptr().add(IQ1S_BLOCK * i));
            for (col, a) in acols.iter().zip(&mut accd) {
                *a = iq1s_block_step(&pack, col.add(Q8K_STRIDE * i), *a);
            }
        }
        // Reduction order is load-bearing: the one-column kernel's, per column.
        let mut out = [0.0f32; C];
        for (o, a) in out.iter_mut().zip(&accd) {
            *o = 0.125 * hsum_float_8(*a);
        }
        out
    }
}

/// Mirror of `dot_iq1s_q8k_avx2`, bit-identical: the integer lanes are exact sums (no
/// saturation, no wrap — the kernel's bounds), so each lane's integer is computed directly in
/// the kernel's split — lane `l` takes, per 64-value group, the values `[0..4) ∪ [8..12)`
/// (even `l%4`) or `[4..8) ∪ [12..16)` (odd `l%4`), plus `16·(l/4)`, of sub-block
/// `2·group + (l%4)/2`, and the delta fold of sub-block `l` — and the one fmadd per block
/// and lane, the closing `0.125` and the hsum's order are the kernel's.
fn dot_iq1s_q8k_emul(wrow: &[u8], acol: &[u8], nb: usize) -> f32 {
    // The two value sets a 128-bit lane's i32 halves take of a sub-block.
    const LANE_A: [usize; 8] = [0, 1, 2, 3, 8, 9, 10, 11];
    const LANE_B: [usize; 8] = [4, 5, 6, 7, 12, 13, 14, 15];
    let mut accd = [0.0f32; 8];
    for i in 0..nb {
        let blk = &wrow[IQ1S_BLOCK * i..IQ1S_BLOCK * (i + 1)];
        let d = half_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let qh = |b: usize| u16::from_le_bytes([blk[34 + 2 * b], blk[35 + 2 * b]]);
        // 2s + 1 and the delta of sub-block b.
        let sc = |b: usize| 2 * i32::from((qh(b) >> 12) & 7) + 1;
        let delta = |b: usize| if qh(b) & 0x8000 != 0 { -1 } else { 1 };

        let col = &acol[Q8K_STRIDE * i..Q8K_STRIDE * (i + 1)];
        let dy = f32::from_le_bytes(col[0..4].try_into().unwrap());
        let q8 = |n: usize| i32::from(col[8 + n] as i8);
        // Sub-block b's 32-value code sum, from its two 16-value bsums.
        let bsum32 = |b: usize| {
            (0..2)
                .map(|t| {
                    i32::from(i16::from_le_bytes([
                        col[264 + 4 * b + 2 * t],
                        col[265 + 4 * b + 2 * t],
                    ]))
                })
                .sum::<i32>()
        };
        for (l, a) in accd.iter_mut().enumerate() {
            let (half, q) = (l / 4, l % 4);
            let bases: &[usize; 8] = if q % 2 == 0 { &LANE_A } else { &LANE_B };
            let mut sumi = 0i64;
            for g in 0..4 {
                let sub = 2 * g + q / 2;
                let mut part = 0i64;
                for &base in bases {
                    let v = base + 16 * half;
                    let idx = usize::from(blk[2 + 4 * sub + v / 8])
                        | (((qh(sub) >> (3 * (v / 8))) & 7) as usize) << 8;
                    let u = i64::from(IQ1S_GRID[idx].to_le_bytes()[v % 8] as i8) + 1;
                    part += u * i64::from(q8(32 * sub + v));
                }
                sumi += i64::from(8 * sc(sub)) * part;
            }
            sumi += i64::from(bsum32(l) * (sc(l) * (delta(l) - 8)));
            *a = (d * dy).mul_add(sumi as f32, *a);
        }
    }
    let t = [
        accd[0] + accd[4],
        accd[1] + accd[5],
        accd[2] + accd[6],
        accd[3] + accd[7],
    ];
    0.125 * ((t[0] + t[2]) + (t[1] + t[3]))
}

// --------------------------------------------------------- MXFP4 x Q8_2_X4

/// `block_mxfp4` (ggml-common.h:183): E8M0 scale e @0, qs[16] @1 — value j (j < 16)
/// the low nibble of qs[j], value j + 16 the high nibble.
const MXFP4_BLOCK: usize = 17;

/// ik's `kvalues_mxfp4_unsigned` (iqk_gemm_legacy_quants.cpp:621): the doubled E2M1
/// values plus 12, maddubs' unsigned side; the 12 comes back out through the
/// activation block sums.
static KVALUES_MXFP4_U: [u8; 16] = {
    let mut t = [0u8; 16];
    let mut i = 0;
    while i < 16 {
        t[i] = (KVALUES_MXFP4[i] + 12) as u8;
        i += 1;
    }
    t
};

/// One 32-value 4-bit block's codes through a 16-entry byte table: the low nibble of
/// qs[j] is value j, the high nibble value j + 16 (ik's `Dequantizer4bit::dequant` plus
/// the dequantizer's own `shuffle_epi8`, iqk_gemm_legacy_quants.cpp:568 and the
/// `prepare16` shuffle, iqk_common.h:539). MXFP4 passes its offset table, IQ4_NL the
/// signed kvalues, IQ4_XS the kvalues plus maddubs' offset.
///
/// # Safety
/// `qs` must point at 16 readable bytes; AVX2 must be present.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn lut4_codes(qs: *const u8, m4: __m256i, table: __m256i) -> __m256i {
    // SAFETY: 16 readable bytes per contract; the rest is register-only.
    unsafe {
        let aux128 = _mm_loadu_si128(qs as *const __m128i);
        let nib = _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16::<4>(aux128), aux128), m4);
        _mm256_shuffle_epi8(table, nib)
    }
}

/// AVX2+FMA row dot: MXFP4 weights against a Q8_2_X4 column.
///
/// TWIN of [`dot_q5f0_q82x4_avx2`] — ik runs both through `mul_mat_qX_1_q8_2_T`,
/// and they differ in four places only: block width (17 vs 22), the codes
/// (a table lookup of the nibbles vs the qh fold), the scale gather (four E8M0
/// halves, `ScaleHelperQ_0_1_MXFP4::prepare4`, vs four f16 through
/// `_mm_cvtph_ps`), and the min (-12 vs -16). Every edit outside those four must
/// be made in both. i16: codes 0..24 against activations in [-128, 127], a
/// maddubs pair peaks at 2·24·128 = 6144; saturation is unreachable. Also TWIN of
/// [`dot_mxfp4_q82x4_tile_avx2`], its column tile.
///
/// # Safety
/// Caller must ensure AVX2+FMA are available and buffers match `nb` blocks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_mxfp4_q82x4_avx2(wrow: &[u8], acol: &[u8], nb: usize) -> f32 {
    // SAFETY: AVX2+FMA present (checked by dot_row) and both slices hold the
    // validated lengths per contract.
    unsafe {
        let m4 = _mm256_set1_epi8(0xF);
        // SAFETY: 16 readable bytes of the static table.
        let t128 = _mm_loadu_si128(KVALUES_MXFP4_U.as_ptr() as *const __m128i);
        let table = _mm256_set_m128i(t128, t128);
        let m1 = _mm256_set1_epi16(1);
        let ones = _mm_set1_epi32(1);
        let min12 = _mm_set1_ps(-12.0f32);
        let mut acc = _mm256_setzero_ps();
        let mut accm = _mm_setzero_ps();

        let nbg = nb / 4;
        for i in 0..nbg {
            let b0 = wrow.as_ptr().add((4 * i) * MXFP4_BLOCK);
            let b1 = b0.add(MXFP4_BLOCK);
            let b2 = b1.add(MXFP4_BLOCK);
            let b3 = b2.add(MXFP4_BLOCK);
            let qx0 = lut4_codes(b0.add(1), m4, table);
            let qx1 = lut4_codes(b1.add(1), m4, table);
            let qx2 = lut4_codes(b2.add(1), m4, table);
            let qx3 = lut4_codes(b3.add(1), m4, table);
            // ScaleHelperQ_0_1_MXFP4::prepare4: (e - 1) << 23, with the two
            // subnormal halves for e = 0 and e = 1.
            let packed = u32::from(*b0)
                | (u32::from(*b1) << 8)
                | (u32::from(*b2) << 16)
                | (u32::from(*b3) << 24);
            let e32 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(packed as i32));
            let r = _mm_slli_epi32::<23>(_mm_sub_epi32(e32, ones));
            let r = _mm_blendv_epi8(
                r,
                _mm_set1_epi32(0x0020_0000),
                _mm_cmpeq_epi32(e32, _mm_setzero_si128()),
            );
            let r = _mm_blendv_epi8(r, _mm_set1_epi32(0x0040_0000), _mm_cmpeq_epi32(e32, ones));
            let s4 = _mm_castsi128_ps(r);
            let other = _mm256_set_m128(_mm_mul_ps(s4, min12), s4);

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
            let wb = wrow.as_ptr().add(i * MXFP4_BLOCK);
            let dw = e8m0_to_f32_half(*wb);
            let qx0 = lut4_codes(wb.add(1), m4, table);
            let tb = nb4 / 4 * Q82X4_STRIDE + (i - nb4) * Q82_BLOCK;
            let da = bf16_bits_to_f32(u16::from_le_bytes([acol[tb], acol[tb + 1]]));
            let ma = i16::from_le_bytes([acol[tb + 2], acol[tb + 3]]) as f32;
            let d = dw * da;
            let corr = (-12.0f32 * dw) * (da * ma) * 0.25f32;
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

/// The MXFP4 tile: one row's blocks against up to [`TILE_COLS`] Q8_2_X4
/// columns. Per x4 group the four blocks' codes and the E8M0 scale pair
/// (`s4`, `s4 · -12`) once; per column its bf16 scales and sums, its `accm`
/// add and its one FMA; per tail block the codes and `dw` once, then per
/// column the one-column tail step. Each column keeps its own `acc` and
/// `accm` and ends in the one-column kernel's reduction, so column `c` sees
/// the one-column kernel's float operations in its order.
///
/// TWIN of [`dot_mxfp4_q82x4_avx2`]: any edit to the float or integer steps
/// of one is made in both.
///
/// # Safety
/// CPU must support AVX2+FMA; `wrow` must hold `nb` MXFP4 blocks and each
/// `acols[c]` point at `col_bytes(MXFP4, 32 · nb)` readable bytes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_mxfp4_q82x4_tile_avx2<const C: usize>(
    wrow: &[u8],
    acols: &[*const u8; C],
    nb: usize,
) -> [f32; C] {
    const { assert!(C != 0 && C <= TILE_COLS) };
    // SAFETY: AVX2+FMA present, `wrow` holds nb blocks and every column the
    // Q8_2_X4 groups and q8_2 tails of nb blocks per contract.
    unsafe {
        let m4 = _mm256_set1_epi8(0xF);
        // SAFETY: 16 readable bytes of the static table.
        let t128 = _mm_loadu_si128(KVALUES_MXFP4_U.as_ptr() as *const __m128i);
        let table = _mm256_set_m128i(t128, t128);
        let m1 = _mm256_set1_epi16(1);
        let ones = _mm_set1_epi32(1);
        let min12 = _mm_set1_ps(-12.0f32);
        let mut acc = [_mm256_setzero_ps(); C];
        let mut accm = [_mm_setzero_ps(); C];

        let nbg = nb / 4;
        for i in 0..nbg {
            let b0 = wrow.as_ptr().add((4 * i) * MXFP4_BLOCK);
            let b1 = b0.add(MXFP4_BLOCK);
            let b2 = b1.add(MXFP4_BLOCK);
            let b3 = b2.add(MXFP4_BLOCK);
            // SAFETY: 16 readable bytes at offset 1 of each validated block.
            let qx0 = lut4_codes(b0.add(1), m4, table);
            let qx1 = lut4_codes(b1.add(1), m4, table);
            let qx2 = lut4_codes(b2.add(1), m4, table);
            let qx3 = lut4_codes(b3.add(1), m4, table);
            // ScaleHelperQ_0_1_MXFP4::prepare4: (e - 1) << 23, with the two
            // subnormal halves for e = 0 and e = 1.
            let packed = u32::from(*b0)
                | (u32::from(*b1) << 8)
                | (u32::from(*b2) << 16)
                | (u32::from(*b3) << 24);
            let e32 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(packed as i32));
            let r = _mm_slli_epi32::<23>(_mm_sub_epi32(e32, ones));
            let r = _mm_blendv_epi8(
                r,
                _mm_set1_epi32(0x0020_0000),
                _mm_cmpeq_epi32(e32, _mm_setzero_si128()),
            );
            let r = _mm_blendv_epi8(r, _mm_set1_epi32(0x0040_0000), _mm_cmpeq_epi32(e32, ones));
            let s4 = _mm_castsi128_ps(r);
            let other = _mm256_set_m128(_mm_mul_ps(s4, min12), s4);

            for ((col, a), am) in acols.iter().zip(acc.iter_mut()).zip(accm.iter_mut()) {
                // SAFETY: 8 readable bytes at the head of the column's group i.
                let g = col.add(i * Q82X4_STRIDE);
                let aux_d = _mm_castsi128_ps(_mm_slli_epi32::<16>(_mm_cvtepu16_epi32(
                    _mm_loadl_epi64(g as *const __m128i),
                )));
                // SAFETY: 8 readable bytes at offset 8 of the group.
                let aux_m = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_loadl_epi64(
                    g.add(8) as *const __m128i
                )));
                let prep = _mm256_set_m128(_mm_mul_ps(aux_d, aux_m), aux_d);
                let s12 = _mm256_mul_ps(other, prep);
                *am = _mm_add_ps(*am, _mm256_extractf128_ps(s12, 1));
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
                *a = _mm256_fmadd_ps(dall, _mm256_cvtepi32_ps(pall), *a);
            }
        }

        let nb4 = 4 * nbg;
        for i in nb4..nb {
            let wb = wrow.as_ptr().add(i * MXFP4_BLOCK);
            let dw = e8m0_to_f32_half(*wb);
            let qx0 = lut4_codes(wb.add(1), m4, table);
            let tb = nb4 / 4 * Q82X4_STRIDE + (i - nb4) * Q82_BLOCK;
            for ((col, a), am) in acols.iter().zip(acc.iter_mut()).zip(accm.iter_mut()) {
                // SAFETY: the tail block's 36 bytes are inside the column.
                let t = col.add(tb);
                let da = bf16_bits_to_f32(u16::from_le_bytes([*t, *t.add(1)]));
                let ma = i16::from_le_bytes([*t.add(2), *t.add(3)]) as f32;
                let d = dw * da;
                let corr = (-12.0f32 * dw) * (da * ma) * 0.25f32;
                *am = _mm_add_ps(*am, _mm_set1_ps(corr));
                let qs = _mm256_loadu_si256(t.add(4) as *const __m256i);
                let p0 = _mm256_madd_epi16(m1, _mm256_maddubs_epi16(qx0, qs));
                *a = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(p0), *a);
            }
        }

        // Reduction order is load-bearing: the one-column kernel's, per column.
        let mut out = [0.0f32; C];
        for ((o, a), am) in out.iter_mut().zip(&acc).zip(&accm) {
            let sum = _mm_add_ps(_mm256_castps256_ps128(*a), _mm256_extractf128_ps(*a, 1));
            let x = _mm_add_ps(sum, *am);
            let x = _mm_add_ps(x, _mm_movehl_ps(x, x));
            *o = _mm_cvtss_f32(_mm_add_ss(x, _mm_movehdup_ps(x)));
        }
        out
    }
}

/// Scalar decode of one 32-value 4-bit block's codes, `lut4_codes` lane for lane.
#[inline]
fn lut4_codes_scalar(qs: &[u8], table: &[u8; 16]) -> [u8; 32] {
    std::array::from_fn(|v| {
        let nib = if v < 16 { qs[v] & 0xF } else { qs[v - 16] >> 4 };
        table[usize::from(nib)]
    })
}

/// Emulates `dot_mxfp4_q82x4_avx2` with bit identity: `dot_q5f0_q82x4_emul`
/// with the four twin differences.
fn dot_mxfp4_q82x4_emul(wrow: &[u8], acol: &[u8], nb: usize) -> f32 {
    let mut acc = [0.0f32; 8];
    let mut accm = [0.0f32; 4];

    let nbg = nb / 4;
    for i in 0..nbg {
        let mut qx = [[0u8; 32]; 4];
        let mut dw = [0.0f32; 4];
        for j in 0..4 {
            let wb = &wrow[(4 * i + j) * MXFP4_BLOCK..(4 * i + j + 1) * MXFP4_BLOCK];
            qx[j] = lut4_codes_scalar(&wb[1..], &KVALUES_MXFP4_U);
            dw[j] = e8m0_to_f32_half(wb[0]);
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
            accm[l] += (dw[l] * -12.0f32) * (da[l] * ma[l]);
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
        let wb = &wrow[i * MXFP4_BLOCK..(i + 1) * MXFP4_BLOCK];
        let dw = e8m0_to_f32_half(wb[0]);
        let code = lut4_codes_scalar(&wb[1..], &KVALUES_MXFP4_U);
        let tb = nb4 / 4 * Q82X4_STRIDE + (i - nb4) * Q82_BLOCK;
        let blk = &acol[tb..tb + Q82_BLOCK];
        let da = bf16_bits_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let ma = i16::from_le_bytes([blk[2], blk[3]]) as f32;
        let d = dw * da;
        let corr = (-12.0f32 * dw) * (da * ma) * 0.25f32;
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

// ------------------------------------------------------------- NVFP4 x Q8_2_X4

/// The three byte layouts of an NVFP4 row, each 16-value group one UE4M3 scale and 8 code bytes.
/// `NV_BLOCKS` is mainline's `block_nvfp4` ([`GgmlType::NVFP4`], llama.cpp-mainline
/// ggml-common.h:221-227): four scales `d[4]` @0 then `qs[32]` @4, a 64-value block of four groups;
/// a group's value `j` (j < 8) is the low nibble of its code byte `j` and value `j + 8` the high
/// nibble (split order). `NV_PLANES` is a row held in place as two runs, its codes (8 bytes a
/// group) and its scales (one byte a group), and `NV_ROWS` the 9-byte groups of
/// [`GgmlType::NVFP4_ROWS`] (scale, then 8 codes); both are ModelOpt's pair order: value `2i` the
/// low nibble of code byte `i`, `2i + 1` the high nibble. One kernel reads all three, so a row
/// holds the same bits in each.
const NV_BLOCKS: u8 = 0;
const NV_PLANES: u8 = 1;
const NV_ROWS: u8 = 2;

/// `sizeof(block_nvfp4)`: 36 bytes / 64 values.
const NVFP4_BLOCK: usize = 36;

/// One 16-value group of the `NV_ROWS` layout: the scale byte @0, then 8 code bytes.
const NVFP4_GROUP: usize = 9;

/// 2^119: a UE4M3 byte placed at bit 20 of an f32 reads as the scale times 2^-120, and the doubled
/// code table needs the scale halved.
const NVFP4_TWO_119: f32 = f32::from_bits(246 << 23);

/// The largest UE4M3 byte that is a scale: a byte with bit 7 set, and 0x7F (E4M3's NaN), are none.
const UE4M3_MAX: u8 = 0x7E;

/// A UE4M3 scale byte's half value, `gguf::quant::ue4m3_to_f32_half` without its refusal. The byte
/// at bit 20 of an f32 is the scale times 2^-120 for exponent 0 (through the f32 subnormal) and
/// above alike, so the one multiply is exact for every byte up to [`UE4M3_MAX`]; a larger byte
/// gives a value nobody reads, because the caller poisons the row.
#[inline(always)]
fn nvfp4_scale_half(byte: u8) -> f32 {
    f32::from_bits(u32::from(byte) << 20) * NVFP4_TWO_119
}

/// The doubled E2M1 values as the table lookup's signed bytes.
#[cfg(target_arch = "x86_64")]
static KVALUES_NVFP4: [u8; 16] = {
    let mut t = [0u8; 16];
    let mut i = 0;
    while i < 16 {
        t[i] = KVALUES_MXFP4[i] as u8;
        i += 1;
    }
    t
};

/// The 16 code bytes of 32-value unit `u` (two groups) of a row in layout `L`.
///
/// # Safety
/// The unit's code bytes must be readable at their place in `w`.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn nvfp4_unit_codes<const L: u8>(w: *const u8, u: usize) -> __m128i {
    // SAFETY: per contract; `NV_ROWS` reads the two 8-byte code runs of the unit's groups.
    unsafe {
        match L {
            NV_PLANES => _mm_loadu_si128(w.add(16 * u) as *const __m128i),
            NV_BLOCKS => {
                _mm_loadu_si128(w.add(NVFP4_BLOCK * (u / 2) + 4 + 16 * (u % 2)) as *const __m128i)
            }
            _ => {
                let b = w.add(2 * NVFP4_GROUP * u);
                _mm_unpacklo_epi64(
                    _mm_loadl_epi64(b.add(1) as *const __m128i),
                    _mm_loadl_epi64(b.add(NVFP4_GROUP + 1) as *const __m128i),
                )
            }
        }
    }
}

/// The code bytes of units `u` and `u + 1` (`u` even), unit `u` in the low 128-bit lane.
///
/// # Safety
/// The two units' code bytes must be readable at their place in `w`.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn nvfp4_unit_pair<const L: u8>(w: *const u8, u: usize) -> __m256i {
    // SAFETY: per contract.
    unsafe {
        match L {
            NV_PLANES => _mm256_loadu_si256(w.add(16 * u) as *const __m256i),
            NV_BLOCKS => _mm256_loadu_si256(w.add(NVFP4_BLOCK * (u / 2) + 4) as *const __m256i),
            _ => _mm256_set_m128i(nvfp4_unit_codes::<L>(w, u + 1), nvfp4_unit_codes::<L>(w, u)),
        }
    }
}

/// The signed doubled E2M1 values of the units in `raw`'s two 128-bit lanes (16 code bytes each),
/// as two vectors: the lanes' values 0 to 15, then their values 16 to 31. Pair order puts value
/// `2i` in the low nibble and `2i + 1` in the high, so one byte unpack of the nibble vectors
/// restores value order; split order holds eight values in each nibble run of a group, so a
/// 64-bit unpack does.
///
/// # Safety
/// AVX2 must be present.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn nvfp4_values<const L: u8>(
    raw: __m256i,
    m4: __m256i,
    table: __m256i,
) -> (__m256i, __m256i) {
    // SAFETY: register-only.
    unsafe {
        let lo = _mm256_and_si256(raw, m4);
        let hi = _mm256_and_si256(_mm256_srli_epi16::<4>(raw), m4);
        let (a, b) = if L == NV_BLOCKS {
            (_mm256_unpacklo_epi64(lo, hi), _mm256_unpackhi_epi64(lo, hi))
        } else {
            (_mm256_unpacklo_epi8(lo, hi), _mm256_unpackhi_epi8(lo, hi))
        };
        (_mm256_shuffle_epi8(table, a), _mm256_shuffle_epi8(table, b))
    }
}

/// `madd(1, maddubs(|w|, sign(a, w)))`: the code's sign folded into the activation, so the weight
/// side is unsigned, and four adjacent products summed to each i32 lane. A maddubs pair peaks at
/// 2·12·128 = 3072; saturation is unreachable. `sign_epi8` keeps -128 wrapped to -128, and the
/// encoder never writes it.
///
/// # Safety
/// AVX2 must be present.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn nvfp4_lanes(w: __m256i, a: __m256i, m1: __m256i) -> __m256i {
    // SAFETY: register-only.
    unsafe { nvfp4_lanes_abs(_mm256_sign_epi8(w, w), w, a, m1) }
}

/// [`nvfp4_lanes`] with `|w|` (`sign_epi8(w, w)`) given, so a caller that dots one weight vector
/// against several columns takes it once.
///
/// # Safety
/// AVX2 must be present.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn nvfp4_lanes_abs(abs_w: __m256i, w: __m256i, a: __m256i, m1: __m256i) -> __m256i {
    // SAFETY: register-only.
    unsafe { _mm256_madd_epi16(m1, _mm256_maddubs_epi16(abs_w, _mm256_sign_epi8(a, w))) }
}

/// The four units' i32 lane sums of an x4 group (`pj` unit `j`'s, four lanes a 16-value group)
/// combined as the one-column kernel combines them: `[A0 A1 A2 A3 | B0 B1 B2 B3]`, `Aj` the sum of
/// unit `j`'s first 16 values and `Bj` of its second 16.
///
/// # Safety
/// AVX2 must be present.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn nvfp4_group_sums(p0: __m256i, p1: __m256i, p2: __m256i, p3: __m256i) -> __m256i {
    // SAFETY: register-only.
    unsafe {
        let p01 = _mm256_add_epi32(_mm256_unpacklo_epi32(p0, p1), _mm256_unpackhi_epi32(p0, p1));
        let p23 = _mm256_add_epi32(_mm256_unpacklo_epi32(p2, p3), _mm256_unpackhi_epi32(p2, p3));
        _mm256_add_epi32(
            _mm256_unpacklo_epi64(p01, p23),
            _mm256_unpackhi_epi64(p01, p23),
        )
    }
}

/// The offset that makes the doubled E2M1 values maddubs' unsigned side ([`KVALUES_MXFP4_U`]).
#[cfg(target_arch = "x86_64")]
const NVFP4_OFFSET: i8 = 12;

/// One column's share of the NVFP4 tile for one x4 group, built once for every row a run of rows
/// dots against the column ([`nvfp4_prep_avx2`]). `corr` is [`NVFP4_OFFSET`] times each 16-value
/// group's sum of activation codes in [`nvfp4_group_sums`]' lane order, and `d` the group's four
/// bf16 activation scales as f32 in both halves. The tile dots the offset codes `w + 12` with the
/// column's codes as they are and takes `corr` off: `Σ (w + 12)·a − 12·Σ a = Σ w·a`, the
/// one-column kernel's integer sum, so the float steps that follow are its own.
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct Nvfp4Prep {
    corr: [i32; 8],
    d: [f32; 8],
}

#[cfg(target_arch = "x86_64")]
impl Nvfp4Prep {
    const ZERO: Nvfp4Prep = Nvfp4Prep {
        corr: [0; 8],
        d: [0.0; 8],
    };
}

/// The x4 groups of one column the stack holds a prep for, a tile's [`TILE_COLS`] columns each:
/// `k` up to 8,192. A longer row's prep is a buffer of the call.
#[cfg(target_arch = "x86_64")]
const NVFP4_PREP_GROUPS: usize = 64;

/// The rows a planes call dots against one run of columns before the next run when its columns
/// take more than one run, so a block's weight bytes are read from DRAM once and from the core's
/// cache for every later run.
#[cfg(target_arch = "x86_64")]
const NVFP4_ROW_BLOCK: usize = 64;

/// The tile's [`Nvfp4Prep`] of the `out.len()` x4 groups of the column at `acol` (`nb` units), and
/// whether any code of the column is -128: no encoder writes it, the one-column kernel's sign fold
/// wraps it, and the tile's offset dot would read it as itself, so the caller refuses the column.
///
/// # Safety
/// AVX2+FMA must be present and `acol` hold the Q8_2_X4 groups and q8_2 tails of `nb` units, with
/// `out.len() == nb / 4`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn nvfp4_prep_avx2(acol: *const u8, nb: usize, out: &mut [Nvfp4Prep]) -> bool {
    // SAFETY: AVX2+FMA present; every read is inside the column's groups and tails per contract,
    // and every store inside an aligned `Nvfp4Prep` of `out`.
    unsafe {
        let off = _mm256_set1_epi8(NVFP4_OFFSET);
        let m1 = _mm256_set1_epi16(1);
        let min = _mm256_set1_epi8(i8::MIN);
        let mut bad = _mm256_setzero_si256();
        for (i, p) in out.iter_mut().enumerate() {
            let g = acol.add(i * Q82X4_STRIDE);
            let aux_d = _mm_castsi128_ps(_mm_slli_epi32::<16>(_mm_cvtepu16_epi32(
                _mm_loadl_epi64(g as *const __m128i),
            )));
            _mm256_store_ps(p.d.as_mut_ptr(), _mm256_set_m128(aux_d, aux_d));
            let a0 = _mm256_loadu_si256(g.add(16) as *const __m256i);
            let a1 = _mm256_loadu_si256(g.add(48) as *const __m256i);
            let a2 = _mm256_loadu_si256(g.add(80) as *const __m256i);
            let a3 = _mm256_loadu_si256(g.add(112) as *const __m256i);
            bad = _mm256_or_si256(
                bad,
                _mm256_or_si256(
                    _mm256_or_si256(_mm256_cmpeq_epi8(a0, min), _mm256_cmpeq_epi8(a1, min)),
                    _mm256_or_si256(_mm256_cmpeq_epi8(a2, min), _mm256_cmpeq_epi8(a3, min)),
                ),
            );
            let corr = nvfp4_group_sums(
                _mm256_madd_epi16(m1, _mm256_maddubs_epi16(off, a0)),
                _mm256_madd_epi16(m1, _mm256_maddubs_epi16(off, a1)),
                _mm256_madd_epi16(m1, _mm256_maddubs_epi16(off, a2)),
                _mm256_madd_epi16(m1, _mm256_maddubs_epi16(off, a3)),
            );
            _mm256_store_si256(p.corr.as_mut_ptr() as *mut __m256i, corr);
        }
        let tails = acol.add(out.len() * Q82X4_STRIDE);
        for t in 0..nb - 4 * out.len() {
            let qs = _mm256_loadu_si256(tails.add(t * Q82_BLOCK + 4) as *const __m256i);
            bad = _mm256_or_si256(bad, _mm256_cmpeq_epi8(qs, min));
        }
        _mm256_testz_si256(bad, bad) == 0
    }
}

/// The eight scale bytes of the 4-unit group `g` in memory order: `NV_PLANES` reads the scale
/// plane `sc`, `NV_BLOCKS` the `d[4]` of the group's two blocks, `NV_ROWS` every ninth byte.
///
/// # Safety
/// The group's eight scales must be readable at their place.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn nvfp4_group_scales<const L: u8>(w: *const u8, sc: *const u8, g: usize) -> u64 {
    // SAFETY: per contract.
    unsafe {
        match L {
            NV_PLANES => (sc.add(8 * g) as *const u64).read_unaligned(),
            NV_BLOCKS => {
                let b = w.add(2 * NVFP4_BLOCK * g);
                u64::from((b as *const u32).read_unaligned())
                    | u64::from((b.add(NVFP4_BLOCK) as *const u32).read_unaligned()) << 32
            }
            _ => {
                let b = w.add(8 * NVFP4_GROUP * g);
                let mut v = 0u64;
                for t in 0..8 {
                    v |= u64::from(*b.add(NVFP4_GROUP * t)) << (8 * t);
                }
                v
            }
        }
    }
}

/// Scale byte `t` of a row (one a 16-value group).
///
/// # Safety
/// Scale `t` must be readable at its place.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn nvfp4_scale_byte<const L: u8>(w: *const u8, sc: *const u8, t: usize) -> u8 {
    // SAFETY: per contract.
    unsafe {
        match L {
            NV_PLANES => *sc.add(t),
            NV_BLOCKS => *w.add(NVFP4_BLOCK * (t / 4) + t % 4),
            _ => *w.add(NVFP4_GROUP * t),
        }
    }
}

/// AVX2+FMA row dot: NVFP4 weights against a Q8_2_X4 column, the tensor scale left out.
///
/// `w` is the row in layout `L` (the code plane for `NV_PLANES`), `sc` the scale plane (read for
/// `NV_PLANES` alone). All three walk the same 32-value units and the same float order, so a row's
/// forms give the same bits.
///
/// Per 4-unit group: two ymm loads of codes, unpacked to value order and looked up in the signed
/// table; per unit the IQ4_NL kernel's sign-folded dot; the i32 lane sums combined as IQ4_NL
/// combines them, which leaves `[A0 A1 A2 A3 | B0 B1 B2 B3]`, `Aj` the first 16 values of unit
/// `j` and `Bj` the second 16. The scale vector is the eight UE4M3 bytes shuffled to
/// `[s0 s2 s4 s6 | s1 s3 s5 s7]`, widened, re-biased exactly ([`nvfp4_scale_half`]) and times
/// the four bf16 activation scales in both halves: each product is a 4-bit by an 8-bit
/// significand, exact. One fmadd a group. A tail unit (`nb % 4` of them) takes its two scales
/// as scalars and one fmadd. The sums are integers up to 4·12·128, exact, so the f32 order is the
/// kernel's alone and the reduction order is load-bearing.
///
/// A scale byte that is no UE4M3 scale ([`UE4M3_MAX`]) makes the whole row NaN. This is a twin of
/// [`dot_nvfp4_q82x4_emul`], its scalar mirror, and of [`dot_nvfp4_q82x4_tile_avx2`], its column
/// tile: any edit to a float or integer step of one is made in all three.
///
/// # Safety
/// Caller must ensure AVX2+FMA are available and the row (`nb` units in layout `L`) and the column
/// (`col_bytes(NVFP4, 32 · nb)` bytes) are readable.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_nvfp4_q82x4_avx2<const L: u8>(
    w: *const u8,
    sc: *const u8,
    acol: &[u8],
    nb: usize,
) -> f32 {
    // SAFETY: AVX2+FMA present per contract; the row and the column hold nb units.
    unsafe {
        let m4 = _mm256_set1_epi8(0xF);
        // SAFETY: 16 readable bytes of the static table.
        let t128 = _mm_loadu_si128(KVALUES_NVFP4.as_ptr() as *const __m128i);
        let table = _mm256_set_m128i(t128, t128);
        let m1 = _mm256_set1_epi16(1);
        let two119 = _mm256_set1_ps(NVFP4_TWO_119);
        let split = _mm_setr_epi8(0, 2, 4, 6, 1, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1);
        let mut acc = _mm256_setzero_ps();
        let mut top = _mm_setzero_si128();

        let nbg = nb / 4;
        for i in 0..nbg {
            let (a01, b01) = nvfp4_values::<L>(nvfp4_unit_pair::<L>(w, 4 * i), m4, table);
            let (a23, b23) = nvfp4_values::<L>(nvfp4_unit_pair::<L>(w, 4 * i + 2), m4, table);
            let q0 = _mm256_permute2x128_si256::<0x20>(a01, b01);
            let q1 = _mm256_permute2x128_si256::<0x31>(a01, b01);
            let q2 = _mm256_permute2x128_si256::<0x20>(a23, b23);
            let q3 = _mm256_permute2x128_si256::<0x31>(a23, b23);

            let sb = _mm_cvtsi64_si128(nvfp4_group_scales::<L>(w, sc, i) as i64);
            top = _mm_max_epu8(top, sb);
            let half = _mm256_mul_ps(
                _mm256_castsi256_ps(_mm256_slli_epi32::<20>(_mm256_cvtepu8_epi32(
                    _mm_shuffle_epi8(sb, split),
                ))),
                two119,
            );
            // SAFETY: 8 readable bytes at the head of the validated group.
            let g = acol.as_ptr().add(i * Q82X4_STRIDE);
            let aux_d = _mm_castsi128_ps(_mm_slli_epi32::<16>(_mm_cvtepu16_epi32(
                _mm_loadl_epi64(g as *const __m128i),
            )));
            let s = _mm256_mul_ps(half, _mm256_set_m128(aux_d, aux_d));

            // SAFETY: 32 readable bytes at each offset inside the group.
            let p0 = nvfp4_lanes(q0, _mm256_loadu_si256(g.add(16) as *const __m256i), m1);
            let p1 = nvfp4_lanes(q1, _mm256_loadu_si256(g.add(48) as *const __m256i), m1);
            let p2 = nvfp4_lanes(q2, _mm256_loadu_si256(g.add(80) as *const __m256i), m1);
            let p3 = nvfp4_lanes(q3, _mm256_loadu_si256(g.add(112) as *const __m256i), m1);
            let p01 =
                _mm256_add_epi32(_mm256_unpacklo_epi32(p0, p1), _mm256_unpackhi_epi32(p0, p1));
            let p23 =
                _mm256_add_epi32(_mm256_unpacklo_epi32(p2, p3), _mm256_unpackhi_epi32(p2, p3));
            let pall = _mm256_add_epi32(
                _mm256_unpacklo_epi64(p01, p23),
                _mm256_unpackhi_epi64(p01, p23),
            );
            acc = _mm256_fmadd_ps(s, _mm256_cvtepi32_ps(pall), acc);
        }

        let nb4 = 4 * nbg;
        let mut tail_top = 0u8;
        for i in nb4..nb {
            let (s0, s1) = (
                nvfp4_scale_byte::<L>(w, sc, 2 * i),
                nvfp4_scale_byte::<L>(w, sc, 2 * i + 1),
            );
            tail_top = tail_top.max(s0).max(s1);
            let tb = nb4 / 4 * Q82X4_STRIDE + (i - nb4) * Q82_BLOCK;
            let da = bf16_bits_to_f32(u16::from_le_bytes([acol[tb], acol[tb + 1]]));
            let s = _mm256_set_m128(
                _mm_set1_ps(nvfp4_scale_half(s1) * da),
                _mm_set1_ps(nvfp4_scale_half(s0) * da),
            );
            let (a, b) = nvfp4_values::<L>(
                _mm256_set_m128i(_mm_setzero_si128(), nvfp4_unit_codes::<L>(w, i)),
                m4,
                table,
            );
            let q = _mm256_permute2x128_si256::<0x20>(a, b);
            // SAFETY: 32 readable bytes at offset 4 of the tail block.
            let qs = _mm256_loadu_si256(acol.as_ptr().add(tb + 4) as *const __m256i);
            acc = _mm256_fmadd_ps(s, _mm256_cvtepi32_ps(nvfp4_lanes(q, qs, m1)), acc);
        }

        let max = _mm_set1_epi8(UE4M3_MAX as i8);
        let scales_ok = _mm_movemask_epi8(_mm_cmpeq_epi8(_mm_max_epu8(top, max), max)) == 0xFFFF;
        if !scales_ok || tail_top > UE4M3_MAX {
            return f32::NAN;
        }
        hsum_float_8(acc)
    }
}

/// The NVFP4 tile: one row in layout `L` against up to [`TILE_COLS`] Q8_2_X4 columns, each with
/// its [`Nvfp4Prep`] of every x4 group. Per x4 group the codes are unpacked and looked up as
/// [`NVFP4_OFFSET`] plus their value, never negative, and the scale vector built once; per column
/// maddubs takes the column's codes as they are, the lane sums lose the prep's `corr`, and the
/// prep's `d` and one fmadd follow. Per tail unit the codes, `|w|` and the two weight scales once,
/// then per column the one-column kernel's sign-folded tail step. Each column's integer sums are
/// the one-column kernel's and it keeps its own `acc`, ending in that kernel's reduction, so column
/// `j` sees the one-column kernel's float operations in its order. A scale byte that is no UE4M3
/// scale makes every column NaN, as it makes the one column.
///
/// TWIN of [`dot_nvfp4_q82x4_avx2`]: any edit to a float or integer step of one is made in both.
///
/// # Safety
/// Caller must ensure AVX2+FMA are available, the row (`nb` units in layout `L`, `sc` its scale
/// plane for `NV_PLANES`) is readable, each `acols[j]` points at `col_bytes(NVFP4, 32 · nb)`
/// readable bytes and each `preps[j]` at the `nb / 4` preps [`nvfp4_prep_avx2`] built of that
/// column, none of whose codes is -128.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_nvfp4_q82x4_tile_avx2<const L: u8, const C: usize>(
    w: *const u8,
    sc: *const u8,
    acols: &[*const u8; C],
    preps: &[*const Nvfp4Prep; C],
    nb: usize,
) -> [f32; C] {
    const { assert!(C != 0 && C <= TILE_COLS) };
    // SAFETY: AVX2+FMA present per contract; the row, every column and every prep hold nb units.
    unsafe {
        let m4 = _mm256_set1_epi8(0xF);
        // SAFETY: 16 readable bytes of each static table.
        let u128 = _mm_loadu_si128(KVALUES_MXFP4_U.as_ptr() as *const __m128i);
        let table_u = _mm256_set_m128i(u128, u128);
        let m1 = _mm256_set1_epi16(1);
        let two119 = _mm256_set1_ps(NVFP4_TWO_119);
        let split = _mm_setr_epi8(0, 2, 4, 6, 1, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1);
        let mut acc = [_mm256_setzero_ps(); C];
        let mut top = _mm_setzero_si128();

        let nbg = nb / 4;
        for i in 0..nbg {
            let (a01, b01) = nvfp4_values::<L>(nvfp4_unit_pair::<L>(w, 4 * i), m4, table_u);
            let (a23, b23) = nvfp4_values::<L>(nvfp4_unit_pair::<L>(w, 4 * i + 2), m4, table_u);
            let u0 = _mm256_permute2x128_si256::<0x20>(a01, b01);
            let u1 = _mm256_permute2x128_si256::<0x31>(a01, b01);
            let u2 = _mm256_permute2x128_si256::<0x20>(a23, b23);
            let u3 = _mm256_permute2x128_si256::<0x31>(a23, b23);

            let sb = _mm_cvtsi64_si128(nvfp4_group_scales::<L>(w, sc, i) as i64);
            top = _mm_max_epu8(top, sb);
            let half = _mm256_mul_ps(
                _mm256_castsi256_ps(_mm256_slli_epi32::<20>(_mm256_cvtepu8_epi32(
                    _mm_shuffle_epi8(sb, split),
                ))),
                two119,
            );

            for ((col, prep), a) in acols.iter().zip(preps).zip(acc.iter_mut()) {
                // SAFETY: the column's group i and its prep i are readable per contract.
                let g = col.add(i * Q82X4_STRIDE);
                let p = &*prep.add(i);
                let s = _mm256_mul_ps(half, _mm256_load_ps(p.d.as_ptr()));
                // SAFETY: 32 readable bytes at each offset inside the group.
                let p0 = _mm256_madd_epi16(
                    m1,
                    _mm256_maddubs_epi16(u0, _mm256_loadu_si256(g.add(16) as *const __m256i)),
                );
                let p1 = _mm256_madd_epi16(
                    m1,
                    _mm256_maddubs_epi16(u1, _mm256_loadu_si256(g.add(48) as *const __m256i)),
                );
                let p2 = _mm256_madd_epi16(
                    m1,
                    _mm256_maddubs_epi16(u2, _mm256_loadu_si256(g.add(80) as *const __m256i)),
                );
                let p3 = _mm256_madd_epi16(
                    m1,
                    _mm256_maddubs_epi16(u3, _mm256_loadu_si256(g.add(112) as *const __m256i)),
                );
                let pall = _mm256_sub_epi32(
                    nvfp4_group_sums(p0, p1, p2, p3),
                    _mm256_load_si256(p.corr.as_ptr() as *const __m256i),
                );
                *a = _mm256_fmadd_ps(s, _mm256_cvtepi32_ps(pall), *a);
            }
        }

        let nb4 = 4 * nbg;
        let mut tail_top = 0u8;
        if nb4 < nb {
            // SAFETY: 16 readable bytes of the static table.
            let t128 = _mm_loadu_si128(KVALUES_NVFP4.as_ptr() as *const __m128i);
            let table = _mm256_set_m128i(t128, t128);
            for i in nb4..nb {
                let (s0, s1) = (
                    nvfp4_scale_byte::<L>(w, sc, 2 * i),
                    nvfp4_scale_byte::<L>(w, sc, 2 * i + 1),
                );
                tail_top = tail_top.max(s0).max(s1);
                let (h0, h1) = (nvfp4_scale_half(s0), nvfp4_scale_half(s1));
                let tb = nb4 / 4 * Q82X4_STRIDE + (i - nb4) * Q82_BLOCK;
                let (lo, hi) = nvfp4_values::<L>(
                    _mm256_set_m128i(_mm_setzero_si128(), nvfp4_unit_codes::<L>(w, i)),
                    m4,
                    table,
                );
                let q = _mm256_permute2x128_si256::<0x20>(lo, hi);
                let n = _mm256_sign_epi8(q, q);
                for (col, a) in acols.iter().zip(acc.iter_mut()) {
                    // SAFETY: the tail block's 36 bytes are inside the column.
                    let t = col.add(tb);
                    let da = bf16_bits_to_f32(u16::from_le_bytes([*t, *t.add(1)]));
                    let s = _mm256_set_m128(_mm_set1_ps(h1 * da), _mm_set1_ps(h0 * da));
                    // SAFETY: 32 readable bytes at offset 4 of the tail block.
                    let qs = _mm256_loadu_si256(t.add(4) as *const __m256i);
                    *a = _mm256_fmadd_ps(s, _mm256_cvtepi32_ps(nvfp4_lanes_abs(n, q, qs, m1)), *a);
                }
            }
        }

        let max = _mm_set1_epi8(UE4M3_MAX as i8);
        let scales_ok = _mm_movemask_epi8(_mm_cmpeq_epi8(_mm_max_epu8(top, max), max)) == 0xFFFF;
        if !scales_ok || tail_top > UE4M3_MAX {
            return [f32::NAN; C];
        }
        // Reduction order is load-bearing: the one-column kernel's, per column.
        let mut out = [0.0f32; C];
        for (o, a) in out.iter_mut().zip(&acc) {
            *o = hsum_float_8(*a);
        }
        out
    }
}

/// Unit `u`'s 32 signed doubled E2M1 values in value order, as `nvfp4_values` builds them.
fn nvfp4_values_scalar<const L: u8>(w: &[u8], u: usize) -> [u8; 32] {
    std::array::from_fn(|v| {
        // The unit's sub-block (a 16-value group) and the value in it.
        let (group, j) = (v / 16, v % 16);
        let nib = if L == NV_BLOCKS {
            let byte = w[NVFP4_BLOCK * (u / 2) + 4 + 8 * (2 * (u % 2) + group) + j % 8];
            if j < 8 { byte & 0xF } else { byte >> 4 }
        } else {
            let byte = if L == NV_PLANES {
                w[16 * u + 8 * group + j / 2]
            } else {
                w[NVFP4_GROUP * (2 * u + group) + 1 + j / 2]
            };
            if j % 2 == 0 { byte & 0xF } else { byte >> 4 }
        };
        KVALUES_MXFP4[usize::from(nib)] as u8
    })
}

/// `nvfp4_lanes` on arrays: lane `l` of the result sums values `4l..4l + 4` of the unit.
fn nvfp4_lanes_scalar(w: &[u8; 32], a: &[u8; 32]) -> [i32; 8] {
    let abs_w: [u8; 32] = std::array::from_fn(|n| (w[n] as i8).wrapping_abs() as u8);
    let sgn_a: [u8; 32] = std::array::from_fn(|n| emul_sign_epi8(a[n], w[n] as i8));
    emul_madd1(emul_maddubs(abs_w, sgn_a))
}

/// Scale byte `t` of a row, as `nvfp4_scale_byte` reads it.
fn nvfp4_scale_byte_scalar<const L: u8>(w: &[u8], sc: &[u8], t: usize) -> u8 {
    match L {
        NV_PLANES => sc[t],
        NV_BLOCKS => w[NVFP4_BLOCK * (t / 4) + t % 4],
        _ => w[NVFP4_GROUP * t],
    }
}

/// Emulates `dot_nvfp4_q82x4_avx2` with bit identity: the integer sums are exact either way, the
/// scale vector, the group and tail fmadds and the final reduction are the kernel's.
fn dot_nvfp4_q82x4_emul<const L: u8>(w: &[u8], sc: &[u8], acol: &[u8], nb: usize) -> f32 {
    let mut acc = [0.0f32; 8];
    let mut top = 0u8;

    let nbg = nb / 4;
    for i in 0..nbg {
        let g = &acol[i * Q82X4_STRIDE..(i + 1) * Q82X4_STRIDE];
        let mut s = [0.0f32; 8];
        for (l, sl) in s.iter_mut().enumerate() {
            // Lane l is unit l % 4's first (l < 4) or second sub-block.
            let byte = nvfp4_scale_byte_scalar::<L>(w, sc, 8 * i + 2 * (l % 4) + l / 4);
            top = top.max(byte);
            let da = bf16_bits_to_f32(u16::from_le_bytes([g[2 * (l % 4)], g[2 * (l % 4) + 1]]));
            *sl = nvfp4_scale_half(byte) * da;
        }
        let mut p = [[0i32; 8]; 4];
        for (j, pj) in p.iter_mut().enumerate() {
            let mut qs = [0u8; 32];
            qs.copy_from_slice(&g[16 + 32 * j..16 + 32 * j + 32]);
            *pj = nvfp4_lanes_scalar(&nvfp4_values_scalar::<L>(w, 4 * i + j), &qs);
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
            acc[l] = s[l].mul_add(pall[l] as f32, acc[l]);
        }
    }

    let nb4 = 4 * nbg;
    for i in nb4..nb {
        let (s0, s1) = (
            nvfp4_scale_byte_scalar::<L>(w, sc, 2 * i),
            nvfp4_scale_byte_scalar::<L>(w, sc, 2 * i + 1),
        );
        top = top.max(s0).max(s1);
        let tb = nb4 / 4 * Q82X4_STRIDE + (i - nb4) * Q82_BLOCK;
        let blk = &acol[tb..tb + Q82_BLOCK];
        let da = bf16_bits_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let s = [nvfp4_scale_half(s0) * da, nvfp4_scale_half(s1) * da];
        let mut qs = [0u8; 32];
        qs.copy_from_slice(&blk[4..36]);
        let p0 = nvfp4_lanes_scalar(&nvfp4_values_scalar::<L>(w, i), &qs);
        for l in 0..8 {
            acc[l] = s[l / 4].mul_add(p0[l] as f32, acc[l]);
        }
    }

    if top > UE4M3_MAX {
        return f32::NAN;
    }
    let t = [
        acc[0] + acc[4],
        acc[1] + acc[5],
        acc[2] + acc[6],
        acc[3] + acc[7],
    ];
    (t[0] + t[2]) + (t[1] + t[3])
}

/// The shape contract of the plane entries: `k` whole 32-value blocks, a code plane of `k / 2`
/// bytes, a scale plane of `k / 16` and a column of `col_bytes(NVFP4_ROWS, k)` at least. A caller
/// that breaks it has a layout bug, so each refusal is a named panic. Returns the unit count.
fn check_planes(codes: usize, scales: usize, acol: usize, k: usize) -> usize {
    const FN: &str = "dot_nvfp4_planes_row";
    assert!(
        k.is_multiple_of(32),
        "{FN}: k = {k} is not a whole number of 32-value blocks"
    );
    assert!(
        codes >= k / 2,
        "{FN}: the code plane holds {codes} bytes, k = {k} needs {}",
        k / 2
    );
    assert!(
        scales >= k / 16,
        "{FN}: the scale plane holds {scales} bytes, k = {k} needs {}",
        k / 16
    );
    let need = col_bytes(GgmlType::NVFP4_ROWS, k);
    assert!(
        acol >= need,
        "{FN}: the activation column holds {acol} bytes, k = {k} needs {need}"
    );
    k / 32
}

/// One NVFP4 row, held in place as two planes, against a quantized activation column: `codes`
/// the row's `k / 2` code bytes (ModelOpt's pair order: value `2i` the low nibble of byte `i`,
/// `2i + 1` the high), `scales` its `k / 16` UE4M3 bytes (one a 16-value group), `acol` a
/// [`quantize_col`] column of [`GgmlType::NVFP4_ROWS`] (Q8_2_X4, MXFP4's). Each value is `e2m1 ·
/// scale`; the result is the dot without the tensor scale, which is the caller's one multiply
/// (`weight_global_scale` is a divisor, so the caller multiplies by its reciprocal).
///
/// AVX2 where the CPU has AVX2 and FMA, else its bit-identical scalar mirror
/// ([`dot_nvfp4_planes_row_scalar`]). A scale byte that is no UE4M3 scale (bit 7 set, or 0x7F)
/// makes the result NaN. A plane shorter than `k` implies, a `k` that is not a multiple of 32
/// or a short column is a named panic: each is a layout bug at the caller.
#[must_use]
pub fn dot_nvfp4_planes_row(codes: &[u8], scales: &[u8], acol: &[u8], k: usize) -> f32 {
    let nb = check_planes(codes.len(), scales.len(), acol.len(), k);
    #[cfg(target_arch = "x86_64")]
    if has_features(GgmlType::NVFP4_ROWS) {
        // SAFETY: AVX2+FMA were just detected; check_planes sized both planes and the column.
        return unsafe {
            dot_nvfp4_q82x4_avx2::<NV_PLANES>(codes.as_ptr(), scales.as_ptr(), acol, nb)
        };
    }
    dot_nvfp4_q82x4_emul::<NV_PLANES>(codes, scales, acol, nb)
}

/// The scalar mirror behind [`dot_nvfp4_planes_row`]: the fallback without AVX2 and the oracle
/// its bit-identity gate compares against.
#[must_use]
pub fn dot_nvfp4_planes_row_scalar(codes: &[u8], scales: &[u8], acol: &[u8], k: usize) -> f32 {
    let nb = check_planes(codes.len(), scales.len(), acol.len(), k);
    dot_nvfp4_q82x4_emul::<NV_PLANES>(codes, scales, acol, nb)
}

/// The AVX2 kernel of [`dot_nvfp4_planes_row`], called directly. Panics without AVX2+FMA.
#[cfg(target_arch = "x86_64")]
#[must_use]
pub fn dot_nvfp4_planes_row_avx2(codes: &[u8], scales: &[u8], acol: &[u8], k: usize) -> f32 {
    let nb = check_planes(codes.len(), scales.len(), acol.len(), k);
    assert!(
        has_features(GgmlType::NVFP4_ROWS),
        "dot_nvfp4_planes_row_avx2 called on a CPU without avx2+fma"
    );
    // SAFETY: asserted just above; check_planes sized both planes and the column.
    unsafe { dot_nvfp4_q82x4_avx2::<NV_PLANES>(codes.as_ptr(), scales.as_ptr(), acol, nb) }
}

/// The shape contract of [`dot_nvfp4_planes_row_cols`]: one to [`TILE_COLS`] columns, one output
/// each, and [`check_planes`]' contract for every column. A caller that breaks it has a layout bug,
/// so each refusal is a named panic.
fn check_planes_cols(codes: usize, scales: usize, acols: &[&[u8]], k: usize, outs: usize) {
    const FN: &str = "dot_nvfp4_planes_row_cols";
    let c = acols.len();
    assert!(
        (1..=TILE_COLS).contains(&c) && outs == c,
        "{FN}: {c} columns and {outs} outputs, a tile takes 1..={TILE_COLS} columns and one output each"
    );
    for a in acols {
        check_planes(codes, scales, a.len(), k);
    }
}

/// [`dot_nvfp4_q82x4_tile_avx2`] at a run of `m` columns, 2 to [`TILE_COLS`]: the run's first `m`
/// slots of `cols` and `preps`, its values in the first `m` slots of the result.
///
/// # Safety
/// AVX2+FMA must be present and the tile's contract hold for the row at `w`/`sc` and the run's
/// columns and preps.
#[cfg(target_arch = "x86_64")]
unsafe fn nvfp4_tile_run(
    m: usize,
    (w, sc): (*const u8, *const u8),
    cols: &[*const u8; TILE_COLS],
    preps: &[*const Nvfp4Prep; TILE_COLS],
    nb: usize,
) -> [f32; TILE_COLS] {
    /// # Safety
    /// As [`nvfp4_tile_run`], with `m == C`.
    unsafe fn run<const C: usize>(
        (w, sc): (*const u8, *const u8),
        cols: &[*const u8; TILE_COLS],
        preps: &[*const Nvfp4Prep; TILE_COLS],
        nb: usize,
    ) -> [f32; TILE_COLS] {
        let c: [*const u8; C] = std::array::from_fn(|j| cols[j]);
        let p: [*const Nvfp4Prep; C] = std::array::from_fn(|j| preps[j]);
        // SAFETY: per contract.
        let v = unsafe { dot_nvfp4_q82x4_tile_avx2::<NV_PLANES, C>(w, sc, &c, &p, nb) };
        let mut out = [0.0f32; TILE_COLS];
        out[..C].copy_from_slice(&v);
        out
    }
    // SAFETY: per contract; each arm's `C` is the run's column count.
    unsafe {
        match m {
            2 => run::<2>((w, sc), cols, preps, nb),
            3 => run::<3>((w, sc), cols, preps, nb),
            4 => run::<4>((w, sc), cols, preps, nb),
            5 => run::<5>((w, sc), cols, preps, nb),
            6 => run::<6>((w, sc), cols, preps, nb),
            7 => run::<7>((w, sc), cols, preps, nb),
            8 => run::<8>((w, sc), cols, preps, nb),
            c => panic!("dot_nvfp4_planes_rows_cols: no tile of {c} columns"),
        }
    }
}

/// The planes tile over `rows` against `ncols` columns, two or more: the columns in balanced runs
/// of at most [`TILE_COLS`]. A run's preps are built once for the call when one run holds every
/// column, else once for each [`NVFP4_ROW_BLOCK`] rows, so a block's weight rows are read from
/// DRAM once whatever the run count. A column holding the code -128 is refused by name.
///
/// # Safety
/// AVX2+FMA must be present and [`dot_nvfp4_planes_rows_cols`]' shape checks have passed for these
/// arguments.
#[cfg(target_arch = "x86_64")]
unsafe fn nvfp4_planes_tiled<'a>(
    (codes, scales): (&[u8], &[u8]),
    rows: Range<usize>,
    k: usize,
    (ncols, col): (usize, &impl Fn(usize) -> &'a [u8]),
    emit: &mut impl FnMut(usize, usize, f32),
) {
    let (nb, code_row, scale_row) = (k / 32, k / 2, k / 16);
    let nbg = nb / 4;
    let mut stack = [Nvfp4Prep::ZERO; TILE_COLS * NVFP4_PREP_GROUPS];
    let mut heap = Vec::new();
    let prep: &mut [Nvfp4Prep] = if nbg <= NVFP4_PREP_GROUPS {
        &mut stack[..TILE_COLS * nbg]
    } else {
        heap.resize(TILE_COLS * nbg, Nvfp4Prep::ZERO);
        &mut heap
    };
    let runs = ncols.div_ceil(TILE_COLS);
    let (base, extra) = (ncols / runs, ncols % runs);
    let block = if runs == 1 {
        rows.len()
    } else {
        NVFP4_ROW_BLOCK
    };
    let mut r0 = rows.start;
    while r0 < rows.end {
        let r1 = (r0 + block).min(rows.end);
        let mut t0 = 0;
        for run in 0..runs {
            let m = base + usize::from(run < extra);
            let cols: [*const u8; TILE_COLS] =
                std::array::from_fn(|j| col(t0 + j.min(m - 1)).as_ptr());
            if runs > 1 || r0 == rows.start {
                for (j, &c) in cols[..m].iter().enumerate() {
                    // SAFETY: AVX2+FMA present; the column holds nb units (checked by the entry)
                    // and its prep slice is nb / 4 cells.
                    if unsafe { nvfp4_prep_avx2(c, nb, &mut prep[j * nbg..(j + 1) * nbg]) } {
                        panic!(
                            "dot_nvfp4_planes_rows_cols: column {} holds the code -128, which no \
                             encoder writes",
                            t0 + j
                        );
                    }
                }
            }
            let preps: [*const Nvfp4Prep; TILE_COLS] =
                std::array::from_fn(|j| prep[j.min(m - 1) * nbg..].as_ptr());
            for r in r0..r1 {
                let row = (
                    codes[r * code_row..].as_ptr(),
                    scales[r * scale_row..].as_ptr(),
                );
                // SAFETY: AVX2+FMA present; row r lies inside both planes, every column of the run
                // holds nb units and its preps were built of it just above or for an earlier block.
                let v = unsafe { nvfp4_tile_run(m, row, &cols, &preps, nb) };
                for (j, &v) in v[..m].iter().enumerate() {
                    emit(r, t0 + j, v);
                }
            }
            t0 += m;
        }
        r0 = r1;
    }
}

/// Rows `rows` of an NVFP4 matrix held as two planes (`codes`, `k / 2` bytes a row; `scales`,
/// `k / 16`) against `ncols` quantized activation columns, `col(j)` column `j`: `emit(r, j, v)`
/// with `v` the value [`dot_nvfp4_planes_row`] gives row `r` and column `j`, bit for bit.
///
/// With two or more columns and AVX2 and FMA, the columns go in balanced runs of up to
/// [`TILE_COLS`] through the planes tile (`dot_nvfp4_q82x4_tile_avx2`): each x4 group of a row is
/// unpacked once a run, and each column's activation side of the dot (its code sums and scales) is
/// built once for the rows of the call, where the one-column kernel folds each code's sign into the
/// activation a row at a time. One column, or a CPU without AVX2 and FMA, dots each row through
/// [`dot_nvfp4_planes_row`]. Rows past either plane are a named panic, as is every shape refusal of
/// [`dot_nvfp4_planes_row`] for any column, and a column holding the code -128 on the tile (no
/// encoder writes it; the one-column kernel's sign fold wraps it); a scale byte that is no UE4M3
/// scale makes the row's values NaN.
pub fn dot_nvfp4_planes_rows_cols<'a>(
    (codes, scales): (&[u8], &[u8]),
    rows: Range<usize>,
    k: usize,
    (ncols, col): (usize, impl Fn(usize) -> &'a [u8]),
    mut emit: impl FnMut(usize, usize, f32),
) {
    let (code_row, scale_row) = (k / 2, k / 16);
    for j in 0..ncols {
        check_planes(code_row, scale_row, col(j).len(), k);
    }
    assert!(
        rows.end * code_row <= codes.len() && rows.end * scale_row <= scales.len(),
        "dot_nvfp4_planes_rows_cols: rows {rows:?} of {k} values pass the planes' {} code and {} \
         scale bytes",
        codes.len(),
        scales.len()
    );
    #[cfg(target_arch = "x86_64")]
    if ncols > 1 && has_features(GgmlType::NVFP4_ROWS) {
        // SAFETY: AVX2+FMA were just detected; every row lies inside both planes and every column
        // passed check_planes at k.
        unsafe { nvfp4_planes_tiled((codes, scales), rows, k, (ncols, &col), &mut emit) };
        return;
    }
    for r in rows {
        let row_codes = &codes[r * code_row..(r + 1) * code_row];
        let row_scales = &scales[r * scale_row..(r + 1) * scale_row];
        for j in 0..ncols {
            emit(r, j, dot_nvfp4_planes_row(row_codes, row_scales, col(j), k));
        }
    }
}

/// One NVFP4 row, held in place as two planes, against up to [`TILE_COLS`] quantized activation
/// columns: `out[j]` is `dot_nvfp4_planes_row(codes, scales, acols[j], k)`, bit for bit, through
/// [`dot_nvfp4_planes_rows_cols`] over the one row. A column count outside `1..=TILE_COLS`, an
/// `out` of another length and every shape refusal of [`dot_nvfp4_planes_row`] are named panics;
/// a scale byte that is no UE4M3 scale makes every value NaN.
pub fn dot_nvfp4_planes_row_cols(
    codes: &[u8],
    scales: &[u8],
    acols: &[&[u8]],
    k: usize,
    out: &mut [f32],
) {
    check_planes_cols(codes.len(), scales.len(), acols, k, out.len());
    dot_nvfp4_planes_rows_cols(
        (&codes[..k / 2], &scales[..k / 16]),
        0..1,
        k,
        (acols.len(), |j| acols[j]),
        |_, j, v| out[j] = v,
    );
}

// --------------------------------------------------------- IQ4_NL x Q8_2_X4

/// `block_iq4_nl` (ggml-common.h:586): f16 d @0, qs[16] @2 — value j (j < 16) the low
/// nibble of qs[j], value j + 16 the high nibble.
const IQ4NL_BLOCK: usize = 18;

/// AVX2+FMA+F16C row dot: IQ4_NL weights against a Q8_2_X4 column.
///
/// ik's form (`mul_mat_qX_0_q8_0_T<IQ4_NL_UnpackerS, _, block_q8_2>`,
/// `AccumT<MinusType0>`): the codes are the SIGNED `kvalues_iq4nl` bytes, so the dot
/// folds each code's sign into the activation — `maddubs(|w|, sign(q8, w))` — and no
/// offset correction exists at all: no min, no `accm`, one f32 scale per 4-block group
/// (`dw · da` per lane, broadcast to the eight accumulator lanes) and one per tail block,
/// each folded with one fmadd, reduced by `hsum_float_8`. i16: |w| <= 127 against a
/// sign-folded activation in [-128, 127], a maddubs pair peaks at 2·127·128 = 32512;
/// saturation is unreachable (`sign_epi8` keeps -128 wrapped to -128). i32: a `pall`
/// lane sums 8 i16 pairs, < 2^24.
///
/// TWIN of [`dot_q5f0_q82x4_avx2`] with five differences only: block width (18 vs 22),
/// the codes (the signed table through [`lut4_codes`] vs the qh fold), the dot (the
/// sign fold vs plain maddubs), the scales (the 4-lane `dw · da` only vs the (d, d·min)
/// pair), and no tail correction. Every edit outside those five must be made in both.
/// Also TWIN of [`dot_iq4nl_q82x4_tile_avx2`], its column tile.
///
/// # Safety
/// Caller must ensure AVX2+FMA+F16C are available and buffers match `nb` blocks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
unsafe fn dot_iq4nl_q82x4_avx2(wrow: &[u8], acol: &[u8], nb: usize) -> f32 {
    // SAFETY: AVX2+FMA+F16C present (checked by dot_row) and both slices hold the
    // validated lengths per contract.
    unsafe {
        let m4 = _mm256_set1_epi8(0xF);
        // SAFETY: 16 readable bytes of the static table.
        let t128 = _mm_loadu_si128(KVALUES_IQ4NL.as_ptr() as *const __m128i);
        let table = _mm256_set_m128i(t128, t128);
        let m1 = _mm256_set1_epi16(1);
        let mut acc = _mm256_setzero_ps();

        let nbg = nb / 4;
        for i in 0..nbg {
            let b0 = (4 * i) * IQ4NL_BLOCK;
            let b1 = (4 * i + 1) * IQ4NL_BLOCK;
            let b2 = (4 * i + 2) * IQ4NL_BLOCK;
            let b3 = (4 * i + 3) * IQ4NL_BLOCK;
            // SAFETY: 16 readable bytes inside each of the four validated blocks.
            let qx0 = lut4_codes(wrow.as_ptr().add(b0 + 2), m4, table);
            let qx1 = lut4_codes(wrow.as_ptr().add(b1 + 2), m4, table);
            let qx2 = lut4_codes(wrow.as_ptr().add(b2 + 2), m4, table);
            let qx3 = lut4_codes(wrow.as_ptr().add(b3 + 2), m4, table);
            // ScaleHelperQ_0::prepare4: the four blocks' f16 d's through one cvtph.
            // SAFETY: 8 readable bytes inside the validated row.
            let mut scales8 = [0u8; 8];
            scales8[0..2].copy_from_slice(&wrow[b0..b0 + 2]);
            scales8[2..4].copy_from_slice(&wrow[b1..b1 + 2]);
            scales8[4..6].copy_from_slice(&wrow[b2..b2 + 2]);
            scales8[6..8].copy_from_slice(&wrow[b3..b3 + 2]);
            let other = _mm_cvtph_ps(_mm_loadl_epi64(scales8.as_ptr() as *const __m128i));

            // SAFETY: 8 readable bytes at the head of the validated group
            // (convert_scales_s: the group's four bf16 d's).
            let g = acol.as_ptr().add(i * Q82X4_STRIDE);
            let aux_d = _mm_castsi128_ps(_mm_slli_epi32::<16>(_mm_cvtepu16_epi32(
                _mm_loadl_epi64(g as *const __m128i),
            )));
            // MinusType0::compute: s12 = dw · da, broadcast to both halves.
            let dall = _mm256_set_m128(_mm_mul_ps(other, aux_d), _mm_mul_ps(other, aux_d));

            // SignedDot: |w| on maddubs' unsigned side, the code's sign folded into
            // the activation. SAFETY: 32 readable bytes at each offset inside the group.
            let p0 = _mm256_madd_epi16(
                m1,
                _mm256_maddubs_epi16(
                    _mm256_sign_epi8(qx0, qx0),
                    _mm256_sign_epi8(_mm256_loadu_si256(g.add(16) as *const __m256i), qx0),
                ),
            );
            let p1 = _mm256_madd_epi16(
                m1,
                _mm256_maddubs_epi16(
                    _mm256_sign_epi8(qx1, qx1),
                    _mm256_sign_epi8(_mm256_loadu_si256(g.add(48) as *const __m256i), qx1),
                ),
            );
            let p2 = _mm256_madd_epi16(
                m1,
                _mm256_maddubs_epi16(
                    _mm256_sign_epi8(qx2, qx2),
                    _mm256_sign_epi8(_mm256_loadu_si256(g.add(80) as *const __m256i), qx2),
                ),
            );
            let p3 = _mm256_madd_epi16(
                m1,
                _mm256_maddubs_epi16(
                    _mm256_sign_epi8(qx3, qx3),
                    _mm256_sign_epi8(_mm256_loadu_si256(g.add(112) as *const __m256i), qx3),
                ),
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
            let wb = wrow.as_ptr().add(i * IQ4NL_BLOCK);
            let dw = f16c_to_f32((wb as *const u16).read_unaligned());
            let qx0 = lut4_codes(wb.add(2), m4, table);
            let tb = nb4 / 4 * Q82X4_STRIDE + (i - nb4) * Q82_BLOCK;
            let da = bf16_bits_to_f32(u16::from_le_bytes([acol[tb], acol[tb + 1]]));
            let d = dw * da;
            // SAFETY: 32 readable bytes at offset 4 of the tail block.
            let qs = _mm256_loadu_si256(acol.as_ptr().add(tb + 4) as *const __m256i);
            let p0 = _mm256_madd_epi16(
                m1,
                _mm256_maddubs_epi16(_mm256_sign_epi8(qx0, qx0), _mm256_sign_epi8(qs, qx0)),
            );
            acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(p0), acc);
        }

        hsum_float_8(acc)
    }
}

/// The IQ4_NL tile: one row's blocks against up to [`TILE_COLS`] Q8_2_X4
/// columns. Per x4 group the four blocks' signed codes, their `|w|` and the
/// f16 scales once; per column its bf16 scales, the sign-folded dots and its
/// one fmadd; per tail block the codes, `|w|` and `dw` once, then per column
/// the one-column tail step. Each column keeps its own `acc` and ends in
/// `hsum_float_8`, so column `c` sees the one-column kernel's float
/// operations in its order.
///
/// TWIN of [`dot_iq4nl_q82x4_avx2`]: any edit to the float or integer steps
/// of one is made in both.
///
/// # Safety
/// CPU must support AVX2+FMA+F16C; `wrow` must hold `nb` IQ4_NL blocks and
/// each `acols[c]` point at `col_bytes(IQ4_NL, 32 · nb)` readable bytes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
unsafe fn dot_iq4nl_q82x4_tile_avx2<const C: usize>(
    wrow: &[u8],
    acols: &[*const u8; C],
    nb: usize,
) -> [f32; C] {
    const { assert!(C != 0 && C <= TILE_COLS) };
    // SAFETY: AVX2+FMA+F16C present, `wrow` holds nb blocks and every column
    // the Q8_2_X4 groups and q8_2 tails of nb blocks per contract.
    unsafe {
        let m4 = _mm256_set1_epi8(0xF);
        // SAFETY: 16 readable bytes of the static table.
        let t128 = _mm_loadu_si128(KVALUES_IQ4NL.as_ptr() as *const __m128i);
        let table = _mm256_set_m128i(t128, t128);
        let m1 = _mm256_set1_epi16(1);
        let mut acc = [_mm256_setzero_ps(); C];

        let nbg = nb / 4;
        for i in 0..nbg {
            let b0 = (4 * i) * IQ4NL_BLOCK;
            let b1 = (4 * i + 1) * IQ4NL_BLOCK;
            let b2 = (4 * i + 2) * IQ4NL_BLOCK;
            let b3 = (4 * i + 3) * IQ4NL_BLOCK;
            // SAFETY: 16 readable bytes inside each of the four validated blocks.
            let qx0 = lut4_codes(wrow.as_ptr().add(b0 + 2), m4, table);
            let qx1 = lut4_codes(wrow.as_ptr().add(b1 + 2), m4, table);
            let qx2 = lut4_codes(wrow.as_ptr().add(b2 + 2), m4, table);
            let qx3 = lut4_codes(wrow.as_ptr().add(b3 + 2), m4, table);
            let ax0 = _mm256_sign_epi8(qx0, qx0);
            let ax1 = _mm256_sign_epi8(qx1, qx1);
            let ax2 = _mm256_sign_epi8(qx2, qx2);
            let ax3 = _mm256_sign_epi8(qx3, qx3);
            // ScaleHelperQ_0::prepare4: the four blocks' f16 d's through one cvtph.
            let mut scales8 = [0u8; 8];
            scales8[0..2].copy_from_slice(&wrow[b0..b0 + 2]);
            scales8[2..4].copy_from_slice(&wrow[b1..b1 + 2]);
            scales8[4..6].copy_from_slice(&wrow[b2..b2 + 2]);
            scales8[6..8].copy_from_slice(&wrow[b3..b3 + 2]);
            // SAFETY: 8 readable bytes in the local staging buffer.
            let other = _mm_cvtph_ps(_mm_loadl_epi64(scales8.as_ptr() as *const __m128i));

            for (col, a) in acols.iter().zip(acc.iter_mut()) {
                // SAFETY: 8 readable bytes at the head of the column's group i
                // (convert_scales_s: the group's four bf16 d's).
                let g = col.add(i * Q82X4_STRIDE);
                let aux_d = _mm_castsi128_ps(_mm_slli_epi32::<16>(_mm_cvtepu16_epi32(
                    _mm_loadl_epi64(g as *const __m128i),
                )));
                // MinusType0::compute: s12 = dw · da, broadcast to both halves.
                let dall = _mm256_set_m128(_mm_mul_ps(other, aux_d), _mm_mul_ps(other, aux_d));

                // SignedDot. SAFETY: 32 readable bytes at each offset inside the group.
                let p0 = _mm256_madd_epi16(
                    m1,
                    _mm256_maddubs_epi16(
                        ax0,
                        _mm256_sign_epi8(_mm256_loadu_si256(g.add(16) as *const __m256i), qx0),
                    ),
                );
                let p1 = _mm256_madd_epi16(
                    m1,
                    _mm256_maddubs_epi16(
                        ax1,
                        _mm256_sign_epi8(_mm256_loadu_si256(g.add(48) as *const __m256i), qx1),
                    ),
                );
                let p2 = _mm256_madd_epi16(
                    m1,
                    _mm256_maddubs_epi16(
                        ax2,
                        _mm256_sign_epi8(_mm256_loadu_si256(g.add(80) as *const __m256i), qx2),
                    ),
                );
                let p3 = _mm256_madd_epi16(
                    m1,
                    _mm256_maddubs_epi16(
                        ax3,
                        _mm256_sign_epi8(_mm256_loadu_si256(g.add(112) as *const __m256i), qx3),
                    ),
                );
                let p01 =
                    _mm256_add_epi32(_mm256_unpacklo_epi32(p0, p1), _mm256_unpackhi_epi32(p0, p1));
                let p23 =
                    _mm256_add_epi32(_mm256_unpacklo_epi32(p2, p3), _mm256_unpackhi_epi32(p2, p3));
                let pall = _mm256_add_epi32(
                    _mm256_unpacklo_epi64(p01, p23),
                    _mm256_unpackhi_epi64(p01, p23),
                );
                *a = _mm256_fmadd_ps(dall, _mm256_cvtepi32_ps(pall), *a);
            }
        }

        let nb4 = 4 * nbg;
        for i in nb4..nb {
            let wb = wrow.as_ptr().add(i * IQ4NL_BLOCK);
            let dw = f16c_to_f32((wb as *const u16).read_unaligned());
            let qx0 = lut4_codes(wb.add(2), m4, table);
            let ax0 = _mm256_sign_epi8(qx0, qx0);
            let tb = nb4 / 4 * Q82X4_STRIDE + (i - nb4) * Q82_BLOCK;
            for (col, a) in acols.iter().zip(acc.iter_mut()) {
                // SAFETY: the tail block's 36 bytes are inside the column.
                let t = col.add(tb);
                let da = bf16_bits_to_f32(u16::from_le_bytes([*t, *t.add(1)]));
                let d = dw * da;
                let qs = _mm256_loadu_si256(t.add(4) as *const __m256i);
                let p0 =
                    _mm256_madd_epi16(m1, _mm256_maddubs_epi16(ax0, _mm256_sign_epi8(qs, qx0)));
                *a = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(p0), *a);
            }
        }

        // Reduction order is load-bearing: the one-column kernel's, per column.
        let mut out = [0.0f32; C];
        for (o, a) in out.iter_mut().zip(&acc) {
            *o = hsum_float_8(*a);
        }
        out
    }
}

/// Mirror of `dot_iq4nl_q82x4_avx2`, bit-identical: the integer lanes are exact sums
/// (no saturation, no wrap — the kernel's bounds), so they are computed directly; the
/// group fmadds, the tail fmadds and the final reduction are the kernel's. The sign
/// fold is `sign_epi8` lane for lane: a -128 activation stays -128 under negation.
fn dot_iq4nl_q82x4_emul(wrow: &[u8], acol: &[u8], nb: usize) -> f32 {
    let mut acc = [0.0f32; 8];

    let signed_codes = |blk: &[u8]| -> [u8; 32] {
        std::array::from_fn(|v| {
            let nib = if v < 16 {
                blk[2 + v] & 0xF
            } else {
                blk[2 + v - 16] >> 4
            };
            KVALUES_IQ4NL[usize::from(nib)] as u8
        })
    };

    let nbg = nb / 4;
    for i in 0..nbg {
        let mut qx = [[0u8; 32]; 4];
        let mut dw = [0.0f32; 4];
        for j in 0..4 {
            let wb = &wrow[(4 * i + j) * IQ4NL_BLOCK..(4 * i + j + 1) * IQ4NL_BLOCK];
            qx[j] = signed_codes(wb);
            dw[j] = half_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
        }
        let g = &acol[i * Q82X4_STRIDE..(i + 1) * Q82X4_STRIDE];
        let mut dall = [0.0f32; 4];
        for l in 0..4 {
            let da = bf16_bits_to_f32(u16::from_le_bytes([g[2 * l], g[2 * l + 1]]));
            dall[l] = dw[l] * da;
        }
        let mut p = [[0i32; 8]; 4];
        for (j, pj) in p.iter_mut().enumerate() {
            let mut qs = [0u8; 32];
            qs.copy_from_slice(&g[16 + 32 * j..16 + 32 * j + 32]);
            // sign_epi8(w, w) and sign_epi8(q8, w), lane for lane (0x80 wraps to itself).
            let abs_w: [u8; 32] = std::array::from_fn(|n| (qx[j][n] as i8).wrapping_abs() as u8);
            let sgn_q: [u8; 32] = std::array::from_fn(|n| {
                let w = qx[j][n] as i8;
                let a = qs[n] as i8;
                (if w < 0 {
                    a.wrapping_neg()
                } else if w == 0 {
                    0
                } else {
                    a
                }) as u8
            });
            *pj = emul_madd1(emul_maddubs(abs_w, sgn_q));
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
            acc[l] = dall[l % 4].mul_add(pall[l] as f32, acc[l]);
        }
    }

    let nb4 = 4 * nbg;
    for i in nb4..nb {
        let wb = &wrow[i * IQ4NL_BLOCK..(i + 1) * IQ4NL_BLOCK];
        let dw = half_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
        let code = signed_codes(wb);
        let tb = nb4 / 4 * Q82X4_STRIDE + (i - nb4) * Q82_BLOCK;
        let blk = &acol[tb..tb + Q82_BLOCK];
        let da = bf16_bits_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let d = dw * da;
        let mut qs = [0u8; 32];
        qs.copy_from_slice(&blk[4..36]);
        let abs_w: [u8; 32] = std::array::from_fn(|n| (code[n] as i8).wrapping_abs() as u8);
        let sgn_q: [u8; 32] = std::array::from_fn(|n| {
            let w = code[n] as i8;
            let a = qs[n] as i8;
            (if w < 0 {
                a.wrapping_neg()
            } else if w == 0 {
                0
            } else {
                a
            }) as u8
        });
        let p0 = emul_madd1(emul_maddubs(abs_w, sgn_q));
        for l in 0..8 {
            acc[l] = d.mul_add(p0[l] as f32, acc[l]);
        }
    }

    let t = [
        acc[0] + acc[4],
        acc[1] + acc[5],
        acc[2] + acc[6],
        acc[3] + acc[7],
    ];
    (t[0] + t[2]) + (t[1] + t[3])
}

// ------------------------------------------------------------ IQ4_XS x Q8_K

/// `block_iq4_xs` (ggml-common.h:602): f16 d @0, u16 scales_h @2, u8 scales_l[4] @4,
/// qs[128] @8. Sub-block b's 32 values are the nibbles of qs[16b..16b+16] (low nibbles
/// first), scaled by the 6-bit scale ls[b] = low nibble b of scales_l | high 2 bits from
/// scales_h, minus 32.
const IQ4XS_BLOCK: usize = 136;

/// AVX2+FMA+F16C row dot: IQ4_XS weights against a Q8_K column.
///
/// # Safety
/// Caller must ensure AVX2+FMA+F16C are available and buffers match `nb` blocks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
unsafe fn dot_iq4xs_q8k_avx2(wrow: &[u8], acol: &[u8], nb: usize) -> f32 {
    // SAFETY: the caller validates the ISA and both buffers for nb blocks.
    unsafe { dot_iq4xs_q8k_tile_avx2(wrow, &[acol.as_ptr()], nb)[0] }
}

/// The IQ4_XS tile: unpack each weight block once for up to [`TILE_COLS`] Q8_K
/// columns. Signed table values put their magnitude on maddubs' unsigned side and
/// fold their sign into the activation. A pair is bounded by 2 * 127 * 128 = 32512,
/// so no i16 saturation occurs. Each i32 lane is bounded by
/// 8 * 32 * 4 * 127 * 128 = 16646144; the f32 conversion and one fmadd per block
/// follow ik's `mul_mat_qX_K_q8_K_T<DequantizerIQ4XS>`.
///
/// # Safety
/// CPU must support AVX2+FMA+F16C; `wrow` must hold `nb` IQ4_XS blocks and
/// each `acols[c]` point at `nb` Q8_K blocks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
unsafe fn dot_iq4xs_q8k_tile_avx2<const C: usize>(
    wrow: &[u8],
    acols: &[*const u8; C],
    nb: usize,
) -> [f32; C] {
    const { assert!(C != 0 && C <= TILE_COLS) };
    // SAFETY: AVX2+FMA+F16C present, `wrow` holds nb blocks and every column
    // nb Q8_K blocks per contract.
    unsafe {
        let m4 = _mm256_set1_epi8(0xF);
        // SAFETY: 16 readable bytes of the static table.
        let t128 = _mm_loadu_si128(KVALUES_IQ4NL.as_ptr() as *const __m128i);
        let table = _mm256_set_m128i(t128, t128);
        // ScaleIQ4XS::make_scales: the 6-bit split's shifts, masks and nibble shuffle.
        let hshift = _mm_set_epi32(12, 8, 4, 0);
        let lshift = _mm_set_epi32(4, 0, 4, 0);
        let hmask = _mm_set1_epi16(0x03);
        let lmask = _mm_set1_epi8(0xf);
        let lshuffle = _mm_set_epi32(0x0703_0602, 0x0501_0400, 0x0703_0602, 0x0501_0400);
        let m32 = _mm_set1_epi16(-32);
        let mut accd = [_mm256_setzero_ps(); C];

        for i in 0..nb {
            let blk = wrow.as_ptr().add(IQ4XS_BLOCK * i);
            // SAFETY: one unaligned u16, u16 and u32 read inside the block.
            let d = f16c_to_f32((blk as *const u16).read_unaligned());
            let scales_h = u32::from((blk.add(2) as *const u16).read_unaligned());
            let scales_l = (blk.add(4) as *const u32).read_unaligned();
            // make_scales: ls[j] - 32 on the eight i16 lanes.
            let tmp32 = scales_h | (scales_h << 14);
            let sh = _mm_slli_epi16::<4>(_mm_and_si128(
                _mm_srlv_epi32(_mm_set1_epi32(tmp32 as i32), hshift),
                hmask,
            ));
            let sl = _mm_and_si128(
                _mm_srlv_epi32(_mm_set1_epi32(scales_l as i32), lshift),
                lmask,
            );
            let scales128 = _mm_add_epi16(
                _mm_or_si128(sh, _mm_cvtepi8_epi16(_mm_shuffle_epi8(sl, lshuffle))),
                m32,
            );
            let all_scales = _mm256_set_m128i(scales128, scales128);

            let qs = blk.add(8);
            let mut values = [_mm256_setzero_si256(); 8];
            let mut scales = [_mm256_setzero_si256(); 8];
            for b in 0..8 {
                // SAFETY: 16 code bytes of sub-block b inside the block.
                values[b] = lut4_codes(qs.add(16 * b), m4, table);
                // set_scales_8: sub-block b's scale on every i16 lane.
                scales[b] = _mm256_shuffle_epi8(
                    all_scales,
                    _mm256_set1_epi16(((2 * b) | ((2 * b + 1) << 8)) as i16),
                );
            }

            for (col, a) in acols.iter().zip(accd.iter_mut()) {
                // SAFETY: the column's block i holds d @0 and 256 codes @8.
                let col = col.add(Q8K_STRIDE * i);
                let dy = (col as *const f32).read_unaligned();
                let mut sumi = _mm256_setzero_si256();
                for (b, (&value, &scale)) in values.iter().zip(&scales).enumerate() {
                    let q8 = _mm256_loadu_si256(col.add(8 + 32 * b) as *const __m256i);
                    let pair = _mm256_maddubs_epi16(
                        _mm256_sign_epi8(value, value),
                        _mm256_sign_epi8(q8, value),
                    );
                    sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(scale, pair));
                }
                *a = _mm256_fmadd_ps(_mm256_set1_ps(d * dy), _mm256_cvtepi32_ps(sumi), *a);
            }
        }

        // Reduction order is load-bearing: the one-column kernel's, per column.
        let mut out = [0.0f32; C];
        for (o, a) in out.iter_mut().zip(&accd) {
            *o = hsum_float_8(*a);
        }
        out
    }
}

/// Mirror of the IQ4_XS tile's signed dot: sign_epi8 wraps a -128 activation
/// under negation. The integer lanes, per-block fmadd and final reduction match AVX2.
fn dot_iq4xs_q8k_emul(wrow: &[u8], acol: &[u8], nb: usize) -> f32 {
    let mut accd = [0.0f32; 8];
    for i in 0..nb {
        let blk = &wrow[IQ4XS_BLOCK * i..IQ4XS_BLOCK * (i + 1)];
        let d = half_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let scales_h = u16::from_le_bytes([blk[2], blk[3]]);
        let scales_l = &blk[4..8];
        let col = &acol[Q8K_STRIDE * i..Q8K_STRIDE * (i + 1)];
        let dy = f32::from_le_bytes(col[0..4].try_into().unwrap());
        let mut sumi = [0i32; 8];
        for b in 0..8 {
            let lo = i32::from((scales_l[b / 2] >> (4 * (b % 2))) & 0x0f);
            let hi = i32::from((scales_h >> (2 * b)) & 3);
            let scale = (lo | (hi << 4)) - 32;
            for (l, sum) in sumi.iter_mut().enumerate() {
                for n in 4 * l..4 * l + 4 {
                    let byte = blk[8 + 16 * b + (n % 16)];
                    let nib = if n < 16 { byte & 0xf } else { byte >> 4 };
                    let w = KVALUES_IQ4NL[usize::from(nib)];
                    let q = emul_sign_epi8(col[8 + 32 * b + n], w) as i8;
                    *sum += scale * i32::from(w.wrapping_abs() as u8) * i32::from(q);
                }
            }
        }
        for (a, &si) in accd.iter_mut().zip(&sumi) {
            *a = (d * dy).mul_add(si as f32, *a);
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

// ------------------------------------------------- Q8_0 x act cells
// Sign fold: |w| is maddubs' u8 side, sign(w) is folded into the activation
// bytes, so |w|*(a*sign(w)) = w*a exactly, with no +128 offset to compensate.
// Integer adds are associative, tree shape is free; the f64 epilogue is scalar,
// in the mirror's order — bit identity is algebra, not a port.
// DOMAIN: activation codes must lie in [-127, 127]. -128 under a negative
// weight wraps in `sign_epi8` and flips that term's sign. The producer
// (`model::arch::deepseek2::attn::quantize_act`) clamps at -127; a debug_assert
// re-checks here.
// Weight codes may be -128 (|w| = 128 is a legal u8 magnitude).
// i16: a maddubs pair peaks at 2*128*127 = 32512 <= 32767; saturation is
// unreachable on any i8 x i8 input in this form. i32: 32 terms <= 516128.

// The weight block is a GGUF format and lives with the others in `gguf::quant`;
// the cell kernels take it, so it keeps this crate's path too.
pub use gguf::quant::Q8Block;

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
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        // SAFETY: AVX2+FMA detected just now; asserts bound every index the kernel touches.
        unsafe { q_nope2_cells_avx2_inner(whead, acol, j0, j_end, out) }
    } else {
        q_nope2_cells_scalar(whead, acol, j0, j_end, out);
    }
    #[cfg(not(target_arch = "x86_64"))]
    q_nope2_cells_scalar(whead, acol, j0, j_end, out);
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
#[cfg(target_arch = "x86_64")]
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
#[cfg(target_arch = "x86_64")]
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
/// half onto lower, `movehl`, `movehdup`). Without AVX2+FMA the same lanes run
/// one value at a time, `mul_add` for each fused add: the same bits. `None`
/// when `k` is 0 or not a multiple of 8; the caller keeps its scalar loop for that.
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
    if x.is_empty() || !x.len().is_multiple_of(8) {
        return None;
    }
    let (w, _) = wrow.as_chunks::<4>();
    let w = |i: usize| f32::from_le_bytes(w[i]);
    let mut acc: [f32; 8] = std::array::from_fn(|l| x[l] * w(l));
    for i in 1..x.len() / 8 {
        for (l, a) in acc.iter_mut().enumerate() {
            *a = x[i * 8 + l].mul_add(w(i * 8 + l), *a);
        }
    }
    let s: [f32; 4] = std::array::from_fn(|l| acc[l] + acc[l + 4]);
    Some((s[0] + s[2]) + (s[1] + s[3]))
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
/// where in the block it sits. Without AVX2+FMA one lane at a time, `v_expf` as
/// [`expf_ik_scalar`]: the same bits.
pub fn swiglu(gate: &[f32], up: &[f32], out: &mut [f32]) {
    assert!(gate.len() == up.len() && gate.len() == out.len());
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        // SAFETY: the features were just detected; the slices are equal length.
        unsafe { swiglu_avx2(gate, up, out) };
        return;
    }
    for (o, (&g, &u)) in out.iter_mut().zip(gate.iter().zip(up)) {
        *o = g / (1.0 + expf_ik_scalar(0.0 - g)) * u;
    }
}

/// ik's V4.1 SwiGLU, `mul_mat_up_gate_NxM` (`ggml/src/iqk/iqk_mul_mat.cpp`
/// :155-156 and :168-171 of the tree the V4.1 oracle was built from):
/// `out[i] = clamp(up[i], -limit, limit) · min(silu(gate[i]), limit)`. The
/// clamp bites on silu's output, not on the gate, and neither clamp applies
/// when `limit <= 1e-6` — then this is [`swiglu`]. The `min`/`max` are ik's
/// `std::min`/`std::max` operand orders: a NaN silu passes its clamp, a NaN up
/// becomes `limit`. Silu is [`swiglu`]'s `v_silu` lanes, tail included.
///
/// Returns `false` when the clamp met a `gate` or `up` value that is not
/// finite, which it would write as a finite value. Below the clamp's limit
/// every such value stays non-finite in `out`, as in [`swiglu`], and the
/// result is `true`.
pub fn swiglu_clamp(gate: &[f32], up: &[f32], limit: f32, out: &mut [f32]) -> bool {
    assert!(gate.len() == up.len() && gate.len() == out.len());
    // ik's own test, so a NaN limit clamps nothing either.
    if limit > 1e-6 {
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            // SAFETY: the features were just detected; the slices are equal length.
            return unsafe { swiglu_clamp_avx2(gate, up, limit, out) };
        }
        // Absolute f32 encodings put every finite value below infinity and NaN.
        let mut max_bits = 0u32;
        for (o, (&g, &u)) in out.iter_mut().zip(gate.iter().zip(up)) {
            max_bits = max_bits
                .max(g.to_bits() & 0x7fff_ffff)
                .max(u.to_bits() & 0x7fff_ffff);
            let s = g / (1.0 + expf_ik_scalar(0.0 - g));
            let s = if limit < s { limit } else { s };
            let c = if u < limit { u } else { limit };
            let c = if -limit < c { c } else { -limit };
            *o = c * s;
        }
        max_bits < f32::INFINITY.to_bits()
    } else {
        swiglu(gate, up, out);
        true
    }
}

/// Each lane's largest absolute f32 encoding after another `g` and `u`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
fn max_abs_bits(acc: __m256i, g: __m256, u: __m256) -> __m256i {
    let mask = _mm256_set1_epi32(i32::MAX);
    let g = _mm256_and_si256(_mm256_castps_si256(g), mask);
    let u = _mm256_and_si256(_mm256_castps_si256(u), mask);
    _mm256_max_epi32(acc, _mm256_max_epi32(g, u))
}

/// [`swiglu_clamp`]'s vector body, with its flag.
///
/// # Safety
/// The CPU must support AVX2 and FMA, and `gate`, `up` and `out` must all be
/// the same length — the loads and the store share one index.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn swiglu_clamp_avx2(gate: &[f32], up: &[f32], limit: f32, out: &mut [f32]) -> bool {
    let n = gate.len();
    let full = n / 8 * 8;
    let mut i = 0;
    // Keep the finite accumulator and clamp constants out of the expf pass.
    while i < full {
        // SAFETY: i + 8 <= full <= gate.len().
        let g = unsafe { _mm256_loadu_ps(gate.as_ptr().add(i)) };
        // SAFETY: the caller provides AVX2 and FMA.
        let s = unsafe { v_silu(g) };
        // SAFETY: i + 8 <= full <= out.len().
        unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(i), s) };
        i += 8;
    }
    let (mut gp, mut upad, mut spad, mut op) = ([0.0f32; 8], [0.0f32; 8], [0.0f32; 8], [0.0f32; 8]);
    if full < n {
        gp[..n - full].copy_from_slice(&gate[full..]);
        // SAFETY: gp holds eight f32 values.
        let g = unsafe { _mm256_loadu_ps(gp.as_ptr()) };
        // SAFETY: the caller provides AVX2 and FMA.
        let s = unsafe { v_silu(g) };
        // SAFETY: spad holds eight f32 values.
        unsafe { _mm256_storeu_ps(spad.as_mut_ptr(), s) };
    }

    // The min/max operand orders are ik's std::min/std::max orders.
    let hi = _mm256_set1_ps(limit);
    let lo = _mm256_set1_ps(-limit);
    let mut acc = _mm256_setzero_si256();
    i = 0;
    while i < full {
        // SAFETY: i + 8 <= full <= gate.len().
        let g = unsafe { _mm256_loadu_ps(gate.as_ptr().add(i)) };
        // SAFETY: i + 8 <= full <= up.len().
        let u = unsafe { _mm256_loadu_ps(up.as_ptr().add(i)) };
        // SAFETY: the first pass wrote these eight values, all inside out.
        let s = unsafe { _mm256_loadu_ps(out.as_ptr().add(i)) };
        acc = max_abs_bits(acc, g, u);
        let s = _mm256_min_ps(hi, s);
        let c = _mm256_max_ps(_mm256_min_ps(u, hi), lo);
        // SAFETY: i + 8 <= full <= out.len().
        unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(i), _mm256_mul_ps(c, s)) };
        i += 8;
    }
    if full < n {
        upad[..n - full].copy_from_slice(&up[full..]);
        // SAFETY: gp holds eight f32 values.
        let g = unsafe { _mm256_loadu_ps(gp.as_ptr()) };
        // SAFETY: upad holds eight f32 values.
        let u = unsafe { _mm256_loadu_ps(upad.as_ptr()) };
        // SAFETY: the first pass wrote all eight values of spad.
        let s = unsafe { _mm256_loadu_ps(spad.as_ptr()) };
        acc = max_abs_bits(acc, g, u);
        let s = _mm256_min_ps(hi, s);
        let c = _mm256_max_ps(_mm256_min_ps(u, hi), lo);
        // SAFETY: op holds eight f32 values.
        unsafe { _mm256_storeu_ps(op.as_mut_ptr(), _mm256_mul_ps(c, s)) };
        out[full..].copy_from_slice(&op[..n - full]);
    }
    let finite = _mm256_cmpgt_epi32(_mm256_castps_si256(_mm256_set1_ps(f32::INFINITY)), acc);
    _mm256_movemask_epi8(finite) == -1
}

/// The gpt-oss / MiniMax SwiGLU for one value, the rule the card runs
/// (`bloomery_gpu::kquant::act::swiglu_oai`): `x = min(gate, limit)`, `y =
/// clamp(up, -limit, limit)`, `x / (1 + e^(−alpha·x)) · (y + 1)` — llama.cpp's
/// `ggml_compute_forward_swiglu_oai` (and ik's `MulMat::swiglu_oai`) term for
/// term, with the exponential [`expf_ik_scalar`] where they call libm's. The
/// `min` and the clamp are the `std::min` and `std::clamp` comparisons.
#[inline]
#[must_use]
pub fn swiglu_oai_1(gate: f32, up: f32, alpha: f32, limit: f32) -> f32 {
    let x = if limit < gate { limit } else { gate };
    let y = if up < -limit {
        -limit
    } else if limit < up {
        limit
    } else {
        up
    };
    x / (1.0 + expf_ik_scalar(alpha * (0.0 - x))) * (y + 1.0)
}

/// `out[i] = swiglu_oai_1(gate[i], up[i], alpha, limit)`, AVX2 where available:
/// [`swiglu`]'s `v_expf` lanes (a tail shorter than eight goes through the
/// same lanes on a zero-padded copy, so a value never depends on where in the
/// block it sits) with the OAI rule's clamps, the same bits as the scalar
/// [`swiglu_oai_1`]. `limit` must not be negative (the clamp of `up` would have
/// its bounds reversed).
///
/// Returns `false` when `gate` or `up` holds a value that is not finite: the
/// clamps would write some of those as finite values, so the caller refuses the
/// block by name instead of using `out`.
pub fn swiglu_oai(gate: &[f32], up: &[f32], alpha: f32, limit: f32, out: &mut [f32]) -> bool {
    assert!(gate.len() == up.len() && gate.len() == out.len());
    assert!(
        limit >= 0.0 && limit.is_finite(),
        "swiglu_oai: the limit must be finite and not negative, got {limit}"
    );
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        // SAFETY: the features were just detected; the slices are equal length.
        return unsafe { swiglu_oai_avx2(gate, up, alpha, limit, out) };
    }
    // Absolute f32 encodings put every finite value below infinity and NaN.
    let mut max_bits = 0u32;
    for (o, (&g, &u)) in out.iter_mut().zip(gate.iter().zip(up)) {
        max_bits = max_bits
            .max(g.to_bits() & 0x7fff_ffff)
            .max(u.to_bits() & 0x7fff_ffff);
        *o = swiglu_oai_1(g, u, alpha, limit);
    }
    max_bits < f32::INFINITY.to_bits()
}

/// [`swiglu_oai_1`] over eight lanes; `(alpha, limit, -limit)` broadcast. The
/// `min` and `max` operand orders are the scalar rule's comparisons.
///
/// # Safety
/// The CPU must support AVX2 and FMA. Register-only: no memory is touched.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn v_swiglu_oai(g: __m256, u: __m256, (a, hi, lo): (__m256, __m256, __m256)) -> __m256 {
    let x = _mm256_min_ps(hi, g);
    let y = _mm256_max_ps(_mm256_min_ps(u, hi), lo);
    // SAFETY: same target features as the caller.
    let e = unsafe { v_expf(_mm256_mul_ps(a, _mm256_sub_ps(_mm256_setzero_ps(), x))) };
    let s = _mm256_div_ps(x, _mm256_add_ps(_mm256_set1_ps(1.0), e));
    _mm256_mul_ps(s, _mm256_add_ps(y, _mm256_set1_ps(1.0)))
}

/// [`swiglu_oai`]'s vector body, with its flag.
///
/// # Safety
/// The CPU must support AVX2 and FMA, and `gate`, `up` and `out` must all be
/// the same length — the loads and the store share one index.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn swiglu_oai_avx2(
    gate: &[f32],
    up: &[f32],
    alpha: f32,
    limit: f32,
    out: &mut [f32],
) -> bool {
    let n = gate.len();
    let full = n / 8 * 8;
    let (hi, lo, a) = (
        _mm256_set1_ps(limit),
        _mm256_set1_ps(-limit),
        _mm256_set1_ps(alpha),
    );
    let mut acc = _mm256_setzero_si256();
    let mut i = 0;
    while i < full {
        // SAFETY: i + 8 <= full <= gate.len().
        let g = unsafe { _mm256_loadu_ps(gate.as_ptr().add(i)) };
        // SAFETY: i + 8 <= full <= up.len().
        let u = unsafe { _mm256_loadu_ps(up.as_ptr().add(i)) };
        acc = max_abs_bits(acc, g, u);
        // SAFETY: this function requires AVX2 and FMA.
        let v = unsafe { v_swiglu_oai(g, u, (a, hi, lo)) };
        // SAFETY: i + 8 <= full <= out.len().
        unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(i), v) };
        i += 8;
    }
    if full < n {
        let (mut gp, mut upad, mut op) = ([0.0f32; 8], [0.0f32; 8], [0.0f32; 8]);
        gp[..n - full].copy_from_slice(&gate[full..]);
        upad[..n - full].copy_from_slice(&up[full..]);
        // SAFETY: gp holds eight f32 values.
        let g = unsafe { _mm256_loadu_ps(gp.as_ptr()) };
        // SAFETY: upad holds eight f32 values.
        let u = unsafe { _mm256_loadu_ps(upad.as_ptr()) };
        acc = max_abs_bits(acc, g, u);
        // SAFETY: this function requires AVX2 and FMA.
        let v = unsafe { v_swiglu_oai(g, u, (a, hi, lo)) };
        // SAFETY: op holds eight writable f32 values.
        unsafe { _mm256_storeu_ps(op.as_mut_ptr(), v) };
        out[full..].copy_from_slice(&op[..n - full]);
    }
    let finite = _mm256_cmpgt_epi32(_mm256_castps_si256(_mm256_set1_ps(f32::INFINITY)), acc);
    _mm256_movemask_epi8(finite) == -1
}

/// # Safety
/// The CPU must support AVX2 and FMA, and `gate`, `up` and `out` must all be
/// the same length — the loads and the store share one index.
#[cfg(target_arch = "x86_64")]
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
#[cfg(target_arch = "x86_64")]
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
#[cfg(target_arch = "x86_64")]
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
