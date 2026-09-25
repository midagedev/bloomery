//! qdot — fused quantized row dots for host CPU: Q3_K x Q8_K, Q4_K/Q5_K x Q8_2_X4,
//! Q6_K x Q8_2_X4, Q5_0 x Q8_2_X4, Q5_1 x Q8_2_X4, IQ3_XXS x Q8_K, MXFP4 x Q8_2_X4,
//! and Q8_0 x act cell kernels.
//!
//! Dots quantized weight rows directly against quantized activation columns without
//! materializing f32 weights, scaling once per block:
//! - Q3_K x Q8_K: port of `ggml_vec_dot_q3_K_q8_K` (ggml-quants.c:6482).
//! - Q4_K x Q8_2_X4: port of ik's `mul_mat_qX_K_q8_2_X4_T` (iqk_gemm_kquants.cpp:783).
//! - Q6_K x Q8_2_X4: port of ik's `mul_mat_qY_K_q8_2_X4_T` (iqk_gemm_kquants.cpp:938).
//! - Q5_K x Q8_2_X4: the Q4_K template with ik's `DequantizerQ5K_AVX2` (iqk_gemm_kquants.cpp:763).
//! - Q5_0 x Q8_2_X4: port of ik's `mul_mat_qX_1_q8_2_T<Q5_0_1_Unpacker>` (iqk_gemm_legacy_quants.cpp:507).
//! - Q5_1 x Q8_2_X4: port of ik's `mul_mat_qX_1_q8_2_T<Q5_1_Unpacker>` (iqk_gemm_legacy_quants.cpp:804).
//! - IQ3_XXS x Q8_K: port of ik's `mul_mat_qX_K_q8_K_IQ_N<DequantizerIQ3XXS, 1>`
//!   (iqk_gemm_iquants.cpp:787, :494), the body its AVX2 (non-AVX512) build runs.
//! - MXFP4 x Q8_2_X4: port of ik's `mul_mat_qX_1_q8_2_T<MXFP4_Unpacker>` (iqk_gemm_legacy_quants.cpp:779).
//! - Q8_0 x act cells: fused cell kernel for `q_nope2_absorbed` (model::arch::deepseek2::attn).
//! - Q3_K, Q4_K and Q5_K tiles: one weight row against up to [`TILE_COLS`] columns, each
//!   block unpacked once, every column bit-identical to its one-column kernel ([`dot_row_cols`]).
//! - Q3_K row-lane tile: [`Q3K_R8_ROWS`] rows repacked into one lane-interleaved layout
//!   ([`repack_q3k_r8`]) against up to [`TILE_COLS`] columns, every (row, column) value
//!   bit-identical to [`dot_row`] ([`dot_q3k_r8_cols`]).
//!
//! Super-block geometry (block_q3_K, 110 bytes / 256 values): hmask[32] @+0,
//! qs[64] @+32, scales[12] @+96, f16 d @+108.
//!
//! Each AVX2 kernel has a bit-identical scalar mirror for fallback and verification.

use std::arch::x86_64::*;
use std::fmt;

use gguf::GgmlType;
use gguf::iq_tables::{IQ3XXS_GRID, KSIGNS_IQ2XS};
use gguf::quant::{KVALUES_MXFP4, e8m0_to_f32_half, half_to_f32};

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
    /// A [`dot_row_cols`] call's column count is outside `1..=TILE_COLS`, or
    /// its output slice does not hold one value per column.
    TileShape { cols: usize, outs: usize },
    /// A row-lane repack's row count is not a multiple of [`Q3K_R8_ROWS`].
    RowGroup { rows: usize },
    /// A row-lane buffer is not exactly the bytes its rows and `k` make;
    /// `buf` names which: `"repack source"` or `"repack destination"`
    /// ([`repack_q3k_r8`]'s `rows` or `out`), or `"tile group"` (the group
    /// handed to [`dot_q3k_r8_cols`]).
    RowGroupBytes {
        buf: &'static str,
        have: usize,
        need: usize,
    },
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
        }
    }
}

impl std::error::Error for QdotError {}

// ------------------------------------------------------------ public API

/// The weight types this crate has a kernel and an activation format for — the
/// one list every entry point checks against.
fn has_kernel(w: GgmlType) -> bool {
    matches!(
        w,
        GgmlType::Q3_K
            | GgmlType::Q4_K
            | GgmlType::Q5_K
            | GgmlType::Q5_0
            | GgmlType::Q5_1
            | GgmlType::Q6_K
            | GgmlType::IQ3_XXS
            | GgmlType::MXFP4
    )
}

/// Whether the fused path handles this weight type on this machine: the type
/// is supported and the CPU has the required features.
#[must_use]
pub fn supports(w: GgmlType) -> bool {
    has_kernel(w) && has_features(w)
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
/// Q3_K needs avx2+f16c; MXFP4 needs avx2+fma; every other type avx2+fma+f16c.
fn has_features(w: GgmlType) -> bool {
    // Per arm, so a call checks only the features its kernel needs: `dot_row`
    // asks once per row.
    let avx2 = || std::arch::is_x86_feature_detected!("avx2");
    let fma = || std::arch::is_x86_feature_detected!("fma");
    let f16c = || std::arch::is_x86_feature_detected!("f16c");
    match w {
        // dot_q3k_q8k_avx2: enable = "avx2", "f16c"
        GgmlType::Q3_K => avx2() && f16c(),
        // dot_mxfp4_q82x4_avx2: enable = "avx2", "fma"
        GgmlType::MXFP4 => avx2() && fma(),
        // dot_q4k/q5k/q6k_q82x4_avx2, dot_q5f0/q5f1_q82x4_avx2, dot_iq3xxs_q8k_avx2:
        // enable = "avx2", "fma", "f16c"
        GgmlType::Q4_K
        | GgmlType::Q5_K
        | GgmlType::Q6_K
        | GgmlType::Q5_0
        | GgmlType::Q5_1
        | GgmlType::IQ3_XXS => avx2() && fma() && f16c(),
        _ => false,
    }
}

/// The `k` contract: `k` must be a multiple of the weight format's block size
/// (256 for K-quants and IQ3_XXS, 32 for Q5_0, Q5_1 and MXFP4).
pub fn k_granularity(w: GgmlType) -> usize {
    match w {
        GgmlType::Q5_0 | GgmlType::Q5_1 | GgmlType::MXFP4 => 32,
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
        GgmlType::Q5_0 | GgmlType::Q5_1 | GgmlType::MXFP4 => {
            (k / 128) * Q82X4_STRIDE + ((k % 128) / 32) * Q82_BLOCK
        }
        GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q6_K => (k / 128) * Q82X4_STRIDE,
        _ => (k / 256) * Q8K_STRIDE,
    }
}

/// Quantize one activation column into the block format `w` implies.
///
/// Q3_K and IQ3_XXS use `block_q8_K` (296 B/256 values); Q4_K, Q5_K, Q5_0, Q5_1, Q6_K
/// and MXFP4 use `block_q8_2_x4` (144 B/128 values), with 36-byte `block_q8_2` tails for
/// Q5_0/Q5_1/MXFP4.
/// `out.len()` must equal `col_bytes(w, x.len())`; panics otherwise, and by
/// name on a non-finite activation value (undefined input is refused, never
/// encoded). Uses the AVX2 encoders when the CPU has AVX2 — byte-identical to
/// the scalar mirrors.
pub fn quantize_col(w: GgmlType, x: &[f32], out: &mut [u8]) {
    quantize_col_check(w, x, out);
    let avx2 = std::arch::is_x86_feature_detected!("avx2");
    match w {
        GgmlType::Q4_K
        | GgmlType::Q5_K
        | GgmlType::Q5_0
        | GgmlType::Q5_1
        | GgmlType::Q6_K
        | GgmlType::MXFP4 => {
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
        GgmlType::Q4_K
        | GgmlType::Q5_K
        | GgmlType::Q5_0
        | GgmlType::Q5_1
        | GgmlType::Q6_K
        | GgmlType::MXFP4 => {
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
        (GgmlType::IQ3_XXS, true) => {
            // SAFETY: AVX2+FMA+F16C were just detected; lengths validated above.
            Ok(unsafe { dot_iq3xxs_q8k_avx2(wrow, acol, nb) })
        }
        (GgmlType::MXFP4, true) => {
            // SAFETY: AVX2+FMA were just detected; lengths validated above.
            Ok(unsafe { dot_mxfp4_q82x4_avx2(wrow, acol, nb) })
        }
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
        GgmlType::Q5_K => dot_q5k_q82x4_emul(wrow, acol, nb),
        GgmlType::Q6_K => dot_q6k_q82x4_emul(wrow, acol, nb),
        GgmlType::Q5_0 => dot_q5f0_q82x4_emul(wrow, acol, nb),
        GgmlType::Q5_1 => dot_q5f1_q82x4_emul(wrow, acol, nb),
        GgmlType::IQ3_XXS => dot_iq3xxs_q8k_emul(wrow, acol, nb),
        GgmlType::MXFP4 => dot_mxfp4_q82x4_emul(wrow, acol, nb),
        _ => dot_q3k_q8k_scalar(wrow, acol, nb),
    }
}

/// Direct AVX2 kernel call behind [`dot_row`]; panics if required features are missing.
pub fn dot_row_avx2(w: GgmlType, wrow: &[u8], acol: &[u8], k: usize) -> Result<f32, QdotError> {
    let nb = check_row(w, wrow.len(), acol.len(), k)?;
    assert!(
        has_features(w),
        "dot_row_avx2 called on a CPU without the kernel's ISA \
         (avx2+f16c for Q3_K, avx2+fma for MXFP4, avx2+fma+f16c for the rest)"
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
        GgmlType::IQ3_XXS => {
            // SAFETY: asserted just above; lengths validated by check_row.
            Ok(unsafe { dot_iq3xxs_q8k_avx2(wrow, acol, nb) })
        }
        GgmlType::MXFP4 => {
            // SAFETY: asserted just above; lengths validated by check_row.
            Ok(unsafe { dot_mxfp4_q82x4_avx2(wrow, acol, nb) })
        }
        _ => {
            // SAFETY: asserted just above; lengths validated by check_row.
            Ok(unsafe { dot_q3k_q8k_avx2(wrow, acol, nb) })
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
    Q3K,
    Q4K,
    Q5K,
}

/// The tile kernel `w` takes on this machine: a type with one and the ISA its
/// kernel needs (the one-column kernel's, [`has_features`]).
fn tile_kind(w: GgmlType) -> Option<TileKind> {
    let kind = match w {
        GgmlType::Q3_K => TileKind::Q3K,
        GgmlType::Q4_K => TileKind::Q4K,
        GgmlType::Q5_K => TileKind::Q5K,
        _ => return None,
    };
    has_features(w).then_some(kind)
}

/// Whether [`dot_row_cols`] runs a tile kernel for `w` on this machine
/// (Q3_K, Q4_K and Q5_K with their ISA); otherwise it dots each column
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
    match c {
        2 => tile::<2>(kind, wrow, acols, nb, out),
        3 => tile::<3>(kind, wrow, acols, nb, out),
        4 => tile::<4>(kind, wrow, acols, nb, out),
        5 => tile::<5>(kind, wrow, acols, nb, out),
        6 => tile::<6>(kind, wrow, acols, nb, out),
        7 => tile::<7>(kind, wrow, acols, nb, out),
        _ => tile::<8>(kind, wrow, acols, nb, out),
    }
    Ok(())
}

/// The `C`-column tile of `kind` over validated inputs: `acols` holds `C`
/// columns of at least the activation bytes `nb` blocks need, `wrow` the
/// row's `nb` blocks, `out` `C` values.
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
    };
    out.copy_from_slice(&v);
}

/// Rows one group of the row-lane Q3_K layout interleaves ([`repack_q3k_r8`]).
pub const Q3K_R8_ROWS: usize = 8;

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
        GgmlType::Q4_K => (k / 256, (k / 256) * Q4K_BLOCK),
        GgmlType::Q5_K => (k / 256, (k / 256) * Q5K_BLOCK),
        GgmlType::IQ3_XXS => (k / 256, (k / 256) * IQ3XXS_BLOCK),
        GgmlType::MXFP4 => (k / 32, (k / 32) * MXFP4_BLOCK),
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

/// A NaN or an infinity in an activation block is undefined input: every
/// encoder path refuses it here, by name, instead of quantizing the block to a
/// defined value (a lane-wise max would skip the NaN; an infinity would give a
/// zero scale). `x` is the block; the message names the first offending value.
#[cold]
#[inline(never)]
fn non_finite_activation(x: &[f32]) -> ! {
    let (i, v) = x
        .iter()
        .enumerate()
        .find(|(_, v)| !v.is_finite())
        .map_or((usize::MAX, f32::NAN), |(i, &v)| (i, v));
    panic!(
        "qdot: non-finite activation value {v} at offset {i} of a {}-value block",
        x.len()
    );
}

/// Quantizes one 256-value block into 296-byte `block_q8_K` layout. Refuses a
/// block with a non-finite value ([`non_finite_activation`]).
fn quantize_q8k_block(x: &[f32], out: &mut [u8]) {
    debug_assert_eq!(x.len(), 256);
    debug_assert_eq!(out.len(), Q8K_STRIDE);
    let mut max = 0.0f32;
    let mut amax = 0.0f32;
    for &v in x {
        if !v.is_finite() {
            non_finite_activation(x);
        }
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

/// Reduce lanes to the `as i8` wrap range [-128, 127] — the q8_K encoder's.
/// In-domain codes never reach the wrap (|iscale * v| stays under 127.5
/// before rounding); the ±inf/NaN products of an overflowing scale extract to
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
/// AVX (`_mm256_castps256_ps128`, `_mm256_extractf128_ps`) and SSE3
/// (`_mm_movehdup_ps`) must be available on the target; being
/// `#[inline(always)]`, it takes them from its caller's features.
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
            non_finite_activation(x);
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
/// CPU must support AVX2+F16C; buffers must hold `nb` super-blocks.
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

/// The q8_2 block's code multiplier from its bf16 scale bits: `1/d`, and 0
/// where `d` is 0 or `1/d` overflows (a block whose scale is at or below
/// about 2^-128 flushes to zero codes and a zero sum). Codes are `nearest_int(v
/// * id)` saturated to i8, the sum over the unsaturated ints, which is ik's
/// x86 rule (`_mm256_cvtps_epi32` then the saturating packs): with a normal
/// `d` the bf16 round-down is at most 2^-8 relative and every code stays in
/// ±127, but a subnormal `d` keeps fewer bits and `v * id` reaches 128 —
/// saturated to 127, not wrapped to -128 (`hw_q82_codes_at_bf16_round_down`).
#[inline(always)]
fn q82_inverse(t: u16) -> f32 {
    let d = bf16_bits_to_f32(t);
    let id = if d > 0.0 { 1.0 / d } else { 0.0 };
    if id.is_finite() { id } else { 0.0 }
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
                if !v.is_finite() {
                    non_finite_activation(xb);
                }
                amax = amax.max(v.abs());
            }
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
        let mut amax = 0.0f32;
        for &v in xb {
            if !v.is_finite() {
                non_finite_activation(xb);
            }
            amax = amax.max(v.abs());
        }
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
/// byte-identical to the scalar loops in `quantize_q82x4_col`. `v * id` is a
/// plain multiply (no FMA); `id` is [`q82_inverse`]'s, so every product is
/// finite and [`v_nearest_int`] exact; the codes are the saturating packs of
/// the unsaturated ints — the scalar's clamp — and the sum is over those
/// ints, as the scalar's.
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
            non_finite_activation(x);
        }
        // The bf16 scale round-trip stays scalar: one conversion per block.
        let t = fp32_to_bf16_bits(amax / 127.0);
        let idv = _mm256_set1_ps(q82_inverse(t));
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

/// f16 bits to f32 through F16C (`vcvtph2ps`), the reference's `GGML_FP16_TO_FP32`
/// on this build. The conversion is exact, so it equals [`half_to_f32`] (the scalar
/// mirrors' converter) on every f16 but a NaN's payload.
///
/// # Safety
/// F16C must be available on the target; being `#[inline(always)]`, it takes it
/// from its caller's features.
#[inline(always)]
unsafe fn f16c_to_f32(h: u16) -> f32 {
    // SAFETY: register-only intrinsics, no memory access.
    unsafe { _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(i32::from(h)))) }
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
/// Caller must ensure AVX2+FMA+F16C are available and buffers match `nb` blocks.
#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
unsafe fn dot_q4k_q82x4_avx2(wrow: &[u8], acol: &[u8], nb: usize) -> f32 {
    // SAFETY: AVX2+FMA+F16C present and both slices hold nb blocks per contract.
    unsafe {
        let ml = _mm256_set1_epi8(0xF);
        let mut accd = _mm256_setzero_ps();

        for i in 0..nb {
            let wb = &wrow[i * Q4K_BLOCK..(i + 1) * Q4K_BLOCK];

            let d = f16c_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
            let dmin = f16c_to_f32(u16::from_le_bytes([wb[2], wb[3]]));

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

        let d = half_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
        let dmin = half_to_f32(u16::from_le_bytes([wb[2], wb[3]]));

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
/// Caller must ensure AVX2+FMA+F16C are available and buffers match `nb` blocks.
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
#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
unsafe fn dot_q5k_q82x4_avx2(wrow: &[u8], acol: &[u8], nb: usize) -> f32 {
    // SAFETY: AVX2+FMA+F16C present and both slices hold nb blocks per contract.
    unsafe {
        let ml = _mm256_set1_epi8(0xF);
        let mh = _mm256_set1_epi8(0x10);
        let mut accd = _mm256_setzero_ps();

        for i in 0..nb {
            let wb = &wrow[i * Q5K_BLOCK..(i + 1) * Q5K_BLOCK];

            let d = f16c_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
            let dmin = f16c_to_f32(u16::from_le_bytes([wb[2], wb[3]]));

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

            let d = f16c_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
            let dmin = f16c_to_f32(u16::from_le_bytes([wb[2], wb[3]]));

            let utmp = make_q4_scales(&wb[4..16]);
            // SAFETY: 8 readable bytes inside the local u32[4].
            let mins_v = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(
                utmp.as_ptr().add(2) as *const __m128i,
            )));
            let mins = _mm256_mul_ps(_mm256_set1_ps(-dmin), mins_v);

            let mut dy = [_mm256_setzero_ps(); C];
            for ((col, dyc), acc) in acols.iter().zip(dy.iter_mut()).zip(accd.iter_mut()) {
                // SAFETY: 8 readable bytes at the head of each of the column's two groups.
                let g0 = col.add((2 * i) * Q82X4_STRIDE);
                let g1 = col.add((2 * i + 1) * Q82X4_STRIDE);
                let d4_1 = _mm_cvtepu16_epi32(_mm_loadl_epi64(g0 as *const __m128i));
                let d4_2 = _mm_cvtepu16_epi32(_mm_loadl_epi64(g1 as *const __m128i));
                *dyc = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_set_m128i(d4_2, d4_1), 16));
                // SAFETY: 8 readable bytes at offset 8 of each group.
                let m4_1 = _mm_cvtepi16_epi32(_mm_loadl_epi64(g0.add(8) as *const __m128i));
                let m4_2 = _mm_cvtepi16_epi32(_mm_loadl_epi64(g1.add(8) as *const __m128i));
                let myi = _mm256_set_m128i(m4_2, m4_1);
                let my = _mm256_mul_ps(*dyc, _mm256_cvtepi32_ps(myi));
                *acc = _mm256_fmadd_ps(my, mins, *acc);
            }

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

// ------------------------------------------------------------ IQ3_XXS x Q8_K

/// `block_iq3_xxs` (ggml-common.h): f16 d @0, qs[96] @2. Bytes 0..64 of qs index
/// [`IQ3XXS_GRID`] (four magnitudes each); bytes 64..96 are eight u32 words, one per
/// 32-value sub-block: bits 7l..7l+7 the [`KSIGNS_IQ2XS`] index of its eight-value
/// group l, bits 28..32 the scale s, weight 2s + 1.
const IQ3XXS_BLOCK: usize = 98;

/// ik's `keven_signs` (iqk_common.h): entry i is [`KSIGNS_IQ2XS`]`[i]` spread to one
/// byte per bit, 0xff where the bit is set and 0x01 where it is clear — the
/// `sign_epi8` operand that negates or keeps each of eight grid values.
static KEVEN_SIGNS: [u64; 128] = keven_signs();

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
/// # Safety
/// Caller must ensure AVX2+FMA+F16C are available and buffers match `nb` blocks.
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
                    let signs = _mm256_set_epi64x(
                        KEVEN_SIGNS[((aux >> 21) & 127) as usize] as i64,
                        KEVEN_SIGNS[((aux >> 14) & 127) as usize] as i64,
                        KEVEN_SIGNS[((aux >> 7) & 127) as usize] as i64,
                        KEVEN_SIGNS[(aux & 127) as usize] as i64,
                    );
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

/// One 32-value MXFP4 block's codes as unsigned bytes 0..24 (`MXFP4_Dequantizer`).
///
/// # Safety
/// `qs` must point at 16 readable bytes; AVX2 must be present.
#[inline(always)]
unsafe fn mxfp4_codes(qs: *const u8, m4: __m256i, table: __m256i) -> __m256i {
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
/// maddubs pair peaks at 2·24·128 = 6144; saturation is unreachable.
///
/// # Safety
/// Caller must ensure AVX2+FMA are available and buffers match `nb` blocks.
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
            let qx0 = mxfp4_codes(b0.add(1), m4, table);
            let qx1 = mxfp4_codes(b1.add(1), m4, table);
            let qx2 = mxfp4_codes(b2.add(1), m4, table);
            let qx3 = mxfp4_codes(b3.add(1), m4, table);
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
            let qx0 = mxfp4_codes(wb.add(1), m4, table);
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

/// Scalar decode of one MXFP4 block's codes, `mxfp4_codes` lane for lane.
#[inline]
fn mxfp4_codes_scalar(qs: &[u8]) -> [u8; 32] {
    std::array::from_fn(|v| {
        let nib = if v < 16 { qs[v] & 0xF } else { qs[v - 16] >> 4 };
        KVALUES_MXFP4_U[usize::from(nib)]
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
            qx[j] = mxfp4_codes_scalar(&wb[1..]);
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
        let code = mxfp4_codes_scalar(&wb[1..]);
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

/// ik's V4.1 SwiGLU, `mul_mat_up_gate_NxM` (`ggml/src/iqk/iqk_mul_mat.cpp`
/// :155-156 and :168-171 of the tree the V4.1 oracle was built from):
/// `out[i] = clamp(up[i], -limit, limit) · min(silu(gate[i]), limit)`. The
/// clamp bites on silu's output, not on the gate, and neither clamp applies
/// when `limit <= 1e-6` — then this is [`swiglu`]. The `min`/`max` are ik's
/// `std::min`/`std::max` operand orders: a NaN silu passes its clamp, a NaN up
/// becomes `limit`. Silu is [`swiglu`]'s `v_silu` lanes, tail included.
pub fn swiglu_clamp(gate: &[f32], up: &[f32], limit: f32, out: &mut [f32]) {
    assert!(gate.len() == up.len() && gate.len() == out.len());
    // ik's own test, so a NaN limit clamps nothing either.
    if limit > 1e-6 {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            // SAFETY: the features were just detected; the slices are equal length.
            unsafe { swiglu_clamp_avx2(gate, up, limit, out) };
            return;
        }
        for (o, (&g, &u)) in out.iter_mut().zip(gate.iter().zip(up)) {
            let s = g / (1.0 + (-g).exp());
            let s = if limit < s { limit } else { s };
            let c = if u < limit { u } else { limit };
            let c = if -limit < c { c } else { -limit };
            *o = c * s;
        }
    } else {
        swiglu(gate, up, out);
    }
}

/// # Safety
/// The CPU must support AVX2 and FMA, and `gate`, `up` and `out` must all be
/// the same length — the loads and the store share one index.
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn swiglu_clamp_avx2(gate: &[f32], up: &[f32], limit: f32, out: &mut [f32]) {
    let n = gate.len();
    let full = n / 8 * 8;
    // `min_ps(a, b)` is `a < b ? a : b` and `max_ps(a, b)` is `a > b ? a : b`,
    // so `min_ps(hi, s)` is `std::min(s, limit)`, `min_ps(u, hi)` is
    // `std::min(limit, u)` and `max_ps(c, lo)` is `std::max(-limit, c)`.
    let hi = _mm256_set1_ps(limit);
    let lo = _mm256_set1_ps(-limit);
    let mut i = 0;
    while i < full {
        // SAFETY: `i + 8 <= full <= n` for all three slices.
        unsafe {
            let s = _mm256_min_ps(hi, v_silu(_mm256_loadu_ps(gate.as_ptr().add(i))));
            let c = _mm256_max_ps(_mm256_min_ps(_mm256_loadu_ps(up.as_ptr().add(i)), hi), lo);
            _mm256_storeu_ps(out.as_mut_ptr().add(i), _mm256_mul_ps(c, s));
        }
        i += 8;
    }
    if full < n {
        let (mut gp, mut upad, mut op) = ([0.0f32; 8], [0.0f32; 8], [0.0f32; 8]);
        gp[..n - full].copy_from_slice(&gate[full..]);
        upad[..n - full].copy_from_slice(&up[full..]);
        // SAFETY: the three arrays are eight f32 each.
        unsafe {
            let s = _mm256_min_ps(hi, v_silu(_mm256_loadu_ps(gp.as_ptr())));
            let c = _mm256_max_ps(_mm256_min_ps(_mm256_loadu_ps(upad.as_ptr()), hi), lo);
            _mm256_storeu_ps(op.as_mut_ptr(), _mm256_mul_ps(c, s));
        }
        out[full..].copy_from_slice(&op[..n - full]);
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
