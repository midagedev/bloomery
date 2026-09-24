//! Scalar reference dequantization for the ggml tensor types stage 1 needs.
//!
//! This is the *reference* path, not the fast path — the fast paths live in
//! `crates/q3k-cpu` (AVX2) and `crates/q3k-gemv` (CUDA) and are untouched.
//! Every decode below is a port of the corresponding
//! `dequantize_row_*` in the vendored ik_llama.cpp checkout
//! (`$IK/ggml/src/ggml-quants.c`, IK=/home/user/ik_llama.cpp), line numbers
//! cited per function. Nothing here is invented.
//!
//! FMA parity (verified by disassembling the compiled library,
//! `$IK/build/ggml/src/libggml.so`): the compiled q5_1 uses
//! `vfmadd132ps` (x0*d + m fused), q4_K and q5_K use `vfmsub132ps`
//! (q*d1 - m1 fused); q5_0, q3_K and q6_K contain no fused ops. The Rust
//! ports mirror exactly that: `mul_add` where the library fused, separate
//! multiplies where it did not. For these types the mirror does not decide
//! the bits: an f16 scale times the small integer codes stays within f32's
//! 24 significant bits, so every product is exact, the single add or
//! subtract is the only rounding, and the fused and unfused forms agree.
//! A type whose product rounds would need the mirror to match the oracle.
//!
//! MXFP4 is not in the vendored checkout's `ggml-quants.c`: its
//! `dequantize_row_mxfp4` lives in ik's `ggml/src/iqk/iqk_quantize.cpp:4233`,
//! and is the `to_float` of `[GGML_TYPE_MXFP4]` (ggml.c:1311). Its one
//! multiply, a power of two times an int8 of magnitude at most 12, is exact
//! (the subnormal scales included), so no fusion question arises.

use std::fmt;

/// GGUF v3 / ggml tensor type tags, values from `enum ggml_type`
/// (ggml.h:391; F32=0 … Q6_K=14 at ggml.h:392-406, IQ2_XXS=16 … IQ4_XS=23
/// at ggml.h:408-415, I8=24 … F64=28 at ggml.h:416-420, IQ1_M=29 at
/// ggml.h:421, BF16=30 at ggml.h:422, MXFP4=39 at ggml.h:427).
///
/// TQ1_0=34 and TQ2_0=35 are mainline ggml's (llama.cpp `930e2fa59`,
/// ggml.h:424-425); the ik tree leaves 34 and 35 unassigned, so no id means
/// two things.
///
/// A named type is one this build can size ([`GgmlType::blck_size`],
/// [`GgmlType::type_size`]); whether it can decode it is a separate question
/// ([`GgmlType::has_dequant`]). `Unknown` carries every other tag: the ids
/// the enum does not name (q4_0, q4_1, q8_1, q2_K, q8_K among them), the
/// slots ggml removed (4, 5, 31-33, 36-38) and the ids past the table. The
/// loader accepts such tensors only far enough to name them, and every
/// size/dequant entry point rejects them with an error.
// Variant names mirror ggml's type names verbatim (Q3_K, not Q3K), the way
// libc-style bindings keep the C spelling.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum GgmlType {
    F32,
    F16,
    Q5_0,
    Q5_1,
    Q8_0,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    BF16,
    MXFP4,
    IQ2_XXS,
    IQ2_XS,
    IQ3_XXS,
    IQ1_S,
    IQ4_NL,
    IQ3_S,
    IQ2_S,
    IQ4_XS,
    I8,
    I16,
    I32,
    I64,
    F64,
    IQ1_M,
    TQ1_0,
    TQ2_0,
    Unknown(u32),
}

impl GgmlType {
    #[must_use]
    pub fn from_u32(v: u32) -> Self {
        match v {
            0 => GgmlType::F32,
            1 => GgmlType::F16,
            6 => GgmlType::Q5_0,
            7 => GgmlType::Q5_1,
            8 => GgmlType::Q8_0,
            11 => GgmlType::Q3_K,
            12 => GgmlType::Q4_K,
            13 => GgmlType::Q5_K,
            14 => GgmlType::Q6_K,
            16 => GgmlType::IQ2_XXS,
            17 => GgmlType::IQ2_XS,
            18 => GgmlType::IQ3_XXS,
            19 => GgmlType::IQ1_S,
            20 => GgmlType::IQ4_NL,
            21 => GgmlType::IQ3_S,
            22 => GgmlType::IQ2_S,
            23 => GgmlType::IQ4_XS,
            24 => GgmlType::I8,
            25 => GgmlType::I16,
            26 => GgmlType::I32,
            27 => GgmlType::I64,
            28 => GgmlType::F64,
            29 => GgmlType::IQ1_M,
            30 => GgmlType::BF16,
            34 => GgmlType::TQ1_0,
            35 => GgmlType::TQ2_0,
            39 => GgmlType::MXFP4,
            other => GgmlType::Unknown(other),
        }
    }

    pub fn as_u32(self) -> u32 {
        match self {
            GgmlType::F32 => 0,
            GgmlType::F16 => 1,
            GgmlType::Q5_0 => 6,
            GgmlType::Q5_1 => 7,
            GgmlType::Q8_0 => 8,
            GgmlType::Q3_K => 11,
            GgmlType::Q4_K => 12,
            GgmlType::Q5_K => 13,
            GgmlType::Q6_K => 14,
            GgmlType::IQ2_XXS => 16,
            GgmlType::IQ2_XS => 17,
            GgmlType::IQ3_XXS => 18,
            GgmlType::IQ1_S => 19,
            GgmlType::IQ4_NL => 20,
            GgmlType::IQ3_S => 21,
            GgmlType::IQ2_S => 22,
            GgmlType::IQ4_XS => 23,
            GgmlType::I8 => 24,
            GgmlType::I16 => 25,
            GgmlType::I32 => 26,
            GgmlType::I64 => 27,
            GgmlType::F64 => 28,
            GgmlType::IQ1_M => 29,
            GgmlType::BF16 => 30,
            GgmlType::TQ1_0 => 34,
            GgmlType::TQ2_0 => 35,
            GgmlType::MXFP4 => 39,
            GgmlType::Unknown(v) => v,
        }
    }

    /// `ggml_type_name` string (type_traits table, ggml.c:620 — entries at
    /// ggml.c:657/667/756/777/819/912/938/998/1485, mxfp4 at ggml.c:1307;
    /// i8…f64 at ggml.c:621-649, the i-quants at ggml.c:1055-1293; tq1_0 and
    /// tq2_0 from mainline's table, llama.cpp ggml.c:912/920).
    /// Used to name the oracle
    /// dumps `$BLOOMERY_DATA/ref/<name>.raw`.
    pub fn name(self) -> Option<&'static str> {
        match self {
            GgmlType::F32 => Some("f32"),
            GgmlType::F16 => Some("f16"),
            GgmlType::Q5_0 => Some("q5_0"),
            GgmlType::Q5_1 => Some("q5_1"),
            GgmlType::Q8_0 => Some("q8_0"),
            GgmlType::Q3_K => Some("q3_K"),
            GgmlType::Q4_K => Some("q4_K"),
            GgmlType::Q5_K => Some("q5_K"),
            GgmlType::Q6_K => Some("q6_K"),
            GgmlType::BF16 => Some("bf16"),
            GgmlType::MXFP4 => Some("mxfp4"),
            GgmlType::IQ2_XXS => Some("iq2_xxs"),
            GgmlType::IQ2_XS => Some("iq2_xs"),
            GgmlType::IQ3_XXS => Some("iq3_xxs"),
            GgmlType::IQ1_S => Some("iq1_s"),
            GgmlType::IQ4_NL => Some("iq4_nl"),
            GgmlType::IQ3_S => Some("iq3_s"),
            GgmlType::IQ2_S => Some("iq2_s"),
            GgmlType::IQ4_XS => Some("iq4_xs"),
            GgmlType::I8 => Some("i8"),
            GgmlType::I16 => Some("i16"),
            GgmlType::I32 => Some("i32"),
            GgmlType::I64 => Some("i64"),
            GgmlType::F64 => Some("f64"),
            GgmlType::IQ1_M => Some("iq1_m"),
            GgmlType::TQ1_0 => Some("tq1_0"),
            GgmlType::TQ2_0 => Some("tq2_0"),
            GgmlType::Unknown(_) => None,
        }
    }

    /// `blck_size` from ggml's type_traits table (ggml.c:620): F32/F16/BF16 = 1
    /// (ggml.c:657/667/1485), Q5_0/Q5_1/Q8_0 = QK5_0/QK5_1/QK8_0 = 32
    /// (ggml.c:756/777/819, QK5_0/QK5_1/QK8_0 at ggml-common.h:195/210/233),
    /// K-quants = QK_K = 256 (ggml.c:912/938/998, QK_K at ggml-common.h:79),
    /// MXFP4 = QK_MXFP4 = 32 (ggml.c:1308, ggml-common.h:182); the integer
    /// types and F64 = 1 (ggml.c:621-649), IQ4_NL = QK4_NL = 32 (ggml.c:1276),
    /// the other i-quants and TQ1_0/TQ2_0 = QK_K = 256 (ggml.c:1055-1293,
    /// mainline ggml.c:912/920).
    ///
    /// This match is the single owner of those numbers for the Rust side.
    pub fn blck_size(self) -> Option<u64> {
        match self {
            GgmlType::F32
            | GgmlType::F16
            | GgmlType::BF16
            | GgmlType::I8
            | GgmlType::I16
            | GgmlType::I32
            | GgmlType::I64
            | GgmlType::F64 => Some(1),
            GgmlType::Q5_0
            | GgmlType::Q5_1
            | GgmlType::Q8_0
            | GgmlType::MXFP4
            | GgmlType::IQ4_NL => Some(32),
            GgmlType::Q3_K
            | GgmlType::Q4_K
            | GgmlType::Q5_K
            | GgmlType::Q6_K
            | GgmlType::IQ2_XXS
            | GgmlType::IQ2_XS
            | GgmlType::IQ3_XXS
            | GgmlType::IQ1_S
            | GgmlType::IQ3_S
            | GgmlType::IQ2_S
            | GgmlType::IQ4_XS
            | GgmlType::IQ1_M
            | GgmlType::TQ1_0
            | GgmlType::TQ2_0 => Some(256),
            GgmlType::Unknown(_) => None,
        }
    }

    /// `type_size` (bytes per block) from the same type_traits table:
    /// `sizeof(float)` = 4 (F32), `sizeof(ggml_fp16_t)` = 2 (F16),
    /// `sizeof(ggml_bf16_t)` = 2 (BF16, one `uint16_t`, ggml.h:380),
    /// `sizeof(block_q5_0)` = 22, `sizeof(block_q5_1)` = 24,
    /// `sizeof(block_q8_0)` = 34, `sizeof(block_q3_K)` = 110,
    /// `sizeof(block_q4_K)` = 144, `sizeof(block_q5_K)` = 176,
    /// `sizeof(block_q6_K)` = 210 — block layouts and static_asserts in
    /// ggml-common.h:327-332 (q3_K), 348-353 (q4_K), 373-378 (q5_K),
    /// 388-393 (q6_K), 196-216 (q5_0/q5_1), 233-238 (q8_0: one f16 `d` and
    /// 32 int8 codes), `sizeof(block_mxfp4)` = 17 (ggml-common.h:183-187: one
    /// E8M0 byte `e`, then 16 bytes of 4-bit codes); the block structs'
    /// static_asserts in ggml-common.h give `sizeof(block_iq2_xxs)` = 66 (:443),
    /// `block_iq2_xs` = 74 (:458), `block_iq2_s` = 82 (:474), `block_iq3_xxs`
    /// = 98 (:492), `block_iq3_s` = 110 (:510), `block_iq1_s` = 50 (:526),
    /// `block_iq1_m` = 56 (:540), `block_iq4_nl` = 18 (:590), `block_iq4_xs` =
    /// 136 (:608), and mainline's `block_tq1_0` = 54 (:281), `block_tq2_0` =
    /// 66 (:288); I8/I16/I32/I64/F64 are their C scalars, 1/2/4/8/8.
    pub fn type_size(self) -> Option<u64> {
        match self {
            GgmlType::F32 => Some(4),
            GgmlType::F16 => Some(2),
            GgmlType::BF16 => Some(2),
            GgmlType::Q5_0 => Some(22),
            GgmlType::Q5_1 => Some(24),
            GgmlType::Q8_0 => Some(34),
            GgmlType::Q3_K => Some(110),
            GgmlType::Q4_K => Some(144),
            GgmlType::Q5_K => Some(176),
            GgmlType::Q6_K => Some(210),
            GgmlType::MXFP4 => Some(17),
            GgmlType::IQ2_XXS => Some(66),
            GgmlType::IQ2_XS => Some(74),
            GgmlType::IQ3_XXS => Some(98),
            GgmlType::IQ1_S => Some(50),
            GgmlType::IQ4_NL => Some(18),
            GgmlType::IQ3_S => Some(110),
            GgmlType::IQ2_S => Some(82),
            GgmlType::IQ4_XS => Some(136),
            GgmlType::I8 => Some(1),
            GgmlType::I16 => Some(2),
            GgmlType::I32 => Some(4),
            GgmlType::I64 => Some(8),
            GgmlType::F64 => Some(8),
            GgmlType::IQ1_M => Some(56),
            GgmlType::TQ1_0 => Some(54),
            GgmlType::TQ2_0 => Some(66),
            GgmlType::Unknown(_) => None,
        }
    }

    /// Whether [`dequant_row`] decodes this type. A named type it does not
    /// decode is sized — a file holding it opens, and its bytes can be moved —
    /// but every decode of it is [`QuantError::Unsupported`].
    #[must_use]
    pub fn has_dequant(self) -> bool {
        match self {
            GgmlType::F32
            | GgmlType::F16
            | GgmlType::Q5_0
            | GgmlType::Q5_1
            | GgmlType::Q8_0
            | GgmlType::Q3_K
            | GgmlType::Q4_K
            | GgmlType::Q5_K
            | GgmlType::Q6_K
            | GgmlType::BF16
            | GgmlType::MXFP4 => true,
            GgmlType::IQ2_XXS
            | GgmlType::IQ2_XS
            | GgmlType::IQ3_XXS
            | GgmlType::IQ1_S
            | GgmlType::IQ4_NL
            | GgmlType::IQ3_S
            | GgmlType::IQ2_S
            | GgmlType::IQ4_XS
            | GgmlType::I8
            | GgmlType::I16
            | GgmlType::I32
            | GgmlType::I64
            | GgmlType::F64
            | GgmlType::IQ1_M
            | GgmlType::TQ1_0
            | GgmlType::TQ2_0
            | GgmlType::Unknown(_) => false,
        }
    }
}

impl fmt::Display for GgmlType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.name() {
            Some(n) => write!(f, "{n}"),
            None => write!(f, "ggml-type-{}", self.as_u32()),
        }
    }
}

/// Errors from the reference dequantizer. No panic path: length problems are
/// caller errors, not bugs.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum QuantError {
    #[error("ggml type {0} has no reference dequantizer in this build")]
    Unsupported(GgmlType),
    #[error("ggml type {0} has no CPU matmul in this build, so no activation format")]
    NoActivationFormat(GgmlType),
    #[error("dequant {ty}: dst length {dst} is not a multiple of the block size {blck}")]
    UnalignedDst {
        ty: GgmlType,
        dst: usize,
        blck: usize,
    },
    #[error("dequant {ty}: src has {src} bytes, needs {need} for {n} values")]
    ShortSrc {
        ty: GgmlType,
        src: usize,
        need: usize,
        n: usize,
    },
}

/// Dequantize `n = dst.len()` values of one contiguous span of type `ty`.
/// `src` must hold at least `type_size * n / blck_size` bytes (callers pass
/// the exact row slice; a longer slice is accepted).
///
/// Scalar on purpose — this is what the oracle gate compares against ggml's
/// `to_float`, so the arithmetic shape mirrors the C source exactly.
pub fn dequant_row(ty: GgmlType, src: &[u8], dst: &mut [f32]) -> Result<(), QuantError> {
    if !ty.has_dequant() {
        return Err(QuantError::Unsupported(ty));
    }
    let blck = ty.blck_size().ok_or(QuantError::Unsupported(ty))?.max(1) as usize;
    let tsz = ty.type_size().ok_or(QuantError::Unsupported(ty))? as usize;
    if !dst.len().is_multiple_of(blck) {
        return Err(QuantError::UnalignedDst {
            ty,
            dst: dst.len(),
            blck,
        });
    }
    let nblocks = dst.len() / blck;
    let need = nblocks * tsz;
    if src.len() < need {
        return Err(QuantError::ShortSrc {
            ty,
            src: src.len(),
            need,
            n: dst.len(),
        });
    }
    match ty {
        GgmlType::F32 => {
            for (o, c) in dst.iter_mut().zip(src.as_chunks::<4>().0) {
                *o = f32::from_le_bytes(*c);
            }
        }
        GgmlType::F16 => {
            for (o, c) in dst.iter_mut().zip(src.as_chunks::<2>().0) {
                *o = half_to_f32(u16::from_le_bytes(*c));
            }
        }
        GgmlType::Q5_0 => dequant_q5_0(&src[..need], dst),
        GgmlType::Q5_1 => dequant_q5_1(&src[..need], dst),
        GgmlType::Q3_K => dequant_q3_k(&src[..need], dst),
        GgmlType::Q4_K => dequant_q4_k(&src[..need], dst),
        GgmlType::Q5_K => dequant_q5_k(&src[..need], dst),
        GgmlType::Q6_K => dequant_q6_k(&src[..need], dst),
        GgmlType::Q8_0 => dequant_q8_0(&src[..need], dst),
        GgmlType::BF16 => dequant_bf16(&src[..need], dst),
        GgmlType::MXFP4 => dequant_mxfp4(&src[..need], dst),
        GgmlType::IQ2_XXS
        | GgmlType::IQ2_XS
        | GgmlType::IQ3_XXS
        | GgmlType::IQ1_S
        | GgmlType::IQ4_NL
        | GgmlType::IQ3_S
        | GgmlType::IQ2_S
        | GgmlType::IQ4_XS
        | GgmlType::I8
        | GgmlType::I16
        | GgmlType::I32
        | GgmlType::I64
        | GgmlType::F64
        | GgmlType::IQ1_M
        | GgmlType::TQ1_0
        | GgmlType::TQ2_0
        | GgmlType::Unknown(_) => return Err(QuantError::Unsupported(ty)),
    }
    Ok(())
}

/// IEEE-754 half to float, integer-only. Same routine `crates/q3k-cpu`
/// (src/main.rs:47) verified against ggml's `GGML_FP16_TO_FP32` at 1e-7 in
/// stage 0; copied (not imported) because q3k-cpu is a binary crate and its
/// tree is out of bounds for this round. The conversion is exact — every
/// f16 is representable in f32 — so it matches ggml's table lookup bit for
/// bit.
#[inline]
#[must_use]
pub fn half_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) as u32) << 31;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x3ff) as u32;
    let mag = if exp == 0x1f {
        0x7f80_0000 | (mant << 13)
    } else if exp == 0 {
        if mant == 0 {
            0
        } else {
            let mut e = 127 - 14;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            (e << 23) | ((m & 0x3ff) << 13)
        }
    } else {
        ((exp + 112) << 23) | (mant << 13)
    };
    f32::from_bits(sign | mag)
}

/// f32 → f16 bits, round-to-nearest-even — `vcvtps2ph $0x0` on the reference build (AVX2 +
/// F16C, not AVX-512). Subnormals and ties follow IEEE; NaN/inf collapse to inf (no NaN
/// reaches the gated graph).
///
/// The inverse of [`half_to_f32`]: every f16 but NaN comes back as its own bits. The
/// engine's one f32 → f16 rounding: the CPU and GPU KV caches, the GPU tensor-core query
/// rows and the Q8_0 requant's scales all round through here.
pub fn f32_to_f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let a = b & 0x7fff_ffff;
    if a >= 0x7f80_0000 {
        return sign | 0x7c00;
    }
    let exp = ((a >> 23) as i32) - 127;
    let frac = a & 0x007f_ffff;
    if exp > 15 {
        // Past f16's finite range; unreachable in the gated graph.
        return sign | 0x7c00;
    }
    if exp >= -14 {
        // Normal f16: keep 11 significand bits; round the dropped 13 with the ties-to-even carry `v + 0x0fff + ((v >> 13) & 1)`.
        let v = (((exp + 15) as u32) << 23) | frac;
        let t = v + 0x0fff + ((v >> 13) & 1);
        let h = t >> 13;
        if h & 0x7c00 == 0x7c00 {
            return sign | 0x7c00;
        }
        return sign | h as u16;
    }
    // Subnormal f16: value in units of 2^-24, round-to-nearest-even on the shift.
    if exp < -25 {
        return sign;
    }
    let shift = (-1 - exp) as u32;
    let v = 0x0080_0000 | frac;
    let half = 1u32 << (shift - 1);
    let rem = v & ((1 << shift) - 1);
    let mut h = v >> shift;
    if rem > half || (rem == half && (h & 1) == 1) {
        h += 1; // a carry into the normal range is correct IEEE behaviour
    }
    sign | h as u16
}

// ------------------------------------------------------------ legacy quants

/// Port of `dequantize_row_q5_0` (ggml-quants.c:1628). Block geometry
/// (block_q5_0, ggml-common.h:196, 22 bytes / 32 values):
/// d f16 @0, qh u32 @2 (5th bits), qs[16] @6 (low/high nibble pairs).
fn dequant_q5_0(src: &[u8], dst: &mut [f32]) {
    for (blk, out) in src
        .as_chunks::<22>()
        .0
        .iter()
        .zip(dst.as_chunks_mut::<32>().0)
    {
        let d = half_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let qh = u32::from_le_bytes([blk[2], blk[3], blk[4], blk[5]]);
        let qs = &blk[6..22];
        let (lo, hi) = out.split_at_mut(16);
        for j in 0..16 {
            // C: xh_0 = ((qh >> j) << 4) & 0x10; xh_1 = (qh >> (j+12)) & 0x10
            let xh0 = ((qh >> j) << 4) & 0x10;
            let xh1 = (qh >> (j + 12)) & 0x10;
            let x0 = ((qs[j] & 0x0f) as i32 | xh0 as i32) - 16;
            let x1 = ((qs[j] >> 4) as i32 | xh1 as i32) - 16;
            lo[j] = x0 as f32 * d;
            hi[j] = x1 as f32 * d;
        }
    }
}

/// Port of `dequantize_row_q5_1` (ggml-quants.c:1654). Block geometry
/// (block_q5_1, ggml-common.h:209, 24 bytes / 32 values):
/// d f16 @0, m f16 @2, qh u32 @4, qs[16] @8.
///
/// The compiled library fuses `x0*d + m` into one `vfmadd` — mirrored here
/// with `mul_add` (see the module comment for the objdump evidence).
fn dequant_q5_1(src: &[u8], dst: &mut [f32]) {
    for (blk, out) in src
        .as_chunks::<24>()
        .0
        .iter()
        .zip(dst.as_chunks_mut::<32>().0)
    {
        let d = half_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let m = half_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
        let qh = u32::from_le_bytes([blk[4], blk[5], blk[6], blk[7]]);
        let qs = &blk[8..24];
        let (lo, hi) = out.split_at_mut(16);
        for j in 0..16 {
            let xh0 = ((qh >> j) << 4) & 0x10;
            let xh1 = (qh >> (j + 12)) & 0x10;
            let x0 = (qs[j] & 0x0f) as i32 | xh0 as i32;
            let x1 = (qs[j] >> 4) as i32 | xh1 as i32;
            lo[j] = (x0 as f32).mul_add(d, m);
            hi[j] = (x1 as f32).mul_add(d, m);
        }
    }
}

// --------------------------------------------------------------- K-quants

/// Port of `get_scale_min_k4` (ggml-quants.c:2042), verbatim.
#[inline]
fn get_scale_min_k4(j: usize, q: &[u8; 12]) -> (i32, i32) {
    if j < 4 {
        ((q[j] & 63) as i32, (q[j + 4] & 63) as i32)
    } else {
        let d = (q[j + 4] & 0x0f) as i32 | (((q[j - 4] as i32) >> 6) << 4);
        let m = ((q[j + 4] as i32) >> 4) | (((q[j] as i32) >> 6) << 4);
        (d, m)
    }
}

/// Port of `dequantize_row_q3_K` (ggml-quants.c:2571). Block geometry
/// (block_q3_K, ggml-common.h:327, 110 bytes / 256 values):
/// hmask[32] @0, qs[64] @32, scales[12] @96 (6-bit packed), d f16 @108.
///
/// Weight k = 128c+32f+l (c = 128-value half, f = 16-value field) reads
/// qs byte 32c+16f+l, field `2f` bits; its high bit is hmask byte
/// 32c+l bit f (m = 1<<f); its sub-block scale is scales[8c+2f+l/16]-32
/// after the aux[] unpack, times d.
fn dequant_q3_k(src: &[u8], dst: &mut [f32]) {
    const KMASK1: u32 = 0x0303_0303;
    const KMASK2: u32 = 0x0f0f_0f0f;
    for (blk, out) in src
        .as_chunks::<110>()
        .0
        .iter()
        .zip(dst.as_chunks_mut::<256>().0)
    {
        let hm = &blk[0..32];
        let qs = &blk[32..96];
        let d_all = half_to_f32(u16::from_le_bytes([blk[108], blk[109]]));

        // 12 packed scale bytes -> 16 int8 scales via the aux[] shuffle,
        // verbatim from the C: only aux[0..3] are loaded (memcpy of 12
        // bytes); aux[3] is never read before being fully computed.
        let mut aux = [0u32; 4];
        for i in 0..3 {
            let s = &blk[96 + 4 * i..96 + 4 * i + 4];
            aux[i] = u32::from_le_bytes([s[0], s[1], s[2], s[3]]);
        }
        let tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
        aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
        aux[0] = (aux[0] & KMASK2) | ((tmp & KMASK1) << 4); // C: (tmp >> 0) — identity shift dropped
        aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
        let scales: [i8; 16] = core::array::from_fn(|i| aux[i / 4].to_le_bytes()[i % 4] as i8);

        let mut is = 0usize;
        let mut m = 1u8;
        let mut yi = 0usize;
        for half in 0..2 {
            let q = &qs[32 * half..32 * half + 32];
            let mut shift = 0u32;
            for _field in 0..4 {
                for half16 in 0..2 {
                    let dl = d_all * (scales[is] as i32 - 32) as f32;
                    is += 1;
                    for l in 0..16 {
                        let qv = (q[16 * half16 + l] >> shift) & 3;
                        let hv = if hm[16 * half16 + l] & m != 0 { 0 } else { 4 };
                        out[yi] = dl * (qv as i32 - hv) as f32;
                        yi += 1;
                    }
                }
                shift += 2;
                m <<= 1;
            }
        }
    }
}

/// Port of `dequantize_row_q4_K` (ggml-quants.c:2807). Block geometry
/// (block_q4_K, ggml-common.h:348, 144 bytes / 256 values):
/// d f16 @0, dmin f16 @2, scales[12] @4 (6-bit, via get_scale_min_k4),
/// qs[128] @16 (low nibble = first 32 of each 64, high nibble = second).
///
/// The compiled library computes `d1*(q&0xF) - m1` with one `vfmsub` —
/// mirrored with `mul_add(…, -m1)`.
fn dequant_q4_k(src: &[u8], dst: &mut [f32]) {
    for (blk, out) in src
        .as_chunks::<144>()
        .0
        .iter()
        .zip(dst.as_chunks_mut::<256>().0)
    {
        let d = half_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let dmin = half_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
        let scales: [u8; 12] = std::array::from_fn(|i| blk[4 + i]);
        let qs = &blk[16..144];

        let mut is = 0usize;
        for j in 0..4 {
            let (sc, mi) = get_scale_min_k4(is, &scales);
            let d1 = d * sc as f32;
            let m1 = dmin * mi as f32;
            let (sc, mi) = get_scale_min_k4(is + 1, &scales);
            let d2 = d * sc as f32;
            let m2 = dmin * mi as f32;
            let q = &qs[32 * j..32 * j + 32];
            let o = &mut out[64 * j..64 * j + 64];
            for l in 0..32 {
                o[l] = ((q[l] & 0x0f) as f32).mul_add(d1, -m1);
                o[l + 32] = ((q[l] >> 4) as f32).mul_add(d2, -m2);
            }
            is += 2;
        }
    }
}

/// Port of `dequantize_row_q5_K` (ggml-quants.c:3025). Block geometry
/// (block_q5_K, ggml-common.h:367, 176 bytes / 256 values):
/// d f16 @0, dmin f16 @2, scales[12] @4 (q4_K's 6-bit packing, via
/// get_scale_min_k4), qh[32] @16, qs[128] @48 (nibbles as in q4_K).
///
/// Value 64j+l takes bit 2j of qh[l] as its fifth bit, value 64j+32+l bit
/// 2j+1. The compiled library computes `d1*q - m1` with one `vfmsub`, as it
/// does for q4_K — mirrored with `mul_add(…, -m1)`. The mirror is for the
/// reader, not the bits: `d*sc` and `d*sc*q` (at most 11+6+5 significant
/// bits) and `dmin*m` are exact in f32, so the subtraction is the only
/// rounding and the fused and unfused forms agree.
fn dequant_q5_k(src: &[u8], dst: &mut [f32]) {
    for (blk, out) in src
        .as_chunks::<176>()
        .0
        .iter()
        .zip(dst.as_chunks_mut::<256>().0)
    {
        let d = half_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let dmin = half_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
        let scales: [u8; 12] = std::array::from_fn(|i| blk[4 + i]);
        let qh = &blk[16..48];
        let qs = &blk[48..176];

        let mut is = 0usize;
        let mut u1 = 1u8;
        let mut u2 = 2u8;
        for j in 0..4 {
            let (sc, mi) = get_scale_min_k4(is, &scales);
            let d1 = d * sc as f32;
            let m1 = dmin * mi as f32;
            let (sc, mi) = get_scale_min_k4(is + 1, &scales);
            let d2 = d * sc as f32;
            let m2 = dmin * mi as f32;
            let ql = &qs[32 * j..32 * j + 32];
            let o = &mut out[64 * j..64 * j + 64];
            for l in 0..32 {
                let h1 = if qh[l] & u1 != 0 { 16 } else { 0 };
                let h2 = if qh[l] & u2 != 0 { 16 } else { 0 };
                o[l] = (((ql[l] & 0x0f) + h1) as f32).mul_add(d1, -m1);
                o[l + 32] = (((ql[l] >> 4) + h2) as f32).mul_add(d2, -m2);
            }
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

/// Port of `dequantize_row_q6_K` (ggml-quants.c:3243). Block geometry
/// (block_q6_K, ggml-common.h:388, 210 bytes / 256 values):
/// ql[128] @0 (low nibble = values 0-63 of each 128, high = 64-127),
/// qh[64] @128 (2 high bits per value), scales int8[16] @192, d f16 @208.
///
/// Value 128c+32a+l: ql byte 64c+32a+l nibble a (low: &0xF, high: >>4),
/// high bits (qh[32c+l] >> 2a) & 3, minus 32; scale scales[8c+2a]-ish —
/// per the C: sc[is + 2a] where is = l/16.
fn dequant_q6_k(src: &[u8], dst: &mut [f32]) {
    for (blk, out) in src
        .as_chunks::<210>()
        .0
        .iter()
        .zip(dst.as_chunks_mut::<256>().0)
    {
        let d = half_to_f32(u16::from_le_bytes([blk[208], blk[209]]));
        let ql = &blk[0..128];
        let qh = &blk[128..192];
        let sc = &blk[192..208];

        let mut base = 0usize;
        for half in 0..2 {
            let ql = &ql[64 * half..64 * half + 64];
            let qh = &qh[32 * half..32 * half + 32];
            let sc = &sc[8 * half..8 * half + 8];
            for l in 0..32 {
                let is_ = l / 16;
                let q1 = (ql[l] & 0x0f) as i32 | (((qh[l] as i32) & 3) << 4);
                let q2 = (ql[l + 32] & 0x0f) as i32 | (((qh[l] as i32) >> 2 & 3) << 4);
                let q3 = (ql[l] >> 4) as i32 | (((qh[l] as i32) >> 4 & 3) << 4);
                let q4 = (ql[l + 32] >> 4) as i32 | (((qh[l] as i32) >> 6 & 3) << 4);
                // C: y = d * sc[i] * q — left-associated, no fusion measured.
                // scales is int8[] in the C struct; reinterpret the byte signed.
                out[base + l] = (d * sc[is_] as i8 as f32) * (q1 - 32) as f32;
                out[base + l + 32] = (d * sc[is_ + 2] as i8 as f32) * (q2 - 32) as f32;
                out[base + l + 64] = (d * sc[is_ + 4] as i8 as f32) * (q3 - 32) as f32;
                out[base + l + 96] = (d * sc[is_ + 6] as i8 as f32) * (q4 - 32) as f32;
            }
            base += 128;
        }
    }
}

/// Port of `dequantize_row_q8_0` (ggml-quants.c:1703). Block geometry
/// (block_q8_0, ggml-common.h:233, 34 bytes / 32 values): d f16 @0,
/// qs int8[32] @2; `y = qs·d`.
///
/// An 8-bit code times the f16 scale's 11-bit significand is exact in f32,
/// so the port matches ggml bit for bit whatever the compiler does with the
/// multiply.
fn dequant_q8_0(src: &[u8], dst: &mut [f32]) {
    for (blk, out) in src
        .as_chunks::<34>()
        .0
        .iter()
        .zip(dst.as_chunks_mut::<32>().0)
    {
        let d = half_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        for (o, &q) in out.iter_mut().zip(&blk[2..34]) {
            *o = q as i8 as f32 * d;
        }
    }
}

/// Port of `ggml_bf16_to_fp32_row` (ggml.c:445): a bf16 is the high half of
/// an f32 (`ggml_compute_bf16_to_fp32`, ggml-impl.h:89 — the AVX2 path at
/// ggml.c:457 is the same shift), so the conversion is exact.
fn dequant_bf16(src: &[u8], dst: &mut [f32]) {
    for (o, c) in dst.iter_mut().zip(src.as_chunks::<2>().0) {
        *o = f32::from_bits(u32::from(u16::from_le_bytes(*c)) << 16);
    }
}

// ------------------------------------------------------------------- MXFP4

/// The E2M1 code values doubled, `kvalues_mxfp4` (ggml-common.h:2250-2252):
/// codes 0-7 are 0, 0.5, 1, 1.5, 2, 3, 4, 6 times two, 8-15 the same negated.
pub const KVALUES_MXFP4: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

/// The block scale of an MXFP4 block, `ggml_e8m0_to_fp32_half` (ggml-impl.h:40-45):
/// `x >= 2 ? (x - 1) << 23 : {0x00200000, 0x00400000}[x]` as f32 bits, which is
/// 2^(E-128) for every E — the E8M0 scale 2^(E-127) halved, the half that the
/// doubled code table puts back. E = 0 and 1 are the two subnormal scales; E = 255
/// is 2^127, not NaN (ggml's decode does not special-case it).
#[inline]
#[must_use]
pub fn e8m0_to_f32_half(e: u8) -> f32 {
    let bits = if e >= 2 {
        u32::from(e - 1) << 23
    } else if e == 1 {
        0x0040_0000
    } else {
        0x0020_0000
    };
    f32::from_bits(bits)
}

/// Port of `dequantize_row_mxfp4` (iqk_quantize.cpp:4233). Block geometry
/// (block_mxfp4, ggml-common.h:183, 17 bytes / 32 values): E8M0 scale `e` @0,
/// qs[16] @1; value j (j < 16) is the low nibble of qs[j], value j+16 the high
/// nibble, each `d * kvalues_mxfp4[code]` with `d` = [`e8m0_to_f32_half`]`(e)`.
fn dequant_mxfp4(src: &[u8], dst: &mut [f32]) {
    for (blk, out) in src
        .as_chunks::<17>()
        .0
        .iter()
        .zip(dst.as_chunks_mut::<32>().0)
    {
        let d = e8m0_to_f32_half(blk[0]);
        let (lo, hi) = out.split_at_mut(16);
        for (j, &q) in blk[1..].iter().enumerate() {
            lo[j] = d * f32::from(KVALUES_MXFP4[usize::from(q & 0x0f)]);
            hi[j] = d * f32::from(KVALUES_MXFP4[usize::from(q >> 4)]);
        }
    }
}

/// Round-trip f32 activations through ggml's Q8_K activation quantization.
///
/// ggml does not multiply K-quant weights by f32 activations. Before a
/// `ggml_vec_dot_q*_K_q8_K` it quantizes the activation row to Q8_K — 256 values per block,
/// one f16-ish scale, int8 codes — and does the dot in integers. An f32 reference therefore
/// does NOT reproduce ggml's output: on the oracle's own inputs the exact-f32 matmul is
/// percent-level off, uniformly across tokens. That is the quantization, not an error.
///
/// So the stage-1 reference quantizes activations too, and the gates stay tight instead of
/// being opened to 1e-1 to make room for a difference we understand.
///
/// Port of `quantize_row_q8_K_ref` (ggml-quants.c:3974): the scale comes from the SIGNED
/// value with the largest magnitude, `iscale = -127 / max`, and `d = 1 / iscale`. Using
/// `amax` instead of `max` flips the sign of every code when the extreme value is positive.
///
/// `x.len()` must be a multiple of 256; ggml's K-quant rows always are.
pub fn quantize_row_q8_k_roundtrip(x: &[f32], out: &mut [f32]) {
    assert_eq!(x.len(), out.len(), "q8_K round-trip needs matching lengths");
    assert!(x.len().is_multiple_of(256), "q8_K blocks are 256 values");
    for (xb, ob) in x.chunks_exact(256).zip(out.chunks_exact_mut(256)) {
        let mut amax = 0.0f32;
        let mut max = 0.0f32;
        for &v in xb {
            let ax = v.abs();
            if ax > amax {
                amax = ax;
                max = v;
            }
        }
        if amax == 0.0 {
            ob.fill(0.0);
            continue;
        }
        let iscale = -127.0f32 / max;
        let d = 1.0f32 / iscale;
        for (o, &v) in ob.iter_mut().zip(xb) {
            // ggml's nearest_int, then the same clamp at +127 (and only at +127: the
            // negative side reaches -127 exactly by construction of iscale).
            let q = (iscale * v).round_ties_even().min(127.0);
            *o = d * q;
        }
    }
}

/// Round-trip f32 activations through ik's **Q8_2_X4** activation quantization.
///
/// ggml does not use one activation format for all weights. `ggml.c`'s type-traits table
/// gives each weight type its own `vec_dot_type`, and on this build (ik_llama.cpp with
/// `GGML_USE_IQK_MULMAT=ON`) they are:
///
/// | weight | activation |
/// |---|---|
/// | Q3_K | `Q8_K` |
/// | Q4_K, Q5_K, Q6_K, Q5_0, Q5_1, Q8_0 | `Q8_2_X4` |
/// | F32, F16 | none |
///
/// Using Q8_K for all of them is wrong by ~1e-3 on the Q5_1 down projection; the
/// error grows with the row length like √k, the signature of
/// activation-quantization noise rather than a logic error.
///
/// Block geometry (`ggml-common.h`, `block_q8_2`): 32 values, `d` as **bf16** (not f16),
/// `s` a sum this reference does not need, then 32 int8 codes. `_x4` interleaves four such
/// blocks; interleaving changes only the byte layout, so a value-level round trip does not
/// see it.
///
/// Three details that each change the last bits (`iqk_quantize.cpp:1005`):
///   * the scale is `amax / 127` from the **unsigned** max — unlike Q8_K, whose scale comes
///     from the signed extreme;
///   * `d` is converted to bf16 **and read back** before it is used to code, so the codes
///     are computed against the stored scale, not the exact one;
///   * rounding is round-half-to-even.
pub fn quantize_row_q8_2_x4_roundtrip(x: &[f32], out: &mut [f32]) {
    assert_eq!(x.len(), out.len(), "q8_2 round-trip needs matching lengths");
    assert!(x.len().is_multiple_of(32), "q8_2 blocks are 32 values");
    for (xb, ob) in x.chunks_exact(32).zip(out.chunks_exact_mut(32)) {
        let mut amax = 0.0f32;
        for &v in xb {
            amax = amax.max(v.abs());
        }
        // ggml_compute_fp32_to_bf16 (ggml-impl.h:106), then straight back: bf16 -> f32 is
        // the bits in the high half. NaN cannot occur here (amax is finite and non-negative).
        let d_exact = amax / 127.0f32;
        let bits = d_exact.to_bits();
        let bf16 = ((bits + (0x7fff + ((bits >> 16) & 1))) >> 16) as u16;
        let d = f32::from_bits((bf16 as u32) << 16);
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        for (o, &v) in ob.iter_mut().zip(xb) {
            *o = d * (v * id).round_ties_even();
        }
    }
}

/// The activation format ggml quantizes to before a dot with this weight type, as the
/// type-traits table defines it. `Ok(None)` means the activations stay f32.
///
/// This lives beside the two round-trip functions so the mapping has one owner: a caller
/// that picks the activation format itself will pick it differently somewhere else, and the
/// difference shows up as a 1e-3 numeric drift nobody can place.
///
/// A weight type no CPU matmul here takes is refused, not given a format: Q8_0 and BF16
/// activations are Q8_2_X4 and BF16 in ggml (`vec_dot_type`, ggml.c:828-837 on this AVX2
/// IQK build, ggml.c:1494), neither of which this engine encodes, and f32 would be a
/// plausible wrong answer. MXFP4's is Q8_2_X4 on AVX2 (ggml.c:1316), but no CPU MXFP4
/// matmul exists here, so it is refused the same way, as is every type [`dequant_row`]
/// does not decode.
pub fn activation_format(weight: GgmlType) -> Result<Option<ActivationFormat>, QuantError> {
    match weight {
        GgmlType::Q3_K => Ok(Some(ActivationFormat::Q8K)),
        GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q6_K | GgmlType::Q5_0 | GgmlType::Q5_1 => {
            Ok(Some(ActivationFormat::Q8_2X4))
        }
        GgmlType::F32 | GgmlType::F16 => Ok(None),
        GgmlType::Q8_0
        | GgmlType::BF16
        | GgmlType::MXFP4
        | GgmlType::IQ2_XXS
        | GgmlType::IQ2_XS
        | GgmlType::IQ3_XXS
        | GgmlType::IQ1_S
        | GgmlType::IQ4_NL
        | GgmlType::IQ3_S
        | GgmlType::IQ2_S
        | GgmlType::IQ4_XS
        | GgmlType::I8
        | GgmlType::I16
        | GgmlType::I32
        | GgmlType::I64
        | GgmlType::F64
        | GgmlType::IQ1_M
        | GgmlType::TQ1_0
        | GgmlType::TQ2_0
        | GgmlType::Unknown(_) => Err(QuantError::NoActivationFormat(weight)),
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ActivationFormat {
    Q8K,
    Q8_2X4,
}

/// Apply whichever activation quantization `weight` implies, in place of a copy; a weight
/// type [`activation_format`] refuses writes nothing.
pub fn quantize_activations(
    weight: GgmlType,
    x: &[f32],
    out: &mut [f32],
) -> Result<(), QuantError> {
    match activation_format(weight)? {
        Some(ActivationFormat::Q8K) => quantize_row_q8_k_roundtrip(x, out),
        Some(ActivationFormat::Q8_2X4) => quantize_row_q8_2_x4_roundtrip(x, out),
        None => out.copy_from_slice(x),
    }
    Ok(())
}

/// One Q8_0 block over 32 values: f16 scale bits then 32 int8 codes (34 bytes).
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Q8Block {
    /// f16 bits of the block scale (convert with `half_to_f32`).
    pub d: u16,
    /// The int8 codes: any i8. A weight code of -128 is legal; only activation codes are
    /// held to [-127, 127] (the DOMAIN note in qdot's Q8_0 x Q8_0 kernel).
    pub q: [i8; 32],
}

impl Q8Block {
    /// The block a file's 34 bytes hold: the scale's f16 bits little-endian, then the
    /// codes (`block_q8_0`, ggml-common.h).
    #[must_use]
    pub fn from_bytes(b: &[u8; 34]) -> Q8Block {
        let mut q = [0i8; 32];
        for (code, &byte) in q.iter_mut().zip(&b[2..]) {
            *code = i8::from_le_bytes([byte]);
        }
        Q8Block {
            d: u16::from_le_bytes([b[0], b[1]]),
            q,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{GgmlType, QuantError, activation_format, dequant_row};

    /// `(id, ggml_type_name, ggml_blck_size, ggml_type_size)` for every id
    /// [`GgmlType`] names, as the compiled libraries answer: ik_llama.cpp
    /// `c10fbbcc` (`build/ggml/src/libggml.so`) and, for tq1_0/tq2_0, which
    /// the ik table does not have, mainline llama.cpp `930e2fa59`
    /// (`build/bin/libggml-base.so`). The two libraries agree on every other
    /// row here.
    const GGML_TABLE: &[(u32, &str, u64, u64)] = &[
        (0, "f32", 1, 4),
        (1, "f16", 1, 2),
        (6, "q5_0", 32, 22),
        (7, "q5_1", 32, 24),
        (8, "q8_0", 32, 34),
        (11, "q3_K", 256, 110),
        (12, "q4_K", 256, 144),
        (13, "q5_K", 256, 176),
        (14, "q6_K", 256, 210),
        (16, "iq2_xxs", 256, 66),
        (17, "iq2_xs", 256, 74),
        (18, "iq3_xxs", 256, 98),
        (19, "iq1_s", 256, 50),
        (20, "iq4_nl", 32, 18),
        (21, "iq3_s", 256, 110),
        (22, "iq2_s", 256, 82),
        (23, "iq4_xs", 256, 136),
        (24, "i8", 1, 1),
        (25, "i16", 1, 2),
        (26, "i32", 1, 4),
        (27, "i64", 1, 8),
        (28, "f64", 1, 8),
        (29, "iq1_m", 256, 56),
        (30, "bf16", 1, 2),
        (34, "tq1_0", 256, 54),
        (35, "tq2_0", 256, 66),
        (39, "mxfp4", 32, 17),
    ];

    /// Ids the enum leaves `Unknown`: types it does not name yet (q4_0, q4_1,
    /// q8_1, q2_K, q8_K), slots ggml removed (4, 5, 31-33, 36-38; ik reuses
    /// 31-33 and 36 for repacked and BitNet types mainline does not have),
    /// and the first ids past both tables.
    const UNKNOWN_IDS: &[u32] = &[
        2, 3, 4, 5, 9, 10, 15, 31, 32, 33, 36, 37, 38, 40, 41, 42, 99,
    ];

    /// Every named id round-trips through the enum and carries ggml's name,
    /// block size and block bytes; every other id stays `Unknown` with no
    /// name or size.
    #[test]
    fn every_named_type_matches_ggml_table() {
        for &(id, name, blck, bytes) in GGML_TABLE {
            let ty = GgmlType::from_u32(id);
            assert!(
                !matches!(ty, GgmlType::Unknown(_)),
                "id {id} ({name}) is not named"
            );
            assert_eq!(ty.as_u32(), id, "{name}: id round trip");
            assert_eq!(ty.name(), Some(name), "id {id}: name");
            assert_eq!(ty.to_string(), name, "id {id}: Display");
            assert_eq!(ty.blck_size(), Some(blck), "{name}: block size");
            assert_eq!(ty.type_size(), Some(bytes), "{name}: block bytes");
        }
        for &id in UNKNOWN_IDS {
            let ty = GgmlType::from_u32(id);
            assert_eq!(ty, GgmlType::Unknown(id), "id {id}");
            assert_eq!(ty.as_u32(), id);
            assert_eq!(
                (ty.name(), ty.blck_size(), ty.type_size()),
                (None, None, None)
            );
            assert_eq!(ty.to_string(), format!("ggml-type-{id}"));
        }
    }

    /// A type the reference dequantizer does not decode is refused by name,
    /// whatever the lengths it is handed, and has no activation format.
    #[test]
    fn undecoded_types_are_refused_by_name() {
        for &(id, name, blck, bytes) in GGML_TABLE {
            let ty = GgmlType::from_u32(id);
            if ty.has_dequant() {
                continue;
            }
            let src = vec![0u8; bytes as usize];
            let mut dst = vec![0.0f32; blck as usize];
            let e = dequant_row(ty, &src, &mut dst).expect_err(name);
            assert_eq!(e, QuantError::Unsupported(ty));
            assert!(e.to_string().contains(name), "{e}");
            assert_eq!(
                dequant_row(ty, &[], &mut [0.0; 3]),
                Err(QuantError::Unsupported(ty)),
                "{name}: refused before any length check"
            );
            assert_eq!(
                activation_format(ty),
                Err(QuantError::NoActivationFormat(ty))
            );
        }
    }

    /// An E2M1 code (sign, two exponent bits, one mantissa bit) as the OCP
    /// microscaling spec defines it, from its bits and not from ggml's table:
    /// 0 and 0.5 below exponent 1, `(1 + m/2) · 2^(e-1)` above. The one
    /// departure is ggml's: its table maps code 8, the spec's −0, to +0.
    fn e2m1(code: u8) -> f64 {
        let (e, m) = (i32::from((code >> 1) & 3), f64::from(code & 1));
        let mag = if e == 0 {
            m * 0.5
        } else {
            (1.0 + m * 0.5) * 2f64.powi(e - 1)
        };
        if code & 8 != 0 && mag != 0.0 {
            -mag
        } else {
            mag
        }
    }

    /// MXFP4 over its whole domain: every scale byte E, every code in both
    /// nibbles, decodes to `e2m1(code) · 2^(E−127)`, computed in f64 from the
    /// spec's definitions. Every such value is an f32 (subnormals at E = 0 and
    /// 1 included, 6 · 2^128 = inf), so the comparison is on bits.
    #[test]
    fn mxfp4_decodes_every_scale_and_code() {
        let mut src = Vec::with_capacity(256 * 17);
        for e in 0..=255u8 {
            src.push(e);
            // Low nibble j, high nibble 15 − j: each block holds every code twice.
            src.extend((0..16u8).map(|j| j | ((15 - j) << 4)));
        }
        let mut dst = vec![0.0f32; 256 * 32];
        dequant_row(GgmlType::MXFP4, &src, &mut dst).unwrap();
        for (e, out) in dst.as_chunks::<32>().0.iter().enumerate() {
            let scale = 2f64.powi(e as i32 - 127);
            for (i, &got) in out.iter().enumerate() {
                let code = if i < 16 { i } else { 15 - (i - 16) } as u8;
                let want = (e2m1(code) * scale) as f32;
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "E {e} value {i} (code {code}): got {got:e}, want {want:e}"
                );
            }
        }
    }
}
