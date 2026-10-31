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
//!
//! Q2_0 is mainline ggml's alone (ik has no Q2_0): its decode ports
//! llama.cpp's `dequantize_row_q2_0`, whose one product is exact.
//!
//! NVFP4 is mainline ggml's too: its decode ports llama.cpp's
//! `dequantize_row_nvfp4`, whose one product — a UE4M3 sub-block scale times
//! the doubled E2M1 code table `kvalues_mxfp4` — is exact, subnormal scales
//! included. A scale byte 0x7F or ≥ 0x80, which mainline's encoder never
//! writes and its decoder folds to 0 or drops bit 7 of, is refused by name
//! instead of decoded.
//!
//! Q2_K, IQ2_XXS, IQ2_XS, IQ2_S, IQ1_S, IQ1_M, IQ3_XXS, IQ3_S, IQ4_NL and IQ4_XS are ports of the
//! same-named functions in that file, whose bodies are identical in mainline ggml. None has a product
//! that rounds, so the order of their multiplies does not decide the bits:
//! the scale factors (`d · (0.5 + s) · 0.25`, `d · (0.5 + s) · 0.5`,
//! `d · (1 + 2s)`, `d · (ls − 32)`, `d · sc`) are an f16 significand times at
//! most five bits,
//! and a codebook or code value adds at most seven more (IQ4_XS's
//! `kvalues_iq4nl` reaches 127), 23 bits in the worst case; IQ4_NL's
//! `d · kvalues_iq4nl[code]` is 11 + 7 bits. Q2_K's
//! `dl · q − ml` rounds once, at the subtraction, fused or not. The test
//! `iq_products_are_exact` checks this over every finite f16 scale. The
//! codebooks live in [`crate::iq_tables`], generated from ggml's header.

use std::fmt;

use crate::iq_tables::{
    IQ1S_GRID, IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID, KMASK_IQ2XS,
    KSIGNS_IQ2XS, KVALUES_IQ4NL,
};
use crate::qtype::HostAct;

/// GGUF v3 / ggml tensor type tags, values from `enum ggml_type`
/// (ggml.h:391; F32=0 … Q6_K=14 at ggml.h:392-406, IQ2_XXS=16 … IQ4_XS=23
/// at ggml.h:408-415, I8=24 … F64=28 at ggml.h:416-420, IQ1_M=29 at
/// ggml.h:421, BF16=30 at ggml.h:422, MXFP4=39 at ggml.h:427).
///
/// NVFP4=40 is mainline ggml's (llama.cpp-mainline ggml.h:430); ik has no
/// NVFP4.
///
/// TQ1_0=34 and TQ2_0=35 are mainline ggml's (llama.cpp `930e2fa59`,
/// ggml.h:424-425); the ik tree leaves 34 and 35 unassigned, so no id means
/// two things. Q2_0=42 is mainline's too (llama.cpp `f2cc7282c`, ggml.h:432);
/// ik has no Q2_0. Q1_0=41 is mainline's q1_0, which ik spells q1_0_g128 at the same id with
/// the same 128-value, 18-byte block.
///
/// Q8_1=9 and Q8_K=15 are ggml's activation forms (ggml.h:399, :405): named so the table of
/// ggml's types is whole, sized, and decoded by nothing here.
///
/// F8E4M3_ROWS=2000, MXFP8=2001, NVFP4_ROWS=2002, FP8_B128=2003, MLX_A4=2004, MLX_A8=2005,
/// MLX_A4H=2006 and MLX_A8H=2007 are bloomery's own value types, in its range of ids no
/// ggml table holds (2000 up, one the next after another; 1000 up holds its re-layouts of ggml
/// types): their rows are `qtype::OWN_ROWS`, beside ggml's. MLX_A4H and MLX_A8H are MLX_A4's
/// and MLX_A8's groups whose scale and bias are F16 bits, where MLX_A4's and MLX_A8's are BF16
/// bits: two storages of one affine rule, two types, since a block's bytes do not say which.
///
/// A named type is one this build can size ([`GgmlType::blck_size`],
/// [`GgmlType::type_size`]): it has a row in [`crate::QROWS`], the one owner
/// of the facts a type has. Whether it can decode is a separate question
/// ([`GgmlType::has_dequant`]), and Q4_0 and Q4_1 are named, sized and
/// decoded by nothing here. `Unknown` carries every other tag: the
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
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    BF16,
    MXFP4,
    NVFP4,
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
    Q2_0,
    Q4_0,
    Q4_1,
    Q1_0,
    Q8_1,
    Q8_K,
    F8E4M3_ROWS,
    MXFP8,
    NVFP4_ROWS,
    FP8_B128,
    MLX_A4,
    MLX_A8,
    MLX_A4H,
    MLX_A8H,
    Unknown(u32),
}

impl GgmlType {
    #[must_use]
    pub const fn from_u32(v: u32) -> Self {
        match v {
            0 => GgmlType::F32,
            1 => GgmlType::F16,
            2 => GgmlType::Q4_0,
            3 => GgmlType::Q4_1,
            6 => GgmlType::Q5_0,
            7 => GgmlType::Q5_1,
            8 => GgmlType::Q8_0,
            9 => GgmlType::Q8_1,
            10 => GgmlType::Q2_K,
            11 => GgmlType::Q3_K,
            12 => GgmlType::Q4_K,
            13 => GgmlType::Q5_K,
            14 => GgmlType::Q6_K,
            15 => GgmlType::Q8_K,
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
            40 => GgmlType::NVFP4,
            41 => GgmlType::Q1_0,
            42 => GgmlType::Q2_0,
            2000 => GgmlType::F8E4M3_ROWS,
            2001 => GgmlType::MXFP8,
            2002 => GgmlType::NVFP4_ROWS,
            2003 => GgmlType::FP8_B128,
            2004 => GgmlType::MLX_A4,
            2005 => GgmlType::MLX_A8,
            2006 => GgmlType::MLX_A4H,
            2007 => GgmlType::MLX_A8H,
            other => GgmlType::Unknown(other),
        }
    }

    pub const fn as_u32(self) -> u32 {
        match self {
            GgmlType::F32 => 0,
            GgmlType::F16 => 1,
            GgmlType::Q4_0 => 2,
            GgmlType::Q4_1 => 3,
            GgmlType::Q5_0 => 6,
            GgmlType::Q5_1 => 7,
            GgmlType::Q8_0 => 8,
            GgmlType::Q8_1 => 9,
            GgmlType::Q2_K => 10,
            GgmlType::Q3_K => 11,
            GgmlType::Q4_K => 12,
            GgmlType::Q5_K => 13,
            GgmlType::Q6_K => 14,
            GgmlType::Q8_K => 15,
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
            GgmlType::NVFP4 => 40,
            GgmlType::Q1_0 => 41,
            GgmlType::Q2_0 => 42,
            GgmlType::F8E4M3_ROWS => 2000,
            GgmlType::MXFP8 => 2001,
            GgmlType::NVFP4_ROWS => 2002,
            GgmlType::FP8_B128 => 2003,
            GgmlType::MLX_A4 => 2004,
            GgmlType::MLX_A8 => 2005,
            GgmlType::MLX_A4H => 2006,
            GgmlType::MLX_A8H => 2007,
            GgmlType::Unknown(v) => v,
        }
    }

    /// `ggml_type_name`, from the type's row; `None` for [`GgmlType::Unknown`].
    /// Names the oracle dumps `$BLOOMERY_DATA/ref/<name>.raw`.
    #[must_use]
    pub fn name(self) -> Option<&'static str> {
        self.row().map(|r| r.name)
    }

    /// `blck_size` from ggml's type_traits table, as the type's row holds it (values per
    /// block); `None` for [`GgmlType::Unknown`]. The row is the single owner of the number for
    /// the Rust side, with the source lines of every type.
    #[must_use]
    pub fn blck_size(self) -> Option<u64> {
        self.row().map(|r| r.blck)
    }

    /// `type_size` (bytes per block) from the same table, as the type's row holds it; `None`
    /// for [`GgmlType::Unknown`].
    #[must_use]
    pub fn type_size(self) -> Option<u64> {
        self.row().map(|r| r.bytes)
    }

    /// Whether [`dequant_row`] decodes this type. A named type it does not
    /// decode is sized — a file holding it opens, and its bytes can be moved —
    /// but every decode of it is [`QuantError::Unsupported`].
    #[must_use]
    pub fn has_dequant(self) -> bool {
        self.row().is_some_and(|r| r.dequant)
    }

    /// Whether a tensor of this type is read with the f32 values of `<stem>.scale` beside
    /// `<stem>.weight`: NVFP4 (one a tensor, or one an expert of a stack) and bloomery's F8
    /// and NVFP4 rows (one a tensor). The writer that stores the scale and the loader that
    /// reads it both ask here.
    #[must_use]
    pub const fn reads_scale(self) -> bool {
        matches!(
            self,
            GgmlType::NVFP4 | GgmlType::F8E4M3_ROWS | GgmlType::NVFP4_ROWS
        )
    }
}

/// ggml's name for a type this enum names, so a refusal of such a tensor names its type;
/// `ggml-type-<id>` for any other id.
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
    /// A block scale byte the format leaves undefined, or one a value of its
    /// block leaves f32's range at: no value is written for it.
    #[error("dequant {ty}: block {block} sub-block {sub} holds scale byte {byte:#04x}, {why}")]
    UndefinedScale {
        ty: GgmlType,
        block: usize,
        sub: usize,
        byte: u8,
        why: &'static str,
    },
    /// A block's f32 scale that is not finite, or one a value of its block
    /// leaves f32's range at ([`check_fp8_b128`]): no value is written for
    /// it.
    #[error("dequant {ty}: block {block} holds f32 scale bits {bits:#010x}, {why}")]
    UndefinedF32Scale {
        ty: GgmlType,
        block: usize,
        bits: u32,
        why: &'static str,
    },
    /// An MLX affine block's scale or bias (BF16 or F16 bits, as its type
    /// stores them) that is not finite, or a scale whose largest product with
    /// a code leaves f32's range at ([`check_mlx_affine`]): no value is
    /// written for its group.
    #[error("dequant {ty}: block {block} holds {field} bits {bits:#06x}, {why}")]
    UndefinedAffinePair {
        ty: GgmlType,
        block: usize,
        field: &'static str,
        bits: u16,
        why: &'static str,
    },
    /// A byte that is no UE4M3 scale, outside a block ([`ue4m3_to_f32_half`]).
    #[error("{byte:#04x} is no UE4M3 scale: {why}")]
    NotUe4m3 { byte: u8, why: &'static str },
    /// An F8_E4M3 byte that holds no value ([`E4m3Rows`]), at its index in
    /// the span.
    #[error("f8_e4m3 value {at} is byte {byte:#04x}, E4M3's NaN")]
    E4m3Nan { at: usize, byte: u8 },
    /// An F8_E4M3 value whose product with the tensor scale is not finite.
    #[error("f8_e4m3 value {at} is byte {byte:#04x}, its scaled value is not finite")]
    E4m3NonFinite { at: usize, byte: u8 },
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
        GgmlType::Q2_K => dequant_q2_k(&src[..need], dst),
        GgmlType::Q3_K => dequant_q3_k(&src[..need], dst),
        GgmlType::Q4_K => dequant_q4_k(&src[..need], dst),
        GgmlType::Q5_K => dequant_q5_k(&src[..need], dst),
        GgmlType::Q6_K => dequant_q6_k(&src[..need], dst),
        GgmlType::Q8_0 => dequant_q8_0(&src[..need], dst),
        GgmlType::BF16 => dequant_bf16(&src[..need], dst),
        GgmlType::MXFP4 => dequant_mxfp4(&src[..need], dst),
        GgmlType::NVFP4 => dequant_nvfp4(&src[..need], dst)?,
        GgmlType::IQ2_XXS => dequant_iq2_xxs(&src[..need], dst),
        GgmlType::IQ2_XS => dequant_iq2_xs(&src[..need], dst),
        GgmlType::IQ2_S => dequant_iq2_s(&src[..need], dst),
        GgmlType::IQ1_S => dequant_iq1_s(&src[..need], dst),
        GgmlType::IQ1_M => dequant_iq1_m(&src[..need], dst),
        GgmlType::IQ3_XXS => dequant_iq3_xxs(&src[..need], dst),
        GgmlType::IQ3_S => dequant_iq3_s(&src[..need], dst),
        GgmlType::IQ4_NL => dequant_iq4_nl(&src[..need], dst),
        GgmlType::IQ4_XS => dequant_iq4_xs(&src[..need], dst),
        GgmlType::Q2_0 => dequant_q2_0(&src[..need], dst),
        GgmlType::MXFP8 => dequant_mxfp8(&src[..need], dst)?,
        GgmlType::FP8_B128 => dequant_fp8_b128(&src[..need], dst)?,
        GgmlType::MLX_A4 => dequant_mlx_a4(ty, AffinePair::Bf16, &src[..need], dst)?,
        GgmlType::MLX_A8 => dequant_mlx_a8(ty, AffinePair::Bf16, &src[..need], dst)?,
        GgmlType::MLX_A4H => dequant_mlx_a4(ty, AffinePair::F16, &src[..need], dst)?,
        GgmlType::MLX_A8H => dequant_mlx_a8(ty, AffinePair::F16, &src[..need], dst)?,
        GgmlType::Q4_0
        | GgmlType::Q4_1
        | GgmlType::I8
        | GgmlType::I16
        | GgmlType::I32
        | GgmlType::I64
        | GgmlType::F64
        | GgmlType::TQ1_0
        | GgmlType::TQ2_0
        | GgmlType::Q1_0
        | GgmlType::Q8_1
        | GgmlType::Q8_K
        | GgmlType::F8E4M3_ROWS
        | GgmlType::NVFP4_ROWS
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
/// F16C, not AVX-512). Subnormals, ties and infinities follow IEEE; a NaN stays a NaN of
/// its sign (quiet, payload dropped: `sign | 0x7e00`), so a cache row or query staged from
/// a refused value is never a plausible number.
///
/// The inverse of [`half_to_f32`]: every f16 but NaN comes back as its own bits. The
/// engine's one f32 → f16 rounding: the CPU and GPU KV caches, the GPU tensor-core query
/// rows and the Q8_0 requant's scales all round through here.
pub fn f32_to_f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let a = b & 0x7fff_ffff;
    if a > 0x7f80_0000 {
        return sign | 0x7e00;
    }
    let exp = ((a >> 23) as i32) - 127;
    let frac = a & 0x007f_ffff;
    if exp > 15 {
        // Past f16's finite range, infinity included.
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

/// Converts fp32 to bf16 bits with round to nearest even.
#[inline]
pub fn fp32_to_bf16_bits(s: f32) -> u16 {
    let u = s.to_bits();
    if (u & 0x7fff_ffff) > 0x7f80_0000 {
        ((u >> 16) | 64) as u16
    } else {
        (u.wrapping_add(0x7fff + ((u >> 16) & 1)) >> 16) as u16
    }
}

/// Converts bf16 bits to f32.
#[inline]
pub fn bf16_bits_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
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

/// Port of `dequantize_row_q2_K` (ggml-quants.c:2168). Block geometry
/// (block_q2_K, ggml-common.h:309-314, 84 bytes / 256 values): scales[16] @0
/// (low nibble the scale, high nibble the min, one byte per 16 values),
/// qs[64] @16, d f16 @80, dmin f16 @82 (`GGML_SCALE_TYPE1`, :16).
///
/// Weight k = 128c+32f+16h+l (c = 128-value half, f = 2-bit field, h = 16-value
/// half) reads qs byte 32c+16h+l, bits 2f..2f+2, with scale byte 8c+2f+h:
/// `d·(sc & 15)·q − dmin·(sc >> 4)`. The product is exact, so the one rounding
/// is the subtraction whether or not a compiler fuses it.
fn dequant_q2_k(src: &[u8], dst: &mut [f32]) {
    for (blk, out) in src
        .as_chunks::<84>()
        .0
        .iter()
        .zip(dst.as_chunks_mut::<256>().0)
    {
        let d = half_to_f32(u16::from_le_bytes([blk[80], blk[81]]));
        let min = half_to_f32(u16::from_le_bytes([blk[82], blk[83]]));
        let scales = &blk[0..16];
        let qs = &blk[16..80];
        let mut is = 0usize;
        let mut y = 0usize;
        for half in 0..2 {
            let q = &qs[32 * half..32 * half + 32];
            for field in 0..4 {
                let shift = 2 * field;
                for h in 0..2 {
                    let sc = scales[is];
                    is += 1;
                    let dl = d * f32::from(sc & 0x0f);
                    let ml = min * f32::from(sc >> 4);
                    for &byte in &q[16 * h..16 * h + 16] {
                        out[y] = dl * f32::from((byte >> shift) & 3) - ml;
                        y += 1;
                    }
                }
            }
        }
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

/// Port of mainline's `dequantize_row_q2_0` (llama.cpp `f2cc7282c`
/// ggml-quants.c:439; ik has no Q2_0). Block geometry (block_q2_0,
/// ggml-common.h:187-192, 18 bytes / 64 values): d f16 @0, qs[16] @2.
///
/// Value j is bits `2·(j % 4)` and up of `qs[j / 4]`, the low pair first, and
/// decodes to `(q − 1)·d`: codes 0..3 are −1, 0, +1, +2. An f16 times an
/// integer of magnitude at most 2 is exact in f32.
fn dequant_q2_0(src: &[u8], dst: &mut [f32]) {
    for (blk, out) in src
        .as_chunks::<18>()
        .0
        .iter()
        .zip(dst.as_chunks_mut::<64>().0)
    {
        let d = half_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        for (&byte, o4) in blk[2..].iter().zip(out.as_chunks_mut::<4>().0) {
            for (k, o) in o4.iter_mut().enumerate() {
                *o = (i32::from((byte >> (2 * k)) & 3) - 1) as f32 * d;
            }
        }
    }
}

/// Port of `ggml_bf16_to_fp32_row` (ggml.c:445): a bf16 is the high half of
/// an f32 (`ggml_compute_bf16_to_fp32`, ggml-impl.h:89 — the AVX2 path at
/// ggml.c:457 is the same shift), so the conversion is exact.
fn dequant_bf16(src: &[u8], dst: &mut [f32]) {
    for (o, c) in dst.iter_mut().zip(src.as_chunks::<2>().0) {
        *o = bf16_bits_to_f32(u16::from_le_bytes(*c));
    }
}

// ------------------------------------------------------------------- MXFP4

/// The E2M1 code values doubled, `kvalues_mxfp4` (ggml-common.h:2250-2252):
/// codes 0-7 are 0, 0.5, 1, 1.5, 2, 3, 4, 6 times two, 8-15 the same negated.
pub const KVALUES_MXFP4: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

/// The two doubled E2M1 values of one FP4 code byte: the low nibble's
/// [`KVALUES_MXFP4`] entry, then the high nibble's. The one host decode of a
/// code byte, which MXFP4 and NVFP4 share (they differ only in their scales
/// and in which values a byte's two nibbles are).
#[inline]
#[must_use]
pub fn fp4_pair(byte: u8) -> [i8; 2] {
    [
        KVALUES_MXFP4[usize::from(byte & 0x0f)],
        KVALUES_MXFP4[usize::from(byte >> 4)],
    ]
}

/// One 16-value group of NVFP4 rows' codes ([`GgmlType::NVFP4_ROWS`]: value `2i` the low nibble
/// of byte `i`, value `2i + 1` its high nibble) in the order an NVFP4 block holds them (ggml's
/// `block_nvfp4`, the card's code plane: byte `j` holds value `j` low and value `j + 8` high), its
/// nibbles moved and none changed. The one rule of that move: a reader of a checkpoint's code
/// plane runs it, and qdot's AVX2 body is held to it bit for bit.
#[inline]
#[must_use]
pub fn nvfp4_group_to_block(g: &[u8; 8]) -> [u8; 8] {
    let spread = |half: [u8; 4]| {
        let x = u64::from(u32::from_le_bytes(half));
        let x = (x | x << 16) & 0x0000_ffff_0000_ffff;
        let x = (x | x << 8) & 0x00ff_00ff_00ff_00ff;
        (x | x << 4) & 0x0f0f_0f0f_0f0f_0f0f
    };
    let [a, b, c, d, e, f, h, i] = *g;
    (spread([a, b, c, d]) | spread([e, f, h, i]) << 4).to_le_bytes()
}

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
            let [l, h] = fp4_pair(q);
            lo[j] = d * f32::from(l);
            hi[j] = d * f32::from(h);
        }
    }
}

// ------------------------------------------------------------------- NVFP4

/// An NVFP4 sub-block scale, `ggml_ue4m3_to_fp32` (llama.cpp-mainline
/// ggml-impl.h:512-527): unsigned E4M3, bias 7, a subnormal `m · 2^-9` at
/// exponent 0, times 0.5 — the half the doubled code table
/// ([`KVALUES_MXFP4`], mainline's `kvalues_fp4`, ggml-common.h:1126-1129)
/// puts back. The byte has no sign bit: a byte with bit 7 set, and 0x7F
/// (E4M3's NaN), are not scales. Mainline decodes 0x7F to 0 and drops bit 7;
/// its encoder (`ggml_fp32_to_ue4m3`, ggml-impl.h:529-565) writes neither,
/// so each is refused here by name instead.
///
/// # Errors
///
/// [`QuantError::NotUe4m3`], naming the byte and why it is no scale.
pub fn ue4m3_to_f32_half(byte: u8) -> Result<f32, QuantError> {
    ue4m3(byte).map_err(|why| QuantError::NotUe4m3 { byte, why })
}

/// [`ue4m3_to_f32_half`] with the reason alone, for a caller that names the
/// block and sub-block itself.
fn ue4m3(byte: u8) -> Result<f32, &'static str> {
    if byte & 0x80 != 0 {
        return Err("bit 7 set: UE4M3 has no sign bit");
    }
    if byte == 0x7F {
        return Err("E4M3's NaN");
    }
    let (exp, man) = (i32::from(byte >> 3), f32::from(byte & 7));
    let raw = if exp == 0 {
        man * 2f32.powi(-9)
    } else {
        (1.0 + man / 8.0) * 2f32.powi(exp - 7)
    };
    Ok(raw * 0.5)
}

/// Rows of bloomery's F8 type ([`GgmlType::F8E4M3_ROWS`], one byte a value) decoded: each byte an
/// F8_E4M3 value (OCP E4M3FN: a sign bit, four exponent bits of bias 7, three
/// mantissa bits, `m · 2^-9` at exponent 0) times the tensor's scale, the one
/// rounding of the exact value's product with the scale. A span holding
/// 0x7F or 0xFF (E4M3's NaNs), or a scaled value that is not finite, is
/// refused by name before any value is written.
pub struct E4m3Rows {
    /// Each byte's value times the scale, including non-finite results.
    table: [f32; 256],
}

impl E4m3Rows {
    /// The decode of a tensor whose scale is `scale`. Load-time: the table
    /// is built here, once.
    #[must_use]
    pub fn new(scale: f32) -> E4m3Rows {
        let mut table = [f32::NAN; 256];
        for (byte, v) in (0..=255u8).zip(&mut table) {
            // UE4M3 is E4M3 without its sign, at half the value.
            if let Ok(half) = ue4m3(byte & 0x7F) {
                let twice = if byte & 0x80 == 0 { 2.0 } else { -2.0 };
                *v = twice * half * scale;
            }
        }
        E4m3Rows { table }
    }

    /// Whether every byte in `src` has a finite value at the tensor's scale.
    ///
    /// # Errors
    ///
    /// The first non-finite scaled value, [`QuantError::E4m3Nan`] for a NaN
    /// byte or [`QuantError::E4m3NonFinite`] otherwise, naming its index.
    pub fn check(&self, src: &[u8]) -> Result<(), QuantError> {
        if let Some(at) = src
            .iter()
            .position(|&b| !self.table[usize::from(b)].is_finite())
        {
            let byte = src[at];
            return Err(if byte & 0x7F == 0x7F {
                QuantError::E4m3Nan { at, byte }
            } else {
                QuantError::E4m3NonFinite { at, byte }
            });
        }
        Ok(())
    }

    /// `src`'s bytes decoded into `dst`, one value a byte.
    ///
    /// # Errors
    ///
    /// [`QuantError::ShortSrc`] when `src` holds fewer bytes than `dst`
    /// values. The first byte whose scaled value is not finite is
    /// [`QuantError::E4m3Nan`] for a NaN byte or [`QuantError::E4m3NonFinite`]
    /// otherwise, before any value of the span is written.
    pub fn decode(&self, src: &[u8], dst: &mut [f32]) -> Result<(), QuantError> {
        let src = src.get(..dst.len()).ok_or(QuantError::ShortSrc {
            ty: GgmlType::F8E4M3_ROWS,
            src: src.len(),
            need: dst.len(),
            n: dst.len(),
        })?;
        self.check(src)?;
        for (o, &b) in dst.iter_mut().zip(src) {
            *o = self.table[usize::from(b)];
        }
        Ok(())
    }
}

/// Rows of bloomery's NVFP4 row type ([`GgmlType::NVFP4_ROWS`], 9 bytes a 16 values) decoded: each
/// 16-value group a UE4M3 scale byte then 8 code bytes, value `2i` the low
/// nibble of code byte `i` and `2i + 1` its high nibble, each value
/// `e2m1 · scale` (exact: a 2-bit by a 4-bit significand) times the tensor's
/// scale, that product's one rounding. A scale byte with bit 7 set or 0x7F
/// is no scale: a span holding one or a non-finite scaled value is refused by name.
pub struct Nvfp4Rows {
    /// Each scale byte's half value (the doubled code table puts the half
    /// back); a byte that is no scale is refused before its entry is read.
    halves: [f32; 256],
    scale: f32,
}

impl Nvfp4Rows {
    /// The decode of a tensor whose scale is `scale`. Load-time: the scale
    /// table is built here, once.
    #[must_use]
    pub fn new(scale: f32) -> Nvfp4Rows {
        let mut halves = [f32::NAN; 256];
        for (byte, h) in (0..=255u8).zip(&mut halves) {
            if let Ok(half) = ue4m3(byte) {
                *h = half;
            }
        }
        Nvfp4Rows { halves, scale }
    }

    /// `src`'s groups decoded into `dst`, 16 values a group.
    ///
    /// # Errors
    ///
    /// [`QuantError::UnalignedDst`] when `dst` is not whole groups,
    /// [`QuantError::ShortSrc`] when `src` holds fewer groups, and
    /// [`QuantError::UndefinedScale`] for the first group whose scale byte is
    /// no UE4M3 scale or whose scaled value is not finite (its index the block,
    /// sub-block 0), before any value of the span is written.
    pub fn decode(&self, src: &[u8], dst: &mut [f32]) -> Result<(), QuantError> {
        let ty = GgmlType::NVFP4_ROWS;
        let need = Self::group_count(dst.len())? * 9;
        let src = src.get(..need).ok_or(QuantError::ShortSrc {
            ty,
            src: src.len(),
            need,
            n: dst.len(),
        })?;
        let groups = src.as_chunks::<9>().0.iter().map(|g| {
            let [scale, codes @ ..] = g;
            (*scale, codes)
        });
        self.decode_groups(groups, dst)
    }

    /// Separate code and block-scale planes decoded with the same rounding
    /// and refusals as [`Self::decode`], without packing an intermediate row.
    pub fn decode_planes(
        &self,
        codes: &[u8],
        scales: &[u8],
        dst: &mut [f32],
    ) -> Result<(), QuantError> {
        let groups = Self::plane_groups(codes, scales, dst.len())?;
        self.decode_groups(groups, dst)
    }

    /// [`Self::decode_planes`]'s refusals of a span of `values` values alone,
    /// by the same names, for a caller that decodes the planes elsewhere at
    /// [`Self::scale`].
    pub fn check_planes(
        &self,
        codes: &[u8],
        scales: &[u8],
        values: usize,
    ) -> Result<(), QuantError> {
        self.refusal(Self::plane_groups(codes, scales, values)?)
    }

    /// The tensor scale every value is multiplied by.
    #[must_use]
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// The `(scale byte, codes)` groups of a span of `values` values in
    /// separate planes; a span of partial groups or a short plane refused.
    fn plane_groups<'a>(
        codes: &'a [u8],
        scales: &'a [u8],
        values: usize,
    ) -> Result<impl Iterator<Item = (u8, &'a [u8; 8])> + Clone, QuantError> {
        let groups = Self::group_count(values)?;
        let take = |src: &[u8], need| {
            if src.len() < need {
                Err(QuantError::ShortSrc {
                    ty: GgmlType::NVFP4_ROWS,
                    src: src.len(),
                    need,
                    n: values,
                })
            } else {
                Ok(())
            }
        };
        take(codes, groups * 8)?;
        take(scales, groups)?;
        Ok(codes[..groups * 8]
            .as_chunks::<8>()
            .0
            .iter()
            .zip(&scales[..groups])
            .map(|(codes, &scale)| (scale, codes)))
    }

    fn group_count(values: usize) -> Result<usize, QuantError> {
        if !values.is_multiple_of(16) {
            return Err(QuantError::UnalignedDst {
                ty: GgmlType::NVFP4_ROWS,
                dst: values,
                blck: 16,
            });
        }
        Ok(values / 16)
    }

    /// The first group that is no span of values: its scale byte no UE4M3
    /// scale, or a value times the tensor scale outside f32's range.
    fn refusal<'a>(
        &self,
        groups: impl Iterator<Item = (u8, &'a [u8; 8])>,
    ) -> Result<(), QuantError> {
        let undefined = groups.enumerate().find_map(|(block, (byte, codes))| {
            let half = self.halves[usize::from(byte)];
            let why = if half.is_nan() {
                ue4m3(byte).err()
            } else if (half * 12.0 * self.scale).is_finite() {
                // 12 is the largest doubled code; `half · v` is exact and its
                // rounded product with the scale grows with |v|, so every
                // value of the group is finite.
                None
            } else {
                codes
                    .iter()
                    .flat_map(|&q| fp4_pair(q))
                    .any(|v| !(half * f32::from(v) * self.scale).is_finite())
                    .then_some("a value times the tensor scale leaves f32's range")
            };
            why.map(|why| (block, byte, why))
        });
        if let Some((block, byte, why)) = undefined {
            return Err(QuantError::UndefinedScale {
                ty: GgmlType::NVFP4_ROWS,
                block,
                sub: 0,
                byte,
                why,
            });
        }
        Ok(())
    }

    fn decode_groups<'a>(
        &self,
        groups: impl Iterator<Item = (u8, &'a [u8; 8])> + Clone,
        dst: &mut [f32],
    ) -> Result<(), QuantError> {
        self.refusal(groups.clone())?;
        for ((byte, codes), out) in groups.zip(dst.as_chunks_mut::<16>().0) {
            let half = self.halves[usize::from(byte)];
            for (pair, &q) in out.as_chunks_mut::<2>().0.iter_mut().zip(codes) {
                let [l, h] = fp4_pair(q);
                pair[0] = half * f32::from(l) * self.scale;
                pair[1] = half * f32::from(h) * self.scale;
            }
        }
        Ok(())
    }
}

// ------------------------------------------------------------------- MXFP8

/// Whether bloomery's MXFP8 blocks `src` ([`GgmlType::MXFP8`], whole 33-byte
/// blocks: the block's E8M0 scale byte, then its 32 F8_E4M3 bytes) hold only values: a scale byte 0xFF (E8M0's NaN) is
/// [`QuantError::UndefinedScale`] naming its block, a value byte 0x7F or
/// 0xFF (E4M3's NaN) [`QuantError::E4m3Nan`] naming its index in the span,
/// a scale a value of its block leaves f32's range at (a product of
/// 2^128 or more) [`QuantError::UndefinedScale`] naming its block, and a
/// partial last block [`QuantError::ShortSrc`].
///
/// # Errors
///
/// The first such byte, or the partial block.
pub fn check_mxfp8(src: &[u8]) -> Result<(), QuantError> {
    let ty = GgmlType::MXFP8;
    let (blocks, tail) = src.as_chunks::<33>();
    if !tail.is_empty() {
        return Err(QuantError::ShortSrc {
            ty,
            src: src.len(),
            need: src.len().next_multiple_of(33),
            n: 32 * (blocks.len() + 1),
        });
    }
    for (block, blk) in blocks.iter().enumerate() {
        if blk[0] == 0xFF {
            return Err(QuantError::UndefinedScale {
                ty,
                block,
                sub: 0,
                byte: 0xFF,
                why: "E8M0's NaN",
            });
        }
        if let Some(j) = blk[1..].iter().position(|&b| b & 0x7F == 0x7F) {
            return Err(QuantError::E4m3Nan {
                at: 32 * block + j,
                byte: blk[1 + j],
            });
        }
        // A value of exponent field F >= 1 lies in [2^(F - 7), 2^(F - 6)) and
        // one of field 0 below 2^-6, so times 2^(E - 127) a value reaches
        // 2^128 exactly when E + F >= 262. A quantizer of f32 weights writes
        // E <= 246, which no field reaches.
        let top = blk[1..].iter().map(|&b| (b & 0x7F) >> 3).max().unwrap_or(0);
        if u16::from(blk[0]) + u16::from(top) >= 262 {
            return Err(QuantError::UndefinedScale {
                ty,
                block,
                sub: 0,
                byte: blk[0],
                why: "a value times it leaves f32's range",
            });
        }
    }
    Ok(())
}

/// bloomery's MXFP8 blocks decoded (OCP MX with F8_E4M3 elements): value `j`
/// of a block is its byte `1 + j` as F8_E4M3 ([`E4m3Rows`] at scale 1, exact)
/// times the block's scale `2^(E − 127)` (twice [`e8m0_to_f32_half`]), `E`
/// its byte 0 — one rounding, none at all for a normal product.
/// [`check_mxfp8`]'s refusals come before any value is written; `src` holds
/// `dst`'s blocks exactly ([`dequant_row`]).
fn dequant_mxfp8(src: &[u8], dst: &mut [f32]) -> Result<(), QuantError> {
    check_mxfp8(src)?;
    let e4m3 = E4m3Rows::new(1.0);
    for (blk, out) in src
        .as_chunks::<33>()
        .0
        .iter()
        .zip(dst.as_chunks_mut::<32>().0)
    {
        let d = 2.0 * e8m0_to_f32_half(blk[0]);
        for (o, &b) in out.iter_mut().zip(&blk[1..]) {
            *o = e4m3.table[usize::from(b)] * d;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------- FP8_B128

/// Whether bloomery's FP8 blocks of 128 `src` ([`GgmlType::FP8_B128`], whole
/// 132-byte blocks: the block's f32 scale, little-endian, then its 128
/// F8_E4M3 bytes) hold only values: a scale that is not finite, or one a
/// value of its block times it leaves f32's range at, is
/// [`QuantError::UndefinedF32Scale`] naming its block, a value byte 0x7F or
/// 0xFF (E4M3's NaN) [`QuantError::E4m3Nan`] naming its index in the span,
/// and a partial last block [`QuantError::ShortSrc`].
///
/// # Errors
///
/// The first such block or byte, or the partial block.
pub fn check_fp8_b128(src: &[u8]) -> Result<(), QuantError> {
    let ty = GgmlType::FP8_B128;
    let (blocks, tail) = src.as_chunks::<132>();
    if !tail.is_empty() {
        return Err(QuantError::ShortSrc {
            ty,
            src: src.len(),
            need: src.len().next_multiple_of(132),
            n: 128 * (blocks.len() + 1),
        });
    }
    let e4m3 = E4m3Rows::new(1.0);
    for (block, blk) in blocks.iter().enumerate() {
        let (scale, values) = blk.split_at(4);
        let bits = u32::from_le_bytes([scale[0], scale[1], scale[2], scale[3]]);
        let refuse = |why| QuantError::UndefinedF32Scale {
            ty,
            block,
            bits,
            why,
        };
        let d = f32::from_bits(bits);
        if !d.is_finite() {
            return Err(refuse("not finite"));
        }
        if let Some(j) = values.iter().position(|&b| b & 0x7F == 0x7F) {
            return Err(QuantError::E4m3Nan {
                at: 128 * block + j,
                byte: values[j],
            });
        }
        // The product rounds up in magnitude with the value, so the block's
        // largest value bounds every product's range.
        let top = values.iter().map(|&b| b & 0x7F).max().unwrap_or(0);
        if !(e4m3.table[usize::from(top)] * d).is_finite() {
            return Err(refuse("a value times it leaves f32's range"));
        }
    }
    Ok(())
}

/// bloomery's FP8 blocks of 128 decoded: value `j` of a block is its byte
/// `4 + j` as F8_E4M3 ([`E4m3Rows`] at scale 1, exact) times the block's f32
/// scale, its bytes 0..4, that product's one rounding. A checkpoint's FP8
/// weight with one scale a 128 by 128 block is held as these blocks, each row's
/// 128-value block carrying the scale of the square it lies in.
/// [`check_fp8_b128`]'s refusals come before any value is written; `src`
/// holds `dst`'s blocks exactly ([`dequant_row`]).
fn dequant_fp8_b128(src: &[u8], dst: &mut [f32]) -> Result<(), QuantError> {
    check_fp8_b128(src)?;
    let e4m3 = E4m3Rows::new(1.0);
    for (blk, out) in src
        .as_chunks::<132>()
        .0
        .iter()
        .zip(dst.as_chunks_mut::<128>().0)
    {
        let d = f32::from_le_bytes([blk[0], blk[1], blk[2], blk[3]]);
        for (o, &b) in out.iter_mut().zip(&blk[4..]) {
            *o = e4m3.table[usize::from(b)] * d;
        }
    }
    Ok(())
}

// ----------------------------------------------------------------- MLX affine

/// How an MLX affine type stores the scale and the bias that end each group:
/// both as BF16 bits ([`GgmlType::MLX_A4`], [`GgmlType::MLX_A8`]) or both as
/// F16 bits ([`GgmlType::MLX_A4H`], [`GgmlType::MLX_A8H`]). Either widens to
/// f32 exactly, so one affine rule decodes all four.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AffinePair {
    Bf16,
    F16,
}

impl AffinePair {
    /// The f32 of one stored scale or bias.
    fn widen(self, bits: u16) -> f32 {
        match self {
            AffinePair::Bf16 => bf16_bits_to_f32(bits),
            AffinePair::F16 => half_to_f32(bits),
        }
    }
}

/// Whether MLX's affine blocks `src` hold only values: each group's scale
/// or bias that is not finite, or a scale whose largest product with a code
/// leaves f32's range at, is [`QuantError::UndefinedAffinePair`] naming its
/// block, and a partial last block [`QuantError::ShortSrc`].
/// [`GgmlType::MLX_A4`] and [`GgmlType::MLX_A4H`] blocks are whole 36-byte
/// groups (8 little-endian U32 words of codes, then the scale and bias),
/// [`GgmlType::MLX_A8`] and [`GgmlType::MLX_A8H`] 68-byte (16 words, then the
/// pair), the pair BF16 bits for the first of each and F16 bits for the `H`
/// one; the largest product sits at the group's top code (15, 255), the value
/// a finite bias and scale cannot overflow past — the sum is linear in the
/// code, so an endpoint bounds it.
///
/// # Errors
///
/// The first such block, or the partial block.
pub fn check_mlx_affine(ty: GgmlType, src: &[u8]) -> Result<(), QuantError> {
    match ty {
        GgmlType::MLX_A4 => check_affine_blocks::<36, 15>(ty, AffinePair::Bf16, src),
        GgmlType::MLX_A8 => check_affine_blocks::<68, 255>(ty, AffinePair::Bf16, src),
        GgmlType::MLX_A4H => check_affine_blocks::<36, 15>(ty, AffinePair::F16, src),
        GgmlType::MLX_A8H => check_affine_blocks::<68, 255>(ty, AffinePair::F16, src),
        other => Err(QuantError::Unsupported(other)),
    }
}

fn check_affine_blocks<const N: usize, const TOP: u32>(
    ty: GgmlType,
    pair: AffinePair,
    src: &[u8],
) -> Result<(), QuantError> {
    let (blocks, tail) = src.as_chunks::<N>();
    if !tail.is_empty() {
        return Err(QuantError::ShortSrc {
            ty,
            src: src.len(),
            need: src.len().next_multiple_of(N),
            n: 64 * (blocks.len() + 1),
        });
    }
    for (block, blk) in blocks.iter().enumerate() {
        let scale = u16::from_le_bytes([blk[N - 4], blk[N - 3]]);
        let bias = u16::from_le_bytes([blk[N - 2], blk[N - 1]]);
        let refuse =
            |field: &'static str, bits: u16, why: &'static str| QuantError::UndefinedAffinePair {
                ty,
                block,
                field,
                bits,
                why,
            };
        if !pair.widen(scale).is_finite() {
            return Err(refuse("scale", scale, "not finite"));
        }
        if !pair.widen(bias).is_finite() {
            return Err(refuse("bias", bias, "not finite"));
        }
        let s = pair.widen(scale);
        let b = pair.widen(bias);
        if !(s * TOP as f32 + b).is_finite() {
            return Err(refuse(
                "scale",
                scale,
                "a value times it leaves f32's range",
            ));
        }
    }
    Ok(())
}

/// MLX's affine 4-bit groups of 64 ([`GgmlType::MLX_A4`] and
/// [`GgmlType::MLX_A4H`], 36 bytes: the group's 8 little-endian U32 words of
/// codes, then its scale and bias as `pair` stores them) decoded: value `i`
/// is the 4 bits at bit `4·(i mod 8)` of word `i/8` (low nibble first) as
/// `s·q + b` in f32, `s` and `b` the widened pair. The product of a scale
/// (an 8-bit significand in BF16, 11 in F16) and a 4-bit code is exact in
/// f32, so the sum rounds once; MLX's own `dequantize` returns that value
/// rounded once more to bf16 (measured, mlx 0.32.1: `mx.quantize` then
/// `mx.dequantize`, every bit explained by the single rounding). This f32
/// value is the oracle the Metal kernels gate compares against.
/// [`check_mlx_affine`]'s refusals come before any value is written; `src`
/// holds `dst`'s blocks exactly ([`dequant_row`]).
fn dequant_mlx_a4(
    ty: GgmlType,
    pair: AffinePair,
    src: &[u8],
    dst: &mut [f32],
) -> Result<(), QuantError> {
    check_mlx_affine(ty, src)?;
    for (blk, out) in src
        .as_chunks::<36>()
        .0
        .iter()
        .zip(dst.as_chunks_mut::<64>().0)
    {
        let s = pair.widen(u16::from_le_bytes([blk[32], blk[33]]));
        let b = pair.widen(u16::from_le_bytes([blk[34], blk[35]]));
        for (w, word) in blk[..32].as_chunks::<4>().0.iter().enumerate() {
            let word = u32::from_le_bytes(*word);
            for j in 0..8 {
                out[w * 8 + j] = s * ((word >> (4 * j)) & 0xf) as f32 + b;
            }
        }
    }
    Ok(())
}

/// MLX's affine 8-bit groups of 64 ([`GgmlType::MLX_A8`] and
/// [`GgmlType::MLX_A8H`], 68 bytes: 16 little-endian U32 words of one-byte
/// codes, then the scale and bias as `pair` stores them) decoded: value `i`
/// is byte `i` as `s·q + b` in f32, the same sum [`dequant_mlx_a4`] forms (an
/// 8-bit code still exact in the product, 19 bits at most).
fn dequant_mlx_a8(
    ty: GgmlType,
    pair: AffinePair,
    src: &[u8],
    dst: &mut [f32],
) -> Result<(), QuantError> {
    check_mlx_affine(ty, src)?;
    for (blk, out) in src
        .as_chunks::<68>()
        .0
        .iter()
        .zip(dst.as_chunks_mut::<64>().0)
    {
        let s = pair.widen(u16::from_le_bytes([blk[64], blk[65]]));
        let b = pair.widen(u16::from_le_bytes([blk[66], blk[67]]));
        for (w, word) in blk[..64].as_chunks::<4>().0.iter().enumerate() {
            let word = u32::from_le_bytes(*word);
            for j in 0..4 {
                out[w * 4 + j] = s * ((word >> (8 * j)) & 0xff) as f32 + b;
            }
        }
    }
    Ok(())
}

/// Port of `dequantize_row_nvfp4` (llama.cpp-mainline ggml-quants.c:589-612).
/// Block geometry (block_nvfp4, ggml-common.h:221-227, 36 bytes / 64
/// values): four UE4M3 scales `d[4]` @0, `qs[32]` @4; sub-block `s` holds
/// values `16s..16s+16`, its value `j` (j < 8) the low nibble of
/// `qs[8s + j]` and value `j + 8` the high nibble, each
/// `d * kvalues_fp4[code]` with `d` = [`ue4m3_to_f32_half`]`(d[s])`. The
/// tensor's per-tensor scale (`<name>.scale`, an f32 the graph multiplies
/// the product by) is not part of the block. A 4-bit significand times a
/// code of at most 4 bits, so every product is exact.
fn dequant_nvfp4(src: &[u8], dst: &mut [f32]) -> Result<(), QuantError> {
    for (block, (blk, out)) in src
        .as_chunks::<36>()
        .0
        .iter()
        .zip(dst.as_chunks_mut::<64>().0)
        .enumerate()
    {
        for (sub, (out, qs)) in out
            .as_chunks_mut::<16>()
            .0
            .iter_mut()
            .zip(blk[4..].as_chunks::<8>().0)
            .enumerate()
        {
            let byte = blk[sub];
            let d = ue4m3(byte).map_err(|why| QuantError::UndefinedScale {
                ty: GgmlType::NVFP4,
                block,
                sub,
                byte,
                why,
            })?;
            let (lo, hi) = out.split_at_mut(8);
            for (j, &q) in qs.iter().enumerate() {
                let [l, h] = fp4_pair(q);
                lo[j] = d * f32::from(l);
                hi[j] = d * f32::from(h);
            }
        }
    }
    Ok(())
}

// --------------------------------------------------------------- i-quants

/// `±1.0` for bit `j` of a sign byte, ggml's `signs & kmask_iq2xs[j] ? -1.f : 1.f`.
#[inline]
fn iq_sign(signs: u8, j: usize) -> f32 {
    if signs & KMASK_IQ2XS[j] != 0 {
        -1.0
    } else {
        1.0
    }
}

/// Port of `dequantize_row_iq2_xxs` (ggml-quants.c:3692). Block geometry
/// (block_iq2_xxs, ggml-common.h:439-442, 66 bytes / 256 values): d f16 @0,
/// qs u16[32] @2, read as two u32 words a 32-value sub-block.
///
/// Sub-block ib32's words are `idx` (bytes `2 + 8·ib32 ..`) and `aux` (the
/// next four): byte l of `idx` indexes [`IQ2XXS_GRID`] (eight magnitudes, one
/// per byte) for group l's eight values, bits `7l .. 7l + 7` of `aux` index
/// [`KSIGNS_IQ2XS`] for its signs, and bits 28..32 are the scale `s`; each
/// value is `db·grid[j]·(±1)` with `db = d·(0.5 + s)·0.25`, left to right as
/// the C.
fn dequant_iq2_xxs(src: &[u8], dst: &mut [f32]) {
    for (blk, out) in src
        .as_chunks::<66>()
        .0
        .iter()
        .zip(dst.as_chunks_mut::<256>().0)
    {
        let d = half_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let words = blk[2..66].as_chunks::<8>().0;
        for (w, o32) in words.iter().zip(out.as_chunks_mut::<32>().0) {
            let aux = u32::from_le_bytes([w[4], w[5], w[6], w[7]]);
            let db = d * (0.5 + (aux >> 28) as f32) * 0.25;
            for (l, o8) in o32.as_chunks_mut::<8>().0.iter_mut().enumerate() {
                let grid = IQ2XXS_GRID[usize::from(w[l])].to_le_bytes();
                let signs = KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize];
                for (j, o) in o8.iter_mut().enumerate() {
                    *o = db * f32::from(grid[j]) * iq_sign(signs, j);
                }
            }
        }
    }
}

/// Port of `dequantize_row_iq2_xs` (ggml-quants.c:3720). Block geometry
/// (block_iq2_xs, ggml-common.h:453-458, 74 bytes / 256 values): d f16 @0,
/// qs u16[32] @2, scales u8[8] @66.
///
/// Group g = 4·ib32 + l of eight values (ib32 the 32-value sub-block) reads
/// `qs[g]`: its low 9 bits index [`IQ2XS_GRID`] (eight magnitudes, one per
/// byte), its high 7 bits index [`KSIGNS_IQ2XS`] (eight sign bits, the eighth
/// the parity of the seven stored). The scale is `d·(0.5 + s)·0.25` with `s`
/// the low nibble of `scales[ib32]` for l < 2, the high nibble for l ≥ 2;
/// each value is `db·grid[j]·(±1)`, left to right as the C.
fn dequant_iq2_xs(src: &[u8], dst: &mut [f32]) {
    for (blk, out) in src
        .as_chunks::<74>()
        .0
        .iter()
        .zip(dst.as_chunks_mut::<256>().0)
    {
        let d = half_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let qs = blk[2..66].as_chunks::<2>().0;
        let scales = &blk[66..74];
        for (ib32, (&s, o32)) in scales.iter().zip(out.as_chunks_mut::<32>().0).enumerate() {
            let db = [
                d * (0.5 + f32::from(s & 0x0f)) * 0.25,
                d * (0.5 + f32::from(s >> 4)) * 0.25,
            ];
            for (l, o8) in o32.as_chunks_mut::<8>().0.iter_mut().enumerate() {
                let q = u16::from_le_bytes(qs[4 * ib32 + l]);
                let grid = IQ2XS_GRID[usize::from(q & 511)].to_le_bytes();
                let signs = KSIGNS_IQ2XS[usize::from(q >> 9)];
                for (j, o) in o8.iter_mut().enumerate() {
                    *o = db[l / 2] * f32::from(grid[j]) * iq_sign(signs, j);
                }
            }
        }
    }
}

/// Port of `dequantize_row_iq3_xxs` (ggml-quants.c:3779). Block geometry
/// (block_iq3_xxs, ggml-common.h:488-492, 98 bytes / 256 values): d f16 @0,
/// qs[96] @2 — bytes 0..64 are [`IQ3XXS_GRID`] indices, bytes 64..96 the
/// eight u32 words `scales_and_signs`, one per 32-value sub-block.
///
/// Sub-block ib32's word `aux`: bits 28..32 are the scale `s`, bits 7l..7l+7
/// the [`KSIGNS_IQ2XS`] index of group l. Group l's eight values are grid
/// entries `qs[8·ib32 + 2l]` (values 0..4) and `qs[8·ib32 + 2l + 1]` (4..8),
/// four magnitudes each; value `db·grid[j]·(±1)` with `db = d·(0.5 + s)·0.5`.
fn dequant_iq3_xxs(src: &[u8], dst: &mut [f32]) {
    for (blk, out) in src
        .as_chunks::<98>()
        .0
        .iter()
        .zip(dst.as_chunks_mut::<256>().0)
    {
        let d = half_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let qs = &blk[2..66];
        let words = blk[66..98].as_chunks::<4>().0;
        for (ib32, (w, o32)) in words.iter().zip(out.as_chunks_mut::<32>().0).enumerate() {
            let aux = u32::from_le_bytes(*w);
            let db = d * (0.5 + (aux >> 28) as f32) * 0.5;
            for (l, o8) in o32.as_chunks_mut::<8>().0.iter_mut().enumerate() {
                let signs = KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize];
                let g1 = IQ3XXS_GRID[usize::from(qs[8 * ib32 + 2 * l])].to_le_bytes();
                let g2 = IQ3XXS_GRID[usize::from(qs[8 * ib32 + 2 * l + 1])].to_le_bytes();
                for j in 0..4 {
                    o8[j] = db * f32::from(g1[j]) * iq_sign(signs, j);
                    o8[j + 4] = db * f32::from(g2[j]) * iq_sign(signs, j + 4);
                }
            }
        }
    }
}

/// Port of `dequantize_row_iq3_s` (ggml-quants.c:3811). Block geometry
/// (block_iq3_s, ggml-common.h:503-510, 110 bytes / 256 values): d f16 @0,
/// qs[64] @2, qh[8] @66, signs[32] @74, scales[4] @106.
///
/// Sub-block ib32 (32 values, eight grid words of four magnitudes) reads
/// `qs[8·ib32 + m]` for m in 0..8, extended to a 9-bit [`IQ3S_GRID`] index by
/// bit m of `qh[ib32]` (value 256). Word m holds values 4m..4m+4, and its
/// sign is bit `(4m + j) % 8` of `signs[4·ib32 + m / 2]`, set meaning negative.
/// The scale is nibble `ib32 & 1` of `scales[ib32 / 2]`, low nibble first,
/// and the value `db·grid[j]·(±1)` has `db = d·(1 + 2s)`.
fn dequant_iq3_s(src: &[u8], dst: &mut [f32]) {
    for (blk, out) in src
        .as_chunks::<110>()
        .0
        .iter()
        .zip(dst.as_chunks_mut::<256>().0)
    {
        let d = half_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let qs = &blk[2..66];
        let qh = &blk[66..74];
        let signs = &blk[74..106];
        let scales = &blk[106..110];
        for (ib32, o32) in out.as_chunks_mut::<32>().0.iter_mut().enumerate() {
            let s = (scales[ib32 / 2] >> (4 * (ib32 % 2))) & 0x0f;
            let db = d * (1 + 2 * i32::from(s)) as f32;
            for (m, o4) in o32.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                let hi = usize::from((qh[ib32] >> m) & 1) << 8;
                let g = IQ3S_GRID[usize::from(qs[8 * ib32 + m]) | hi].to_le_bytes();
                let sg = signs[4 * ib32 + m / 2];
                for (j, o) in o4.iter_mut().enumerate() {
                    *o = db * f32::from(g[j]) * iq_sign(sg, 4 * (m % 2) + j);
                }
            }
        }
    }
}

/// Port of `dequantize_row_iq2_s` (ggml-quants.c:3747). Block geometry
/// (block_iq2_s, ggml-common.h:468-473, 82 bytes / 256 values): d f16 @0,
/// qs[64] @2 (32 grid index bytes, then 32 sign bytes), qh[8] @66, scales[8] @74.
///
/// Group `l` (eight values) of sub-block ib32 reads grid index byte `qs[4·ib32 + l]` extended
/// to ten bits by bits `2l .. 2l + 2` of `qh[ib32]` (value 256 and 512) into [`IQ2S_GRID`]
/// (eight magnitudes, one per byte), and its sign byte `qs[32 + 4·ib32 + l]`, bit j set
/// meaning value j negative. The scale is the low nibble of `scales[ib32]` for l < 2 and the
/// high nibble for l >= 2; each value is `db·grid[j]·(±1)` with `db = d·(0.5 + s)·0.25`,
/// left to right as the C.
fn dequant_iq2_s(src: &[u8], dst: &mut [f32]) {
    for (blk, out) in src
        .as_chunks::<82>()
        .0
        .iter()
        .zip(dst.as_chunks_mut::<256>().0)
    {
        let d = half_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let qs = &blk[2..34];
        let signs = &blk[34..66];
        let qh = &blk[66..74];
        let scales = &blk[74..82];
        for (ib32, o32) in out.as_chunks_mut::<32>().0.iter_mut().enumerate() {
            let db = [
                d * (0.5 + f32::from(scales[ib32] & 0x0f)) * 0.25,
                d * (0.5 + f32::from(scales[ib32] >> 4)) * 0.25,
            ];
            for (l, o8) in o32.as_chunks_mut::<8>().0.iter_mut().enumerate() {
                let hi = (usize::from(qh[ib32]) << (8 - 2 * l)) & 0x300;
                let grid = IQ2S_GRID[usize::from(qs[4 * ib32 + l]) | hi].to_le_bytes();
                let sg = signs[4 * ib32 + l];
                for (j, o) in o8.iter_mut().enumerate() {
                    *o = db[l / 2] * f32::from(grid[j]) * iq_sign(sg, j);
                }
            }
        }
    }
}

/// `IQ1S_DELTA` (ggml-common.h:1434): the shift of an IQ1_S / IQ1_M group's grid values.
const IQ1S_DELTA: f32 = 0.125;

/// [`IQ1S_GRID`] entry `index` as the eight signed values (-1, 0 or 1) of a group.
#[inline]
fn iq1s_group(index: usize) -> [i8; 8] {
    IQ1S_GRID[index].to_le_bytes().map(|b| b as i8)
}

/// Port of `dequantize_row_iq1_s` (ggml-quants.c:3854). Block geometry
/// (block_iq1_s, ggml-common.h:521-525, 50 bytes / 256 values): d f16 @0, qs[32] @2,
/// qh u16[8] @34.
///
/// Sub-block ib32 (32 values, four groups of eight) reads `qh[ib32]`, a little-endian u16:
/// group `l`'s [`IQ1S_GRID`] index is `qs[4·ib32 + l]` plus bits `3l .. 3l + 3` of `qh`
/// as its high three bits, bits 12..15 are the scale `s` (`dl = d·(2s + 1)`) and bit 15
/// the sign of the delta. Each value is `dl·(grid[j] + delta)` with
/// `delta = ±IQ1S_DELTA`, left to right as the C.
fn dequant_iq1_s(src: &[u8], dst: &mut [f32]) {
    for (blk, out) in src
        .as_chunks::<50>()
        .0
        .iter()
        .zip(dst.as_chunks_mut::<256>().0)
    {
        let d = half_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let qs = &blk[2..34];
        let qh = blk[34..50].as_chunks::<2>().0;
        for (ib, o32) in out.as_chunks_mut::<32>().0.iter_mut().enumerate() {
            let h = u16::from_le_bytes(qh[ib]);
            let dl = d * (2 * i32::from((h >> 12) & 7) + 1) as f32;
            let delta = if h & 0x8000 != 0 {
                -IQ1S_DELTA
            } else {
                IQ1S_DELTA
            };
            for (l, o8) in o32.as_chunks_mut::<8>().0.iter_mut().enumerate() {
                let index = usize::from(qs[4 * ib + l]) | (usize::from((h >> (3 * l)) & 7) << 8);
                let grid = iq1s_group(index);
                for (j, o) in o8.iter_mut().enumerate() {
                    *o = dl * (f32::from(grid[j]) + delta);
                }
            }
        }
    }
}

/// Port of `dequantize_row_iq1_m` (ggml-quants.c:3879). Block geometry
/// (block_iq1_m, ggml-common.h:535-539, 56 bytes / 256 values): qs[32] @0, qh[16] @32,
/// scales[8] @48, read as four little-endian u16 `sc`.
///
/// The block's f16 scale `d` has no field of its own: its four nibbles are the top nibbles of
/// `sc[0..4]`, low nibble first. Sub-block ib (32 values) has two scales, the 3-bit fields
/// `6·(ib % 2)` and `6·(ib % 2) + 3` of `sc[ib / 2]` (`dl = d·(2s + 1)`, the first for groups
/// 0 and 1, the second for 2 and 3). Group `l` reads grid index byte `qs[4·ib + l]` plus
/// three high bits from `qh[2·ib + l / 2]` (bits 0..3 for even `l`, bits 4..7 for odd `l`),
/// and a delta sign from bit 3 (even `l`) or bit 7 (odd `l`) of that byte. Each value is
/// `dl·(grid[j] + delta)` as in IQ1_S.
fn dequant_iq1_m(src: &[u8], dst: &mut [f32]) {
    for (blk, out) in src
        .as_chunks::<56>()
        .0
        .iter()
        .zip(dst.as_chunks_mut::<256>().0)
    {
        let qs = &blk[0..32];
        let qh = &blk[32..48];
        let sc: [u16; 4] =
            std::array::from_fn(|i| u16::from_le_bytes([blk[48 + 2 * i], blk[49 + 2 * i]]));
        let bits =
            (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
        let d = half_to_f32(bits);
        for (ib, o32) in out.as_chunks_mut::<32>().0.iter_mut().enumerate() {
            let shift = 6 * (ib % 2);
            let dl = [
                d * (2 * i32::from((sc[ib / 2] >> shift) & 7) + 1) as f32,
                d * (2 * i32::from((sc[ib / 2] >> (shift + 3)) & 7) + 1) as f32,
            ];
            for (l, o8) in o32.as_chunks_mut::<8>().0.iter_mut().enumerate() {
                let h = qh[2 * ib + l / 2];
                let (high, flag) = if l % 2 == 0 {
                    (h & 0x07, 0x08)
                } else {
                    ((h >> 4) & 0x07, 0x80)
                };
                let grid = iq1s_group(usize::from(qs[4 * ib + l]) | (usize::from(high) << 8));
                let delta = if h & flag != 0 {
                    -IQ1S_DELTA
                } else {
                    IQ1S_DELTA
                };
                for (j, o) in o8.iter_mut().enumerate() {
                    *o = dl[l / 2] * (f32::from(grid[j]) + delta);
                }
            }
        }
    }
}

/// Port of `dequantize_row_iq4_nl` (ggml-quants.c:3931). Block geometry
/// (block_iq4_nl, ggml-common.h:585-589, 18 bytes / 32 values): d f16 @0,
/// qs[16] @2. Value j < 16 is `d·kvalues_iq4nl[qs[j] & 0xf]` and value j + 16
/// is `d·kvalues_iq4nl[qs[j] >> 4]` ([`KVALUES_IQ4NL`]).
fn dequant_iq4_nl(src: &[u8], dst: &mut [f32]) {
    for (blk, out) in src
        .as_chunks::<18>()
        .0
        .iter()
        .zip(dst.as_chunks_mut::<32>().0)
    {
        let d = half_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let (lo, hi) = out.split_at_mut(16);
        for (j, &byte) in blk[2..18].iter().enumerate() {
            lo[j] = d * f32::from(KVALUES_IQ4NL[usize::from(byte & 0x0f)]);
            hi[j] = d * f32::from(KVALUES_IQ4NL[usize::from(byte >> 4)]);
        }
    }
}

/// Port of `dequantize_row_iq4_xs` (ggml-quants.c:3949). Block geometry
/// (block_iq4_xs, ggml-common.h:602-608, 136 bytes / 256 values): d f16 @0,
/// scales_h u16 @2, scales_l[4] @4, qs[128] @8.
///
/// Sub-block ib (32 values) has the 6-bit scale `ls` = nibble ib%2 of
/// `scales_l[ib/2]` | bits 2ib..2ib+2 of `scales_h` << 4, and `dl = d·(ls − 32)`;
/// value j < 16 is the low nibble of `qs[16·ib + j]`, value j + 16 its high
/// nibble, each `dl·kvalues_iq4nl[code]` ([`KVALUES_IQ4NL`]).
fn dequant_iq4_xs(src: &[u8], dst: &mut [f32]) {
    for (blk, out) in src
        .as_chunks::<136>()
        .0
        .iter()
        .zip(dst.as_chunks_mut::<256>().0)
    {
        let d = half_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let scales_h = u16::from_le_bytes([blk[2], blk[3]]);
        let scales_l = &blk[4..8];
        let qs = blk[8..136].as_chunks::<16>().0;
        for (ib, (q, o32)) in qs.iter().zip(out.as_chunks_mut::<32>().0).enumerate() {
            let ls = i32::from((scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0f)
                | (i32::from((scales_h >> (2 * ib)) & 3) << 4);
            let dl = d * (ls - 32) as f32;
            let (lo, hi) = o32.split_at_mut(16);
            for (j, &byte) in q.iter().enumerate() {
                lo[j] = dl * f32::from(KVALUES_IQ4NL[usize::from(byte & 0x0f)]);
                hi[j] = dl * f32::from(KVALUES_IQ4NL[usize::from(byte >> 4)]);
            }
        }
    }
}

/// Refuse a non-finite activation by quantizer name and block-relative offset.
#[cold]
#[inline(never)]
pub fn non_finite_activation(x: &[f32], quantizer: &str) -> ! {
    let (i, v) = x
        .iter()
        .enumerate()
        .find(|(_, v)| !v.is_finite())
        .expect("the caller already found a non-finite activation");
    panic!(
        "{quantizer}: non-finite activation value {v} at offset {i} of a {}-value block",
        x.len()
    );
}

/// Signed maximum-magnitude nonzero value (first on ties), or positive zero.
/// Refuses non-finite input by quantizer name and block-relative offset.
#[inline]
pub fn activation_max(x: &[f32], quantizer: &str) -> f32 {
    let mut max = 0.0f32;
    let mut amax = 0u32;
    for &v in x {
        // Magnitude bits order NaNs above infinity, so neither can disappear in the max.
        let ax = v.to_bits() & 0x7fff_ffff;
        if ax > amax {
            amax = ax;
            max = v;
        }
    }
    if !max.is_finite() {
        non_finite_activation(x, quantizer);
    }
    max
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
    for (xb, ob) in x
        .as_chunks::<256>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<256>().0)
    {
        let max = activation_max(xb, "quantize_row_q8_k_roundtrip");
        if max == 0.0 {
            ob.fill(0.0);
            continue;
        }
        let iscale = -127.0f32 / max;
        let d = 1.0f32 / iscale;
        for (o, &v) in ob.iter_mut().zip(xb) {
            // Clamped at +127 only: the negative side reaches -127 exactly by construction
            // of iscale.
            *o = d * f32::from(nearest_int(iscale * v).min(127) as i8);
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
/// | Q3_K, IQ3_XXS, IQ4_XS | `Q8_K` |
/// | Q4_K, Q5_K, Q6_K, Q5_0, Q5_1, Q8_0, MXFP4, IQ4_NL | `Q8_2_X4` |
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
///   * rounding is round-half-to-even, the codes saturated to i8, with [`q82_inverse`]'s
///     multiplier.
pub fn quantize_row_q8_2_x4_roundtrip(x: &[f32], out: &mut [f32]) {
    assert_eq!(x.len(), out.len(), "q8_2 round-trip needs matching lengths");
    assert!(x.len().is_multiple_of(32), "q8_2 blocks are 32 values");
    for (xb, ob) in x
        .as_chunks::<32>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<32>().0)
    {
        let amax = activation_max(xb, "quantize_row_q8_2_x4_roundtrip").abs();
        let t = fp32_to_bf16_bits(amax / 127.0);
        let (d, id) = (bf16_bits_to_f32(t), q82_inverse(t));
        for (o, &v) in ob.iter_mut().zip(xb) {
            *o = d * f32::from(nearest_int(v * id).clamp(-128, 127) as i8);
        }
    }
}

/// Round-trip f32 activations through mainline ggml's x86 **Q8_0** activation quantization,
/// the column a Q2_0 weight's dot reads: `quantize_row_q8_0` (llama.cpp `f2cc7282c`
/// ggml-cpu/arch/x86/quants.c:302). Per 32 values `d = f16(amax / 127)`, rounded to nearest
/// even, and each code `round_half_even(v * id)` with `id = 127 / amax`. That is not
/// `quantize_row_q8_0_ref`'s rule, which codes with `1 / d` of the f32 `d = amax / 127`, before
/// its f16 rounding, and `roundf`. A block whose multiplier overflows (amax below about 2^-121)
/// restores to zero ([`q80_inverse`]).
pub fn quantize_row_q8_0_roundtrip(x: &[f32], out: &mut [f32]) {
    assert_eq!(x.len(), out.len(), "q8_0 round-trip needs matching lengths");
    assert!(x.len().is_multiple_of(32), "q8_0 blocks are 32 values");
    let (blocks, _) = x.as_chunks::<32>();
    for (xb, ob) in blocks.iter().zip(out.as_chunks_mut::<32>().0) {
        let block = quantize_q8_0(xb);
        let d = half_to_f32(block.d);
        for (o, &q) in ob.iter_mut().zip(&block.q) {
            *o = d * f32::from(q);
        }
    }
}

/// The activation format ggml quantizes to before a dot with this weight type, as the
/// type-traits table defines it. `Ok(None)` means the activations stay f32.
///
/// The mapping is each type's row ([`crate::QRow::host_act`]) and has no other owner: a caller
/// that picks the activation format itself will pick it differently somewhere else, and the
/// difference shows up as a 1e-3 numeric drift nobody can place.
///
/// MXFP4 takes Q8_2_X4 (ggml.c:1316 on this AVX2 IQK build; ik's
/// `iqk_set_kernels_legacy_quants` pairs it with `mul_mat_qX_1_q8_2_T`) and IQ3_XXS takes
/// Q8_K (ggml.c:1116; `iqk_set_kernels_iquants` refuses any other activation type), as does
/// IQ3_S (ggml.c:1142, the same entry; a single column never takes ik's Q8_K_R8 repack,
/// which `is_dequant_better`, iqk_mul_mat.cpp:252, allows from 32 columns). Q8_0
/// takes Q8_2_X4 too (ggml.c:831 on this AVX2 IQK build; the legacy-quants entry pairs it
/// with `mul_mat_qX_0_q8_0_T<Q8_0_Unpacker, _, block_q8_2>`), the column qdot's Q8_0 kernel
/// reads. IQ4_NL takes Q8_2_X4 on AVX2 (ggml.c:1286-1288; the legacy-quants entry pairs it
/// with `mul_mat_qX_0_q8_0_T<IQ4_NL_UnpackerS, _, block_q8_2>`) and IQ4_XS takes Q8_K
/// (ggml.c:1302; `iqk_set_kernels_kquants` accepts Q8_K only for it).
///
/// A weight type no CPU matmul here takes is refused, not given a format: BF16 activations
/// are BF16 in ggml (`vec_dot_type`, ggml.c:1494), which this engine does not encode, and
/// f32 would be a plausible wrong answer. So is every type [`dequant_row`] does not decode. Q2_K and
/// IQ2_XS take Q8_K in ggml (ggml.c:895, :1105 on the IQK build; `iqk_set_kernels_kquants` and
/// `iqk_set_kernels_iquants` accept Q8_K only for them), and IQ2_XXS too (ggml.c:1064, mainline's
/// ggml-cpu.c:340). IQ2_S, IQ1_S and IQ1_M take Q8_K as well (ik ggml.c:1159-1170, :1185-1196 and
/// :1211-1222, the entries' `vec_dot_type`; mainline's ggml-cpu.c names the same). Q2_0, mainline's alone, takes Q8_0 (`f2cc7282c` ggml-cpu.c:237), the column
/// qdot's Q2_0 kernel reads. NVFP4, mainline's alone, takes Q8_0 there (`ggml_vec_dot_nvfp4_q8_0`) and Q8_2_X4 here, the
/// column qdot's NVFP4 kernel reads, so the `gate-quant` gap between the two activations is part of its band. bloomery's
/// NVFP4 rows take Q8_2_X4 too, the same column.
pub fn activation_format(weight: GgmlType) -> Result<Option<ActivationFormat>, QuantError> {
    match weight.row().map(|r| r.host_act) {
        Some(HostAct::Q8K) => Ok(Some(ActivationFormat::Q8K)),
        Some(HostAct::Q8_2X4) => Ok(Some(ActivationFormat::Q8_2X4)),
        Some(HostAct::Q8_0) => Ok(Some(ActivationFormat::Q8_0)),
        Some(HostAct::F32) => Ok(None),
        Some(HostAct::None) | None => Err(QuantError::NoActivationFormat(weight)),
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ActivationFormat {
    Q8K,
    Q8_2X4,
    Q8_0,
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
        Some(ActivationFormat::Q8_0) => quantize_row_q8_0_roundtrip(x, out),
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

/// ggml's `nearest_int` (ggml-quants.c:1726): round-to-nearest via the
/// 2^23 + 2^22 f32 mantissa trick. Identical to `round_ties_even` on the
/// |v| <= 127 domain the activation codes live in. Every activation encoder
/// rounds through this, the f32 round trips included: without SSE4.1 in the
/// baseline target `round_ties_even` is a libm `rintf` call per element. An
/// infinite or NaN product extracts to -2^22 or 0, whose low byte is 0, so an
/// overflowing Q8_K scale gives zero `as i8` codes, as ggml's do.
#[inline]
pub fn nearest_int(fval: f32) -> i32 {
    let val = fval + 12582912.0;
    let i = f32::to_bits(val);
    ((i & 0x007f_ffff) as i32) - 0x0040_0000
}

/// The q8_2 block's code multiplier from its bf16 scale bits: `1/d`, and 0
/// where `d` is 0 or `1/d` overflows (a block whose scale is at or below
/// about 2^-128 flushes to zero codes and a zero sum). Codes are `nearest_int(v
/// * id)` saturated to i8, the sum over the unsaturated ints, which is ik's
/// x86 rule (`_mm256_cvtps_epi32` then the saturating packs): with a normal
/// `d` the bf16 round-down is at most 2^-8 relative and every code stays in
/// ±127, but a subnormal `d` keeps fewer bits and `v * id` reaches 128,
/// saturated to 127, not wrapped to -128 (`hw_q82_codes_at_bf16_round_down`).
#[inline(always)]
pub fn q82_inverse(t: u16) -> f32 {
    let d = bf16_bits_to_f32(t);
    let id = if d > 0.0 { 1.0 / d } else { 0.0 };
    if id.is_finite() { id } else { 0.0 }
}

/// The q8_0 block's code multiplier from its max |value| `amax`: `127 / amax`,
/// 0 for a zero block, as mainline's x86 `quantize_row_q8_0` (llama.cpp
/// `f2cc7282c` ggml-cpu/arch/x86/quants.c:333) computes it, from `amax` and
/// not from the stored f16 scale. 0 too where `127 / amax` overflows (amax
/// below about 2^-121), where mainline's codes are the integer-indefinite
/// -128 and ours 0. No dot tells them apart: that block's f16 `d` is exactly
/// +0, so `fma(d, s, sumi)` adds a signed zero to `sumi`, whatever `s` is, and
/// `sumi` is never -0 (it starts at +0, and no step rounds to -0), so it keeps
/// its value.
#[inline(always)]
pub fn q80_inverse(amax: f32) -> f32 {
    let id = if amax != 0.0 { 127.0 / amax } else { 0.0 };
    if id.is_finite() { id } else { 0.0 }
}

/// Quantize one 32-value block by ggml's x86 Q8_0 rule, not its weight-file reference rule:
/// `d = f16(amax / 127)`, codes `nearest_int(v * id)` with [`q80_inverse`]'s `id`.
/// Non-finite input is refused by name before a scale or code is written.
#[inline]
#[must_use]
pub fn quantize_q8_0(x: &[f32]) -> Q8Block {
    assert_eq!(x.len(), 32, "quantize_q8_0: one 32-value block at a time");
    let amax = activation_max(x, "quantize_q8_0").abs();
    let id = q80_inverse(amax);
    let mut q = [0i8; 32];
    for (q, &v) in q.iter_mut().zip(x) {
        *q = nearest_int(v * id).clamp(-128, 127) as i8;
    }
    Q8Block {
        d: f32_to_f16_bits(amax / 127.0),
        q,
    }
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

    /// The block's 34 file bytes, as [`Q8Block::from_bytes`] reads them.
    #[must_use]
    pub fn to_bytes(&self) -> [u8; 34] {
        let mut b = [0u8; 34];
        b[..2].copy_from_slice(&self.d.to_le_bytes());
        for (byte, &code) in b[2..].iter_mut().zip(&self.q) {
            *byte = code.to_le_bytes()[0];
        }
        b
    }

    /// ggml's reference weight rule, the one `llama-quantize` and the converters write Q8_0
    /// weights with (`quantize_row_q8_0_ref`, ggml-quants.c): `d = amax / 127`, stored as
    /// f16, and each code `roundf(x · id)` with `id = 1 / d` of the f32 `d`. The activation
    /// rule ([`quantize_row_q8_0_roundtrip`]) and the engine's requant
    /// (`model::quant::quantize_q8_0`) code with `127 / amax` and round half to even, and
    /// differ from it on about one code in a thousand. `None` when a value is not finite, which
    /// the rule would write as a code of 0, or when the scale is past f16's range, which would
    /// store an infinite `d`.
    #[must_use]
    pub fn quantize_ref(x: &[f32; 32]) -> Option<Q8Block> {
        if !x.iter().all(|v| v.is_finite()) {
            return None;
        }
        let amax = x.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        let d = amax / 127.0;
        let id = if d == 0.0 { 0.0 } else { 1.0 / d };
        let stored_d = f32_to_f16_bits(d);
        if !id.is_finite() || !half_to_f32(stored_d).is_finite() {
            return None;
        }
        Some(Q8Block {
            d: stored_d,
            q: x.map(|v| (v * id).round() as i8),
        })
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn activation_roundtrips_refuse_non_finite_blocks() {
        let mut accepted = Vec::new();
        for (weight, n, owner) in [
            (GgmlType::Q2_0, 32, "quantize_q8_0"),
            (GgmlType::Q3_K, 256, "quantize_row_q8_k_roundtrip"),
            (GgmlType::Q4_K, 128, "quantize_row_q8_2_x4_roundtrip"),
        ] {
            for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                for offset in 0..n {
                    let mut x = vec![1.0; n];
                    x[offset] = bad;
                    for x in [x, vec![bad; n]] {
                        let Err(err) = std::panic::catch_unwind(|| {
                            super::quantize_activations(weight, &x, &mut vec![0.0; n]).unwrap();
                        }) else {
                            accepted.push((weight, offset, bad));
                            continue;
                        };
                        let msg = err.downcast_ref::<String>().unwrap();
                        assert!(
                            msg.contains(&format!("{owner}: non-finite activation value")),
                            "{msg}"
                        );
                        assert!(msg.contains("at offset "), "{msg}");
                    }
                }
            }
            let mut out = vec![0.0; n];
            super::quantize_activations(weight, &vec![127.0; n], &mut out).unwrap();
            assert_eq!(out, vec![127.0; n]);
            super::quantize_activations(weight, &vec![0.0; n], &mut out).unwrap();
            assert_eq!(out, vec![0.0; n]);
        }
        assert!(
            accepted.is_empty(),
            "non-finite inputs accepted: {accepted:?}"
        );
        let mut out = [0.0; 32];
        super::quantize_row_q8_0_roundtrip(&[127.0; 32], &mut out);
        assert_eq!(out, [127.0; 32]);
        let mut x = [127.0; 32];
        x[..5].copy_from_slice(&[0.5, 1.5, 2.5, -1.5, -127.0]);
        let block = super::quantize_q8_0(&x);
        assert_eq!(block.d, 0x3c00);
        assert_eq!(&block.q[..5], &[0, 2, 2, -2, -127]);
        for x in [[0.0; 32], [f32::from_bits(1); 32]] {
            assert_eq!(super::quantize_q8_0(&x).to_bytes(), [0; 34]);
        }
        // A multiplier that overflows restores to zero in every format.
        for (weight, n) in [
            (GgmlType::Q2_0, 32),
            (GgmlType::Q3_K, 256),
            (GgmlType::Q4_K, 128),
        ] {
            let x: Vec<f32> = (0..n)
                .map(|i| if i % 2 == 0 { 1e-38 } else { -1e-38 })
                .collect();
            let mut out = vec![1.0; n];
            super::quantize_activations(weight, &x, &mut out).unwrap();
            assert!(out.iter().all(|&v| v == 0.0), "{weight:?}: {out:?}");
        }
        // amax / 127 = 0x00407f00 rounds down to the bf16 scale 2^-127, so amax * id is 127.98:
        // the code saturates at 127 as the q8_2 encoders' does.
        let mut out = [0.0; 32];
        super::quantize_row_q8_2_x4_roundtrip(&[f32::from_bits(0x037f_f808); 32], &mut out);
        assert_eq!(out, [127.0 * f32::from_bits(0x0040_0000); 32]);
    }

    use super::{
        GgmlType, QuantError, activation_format, check_fp8_b128, check_mlx_affine, check_mxfp8,
        dequant_row,
    };
    use crate::qtype::{HostAct, QROWS};

    /// `(id, ggml_type_name, ggml_blck_size, ggml_type_size)` for every id
    /// [`GgmlType`] names, as the compiled libraries answer: ik_llama.cpp
    /// `c10fbbcc` (`build/ggml/src/libggml.so`) and, for tq1_0/tq2_0, which
    /// the ik table does not have, mainline llama.cpp `930e2fa59`
    /// (`build/bin/libggml-base.so`), and for q2_0 mainline `f2cc7282c`. ik's
    /// and mainline's libraries agree on every other row here. nvfp4's row
    /// is read from llama.cpp-mainline's source (ggml.c:760-764,
    /// ggml-common.h:221-227), not a compiled library; so are q1_0's, q8_1's and q8_K's
    /// (llama.cpp `36a73916`: ggml-common.h:180-185, :258-269, :371-376). q8_K is mainline's
    /// 292 B block; ik's is 296 (it adds `float sum`).
    const GGML_TABLE: &[(u32, &str, u64, u64)] = &[
        (0, "f32", 1, 4),
        (1, "f16", 1, 2),
        (2, "q4_0", 32, 18),
        (3, "q4_1", 32, 20),
        (6, "q5_0", 32, 22),
        (7, "q5_1", 32, 24),
        (8, "q8_0", 32, 34),
        (9, "q8_1", 32, 36),
        (10, "q2_K", 256, 84),
        (11, "q3_K", 256, 110),
        (12, "q4_K", 256, 144),
        (13, "q5_K", 256, 176),
        (14, "q6_K", 256, 210),
        (15, "q8_K", 256, 292),
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
        (40, "nvfp4", 64, 36),
        (41, "q1_0", 128, 18),
        (42, "q2_0", 64, 18),
    ];

    /// Ids the enum leaves `Unknown`: types it does not name yet (ik's q8_2_x4 at 99),
    /// slots ggml removed (4, 5, 31-33, 36-38; ik reuses 31-33 and
    /// 36 for repacked and BitNet types mainline does not have), and 43, the
    /// first id past mainline's table.
    // PIN(2026-09-24): 10 (q2_K) left this list when the enum named it for the GLM-5.3 MTP layer.
    // PIN(2026-10-12): 42 (q2_0) left this list when the enum named it.
    // PIN(2026-10-16): 40 (nvfp4) left this list when the enum named it for Nemotron 3.5 Bolt.
    // PIN(2026-10-18): 2 (q4_0) and 3 (q4_1) left this list when the enum named them, with a row each; both
    // are sized and refused by name.
    // PIN(2026-10-19): 9 (q8_1), 15 (q8_K) and 41 (q1_0) left this list when the enum named them, with a row
    // each, so every type in ggml's table has one; all three are sized and refused by name.
    const UNKNOWN_IDS: &[u32] = &[4, 5, 31, 32, 33, 36, 37, 38, 43, 99];

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
            assert!(ty.row().is_none(), "id {id}");
            let shown = crate::ggml_type_info(id)
                .map_or_else(|| format!("ggml-type-{id}"), |(n, _, _)| n.to_owned());
            assert_eq!(ty.to_string(), shown, "id {id}: Display");
        }
    }

    /// The rows are ggml's table and nothing else: as many rows as the table has ids, in tag
    /// order, so each id of the table has its row (`every_named_type_matches_ggml_table` holds
    /// each row's name and sizes to it). A tag the enum does not name has no row, whatever tag an
    /// `Unknown` holds.
    #[test]
    fn the_rows_are_ggmls_table() {
        assert_eq!(
            QROWS.len(),
            GGML_TABLE.len(),
            "a row for each id of the table and no other"
        );
        assert!(
            QROWS
                .windows(2)
                .all(|w| w[0].ty.as_u32() < w[1].ty.as_u32()),
            "rows are in tag order"
        );
        for id in 0..128 {
            let named = GGML_TABLE.iter().any(|t| t.0 == id);
            assert_eq!(GgmlType::from_u32(id).row().is_some(), named, "id {id}");
            assert!(GgmlType::Unknown(id).row().is_none(), "Unknown({id})");
        }
    }

    /// A row is what the dispatchers it feeds do: a type decodes when a CPU matmul takes it, and
    /// besides those only BF16 decodes (named here, not read from the row), and
    /// `dequant_row` decodes one zero block exactly when the row says so.
    #[test]
    fn a_row_is_what_its_dispatchers_do() {
        for row in QROWS {
            let expected = row.host_act != HostAct::None || row.ty == GgmlType::BF16;
            assert_eq!(row.dequant, expected, "{}: decode support", row.name);
            let zero_block = dequant_row(
                row.ty,
                &vec![0u8; row.bytes as usize],
                &mut vec![0.0f32; row.blck as usize],
            );
            assert_eq!(zero_block.is_ok(), expected, "{}: dequant", row.name);
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

    /// The activation format of every type a CPU matmul takes, as ggml's `vec_dot_type` on
    /// this AVX2 IQK build names it (Q2_0's, which ik lacks, as mainline's names it), and BF16
    /// refused by name.
    // PIN(2026-09-27): Q8_0 takes Q8_2_X4, no longer refused — qdot fuses it (q8qdot).
    // PIN(2026-10-11): Q2_K and IQ2_XS take Q8_K, no longer refused — qdot fuses them (qw2h); ggml.c:895
    // and :1105 name Q8_K as their vec_dot_type, and ik's kernel tables accept no other column.
    // PIN(2026-10-12): Q2_0 takes Q8_0, no longer refused: qdot fuses it, and mainline's ggml-cpu.c:237
    // (`f2cc7282c`) names Q8_0 as its vec_dot_type.
    // PIN(2026-10-16): IQ2_XXS takes Q8_K, no longer refused: qdot fuses it, and ggml.c:1064 (mainline's
    // ggml-cpu.c:340) names Q8_K as its vec_dot_type.
    // PIN(2026-10-25): IQ2_S, IQ1_S and IQ1_M take Q8_K, no longer refused: qdot decodes them through its generic
    // host dot, and ik's ggml.c names Q8_K as their vec_dot_type.
    // PIN(2026-10-23): NVFP4 takes Q8_2_X4, no longer refused: qdot fuses it against MXFP4's column. Mainline's
    // vec_dot_type for it is Q8_0; `gate-quant` clause (c) holds the encoder to its own half step and clause (d)
    // adds the gap to ggml's column to the band. NVFP4_ROWS, bloomery's own type with no ggml row, takes the same
    // column.
    #[test]
    fn activation_formats_are_ggmls_vec_dot_types() {
        use super::ActivationFormat::{Q8_0, Q8_2X4, Q8K};
        for (ty, want) in [
            (GgmlType::Q2_K, Some(Q8K)),
            (GgmlType::Q3_K, Some(Q8K)),
            (GgmlType::IQ2_XXS, Some(Q8K)),
            (GgmlType::IQ2_XS, Some(Q8K)),
            (GgmlType::IQ3_XXS, Some(Q8K)),
            (GgmlType::IQ3_S, Some(Q8K)),
            (GgmlType::IQ2_S, Some(Q8K)),
            (GgmlType::IQ1_S, Some(Q8K)),
            (GgmlType::IQ1_M, Some(Q8K)),
            (GgmlType::Q4_K, Some(Q8_2X4)),
            (GgmlType::Q5_K, Some(Q8_2X4)),
            (GgmlType::Q6_K, Some(Q8_2X4)),
            (GgmlType::Q5_0, Some(Q8_2X4)),
            (GgmlType::Q5_1, Some(Q8_2X4)),
            (GgmlType::Q8_0, Some(Q8_2X4)),
            (GgmlType::MXFP4, Some(Q8_2X4)),
            (GgmlType::NVFP4, Some(Q8_2X4)),
            (GgmlType::Q2_0, Some(Q8_0)),
            (GgmlType::IQ4_XS, Some(Q8K)),
            (GgmlType::IQ4_NL, Some(Q8_2X4)),
            (GgmlType::NVFP4_ROWS, Some(Q8_2X4)),
            (GgmlType::F32, None),
            (GgmlType::F16, None),
        ] {
            assert_eq!(activation_format(ty), Ok(want), "{ty}");
        }
        assert_eq!(
            activation_format(GgmlType::BF16),
            Err(QuantError::NoActivationFormat(GgmlType::BF16))
        );
    }

    /// Every product the six i-quant/Q2_K ports form is exact in f32 — the
    /// module doc's claim that the multiply order cannot decide the bits —
    /// checked for every finite f16 scale `d` against every scale factor and
    /// code value the format can hold: the f32 product equals the f64 one.
    #[test]
    fn iq_products_are_exact() {
        use crate::iq_tables::{
            IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID, KVALUES_IQ4NL,
        };
        // IQ2_XXS's and IQ2_S's grid bytes and factor are IQ2_XS's: all go through `g2` and `db2`.
        let mut g2: Vec<u8> = IQ2XS_GRID
            .iter()
            .chain(&IQ2XXS_GRID)
            .chain(&IQ2S_GRID)
            .flat_map(|w| w.to_le_bytes())
            .collect();
        let mut g3: Vec<u8> = IQ3XXS_GRID.iter().flat_map(|w| w.to_le_bytes()).collect();
        let mut g3s: Vec<u8> = IQ3S_GRID.iter().flat_map(|w| w.to_le_bytes()).collect();
        g2.sort_unstable();
        g2.dedup();
        g3.sort_unstable();
        g3.dedup();
        g3s.sort_unstable();
        g3s.dedup();
        let exact = |a: f32, b: f32| f64::from(a * b) == f64::from(a) * f64::from(b);
        for bits in (0u16..0x7c00).chain(0x8000..0xfc00) {
            let d = super::half_to_f32(bits);
            assert!(
                KVALUES_IQ4NL.iter().all(|&k| exact(d, f32::from(k))),
                "iq4_nl {bits:#06x}"
            );
            for s in 0..16u8 {
                let s = f32::from(s);
                let db2 = d * (0.5 + s) * 0.25;
                let db3 = d * (0.5 + s) * 0.5;
                assert!(exact(d, 0.5 + s), "d·(0.5+s) at {bits:#06x}");
                assert!(
                    g2.iter().all(|&g| exact(db2, f32::from(g))),
                    "iq2_xs and iq2_xxs {bits:#06x}"
                );
                assert!(
                    g3.iter().all(|&g| exact(db3, f32::from(g))),
                    "iq3_xxs {bits:#06x}"
                );
                // iq1_s and iq1_m: dl·(grid + delta), grid in -1..=1 and delta ±IQ1S_DELTA, with s the 3-bit scale.
                let dl1 = d * (1 + 2 * (s as i32 & 7)) as f32;
                assert!(
                    exact(d, (1 + 2 * (s as i32 & 7)) as f32),
                    "iq1 d·(2s+1) at {bits:#06x}"
                );
                assert!(
                    [-1.0f32, 0.0, 1.0].iter().all(|&g| {
                        [-super::IQ1S_DELTA, super::IQ1S_DELTA]
                            .iter()
                            .all(|&delta| exact(dl1, g + delta))
                    }),
                    "iq1 {bits:#06x}"
                );
                let db3s = d * (1 + 2 * s as i32) as f32;
                assert!(
                    exact(d, (1 + 2 * s as i32) as f32),
                    "d·(1+2s) at {bits:#06x}"
                );
                assert!(
                    g3s.iter().all(|&g| exact(db3s, f32::from(g))),
                    "iq3_s {bits:#06x}"
                );
                // q2_K: d·sc and min·m, then dl·q for the 2-bit codes.
                assert!(exact(d, s));
                assert!(
                    (0..4u8).all(|q| exact(d * s, f32::from(q))),
                    "q2_K {bits:#06x}"
                );
            }
            for ls in -32..32i32 {
                let dl = d * ls as f32;
                assert!(exact(d, ls as f32));
                assert!(
                    KVALUES_IQ4NL.iter().all(|&k| exact(dl, f32::from(k))),
                    "iq4_xs {bits:#06x} ls {ls}"
                );
            }
        }
    }

    /// The codebooks' structure, as ggml builds them: every IQ2_XS,
    /// IQ2_XXS and IQ2_S grid byte is one of 8, 25, 43; every IQ1_S grid byte is 0, 1 or 255 (-1); every IQ3_XXS grid byte is 4 + 8k or 62; every
    /// IQ3_S grid byte is odd and at most 15; a `ksigns_iq2xs` entry is its index with bit 7 set to the parity of the
    /// seven, so the eight signs of a group always flip an even number.
    #[test]
    fn iq_tables_have_ggml_structure() {
        use crate::iq_tables::{
            IQ1S_GRID, IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID, KMASK_IQ2XS,
            KSIGNS_IQ2XS,
        };
        for &w in IQ2XS_GRID.iter().chain(&IQ2XXS_GRID).chain(&IQ2S_GRID) {
            assert!(
                w.to_le_bytes().iter().all(|b| [8, 25, 43].contains(b)),
                "{w:#x}"
            );
        }
        for w in IQ1S_GRID {
            assert!(
                w.to_le_bytes().iter().all(|b| [0, 1, 255].contains(b)),
                "{w:#x}"
            );
        }
        for w in IQ3XXS_GRID {
            assert!(
                w.to_le_bytes()
                    .iter()
                    .all(|&b| b == 62 || (b >= 4 && b % 8 == 4 && b <= 52)),
                "{w:#x}"
            );
        }
        for w in IQ3S_GRID {
            assert!(
                w.to_le_bytes().iter().all(|&b| b % 2 == 1 && b <= 15),
                "{w:#x}"
            );
        }
        for (i, &s) in KSIGNS_IQ2XS.iter().enumerate() {
            let i = i as u8;
            assert_eq!(s & 0x7f, i);
            assert_eq!(s >> 7, (i.count_ones() & 1) as u8, "entry {i}");
        }
        assert_eq!(KMASK_IQ2XS, core::array::from_fn(|j| 1u8 << j));
    }

    /// IQ4_NL over its block layout: every code in both nibbles of each byte
    /// position, at scales that are a normal, a subnormal, a negative and a
    /// zero f16, decodes to `d·kvalues_iq4nl[code]` computed in f64 (exact,
    /// [`iq_products_are_exact`]), low nibble of byte j to value j and high
    /// nibble to value j + 16. A 160-value row (a PLE table row, five blocks,
    /// 90 bytes) decodes block by block; one byte short, or a row that is not
    /// whole blocks, is refused by name.
    #[test]
    fn iq4_nl_decodes_every_code() {
        use crate::iq_tables::KVALUES_IQ4NL;
        let scales: [u16; 5] = [0x3c00, 0x0001, 0xb555, 0x0000, 0x7bff];
        let mut src = Vec::new();
        for (i, &d) in scales.iter().enumerate() {
            src.extend(d.to_le_bytes());
            // Byte j holds code (j + i) mod 16 low and (15 − j + 2i) mod 16 high.
            src.extend(
                (0..16u8).map(|j| ((j + i as u8) & 15) | (((15 - j + 2 * i as u8) & 15) << 4)),
            );
        }
        assert_eq!(src.len(), 90);
        let mut dst = vec![0.0f32; 160];
        dequant_row(GgmlType::IQ4_NL, &src, &mut dst).unwrap();
        for (i, (&d, out)) in scales.iter().zip(dst.as_chunks::<32>().0).enumerate() {
            let d = f64::from(super::half_to_f32(d));
            for j in 0..16usize {
                let lo = (j + i) & 15;
                let hi = (15 - j + 2 * i) & 15;
                for (at, code) in [(j, lo), (j + 16, hi)] {
                    let want = (d * f64::from(KVALUES_IQ4NL[code])) as f32;
                    assert_eq!(
                        out[at].to_bits(),
                        want.to_bits(),
                        "block {i} value {at} (code {code}): got {:e}, want {want:e}",
                        out[at]
                    );
                }
            }
        }
        assert_eq!(
            dequant_row(GgmlType::IQ4_NL, &src[..89], &mut dst),
            Err(QuantError::ShortSrc {
                ty: GgmlType::IQ4_NL,
                src: 89,
                need: 90,
                n: 160
            })
        );
        assert_eq!(
            dequant_row(GgmlType::IQ4_NL, &src, &mut dst[..150]),
            Err(QuantError::UnalignedDst {
                ty: GgmlType::IQ4_NL,
                dst: 150,
                blck: 32
            })
        );
    }

    /// `dequantize_row_q2_0` of mainline llama.cpp `f2cc7282c` over the blocks
    /// of [`q2_0_matches_mainline`] at d = 1.0 (f16 `0x3c00`), printed through
    /// `ggml_get_type_traits(GGML_TYPE_Q2_0)->to_float` on x86_64 Linux; one
    /// line per 16 values, four lines per block.
    #[rustfmt::skip]
    const Q2_0_MAINLINE: [i8; 256] = [
        -1, -1,  2,  2,  0, -1,  1,  2,  1, -1,  0,  2,  2, -1, -1,  2,
        -1,  0,  2,  1,  0,  0,  1,  1,  1,  0,  0,  1,  2,  0, -1,  1,
        -1,  1,  2,  0,  0,  1,  1,  0,  1,  1,  0,  0,  2,  1, -1,  0,
        -1,  2,  2, -1,  0,  2,  1, -1,  1,  2,  0, -1,  2,  2, -1, -1,
         0,  0,  1,  1, -1,  0,  2,  1,  2,  0, -1,  1,  1,  0,  0,  1,
         0, -1,  1,  2, -1, -1,  2,  2,  2, -1, -1,  2,  1, -1,  0,  2,
         0,  2,  1, -1, -1,  2,  2, -1,  2,  2, -1, -1,  1,  2,  0, -1,
         0,  1,  1,  0, -1,  1,  2,  0,  2,  1, -1,  0,  1,  1,  0,  0,
         1,  1,  0,  0,  2,  1, -1,  0, -1,  1,  2,  0,  0,  1,  1,  0,
         1,  2,  0, -1,  2,  2, -1, -1, -1,  2,  2, -1,  0,  2,  1, -1,
         1, -1,  0,  2,  2, -1, -1,  2, -1, -1,  2,  2,  0, -1,  1,  2,
         1,  0,  0,  1,  2,  0, -1,  1, -1,  0,  2,  1,  0,  0,  1,  1,
         2,  2, -1, -1,  1,  2,  0, -1,  0,  2,  1, -1, -1,  2,  2, -1,
         2,  1, -1,  0,  1,  1,  0,  0,  0,  1,  1,  0, -1,  1,  2,  0,
         2,  0, -1,  1,  1,  0,  0,  1,  0,  0,  1,  1, -1,  0,  2,  1,
         2, -1, -1,  2,  1, -1,  0,  2,  0, -1,  1,  2, -1, -1,  2,  2,
    ];

    /// Q2_0 against mainline ggml's decode: four blocks whose code byte j is
    /// `(j | (15 − j) << 4) ^ 0x55·i`, so every position holds each of the four
    /// codes once across the blocks and no two bytes of a block are equal. At
    /// d = 1.0 the row is [`Q2_0_MAINLINE`]. Each pass then gives the four
    /// blocks four of the scales below (subnormal, negative, zero, negative-zero
    /// and largest finite among them), rotated a step a pass, and every value is
    /// that one times its block's d, computed in f64 (exact, the module doc),
    /// down to the sign of a zero. A row one byte short, or not whole blocks, is
    /// refused by name.
    #[test]
    fn q2_0_matches_mainline() {
        let blocks = |d: [u16; 4]| -> Vec<u8> {
            (0..4u8)
                .flat_map(|i| {
                    let codes = (0..16u8).map(move |j| (j | ((15 - j) << 4)) ^ (0x55 * i));
                    d[usize::from(i)].to_le_bytes().into_iter().chain(codes)
                })
                .collect()
        };
        let scales = [
            0x3c00u16, 0x0001, 0x03ff, 0x0400, 0xb555, 0x0000, 0x8000, 0x7bff, 0xfbff,
        ];
        let mut dst = [0.0f32; 256];
        for s in 0..scales.len() {
            let ds: [u16; 4] = std::array::from_fn(|i| scales[(s + i) % scales.len()]);
            dequant_row(GgmlType::Q2_0, &blocks(ds), &mut dst).unwrap();
            for (p, (&got, &l)) in dst.iter().zip(&Q2_0_MAINLINE).enumerate() {
                let d = ds[p / 64];
                let want = (f64::from(l) * f64::from(super::half_to_f32(d))) as f32;
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "d {d:#06x} value {p}: got {got:e}, want {want:e}"
                );
            }
        }
        let src = blocks([0x3c00; 4]);
        assert_eq!(
            dequant_row(GgmlType::Q2_0, &src[..71], &mut dst),
            Err(QuantError::ShortSrc {
                ty: GgmlType::Q2_0,
                src: 71,
                need: 72,
                n: 256
            })
        );
        assert_eq!(
            dequant_row(GgmlType::Q2_0, &src, &mut dst[..96]),
            Err(QuantError::UnalignedDst {
                ty: GgmlType::Q2_0,
                dst: 96,
                blck: 64
            })
        );
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

    /// An unsigned E4M3 scale byte's value from the format's definition, not
    /// from ggml's decode: `m · 2^-9` at exponent 0, `(1 + m/8) · 2^(e-7)`
    /// above.
    fn ue4m3(byte: u8) -> f64 {
        let (e, m) = (i32::from(byte >> 3), f64::from(byte & 7));
        if e == 0 {
            m * 2f64.powi(-9)
        } else {
            (1.0 + m / 8.0) * 2f64.powi(e - 7)
        }
    }

    /// One NVFP4 block built by hand: scales 1, 2, 2^-9 and 448 (bytes 0x38,
    /// 0x40, 0x01, 0x7E), every code byte 0x71 (low nibble code 1 = 0.5, high
    /// nibble code 7 = 6). Each sub-block is eight 0.5s then eight 6s at its
    /// scale. A swapped nibble order or a scale not halved against the
    /// doubled code table fails it.
    #[test]
    fn nvfp4_decodes_a_hand_built_block() {
        let mut src = vec![0x38, 0x40, 0x01, 0x7E];
        src.extend([0x71u8; 32]);
        let mut dst = [0.0f32; 64];
        dequant_row(GgmlType::NVFP4, &src, &mut dst).unwrap();
        let tiny = 2f32.powi(-9);
        for (s, scale) in [1.0, 2.0, tiny, 448.0].into_iter().enumerate() {
            let sub = &dst[16 * s..16 * s + 16];
            assert!(sub[..8].iter().all(|&v| v == 0.5 * scale), "{s}: {sub:?}");
            assert!(sub[8..].iter().all(|&v| v == 6.0 * scale), "{s}: {sub:?}");
        }
        assert_eq!((dst[0], dst[15], dst[63]), (0.5, 6.0, 2688.0));
    }

    /// NVFP4 over its defined domain: 32 blocks whose 128 sub-block scales
    /// run through every defined byte 0x00..=0x7E, each sub-block holding
    /// every code once (low nibble j + sub, high nibble 15 − j + sub, mod 16,
    /// so the four sub-blocks' bytes differ and a sub-block read at another
    /// offset fails), decode to `e2m1(code) · ue4m3(scale)` computed in f64
    /// from the formats' definitions. Every such product is an f32, so the
    /// comparison is on bits.
    #[test]
    fn nvfp4_decodes_every_defined_scale_and_code() {
        let scales: Vec<u8> = (0..128u8).map(|i| i % 127).collect();
        let mut src = Vec::with_capacity(32 * 36);
        for d in scales.chunks(4) {
            src.extend_from_slice(d);
            for sub in 0..4u8 {
                src.extend((0..8u8).map(|j| ((j + sub) & 15) | (((15 - j + sub) & 15) << 4)));
            }
        }
        let mut dst = vec![0.0f32; 32 * 64];
        dequant_row(GgmlType::NVFP4, &src, &mut dst).unwrap();
        for (s, out) in dst.as_chunks::<16>().0.iter().enumerate() {
            let scale = ue4m3(scales[s]);
            for (i, &got) in out.iter().enumerate() {
                let base = if i < 8 { i } else { 15 - (i - 8) };
                let code = ((base + s % 4) & 15) as u8;
                let want = (e2m1(code) * scale) as f32;
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "scale {:#04x} value {i} (code {code}): got {got:e}, want {want:e}",
                    scales[s]
                );
            }
        }
    }

    /// The public decode refuses a byte that is no scale with the crate's error, naming it.
    #[test]
    fn ue4m3_to_f32_half_names_a_byte_that_is_no_scale() {
        assert_eq!(super::ue4m3_to_f32_half(0x38), Ok(0.5));
        for (byte, why) in [
            (0x7Fu8, "E4M3's NaN"),
            (0x80, "bit 7 set: UE4M3 has no sign bit"),
        ] {
            assert_eq!(
                super::ue4m3_to_f32_half(byte),
                Err(QuantError::NotUe4m3 { byte, why })
            );
        }
    }

    /// A scale byte UE4M3 does not define — 0x7F, E4M3's NaN, or any byte
    /// with bit 7 set — is refused naming the block, the sub-block and the
    /// byte. Decoding it as mainline does (0x7F to 0, bit 7 dropped) passes
    /// this test's decode, and fails it.
    #[test]
    fn nvfp4_refuses_an_undefined_scale_by_name() {
        for (byte, why) in [
            (0x7Fu8, "NaN"),
            (0x80, "no sign bit"),
            (0xB8, "no sign bit"),
            (0xFF, "no sign bit"),
        ] {
            let mut src = vec![0x38u8; 72];
            src[36 + 2] = byte;
            let mut dst = [0.0f32; 128];
            let e = dequant_row(GgmlType::NVFP4, &src, &mut dst).expect_err("undefined");
            assert_eq!(
                e,
                QuantError::UndefinedScale {
                    ty: GgmlType::NVFP4,
                    block: 1,
                    sub: 2,
                    byte,
                    why: super::ue4m3(byte).unwrap_err(),
                }
            );
            let text = e.to_string();
            assert!(
                text.contains("nvfp4")
                    && text.contains("block 1 sub-block 2")
                    && text.contains(why),
                "{text}"
            );
        }
    }

    /// Signed E4M3 reference bits from the existing f64 oracle, grouped by scale.
    fn e4m3_reference() -> Vec<(u32, u8, Option<u32>)> {
        [0x3F80_0000u32, 0x3951_0000]
            .into_iter()
            .flat_map(|scale| {
                let factor = f64::from(f32::from_bits(scale));
                (0..=u8::MAX).map(move |byte| {
                    let magnitude = byte & 0x7F;
                    let value = (magnitude != 0x7F).then(|| {
                        let sign = if byte & 0x80 == 0 { 1.0 } else { -1.0 };
                        ((sign * ue4m3(magnitude) * factor) as f32).to_bits()
                    });
                    (scale, byte, value)
                })
            })
            .collect()
    }

    /// bloomery's F8 rows against the reference over all 256 bytes at scale
    /// 1 and at the checkpoint's PLE scale: every byte that holds a value
    /// decodes to the reference's bits (the sign of zero included); 0x7F and
    /// 0xFF are refused naming the byte and its index, and a short span by
    /// its lengths. A wrong exponent bias, a subnormal read as normal, a
    /// dropped sign or a scale applied twice fails it.
    #[test]
    fn e4m3_rows_decode_every_byte_to_the_reference() {
        let reference = e4m3_reference();
        assert_eq!(reference.len(), 512);
        for scale in [0x3F80_0000u32, 0x3951_0000] {
            let rows = super::E4m3Rows::new(f32::from_bits(scale));
            for &(_, byte, want) in reference.iter().filter(|r| r.0 == scale) {
                let mut two = [0.0f32; 2];
                match (want, rows.decode(&[0x38, byte], &mut two)) {
                    (Some(bits), Ok(())) => assert_eq!(
                        two[1].to_bits(),
                        bits,
                        "scale {scale:08x} byte {byte:#04x}: got {:e}",
                        two[1]
                    ),
                    (None, Err(e)) => {
                        assert_eq!(e, QuantError::E4m3Nan { at: 1, byte });
                        assert!(e.to_string().contains("value 1 is byte"), "{e}");
                        assert_eq!(two, [0.0; 2], "NaN refusal must leave dst unchanged");
                    }
                    (want, got) => panic!("byte {byte:#04x}: want {want:?}, got {got:?}"),
                }
            }
        }
        assert!(matches!(
            super::E4m3Rows::new(1.0).decode(&[0x38], &mut [0.0; 2]),
            Err(QuantError::ShortSrc {
                src: 1,
                need: 2,
                ..
            })
        ));
    }

    #[test]
    fn e4m3_rows_refuse_nonfinite_products_before_writing() {
        for scale in [f32::MAX, -f32::MAX, f32::MAX / 2.0, -f32::MAX / 2.0] {
            let rows = super::E4m3Rows::new(scale);
            for byte in [0x40u8, 0xC0, 0x7E, 0xFE, 0x38, 0xB8, 0x00, 0x80] {
                let sign = if byte & 0x80 == 0 { 1.0 } else { -1.0 };
                let want = (sign * ue4m3(byte & 0x7F) * f64::from(scale)) as f32;
                let mut dst = [f32::from_bits(0x7FC0_1234), -0.0];
                let before = dst.map(f32::to_bits);
                let got = rows.decode(&[0x38, byte, 0x7F], &mut dst);
                if want.is_finite() {
                    got.unwrap();
                    assert_eq!(dst.map(f32::to_bits), [scale.to_bits(), want.to_bits()]);
                } else {
                    let error = got.expect_err("a scaled value outside f32 must be refused");
                    let text = error.to_string();
                    assert!(
                        text.contains("value 1") && text.contains("not finite"),
                        "{text}"
                    );
                    assert!(text.contains(&format!("{byte:#04x}")), "{text}");
                    assert_eq!(dst.map(f32::to_bits), before);
                }
            }
        }
        for scale in [f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
            let mut dst = [42.0, -0.0];
            let before = dst.map(f32::to_bits);
            let error = super::E4m3Rows::new(scale)
                .decode(&[0x38, 0x00], &mut dst)
                .expect_err("a non-finite scale must be refused");
            let text = error.to_string();
            assert!(
                text.contains("value 0") && text.contains("not finite"),
                "{text}"
            );
            assert_eq!(dst.map(f32::to_bits), before);
        }
    }

    /// A split holding a PLE table in bloomery's F8 rows and its `.scale`,
    /// opened through `Split::open`: each row, decoded at the scale read from
    /// the file, is the reference's (every byte that holds a value appears).
    #[test]
    fn a_split_in_f8_rows_decodes_to_the_reference() {
        use crate::write::{TensorDecl, write_file};
        const SCALE: u32 = 0x3951_0000;
        let want: Vec<(u8, u32)> = e4m3_reference()
            .into_iter()
            .filter_map(|(s, b, v)| (s == SCALE).then_some((b, v?)))
            .collect();
        let table: Vec<u8> = want.iter().map(|w| w.0).cycle().take(2 * 160).collect();
        let decl = |name: &str, dims: Vec<u64>, type_id, nbytes| TensorDecl {
            name: name.to_string(),
            dims,
            type_id,
            nbytes,
        };
        let path = std::env::temp_dir().join(format!("gguf-f8-rows-{}.gguf", std::process::id()));
        write_file(
            &path,
            &[],
            &[
                (
                    decl(
                        "t.weight",
                        vec![160, 2],
                        GgmlType::F8E4M3_ROWS.as_u32(),
                        320,
                    ),
                    table,
                ),
                (decl("t.scale", vec![1], 0, 4), SCALE.to_le_bytes().to_vec()),
            ],
        );
        let split = crate::Split::open(&path);
        std::fs::remove_file(&path).unwrap();
        let split = split.unwrap();
        let data = |name: &str| {
            let (s, t) = split.find(name).unwrap();
            split.shard(s).unwrap().data(t).unwrap()
        };
        let scale = f32::from_le_bytes(data("t.scale").try_into().unwrap());
        let rows = super::E4m3Rows::new(scale);
        for (r, src) in data("t.weight").as_chunks::<160>().0.iter().enumerate() {
            let mut got = [0.0f32; 160];
            rows.decode(src, &mut got).unwrap();
            for (i, (g, &b)) in got.iter().zip(src).enumerate() {
                let bits = want.iter().find(|w| w.0 == b).unwrap().1;
                assert_eq!(g.to_bits(), bits, "row {r} value {i} byte {b:#04x}");
            }
        }
    }

    /// bloomery's MXFP8 blocks read through a split (its row sizes the type
    /// at 33 bytes a 32 values) decode to the f64 reference, the E4M3
    /// value times `2^(E − 127)` rounded once to f32, for every scale byte
    /// below 0xFF (the subnormal `2^-127` at 0 and the subnormal products
    /// among them) and every value byte that holds a value: bit for bit, the
    /// sign of zero included, up to the largest finite product (1.875 · 2^127).
    /// A NaN scale byte, a NaN value byte, a scale a value of its block
    /// leaves f32's range at, a partial block and a short source are refused
    /// by name, and a refusal writes no value. The host tier's matmul has no
    /// MXFP8 kernel and refuses the type by name.
    #[test]
    fn mxfp8_blocks_decode_to_the_reference() {
        use crate::write::{TensorDecl, write_file};
        let ty = GgmlType::MXFP8;
        let values: Vec<u8> = (0..=u8::MAX).filter(|b| b & 0x7F != 0x7F).collect();
        // One block a scale byte 0..=254: 255 blocks, 15 a row, 17 rows. A
        // block holds the values whose product with its scale is a finite f32.
        let mut blocks = Vec::new();
        for e in 0..=254u8 {
            let scale = 2f64.powi(i32::from(e) - 127);
            let held: Vec<u8> = values
                .iter()
                .copied()
                .filter(|b| ue4m3(b & 0x7F) * scale <= f64::from(f32::MAX))
                .collect();
            blocks.push(e);
            for j in 0..32 {
                blocks.push(held[(32 * usize::from(e) + j) % held.len()]);
            }
        }
        let path = std::env::temp_dir().join(format!("gguf-mxfp8-{}.gguf", std::process::id()));
        let decl = TensorDecl {
            name: "t.weight".to_string(),
            dims: vec![480, 17],
            type_id: GgmlType::MXFP8.as_u32(),
            nbytes: blocks.len() as u64,
        };
        write_file(&path, &[], &[(decl, blocks.clone())]);
        let split = crate::Split::open(&path);
        std::fs::remove_file(&path).unwrap();
        let split = split.unwrap();
        let (s, t) = split.find("t.weight").unwrap();
        assert_eq!((t.ty, t.nbytes), (ty, 255 * 33));
        assert_eq!((ty.blck_size(), ty.type_size()), (Some(32), Some(33)));
        let data = split.shard(s).unwrap().data(t).unwrap();
        let mut got = vec![0.0f32; 255 * 32];
        dequant_row(ty, data, &mut got).unwrap();
        for (e, (blk, out)) in blocks
            .as_chunks::<33>()
            .0
            .iter()
            .zip(got.chunks(32))
            .enumerate()
        {
            let scale = 2f64.powi(e as i32 - 127);
            for (j, (&b, g)) in blk[1..].iter().zip(out).enumerate() {
                let sign = if b & 0x80 == 0 { 1.0 } else { -1.0 };
                let want = (sign * ue4m3(b & 0x7F) * scale) as f32;
                assert_eq!(g.to_bits(), want.to_bits(), "E {e} value {j} byte {b:#04x}");
            }
        }
        let refused = |src: &[u8], n: usize| {
            let mut dst = vec![0.0f32; n];
            let e = dequant_row(ty, src, &mut dst).expect_err("refused");
            assert!(dst.iter().all(|v| *v == 0.0), "{e}: a value was written");
            e
        };
        let mut nan_scale = blocks[..66].to_vec();
        nan_scale[33] = 0xFF;
        let e = refused(&nan_scale, 64);
        assert_eq!(
            e,
            QuantError::UndefinedScale {
                ty,
                block: 1,
                sub: 0,
                byte: 0xFF,
                why: "E8M0's NaN"
            }
        );
        assert!(e.to_string().contains("block 1"), "{e}");
        for byte in [0x7Fu8, 0xFF] {
            let mut nan_value = blocks[..66].to_vec();
            nan_value[33 + 1 + 5] = byte;
            assert_eq!(
                refused(&nan_value, 64),
                QuantError::E4m3Nan { at: 37, byte }
            );
        }
        // At E = 254 byte 0x3F (1.875) is the largest finite value and 0x40
        // (2) reaches 2^128; at E = 247 byte 0xF8 (-256) does.
        let mut edge = vec![254u8];
        edge.extend([0x3F, 0xBF].repeat(16));
        let mut out = [0.0f32; 32];
        dequant_row(ty, &edge, &mut out).unwrap();
        assert!(out.iter().all(|v| v.abs() == 1.875 * 2f32.powi(127)));
        for (scale, byte) in [(254u8, 0x40u8), (247, 0xF8)] {
            let mut big = blocks[..66].to_vec();
            big[33] = scale;
            big[33 + 1 + 9] = byte;
            assert_eq!(
                refused(&big, 64),
                QuantError::UndefinedScale {
                    ty,
                    block: 1,
                    sub: 0,
                    byte: scale,
                    why: "a value times it leaves f32's range"
                }
            );
        }
        assert!(matches!(
            check_mxfp8(&blocks[..40]),
            Err(QuantError::ShortSrc {
                src: 40,
                need: 66,
                n: 64,
                ..
            })
        ));
        assert!(matches!(
            refused(&blocks[..40], 64),
            QuantError::ShortSrc {
                src: 40,
                need: 66,
                ..
            }
        ));
        assert!(matches!(
            refused(&blocks[..66], 48),
            QuantError::UnalignedDst {
                dst: 48,
                blck: 32,
                ..
            }
        ));
        assert_eq!(
            activation_format(ty),
            Err(QuantError::NoActivationFormat(ty))
        );
    }

    /// bloomery's FP8 blocks of 128 read through a split (its row sizes the
    /// type at 132 bytes a 128 values) decode to the f64 reference, the E4M3
    /// value times the block's f32 scale rounded once to f32, bit for bit, the
    /// sign of zero included: every value byte that holds a value, at scales
    /// that are a BF16 widened (a checkpoint's `weight_scale_inv`), an f32 with
    /// a full significand, a negative one, zero, a subnormal and one at which
    /// 448, the largest value, stays finite. A scale that is NaN or infinite, a value byte
    /// that is E4M3's NaN, a scale a value of its block leaves f32's range
    /// at, a partial block and a short source are refused by name, and a
    /// refusal writes no value. The host tier's matmul has no FP8_B128
    /// kernel and refuses the type by name.
    #[test]
    fn fp8_b128_blocks_decode_to_the_reference() {
        use crate::write::{TensorDecl, write_file};
        let ty = GgmlType::FP8_B128;
        let values: Vec<u8> = (0..=u8::MAX).filter(|b| b & 0x7F != 0x7F).collect();
        let scales = [
            f32::from_bits(0x3951 << 16),
            1.0 / 3.0,
            -0.007_812_5,
            0.0,
            f32::from_bits(1),
            f32::MAX / 512.0,
        ];
        let mut blocks = Vec::new();
        for (i, d) in scales.iter().enumerate() {
            for half in 0..2 {
                blocks.extend(d.to_le_bytes());
                blocks.extend((0..128).map(|j| values[(128 * half + j + i) % values.len()]));
            }
        }
        let path = std::env::temp_dir().join(format!("gguf-fp8b-{}.gguf", std::process::id()));
        let decl = TensorDecl {
            name: "t.weight".to_string(),
            dims: vec![256, 6],
            type_id: ty.as_u32(),
            nbytes: blocks.len() as u64,
        };
        write_file(&path, &[], &[(decl, blocks.clone())]);
        let split = crate::Split::open(&path);
        std::fs::remove_file(&path).unwrap();
        let split = split.unwrap();
        let (s, t) = split.find("t.weight").unwrap();
        assert_eq!((t.ty, t.nbytes), (ty, 12 * 132));
        assert_eq!((ty.blck_size(), ty.type_size()), (Some(128), Some(132)));
        let data = split.shard(s).unwrap().data(t).unwrap();
        let mut got = vec![0.0f32; 12 * 128];
        dequant_row(ty, data, &mut got).unwrap();
        for (b, (blk, out)) in blocks
            .as_chunks::<132>()
            .0
            .iter()
            .zip(got.chunks(128))
            .enumerate()
        {
            let d = f64::from(scales[b / 2]);
            for (j, (&q, g)) in blk[4..].iter().zip(out).enumerate() {
                let sign = if q & 0x80 == 0 { 1.0 } else { -1.0 };
                let want = (sign * ue4m3(q & 0x7F) * d) as f32;
                assert_eq!(
                    g.to_bits(),
                    want.to_bits(),
                    "block {b} value {j} byte {q:#04x}"
                );
            }
        }
        let refused = |src: &[u8], n: usize| {
            let mut dst = vec![0.0f32; n];
            let e = dequant_row(ty, src, &mut dst).expect_err("refused");
            assert!(dst.iter().all(|v| *v == 0.0), "{e}: a value was written");
            e
        };
        for (d, why) in [
            (f32::NAN, "not finite"),
            (f32::INFINITY, "not finite"),
            (f32::MAX / 2.0, "a value times it leaves f32's range"),
        ] {
            let mut bad = blocks[..264].to_vec();
            bad[132..136].copy_from_slice(&d.to_le_bytes());
            let e = refused(&bad, 256);
            assert_eq!(
                e,
                QuantError::UndefinedF32Scale {
                    ty,
                    block: 1,
                    bits: d.to_bits(),
                    why
                }
            );
            assert!(e.to_string().contains("block 1"), "{e}");
        }
        for byte in [0x7Fu8, 0xFF] {
            let mut nan_value = blocks[..264].to_vec();
            nan_value[132 + 4 + 5] = byte;
            assert_eq!(
                refused(&nan_value, 256),
                QuantError::E4m3Nan { at: 133, byte }
            );
        }
        assert!(matches!(
            check_fp8_b128(&blocks[..140]),
            Err(QuantError::ShortSrc {
                src: 140,
                need: 264,
                n: 256,
                ..
            })
        ));
        assert!(matches!(
            refused(&blocks[..140], 256),
            QuantError::ShortSrc {
                src: 140,
                need: 264,
                ..
            }
        ));
        assert!(matches!(
            refused(&blocks[..264], 192),
            QuantError::UnalignedDst {
                dst: 192,
                blck: 128,
                ..
            }
        ));
        assert_eq!(
            activation_format(ty),
            Err(QuantError::NoActivationFormat(ty))
        );
    }

    /// ModelOpt's NVFP4 value of a row stored as ModelOpt stores it, from the
    /// definitions: value `i` is `e2m1` of nibble `i` of the code bytes (value
    /// `2i` low, `2i + 1` high) times the E4M3 scale byte of its 16-value
    /// group times `scale`, the exact product rounded once to f32.
    fn modelopt_row(codes: &[u8], scales: &[u8], scale: f32) -> Vec<u32> {
        (0..codes.len() * 2)
            .map(|i| {
                let code = (codes[i / 2] >> (4 * (i % 2))) & 15;
                let v = e2m1(code) * ue4m3(scales[i / 16]) * f64::from(scale);
                (v as f32).to_bits()
            })
            .collect()
    }

    /// A ModelOpt row (`codes`, a byte two values, and `scales`, a byte a 16
    /// values) in bloomery's NVFP4 rows: each group's scale byte, then its 8
    /// code bytes as stored.
    fn nvfp4_rows(codes: &[u8], scales: &[u8]) -> Vec<u8> {
        codes
            .as_chunks::<8>()
            .0
            .iter()
            .zip(scales)
            .flat_map(|(c, &s)| std::iter::once(s).chain(c.iter().copied()))
            .collect()
    }

    /// bloomery's NVFP4 rows against the ModelOpt rule computed from the
    /// checkpoint's own layout: every scale byte that is a scale, each with
    /// every code at every place of its group, at scale 1 and at the PLE
    /// table's `weight_scale_2` (0x380B6DB7); and the first four rows of PLE
    /// parts 0 and 101 of nvidia/Qwen3.8-Flash-Next-FP4-Local b9ce18a6
    /// (`model.language_model.layers.1.ple.ple_embedding.ngram_embedding.
    /// shard_{0,101}.weight` U8 [2500012, 80] and `.weight_scale` F8_E4M3
    /// [2500012, 10], range-read from tensors/model-00015-of-00021 and
    /// -00017-of-00021; the fixture holds each part's 320 code bytes then its
    /// 40 scale bytes). Swapped nibbles, a scale from the wrong group, a
    /// scale not halved against the doubled code table or the tensor scale
    /// applied before the group's fails it.
    #[test]
    fn nvfp4_rows_decode_to_the_modelopt_rule() {
        let defined: Vec<u8> = (0..0x7Fu8).collect();
        let codes: Vec<u8> = defined
            .iter()
            .flat_map(|&b| {
                (0..8u8).map(move |i| ((2 * i + b) & 15) | (((2 * i + 1 + b) & 15) << 4))
            })
            .collect();
        let mut rows = vec![(codes, defined)];
        let fixture = include_bytes!("../tests/fixtures/qwen38-fp4l-ple-rows.bin");
        for part in fixture.as_chunks::<360>().0 {
            let (c, s) = part.split_at(320);
            rows.extend(
                c.as_chunks::<80>()
                    .0
                    .iter()
                    .zip(s.as_chunks::<10>().0)
                    .map(|(c, s)| (c.to_vec(), s.to_vec())),
            );
        }
        assert_eq!(rows.len(), 9);
        for scale in [1.0f32, f32::from_bits(0x380B_6DB7)] {
            let decode = super::Nvfp4Rows::new(scale);
            for (r, (codes, scales)) in rows.iter().enumerate() {
                let want = modelopt_row(codes, scales, scale);
                let mut got = vec![0.0f32; want.len()];
                for separate in [false, true] {
                    got.fill(f32::NAN);
                    if separate {
                        decode.decode_planes(codes, scales, &mut got).unwrap();
                    } else {
                        decode.decode(&nvfp4_rows(codes, scales), &mut got).unwrap();
                    }
                    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                        assert_eq!(
                            g.to_bits(),
                            *w,
                            "scale {scale:e} row {r} value {i} separate={separate}: got {g:e}"
                        );
                    }
                }
            }
        }
    }

    /// A scale byte that is no UE4M3 scale (0x7F, or bit 7 set) is refused
    /// naming its group and the byte, before a value is written; so is a span
    /// that is not whole groups, or one `src` is too short for.
    #[test]
    fn nvfp4_rows_refuse_by_name() {
        let ty = GgmlType::NVFP4_ROWS;
        let decode = super::Nvfp4Rows::new(1.0);
        for byte in [0x7Fu8, 0x80, 0xB8, 0xFF] {
            let mut src = [0x38u8; 45];
            src[27] = byte;
            let mut dst = [0.0f32; 80];
            let e = decode.decode(&src, &mut dst).expect_err("undefined");
            assert_eq!(
                e,
                QuantError::UndefinedScale {
                    ty,
                    block: 3,
                    sub: 0,
                    byte,
                    why: super::ue4m3(byte).unwrap_err(),
                }
            );
            assert!(e.to_string().contains("block 3 sub-block 0"), "{e}");
            assert_eq!(dst, [0.0; 80], "a refusal must leave dst unchanged");
            let mut scales = [0x38; 5];
            scales[3] = byte;
            let checked = decode
                .check_planes(&[0x38; 40], &scales, 80)
                .expect_err("undefined");
            assert_eq!(checked.to_string(), e.to_string());
            assert_eq!(decode.decode_planes(&[0x38; 40], &scales, &mut dst), Err(e));
            assert_eq!(dst, [0.0; 80], "a refusal must leave dst unchanged");
        }
        for (code, byte, divisor) in [(0x77, 0x38, 8.0), (0x22, 0x40, 2.0)] {
            for scale in [f32::MAX, -f32::MAX] {
                let mut src = [0x11; 45];
                for g in src.as_chunks_mut::<9>().0 {
                    g[0] = 0x38;
                }
                src[27] = byte;
                src[28..36].fill(code);
                let mut dst = [0.0; 80];
                super::Nvfp4Rows::new(scale / divisor)
                    .decode(&src, &mut dst)
                    .unwrap();
                dst.fill(0.0);
                let e = super::Nvfp4Rows::new(scale)
                    .decode(&src, &mut dst)
                    .unwrap_err();
                assert!(
                    matches!(e, QuantError::UndefinedScale { block: 3, .. }),
                    "{e}"
                );
                assert!(e.to_string().contains("leaves f32's range"), "{e}");
                assert_eq!(dst, [0.0; 80], "a refusal must leave dst unchanged");
                let (codes, scales): (Vec<u8>, Vec<u8>) = src
                    .as_chunks::<9>()
                    .0
                    .iter()
                    .map(|g| (g[1..].to_vec(), g[0]))
                    .fold((vec![], vec![]), |(mut c, mut s), (g, b)| {
                        c.extend(g);
                        s.push(b);
                        (c, s)
                    });
                let planes = super::Nvfp4Rows::new(scale).check_planes(&codes, &scales, 80);
                assert_eq!(planes, Err(e));
            }
        }
        assert_eq!(
            decode.decode(&[0x38; 9], &mut [0.0; 8]),
            Err(QuantError::UnalignedDst {
                ty,
                dst: 8,
                blck: 16
            })
        );
        assert!(matches!(
            decode.decode_planes(&[0x38; 8], &[0x38], &mut [0.0; 8]),
            Err(QuantError::UnalignedDst { dst: 8, .. })
        ));
        assert!(matches!(
            decode.check_planes(&[0x38; 7], &[0x38], 16),
            Err(QuantError::ShortSrc { src: 7, .. })
        ));
        assert!(matches!(
            decode.decode_planes(&[0x38; 7], &[0x38], &mut [0.0; 16]),
            Err(QuantError::ShortSrc {
                src: 7,
                need: 8,
                ..
            })
        ));
        assert!(matches!(
            decode.decode_planes(&[0x38; 8], &[], &mut [0.0; 16]),
            Err(QuantError::ShortSrc {
                src: 0,
                need: 1,
                ..
            })
        ));
        assert_eq!(
            decode.decode(&[0x38; 9], &mut [0.0; 32]),
            Err(QuantError::ShortSrc {
                ty,
                src: 9,
                need: 18,
                n: 32
            })
        );
    }

    /// IQ2_XXS against a block ggml made: `quantize_iq2_xxs` over 256
    /// seeded values with a seeded importance matrix, and `to_float`
    /// (`dequantize_row_iq2_xxs`) of it, from llama.cpp a0e6dc7's
    /// ggml-quants.c (its IQ2_XXS quantizer, dequant and tables are ik's,
    /// `gen-iq-tables.py` holding the grid equal in both trees). Its eight
    /// sub-blocks carry scales 9, 2, 1, 2, 15, 2, 3, 2, so a scale read from
    /// a neighbour moves values. Every value equals ggml's bit for bit.
    #[test]
    fn iq2_xxs_matches_a_ggml_made_block() {
        #[rustfmt::skip]
        const BLOCK: [u8; 66] = [
            203, 24, 0, 0, 3, 0, 243, 183, 141, 153, 95, 14, 177, 155, 146, 93,
            82, 41, 247, 202, 70, 114, 250, 40, 11, 19, 172, 184, 225, 192, 191, 250,
            194, 34, 0, 0, 3, 0, 53, 106, 19, 246, 139, 12, 0, 192, 134, 152,
            90, 40, 39, 23, 3, 0, 190, 72, 157, 56, 151, 78, 4, 186, 222, 246,
            131, 42,
        ];
        #[rustfmt::skip]
        const WANT: [u32; 256] = [
            0xbd36_2200, 0xbd36_2200, 0x3d36_2200, 0x3d36_2200, 0xbd36_2200, 0xbd36_2200,
            0xbd36_2200, 0xbd36_2200, 0xbd36_2200, 0xbd36_2200, 0xbd36_2200, 0xbd36_2200,
            0x3d36_2200, 0xbd36_2200, 0xbd36_2200, 0x3d36_2200, 0x3d36_2200, 0xbe74_bdb0,
            0xbd36_2200, 0x3d36_2200, 0xbd36_2200, 0xbd36_2200, 0x3d36_2200, 0x3d36_2200,
            0x3d36_2200, 0x3d36_2200, 0xbd36_2200, 0xbd36_2200, 0x3d36_2200, 0x3d36_2200,
            0xbd36_2200, 0xbd36_2200, 0x3c3f_b800, 0xbd80_cfa0, 0x3d15_c7c0, 0x3c3f_b800,
            0xbd15_c7c0, 0x3d15_c7c0, 0x3d15_c7c0, 0x3c3f_b800, 0xbc3f_b800, 0xbd80_cfa0,
            0x3d15_c7c0, 0xbd15_c7c0, 0xbc3f_b800, 0xbc3f_b800, 0x3c3f_b800, 0xbc3f_b800,
            0xbd15_c7c0, 0x3d80_cfa0, 0x3d15_c7c0, 0xbc3f_b800, 0x3c3f_b800, 0x3c3f_b800,
            0xbd15_c7c0, 0xbd15_c7c0, 0x3c3f_b800, 0xbc3f_b800, 0x3c3f_b800, 0xbc3f_b800,
            0x3c3f_b800, 0x3d15_c7c0, 0xbc3f_b800, 0xbd15_c7c0, 0x3be6_1000, 0xbbe6_1000,
            0x3d1a_92c0, 0xbbe6_1000, 0xbcb3_bc80, 0xbd1a_92c0, 0xbcb3_bc80, 0xbd1a_92c0,
            0xbbe6_1000, 0x3d1a_92c0, 0x3cb3_bc80, 0x3be6_1000, 0xbd1a_92c0, 0x3be6_1000,
            0xbd1a_92c0, 0xbcb3_bc80, 0x3d1a_92c0, 0x3be6_1000, 0xbbe6_1000, 0xbbe6_1000,
            0x3d1a_92c0, 0xbd1a_92c0, 0x3be6_1000, 0xbbe6_1000, 0x3d1a_92c0, 0x3be6_1000,
            0x3be6_1000, 0xbd1a_92c0, 0xbbe6_1000, 0x3be6_1000, 0x3d1a_92c0, 0x3be6_1000,
            0xbd15_c7c0, 0xbd15_c7c0, 0xbc3f_b800, 0xbd15_c7c0, 0xbd80_cfa0, 0xbd80_cfa0,
            0x3c3f_b800, 0x3d15_c7c0, 0xbd15_c7c0, 0x3c3f_b800, 0xbd15_c7c0, 0x3d80_cfa0,
            0xbd80_cfa0, 0xbc3f_b800, 0xbd15_c7c0, 0xbd15_c7c0, 0xbc3f_b800, 0xbc3f_b800,
            0x3c3f_b800, 0xbc3f_b800, 0x3d15_c7c0, 0x3d15_c7c0, 0x3c3f_b800, 0xbd80_cfa0,
            0x3d15_c7c0, 0xbd80_cfa0, 0xbd15_c7c0, 0x3c3f_b800, 0xbc3f_b800, 0x3d80_cfa0,
            0x3d15_c7c0, 0xbd15_c7c0, 0xbd94_9500, 0x3d94_9500, 0xbd94_9500, 0x3d94_9500,
            0xbd94_9500, 0xbd94_9500, 0x3d94_9500, 0x3d94_9500, 0x3d94_9500, 0x3d94_9500,
            0xbd94_9500, 0x3d94_9500, 0xbd94_9500, 0x3d94_9500, 0xbd94_9500, 0xbd94_9500,
            0xbd94_9500, 0x3ec7_a838, 0xbd94_9500, 0xbd94_9500, 0x3d94_9500, 0x3d94_9500,
            0xbd94_9500, 0x3d94_9500, 0x3d94_9500, 0x3d94_9500, 0x3d94_9500, 0x3d94_9500,
            0xbd94_9500, 0xbd94_9500, 0x3d94_9500, 0x3d94_9500, 0x3c3f_b800, 0xbd80_cfa0,
            0xbc3f_b800, 0x3d15_c7c0, 0x3c3f_b800, 0x3c3f_b800, 0x3c3f_b800, 0x3d15_c7c0,
            0xbc3f_b800, 0x3d15_c7c0, 0x3c3f_b800, 0x3d15_c7c0, 0xbc3f_b800, 0xbc3f_b800,
            0x3c3f_b800, 0xbc3f_b800, 0x3c3f_b800, 0xbc3f_b800, 0x3c3f_b800, 0xbc3f_b800,
            0x3c3f_b800, 0xbc3f_b800, 0xbc3f_b800, 0x3c3f_b800, 0x3d15_c7c0, 0xbd80_cfa0,
            0x3d15_c7c0, 0x3c3f_b800, 0x3c3f_b800, 0x3d80_cfa0, 0xbd15_c7c0, 0x3d15_c7c0,
            0x3c86_3400, 0xbdb4_55e0, 0xbc86_3400, 0xbd51_b140, 0xbc86_3400, 0xbd51_b140,
            0x3c86_3400, 0xbc86_3400, 0xbc86_3400, 0x3c86_3400, 0x3d51_b140, 0x3c86_3400,
            0xbd51_b140, 0x3c86_3400, 0x3c86_3400, 0x3c86_3400, 0xbc86_3400, 0x3db4_55e0,
            0xbc86_3400, 0x3c86_3400, 0xbc86_3400, 0xbc86_3400, 0xbc86_3400, 0xbc86_3400,
            0x3c86_3400, 0x3c86_3400, 0xbc86_3400, 0x3c86_3400, 0x3c86_3400, 0x3c86_3400,
            0xbc86_3400, 0x3c86_3400, 0x3c3f_b800, 0xbc3f_b800, 0xbd15_c7c0, 0xbc3f_b800,
            0xbd80_cfa0, 0x3c3f_b800, 0xbc3f_b800, 0xbd15_c7c0, 0xbc3f_b800, 0x3d15_c7c0,
            0xbc3f_b800, 0xbd80_cfa0, 0x3c3f_b800, 0xbc3f_b800, 0xbd15_c7c0, 0xbc3f_b800,
            0xbd80_cfa0, 0xbd80_cfa0, 0xbc3f_b800, 0xbc3f_b800, 0x3c3f_b800, 0x3c3f_b800,
            0x3c3f_b800, 0x3c3f_b800, 0x3d80_cfa0, 0x3c3f_b800, 0xbd15_c7c0, 0x3d80_cfa0,
            0xbc3f_b800, 0x3d15_c7c0, 0xbd15_c7c0, 0xbd15_c7c0,
        ];
        let mut dst = [0.0f32; 256];
        dequant_row(GgmlType::IQ2_XXS, &BLOCK, &mut dst).expect("one whole block");
        for (i, (&got, &want)) in dst.iter().zip(&WANT).enumerate() {
            assert_eq!(got.to_bits(), want, "value {i}: {got:e}");
        }
    }

    /// IQ2_XXS over its block layout: eight blocks whose 64 sub-blocks take
    /// every grid index once (group l of sub-block `q` indexes `4q + l`), the
    /// sign index `(37·(4q + l) + 5) mod 128` and the scale `q mod 16`, under
    /// a normal and a negative `d`; every value is `d·(0.5 + s)·0.25·grid[j]`
    /// negated where bit j of `ksigns_iq2xs[sign index]` is set, computed in
    /// f64 (exact, [`iq_products_are_exact`]). One byte short of a block is
    /// refused by name.
    #[test]
    fn iq2_xxs_decodes_every_index_sign_and_scale() {
        use crate::iq_tables::{IQ2XXS_GRID, KSIGNS_IQ2XS};
        let ds: [u16; 8] = [
            0x3c00, 0x2e66, 0xb555, 0x1001, 0x3c00, 0xc400, 0x0401, 0x7bff,
        ];
        let mut src = Vec::new();
        for (b, &d) in ds.iter().enumerate() {
            src.extend(d.to_le_bytes());
            for ib in 0..8 {
                let q = 8 * b + ib;
                let mut aux = (q as u32 % 16) << 28;
                for l in 0..4 {
                    src.push((4 * q + l) as u8);
                    aux |= ((37 * (4 * q + l) as u32 + 5) % 128) << (7 * l);
                }
                src.extend(aux.to_le_bytes());
            }
        }
        let mut dst = vec![0.0f32; 256 * ds.len()];
        dequant_row(GgmlType::IQ2_XXS, &src, &mut dst).expect("eight whole blocks");
        for (v, &got) in dst.iter().enumerate() {
            let (q, l, j) = (v / 32, (v / 8) % 4, v % 8);
            let d = f64::from(super::half_to_f32(ds[q / 8]));
            let g = IQ2XXS_GRID[4 * q + l].to_le_bytes()[j];
            let signs = KSIGNS_IQ2XS[(37 * (4 * q + l) + 5) % 128];
            let neg = if signs >> j & 1 != 0 { -1.0 } else { 1.0 };
            let want = d * (0.5 + (q % 16) as f64) * 0.25 * f64::from(g) * neg;
            assert_eq!(f64::from(got), want, "value {v}");
        }
        assert_eq!(
            dequant_row(GgmlType::IQ2_XXS, &src[..65], &mut [0.0f32; 256]),
            Err(QuantError::ShortSrc {
                ty: GgmlType::IQ2_XXS,
                src: 65,
                need: 66,
                n: 256
            })
        );
    }

    /// Eight blocks of Qwen3.8's layer-0 shared-expert gate: the checkpoint's BF16 values
    /// (`model.language_model.layers.0.mlp.shared_expert.gate_proj.weight` of
    /// nvidia/Qwen3.8-Flash-Next-NVFP4 at `fc694b54`) and the Q8_0 block unsloth's
    /// `UD-Q4_K_XL` (`766911a6`) holds for them in `blk.0.ffn_gate_shexp.weight`, 64 + 34
    /// bytes a block. In the first six the `127 / amax` rule codes some value differently.
    #[test]
    fn the_reference_rule_writes_ggml_s_weight_blocks() {
        let fixture = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/qwen38-shexp-q8_0-blocks.bin"
        ))
        .unwrap();
        let (blocks, rest) = fixture.as_chunks::<98>();
        assert!(rest.is_empty() && blocks.len() == 8);
        for (i, b) in blocks.iter().enumerate() {
            let x: [f32; 32] = std::array::from_fn(|j| {
                f32::from_bits(u32::from(u16::from_le_bytes([b[2 * j], b[2 * j + 1]])) << 16)
            });
            let want: &[u8; 34] = b[64..].try_into().unwrap();
            let got = super::Q8Block::quantize_ref(&x).unwrap();
            assert_eq!(&got.to_bytes(), want, "block {i}");
            assert_eq!(super::Q8Block::from_bytes(want), got);
            let other = crate::quant::quantize_row_q8_0_roundtrip;
            let mut rt = [0.0f32; 32];
            other(&x, &mut rt);
            let d = super::half_to_f32(got.d);
            let differs = rt.iter().zip(&got.q).any(|(&r, &q)| r != d * f32::from(q));
            assert_eq!(differs, i < 6, "block {i}: the two rules");
        }
        assert_eq!(
            super::Q8Block::quantize_ref(&[0.0; 32]).unwrap().to_bytes(),
            [0; 34]
        );
        for bad in [f32::NAN, f32::INFINITY] {
            let mut x = [1.0; 32];
            x[7] = bad;
            assert_eq!(super::Q8Block::quantize_ref(&x), None, "{bad}");
        }
        // Finite values whose scale `amax / 127` is past f16's range.
        assert_eq!(super::Q8Block::quantize_ref(&[1e10; 32]), None);
    }
    // MLX's affine blocks pinned from real mx.quantize output: pip mlx 0.32.1,
    // mx.quantize(w, group_size=64, bits=4|8, mode="affine") over a two-row LCG weight
    // (x = (1103515245·x + 12345) mod 2^31, value ((x >> 7) mod 2000 − 1000) / 347),
    // then mx.dequantize(q, s, b, group_size=64, bits). The words are the returned
    // `.weight` (LE U32), the scales and biases its `.scales`/`.biases` BF16 bits.
    const A4_WORDS: [[u32; 16]; 2] = [
        [
            0xd450dfe9, 0x3a4bf290, 0x4114bb68, 0x16bb6dc0, 0x98fff4e1, 0xf2a4b15b, 0x1fd471af,
            0x7454684a, 0x4e4502fe, 0xc5b8d467, 0xb22876da, 0xf049b6bc, 0x6d1c225f, 0x1de1ff29,
            0xb6364005, 0x2b59f488,
        ],
        [
            0xb23f5b5e, 0xe48dfd81, 0xd9585ae5, 0xbf4f8b8b, 0x75d23f87, 0x70ed02c7, 0xd14619b7,
            0x256985fd, 0x7f58f9ed, 0xe44f7999, 0x36491eca, 0x8c8ad9e0, 0xf204b4a5, 0x5b4f4425,
            0xfaacfea1, 0x2b692460,
        ],
    ];
    const A4_SCALES: [[u16; 2]; 2] = [[0xbeb4, 0xbeb7], [0x3eb8, 0x3eb4]];
    const A4_BIASES: [[u16; 2]; 2] = [[0x4034, 0x4037], [0xc038, 0xc034]];
    const A4_MX_DEQ: [[u32; 128]; 2] = [
        [
            0xbeb40000, 0xc0070000, 0xc01e0000, 0xbfe10000, 0x40340000, 0x3f870000, 0x3fb40000,
            0xbfe10000, 0x40340000, 0xbeb40000, 0x40070000, 0xc01e0000, 0xbf870000, 0x3fb40000,
            0xbf340000, 0x3fe10000, 0x00000000, 0x3f340000, 0xbf870000, 0xbf870000, 0x3fb40000,
            0x401e0000, 0x401e0000, 0x3fb40000, 0x40340000, 0xbfb40000, 0xbfe10000, 0x3f340000,
            0xbf870000, 0xbf870000, 0x3f340000, 0x401e0000, 0x401e0000, 0xc0070000, 0x3fb40000,
            0xc01e0000, 0xc01e0000, 0xc01e0000, 0x00000000, 0xbeb40000, 0xbf870000, 0x3f870000,
            0x401e0000, 0xbf870000, 0x3fb40000, 0xbf340000, 0x40070000, 0xc01e0000, 0xc01e0000,
            0xbf340000, 0x401e0000, 0x3eb40000, 0x3fb40000, 0xbfe10000, 0xc01e0000, 0x401e0000,
            0xbf340000, 0x3fb40000, 0x00000000, 0x3f340000, 0x3fb40000, 0x3f870000, 0x3fb40000,
            0x3eb40000, 0xc0090000, 0xc0200000, 0x40090000, 0x40370000, 0x3f890000, 0x3fb70000,
            0xc0090000, 0x3fb70000, 0x3eb70000, 0x3f370000, 0x3fb70000, 0xbfe50000, 0x00000000,
            0xbf890000, 0x3f890000, 0xbfb70000, 0xbf370000, 0xbfe50000, 0x3f370000, 0x3eb70000,
            0x00000000, 0x40090000, 0x40090000, 0xbf890000, 0xbfb70000, 0xbf890000, 0x3f370000,
            0xbf890000, 0xbeb70000, 0x3fb70000, 0x40370000, 0xc0200000, 0xc0200000, 0x3f890000,
            0x40090000, 0x40090000, 0xbfb70000, 0x40200000, 0xbfe50000, 0x3f370000, 0xbeb70000,
            0x40090000, 0xc0200000, 0xc0200000, 0x40200000, 0xc0090000, 0xbfe50000, 0x40200000,
            0x3f890000, 0x40370000, 0x40370000, 0x3fb70000, 0x3f370000, 0x3fe50000, 0x3f370000,
            0xbf890000, 0x00000000, 0x00000000, 0x3fb70000, 0xc0200000, 0xbeb70000, 0x3f890000,
            0xbf890000, 0x40090000,
        ],
        [
            0x400a0000, 0xbf8a0000, 0x3f8a0000, 0xbf8a0000, 0x40210000, 0xbfe60000, 0xc00a0000,
            0x3f8a0000, 0xc0210000, 0x00000000, 0x3fe60000, 0x40210000, 0x3fe60000, 0x00000000,
            0xbfb80000, 0x400a0000, 0xbf8a0000, 0x400a0000, 0x3f380000, 0xbf8a0000, 0x00000000,
            0xbf8a0000, 0x3eb80000, 0x3fe60000, 0x3f8a0000, 0x00000000, 0x3f8a0000, 0x00000000,
            0x40210000, 0xbfb80000, 0x40210000, 0x3f8a0000, 0xbeb80000, 0x00000000, 0x40210000,
            0xbfe60000, 0xc00a0000, 0x3fe60000, 0xbf8a0000, 0xbeb80000, 0xbeb80000, 0x3fb80000,
            0xc00a0000, 0xc0380000, 0x3fe60000, 0x400a0000, 0xc0380000, 0xbeb80000, 0xbeb80000,
            0x3f8a0000, 0x3eb80000, 0xc0210000, 0xbf380000, 0xbfb80000, 0xc0210000, 0x3fe60000,
            0x3fe60000, 0x40210000, 0xbf8a0000, 0x00000000, 0x3eb80000, 0xbf380000, 0xbf8a0000,
            0xc00a0000, 0x3fe10000, 0x40070000, 0x3eb40000, 0x401e0000, 0x00000000, 0xbf870000,
            0x401e0000, 0xbeb40000, 0x3eb40000, 0x3eb40000, 0x3eb40000, 0xbeb40000, 0x401e0000,
            0xbfb40000, 0xbfb40000, 0x40070000, 0x3f340000, 0x3fb40000, 0x40070000, 0xc01e0000,
            0x3eb40000, 0xbfb40000, 0xbf340000, 0xbfe10000, 0xc0340000, 0x40070000, 0x3eb40000,
            0x3fe10000, 0x3f340000, 0x00000000, 0x3fb40000, 0x00000000, 0xbf870000, 0x3f340000,
            0xbfb40000, 0x3f870000, 0xbfb40000, 0xc0340000, 0xc0070000, 0x401e0000, 0xbf870000,
            0xc0070000, 0xbfb40000, 0xbfb40000, 0x401e0000, 0xbfb40000, 0x3f870000, 0xbf870000,
            0xc01e0000, 0x3f340000, 0x40070000, 0x401e0000, 0x3fb40000, 0x3f340000, 0x3f340000,
            0x401e0000, 0xc0340000, 0xbf340000, 0xbfb40000, 0xc0070000, 0x3eb40000, 0xbf340000,
            0x3f870000, 0xc0070000,
        ],
    ];
    const A8_WORDS: [[u32; 32]; 2] = [
        [
            0xcfebe08e, 0xd03b5100, 0xf7248802, 0x34a23ab6, 0xaaaa5a7d, 0x440a0a3e, 0x5fd3bb07,
            0x1058b3ac, 0xfd47e60f, 0x9186f0f5, 0xb20e4fb5, 0xfc21a03e, 0x680ca4e9, 0x10ffd242,
            0x65793c9c, 0x7746493f, 0x0023efe0, 0x45df3a4a, 0xd63c5d77, 0xbf4db97a, 0x6f65d29b,
            0xab1d2088, 0xab66b8bf, 0xff003c8c, 0x1b264af8, 0x5ccc11c9, 0xfcff2499, 0x0bcfde0f,
            0x3a01024e, 0xad673168, 0xf2457a80, 0x1ab84c95,
        ],
        [
            0x53ad4cdc, 0xb92037fe, 0xf4d0830e, 0xe74588d9, 0x57a6de54, 0xcd985279, 0x7bae86b1,
            0xadff3cfe, 0x2afe856c, 0x7149cc20, 0x0827c473, 0x6f00e7d3, 0x1390b270, 0xd4103861,
            0x884ef5ca, 0x254a6297, 0xf891e0cf, 0x75eb5781, 0x77959499, 0xe6403bfc, 0x0bdbc99c,
            0x37693f8c, 0xcd90e500, 0x86c37b9b, 0xaf3ba54f, 0xfc270848, 0x44401f51, 0x4fb139f9,
            0xffe59e10, 0xfe9ca4be, 0x22426606, 0x22b35a8b,
        ],
    ];
    const A8_SCALES: [[u16; 2]; 2] = [[0xbcb4, 0xbcb6], [0x3cb7, 0x3cb3]];
    const A8_BIASES: [[u16; 2]; 2] = [[0x4034, 0x4037], [0xc038, 0xc034]];
    const A8_MX_DEQ: [[u32; 128]; 2] = [
        [
            0xbe9e0000, 0xc0070000, 0xc0160000, 0xbfde0000, 0x40340000, 0x3f840000, 0x3fc20000,
            0xbfe10000, 0x40310000, 0xbe340000, 0x40010000, 0xc0270000, 0xbf980000, 0x3fc50000,
            0xbf3f0000, 0x3fd60000, 0x3d870000, 0x3f560000, 0xbf6c0000, 0xbf6c0000, 0x3fba0000,
            0x40260000, 0x40260000, 0x3fa90000, 0x402a0000, 0xbfa60000, 0xbfe90000, 0x3f3a0000,
            0xbf780000, 0xbf8f0000, 0x3f610000, 0x401e0000, 0x401f0000, 0xc00f0000, 0x3fa00000,
            0xc0300000, 0xc0250000, 0xc01e0000, 0xbe070000, 0xbebf0000, 0xbf950000, 0x3f8a0000,
            0x40200000, 0xbf8d0000, 0x3fba0000, 0xbf340000, 0x40060000, 0xc02e0000, 0xc0140000,
            0xbf4a0000, 0x40230000, 0x3f070000, 0x3fae0000, 0xbfe70000, 0xc0330000, 0x401e0000,
            0xbf1e0000, 0x3fbf0000, 0x3e1e0000, 0x3f180000, 0x3fb70000, 0x3f9b0000, 0x3fa30000,
            0x3e4a0000, 0xc0080000, 0xc01d0000, 0x40050000, 0x40370000, 0x3f9c0000, 0x3fc90000,
            0xc0060000, 0x3faa0000, 0x3e5d0000, 0x3f4b0000, 0x3fc30000, 0xbff30000, 0x3e180000,
            0xbfa00000, 0x3f930000, 0xbfb10000, 0xbf160000, 0xbfe70000, 0x3f1e0000, 0x3ec90000,
            0xbe260000, 0x400a0000, 0x400e0000, 0xbf710000, 0xbfb10000, 0xbf9d0000, 0x3f180000,
            0xbf710000, 0xbe800000, 0x3fc30000, 0x40370000, 0xc0340000, 0xc02a0000, 0x3f9c0000,
            0x40010000, 0x40110000, 0xbfce0000, 0x401f0000, 0xbfd60000, 0x3f510000, 0xbf0a0000,
            0x40040000, 0xc0340000, 0xc02f0000, 0x40220000, 0xc0050000, 0xbfdf0000, 0x40270000,
            0x3f900000, 0x40340000, 0x40360000, 0x3fc90000, 0x3f0c0000, 0x3fe30000, 0x3f120000,
            0xbf7c0000, 0x3c800000, 0x3e180000, 0x3faa0000, 0xc0210000, 0xbee70000, 0x3f960000,
            0xbf9d0000, 0x40120000,
        ],
        [
            0x40030000, 0xbf970000, 0x3f7d0000, 0xbf830000, 0x40330000, 0xbfd30000, 0xc00a0000,
            0x3fa10000, 0xc0240000, 0x3d520000, 0x3fe30000, 0x40250000, 0x3ffc0000, 0x3e270000,
            0xbfab0000, 0x40120000, 0xbf800000, 0x40050000, 0x3f550000, 0xbf6e0000, 0xbe300000,
            0xbf860000, 0x3f050000, 0x3fda0000, 0x3f8a0000, 0x3df20000, 0x3f820000, 0xbe020000,
            0x40330000, 0xbfc40000, 0x40350000, 0x3f7d0000, 0xbeed0000, 0x3dc50000, 0x40330000,
            0xbff80000, 0xc00a0000, 0x3fd70000, 0xbf9f0000, 0xbeb40000, 0xbe9d0000, 0x3fc00000,
            0xc0000000, 0xc02d0000, 0x3feb0000, 0x40120000, 0xc0380000, 0xbeca0000, 0xbebf0000,
            0x3f8d0000, 0x3eaf0000, 0xc01d0000, 0xbf350000, 0xbfd00000, 0xc0210000, 0x3fee0000,
            0x3fd20000, 0x40260000, 0xbf910000, 0x3e270000, 0x3eff0000, 0xbf300000, 0xbf9c0000,
            0xc0030000, 0x3fdb0000, 0x40050000, 0x3eb60000, 0x40270000, 0x3bcc0000, 0xbf690000,
            0x40150000, 0xbe830000, 0x3f080000, 0x3ed80000, 0x3ee30000, 0xbe590000, 0x402c0000,
            0xbfc30000, 0xbfb50000, 0x400e0000, 0x3f190000, 0x3fca0000, 0x3ffd0000, 0xc0250000,
            0x3e7c0000, 0xbfb80000, 0xbf050000, 0xbfce0000, 0xc0340000, 0x400c0000, 0x3eab0000,
            0x3fd50000, 0x3f130000, 0xbe000000, 0x3fb90000, 0x3dec0000, 0xbf8b0000, 0x3f4b0000,
            0xbfc30000, 0x3f810000, 0xbf9f0000, 0xc0290000, 0xbffb0000, 0x402c0000, 0xbf850000,
            0xc0090000, 0xbfb50000, 0xbfaa0000, 0x40280000, 0xbfc90000, 0x3f870000, 0xbf8b0000,
            0xc01e0000, 0x3f240000, 0x400c0000, 0x40310000, 0x3fab0000, 0x3f450000, 0x3f190000,
            0x402f0000, 0xc02c0000, 0xbf150000, 0xbfaf0000, 0xc0040000, 0x3e660000, 0xbf590000,
            0x3f8d0000, 0xc0040000,
        ],
    ];
    // The exact f32 oracle (`s·q + b`, the product exact) and MLX's own bf16 outputs.
    const A4_F32: [[u32; 128]; 2] = [
        [
            0xbeb40000, 0xc0070000, 0xc01d8000, 0xbfe10000, 0x40340000, 0x3f870000, 0x3fb40000,
            0xbfe10000, 0x40340000, 0xbeb40000, 0x40070000, 0xc01d8000, 0xbf870000, 0x3fb40000,
            0xbf340000, 0x3fe10000, 0x00000000, 0x3f340000, 0xbf870000, 0xbf870000, 0x3fb40000,
            0x401d8000, 0x401d8000, 0x3fb40000, 0x40340000, 0xbfb40000, 0xbfe10000, 0x3f340000,
            0xbf870000, 0xbf870000, 0x3f340000, 0x401d8000, 0x401d8000, 0xc0070000, 0x3fb40000,
            0xc01d8000, 0xc01d8000, 0xc01d8000, 0x00000000, 0xbeb40000, 0xbf870000, 0x3f870000,
            0x401d8000, 0xbf870000, 0x3fb40000, 0xbf340000, 0x40070000, 0xc01d8000, 0xc01d8000,
            0xbf340000, 0x401d8000, 0x3eb40000, 0x3fb40000, 0xbfe10000, 0xc01d8000, 0x401d8000,
            0xbf340000, 0x3fb40000, 0x00000000, 0x3f340000, 0x3fb40000, 0x3f870000, 0x3fb40000,
            0x3eb40000, 0xc0094000, 0xc0202000, 0x40094000, 0x40370000, 0x3f894000, 0x3fb70000,
            0xc0094000, 0x3fb70000, 0x3eb70000, 0x3f370000, 0x3fb70000, 0xbfe4c000, 0x00000000,
            0xbf894000, 0x3f894000, 0xbfb70000, 0xbf370000, 0xbfe4c000, 0x3f370000, 0x3eb70000,
            0x00000000, 0x40094000, 0x40094000, 0xbf894000, 0xbfb70000, 0xbf894000, 0x3f370000,
            0xbf894000, 0xbeb70000, 0x3fb70000, 0x40370000, 0xc0202000, 0xc0202000, 0x3f894000,
            0x40094000, 0x40094000, 0xbfb70000, 0x40202000, 0xbfe4c000, 0x3f370000, 0xbeb70000,
            0x40094000, 0xc0202000, 0xc0202000, 0x40202000, 0xc0094000, 0xbfe4c000, 0x40202000,
            0x3f894000, 0x40370000, 0x40370000, 0x3fb70000, 0x3f370000, 0x3fe4c000, 0x3f370000,
            0xbf894000, 0x00000000, 0x00000000, 0x3fb70000, 0xc0202000, 0xbeb70000, 0x3f894000,
            0xbf894000, 0x40094000,
        ],
        [
            0x400a0000, 0xbf8a0000, 0x3f8a0000, 0xbf8a0000, 0x40210000, 0xbfe60000, 0xc00a0000,
            0x3f8a0000, 0xc0210000, 0x00000000, 0x3fe60000, 0x40210000, 0x3fe60000, 0x00000000,
            0xbfb80000, 0x400a0000, 0xbf8a0000, 0x400a0000, 0x3f380000, 0xbf8a0000, 0x00000000,
            0xbf8a0000, 0x3eb80000, 0x3fe60000, 0x3f8a0000, 0x00000000, 0x3f8a0000, 0x00000000,
            0x40210000, 0xbfb80000, 0x40210000, 0x3f8a0000, 0xbeb80000, 0x00000000, 0x40210000,
            0xbfe60000, 0xc00a0000, 0x3fe60000, 0xbf8a0000, 0xbeb80000, 0xbeb80000, 0x3fb80000,
            0xc00a0000, 0xc0380000, 0x3fe60000, 0x400a0000, 0xc0380000, 0xbeb80000, 0xbeb80000,
            0x3f8a0000, 0x3eb80000, 0xc0210000, 0xbf380000, 0xbfb80000, 0xc0210000, 0x3fe60000,
            0x3fe60000, 0x40210000, 0xbf8a0000, 0x00000000, 0x3eb80000, 0xbf380000, 0xbf8a0000,
            0xc00a0000, 0x3fe10000, 0x40070000, 0x3eb40000, 0x401d8000, 0x00000000, 0xbf870000,
            0x401d8000, 0xbeb40000, 0x3eb40000, 0x3eb40000, 0x3eb40000, 0xbeb40000, 0x401d8000,
            0xbfb40000, 0xbfb40000, 0x40070000, 0x3f340000, 0x3fb40000, 0x40070000, 0xc01d8000,
            0x3eb40000, 0xbfb40000, 0xbf340000, 0xbfe10000, 0xc0340000, 0x40070000, 0x3eb40000,
            0x3fe10000, 0x3f340000, 0x00000000, 0x3fb40000, 0x00000000, 0xbf870000, 0x3f340000,
            0xbfb40000, 0x3f870000, 0xbfb40000, 0xc0340000, 0xc0070000, 0x401d8000, 0xbf870000,
            0xc0070000, 0xbfb40000, 0xbfb40000, 0x401d8000, 0xbfb40000, 0x3f870000, 0xbf870000,
            0xc01d8000, 0x3f340000, 0x40070000, 0x401d8000, 0x3fb40000, 0x3f340000, 0x3f340000,
            0x401d8000, 0xc0340000, 0xbf340000, 0xbfb40000, 0xc0070000, 0x3eb40000, 0xbf340000,
            0x3f870000, 0xc0070000,
        ],
    ];
    const A8_F32: [[u32; 128]; 2] = [
        [
            0xbe9d8000, 0xc0070000, 0xc0167800, 0xbfde3000, 0x40340000, 0x3f843000, 0x3fc21000,
            0xbfe10000, 0x40313000, 0xbe340000, 0x40016000, 0xc0275800, 0xbf97e000, 0x3fc4e000,
            0xbf3f4000, 0x3fd5c000, 0x3d870000, 0x3f55c000, 0xbf6c4000, 0xbf6c4000, 0x3fb9a000,
            0x4025f000, 0x4025f000, 0x3fa8c000, 0x402a2800, 0xbfa5f000, 0xbfe97000, 0x3f39a000,
            0xbf778000, 0xbf8f7000, 0x3f610000, 0x401d8000, 0x401ee800, 0xc00f7000, 0x3fa05000,
            0xc02fc800, 0xc0248800, 0xc01d8000, 0xbe070000, 0xbebf4000, 0xbf951000, 0x3f89d000,
            0x40205000, 0xbf8ca000, 0x3fb9a000, 0xbf340000, 0x40059800, 0xc02e6000, 0xc013a800,
            0xbf4a8000, 0x40232000, 0x3f070000, 0x3fae6000, 0xbfe6a000, 0xc0329800, 0x401d8000,
            0xbf1d8000, 0x3fbf4000, 0x3e1d8000, 0x3f17e000, 0x3fb6d000, 0x3f9ab000, 0x3fa32000,
            0x3e4a8000, 0xc0078000, 0xc01cd400, 0x40053c00, 0x40370000, 0x3f9b9000, 0x3fc91000,
            0xc0061400, 0x3fa9c800, 0x3e5cc000, 0x3f4b1000, 0x3fc36000, 0xbff29000, 0x3e188000,
            0xbfa01800, 0x3f930800, 0xbfb12800, 0xbf159000, 0xbfe73000, 0x3f1d9000, 0x3ec96000,
            0xbe260000, 0x40098000, 0x400dc400, 0xbf709000, 0xbfb12800, 0xbf9d4000, 0x3f17e000,
            0xbf709000, 0xbe808000, 0x3fc36000, 0x40370000, 0xc0339400, 0xc029a000, 0x3f9b9000,
            0x4000f800, 0x40109c00, 0xbfcd9800, 0x401ed400, 0xbfd62000, 0x3f50c000, 0xbf0a3000,
            0x4003d000, 0xc0339400, 0xc02f5000, 0x4021ac00, 0xc004a800, 0xbfdea800, 0x40275c00,
            0x3f903000, 0x40342800, 0x40359400, 0x3fc91000, 0x3f0c8000, 0x3fe2a800, 0x3f123000,
            0xbf7bf000, 0x3c800000, 0x3e188000, 0x3fa9c800, 0xc0211800, 0xbee6e000, 0x3f95e000,
            0xbf9d4000, 0x40120800,
        ],
        [
            0x40028800, 0xbf96b000, 0x3f7d5800, 0xbf82ac00, 0x40332400, 0xbfd2bc00, 0xc00a4000,
            0x3fa0fc00, 0xc023fc00, 0x3d528000, 0x3fe2c000, 0x4024d800, 0x3ffc7c00, 0x3e270000,
            0xbfaab400, 0x40124200, 0xbf7fa000, 0x40056400, 0x3f555000, 0xbf6e7800, 0xbe302000,
            0xbf858800, 0x3f054000, 0x3fda2c00, 0x3f8a1c00, 0x3df28000, 0x3f818800, 0xbe026000,
            0x40332400, 0xbfc47000, 0x40349200, 0x3f7d5800, 0xbeecc000, 0x3dc4c000, 0x40332400,
            0xbff7e800, 0xc00a4000, 0x3fd75000, 0xbf9f4400, 0xbeb39000, 0xbe9cb000, 0x3fc07000,
            0xc0003e00, 0xc02c9000, 0x3feb5400, 0x40124200, 0xc0380000, 0xbeca7000, 0xbebf0000,
            0x3f8cf800, 0x3eaf0000, 0xc01cd600, 0xbf354800, 0xbfcfe000, 0xc0212000, 0x3fee3000,
            0x3fd19800, 0x40264600, 0xbf90f800, 0x3e270000, 0x3eff1000, 0xbf2f9000, 0xbf9c6800,
            0xc0031a00, 0x3fdaf400, 0x40054000, 0x3eb63000, 0x4026d000, 0x3bcc0000, 0xbf695800,
            0x4014a200, 0xbe831000, 0x3f07d800, 0x3ed7c000, 0x3ee2f000, 0xbe596000, 0x402c6800,
            0xbfc2fc00, 0xbfb50000, 0x400da400, 0x3f18a000, 0x3fca2c00, 0x3ffc8400, 0xc0249e00,
            0x3e7c8000, 0xbfb7cc00, 0xbf04a800, 0xbfce2c00, 0xc0340000, 0x400c3e00, 0x3eab0000,
            0x3fd55c00, 0x3f130800, 0xbdffc000, 0x3fb96400, 0x3dec8000, 0xbf8b0c00, 0x3f4af800,
            0xbfc2fc00, 0x3f817400, 0xbf9ea000, 0xc028d000, 0xbffaec00, 0x402c6800, 0xbf857400,
            0xc008a600, 0xbfb50000, 0xbfa9d000, 0x40283600, 0xbfc89400, 0x3f870c00, 0xbf8b0c00,
            0xc01da000, 0x3f23d000, 0x400c3e00, 0x40309a00, 0x3fab6800, 0x3f456000, 0x3f18a000,
            0x402f3400, 0xc02b9c00, 0xbf157000, 0xbfaf6800, 0xc0047400, 0x3e662000, 0xbf589000,
            0x3f8ca400, 0xc0047400,
        ],
    ];

    /// The pinned MLX blocks decode to the exact f32 `s·q + b` and, rounded
    /// once to bf16, to MLX's own `dequantize` bit for bit; the rows are named
    /// and sized as the table holds them. Swapping the nibble order
    /// (`4 * j` → `4 * (7 - j)` in `dequant_mlx_a4`) or the scale·bias pair is
    /// red here.
    #[test]
    fn mlx_blocks_decode_mx_quantize_output_bit_for_bit() {
        let run = |ty: GgmlType,
                   wpg: usize,
                   words: &[[u32; 32]; 2],
                   scales: &[[u16; 2]; 2],
                   biases: &[[u16; 2]; 2],
                   f32_bits: &[[u32; 128]; 2],
                   mx_bits: &[[u32; 128]; 2]| {
            assert_eq!(
                ty.name(),
                Some(if ty == GgmlType::MLX_A4 {
                    "mlx_a4"
                } else {
                    "mlx_a8"
                })
            );
            assert_eq!(ty.blck_size(), Some(64));
            assert_eq!(
                ty.type_size(),
                Some(if ty == GgmlType::MLX_A4 { 36 } else { 68 })
            );
            let mut src = Vec::new();
            for r in 0..2 {
                for g in 0..2 {
                    for w in &words[r][g * wpg..(g + 1) * wpg] {
                        src.extend_from_slice(&w.to_le_bytes());
                    }
                    src.extend_from_slice(&scales[r][g].to_le_bytes());
                    src.extend_from_slice(&biases[r][g].to_le_bytes());
                }
            }
            let mut got = vec![0.0f32; 256];
            dequant_row(ty, &src, &mut got).unwrap();
            for r in 0..2 {
                for i in 0..128 {
                    let v = got[r * 128 + i];
                    assert_eq!(v.to_bits(), f32_bits[r][i], "{ty} row {r} value {i}");
                    assert_eq!(
                        u32::from(super::fp32_to_bf16_bits(v)) << 16,
                        mx_bits[r][i],
                        "{ty} row {r} value {i}: mx.dequantize's bf16"
                    );
                }
            }
        };
        run(
            GgmlType::MLX_A4,
            8,
            &{
                let mut w = [[0u32; 32]; 2];
                for r in 0..2 {
                    w[r][..16].copy_from_slice(&A4_WORDS[r]);
                }
                w
            },
            &A4_SCALES,
            &A4_BIASES,
            &A4_F32,
            &A4_MX_DEQ,
        );
        run(
            GgmlType::MLX_A8,
            16,
            &A8_WORDS,
            &A8_SCALES,
            &A8_BIASES,
            &A8_F32,
            &A8_MX_DEQ,
        );
    }

    /// An MLX affine block whose BF16 scale or bias is not finite, or whose
    /// top code's sum leaves f32, is refused by name before any value is
    /// written; a partial block and a CPU matmul's activation format are
    /// refused too.
    #[test]
    fn mlx_blocks_refuse_undefined_pairs_and_activations() {
        let block = |ty: GgmlType, scale: u16, bias: u16| -> Vec<u8> {
            let codes = ty.type_size().unwrap() as usize - 4;
            let mut b = vec![0x11u8; codes];
            b.extend_from_slice(&scale.to_le_bytes());
            b.extend_from_slice(&bias.to_le_bytes());
            b
        };
        let refused = |ty: GgmlType, src: &[u8], want: &str| {
            let mut dst = vec![0.0f32; 64];
            let e = dequant_row(ty, src, &mut dst).unwrap_err();
            assert!(e.to_string().contains(want), "{ty}: {e}");
        };
        // 0x7fc0: bf16 NaN; 0x7f80: bf16 inf; 0x7f7f: bf16's largest finite,
        // whose product with the 4-bit top code 15 overflows.
        refused(
            GgmlType::MLX_A4,
            &block(GgmlType::MLX_A4, 0x7fc0, 0),
            "block 0 holds scale bits 0x7fc0, not finite",
        );
        refused(
            GgmlType::MLX_A8,
            &block(GgmlType::MLX_A8, 0x3f80, 0x7f80),
            "block 0 holds bias bits 0x7f80, not finite",
        );
        for ty in [GgmlType::MLX_A4, GgmlType::MLX_A8] {
            let e = check_mlx_affine(ty, &block(ty, 0x7f7f, 0)).unwrap_err();
            assert_eq!(
                e,
                QuantError::UndefinedAffinePair {
                    ty,
                    block: 0,
                    field: "scale",
                    bits: 0x7f7f,
                    why: "a value times it leaves f32's range",
                },
                "{ty}"
            );
        }
        let e = dequant_row(
            GgmlType::MLX_A4,
            &block(GgmlType::MLX_A4, 0, 0)[..35],
            &mut vec![0.0; 64],
        )
        .unwrap_err();
        assert!(matches!(e, QuantError::ShortSrc { .. }), "{e}");
        assert_eq!(
            activation_format(GgmlType::MLX_A4),
            Err(QuantError::NoActivationFormat(GgmlType::MLX_A4))
        );
        assert_eq!(
            activation_format(GgmlType::MLX_A8),
            Err(QuantError::NoActivationFormat(GgmlType::MLX_A8))
        );
    }

    /// One MLX affine group: `words` as the code words, then the scale and the bias bits.
    fn affine_group(words: &[u32], scale: u16, bias: u16) -> Vec<u8> {
        let mut b: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        b.extend_from_slice(&scale.to_le_bytes());
        b.extend_from_slice(&bias.to_le_bytes());
        b
    }

    /// The F16-pair types are named and sized as their BF16 twins and decode to the hand values
    /// of `s·q + b` with `s` and `b` widened from F16 bits: 0x3400 is 0.25, 0xb800 is -0.5, 0x3c00
    /// is 1, 0xc000 is -2, and 0x0001 is the smallest subnormal, 2^-24. The same bytes read as the
    /// BF16 type are other values, so the two types are two meanings.
    #[test]
    fn mlx_f16_pair_blocks_decode_to_hand_values() {
        for (ty, twin, name, size) in [
            (GgmlType::MLX_A4H, GgmlType::MLX_A4, "mlx_a4h", 36),
            (GgmlType::MLX_A8H, GgmlType::MLX_A8, "mlx_a8h", 68),
        ] {
            assert_eq!(ty.name(), Some(name));
            assert_eq!(ty.blck_size(), Some(64));
            assert_eq!(ty.type_size(), Some(size));
            assert_eq!(ty.type_size(), twin.type_size());
            assert!(ty.has_dequant());
            assert_eq!(GgmlType::from_u32(ty.as_u32()), ty);
        }
        // 4-bit: word 0 holds codes 0..=7 low nibble first, word 1 codes 8..=15.
        let mut words = [0u32; 8];
        words[0] = 0x7654_3210;
        words[1] = 0xfedc_ba98;
        let src = affine_group(&words, 0x3400, 0xb800);
        let mut got = [0.0f32; 64];
        dequant_row(GgmlType::MLX_A4H, &src, &mut got).unwrap();
        for (i, v) in got.iter().enumerate() {
            let want = 0.25 * (if i < 16 { i as f32 } else { 0.0 }) - 0.5;
            assert_eq!(v.to_bits(), want.to_bits(), "mlx_a4h value {i}");
        }
        let mut as_bf16 = [0.0f32; 64];
        dequant_row(GgmlType::MLX_A4, &src, &mut as_bf16).unwrap();
        assert_ne!(as_bf16[5].to_bits(), got[5].to_bits());
        // The smallest subnormal scale times the top code, no bias.
        let mut words = [0u32; 8];
        words[7] = 0xf000_0000;
        let src = affine_group(&words, 0x0001, 0x0000);
        dequant_row(GgmlType::MLX_A4H, &src, &mut got).unwrap();
        assert_eq!(got[63].to_bits(), (15.0f32 * 2.0f32.powi(-24)).to_bits());
        assert_eq!(got[62].to_bits(), 0.0f32.to_bits());

        // 8-bit: byte j of word w is value 4w + j.
        let mut words = [0u32; 16];
        words[0] = 0x0302_0100;
        words[1] = 0xfffe_8001;
        let src = affine_group(&words, 0x3c00, 0xc000);
        let mut got = [0.0f32; 64];
        dequant_row(GgmlType::MLX_A8H, &src, &mut got).unwrap();
        let codes = [0u32, 1, 2, 3, 1, 128, 254, 255];
        for (i, v) in got.iter().enumerate() {
            let q = codes.get(i).copied().unwrap_or(0) as f32;
            assert_eq!(v.to_bits(), (q - 2.0).to_bits(), "mlx_a8h value {i}");
        }
        let mut as_bf16 = [0.0f32; 64];
        dequant_row(GgmlType::MLX_A8, &src, &mut as_bf16).unwrap();
        assert_ne!(as_bf16[7].to_bits(), got[7].to_bits());
        // F16's largest finite scale times the top code stays inside f32.
        let src = affine_group(&words, 0x7bff, 0x0000);
        dequant_row(GgmlType::MLX_A8H, &src, &mut got).unwrap();
        assert_eq!(got[7].to_bits(), (65504.0f32 * 255.0).to_bits());
    }

    /// An F16-pair group whose scale or bias is not finite is refused by name before any value is
    /// written (0x7c00 inf, 0xfc00 -inf, 0x7e00 NaN), a partial group is `ShortSrc`, and the types
    /// have no activation format.
    #[test]
    fn mlx_f16_pair_blocks_refuse_undefined_pairs() {
        for ty in [GgmlType::MLX_A4H, GgmlType::MLX_A8H] {
            let codes = ty.type_size().unwrap() as usize / 4 - 1;
            let words = vec![0x1111_1111u32; codes];
            for (scale, bias, field, bits) in [
                (0x7c00u16, 0u16, "scale", 0x7c00u16),
                (0x3c00, 0xfc00, "bias", 0xfc00),
                (0x7e00, 0, "scale", 0x7e00),
            ] {
                let src = affine_group(&words, scale, bias);
                let mut dst = vec![1.0f32; 64];
                let e = dequant_row(ty, &src, &mut dst).unwrap_err();
                assert_eq!(
                    e,
                    QuantError::UndefinedAffinePair {
                        ty,
                        block: 0,
                        field,
                        bits,
                        why: "not finite",
                    },
                    "{ty}"
                );
                assert!(dst.iter().all(|&v| v == 1.0), "{ty}: a value was written");
            }
            let ok = affine_group(&words, 0x3c00, 0x0000);
            let e = dequant_row(ty, &ok[..ok.len() - 1], &mut vec![0.0; 64]).unwrap_err();
            assert!(matches!(e, QuantError::ShortSrc { .. }), "{e}");
            assert_eq!(
                activation_format(ty),
                Err(QuantError::NoActivationFormat(ty))
            );
        }
    }
}
