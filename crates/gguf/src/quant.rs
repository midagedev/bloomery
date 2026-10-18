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
//! Q2_K, IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S, IQ4_NL and IQ4_XS are ports of the same-named
//! functions in that file, whose bodies are identical in mainline ggml. None has a product
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
    IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS, KVALUES_IQ4NL,
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
/// ik has no Q2_0.
///
/// A named type is one this build can size ([`GgmlType::blck_size`],
/// [`GgmlType::type_size`]): it has a row in [`crate::QROWS`], the one owner
/// of the facts a type has. Whether it can decode is a separate question
/// ([`GgmlType::has_dequant`]), and Q4_0 and Q4_1 are named, sized and
/// decoded by nothing here. `Unknown` carries every other tag: the ids
/// the enum does not name (q8_1 and q8_K among them), the
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
            10 => GgmlType::Q2_K,
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
            40 => GgmlType::NVFP4,
            42 => GgmlType::Q2_0,
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
            GgmlType::Q2_K => 10,
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
            GgmlType::NVFP4 => 40,
            GgmlType::Q2_0 => 42,
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
}

/// ggml's name for a type this enum names, or for an id ggml's table sizes
/// ([`crate::ggml_type_info`]: q8_1, q8_K), so a refusal of such a
/// tensor names its type; `ggml-type-<id>` for any other id.
impl fmt::Display for GgmlType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
            .name()
            .or_else(|| crate::ggml_type_info(self.as_u32()).map(|(n, _, _)| n))
        {
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
    /// A block scale byte the format leaves undefined: no value is written
    /// for it.
    #[error("dequant {ty}: block {block} sub-block {sub} holds scale byte {byte:#04x}, {why}")]
    UndefinedScale {
        ty: GgmlType,
        block: usize,
        sub: usize,
        byte: u8,
        why: &'static str,
    },
    /// A byte that is no UE4M3 scale, outside a block ([`ue4m3_to_f32_half`]).
    #[error("{byte:#04x} is no UE4M3 scale: {why}")]
    NotUe4m3 { byte: u8, why: &'static str },
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
        GgmlType::IQ3_XXS => dequant_iq3_xxs(&src[..need], dst),
        GgmlType::IQ3_S => dequant_iq3_s(&src[..need], dst),
        GgmlType::IQ4_NL => dequant_iq4_nl(&src[..need], dst),
        GgmlType::IQ4_XS => dequant_iq4_xs(&src[..need], dst),
        GgmlType::Q2_0 => dequant_q2_0(&src[..need], dst),
        GgmlType::Q4_0
        | GgmlType::Q4_1
        | GgmlType::IQ1_S
        | GgmlType::IQ2_S
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
/// F16C, not AVX-512). Subnormals, ties and infinities follow IEEE; a NaN stays a NaN of
/// its sign (quiet, payload dropped), so a cache row or query staged from a refused value
/// is never a plausible number.
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
        *o = f32::from_bits(u32::from(u16::from_le_bytes(*c)) << 16);
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
///   * rounding is round-half-to-even.
pub fn quantize_row_q8_2_x4_roundtrip(x: &[f32], out: &mut [f32]) {
    assert_eq!(x.len(), out.len(), "q8_2 round-trip needs matching lengths");
    assert!(x.len().is_multiple_of(32), "q8_2 blocks are 32 values");
    for (xb, ob) in x
        .as_chunks::<32>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<32>().0)
    {
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

/// Round-trip f32 activations through mainline ggml's x86 **Q8_0** activation quantization,
/// the column a Q2_0 weight's dot reads: `quantize_row_q8_0` (llama.cpp `f2cc7282c`
/// ggml-cpu/arch/x86/quants.c:302). Per 32 values `d = f16(amax / 127)`, rounded to nearest
/// even, and each code `round_half_even(v * id)` with `id = 127 / amax`. That is not
/// `quantize_row_q8_0_ref`'s rule, which codes with `1 / d` of the f32 `d = amax / 127`, before
/// its f16 rounding, and `roundf`. A block whose multiplier overflows (amax below about 2^-121)
/// restores to zero, as qdot's encoder flushes it.
pub fn quantize_row_q8_0_roundtrip(x: &[f32], out: &mut [f32]) {
    assert_eq!(x.len(), out.len(), "q8_0 round-trip needs matching lengths");
    assert!(x.len().is_multiple_of(32), "q8_0 blocks are 32 values");
    let (blocks, _) = x.as_chunks::<32>();
    for (xb, ob) in blocks.iter().zip(out.as_chunks_mut::<32>().0) {
        let amax = xb.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        let d = half_to_f32(f32_to_f16_bits(amax / 127.0));
        let id = if amax > 0.0 { 127.0 / amax } else { 0.0 };
        let id = if id.is_finite() { id } else { 0.0 };
        for (o, &v) in ob.iter_mut().zip(xb) {
            *o = d * (v * id).round_ties_even();
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
/// ggml-cpu.c:340); the other i-quants take Q8_K too and have no CPU matmul here. Q2_0, mainline's alone, takes Q8_0 (`f2cc7282c` ggml-cpu.c:237), the column
/// qdot's Q2_0 kernel reads. NVFP4 has a reference decode and no CPU matmul here.
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
    use crate::qtype::{HostAct, QROWS};

    /// `(id, ggml_type_name, ggml_blck_size, ggml_type_size)` for every id
    /// [`GgmlType`] names, as the compiled libraries answer: ik_llama.cpp
    /// `c10fbbcc` (`build/ggml/src/libggml.so`) and, for tq1_0/tq2_0, which
    /// the ik table does not have, mainline llama.cpp `930e2fa59`
    /// (`build/bin/libggml-base.so`), and for q2_0 mainline `f2cc7282c`. ik's
    /// and mainline's libraries agree on every other row here. nvfp4's row
    /// is read from llama.cpp-mainline's source (ggml.c:760-764,
    /// ggml-common.h:221-227), not a compiled library.
    const GGML_TABLE: &[(u32, &str, u64, u64)] = &[
        (0, "f32", 1, 4),
        (1, "f16", 1, 2),
        (2, "q4_0", 32, 18),
        (3, "q4_1", 32, 20),
        (6, "q5_0", 32, 22),
        (7, "q5_1", 32, 24),
        (8, "q8_0", 32, 34),
        (10, "q2_K", 256, 84),
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
        (40, "nvfp4", 64, 36),
        (42, "q2_0", 64, 18),
    ];

    /// Ids the enum leaves `Unknown`: types it does not name yet (q8_1, q8_K;
    /// mainline's q1_0 at 41, ik's q1_0_g128 at 41 and q8_2_x4
    /// at 99), slots ggml removed (4, 5, 31-33, 36-38; ik reuses 31-33 and
    /// 36 for repacked and BitNet types mainline does not have), and 43, the
    /// first id past mainline's table.
    // PIN(2026-09-24): 10 (q2_K) left this list when the enum named it for the GLM-5.3 MTP layer.
    // PIN(2026-10-12): 42 (q2_0) left this list when the enum named it.
    // PIN(2026-10-16): 40 (nvfp4) left this list when the enum named it for Nemotron 3.5 Bolt.
    // PIN(2026-10-18): 2 (q4_0) and 3 (q4_1) left this list when the enum named them, with a row each; both
    // are sized and refused by name.
    const UNKNOWN_IDS: &[u32] = &[4, 5, 9, 15, 31, 32, 33, 36, 37, 38, 41, 43, 99];

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

    /// The rows are ggml's table and nothing else: one row for each id of the table, in tag
    /// order, each with the table's name and sizes. A tag the enum does not name has no row,
    /// whatever tag an `Unknown` holds.
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
        for &(id, name, blck, bytes) in GGML_TABLE {
            let row = GgmlType::from_u32(id)
                .row()
                .unwrap_or_else(|| panic!("id {id} ({name}) has no row"));
            assert_eq!(
                (row.ty.as_u32(), row.name, row.blck, row.bytes),
                (id, name, blck, bytes),
                "id {id}"
            );
        }
        for id in 0..128 {
            let named = GGML_TABLE.iter().any(|t| t.0 == id);
            assert_eq!(GgmlType::from_u32(id).row().is_some(), named, "id {id}");
            assert!(GgmlType::Unknown(id).row().is_none(), "Unknown({id})");
        }
    }

    /// A row is what the dispatchers it feeds do: `dequant` is whether `dequant_row` decodes one
    /// zero block, a type a CPU matmul takes has a reference decode to be checked against, and
    /// the tag and the name round-trip through the enum.
    #[test]
    fn a_row_is_what_its_dispatchers_do() {
        for row in QROWS {
            let (ty, name) = (row.ty, row.name);
            assert_eq!(
                GgmlType::from_u32(ty.as_u32()),
                ty,
                "{name}: tag round trip"
            );
            assert_eq!(ty.to_string(), name, "{name}: Display");
            let zero_block = dequant_row(
                ty,
                &vec![0u8; row.bytes as usize],
                &mut vec![0.0f32; row.blck as usize],
            );
            assert_eq!(
                !matches!(zero_block, Err(QuantError::Unsupported(_))),
                row.dequant,
                "{name}: dequant"
            );
            assert!(
                row.dequant || row.host_act == HostAct::None,
                "{name}: a CPU matmul with no reference decode to check it"
            );
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
            (GgmlType::Q4_K, Some(Q8_2X4)),
            (GgmlType::Q5_K, Some(Q8_2X4)),
            (GgmlType::Q6_K, Some(Q8_2X4)),
            (GgmlType::Q5_0, Some(Q8_2X4)),
            (GgmlType::Q5_1, Some(Q8_2X4)),
            (GgmlType::Q8_0, Some(Q8_2X4)),
            (GgmlType::MXFP4, Some(Q8_2X4)),
            (GgmlType::Q2_0, Some(Q8_0)),
            (GgmlType::IQ4_XS, Some(Q8K)),
            (GgmlType::IQ4_NL, Some(Q8_2X4)),
            (GgmlType::F32, None),
            (GgmlType::F16, None),
        ] {
            assert_eq!(activation_format(ty), Ok(want), "{ty}");
        }
        assert_eq!(
            activation_format(GgmlType::BF16),
            Err(QuantError::NoActivationFormat(GgmlType::BF16))
        );
        // No CPU matmul takes NVFP4: dequant_row reads it, activation_format
        // refuses it by name, as it refuses BF16.
        assert_eq!(
            activation_format(GgmlType::NVFP4),
            Err(QuantError::NoActivationFormat(GgmlType::NVFP4))
        );
    }

    /// Every product the six i-quant/Q2_K ports form is exact in f32 — the
    /// module doc's claim that the multiply order cannot decide the bits —
    /// checked for every finite f16 scale `d` against every scale factor and
    /// code value the format can hold: the f32 product equals the f64 one.
    #[test]
    fn iq_products_are_exact() {
        use crate::iq_tables::{IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID, KVALUES_IQ4NL};
        // IQ2_XXS's grid bytes and factor are IQ2_XS's: both go through `g2` and `db2`.
        let mut g2: Vec<u8> = IQ2XS_GRID
            .iter()
            .chain(&IQ2XXS_GRID)
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

    /// The codebooks' structure, as ggml builds them: every IQ2_XS and
    /// IQ2_XXS grid byte is one of 8, 25, 43; every IQ3_XXS grid byte is 4 + 8k or 62; every
    /// IQ3_S grid byte is odd and at most 15; a `ksigns_iq2xs` entry is its index with bit 7 set to the parity of the
    /// seven, so the eight signs of a group always flip an even number.
    #[test]
    fn iq_tables_have_ggml_structure() {
        use crate::iq_tables::{
            IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS,
        };
        for &w in IQ2XS_GRID.iter().chain(&IQ2XXS_GRID) {
            assert!(
                w.to_le_bytes().iter().all(|b| [8, 25, 43].contains(b)),
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
}
