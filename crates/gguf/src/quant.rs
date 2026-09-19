//! Scalar reference dequantization for the ggml tensor types stage 1 needs.
//!
//! This is the *reference* path, not the fast path — the fast paths live in
//! `crates/q3k-cpu` (AVX2) and `crates/q3k-gemv` (CUDA) and are untouched.
//! Every decode below is a port of the corresponding
//! `dequantize_row_*` in the vendored ik_llama.cpp checkout
//! (`$IK/ggml/src/ggml-quants.c`, IK=/home/user/ik_llama.cpp), line numbers
//! cited per function. Nothing here is invented.
//!
//! FMA parity (measured on the box, 2026-09-19, `objdump -d --disassemble=…`
//! on `$IK/build/ggml/src/libggml.so`): the compiled q5_1 uses
//! `vfmadd132ps` (x0*d + m fused) and q4_K uses `vfmsub132ps` (q*d1 - m1
//! fused); q5_0, q3_K and q6_K contain no fused ops. The Rust ports mirror
//! exactly that: `mul_add` where the library fused, separate multiplies
//! where it did not. Diverging here would show up as ~1-ulp differences
//! against the oracle — the gate is 1e-6 absolute, so the mirror matters.

use std::fmt;

/// GGUF v3 / ggml tensor type tags, values from `enum ggml_type`
/// (ggml.h:391; F32=0 … Q6_K=14 seen at ggml.h:392-404).
///
/// `Unknown` carries any tag this build does not model; the loader accepts
/// such tensors only far enough to name them, and every size/dequant entry
/// point rejects them with an error.
// Variant names mirror ggml's type names verbatim (Q3_K, not Q3K), the way
// libc-style bindings keep the C spelling.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum GgmlType {
    F32,
    F16,
    Q5_0,
    Q5_1,
    Q3_K,
    Q4_K,
    Q5_K, // tag 13, present in the enum for the citation table only
    Q6_K,
    Unknown(u32),
}

impl GgmlType {
    pub fn from_u32(v: u32) -> Self {
        match v {
            0 => GgmlType::F32,
            1 => GgmlType::F16,
            6 => GgmlType::Q5_0,
            7 => GgmlType::Q5_1,
            11 => GgmlType::Q3_K,
            12 => GgmlType::Q4_K,
            13 => GgmlType::Q5_K,
            14 => GgmlType::Q6_K,
            other => GgmlType::Unknown(other),
        }
    }

    pub fn as_u32(self) -> u32 {
        match self {
            GgmlType::F32 => 0,
            GgmlType::F16 => 1,
            GgmlType::Q5_0 => 6,
            GgmlType::Q5_1 => 7,
            GgmlType::Q3_K => 11,
            GgmlType::Q4_K => 12,
            GgmlType::Q5_K => 13,
            GgmlType::Q6_K => 14,
            GgmlType::Unknown(v) => v,
        }
    }

    /// `ggml_type_name` string (type_traits table, ggml.c:620 — entries at
    /// ggml.c:657/667/756/777/912/938/998). Used to name the oracle dumps
    /// `$MULLE_DATA/ref/<name>.raw`.
    pub fn name(self) -> Option<&'static str> {
        match self {
            GgmlType::F32 => Some("f32"),
            GgmlType::F16 => Some("f16"),
            GgmlType::Q5_0 => Some("q5_0"),
            GgmlType::Q5_1 => Some("q5_1"),
            GgmlType::Q3_K => Some("q3_K"),
            GgmlType::Q4_K => Some("q4_K"),
            GgmlType::Q5_K => Some("q5_K"),
            GgmlType::Q6_K => Some("q6_K"),
            GgmlType::Unknown(_) => None,
        }
    }

    /// `blck_size` from ggml's type_traits table (ggml.c:620): F32/F16 = 1
    /// (ggml.c:657/667), Q5_0/Q5_1 = QK5_0/QK5_1 = 32 (ggml.c:756/777,
    /// QK5_0/QK5_1 at ggml-common.h:195/210), K-quants = QK_K = 256
    /// (ggml.c:912/938/998, QK_K at ggml-common.h:79).
    ///
    /// This match is the single owner of those numbers for the Rust side.
    pub fn blck_size(self) -> Option<u64> {
        match self {
            GgmlType::F32 | GgmlType::F16 => Some(1),
            GgmlType::Q5_0 | GgmlType::Q5_1 => Some(32),
            GgmlType::Q3_K | GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q6_K => Some(256),
            GgmlType::Unknown(_) => None,
        }
    }

    /// `type_size` (bytes per block) from the same type_traits table:
    /// `sizeof(float)` = 4 (F32), `sizeof(ggml_fp16_t)` = 2 (F16),
    /// `sizeof(block_q5_0)` = 22, `sizeof(block_q5_1)` = 24,
    /// `sizeof(block_q3_K)` = 110, `sizeof(block_q4_K)` = 144,
    /// `sizeof(block_q5_K)` = 176, `sizeof(block_q6_K)` = 210 —
    /// block layouts and static_asserts in ggml-common.h:327-332 (q3_K),
    /// 348-353 (q4_K), 373-378 (q5_K), 388-393 (q6_K), 196-216 (q5_0/q5_1).
    pub fn type_size(self) -> Option<u64> {
        match self {
            GgmlType::F32 => Some(4),
            GgmlType::F16 => Some(2),
            GgmlType::Q5_0 => Some(22),
            GgmlType::Q5_1 => Some(24),
            GgmlType::Q3_K => Some(110),
            GgmlType::Q4_K => Some(144),
            GgmlType::Q5_K => Some(176),
            GgmlType::Q6_K => Some(210),
            GgmlType::Unknown(_) => None,
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
        GgmlType::Q6_K => dequant_q6_k(&src[..need], dst),
        GgmlType::Q5_K | GgmlType::Unknown(_) => return Err(QuantError::Unsupported(ty)),
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
        let scales: &[u8; 12] = blk[4..16].try_into().unwrap();
        let qs = &blk[16..144];

        let mut is = 0usize;
        for j in 0..4 {
            let (sc, mi) = get_scale_min_k4(is, scales);
            let d1 = d * sc as f32;
            let m1 = dmin * mi as f32;
            let (sc, mi) = get_scale_min_k4(is + 1, scales);
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
