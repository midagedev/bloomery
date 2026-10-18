//! The generic host dot: a type with no hand kernel here runs its block decoder through one
//! kernel per activation form.
//!
//! [`HostDecode`] is what a type is on the host: the float factors of a block, the integer scale
//! and min of each 16 values, and the 32 signed codes of each 32-value sub-block, read the way
//! ggml's dequant reads them, `value = (d·scale)·code − dmin·min` with the multiplies in that
//! order. The kernels do the rest and know nothing of a type: [`dot_avx2`] and [`dot_scalar`]
//! take the codes against the activation column's codes (`maddubs`), apply the integer scales
//! (`madd`), and close each block with one float step.
//!
//! Two activation forms are taken, [`HostAct::Q8K`] (256-value blocks) and [`HostAct::Q8_2X4`]
//! (32-value blocks); [`Entry::of`] refuses a decoder of another form when it is built. The float
//! steps are the hand kernels' own, so a type already hand-written gives the same bits here:
//!
//! - Q8_K: per block, the min fold `fma(−dmin·dy, Σ min·bsum)` when the type has mins, then
//!   `fma(d·dy, Σ scale·pair)`, both on eight lanes where lane `l` holds bytes `4l..4l + 4` of
//!   every sub-block; the row closes through `hsum_float_8`.
//! - Q8_2_X4: per group of four blocks the four block sums folded across lanes and one
//!   `fma(dw·da, ·)`, per tail block `fma(dw·da, ·)`; the row closes through `hsum_float_8`.
//!
//! The decode is the only per-type code, in two forms. [`HostDecode::codes`] is the definition
//! (a scalar decode) and also what the default [`HostDecode::codes_avx2`] loads, so a type with
//! nothing else is correct on both paths at once, at the speed of its scalar decode. A type that
//! a placement leaves on the host overrides the vector forms; the vector decode and the scalar
//! one are tied bit for bit by the tests, so an override cannot drift from the definition.
//!
//! A type joins by an `impl HostDecode` and one [`Entry::of`] line in [`ENTRIES`]; its row
//! (`gguf::QROWS`) already carries the block size, bytes and activation form, which the entry is
//! checked against at build time. A type with a hand kernel is never listed: the hand arms of
//! the dispatch are matched first.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use gguf::iq_tables::KVALUES_IQ4NL;
use gguf::quant::half_to_f32;
use gguf::{GgmlType, HostAct};

use super::{
    Q8K_STRIDE, Q82_BLOCK, Q82X4_STRIDE, bf16_bits_to_f32, emul_add_epi32, emul_madd1,
    emul_maddubs, emul_sign_epi8, emul_unpackhi_epi32_i32, emul_unpackhi_epi64_i32,
    emul_unpacklo_epi32_i32, emul_unpacklo_epi64_i32, quantize_q8k_block, quantize_q82x4_col,
};
#[cfg(target_arch = "x86_64")]
use super::{
    hsum_float_8, lut4_codes, q2k_values, quantize_q8k_block_avx2, quantize_q82x4_col_avx2,
};

/// What a type is on the host.
///
/// A block holds [`Self::VALUES`] values in 32-value sub-blocks. The weight of value `v` in the
/// 16-value group `g` is `(d·scale(g))·code(v) − dmin·min(g)` in f32, as `head` gives `d` and
/// `dmin`. The codes are in `-127..=127`: the dot folds a code's sign into the activation and
/// `maddubs` must not saturate. A type with no integer scale (a 32-value block, whose only
/// factor is `d`) returns 1 from [`Self::scale`] and has no mins.
pub trait HostDecode {
    /// The type this decodes.
    const TY: GgmlType;
    /// The activation form the type's dot reads: [`HostAct::Q8K`] or [`HostAct::Q8_2X4`].
    const ACT: HostAct;
    /// Bytes in one block.
    const BYTES: usize;
    /// Values in one block: 256 for the Q8_K form, 32 for the Q8_2_X4 form.
    const VALUES: usize;
    /// Every code is non-negative, so the dot skips the sign fold.
    const UNSIGNED: bool = false;
    /// The type has a min term ([`Self::min`] and `dmin`); only the Q8_K form carries one.
    const MINS: bool = false;

    /// The block's float factors `(d, dmin)`; `dmin` is 0 for a type without mins.
    fn head(blk: &[u8]) -> (f32, f32);
    /// The integer scale of the 16-value group `group`.
    fn scale(blk: &[u8], group: usize) -> i32;
    /// The integer min of the 16-value group `group`.
    fn min(_blk: &[u8], _group: usize) -> i32 {
        0
    }
    /// Sub-block `sub`'s 32 codes, in value order.
    fn codes(blk: &[u8], sub: usize) -> [i8; 32];

    /// [`Self::codes`] in a register. The default loads the scalar decode.
    ///
    /// # Safety
    /// AVX2 must be available (taken from the caller, being `#[inline(always)]`); `blk` must be a
    /// whole block of the type and `sub` below `VALUES / 32`.
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn codes_avx2(blk: &[u8], sub: usize) -> __m256i {
        let codes = Self::codes(blk, sub);
        // SAFETY: 32 readable bytes of `codes`; the load is unaligned.
        unsafe { _mm256_loadu_si256(codes.as_ptr().cast()) }
    }

    /// Sub-block `sub`'s scales in a register: group `2·sub` on the eight low i16 lanes, group
    /// `2·sub + 1` on the eight high ones.
    ///
    /// # Safety
    /// AVX2 must be available (taken from the caller, being `#[inline(always)]`); `blk` must be a
    /// whole block of the type and `sub` below `VALUES / 32`.
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn scales_avx2(blk: &[u8], sub: usize) -> __m256i {
        let (lo, hi) = (Self::scale(blk, 2 * sub), Self::scale(blk, 2 * sub + 1));
        // SAFETY: register-only; AVX2 per contract.
        unsafe { _mm256_set_m128i(_mm_set1_epi16(hi as i16), _mm_set1_epi16(lo as i16)) }
    }
}

// ------------------------------------------------------------------ the entry

/// One decoder, ready for the dispatch: the type's block geometry and the kernels over it.
#[derive(Clone, Copy)]
pub struct Entry {
    /// The type the decoder reads.
    pub ty: GgmlType,
    /// Bytes in one block.
    pub bytes: usize,
    /// Values in one block.
    pub values: usize,
    /// The activation form the dot reads.
    pub act: HostAct,
    scalar: fn(&[u8], &[u8], usize) -> f32,
    #[cfg(target_arch = "x86_64")]
    avx2: unsafe fn(&[u8], &[u8], usize) -> f32,
}

/// The entries the dispatch falls to for a type with no hand kernel.
pub static ENTRIES: &[Entry] = &[];

const _: () = {
    let mut i = 0;
    while i < ENTRIES.len() {
        assert!(
            ENTRIES[i].agrees_with_row(),
            "a generic entry disagrees with its type's QROWS row (bytes, values or activation form)"
        );
        i += 1;
    }
};

/// The entry of `w` in [`ENTRIES`].
pub fn entry(w: GgmlType) -> Option<&'static Entry> {
    ENTRIES.iter().find(|e| e.ty == w)
}

const fn same_act(a: HostAct, b: HostAct) -> bool {
    matches!(
        (a, b),
        (HostAct::Q8K, HostAct::Q8K)
            | (HostAct::Q8_2X4, HostAct::Q8_2X4)
            | (HostAct::Q8_0, HostAct::Q8_0)
            | (HostAct::F32, HostAct::F32)
            | (HostAct::None, HostAct::None)
    )
}

impl Entry {
    /// The entry of decoder `D`. A form the kernels do not take, or a geometry the form cannot
    /// hold, stops the build.
    pub const fn of<D: HostDecode>() -> Entry {
        const { assert!(takes::<D>(), "{}", FORM_RULE) };
        Entry {
            ty: D::TY,
            bytes: D::BYTES,
            values: D::VALUES,
            act: D::ACT,
            scalar: dot_scalar::<D>,
            #[cfg(target_arch = "x86_64")]
            avx2: dot_avx2::<D>,
        }
    }

    /// Whether the type's row names this entry's block bytes, block values and activation form.
    pub const fn agrees_with_row(&self) -> bool {
        match self.ty.row() {
            Some(row) => {
                row.bytes == self.bytes as u64
                    && row.blck == self.values as u64
                    && same_act(row.host_act, self.act)
            }
            None => false,
        }
    }

    /// Bytes of one quantized activation column of `k` values in the entry's form.
    pub fn col_bytes(&self, k: usize) -> usize {
        match self.act {
            HostAct::Q8K => (k / 256) * Q8K_STRIDE,
            HostAct::Q8_2X4 => (k / 128) * Q82X4_STRIDE + ((k % 128) / 32) * Q82_BLOCK,
            other => unreachable!("an entry takes no {other:?} activation"),
        }
    }

    /// Quantizes one activation column into the entry's form; `avx2` picks the vector encoder
    /// (byte-identical to the scalar one). `out` is `col_bytes(x.len())` bytes.
    #[cfg_attr(
        not(target_arch = "x86_64"),
        expect(
            unused_variables,
            reason = "only the x86_64 build has a vector encoder"
        )
    )]
    pub fn quantize(&self, x: &[f32], out: &mut [u8], avx2: bool) {
        match self.act {
            HostAct::Q8K => {
                let (xblocks, _) = x.as_chunks::<256>();
                let (oblocks, _) = out.as_chunks_mut::<Q8K_STRIDE>();
                for (xb, ob) in xblocks.iter().zip(oblocks.iter_mut()) {
                    #[cfg(target_arch = "x86_64")]
                    if avx2 {
                        // SAFETY: the caller saw AVX2.
                        unsafe { quantize_q8k_block_avx2(xb, ob) };
                        continue;
                    }
                    quantize_q8k_block(xb, ob);
                }
            }
            HostAct::Q8_2X4 => {
                #[cfg(target_arch = "x86_64")]
                if avx2 {
                    // SAFETY: the caller saw AVX2.
                    unsafe { quantize_q82x4_col_avx2(x, out) };
                    return;
                }
                quantize_q82x4_col(x, out);
            }
            other => unreachable!("an entry takes no {other:?} activation"),
        }
    }

    /// The scalar mirror of the AVX2 dot over `nb` blocks, bit for bit.
    pub fn dot_scalar(&self, wrow: &[u8], acol: &[u8], nb: usize) -> f32 {
        (self.scalar)(wrow, acol, nb)
    }

    /// The AVX2 dot over `nb` blocks.
    ///
    /// # Safety
    /// AVX2 and FMA must be available; `wrow` must hold `nb` blocks and `acol` a column of
    /// `col_bytes(nb · values)` bytes.
    #[cfg(target_arch = "x86_64")]
    pub unsafe fn dot_avx2(&self, wrow: &[u8], acol: &[u8], nb: usize) -> f32 {
        // SAFETY: the contract above is this call's.
        unsafe { (self.avx2)(wrow, acol, nb) }
    }
}

// ----------------------------------------------------------------- the kernels

/// What the kernels take, for the build to refuse anything else.
const FORM_RULE: &str = "the generic kernels take the Q8_K form with 256-value blocks and the \
                         Q8_2_X4 form with 32-value blocks and no mins";

/// Whether the kernels take decoder `D`: its form and geometry agree.
const fn takes<D: HostDecode>() -> bool {
    match D::ACT {
        HostAct::Q8K => D::VALUES == 256,
        HostAct::Q8_2X4 => D::VALUES == 32 && !D::MINS,
        _ => false,
    }
}

/// The row dot of decoder `D` over `nb` blocks, scalar.
pub fn dot_scalar<D: HostDecode>(wrow: &[u8], acol: &[u8], nb: usize) -> f32 {
    const { assert!(takes::<D>(), "{}", FORM_RULE) };
    match D::ACT {
        HostAct::Q8K => dot_q8k_scalar::<D>(wrow, acol, nb),
        HostAct::Q8_2X4 => dot_q82x4_scalar::<D>(wrow, acol, nb),
        other => unreachable!("a generic decoder takes no {other:?} activation"),
    }
}

/// The row dot of decoder `D` over `nb` blocks, AVX2+FMA.
///
/// # Safety
/// AVX2 and FMA must be available; `wrow` must hold `nb` blocks of `D` and `acol` the activation
/// column of `nb · D::VALUES` values in `D::ACT`'s form.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn dot_avx2<D: HostDecode>(wrow: &[u8], acol: &[u8], nb: usize) -> f32 {
    const { assert!(takes::<D>(), "{}", FORM_RULE) };
    match D::ACT {
        // SAFETY: the contract above is the callee's.
        HostAct::Q8K => unsafe { dot_q8k_avx2::<D>(wrow, acol, nb) },
        // SAFETY: as above.
        HostAct::Q8_2X4 => unsafe { dot_q82x4_avx2::<D>(wrow, acol, nb) },
        other => unreachable!("a generic decoder takes no {other:?} activation"),
    }
}

/// `hsum_float_8` on an array: its reduction order.
fn hsum8(a: [f32; 8]) -> f32 {
    let t = [a[0] + a[4], a[1] + a[5], a[2] + a[6], a[3] + a[7]];
    (t[0] + t[2]) + (t[1] + t[3])
}

/// `maddubs`' two operands for one sub-block: `|w|` and the activation with `w`'s sign folded in
/// (the codes as they are, when the type is unsigned).
fn pair_operands<D: HostDecode>(codes: &[i8; 32], q8: &[u8]) -> ([u8; 32], [u8; 32]) {
    let abs_w = std::array::from_fn(|n| {
        if D::UNSIGNED {
            codes[n] as u8
        } else {
            codes[n].wrapping_abs() as u8
        }
    });
    let sgn_y = std::array::from_fn(|n| {
        if D::UNSIGNED {
            q8[n]
        } else {
            emul_sign_epi8(q8[n], codes[n])
        }
    });
    (abs_w, sgn_y)
}

/// Scalar mirror of [`dot_q8k_avx2`].
fn dot_q8k_scalar<D: HostDecode>(wrow: &[u8], acol: &[u8], nb: usize) -> f32 {
    let mut acc = [0.0f32; 8];
    for i in 0..nb {
        let blk = &wrow[D::BYTES * i..D::BYTES * (i + 1)];
        let (d, dmin) = D::head(blk);
        let col = &acol[Q8K_STRIDE * i..Q8K_STRIDE * (i + 1)];
        let dy = f32::from_le_bytes([col[0], col[1], col[2], col[3]]);
        if D::MINS {
            let bsum =
                |t: usize| i32::from(i16::from_le_bytes([col[264 + 2 * t], col[265 + 2 * t]]));
            for (l, a) in acc.iter_mut().enumerate() {
                let m = D::min(blk, 2 * l) * bsum(2 * l) + D::min(blk, 2 * l + 1) * bsum(2 * l + 1);
                *a = ((-dmin) * dy).mul_add(m as f32, *a);
            }
        }
        let mut sumi = [0i32; 8];
        for b in 0..8 {
            let (abs_w, sgn_y) =
                pair_operands::<D>(&D::codes(blk, b), &col[8 + 32 * b..8 + 32 * b + 32]);
            let pair = emul_maddubs(abs_w, sgn_y);
            for (l, s) in sumi.iter_mut().enumerate() {
                for p in [2 * l, 2 * l + 1] {
                    *s = s.wrapping_add(D::scale(blk, 2 * b + p / 8) * i32::from(pair[p]));
                }
            }
        }
        for (a, &si) in acc.iter_mut().zip(&sumi) {
            *a = (d * dy).mul_add(si as f32, *a);
        }
    }
    hsum8(acc)
}

/// 32 bytes at the head of `bytes`, unaligned.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn load256(bytes: &[u8]) -> __m256i {
    let head = &bytes[..32];
    // SAFETY: `head` is 32 readable bytes; the load is unaligned.
    unsafe { _mm256_loadu_si256(head.as_ptr().cast()) }
}

/// `maddubs(|w|, sign(q8, w))`, or `maddubs(w, q8)` for an unsigned type: the i16 pair sums of one
/// sub-block's codes against the column's.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn pair_sums<D: HostDecode>(codes: __m256i, q8: __m256i) -> __m256i {
    if D::UNSIGNED {
        _mm256_maddubs_epi16(codes, q8)
    } else {
        _mm256_maddubs_epi16(_mm256_sign_epi8(codes, codes), _mm256_sign_epi8(q8, codes))
    }
}

/// Q8_K form, AVX2+FMA. The min fold comes first, as in the hand kernels, then the eight
/// sub-blocks' `madd(scale, maddubs(|w|, sign(q8, w)))`, then the codes' fma.
///
/// # Safety
/// AVX2 and FMA must be available; `wrow` must hold `nb` blocks and `acol` `nb` Q8_K blocks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_q8k_avx2<D: HostDecode>(wrow: &[u8], acol: &[u8], nb: usize) -> f32 {
    let mut accd = _mm256_setzero_ps();
    for i in 0..nb {
        let blk = &wrow[D::BYTES * i..D::BYTES * (i + 1)];
        let (d, dmin) = D::head(blk);
        let col = &acol[Q8K_STRIDE * i..Q8K_STRIDE * (i + 1)];
        let dy = f32::from_le_bytes([col[0], col[1], col[2], col[3]]);
        if D::MINS {
            let mut mins = [0u8; 32];
            for g in 0..16 {
                mins[2 * g..2 * g + 2].copy_from_slice(&(D::min(blk, g) as i16).to_le_bytes());
            }
            let bsums = load256(&col[264..]);
            accd = _mm256_fmadd_ps(
                _mm256_set1_ps((-dmin) * dy),
                _mm256_cvtepi32_ps(_mm256_madd_epi16(load256(&mins), bsums)),
                accd,
            );
        }
        let mut sumi = _mm256_setzero_si256();
        for b in 0..8 {
            // SAFETY: AVX2 per contract; `blk` is a whole block and `b` is below 8.
            let (codes, scales) = unsafe { (D::codes_avx2(blk, b), D::scales_avx2(blk, b)) };
            let pair = pair_sums::<D>(codes, load256(&col[8 + 32 * b..]));
            sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(scales, pair));
        }
        accd = _mm256_fmadd_ps(_mm256_set1_ps(d * dy), _mm256_cvtepi32_ps(sumi), accd);
    }
    // SAFETY: AVX is on in this function.
    unsafe { hsum_float_8(accd) }
}

/// Scalar mirror of [`dot_q82x4_avx2`].
fn dot_q82x4_scalar<D: HostDecode>(wrow: &[u8], acol: &[u8], nb: usize) -> f32 {
    let mut acc = [0.0f32; 8];
    let block_lanes = |blk: &[u8], q8: &[u8]| -> [i32; 8] {
        let (abs_w, sgn_y) = pair_operands::<D>(&D::codes(blk, 0), q8);
        emul_madd1(emul_maddubs(abs_w, sgn_y))
    };
    let nbg = nb / 4;
    for i in 0..nbg {
        let g = &acol[i * Q82X4_STRIDE..(i + 1) * Q82X4_STRIDE];
        let mut p = [[0i32; 8]; 4];
        let mut dall = [0.0f32; 4];
        for (k, (pk, dk)) in p.iter_mut().zip(&mut dall).enumerate() {
            let blk = &wrow[D::BYTES * (4 * i + k)..D::BYTES * (4 * i + k + 1)];
            let da = bf16_bits_to_f32(u16::from_le_bytes([g[2 * k], g[2 * k + 1]]));
            *dk = D::head(blk).0 * da;
            *pk = block_lanes(blk, &g[16 + 32 * k..16 + 32 * k + 32]);
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
            *a = dall[l % 4].mul_add(pall[l] as f32, *a);
        }
    }
    let nb4 = 4 * nbg;
    for i in nb4..nb {
        let blk = &wrow[D::BYTES * i..D::BYTES * (i + 1)];
        let tb = nb4 / 4 * Q82X4_STRIDE + (i - nb4) * Q82_BLOCK;
        let tail = &acol[tb..tb + Q82_BLOCK];
        let da = bf16_bits_to_f32(u16::from_le_bytes([tail[0], tail[1]]));
        let d = D::head(blk).0 * da;
        let lanes = block_lanes(blk, &tail[4..36]);
        for (a, &p) in acc.iter_mut().zip(&lanes) {
            *a = d.mul_add(p as f32, *a);
        }
    }
    hsum8(acc)
}

/// Q8_2_X4 form, AVX2+FMA: per group of four blocks the four sign-folded `madd(1, maddubs)`
/// vectors folded across lanes by unpack and add, one fma with the four `dw·da` products; each
/// tail block one fma with its own `dw·da`.
///
/// # Safety
/// AVX2 and FMA must be available; `wrow` must hold `nb` blocks and `acol` the column's
/// `nb / 4` Q8_2_X4 groups followed by `nb % 4` Q8_2 tail blocks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_q82x4_avx2<D: HostDecode>(wrow: &[u8], acol: &[u8], nb: usize) -> f32 {
    let m1 = _mm256_set1_epi16(1);
    let mut acc = _mm256_setzero_ps();
    let nbg = nb / 4;
    for i in 0..nbg {
        let g = &acol[i * Q82X4_STRIDE..(i + 1) * Q82X4_STRIDE];
        let mut p = [_mm256_setzero_si256(); 4];
        let mut dw = [0.0f32; 4];
        for (k, (pk, dwk)) in p.iter_mut().zip(&mut dw).enumerate() {
            let blk = &wrow[D::BYTES * (4 * i + k)..D::BYTES * (4 * i + k + 1)];
            *dwk = D::head(blk).0;
            // SAFETY: AVX2 per contract; `blk` is a whole block.
            let codes = unsafe { D::codes_avx2(blk, 0) };
            let pair = pair_sums::<D>(codes, load256(&g[16 + 32 * k..]));
            *pk = _mm256_madd_epi16(m1, pair);
        }
        // The group's four bf16 scales are its first eight bytes.
        let aux = _mm_cvtsi64_si128(i64::from_le_bytes([
            g[0], g[1], g[2], g[3], g[4], g[5], g[6], g[7],
        ]));
        let aux_d = _mm_castsi128_ps(_mm_slli_epi32::<16>(_mm_cvtepu16_epi32(aux)));
        let m = _mm_mul_ps(_mm_set_ps(dw[3], dw[2], dw[1], dw[0]), aux_d);
        let dall = _mm256_set_m128(m, m);
        let p01 = _mm256_add_epi32(
            _mm256_unpacklo_epi32(p[0], p[1]),
            _mm256_unpackhi_epi32(p[0], p[1]),
        );
        let p23 = _mm256_add_epi32(
            _mm256_unpacklo_epi32(p[2], p[3]),
            _mm256_unpackhi_epi32(p[2], p[3]),
        );
        let pall = _mm256_add_epi32(
            _mm256_unpacklo_epi64(p01, p23),
            _mm256_unpackhi_epi64(p01, p23),
        );
        acc = _mm256_fmadd_ps(dall, _mm256_cvtepi32_ps(pall), acc);
    }
    let nb4 = 4 * nbg;
    for i in nb4..nb {
        let blk = &wrow[D::BYTES * i..D::BYTES * (i + 1)];
        let tb = nb4 / 4 * Q82X4_STRIDE + (i - nb4) * Q82_BLOCK;
        let da = bf16_bits_to_f32(u16::from_le_bytes([acol[tb], acol[tb + 1]]));
        let d = D::head(blk).0 * da;
        // SAFETY: AVX2 per contract; `blk` is a whole block.
        let codes = unsafe { D::codes_avx2(blk, 0) };
        let p0 = _mm256_madd_epi16(m1, pair_sums::<D>(codes, load256(&acol[tb + 4..])));
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(p0), acc);
    }
    // SAFETY: AVX is on in this function.
    unsafe { hsum_float_8(acc) }
}

// ---------------------------------------------------------------- the decoders

/// The little-endian f16 at `off`.
fn half_at(blk: &[u8], off: usize) -> f32 {
    half_to_f32(u16::from_le_bytes([blk[off], blk[off + 1]]))
}

/// The 32 values of a 4-bit sub-block through the signed `kvalues_iq4nl` table: the low nibble
/// of `qs[j]` is value `j`, the high nibble value `j + 16`.
fn iq4nl_codes(qs: &[u8]) -> [i8; 32] {
    std::array::from_fn(|n| {
        let nib = if n < 16 {
            qs[n] & 0x0f
        } else {
            qs[n - 16] >> 4
        };
        KVALUES_IQ4NL[usize::from(nib)]
    })
}

/// [`iq4nl_codes`] in a register.
///
/// # Safety
/// AVX2 must be available (taken from the caller, being `#[inline(always)]`); `qs` must hold 16
/// bytes.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn iq4nl_codes_avx2(qs: &[u8]) -> __m256i {
    // SAFETY: 16 readable bytes of the static table.
    let t128 = unsafe { _mm_loadu_si128(KVALUES_IQ4NL.as_ptr().cast()) };
    // SAFETY: register-only; AVX2 per contract.
    let table = unsafe { _mm256_set_m128i(t128, t128) };
    // SAFETY: 16 readable bytes of `qs` per contract; AVX2 per contract.
    unsafe { lut4_codes(qs.as_ptr(), _mm256_set1_epi8(0x0f), table) }
}

/// Q2_K (`block_q2_K`): scales[16] @0, one byte per 16 values with the scale in the low nibble
/// and the min in the high; qs[64] @16, sub-block `4j + k` being bits `2k` of qs bytes
/// `32j..32j + 32`; f16 d @80, f16 dmin @82.
pub struct Q2k;

impl HostDecode for Q2k {
    const TY: GgmlType = GgmlType::Q2_K;
    const ACT: HostAct = HostAct::Q8K;
    const BYTES: usize = 84;
    const VALUES: usize = 256;
    const UNSIGNED: bool = true;
    const MINS: bool = true;

    fn head(blk: &[u8]) -> (f32, f32) {
        (half_at(blk, 80), half_at(blk, 82))
    }
    fn scale(blk: &[u8], group: usize) -> i32 {
        i32::from(blk[group] & 0x0f)
    }
    fn min(blk: &[u8], group: usize) -> i32 {
        i32::from(blk[group] >> 4)
    }
    fn codes(blk: &[u8], sub: usize) -> [i8; 32] {
        let (j, k) = (sub / 4, sub % 4);
        std::array::from_fn(|n| ((blk[16 + 32 * j + n] >> (2 * k)) & 3) as i8)
    }
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn codes_avx2(blk: &[u8], sub: usize) -> __m256i {
        // SAFETY: AVX2 per contract; `blk` is a whole block and `sub` is below 8.
        unsafe { q2k_values(blk.as_ptr(), sub) }
    }
}

/// IQ4_XS (`block_iq4_xs`): f16 d @0, u16 scales_h @2, u8 scales_l[4] @4, qs[128] @8. Sub-block
/// `b` is the nibbles of qs[16b..16b + 16] through `kvalues_iq4nl`; its scale is the 6-bit
/// `ls[b]` (nibble `b` of scales_l, two bits of scales_h) minus 32.
pub struct Iq4Xs;

impl HostDecode for Iq4Xs {
    const TY: GgmlType = GgmlType::IQ4_XS;
    const ACT: HostAct = HostAct::Q8K;
    const BYTES: usize = 136;
    const VALUES: usize = 256;

    fn head(blk: &[u8]) -> (f32, f32) {
        (half_at(blk, 0), 0.0)
    }
    fn scale(blk: &[u8], group: usize) -> i32 {
        let b = group / 2;
        let scales_h = u16::from_le_bytes([blk[2], blk[3]]);
        let lo = i32::from((blk[4 + b / 2] >> (4 * (b % 2))) & 0x0f);
        let hi = i32::from((scales_h >> (2 * b)) & 3);
        (lo | (hi << 4)) - 32
    }
    fn codes(blk: &[u8], sub: usize) -> [i8; 32] {
        iq4nl_codes(&blk[8 + 16 * sub..8 + 16 * sub + 16])
    }
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn codes_avx2(blk: &[u8], sub: usize) -> __m256i {
        // SAFETY: AVX2 per contract.
        unsafe { iq4nl_codes_avx2(&blk[8 + 16 * sub..8 + 16 * sub + 16]) }
    }
}

/// IQ4_NL (`block_iq4_nl`): f16 d @0, qs[16] @2; one 32-value block, no integer scale.
pub struct Iq4Nl;

impl HostDecode for Iq4Nl {
    const TY: GgmlType = GgmlType::IQ4_NL;
    const ACT: HostAct = HostAct::Q8_2X4;
    const BYTES: usize = 18;
    const VALUES: usize = 32;

    fn head(blk: &[u8]) -> (f32, f32) {
        (half_at(blk, 0), 0.0)
    }
    fn scale(_blk: &[u8], _group: usize) -> i32 {
        1
    }
    fn codes(blk: &[u8], _sub: usize) -> [i8; 32] {
        iq4nl_codes(&blk[2..18])
    }
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn codes_avx2(blk: &[u8], _sub: usize) -> __m256i {
        // SAFETY: AVX2 per contract.
        unsafe { iq4nl_codes_avx2(&blk[2..18]) }
    }
}
