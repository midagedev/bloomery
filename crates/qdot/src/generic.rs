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
//! the dispatch are matched first. The listed types are IQ2_S and IQ1_M, each with its
//! vector decode; IQ1_S and IQ1_M put the group delta into the code (`8·grid ± 1`, float factor
//! `d / 8`), so they have no min term and read no column sums. Q2_K, IQ4_XS and IQ4_NL are the
//! decoders the tests hold to their hand kernels. IQ1_S's decoder
//! ([`Iq1s`]) is unlisted — its hand kernel owns the dispatch — but stays with the tests that
//! hold its decode, and the rate bin dots it as the pre-port arm.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use gguf::iq_tables::{IQ1S_GRID, IQ2S_GRID, KVALUES_IQ4NL};
use gguf::quant::half_to_f32;
use gguf::{GgmlType, HostAct};

use super::{
    Q8K_STRIDE, Q82_BLOCK, Q82X4_STRIDE, bf16_bits_to_f32, emul_add_epi32, emul_madd1,
    emul_maddubs, emul_sign_epi8, emul_unpackhi_epi32_i32, emul_unpackhi_epi64_i32,
    emul_unpacklo_epi32_i32, emul_unpacklo_epi64_i32, lut4_codes_scalar, quantize_q8k_block,
    quantize_q82x4_col,
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
pub static ENTRIES: &[Entry] = &[Entry::of::<Iq2s>(), Entry::of::<Iq1m>()];

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
    /// (byte-identical to the scalar one), and a CPU without AVX2 takes the scalar encoder
    /// whatever `avx2` says.
    ///
    /// # Panics
    /// When `x` is not a whole number of the entry's blocks, or `out` is not `col_bytes(x.len())`
    /// bytes.
    #[cfg_attr(
        not(target_arch = "x86_64"),
        expect(
            unused_variables,
            reason = "only the x86_64 build has a vector encoder"
        )
    )]
    pub fn quantize(&self, x: &[f32], out: &mut [u8], avx2: bool) {
        #[cfg(target_arch = "x86_64")]
        let avx2 = avx2 && std::arch::is_x86_feature_detected!("avx2");
        assert!(
            x.len().is_multiple_of(self.values),
            "generic qdot: incomplete activation block"
        );
        assert_eq!(
            out.len(),
            self.col_bytes(x.len()),
            "generic qdot: wrong activation buffer length"
        );
        match self.act {
            HostAct::Q8K => {
                let (xblocks, _) = x.as_chunks::<256>();
                let (oblocks, _) = out.as_chunks_mut::<Q8K_STRIDE>();
                for (xb, ob) in xblocks.iter().zip(oblocks.iter_mut()) {
                    #[cfg(target_arch = "x86_64")]
                    if avx2 {
                        // SAFETY: AVX2 was detected above.
                        unsafe { quantize_q8k_block_avx2(xb, ob) };
                        continue;
                    }
                    quantize_q8k_block(xb, ob);
                }
            }
            HostAct::Q8_2X4 => {
                #[cfg(target_arch = "x86_64")]
                if avx2 {
                    // SAFETY: AVX2 was detected above; the lengths were checked above.
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
    lut4_codes_scalar(qs, &KVALUES_IQ4NL.map(|c| c as u8)).map(|c| c as i8)
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

/// The 32 values of a sub-block as `q`'s grid magnitudes with the sign of bit `n` of `sign_bytes`
/// on value `n`: ik's `SignHelper::sign_4_values`, the four sign bytes' bits to `±1` per value
/// through `sign_epi8`.
///
/// # Safety
/// AVX2 must be available (taken from the caller, being `#[inline(always)]`).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn signed_by_bits(q: __m256i, sign_bytes: u32) -> __m256i {
    // SAFETY: register-only; AVX2 per contract.
    unsafe {
        // Value n of the 32 selects sign byte n / 8, bit n % 8.
        let pick = _mm256_set_epi64x(
            0x0303_0303_0303_0303,
            0x0202_0202_0202_0202,
            0x0101_0101_0101_0101,
            0,
        );
        let bit = _mm256_set1_epi64x(0x8040_2010_0804_0201u64 as i64);
        let bits = _mm256_and_si256(
            _mm256_shuffle_epi8(_mm256_set1_epi32(sign_bytes as i32), pick),
            bit,
        );
        let signs = _mm256_or_si256(_mm256_cmpeq_epi8(bits, bit), _mm256_set1_epi8(1));
        _mm256_sign_epi8(q, signs)
    }
}

/// An IQ1 sub-block's 32 codes `8·g + delta`: `g` in `-1..=1` is the byte of the four `grid`
/// entries of [`IQ1S_GRID`] and `delta` is `±1` on each byte.
///
/// # Safety
/// AVX2 must be available (taken from the caller, being `#[inline(always)]`); every `grid` entry
/// must be below 2048.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn iq1_codes_avx2(grid: [usize; 4], delta: __m256i) -> __m256i {
    // SAFETY: register-only; AVX2 and the index bound per contract.
    unsafe {
        let g = _mm256_set_epi64x(
            IQ1S_GRID[grid[3]] as i64,
            IQ1S_GRID[grid[2]] as i64,
            IQ1S_GRID[grid[1]] as i64,
            IQ1S_GRID[grid[0]] as i64,
        );
        // Bytes 0, 1 and 0xff times eight: the shift moves a byte's low bits up and the mask drops
        // the bits that crossed from the byte below.
        let eights = _mm256_and_si256(_mm256_slli_epi16::<3>(g), _mm256_set1_epi8(0xF8u8 as i8));
        _mm256_add_epi8(eights, delta)
    }
}

/// IQ2_S (`block_iq2_s`): f16 d @0, grid index bytes qs[32] @2, sign bytes @34, qh[8] @66,
/// scales[8] @74. Group `l` (eight values) of sub-block `b` reads the grid entry
/// `qs[4b + l] | (qh[b] << (8 - 2l)) & 0x300` and the sign byte `34 + 4b + l`; the 16 values of
/// groups `2h` and `2h + 1` share the nibble `h` of `scales[b]`. A value is
/// `d·(0.5 + s)·0.25·grid·sign`, so the code is the signed grid byte (at most 43), the scale
/// `2s + 1` and the float factor `d / 8`.
pub struct Iq2s;

impl Iq2s {
    /// The grid entry of group `l` of sub-block `sub`.
    fn index(blk: &[u8], sub: usize, l: usize) -> usize {
        usize::from(blk[2 + 4 * sub + l]) | ((usize::from(blk[66 + sub]) << (8 - 2 * l)) & 0x300)
    }
}

impl HostDecode for Iq2s {
    const TY: GgmlType = GgmlType::IQ2_S;
    const ACT: HostAct = HostAct::Q8K;
    const BYTES: usize = 82;
    const VALUES: usize = 256;

    fn head(blk: &[u8]) -> (f32, f32) {
        (half_at(blk, 0) * 0.125, 0.0)
    }
    fn scale(blk: &[u8], group: usize) -> i32 {
        2 * i32::from((blk[74 + group / 2] >> (4 * (group % 2))) & 0x0f) + 1
    }
    fn codes(blk: &[u8], sub: usize) -> [i8; 32] {
        std::array::from_fn(|n| {
            let (l, j) = (n / 8, n % 8);
            let mag = IQ2S_GRID[Self::index(blk, sub, l)].to_le_bytes()[j] as i8;
            if blk[34 + 4 * sub + l] >> j & 1 != 0 {
                -mag
            } else {
                mag
            }
        })
    }
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn codes_avx2(blk: &[u8], sub: usize) -> __m256i {
        let at = 34 + 4 * sub;
        let signs = u32::from_le_bytes([blk[at], blk[at + 1], blk[at + 2], blk[at + 3]]);
        // SAFETY: register-only; AVX2 per contract. Each index is ten bits, below the grid's 1024
        // entries.
        unsafe {
            let q = _mm256_set_epi64x(
                IQ2S_GRID[Self::index(blk, sub, 3)] as i64,
                IQ2S_GRID[Self::index(blk, sub, 2)] as i64,
                IQ2S_GRID[Self::index(blk, sub, 1)] as i64,
                IQ2S_GRID[Self::index(blk, sub, 0)] as i64,
            );
            signed_by_bits(q, signs)
        }
    }
}

/// IQ1_S (`block_iq1_s`): f16 d @0, qs[32] @2, qh u16[8] @34. Group `l` of sub-block `b` reads
/// the grid entry `qs[4b + l] | (qh[b] >> 3l & 7) << 8` (eight values in `-1..=1`); bits 12..15
/// of `qh[b]` are the scale `s` and bit 15 the sign of the delta. A value is
/// `d·(2s + 1)·(grid ± 1/8)`, so the code is `8·grid ± 1` (at most 9), the scale `2s + 1` and
/// the float factor `d / 8`: the delta is inside the integer sum, which needs no min term.
pub struct Iq1s;

impl Iq1s {
    fn qh(blk: &[u8], sub: usize) -> u16 {
        u16::from_le_bytes([blk[34 + 2 * sub], blk[35 + 2 * sub]])
    }
    /// The grid entry of group `l` of sub-block `sub`.
    fn index(blk: &[u8], sub: usize, l: usize) -> usize {
        usize::from(blk[2 + 4 * sub + l]) | (usize::from((Self::qh(blk, sub) >> (3 * l)) & 7) << 8)
    }
    /// The delta of sub-block `sub`: 1, or -1 when bit 15 of its `qh` is set.
    fn delta(blk: &[u8], sub: usize) -> i8 {
        if Self::qh(blk, sub) & 0x8000 != 0 {
            -1
        } else {
            1
        }
    }
}

impl HostDecode for Iq1s {
    const TY: GgmlType = GgmlType::IQ1_S;
    const ACT: HostAct = HostAct::Q8K;
    const BYTES: usize = 50;
    const VALUES: usize = 256;

    fn head(blk: &[u8]) -> (f32, f32) {
        (half_at(blk, 0) * 0.125, 0.0)
    }
    fn scale(blk: &[u8], group: usize) -> i32 {
        2 * i32::from((Self::qh(blk, group / 2) >> 12) & 7) + 1
    }
    fn codes(blk: &[u8], sub: usize) -> [i8; 32] {
        let delta = Self::delta(blk, sub);
        std::array::from_fn(|n| {
            let g = IQ1S_GRID[Self::index(blk, sub, n / 8)].to_le_bytes()[n % 8] as i8;
            8 * g + delta
        })
    }
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn codes_avx2(blk: &[u8], sub: usize) -> __m256i {
        // SAFETY: register-only; AVX2 per contract.
        unsafe {
            iq1_codes_avx2(
                std::array::from_fn(|l| Self::index(blk, sub, l)),
                _mm256_set1_epi8(Self::delta(blk, sub)),
            )
        }
    }
}

/// IQ1_M (`block_iq1_m`): qs[32] @0, qh[16] @32, scales[8] @48 read as four u16 `sc`. The f16
/// `d` has no field: its nibbles are the top nibbles of `sc[0..4]`, low nibble first. Group `l`
/// of sub-block `b` reads the grid entry `qs[4b + l]` with three high bits from `qh[2b + l / 2]`
/// (bits 0..3 for even `l`, 4..7 for odd) and a delta sign from bit 3 (even) or bit 7 (odd) of
/// that byte; the 16 values of groups `2h` and `2h + 1` have the scale
/// `2·(sc[b / 2] >> (6·(b % 2) + 3h) & 7) + 1`. Code, scale and float factor are IQ1_S's.
pub struct Iq1m;

impl Iq1m {
    fn sc(blk: &[u8], i: usize) -> u16 {
        u16::from_le_bytes([blk[48 + 2 * i], blk[49 + 2 * i]])
    }
    /// The block's f16 scale, from the top nibbles of the four `sc`.
    fn d_bits(blk: &[u8]) -> u16 {
        (Self::sc(blk, 0) >> 12)
            | ((Self::sc(blk, 1) >> 8) & 0x00f0)
            | ((Self::sc(blk, 2) >> 4) & 0x0f00)
            | (Self::sc(blk, 3) & 0xf000)
    }
    /// The grid entry of group `l` of sub-block `sub`, and whether its delta is negative.
    fn group(blk: &[u8], sub: usize, l: usize) -> (usize, bool) {
        let h = blk[32 + 2 * sub + l / 2];
        let (high, flag) = if l.is_multiple_of(2) {
            (h & 0x07, 0x08)
        } else {
            ((h >> 4) & 0x07, 0x80)
        };
        (
            usize::from(blk[4 * sub + l]) | (usize::from(high) << 8),
            h & flag != 0,
        )
    }
}

impl HostDecode for Iq1m {
    const TY: GgmlType = GgmlType::IQ1_M;
    const ACT: HostAct = HostAct::Q8K;
    const BYTES: usize = 56;
    const VALUES: usize = 256;

    fn head(blk: &[u8]) -> (f32, f32) {
        (half_to_f32(Self::d_bits(blk)) * 0.125, 0.0)
    }
    fn scale(blk: &[u8], group: usize) -> i32 {
        let sub = group / 2;
        let shift = 6 * (sub % 2) + 3 * (group % 2);
        2 * i32::from((Self::sc(blk, sub / 2) >> shift) & 7) + 1
    }
    fn codes(blk: &[u8], sub: usize) -> [i8; 32] {
        std::array::from_fn(|n| {
            let (index, negative) = Self::group(blk, sub, n / 8);
            let g = IQ1S_GRID[index].to_le_bytes()[n % 8] as i8;
            8 * g + if negative { -1 } else { 1 }
        })
    }
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn codes_avx2(blk: &[u8], sub: usize) -> __m256i {
        let groups: [(usize, bool); 4] = std::array::from_fn(|l| Self::group(blk, sub, l));
        // A group's delta byte, eight times: 1, or -1 for a negative one.
        let delta = |negative: bool| {
            if negative {
                -1i64
            } else {
                0x0101_0101_0101_0101
            }
        };
        // SAFETY: register-only; AVX2 per contract.
        unsafe {
            iq1_codes_avx2(
                groups.map(|(index, _)| index),
                _mm256_set_epi64x(
                    delta(groups[3].1),
                    delta(groups[2].1),
                    delta(groups[1].1),
                    delta(groups[0].1),
                ),
            )
        }
    }
}
