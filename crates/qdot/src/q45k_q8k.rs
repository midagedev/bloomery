//! Q4_K and Q5_K weight rows against q8_K activation columns: the rule the
//! host tier's routed down reads (mainline's `vec_dot_type` for both types),
//! as a column tile of 1..=[`TILE_COLS`] columns and its scalar mirror.
//!
//! The rule, per super-block `b` of 256 values, sub-block `s` of 32 (scale
//! `sc_s`, min `m_s`, six bits each), weight code `q_i` (0..15, or 0..31 with
//! Q5_K's high bit) and column code `a_i` (i8) with the column's f32 scale
//! `dy_b`:
//!
//! - `sumi_b = Σ_i sc_(i/32) · q_i · a_i` and `mini_b = Σ_i m_(i/32) · a_i`,
//!   both exact in i32 (order free);
//! - `acc = fma(dy_b · d_b, sumi_b as f32, acc)`, then
//!   `acc = fma(−(dy_b · dmin_b), mini_b as f32, acc)`, from `acc = 0` in
//!   super-block order; the value is `acc`.
//!
//! A column is [`quantize_col_q8k`]'s bytes: `block_q8_K` of 296 bytes, f32
//! `d` at 0, codes at 8..264, the 16 sums of 16 codes as i16 at 264..296. The
//! tile reads the sums for the min term; the mirror forms it from the codes,
//! so the two agree only on a column whose sums are its codes' — every column
//! the encoder writes.

use std::arch::x86_64::*;

use gguf::GgmlType;
use gguf::quant::half_to_f32;

use super::{
    Q4K_BLOCK, Q5K_BLOCK, Q8K_STRIDE, QdotError, TILE_COLS, hsum_i32, hsum8_i32, make_q4_scales,
    quantize_q8k_block, quantize_q8k_block_avx2,
};

/// Whether `w` has a q8_K column kernel here (Q4_K and Q5_K). On a CPU
/// without AVX2+FMA+F16C the kernel is the scalar mirror.
#[must_use]
pub fn has_q8k_kernel(w: GgmlType) -> bool {
    matches!(w, GgmlType::Q4_K | GgmlType::Q5_K)
}

/// Bytes of one q8_K column of `k` values. Panics by name when `k` is not
/// whole 256-value super-blocks.
#[must_use]
pub fn q8k_col_bytes(k: usize) -> usize {
    assert!(
        k.is_multiple_of(256),
        "q8_K activation columns are whole 256-value super-blocks, k = {k}"
    );
    (k / 256) * Q8K_STRIDE
}

/// One activation column into q8_K blocks ([`q8k_col_bytes`] of `x.len()`
/// bytes, else a named panic). A block holding a NaN or an infinity is refused
/// by name, never encoded. The AVX2 encoder when the CPU has it — bit for bit
/// the scalar one ([`quantize_col_q8k_scalar`]).
pub fn quantize_col_q8k(x: &[f32], out: &mut [u8]) {
    check_q8k_col(x, out);
    let (xblocks, _) = x.as_chunks::<256>();
    let (oblocks, _) = out.as_chunks_mut::<296>();
    if std::arch::is_x86_feature_detected!("avx2") {
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

/// The scalar encoder behind [`quantize_col_q8k`].
pub fn quantize_col_q8k_scalar(x: &[f32], out: &mut [u8]) {
    check_q8k_col(x, out);
    let (xblocks, _) = x.as_chunks::<256>();
    let (oblocks, _) = out.as_chunks_mut::<296>();
    for (xb, ob) in xblocks.iter().zip(oblocks.iter_mut()) {
        quantize_q8k_block(xb, ob);
    }
}

/// The encoders' shape contract. A non-finite value is refused inside each
/// block's encoder.
fn check_q8k_col(x: &[f32], out: &[u8]) {
    assert_eq!(
        out.len(),
        q8k_col_bytes(x.len()),
        "out must be q8k_col_bytes(x.len())"
    );
}

/// One Q4_K or Q5_K weight row against up to [`TILE_COLS`] q8_K columns:
/// `out[c]` is the module's rule for column `acols[c]`, each column bit for
/// bit what a call of any other width gives it and what
/// [`dot_q45k_q8k_cols_scalar`] gives. A column must come from
/// [`quantize_col_q8k`] (the tile reads its code sums). Refused by name: a
/// type without a q8_K kernel, a column count outside `1..=TILE_COLS` or an
/// `out` of another length, a `k` off the 256-value grid, a short row or a
/// short column.
pub fn dot_q45k_q8k_cols(
    w: GgmlType,
    wrow: &[u8],
    acols: &[&[u8]],
    k: usize,
    out: &mut [f32],
) -> Result<(), QdotError> {
    let nb = check_cols(w, wrow.len(), acols, k, out.len())?;
    if !(std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("fma")
        && std::arch::is_x86_feature_detected!("f16c"))
    {
        scalar(w, wrow, acols, nb, out);
        return Ok(());
    }
    match acols.len() {
        1 => tile::<1>(w, wrow, acols, nb, out),
        2 => tile::<2>(w, wrow, acols, nb, out),
        3 => tile::<3>(w, wrow, acols, nb, out),
        4 => tile::<4>(w, wrow, acols, nb, out),
        5 => tile::<5>(w, wrow, acols, nb, out),
        6 => tile::<6>(w, wrow, acols, nb, out),
        7 => tile::<7>(w, wrow, acols, nb, out),
        _ => tile::<8>(w, wrow, acols, nb, out),
    }
    Ok(())
}

/// The scalar mirror behind [`dot_q45k_q8k_cols`]: the rule as the module
/// doc states it, the min term formed from the codes. The fallback without
/// AVX2+FMA+F16C, and the gate's second path.
pub fn dot_q45k_q8k_cols_scalar(
    w: GgmlType,
    wrow: &[u8],
    acols: &[&[u8]],
    k: usize,
    out: &mut [f32],
) -> Result<(), QdotError> {
    let nb = check_cols(w, wrow.len(), acols, k, out.len())?;
    scalar(w, wrow, acols, nb, out);
    Ok(())
}

/// The shape contract of both entries; the super-block count on success.
fn check_cols(
    w: GgmlType,
    wrow_len: usize,
    acols: &[&[u8]],
    k: usize,
    outs: usize,
) -> Result<usize, QdotError> {
    let block = match w {
        GgmlType::Q4_K => Q4K_BLOCK,
        GgmlType::Q5_K => Q5K_BLOCK,
        _ => return Err(QdotError::NoQ8kKernel(w)),
    };
    let c = acols.len();
    if c == 0 || c > TILE_COLS || outs != c {
        return Err(QdotError::TileShape { cols: c, outs });
    }
    if !k.is_multiple_of(256) {
        return Err(QdotError::UnalignedK { k, gran: 256 });
    }
    let nb = k / 256;
    if wrow_len < nb * block {
        return Err(QdotError::ShortWeightRow {
            have: wrow_len,
            need: nb * block,
            k,
        });
    }
    let need = nb * Q8K_STRIDE;
    if let Some(a) = acols.iter().find(|a| a.len() < need) {
        return Err(QdotError::ShortActivationCol {
            have: a.len(),
            need,
            k,
        });
    }
    Ok(nb)
}

/// The `C`-column tile over validated inputs.
fn tile<const C: usize>(w: GgmlType, wrow: &[u8], acols: &[&[u8]], nb: usize, out: &mut [f32]) {
    let cols: [*const u8; C] = std::array::from_fn(|j| acols[j].as_ptr());
    let v = if w == GgmlType::Q5_K {
        // SAFETY: the caller saw AVX2+FMA+F16C; check_cols sized the row for
        // nb Q5_K super-blocks and every column for nb q8_K blocks.
        unsafe { q45k_q8k_tile_avx2::<C, true>(wrow, &cols, nb) }
    } else {
        // SAFETY: as the Q5_K arm, for Q4_K.
        unsafe { q45k_q8k_tile_avx2::<C, false>(wrow, &cols, nb) }
    };
    out.copy_from_slice(&v);
}

/// Six-bit scale and min of each of a super-block's eight sub-blocks.
fn scales_mins(blk: &[u8]) -> ([i32; 8], [i32; 8]) {
    let u = make_q4_scales(&blk[4..16]);
    let byte = |w: u32, i: usize| ((w >> (8 * i)) & 0xFF) as i32;
    let sc = std::array::from_fn(|s| byte(u[s / 4], s % 4));
    let m = std::array::from_fn(|s| byte(u[2 + s / 4], s % 4));
    (sc, m)
}

/// Code `i` of a Q4_K (`Q5 = false`) or Q5_K super-block: chunk `i / 64`'s
/// 32 bytes, the low nibble for its first 32 values and the high nibble for
/// the rest; Q5_K adds bit `i / 32` of `qh[i % 32]` as bit 4.
fn code(blk: &[u8], q5: bool, i: usize) -> i32 {
    let qs = if q5 { 48 } else { 16 };
    let (chunk, r) = (i / 64, i % 64);
    let byte = blk[qs + 32 * chunk + r % 32];
    let low = i32::from(if r < 32 { byte & 0xF } else { byte >> 4 });
    if q5 {
        low | (i32::from((blk[16 + r % 32] >> (i / 32)) & 1) << 4)
    } else {
        low
    }
}

/// The rule, column by column, over validated inputs.
fn scalar(w: GgmlType, wrow: &[u8], acols: &[&[u8]], nb: usize, out: &mut [f32]) {
    let (block, q5) = if w == GgmlType::Q5_K {
        (Q5K_BLOCK, true)
    } else {
        (Q4K_BLOCK, false)
    };
    for (o, acol) in out.iter_mut().zip(acols) {
        let mut acc = 0.0f32;
        for sb in 0..nb {
            let blk = &wrow[sb * block..(sb + 1) * block];
            let (sc, m) = scales_mins(blk);
            let d = half_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
            let dmin = half_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
            let a = &acol[sb * Q8K_STRIDE..(sb + 1) * Q8K_STRIDE];
            let dy = f32::from_le_bytes([a[0], a[1], a[2], a[3]]);
            let (mut sumi, mut mini) = (0i32, 0i32);
            for i in 0..256 {
                let ai = i32::from(a[8 + i] as i8);
                sumi += sc[i / 32] * code(blk, q5, i) * ai;
                mini += m[i / 32] * ai;
            }
            acc = (dy * d).mul_add(sumi as f32, acc);
            acc = (-(dy * dmin)).mul_add(mini as f32, acc);
        }
        *o = acc;
    }
}

/// Byte pairs `[2s, 2s + 1]` sixteen times: sub-block `s`'s i16 scale from
/// a vector of the eight scales, broadcast.
static SCALE_SHUF: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut s = 0;
    while s < 8 {
        let mut b = 0;
        while b < 32 {
            t[32 * s + b] = (2 * s + b % 2) as u8;
            b += 1;
        }
        s += 1;
    }
    t
};

/// Q5_K's bit 4 of one 32-value sub-block: `base` holds the sub-block's qh
/// bit at bit `4 − S` of each byte.
///
/// # Safety
/// AVX2 must be available on the target.
#[inline(always)]
unsafe fn q5_bit<const S: i32>(v: __m256i, base: __m256i, mh: __m256i) -> __m256i {
    // SAFETY: register-only intrinsics, no memory access.
    unsafe { _mm256_or_si256(v, _mm256_and_si256(_mm256_slli_epi16::<S>(base), mh)) }
}

/// AVX2 tile: one Q4_K (`Q5 = false`) or Q5_K weight row against `C` q8_K
/// columns, value `c` the rule of column `c`, equal to the scalar mirror bit
/// for bit.
///
/// Per super-block the weight's scales, mins, `d` and `dmin` are read once,
/// each 64-value chunk's codes unpacked once and each sub-block's scale
/// broadcast once, then each column takes, per sub-block, one maddubs of the
/// codes against its codes (pairs within 2·31·128), one madd with the scale
/// (pairs within 2·63·7936) and one add into its `sumi`, and one madd of the
/// doubled mins against its 16 code sums for its `mini`. Both are exact:
/// Q4_K's |sumi| stays under 63·15·128·256 < 2^31 (it can pass 2^24, where the
/// convert below rounds as `as f32` does), Q5_K's under 63·31·128·256, |mini|
/// under 63·32768·16. The integers are reduced across lanes by the 8×8
/// transpose-add ([`hsum8_i32`]) — `sumi` and `mini` of up to four columns in
/// one, else one each; a lone column's by two lane sums — and lane `c` takes
/// the rule's two fmas in super-block order. Lanes past `C` see a zero `dy`
/// and are never returned.
///
/// # Safety
/// CPU must support AVX2+FMA+F16C; `wrow` must hold `nb` super-blocks of its
/// type and each `acols[c]` point at `nb` readable q8_K blocks.
#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
unsafe fn q45k_q8k_tile_avx2<const C: usize, const Q5: bool>(
    wrow: &[u8],
    acols: &[*const u8; C],
    nb: usize,
) -> [f32; C] {
    const { assert!(C != 0 && C <= TILE_COLS) };
    // SAFETY: AVX2+FMA+F16C present, `wrow` holds nb super-blocks and every
    // column nb q8_K blocks per contract; every load below is inside one of
    // them or the static shuffle table.
    unsafe {
        let (block, qs_off) = if Q5 { (Q5K_BLOCK, 48) } else { (Q4K_BLOCK, 16) };
        let ml = _mm256_set1_epi8(0xF);
        let mh = _mm256_set1_epi8(0x10);
        let zero = _mm256_setzero_si256();
        let shuf: [__m256i; 8] = std::array::from_fn(|s| {
            // SAFETY: 32-byte load inside the 256-byte static table.
            _mm256_loadu_si256(SCALE_SHUF.as_ptr().add(32 * s) as *const __m256i)
        });
        // Lane c is column c's accumulator; a lone column's is `acc1`.
        let mut acc = _mm256_setzero_ps();
        let mut acc1 = 0.0f32;
        for sb in 0..nb {
            let base = wrow.as_ptr().add(block * sb);
            let utmp = make_q4_scales(std::slice::from_raw_parts(base.add(4), 12));
            // Lanes 0..8 the scales, 8..16 the mins, as i16.
            let sm = _mm256_cvtepu8_epi16(_mm_set_epi32(
                utmp[3] as i32,
                utmp[2] as i32,
                utmp[1] as i32,
                utmp[0] as i32,
            ));
            let sc128 = _mm256_castsi256_si128(sm);
            let scales = _mm256_set_m128i(sc128, sc128);
            let m128 = _mm256_extracti128_si256(sm, 1);
            // [m0, m0, m1, m1, ..., m7, m7] against the 16 code sums in order.
            let mins = _mm256_set_m128i(
                _mm_unpackhi_epi16(m128, m128),
                _mm_unpacklo_epi16(m128, m128),
            );
            let d = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(i32::from(
                (base as *const u16).read_unaligned(),
            ))));
            let dmin = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(i32::from(
                (base.add(2) as *const u16).read_unaligned(),
            ))));
            let qh = if Q5 {
                _mm256_loadu_si256(base.add(16) as *const __m256i)
            } else {
                zero
            };
            let blk: [*const u8; C] = std::array::from_fn(|c| acols[c].add(sb * Q8K_STRIDE));

            let mut sumi = [zero; C];
            for chunk in 0..4 {
                let bits = _mm256_loadu_si256(base.add(qs_off + 32 * chunk) as *const __m256i);
                let mut lo = _mm256_and_si256(bits, ml);
                let mut hi = _mm256_and_si256(_mm256_srli_epi16::<4>(bits), ml);
                if Q5 {
                    // Sub-blocks 2·chunk and 2·chunk + 1 take qh bits
                    // 2·chunk and 2·chunk + 1: bit 0..3 of `hb` after the
                    // shift by 4 for the second half.
                    let hb = if chunk < 2 {
                        qh
                    } else {
                        _mm256_srli_epi16::<4>(qh)
                    };
                    if chunk % 2 == 0 {
                        lo = q5_bit::<4>(lo, hb, mh);
                        hi = q5_bit::<3>(hi, hb, mh);
                    } else {
                        lo = q5_bit::<2>(lo, hb, mh);
                        hi = q5_bit::<1>(hi, hb, mh);
                    }
                }
                let s_lo = _mm256_shuffle_epi8(scales, shuf[2 * chunk]);
                let s_hi = _mm256_shuffle_epi8(scales, shuf[2 * chunk + 1]);
                for (s, b) in sumi.iter_mut().zip(&blk) {
                    let q8 = b.add(8 + 64 * chunk);
                    let p_lo = _mm256_madd_epi16(
                        s_lo,
                        _mm256_maddubs_epi16(lo, _mm256_loadu_si256(q8 as *const __m256i)),
                    );
                    let p_hi = _mm256_madd_epi16(
                        s_hi,
                        _mm256_maddubs_epi16(hi, _mm256_loadu_si256(q8.add(32) as *const __m256i)),
                    );
                    *s = _mm256_add_epi32(*s, _mm256_add_epi32(p_lo, p_hi));
                }
            }
            let mini: [__m256i; C] = std::array::from_fn(|c| {
                _mm256_madd_epi16(mins, _mm256_loadu_si256(blk[c].add(264) as *const __m256i))
            });
            if C == 1 {
                // One column: two lane sums in place of the transpose.
                let (si, mi) = (hsum_i32(sumi[0]), hsum_i32(mini[0]));
                let dy = (blk[0] as *const f32).read_unaligned();
                acc1 = (dy * d).mul_add(si as f32, acc1);
                acc1 = (-(dy * dmin)).mul_add(mi as f32, acc1);
                continue;
            }
            let (tot, totm) = if C <= 4 {
                let mut s8 = [zero; TILE_COLS];
                s8[..C].copy_from_slice(&sumi);
                s8[4..4 + C].copy_from_slice(&mini);
                let t = hsum8_i32(&s8);
                (t, _mm256_permute2x128_si256::<0x11>(t, t))
            } else {
                let mut s8 = [zero; TILE_COLS];
                s8[..C].copy_from_slice(&sumi);
                let mut m8 = [zero; TILE_COLS];
                m8[..C].copy_from_slice(&mini);
                (hsum8_i32(&s8), hsum8_i32(&m8))
            };
            let mut dcol = [0.0f32; TILE_COLS];
            for (dc, b) in dcol.iter_mut().zip(&blk) {
                *dc = (*b as *const f32).read_unaligned();
            }
            let dy = _mm256_loadu_ps(dcol.as_ptr());
            let t = _mm256_mul_ps(dy, _mm256_set1_ps(d));
            let tm = _mm256_mul_ps(dy, _mm256_set1_ps(dmin));
            acc = _mm256_fmadd_ps(t, _mm256_cvtepi32_ps(tot), acc);
            acc = _mm256_fnmadd_ps(tm, _mm256_cvtepi32_ps(totm), acc);
        }
        let mut lanes = [0.0f32; TILE_COLS];
        _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
        if C == 1 {
            lanes[0] = acc1;
        }
        let mut out = [0.0f32; C];
        out.copy_from_slice(&lanes[..C]);
        out
    }
}
