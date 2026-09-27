//! Q8_0 rows for Walk A, read in the file's own layout: ggml's `block_q8_0`,
//! 34 bytes for 32 values — the scale `d` (f16) and 32 signed codes — so
//! eight blocks are one 256-value super-block of 272 bytes (68 words), and
//! a row of `256 · n_sb` values is `68 · n_sb` words with no repack.
//!
//! Sub-block `s` of a super-block is block `s`, at byte `34·s`: for `s`
//! even it starts a word (`d` the low half, the codes two bytes in), for `s`
//! odd it starts half a word in (`d` the high half, the codes on the next
//! word). Either way its nine words start at word `17·s >> 1`, and code
//! word `i` is the 64-bit pair of words `i`, `i + 1` shifted right by `16`
//! (`s` even) or `32` (`s` odd). The codes are the values themselves
//! (`d·code`), so `cda = d` and there is no offset to fold: `cdb = 0`.

use super::walk::SbDecode;
use crate::cores::half_to_f32;

/// The Q8_0 format marker (module doc).
pub struct Q8_0;

/// Four codes of a block from two consecutive words: the little-endian pair
/// `(lo, hi)` shifted right by `sh` — 16 when the block starts a word (its
/// codes begin two bytes into `lo`), 32 when it starts half a word in (its
/// codes begin on `hi`).
#[inline(always)]
#[must_use]
pub fn q8_0_code(lo: u32, hi: u32, sh: u32) -> u32 {
    (((u64::from(hi) << 32) | u64::from(lo)) >> sh) as u32
}

impl SbDecode for Q8_0 {
    const WORDS: usize = 68;

    #[inline(always)]
    unsafe fn decode(w: &[u32], wk: usize, s: usize) -> ([u32; 8], f32, f32) {
        let b = wk + ((17 * s) >> 1);
        let odd = s as u32 & 1;
        // SAFETY: b + 8 <= wk + 59 + 8 = wk + 67 (s < 8), inside the
        // super-block, which is inside `w` by this fn's contract.
        let v = unsafe {
            [
                *w.get_unchecked(b),
                *w.get_unchecked(b + 1),
                *w.get_unchecked(b + 2),
                *w.get_unchecked(b + 3),
                *w.get_unchecked(b + 4),
                *w.get_unchecked(b + 5),
                *w.get_unchecked(b + 6),
                *w.get_unchecked(b + 7),
                *w.get_unchecked(b + 8),
            ]
        };
        let d = half_to_f32((v[0] >> (16 * odd)) as u16);
        let sh = 16 + 16 * odd;
        let vi = [
            q8_0_code(v[0], v[1], sh),
            q8_0_code(v[1], v[2], sh),
            q8_0_code(v[2], v[3], sh),
            q8_0_code(v[3], v[4], sh),
            q8_0_code(v[4], v[5], sh),
            q8_0_code(v[5], v[6], sh),
            q8_0_code(v[6], v[7], sh),
            q8_0_code(v[7], v[8], sh),
        ];
        (vi, d, 0.0)
    }
}
