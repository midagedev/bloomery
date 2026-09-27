//! Q5_K's super-block for Walk A: ggml's `block_q5_K`, 176 bytes (44
//! words) for 256 values — `d` and `dmin` (f16) in word 0, the 12 scale
//! bytes in words 1..4 (Q4_K's packing, `get_scale_min_k4`), the high-bit
//! plane `qh[32]` in words 4..12 and the nibble plane `qs[128]` in words
//! 12..44 (`gguf::quant::dequant_row`'s layout, the host's owner of it).
//!
//! Value `l` (0..32) of sub-block `s` is `d·sc_s·(nib + 16·h) − dmin·m_s`,
//! its nibble the low (`s` even) or high (`s` odd) half of `qs[32·(s >> 1)
//! + l]` and `h` bit `s` of `qh[l]`. Qs word `i` of the sub-block and qh
//! word `i` cover the same four values `4·i ..`, so a code word is the
//! nibbles with each qh byte's bit `s` moved to bit 4: codes 0..31, exact
//! as signed bytes against the q8 plane. There is no offset to fold, so
//! `cdb = −dmin·m` alone.

use super::walk::SbDecode;
use crate::cores::{half_to_f32, q4k_scale_min};

/// The Q5_K format marker (module doc).
pub struct Q5k;

/// Word offset of `qh` inside a Q5_K super-block.
const QH_WORD: usize = 4;
/// Word offset of `qs` inside a Q5_K super-block.
const QS_WORD: usize = 12;

// 176 bytes: the header word, three scale words, 8 qh words, 32 qs words.
const _: () = assert!(QS_WORD + 32 == 44 && QH_WORD + 8 == QS_WORD);

/// Four Q5_K codes (bytes 0..31): the nibbles at `nib` of the qs word, bit
/// `s` of each byte of the qh word as bit 4.
#[inline(always)]
#[must_use]
pub fn q5k_code(qs: u32, qh: u32, nib: u32, s: u32) -> u32 {
    ((qs >> nib) & 0x0f0f_0f0f) | (((qh >> s) & 0x0101_0101) << 4)
}

impl SbDecode for Q5k {
    const WORDS: usize = 44;

    #[inline(always)]
    unsafe fn decode(w: &[u32], wk: usize, s: usize) -> ([u32; 8], f32, f32) {
        // SAFETY: wk + 3 < wk + 44 <= w.len() by this fn's contract.
        let (w0, w1, w2, w3) = unsafe {
            (
                *w.get_unchecked(wk),
                *w.get_unchecked(wk + 1),
                *w.get_unchecked(wk + 2),
                *w.get_unchecked(wk + 3),
            )
        };
        let d = half_to_f32((w0 & 0xffff) as u16);
        let dmin = half_to_f32((w0 >> 16) as u16);
        let (sc, mi) = q4k_scale_min(s, w1, w2, w3);
        let cda = d * sc as f32;
        let cdb = -(dmin * mi as f32);
        let qh = wk + QH_WORD;
        let qs = wk + QS_WORD + 8 * (s >> 1);
        let nib = (s as u32 & 1) * 4;
        let hb = s as u32;
        // SAFETY: qh + 7 = wk + 11 and qs + 7 <= wk + 12 + 24 + 7 = wk + 43,
        // both inside the super-block, which is inside `w` by this fn's
        // contract (s < 8).
        let vi = unsafe {
            [
                q5k_code(*w.get_unchecked(qs), *w.get_unchecked(qh), nib, hb),
                q5k_code(*w.get_unchecked(qs + 1), *w.get_unchecked(qh + 1), nib, hb),
                q5k_code(*w.get_unchecked(qs + 2), *w.get_unchecked(qh + 2), nib, hb),
                q5k_code(*w.get_unchecked(qs + 3), *w.get_unchecked(qh + 3), nib, hb),
                q5k_code(*w.get_unchecked(qs + 4), *w.get_unchecked(qh + 4), nib, hb),
                q5k_code(*w.get_unchecked(qs + 5), *w.get_unchecked(qh + 5), nib, hb),
                q5k_code(*w.get_unchecked(qs + 6), *w.get_unchecked(qh + 6), nib, hb),
                q5k_code(*w.get_unchecked(qs + 7), *w.get_unchecked(qh + 7), nib, hb),
            ]
        };
        (vi, cda, cdb)
    }
}
