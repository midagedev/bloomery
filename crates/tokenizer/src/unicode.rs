//! The reference's Unicode handling, byte for byte: its lenient UTF-8 decoder,
//! the GPT-2 byte <-> codepoint table, and the "collapsed" byte its regex
//! fallback matches on.

use crate::collapse_table::RUNS;

/// U+FFFD, what the reference substitutes for a byte it cannot decode.
pub(crate) const REPLACEMENT: u32 = 0xFFFD;

/// Decode `bytes` the way the reference's `unicode_cpts_from_utf8` does: a lead
/// byte is trusted for its length, each continuation byte must be `10xxxxxx`,
/// overlong forms and surrogates pass through, and a byte that starts no valid
/// sequence becomes U+FFFD and advances one byte.
///
/// A four-byte form above U+10FFFF also becomes U+FFFD here; the reference
/// throws on it when it re-encodes the word.
pub(crate) fn decode_lenient(bytes: &[u8], out: &mut Vec<u32>) {
    out.clear();
    out.reserve(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        let (cpt, n) = decode_one(bytes, i);
        out.push(cpt);
        i += n;
    }
}

fn decode_one(b: &[u8], i: usize) -> (u32, usize) {
    let cont = |k: usize| b.get(i + k).is_some_and(|&c| c & 0xC0 == 0x80);
    let lead = b[i];
    let bits = |k: usize| u32::from(b[i + k] & 0x3F);
    if lead & 0x80 == 0 {
        (u32::from(lead), 1)
    } else if lead & 0x40 == 0 {
        (REPLACEMENT, 1)
    } else if lead & 0x20 == 0 {
        if cont(1) {
            ((u32::from(lead & 0x1F) << 6) | bits(1), 2)
        } else {
            (REPLACEMENT, 1)
        }
    } else if lead & 0x10 == 0 {
        if cont(1) && cont(2) {
            ((u32::from(lead & 0x0F) << 12) | (bits(1) << 6) | bits(2), 3)
        } else {
            (REPLACEMENT, 1)
        }
    } else if lead & 0x08 == 0 {
        if cont(1) && cont(2) && cont(3) {
            let cpt = (u32::from(lead & 0x07) << 18) | (bits(1) << 12) | (bits(2) << 6) | bits(3);
            (if cpt > 0x10FFFF { REPLACEMENT } else { cpt }, 4)
        } else {
            (REPLACEMENT, 1)
        }
    } else {
        (REPLACEMENT, 1)
    }
}

/// Append the UTF-8 form of `cpt` (<= U+10FFFF; surrogates are encoded as
/// three bytes, as the reference does).
pub(crate) fn encode_cpt(cpt: u32, out: &mut Vec<u8>) {
    // The casts below keep the low bits of a value already masked or shifted
    // into byte range.
    if cpt <= 0x7F {
        out.push(cpt as u8);
    } else if cpt <= 0x7FF {
        out.extend_from_slice(&[0xC0 | (cpt >> 6) as u8, 0x80 | (cpt & 0x3F) as u8]);
    } else if cpt <= 0xFFFF {
        out.extend_from_slice(&[
            0xE0 | (cpt >> 12) as u8,
            0x80 | ((cpt >> 6) & 0x3F) as u8,
            0x80 | (cpt & 0x3F) as u8,
        ]);
    } else {
        out.extend_from_slice(&[
            0xF0 | ((cpt >> 18) & 0x07) as u8,
            0x80 | ((cpt >> 12) & 0x3F) as u8,
            0x80 | ((cpt >> 6) & 0x3F) as u8,
            0x80 | (cpt & 0x3F) as u8,
        ]);
    }
}

/// The byte a codepoint collapses to: itself below 0x80, otherwise its
/// category byte from the reference's table (whitespace first).
pub(crate) fn collapse(cpt: u32) -> u8 {
    match u8::try_from(cpt) {
        Ok(b) if b < 0x80 => b,
        _ => {
            let i = RUNS.partition_point(|&(start, _)| start <= cpt);
            // RUNS starts at 0x80 and `cpt` is at least that, so `i >= 1`.
            RUNS[i - 1].1
        }
    }
}

/// GPT-2's byte-level alphabet: the codepoint each byte is written as inside
/// a vocabulary string (`unicode_byte_to_utf8_map`).
pub(crate) fn byte_to_cpt_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut assigned = [false; 256];
    for b in (0x21..=0x7E).chain(0xA1..=0xAC).chain(0xAE..=0xFF) {
        table[b] = b as u32;
        assigned[b] = true;
    }
    let mut n = 0;
    for b in 0..256 {
        if !assigned[b] {
            table[b] = 256 + n;
            n += 1;
        }
    }
    table
}
