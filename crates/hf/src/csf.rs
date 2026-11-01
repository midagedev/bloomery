//! NVFP4 block scales a ModelOpt checkpoint stores compressed, read back as the logical
//! `weight_scale`: the `R x C` F8_E4M3 bytes, one a 16-value group, a plain checkpoint stores.
//!
//! `quantized_layers[...].weight_scale_encoding` names one of two encodings, each a byte
//! `base + code` (a row's base byte and a 4-bit code, low nibble first) with whole bytes stored
//! apart where the 16-wide window misses:
//!
//! - `csf`: `<scale>.nvfp4_csf_fixed` U8 `[R/16, 16 (1 + C/2)]`, each row a 16-row slab, its 16
//!   base bytes then the slab's codes row by row; `<scale>.nvfp4_csf_exceptions` U32, sorted and
//!   unique, the low 24 bits a flat position `r C + c` and the high 8 the byte there.
//! - `csf_32x2_v1`: blocks of 32 rows by 2 columns (the last block of an odd `C` one wide), tiles
//!   of 4 block rows by 2 block columns, row-major, pages of 32 tiles. `<scale>.csf32_bases` U8
//!   `[R]`; `.csf32_codes` U8 `[R C / 2]`, the blocks in stream order (a tile's first block
//!   column's block rows, then its second's), each block's codes row-major; `.csf32_page_bases`
//!   U32, twice the records before each page; `.csf32_offsets` U16 `[blocks + pages]`, each page a
//!   prefix of its blocks' records from 0 with a terminal entry; `.csf32_exceptions` U16, a
//!   block's records in increasing position, the low byte the place in the block (`2 row + col`)
//!   and the high byte the byte there.
//!
//! The rules are those of the encoders' published decoders (FlashInfer's `_csf_blocks.py`,
//! vLLM's `csf_scales.py`). Every part a rule does not allow is refused by name: a base above
//! 240, a part of the wrong size, a page base or a page's first prefix that does not match, a
//! prefix that falls or runs past the records, a record outside its block or out of order, an
//! exception position out of range, repeated or unsorted, and records or codes left over.

use safetensors::{Dtype, Raw};

/// The largest base: `base + 15` must stay a byte.
const MAX_BASE: u8 = 240;
const BLOCK_ROWS: usize = 32;
const TILE_BLOCK_ROWS: usize = 4;
const TILE_BLOCK_COLS: usize = 2;
const PAGE_TILES: usize = 32;

/// Why a compressed scale was refused.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Safetensors(#[from] safetensors::Error),
    #[error("{name}: {what}")]
    Rule { name: String, what: String },
}

/// A scale encoding `weight_scale_encoding` names: `csf` or `csf_32x2_v1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Encoding {
    Csf,
    Csf32,
}

impl Encoding {
    /// The encoding of the block scales of NVFP4 stored as `format`; `None` for a format that
    /// stores them plain.
    pub(crate) fn of(format: crate::quant_config::Format) -> Option<Encoding> {
        use crate::quant_config::Format;
        match format {
            Format::Nvfp4Csf => Some(Encoding::Csf),
            Format::Nvfp4Csf32 => Some(Encoding::Csf32),
            _ => None,
        }
    }
}

/// The logical scale `scale` (`<stem>.weight_scale`) of a weight of `rows` by `cols` scale
/// groups, its parts looked up by name through `get` and decoded by `encoding`: `rows · cols`
/// bytes, row-major.
///
/// # Errors
///
/// A part `get` refuses or lacks, a part of the wrong dtype or size, and every rule the module
/// header lists, by name.
pub(crate) fn weight_scale<'a>(
    get: impl Fn(&str) -> Result<Raw<'a>, safetensors::Error>,
    scale: &str,
    encoding: Encoding,
    rows: usize,
    cols: usize,
) -> Result<Vec<u8>, Error> {
    let size = rows
        .checked_mul(cols)
        .ok_or_else(|| rule(scale, "scale size overflow"))?;
    let part = |suffix: &str, want: Dtype| -> Result<(String, &'a [u8]), Error> {
        let name = format!("{scale}.{suffix}");
        let raw = get(&name)?;
        if raw.dtype != want {
            return Err(safetensors::Error::Kind {
                name,
                have: raw.dtype.name(),
                want: want.name(),
            }
            .into());
        }
        Ok((name, raw.bytes))
    };
    match encoding {
        Encoding::Csf => {
            let (name, fixed) = part("nvfp4_csf_fixed", Dtype::U8)?;
            let (_, exceptions) = part("nvfp4_csf_exceptions", Dtype::U32)?;
            csf(&name, fixed, words::<4>(exceptions), rows, cols, size)
        }
        Encoding::Csf32 => {
            let (name, bases) = part("csf32_bases", Dtype::U8)?;
            let (_, codes) = part("csf32_codes", Dtype::U8)?;
            let (_, page_bases) = part("csf32_page_bases", Dtype::U32)?;
            let (_, offsets) = part("csf32_offsets", Dtype::U16)?;
            let (_, exceptions) = part("csf32_exceptions", Dtype::U16)?;
            let parts = Csf32 {
                bases,
                codes,
                page_bases: &words::<4>(page_bases),
                offsets: &words::<2>(offsets),
                exceptions: &words::<2>(exceptions),
            };
            csf32(&name, &parts, rows, cols, size)
        }
    }
}

/// Little-endian words of `N` bytes, widened to u32; a trailing partial word is left for the
/// caller's size check to refuse.
fn words<const N: usize>(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<N>()
        .0
        .iter()
        .map(|w| w.iter().rev().fold(0u32, |a, &b| a << 8 | u32::from(b)))
        .collect()
}

/// A refusal of the part `name`.
fn rule(name: &str, what: impl Into<String>) -> Error {
    Error::Rule {
        name: name.to_string(),
        what: what.into(),
    }
}

/// Code `k` of a stream of 4-bit codes, low nibble first.
fn nibble(codes: &[u8], k: usize) -> u8 {
    codes[k / 2] >> (4 * (k % 2)) & 15
}

/// `csf`: the fixed stream `fixed` (named `name`) and its exception words, for `rows` by `cols`.
fn csf(
    name: &str,
    fixed: &[u8],
    exceptions: Vec<u32>,
    rows: usize,
    cols: usize,
    size: usize,
) -> Result<Vec<u8>, Error> {
    if rows == 0 || !rows.is_multiple_of(16) || cols == 0 || !cols.is_multiple_of(2) {
        return Err(rule(
            name,
            format!("{rows} by {cols} scales: csf needs rows a multiple of 16 and even columns"),
        ));
    }
    let slab = (cols / 2)
        .checked_add(1)
        .and_then(|n| n.checked_mul(16))
        .ok_or_else(|| rule(name, "slab size overflow"))?;
    let fixed_size = (rows / 16)
        .checked_mul(slab)
        .ok_or_else(|| rule(name, "fixed size overflow"))?;
    if fixed.len() != fixed_size {
        return Err(rule(
            name,
            format!(
                "{} bytes; {rows} by {cols} scales are {} slabs of {slab}",
                fixed.len(),
                rows / 16
            ),
        ));
    }
    let mut out = vec![0u8; size];
    for (s, slab) in fixed.chunks_exact(slab).enumerate() {
        let (bases, codes) = slab.split_at(16);
        if let Some(r) = bases.iter().position(|&b| b > MAX_BASE) {
            return Err(rule(
                name,
                format!("row {} base {} is above {MAX_BASE}", 16 * s + r, bases[r]),
            ));
        }
        for (r, &base) in bases.iter().enumerate() {
            let row = &mut out[(16 * s + r) * cols..][..cols];
            for (c, v) in row.iter_mut().enumerate() {
                *v = base + nibble(codes, r * cols + c);
            }
        }
    }
    let mut prev = None;
    for (i, &w) in exceptions.iter().enumerate() {
        let at = (w & 0xFF_FFFF) as usize;
        if at >= size || prev.is_some_and(|p| at <= p) {
            return Err(rule(
                name,
                format!(
                    "exception {i} at position {at}: positions are increasing and below {}",
                    size
                ),
            ));
        }
        prev = Some(at);
        out[at] = (w >> 24) as u8;
    }
    Ok(out)
}

/// The five parts of a `csf_32x2_v1` scale, its words widened.
struct Csf32<'p> {
    bases: &'p [u8],
    codes: &'p [u8],
    page_bases: &'p [u32],
    offsets: &'p [u32],
    exceptions: &'p [u32],
}

/// `csf_32x2_v1`: `parts` (named `name` by their bases) for `rows` by `cols`.
fn csf32(
    name: &str,
    parts: &Csf32<'_>,
    rows: usize,
    cols: usize,
    size: usize,
) -> Result<Vec<u8>, Error> {
    if rows == 0 || !rows.is_multiple_of(BLOCK_ROWS) || cols == 0 {
        return Err(rule(
            name,
            format!("{rows} by {cols} scales: csf32 needs rows a multiple of 32"),
        ));
    }
    let (block_rows, block_cols) = (rows / BLOCK_ROWS, cols.div_ceil(2));
    let tile_cols = block_cols.div_ceil(TILE_BLOCK_COLS);
    let tiles = block_rows
        .div_ceil(TILE_BLOCK_ROWS)
        .checked_mul(tile_cols)
        .ok_or_else(|| rule(name, "tile count overflow"))?;
    let pages = tiles.div_ceil(PAGE_TILES);
    let blocks = block_rows
        .checked_mul(block_cols)
        .ok_or_else(|| rule(name, "block count overflow"))?;
    let offset_count = blocks
        .checked_add(pages)
        .ok_or_else(|| rule(name, "offset count overflow"))?;
    let Csf32 {
        bases,
        codes,
        page_bases,
        offsets,
        exceptions,
    } = *parts;
    let sizes = [
        ("bases", bases.len(), rows),
        ("codes", codes.len(), size / 2),
        ("page bases", page_bases.len(), pages),
        ("offsets", offsets.len(), offset_count),
    ];
    for (part, have, want) in sizes {
        if have != want {
            return Err(rule(
                name,
                format!("{part} hold {have} entries; {rows} by {cols} scales need {want}"),
            ));
        }
    }
    if let Some(r) = bases.iter().position(|&b| b > MAX_BASE) {
        return Err(rule(
            name,
            format!("row {r} base {} is above {MAX_BASE}", bases[r]),
        ));
    }
    let mut out = vec![0u8; size];
    let mut field = 0;
    let mut entry = 0;
    let mut origin = 0usize;
    let mut page_end = 0;
    let mut block = 0;
    for tile in 0..tiles {
        if tile.is_multiple_of(PAGE_TILES) {
            if tile > 0 {
                entry += 1;
                origin += page_end;
            }
            let page = tile / PAGE_TILES;
            let base = origin
                .checked_mul(2)
                .ok_or_else(|| rule(name, "page base overflow"))?;
            if page_bases[page] as usize != base || offsets[entry] != 0 {
                return Err(rule(
                    name,
                    format!(
                        "page {page}: base {} and first prefix {}; the records before it make {} and 0",
                        page_bases[page], offsets[entry], base
                    ),
                ));
            }
        }
        let (r, c) = (
            tile / tile_cols * TILE_BLOCK_ROWS,
            tile % tile_cols * TILE_BLOCK_COLS,
        );
        for bc in c..block_cols.min(c + TILE_BLOCK_COLS) {
            for br in r..block_rows.min(r + TILE_BLOCK_ROWS) {
                let (r0, c0) = (br * BLOCK_ROWS, 2 * bc);
                let w = (cols - c0).min(2);
                for i in 0..BLOCK_ROWS {
                    for j in 0..w {
                        out[(r0 + i) * cols + c0 + j] =
                            bases[r0 + i] + nibble(codes, field + i * w + j);
                    }
                }
                field += BLOCK_ROWS * w;
                let (lo, hi) = (offsets[entry] as usize, offsets[entry + 1] as usize);
                let end = origin
                    .checked_add(hi)
                    .ok_or_else(|| rule(name, "exception offset overflow"))?;
                if hi < lo || end > exceptions.len() {
                    return Err(rule(
                        name,
                        format!(
                            "block {block}: prefixes {lo} to {hi} from record {origin} of {}",
                            exceptions.len()
                        ),
                    ));
                }
                let mut prev = None;
                for &rec in &exceptions[origin + lo..end] {
                    let at = (rec & 0xFF) as usize;
                    if at / 2 >= BLOCK_ROWS || at % 2 >= w || prev.is_some_and(|p| at <= p) {
                        return Err(rule(
                            name,
                            format!(
                                "block {block}: record at {at} is outside its {BLOCK_ROWS} by {w} block or out of order"
                            ),
                        ));
                    }
                    prev = Some(at);
                    out[(r0 + at / 2) * cols + c0 + at % 2] = (rec >> 8) as u8;
                }
                page_end = hi;
                entry += 1;
                block += 1;
            }
        }
    }
    if origin + page_end != exceptions.len() || field != size {
        return Err(rule(
            name,
            format!(
                "{} of {} records and {field} of {} codes read",
                origin + page_end,
                exceptions.len(),
                size
            ),
        ));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::{Encoding, Error, weight_scale};
    use safetensors::{Safetensors, write};
    use sha2::{Digest, Sha256};

    /// The fixture `tests/fixtures/csf_scales.py fixture` writes.
    fn fixture() -> Safetensors {
        Safetensors::parse(include_bytes!("../tests/fixtures/csf-scales.safetensors").to_vec())
            .unwrap()
    }

    /// The encoders' outputs decode to the bytes they encoded: a seeded `csf` scale with its
    /// exceptions and a seeded `csf_32x2_v1` scale of an odd column count over two pages. The
    /// checkpoint's own `csf` scale (layer 0 expert 0 gate_proj of Qwen3.8-Flash-Next-FP4-Local)
    /// decodes to the plain `weight_scale` of the uncompressed copy of the same weights
    /// (`csf_scales.py plain`: 102,400 bytes, sha256 4256235...). Reading a block column before
    /// a block row, a page's records from the arena's start or the code nibbles high first fails
    /// one of them.
    #[test]
    fn compressed_scales_decode_to_the_bytes_they_encode() {
        let st = fixture();
        let get = |n: &str| st.raw(n);
        for (stem, encoding, rows, cols) in [
            ("syn_csf", Encoding::Csf, 48, 20),
            ("syn_csf32", Encoding::Csf32, 384, 43),
            ("syn_csf32k", Encoding::Csf32, 64, 8),
        ] {
            let scale = format!("{stem}.weight_scale");
            let got = weight_scale(get, &scale, encoding, rows, cols).unwrap();
            assert_eq!(got, st.raw(&scale).unwrap().bytes, "{stem}");
        }
        let got = weight_scale(get, "qwen.weight_scale", Encoding::Csf, 640, 160).unwrap();
        assert_eq!(
            format!("{:x}", Sha256::digest(&got)),
            "425623551a68b087cc6a8827d42e43006ceaca909b874220491ec4f55db74c96"
        );
    }

    /// Each structural rule refuses by name: a base above 240, a part of the wrong size, a page
    /// base that does not count the records before it, a falling prefix, a record outside its
    /// block or out of order, records left over, and an exception position out of range or
    /// repeated.
    #[test]
    fn a_compressed_scale_breaking_a_rule_is_refused_by_name() {
        for (encoding, rows, cols) in [
            (Encoding::Csf, 16, usize::MAX - 1),
            (Encoding::Csf, usize::MAX / 16 * 16, 2),
            (Encoding::Csf32, 32, usize::MAX),
            (Encoding::Csf32, usize::MAX / 32 * 32, 2),
        ] {
            let get = |n: &str| {
                use safetensors::Dtype;
                let dtype = if n.ends_with("offsets") || n.ends_with("csf32_exceptions") {
                    Dtype::U16
                } else if n.ends_with("exceptions") || n.ends_with("page_bases") {
                    Dtype::U32
                } else {
                    Dtype::U8
                };
                Ok(safetensors::Raw {
                    dtype,
                    shape: &[],
                    bytes: &[],
                })
            };
            let e = weight_scale(get, "huge.weight_scale", encoding, rows, cols).unwrap_err();
            assert!(e.to_string().contains("huge.weight_scale"), "{e}");
            assert!(e.to_string().contains("size overflow"), "{e}");
        }
        let st = fixture();
        let parts: Vec<(String, &str, Vec<usize>, Vec<u8>)> = st
            .names()
            .filter(|n| n.starts_with("syn_csf"))
            .map(|n| {
                let raw = st.raw(n).unwrap();
                (
                    n.to_string(),
                    raw.dtype.name(),
                    raw.shape.to_vec(),
                    raw.bytes.to_vec(),
                )
            })
            .collect();
        let refuse = |part: &str, edit: &dyn Fn(&mut Vec<u8>), want: &str| {
            let (stem, encoding, rows, cols) = if part.starts_with("syn_csf32") {
                ("syn_csf32", Encoding::Csf32, 384, 43)
            } else {
                ("syn_csf", Encoding::Csf, 48, 20)
            };
            let mut edited = parts.clone();
            let t = edited.iter_mut().find(|t| t.0 == part).unwrap();
            edit(&mut t.3);
            t.2 = vec![t.3.len() / st.raw(part).unwrap().dtype.size()];
            let entries: Vec<(&str, &str, &[usize], &[u8])> = edited
                .iter()
                .map(|(n, d, s, b)| (n.as_str(), *d, s.as_slice(), b.as_slice()))
                .collect();
            let st = Safetensors::parse(write(&entries)).unwrap();
            let e = weight_scale(
                |n| st.raw(n),
                &format!("{stem}.weight_scale"),
                encoding,
                rows,
                cols,
            )
            .expect_err(part);
            assert!(
                matches!(e, Error::Rule { .. }) && e.to_string().contains(want),
                "{part}: {e}"
            );
        };
        let fixed = "syn_csf.weight_scale.nvfp4_csf_fixed";
        let exc = "syn_csf.weight_scale.nvfp4_csf_exceptions";
        let c32 = |p: &str| format!("syn_csf32.weight_scale.csf32_{p}");
        refuse(fixed, &|b| b[3] = 241, "base 241 is above 240");
        refuse(&c32("bases"), &|b| b[7] = 250, "row 7 base 250");
        refuse(fixed, &|b| b.truncate(b.len() - 16), "slabs of 176");
        refuse(&c32("codes"), &|b| b.truncate(b.len() - 1), "codes hold");
        refuse(&c32("page_bases"), &|b| b[4] ^= 2, "page 1: base");
        refuse(
            &c32("offsets"),
            &|b| b[2..4].copy_from_slice(&u16::MAX.to_le_bytes()),
            "block 0: prefixes",
        );
        refuse(&c32("exceptions"), &|b| b[0] = 64, "outside its 32 by");
        refuse(&c32("exceptions"), &|b| b[2] = b[0], "out of order");
        refuse(
            &c32("exceptions"),
            &|b| b.extend_from_slice(&[0, 0]),
            "records and",
        );
        refuse(
            exc,
            &|b| b[0..3].copy_from_slice(&[0xC0, 0x03, 0]),
            "below 960",
        );
        refuse(
            exc,
            &|b| {
                let first = b[0..4].to_vec();
                b[4..8].copy_from_slice(&first);
            },
            "positions are increasing",
        );
    }
}
