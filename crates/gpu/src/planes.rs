//! The one splitter of a file tensor's records into the two planes a card
//! kernel reads.
//!
//! Five file formats (Q8_0, NVFP4, MXFP4, MXFP8, FP8_B128) store a tensor as
//! records of `sub` blocks, each block its `head` scale bytes and then its
//! `code` value bytes. The card layout keeps the two apart: a plane of the
//! codes in record order and a plane of the scale bytes in record order, each
//! the file's own bytes read as little-endian words. The split is a byte
//! de-interleave with a validity check, never arithmetic, and [`Format`] names
//! the record: [`split_rows`] is the one routine that does it, over whole rows,
//! in 32 to 128 byte moves per block the compiler keeps as vector loads and
//! stores. The two callers differ in where the planes go: [`split_vecs`] into
//! two host vectors (a host packing for a caller with no ring), and
//! [`fill_piece`] into one upload slot's two regions, so the load's bytes are
//! read once from the file and written once, into the pinned memory the device
//! copies from.

use crate::GpuError;
use crate::mxfp4::{Fp4, Mxfp4, Nvfp4};
use crate::upload::{TEAM_MIN, pool_map};
use gguf::quant::{check_fp8_b128, check_mxfp8};

/// A file format's record and its two planes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    /// 34 bytes: the f16 scale, 32 codes. Planes: the codes as words, the scale bits.
    Q8_0,
    /// 36 bytes: 4 UE4M3 scale bytes, 32 code bytes (64 values).
    Nvfp4,
    /// 17 bytes: the E8M0 scale byte, 16 code bytes (32 values).
    Mxfp4,
    /// 132 bytes: four blocks of a scale byte and 32 codes (128 values).
    Mxfp8,
    /// 132 bytes: the f32 block scale, 128 codes.
    Fp8B128,
}

impl Format {
    /// Blocks a record holds, the scale bytes and the code bytes of each.
    const fn dims(self) -> (usize, usize, usize) {
        match self {
            Format::Q8_0 => (1, 2, 32),
            Format::Nvfp4 => (1, 4, 32),
            Format::Mxfp4 => (1, 1, 16),
            Format::Mxfp8 => (4, 1, 32),
            Format::Fp8B128 => (1, 4, 128),
        }
    }

    /// File bytes a record takes.
    #[must_use]
    pub const fn record_bytes(self) -> usize {
        let (sub, head, code) = self.dims();
        sub * (head + code)
    }

    /// Bytes a record adds to the code plane.
    #[must_use]
    pub const fn code_bytes(self) -> usize {
        let (sub, _, code) = self.dims();
        sub * code
    }

    /// Bytes a record adds to the scale plane.
    #[must_use]
    pub const fn head_bytes(self) -> usize {
        let (sub, head, _) = self.dims();
        sub * head
    }
}

/// `format`'s rows of `row_recs` records, the plane bytes of one row:
/// `(file bytes, code bytes, scale bytes)`.
fn row_widths(format: Format, row_recs: usize) -> Result<(usize, usize, usize), String> {
    if row_recs == 0 {
        return Err(format!("a {format:?} row of no records"));
    }
    let wide = |per: usize| {
        row_recs
            .checked_mul(per)
            .ok_or_else(|| format!("{row_recs} records of {per} bytes a row pass usize"))
    };
    Ok((
        wide(format.record_bytes())?,
        wide(format.code_bytes())?,
        wide(format.head_bytes())?,
    ))
}

/// The refusal of an FP4 row's scale bytes (`F::scale_ok`), naming its row,
/// block and, for a format with several scale bytes a block, sub-block.
fn fp4_scales<F: Fp4>(row: usize, bytes: &[u8]) -> Result<(), String> {
    for (b, blk) in bytes.chunks_exact(F::BLOCK_BYTES).enumerate() {
        for (s, &byte) in blk[..F::SCALE_BYTES].iter().enumerate() {
            F::scale_ok(byte).map_err(|why| {
                let sub = if F::SCALE_BYTES > 1 {
                    format!(" sub-block {s}")
                } else {
                    String::new()
                };
                format!("row {row} block {b}{sub}: scale byte {why}")
            })?;
        }
    }
    Ok(())
}

/// The validity check of one file row of `format`: the bytes the card kernels
/// are not defined on are refused here, naming `row`.
fn check_row(format: Format, row: usize, bytes: &[u8]) -> Result<(), String> {
    match format {
        Format::Q8_0 => Ok(()),
        Format::Nvfp4 => fp4_scales::<Nvfp4>(row, bytes),
        Format::Mxfp4 => fp4_scales::<Mxfp4>(row, bytes),
        Format::Mxfp8 => check_mxfp8(bytes).map_err(|why| format!("row {row}: {why}")),
        Format::Fp8B128 => check_fp8_b128(bytes).map_err(|why| format!("row {row}: {why}")),
    }
}

/// The records of `src` de-interleaved: each block's `HEAD` scale bytes into
/// `heads`, its `CODE` value bytes into `codes`, in record order. The moves
/// have constant lengths, so each is a vector load and store.
#[inline(always)]
fn deinterleave<const SUB: usize, const HEAD: usize, const CODE: usize>(
    src: &[u8],
    codes: &mut [u8],
    heads: &mut [u8],
) {
    let records = src
        .chunks_exact(SUB * (HEAD + CODE))
        .zip(codes.chunks_exact_mut(SUB * CODE))
        .zip(heads.chunks_exact_mut(SUB * HEAD));
    for ((rec, c), h) in records {
        for s in 0..SUB {
            let blk = &rec[s * (HEAD + CODE)..][..HEAD + CODE];
            h[s * HEAD..][..HEAD].copy_from_slice(&blk[..HEAD]);
            c[s * CODE..][..CODE].copy_from_slice(&blk[HEAD..]);
        }
    }
}

/// Rows `first_row ..` of `format`, `src` their whole file bytes (rows of
/// `row_recs` records): row `r`'s code bytes into the next row's worth of
/// `codes` and its scale bytes into `heads`. The first row a check refuses is
/// the error, naming its row of the tensor; the planes of the rows before it
/// are written.
pub(crate) fn split_rows(
    format: Format,
    row_recs: usize,
    first_row: usize,
    src: &[u8],
    codes: &mut [u8],
    heads: &mut [u8],
) -> Result<(), String> {
    let (row, code_row, head_row) = row_widths(format, row_recs)?;
    let rows = src.len() / row;
    if rows * row != src.len() || codes.len() != rows * code_row || heads.len() != rows * head_row {
        return Err(format!(
            "{} bytes into {} and {} plane bytes are not whole rows of {row_recs} {format:?} records",
            src.len(),
            codes.len(),
            heads.len()
        ));
    }
    let each = src
        .chunks_exact(row)
        .zip(codes.chunks_exact_mut(code_row))
        .zip(heads.chunks_exact_mut(head_row));
    for (r, ((file, c), h)) in each.enumerate() {
        check_row(format, first_row + r, file)?;
        match format {
            Format::Q8_0 => deinterleave::<1, 2, 32>(file, c, h),
            Format::Nvfp4 => deinterleave::<1, 4, 32>(file, c, h),
            Format::Mxfp4 => deinterleave::<1, 1, 16>(file, c, h),
            Format::Mxfp8 => deinterleave::<4, 1, 32>(file, c, h),
            Format::Fp8B128 => deinterleave::<1, 4, 128>(file, c, h),
        }
    }
    Ok(())
}

/// [`split_rows`] over the resident worker pool: a run of rows a task, tasks
/// of about a quarter of a thread's share so that uneven rows keep every
/// core busy, and the first refusal in row order is the error. A short input
/// splits on the calling thread ([`TEAM_MIN`]).
fn split_pooled(
    format: Format,
    row_recs: usize,
    first_row: usize,
    src: &[u8],
    codes: &mut [u8],
    heads: &mut [u8],
) -> Result<(), String> {
    let (row, code_row, head_row) = row_widths(format, row_recs)?;
    if src.len() < TEAM_MIN {
        return split_rows(format, row_recs, first_row, src, codes, heads);
    }
    let rows = src.len() / row;
    let run = rows.div_ceil(4 * threads::pool().threads()).max(1);
    let tasks: Vec<_> = src
        .chunks(run * row)
        .zip(codes.chunks_mut(run * code_row))
        .zip(heads.chunks_mut(run * head_row))
        .enumerate()
        .map(|(i, ((s, c), h))| (first_row + i * run, s, c, h))
        .collect();
    pool_map(tasks, |(first, s, c, h)| {
        split_rows(format, row_recs, first, s, c, h)
    })
    .into_iter()
    .find_map(Result::err)
    .map_or(Ok(()), Err)
}

/// The planes of `rows` rows of `format` (`bytes` exactly their file bytes,
/// rows of `row_recs` records) as host words: `(codes, scales)`, each the
/// plane's bytes as little-endian words. A refused row is `what`'s error.
pub(crate) fn split_vecs(
    what: &'static str,
    format: Format,
    row_recs: usize,
    rows: usize,
    bytes: &[u8],
) -> Result<(Vec<u32>, Vec<u32>), GpuError> {
    let (row, code_row, head_row) =
        row_widths(format, row_recs).map_err(|e| GpuError::shape(what, e))?;
    let size = |per: usize| {
        rows.checked_mul(per)
            .ok_or_else(|| GpuError::shape(what, format!("{rows} rows of {per} bytes pass usize")))
    };
    if size(row)? != bytes.len() {
        return Err(GpuError::shape(
            what,
            format!("{} bytes are not {rows} rows of {row}", bytes.len()),
        ));
    }
    let (mut codes, mut heads) = (vec![0u8; size(code_row)?], vec![0u8; size(head_row)?]);
    if !codes.len().is_multiple_of(4) || !heads.len().is_multiple_of(4) {
        let detail = format!(
            "{} and {} plane bytes are not whole words",
            codes.len(),
            heads.len()
        );
        return Err(GpuError::shape(what, detail));
    }
    split_pooled(format, row_recs, 0, bytes, &mut codes, &mut heads)
        .map_err(|e| GpuError::shape(what, e))?;
    let words = |plane: Vec<u8>| -> Vec<u32> {
        plane
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| u32::from_le_bytes(*w))
            .collect()
    };
    Ok((words(codes), words(heads)))
}

/// One upload piece: `src`'s whole rows split into `out`, the codes of the rows
/// in `out`'s first `rows * code_row` bytes and their scale bytes after them,
/// the two regions a slot's two device copies read. A refused row is the error.
pub(crate) fn fill_piece(
    format: Format,
    row_recs: usize,
    first_row: usize,
    src: &[u8],
    out: &mut [u8],
) -> Result<(), String> {
    let (row, code_row, _) = row_widths(format, row_recs)?;
    let rows = src.len() / row;
    let (codes, heads) = out.split_at_mut(rows * code_row);
    split_pooled(format, row_recs, first_row, src, codes, heads)
}

/// Rows a slot of `slot_bytes` holds as one piece: whole rows, at least one;
/// `None` when one row's two planes do not fit the slot.
pub(crate) fn piece_rows(format: Format, row_recs: usize, slot_bytes: usize) -> Option<usize> {
    let (_, code_row, head_row) = row_widths(format, row_recs).ok()?;
    let rows = slot_bytes / (code_row + head_row).max(1);
    (rows > 0).then_some(rows)
}

#[cfg(test)]
mod tests {
    use super::{Format, fill_piece, piece_rows, split_rows, split_vecs};
    use crate::weights::q8_0_planes;
    use gguf::quant::Q8Block;

    /// A repeatable byte stream: an xorshift of `seed`, so a test names its input.
    fn stream(seed: u64, n: usize) -> Vec<u8> {
        let mut x = seed | 1;
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (x >> 24) as u8
            })
            .collect()
    }

    /// The valid file bytes of `rows` rows of `row_recs` records of `format`:
    /// random everywhere, then the scale bytes a format defines (UE4M3 below
    /// 0x7F, E8M0 inside its range, a finite f32 block scale) and, for the E4M3
    /// codes, no NaN payload.
    fn file(format: Format, row_recs: usize, rows: usize, seed: u64) -> Vec<u8> {
        let (_, head, code) = format.dims();
        let mut out = stream(seed, rows * row_recs * format.record_bytes());
        for blk in out.chunks_exact_mut(head + code) {
            let (h, c) = blk.split_at_mut(head);
            match format {
                Format::Q8_0 => {}
                Format::Nvfp4 => {
                    for b in h.iter_mut() {
                        *b &= 0x7E;
                    }
                }
                Format::Mxfp4 => h[0] = 2 + h[0] % 200,
                Format::Mxfp8 | Format::Fp8B128 => {
                    if format == Format::Mxfp8 {
                        h[0] = 100 + h[0] % 20;
                    } else {
                        h.copy_from_slice(&1.5f32.to_le_bytes());
                    }
                    for b in c.iter_mut() {
                        if *b & 0x7F == 0x7F {
                            *b ^= 0x40;
                        }
                    }
                }
            }
        }
        out
    }

    /// The scalar splitters the loader had before [`split_rows`], one per file
    /// format: the codes and scale bytes as the words (the Q8_0 scales as the
    /// bytes of their u16s) of the planes they built.
    fn reference(format: Format, src: &[u8]) -> (Vec<u32>, Vec<u8>) {
        let le = |b: &[u8]| -> Vec<u32> {
            b.as_chunks::<4>()
                .0
                .iter()
                .map(|w| u32::from_le_bytes(*w))
                .collect()
        };
        let (mut qs, mut e) = (Vec::new(), Vec::new());
        match format {
            Format::Q8_0 => {
                let blocks: Vec<Q8Block> = src
                    .as_chunks::<34>()
                    .0
                    .iter()
                    .map(Q8Block::from_bytes)
                    .collect();
                let (words, scales) = q8_0_planes(&blocks);
                qs = words;
                e = scales.iter().flat_map(|d| d.to_le_bytes()).collect();
            }
            Format::Nvfp4 | Format::Mxfp4 => {
                let (block, scales) = if format == Format::Nvfp4 {
                    (36, 4)
                } else {
                    (17, 1)
                };
                for blk in src.chunks_exact(block) {
                    let (head, codes) = blk.split_at(scales);
                    e.extend_from_slice(head);
                    qs.extend(le(codes));
                }
            }
            Format::Mxfp8 => {
                for group in src.as_chunks::<132>().0 {
                    for blk in group.as_chunks::<33>().0 {
                        e.push(blk[0]);
                        qs.extend(le(&blk[1..]));
                    }
                }
            }
            Format::Fp8B128 => {
                for group in src.as_chunks::<132>().0 {
                    let (scale, values) = group.split_at(4);
                    e.extend_from_slice(scale);
                    qs.extend(le(values));
                }
            }
        }
        (qs, e)
    }

    /// The pieces of an upload put back: `src` cut into pieces of the rows a
    /// slot of `slot` bytes holds, each through [`fill_piece`], the code and
    /// scale regions of each appended in turn as the device copies lay them out.
    fn through_pieces(
        format: Format,
        row_recs: usize,
        src: &[u8],
        slot: usize,
    ) -> (Vec<u8>, Vec<u8>) {
        let row = row_recs * format.record_bytes();
        let n = piece_rows(format, row_recs, slot).expect("one row fits the slot");
        let (mut codes, mut heads) = (Vec::new(), Vec::new());
        let mut first = 0;
        for run in src.chunks(n * row) {
            let rows = run.len() / row;
            let mut out = vec![0u8; rows * row_recs * (format.code_bytes() + format.head_bytes())];
            fill_piece(format, row_recs, first, run, &mut out).unwrap_or_else(|e| panic!("{e}"));
            let (c, h) = out.split_at(rows * row_recs * format.code_bytes());
            codes.extend_from_slice(c);
            heads.extend_from_slice(h);
            first += rows;
        }
        (codes, heads)
    }

    /// Every format's splitter gives the planes of the scalar splitter it
    /// replaced, on random valid bytes: cut into the upload's slot-sized pieces
    /// (the largest past the pool's threshold, so its tasks run on the
    /// workers), and, for the formats a host caller takes as words, whole.
    #[test]
    fn every_format_splits_to_the_scalar_planes_whole_and_in_pieces() {
        let words = |b: &[u8]| -> Vec<u32> {
            b.as_chunks::<4>()
                .0
                .iter()
                .map(|w| u32::from_le_bytes(*w))
                .collect()
        };
        for (format, row_recs, rows) in [
            (Format::Q8_0, 128, 6144),
            (Format::Q8_0, 4, 5),
            (Format::Nvfp4, 64, 5000),
            (Format::Nvfp4, 2, 3),
            (Format::Mxfp4, 8, 9000),
            (Format::Mxfp8, 32, 2500),
            (Format::Mxfp8, 1, 7),
            (Format::Fp8B128, 32, 2500),
        ] {
            let src = file(format, row_recs, rows, 0x5eed + rows as u64);
            let (want_q, want_e) = reference(format, &src);
            if format != Format::Q8_0 {
                let (q, e) = split_vecs("test", format, row_recs, rows, &src)
                    .unwrap_or_else(|e| panic!("{e}"));
                assert_eq!(q, want_q, "{format:?} codes whole");
                assert_eq!(e, words(&want_e), "{format:?} scales whole");
            }
            let row = row_recs * format.record_bytes();
            for slot in [3 * row * 2, 4 << 20, 32 << 20] {
                let (c, h) = through_pieces(format, row_recs, &src, slot);
                assert_eq!(words(&c), want_q, "{format:?} codes in pieces of {slot}");
                assert_eq!(h, want_e, "{format:?} scales in pieces of {slot}");
            }
        }
    }

    /// A row a check refuses is named by its row of the tensor, whichever
    /// piece or pool task it falls in, and the first such row is the one named.
    #[test]
    fn a_refused_row_is_named_by_its_row_of_the_tensor() {
        let rows = 16384;
        let mut src = file(Format::Nvfp4, 2, rows, 7);
        let row = 2 * 36;
        for r in [rows - 1, rows / 2] {
            src[r * row + 36 + 1] = 0xFF;
        }
        let want = format!("row {} block 1 sub-block 1", rows / 2);
        let err = split_vecs("test", Format::Nvfp4, 2, rows, &src).unwrap_err();
        assert!(err.to_string().contains(&want), "{err}");
        let n = piece_rows(Format::Nvfp4, 2, 1 << 20).expect("one row fits the slot");
        let mut first = 0;
        let mut named = None;
        for run in src.chunks(n * row) {
            let mut out = vec![0u8; run.len() / row * 2 * 36];
            if let Err(e) = fill_piece(Format::Nvfp4, 2, first, run, &mut out) {
                named = Some(e);
                break;
            }
            first += run.len() / row;
        }
        let named = named.expect("a piece refused its row");
        assert!(named.starts_with(&want), "{named}");
        let mut mx = file(Format::Mxfp8, 1, 4, 9);
        mx[132 + 33 + 1 + 3] = 0xFF;
        let (mut c, mut h) = (vec![0u8; 4 * 128], vec![0u8; 4 * 4]);
        let err = split_rows(Format::Mxfp8, 1, 0, &mx, &mut c, &mut h).unwrap_err();
        assert!(err.starts_with("row 1:"), "{err}");
    }
}
