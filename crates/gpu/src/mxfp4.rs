//! MXFP4 weights against q8_1 activations: the card layout of an MXFP4
//! tensor, one lane's share of a row's dot (device-callable), and the same
//! rule on the host.
//!
//! The file format is ggml's `block_mxfp4`: 17 bytes per 32 values, byte 0
//! the E8M0 scale `E`, bytes 1..17 the codes; value `j < 16` is the low
//! nibble of code byte `j` and value `j + 16` its high nibble; code `c` means
//! `KVALUES_MXFP4[c] · 2^(E − 128)` (`gguf::quant::e8m0_to_f32_half`).
//!
//! The card layout ([`repack`]) splits a tensor into two word planes, the
//! same 17 bytes per block: `qs`, each block's 16 code bytes as four
//! little-endian words, blocks in row order (block `b` of row `r` at vector
//! `r · n_blk + b`, one 16-byte load); and `e`, the scale bytes packed four
//! to a word in the same order. The repack refuses a scale outside
//! [`E_MIN`]..=[`E_MAX`], so the kernel's scale is a normal f32 power of two.
//!
//! The file layout is also read as it is ([`lane_partial_file`]): a block `g`
//! of a stream of whole rows starts at byte `17g` and is read through the
//! five words that cover it, so the stack stays the file's blocks
//! (`CardFormat::KQuant`). That path takes every scale byte by ggml's half
//! rule ([`scale_half`]); the planes path takes the normal range only.
//!
//! Activations are the q8_1 quantizer's (`q8_1_quant_block`): per 128-value
//! block a scale `d_x` and int8 codes, stored in the Q4_K gemv's permutation,
//! where value-order word `v` of a column sits at slot `256·(v >> 8) +
//! 32·(v & 7) + 8·((v >> 6) & 3) + ((v >> 3) & 7)`. Word `i` of 32-value
//! block `b` (`v = 8b + i`) is then at `256·(b >> 5) + 32·i + (b & 31)`: a
//! lane that owns block `32g + L` reads its eight words lane-consecutive.
//!
//! The rule, one row against one column, both sides:
//! - lane `L` of the row's warp takes blocks `b = L, L + 32, …` ascending;
//! - block sum `s_b = Σ_j KVALUES_MXFP4[c_j] · q_j` over the 32 values, an
//!   exact i32 (`|s_b| ≤ 32 · 12 · 127`), by `dp4a` on the device
//!   ([`decode_block`], then [`block_fma`]);
//! - `f = fma(d_w · d_x, s_b as f32, f)` from 0, `d_w = 2^(E − 128)` and
//!   `d_x` the scale of the 128-value block holding `b` ([`block_fma`]);
//! - the caller sums the 32 lane values with the warp butterfly
//!   (`warp::reduce_sum_f32`).

use crate::GpuError;
use crate::kquant::q8_0::q8_0_code;
use cuda_device::dotprod::dp4a_s32;
use cuda_device::prmt::prmt;
use cuda_device::vector::U32x4;
use gguf::quant::KVALUES_MXFP4;

/// Values per MXFP4 block.
pub const BLOCK_VALUES: usize = 32;
/// File bytes per MXFP4 block: the scale byte and 16 code bytes.
pub const BLOCK_BYTES: usize = 17;
/// Code words per block in the card layout's `qs` plane.
pub const QS_WORDS: usize = 4;
/// The smallest scale byte the card layout takes: `E = 0` and `1` are the
/// subnormal scales of ggml's half rule.
pub const E_MIN: u8 = 2;
/// The largest: `E = 255` is NaN in the OCP MX format (ggml decodes it as
/// `2^127`).
pub const E_MAX: u8 = 254;

/// `KVALUES_MXFP4` bytes `4i .. 4i + 4` as a little-endian word: the four
/// `prmt` table words.
const fn table_word(i: usize) -> u32 {
    let k = KVALUES_MXFP4;
    (k[4 * i] as u8 as u32)
        | ((k[4 * i + 1] as u8 as u32) << 8)
        | ((k[4 * i + 2] as u8 as u32) << 16)
        | ((k[4 * i + 3] as u8 as u32) << 24)
}
const KV0: u32 = table_word(0);
const KV1: u32 = table_word(1);
const KV2: u32 = table_word(2);
const KV3: u32 = table_word(3);

/// The eight codes of one `qs` word decoded to int8 values: the low nibbles
/// of its four bytes (values `4i .. 4i + 4` of the block for word `i`) and
/// the high nibbles (values `16 + 4i ..`), each as four bytes in byte order.
///
/// A register table read by byte permute: `prmt` in its generic mode indexes
/// eight bytes of two words with the low three bits of each selector nibble
/// (bit 3 would ask for sign replication, so it is masked off), once into
/// the codes `0..8` and once into `8..16`; a third permute picks between the
/// two by each code's bit 3, and two more gather the low and the high
/// nibbles. No memory is touched: a shared table would cost a load per code.
#[inline(always)]
fn decode_word(qs: u32) -> (u32, u32) {
    let sel = qs & 0x7777_7777;
    let pick = 0x3210_3210 | ((qs & 0x8888_8888) >> 1);
    let a = prmt(prmt(KV0, KV1, sel), prmt(KV2, KV3, sel), pick);
    let b = prmt(
        prmt(KV0, KV1, sel >> 16),
        prmt(KV2, KV3, sel >> 16),
        pick >> 16,
    );
    (prmt(a, b, 0x6420), prmt(a, b, 0x7531))
}

/// `2^(E − 128)` for a scale byte `E` in [`E_MIN`]..=[`E_MAX`]: the f32 whose
/// exponent field is `E − 1` — ggml's half rule on its normal range.
#[inline(always)]
fn scale_of(e: u32) -> f32 {
    f32::from_bits((e - 1) << 23)
}

/// `2^(E − 128)` for any scale byte `E`: ggml's half rule
/// (`gguf::quant::e8m0_to_f32_half`), `E >= 2` the f32 whose exponent field
/// is `E − 1` (255 is `2^127`), and `E = 0`, `1` the subnormals
/// `2^-129`, `2^-128`.
#[inline(always)]
fn scale_half(e: u32) -> f32 {
    f32::from_bits(if e >= 2 {
        (e - 1) << 23
    } else {
        0x0020_0000 << e
    })
}

/// The eight int8 words one block's four `qs` words decode to, in value
/// order: the low-nibble words of `qs[0..4]`, then their high-nibble words
/// ([`decode_word`]). The first half of the block rule.
#[inline(always)]
pub fn decode_block(qs: [u32; 4]) -> [u32; 8] {
    let (l0, h0) = decode_word(qs[0]);
    let (l1, h1) = decode_word(qs[1]);
    let (l2, h2) = decode_word(qs[2]);
    let (l3, h3) = decode_word(qs[3]);
    [l0, l1, l2, l3, h0, h1, h2, h3]
}

/// The second half of the block rule, one block against one column: the
/// exact i32 block sum of the decoded words `dec` ([`decode_block`]) with the
/// column's eight activation words `x` (value order), by eight `dp4a`, then
/// `fma(d_w · d_x, s, f)`.
#[inline(always)]
pub fn block_fma(dec: &[u32; 8], x: &[u32; 8], dw: f32, dx: f32, f: f32) -> f32 {
    let s = dp4a_s32(dec[0], x[0], 0);
    let s = dp4a_s32(dec[1], x[1], s);
    let s = dp4a_s32(dec[2], x[2], s);
    let s = dp4a_s32(dec[3], x[3], s);
    let s = dp4a_s32(dec[4], x[4], s);
    let s = dp4a_s32(dec[5], x[5], s);
    let s = dp4a_s32(dec[6], x[6], s);
    let s = dp4a_s32(dec[7], x[7], s);
    (dw * dx).mul_add(s as f32, f)
}

/// Block `g` of a file-layout stream (the rows' 17-byte blocks back to back as
/// little-endian words): its scale by [`scale_half`] and its four `qs` words.
///
/// The block starts at byte `17g`, word `17g >> 2`, `17g & 3 = g & 3` bytes
/// in, and its last byte is in the fifth word from there for every `g`, so
/// five words cover it and none lies past the block's own last byte. The
/// scale is byte `g & 3` of the first word; code word `i` is the
/// little-endian pair of words `i`, `i + 1` shifted right by `8 · (g & 3) + 8`
/// ([`q8_0_code`], whose shift reaches 32 for `g & 3 = 3`).
///
/// The caller keeps `w` holding the stream's word of block `g`'s last byte,
/// `(17g + 16) >> 2` (rust-quality R28: a crate-internal fn over unchecked
/// reads under a caller-kept boundary).
#[inline(always)]
pub(crate) fn file_block(w: &[u32], g: usize) -> (f32, [u32; 4]) {
    let at = BLOCK_BYTES * g;
    let i = at >> 2;
    let o = (at & 3) as u32;
    // SAFETY: words i .. i + 4 end at (17g + 16) >> 2, inside `w` by this
    // fn's contract.
    let v = unsafe {
        [
            *w.get_unchecked(i),
            *w.get_unchecked(i + 1),
            *w.get_unchecked(i + 2),
            *w.get_unchecked(i + 3),
            *w.get_unchecked(i + 4),
        ]
    };
    let sh = 8 * o + 8;
    (
        scale_half((v[0] >> (8 * o)) & 0xff),
        [
            q8_0_code(v[0], v[1], sh),
            q8_0_code(v[1], v[2], sh),
            q8_0_code(v[2], v[3], sh),
            q8_0_code(v[3], v[4], sh),
        ],
    )
}

/// Lane `lane`'s partial of one row of a file-layout stream against one
/// q8_1 column, the module doc's rule: row `row`'s block `b` is block `row ·
/// n_blk + b` of the stream ([`file_block`]), the column's words are read as
/// in [`lane_partials`] from `q0` and its scales from `d0`.
///
/// The caller keeps `n_blk` a multiple of 4 (so a row is whole words), `w`
/// holding the stream through the last block of row `row` ([`file_block`]'s
/// boundary), `q.len() >= q0 + 256 · ceil(n_blk / 32)`, `d8.len() >= d0 +
/// n_blk / 4` and `lane < 32` (rust-quality R28).
#[allow(
    clippy::too_many_arguments,
    reason = "device core: it is handed a kernel entry's flat arguments (rust-quality R8)"
)]
#[inline(always)]
pub(crate) fn lane_partial_file(
    w: &[u32],
    row: usize,
    n_blk: usize,
    q: &[u32],
    d8: &[f32],
    q0: usize,
    d0: usize,
    lane: usize,
) -> f32 {
    let g0 = row * n_blk;
    let mut f = 0.0f32;
    let mut b = lane;
    while b < n_blk {
        // b < n_blk, so block g0 + b is inside the row, inside `w` by this
        // fn's contract.
        let (dw, qs) = file_block(w, g0 + b);
        let dec = decode_block(qs);
        let qb = q0 + 256 * (b >> 5) + (b & 31);
        // SAFETY: qb + 224 < q0 + 256·ceil(n_blk/32) for b < n_blk, so the
        // eight words qb + 32i are inside the column, and scale d0 + b/4
        // inside its scales, both by this fn's contract.
        let (x, dx) = unsafe {
            (
                [
                    *q.get_unchecked(qb),
                    *q.get_unchecked(qb + 32),
                    *q.get_unchecked(qb + 64),
                    *q.get_unchecked(qb + 96),
                    *q.get_unchecked(qb + 128),
                    *q.get_unchecked(qb + 160),
                    *q.get_unchecked(qb + 192),
                    *q.get_unchecked(qb + 224),
                ],
                *d8.get_unchecked(d0 + (b >> 2)),
            )
        };
        f = block_fma(&dec, &x, dw, dx, f);
        b += 32;
    }
    f
}

/// Lane `lane`'s partial sums of one MXFP4 row against `M` q8_1 columns, the
/// module doc's rule: each block's codes are loaded and decoded once and
/// dotted with every column whose bit is set in `mask`. Column `c` reads its
/// q8_1 words from `q0 + c · q_col` and its scales from `d0 + c · d_col`;
/// columns `M..8` and columns outside `mask` stay 0.0.
///
/// # Safety
///
/// `qs` holds at least `(row + 1) · n_blk` vectors and `e` at least
/// `(row + 1) · n_blk / 4` words, `n_blk` a multiple of 4; for every column
/// `c < M`, `q.len() >= q0 + c · q_col + 256 · ceil(n_blk / 32)` and
/// `d8.len() >= d0 + c · d_col + n_blk / 4`; `lane < 32`; every scale byte of
/// the row is in [`E_MIN`]..=[`E_MAX`] (the repack's check).
#[allow(
    clippy::too_many_arguments,
    reason = "device core: it is handed a kernel entry's flat arguments (rust-quality R8)"
)]
#[inline(always)]
pub unsafe fn lane_partials<const M: usize>(
    qs: &[U32x4],
    e: &[u32],
    row: usize,
    n_blk: usize,
    q: &[u32],
    d8: &[f32],
    q0: usize,
    q_col: usize,
    d0: usize,
    d_col: usize,
    mask: u32,
    lane: usize,
) -> [f32; 8] {
    const {
        assert!(M > 0);
        assert!(M <= 8);
    };
    let mut f = [0.0f32; 8];
    let blk0 = row * n_blk;
    let mut b = lane;
    while b < n_blk {
        // SAFETY: b < n_blk, so vector blk0 + b and scale word (blk0 + b)/4
        // are inside the row, inside qs and e by this fn's contract.
        let (v, ew) = unsafe {
            (
                *qs.get_unchecked(blk0 + b),
                *e.get_unchecked((blk0 + b) >> 2),
            )
        };
        let dw = scale_of((ew >> (8 * (b & 3))) & 0xff);
        let dec = decode_block(v.0);
        let slot = 256 * (b >> 5) + (b & 31);
        macro_rules! col {
            ($c:literal) => {
                if M > $c && (mask >> $c) & 1 != 0 {
                    let qb = q0 + $c * q_col + slot;
                    // SAFETY: slot + 224 < 256·ceil(n_blk/32) for b < n_blk,
                    // so the eight words qb + 32i are inside column $c's
                    // words, and scale d0 + $c·d_col + b/4 inside its
                    // scales, both inside q and d8 by this fn's contract.
                    let (x, dx) = unsafe {
                        (
                            [
                                *q.get_unchecked(qb),
                                *q.get_unchecked(qb + 32),
                                *q.get_unchecked(qb + 64),
                                *q.get_unchecked(qb + 96),
                                *q.get_unchecked(qb + 128),
                                *q.get_unchecked(qb + 160),
                                *q.get_unchecked(qb + 192),
                                *q.get_unchecked(qb + 224),
                            ],
                            *d8.get_unchecked(d0 + $c * d_col + (b >> 2)),
                        )
                    };
                    f[$c] = block_fma(&dec, &x, dw, dx, f[$c]);
                }
            };
        }
        col!(0);
        col!(1);
        col!(2);
        col!(3);
        col!(4);
        col!(5);
        col!(6);
        col!(7);
        b += 32;
    }
    f
}

/// Lane `lane`'s partial sums of one file-layout row against `m` q8_1 columns
/// of a stream of columns, the module doc's rule: each block is read through
/// [`file_block`] and decoded once and dotted with columns `cols[0] .. cols[m]`,
/// column `c` of the stream at `cols[c] · q_col` words of `q` and `cols[c] ·
/// d_col` scales of `d8`. Column `c`'s value is [`lane_partial_file`]'s for that
/// column bit for bit; columns `m..8` stay 0.0. `m` is warp-uniform. The column
/// offsets are 32-bit, so each stays in a register across the block loop.
///
/// The caller keeps `1 <= m <= 8`, `n_blk` a multiple of 4 below 2^32, `w`
/// holding the stream through the last block of row `row` ([`file_block`]'s
/// boundary), and for every `c < m`: `cols[c] · q_col + 256 · ceil(n_blk / 32)
/// <= q.len()` and `cols[c] · d_col + n_blk / 4 <= d8.len()`, both below 2^32
/// (no offset wraps); `lane < 32` (rust-quality R28).
#[allow(
    clippy::too_many_arguments,
    reason = "device core: it is handed a kernel entry's flat arguments (rust-quality R8)"
)]
#[inline(always)]
pub(crate) unsafe fn lane_partials_file_cols(
    w: &[u32],
    row: usize,
    n_blk: usize,
    q: &[u32],
    d8: &[f32],
    cols: [u32; 8],
    q_col: u32,
    d_col: u32,
    m: usize,
    lane: usize,
) -> [f32; 8] {
    let mut f = [0.0f32; 8];
    let g0 = row * n_blk;
    let n = n_blk as u32;
    let mut b = lane as u32;
    while b < n {
        // b < n_blk, so block g0 + b is inside the row, inside `w` by this
        // fn's contract.
        let (dw, qs) = file_block(w, g0 + b as usize);
        let dec = decode_block(qs);
        let slot = 256 * (b >> 5) + (b & 31);
        macro_rules! col {
            ($c:literal) => {
                if m > $c {
                    let qb = (cols[$c] * q_col + slot) as usize;
                    // SAFETY: slot + 224 < 256·ceil(n_blk/32) for b < n_blk,
                    // so the eight words qb + 32i are inside column cols[$c],
                    // and scale cols[$c]·d_col + b/4 inside its scales, both
                    // by this fn's contract.
                    let (x, dx) = unsafe {
                        (
                            [
                                *q.get_unchecked(qb),
                                *q.get_unchecked(qb + 32),
                                *q.get_unchecked(qb + 64),
                                *q.get_unchecked(qb + 96),
                                *q.get_unchecked(qb + 128),
                                *q.get_unchecked(qb + 160),
                                *q.get_unchecked(qb + 192),
                                *q.get_unchecked(qb + 224),
                            ],
                            *d8.get_unchecked((cols[$c] * d_col + (b >> 2)) as usize),
                        )
                    };
                    f[$c] = block_fma(&dec, &x, dw, dx, f[$c]);
                }
            };
        }
        col!(0);
        col!(1);
        col!(2);
        col!(3);
        col!(4);
        col!(5);
        col!(6);
        col!(7);
        b += 32;
    }
    f
}

// ------------------------------------------------------------------ host

/// The rule's block sum on the host: the 32 codes of a file block
/// (`blk[1..17]`, the scale byte ignored) against 32 int8 activation codes
/// in value order.
#[must_use]
pub fn block_sum_host(blk: &[u8; BLOCK_BYTES], xq: &[i8; BLOCK_VALUES]) -> i32 {
    let kv = |c: u8| i32::from(KVALUES_MXFP4[usize::from(c)]);
    let mut s = 0i32;
    for (j, &byte) in blk[1..].iter().enumerate() {
        s += kv(byte & 0x0f) * i32::from(xq[j]) + kv(byte >> 4) * i32::from(xq[j + 16]);
    }
    s
}

/// Lane `lane`'s partial of one row on the host, the module doc's rule:
/// `row` the row's file bytes (`17 · n_blk`), `xq` the column's int8 codes in
/// value order (`32 · n_blk`), `d8` its 128-value block scales.
#[must_use]
pub fn lane_partial_host(row: &[u8], xq: &[i8], d8: &[f32], lane: usize) -> f32 {
    let blocks = row.as_chunks::<BLOCK_BYTES>().0;
    let codes = xq.as_chunks::<BLOCK_VALUES>().0;
    let mut f = 0.0f32;
    let mut b = lane;
    while b < blocks.len() {
        let dw = gguf::quant::e8m0_to_f32_half(blocks[b][0]);
        let s = block_sum_host(&blocks[b], &codes[b]);
        f = (dw * d8[b / 4]).mul_add(s as f32, f);
        b += 32;
    }
    f
}

/// An MXFP4 tensor in the card layout (the module doc): `qs` holds `4 ·
/// n_blk` words per row, `e` holds `n_blk / 4` words per row.
pub struct Planes {
    pub qs: Vec<u32>,
    pub e: Vec<u32>,
}

/// The card layout of `rows` MXFP4 rows of `k` values from their file bytes.
/// `k` must be a positive multiple of 128 (whole scale words per row) and
/// `bytes` exactly the rows' `17 · k / 32 · rows` bytes; a scale byte outside
/// [`E_MIN`]..=[`E_MAX`] is refused, naming its row and block.
pub fn repack(bytes: &[u8], rows: usize, k: usize) -> Result<Planes, GpuError> {
    let what = "mxfp4::repack";
    if k == 0 || !k.is_multiple_of(128) {
        return Err(GpuError::shape(
            what,
            format!("k {k} is not a positive multiple of 128"),
        ));
    }
    let n_blk = k / BLOCK_VALUES;
    if bytes.len() != rows * n_blk * BLOCK_BYTES {
        return Err(GpuError::shape(
            what,
            format!(
                "{} bytes for {rows} rows of {k} values, want {}",
                bytes.len(),
                rows * n_blk * BLOCK_BYTES
            ),
        ));
    }
    let blocks = bytes.as_chunks::<BLOCK_BYTES>().0;
    let mut qs = Vec::with_capacity(blocks.len() * QS_WORDS);
    let mut e = vec![0u32; blocks.len() / 4];
    for (i, blk) in blocks.iter().enumerate() {
        if !(E_MIN..=E_MAX).contains(&blk[0]) {
            return Err(GpuError::shape(
                what,
                format!(
                    "row {} block {}: scale byte {} is outside {E_MIN}..={E_MAX}",
                    i / n_blk,
                    i % n_blk,
                    blk[0]
                ),
            ));
        }
        e[i / 4] |= u32::from(blk[0]) << (8 * (i % 4));
        for w in blk[1..].as_chunks::<4>().0 {
            qs.push(u32::from_le_bytes(*w));
        }
    }
    Ok(Planes { qs, e })
}
