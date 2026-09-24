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
//!   exact i32 (`|s_b| ≤ 32 · 12 · 127`), by `dp4a` on the device;
//! - `f = fma(d_w · d_x, s_b as f32, f)` from 0, `d_w = 2^(E − 128)` and
//!   `d_x` the scale of the 128-value block holding `b`;
//! - the caller sums the 32 lane values with the warp butterfly
//!   (`warp::reduce_sum_f32`).

use crate::GpuError;
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
        let [w0, w1, w2, w3] = v.0;
        let (l0, h0) = decode_word(w0);
        let (l1, h1) = decode_word(w1);
        let (l2, h2) = decode_word(w2);
        let (l3, h3) = decode_word(w3);
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
                    let s = dp4a_s32(l0, x[0], 0);
                    let s = dp4a_s32(l1, x[1], s);
                    let s = dp4a_s32(l2, x[2], s);
                    let s = dp4a_s32(l3, x[3], s);
                    let s = dp4a_s32(h0, x[4], s);
                    let s = dp4a_s32(h1, x[5], s);
                    let s = dp4a_s32(h2, x[6], s);
                    let s = dp4a_s32(h3, x[7], s);
                    f[$c] = (dw * dx).mul_add(s as f32, f[$c]);
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
