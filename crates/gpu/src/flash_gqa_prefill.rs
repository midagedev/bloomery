//! Grouped-query flash attention for a prompt's ubatch: `T` query rows at
//! consecutive positions, each attending over its own live key count
//! (`n_keys[t]`, the causal limit of its position), [`GROUP`] query heads
//! sharing one key/value head of [`HEAD`] values over the cache planes
//! `[n_kv][ctx][HEAD]` f16 that `rope_neox::head_norm_neox_append` writes.
//!
//! Geometry: block `(query tile, kv head)` owns [`POSITIONS`] consecutive
//! rows and the group's eight heads of each — 64 query rows. Each of its four
//! warps owns two positions, one m16 fragment: rows `0..8` are the first
//! position's heads, rows `8..16` the second's. The block walks its keys once,
//! in [`KEY_TILE`]-key tiles at absolute positions (tile `i` holds keys
//! `[64·i, 64·i + 64)` whatever the ubatch's first position), ascending, and
//! stages each tile's key and value rows into shared memory once for all of
//! its rows: the value tile is copied (`cp.async`) while the scores run on the
//! key tile, the next key tile while the values are accumulated. A staged row
//! is the cache row's 256 bytes unpadded, its sixteen 16-byte chunks permuted
//! within each 128-byte half by `chunk ^ (row & 7)` ([`staged_word`]): each
//! 32-byte source sector lands in one 32-byte shared sector, and the eight rows
//! of one `ldmatrix` phase hit the thirty-two banks once.
//!
//! Per tile and warp:
//! 1. `S = Q·Kᵀ` on `mma.m16n8k16`: the query rows rounded to f16 once per
//!    block (`f32x2_to_f16x2_bits`: `f32_to_f16_bits` of each value, nearest
//!    even) and held in registers, the head's eight k16 steps ascending from
//!    zero, then `s = S · scale` rounded
//!    — the scores `gqa_flash_seg_mma` computes for the same row and key, bit
//!    for bit. A key at or past the row's count is `−inf`.
//! 2. Per query row: the tile max (the lane's 16 values, then the xor
//!    butterfly over the row's four lanes), `m' = max(m, tile max)`, on a bump
//!    `f = dev_exp(m − m')` (0 for the first) else `f = 1`; the weights
//!    `p = exp_weight(s − m')` (0 for a masked key) rounded to f16; the lane's
//!    running sum `l = l·f + Σ p̂` of the rounded weights, and the output
//!    accumulators times `f` — skipped when every lane of the warp has
//!    `f = 1`, a multiply that would return every accumulator unchanged.
//!
//! A tile whose keys are all below both of a warp's counts (every tile but
//! the one holding the warp's first count, and the ones past it) runs the
//! same arithmetic without the per-key mask, and a tile whose keys are all
//! below the block's largest count is staged without the per-copy bound:
//! on such a tile every mask and bound the general path tests is true, so
//! both paths evaluate the same expressions and write the same bits.
//! 3. `O += P̂·V` on `mma.m16n8k16`, the rounded weights as the A fragment
//!    straight from the score registers, the value tile through
//!    `ldmatrix.trans`, f32 accumulation, the tile's four k16 steps ascending.
//!
//! After the last tile: the row sum over its four lanes (`(l + l^1) + l^2`)
//! and `o · (1/l)`. Summing the rounded weights makes the normalizer the sum
//! of the weights the values were multiplied by, so the f16 rounding of `p`
//! moves the output by the weights' deviation times `|v − o|`, not times `|v|`.
//!
//! Reduction structure, the fixed contract (reruns are bit-identical): a
//! row's arithmetic depends only on its own query row, the key and value rows
//! below its count and the fixed tile partition — not on `T`, the first
//! position, the rows that share its block or the grid. Nothing is reduced
//! across rows. A tile at or past both of a warp's counts is skipped by that
//! warp (its state is not touched); a tile past only the warp's first row's
//! count gives that row `−inf` at every key, so its max, rescale and sum are
//! unchanged and its output rows gain the products of all-zero weights — the
//! tensor core returns the accumulator unchanged.
//!
//! Keys at or past the block's largest count are never loaded: the staging
//! writes zeros for them, so a padded cache row holding a NaN pattern reaches
//! no result. Keys between a row's count and the block's largest are loaded
//! and masked — rows a prefill has just written, since the block's rows are
//! consecutive positions; a non-finite value in such a key's value row still
//! reaches the masking row's output through its zero weight (`0 · NaN`), so
//! it shows as NaN rather than being dropped. A count of zero or past `ctx`
//! on a row below `T` raises
//! [`FaultSite::KeyCount`], and that row's output is NaN; no key is read for
//! it.
//!
//! [`FlashGqaPrefill::enqueue_split`] serves the same attention to a group of
//! any positive multiple of [`GROUP`] query heads a key head: one plain launch
//! a sub-group of eight, the rows moved through a caller-owned
//! [`SplitScratch`].
//!
//! [`FlashGqaPrefill::enqueue_256`] runs the same attention over heads of
//! [`HEAD_256`] values in eight warps a block (`gqa_prefill_flash_256`, the
//! section before the kernels). [`FlashGqaPrefill::enqueue_256_p4`] runs it
//! for groups of any multiple of four query heads (`gqa_prefill_flash_256_p4`):
//! a block takes four heads of sixteen positions, an m16 fragment four
//! positions × four heads, and a key head's group is `group / 4` blocks; a
//! row's bits are the eight-head kernel's for the same row. Its body is
//! `prefill_256_p`, a PACK-generic form of the eight-head kernel's.
//! [`FlashGqaPrefill::enqueue_256_p2`] runs the same body in packs of two
//! (`gqa_prefill_flash_256_p2`) for a group that is even and not a multiple of
//! four (Qwen3.5-9B's 24/4): a block takes two heads of thirty-two positions,
//! an m16 fragment eight positions × two heads.
//!
//! [`FlashGqaPrefill::enqueue_256_p2_lower`] (`gqa_prefill_flash_256_p2_lower`)
//! is the pack-of-two walk with a lower bound a row: row `t` attends over keys
//! `[k_first[t], n_keys[t])`, a window the host sets (an encoder's symmetric
//! sliding window; a global layer of a non-causal pass is `k_first = 0`,
//! `n_keys = n`). The block starts at the tile holding its smallest first key
//! and stages zeros below it, as it does at and past its largest count; a key
//! below a row's first is `−inf` and weighs 0, as one at or past its count. A
//! tile below every first key of a warp's rows is skipped by that warp, and a
//! tile that only some rows of a lane see leaves the others' state untouched
//! (their max stays `−inf`, no rescale, zero weights), so over finite cache
//! values a row's bits are still those of its one-row launch. A non-finite
//! value below a row's first key that the block stages for another row
//! reaches that row through its zero weight, as one between a row's count and
//! the block's largest does. A first key at or past the row's count is
//! refused as a bad count is.
//!
//! The `_q8` entries ([`FlashGqaPrefill::enqueue_q8`] and its eight-head
//! `_256` kin; no packed q8 entry exists) run the same walks over the Q8_0
//! cache the quantizing appends write (`rope_neox`'s two-plane layout: a
//! codes plane of `head/4` u32 a row, a scales plane of `head/32` u16, a
//! pair of planes a side). The `_256_p4`, `_256_p2` and `_256_q8` entries
//! are the instances of one body, `prefill_256_p`, over [`F16Kv`] and
//! [`Q8Kv`]; `gqa_prefill_flash_q8` holds `gqa_prefill_flash`'s body with the
//! q8 staging, because moving that f16 entry's kernel body into a helper
//! changes the entry's code at the toolchain pin. The
//! tensor cores need the f16 tile the walk reads, so a q8 tile's staging
//! becomes two steps: `cp.async` carries each tile's
//! raw code words into the low half of the f16 tile's own rows (a code row
//! is half a staged row's words) and its scales into a small scales tile,
//! and a dequant step widens the codes in place — each warp owns whole
//! staged rows, reads its rows' code words and scales into registers,
//! syncs the warp (a row's staged writes may land on its own raw words),
//! and writes the dequantized f16 pairs to the permuted layout the
//! `ldmatrix` addresses walk ([`flash_gqa::q8_word_values`] through
//! `f32x2_to_f16x2_bits`): every key below the block's largest count as the
//! f16 rounding of `code·d`, a key past it four zero words — the f16 walk's
//! own zeros, so a refused block's NaN scale reaches a row's output only
//! through a key below its count, never past it. The raw copies keep the
//! f16 walk's overlap (the value side's landing under the scores, the next
//! key tile's under the value product); the dequants add the two barriers
//! the in-place widening needs. A q8 256 block's dynamic shared memory grows
//! by the two scales tiles (100,352 bytes, still inside the card's opt-in
//! window); the 128 entry's are static. The query staging, the scores, the
//! softmax and the value products are the f16 walk's — the q8 read's
//! dequantized values are not the f16 cache's bits, and the owning gates
//! hold these entries against the dequantized cache with the f16 rounding
//! of the widened tiles said in their bands.

use crate::fault::{FaultSink, FaultSite};
use crate::flash::{f32x2_to_f16x2_bits, half_bits_to_f32};
use crate::flash_gqa::{F16Kv, GROUP, HEAD, KvSrc, PACK_2, PACK_4, Q8Kv, q8_word_values};
use crate::rope_neox::q8_plane_lens;
use crate::tensor::{Pitched, dtod_2d};
use crate::{GpuError, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::async_copy::{
    cp_async_ca_8, cp_async_cg_16, cp_async_commit_group, cp_async_wait_group,
};
use cuda_device::float::{add_rn_f32, mul_rn_f32};
use cuda_device::ptx_asm;
use cuda_device::vector::U32x4;
use cuda_device::{
    DisjointSlice, DynamicSharedArray, SharedArray, kernel, launch_bounds, launch_contract, shared,
    thread, warp, wmma,
};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Keys one tile stages. Tile `i` holds keys `[64·i, 64·i + 64)`.
pub const KEY_TILE: usize = 64;
/// Query positions one block covers: its four warps take two each.
pub const POSITIONS: usize = 8;
const WARPS: usize = POSITIONS / 2;
const THREADS: usize = WARPS * 32;
const THREADS_U32: u32 = THREADS as u32;
const _: () = assert!(THREADS_U32 as usize == THREADS);
/// Query rows one block covers: every position's group of heads, staged
/// through the key tile before the first key tile lands.
const Q_ROWS: usize = POSITIONS * GROUP;
/// 16-byte chunks of one cache row, and per thread per staged tile.
const ROW_CHUNKS: usize = HEAD / 8;
const CHUNKS: usize = KEY_TILE * ROW_CHUNKS / THREADS;
/// f16 pairs of one query row, and per thread in the query staging.
const Q_PAIRS: usize = HEAD / 2;
const Q_WORDS: usize = Q_ROWS * Q_PAIRS / THREADS;
/// Staged query rows one pass of the block's loads covers, and the passes
/// that cover one position's group of heads.
const Q_ROW_STEP: usize = THREADS / Q_PAIRS;
const Q_POS_PASSES: usize = GROUP / Q_ROW_STEP;
/// The score's k16 steps over the head, and its n8 tiles of keys.
const QK_STEPS: usize = HEAD / 16;
const KEY_NT: usize = KEY_TILE / 8;
/// The value product's k16 steps over a tile's keys, and its n16 pairs of
/// dims (one `ldmatrix.x4.trans` each).
const PV_STEPS: usize = KEY_TILE / 16;
const DIM_PAIRS: usize = HEAD / 16;

/// Keys one pass of the block's copies covers: a tile is staged in
/// [`CHUNKS`] passes, thread `tid` copying column `tid % ROW_CHUNKS` of key
/// `tid / ROW_CHUNKS` of each.
const PASS_KEYS: usize = THREADS / ROW_CHUNKS;
/// u32 words of one cache row, and of one tile of rows — in the cache plane
/// and staged alike.
const ROW_WORDS: usize = HEAD / 2;
const TILE_WORDS: usize = KEY_TILE * ROW_WORDS;
const KEY_TILE_U32: u32 = KEY_TILE as u32;

/// The tile permutation, expanded inside each width's address helper.
macro_rules! staged {
    (@offset $xor:expr, $w:expr) => {
        ((($w >> 2) ^ $xor) << 2) | ($w & 3)
    };
    ($row:expr, $w:expr, $row_words:expr) => {
        $row * $row_words + staged!(@offset $row & 7, $w)
    };
}

/// The staged word of word `w` of row `row` in a tile of `row_words`-word
/// rows: `flash_gqa_prefill`'s permutation within each 128-byte line.
#[inline(always)]
pub(crate) const fn staged(row: usize, w: usize, row_words: usize) -> usize {
    staged!(row, w, row_words)
}

/// The layout's facts over a tile of `rows` rows of `row_words` words: every
/// chunk stays in its 128-byte line and every logical 32-byte pair of chunks
/// in one aligned 32-byte sector; word `w` of row `row` is the same word of
/// row `row % 8` in the first line plus whole rows and whole lines, so an
/// address `8·i` rows or a line further on is a lane's address plus a
/// constant.
pub(crate) const fn layout_holds(rows: usize, row_words: usize) -> bool {
    let mut row = 0;
    while row < rows {
        let mut w = 0;
        while w < row_words {
            let at = staged(row, w, row_words) - row * row_words;
            let mate = staged(row, w ^ 4, row_words) - row * row_words;
            let base = staged(row % 8, w % LINE, row_words);
            if at / LINE != w / LINE
                || at / 8 != mate / 8
                || staged(row, w, row_words) != base + (row - row % 8) * row_words + (w - w % LINE)
            {
                return false;
            }
            w += 1;
        }
        row += 1;
    }
    true
}

/// The staged word in a row of `HEAD` values.
#[inline(always)]
const fn staged_word(row: usize, w: usize) -> usize {
    staged!(row, w, ROW_WORDS)
}

/// u32 words of half a staged row: the XOR never crosses it. An `ldmatrix`
/// step reads a pair of chunks (8 words), so a half-row is HALF_PAIRS steps.
const HALF_ROW: usize = ROW_WORDS / 2;
const HALF_PAIRS: usize = HALF_ROW / 8;

const _: () = assert!(GROUP == 8 && Q_ROWS == WARPS * 16 && Q_ROWS <= KEY_TILE);
const _: () = assert!(PASS_KEYS * ROW_CHUNKS == THREADS && PASS_KEYS * CHUNKS == KEY_TILE);
// A row is whole 128-byte lines, so a row's 32-byte sectors and halves are
// aligned whenever the tile base is (the tiles are 128-byte aligned).
const _: () = assert!((ROW_WORDS * 4).is_multiple_of(128));
// The XOR takes the low three chunk bits: a row needs whole groups of eight
// chunks for it to stay inside the row.
const _: () = assert!(ROW_CHUNKS.is_multiple_of(8));
// Tile rows start at key 64·i, so a row's `row & 7` is its key's; eight
// consecutive rows (an `ldmatrix` phase) take every value of it once.
const _: () = assert!(KEY_TILE.is_multiple_of(8));
// A thread's copies of a tile are PASS_KEYS rows apart and so share `row & 7`:
// its destinations are one constant plus a pass stride.
const _: () = assert!(PASS_KEYS.is_multiple_of(8));
const _: () = assert!(layout_holds(KEY_TILE, ROW_WORDS));
const _: () = assert!(CHUNKS * THREADS == KEY_TILE * ROW_CHUNKS);
const _: () = assert!(Q_WORDS * THREADS == Q_ROWS * Q_PAIRS);
// Pass `e` of a thread is row `Q_ROW_STEP·e + tid / Q_PAIRS`: position
// `e / Q_POS_PASSES`, head `Q_ROW_STEP·(e % Q_POS_PASSES) + tid / Q_PAIRS`.
const _: () = assert!(Q_ROW_STEP * Q_PAIRS == THREADS && Q_POS_PASSES * Q_ROW_STEP == GROUP);
const _: () = assert!(Q_WORDS == Q_POS_PASSES * POSITIONS);
const _: () = assert!(KEY_TILE == 4 * 16 && HEAD.is_multiple_of(16));
// `gqa_prefill_flash`'s launch contract spells GROUP and HEAD out as 8 and
// 128.
const _: () = assert!(GROUP == 8 && HEAD == 128);

/// Blocks a ubatch of `t` rows over `n_kv` key heads launches.
#[must_use]
fn blocks_for(t: usize, n_kv: usize) -> usize {
    t.div_ceil(POSITIONS) * n_kv
}

/// `exp(x)` for a weight that is rounded to f16 next: `ex2.approx.ftz` on
/// `x · log2 e`. It runs the same `MUFU.EX2` on the same argument as
/// [`dev_exp`](crate::flash::dev_exp) wherever that argument is at least
/// −126; below it `dev_exp` returns a value under 2^-126 and this returns 0,
/// and both round to the f16 +0 (anything under 2^-25 does). So the rounded weight — the only thing a
/// weight reaches — is `dev_exp`'s bit for bit. Never for a rescale factor,
/// which multiplies the accumulators unrounded.
#[inline(always)]
pub(crate) fn exp_weight(x: f32) -> f32 {
    cuda_device::float::ex2_approx_ftz_f32(x * std::f32::consts::LOG2_E)
}

/// `max.f32`: one instruction, where `f32::max` lowers to a compare, a NaN
/// test and a select. Both return the other operand when one is NaN; they
/// may differ only in the sign of a zero result, and the walk's maxima reach
/// the bits only through `s − m` (unchanged by the sign of a zero `m` unless
/// `s` is a zero too, where `ex2(±0) = 1`) and through `m' > m`, which reads
/// the two zeros as equal. The walk's running maxima start at −∞ and are
/// never NaN, so the two-NaN case does not occur.
#[inline(always)]
pub(crate) fn fmax(a: f32, b: f32) -> f32 {
    let r: f32;
    // SAFETY: a register-only arithmetic instruction; no memory is touched.
    unsafe {
        ptx_asm!(
            "max.f32 %0, %1, %2;",
            out("=f") r,
            in("f") a,
            in("f") b,
            options(register_only),
        );
    }
    r
}

// ------------------------------------------------ the 256-value head body
//
// `gqa_prefill_flash_256`: the same walk over heads of `HEAD_256` values with
// eight warps a block. Warp `w` owns m16 row slice `w % 4` — the same two
// positions × eight heads as the 128 body's warp `w % 4` — and dim half
// `w / 4`: it computes the half-dim scores `S_w` of its sixteen rows over its
// own 128 dims, publishes them in shared memory, and adds its partner's
// (`w ^ 4`, the other half): a rounded add is commutative, so `S_w + S_mate`
// is `S_lo + S_hi`, the sum `flash_gqa::gqa_flash_seg_mma_256` forms from its
// two slice accumulators, and a key's score is that pass's bit for bit. Both
// warps of a pair then run the same softmax on the same bits and each
// accumulates `P̂·V` over its own 128 dims.

/// Values per head, and those of one warp's half.
pub const HEAD_256: usize = 256;
const HALF_256: usize = HEAD_256 / 2;
/// Warps and threads of a block: four row slices × two dim halves.
const SLICES_256: usize = POSITIONS / 2;
const WARPS_256: usize = 2 * SLICES_256;
const THREADS_256: usize = WARPS_256 * 32;
const THREADS_256_U32: u32 = THREADS_256 as u32;
const _: () = assert!(THREADS_256_U32 as usize == THREADS_256 && THREADS_256 == 256);
/// Query rows one block covers, as in the 128 body.
const Q_ROWS_256: usize = POSITIONS * GROUP;
/// u32 words of one cache row (and of one staged row), and of a tile.
const ROW_WORDS_256: usize = HEAD_256 / 2;
const TILE_WORDS_256: usize = KEY_TILE * ROW_WORDS_256;
/// 16-byte chunks of one row, per thread per staged tile, and the keys one
/// pass of the block's copies covers.
const ROW_CHUNKS_256: usize = HEAD_256 / 8;
const CHUNKS_256: usize = KEY_TILE * ROW_CHUNKS_256 / THREADS_256;
const PASS_KEYS_256: usize = THREADS_256 / ROW_CHUNKS_256;
/// f16 pairs of one query row, per thread in the query staging, the rows one
/// pass covers and the passes per position's group of heads.
const Q_PAIRS_256: usize = HEAD_256 / 2;
const Q_WORDS_256: usize = Q_ROWS_256 * Q_PAIRS_256 / THREADS_256;
const Q_ROW_STEP_256: usize = THREADS_256 / Q_PAIRS_256;
const Q_POS_PASSES_256: usize = GROUP / Q_ROW_STEP_256;
/// Passes of the query staging that cover eight staged rows (the
/// PACK-generic body's `q_at`).
const Q_OCT_PASSES_256: usize = 8 / Q_ROW_STEP_256;
/// Rows of one m16 row slice: `SLICE_ROWS / PACK` positions of `PACK` heads.
const SLICE_ROWS: usize = 16;
/// A warp's k16 steps over its half of the head, and its n16 pairs of dims.
const QK_STEPS_256: usize = HALF_256 / 16;
const DIM_PAIRS_256: usize = HALF_256 / 16;
/// u32 words of a 128-byte line of a staged row (the XOR's reach) and the
/// `ldmatrix` chunk pairs one line holds.
const LINE: usize = 32;
const LINE_PAIRS: usize = LINE / 8;
/// f32 of one warp's published half-dim scores: 32 lanes × the fragment's
/// `KEY_NT · 4` values.
const EXCH_WARP: usize = 32 * KEY_NT * 4;
/// Bytes of dynamic shared memory a block takes: the key tile, the value
/// tile and the score exchange, in that order, each 128-byte aligned.
const DYN_BYTES_256: usize = 2 * TILE_WORDS_256 * 4 + WARPS_256 * EXCH_WARP * 4;
const DYN_BYTES_256_U32: u32 = DYN_BYTES_256 as u32;
const _: () = assert!(DYN_BYTES_256_U32 as usize == DYN_BYTES_256);
// `#[launch_contract(dynamic_shared = ...)]` takes a literal: the kernel's
// declaration spells the byte count out, and this is the two agreeing.
const _: () = assert!(DYN_BYTES_256 == 98_304);
// The launch contract spells GROUP and HEAD_256 out as 8 and 256.
const _: () = assert!(GROUP == 8 && HEAD_256 == 256);

/// The staged word in a row of `HEAD_256` values.
#[inline(always)]
const fn staged_word_256(row: usize, w: usize) -> usize {
    staged!(row, w, ROW_WORDS_256)
}

const _: () = assert!(Q_ROWS_256 == SLICES_256 * 16 && Q_ROWS_256 <= KEY_TILE);
const _: () = assert!(PASS_KEYS_256 * ROW_CHUNKS_256 == THREADS_256);
const _: () = assert!(PASS_KEYS_256 * CHUNKS_256 == KEY_TILE && PASS_KEYS_256.is_multiple_of(8));
const _: () = assert!((ROW_WORDS_256 * 4).is_multiple_of(128) && ROW_CHUNKS_256.is_multiple_of(8));
const _: () = assert!(layout_holds(KEY_TILE, ROW_WORDS_256));
const _: () = assert!(Q_WORDS_256 * THREADS_256 == Q_ROWS_256 * Q_PAIRS_256);
const _: () = assert!(Q_ROW_STEP_256 * Q_PAIRS_256 == THREADS_256);
const _: () = assert!(Q_POS_PASSES_256 * Q_ROW_STEP_256 == GROUP);
const _: () = assert!(Q_WORDS_256 == Q_POS_PASSES_256 * POSITIONS);
// A tile holds the block's staged query rows before the first key tile.
const _: () = assert!(Q_ROWS_256 * Q_PAIRS_256 == TILE_WORDS_256);
// A warp's half is whole lines, and its k16 steps and dim pairs are whole
// line pairs.
const _: () =
    assert!((HALF_256 / 2).is_multiple_of(LINE) && QK_STEPS_256.is_multiple_of(LINE_PAIRS));

/// The staging's word of pass `e`, `staged_word_256(Q_ROW_STEP_256·e + r0,
/// wd)`, is the word of pass `e % Q_OCT_PASSES_256` plus `e /
/// Q_OCT_PASSES_256` groups of eight rows, for every pass, thread row and word.
const fn q_oct_hold() -> bool {
    let mut e = 0;
    while e < Q_WORDS_256 {
        let mut r0 = 0;
        while r0 < Q_ROW_STEP_256 {
            let mut wd = 0;
            while wd < Q_PAIRS_256 {
                let at = staged_word_256(Q_ROW_STEP_256 * e + r0, wd);
                let base = staged_word_256(Q_ROW_STEP_256 * (e % Q_OCT_PASSES_256) + r0, wd);
                if at != base + (e / Q_OCT_PASSES_256) * 8 * ROW_WORDS_256 {
                    return false;
                }
                wd += 1;
            }
            r0 += 1;
        }
        e += 1;
    }
    true
}

/// The slice position (`0 .. SLICE_ROWS / PACK`) of fragment row `g + 8·h`,
/// `g = lane / 4 < 8`: rows are `position·PACK + head`. A pack of eight is one
/// position a row half, so the lane's index does not enter.
#[inline(always)]
pub(crate) const fn row_pos<const PACK: usize>(g: usize, h: usize) -> usize {
    if PACK >= 8 { h } else { (g + 8 * h) / PACK }
}

/// The head within the pack of fragment row `g + 8·h` (the same for both
/// halves, as `PACK` divides eight).
#[inline(always)]
pub(crate) const fn row_head<const PACK: usize>(g: usize) -> usize {
    if PACK >= 8 { g } else { g % PACK }
}

/// The fragment row maps of a prefill block at `PACK`: every fragment row `16·rs +
/// g + 8·h` of every warp is the staged row of slice position `row_pos` and
/// head `row_head` of its slice, each staged row `lp·PACK + j` of the block
/// once over `slices` slices.
pub(crate) const fn pack_rows_hold<const PACK: usize>(slices: usize) -> bool {
    if slices == 0 || slices > 4 || PACK == 0 || !8usize.is_multiple_of(PACK) {
        return false;
    }
    let pps = SLICE_ROWS / PACK;
    let mut seen = 0u64;
    let mut rs = 0;
    while rs < slices {
        let mut g = 0;
        while g < 8 {
            let mut h = 0;
            while h < 2 {
                let (p, j) = (row_pos::<PACK>(g, h), row_head::<PACK>(g));
                let row = (pps * rs + p) * PACK + j;
                if p >= pps || j >= PACK || row != 16 * rs + g + 8 * h || seen & (1 << row) != 0 {
                    return false;
                }
                seen |= 1 << row;
                h += 1;
            }
            g += 1;
        }
        rs += 1;
    }
    seen == u64::MAX >> (64 - SLICE_ROWS * slices)
}

/// Every query staging pass of the packed 256 body covers its row once.
const fn pack_staging_holds<const PACK: usize>() -> bool {
    if !pack_rows_hold::<PACK>(SLICES_256) {
        return false;
    }
    let mut staged = 0u64;
    let mut e = 0;
    while e < Q_WORDS_256 {
        let mut r0 = 0;
        while r0 < Q_ROW_STEP_256 {
            let row = Q_ROW_STEP_256 * e + r0;
            let (lp, j) = (row / PACK, row % PACK);
            if lp >= Q_ROWS_256 / PACK || j >= PACK || row != lp * PACK + j {
                return false;
            }
            staged |= 1 << row;
            r0 += 1;
        }
        e += 1;
    }
    Q_ROWS_256 == 64 && staged == u64::MAX
}

const _: () = assert!(Q_OCT_PASSES_256 * Q_ROW_STEP_256 == 8);
const _: () = assert!(q_oct_hold());
// The packed layouts, and at `GROUP` the eight-head entry's staged rows.
const _: () = assert!(
    pack_staging_holds::<GROUP>()
        && pack_staging_holds::<PACK_4>()
        && pack_staging_holds::<PACK_2>()
        && pack_staging_holds::<1>()
);
// The `_p4` instance of `prefill_256_p` (its generic const blocks run only
// when the device build instantiates it, this one in every check).
const _: () = assert!(
    SLICE_ROWS.is_multiple_of(PACK_4) && SLICES_256 * (SLICE_ROWS / PACK_4) == Q_ROWS_256 / PACK_4
);
// The `_p2` instance's, likewise.
const _: () = assert!(
    SLICE_ROWS.is_multiple_of(PACK_2) && SLICES_256 * (SLICE_ROWS / PACK_2) == Q_ROWS_256 / PACK_2
);

// ----------------------------------------------------- the q8_0 tile staging
//
// A q8 cache's tile staging in both bodies (the module doc's `_q8`
// paragraph): the raw code words of a tile's rows land by `cp.async` in the
// low half of the rows' own spans of the f16 tile and the rows' scales in a
// small scales tile, and the in-place widening below writes the dequantized
// f16 pairs to the permuted layout the `ldmatrix` addresses walk.

/// Word `w`'s offset within its row of a staged tile — the in-row half of
/// [`staged_word`]/[`staged_word_256`]: the 16-byte chunk `w/4` stored at
/// chunk `(w/4) ^ xor`, `xor` the row's `row & 7`.
#[inline(always)]
pub(crate) const fn staged_off(xor: usize, w: usize) -> usize {
    staged!(@offset xor, w)
}

/// One staged row's in-place widening, as a macro: a pointer into the
/// dynamic shared window, which the 256 body's tiles are, cannot be passed
/// to a device helper fn (the toolchain types it in the shared address
/// space and the helper's parameter in the generic one), so the body expands
/// at each call site — the file's own `stage!` shape.
/// Lane `$lane` reads the row's raw code words `$lane + 32·i` (`i < NW` — the
/// row's `NW·32` words, its low half, landed by the copies) and its scale
/// `lane/8 + 4·i`, the warp syncs — a row's staged writes may land on its
/// own raw words — and each word's four widened values ([`q8_word_values`])
/// go as two staged words to the permuted layout: `$tile` the staged tile,
/// `$base` the row's first word, `$scales` the row's `4·NW` u16 scales,
/// `$xor` the row's `row & 7`, `$lane` the calling lane. All 32 lanes of a
/// warp, together, on one row that is that warp's alone.
///
/// SAFETY: the row's span of `$tile` is this warp's alone, `NW·32` is at
/// most half the tile's row stride, `$scales .. +4·NW` lies inside the
/// scales tile, `$xor < 8`, and the raw words were published by the barrier
/// before this expansion.
macro_rules! dequant_row {
    ($nw:literal, $tile:expr, $base:expr, $scales:expr, $xor:expr, $lane:expr) => {{
        let mut cw = [0u32; $nw];
        for i in 0..$nw {
            cuda_device::thread::__unroll_config::<0>();
            // SAFETY: lane + 32·i < NW·32 <= half the row's words, this
            // warp's own row of the tile.
            unsafe { cw[i] = *$tile.add($base + $lane + 32 * i) };
        }
        warp::sync_mask(u32::MAX);
        for i in 0..$nw {
            cuda_device::thread::__unroll_config::<0>();
            // SAFETY: lane/8 + 4·i < 4·NW: inside the row's scales.
            let d = half_bits_to_f32(unsafe { *$scales.add($lane / 8 + 4 * i) });
            let v = q8_word_values(cw[i], d);
            let c = $lane + 32 * i;
            // SAFETY: staged words 2·c and 2·c + 1 of the row, inside its
            // span.
            unsafe {
                *$tile.add($base + staged_off($xor, 2 * c)) = f32x2_to_f16x2_bits(v[0], v[1]);
                *$tile.add($base + staged_off($xor, 2 * c + 1)) = f32x2_to_f16x2_bits(v[2], v[3]);
            }
        }
    }};
}

/// A staged row's zero words, the dead key's form, as a macro (the same
/// boundary as `dequant_row!`'s): lane `$lane` writes the row's staged
/// words `$lane + 32·j` (`j < 2·NW`) as zero — the f16 walk's own zeros for
/// a key at or past the block's largest count, so a padded row's NaN scale
/// reaches no result.
///
/// SAFETY: the row's span of `$tile` is this warp's alone, `2·NW·32` the
/// row's staged word count, `$xor < 8`.
macro_rules! zero_row {
    ($nw:literal, $tile:expr, $base:expr, $xor:expr, $lane:expr) => {{
        for j in 0..2 * $nw {
            cuda_device::thread::__unroll_config::<0>();
            // SAFETY: lane + 32·j < 64·NW: inside the row's span, this
            // warp's own row.
            unsafe { *$tile.add($base + staged_off($xor, $lane + 32 * j)) = 0 };
        }
    }};
}

/// The online softmax of the module doc's step 2 over one key tile's scores,
/// written once for every prefill body and expanded in each, as
/// `dequant_row!` is: the body's locals are the arguments. A warp whose tile
/// is inside every row's bounds (`unmasked`) takes the expansion without
/// the per-key mask, every other one the masked expansion; on such a
/// tile both evaluate the same expressions. `key0` is the lane's first key
/// (`KEY_TILE·kb + 2·t4`), `tiles` the tile's n8 tiles and k16 steps,
/// `scores` the lane's `[[f32; 4]; n8 tiles]` raw scores (scaled in place),
/// `counts` its two rows' counts, optional `lower` their first keys when
/// enabled, `state` the rows' running max, lane sum and
/// this tile's rescale factor (`f` starts at 1), `weights` the value
/// product's A fragments.
///
/// Register `j` of n8 tile `nt` is row half `j / 2`, key
/// `key0 + 8·nt + j % 2`. The weights go out rounded to f16 pairs as the A
/// fragment of the value product: k16 step `kk2` is n8 tiles `2·kk2` and
/// `2·kk2 + 1`, registers `(0, 1)` then `(2, 3)` of each, low key first, and
/// the lane's sum takes the rounded values in that order. The mask is the
/// key's position, not its score: a NaN score stays NaN.
macro_rules! prefill_softmax {
    (
        unmasked: $unmasked:expr,
        key0: $key0:expr,
        tiles: ($key_nt:expr, $pv_steps:expr),
        scores: $sc:ident,
        counts: $cnt:ident,
        $(lower: ($lower:expr, $first:ident),)?
        scale: $scale:expr,
        state: ($m:ident, $l:ident, $f:ident),
        weights: $pa:ident $(,)?
    ) => {
        if $unmasked {
            prefill_softmax!(
                @tile false, $key0, $key_nt, $pv_steps, $sc, $cnt, $scale, $m, $l, $f, $pa $(, ($lower, $first))?
            );
        } else {
            prefill_softmax!(
                @tile true, $key0, $key_nt, $pv_steps, $sc, $cnt, $scale, $m, $l, $f, $pa $(, ($lower, $first))?
            );
        }
    };
    (
        @tile $masked:literal, $key0:expr, $key_nt:expr, $pv_steps:expr, $sc:ident,
        $cnt:ident, $scale:expr, $m:ident, $l:ident, $f:ident, $pa:ident
        $(, ($lower:expr, $first:ident))?
    ) => {{
        let kb0 = $key0;
        let mut tmax = [f32::NEG_INFINITY; 2];
        for nt in 0..$key_nt {
            cuda_device::thread::__unroll_config::<0>();
            for j in 0usize..4 {
                cuda_device::thread::__unroll_config::<0>();
                let key = kb0 + (8 * nt + j % 2) as u32;
                let v = if !$masked || (key < $cnt[j / 2] $(&& (!$lower || key >= $first[j / 2]))?) {
                    cuda_device::float::mul_rn_f32($sc[nt][j], $scale)
                } else {
                    f32::NEG_INFINITY
                };
                $sc[nt][j] = v;
                tmax[j / 2] = $crate::flash_gqa_prefill::fmax(tmax[j / 2], v);
            }
        }
        let mut mn = [f32::NEG_INFINITY; 2];
        for h in 0usize..2 {
            cuda_device::thread::__unroll_config::<0>();
            let mut x = tmax[h];
            x = $crate::flash_gqa_prefill::fmax(x, cuda_device::warp::shuffle_xor_f32(x, 1));
            x = $crate::flash_gqa_prefill::fmax(x, cuda_device::warp::shuffle_xor_f32(x, 2));
            let m_new = $crate::flash_gqa_prefill::fmax($m[h], x);
            if m_new > $m[h] {
                $f[h] = if $m[h] > f32::NEG_INFINITY {
                    $crate::flash::dev_exp($m[h] - m_new)
                } else {
                    0.0
                };
                $m[h] = m_new;
            }
            mn[h] = $m[h];
            $l[h] = cuda_device::float::mul_rn_f32($l[h], $f[h]);
        }
        for kk2 in 0..$pv_steps {
            cuda_device::thread::__unroll_config::<0>();
            for e in 0usize..4 {
                cuda_device::thread::__unroll_config::<0>();
                let nt = 2 * kk2 + e / 2;
                let j0 = 2 * (e % 2);
                let h = e % 2;
                let key = kb0 + (8 * nt) as u32;
                let p0 = if !$masked || (key < $cnt[h] $(&& (!$lower || key >= $first[h]))?) {
                    $crate::flash_gqa_prefill::exp_weight($sc[nt][j0] - mn[h])
                } else {
                    0.0
                };
                let p1 = if !$masked || (key + 1 < $cnt[h] $(&& (!$lower || key + 1 >= $first[h]))?) {
                    $crate::flash_gqa_prefill::exp_weight($sc[nt][j0 + 1] - mn[h])
                } else {
                    0.0
                };
                let packed = cuda_device::convert::cvt_f16x2_f32(p0, p1);
                let (r0, r1) = cuda_device::convert::cvt_f32x2_f16x2(packed);
                $l[h] = cuda_device::float::add_rn_f32(
                    cuda_device::float::add_rn_f32($l[h], r0),
                    r1,
                );
                $pa[kk2][e] = packed;
            }
        }
    }};
}
pub(crate) use prefill_softmax;

/// The row counts a PACK-generic body's walk uses, written once and expanded
/// in each, as `dequant_row!` is: it binds the `out` names in the caller. Slice `s` is
/// positions `t0 + s·slice_pos ..` of the block; both loop bounds are
/// constants, so every copy's `s` and `i` are literals and no count is kept
/// in an array indexed at run time. A row below `rows` reads its count from
/// `n_keys`, refused (count 0, refused) when it is zero or past `ctx`; a row
/// past `rows` is count 0, not refused. Bound: `hi`, the block's largest
/// count; `warp_hi`, `warp_lo` and `any_bad`, the largest, smallest and any
/// refusal of the warp's own slice `rs` (the walk's warp-uniform bounds);
/// `cnt` and `bad`, the lane's two rows' own (`row_pos` of `g`). With `lower`,
/// the first keys follow the same walk: `lo` is the block's smallest live
/// first key, `warp_first_lo` and `warp_first_hi` the slice's bounds, and
/// `first` the lane's two rows'. A first key at or past its count is refused.
///
/// SAFETY: `n_keys` holds at least `rows` counts, and with `lower`, `k_first`
/// holds at least `rows` first keys (the caller's launch contract).
macro_rules! prefill_pack_counts {
    (
        out: ($hi:ident, $warp_hi:ident, $warp_lo:ident, $any_bad:ident, $cnt:ident, $bad:ident),
        shape: ($slices:expr, $slice_pos:ident, $pack:ident),
        rows: ($t0:ident, $rows:ident, $rs:ident, $g:ident),
        counts: ($n_keys:ident, $ctx:ident),
        lower: ($lower:expr, $k_first:ident),
        first: ($lo:ident, $warp_first_lo:ident, $warp_first_hi:ident, $first:ident) $(,)?
    ) => {
        let mut $hi = 0u32;
        let mut $warp_hi = 0u32;
        let mut $warp_lo = u32::MAX;
        let mut $lo = u32::MAX;
        let mut $warp_first_lo = u32::MAX;
        let mut $warp_first_hi = 0u32;
        let mut $first = [0u32; 2];
        let mut $any_bad = false;
        let mut $cnt = [0u32; 2];
        let mut $bad = [false; 2];
        for s in 0..$slices {
            cuda_device::thread::__unroll_config::<0>();
            for i in 0..$slice_pos {
                cuda_device::thread::__unroll_config::<0>();
                let t = $t0 + s * $slice_pos + i;
                let (c, k0, refused) = if t < $rows {
                    // SAFETY: t < t_rows <= n_keys_buf.len() (launch contract).
                    let c = unsafe { *$n_keys.get_unchecked(t) };
                    let k0 = if $lower {
                        // SAFETY: t < rows <= k_first.len() (the LOWER launch contract).
                        unsafe { *$k_first.get_unchecked(t) }
                    } else {
                        0
                    };
                    if c == 0 || c > $ctx || ($lower && k0 >= c) {
                        (0, 0, true)
                    } else {
                        (c, k0, false)
                    }
                } else {
                    (0, 0, false)
                };
                $hi = $hi.max(c);
                if $lower && c > 0 {
                    $lo = $lo.min(k0);
                }
                if s == $rs {
                    $warp_hi = $warp_hi.max(c);
                    $warp_lo = $warp_lo.min(c);
                    if $lower && c > 0 {
                        $warp_first_lo = $warp_first_lo.min(k0);
                        $warp_first_hi = $warp_first_hi.max(k0);
                    }
                    $any_bad = $any_bad || refused;
                    if i == $crate::flash_gqa_prefill::row_pos::<$pack>($g, 0) {
                        $cnt[0] = c;
                        $first[0] = k0;
                        $bad[0] = refused;
                    }
                    if i == $crate::flash_gqa_prefill::row_pos::<$pack>($g, 1) {
                        $cnt[1] = c;
                        $first[1] = k0;
                        $bad[1] = refused;
                    }
                }
            }
        }
    };
}
pub(crate) use prefill_pack_counts;

/// The end of a PACK-generic body's walk, written once and expanded in each,
/// as `dequant_row!` is: each row's sum over its four lanes `(l + l^1) +
/// l^2`, then `o · (1/l)` into `y`, NaN for a refused row (`bad`). The lane's
/// value `j` of n8 tile `nd` (`nd < dims`, the accumulator's tiles) is row
/// half `j / 2`, position `row0 + row_pos(g, j / 2)` and head `head0 +
/// row_head(g)`, dim `col0 + 8·nd + 2·t4 + j % 2` of a `head`-value row; a
/// position at or past `rows` writes nothing.
///
/// SAFETY: every dim the expansion writes is below `head` and owned by this
/// lane alone (the caller's `col0`), `head0 + PACK <= n_head`, and `y` holds
/// `rows · n_head · head` values (the caller's launch contract).
macro_rules! prefill_pack_store {
    (
        acc: ($o:ident, $l:ident, $bad:ident, $dims:expr),
        rows: ($row0:expr, $rows:ident, $pack:ident, $g:ident, $t4:ident),
        cols: ($col0:expr, $head:expr),
        out: ($y:ident, $n_head:ident, $head0:ident) $(,)?
    ) => {
        let mut inv = [0.0f32; 2];
        for h in 0usize..2 {
            cuda_device::thread::__unroll_config::<0>();
            let a =
                cuda_device::float::add_rn_f32($l[h], cuda_device::warp::shuffle_xor_f32($l[h], 1));
            let s = cuda_device::float::add_rn_f32(a, cuda_device::warp::shuffle_xor_f32(a, 2));
            inv[h] = 1.0 / s;
        }
        for nd in 0..$dims {
            cuda_device::thread::__unroll_config::<0>();
            for j in 0usize..4 {
                cuda_device::thread::__unroll_config::<0>();
                let h = j / 2;
                let t = $row0 + $crate::flash_gqa_prefill::row_pos::<$pack>($g, h);
                if t < $rows {
                    let d = $col0 + 8 * nd + 2 * $t4 + j % 2;
                    let v = if $bad[h] {
                        f32::NAN
                    } else {
                        cuda_device::float::mul_rn_f32($o[nd][j], inv[h])
                    };
                    // SAFETY: t < t_rows, head0 + row_head(g) < head0 + PACK
                    // <= n_head and d < head: below t_rows·n_head·head <=
                    // y.len() (launch contract); one owner lane per value.
                    unsafe {
                        *$y.get_unchecked_mut(
                            (t * $n_head
                                + $head0
                                + $crate::flash_gqa_prefill::row_head::<$pack>($g))
                                * $head
                                + d,
                        ) = v;
                    }
                }
            }
        }
    };
}
pub(crate) use prefill_pack_store;

/// One 256-value-head scales tile of the q8 staging: `KEY_TILE` rows of
/// `HEAD_256/32` u16.
const Q8_SCALE_TILE_256: usize = KEY_TILE * (HEAD_256 / 32);
/// The q8 bytes of dynamic shared memory a 256-value-head block adds to the
/// f16 window: its two scales tiles.
const Q8_SCALES_256: usize = 2 * Q8_SCALE_TILE_256;
/// A q8 256 block's dynamic shared window: the f16 [`DYN_BYTES_256`] plus the
/// two scales tiles — 100,352 bytes, inside the card's opt-in window
/// (101,376).
const DYN_BYTES_256_Q8: usize = DYN_BYTES_256 + Q8_SCALES_256 * 2;
const DYN_BYTES_256_Q8_U32: u32 = DYN_BYTES_256_Q8 as u32;
const _: () = assert!(DYN_BYTES_256_Q8_U32 as usize == DYN_BYTES_256_Q8);
const _: () = assert!(DYN_BYTES_256_Q8 == 100_352);

/// The attention of the module doc over heads of [`HEAD_256`] values, eight
/// warps a block (the section above), in blocks of `PACK` query heads, `packs`
/// of them per key head, over a cache in the format `S`: the body of
/// `gqa_prefill_flash_256_p4` (`PACK = 4`), `gqa_prefill_flash_256_p2`
/// (`PACK = 2`), `gqa_prefill_flash_256_p1_lower` (`PACK = 1`) and,
/// over [`Q8Kv`], `gqa_prefill_flash_256_q8` (`PACK = 8`).
/// `gqa_prefill_flash_256` keeps its own body; this PACK-generic one computes
/// each row as that body does ([`pack_staging_holds`] at `GROUP` is its layout).
/// Block `b` is unit `r = b % (n_kv · packs)` — key head `r / packs`, query
/// heads `head0 = r·PACK ..` — and query tile `n_tiles − 1 − b / (n_kv ·
/// packs)`, rows `(Q_ROWS_256 / PACK)·tile ..` of `t_rows`; a row slice is
/// `SLICE_ROWS / PACK` positions × `PACK` heads ([`pack_staging_holds`]). Row `t`'s query head `head0 + j` is read at `q[(t ·
/// n_head + head0 + j)·256 ..]` and its output written at the same index of
/// `y`. The key tile, the value tile and the score exchange live in the
/// block's dynamic shared memory, and a q8 cache's two scales tiles after
/// them. With `LOWER`, row `t` also reads its first key `k_first_buf[t]` and
/// masks the keys below it (the module doc's lower-bound paragraph); without,
/// `k_first_buf` is never read. The formats differ only in a tile's staging:
/// the f16 rows land by `cp.async` in the permuted layout; a q8 tile's raw code words land in the
/// low halves of its rows and its scales in its scales tile, and the
/// in-place widening (`dequant_row`, `zero_row`) and its barrier run before
/// the tile is read. Every guard
/// around a shuffle or a barrier
/// is block- or warp-pair-uniform: `live` and the unmasked path read the
/// slice's largest and smallest counts and first keys; a lane's two rows
/// mask by their own.
///
/// SAFETY: the entry's launch contract at `n_kv · packs · PACK` query heads
/// over `src`'s planes (every buffer bound it names), a block of
/// [`THREADS_256`] threads and [`DYN_BYTES_256`] bytes of dynamic shared
/// memory for an f16 cache, [`DYN_BYTES_256_Q8`] for a q8 one, whose window
/// starts at the block's shared base (no static shared memory in the entry).
#[inline(always)]
#[allow(
    clippy::too_many_arguments,
    reason = "a kernel entry's flat arguments, handed on (rust-quality R8)"
)]
#[allow(
    clippy::needless_range_loop,
    reason = "a range `for` is the loop the unroller recognizes; an iterator loop keeps its array in a local depot"
)]
unsafe fn prefill_256_p<const PACK: usize, const SLICE_POS: usize, const LOWER: bool, S: KvSrc>(
    q: &[f32],
    src: S,
    n_keys_buf: &[u32],
    k_first_buf: &[u32],
    scale: f32,
    n_kv: u32,
    ctx: u32,
    t_rows: u32,
    packs: usize,
    fault: FaultSink,
    mut y: DisjointSlice<f32>,
) {
    const { assert!(pack_staging_holds::<PACK>() && SLICE_POS * PACK == SLICE_ROWS) };
    const { assert!(SLICES_256 * SLICE_POS == Q_ROWS_256 / PACK) };
    // Positions a block covers, and a row slice's.
    let positions = Q_ROWS_256 / PACK;
    let pps = SLICE_POS;
    // PACK = 1 takes its own position indexing, without dividing by qpp.
    let qpp = PACK / Q_ROW_STEP_256;
    // The q8 staging: code 16-byte chunks a staged row takes, a thread's
    // count of a tile's, and the rows of a tile each warp widens.
    const ROW_CCHUNKS: usize = HEAD_256 / 16;
    const TILE_CCHUNKS: usize = KEY_TILE * ROW_CCHUNKS / THREADS_256;
    const _: () = assert!(TILE_CCHUNKS * THREADS_256 == KEY_TILE * ROW_CCHUNKS);
    const WID_ROWS: usize = KEY_TILE / WARPS_256;
    const _: () = assert!(WID_ROWS * WARPS_256 == KEY_TILE);

    let b = thread::blockIdx_x() as usize;
    let nkv = n_kv as usize;
    let rows = t_rows as usize;
    let n_tiles = rows.div_ceil(positions);
    let units = nkv * packs;
    if b >= n_tiles * units {
        return; // block-uniform
    }
    let qt = n_tiles - 1 - b / units;
    let r = b % units;
    let kh = r / packs;
    let head0 = r * PACK;
    let tid = thread::threadIdx_x() as usize;
    let lane = warp::lane_id() as usize;
    let wid = tid / 32;
    // The warp's row slice (its positions) and dim half.
    let rs = wid % SLICES_256;
    let dh = wid / SLICES_256;
    let g = lane / 4;
    let t4 = lane % 4;
    let n_head = units * PACK;
    let t0 = qt * positions;

    // The block's dynamic shared memory, DYN_BYTES_256 bytes (the launch
    // contract): the key tile, the value tile and the exchange at byte
    // offsets 0, TILE_WORDS_256·4 and 2·TILE_WORDS_256·4, every write
    // ordered before its reads by a block barrier (the cp.async copies by
    // the wait before it). The base is declared 16-byte aligned — what
    // the 16-byte copies and `ldmatrix` rows need; the bundle's other
    // kernels share that declaration, and a wider one pads their static
    // shared memory. With no static shared memory here the base is the
    // window's start, so the tiles' 128-byte lines are the layout's.
    let (kt, vt, ex) = (
        DynamicSharedArray::<u32, 16>::get(),
        DynamicSharedArray::<u32, 16>::offset(TILE_WORDS_256 * 4),
        DynamicSharedArray::<f32, 16>::offset(2 * TILE_WORDS_256 * 4),
    );
    // A q8 cache's two scales tiles, past the exchange: 16-byte aligned (their
    // offset DYN_BYTES_256 is), so the 16-byte scale copies land aligned.
    let sk =
        DynamicSharedArray::<u16, 16>::offset(2 * TILE_WORDS_256 * 4 + WARPS_256 * EXCH_WARP * 4);
    let sv = DynamicSharedArray::<u16, 16>::offset(
        2 * TILE_WORDS_256 * 4 + WARPS_256 * EXCH_WARP * 4 + Q8_SCALE_TILE_256 * 2,
    );

    // Every row's count and, with LOWER, first key as the walk uses them.
    prefill_pack_counts! {
        out: (hi, warp_hi, warp_lo, any_bad, cnt, bad),
        shape: (SLICES_256, SLICE_POS, PACK),
        rows: (t0, rows, rs, g),
        counts: (n_keys_buf, ctx),
        lower: (LOWER, k_first_buf),
        first: (lo, warp_first_lo, warp_first_hi, first),
    }
    if lane == 0 && dh == 0 && any_bad {
        fault.raise(FaultSite::KeyCount);
    }

    // The block's query rows, rounded to f16 pairs, into the key tile:
    // staged row `lp·PACK + j` is position `t0 + lp`'s head `head0 + j`, so
    // row slice `r` is rows `16·r ..`. A row past `t_rows` is zero. Thread
    // `tid` stages word `tid % Q_PAIRS_256` of every row its passes cover —
    // pass `e` is staged row `Q_ROW_STEP_256·e + r0` (two positions when
    // PACK = 1), every load issued before the first conversion.
    let wd = tid % Q_PAIRS_256;
    let r0 = tid / Q_PAIRS_256;
    let q_first = (t0 * n_head + head0 + r0) * Q_PAIRS_256 + wd;
    let q_pos = n_head * Q_PAIRS_256;
    let q64 = q.as_ptr().cast::<u64>();
    let mut raw = [0u64; Q_WORDS_256];
    if PACK == 1 {
        for e in 0..Q_WORDS_256 {
            cuda_device::thread::__unroll_config::<0>();
            let t = t0 + Q_ROW_STEP_256 * e + r0;
            if t < rows {
                let at = (t * n_head + head0) * Q_PAIRS_256 + wd;
                // SAFETY: t < t_rows and head0 < n_head, inside q. A row
                // is 128 u64, and q starts 8-byte aligned (host-checked).
                raw[e] = unsafe { *q64.add(at) };
            }
        }
    } else {
        for e in 0..Q_WORDS_256 {
            cuda_device::thread::__unroll_config::<0>();
            let t = t0 + e / qpp;
            if t < rows {
                let at = q_first + (e / qpp) * q_pos + Q_ROW_STEP_256 * (e % qpp) * Q_PAIRS_256;
                // SAFETY: t < t_rows and the head head0 + Q_ROW_STEP_256·(e % qpp)
                // + r0 < head0 + PACK <= n_head, so the row lies inside q (launch
                // contract); a row is 256 f32 = 128 u64 and q starts 8-byte
                // aligned (host-checked), so the u64 read is aligned.
                raw[e] = unsafe { *q64.add(at) };
            }
        }
    }
    // The staged word of each pass of the first eight rows; a later pass's is
    // it plus whole groups of eight rows (layout_holds).
    let mut q_at = [0usize; Q_OCT_PASSES_256];
    for j in 0..Q_OCT_PASSES_256 {
        cuda_device::thread::__unroll_config::<0>();
        q_at[j] = staged_word_256(Q_ROW_STEP_256 * j + r0, wd);
    }
    for e in 0..Q_WORDS_256 {
        cuda_device::thread::__unroll_config::<0>();
        let pair = f32x2_to_f16x2_bits(
            f32::from_bits(raw[e] as u32),
            f32::from_bits((raw[e] >> 32) as u32),
        );
        // SAFETY: row Q_ROW_STEP_256·e + r0 < Q_ROWS_256 <= KEY_TILE and
        // wd < ROW_WORDS_256: inside KT; one owner per word.
        unsafe {
            *kt.add(q_at[e % Q_OCT_PASSES_256] + (e / Q_OCT_PASSES_256) * 8 * ROW_WORDS_256) = pair
        };
    }
    thread::sync_threads();

    // This warp's sixteen query rows over its half, one A fragment per
    // k16 step.
    let mut qa = [[0u32; 4]; QK_STEPS_256];
    for kk in 0..QK_STEPS_256 {
        cuda_device::thread::__unroll_config::<0>();
        let row = rs * 16 + lane % 16;
        // SAFETY: row < Q_ROWS_256 and words HALF_256/2·dh + 8·kk +
        // 4·(lane/16) .. + 4 <= ROW_WORDS_256 are one whole chunk, which
        // staged_word_256 keeps whole and 16-byte aligned inside KT,
        // published by the barrier above; every lane issues the load.
        qa[kk] = unsafe {
            let p = kt.add(staged_word_256(
                row,
                (HALF_256 / 2) * dh + 8 * kk + 4 * (lane / 16),
            ));
            wmma::ldmatrix_x4_shared_u32(shared::cvta_generic_to_shared_u32(
                p.cast_const().cast::<u8>(),
            ))
        };
    }
    // Every warp has its fragments before the first key tile lands.
    thread::sync_threads();

    let [k_rows, k_scales, v_rows, v_scales] = src.words();
    // The f16 copies: thread `tid`'s copies of a tile are pass `i`'s key
    // `tid / ROW_CHUNKS_256 + PASS_KEYS_256·i`, 16-byte column `tid %
    // ROW_CHUNKS_256`.
    let src_word = kh * ctx as usize * ROW_WORDS_256 + 4 * tid;
    let k_src = k_rows.wrapping_add(src_word);
    let v_src = v_rows.wrapping_add(src_word);
    let dst_col = 4 * (tid % ROW_CHUNKS_256);
    let key_of_pass0 = (tid / ROW_CHUNKS_256) as u32;
    // The q8 copies: the planes' first rows of this key head, codes and
    // scales a side (the scales' u32 half-words: a row's HEAD_256/32 u16).
    let rowq = kh * ctx as usize;
    let kq_src = k_rows.wrapping_add(rowq * (HEAD_256 / 4));
    let kd_src = k_scales.wrapping_add(rowq * (HEAD_256 / 32) / 2);
    let vq_src = v_rows.wrapping_add(rowq * (HEAD_256 / 4));
    let vd_src = v_scales.wrapping_add(rowq * (HEAD_256 / 32) / 2);
    let tiles = hi.div_ceil(KEY_TILE_U32);
    // The walk's first tile: the one holding the block's smallest first key.
    let kb_first = if LOWER {
        (lo / KEY_TILE_U32).min(tiles)
    } else {
        0
    };
    // Stage tile `kb` of the plane `src` into `dst` (the 128 body's
    // `stage!` at this row width). A key below `lo` is staged as a key past
    // `hi` is.
    macro_rules! stage {
        ($dst:expr, $src:expr, $kb:expr, $full:literal) => {{
            let (dst, src, kb): (*mut u32, *const u32, u32) = ($dst, $src, $kb);
            let tile_src = src.wrapping_add(kb as usize * TILE_WORDS_256);
            let key0 = kb * KEY_TILE_U32 + key_of_pass0;
            for i in 0..CHUNKS_256 {
                cuda_device::thread::__unroll_config::<0>();
                let at = staged_word_256(tid / ROW_CHUNKS_256 + i * PASS_KEYS_256, dst_col);
                // SAFETY: the key is below KEY_TILE and words dst_col ..
                // + 4 are one whole chunk, which staged_word_256 keeps
                // whole: the 16 destination bytes are inside the tile and
                // 16-byte aligned.
                let d = unsafe { dst.add(at) };
                let key = key0 + ((i * PASS_KEYS_256) as u32);
                if $full || (key < hi && (!LOWER || key >= lo)) {
                    // SAFETY: the key is below hi <= ctx, so its row is
                    // inside key head kh's plane (launch contract); rows
                    // are 512 bytes and both planes start 16-byte aligned
                    // (host-checked), so the source is 16-byte aligned.
                    unsafe {
                        cp_async_cg_16(d, tile_src.wrapping_add(i * PASS_KEYS_256 * ROW_WORDS_256))
                    };
                } else {
                    // SAFETY: `d` is 16 bytes inside the tile, 16-byte
                    // aligned, this thread's chunk alone, and no warp
                    // reads the tile before the barrier after this
                    // stage's wait.
                    unsafe { *d.cast::<U32x4>() = U32x4::splat(0) };
                }
            }
            // SAFETY: closes this thread's copies above as one group.
            unsafe { cp_async_commit_group() };
        }};
    }
    // Stage tile `kb` of one side's q8 planes (the module doc's `_q8`
    // paragraph): the code words as 16-byte `cp.async` copies into the low
    // half of the f16 tile's rows (four chunks a thread), the scales as one
    // 16-byte copy a row, a key at or past `hi` (or below `lo` with LOWER)
    // zeroed and never read, committed as one group. `full` (a literal) is
    // the tile lying below `hi` and, with LOWER, at or above `lo`.
    macro_rules! stage_q8 {
        ($tile:expr, $sk:expr, $qs:expr, $ds:expr, $kb:expr, $full:literal) => {{
            let (tile, sk, qs, ds, kb): (*mut u32, *mut u16, *const u32, *const u32, u32) =
                ($tile, $sk, $qs, $ds, $kb);
            let sk32 = sk.cast::<u32>();
            for i in 0..TILE_CCHUNKS {
                cuda_device::thread::__unroll_config::<0>();
                let e = tid + THREADS_256 * i;
                let row = e / ROW_CCHUNKS;
                let col = e - row * ROW_CCHUNKS;
                // SAFETY: row < KEY_TILE and words 4·col .. + 4 are one whole
                // chunk of the row's low half: the 16 destination bytes are
                // inside the tile and 16-byte aligned.
                let d = unsafe { tile.add(row * ROW_WORDS_256 + 4 * col) };
                let key = kb * KEY_TILE_U32 + (row as u32);
                if $full || (key < hi && (!LOWER || key >= lo)) {
                    // SAFETY: the key is below hi <= ctx, so its row is inside
                    // this key head's codes plane (launch contract); rows are
                    // HEAD_256/4 words and the plane starts 16-byte aligned
                    // (host-checked), so the source is 16-byte aligned.
                    unsafe {
                        cp_async_cg_16(
                            d,
                            qs.wrapping_add(
                                (kb as usize * KEY_TILE + row) * (HEAD_256 / 4) + 4 * col,
                            ),
                        )
                    };
                } else {
                    // SAFETY: `d` is 16 bytes inside the tile, 16-byte
                    // aligned, this thread's chunk alone, and no warp reads
                    // the tile before the barrier after this stage's wait (the
                    // widening writes the dead row's zeros over it).
                    unsafe { *d.cast::<U32x4>() = U32x4::splat(0) };
                }
            }
            if tid < KEY_TILE {
                // SAFETY: the row's HEAD_256/32 u16 are one aligned 16-byte
                // chunk of the scales tile, this thread's row alone.
                let d = unsafe { sk32.add(4 * tid) };
                let key = kb * KEY_TILE_U32 + (tid as u32);
                if $full || (key < hi && (!LOWER || key >= lo)) {
                    // SAFETY: the key is below hi <= ctx: its scales are
                    // inside the plane (launch contract), one 16-byte aligned
                    // chunk of HEAD_256/32 u16.
                    unsafe {
                        cp_async_cg_16(d, ds.wrapping_add((kb as usize * KEY_TILE + tid) * 4))
                    };
                } else {
                    // SAFETY: this thread's chunk alone, read by no one (the
                    // dead row widens to zeros).
                    unsafe { *d.cast::<U32x4>() = U32x4::splat(0) };
                }
            }
            // SAFETY: closes this thread's copies above as one group.
            unsafe { cp_async_commit_group() };
        }};
    }
    // Stage tile `kb` of the key (`K`) or value (`V`) side in the cache's
    // format.
    macro_rules! stage_side {
        (K, $kb:expr, $full:literal) => {
            if S::Q8 {
                stage_q8!(kt, sk, kq_src, kd_src, $kb, $full);
            } else {
                stage!(kt, k_src, $kb, $full);
            }
        };
        (V, $kb:expr, $full:literal) => {
            if S::Q8 {
                stage_q8!(vt, sv, vq_src, vd_src, $kb, $full);
            } else {
                stage!(vt, v_src, $kb, $full);
            }
        };
    }
    // A q8 tile's in-place widening, `$tile` with its scales `$scales`:
    // warp `wid` widens its own rows (`dequant_row` reads before its warp
    // sync and writes after), a key at or past `hi` (or below `lo` with
    // LOWER) four zero words, then a barrier publishes the tile.
    macro_rules! widen {
        ($tile:expr, $scales:expr, $kb:expr) => {{
            let rows0 = wid * WID_ROWS;
            for r in 0..WID_ROWS {
                let row = rows0 + r;
                let base = row * ROW_WORDS_256;
                let key = $kb * KEY_TILE_U32 + (row as u32);
                if key < hi && (!LOWER || key >= lo) {
                    // SAFETY: the row is this warp's alone, its raw words and
                    // scales published by the barrier before this expansion.
                    dequant_row!(2, $tile, base, $scales.wrapping_add(row * 8), row & 7, lane);
                } else {
                    // SAFETY: the row is this warp's alone in the tile.
                    zero_row!(2, $tile, base, row & 7, lane);
                }
            }
            thread::sync_threads();
        }};
    }
    if kb_first < tiles {
        if KEY_TILE_U32 * (kb_first + 1) <= hi && (!LOWER || KEY_TILE_U32 * kb_first >= lo) {
            stage_side!(K, kb_first, true);
        } else {
            stage_side!(K, kb_first, false);
        }
    }

    // This lane's `ldmatrix` addresses in the first sixteen rows and the
    // first line of the warp's half, one per chunk pair `j` (the 128
    // body's): the half starts HALF_256/2 words, whole lines, into a row,
    // so every other address of the walk is one of these plus a constant.
    let k_row = lane % 8 + 8 * (lane / 16);
    let v_row = lane % 8 + 8 * ((lane / 8) % 2);
    let half_bytes = (4 * (HALF_256 / 2) * dh) as u32;
    let mut k_at = [0u32; LINE_PAIRS];
    let mut v_at = [0u32; LINE_PAIRS];
    for j in 0..LINE_PAIRS {
        cuda_device::thread::__unroll_config::<0>();
        // SAFETY: k_row, v_row < 16 <= KEY_TILE and words 8·j + 4 .. + 4
        // <= LINE: inside the tiles; only the addresses are formed here
        // (the half's HALF_256/2 more words stay inside the row).
        unsafe {
            k_at[j] = shared::cvta_generic_to_shared_u32(
                kt.add(staged_word_256(k_row, 8 * j + 4 * ((lane / 8) % 2)))
                    .cast_const()
                    .cast::<u8>(),
            ) + half_bytes;
            v_at[j] = shared::cvta_generic_to_shared_u32(
                vt.add(staged_word_256(v_row, 8 * j + 4 * (lane / 16)))
                    .cast_const()
                    .cast::<u8>(),
            ) + half_bytes;
        }
    }
    // This lane's slots of its own and its partner's published scores.
    let ex_own = ex.wrapping_add(wid * EXCH_WARP + lane);
    let ex_mate = ex.wrapping_add((wid ^ SLICES_256) * EXCH_WARP + lane);

    let t4k = 2 * t4 as u32;
    let mut m = [f32::NEG_INFINITY; 2];
    let mut l = [0.0f32; 2];
    let mut o = [[0.0f32; 4]; 2 * DIM_PAIRS_256];
    let mut kb = kb_first;
    while kb < tiles {
        let end = KEY_TILE_U32 * (kb + 1);
        // Key tile kb has landed, and every warp is done reading the
        // value tile of kb − 1 and the exchange of kb − 1.
        // SAFETY: waits for this thread's outstanding groups; the barrier
        // then publishes every thread's copies.
        unsafe { cp_async_wait_group(0) };
        thread::sync_threads();
        // A q8 key tile landed as raw codes and scales: widened in place
        // before the scores read it.
        if S::Q8 {
            widen!(kt, sk, kb);
        }
        if end <= hi && (!LOWER || KEY_TILE_U32 * kb >= lo) {
            stage_side!(V, kb, true);
        } else {
            stage_side!(V, kb, false);
        }
        // Both warps of a pair hold the same rows, so the same `live`.
        let live = KEY_TILE_U32 * kb < warp_hi && (!LOWER || end > warp_first_lo);

        // ---- S_w = Q·Kᵀ over the warp's half, published.
        let mut sc = [[0.0f32; 4]; KEY_NT];
        if live {
            for kk in 0..QK_STEPS_256 {
                cuda_device::thread::__unroll_config::<0>();
                for nj in 0..KEY_NT / 2 {
                    cuda_device::thread::__unroll_config::<0>();
                    // Key row 16·nj + k_row, chunk 2·kk + (lane/8) % 2 of
                    // the warp's half.
                    let at = k_at[kk % LINE_PAIRS]
                        + (4 * (16 * nj * ROW_WORDS_256 + LINE * (kk / LINE_PAIRS))) as u32;
                    // SAFETY: the row is below KEY_TILE and the chunk is
                    // whole inside KT (staged_word_256,
                    // layout_holds), published by the barrier
                    // above; every lane issues the load, then both
                    // `mma.sync` with its own fragments.
                    unsafe {
                        let bf = wmma::ldmatrix_x4_shared_u32(at);
                        sc[2 * nj] = wmma::mma_m16n8k16_f32_f16(sc[2 * nj], qa[kk], [bf[0], bf[1]]);
                        sc[2 * nj + 1] =
                            wmma::mma_m16n8k16_f32_f16(sc[2 * nj + 1], qa[kk], [bf[2], bf[3]]);
                    }
                }
            }
            for nt in 0..KEY_NT {
                cuda_device::thread::__unroll_config::<0>();
                for j in 0usize..4 {
                    cuda_device::thread::__unroll_config::<0>();
                    // SAFETY: (4·nt + j)·32 + lane < EXCH_WARP: this
                    // lane's own slot of this warp's exchange.
                    unsafe { *ex_own.add((4 * nt + j) * 32) = sc[nt][j] };
                }
            }
        }
        // Both halves are published.
        thread::sync_threads();

        let mut pa = [[0u32; 4]; PV_STEPS];
        let mut f = [1.0f32; 2];
        if live {
            // S = S_w + S_mate = S_lo + S_hi: a rounded add is
            // commutative, so both warps of the pair hold the same bits.
            for nt in 0..KEY_NT {
                cuda_device::thread::__unroll_config::<0>();
                for j in 0usize..4 {
                    cuda_device::thread::__unroll_config::<0>();
                    // SAFETY: the partner's slot for the same fragment
                    // value, written before the barrier above.
                    let mate = unsafe { *ex_mate.add((4 * nt + j) * 32) };
                    sc[nt][j] = add_rn_f32(sc[nt][j], mate);
                }
            }

            prefill_softmax! {
                unmasked: end <= warp_lo && (!LOWER || KEY_TILE_U32 * kb >= warp_first_hi),
                key0: KEY_TILE_U32 * kb + t4k,
                tiles: (KEY_NT, PV_STEPS),
                scores: sc,
                counts: cnt,
                lower: (LOWER, first),
                scale: scale,
                state: (m, l, f),
                weights: pa,
            }

            if warp::any(f[0] != 1.0 || f[1] != 1.0) {
                for nd in 0..2 * DIM_PAIRS_256 {
                    cuda_device::thread::__unroll_config::<0>();
                    for j in 0usize..4 {
                        cuda_device::thread::__unroll_config::<0>();
                        o[nd][j] = mul_rn_f32(o[nd][j], f[j / 2]);
                    }
                }
            }
        }

        // The value tile has landed, and every warp is done reading the
        // key tile.
        // SAFETY: waits until none of this thread's groups is pending —
        // its copies of value tile kb; the barrier then publishes every
        // thread's copies before the value product reads VT.
        unsafe { cp_async_wait_group(0) };
        thread::sync_threads();
        // A q8 value tile likewise, before the value product reads it.
        if S::Q8 {
            widen!(vt, sv, kb);
        }
        if kb + 1 < tiles {
            if end + KEY_TILE_U32 <= hi {
                stage_side!(K, kb + 1, true);
            } else {
                stage_side!(K, kb + 1, false);
            }
        }

        // ---- O += P̂·V over the warp's half.
        if live {
            for kk2 in 0..PV_STEPS {
                cuda_device::thread::__unroll_config::<0>();
                for nd in 0..DIM_PAIRS_256 {
                    cuda_device::thread::__unroll_config::<0>();
                    // Value row 16·kk2 + v_row, chunk 2·nd + lane/16 of
                    // the warp's half.
                    let at = v_at[nd % LINE_PAIRS]
                        + (4 * (16 * kk2 * ROW_WORDS_256 + LINE * (nd / LINE_PAIRS))) as u32;
                    // SAFETY: the row is below KEY_TILE and the chunk is
                    // whole inside VT, published by the barrier above;
                    // every lane issues the load, then both `mma.sync`
                    // with its own fragments.
                    unsafe {
                        let bf = wmma::ldmatrix_x4_trans_shared_u32(at);
                        o[2 * nd] = wmma::mma_m16n8k16_f32_f16(o[2 * nd], pa[kk2], [bf[0], bf[1]]);
                        o[2 * nd + 1] =
                            wmma::mma_m16n8k16_f32_f16(o[2 * nd + 1], pa[kk2], [bf[2], bf[3]]);
                    }
                }
            }
        }
        kb += 1;
    }

    // ---- the row sums over the four lanes of a row, then o · (1/l).
    prefill_pack_store! {
        acc: (o, l, bad, 2 * DIM_PAIRS_256),
        rows: (t0 + pps * rs, rows, PACK, g, t4),
        cols: (HALF_256 * dh, HEAD_256),
        out: (y, n_head, head0),
    }
}

#[cuda_module]
mod flash_gqa_prefill_kernels {
    use super::*;
    use cuda_device::async_copy::{cp_async_cg_16, cp_async_commit_group, cp_async_wait_group};
    use cuda_device::float::{add_rn_f32, mul_rn_f32};
    use cuda_device::vector::U32x4;

    /// The module doc's attention. Block `b` is key head `b % n_kv` and query
    /// tile `n_tiles − 1 − b / n_kv` (the deepest tiles first), rows
    /// `POSITIONS·tile ..` of `t_rows`; row `t`'s query heads `kh·GROUP ..`
    /// are read at `q[(t·n_head + kh·GROUP + g)·HEAD ..]` and its output
    /// written at the same index of `y`. Every guard below is block- or
    /// warp-uniform.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[allow(
        clippy::needless_range_loop,
        reason = "a range `for` is the loop the unroller recognizes; an iterator loop keeps its array in a local depot"
    )]
    #[kernel]
    #[launch_bounds(128, 2)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        requires = (
            n_keys_buf.len() >= t_rows,
            q.len() >= t_rows * n_kv * 8 * 128,
            kc.len() >= n_kv * ctx * 128,
            vc.len() >= n_kv * ctx * 128,
            y.len() >= t_rows * n_kv * 8 * 128
        )
    )]
    pub fn gqa_prefill_flash(
        q: &[f32],
        kc: &[u16],
        vc: &[u16],
        n_keys_buf: &[u32],
        scale: f32,
        n_kv: u32,
        ctx: u32,
        t_rows: u32,
        fault: FaultSink,
        mut y: DisjointSlice<f32>,
    ) {
        // 128-byte aligned: the layout's sectors and halves are aligned only
        // if the tile base is.
        static mut KT: SharedArray<u32, TILE_WORDS, 128> = SharedArray::UNINIT;
        static mut VT: SharedArray<u32, TILE_WORDS, 128> = SharedArray::UNINIT;

        let b = thread::blockIdx_x() as usize;
        let nkv = n_kv as usize;
        let rows = t_rows as usize;
        let n_tiles = rows.div_ceil(POSITIONS);
        if b >= n_tiles * nkv {
            return; // block-uniform
        }
        let qt = n_tiles - 1 - b / nkv;
        let kh = b % nkv;
        let tid = thread::threadIdx_x() as usize;
        let lane = warp::lane_id() as usize;
        let wid = tid / 32;
        let g = lane / 4;
        let t4 = lane % 4;
        let n_head = nkv * GROUP;
        let t0 = qt * POSITIONS;

        // SAFETY: block-shared, TILE_WORDS words each; every write is ordered
        // before its reads by a block barrier (the cp.async copies by the
        // wait before it).
        let (kt, vt) = unsafe {
            (
                SharedArray::as_raw_mut_ptr(&raw mut KT),
                SharedArray::as_raw_mut_ptr(&raw mut VT),
            )
        };

        // Every row's count as the walk uses it: 0 for a row past `t_rows`
        // and for a refused one; the block's largest bounds the keys loaded.
        // Keys and counts are u32: the host keeps `ctx + KEY_TILE` in range.
        let mut hi = 0u32;
        let mut cnt = [0u32; 2];
        let mut bad = [false; 2];
        for lp in 0..POSITIONS {
            cuda_device::thread::__unroll_config::<0>();
            let t = t0 + lp;
            let (c, refused) = if t < rows {
                // SAFETY: t < t_rows <= n_keys_buf.len() (launch contract).
                let c = unsafe { *n_keys_buf.get_unchecked(t) };
                if c == 0 || c > ctx {
                    (0, true)
                } else {
                    (c, false)
                }
            } else {
                (0, false)
            };
            hi = hi.max(c);
            if lp / 2 == wid {
                cnt[lp % 2] = c;
                bad[lp % 2] = refused;
            }
        }
        if lane == 0 && (bad[0] || bad[1]) {
            fault.raise(FaultSite::KeyCount);
        }
        let warp_hi = cnt[0].max(cnt[1]);
        let warp_lo = cnt[0].min(cnt[1]);

        // The block's query rows, rounded to f16 pairs, into the key tile:
        // staged row `lp·GROUP + g` is position `t0 + lp`'s head `kh·GROUP +
        // g`, which puts position `2·w + h` at rows `16·w + 8·h ..` — warp
        // `w`'s fragment. A row past `t_rows` is zero. Thread `tid` stages
        // word `tid % Q_PAIRS` of every row its passes cover, and every load
        // is issued before the first conversion.
        let wd = tid % Q_PAIRS;
        let r0 = tid / Q_PAIRS;
        // u64 words of q: this thread's word of position t0's head kh·GROUP +
        // r0, and the step from one position to the next.
        let q_first = (t0 * n_head + kh * GROUP + r0) * Q_PAIRS + wd;
        let q_pos = n_head * Q_PAIRS;
        let q64 = q.as_ptr().cast::<u64>();
        let mut raw = [0u64; Q_WORDS];
        for e in 0..Q_WORDS {
            cuda_device::thread::__unroll_config::<0>();
            let t = t0 + e / Q_POS_PASSES;
            if t < rows {
                let at = q_first
                    + (e / Q_POS_PASSES) * q_pos
                    + Q_ROW_STEP * (e % Q_POS_PASSES) * Q_PAIRS;
                // SAFETY: t < t_rows and the head kh·GROUP + Q_ROW_STEP·(e %
                // Q_POS_PASSES) + r0 < n_head, so the row lies inside q
                // (launch contract); a row is HEAD f32 = HEAD/2 u64 and q
                // starts 8-byte aligned (host-checked), so the u64 read of
                // values 2·wd, 2·wd + 1 is aligned.
                raw[e] = unsafe { *q64.add(at) };
            }
        }
        // The staged word of each pass of the first position; a later
        // position's is it plus whole groups of rows (layout_holds).
        let mut q_at = [0usize; Q_POS_PASSES];
        for j in 0..Q_POS_PASSES {
            cuda_device::thread::__unroll_config::<0>();
            q_at[j] = staged_word(Q_ROW_STEP * j + r0, wd);
        }
        for e in 0..Q_WORDS {
            cuda_device::thread::__unroll_config::<0>();
            let pair = f32x2_to_f16x2_bits(
                f32::from_bits(raw[e] as u32),
                f32::from_bits((raw[e] >> 32) as u32),
            );
            // SAFETY: row Q_ROW_STEP·e + r0 < Q_ROWS <= KEY_TILE and wd <
            // ROW_WORDS: inside KT; one owner per word.
            unsafe {
                *kt.add(q_at[e % Q_POS_PASSES] + (e / Q_POS_PASSES) * GROUP * ROW_WORDS) = pair
            };
        }
        thread::sync_threads();

        // This warp's 16 query rows as A fragments, one per k16 step.
        let mut qa = [[0u32; 4]; QK_STEPS];
        for kk in 0..QK_STEPS {
            cuda_device::thread::__unroll_config::<0>();
            let row = wid * 16 + lane % 16;
            // SAFETY: row < Q_ROWS and words 8·kk + 4·(lane/16) .. + 4 <=
            // HEAD/2 are one whole chunk, which staged_word keeps whole and
            // 16-byte aligned inside KT, published by the barrier above;
            // every lane issues the load.
            qa[kk] = unsafe {
                let p = kt.add(staged_word(row, 8 * kk + 4 * (lane / 16)));
                wmma::ldmatrix_x4_shared_u32(shared::cvta_generic_to_shared_u32(
                    p.cast_const().cast::<u8>(),
                ))
            };
        }
        // Every warp has its fragments before the first key tile lands.
        thread::sync_threads();

        // Thread `tid`'s copies of a tile: pass `i` is key `tid / ROW_CHUNKS
        // + PASS_KEYS·i`, 16-byte column `tid % ROW_CHUNKS` — source word
        // `4·tid + PASS_KEYS·ROW_WORDS·i` past the tile's first row, so a tile
        // is one pointer step and each pass a constant offset from it.
        let src_word = kh * ctx as usize * ROW_WORDS + 4 * tid;
        let k_src = kc.as_ptr().cast::<u32>().wrapping_add(src_word);
        let v_src = vc.as_ptr().cast::<u32>().wrapping_add(src_word);
        let dst_col = 4 * (tid % ROW_CHUNKS);
        let key_of_pass0 = (tid / ROW_CHUNKS) as u32;
        let tiles = hi.div_ceil(KEY_TILE_U32);
        // Stage tile `kb` of the plane `src` into `dst`: 16-byte copies of
        // the keys below `hi`, zeros past it, committed as one group. `full`
        // (a literal) is the tile lying below `hi`, where no copy is tested.
        macro_rules! stage {
            ($dst:expr, $src:expr, $kb:expr, $full:literal) => {{
                let (dst, src, kb): (*mut u32, *const u32, u32) = ($dst, $src, $kb);
                let tile_src = src.wrapping_add(kb as usize * TILE_WORDS);
                let key0 = kb * KEY_TILE_U32 + key_of_pass0;
                for i in 0..CHUNKS {
                    cuda_device::thread::__unroll_config::<0>();
                    let at = staged_word(tid / ROW_CHUNKS + i * PASS_KEYS, dst_col);
                    // SAFETY: key tid / ROW_CHUNKS + PASS_KEYS·i < KEY_TILE and
                    // words 4·(tid % ROW_CHUNKS) .. + 4 are one whole chunk,
                    // which staged_word keeps whole: the 16 destination bytes
                    // are inside the tile and 16-byte aligned.
                    let d = unsafe { dst.add(at) };
                    if $full || key0 + ((i * PASS_KEYS) as u32) < hi {
                        // SAFETY: the key is below hi <= ctx (every key of a
                        // full tile is), so its row is inside key head kh's
                        // plane (launch contract); rows are HEAD f16 = 256
                        // bytes and both planes start 16-byte aligned
                        // (host-checked), so the source is 16-byte aligned.
                        unsafe {
                            cp_async_cg_16(d, tile_src.wrapping_add(i * PASS_KEYS * ROW_WORDS))
                        };
                    } else {
                        // SAFETY: `d` is 16 bytes inside the tile and 16-byte
                        // aligned (formed above), the chunk of key
                        // tid / ROW_CHUNKS + PASS_KEYS·i, column tid % ROW_CHUNKS
                        // — no other thread's copy or store names it — and no
                        // warp reads the tile before the barrier after this
                        // stage's wait.
                        unsafe { *d.cast::<U32x4>() = U32x4::splat(0) };
                    }
                }
                // SAFETY: closes this thread's copies above as one group.
                unsafe { cp_async_commit_group() };
            }};
        }
        if tiles > 0 {
            if KEY_TILE_U32 <= hi {
                stage!(kt, k_src, 0, true);
            } else {
                stage!(kt, k_src, 0, false);
            }
        }

        // This lane's `ldmatrix` addresses in the first sixteen rows and the
        // first half-row, one per chunk pair `j`: the scores' key row and
        // chunk `2·j + (lane/8) % 2`, the values' row and chunk `2·j +
        // lane/16`. Every other address of the walk is one of these plus a
        // constant (`layout_holds`).
        let k_row = lane % 8 + 8 * (lane / 16);
        let v_row = lane % 8 + 8 * ((lane / 8) % 2);
        let mut k_at = [0u32; HALF_PAIRS];
        let mut v_at = [0u32; HALF_PAIRS];
        for j in 0..HALF_PAIRS {
            cuda_device::thread::__unroll_config::<0>();
            // SAFETY: k_row, v_row < 16 <= KEY_TILE and words 8·j + 4 .. + 4
            // <= HALF_ROW: inside the tiles; only the addresses are formed here.
            unsafe {
                k_at[j] = shared::cvta_generic_to_shared_u32(
                    kt.add(staged_word(k_row, 8 * j + 4 * ((lane / 8) % 2)))
                        .cast_const()
                        .cast::<u8>(),
                );
                v_at[j] = shared::cvta_generic_to_shared_u32(
                    vt.add(staged_word(v_row, 8 * j + 4 * (lane / 16)))
                        .cast_const()
                        .cast::<u8>(),
                );
            }
        }

        let t4k = 2 * t4 as u32;
        let mut m = [f32::NEG_INFINITY; 2];
        let mut l = [0.0f32; 2];
        let mut o = [[0.0f32; 4]; 2 * DIM_PAIRS];
        let mut kb = 0u32;
        while kb < tiles {
            // One past the tile's last key.
            let end = KEY_TILE_U32 * (kb + 1);
            // Key tile kb has landed, and every warp is done reading the
            // value tile of kb − 1.
            // SAFETY: waits for this thread's outstanding groups; the barrier
            // then publishes every thread's copies.
            unsafe { cp_async_wait_group(0) };
            thread::sync_threads();
            if end <= hi {
                stage!(vt, v_src, kb, true);
            } else {
                stage!(vt, v_src, kb, false);
            }
            let live = KEY_TILE_U32 * kb < warp_hi;

            // ---- S = Q·Kᵀ, then the online softmax into f16 weights.
            let mut pa = [[0u32; 4]; PV_STEPS];
            let mut f = [1.0f32; 2];
            if live {
                let mut sc = [[0.0f32; 4]; KEY_NT];
                for kk in 0..QK_STEPS {
                    cuda_device::thread::__unroll_config::<0>();
                    for nj in 0..KEY_NT / 2 {
                        cuda_device::thread::__unroll_config::<0>();
                        // Key row 16·nj + k_row, chunk 2·kk + (lane/8) % 2.
                        let at = k_at[kk % HALF_PAIRS]
                            + (4 * (16 * nj * ROW_WORDS + HALF_ROW * (kk / HALF_PAIRS))) as u32;
                        // SAFETY: the row is below KEY_TILE and the chunk is
                        // whole inside KT (staged_word, layout_holds),
                        // published by the barrier above; every lane issues
                        // the load, then both `mma.sync` with its own
                        // fragments.
                        unsafe {
                            let bf = wmma::ldmatrix_x4_shared_u32(at);
                            sc[2 * nj] =
                                wmma::mma_m16n8k16_f32_f16(sc[2 * nj], qa[kk], [bf[0], bf[1]]);
                            sc[2 * nj + 1] =
                                wmma::mma_m16n8k16_f32_f16(sc[2 * nj + 1], qa[kk], [bf[2], bf[3]]);
                        }
                    }
                }

                prefill_softmax! {
                    unmasked: end <= warp_lo,
                    key0: KEY_TILE_U32 * kb + t4k,
                    tiles: (KEY_NT, PV_STEPS),
                    scores: sc,
                    counts: cnt,
                    scale: scale,
                    state: (m, l, f),
                    weights: pa,
                }

                // The accumulators times f, unless no lane of the warp has a
                // factor other than 1 (the vote keeps the branch uniform).
                if warp::any(f[0] != 1.0 || f[1] != 1.0) {
                    for nd in 0..2 * DIM_PAIRS {
                        cuda_device::thread::__unroll_config::<0>();
                        for j in 0usize..4 {
                            cuda_device::thread::__unroll_config::<0>();
                            o[nd][j] = mul_rn_f32(o[nd][j], f[j / 2]);
                        }
                    }
                }
            }

            // The value tile has landed, and every warp is done reading the
            // key tile.
            // SAFETY: waits until none of this thread's groups is pending —
            // its copies of value tile kb, the one group issued since the
            // wait above; the barrier then publishes every thread's copies
            // before the value product reads VT.
            unsafe { cp_async_wait_group(0) };
            thread::sync_threads();
            if kb + 1 < tiles {
                if end + KEY_TILE_U32 <= hi {
                    stage!(kt, k_src, kb + 1, true);
                } else {
                    stage!(kt, k_src, kb + 1, false);
                }
            }

            // ---- O += P̂·V.
            if live {
                for kk2 in 0..PV_STEPS {
                    cuda_device::thread::__unroll_config::<0>();
                    for nd in 0..DIM_PAIRS {
                        cuda_device::thread::__unroll_config::<0>();
                        // Value row 16·kk2 + v_row, chunk 2·nd + lane/16.
                        let at = v_at[nd % HALF_PAIRS]
                            + (4 * (16 * kk2 * ROW_WORDS + HALF_ROW * (nd / HALF_PAIRS))) as u32;
                        // SAFETY: the row is below KEY_TILE and the chunk is
                        // whole inside VT (staged_word, layout_holds),
                        // published by the barrier above; every lane issues
                        // the load, then both `mma.sync` with its own
                        // fragments.
                        unsafe {
                            let bf = wmma::ldmatrix_x4_trans_shared_u32(at);
                            o[2 * nd] =
                                wmma::mma_m16n8k16_f32_f16(o[2 * nd], pa[kk2], [bf[0], bf[1]]);
                            o[2 * nd + 1] =
                                wmma::mma_m16n8k16_f32_f16(o[2 * nd + 1], pa[kk2], [bf[2], bf[3]]);
                        }
                    }
                }
            }
            kb += 1;
        }

        // ---- the row sums over the four lanes of a row, then o · (1/l).
        let mut inv = [0.0f32; 2];
        for h in 0usize..2 {
            cuda_device::thread::__unroll_config::<0>();
            let a = add_rn_f32(l[h], warp::shuffle_xor_f32(l[h], 1));
            let s = add_rn_f32(a, warp::shuffle_xor_f32(a, 2));
            inv[h] = 1.0 / s;
        }
        for nd in 0..2 * DIM_PAIRS {
            cuda_device::thread::__unroll_config::<0>();
            for j in 0usize..4 {
                cuda_device::thread::__unroll_config::<0>();
                let h = j / 2;
                let t = t0 + 2 * wid + h;
                if t < rows {
                    let d = 8 * nd + 2 * t4 + j % 2;
                    let v = if bad[h] {
                        f32::NAN
                    } else {
                        mul_rn_f32(o[nd][j], inv[h])
                    };
                    // SAFETY: t < t_rows and kh·GROUP + g < n_head, d < HEAD:
                    // the index is below t_rows·n_head·HEAD <= y.len() (launch
                    // contract); one owner lane per value.
                    unsafe {
                        *y.get_unchecked_mut((t * n_head + kh * GROUP + g) * HEAD + d) = v;
                    }
                }
            }
        }
    }

    /// The attention of the module doc over heads of [`HEAD_256`] values,
    /// eight warps a block (the 256 body's section above). Block `b` is key
    /// head `b % n_kv` and query tile `n_tiles − 1 − b / n_kv`, rows
    /// `POSITIONS·tile ..` of `t_rows`; row `t`'s query heads `kh·GROUP ..`
    /// are read at `q[(t·n_head + kh·GROUP + g)·256 ..]` and its output
    /// written at the same index of `y`. The key tile, the value tile and
    /// the score exchange live in the block's dynamic shared memory. Every
    /// guard below is block- or warp-pair-uniform.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[allow(
        clippy::needless_range_loop,
        reason = "a range `for` is the loop the unroller recognizes; an iterator loop keeps its array in a local depot"
    )]
    #[kernel]
    #[launch_bounds(256, 1)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        dynamic_shared = 98304,
        requires = (
            n_keys_buf.len() >= t_rows,
            q.len() >= t_rows * n_kv * 8 * 256,
            kc.len() >= n_kv * ctx * 256,
            vc.len() >= n_kv * ctx * 256,
            y.len() >= t_rows * n_kv * 8 * 256
        )
    )]
    pub fn gqa_prefill_flash_256(
        q: &[f32],
        kc: &[u16],
        vc: &[u16],
        n_keys_buf: &[u32],
        scale: f32,
        n_kv: u32,
        ctx: u32,
        t_rows: u32,
        fault: FaultSink,
        mut y: DisjointSlice<f32>,
    ) {
        let b = thread::blockIdx_x() as usize;
        let nkv = n_kv as usize;
        let rows = t_rows as usize;
        let n_tiles = rows.div_ceil(POSITIONS);
        if b >= n_tiles * nkv {
            return; // block-uniform
        }
        let qt = n_tiles - 1 - b / nkv;
        let kh = b % nkv;
        let tid = thread::threadIdx_x() as usize;
        let lane = warp::lane_id() as usize;
        let wid = tid / 32;
        // The warp's row slice (its two positions) and dim half.
        let rs = wid % SLICES_256;
        let dh = wid / SLICES_256;
        let g = lane / 4;
        let t4 = lane % 4;
        let n_head = nkv * GROUP;
        let t0 = qt * POSITIONS;

        // The block's dynamic shared memory, DYN_BYTES_256 bytes (the launch
        // contract): the key tile, the value tile and the exchange at byte
        // offsets 0, TILE_WORDS_256·4 and 2·TILE_WORDS_256·4, every write
        // ordered before its reads by a block barrier (the cp.async copies by
        // the wait before it). The base is declared 16-byte aligned — what
        // the 16-byte copies and `ldmatrix` rows need; the bundle's other
        // kernels share that declaration, and a wider one pads their static
        // shared memory. With no static shared memory here the base is the
        // window's start, so the tiles' 128-byte lines are the layout's.
        let (kt, vt, ex) = (
            DynamicSharedArray::<u32, 16>::get(),
            DynamicSharedArray::<u32, 16>::offset(TILE_WORDS_256 * 4),
            DynamicSharedArray::<f32, 16>::offset(2 * TILE_WORDS_256 * 4),
        );

        // Every row's count as the walk uses it (the 128 body's rule); the
        // warp's two are its row slice's.
        let mut hi = 0u32;
        let mut cnt = [0u32; 2];
        let mut bad = [false; 2];
        for lp in 0..POSITIONS {
            cuda_device::thread::__unroll_config::<0>();
            let t = t0 + lp;
            let (c, refused) = if t < rows {
                // SAFETY: t < t_rows <= n_keys_buf.len() (launch contract).
                let c = unsafe { *n_keys_buf.get_unchecked(t) };
                if c == 0 || c > ctx {
                    (0, true)
                } else {
                    (c, false)
                }
            } else {
                (0, false)
            };
            hi = hi.max(c);
            if lp / 2 == rs {
                cnt[lp % 2] = c;
                bad[lp % 2] = refused;
            }
        }
        if lane == 0 && dh == 0 && (bad[0] || bad[1]) {
            fault.raise(FaultSite::KeyCount);
        }
        let warp_hi = cnt[0].max(cnt[1]);
        let warp_lo = cnt[0].min(cnt[1]);

        // The block's query rows, rounded to f16 pairs, into the key tile:
        // staged row `lp·GROUP + g` is position `t0 + lp`'s head `kh·GROUP +
        // g`, so row slice `r` is rows `16·r ..`. A row past `t_rows` is
        // zero. Thread `tid` stages word `tid % Q_PAIRS_256` of every row its
        // passes cover, every load issued before the first conversion.
        let wd = tid % Q_PAIRS_256;
        let r0 = tid / Q_PAIRS_256;
        let q_first = (t0 * n_head + kh * GROUP + r0) * Q_PAIRS_256 + wd;
        let q_pos = n_head * Q_PAIRS_256;
        let q64 = q.as_ptr().cast::<u64>();
        let mut raw = [0u64; Q_WORDS_256];
        for e in 0..Q_WORDS_256 {
            cuda_device::thread::__unroll_config::<0>();
            let t = t0 + e / Q_POS_PASSES_256;
            if t < rows {
                let at = q_first
                    + (e / Q_POS_PASSES_256) * q_pos
                    + Q_ROW_STEP_256 * (e % Q_POS_PASSES_256) * Q_PAIRS_256;
                // SAFETY: t < t_rows and the head kh·GROUP + Q_ROW_STEP_256·(e
                // % Q_POS_PASSES_256) + r0 < n_head, so the row lies inside q
                // (launch contract); a row is 256 f32 = 128 u64 and q starts
                // 8-byte aligned (host-checked), so the u64 read is aligned.
                raw[e] = unsafe { *q64.add(at) };
            }
        }
        let mut q_at = [0usize; Q_POS_PASSES_256];
        for j in 0..Q_POS_PASSES_256 {
            cuda_device::thread::__unroll_config::<0>();
            q_at[j] = staged_word_256(Q_ROW_STEP_256 * j + r0, wd);
        }
        for e in 0..Q_WORDS_256 {
            cuda_device::thread::__unroll_config::<0>();
            let pair = f32x2_to_f16x2_bits(
                f32::from_bits(raw[e] as u32),
                f32::from_bits((raw[e] >> 32) as u32),
            );
            // SAFETY: row Q_ROW_STEP_256·e' + r0 < Q_ROWS_256 <= KEY_TILE and
            // wd < ROW_WORDS_256: inside KT; one owner per word.
            unsafe {
                *kt.add(
                    q_at[e % Q_POS_PASSES_256] + (e / Q_POS_PASSES_256) * GROUP * ROW_WORDS_256,
                ) = pair
            };
        }
        thread::sync_threads();

        // This warp's sixteen query rows over its half, one A fragment per
        // k16 step.
        let mut qa = [[0u32; 4]; QK_STEPS_256];
        for kk in 0..QK_STEPS_256 {
            cuda_device::thread::__unroll_config::<0>();
            let row = rs * 16 + lane % 16;
            // SAFETY: row < Q_ROWS_256 and words HALF_256/2·dh + 8·kk +
            // 4·(lane/16) .. + 4 <= ROW_WORDS_256 are one whole chunk, which
            // staged_word_256 keeps whole and 16-byte aligned inside KT,
            // published by the barrier above; every lane issues the load.
            qa[kk] = unsafe {
                let p = kt.add(staged_word_256(
                    row,
                    (HALF_256 / 2) * dh + 8 * kk + 4 * (lane / 16),
                ));
                wmma::ldmatrix_x4_shared_u32(shared::cvta_generic_to_shared_u32(
                    p.cast_const().cast::<u8>(),
                ))
            };
        }
        // Every warp has its fragments before the first key tile lands.
        thread::sync_threads();

        // Thread `tid`'s copies of a tile: pass `i` is key `tid /
        // ROW_CHUNKS_256 + PASS_KEYS_256·i`, 16-byte column `tid %
        // ROW_CHUNKS_256`.
        let src_word = kh * ctx as usize * ROW_WORDS_256 + 4 * tid;
        let k_src = kc.as_ptr().cast::<u32>().wrapping_add(src_word);
        let v_src = vc.as_ptr().cast::<u32>().wrapping_add(src_word);
        let dst_col = 4 * (tid % ROW_CHUNKS_256);
        let key_of_pass0 = (tid / ROW_CHUNKS_256) as u32;
        let tiles = hi.div_ceil(KEY_TILE_U32);
        // Stage tile `kb` of the plane `src` into `dst` (the 128 body's
        // `stage!` at this row width).
        macro_rules! stage {
            ($dst:expr, $src:expr, $kb:expr, $full:literal) => {{
                let (dst, src, kb): (*mut u32, *const u32, u32) = ($dst, $src, $kb);
                let tile_src = src.wrapping_add(kb as usize * TILE_WORDS_256);
                let key0 = kb * KEY_TILE_U32 + key_of_pass0;
                for i in 0..CHUNKS_256 {
                    cuda_device::thread::__unroll_config::<0>();
                    let at = staged_word_256(tid / ROW_CHUNKS_256 + i * PASS_KEYS_256, dst_col);
                    // SAFETY: the key is below KEY_TILE and words dst_col ..
                    // + 4 are one whole chunk, which staged_word_256 keeps
                    // whole: the 16 destination bytes are inside the tile and
                    // 16-byte aligned.
                    let d = unsafe { dst.add(at) };
                    if $full || key0 + ((i * PASS_KEYS_256) as u32) < hi {
                        // SAFETY: the key is below hi <= ctx, so its row is
                        // inside key head kh's plane (launch contract); rows
                        // are 512 bytes and both planes start 16-byte aligned
                        // (host-checked), so the source is 16-byte aligned.
                        unsafe {
                            cp_async_cg_16(
                                d,
                                tile_src.wrapping_add(i * PASS_KEYS_256 * ROW_WORDS_256),
                            )
                        };
                    } else {
                        // SAFETY: `d` is 16 bytes inside the tile, 16-byte
                        // aligned, this thread's chunk alone, and no warp
                        // reads the tile before the barrier after this
                        // stage's wait.
                        unsafe { *d.cast::<U32x4>() = U32x4::splat(0) };
                    }
                }
                // SAFETY: closes this thread's copies above as one group.
                unsafe { cp_async_commit_group() };
            }};
        }
        if tiles > 0 {
            if KEY_TILE_U32 <= hi {
                stage!(kt, k_src, 0, true);
            } else {
                stage!(kt, k_src, 0, false);
            }
        }

        // This lane's `ldmatrix` addresses in the first sixteen rows and the
        // first line of the warp's half, one per chunk pair `j` (the 128
        // body's): the half starts HALF_256/2 words, whole lines, into a row,
        // so every other address of the walk is one of these plus a constant.
        let k_row = lane % 8 + 8 * (lane / 16);
        let v_row = lane % 8 + 8 * ((lane / 8) % 2);
        let half_bytes = (4 * (HALF_256 / 2) * dh) as u32;
        let mut k_at = [0u32; LINE_PAIRS];
        let mut v_at = [0u32; LINE_PAIRS];
        for j in 0..LINE_PAIRS {
            cuda_device::thread::__unroll_config::<0>();
            // SAFETY: k_row, v_row < 16 <= KEY_TILE and words 8·j + 4 .. + 4
            // <= LINE: inside the tiles; only the addresses are formed here
            // (the half's HALF_256/2 more words stay inside the row).
            unsafe {
                k_at[j] = shared::cvta_generic_to_shared_u32(
                    kt.add(staged_word_256(k_row, 8 * j + 4 * ((lane / 8) % 2)))
                        .cast_const()
                        .cast::<u8>(),
                ) + half_bytes;
                v_at[j] = shared::cvta_generic_to_shared_u32(
                    vt.add(staged_word_256(v_row, 8 * j + 4 * (lane / 16)))
                        .cast_const()
                        .cast::<u8>(),
                ) + half_bytes;
            }
        }
        // This lane's slots of its own and its partner's published scores.
        let ex_own = ex.wrapping_add(wid * EXCH_WARP + lane);
        let ex_mate = ex.wrapping_add((wid ^ SLICES_256) * EXCH_WARP + lane);

        let t4k = 2 * t4 as u32;
        let mut m = [f32::NEG_INFINITY; 2];
        let mut l = [0.0f32; 2];
        let mut o = [[0.0f32; 4]; 2 * DIM_PAIRS_256];
        let mut kb = 0u32;
        while kb < tiles {
            let end = KEY_TILE_U32 * (kb + 1);
            // Key tile kb has landed, and every warp is done reading the
            // value tile of kb − 1 and the exchange of kb − 1.
            // SAFETY: waits for this thread's outstanding groups; the barrier
            // then publishes every thread's copies.
            unsafe { cp_async_wait_group(0) };
            thread::sync_threads();
            if end <= hi {
                stage!(vt, v_src, kb, true);
            } else {
                stage!(vt, v_src, kb, false);
            }
            // Both warps of a pair hold the same rows, so the same `live`.
            let live = KEY_TILE_U32 * kb < warp_hi;

            // ---- S_w = Q·Kᵀ over the warp's half, published.
            let mut sc = [[0.0f32; 4]; KEY_NT];
            if live {
                for kk in 0..QK_STEPS_256 {
                    cuda_device::thread::__unroll_config::<0>();
                    for nj in 0..KEY_NT / 2 {
                        cuda_device::thread::__unroll_config::<0>();
                        // Key row 16·nj + k_row, chunk 2·kk + (lane/8) % 2 of
                        // the warp's half.
                        let at = k_at[kk % LINE_PAIRS]
                            + (4 * (16 * nj * ROW_WORDS_256 + LINE * (kk / LINE_PAIRS))) as u32;
                        // SAFETY: the row is below KEY_TILE and the chunk is
                        // whole inside KT (staged_word_256,
                        // layout_holds), published by the barrier
                        // above; every lane issues the load, then both
                        // `mma.sync` with its own fragments.
                        unsafe {
                            let bf = wmma::ldmatrix_x4_shared_u32(at);
                            sc[2 * nj] =
                                wmma::mma_m16n8k16_f32_f16(sc[2 * nj], qa[kk], [bf[0], bf[1]]);
                            sc[2 * nj + 1] =
                                wmma::mma_m16n8k16_f32_f16(sc[2 * nj + 1], qa[kk], [bf[2], bf[3]]);
                        }
                    }
                }
                for nt in 0..KEY_NT {
                    cuda_device::thread::__unroll_config::<0>();
                    for j in 0usize..4 {
                        cuda_device::thread::__unroll_config::<0>();
                        // SAFETY: (4·nt + j)·32 + lane < EXCH_WARP: this
                        // lane's own slot of this warp's exchange.
                        unsafe { *ex_own.add((4 * nt + j) * 32) = sc[nt][j] };
                    }
                }
            }
            // Both halves are published.
            thread::sync_threads();

            let mut pa = [[0u32; 4]; PV_STEPS];
            let mut f = [1.0f32; 2];
            if live {
                // S = S_w + S_mate = S_lo + S_hi: a rounded add is
                // commutative, so both warps of the pair hold the same bits.
                for nt in 0..KEY_NT {
                    cuda_device::thread::__unroll_config::<0>();
                    for j in 0usize..4 {
                        cuda_device::thread::__unroll_config::<0>();
                        // SAFETY: the partner's slot for the same fragment
                        // value, written before the barrier above.
                        let mate = unsafe { *ex_mate.add((4 * nt + j) * 32) };
                        sc[nt][j] = add_rn_f32(sc[nt][j], mate);
                    }
                }

                prefill_softmax! {
                    unmasked: end <= warp_lo,
                    key0: KEY_TILE_U32 * kb + t4k,
                    tiles: (KEY_NT, PV_STEPS),
                    scores: sc,
                    counts: cnt,
                    scale: scale,
                    state: (m, l, f),
                    weights: pa,
                }

                if warp::any(f[0] != 1.0 || f[1] != 1.0) {
                    for nd in 0..2 * DIM_PAIRS_256 {
                        cuda_device::thread::__unroll_config::<0>();
                        for j in 0usize..4 {
                            cuda_device::thread::__unroll_config::<0>();
                            o[nd][j] = mul_rn_f32(o[nd][j], f[j / 2]);
                        }
                    }
                }
            }

            // The value tile has landed, and every warp is done reading the
            // key tile.
            // SAFETY: waits until none of this thread's groups is pending —
            // its copies of value tile kb; the barrier then publishes every
            // thread's copies before the value product reads VT.
            unsafe { cp_async_wait_group(0) };
            thread::sync_threads();
            if kb + 1 < tiles {
                if end + KEY_TILE_U32 <= hi {
                    stage!(kt, k_src, kb + 1, true);
                } else {
                    stage!(kt, k_src, kb + 1, false);
                }
            }

            // ---- O += P̂·V over the warp's half.
            if live {
                for kk2 in 0..PV_STEPS {
                    cuda_device::thread::__unroll_config::<0>();
                    for nd in 0..DIM_PAIRS_256 {
                        cuda_device::thread::__unroll_config::<0>();
                        // Value row 16·kk2 + v_row, chunk 2·nd + lane/16 of
                        // the warp's half.
                        let at = v_at[nd % LINE_PAIRS]
                            + (4 * (16 * kk2 * ROW_WORDS_256 + LINE * (nd / LINE_PAIRS))) as u32;
                        // SAFETY: the row is below KEY_TILE and the chunk is
                        // whole inside VT, published by the barrier above;
                        // every lane issues the load, then both `mma.sync`
                        // with its own fragments.
                        unsafe {
                            let bf = wmma::ldmatrix_x4_trans_shared_u32(at);
                            o[2 * nd] =
                                wmma::mma_m16n8k16_f32_f16(o[2 * nd], pa[kk2], [bf[0], bf[1]]);
                            o[2 * nd + 1] =
                                wmma::mma_m16n8k16_f32_f16(o[2 * nd + 1], pa[kk2], [bf[2], bf[3]]);
                        }
                    }
                }
            }
            kb += 1;
        }

        // ---- the row sums over the four lanes of a row, then o · (1/l).
        let mut inv = [0.0f32; 2];
        for h in 0usize..2 {
            cuda_device::thread::__unroll_config::<0>();
            let a = add_rn_f32(l[h], warp::shuffle_xor_f32(l[h], 1));
            let s = add_rn_f32(a, warp::shuffle_xor_f32(a, 2));
            inv[h] = 1.0 / s;
        }
        for nd in 0..2 * DIM_PAIRS_256 {
            cuda_device::thread::__unroll_config::<0>();
            for j in 0usize..4 {
                cuda_device::thread::__unroll_config::<0>();
                let h = j / 2;
                let t = t0 + 2 * rs + h;
                if t < rows {
                    let d = HALF_256 * dh + 8 * nd + 2 * t4 + j % 2;
                    let v = if bad[h] {
                        f32::NAN
                    } else {
                        mul_rn_f32(o[nd][j], inv[h])
                    };
                    // SAFETY: t < t_rows, kh·GROUP + g < n_head and d <
                    // HEAD_256: below t_rows·n_head·256 <= y.len() (launch
                    // contract); one owner lane per value.
                    unsafe {
                        *y.get_unchecked_mut((t * n_head + kh * GROUP + g) * HEAD_256 + d) = v;
                    }
                }
            }
        }
    }

    /// [`gqa_prefill_flash_256`] in blocks of [`PACK_4`] query heads, `packs`
    /// of them per key head ([`prefill_256_p`] at `PACK_4`): a block is 16
    /// positions × four heads, a row slice four positions × four heads.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256, 1)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        dynamic_shared = 98304,
        requires = (
            n_keys_buf.len() >= t_rows,
            q.len() >= t_rows * n_kv * packs * 4 * 256,
            kc.len() >= n_kv * ctx * 256,
            vc.len() >= n_kv * ctx * 256,
            y.len() >= t_rows * n_kv * packs * 4 * 256
        )
    )]
    pub fn gqa_prefill_flash_256_p4(
        q: &[f32],
        kc: &[u16],
        vc: &[u16],
        n_keys_buf: &[u32],
        scale: f32,
        n_kv: u32,
        ctx: u32,
        t_rows: u32,
        packs: u32,
        fault: FaultSink,
        y: DisjointSlice<f32>,
    ) {
        // SAFETY: the launch contract is `prefill_256_p`'s at `packs · PACK_4`
        // heads per key head, and the block is THREADS_256 threads with
        // DYN_BYTES_256 bytes of dynamic shared memory.
        unsafe {
            prefill_256_p::<PACK_4, { SLICE_ROWS / PACK_4 }, false, _>(
                q,
                F16Kv { k: kc, v: vc },
                n_keys_buf,
                n_keys_buf,
                scale,
                n_kv,
                ctx,
                t_rows,
                packs as usize,
                fault,
                y,
            )
        };
    }

    /// [`gqa_prefill_flash_256`] in blocks of [`PACK_2`] query heads, `packs`
    /// of them per key head ([`prefill_256_p`] at `PACK_2`): a block is 32
    /// positions × two heads, a row slice eight positions × two heads.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256, 1)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        dynamic_shared = 98304,
        requires = (
            n_keys_buf.len() >= t_rows,
            q.len() >= t_rows * n_kv * packs * 2 * 256,
            kc.len() >= n_kv * ctx * 256,
            vc.len() >= n_kv * ctx * 256,
            y.len() >= t_rows * n_kv * packs * 2 * 256
        )
    )]
    pub fn gqa_prefill_flash_256_p2(
        q: &[f32],
        kc: &[u16],
        vc: &[u16],
        n_keys_buf: &[u32],
        scale: f32,
        n_kv: u32,
        ctx: u32,
        t_rows: u32,
        packs: u32,
        fault: FaultSink,
        y: DisjointSlice<f32>,
    ) {
        // SAFETY: the launch contract is `prefill_256_p`'s at `packs · PACK_2`
        // heads per key head, and the block is THREADS_256 threads with
        // DYN_BYTES_256 bytes of dynamic shared memory.
        unsafe {
            prefill_256_p::<PACK_2, { SLICE_ROWS / PACK_2 }, false, _>(
                q,
                F16Kv { k: kc, v: vc },
                n_keys_buf,
                n_keys_buf,
                scale,
                n_kv,
                ctx,
                t_rows,
                packs as usize,
                fault,
                y,
            )
        };
    }

    /// [`gqa_prefill_flash_256_p2`] with a first key a row ([`prefill_256_p`]
    /// at `PACK_2` with `LOWER`): row `t` attends over keys `[k_first[t],
    /// n_keys[t])`, the module doc's lower-bound paragraph.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256, 1)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        dynamic_shared = 98304,
        requires = (
            n_keys_buf.len() >= t_rows,
            k_first_buf.len() >= t_rows,
            q.len() >= t_rows * n_kv * packs * 2 * 256,
            kc.len() >= n_kv * ctx * 256,
            vc.len() >= n_kv * ctx * 256,
            y.len() >= t_rows * n_kv * packs * 2 * 256
        )
    )]
    pub fn gqa_prefill_flash_256_p2_lower(
        q: &[f32],
        kc: &[u16],
        vc: &[u16],
        n_keys_buf: &[u32],
        k_first_buf: &[u32],
        scale: f32,
        n_kv: u32,
        ctx: u32,
        t_rows: u32,
        packs: u32,
        fault: FaultSink,
        y: DisjointSlice<f32>,
    ) {
        // SAFETY: the launch contract is `prefill_256_p`'s at `packs · PACK_2`
        // heads per key head with `k_first_buf` bound at `t_rows`, and the
        // block is THREADS_256 threads with DYN_BYTES_256 bytes of dynamic
        // shared memory.
        unsafe {
            prefill_256_p::<PACK_2, { SLICE_ROWS / PACK_2 }, true, _>(
                q,
                F16Kv { k: kc, v: vc },
                n_keys_buf,
                k_first_buf,
                scale,
                n_kv,
                ctx,
                t_rows,
                packs as usize,
                fault,
                y,
            )
        };
    }

    /// One query head per block pack, with each row's key interval.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256, 1)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        dynamic_shared = 98304,
        requires = (
            n_keys_buf.len() >= t_rows,
            k_first_buf.len() >= t_rows,
            q.len() >= t_rows * n_kv * packs * 256,
            kc.len() >= n_kv * ctx * 256,
            vc.len() >= n_kv * ctx * 256,
            y.len() >= t_rows * n_kv * packs * 256
        )
    )]
    pub fn gqa_prefill_flash_256_p1_lower(
        q: &[f32],
        kc: &[u16],
        vc: &[u16],
        n_keys_buf: &[u32],
        k_first_buf: &[u32],
        scale: f32,
        n_kv: u32,
        ctx: u32,
        t_rows: u32,
        packs: u32,
        fault: FaultSink,
        y: DisjointSlice<f32>,
    ) {
        // SAFETY: the launch contract supplies every plane and count at
        // PACK = 1, THREADS_256 threads and DYN_BYTES_256 shared bytes.
        unsafe {
            prefill_256_p::<1, SLICE_ROWS, true, _>(
                q,
                F16Kv { k: kc, v: vc },
                n_keys_buf,
                k_first_buf,
                scale,
                n_kv,
                ctx,
                t_rows,
                packs as usize,
                fault,
                y,
            )
        };
    }

    /// [`gqa_prefill_flash_256`] over the Q8_0 cache ([`prefill_256_p`] at
    /// `GROUP` over [`Q8Kv`]): the eight-head layout, the two-step staging and
    /// the in-place widening of the module doc's `_q8` paragraph.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256, 1)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        dynamic_shared = 100352,
        requires = (
            n_keys_buf.len() >= t_rows,
            q.len() >= t_rows * n_kv * 8 * 256,
            kq.len() >= n_kv * ctx * 64,
            kd.len() >= n_kv * ctx * 8,
            vq.len() >= n_kv * ctx * 64,
            vd.len() >= n_kv * ctx * 8,
            y.len() >= t_rows * n_kv * 8 * 256
        )
    )]
    pub fn gqa_prefill_flash_256_q8(
        q: &[f32],
        kq: &[u32],
        kd: &[u16],
        vq: &[u32],
        vd: &[u16],
        n_keys_buf: &[u32],
        scale: f32,
        n_kv: u32,
        ctx: u32,
        t_rows: u32,
        fault: FaultSink,
        y: DisjointSlice<f32>,
    ) {
        // SAFETY: the launch contract is `prefill_256_p`'s at `GROUP` heads
        // per key head over the four Q8_0 planes, and the block is
        // THREADS_256 threads with DYN_BYTES_256_Q8 bytes of dynamic shared
        // memory.
        unsafe {
            prefill_256_p::<GROUP, { SLICE_ROWS / GROUP }, false, _>(
                q,
                Q8Kv { kq, kd, vq, vd },
                n_keys_buf,
                n_keys_buf,
                scale,
                n_kv,
                ctx,
                t_rows,
                1,
                fault,
                y,
            )
        };
    }

    /// [`gqa_prefill_flash`] over the Q8_0 cache: the module doc's `_q8`
    /// paragraph at heads of [`HEAD`] values — the walk, the scores, the
    /// softmax and the value products [`gqa_prefill_flash`]'s, each tile's
    /// staging the two-step form (the raw code words into the f16 tile's low
    /// halves and the scales into their static tiles by `cp.async`, then
    /// [`dequant_row`]'s in-place widening, a key past the block's largest
    /// count [`zero_row`]'s four zero words).
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[allow(
        clippy::needless_range_loop,
        reason = "a range `for` is the loop the unroller recognizes; an iterator loop keeps its array in a local depot"
    )]
    #[kernel]
    #[launch_bounds(128, 2)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        requires = (
            n_keys_buf.len() >= t_rows,
            q.len() >= t_rows * n_kv * 8 * 128,
            kq.len() >= n_kv * ctx * 32,
            kd.len() >= n_kv * ctx * 4,
            vq.len() >= n_kv * ctx * 32,
            vd.len() >= n_kv * ctx * 4,
            y.len() >= t_rows * n_kv * 8 * 128
        )
    )]
    pub fn gqa_prefill_flash_q8(
        q: &[f32],
        kq: &[u32],
        kd: &[u16],
        vq: &[u32],
        vd: &[u16],
        n_keys_buf: &[u32],
        scale: f32,
        n_kv: u32,
        ctx: u32,
        t_rows: u32,
        fault: FaultSink,
        mut y: DisjointSlice<f32>,
    ) {
        // 128-byte aligned: the layout's sectors and halves are aligned only
        // if the tile base is.
        static mut KT: SharedArray<u32, TILE_WORDS, 128> = SharedArray::UNINIT;
        static mut VT: SharedArray<u32, TILE_WORDS, 128> = SharedArray::UNINIT;
        // 8-byte aligned: the scales' `cp.async.ca` copies.
        static mut SK: SharedArray<u16, { KEY_TILE * HEAD / 32 }, 8> = SharedArray::UNINIT;
        static mut SV: SharedArray<u16, { KEY_TILE * HEAD / 32 }, 8> = SharedArray::UNINIT;
        // Code 16-byte chunks a staged row takes, and a thread's count of a
        // tile's.
        const ROW_CCHUNKS: usize = HEAD / 16;
        const TILE_CCHUNKS: usize = KEY_TILE * ROW_CCHUNKS / THREADS;
        const _: () = assert!(TILE_CCHUNKS * THREADS == KEY_TILE * ROW_CCHUNKS);
        // The rows of a tile this block's warps widen: KEY_TILE/WARPS a warp.
        const WID_ROWS: usize = KEY_TILE / WARPS;

        let b = thread::blockIdx_x() as usize;
        let nkv = n_kv as usize;
        let rows = t_rows as usize;
        let n_tiles = rows.div_ceil(POSITIONS);
        if b >= n_tiles * nkv {
            return; // block-uniform
        }
        let qt = n_tiles - 1 - b / nkv;
        let kh = b % nkv;
        let tid = thread::threadIdx_x() as usize;
        let lane = warp::lane_id() as usize;
        let wid = tid / 32;
        let g = lane / 4;
        let t4 = lane % 4;
        let n_head = nkv * GROUP;
        let t0 = qt * POSITIONS;

        // SAFETY: block-shared, TILE_WORDS words each and KEY_TILE·HEAD/32
        // u16; every write is ordered before its reads by a block barrier
        // (the cp.async copies by the wait before it).
        let (kt, vt, sk, sv) = unsafe {
            (
                SharedArray::as_raw_mut_ptr(&raw mut KT),
                SharedArray::as_raw_mut_ptr(&raw mut VT),
                SharedArray::as_raw_mut_ptr(&raw mut SK),
                SharedArray::as_raw_mut_ptr(&raw mut SV),
            )
        };

        // Every row's count as the walk uses it: 0 for a row past `t_rows`
        // and for a refused one; the block's largest bounds the keys loaded.
        // Keys and counts are u32: the host keeps `ctx + KEY_TILE` in range.
        let mut hi = 0u32;
        let mut cnt = [0u32; 2];
        let mut bad = [false; 2];
        for lp in 0..POSITIONS {
            cuda_device::thread::__unroll_config::<0>();
            let t = t0 + lp;
            let (c, refused) = if t < rows {
                // SAFETY: t < t_rows <= n_keys_buf.len() (launch contract).
                let c = unsafe { *n_keys_buf.get_unchecked(t) };
                if c == 0 || c > ctx {
                    (0, true)
                } else {
                    (c, false)
                }
            } else {
                (0, false)
            };
            hi = hi.max(c);
            if lp / 2 == wid {
                cnt[lp % 2] = c;
                bad[lp % 2] = refused;
            }
        }
        if lane == 0 && (bad[0] || bad[1]) {
            fault.raise(FaultSite::KeyCount);
        }
        let warp_hi = cnt[0].max(cnt[1]);
        let warp_lo = cnt[0].min(cnt[1]);

        // The block's query rows, rounded to f16 pairs, into the key tile:
        // staged row `lp·GROUP + g` is position `t0 + lp`'s head `kh·GROUP +
        // g`, which puts position `2·w + h` at rows `16·w + 8·h ..` — warp
        // `w`'s fragment. A row past `t_rows` is zero. Thread `tid` stages
        // word `tid % Q_PAIRS` of every row its passes cover, and every load
        // is issued before the first conversion.
        let wd = tid % Q_PAIRS;
        let r0 = tid / Q_PAIRS;
        // u64 words of q: this thread's word of position t0's head kh·GROUP +
        // r0, and the step from one position to the next.
        let q_first = (t0 * n_head + kh * GROUP + r0) * Q_PAIRS + wd;
        let q_pos = n_head * Q_PAIRS;
        let q64 = q.as_ptr().cast::<u64>();
        let mut raw = [0u64; Q_WORDS];
        for e in 0..Q_WORDS {
            cuda_device::thread::__unroll_config::<0>();
            let t = t0 + e / Q_POS_PASSES;
            if t < rows {
                let at = q_first
                    + (e / Q_POS_PASSES) * q_pos
                    + Q_ROW_STEP * (e % Q_POS_PASSES) * Q_PAIRS;
                // SAFETY: t < t_rows and the head kh·GROUP + Q_ROW_STEP·(e %
                // Q_POS_PASSES) + r0 < n_head, so the row lies inside q
                // (launch contract); a row is HEAD f32 = HEAD/2 u64 and q
                // starts 8-byte aligned (host-checked), so the u64 read of
                // values 2·wd, 2·wd + 1 is aligned.
                raw[e] = unsafe { *q64.add(at) };
            }
        }
        // The staged word of each pass of the first position; a later
        // position's is it plus whole groups of rows (layout_holds).
        let mut q_at = [0usize; Q_POS_PASSES];
        for j in 0..Q_POS_PASSES {
            cuda_device::thread::__unroll_config::<0>();
            q_at[j] = staged_word(Q_ROW_STEP * j + r0, wd);
        }
        for e in 0..Q_WORDS {
            cuda_device::thread::__unroll_config::<0>();
            let pair = f32x2_to_f16x2_bits(
                f32::from_bits(raw[e] as u32),
                f32::from_bits((raw[e] >> 32) as u32),
            );
            // SAFETY: row Q_ROW_STEP·e + r0 < Q_ROWS <= KEY_TILE and wd <
            // ROW_WORDS: inside KT; one owner per word.
            unsafe {
                *kt.add(q_at[e % Q_POS_PASSES] + (e / Q_POS_PASSES) * GROUP * ROW_WORDS) = pair
            };
        }
        thread::sync_threads();

        // This warp's 16 query rows as A fragments, one per k16 step.
        let mut qa = [[0u32; 4]; QK_STEPS];
        for kk in 0..QK_STEPS {
            cuda_device::thread::__unroll_config::<0>();
            let row = wid * 16 + lane % 16;
            // SAFETY: row < Q_ROWS and words 8·kk + 4·(lane/16) .. + 4 <=
            // HEAD/2 are one whole chunk, which staged_word keeps whole and
            // 16-byte aligned inside KT, published by the barrier above;
            // every lane issues the load.
            qa[kk] = unsafe {
                let p = kt.add(staged_word(row, 8 * kk + 4 * (lane / 16)));
                wmma::ldmatrix_x4_shared_u32(shared::cvta_generic_to_shared_u32(
                    p.cast_const().cast::<u8>(),
                ))
            };
        }
        // Every warp has its fragments before the first key tile lands.
        thread::sync_threads();

        // The q8 planes' first rows of this key head, codes and scales a
        // side (the scales' u32 half-words: a row's HEAD/32 u16).
        let rowq = kh * ctx as usize;
        let kq_src = kq.as_ptr().wrapping_add(rowq * (HEAD / 4));
        let kd_src = kd
            .as_ptr()
            .cast::<u32>()
            .wrapping_add(rowq * (HEAD / 32) / 2);
        let vq_src = vq.as_ptr().wrapping_add(rowq * (HEAD / 4));
        let vd_src = vd
            .as_ptr()
            .cast::<u32>()
            .wrapping_add(rowq * (HEAD / 32) / 2);
        let tiles = hi.div_ceil(KEY_TILE_U32);
        // Stage tile `kb` of one side's q8 planes: the code words as
        // 16-byte copies into the f16 tile's low halves (four chunks a
        // thread), the scales as one 8-byte copy a row, a key at or past
        // `hi` zeroed and never read, committed as one group. `full` (a
        // literal) is the tile lying below `hi`.
        macro_rules! stage_q8 {
            ($tile:expr, $sk:expr, $qs:expr, $ds:expr, $kb:expr, $full:literal) => {{
                let (tile, sk, qs, ds, kb): (*mut u32, *mut u16, *const u32, *const u32, u32) =
                    ($tile, $sk, $qs, $ds, $kb);
                for i in 0..TILE_CCHUNKS {
                    cuda_device::thread::__unroll_config::<0>();
                    let e = tid + THREADS * i;
                    let row = e / ROW_CCHUNKS;
                    let col = e - row * ROW_CCHUNKS;
                    // SAFETY: row < KEY_TILE and words 4·col .. + 4 are one
                    // whole chunk of the row's low half: the 16 destination
                    // bytes are inside the tile and 16-byte aligned.
                    let d = unsafe { tile.add(row * ROW_WORDS + 4 * col) };
                    if $full || kb * KEY_TILE_U32 + (row as u32) < hi {
                        // SAFETY: the key is below hi <= ctx, so its row is
                        // inside this key head's codes plane (launch
                        // contract); rows are HEAD/4 words and the plane
                        // starts 16-byte aligned (host-checked), so the
                        // source is 16-byte aligned.
                        unsafe {
                            cp_async_cg_16(
                                d,
                                qs.wrapping_add(
                                    (kb as usize * KEY_TILE + row) * (HEAD / 4) + 4 * col,
                                ),
                            )
                        };
                    } else {
                        // SAFETY: `d` is 16 bytes inside the tile and
                        // 16-byte aligned, this thread's chunk alone, and no
                        // warp reads the tile before the barrier after this
                        // stage's wait (the widening writes the dead row's
                        // zeros over it).
                        unsafe { *d.cast::<U32x4>() = U32x4::splat(0) };
                    }
                }
                if tid < KEY_TILE {
                    // SAFETY: the row's HEAD/32 u16 are one aligned 8-byte
                    // half of this thread's row of the scales tile alone.
                    let d = unsafe { sk.cast::<u32>().add(2 * tid) };
                    if $full || kb * KEY_TILE_U32 + (tid as u32) < hi {
                        // SAFETY: the key is below hi <= ctx: its scales are
                        // inside the plane (launch contract), one 8-byte
                        // aligned pair of u32.
                        unsafe {
                            cp_async_ca_8(d, ds.wrapping_add((kb as usize * KEY_TILE + tid) * 2))
                        };
                    } else {
                        // SAFETY: this thread's chunk alone, read by no one
                        // (the dead row widens to zeros).
                        unsafe { *d.cast::<u64>() = 0 };
                    }
                }
                // SAFETY: closes this thread's copies above as one group.
                unsafe { cp_async_commit_group() };
            }};
        }
        if tiles > 0 {
            if KEY_TILE_U32 <= hi {
                stage_q8!(kt, sk, kq_src, kd_src, 0, true);
            } else {
                stage_q8!(kt, sk, kq_src, kd_src, 0, false);
            }
        }

        // This lane's `ldmatrix` addresses in the first sixteen rows and the
        // first half-row, one per chunk pair `j`: the scores' key row and
        // chunk `2·j + (lane/8) % 2`, the values' row and chunk `2·j +
        // lane/16`. Every other address of the walk is one of these plus a
        // constant (`layout_holds`).
        let k_row = lane % 8 + 8 * (lane / 16);
        let v_row = lane % 8 + 8 * ((lane / 8) % 2);
        let mut k_at = [0u32; HALF_PAIRS];
        let mut v_at = [0u32; HALF_PAIRS];
        for j in 0..HALF_PAIRS {
            cuda_device::thread::__unroll_config::<0>();
            // SAFETY: k_row, v_row < 16 <= KEY_TILE and words 8·j + 4 .. + 4
            // <= HALF_ROW: inside the tiles; only the addresses are formed here.
            unsafe {
                k_at[j] = shared::cvta_generic_to_shared_u32(
                    kt.add(staged_word(k_row, 8 * j + 4 * ((lane / 8) % 2)))
                        .cast_const()
                        .cast::<u8>(),
                );
                v_at[j] = shared::cvta_generic_to_shared_u32(
                    vt.add(staged_word(v_row, 8 * j + 4 * (lane / 16)))
                        .cast_const()
                        .cast::<u8>(),
                );
            }
        }

        let t4k = 2 * t4 as u32;
        let mut m = [f32::NEG_INFINITY; 2];
        let mut l = [0.0f32; 2];
        let mut o = [[0.0f32; 4]; 2 * DIM_PAIRS];
        let mut kb = 0u32;
        while kb < tiles {
            // One past the tile's last key.
            let end = KEY_TILE_U32 * (kb + 1);
            // The raw codes and scales of key tile kb have landed, and every
            // warp is done reading the value tile of kb − 1.
            // SAFETY: waits for this thread's outstanding groups; the barrier
            // then publishes every thread's copies.
            unsafe { cp_async_wait_group(0) };
            thread::sync_threads();

            // The key tile's in-place widening: warp `wid` widens its own
            // rows (`dequant_row` reads before its warp sync and writes
            // after), a key at or past `hi` four zero words.
            let rows0 = wid * WID_ROWS;
            for r in 0..WID_ROWS {
                let row = rows0 + r;
                let base = row * ROW_WORDS;
                if kb * KEY_TILE_U32 + (row as u32) < hi {
                    // SAFETY: the row is this warp's alone, its raw words and
                    // scales published by the barrier above.
                    dequant_row!(1, kt, base, sk.wrapping_add(row * 4), row & 7, lane);
                } else {
                    // SAFETY: the row is this warp's alone in the tile.
                    zero_row!(1, kt, base, row & 7, lane);
                }
            }
            thread::sync_threads();

            // The raw codes and scales of value tile kb, issued to overlap
            // the scores.
            if end <= hi {
                stage_q8!(vt, sv, vq_src, vd_src, kb, true);
            } else {
                stage_q8!(vt, sv, vq_src, vd_src, kb, false);
            }
            let live = KEY_TILE_U32 * kb < warp_hi;

            // ---- S = Q·Kᵀ, then the online softmax into f16 weights.
            let mut pa = [[0u32; 4]; PV_STEPS];
            let mut f = [1.0f32; 2];
            if live {
                let mut sc = [[0.0f32; 4]; KEY_NT];
                for kk in 0..QK_STEPS {
                    cuda_device::thread::__unroll_config::<0>();
                    for nj in 0..KEY_NT / 2 {
                        cuda_device::thread::__unroll_config::<0>();
                        // Key row 16·nj + k_row, chunk 2·kk + (lane/8) % 2.
                        let at = k_at[kk % HALF_PAIRS]
                            + (4 * (16 * nj * ROW_WORDS + HALF_ROW * (kk / HALF_PAIRS))) as u32;
                        // SAFETY: the row is below KEY_TILE and the chunk is
                        // whole inside KT (staged_word, layout_holds),
                        // published by the barrier above; every lane issues
                        // the load, then both `mma.sync` with its own
                        // fragments.
                        unsafe {
                            let bf = wmma::ldmatrix_x4_shared_u32(at);
                            sc[2 * nj] =
                                wmma::mma_m16n8k16_f32_f16(sc[2 * nj], qa[kk], [bf[0], bf[1]]);
                            sc[2 * nj + 1] =
                                wmma::mma_m16n8k16_f32_f16(sc[2 * nj + 1], qa[kk], [bf[2], bf[3]]);
                        }
                    }
                }

                prefill_softmax! {
                    unmasked: end <= warp_lo,
                    key0: KEY_TILE_U32 * kb + t4k,
                    tiles: (KEY_NT, PV_STEPS),
                    scores: sc,
                    counts: cnt,
                    scale: scale,
                    state: (m, l, f),
                    weights: pa,
                }

                // The accumulators times f, unless no lane of the warp has a
                // factor other than 1 (the vote keeps the branch uniform).
                if warp::any(f[0] != 1.0 || f[1] != 1.0) {
                    for nd in 0..2 * DIM_PAIRS {
                        cuda_device::thread::__unroll_config::<0>();
                        for j in 0usize..4 {
                            cuda_device::thread::__unroll_config::<0>();
                            o[nd][j] = mul_rn_f32(o[nd][j], f[j / 2]);
                        }
                    }
                }
            }

            // The raw codes and scales of value tile kb have landed, and
            // every warp is done reading the key tile.
            // SAFETY: waits until none of this thread's groups is pending —
            // its copies of value tile kb; the barrier then publishes every
            // thread's copies before the widening reads VT's raw words.
            unsafe { cp_async_wait_group(0) };
            thread::sync_threads();

            // The value tile's in-place widening, the key tile's form.
            let rows0 = wid * WID_ROWS;
            for r in 0..WID_ROWS {
                let row = rows0 + r;
                let base = row * ROW_WORDS;
                if kb * KEY_TILE_U32 + (row as u32) < hi {
                    // SAFETY: as the key tile's widening.
                    dequant_row!(1, vt, base, sv.wrapping_add(row * 4), row & 7, lane);
                } else {
                    // SAFETY: as the key tile's zero row.
                    zero_row!(1, vt, base, row & 7, lane);
                }
            }
            thread::sync_threads();
            if kb + 1 < tiles {
                if end + KEY_TILE_U32 <= hi {
                    stage_q8!(kt, sk, kq_src, kd_src, kb + 1, true);
                } else {
                    stage_q8!(kt, sk, kq_src, kd_src, kb + 1, false);
                }
            }

            // ---- O += P̂·V.
            if live {
                for kk2 in 0..PV_STEPS {
                    cuda_device::thread::__unroll_config::<0>();
                    for nd in 0..DIM_PAIRS {
                        cuda_device::thread::__unroll_config::<0>();
                        // Value row 16·kk2 + v_row, chunk 2·nd + lane/16.
                        let at = v_at[nd % HALF_PAIRS]
                            + (4 * (16 * kk2 * ROW_WORDS + HALF_ROW * (nd / HALF_PAIRS))) as u32;
                        // SAFETY: the row is below KEY_TILE and the chunk is
                        // whole inside VT (staged_word, layout_holds),
                        // published by the barrier above; every lane issues
                        // the load, then both `mma.sync` with its own
                        // fragments.
                        unsafe {
                            let bf = wmma::ldmatrix_x4_trans_shared_u32(at);
                            o[2 * nd] =
                                wmma::mma_m16n8k16_f32_f16(o[2 * nd], pa[kk2], [bf[0], bf[1]]);
                            o[2 * nd + 1] =
                                wmma::mma_m16n8k16_f32_f16(o[2 * nd + 1], pa[kk2], [bf[2], bf[3]]);
                        }
                    }
                }
            }
            kb += 1;
        }

        // ---- the row sums over the four lanes of a row, then o · (1/l).
        let mut inv = [0.0f32; 2];
        for h in 0usize..2 {
            cuda_device::thread::__unroll_config::<0>();
            let a = add_rn_f32(l[h], warp::shuffle_xor_f32(l[h], 1));
            let s = add_rn_f32(a, warp::shuffle_xor_f32(a, 2));
            inv[h] = 1.0 / s;
        }
        for nd in 0..2 * DIM_PAIRS {
            cuda_device::thread::__unroll_config::<0>();
            for j in 0usize..4 {
                cuda_device::thread::__unroll_config::<0>();
                let h = j / 2;
                let t = t0 + 2 * wid + h;
                if t < rows {
                    let d = 8 * nd + 2 * t4 + j % 2;
                    let v = if bad[h] {
                        f32::NAN
                    } else {
                        mul_rn_f32(o[nd][j], inv[h])
                    };
                    // SAFETY: t < t_rows and kh·GROUP + g < n_head, d < HEAD:
                    // the index is below t_rows·n_head·HEAD <= y.len() (launch
                    // contract); one owner lane per value.
                    unsafe {
                        *y.get_unchecked_mut((t * n_head + kh * GROUP + g) * HEAD + d) = v;
                    }
                }
            }
        }
    }
}

/// [`FlashGqaPrefill::enqueue`]'s arguments: `t` rows of `n_head` query heads
/// (roped, unscaled, token-major), the layer's two planes of `n_kv · ctx`
/// rows, each row's live key count on the device (`t` of them), the output
/// rows (token-major) and the fault sink a refused count raises on.
pub struct GqaPrefillArgs<'a> {
    pub q: &'a DeviceBuffer<f32>,
    pub kc: &'a DeviceBuffer<u16>,
    pub vc: &'a DeviceBuffer<u16>,
    pub n_keys: &'a DeviceBuffer<u32>,
    pub scale: f32,
    pub n_head: usize,
    pub n_kv: usize,
    pub ctx: usize,
    pub t: usize,
    pub fault: FaultSink,
    pub y: &'a mut DeviceBuffer<f32>,
}

/// Validate a packed f16 prefill launch and return packs per key head.
pub(crate) fn checked_packed(
    what: &'static str,
    a: &GqaPrefillArgs<'_>,
    (head, tile, pack): (usize, usize, usize),
) -> Result<usize, GpuError> {
    let (n_head, n_kv, ctx, t) = (a.n_head, a.n_kv, a.ctx, a.t);
    if n_kv == 0 || ctx == 0 || t == 0 {
        return Err(GpuError::shape(
            what,
            format!("need n_kv, ctx and t >= 1, got n_kv={n_kv} ctx={ctx} t={t}"),
        ));
    }
    if n_head == 0 || !n_head.is_multiple_of(n_kv * pack) {
        return Err(GpuError::shape(
            what,
            format!(
                "the kernel packs {pack} query heads a block, so the group must be a \
                 multiple of {pack}; got {n_head} heads over {n_kv}"
            ),
        ));
    }
    let packs = n_head / (n_kv * pack);
    let lens = [
        ("q", a.q.len(), t * n_head * head),
        ("kc", a.kc.len(), n_kv * ctx * head),
        ("vc", a.vc.len(), n_kv * ctx * head),
        ("n_keys", a.n_keys.len(), t),
        ("y", a.y.len(), t * n_head * head),
    ];
    if let Some((name, got, need)) = lens.iter().find(|(_, got, need)| got < need) {
        return Err(GpuError::shape(
            what,
            format!("{name}.len() {got} < {need}"),
        ));
    }
    // A window at another offset is a misaligned access on the card.
    let starts = [
        ("q", a.q.cu_deviceptr(), 8u64),
        ("kc", a.kc.cu_deviceptr(), 16),
        ("vc", a.vc.cu_deviceptr(), 16),
    ];
    if let Some((name, at, align)) = starts
        .iter()
        .find(|(_, at, align)| !at.is_multiple_of(*align))
    {
        return Err(GpuError::shape(
            what,
            format!("{name} at {at:#x} is not {align}-byte aligned"),
        ));
    }
    // The walk counts keys in u32 up to one tile past the largest count.
    if ctx > u32::MAX as usize - tile {
        return Err(GpuError::shape(
            what,
            format!("ctx = {ctx}: the key walk counts to ctx + {tile} in u32"),
        ));
    }
    Ok(packs)
}

/// [`GqaPrefillArgs`]' q8_0 form: the same query rows, counts and output,
/// the cache the four planes of the quantizing appends ([`q8_plane_lens`])
/// — per side the codes and the scales — served by the `_q8` entries.
pub struct GqaPrefillQ8Args<'a> {
    pub q: &'a DeviceBuffer<f32>,
    pub kq: &'a DeviceBuffer<u32>,
    pub kd: &'a DeviceBuffer<u16>,
    pub vq: &'a DeviceBuffer<u32>,
    pub vd: &'a DeviceBuffer<u16>,
    pub n_keys: &'a DeviceBuffer<u32>,
    pub scale: f32,
    pub n_head: usize,
    pub n_kv: usize,
    pub ctx: usize,
    pub t: usize,
    pub fault: FaultSink,
    pub y: &'a mut DeviceBuffer<f32>,
}

/// The sub-group buffers [`FlashGqaPrefill::enqueue_split`] moves a group's
/// query rows into and its output rows out of: `rows · n_kv · GROUP · HEAD`
/// f32 each, one plain launch's worth. Load-time only — the split path takes
/// no allocation at a step; a launch of `t` rows touches only the first
/// `t · n_kv · GROUP · HEAD` values of each.
pub struct SplitScratch {
    q: DeviceBuffer<f32>,
    y: DeviceBuffer<f32>,
}

impl SplitScratch {
    /// Allocate the scratch of `rows` rows over `n_kv` key heads.
    /// Load-time only.
    pub fn new(stream: &CudaStream, rows: usize, n_kv: usize) -> Result<Self, GpuError> {
        let len = rows * n_kv * GROUP * HEAD;
        Ok(SplitScratch {
            q: DeviceBuffer::<f32>::zeroed(stream, len)?,
            y: DeviceBuffer::<f32>::zeroed(stream, len)?,
        })
    }

    /// The device bytes both sub-group buffers hold.
    pub fn num_bytes(&self) -> usize {
        self.q.num_bytes() + self.y.num_bytes()
    }
}

/// The loaded module. Owns no stream: each enqueue takes the engine stream.
pub struct FlashGqaPrefill {
    module: flash_gqa_prefill_kernels::LoadedModule,
}

impl FlashGqaPrefill {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<FlashGqaPrefill, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; the launcher checks its launch contract.
        let module = unsafe { crate::shared_module!(flash_gqa_prefill_kernels, ctx)? };
        Ok(FlashGqaPrefill { module })
    }

    /// Enqueue `t` rows' attention: one launch of `blocks_for(t, n_kv)`
    /// blocks. A shape the kernel is not built for (`n_head ≠ n_kv ·
    /// GROUP`), an empty launch, a short buffer, or a `q` not 8-byte aligned
    /// or `kc`/`vc` not 16-byte aligned (the query staging reads u64 words,
    /// the tiles copy 16-byte pieces) is refused by name before any launch.
    /// Asynchronous, allocation-free, capturable.
    pub fn enqueue(&self, stream: &CudaStream, args: GqaPrefillArgs<'_>) -> Result<(), GpuError> {
        let what = "flash_gqa_prefill::enqueue";
        let GqaPrefillArgs {
            q,
            kc,
            vc,
            n_keys,
            scale,
            n_head,
            n_kv,
            ctx,
            t,
            fault,
            y,
        } = args;
        if n_kv == 0 || ctx == 0 || t == 0 {
            return Err(GpuError::shape(
                what,
                format!("need n_kv, ctx and t >= 1, got n_kv={n_kv} ctx={ctx} t={t}"),
            ));
        }
        if n_head != n_kv * GROUP {
            return Err(GpuError::shape(
                what,
                format!(
                    "the kernel groups {GROUP} query heads per key head; got {n_head} over {n_kv}"
                ),
            ));
        }
        let lens = [
            ("q", q.len(), t * n_head * HEAD),
            ("kc", kc.len(), n_kv * ctx * HEAD),
            ("vc", vc.len(), n_kv * ctx * HEAD),
            ("n_keys", n_keys.len(), t),
            ("y", y.len(), t * n_head * HEAD),
        ];
        if let Some((name, got, need)) = lens.iter().find(|(_, got, need)| got < need) {
            return Err(GpuError::shape(
                what,
                format!("{name}.len() {got} < {need}"),
            ));
        }
        // A window at another offset is a misaligned access on the card.
        let starts = [
            ("q", q.cu_deviceptr(), 8u64),
            ("kc", kc.cu_deviceptr(), 16),
            ("vc", vc.cu_deviceptr(), 16),
        ];
        if let Some((name, at, align)) = starts
            .iter()
            .find(|(_, at, align)| !at.is_multiple_of(*align))
        {
            return Err(GpuError::shape(
                what,
                format!("{name} at {at:#x} is not {align}-byte aligned"),
            ));
        }
        // The walk counts keys in u32 up to one tile past the largest count.
        if ctx > u32::MAX as usize - KEY_TILE {
            return Err(GpuError::shape(
                what,
                format!("ctx = {ctx}: the key walk counts to ctx + {KEY_TILE} in u32"),
            ));
        }
        let grid = launch_u32(what, "grid", blocks_for(t, n_kv))?;
        let n_kv = launch_u32(what, "n_kv", n_kv)?;
        let ctx = launch_u32(what, "ctx", ctx)?;
        let t = launch_u32(what, "t", t)?;
        let prep =
            self.module
                .prepare_gqa_prefill_flash(LaunchConfig1D::new(grid, THREADS_U32, 0))?;
        self.module.gqa_prefill_flash(
            stream, &prep, q, kc, vc, n_keys, scale, n_kv, ctx, t, fault, y,
        )?;
        Ok(())
    }

    /// [`FlashGqaPrefill::enqueue`]'s attention for a group of any positive
    /// multiple of [`GROUP`] query heads a key head: sub-group `s` of key head
    /// `kv` is query heads `kv·group + GROUP·s .. kv·group + GROUP·(s+1)`, an
    /// independent group-of-eight problem over the same cache planes. Its
    /// query rows are copied into `scratch.q`, run through one plain
    /// [`FlashGqaPrefill::enqueue`] at `n_kv · GROUP` heads, and the output
    /// rows copied back into `args.y` — a byte copy each way, so every row's
    /// bits are the group-of-eight kernel's for that row and head over the
    /// same planes. `group / GROUP` launches and twice as many 2-D copies,
    /// sub-groups ascending; at `group == GROUP` one plain launch and no
    /// copy. Asynchronous, allocation-free, capturable.
    ///
    /// Refused by name, nothing enqueued, when `n_kv`, `ctx` or `t` is zero,
    /// the group is not a positive multiple of [`GROUP`], `q` or `y` holds
    /// fewer than `t · n_head · HEAD` values, `kc`, `vc` or `n_keys` fewer
    /// than the launch reads, or either of `scratch`'s buffers fewer than
    /// `t · n_kv · GROUP · HEAD`.
    pub fn enqueue_split(
        &self,
        stream: &CudaStream,
        args: GqaPrefillArgs<'_>,
        scratch: &mut SplitScratch,
    ) -> Result<(), GpuError> {
        let what = "flash_gqa_prefill::enqueue_split";
        let GqaPrefillArgs {
            q,
            kc,
            vc,
            n_keys,
            scale,
            n_head,
            n_kv,
            ctx,
            t,
            fault,
            y,
        } = args;
        if n_kv == 0 || t == 0 || ctx == 0 {
            return Err(GpuError::shape(
                what,
                format!("need n_kv, ctx and t >= 1, got n_kv={n_kv} ctx={ctx} t={t}"),
            ));
        }
        if n_head % n_kv != 0 || n_head / n_kv == 0 || (n_head / n_kv) % GROUP != 0 {
            return Err(GpuError::shape(
                what,
                format!(
                    "a group of {n_head} over {n_kv} key heads is not a positive multiple of \
                     {GROUP}"
                ),
            ));
        }
        let group = n_head / n_kv;
        let lens = [
            ("q", q.len(), t * n_head * HEAD),
            ("y", y.len(), t * n_head * HEAD),
            ("kc", kc.len(), n_kv * ctx * HEAD),
            ("vc", vc.len(), n_kv * ctx * HEAD),
            ("n_keys", n_keys.len(), t),
        ];
        if let Some((name, got, need)) = lens.iter().find(|(_, got, need)| got < need) {
            return Err(GpuError::shape(
                what,
                format!("{name}.len() {got} < {need}"),
            ));
        }
        let need = t * n_kv * GROUP * HEAD;
        if scratch.q.len() < need || scratch.y.len() < need {
            return Err(GpuError::shape(
                what,
                format!(
                    "scratch buffers of {} (q) and {} (y) values, the launch needs {need} of each",
                    scratch.q.len(),
                    scratch.y.len()
                ),
            ));
        }
        if group == GROUP {
            return self.enqueue(
                stream,
                GqaPrefillArgs {
                    q,
                    kc,
                    vc,
                    n_keys,
                    scale,
                    n_head,
                    n_kv,
                    ctx,
                    t,
                    fault,
                    y,
                },
            );
        }
        let f = size_of::<f32>();
        let (width, rows, stride) = (GROUP * HEAD * f, t * n_kv, group * HEAD * f);
        for s in 0..group / GROUP {
            let at = s * width;
            dtod_2d(
                stream,
                (q, at),
                (&mut scratch.q, 0),
                Pitched {
                    width,
                    rows,
                    dev_pitch: stride,
                    host_pitch: width,
                },
            )?;
            self.enqueue(
                stream,
                GqaPrefillArgs {
                    q: &scratch.q,
                    kc,
                    vc,
                    n_keys,
                    scale,
                    n_head: n_kv * GROUP,
                    n_kv,
                    ctx,
                    t,
                    fault,
                    y: &mut scratch.y,
                },
            )?;
            dtod_2d(
                stream,
                (&scratch.y, 0),
                (y, at),
                Pitched {
                    width,
                    rows,
                    dev_pitch: width,
                    host_pitch: stride,
                },
            )?;
        }
        Ok(())
    }

    /// [`FlashGqaPrefill::enqueue`] over heads of [`HEAD_256`] values:
    /// `args.q` and `args.y` hold `t · n_head` rows of 256 f32, the planes
    /// `n_kv · ctx` rows of 256 f16. One launch of `blocks_for(t, n_kv)`
    /// blocks of 256 threads with its dynamic shared
    /// memory (98,304 bytes); the same refusals by name before any launch. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_256(
        &self,
        stream: &CudaStream,
        args: GqaPrefillArgs<'_>,
    ) -> Result<(), GpuError> {
        let what = "flash_gqa_prefill::enqueue_256";
        let GqaPrefillArgs {
            q,
            kc,
            vc,
            n_keys,
            scale,
            n_head,
            n_kv,
            ctx,
            t,
            fault,
            y,
        } = args;
        if n_kv == 0 || ctx == 0 || t == 0 {
            return Err(GpuError::shape(
                what,
                format!("need n_kv, ctx and t >= 1, got n_kv={n_kv} ctx={ctx} t={t}"),
            ));
        }
        if n_head != n_kv * GROUP {
            return Err(GpuError::shape(
                what,
                format!(
                    "the kernel groups {GROUP} query heads per key head; got {n_head} over {n_kv}"
                ),
            ));
        }
        let lens = [
            ("q", q.len(), t * n_head * HEAD_256),
            ("kc", kc.len(), n_kv * ctx * HEAD_256),
            ("vc", vc.len(), n_kv * ctx * HEAD_256),
            ("n_keys", n_keys.len(), t),
            ("y", y.len(), t * n_head * HEAD_256),
        ];
        if let Some((name, got, need)) = lens.iter().find(|(_, got, need)| got < need) {
            return Err(GpuError::shape(
                what,
                format!("{name}.len() {got} < {need}"),
            ));
        }
        // A window at another offset is a misaligned access on the card.
        let starts = [
            ("q", q.cu_deviceptr(), 8u64),
            ("kc", kc.cu_deviceptr(), 16),
            ("vc", vc.cu_deviceptr(), 16),
        ];
        if let Some((name, at, align)) = starts
            .iter()
            .find(|(_, at, align)| !at.is_multiple_of(*align))
        {
            return Err(GpuError::shape(
                what,
                format!("{name} at {at:#x} is not {align}-byte aligned"),
            ));
        }
        // The walk counts keys in u32 up to one tile past the largest count.
        if ctx > u32::MAX as usize - KEY_TILE {
            return Err(GpuError::shape(
                what,
                format!("ctx = {ctx}: the key walk counts to ctx + {KEY_TILE} in u32"),
            ));
        }
        let grid = launch_u32(what, "grid", blocks_for(t, n_kv))?;
        let n_kv = launch_u32(what, "n_kv", n_kv)?;
        let ctx = launch_u32(what, "ctx", ctx)?;
        let t = launch_u32(what, "t", t)?;
        let prep = self
            .module
            .prepare_gqa_prefill_flash_256(LaunchConfig1D::new(
                grid,
                THREADS_256_U32,
                DYN_BYTES_256_U32,
            ))?;
        self.module.gqa_prefill_flash_256(
            stream, &prep, q, kc, vc, n_keys, scale, n_kv, ctx, t, fault, y,
        )?;
        Ok(())
    }

    /// [`FlashGqaPrefill::enqueue_256`] for `n_head` query heads over `n_kv`
    /// key heads in blocks of [`PACK_4`]: `n_head` a nonzero multiple of `4 ·
    /// n_kv` (refused by name otherwise). One launch of `⌈t / 16⌉ · n_head / 4`
    /// blocks of 256 threads with its dynamic shared memory (98,304 bytes); the
    /// other refusals are `enqueue_256`'s, by name before any launch.
    /// Asynchronous, allocation-free, capturable.
    pub fn enqueue_256_p4(
        &self,
        stream: &CudaStream,
        args: GqaPrefillArgs<'_>,
    ) -> Result<(), GpuError> {
        self.enqueue_packed(
            "flash_gqa_prefill::enqueue_256_p4",
            stream,
            args,
            PACK_4,
            None,
        )
    }

    /// [`FlashGqaPrefill::enqueue_256_p4`] in blocks of [`PACK_2`]: `n_head` a
    /// nonzero multiple of `2 · n_kv` (refused by name otherwise). One launch
    /// of `⌈t / 32⌉ · n_head / 2` blocks of 256 threads with the same dynamic
    /// shared memory. Asynchronous, allocation-free, capturable.
    pub fn enqueue_256_p2(
        &self,
        stream: &CudaStream,
        args: GqaPrefillArgs<'_>,
    ) -> Result<(), GpuError> {
        self.enqueue_packed(
            "flash_gqa_prefill::enqueue_256_p2",
            stream,
            args,
            PACK_2,
            None,
        )
    }

    /// [`FlashGqaPrefill::enqueue_256_p2`] with a first key a row: row `t`
    /// attends over keys `[k_first[t], n_keys[t])` (the module doc's
    /// lower-bound paragraph), `k_first` on the device, `t` of them (refused by
    /// name when shorter). A first key at or past its row's count raises
    /// [`FaultSite::KeyCount`] and leaves that row NaN. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_256_p2_lower(
        &self,
        stream: &CudaStream,
        args: GqaPrefillArgs<'_>,
        k_first: &DeviceBuffer<u32>,
    ) -> Result<(), GpuError> {
        self.enqueue_packed(
            "flash_gqa_prefill::enqueue_256_p2_lower",
            stream,
            args,
            PACK_2,
            Some(k_first),
        )
    }

    /// The lower-bound flash with one query head per pack, for any nonzero
    /// group, including three. Each row attends over `[k_first[t], n_keys[t])`.
    pub fn enqueue_256_p1_lower(
        &self,
        stream: &CudaStream,
        args: GqaPrefillArgs<'_>,
        k_first: &DeviceBuffer<u32>,
    ) -> Result<(), GpuError> {
        self.enqueue_packed(
            "flash_gqa_prefill::enqueue_256_p1_lower",
            stream,
            args,
            1,
            Some(k_first),
        )
    }

    /// The packed entries' one launcher: the group refused unless a nonzero
    /// multiple of `pack` (1, [`PACK_4`] or [`PACK_2`]), the buffers and their
    /// alignment checked, then that pack's entry, its lower-bound form when
    /// `k_first` is given.
    fn enqueue_packed(
        &self,
        what: &'static str,
        stream: &CudaStream,
        args: GqaPrefillArgs<'_>,
        pack: usize,
        k_first: Option<&DeviceBuffer<u32>>,
    ) -> Result<(), GpuError> {
        let packs = checked_packed(what, &args, (HEAD_256, KEY_TILE, pack))?;
        let GqaPrefillArgs {
            q,
            kc,
            vc,
            n_keys,
            scale,
            n_head: _,
            n_kv,
            ctx,
            t,
            fault,
            y,
        } = args;
        if let Some(first) = k_first
            && first.len() < t
        {
            return Err(GpuError::shape(
                what,
                format!("k_first.len() {} < {t}", first.len()),
            ));
        }
        let grid = launch_u32(what, "grid", t.div_ceil(Q_ROWS_256 / pack) * n_kv * packs)?;
        let packs = launch_u32(what, "packs", packs)?;
        let n_kv = launch_u32(what, "n_kv", n_kv)?;
        let ctx = launch_u32(what, "ctx", ctx)?;
        let t = launch_u32(what, "t", t)?;
        let cfg = LaunchConfig1D::new(grid, THREADS_256_U32, DYN_BYTES_256_U32);
        match (pack, k_first) {
            (1, Some(k_first)) => {
                let prep = self.module.prepare_gqa_prefill_flash_256_p1_lower(cfg)?;
                self.module.gqa_prefill_flash_256_p1_lower(
                    stream, &prep, q, kc, vc, n_keys, k_first, scale, n_kv, ctx, t, packs, fault, y,
                )?;
            }
            (PACK_2, None) => {
                let prep = self.module.prepare_gqa_prefill_flash_256_p2(cfg)?;
                self.module.gqa_prefill_flash_256_p2(
                    stream, &prep, q, kc, vc, n_keys, scale, n_kv, ctx, t, packs, fault, y,
                )?;
            }
            (PACK_2, Some(k_first)) => {
                let prep = self.module.prepare_gqa_prefill_flash_256_p2_lower(cfg)?;
                self.module.gqa_prefill_flash_256_p2_lower(
                    stream, &prep, q, kc, vc, n_keys, k_first, scale, n_kv, ctx, t, packs, fault, y,
                )?;
            }
            (PACK_4, None) => {
                let prep = self.module.prepare_gqa_prefill_flash_256_p4(cfg)?;
                self.module.gqa_prefill_flash_256_p4(
                    stream, &prep, q, kc, vc, n_keys, scale, n_kv, ctx, t, packs, fault, y,
                )?;
            }
            _ => {
                return Err(GpuError::shape(
                    what,
                    format!("no entry packs {pack} query heads a block"),
                ));
            }
        }
        Ok(())
    }

    /// [`FlashGqaPrefill::enqueue`] over the Q8_0 cache (the module doc's
    /// `_q8` paragraph, `gqa_prefill_flash_q8`): `args`' four planes in
    /// place of the f16 pair, one launch of `blocks_for(t, n_kv)` blocks,
    /// the two-step staging and the in-place widening inside it. The same
    /// refusals by name before any launch — the four planes checked at
    /// [`q8_plane_lens`] at [`HEAD`] and their 16-byte alignment (the code
    /// chunks and the scales' copies), `q` at 8. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_q8(
        &self,
        stream: &CudaStream,
        args: GqaPrefillQ8Args<'_>,
    ) -> Result<(), GpuError> {
        let what = "flash_gqa_prefill::enqueue_q8";
        let GqaPrefillQ8Args {
            q,
            kq,
            kd,
            vq,
            vd,
            n_keys,
            scale,
            n_head,
            n_kv,
            ctx,
            t,
            fault,
            y,
        } = args;
        if n_kv == 0 || ctx == 0 || t == 0 {
            return Err(GpuError::shape(
                what,
                format!("need n_kv, ctx and t >= 1, got n_kv={n_kv} ctx={ctx} t={t}"),
            ));
        }
        if n_head != n_kv * GROUP {
            return Err(GpuError::shape(
                what,
                format!(
                    "the kernel groups {GROUP} query heads per key head; got {n_head} over {n_kv}"
                ),
            ));
        }
        let (words, scales) = q8_plane_lens(HEAD, n_kv, ctx);
        let lens = [
            ("q", q.len(), t * n_head * HEAD),
            ("kq", kq.len(), words),
            ("kd", kd.len(), scales),
            ("vq", vq.len(), words),
            ("vd", vd.len(), scales),
            ("n_keys", n_keys.len(), t),
            ("y", y.len(), t * n_head * HEAD),
        ];
        if let Some((name, got, need)) = lens.iter().find(|(_, got, need)| got < need) {
            return Err(GpuError::shape(
                what,
                format!("{name}.len() {got} < {need}"),
            ));
        }
        // A window at another offset is a misaligned access on the card.
        let starts = [
            ("q", q.cu_deviceptr(), 8u64),
            ("kq", kq.cu_deviceptr(), 16),
            ("kd", kd.cu_deviceptr(), 16),
            ("vq", vq.cu_deviceptr(), 16),
            ("vd", vd.cu_deviceptr(), 16),
        ];
        if let Some((name, at, align)) = starts
            .iter()
            .find(|(_, at, align)| !at.is_multiple_of(*align))
        {
            return Err(GpuError::shape(
                what,
                format!("{name} at {at:#x} is not {align}-byte aligned"),
            ));
        }
        // The walk counts keys in u32 up to one tile past the largest count.
        if ctx > u32::MAX as usize - KEY_TILE {
            return Err(GpuError::shape(
                what,
                format!("ctx = {ctx}: the key walk counts to ctx + {KEY_TILE} in u32"),
            ));
        }
        let grid = launch_u32(what, "grid", blocks_for(t, n_kv))?;
        let n_kv = launch_u32(what, "n_kv", n_kv)?;
        let ctx = launch_u32(what, "ctx", ctx)?;
        let t = launch_u32(what, "t", t)?;
        let prep = self
            .module
            .prepare_gqa_prefill_flash_q8(LaunchConfig1D::new(grid, THREADS_U32, 0))?;
        self.module.gqa_prefill_flash_q8(
            stream, &prep, q, kq, kd, vq, vd, n_keys, scale, n_kv, ctx, t, fault, y,
        )?;
        Ok(())
    }

    /// [`FlashGqaPrefill::enqueue_256`] over the Q8_0 cache
    /// (`gqa_prefill_flash_256_q8`): the planes [`q8_plane_lens`] at
    /// [`HEAD_256`], one launch of `blocks_for(t, n_kv)` blocks of 256
    /// threads with its dynamic shared memory (100,352 bytes); the same
    /// refusals by name before any launch. Asynchronous, allocation-free,
    /// capturable.
    pub fn enqueue_256_q8(
        &self,
        stream: &CudaStream,
        args: GqaPrefillQ8Args<'_>,
    ) -> Result<(), GpuError> {
        let what = "flash_gqa_prefill::enqueue_256_q8";
        let GqaPrefillQ8Args {
            q,
            kq,
            kd,
            vq,
            vd,
            n_keys,
            scale,
            n_head,
            n_kv,
            ctx,
            t,
            fault,
            y,
        } = args;
        if n_kv == 0 || ctx == 0 || t == 0 {
            return Err(GpuError::shape(
                what,
                format!("need n_kv, ctx and t >= 1, got n_kv={n_kv} ctx={ctx} t={t}"),
            ));
        }
        if n_head != n_kv * GROUP {
            return Err(GpuError::shape(
                what,
                format!(
                    "the kernel groups {GROUP} query heads per key head; got {n_head} over {n_kv}"
                ),
            ));
        }
        let (words, scales) = q8_plane_lens(HEAD_256, n_kv, ctx);
        let lens = [
            ("q", q.len(), t * n_head * HEAD_256),
            ("kq", kq.len(), words),
            ("kd", kd.len(), scales),
            ("vq", vq.len(), words),
            ("vd", vd.len(), scales),
            ("n_keys", n_keys.len(), t),
            ("y", y.len(), t * n_head * HEAD_256),
        ];
        if let Some((name, got, need)) = lens.iter().find(|(_, got, need)| got < need) {
            return Err(GpuError::shape(
                what,
                format!("{name}.len() {got} < {need}"),
            ));
        }
        // A window at another offset is a misaligned access on the card.
        let starts = [
            ("q", q.cu_deviceptr(), 8u64),
            ("kq", kq.cu_deviceptr(), 16),
            ("kd", kd.cu_deviceptr(), 16),
            ("vq", vq.cu_deviceptr(), 16),
            ("vd", vd.cu_deviceptr(), 16),
        ];
        if let Some((name, at, align)) = starts
            .iter()
            .find(|(_, at, align)| !at.is_multiple_of(*align))
        {
            return Err(GpuError::shape(
                what,
                format!("{name} at {at:#x} is not {align}-byte aligned"),
            ));
        }
        // The walk counts keys in u32 up to one tile past the largest count.
        if ctx > u32::MAX as usize - KEY_TILE {
            return Err(GpuError::shape(
                what,
                format!("ctx = {ctx}: the key walk counts to ctx + {KEY_TILE} in u32"),
            ));
        }
        let grid = launch_u32(what, "grid", blocks_for(t, n_kv))?;
        let n_kv = launch_u32(what, "n_kv", n_kv)?;
        let ctx = launch_u32(what, "ctx", ctx)?;
        let t = launch_u32(what, "t", t)?;
        let prep = self
            .module
            .prepare_gqa_prefill_flash_256_q8(LaunchConfig1D::new(
                grid,
                THREADS_256_U32,
                DYN_BYTES_256_Q8_U32,
            ))?;
        self.module.gqa_prefill_flash_256_q8(
            stream, &prep, q, kq, kd, vq, vd, n_keys, scale, n_kv, ctx, t, fault, y,
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Gpu;
    use gguf::quant::f32_to_f16_bits;

    /// EmbeddingGemma 2's sliding layers: four query heads over two key heads.
    const N_KV: usize = 2;
    const N_HEAD: usize = 4;
    /// Rows and keys: a tail block of 16 rows and a tail tile of 48 keys.
    const T: usize = 1200;
    const ROW: usize = N_HEAD * HEAD_256;
    const NAN16: u16 = 0x7e00;

    /// One launch's inputs: `k_first` picks the lower-bound entry.
    struct Case<'a> {
        q: &'a [f32],
        kc: &'a [u16],
        vc: &'a [u16],
        ctx: usize,
        n_keys: &'a [u32],
        k_first: Option<&'a [u32]>,
    }

    fn run(
        gpu: &Gpu,
        k: &FlashGqaPrefill,
        c: &Case<'_>,
        fault: FaultSink,
    ) -> Result<Vec<f32>, GpuError> {
        let s = gpu.stream();
        let t = c.n_keys.len();
        let q = DeviceBuffer::from_host(s, c.q)?;
        let kc = DeviceBuffer::from_host(s, c.kc)?;
        let vc = DeviceBuffer::from_host(s, c.vc)?;
        let nk = DeviceBuffer::from_host(s, c.n_keys)?;
        let kf = c
            .k_first
            .map(|f| DeviceBuffer::from_host(s, f))
            .transpose()?;
        let mut y = DeviceBuffer::<f32>::zeroed(s, t * ROW)?;
        let args = GqaPrefillArgs {
            q: &q,
            kc: &kc,
            vc: &vc,
            n_keys: &nk,
            scale: 1.0,
            n_head: N_HEAD,
            n_kv: N_KV,
            ctx: c.ctx,
            t,
            fault,
            y: &mut y,
        };
        match &kf {
            Some(kf) => k.enqueue_256_p2_lower(s, args, kf)?,
            None => k.enqueue_256_p2(s, args)?,
        }
        s.synchronize()?;
        Ok(y.to_host_vec(s)?)
    }

    fn same(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
    }

    /// The symmetric window of half-width `w` over `T` keys: row `i` sees `[i − w, i + w]`.
    fn window(w: usize) -> (Vec<u32>, Vec<u32>) {
        (0..T)
            .map(|i| (i.saturating_sub(w) as u32, (i + w + 1).min(T) as u32))
            .unzip()
    }

    /// The lower-bound entry against the plain pack-of-two entry and itself, at half windows of
    /// 256, 300 and 512 keys: each sampled row (the first 32, every 13th, the last 8) of a window
    /// launch bit for bit its one-row launch over a cache holding NaN outside the row's window,
    /// a rerun the same bits; first keys of zero the plain entry's bits; a first key of 192 for
    /// every row the plain entry's bits on the cache from row 192 on; a dominant key at 229
    /// (inside a tile) the output of the rows whose window
    /// holds it and of no other row; a slice skips NaN tiles below all its first keys and masks
    /// a dominant value below its first keys in a live tile; a first key at or past its count
    /// `key_count` at the launch's layer with NaN in that row alone; a short `k_first` refused by
    /// name.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_p2_lower_is_each_rows_window() {
        const LAYER: usize = 9;
        let gpu = Gpu::with_device(0).expect("a Gpu on device 0");
        let k = FlashGqaPrefill::load(gpu.context()).expect("the prefill module");
        let none = gpu.unlabelled_sink();
        let mut st = 0x2545_f491u32;
        let mut next = move || {
            st = st.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (st >> 16) & 0x3ff
        };
        // Multiples of 1/512 in [−1, 1): exact in f16.
        let mut plane = || -> Vec<u16> {
            (0..N_KV * T * HEAD_256)
                .map(|_| f32_to_f16_bits(next() as f32 / 512.0 - 1.0))
                .collect()
        };
        let (kc, vc) = (plane(), plane());
        let q: Vec<f32> = (0..T * ROW)
            .map(|i| ((i * 37) % 1000) as f32 / 1000.0 - 0.5)
            .collect();
        let base = Case {
            q: &q,
            kc: &kc,
            vc: &vc,
            ctx: T,
            n_keys: &[],
            k_first: None,
        };
        let at = |n_keys: &[u32], k_first: Option<&[u32]>, fault: FaultSink| {
            run(
                &gpu,
                &k,
                &Case {
                    n_keys,
                    k_first,
                    ..base
                },
                fault,
            )
        };

        // 300: every other block's first keys straddle a tile boundary.
        for w in [256, 300, 512] {
            let (first, count) = window(w);
            let y = at(&count, Some(&first), none).expect("window launch");
            assert!(
                y.iter().all(|v| v.is_finite()),
                "w={w}: a non-finite output"
            );
            let again = at(&count, Some(&first), none).expect("rerun");
            assert!(same(&y, &again), "w={w}: a rerun moved a bit");
            for i in (0..T).filter(|&i| i < 32 || i.is_multiple_of(13) || i >= T - 8) {
                let (f, c) = (first[i] as usize, count[i] as usize);
                let fence = |p: &[u16]| -> Vec<u16> {
                    let mut p = p.to_vec();
                    for (j, r) in p.chunks_mut(HEAD_256).enumerate() {
                        if !(f..c).contains(&(j % T)) {
                            r.fill(NAN16);
                        }
                    }
                    p
                };
                let (kn, vn) = (fence(&kc), fence(&vc));
                let one = Case {
                    q: &q[i * ROW..(i + 1) * ROW],
                    kc: &kn,
                    vc: &vn,
                    ctx: T,
                    n_keys: &count[i..=i],
                    k_first: Some(&first[i..=i]),
                };
                let y1 = run(&gpu, &k, &one, none).expect("one-row launch");
                assert!(
                    same(&y[i * ROW..(i + 1) * ROW], &y1),
                    "w={w} row {i}: not its one-row launch"
                );
            }
            let zeros = vec![0u32; T];
            let lower = at(&count, Some(&zeros), none).expect("first keys of zero");
            let plain = at(&count, None, none).expect("plain");
            assert!(
                same(&lower, &plain),
                "w={w}: first keys of zero are not the plain entry"
            );
        }

        const SHIFT: usize = 192;
        let count: Vec<u32> = (0..T)
            .map(|i| (SHIFT + 1 + (i * 7) % (T - SHIFT)) as u32)
            .collect();
        let at_shift = vec![SHIFT as u32; T];
        let lower = at(&count, Some(&at_shift), none).expect("shift");
        let cut = |p: &[u16]| -> Vec<u16> {
            p.chunks(T * HEAD_256)
                .flat_map(|h| h[SHIFT * HEAD_256..].iter().copied())
                .collect()
        };
        let (kcut, vcut) = (cut(&kc), cut(&vc));
        let shifted: Vec<u32> = count.iter().map(|c| c - SHIFT as u32).collect();
        let plain = Case {
            q: &q,
            kc: &kcut,
            vc: &vcut,
            ctx: T - SHIFT,
            n_keys: &shifted,
            k_first: None,
        };
        let plain = run(&gpu, &k, &plain, none).expect("plain on the cut cache");
        assert!(
            same(&lower, &plain),
            "a first key of {SHIFT}: not the plain entry from row {SHIFT}"
        );

        // Key 229 scores 512 on every head, every other key at most 64: its weight is 1 and
        // theirs 0, so a row whose window holds it returns its value row exactly.
        const P: usize = 229;
        let (mut kd, mut vd) = (kc.clone(), vc.clone());
        for h in 0..N_KV {
            let r = (h * T + P) * HEAD_256;
            kd[r..r + HEAD_256].fill(f32_to_f16_bits(8.0));
            vd[r..r + HEAD_256].fill(f32_to_f16_bits(1000.0));
        }
        let qd = vec![0.25f32; 96 * ROW];
        let first: Vec<u32> = (0..96).map(|i| (P - 1 + i % 3) as u32).collect();
        let count = vec![T as u32; 96];
        let dom = Case {
            q: &qd,
            kc: &kd,
            vc: &vd,
            ctx: T,
            n_keys: &count,
            k_first: Some(&first),
        };
        let y = run(&gpu, &k, &dom, none).expect("dominant key");
        for (i, r) in y.chunks(ROW).enumerate() {
            if first[i] as usize <= P {
                assert!(
                    r.iter().all(|&v| v == 1000.0),
                    "row {i}: key {P} in its window, not its value"
                );
            } else {
                assert!(
                    r.iter().all(|v| v.abs() <= 1.001),
                    "row {i}: key {P} below its window was read"
                );
            }
        }
        assert!(
            gpu.fault().expect("the fault word").is_none(),
            "a clean launch raised"
        );

        // The first slice keeps tile zero in the block's walk. Later slices skip its NaNs,
        // then include or mask key 64 while an upper bound alone would take unmasked softmax.
        const BLOCK: usize = Q_ROWS_256 / PACK_2;
        const SLICE: usize = SLICE_ROWS / PACK_2;
        const CTX: usize = 3 * KEY_TILE;
        let qs = vec![0.0f32; BLOCK * ROW];
        let ks = vec![0u16; N_KV * CTX * HEAD_256];
        let mut vs = vec![f32_to_f16_bits(1.0); ks.len()];
        for h in 0..N_KV {
            let start = h * CTX * HEAD_256;
            vs[start..start + KEY_TILE * HEAD_256].fill(NAN16);
            let key = start + KEY_TILE * HEAD_256;
            vs[key..key + HEAD_256].fill(f32_to_f16_bits(1000.0));
        }
        let first: Vec<u32> = (0..BLOCK)
            .map(|i| {
                if i < SLICE {
                    0
                } else if i < 2 * SLICE {
                    KEY_TILE as u32
                } else {
                    (KEY_TILE + i % 2) as u32
                }
            })
            .collect();
        let count: Vec<u32> = first
            .iter()
            .enumerate()
            .map(|(i, &f)| {
                if i < SLICE {
                    CTX as u32
                } else {
                    f + KEY_TILE_U32
                }
            })
            .collect();
        let slices = Case {
            q: &qs,
            kc: &ks,
            vc: &vs,
            ctx: CTX,
            n_keys: &count,
            k_first: Some(&first),
        };
        let y = run(&gpu, &k, &slices, none).expect("slices with disjoint first tiles");
        for (i, r) in y.chunks(ROW).enumerate().skip(SLICE) {
            // Zero queries give 64 unit weights: a mean of 64 ones, or of 1000 and 63 ones.
            let want = if first[i] == KEY_TILE_U32 {
                1063.0f32 / 64.0
            } else {
                1.0f32
            };
            assert!(
                r.iter().all(|v| v.to_bits() == want.to_bits()),
                "row {i}: a skipped NaN tile or a masked value reached its output"
            );
        }

        let (good, count) = window(256);
        let mut first = good.clone();
        first[37] = count[37];
        first[500] = count[500] + 5;
        let clean = at(&count, Some(&good), none).expect("clean");
        let sink = gpu.layer_sink(LAYER).expect("a sink");
        let equal = at(&count[37..=37], Some(&first[37..=37]), sink).expect("equal first key");
        let f = gpu
            .take_fault()
            .expect("the fault word")
            .expect("an equal first key raises a fault");
        assert_eq!(f.layer, LAYER as u32);
        assert_eq!(f.site(), Some(FaultSite::KeyCount));
        assert!(equal.iter().all(|v| v.is_nan()));
        let y = at(&count, Some(&first), sink).expect("bad first keys");
        let f = gpu
            .take_fault()
            .expect("the fault word")
            .expect("a raised fault");
        assert!(
            f.layer == LAYER as u32 && f.site() == Some(FaultSite::KeyCount),
            "bad first keys raised {f:?}"
        );
        for (i, (r, c)) in y.chunks(ROW).zip(clean.chunks(ROW)).enumerate() {
            if i == 37 || i == 500 {
                assert!(
                    r.iter().all(|v| v.is_nan()),
                    "row {i}: a bad first key, not NaN"
                );
            } else {
                assert!(same(r, c), "row {i}: moved by another row's bad first key");
            }
        }

        let e = at(&count, Some(&first[..T - 1]), none).expect_err("a short k_first");
        let e = e.to_string();
        assert!(
            e.contains("enqueue_256_p2_lower") && e.contains("k_first"),
            "a short k_first: {e}"
        );
    }
}
