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

use crate::fault::{FaultSink, FaultSite};
use crate::flash::{dev_exp, f32x2_to_f16x2_bits};
use crate::flash_gqa::{GROUP, HEAD};
use crate::{GpuError, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::ptx_asm;
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, shared, thread, warp, wmma,
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

/// The staged word of word `w` of tile row `row`: rows `ROW_WORDS` apart,
/// 16-byte chunk `w / 4` stored at chunk `(w / 4) ^ (row & 7)`. The one owner
/// of the tile layout: every staging store and `cp.async` destination goes
/// through it, and every `ldmatrix` address is a lane base taken from it plus
/// an offset [`staged_offsets_hold`] proves.
#[inline(always)]
const fn staged_word(row: usize, w: usize) -> usize {
    row * ROW_WORDS + ((((w >> 2) ^ (row & 7)) << 2) | (w & 3))
}

/// The layout's sector facts, over the eight rows `row & 7` distinguishes:
/// every logical 32-byte pair of chunks lands in one aligned 32-byte sector,
/// and every chunk stays in its 128-byte half of the row.
const fn staged_pairs_hold() -> bool {
    let mut row = 0;
    while row < 8 {
        let mut w = 0;
        while w < ROW_WORDS {
            let at = staged_word(row, w) - row * ROW_WORDS;
            let mate = staged_word(row, w ^ 4) - row * ROW_WORDS;
            if at / 8 != mate / 8 || at / 32 != w / 32 {
                return false;
            }
            w += 1;
        }
        row += 1;
    }
    true
}

/// u32 words of half a staged row: the XOR never crosses it. An `ldmatrix`
/// step reads a pair of chunks (8 words), so a half-row is HALF_PAIRS steps.
const HALF_ROW: usize = ROW_WORDS / 2;
const HALF_PAIRS: usize = HALF_ROW / 8;

/// The layout's offsets, over a whole tile: word `w` of row `row` is the same
/// word of row `row % 8` in the first half-row plus whole rows and a whole
/// half-row — so an address `8·i` rows or a half-row further on is the same
/// lane's address plus a constant.
const fn staged_offsets_hold() -> bool {
    let mut row = 0;
    while row < KEY_TILE {
        let mut w = 0;
        while w < ROW_WORDS {
            let base = staged_word(row % 8, w % HALF_ROW);
            if staged_word(row, w) != base + (row - row % 8) * ROW_WORDS + (w - w % HALF_ROW) {
                return false;
            }
            w += 1;
        }
        row += 1;
    }
    true
}

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
const _: () = assert!(staged_pairs_hold());
const _: () = assert!(staged_offsets_hold());
const _: () = assert!(CHUNKS * THREADS == KEY_TILE * ROW_CHUNKS);
const _: () = assert!(Q_WORDS * THREADS == Q_ROWS * Q_PAIRS);
// Pass `e` of a thread is row `Q_ROW_STEP·e + tid / Q_PAIRS`: position
// `e / Q_POS_PASSES`, head `Q_ROW_STEP·(e % Q_POS_PASSES) + tid / Q_PAIRS`.
const _: () = assert!(Q_ROW_STEP * Q_PAIRS == THREADS && Q_POS_PASSES * Q_ROW_STEP == GROUP);
const _: () = assert!(Q_WORDS == Q_POS_PASSES * POSITIONS);
const _: () = assert!(KEY_TILE == 4 * 16 && HEAD.is_multiple_of(16));

/// Blocks a ubatch of `t` rows over `n_kv` key heads launches.
#[must_use]
pub fn blocks_for(t: usize, n_kv: usize) -> usize {
    t.div_ceil(POSITIONS) * n_kv
}

/// `exp(x)` for a weight that is rounded to f16 next: `ex2.approx.ftz` on
/// `x · log2 e`. It runs the same `MUFU.EX2` on the same argument as
/// [`dev_exp`] wherever that argument is at least −126; below it `dev_exp`
/// returns a value under 2^-126 and this returns 0, and both round to the f16
/// +0 (anything under 2^-25 does). So the rounded weight — the only thing a
/// weight reaches — is `dev_exp`'s bit for bit. Never for a rescale factor,
/// which multiplies the accumulators unrounded.
#[inline(always)]
fn exp_weight(x: f32) -> f32 {
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
fn fmax(a: f32, b: f32) -> f32 {
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

#[cuda_module]
mod flash_gqa_prefill_kernels {
    use super::*;
    use cuda_device::async_copy::{cp_async_cg_16, cp_async_commit_group, cp_async_wait_group};
    use cuda_device::convert::{cvt_f16x2_f32, cvt_f32x2_f16x2};
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
        let mut lp = 0usize;
        while lp < POSITIONS {
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
            lp += 1;
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
        let mut e = 0usize;
        while e < Q_WORDS {
            cuda_device::thread::__unroll_config::<0>();
            let t = t0 + e / Q_POS_PASSES;
            if t < rows {
                let at = q_first
                    + (e / Q_POS_PASSES) * q_pos
                    + Q_ROW_STEP * (e % Q_POS_PASSES) * Q_PAIRS;
                // SAFETY: t < t_rows and the head kh·GROUP + Q_ROW_STEP·(e %
                // Q_POS_PASSES) + r0 < n_head, so the row lies inside q
                // (launch contract); a row is HEAD f32 = HEAD/2 u64 and the
                // buffer is device-allocated, so the u64 read of values
                // 2·wd, 2·wd + 1 is aligned.
                raw[e] = unsafe { *q64.add(at) };
            }
            e += 1;
        }
        // The staged word of each pass of the first position; a later
        // position's is it plus whole groups of rows (staged_offsets_hold).
        let mut q_at = [0usize; Q_POS_PASSES];
        let mut j = 0usize;
        while j < Q_POS_PASSES {
            cuda_device::thread::__unroll_config::<0>();
            q_at[j] = staged_word(Q_ROW_STEP * j + r0, wd);
            j += 1;
        }
        let mut e = 0usize;
        while e < Q_WORDS {
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
            e += 1;
        }
        thread::sync_threads();

        // This warp's 16 query rows as A fragments, one per k16 step.
        let mut qa = [[0u32; 4]; QK_STEPS];
        let mut kk = 0usize;
        while kk < QK_STEPS {
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
            kk += 1;
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
                let mut i = 0usize;
                while i < CHUNKS {
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
                        // bytes from a device allocation, so the source is
                        // 16-byte aligned.
                        unsafe {
                            cp_async_cg_16(d, tile_src.wrapping_add(i * PASS_KEYS * ROW_WORDS))
                        };
                    } else {
                        // SAFETY: as above, the 16 bytes are this thread's own.
                        unsafe { *d.cast::<U32x4>() = U32x4::splat(0) };
                    }
                    i += 1;
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
        // constant (`staged_offsets_hold`).
        let k_row = lane % 8 + 8 * (lane / 16);
        let v_row = lane % 8 + 8 * ((lane / 8) % 2);
        let mut k_at = [0u32; HALF_PAIRS];
        let mut v_at = [0u32; HALF_PAIRS];
        let mut j = 0usize;
        while j < HALF_PAIRS {
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
            j += 1;
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
                let mut kk = 0usize;
                while kk < QK_STEPS {
                    cuda_device::thread::__unroll_config::<0>();
                    let mut nj = 0usize;
                    while nj < KEY_NT / 2 {
                        cuda_device::thread::__unroll_config::<0>();
                        // Key row 16·nj + k_row, chunk 2·kk + (lane/8) % 2.
                        let at = k_at[kk % HALF_PAIRS]
                            + (4 * (16 * nj * ROW_WORDS + HALF_ROW * (kk / HALF_PAIRS))) as u32;
                        // SAFETY: the row is below KEY_TILE and the chunk is
                        // whole inside KT (staged_word, staged_offsets_hold),
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
                        nj += 1;
                    }
                    kk += 1;
                }

                // The softmax over the tile's scores. `masked` (a literal)
                // tests each key against its row's count; a tile below both
                // counts takes the copy without the tests.
                macro_rules! softmax {
                    ($masked:literal) => {{
                        let kb0 = KEY_TILE_U32 * kb + t4k;
                        // Register j of key tile nt is row half j / 2 (the
                        // warp's position 2·w + j / 2, head g), key
                        // 64·kb + 8·nt + 2·t4 + j % 2.
                        let mut tmax = [f32::NEG_INFINITY; 2];
                        let mut nt = 0usize;
                        while nt < KEY_NT {
                            cuda_device::thread::__unroll_config::<0>();
                            let mut j = 0usize;
                            while j < 4 {
                                cuda_device::thread::__unroll_config::<0>();
                                let key = kb0 + (8 * nt + j % 2) as u32;
                                let v = if !$masked || key < cnt[j / 2] {
                                    mul_rn_f32(sc[nt][j], scale)
                                } else {
                                    f32::NEG_INFINITY
                                };
                                sc[nt][j] = v;
                                tmax[j / 2] = fmax(tmax[j / 2], v);
                                j += 1;
                            }
                            nt += 1;
                        }
                        let mut mn = [f32::NEG_INFINITY; 2];
                        let mut h = 0usize;
                        while h < 2 {
                            cuda_device::thread::__unroll_config::<0>();
                            let mut x = tmax[h];
                            x = fmax(x, warp::shuffle_xor_f32(x, 1));
                            x = fmax(x, warp::shuffle_xor_f32(x, 2));
                            let m_new = fmax(m[h], x);
                            if m_new > m[h] {
                                f[h] = if m[h] > f32::NEG_INFINITY {
                                    dev_exp(m[h] - m_new)
                                } else {
                                    0.0
                                };
                                m[h] = m_new;
                            }
                            mn[h] = m[h];
                            l[h] = mul_rn_f32(l[h], f[h]);
                            h += 1;
                        }
                        // The weights, rounded to f16 pairs as the A fragment
                        // of the value product: k16 step kk2 is key tiles
                        // 2·kk2 and 2·kk2 + 1, registers (0, 1) then (2, 3) of
                        // each, low key first. The lane's sum takes the
                        // rounded values in that order. The mask is the key's
                        // position, not its score: a NaN score stays NaN.
                        let mut kk2 = 0usize;
                        while kk2 < PV_STEPS {
                            cuda_device::thread::__unroll_config::<0>();
                            let mut e = 0usize;
                            while e < 4 {
                                cuda_device::thread::__unroll_config::<0>();
                                let nt = 2 * kk2 + e / 2;
                                let j0 = 2 * (e % 2);
                                let h = e % 2;
                                let key = kb0 + (8 * nt) as u32;
                                let p0 = if !$masked || key < cnt[h] {
                                    exp_weight(sc[nt][j0] - mn[h])
                                } else {
                                    0.0
                                };
                                let p1 = if !$masked || key + 1 < cnt[h] {
                                    exp_weight(sc[nt][j0 + 1] - mn[h])
                                } else {
                                    0.0
                                };
                                let packed = cvt_f16x2_f32(p0, p1);
                                let (r0, r1) = cvt_f32x2_f16x2(packed);
                                l[h] = add_rn_f32(add_rn_f32(l[h], r0), r1);
                                pa[kk2][e] = packed;
                                e += 1;
                            }
                            kk2 += 1;
                        }
                    }};
                }
                if end <= warp_lo {
                    softmax!(false);
                } else {
                    softmax!(true);
                }

                // The accumulators times f, unless no lane of the warp has a
                // factor other than 1 (the vote keeps the branch uniform).
                if warp::any(f[0] != 1.0 || f[1] != 1.0) {
                    let mut nd = 0usize;
                    while nd < 2 * DIM_PAIRS {
                        cuda_device::thread::__unroll_config::<0>();
                        let mut j = 0usize;
                        while j < 4 {
                            cuda_device::thread::__unroll_config::<0>();
                            o[nd][j] = mul_rn_f32(o[nd][j], f[j / 2]);
                            j += 1;
                        }
                        nd += 1;
                    }
                }
            }

            // The value tile has landed, and every warp is done reading the
            // key tile.
            // SAFETY: as at the top of the loop.
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
                let mut kk2 = 0usize;
                while kk2 < PV_STEPS {
                    cuda_device::thread::__unroll_config::<0>();
                    let mut nd = 0usize;
                    while nd < DIM_PAIRS {
                        cuda_device::thread::__unroll_config::<0>();
                        // Value row 16·kk2 + v_row, chunk 2·nd + lane/16.
                        let at = v_at[nd % HALF_PAIRS]
                            + (4 * (16 * kk2 * ROW_WORDS + HALF_ROW * (nd / HALF_PAIRS))) as u32;
                        // SAFETY: the row is below KEY_TILE and the chunk is
                        // whole inside VT (staged_word, staged_offsets_hold),
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
                        nd += 1;
                    }
                    kk2 += 1;
                }
            }
            kb += 1;
        }

        // ---- the row sums over the four lanes of a row, then o · (1/l).
        let mut inv = [0.0f32; 2];
        let mut h = 0usize;
        while h < 2 {
            cuda_device::thread::__unroll_config::<0>();
            let a = add_rn_f32(l[h], warp::shuffle_xor_f32(l[h], 1));
            let s = add_rn_f32(a, warp::shuffle_xor_f32(a, 2));
            inv[h] = 1.0 / s;
            h += 1;
        }
        let mut nd = 0usize;
        while nd < 2 * DIM_PAIRS {
            cuda_device::thread::__unroll_config::<0>();
            let mut j = 0usize;
            while j < 4 {
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
                j += 1;
            }
            nd += 1;
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

/// The loaded module. Owns no stream: each enqueue takes the engine stream.
pub struct FlashGqaPrefill {
    module: flash_gqa_prefill_kernels::LoadedModule,
}

impl FlashGqaPrefill {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<FlashGqaPrefill, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; the launcher checks its launch contract.
        let module = unsafe { flash_gqa_prefill_kernels::load(ctx)? };
        Ok(FlashGqaPrefill { module })
    }

    /// Enqueue `t` rows' attention: one launch of [`blocks_for`]`(t, n_kv)`
    /// blocks. A shape the kernel is not built for (`n_head ≠ n_kv ·
    /// GROUP`), an empty launch or a short buffer is refused by name.
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
}
