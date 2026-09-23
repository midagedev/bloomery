//! The lightning indexer: for every query token, a score for each visible
//! row of its layer's key source, and the `min(n_vis, top_k)` best rows as
//! the list `attn::SelectedRows` reads.
//!
//! The two projections are the caller's gemvs: `indexer.attn_q_b` (q8_0) of
//! the attention's normalized low-rank query gives `q`, and `indexer.proj`
//! (q3_K) of the attention's normed input gives `w`, each in the gemvs'
//! output layout (row-major, a token per column). The rest is two launches:
//!
//! - `ds41_indexer_score` — per token, every block first builds the query:
//!   the rope of each head's last `n_dims` values by the token's table
//!   (`rope::rope_pair_rn`), the Hadamard transform ([`ht_warp`] and
//!   [`HT_SCALE`], the index keys' own), each value then split into
//!   `hi = f16(q)` and `lo = f16(q − hi)` (`q − hi` is exact in f32). The
//!   weights are `w · scale`, `scale = 1/√(n_head · head_dim)` as ik forms it.
//!   Each warp then walks 16-key tiles on the tensor cores: per head the
//!   exact f16 products `hi·k` and `lo·k` are summed into one f32
//!   accumulator row each (`mma.m16n8k16`, eight k-steps), the dot is
//!   `acc_hi + acc_lo`, `relu` keeps `x > 0` and gives `+0` otherwise, and
//!   the head sum is fixed: lane group `g` chains heads `g, 8+g, 16+g, 24+g`
//!   (a product, then three fused multiply-adds), and the eight groups meet
//!   in the xor butterfly over lanes 4, 8, 16. Block `(t, 0)` also writes the
//!   query (after the transform, before the split) and the scaled weights.
//!   The score of every row below `n_vis` lands in the scores buffer, and
//!   the top ten bits of its order-preserving key ([`order_key`]) are counted
//!   in the token's histogram.
//! - `ds41_indexer_topk` — one block per token: the histogram (returned to
//!   zero as it is read) picks the bin of the k-th key, two more passes over
//!   the scores refine it by eleven bits each to the exact k-th key `T`, and
//!   an ordered pass writes every row whose key is above `T`, then the first
//!   rows equal to `T` in row order until `k` rows are taken — ascending
//!   rows, ties at the threshold going to the lower row.
//!
//! The counts are device words, read per launch: token `t`'s `n_vis` at
//! `ints[n_vis_at + t]` and `top_k` at `ints[top_k_at]`, so a captured step
//! replays at any depth and any selection width. `k = min(top_k, stride)`.
//! With `n_vis <= k` the score pass does nothing and the list is the
//! identity `0..n_vis` — the rows the selection would keep, and the prefix
//! order. With `k = 0` nothing is written. The grids depend on the token
//! count and the device alone; work past `n_vis` exits early.
//!
//! The histogram is zero between launches: it is allocated zeroed, the score
//! pass adds into it only when the top-k pass will read it, and that pass
//! zeroes what it reads. Both passes of a step must therefore run, in order,
//! on one stream.

use bloomery_gpu::{DeviceTensor, GpuError, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::atomic::{AtomicOrdering, BlockAtomicU32, DeviceAtomicU32};
use cuda_device::convert::{cvt_f16x2_f32, cvt_f32x2_f16x2};
use cuda_device::float::{add_rn_f32, fma_rn_f32, mul_rn_f32};
use cuda_device::shared::cvta_generic_to_shared_u32;
use cuda_device::wmma::{ldmatrix_x4_shared_u32, mma_m16n8k16_f32_f16};
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread, warp,
};
use cuda_host::cuda_module;
use model::arch::deepseek41::hparams::Hparams;
use std::sync::Arc;

use crate::index_key::{HT_SCALE, PER_LANE, WIDTH, ht_warp};
use crate::rope::rope_pair_rn;

/// Query heads (`attention.indexer.head_count`) the kernels are built for.
pub const HEADS: usize = 32;
/// Values of one head and of one index key (`attention.indexer.key_length`).
pub const HEAD_DIM: usize = WIDTH;
/// Bins of the score pass's histogram: the top ten bits of a key.
pub const HIST_BINS: usize = 1024;

/// Warps of a score block, and its threads.
const SCORE_WARPS: usize = 8;
const SCORE_THREADS: u32 = 256;
const _: () = assert!(SCORE_WARPS * 32 == SCORE_THREADS as usize);
/// Heads one warp builds in the score block's prologue.
const HEADS_PER_WARP: usize = HEADS / SCORE_WARPS;
const _: () = assert!(HEADS_PER_WARP * SCORE_WARPS == HEADS);
/// Keys one warp scores per step: two `mma` n-tiles of eight.
const TILE_KEYS: usize = 16;
/// `mma` m-tiles over the heads: tile `i` holds heads `8i ..`, their `hi`
/// rows first and their `lo` rows after.
const M_TILES: usize = HEADS / 8;
/// `mma` k-steps over a head: sixteen dims each.
const K_STEPS: usize = HEAD_DIM / 16;
/// u32 words between two rows of the staged query tile: 64 words of f16
/// pairs, padded by four so that the eight rows of one `ldmatrix` phase
/// cover the thirty-two banks once.
const Q_ROW_WORDS: usize = HEAD_DIM / 2 + 4;
const _: () = assert!(Q_ROW_WORDS * 4 % 128 == 16);
/// Rows of the staged query tile: `hi` and `lo` of every head.
const Q_ROWS: usize = 2 * HEADS;
/// u32 words of the staged query tile.
const Q_WORDS: usize = Q_ROWS * Q_ROW_WORDS;
/// u32 words of one key row.
const KEY_WORDS: usize = HEAD_DIM / 2;

/// Threads of a top-k block, and its warps.
const TOPK_THREADS: u32 = 512;
const TOPK_WARPS: usize = TOPK_THREADS as usize / 32;
/// Rows one top-k thread reads per batch: sixteen loads in flight.
const LANE_ROWS: usize = 16;
/// Rows the block reads per batch of a refining pass, `tid + 512·m`.
const ROUND_ROWS: usize = TOPK_THREADS as usize * LANE_ROWS;
/// Bins of the two refining passes: eleven bits each.
const FINE_BINS: usize = 2048;
/// Bins a top-k thread owns in the coarse and the fine histograms.
const COARSE_PER: usize = HIST_BINS / TOPK_THREADS as usize;
const FINE_PER: usize = FINE_BINS / TOPK_THREADS as usize;
const _: () = assert!(COARSE_PER * TOPK_THREADS as usize == HIST_BINS);
const _: () = assert!(FINE_PER * TOPK_THREADS as usize == FINE_BINS);

/// The weights' scale ik applies (`build_deepseek4.cpp`, the `lid_weights`
/// scale): `1/√(head_dim · n_head)`, the square root and the quotient each
/// rounded to f32.
#[must_use]
pub fn weights_scale(n_head: usize, head_dim: usize) -> f32 {
    1.0 / ((head_dim * n_head) as f32).sqrt()
}

// ------------------------------------------------------------------ cores

/// The order-preserving key of a score: an unsigned integer that orders as
/// the score does, `-0.0` first made `+0.0` so the two tie.
#[inline(always)]
#[must_use]
pub fn order_key(v: f32) -> u32 {
    let b = v.to_bits();
    let b = if b == 0x8000_0000 { 0 } else { b };
    if b & 0x8000_0000 != 0 {
        !b
    } else {
        b | 0x8000_0000
    }
}

/// `x` where it is above zero, `+0` elsewhere (NaN included).
#[inline(always)]
fn relu(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.0 }
}

/// A block-wide exclusive prefix sum of `v` over thread order in a
/// [`TOPK_THREADS`] block, and the block's total. `wsum` is block-shared
/// scratch of [`TOPK_WARPS`] words. Every thread of the block calls it (it
/// holds two barriers).
///
/// # Safety
/// `wsum` points at [`TOPK_WARPS`] words of this block's shared memory that
/// nothing else uses across the call.
#[inline(always)]
unsafe fn block_scan(v: u32, lane: u32, wid: usize, wsum: *mut u32) -> (u32, u32) {
    let mut x = v;
    let mut off = 1u32;
    while off < 32 {
        let y = warp::shuffle_up(x, off);
        if lane >= off {
            x += y;
        }
        off <<= 1;
    }
    if lane == 31 {
        // SAFETY: wid < TOPK_WARPS words of `wsum` (this fn's contract);
        // lane 31 of warp wid is the slot's only writer.
        unsafe { *wsum.add(wid) = x };
    }
    thread::sync_threads();
    let (mut before, mut total) = (0u32, 0u32);
    let mut w = 0usize;
    while w < TOPK_WARPS {
        // SAFETY: w < TOPK_WARPS; the barrier above published every slot.
        let s = unsafe { *wsum.add(w) };
        if w < wid {
            before += s;
        }
        total += s;
        w += 1;
    }
    thread::sync_threads();
    (before + x - v, total)
}

/// The bin that holds the `need`-th largest key, counting down from the top
/// bin, and how many keys lie in the bins above it. Thread `tid` owns the
/// `PER` bins `top − PER·tid − j`, `j = 0 .. PER`, with counts `c[j]`. The
/// counts must total at least `need >= 1`. Every thread of the block calls
/// it and gets the same pair.
///
/// # Safety
/// As [`block_scan`]; `pick` points at two words of this block's shared
/// memory that nothing else uses across the call.
#[inline(always)]
unsafe fn pick_bin<const PER: usize>(
    c: [u32; PER],
    top: u32,
    need: u32,
    lane: u32,
    tid: usize,
    wsum: *mut u32,
    pick: *mut u32,
) -> (u32, u32) {
    let mut sum = 0u32;
    let mut j = 0usize;
    while j < PER {
        thread::__unroll_config::<0>();
        sum += c[j];
        j += 1;
    }
    // SAFETY: forwarded from this fn's contract.
    let (mut before, _) = unsafe { block_scan(sum, lane, tid / 32, wsum) };
    let mut j = 0usize;
    while j < PER {
        thread::__unroll_config::<0>();
        if before < need && need <= before + c[j] {
            // SAFETY: two words of `pick` (this fn's contract); exactly one
            // (thread, bin) holds the need-th key, so one thread writes.
            unsafe {
                *pick = top - (PER * tid + j) as u32;
                *pick.add(1) = before;
            }
        }
        before += c[j];
        j += 1;
    }
    thread::sync_threads();
    // SAFETY: written before the barrier above.
    let out = unsafe { (*pick, *pick.add(1)) };
    thread::sync_threads();
    out
}

// ---------------------------------------------------------------- kernels

#[cuda_module]
mod indexer_kernels {
    use super::*;

    /// The score pass: block `b` serves token `b / blocks`, and its warps
    /// walk that token's 16-key tiles `j = (b % blocks)·8 + warp`, then every
    /// `8·blocks` after it. See the module doc for the arithmetic. `q` is
    /// the query projection, `[HEADS·HEAD_DIM × tokens]`; `w` the weights'
    /// projection, `[HEADS × tokens]`; `tables` holds each token's rope
    /// table as f32 bits (`n_dims` values, `[cos, sin, …]`, token `t` at
    /// `rope_at + t·rope_stride`); `keys` the key source, `rows` rows of
    /// [`HEAD_DIM`] f16. Writes `q_out` `[tokens × HEADS × HEAD_DIM]` and
    /// `w_out` `[tokens × HEADS]` (block `(t, 0)`), `scores` `[tokens ×
    /// rows]` (rows below `n_vis`), and adds into `hist` `[tokens ×
    /// HIST_BINS]`.
    #[kernel]
    #[launch_bounds(256, 1)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            q.len() >= tokens * 4096,
            w.len() >= tokens * 32,
            ints.len() >= n_vis_at + tokens,
            ints.len() > top_k_at,
            tables.len() + rope_stride >= rope_at + tokens * rope_stride + n_dims,
            keys.len() >= rows * 128,
            q_out.len() >= tokens * 4096,
            w_out.len() >= tokens * 32,
            scores.len() >= tokens * rows,
            hist.len() >= tokens * 1024
        )
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    pub fn ds41_indexer_score(
        q: &[f32],
        w: &[f32],
        ints: &[u32],
        tables: &[u32],
        keys: &[u16],
        tokens: u32,
        blocks: u32,
        n_vis_at: u32,
        top_k_at: u32,
        rope_at: u32,
        rope_stride: u32,
        n_dims: u32,
        rows: u32,
        stride: u32,
        scale: f32,
        mut q_out: DisjointSlice<f32>,
        mut w_out: DisjointSlice<f32>,
        mut scores: DisjointSlice<f32>,
        mut hist: DisjointSlice<u32>,
    ) {
        // The query tile, `hi` and `lo` rows of every head as f16 pairs,
        // and the token's histogram over this block's keys.
        static mut QT: SharedArray<u32, Q_WORDS> = SharedArray::UNINIT;
        static mut HS: SharedArray<u32, HIST_BINS> = SharedArray::UNINIT;

        let bid = thread::blockIdx_x();
        let t = (bid / blocks) as usize;
        let gb = (bid % blocks) as usize;
        let tid = thread::threadIdx_x() as usize;
        let lane = warp::lane_id();
        let wid = tid / 32;
        // SAFETY: t < tokens (the grid is blocks · tokens), so n_vis_at + t
        // < ints.len() by the launch contract.
        let n = unsafe { *ints.get_unchecked(n_vis_at as usize + t) }.min(rows) as usize;
        // SAFETY: top_k_at < ints.len() by the launch contract.
        let k = unsafe { *ints.get_unchecked(top_k_at as usize) }.min(stride) as usize;
        let tiles = n.div_ceil(TILE_KEYS);
        if n <= k || k == 0 || gb * SCORE_WARPS >= tiles {
            return; // block-uniform: no barrier is skipped
        }

        // SAFETY: block-shared statics; the raw forms reach them without a
        // reference, and every access below is bounded and barrier-ordered.
        let (qt, hs) = unsafe {
            (
                SharedArray::as_raw_mut_ptr(&raw mut QT),
                SharedArray::as_raw_mut_ptr(&raw mut HS),
            )
        };
        let mut b = tid;
        while b < HIST_BINS {
            // SAFETY: b < HIST_BINS; thread tid alone writes bins tid + 256·i
            // before the barrier below.
            unsafe { *hs.add(b) = 0 };
            b += SCORE_THREADS as usize;
        }

        // ---- the query: warp `wid` builds heads 4·wid .. 4·wid + 3, lane
        // `lane` values `4·lane ..` of each.
        let tk = tokens as usize;
        let v = PER_LANE * lane as usize;
        let tail0 = HEAD_DIM - n_dims as usize;
        let roped = v >= tail0;
        let mut cs = [0.0f32; PER_LANE];
        if roped {
            let at = rope_at as usize + t * rope_stride as usize + v - tail0;
            let mut i = 0usize;
            while i < PER_LANE {
                thread::__unroll_config::<0>();
                // SAFETY: v − tail0 + i < n_dims, so the word lies inside
                // token t's table, below rope_at + (tokens − 1)·rope_stride +
                // n_dims <= tables.len() (launch contract).
                cs[i] = f32::from_bits(unsafe { *tables.get_unchecked(at + i) });
                i += 1;
            }
        }
        let mut j = 0usize;
        while j < HEADS_PER_WARP {
            thread::__unroll_config::<0>();
            let h = wid * HEADS_PER_WARP + j;
            let mut x = [0.0f32; PER_LANE];
            let mut i = 0usize;
            while i < PER_LANE {
                thread::__unroll_config::<0>();
                // SAFETY: h < HEADS and v + i < HEAD_DIM, so the index is
                // below HEADS·HEAD_DIM·tokens <= q.len() (launch contract).
                x[i] = unsafe { *q.get_unchecked((h * HEAD_DIM + v + i) * tk + t) };
                i += 1;
            }
            if roped {
                let (a0, a1) = rope_pair_rn(x[0], x[1], cs[0], cs[1]);
                let (a2, a3) = rope_pair_rn(x[2], x[3], cs[2], cs[3]);
                x = [a0, a1, a2, a3];
            }
            let hh = ht_warp(x, lane);
            let y = [
                mul_rn_f32(hh[0], HT_SCALE),
                mul_rn_f32(hh[1], HT_SCALE),
                mul_rn_f32(hh[2], HT_SCALE),
                mul_rn_f32(hh[3], HT_SCALE),
            ];
            if gb == 0 {
                let o = (t * HEADS + h) * HEAD_DIM + v;
                // SAFETY: o + 3 < tokens·HEADS·HEAD_DIM <= q_out.len() (launch
                // contract); block (t, 0) alone writes token t's rows, this
                // lane its four values.
                unsafe {
                    *q_out.get_unchecked_mut(o) = y[0];
                    *q_out.get_unchecked_mut(o + 1) = y[1];
                    *q_out.get_unchecked_mut(o + 2) = y[2];
                    *q_out.get_unchecked_mut(o + 3) = y[3];
                }
            }
            let hi01 = cvt_f16x2_f32(y[0], y[1]);
            let hi23 = cvt_f16x2_f32(y[2], y[3]);
            let (h0, h1) = cvt_f32x2_f16x2(hi01);
            let (h2, h3) = cvt_f32x2_f16x2(hi23);
            let lo01 = cvt_f16x2_f32(y[0] - h0, y[1] - h1);
            let lo23 = cvt_f16x2_f32(y[2] - h2, y[3] - h3);
            let row_hi = 16 * (h / 8) + h % 8;
            let at = v / 2;
            // SAFETY: row_hi + 8 < Q_ROWS and at + 1 < HEAD_DIM / 2 <
            // Q_ROW_WORDS keep all four stores inside the tile; this lane
            // alone writes these words, before the barrier below.
            unsafe {
                *qt.add(row_hi * Q_ROW_WORDS + at) = hi01;
                *qt.add(row_hi * Q_ROW_WORDS + at + 1) = hi23;
                *qt.add((row_hi + 8) * Q_ROW_WORDS + at) = lo01;
                *qt.add((row_hi + 8) * Q_ROW_WORDS + at + 1) = lo23;
            }
            j += 1;
        }
        // The weights of the heads this lane's accumulator rows carry: head
        // 8i + lane/4 of m-tile i.
        let g = (lane / 4) as usize;
        let c = (lane % 4) as usize;
        let mut wr = [0.0f32; M_TILES];
        let mut i = 0usize;
        while i < M_TILES {
            thread::__unroll_config::<0>();
            // SAFETY: 8i + g < HEADS, so the index is below HEADS·tokens <=
            // w.len() (launch contract).
            wr[i] = mul_rn_f32(unsafe { *w.get_unchecked((8 * i + g) * tk + t) }, scale);
            i += 1;
        }
        if gb == 0 && wid == 0 {
            let l = lane as usize;
            // SAFETY: l < HEADS bounds the load inside w and the store inside
            // w_out (launch contract); lane l of warp 0 of block (t, 0) is
            // the slot's only writer.
            unsafe {
                *w_out.get_unchecked_mut(t * HEADS + l) =
                    mul_rn_f32(*w.get_unchecked(l * tk + t), scale);
            }
        }
        thread::sync_threads();

        // ---- the keys: B fragments straight from the cache rows. Lane
        // (g, c) holds key g of each n-tile, dims 16s + 2c, +1 and 16s + 2c
        // + 8, +9 of k-step s — words 8s + c and 8s + 4 + c of its row. A
        // row at or past n is not read and stays zero.
        let kw = keys.as_ptr().cast::<u32>();
        // SAFETY: `qt` is this block's shared query tile, a generic address
        // into shared memory, which is what the conversion takes.
        let qbase = unsafe { cvta_generic_to_shared_u32(qt.cast_const().cast::<u8>()) };
        // `ldmatrix` lane roles: row lane % 16 of the m-tile, dims half
        // lane / 16 of the k-step.
        let a_lane = (lane as usize % 16) * Q_ROW_WORDS + 4 * (lane as usize / 16);
        let stride_tiles = blocks as usize * SCORE_WARPS;
        let sbase = t * rows as usize;
        let mut tile = gb * SCORE_WARPS + wid;
        while tile < tiles {
            let key0 = tile * TILE_KEYS;
            let r0 = key0 + g;
            let r1 = r0 + 8;
            let (live0, live1) = (r0 < n, r1 < n);
            let mut b0 = [0u32; 2 * K_STEPS];
            let mut b1 = [0u32; 2 * K_STEPS];
            let mut s = 0usize;
            while s < K_STEPS {
                thread::__unroll_config::<0>();
                if live0 {
                    // SAFETY: r0 < n <= rows and 8s + 4 + c < KEY_WORDS put
                    // both words inside row r0 of `keys`, rows·128 f16
                    // (launch contract), device-allocated and so aligned.
                    unsafe {
                        b0[2 * s] = *kw.add(r0 * KEY_WORDS + 8 * s + c);
                        b0[2 * s + 1] = *kw.add(r0 * KEY_WORDS + 8 * s + 4 + c);
                    }
                }
                if live1 {
                    // SAFETY: r1 < n <= rows and 8s + 4 + c < KEY_WORDS put
                    // both words inside row r1 of `keys`, rows·128 f16
                    // (launch contract), device-allocated and so aligned.
                    unsafe {
                        b1[2 * s] = *kw.add(r1 * KEY_WORDS + 8 * s + c);
                        b1[2 * s + 1] = *kw.add(r1 * KEY_WORDS + 8 * s + 4 + c);
                    }
                }
                s += 1;
            }
            // Per lane: keys 2c and 2c + 1 of n-tile 0, then of n-tile 1.
            let mut p = [0.0f32; 4];
            let mut i = 0usize;
            while i < M_TILES {
                thread::__unroll_config::<0>();
                let mut c0 = [0.0f32; 4];
                let mut c1 = [0.0f32; 4];
                let mut s = 0usize;
                while s < K_STEPS {
                    thread::__unroll_config::<0>();
                    let word = a_lane + 16 * i * Q_ROW_WORDS + 8 * s;
                    // SAFETY: row 16i + lane % 16 < Q_ROWS and words 8s + 4·(lane
                    // / 16) .. + 4 <= HEAD_DIM / 2 are inside the tile, which
                    // the barrier above published; every lane of the warp
                    // reaches this load with the same qualifiers.
                    let a = unsafe { ldmatrix_x4_shared_u32(qbase + (4 * word) as u32) };
                    // SAFETY: the whole warp issues these mma.sync with
                    // fragments it loaded (the loop bounds are warp-uniform).
                    unsafe {
                        c0 = mma_m16n8k16_f32_f16(c0, a, [b0[2 * s], b0[2 * s + 1]]);
                        c1 = mma_m16n8k16_f32_f16(c1, a, [b1[2 * s], b1[2 * s + 1]]);
                    }
                    s += 1;
                }
                // Head 8i + g: accumulator row g is its `hi` part, row g + 8
                // its `lo` part, at keys 2c and 2c + 1.
                let d = [
                    add_rn_f32(c0[0], c0[2]),
                    add_rn_f32(c0[1], c0[3]),
                    add_rn_f32(c1[0], c1[2]),
                    add_rn_f32(c1[1], c1[3]),
                ];
                let mut e = 0usize;
                while e < 4 {
                    thread::__unroll_config::<0>();
                    p[e] = if i == 0 {
                        mul_rn_f32(wr[i], relu(d[e]))
                    } else {
                        fma_rn_f32(wr[i], relu(d[e]), p[e])
                    };
                    e += 1;
                }
                i += 1;
            }
            // The eight lane groups' partial sums meet: xor 4, 8, 16. Every
            // lane of a group ends with the same four scores.
            let mut lv = 0u32;
            while lv < 3 {
                thread::__unroll_config::<0>();
                let off = 4u32 << lv;
                let mut e = 0usize;
                while e < 4 {
                    thread::__unroll_config::<0>();
                    p[e] = add_rn_f32(p[e], warp::shuffle_xor_f32(p[e], off));
                    e += 1;
                }
                lv += 1;
            }
            // Lanes g < 4 publish one score each: key key0 + 8·(g / 2) + 2c
            // + g % 2.
            if g < 4 {
                let key = key0 + 8 * (g / 2) + 2 * c + g % 2;
                let sv = if g == 0 {
                    p[0]
                } else if g == 1 {
                    p[1]
                } else if g == 2 {
                    p[2]
                } else {
                    p[3]
                };
                if key < n {
                    // SAFETY: key < n <= rows, so sbase + key < tokens·rows <=
                    // scores.len() (launch contract); one lane of one warp
                    // scores each key.
                    unsafe { *scores.get_unchecked_mut(sbase + key) = sv };
                    let bin = (order_key(sv) >> 22) as usize;
                    // SAFETY: bin < HIST_BINS, block-shared; every access to
                    // the histogram between the two barriers is atomic.
                    unsafe {
                        BlockAtomicU32::from_ptr(hs.add(bin)).fetch_add(1, AtomicOrdering::Relaxed)
                    };
                }
            }
            tile += stride_tiles;
        }
        thread::sync_threads();
        let hist_ptr = hist.as_mut_ptr();
        let mut b = tid;
        while b < HIST_BINS {
            // SAFETY: b < HIST_BINS; the barrier above ordered every add first.
            let cnt = unsafe { *hs.add(b) };
            if cnt != 0 {
                // SAFETY: t·HIST_BINS + b < tokens·HIST_BINS <= hist.len()
                // (launch contract); every access to it in this launch is
                // atomic.
                unsafe {
                    DeviceAtomicU32::from_ptr(hist_ptr.add(t * HIST_BINS + b))
                        .fetch_add(cnt, AtomicOrdering::Relaxed)
                };
            }
            b += SCORE_THREADS as usize;
        }
    }

    /// The top-k pass: block `t` selects token `t`'s rows into `list[t ·
    /// stride ..]` — see the module doc. Reads the scores and the histogram
    /// the score pass left, and zeroes the histogram's row.
    #[kernel]
    #[launch_bounds(512)]
    #[launch_contract(
        domain = 1,
        block = (512, 1, 1),
        requires = (
            ints.len() >= n_vis_at + tokens,
            ints.len() > top_k_at,
            scores.len() >= tokens * rows,
            hist.len() >= tokens * 1024,
            list.len() >= tokens * stride
        )
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    pub fn ds41_indexer_topk(
        ints: &[u32],
        scores: &[f32],
        tokens: u32,
        n_vis_at: u32,
        top_k_at: u32,
        rows: u32,
        stride: u32,
        mut hist: DisjointSlice<u32>,
        mut list: DisjointSlice<u32>,
    ) {
        static mut FINE: SharedArray<u32, FINE_BINS> = SharedArray::UNINIT;
        static mut WSUM: SharedArray<u32, TOPK_WARPS> = SharedArray::UNINIT;
        static mut WTOT: SharedArray<u32, { 2 * TOPK_WARPS }> = SharedArray::UNINIT;
        static mut PICK: SharedArray<u32, 2> = SharedArray::UNINIT;

        let t = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        let lane = warp::lane_id();
        let wid = tid / 32;
        let _ = tokens;
        // SAFETY: t < tokens (one block per token), so n_vis_at + t <
        // ints.len() by the launch contract.
        let n = unsafe { *ints.get_unchecked(n_vis_at as usize + t) }.min(rows) as usize;
        // SAFETY: top_k_at < ints.len() by the launch contract.
        let k = unsafe { *ints.get_unchecked(top_k_at as usize) }.min(stride) as usize;
        let lbase = t * stride as usize;
        if n <= k {
            let mut i = tid;
            while i < n {
                // SAFETY: i < n <= k <= stride, so lbase + i < tokens·stride
                // <= list.len() (launch contract); thread tid alone writes
                // entries tid + 512·m.
                unsafe { *list.get_unchecked_mut(lbase + i) = i as u32 };
                i += TOPK_THREADS as usize;
            }
            return;
        }
        if k == 0 {
            return;
        }
        // SAFETY: block-shared statics; the raw forms reach them without a
        // reference, and every access below is bounded and barrier-ordered.
        let (fine, wsum, wtot, pick) = unsafe {
            (
                SharedArray::as_raw_mut_ptr(&raw mut FINE),
                SharedArray::as_raw_mut_ptr(&raw mut WSUM),
                SharedArray::as_raw_mut_ptr(&raw mut WTOT),
                SharedArray::as_raw_mut_ptr(&raw mut PICK),
            )
        };
        let sbase = t * rows as usize;
        let need = k as u32;

        // ---- the top ten bits: the score pass's histogram, zeroed as read.
        let hist_ptr = hist.as_mut_ptr();
        let mut hc = [0u32; COARSE_PER];
        let mut j = 0usize;
        while j < COARSE_PER {
            thread::__unroll_config::<0>();
            let at = t * HIST_BINS + HIST_BINS - 1 - (COARSE_PER * tid + j);
            // SAFETY: at < tokens·HIST_BINS <= hist.len() (launch contract);
            // this thread alone reads and zeroes the bin, and the score pass
            // that added into it is ordered before this launch.
            unsafe {
                hc[j] = *hist_ptr.add(at);
                *hist_ptr.add(at) = 0;
            }
            j += 1;
        }
        // SAFETY: WSUM and PICK are used by nothing else across the call.
        let (b1, above1) = unsafe {
            pick_bin::<COARSE_PER>(hc, HIST_BINS as u32 - 1, need, lane, tid, wsum, pick)
        };
        let need2 = need - above1;

        // ---- the next eleven bits, among the keys in bin b1.
        // SAFETY: FINE is used by this pass alone until the next barrier pair.
        unsafe { fine_count(fine, tid) };
        let mut base = 0usize;
        while base < n {
            let mut kk = [0u32; LANE_ROWS];
            let mut m = 0usize;
            while m < LANE_ROWS {
                thread::__unroll_config::<0>();
                let i = base + tid + TOPK_THREADS as usize * m;
                if i < n {
                    // SAFETY: i < n <= rows: sbase + i < tokens·rows <=
                    // scores.len() (launch contract); the score pass wrote
                    // it and is ordered before this launch.
                    kk[m] = order_key(unsafe { *scores.get_unchecked(sbase + i) });
                }
                m += 1;
            }
            let mut m = 0usize;
            while m < LANE_ROWS {
                thread::__unroll_config::<0>();
                let i = base + tid + TOPK_THREADS as usize * m;
                if i < n && kk[m] >> 22 == b1 {
                    // SAFETY: the bin is below FINE_BINS; every access to
                    // FINE between the barriers is atomic.
                    unsafe {
                        BlockAtomicU32::from_ptr(fine.add(((kk[m] >> 11) & 0x7ff) as usize))
                            .fetch_add(1, AtomicOrdering::Relaxed)
                    };
                }
                m += 1;
            }
            base += ROUND_ROWS;
        }
        thread::sync_threads();
        // SAFETY: FINE was published by the barrier above.
        let fc = unsafe { fine_bins(fine, tid) };
        // SAFETY: WSUM and PICK are used by nothing else across the call.
        let (b2, above2) =
            unsafe { pick_bin::<FINE_PER>(fc, FINE_BINS as u32 - 1, need2, lane, tid, wsum, pick) };
        let need3 = need2 - above2;
        let prefix = (b1 << 11) | b2;

        // ---- the last eleven bits, among the keys with that prefix.
        // SAFETY: `fine` is FINE, FINE_BINS words of this block's shared
        // memory; every thread reaches this call, and FINE is used by this
        // pass alone until the next barrier pair.
        unsafe { fine_count(fine, tid) };
        let mut base = 0usize;
        while base < n {
            let mut kk = [0u32; LANE_ROWS];
            let mut m = 0usize;
            while m < LANE_ROWS {
                thread::__unroll_config::<0>();
                let i = base + tid + TOPK_THREADS as usize * m;
                if i < n {
                    // SAFETY: i < n <= rows: sbase + i < tokens·rows <=
                    // scores.len() (launch contract); the score pass wrote
                    // it and is ordered before this launch.
                    kk[m] = order_key(unsafe { *scores.get_unchecked(sbase + i) });
                }
                m += 1;
            }
            let mut m = 0usize;
            while m < LANE_ROWS {
                thread::__unroll_config::<0>();
                let i = base + tid + TOPK_THREADS as usize * m;
                if i < n && kk[m] >> 11 == prefix {
                    // SAFETY: the bin `kk & 0x7ff` is below FINE_BINS; every
                    // access to FINE between the barriers is atomic.
                    unsafe {
                        BlockAtomicU32::from_ptr(fine.add((kk[m] & 0x7ff) as usize))
                            .fetch_add(1, AtomicOrdering::Relaxed)
                    };
                }
                m += 1;
            }
            base += ROUND_ROWS;
        }
        thread::sync_threads();
        // SAFETY: FINE was published by the barrier above.
        let fc = unsafe { fine_bins(fine, tid) };
        // SAFETY: WSUM and PICK are used by nothing else across the call.
        let (b3, above3) =
            unsafe { pick_bin::<FINE_PER>(fc, FINE_BINS as u32 - 1, need3, lane, tid, wsum, pick) };
        // The k-th key, how many keys lie above it, and how many of those
        // equal to it the list takes.
        let thr = (prefix << 11) | b3;
        let c_gt = above1 + above2 + above3;
        let take_eq = need - c_gt;

        // ---- the list, in row order. Warp `wid` owns the contiguous rows
        // `c0 .. c1`: it counts the rows it takes, the block scans the
        // counts, and the warp walks its rows again, writing each row it
        // takes at its place — every row above `T`, and the first
        // `take_eq` rows equal to it.
        let chunk = n.div_ceil(32 * TOPK_WARPS) * 32;
        let c0 = (wid * chunk).min(n);
        let c1 = (c0 + chunk).min(n);
        let l = lane as usize;
        let (mut tg, mut te) = (0u32, 0u32);
        let mut base = c0;
        while base < c1 {
            // SAFETY: c1 <= n <= rows (the fn's contract below).
            let kk = unsafe { lane_keys(scores, sbase, base, c1, l) };
            let mut m = 0usize;
            while m < LANE_ROWS {
                thread::__unroll_config::<0>();
                let live = base + 32 * m + l < c1;
                tg += warp::ballot(live && kk[m] > thr).count_ones();
                te += warp::ballot(live && kk[m] == thr).count_ones();
                m += 1;
            }
            base += 32 * LANE_ROWS;
        }
        if lane == 0 {
            // SAFETY: wid < TOPK_WARPS bounds both slots; lane 0 of warp wid
            // is their only writer, before the barrier.
            unsafe {
                *wtot.add(wid) = tg;
                *wtot.add(TOPK_WARPS + wid) = te;
            }
        }
        thread::sync_threads();
        let (mut og, mut oe) = (0u32, 0u32);
        let mut w = 0usize;
        while w < wid {
            // SAFETY: w < wid < TOPK_WARPS; published by the barrier above.
            unsafe {
                og += *wtot.add(w);
                oe += *wtot.add(TOPK_WARPS + w);
            }
            w += 1;
        }
        let lt = warp::lanemask_lt();
        let mut base = c0;
        while base < c1 {
            // SAFETY: c1 <= n <= rows, so sbase + c1 <= tokens·rows <=
            // scores.len() (the fn's contract below).
            let kk = unsafe { lane_keys(scores, sbase, base, c1, l) };
            let mut m = 0usize;
            while m < LANE_ROWS {
                thread::__unroll_config::<0>();
                let i = base + 32 * m + l;
                let live = i < c1;
                let mg = warp::ballot(live && kk[m] > thr);
                let me = warp::ballot(live && kk[m] == thr);
                let gt_before = og + (mg & lt).count_ones();
                let eq_before = oe + (me & lt).count_ones();
                let is_gt = (mg >> lane) & 1 != 0;
                let is_eq = (me >> lane) & 1 != 0;
                if is_gt || (is_eq && eq_before < take_eq) {
                    let pos = (gt_before + eq_before.min(take_eq)) as usize;
                    // SAFETY: the rows taken number c_gt + take_eq = k and pos
                    // counts those before this one, so pos < k <= stride and
                    // lbase + pos < tokens·stride <= list.len() (launch
                    // contract); each position has one row.
                    unsafe { *list.get_unchecked_mut(lbase + pos) = i as u32 };
                }
                og += mg.count_ones();
                oe += me.count_ones();
                m += 1;
            }
            base += 32 * LANE_ROWS;
        }
    }
}

/// The keys of rows `base + 32·m + lane`, `m = 0 .. LANE_ROWS`, of one
/// token's scores at `scores[sbase ..]`: a row at or past `end` is not read
/// and gives 0. The loads are issued as a batch.
///
/// # Safety
/// `sbase + end <= scores.len()`.
#[inline(always)]
unsafe fn lane_keys(
    scores: &[f32],
    sbase: usize,
    base: usize,
    end: usize,
    lane: usize,
) -> [u32; LANE_ROWS] {
    let mut kk = [0u32; LANE_ROWS];
    let mut m = 0usize;
    while m < LANE_ROWS {
        thread::__unroll_config::<0>();
        let i = base + 32 * m + lane;
        if i < end {
            // SAFETY: i < end, so sbase + i < scores.len() (this fn's
            // contract).
            kk[m] = order_key(unsafe { *scores.get_unchecked(sbase + i) });
        }
        m += 1;
    }
    kk
}

/// Zero the fine histogram and publish it.
///
/// # Safety
/// `fine` points at [`FINE_BINS`] words of this block's shared memory; every
/// thread of the block calls it.
#[inline(always)]
unsafe fn fine_count(fine: *mut u32, tid: usize) {
    let mut j = 0usize;
    while j < FINE_PER {
        thread::__unroll_config::<0>();
        // SAFETY: FINE_PER·tid + j < FINE_BINS (this fn's contract); thread
        // tid alone writes these bins before the barrier.
        unsafe { *fine.add(FINE_PER * tid + j) = 0 };
        j += 1;
    }
    thread::sync_threads();
}

/// This thread's [`FINE_PER`] bins of the fine histogram, from the top bin
/// down: bins `FINE_BINS − 1 − (FINE_PER·tid + j)`.
///
/// # Safety
/// `fine` points at [`FINE_BINS`] published words of this block's shared
/// memory.
#[inline(always)]
unsafe fn fine_bins(fine: *const u32, tid: usize) -> [u32; FINE_PER] {
    let mut c = [0u32; FINE_PER];
    let mut j = 0usize;
    while j < FINE_PER {
        thread::__unroll_config::<0>();
        // SAFETY: the index is below FINE_BINS (this fn's contract).
        c[j] = unsafe { *fine.add(FINE_BINS - 1 - (FINE_PER * tid + j)) };
        j += 1;
    }
    c
}

// -------------------------------------------------------------- launchers

/// What one step of the indexer leaves besides the list, allocated once at
/// load for `tokens` tokens over a `rows`-row key source and reused by every
/// launch: the query after its transform, the scaled weights, the scores and
/// the histogram (zeroed, and zero again after each pair of launches).
pub struct IndexerScratch {
    /// `[tokens × HEADS × HEAD_DIM]`: the query the scores used, before the
    /// f16 split.
    pub q: DeviceBuffer<f32>,
    /// `[tokens × HEADS]`: the scaled weights.
    pub w: DeviceBuffer<f32>,
    /// `[tokens × rows]`: the scores of the rows below each token's `n_vis`.
    pub scores: DeviceBuffer<f32>,
    /// `[tokens × HIST_BINS]`.
    pub hist: DeviceBuffer<u32>,
    tokens: usize,
    rows: usize,
}

impl IndexerScratch {
    /// Allocate for `tokens` tokens over `rows` key rows. Load-time only.
    pub fn new(
        stream: &CudaStream,
        tokens: usize,
        rows: usize,
    ) -> Result<IndexerScratch, GpuError> {
        Ok(IndexerScratch {
            q: DeviceBuffer::zeroed(stream, tokens * HEADS * HEAD_DIM)?,
            w: DeviceBuffer::zeroed(stream, tokens * HEADS)?,
            scores: DeviceBuffer::zeroed(stream, tokens * rows)?,
            hist: DeviceBuffer::zeroed(stream, tokens * HIST_BINS)?,
            tokens,
            rows,
        })
    }

    /// Bytes of device memory the scratch holds.
    #[must_use]
    pub fn device_bytes(&self) -> usize {
        4 * (self.q.len() + self.w.len() + self.scores.len() + self.hist.len())
    }
}

/// One indexer step's inputs and outputs.
pub struct IndexerArgs<'a> {
    /// The query projection, `[HEADS·HEAD_DIM × tokens]`, a token per column.
    pub q: &'a DeviceBuffer<f32>,
    /// The weights' projection, `[HEADS × tokens]`, before the scale.
    pub w: &'a DeviceBuffer<f32>,
    /// Token `t`'s `n_vis` at `ints[n_vis_at + t]`, `top_k` at
    /// `ints[top_k_at]`: the step image, or any buffer laid out so.
    pub ints: &'a DeviceBuffer<u32>,
    pub n_vis_at: usize,
    pub top_k_at: usize,
    /// Token `t`'s rope table — the compressed layers' YaRN table at the
    /// token's position, f32 bits — at `tables[rope_at + t·rope_stride ..]`.
    pub tables: &'a DeviceBuffer<u32>,
    pub rope_at: usize,
    pub rope_stride: usize,
    /// The key source's index keys, `[rows × HEAD_DIM]` f16 bits.
    pub keys: &'a DeviceTensor<u16>,
    pub tokens: usize,
    pub scratch: &'a mut IndexerScratch,
    /// `[tokens × stride]`: token `t`'s list at `t·stride`.
    pub list: &'a mut DeviceBuffer<u32>,
    pub stride: usize,
}

/// The loaded indexer module: `ds41_indexer_score` and `ds41_indexer_topk`.
/// Owns no stream: each enqueue takes the caller's, so the launches order
/// with the rest of the step and capture.
pub struct IndexerKernels {
    module: indexer_kernels::LoadedModule,
    /// Score blocks per token: one per multiprocessor.
    blocks: usize,
    scale: f32,
    n_dims: usize,
}

impl IndexerKernels {
    /// Load this file's device bundle into `ctx` for the model `hp`
    /// describes: its indexer must be [`HEADS`] heads of [`HEAD_DIM`], and
    /// its rope turns an even number of values, at most a head, in pairs of
    /// pairs. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>, hp: &Hparams) -> Result<IndexerKernels, GpuError> {
        Self::with_shape(ctx, hp.indexer.n_head, hp.indexer.head_dim, hp.rope_dims)
    }

    /// [`Self::load`] from the three numbers it reads: `n_head` heads of
    /// `head_dim` and `n_dims` roped values per head. For callers with no
    /// model file (synthetic benches). Load-time only.
    pub fn with_shape(
        ctx: &Arc<CudaContext>,
        n_head: usize,
        head_dim: usize,
        n_dims: usize,
    ) -> Result<IndexerKernels, GpuError> {
        let what = "IndexerKernels::load";
        if n_head != HEADS || head_dim != HEAD_DIM {
            return Err(GpuError::Shape {
                what,
                detail: format!(
                    "the indexer has {n_head} heads of {head_dim}; the kernels are built for {HEADS} \
                     of {HEAD_DIM}"
                ),
            });
        }
        if n_dims == 0 || !n_dims.is_multiple_of(PER_LANE) || n_dims > HEAD_DIM {
            return Err(GpuError::Shape {
                what,
                detail: format!(
                    "the query's rope turns {n_dims} values: a positive multiple of {PER_LANE}, at \
                     most {HEAD_DIM}"
                ),
            });
        }
        let blocks = usize::try_from(ctx.multiprocessor_count()?).map_err(|_| GpuError::Shape {
            what,
            detail: "the multiprocessor count passes usize".to_string(),
        })?;
        // SAFETY: this crate owns the embedded device bundle produced for the
        // module above; the launcher checks its launch contract.
        let module = unsafe { indexer_kernels::load(ctx)? };
        Ok(IndexerKernels {
            module,
            blocks: blocks.max(1),
            scale: weights_scale(n_head, head_dim),
            n_dims,
        })
    }

    /// The weights' scale the score pass applies.
    #[must_use]
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Enqueue one indexer step: the score pass, then the top-k pass. Two
    /// launches whose grids come from the token count and the device, not
    /// from the counts, so a captured graph replays them at any `n_vis` and
    /// `top_k`. Asynchronous, allocation-free, capturable.
    pub fn enqueue(&self, stream: &CudaStream, a: IndexerArgs<'_>) -> Result<(), GpuError> {
        let what = "ds41 IndexerKernels::enqueue";
        let IndexerArgs {
            q,
            w,
            ints,
            n_vis_at,
            top_k_at,
            tables,
            rope_at,
            rope_stride,
            keys,
            tokens,
            scratch,
            list,
            stride,
        } = a;
        let shape = |detail: String| GpuError::Shape { what, detail };
        let rows = keys.rows();
        if tokens == 0 || keys.cols() != HEAD_DIM || rows == 0 {
            return Err(shape(format!(
                "tokens {tokens} and keys {rows}x{}: at least one token, and rows of {HEAD_DIM}",
                keys.cols()
            )));
        }
        if scratch.tokens != tokens || scratch.rows != rows {
            return Err(shape(format!(
                "the scratch serves {} tokens over {} rows, the launch {tokens} over {rows}",
                scratch.tokens, scratch.rows
            )));
        }
        let last_table = rope_at + (tokens - 1) * rope_stride + self.n_dims;
        for (name, have, want) in [
            ("q", q.len(), tokens * HEADS * HEAD_DIM),
            ("w", w.len(), tokens * HEADS),
            ("ints (n_vis)", ints.len(), n_vis_at + tokens),
            ("ints (top_k)", ints.len(), top_k_at + 1),
            ("tables", tables.len(), last_table),
            ("list", list.len(), tokens * stride),
        ] {
            if have < want {
                return Err(shape(format!("{name} holds {have}, want >= {want}")));
            }
        }
        let grid = launch_u32(what, "score grid", self.blocks * tokens)?;
        let prep =
            self.module
                .prepare_ds41_indexer_score(LaunchConfig1D::new(grid, SCORE_THREADS, 0))?;
        let tokens_u = launch_u32(what, "tokens", tokens)?;
        let rows_u = launch_u32(what, "rows", rows)?;
        let stride_u = launch_u32(what, "stride", stride)?;
        let n_vis_u = launch_u32(what, "n_vis_at", n_vis_at)?;
        let top_k_u = launch_u32(what, "top_k_at", top_k_at)?;
        self.module.ds41_indexer_score(
            stream,
            &prep,
            q,
            w,
            ints,
            tables,
            keys.buf(),
            tokens_u,
            launch_u32(what, "blocks", self.blocks)?,
            n_vis_u,
            top_k_u,
            launch_u32(what, "rope_at", rope_at)?,
            launch_u32(what, "rope_stride", rope_stride)?,
            launch_u32(what, "n_dims", self.n_dims)?,
            rows_u,
            stride_u,
            self.scale,
            &mut scratch.q,
            &mut scratch.w,
            &mut scratch.scores,
            &mut scratch.hist,
        )?;
        let prep = self.module.prepare_ds41_indexer_topk(LaunchConfig1D::new(
            tokens_u,
            TOPK_THREADS,
            0,
        ))?;
        self.module.ds41_indexer_topk(
            stream,
            &prep,
            ints,
            &scratch.scores,
            tokens_u,
            n_vis_u,
            top_k_u,
            rows_u,
            stride_u,
            &mut scratch.hist,
            list,
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{order_key, weights_scale};

    /// The weights' scale for the file's indexer is exactly 2⁻⁶.
    #[test]
    fn weights_scale_is_a_power_of_two() {
        assert_eq!(weights_scale(32, 128).to_bits(), (1.0f32 / 64.0).to_bits());
    }

    /// Keys order as the scores do, and the two zeros tie.
    #[test]
    fn order_key_orders() {
        let v = [
            f32::NEG_INFINITY,
            -3.5,
            -1e-30,
            -0.0,
            0.0,
            1e-30,
            2.0,
            f32::INFINITY,
        ];
        for p in v.windows(2) {
            let (a, b) = (order_key(p[0]), order_key(p[1]));
            if p[0] == p[1] {
                assert_eq!(a, b);
            } else {
                assert!(a < b, "{} {}", p[0], p[1]);
            }
        }
    }
}
