//! The lightning indexer: for every query token, a score for each visible
//! row of its layer's key source, and the `min(n_vis, top_k)` best rows as
//! the list a selected-rows attention reads.
//!
//! The two projections are the caller's gemvs: the indexer's query
//! projection of the attention's normalized low-rank query gives `q`, and the
//! head weights' projection of the attention's normed input gives `w`, each in
//! the gemvs' output layout (row-major, a token per column). The rest is two
//! launches:
//!
//! - `ds41_indexer_score` — per token, every block first builds the query:
//!   the rope of each head's last `n_dims` values by the token's table
//!   (`rope_table::rope_pair_rn`), the Hadamard transform ([`ht_warp`] and
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
//!   the top ten bits of its order-preserving key
//!   ([`bloomery_gpu::qsa::order_key`]) are counted in the token's histogram.
//! - `ds41_indexer_topk` — one block per token: the histogram (returned to
//!   zero as it is read) picks the bin of the k-th key, two more passes over
//!   the scores refine it by eleven bits each to the exact k-th key `T` —
//!   `bloomery_gpu::qsa`'s radix select (`pick_bin`, `refine`), shared with
//!   the token-pool selectors — and an ordered pass writes every row whose
//!   key is above `T`, then the first rows equal to `T` in row order until
//!   `k` rows are taken — ascending rows, ties at the threshold going to the
//!   lower row.
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
//!
//! A second score pass serves an indexer whose keys are read at f32 precision
//! and whose heads are [`HEADS_F32`] (ERNIE 5.5's lightning indexer: 64 heads
//! of 128, the query's and the key's first [`TURN_F32`] values turned in NEOX
//! pairs by the token's table row, no transform):
//! `indexer_score_f32` stages a tile of [`F32_KEYS`] keys of the `[rows ×
//! HEAD_DIM]` f32 key cache in shared memory, each lane holding two heads' 32
//! values of the token's turned query in registers, and dots every key on the
//! CUDA cores in f32: per key four warps' partial dots (32 values a warp, two
//! heads a lane, one fused multiply-add a value) meet through shared memory,
//! `relu`, the head weights `w · scale`, and the head sum as one add a lane
//! then the xor butterfly (16, 8, 4, 2, 1). `tokens` query tokens walk the
//! same tile in one launch (a prompt's rows), token `t` scoring the rows
//! below its own `n_vis`; the histogram is the 32-head pass's. The grid comes
//! from the key cache's rows and the token count, so a block whose tile lies
//! past its token's `n_vis` leaves before any barrier.
//!
//! `indexer_window` writes the rows a sliding window forces into a selection
//! list after its top-k — the list `[top_k of 0 .. n_vis | n_vis .. n_win]`,
//! its live count `min(n_vis, top_k) + (n_win − n_vis)` — so one list carries
//! a DSA layer's keys below the cut (the top-k is the identity there: every
//! key) and above it. [`IndexerF32Kernels`] launches the two and the top-k
//! pass.

use crate::fault::{FaultSink, FaultSite};
use crate::qsa::{order_key, pick_bin, refine};
use crate::rope_neox::neox_pair;
use crate::{DeviceTensor, GpuError, launch_u32};
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
use std::sync::Arc;

use crate::index_key::{HT_SCALE, PER_LANE, WIDTH, ht_warp};
use crate::rope_table::rope_pair_rn;

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
/// Bins of the two refining passes: eleven bits each.
const FINE_BINS: usize = 2048;
/// Bins a top-k thread owns in the coarse and the fine histograms.
const COARSE_PER: usize = HIST_BINS / TOPK_THREADS as usize;
const FINE_PER: usize = FINE_BINS / TOPK_THREADS as usize;
const _: () = assert!(COARSE_PER * TOPK_THREADS as usize == HIST_BINS);
const _: () = assert!(FINE_PER * TOPK_THREADS as usize == FINE_BINS);

/// Query heads of the f32-key score pass (`indexer_score_f32`).
pub const HEADS_F32: usize = 64;
/// Values at the front of the f32 pass's query heads and of its keys that
/// turn: NEOX pairs `(i, i + TURN_F32/2)`.
pub const TURN_F32: usize = 64;
/// Keys a block of the f32 pass stages, its threads (eight warps: four a key,
/// two keys a pass), the values a warp dots and the f32 the four warps of a
/// key exchange (`[parity][key][warp][lane][2]`, two passes in flight).
const F32_KEYS: usize = 32;
const F32_THREADS: u32 = 256;
const F32_DIMS: usize = HEAD_DIM / 4;
const F32_PART: usize = 2 * 2 * 4 * 32 * 2;
const _: () =
    assert!(HEADS_F32 == 64 && TURN_F32 / 2 == F32_DIMS && F32_THREADS == 256 && F32_PART == 1024);
/// Threads of the window pass: the most cells one token's window holds.
const WINDOW_THREADS: u32 = 128;

/// The weights' scale ik applies (`build_deepseek4.cpp`, the `lid_weights`
/// scale): `1/√(head_dim · n_head)`, the square root and the quotient each
/// rounded to f32.
#[must_use]
pub fn weights_scale(n_head: usize, head_dim: usize) -> f32 {
    1.0 / ((head_dim * n_head) as f32).sqrt()
}

// ------------------------------------------------------------------ cores

/// `x` where it is above zero, `+0` elsewhere (NaN included).
#[inline(always)]
fn relu(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.0 }
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
        // SAFETY: sbase + n <= scores.len() (n <= rows, the launch
        // contract); FINE is this pass's alone across the call.
        let fc = unsafe { refine(scores, sbase, n, b1, 11, fine, tid) };
        // SAFETY: WSUM and PICK are used by nothing else across the call.
        let (b2, above2) =
            unsafe { pick_bin::<FINE_PER>(fc, FINE_BINS as u32 - 1, need2, lane, tid, wsum, pick) };
        let need3 = need2 - above2;
        let prefix = (b1 << 11) | b2;

        // ---- the last eleven bits, among the keys with that prefix.
        // SAFETY: as the pass above.
        let fc = unsafe { refine(scores, sbase, n, prefix, 0, fine, tid) };
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

    /// The f32-key score pass (module doc): block `b` serves token `t = b /
    /// tiles`'s key tile `b % tiles`, the rows `F32_KEYS·tile ..` of `keys`
    /// (`rows` rows of [`HEAD_DIM`] f32). `q` is the query projection, raw,
    /// `[tokens × HEADS_F32 × HEAD_DIM]` token-major; `w` the weights'
    /// projection `[tokens × HEADS_F32]`, scaled by `scale` here; token `t`'s
    /// rope row is `tables[rope_at + t·rope_stride ..]` ([`TURN_F32`] f32,
    /// `[cos, sin, …]`). Writes `scores[t·rows + key]` for every row below
    /// `ints[n_vis_at + t]`, and adds into `hist[t·HIST_BINS ..]`; a count
    /// past the cache's rows raises [`FaultSite::KeyCount`] and scores
    /// nothing, a score that is not finite raises [`FaultSite::PoolSelect`].
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256, 1)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            q.len() >= tokens * 8192,
            w.len() >= tokens * 64,
            ints.len() >= n_vis_at + tokens,
            ints.len() > top_k_at,
            tables.len() + rope_stride >= rope_at + tokens * rope_stride + 64,
            keys.len() >= rows * 128,
            scores.len() >= tokens * rows,
            hist.len() >= tokens * 1024
        )
    )]
    pub fn indexer_score_f32(
        q: &[f32],
        w: &[f32],
        ints: &[u32],
        tables: &[f32],
        keys: &[f32],
        tokens: u32,
        tiles: u32,
        n_vis_at: u32,
        top_k_at: u32,
        rope_at: u32,
        rope_stride: u32,
        rows: u32,
        scale: f32,
        fault: FaultSink,
        mut scores: DisjointSlice<f32>,
        mut hist: DisjointSlice<u32>,
    ) {
        static mut KS: SharedArray<f32, { F32_KEYS * HEAD_DIM }> = SharedArray::UNINIT;
        static mut PART: SharedArray<f32, F32_PART> = SharedArray::UNINIT;
        static mut HS: SharedArray<u32, HIST_BINS> = SharedArray::UNINIT;

        let bid = thread::blockIdx_x() as usize;
        let tiles = tiles as usize;
        let _ = tokens;
        let (t, tile) = (bid / tiles, bid % tiles);
        let tid = thread::threadIdx_x() as usize;
        let lane = warp::lane_id() as usize;
        let wid = tid / 32;
        // SAFETY: t < tokens (the grid is tiles·tokens), so n_vis_at + t <
        // ints.len() by the launch contract.
        let n_vis = unsafe { *ints.get_unchecked(n_vis_at as usize + t) };
        if n_vis > rows {
            if tid == 0 {
                fault.raise(FaultSite::KeyCount);
            }
            return; // block-uniform: n_vis is the token's
        }
        // SAFETY: top_k_at < ints.len() by the launch contract.
        let k = unsafe { *ints.get_unchecked(top_k_at as usize) } as usize;
        let n = n_vis as usize;
        let key0 = tile * F32_KEYS;
        if n <= k || k == 0 || key0 >= n {
            return; // block-uniform: no barrier is skipped
        }
        let live = F32_KEYS.min(n - key0);
        // SAFETY: block-shared statics; the raw forms reach them without a
        // reference, and every access below is bounded and barrier-ordered.
        let (ks, part, hs) = unsafe {
            (
                SharedArray::as_raw_mut_ptr(&raw mut KS),
                SharedArray::as_raw_mut_ptr(&raw mut PART),
                SharedArray::as_raw_mut_ptr(&raw mut HS),
            )
        };
        let mut b = tid;
        while b < HIST_BINS {
            // SAFETY: b < HIST_BINS; thread tid alone zeroes bins tid + 256·i
            // before the barrier below.
            unsafe { *hs.add(b) = 0 };
            b += F32_THREADS as usize;
        }

        // ---- the query: warp `wid % 4` dots dims 32·(wid % 4) .., lane
        // `lane` heads 2·lane and 2·lane + 1, 32 values a head in registers;
        // the first TURN_F32 turned in NEOX pairs (i, i + TURN_F32/2) by the
        // token's table row, the lower member of a pair at dim i.
        let dq = F32_DIMS * (wid % 4);
        let qbase = t * HEADS_F32 * HEAD_DIM;
        let tab = rope_at as usize + t * rope_stride as usize;
        let mut qq = [[0.0f32; F32_DIMS]; 2];
        let mut i = 0usize;
        while i < F32_DIMS {
            thread::__unroll_config::<0>();
            let d = dq + i;
            let mut hi = 0usize;
            while hi < 2 {
                thread::__unroll_config::<0>();
                let at = qbase + (2 * lane + hi) * HEAD_DIM;
                qq[hi][i] = if d < TURN_F32 {
                    let p = d % (TURN_F32 / 2);
                    // SAFETY: 2·p + 1 < TURN_F32, so both lie inside token t's
                    // table row, below rope_at + t·rope_stride + TURN_F32 <=
                    // tables.len() (launch contract).
                    let (c, s) = unsafe {
                        (
                            *tables.get_unchecked(tab + 2 * p),
                            *tables.get_unchecked(tab + 2 * p + 1),
                        )
                    };
                    // SAFETY: at + p + TURN_F32/2 < qbase + HEADS_F32·HEAD_DIM
                    // <= q.len() (launch contract).
                    let (x0, x1) = unsafe {
                        (
                            *q.get_unchecked(at + p),
                            *q.get_unchecked(at + TURN_F32 / 2 + p),
                        )
                    };
                    let (y0, y1) = neox_pair(x0, x1, c, s);
                    if d < TURN_F32 / 2 { y0 } else { y1 }
                } else {
                    // SAFETY: at + d < qbase + HEADS_F32·HEAD_DIM <= q.len()
                    // (launch contract).
                    unsafe { *q.get_unchecked(at + d) }
                };
                hi += 1;
            }
            i += 1;
        }
        // SAFETY: 2·lane + 1 < HEADS_F32, so both indexes lie below
        // HEADS_F32·tokens <= w.len() (launch contract).
        let (w0, w1) = unsafe {
            (
                *w.get_unchecked(t * HEADS_F32 + 2 * lane),
                *w.get_unchecked(t * HEADS_F32 + 2 * lane + 1),
            )
        };
        let (w0, w1) = (mul_rn_f32(w0, scale), mul_rn_f32(w1, scale));

        // ---- the key tile, staged whole: rows key0 .. key0 + live of `keys`.
        let vals = live * HEAD_DIM;
        let mut at = tid;
        while at < vals {
            // SAFETY: at < live·HEAD_DIM, so key0·HEAD_DIM + at < n·HEAD_DIM
            // <= rows·HEAD_DIM <= keys.len() (launch contract); thread tid
            // alone writes ks[at], at = tid + 256·i, before the barrier below.
            unsafe { *ks.add(at) = *keys.get_unchecked(key0 * HEAD_DIM + at) };
            at += F32_THREADS as usize;
        }
        thread::sync_threads();

        // ---- the dots: warps 4·group .. 4·group + 3 share key `2·it + group`
        // of the pass, warp `wi` its dims 32·wi ..; the partials meet through
        // PART, double-buffered by the pass's parity, and warp 0 of the group
        // finishes the key: relu, the weights, the head sum, the histogram.
        let group = wid / 4;
        let wi = wid % 4;
        let sbase = t * rows as usize;
        let iters = live.div_ceil(2);
        let mut it = 0usize;
        while it < iters {
            let j = 2 * it + group;
            if j < live {
                let kbase = j * HEAD_DIM + F32_DIMS * wi;
                let (mut a0, mut a1) = (0.0f32, 0.0f32);
                let mut i = 0usize;
                while i < F32_DIMS {
                    thread::__unroll_config::<0>();
                    // SAFETY: block-shared, F32_KEYS·HEAD_DIM == the tile,
                    // written before the barrier above; kbase + i < j·HEAD_DIM
                    // + HEAD_DIM <= live·HEAD_DIM.
                    let kd = unsafe { *ks.add(kbase + i) };
                    a0 = fma_rn_f32(qq[0][i], kd, a0);
                    a1 = fma_rn_f32(qq[1][i], kd, a1);
                    i += 1;
                }
                let p = ((it & 1) * 2 + group) * 256 + wi * 64 + 2 * lane;
                // SAFETY: block-shared, F32_PART == 1024 words; p < ((it&1)·2
                // + group)·256 + 256 <= F32_PART, and this lane alone writes
                // its slot before the barrier below.
                unsafe { *part.add(p) = a0 };
                // SAFETY: p + 1 is this lane's second slot, as p above.
                unsafe { *part.add(p + 1) = a1 };
            }
            thread::sync_threads();
            if wi == 0 && j < live {
                let base = ((it & 1) * 2 + group) * 256 + 2 * lane;
                let (mut s0, mut s1) = (0.0f32, 0.0f32);
                let mut r = 0usize;
                while r < 4 {
                    thread::__unroll_config::<0>();
                    // SAFETY: block-shared, F32_PART == 1024 words; base +
                    // 64·r + 1 < ((it&1)·2 + group)·256 + 256 <= F32_PART,
                    // published by the barrier above.
                    let (p0, p1) =
                        unsafe { (*part.add(base + 64 * r), *part.add(base + 64 * r + 1)) };
                    s0 = add_rn_f32(s0, p0);
                    s1 = add_rn_f32(s1, p1);
                    r += 1;
                }
                // The head sum: the lane's pair as one add, then the xor butterfly over
                // all 32 lanes (16, 8, 4, 2, 1): every lane holds two distinct heads.
                let mut v = add_rn_f32(mul_rn_f32(relu(s0), w0), mul_rn_f32(relu(s1), w1));
                let mut lv = 0u32;
                while lv < 5 {
                    thread::__unroll_config::<0>();
                    v = add_rn_f32(v, warp::shuffle_xor_f32(v, 16u32 >> lv));
                    lv += 1;
                }
                if lane == 0 {
                    let key = key0 + j;
                    // SAFETY: key < n <= rows, so sbase + key < tokens·rows <=
                    // scores.len() (launch contract); lane 0 of the group's
                    // warp 0 alone writes the key's score.
                    unsafe { *scores.get_unchecked_mut(sbase + key) = v };
                    if v.is_finite() {
                        let bin = (order_key(v) >> 22) as usize;
                        // SAFETY: bin < HIST_BINS, block-shared; every access
                        // to the histogram between the two barriers is atomic.
                        unsafe {
                            BlockAtomicU32::from_ptr(hs.add(bin))
                                .fetch_add(1, AtomicOrdering::Relaxed)
                        };
                    } else {
                        fault.raise(FaultSite::PoolSelect);
                    }
                }
            }
            it += 1;
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
            b += F32_THREADS as usize;
        }
    }

    /// The forced window of a selection list (module doc): block `t` serves
    /// token `t`, whose candidates are the rows `0 .. n_vis =
    /// ints[n_vis_at + t]`, whose window the rows `n_vis .. n_win =
    /// ints[n_win_at + t]` and whose top-k `ints[top_k_at]`. Thread `i` writes
    /// the cell `i` of the window, `list[t·stride + min(n_vis, top_k) + i] =
    /// n_vis + i`, and thread 0 the list's live count `counts[t] = min(n_vis,
    /// top_k) + (n_win − n_vis)`. A window past the block's threads, a
    /// `n_vis` past `n_win` or a list too narrow for the top-k and the window
    /// raises [`FaultSite::KeyCount`] and writes nothing.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        requires = (
            ints.len() >= n_vis_at + tokens,
            ints.len() >= n_win_at + tokens,
            ints.len() > top_k_at,
            list.len() >= tokens * stride,
            counts.len() >= tokens
        )
    )]
    pub fn indexer_window(
        ints: &[u32],
        tokens: u32,
        n_vis_at: u32,
        n_win_at: u32,
        top_k_at: u32,
        stride: u32,
        fault: FaultSink,
        mut list: DisjointSlice<u32>,
        mut counts: DisjointSlice<u32>,
    ) {
        let t = thread::blockIdx_x() as usize;
        let i = thread::threadIdx_x();
        let _ = tokens;
        // SAFETY: t < tokens, so n_vis_at + t < ints.len() by the launch contract.
        let n_vis = unsafe { *ints.get_unchecked(n_vis_at as usize + t) };
        // SAFETY: t < tokens, so n_win_at + t < ints.len() by the launch contract.
        let n_win = unsafe { *ints.get_unchecked(n_win_at as usize + t) };
        // SAFETY: top_k_at < ints.len() by the launch contract.
        let top_k = unsafe { *ints.get_unchecked(top_k_at as usize) };
        let kept = n_vis.min(top_k);
        if n_vis > n_win || n_win - n_vis > WINDOW_THREADS || kept + (n_win - n_vis) > stride {
            if i == 0 {
                fault.raise(FaultSite::KeyCount);
            }
            return; // block-uniform: the counts are the token's
        }
        let cells = n_win - n_vis;
        if i < cells {
            // SAFETY: kept + i < kept + cells <= stride, so t·stride + kept +
            // i < tokens·stride <= list.len() (launch contract); thread i
            // alone writes the cell.
            unsafe {
                *list.get_unchecked_mut(t * stride as usize + kept as usize + i as usize) =
                    n_vis + i
            };
        }
        if i == 0 {
            // SAFETY: t < tokens <= counts.len() (launch contract); thread 0
            // alone writes the token's count.
            unsafe { *counts.get_unchecked_mut(t) = kept + cells };
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
    /// Load this file's device bundle into `ctx` for an indexer of `n_head`
    /// heads of `head_dim` (the kernels are built for [`HEADS`] of
    /// [`HEAD_DIM`]) and `n_dims` roped values per head: `0` for a layer that
    /// ropes none of them (the score and top-k passes run with no rotation),
    /// else a multiple of four at most a head. Load-time only.
    pub fn with_shape(
        ctx: &Arc<CudaContext>,
        n_head: usize,
        head_dim: usize,
        n_dims: usize,
    ) -> Result<IndexerKernels, GpuError> {
        let what = "IndexerKernels::with_shape";
        if n_head != HEADS || head_dim != HEAD_DIM {
            return Err(GpuError::Shape {
                what,
                detail: format!(
                    "the indexer has {n_head} heads of {head_dim}; the kernels are built for {HEADS} \
                     of {HEAD_DIM}"
                ),
            });
        }
        if !n_dims.is_multiple_of(PER_LANE) || n_dims > HEAD_DIM {
            return Err(GpuError::Shape {
                what,
                detail: format!(
                    "the query's rope turns {n_dims} values: 0, or a positive multiple of \
                     {PER_LANE}, at most {HEAD_DIM}"
                ),
            });
        }
        let blocks = usize::try_from(ctx.multiprocessor_count()?).map_err(|_| GpuError::Shape {
            what,
            detail: "the multiprocessor count passes usize".to_string(),
        })?;
        // SAFETY: this crate owns the embedded device bundle produced for the
        // module above; the launcher checks its launch contract.
        let module = unsafe { crate::shared_module!(indexer_kernels, ctx)? };
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

    /// Enqueue one indexer step: the score pass, then the top-k pass over the
    /// same counts. Two launches whose grids come from the token count and
    /// the device, not from the counts, so a captured graph replays them at
    /// any `n_vis` and `top_k`. Asynchronous, allocation-free, capturable.
    pub fn enqueue(&self, stream: &CudaStream, mut a: IndexerArgs<'_>) -> Result<(), GpuError> {
        self.enqueue_score(stream, &mut a)?;
        let (ints, n_vis_at, top_k_at) = (a.ints, a.n_vis_at, a.top_k_at);
        self.enqueue_topk(stream, &mut a, ints, n_vis_at, top_k_at)
    }

    /// The score pass of [`IndexerKernels::enqueue`]: every visible row's
    /// score into `a.scratch`, and the histogram the top-k pass reads.
    /// Asynchronous, allocation-free, capturable.
    pub fn enqueue_score(
        &self,
        stream: &CudaStream,
        a: &mut IndexerArgs<'_>,
    ) -> Result<(), GpuError> {
        let what = "IndexerKernels::enqueue_score";
        let shape = |detail: String| GpuError::Shape { what, detail };
        let (tokens, keys) = (a.tokens, a.keys);
        let rows = keys.rows();
        if tokens == 0 || keys.cols() != HEAD_DIM || rows == 0 {
            return Err(shape(format!(
                "tokens {tokens} and keys {rows}x{}: at least one token, and rows of {HEAD_DIM}",
                keys.cols()
            )));
        }
        check_scratch(what, a.scratch, tokens, rows)?;
        let last_table = a.rope_at + (tokens - 1) * a.rope_stride + self.n_dims;
        for (name, have, want) in [
            ("q", a.q.len(), tokens * HEADS * HEAD_DIM),
            ("w", a.w.len(), tokens * HEADS),
            ("ints (n_vis)", a.ints.len(), a.n_vis_at + tokens),
            ("ints (top_k)", a.ints.len(), a.top_k_at + 1),
            ("tables", a.tables.len(), last_table),
        ] {
            if have < want {
                return Err(shape(format!("{name} holds {have}, want >= {want}")));
            }
        }
        let grid = launch_u32(what, "score grid", self.blocks * tokens)?;
        let prep =
            self.module
                .prepare_ds41_indexer_score(LaunchConfig1D::new(grid, SCORE_THREADS, 0))?;
        let scratch = &mut *a.scratch;
        self.module.ds41_indexer_score(
            stream,
            &prep,
            a.q,
            a.w,
            a.ints,
            a.tables,
            keys.buf(),
            launch_u32(what, "tokens", tokens)?,
            launch_u32(what, "blocks", self.blocks)?,
            launch_u32(what, "n_vis_at", a.n_vis_at)?,
            launch_u32(what, "top_k_at", a.top_k_at)?,
            launch_u32(what, "rope_at", a.rope_at)?,
            launch_u32(what, "rope_stride", a.rope_stride)?,
            launch_u32(what, "n_dims", self.n_dims)?,
            launch_u32(what, "rows", rows)?,
            launch_u32(what, "stride", a.stride)?,
            self.scale,
            &mut scratch.q,
            &mut scratch.w,
            &mut scratch.scores,
            &mut scratch.hist,
        )?;
        Ok(())
    }

    /// The top-k pass of [`IndexerKernels::enqueue`] over the scores and
    /// histogram in `a.scratch`, into `a.list`, reading token `t`'s visible
    /// count at `ints[n_vis_at + t]` and `top_k` at `ints[top_k_at]`: the
    /// args' own words, or the counts view a consumer's `cand_compact`
    /// (`bloomery_gpu::cand`) wrote. Asynchronous, allocation-free,
    /// capturable.
    pub fn enqueue_topk(
        &self,
        stream: &CudaStream,
        a: &mut IndexerArgs<'_>,
        ints: &DeviceBuffer<u32>,
        n_vis_at: usize,
        top_k_at: usize,
    ) -> Result<(), GpuError> {
        let what = "IndexerKernels::enqueue_topk";
        let shape = |detail: String| GpuError::Shape { what, detail };
        let (tokens, rows) = (a.tokens, a.keys.rows());
        if tokens == 0 || rows == 0 {
            return Err(shape(format!(
                "a top-k of {tokens} tokens over {rows} rows: at least one of each"
            )));
        }
        check_scratch(what, a.scratch, tokens, rows)?;
        for (name, have, want) in [
            ("ints (n_vis)", ints.len(), n_vis_at + tokens),
            ("ints (top_k)", ints.len(), top_k_at + 1),
            ("list", a.list.len(), tokens * a.stride),
        ] {
            if have < want {
                return Err(shape(format!("{name} holds {have}, want >= {want}")));
            }
        }
        let tokens_u = launch_u32(what, "tokens", tokens)?;
        let prep = self.module.prepare_ds41_indexer_topk(LaunchConfig1D::new(
            tokens_u,
            TOPK_THREADS,
            0,
        ))?;
        let scratch = &mut *a.scratch;
        self.module.ds41_indexer_topk(
            stream,
            &prep,
            ints,
            &scratch.scores,
            tokens_u,
            launch_u32(what, "n_vis_at", n_vis_at)?,
            launch_u32(what, "top_k_at", top_k_at)?,
            launch_u32(what, "rows", rows)?,
            launch_u32(what, "stride", a.stride)?,
            &mut scratch.hist,
            &mut *a.list,
        )?;
        Ok(())
    }
}

/// `scratch` against a launch of `tokens` tokens over `rows` key rows.
fn check_scratch(
    what: &'static str,
    scratch: &IndexerScratch,
    tokens: usize,
    rows: usize,
) -> Result<(), GpuError> {
    if scratch.tokens != tokens || scratch.rows != rows {
        return Err(GpuError::Shape {
            what,
            detail: format!(
                "the scratch serves {} tokens over {} rows, the launch {tokens} over {rows}",
                scratch.tokens, scratch.rows
            ),
        });
    }
    Ok(())
}

/// [`IndexerF32Kernels::enqueue_score`]'s arguments: the query projection `q`
/// (`[tokens × HEADS_F32 × HEAD_DIM]` f32, raw, token-major), the weights'
/// projection `w` (`[tokens × HEADS_F32]`, before the scale), the counts
/// `ints` (token `t`'s `n_vis` at `ints[n_vis_at + t]`, `top_k` at
/// `ints[top_k_at]`), the rope table `tables` (token `t`'s row at `rope_at +
/// t·rope_stride`, [`TURN_F32`] f32), the key cache `keys` (`[rows ×
/// HEAD_DIM]` f32), the scores `[tokens × rows]` and the histogram `[tokens ×
/// HIST_BINS]` (zeroed, and zero again once the top-k pass has read it).
pub struct ScoreF32Args<'a> {
    pub q: &'a DeviceBuffer<f32>,
    pub w: &'a DeviceBuffer<f32>,
    pub ints: &'a DeviceBuffer<u32>,
    pub n_vis_at: usize,
    pub top_k_at: usize,
    pub tables: &'a DeviceBuffer<f32>,
    pub rope_at: usize,
    pub rope_stride: usize,
    pub keys: &'a DeviceTensor<f32>,
    pub tokens: usize,
    pub scores: &'a mut DeviceBuffer<f32>,
    pub hist: &'a mut DeviceBuffer<u32>,
    pub fault: FaultSink,
}

/// [`IndexerF32Kernels::enqueue_topk`]'s arguments: the top-k pass
/// (`ds41_indexer_topk`, the module doc's first paragraph) over `scores`
/// (`[tokens × rows]`) into `list` (`[tokens × stride]`), token `t`'s live
/// rows at `ints[n_vis_at + t]` and `top_k` at `ints[top_k_at]`, the
/// histogram the score pass left.
pub struct TopkArgs<'a> {
    pub ints: &'a DeviceBuffer<u32>,
    pub n_vis_at: usize,
    pub top_k_at: usize,
    pub scores: &'a DeviceBuffer<f32>,
    pub tokens: usize,
    pub rows: usize,
    pub stride: usize,
    pub hist: &'a mut DeviceBuffer<u32>,
    pub list: &'a mut DeviceBuffer<u32>,
}

/// [`IndexerF32Kernels::enqueue_window`]'s arguments: token `t`'s `n_vis` at
/// `ints[n_vis_at + t]`, `n_win` at `ints[n_win_at + t]` and `top_k` at
/// `ints[top_k_at]`; the list `[tokens × stride]` the top-k pass wrote the
/// front of and the counts `[tokens]`.
pub struct WindowArgs<'a> {
    pub ints: &'a DeviceBuffer<u32>,
    pub n_vis_at: usize,
    pub n_win_at: usize,
    pub top_k_at: usize,
    pub tokens: usize,
    pub stride: usize,
    pub fault: FaultSink,
    pub list: &'a mut DeviceBuffer<u32>,
    pub counts: &'a mut DeviceBuffer<u32>,
}

/// The f32-key indexer's kernels (module doc): the score pass over
/// [`HEADS_F32`] heads with the query's first [`TURN_F32`] values turned, the
/// top-k pass and the forced window. Owns no stream: each enqueue takes the
/// engine stream, so its launches order with the rest of the step and
/// capture.
pub struct IndexerF32Kernels {
    module: indexer_kernels::LoadedModule,
    scale: f32,
}

impl IndexerF32Kernels {
    /// Load this file's device bundle into `ctx` for the f32-key indexer.
    /// Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<IndexerF32Kernels, GpuError> {
        // SAFETY: this crate owns the embedded device bundle produced for the
        // module above; the launchers check its launch contracts.
        let module = unsafe { crate::shared_module!(indexer_kernels, ctx)? };
        Ok(IndexerF32Kernels {
            module,
            scale: weights_scale(HEADS_F32, HEAD_DIM),
        })
    }

    /// The weights' scale the score pass applies, `1/√(HEADS_F32 · HEAD_DIM)`.
    #[must_use]
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Enqueue the f32-key score pass (module doc): the grid comes from the
    /// key cache's rows and the token count, so a captured graph replays it at
    /// any `n_vis`. Refused before the launch: no token, a key cache of no
    /// rows or not [`HEAD_DIM`] wide, short `q`, `w`, `ints`, `tables`,
    /// `scores` or `hist`. Asynchronous, allocation-free, capturable.
    pub fn enqueue_score(&self, stream: &CudaStream, a: ScoreF32Args<'_>) -> Result<(), GpuError> {
        let what = "IndexerF32Kernels::enqueue_score";
        let shape = |detail: String| GpuError::Shape { what, detail };
        let rows = a.keys.rows();
        if a.tokens == 0 || a.keys.cols() != HEAD_DIM || rows == 0 {
            return Err(shape(format!(
                "tokens {} and keys {}x{}: at least one token, and rows of {HEAD_DIM}",
                a.tokens,
                rows,
                a.keys.cols()
            )));
        }
        let last_table = a.rope_at + (a.tokens - 1) * a.rope_stride + TURN_F32;
        for (name, have, want) in [
            ("q", a.q.len(), a.tokens * HEADS_F32 * HEAD_DIM),
            ("w", a.w.len(), a.tokens * HEADS_F32),
            ("ints (n_vis)", a.ints.len(), a.n_vis_at + a.tokens),
            ("ints (top_k)", a.ints.len(), a.top_k_at + 1),
            ("tables", a.tables.len(), last_table),
            ("scores", a.scores.len(), a.tokens * rows),
            ("hist", a.hist.len(), a.tokens * HIST_BINS),
        ] {
            if have < want {
                return Err(shape(format!("{name} holds {have}, want >= {want}")));
            }
        }
        let grid = launch_u32(what, "grid", rows.div_ceil(F32_KEYS) * a.tokens)?;
        let tiles = launch_u32(what, "tiles", rows.div_ceil(F32_KEYS))?;
        let prep =
            self.module
                .prepare_indexer_score_f32(LaunchConfig1D::new(grid, F32_THREADS, 0))?;
        self.module.indexer_score_f32(
            stream,
            &prep,
            a.q,
            a.w,
            a.ints,
            a.tables,
            a.keys.buf(),
            launch_u32(what, "tokens", a.tokens)?,
            tiles,
            launch_u32(what, "n_vis_at", a.n_vis_at)?,
            launch_u32(what, "top_k_at", a.top_k_at)?,
            launch_u32(what, "rope_at", a.rope_at)?,
            launch_u32(what, "rope_stride", a.rope_stride)?,
            launch_u32(what, "rows", rows)?,
            self.scale,
            a.fault,
            &mut *a.scores,
            &mut *a.hist,
        )?;
        Ok(())
    }

    /// Enqueue the top-k pass over the scores the f32-key pass left (module
    /// doc): `min(n_vis, top_k)` rows a token, ascending, ties to the lower
    /// row, the identity when `n_vis <= top_k`. Refused before the launch: no
    /// token or row, short `ints`, `list` or `hist`. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_topk(&self, stream: &CudaStream, a: TopkArgs<'_>) -> Result<(), GpuError> {
        let what = "IndexerF32Kernels::enqueue_topk";
        let shape = |detail: String| GpuError::Shape { what, detail };
        if a.tokens == 0 || a.rows == 0 {
            return Err(shape(format!(
                "a top-k of {} tokens over {} rows: at least one of each",
                a.tokens, a.rows
            )));
        }
        for (name, have, want) in [
            ("ints (n_vis)", a.ints.len(), a.n_vis_at + a.tokens),
            ("ints (top_k)", a.ints.len(), a.top_k_at + 1),
            ("scores", a.scores.len(), a.tokens * a.rows),
            ("hist", a.hist.len(), a.tokens * HIST_BINS),
            ("list", a.list.len(), a.tokens * a.stride),
        ] {
            if have < want {
                return Err(shape(format!("{name} holds {have}, want >= {want}")));
            }
        }
        let tokens = launch_u32(what, "tokens", a.tokens)?;
        let prep =
            self.module
                .prepare_ds41_indexer_topk(LaunchConfig1D::new(tokens, TOPK_THREADS, 0))?;
        self.module.ds41_indexer_topk(
            stream,
            &prep,
            a.ints,
            a.scores,
            tokens,
            launch_u32(what, "n_vis_at", a.n_vis_at)?,
            launch_u32(what, "top_k_at", a.top_k_at)?,
            launch_u32(what, "rows", a.rows)?,
            launch_u32(what, "stride", a.stride)?,
            &mut *a.hist,
            &mut *a.list,
        )?;
        Ok(())
    }

    /// Enqueue the forced window after the top-k (module doc): one block a
    /// token. Refused before the launch: no token or stride, short `ints`,
    /// `list` or `counts`; a window wider than the block's threads raises
    /// [`FaultSite::KeyCount`] on the device. Asynchronous, allocation-free,
    /// capturable.
    pub fn enqueue_window(&self, stream: &CudaStream, a: WindowArgs<'_>) -> Result<(), GpuError> {
        let what = "IndexerF32Kernels::enqueue_window";
        let shape = |detail: String| GpuError::Shape { what, detail };
        if a.tokens == 0 || a.stride == 0 {
            return Err(shape(format!(
                "need tokens and stride >= 1, got tokens={} stride={}",
                a.tokens, a.stride
            )));
        }
        for (name, have, want) in [
            ("ints (n_vis)", a.ints.len(), a.n_vis_at + a.tokens),
            ("ints (n_win)", a.ints.len(), a.n_win_at + a.tokens),
            ("ints (top_k)", a.ints.len(), a.top_k_at + 1),
            ("list", a.list.len(), a.tokens * a.stride),
            ("counts", a.counts.len(), a.tokens),
        ] {
            if have < want {
                return Err(shape(format!("{name} holds {have}, want >= {want}")));
            }
        }
        let tokens = launch_u32(what, "tokens", a.tokens)?;
        let prep =
            self.module
                .prepare_indexer_window(LaunchConfig1D::new(tokens, WINDOW_THREADS, 0))?;
        self.module.indexer_window(
            stream,
            &prep,
            a.ints,
            tokens,
            launch_u32(what, "n_vis_at", a.n_vis_at)?,
            launch_u32(what, "n_win_at", a.n_win_at)?,
            launch_u32(what, "top_k_at", a.top_k_at)?,
            launch_u32(what, "stride", a.stride)?,
            a.fault,
            &mut *a.list,
            &mut *a.counts,
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::weights_scale;

    /// The weights' scale for the file's indexer is exactly 2⁻⁶.
    #[test]
    fn weights_scale_is_a_power_of_two() {
        assert_eq!(weights_scale(32, 128).to_bits(), (1.0f32 / 64.0).to_bits());
    }
}
