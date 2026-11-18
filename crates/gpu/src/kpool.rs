//! GLM-5.3-Flash's k-pool indexer scores on the card: for every query row
//! that selects, the score of each complete pool it sees, from the pool
//! plane `latent::index_pool` writes. The top-k that turns them into the
//! token list the selected attention reads is `qsa::qsa_topk_high`.
//!
//! The two projections are the caller's gemvs, each in the gemvs' `m`-column
//! output layout, feature-major: value `i` of token `t` of a launch of `tk`
//! tokens is word `i · tk + t`, so `q[(h · DIM + v) · tk + t]` and
//! `w[h · tk + t]` (a token-major GEMM output, word `t · width + i`, is
//! transposed first: `crate::token_major`). `indexer.attn_q_b` (q8_0) of the
//! attention's normed low-rank query gives `q`, [`HEADS`] heads of [`DIM`];
//! `indexer.proj` (f32) of the attention's normed input gives `w`, one
//! weight a head. The indexer has no rope and no transform: the query heads
//! are used as they come.
//!
//! [`kpool_kernels::kpool_score`] — per token, every block first stages the
//! query: each value split into `hi = f16(q)` and `lo = f16(q − hi)` (`q −
//! hi` is exact in f32). The weights are `w · scale`, `scale = 1/√(HEADS ·
//! DIM)` as ik forms it ([`weights_scale`]). Each warp then walks 16-pool
//! tiles on the tensor cores: per head the exact f16 products `hi·k` and
//! `lo·k` are summed into one f32 accumulator row each (`mma.m16n8k16`,
//! eight k-steps), the dot is `acc_hi + acc_lo`, `relu` keeps `x > 0` and
//! gives `+0` otherwise, and the head sum is fixed: lane group `g` chains
//! heads `g, 8+g, 16+g, 24+g` (a product, then three fused multiply-adds),
//! and the eight groups meet in the xor butterfly over lanes 4, 8, 16 —
//! `ds41_indexer_score`'s arithmetic on a query with no rope and no
//! transform. A pool's score depends on its token's query and weights and
//! on the pool alone, in one fixed order, so a token scores the same bits
//! in a launch of one token and of many.
//!
//! A row is named by its live count `c` (its position plus one): it sees the
//! `c / 4` complete pools, and it is scored only when it sees more than
//! `kept` of them ([`crate::qsa::scored`]): the one predicate the top-k pass
//! reads its scores by. The grid depends on the token count and the plane's
//! height alone; work past a row's pools exits early, so a captured step
//! replays at any depth.
//!
//! No silent failure: a head's dot that is not finite (a non-finite query
//! or pool key; `relu` would make it `+0`) and a score that is not finite (a
//! non-finite weight too) raise [`FaultSite::PoolSelect`]; the score is
//! written as computed, and the top-k still writes a defined list. A pool
//! past the row's is read as zeros, so its dot is finite for a finite query.
//!
//! [`kpool_kernels::kpool_score_tail`] scores Hy4-flash's selector
//! (`attention.indexer.kpool_select_tail`) by the same arithmetic, with three
//! differences an architecture fact forces:
//! - the head count is a launch argument, a multiple of eight up to
//!   [`MAX_HEADS`] (Hy4's 16): `heads / 8` m-tiles, lane group `g` chaining
//!   heads `g, 8+g, …` below it, and each warp staging `heads / 8` heads;
//! - the query's last [`ROT`] values of each head turn at the row's position
//!   before the split, NORM mode (pairs `(2i, 2i + 1)` of the tail, through
//!   [`crate::rope_table::rope_pair_rn`] by table pair `i` of row `c − 1` of
//!   the layer's rope table, `[ctx][ROT]` f32, [`crate::rope_table::RopeRows`]'
//!   layout);
//! - a row of live count `c` sees the `(c − 1) / 4` pools that end before its
//!   position and is scored when that is more than `kept`
//!   ([`crate::qsa::scored_tail`]): its own pool is the top-k's tail, scored
//!   or not.

use crate::fault::{FaultSink, FaultSite};
use crate::latent::{INDEX_HEAD, POOL, ROT, pools_for};
use crate::qsa::{scored, scored_tail};
use crate::rope_table::rope_pair_rn;
use crate::tensor::DeviceTensor;
use crate::{GpuError, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::convert::{cvt_f16x2_f32, cvt_f32x2_f16x2};
use cuda_device::float::{add_rn_f32, fma_rn_f32, mul_rn_f32};
use cuda_device::shared::cvta_generic_to_shared_u32;
use cuda_device::wmma::{ldmatrix_x4_shared_u32, mma_m16n8k16_f32_f16};
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread, warp,
};
use cuda_host::cuda_module;
use std::sync::Arc;

/// GLM-5.3-Flash's indexer query heads (`attention.indexer.head_count`):
/// [`kpool_kernels::kpool_score`]'s, fixed at compile time.
pub const HEADS: usize = 32;
/// Values of a query head and of a pool key (`attention.indexer.key_length`).
pub const DIM: usize = INDEX_HEAD;

/// Warps of a score block, and its threads.
const SCORE_WARPS: usize = 8;
const SCORE_THREADS: u32 = 256;
const _: () = assert!(SCORE_WARPS * 32 == SCORE_THREADS as usize);
/// Heads one warp stages in the block's prologue.
const HEADS_PER_WARP: usize = HEADS / SCORE_WARPS;
const _: () = assert!(HEADS_PER_WARP * SCORE_WARPS == HEADS);
/// Query values each lane stages per head.
const PER_LANE: usize = DIM / 32;
const _: () = assert!(PER_LANE == 4 && DIM == 128);
/// Pools one warp scores per step: two `mma` n-tiles of eight.
const TILE_POOLS: usize = 16;
/// `mma` m-tiles over the heads: tile `i` holds heads `8i ..`, their `hi`
/// rows first and their `lo` rows after.
const M_TILES: usize = HEADS / 8;
/// `mma` k-steps over a head: sixteen dims each.
const K_STEPS: usize = DIM / 16;
/// u32 words between two rows of the staged query tile: 64 words of f16
/// pairs, padded by four so that the eight rows of one `ldmatrix` phase
/// cover the thirty-two banks once.
const Q_ROW_WORDS: usize = DIM / 2 + 4;
const _: () = assert!(Q_ROW_WORDS * 4 % 128 == 16);
/// Rows of the staged query tile: `hi` and `lo` of every head.
const Q_ROWS: usize = 2 * HEADS;
/// u32 words of the staged query tile.
const Q_WORDS: usize = Q_ROWS * Q_ROW_WORDS;
/// u32 words of one pool key.
const KEY_WORDS: usize = DIM / 2;
const _: () = assert!(POOL == 4);

/// The most query heads [`kpool_kernels::kpool_score_tail`] takes: the staged
/// tile's rows and the m-tiles a lane walks are sized for it.
pub const MAX_HEADS: usize = HEADS;
/// Heads one warp stages at most, and m-tiles at most.
const MAX_HPW: usize = MAX_HEADS / SCORE_WARPS;
const MAX_M_TILES: usize = MAX_HEADS / 8;
const _: () =
    assert!(ROT.is_multiple_of(PER_LANE) && ROT <= DIM && MAX_HPW == 4 && MAX_M_TILES == 4);

/// The weights' scale ik applies (`build_glm5next.cpp`, `dsa_indexer_weights`):
/// `1/√(head_dim · n_head)`, the square root and the quotient each rounded
/// to f32 — `1/64` exactly for 32 heads of 128.
#[must_use]
pub fn weights_scale(n_head: usize, head_dim: usize) -> f32 {
    1.0 / ((head_dim * n_head) as f32).sqrt()
}

/// Score blocks per token for a plane of `pools` rows: enough for every
/// 16-pool tile of a row that sees all of them, eight tiles a block.
#[must_use]
pub fn blocks_for(pools: usize) -> usize {
    pools.div_ceil(TILE_POOLS * SCORE_WARPS).max(1)
}

/// `x` where it is above zero, `+0` elsewhere (NaN included).
#[inline(always)]
fn relu(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.0 }
}

#[cuda_module]
mod kpool_kernels {
    use super::*;

    /// The score pass: block `b` serves token `t = b / blocks`, and its
    /// warps walk that token's 16-pool tiles `j = (b % blocks)·8 + warp`,
    /// then every `8·blocks` after it (module doc). `q` is the query
    /// projection, `[HEADS·DIM × tokens]`; `w` the weights' projection,
    /// `[HEADS × tokens]`; `n_keys` the tokens' live counts; `pooled` the
    /// pool plane, `pools` rows of [`DIM`] f16. Writes `scores[t·pools + j]`
    /// for the `c / 4` pools of a [`scored`] row and nothing else.
    #[kernel]
    #[launch_bounds(256, 1)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            q.len() >= tokens * 4096,
            w.len() >= tokens * 32,
            n_keys.len() >= tokens,
            pools * 4 >= ctx,
            pooled.len() >= pools * 128,
            scores.len() >= tokens * pools
        )
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    pub fn kpool_score(
        q: &[f32],
        w: &[f32],
        n_keys: &[u32],
        pooled: &[u16],
        tokens: u32,
        blocks: u32,
        ctx: u32,
        pools: u32,
        kept: u32,
        scale: f32,
        fault: FaultSink,
        mut scores: DisjointSlice<f32>,
    ) {
        // The query tile, `hi` and `lo` rows of every head as f16 pairs.
        static mut QT: SharedArray<u32, Q_WORDS> = SharedArray::UNINIT;

        let bid = thread::blockIdx_x();
        let t = (bid / blocks) as usize;
        let gb = (bid % blocks) as usize;
        let tid = thread::threadIdx_x() as usize;
        let lane = warp::lane_id();
        let wid = tid / 32;
        // SAFETY: t < tokens (the grid is blocks · tokens), so t <
        // n_keys.len() by the launch contract.
        let c = unsafe { *n_keys.get_unchecked(t) };
        if !scored(c, ctx, kept) {
            return; // block-uniform: no barrier is skipped
        }
        let n = c as usize / POOL;
        let tiles = n.div_ceil(TILE_POOLS);
        if gb * SCORE_WARPS >= tiles {
            return; // block-uniform
        }

        // SAFETY: the block-shared static; the raw form reaches it without a
        // reference, and every access below is bounded and barrier-ordered.
        let qt = unsafe { SharedArray::as_raw_mut_ptr(&raw mut QT) };

        // ---- the query: warp `wid` stages heads 4·wid .. 4·wid + 3, lane
        // `lane` values `4·lane ..` of each.
        let tk = tokens as usize;
        let v = PER_LANE * lane as usize;
        let mut j = 0usize;
        while j < HEADS_PER_WARP {
            thread::__unroll_config::<0>();
            let h = wid * HEADS_PER_WARP + j;
            let mut y = [0.0f32; PER_LANE];
            let mut i = 0usize;
            while i < PER_LANE {
                thread::__unroll_config::<0>();
                // SAFETY: h < HEADS and v + i < DIM, so the index is below
                // HEADS·DIM·tokens <= q.len() (launch contract).
                y[i] = unsafe { *q.get_unchecked((h * DIM + v + i) * tk + t) };
                i += 1;
            }
            let hi01 = cvt_f16x2_f32(y[0], y[1]);
            let hi23 = cvt_f16x2_f32(y[2], y[3]);
            let (h0, h1) = cvt_f32x2_f16x2(hi01);
            let (h2, h3) = cvt_f32x2_f16x2(hi23);
            let lo01 = cvt_f16x2_f32(y[0] - h0, y[1] - h1);
            let lo23 = cvt_f16x2_f32(y[2] - h2, y[3] - h3);
            let row_hi = 16 * (h / 8) + h % 8;
            let at = v / 2;
            // SAFETY: row_hi + 8 < Q_ROWS and at + 1 < DIM / 2 < Q_ROW_WORDS
            // keep all four stores inside the tile; this lane alone writes
            // these words, before the barrier below.
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
        let cl = (lane % 4) as usize;
        let mut wr = [0.0f32; M_TILES];
        let mut i = 0usize;
        while i < M_TILES {
            thread::__unroll_config::<0>();
            // SAFETY: 8i + g < HEADS, so the index is below HEADS·tokens <=
            // w.len() (launch contract).
            wr[i] = mul_rn_f32(unsafe { *w.get_unchecked((8 * i + g) * tk + t) }, scale);
            i += 1;
        }
        thread::sync_threads();

        // ---- the pools: B fragments straight from the plane's rows. Lane
        // (g, cl) holds pool g of each n-tile, dims 16s + 2cl, +1 and 16s +
        // 2cl + 8, +9 of k-step s — words 8s + cl and 8s + 4 + cl of its
        // row. A row at or past n is not read and stays zero.
        let kw = pooled.as_ptr().cast::<u32>();
        // SAFETY: `qt` is this block's shared query tile, a generic address
        // into shared memory, which is what the conversion takes.
        let qbase = unsafe { cvta_generic_to_shared_u32(qt.cast_const().cast::<u8>()) };
        // `ldmatrix` lane roles: row lane % 16 of the m-tile, dims half
        // lane / 16 of the k-step.
        let a_lane = (lane as usize % 16) * Q_ROW_WORDS + 4 * (lane as usize / 16);
        let stride_tiles = blocks as usize * SCORE_WARPS;
        let sbase = t * pools as usize;
        let mut tile = gb * SCORE_WARPS + wid;
        while tile < tiles {
            let key0 = tile * TILE_POOLS;
            let r0 = key0 + g;
            let r1 = r0 + 8;
            let (live0, live1) = (r0 < n, r1 < n);
            let mut b0 = [0u32; 2 * K_STEPS];
            let mut b1 = [0u32; 2 * K_STEPS];
            let mut s = 0usize;
            while s < K_STEPS {
                thread::__unroll_config::<0>();
                if live0 {
                    // SAFETY: r0 < n <= pools and 8s + 4 + cl < KEY_WORDS put
                    // both words inside row r0 of the plane, pools·128 f16
                    // (launch contract), device-allocated and so aligned.
                    unsafe {
                        b0[2 * s] = *kw.add(r0 * KEY_WORDS + 8 * s + cl);
                        b0[2 * s + 1] = *kw.add(r0 * KEY_WORDS + 8 * s + 4 + cl);
                    }
                }
                if live1 {
                    // SAFETY: as the read above, for row r1 < n.
                    unsafe {
                        b1[2 * s] = *kw.add(r1 * KEY_WORDS + 8 * s + cl);
                        b1[2 * s + 1] = *kw.add(r1 * KEY_WORDS + 8 * s + 4 + cl);
                    }
                }
                s += 1;
            }
            // Per lane: pools 2cl and 2cl + 1 of n-tile 0, then of n-tile 1.
            // A head's dot that is not finite is caught before `relu`, which
            // would make it `+0`.
            let mut p = [0.0f32; 4];
            let mut finite = true;
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
                    // / 16) .. + 4 <= DIM / 2 are inside the tile, which the
                    // barrier above published; every lane of the warp reaches
                    // this load with the same qualifiers.
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
                // its `lo` part, at pools 2cl and 2cl + 1.
                let d = [
                    add_rn_f32(c0[0], c0[2]),
                    add_rn_f32(c0[1], c0[3]),
                    add_rn_f32(c1[0], c1[2]),
                    add_rn_f32(c1[1], c1[3]),
                ];
                finite &= d[0].is_finite() & d[1].is_finite() & d[2].is_finite() & d[3].is_finite();
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
            if !finite {
                fault.raise(FaultSite::PoolSelect);
            }
            // Lanes g < 4 publish one score each: pool key0 + 8·(g / 2) + 2cl
            // + g % 2.
            if g < 4 {
                let key = key0 + 8 * (g / 2) + 2 * cl + g % 2;
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
                    if !sv.is_finite() {
                        fault.raise(FaultSite::PoolSelect);
                    }
                    // SAFETY: key < n <= pools, so sbase + key < tokens·pools <=
                    // scores.len() (launch contract); one lane of one warp
                    // scores each pool.
                    unsafe { *scores.get_unchecked_mut(sbase + key) = sv };
                }
            }
            tile += stride_tiles;
        }
    }

    /// The forced-tail score pass (module doc): [`kpool_score`] over `heads`
    /// query heads (a multiple of eight, at most [`MAX_HEADS`]), the query's
    /// last [`ROT`] values of each head turned by row `c − 1` of `table` (a
    /// row of [`ROT`] f32 per cache position, `ctx` rows) before the split,
    /// and a row of live count `c` scoring the `(c − 1) / 4` pools that end
    /// before its position when it is [`scored_tail`]. `q` is `[heads·DIM ×
    /// tokens]`, `w` `[heads × tokens]`; writes `scores[t·pools + j]` for
    /// those pools and nothing else.
    #[kernel]
    #[launch_bounds(256, 1)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            32 >= heads,
            q.len() >= tokens * heads * 128,
            w.len() >= tokens * heads,
            n_keys.len() >= tokens,
            table.len() >= ctx * 64,
            pools * 4 >= ctx,
            pooled.len() >= pools * 128,
            scores.len() >= tokens * pools
        )
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    pub fn kpool_score_tail(
        q: &[f32],
        w: &[f32],
        n_keys: &[u32],
        table: &[f32],
        pooled: &[u16],
        tokens: u32,
        blocks: u32,
        heads: u32,
        ctx: u32,
        pools: u32,
        kept: u32,
        scale: f32,
        fault: FaultSink,
        mut scores: DisjointSlice<f32>,
    ) {
        // The query tile, `hi` and `lo` rows of every head as f16 pairs.
        static mut QT: SharedArray<u32, Q_WORDS> = SharedArray::UNINIT;

        let bid = thread::blockIdx_x();
        let t = (bid / blocks) as usize;
        let gb = (bid % blocks) as usize;
        let tid = thread::threadIdx_x() as usize;
        let lane = warp::lane_id();
        let wid = tid / 32;
        // SAFETY: t < tokens (the grid is blocks · tokens), so t <
        // n_keys.len() by the launch contract.
        let c = unsafe { *n_keys.get_unchecked(t) };
        if !scored_tail(c, ctx, kept) {
            return; // block-uniform: no barrier is skipped
        }
        let n = (c as usize - 1) / POOL;
        let tiles = n.div_ceil(TILE_POOLS);
        if gb * SCORE_WARPS >= tiles {
            return; // block-uniform
        }
        // Heads a warp stages, and the m-tiles: one per eight heads.
        let m_tiles = heads as usize / 8;

        // SAFETY: the block-shared static; the raw form reaches it without a
        // reference, and every access below is bounded and barrier-ordered.
        let qt = unsafe { SharedArray::as_raw_mut_ptr(&raw mut QT) };

        // ---- the query: warp `wid` stages heads `m_tiles·wid ..` (`m_tiles`
        // of them), lane `lane` values `4·lane ..` of each; lanes past `DIM −
        // ROT` turn theirs by the row's table pairs.
        let tk = tokens as usize;
        let v = PER_LANE * lane as usize;
        let tail0 = DIM - ROT;
        let roped = v >= tail0;
        let mut cs = [0.0f32; PER_LANE];
        if roped {
            let at = (c as usize - 1) * ROT + v - tail0;
            let mut i = 0usize;
            while i < PER_LANE {
                thread::__unroll_config::<0>();
                // SAFETY: c − 1 < ctx and v − tail0 + i < ROT put the word
                // inside row c − 1 of the table, below ctx·ROT <= table.len()
                // (launch contract).
                cs[i] = unsafe { *table.get_unchecked(at + i) };
                i += 1;
            }
        }
        let mut j = 0usize;
        while j < MAX_HPW {
            thread::__unroll_config::<0>();
            if j < m_tiles {
                let h = wid * m_tiles + j;
                let mut y = [0.0f32; PER_LANE];
                let mut i = 0usize;
                while i < PER_LANE {
                    thread::__unroll_config::<0>();
                    // SAFETY: h < 8·m_tiles = heads and v + i < DIM, so the
                    // index is below heads·DIM·tokens <= q.len() (launch
                    // contract).
                    y[i] = unsafe { *q.get_unchecked((h * DIM + v + i) * tk + t) };
                    i += 1;
                }
                if roped {
                    let (a0, a1) = rope_pair_rn(y[0], y[1], cs[0], cs[1]);
                    let (a2, a3) = rope_pair_rn(y[2], y[3], cs[2], cs[3]);
                    y = [a0, a1, a2, a3];
                }
                let hi01 = cvt_f16x2_f32(y[0], y[1]);
                let hi23 = cvt_f16x2_f32(y[2], y[3]);
                let (h0, h1) = cvt_f32x2_f16x2(hi01);
                let (h2, h3) = cvt_f32x2_f16x2(hi23);
                let lo01 = cvt_f16x2_f32(y[0] - h0, y[1] - h1);
                let lo23 = cvt_f16x2_f32(y[2] - h2, y[3] - h3);
                let row_hi = 16 * (h / 8) + h % 8;
                let at = v / 2;
                // SAFETY: h < heads <= MAX_HEADS puts row_hi + 8 < Q_ROWS,
                // and at + 1 < DIM / 2 < Q_ROW_WORDS: all four stores are
                // inside the tile; this lane alone writes these words, before
                // the barrier below.
                unsafe {
                    *qt.add(row_hi * Q_ROW_WORDS + at) = hi01;
                    *qt.add(row_hi * Q_ROW_WORDS + at + 1) = hi23;
                    *qt.add((row_hi + 8) * Q_ROW_WORDS + at) = lo01;
                    *qt.add((row_hi + 8) * Q_ROW_WORDS + at + 1) = lo23;
                }
            }
            j += 1;
        }
        // The weights of the heads this lane's accumulator rows carry: head
        // 8i + lane/4 of m-tile i.
        let g = (lane / 4) as usize;
        let cl = (lane % 4) as usize;
        let mut wr = [0.0f32; MAX_M_TILES];
        let mut i = 0usize;
        while i < MAX_M_TILES {
            thread::__unroll_config::<0>();
            if i < m_tiles {
                // SAFETY: 8i + g < 8·m_tiles = heads, so the index is below
                // heads·tokens <= w.len() (launch contract).
                wr[i] = mul_rn_f32(unsafe { *w.get_unchecked((8 * i + g) * tk + t) }, scale);
            }
            i += 1;
        }
        thread::sync_threads();

        // ---- the pools, as `kpool_score` walks them.
        let kw = pooled.as_ptr().cast::<u32>();
        // SAFETY: `qt` is this block's shared query tile, a generic address
        // into shared memory, which is what the conversion takes.
        let qbase = unsafe { cvta_generic_to_shared_u32(qt.cast_const().cast::<u8>()) };
        let a_lane = (lane as usize % 16) * Q_ROW_WORDS + 4 * (lane as usize / 16);
        let stride_tiles = blocks as usize * SCORE_WARPS;
        let sbase = t * pools as usize;
        let mut tile = gb * SCORE_WARPS + wid;
        while tile < tiles {
            let key0 = tile * TILE_POOLS;
            let r0 = key0 + g;
            let r1 = r0 + 8;
            let (live0, live1) = (r0 < n, r1 < n);
            let mut b0 = [0u32; 2 * K_STEPS];
            let mut b1 = [0u32; 2 * K_STEPS];
            let mut s = 0usize;
            while s < K_STEPS {
                thread::__unroll_config::<0>();
                if live0 {
                    // SAFETY: r0 < n <= pools and 8s + 4 + cl < KEY_WORDS put
                    // both words inside row r0 of the plane, pools·128 f16
                    // (launch contract), device-allocated and so aligned.
                    unsafe {
                        b0[2 * s] = *kw.add(r0 * KEY_WORDS + 8 * s + cl);
                        b0[2 * s + 1] = *kw.add(r0 * KEY_WORDS + 8 * s + 4 + cl);
                    }
                }
                if live1 {
                    // SAFETY: as the read above, for row r1 < n.
                    unsafe {
                        b1[2 * s] = *kw.add(r1 * KEY_WORDS + 8 * s + cl);
                        b1[2 * s + 1] = *kw.add(r1 * KEY_WORDS + 8 * s + 4 + cl);
                    }
                }
                s += 1;
            }
            let mut p = [0.0f32; 4];
            let mut finite = true;
            let mut i = 0usize;
            while i < MAX_M_TILES {
                thread::__unroll_config::<0>();
                if i < m_tiles {
                    let mut c0 = [0.0f32; 4];
                    let mut c1 = [0.0f32; 4];
                    let mut s = 0usize;
                    while s < K_STEPS {
                        thread::__unroll_config::<0>();
                        let word = a_lane + 16 * i * Q_ROW_WORDS + 8 * s;
                        // SAFETY: row 16i + lane % 16 < 16·m_tiles <= Q_ROWS and
                        // words 8s + 4·(lane / 16) .. + 4 <= DIM / 2 are inside
                        // the tile, which the barrier above published; every
                        // lane of the warp reaches this load with the same
                        // qualifiers (i < m_tiles is warp-uniform).
                        let a = unsafe { ldmatrix_x4_shared_u32(qbase + (4 * word) as u32) };
                        // SAFETY: the whole warp issues these mma.sync with
                        // fragments it loaded (the loop bounds and the m-tile
                        // test are warp-uniform).
                        unsafe {
                            c0 = mma_m16n8k16_f32_f16(c0, a, [b0[2 * s], b0[2 * s + 1]]);
                            c1 = mma_m16n8k16_f32_f16(c1, a, [b1[2 * s], b1[2 * s + 1]]);
                        }
                        s += 1;
                    }
                    let d = [
                        add_rn_f32(c0[0], c0[2]),
                        add_rn_f32(c0[1], c0[3]),
                        add_rn_f32(c1[0], c1[2]),
                        add_rn_f32(c1[1], c1[3]),
                    ];
                    finite &=
                        d[0].is_finite() & d[1].is_finite() & d[2].is_finite() & d[3].is_finite();
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
                }
                i += 1;
            }
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
            if !finite {
                fault.raise(FaultSite::PoolSelect);
            }
            if g < 4 {
                let key = key0 + 8 * (g / 2) + 2 * cl + g % 2;
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
                    if !sv.is_finite() {
                        fault.raise(FaultSite::PoolSelect);
                    }
                    // SAFETY: key < n <= pools, so sbase + key < tokens·pools <=
                    // scores.len() (launch contract); one lane of one warp
                    // scores each pool.
                    unsafe { *scores.get_unchecked_mut(sbase + key) = sv };
                }
            }
            tile += stride_tiles;
        }
    }
}

/// [`KpoolKernels::enqueue_score`]'s arguments: the `tokens` rows' query
/// projections (`[HEADS·DIM × tokens]` f32) and weights' projections
/// (`[HEADS × tokens]` f32), their live counts, the pool plane of a cache of
/// `ctx` positions (`[pools_for(ctx)][DIM]` f16), the pools kept, the
/// weights' scale, and the scores (`[tokens][pools_for(ctx)]`).
pub struct ScoreArgs<'a> {
    pub q: &'a DeviceBuffer<f32>,
    pub w: &'a DeviceBuffer<f32>,
    pub n_keys: &'a DeviceBuffer<u32>,
    pub pooled: &'a DeviceTensor<u16>,
    pub tokens: usize,
    pub ctx: usize,
    pub kept: usize,
    pub scale: f32,
    pub fault: FaultSink,
    pub scores: &'a mut DeviceBuffer<f32>,
}

/// [`KpoolKernels::enqueue_score_tail`]'s arguments: [`ScoreArgs`]' with the
/// head count a launch fact (`q` `[heads·DIM × tokens]`, `w` `[heads ×
/// tokens]`) and the layer's rope table (`ctx` rows of [`ROT`] f32, row `p`
/// the position `p`'s `[cos_0, sin_0, …]`), from which a row of live count
/// `c` turns its query by row `c − 1`.
pub struct TailScoreArgs<'a> {
    pub q: &'a DeviceBuffer<f32>,
    pub w: &'a DeviceBuffer<f32>,
    pub n_keys: &'a DeviceBuffer<u32>,
    pub table: &'a DeviceBuffer<f32>,
    pub pooled: &'a DeviceTensor<u16>,
    pub heads: usize,
    pub tokens: usize,
    pub ctx: usize,
    pub kept: usize,
    pub scale: f32,
    pub fault: FaultSink,
    pub scores: &'a mut DeviceBuffer<f32>,
}

/// The loaded module. Owns no stream: each enqueue takes the engine stream.
pub struct KpoolKernels {
    module: kpool_kernels::LoadedModule,
}

impl KpoolKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<KpoolKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; the launcher checks its launch contract.
        let module = unsafe { crate::shared_module!(kpool_kernels, ctx)? };
        Ok(KpoolKernels { module })
    }

    /// Enqueue the score pass for `tokens` rows: [`blocks_for`] the plane's
    /// rows blocks of 256 per token. Refused by name: no token, `kept` or
    /// `ctx` zero, a plane not [`DIM`] wide or short of [`pools_for`]`(ctx)`
    /// rows, a query, weights, counts or scores shorter than the rows need.
    /// Asynchronous, allocation-free, capturable.
    pub fn enqueue_score(&self, stream: &CudaStream, a: ScoreArgs<'_>) -> Result<(), GpuError> {
        let what = "kpool::enqueue_score";
        let pools = pools_for(a.ctx);
        if a.tokens == 0 || a.kept == 0 || a.ctx == 0 || a.pooled.cols() != DIM {
            return Err(GpuError::shape(
                what,
                format!(
                    "need tokens, kept and ctx >= 1 and a {DIM}-wide plane; got tokens={} \
                     kept={} ctx={} plane {}x{}",
                    a.tokens,
                    a.kept,
                    a.ctx,
                    a.pooled.rows(),
                    a.pooled.cols()
                ),
            ));
        }
        let lens = [
            ("q", a.q.len(), a.tokens * HEADS * DIM),
            ("w", a.w.len(), a.tokens * HEADS),
            ("n_keys", a.n_keys.len(), a.tokens),
            ("pooled rows", a.pooled.rows(), pools),
            ("scores", a.scores.len(), a.tokens * pools),
        ];
        if let Some((name, got, need)) = lens.iter().find(|(_, got, need)| got < need) {
            return Err(GpuError::shape(what, format!("{name} {got} < {need}")));
        }
        let blocks = blocks_for(pools);
        let grid = launch_u32(what, "grid", blocks * a.tokens)?;
        let prep = self
            .module
            .prepare_kpool_score(LaunchConfig1D::new(grid, SCORE_THREADS, 0))?;
        self.module.kpool_score(
            stream,
            &prep,
            a.q,
            a.w,
            a.n_keys,
            a.pooled.buf(),
            launch_u32(what, "tokens", a.tokens)?,
            launch_u32(what, "blocks", blocks)?,
            launch_u32(what, "ctx", a.ctx)?,
            launch_u32(what, "pools", pools)?,
            launch_u32(what, "kept", a.kept)?,
            a.scale,
            a.fault,
            a.scores,
        )?;
        Ok(())
    }

    /// Enqueue the forced-tail score pass for `tokens` rows: [`blocks_for`]
    /// the plane's rows blocks of 256 per token. Refused by name: no token,
    /// `kept` or `ctx` zero, a head count that is not a multiple of eight in
    /// `8..=`[`MAX_HEADS`], a plane not [`DIM`] wide or short of
    /// [`pools_for`]`(ctx)` rows, a table short of `ctx` rows of [`ROT`], a
    /// query, weights, counts or scores shorter than the rows need.
    /// Asynchronous, allocation-free, capturable.
    pub fn enqueue_score_tail(
        &self,
        stream: &CudaStream,
        a: TailScoreArgs<'_>,
    ) -> Result<(), GpuError> {
        let what = "kpool::enqueue_score_tail";
        let pools = pools_for(a.ctx);
        let heads_ok = a.heads >= 8 && a.heads <= MAX_HEADS && a.heads.is_multiple_of(8);
        if a.tokens == 0 || a.kept == 0 || a.ctx == 0 || !heads_ok || a.pooled.cols() != DIM {
            return Err(GpuError::shape(
                what,
                format!(
                    "need tokens, kept and ctx >= 1, heads a multiple of 8 in 8..={MAX_HEADS} and \
                     a {DIM}-wide plane; got tokens={} kept={} ctx={} heads={} plane {}x{}",
                    a.tokens,
                    a.kept,
                    a.ctx,
                    a.heads,
                    a.pooled.rows(),
                    a.pooled.cols()
                ),
            ));
        }
        crate::short(
            what,
            &[
                ("q", a.q.len(), a.tokens * a.heads * DIM),
                ("w", a.w.len(), a.tokens * a.heads),
                ("n_keys", a.n_keys.len(), a.tokens),
                ("table", a.table.len(), a.ctx * ROT),
                ("pooled rows", a.pooled.rows(), pools),
                ("scores", a.scores.len(), a.tokens * pools),
            ],
        )?;
        let blocks = blocks_for(pools);
        let grid = launch_u32(what, "grid", blocks * a.tokens)?;
        let prep =
            self.module
                .prepare_kpool_score_tail(LaunchConfig1D::new(grid, SCORE_THREADS, 0))?;
        self.module.kpool_score_tail(
            stream,
            &prep,
            a.q,
            a.w,
            a.n_keys,
            a.table,
            a.pooled.buf(),
            launch_u32(what, "tokens", a.tokens)?,
            launch_u32(what, "blocks", blocks)?,
            launch_u32(what, "heads", a.heads)?,
            launch_u32(what, "ctx", a.ctx)?,
            launch_u32(what, "pools", pools)?,
            launch_u32(what, "kept", a.kept)?,
            a.scale,
            a.fault,
            a.scores,
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_scale_is_one_sixty_fourth_and_the_grid_covers_every_tile() {
        assert_eq!(weights_scale(HEADS, DIM), 1.0 / 64.0);
        assert_eq!(blocks_for(pools_for(16_384)), 32);
        assert_eq!(blocks_for(pools_for(3_136)), 7);
        assert_eq!(blocks_for(1), 1);
        for pools in 1..5000 {
            assert!(blocks_for(pools) * SCORE_WARPS * TILE_POOLS >= pools);
        }
    }

    #[test]
    fn the_forced_tail_scale_is_ggmls_at_sixteen_heads() {
        // `1.0f / sqrtf(float(n_idx * n_idx_head))`, each op rounded to f32.
        let want = 1.0f32 / (2048.0f32).sqrt();
        assert_eq!(weights_scale(16, DIM).to_bits(), want.to_bits());
        // Every head count the forced-tail launch takes stages whole m-tiles
        // on the eight warps and fits the tile.
        for heads in (8..=MAX_HEADS).step_by(8) {
            let m_tiles = heads / 8;
            assert!(m_tiles <= MAX_M_TILES && m_tiles <= MAX_HPW);
            assert_eq!(m_tiles * SCORE_WARPS, heads);
            assert!(16 * m_tiles <= Q_ROWS);
        }
    }
}
