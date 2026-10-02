//! The one device module of the grouped GEMM, its route and its SwiGLU
//! quantizer: every `#[kernel]` entry of the family, so one bundle load
//! serves them all. Each GEMM entry is one expansion of `grouped.rs`'s
//! block; the route and SwiGLU bodies are written out here, and the device
//! helpers every body calls live with their owner (`grouped.rs`,
//! `route.rs`).

use super::grouped::{
    B_COL_W, B_STAGE_W, GEMM_BM, GEMM_NT, GEMM_THREADS, Q3K_D, Q3K_QS, Q3K_SB_BYTES, Q3K_SCALES,
    Q4K_SB_WORDS, Q5K_SB_WORDS, Q6K_D, Q6K_QH, Q6K_SB_BYTES, Q6K_SCALES, S8_STAGE, WT_ROW_W,
    WT_STAGE_W, q3k_code, q3k_scale_word, q5k_code, sbyte, ubyte, win,
};
use super::route::{
    GEMM_MAX_EXPERTS, NO_EXPERT, ROUTE_HIST, ROUTE_THREADS, ROUTE_WARPS, route_id, tile_parts,
    tile_word,
};
use super::{GEMM_BN, GEMM_MAX_SLOTS};
use crate::GpuError;
use crate::fault::{FaultSink, FaultSite};
use cuda_core::CudaContext;
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread, warp,
};
use cuda_host::cuda_module;
use std::sync::Arc;

#[cuda_module]
mod gemm_kernels {
    use super::*;
    use crate::cores::{q4k_scale_min, q6k_dequant};
    use crate::elem::silu_mul;
    use crate::flash::half_bits_to_f32;
    use crate::q8_1_quant_vals;
    use cuda_device::async_copy::{
        cp_async_ca_4, cp_async_cg_16, cp_async_commit_group, cp_async_wait_group,
    };
    use cuda_device::float::{fma_rn_f32, mul_rn_f32};
    use cuda_device::vector::{F32x2, U32x4};
    use cuda_device::wmma::{mma_m16n8k16_s32_s8, mma_m16n8k32_s32_s8};

    /// The route table: `ids` (`n_slots` expert ids, slot `s = token·top_k +
    /// k`) grouped by expert into `cols` — expert 0's slots first, each
    /// expert's in ascending slot order — and cut into tiles of at most
    /// GEMM_BN slots, tile i at `tiles[2i]` (its expert) and `tiles[2i + 1]`
    /// (its `tile_word`: its first index into `cols`, and its length in the
    /// upper 16 bits), experts ascending; `n_tiles[0]` is the tile count.
    ///
    /// An id at or past `n_experts` has no expert to go to: it raises
    /// [`FaultSite::ExpertId`] on `fault` and its slot is left out of the
    /// tiles. The refused slots take the end of `cols` instead, in ascending
    /// slot order after the `n_slots − r` listed ones, and `n_tiles[1]` is
    /// their count `r`: every GEMM over the table writes NaN to each of their
    /// output rows.
    ///
    /// One block, a stable counting sort over chunks. The slots are cut into
    /// `chunks = min(ROUTE_WARPS, ROUTE_HIST / n_experts)` consecutive
    /// ranges, warp `w` owning range `w`, 32 slots a pass in slot order: the
    /// lanes of one pass that share an id find each other with `match.any`,
    /// so a warp counts its range per expert, and later fills it, with no
    /// atomics. Between the two walks, thread `e` turns expert `e`'s counts
    /// into each chunk's first index — the expert's first index plus the
    /// counts of the chunks before — so chunk `w`'s slots of an expert land
    /// after every lower chunk's and before every higher one's, and the
    /// table is the one a single walk over all slots in order builds, the
    /// same bits on every run.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(1024)]
    #[launch_contract(
        domain = 1,
        block = (1024, 1, 1),
        requires = (
            ids.len() >= n_slots,
            cols.len() >= n_slots,
            tiles.len() >= 2 * max_tiles,
            n_tiles.len() >= 2
        )
    )]
    pub fn gemm_route(
        ids: &[u32],
        n_slots: u32,
        n_experts: u32,
        max_tiles: u32,
        mut cols: DisjointSlice<u32>,
        mut tiles: DisjointSlice<u32>,
        mut n_tiles: DisjointSlice<u32>,
        fault: FaultSink,
    ) {
        static mut HIST: SharedArray<u32, ROUTE_HIST> = SharedArray::UNINIT;
        static mut WT: SharedArray<u32, 32> = SharedArray::UNINIT;
        // Per chunk: its refused slots' count, then their first index.
        static mut BAD: SharedArray<u32, ROUTE_WARPS> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let lane = warp::lane_id() as usize;
        let wid = tid / 32;
        let s_n = (n_slots as usize).min(GEMM_MAX_SLOTS);
        let e_n = (n_experts as usize).clamp(1, GEMM_MAX_EXPERTS);
        let chunks = (ROUTE_HIST / e_n).min(ROUTE_WARPS);
        // Each chunk a whole number of passes; the last ones may be short or
        // empty.
        let per = s_n.div_ceil(chunks).div_ceil(32) * 32;
        let lo = wid * per;
        let hi = (lo + per).min(s_n);
        // SAFETY: each `static mut` is this block's own shared allocation,
        // reached raw; every index below is inside its array and every
        // cross-thread read follows a block or warp barrier.
        let (hist, wt, bad_at) = unsafe {
            (
                SharedArray::as_raw_mut_ptr(&raw mut HIST),
                SharedArray::as_raw_mut_ptr(&raw mut WT),
                SharedArray::as_raw_mut_ptr(&raw mut BAD),
            )
        };
        // Warp `wid`'s counts, one per expert: row `wid` of the `chunks`
        // rows of `e_n` words; only warps below `chunks` use it.
        // SAFETY: wid.min(chunks − 1) < chunks and chunks · e_n <=
        // ROUTE_HIST, so the row lies inside HIST.
        let row = unsafe { hist.add(wid.min(chunks - 1) * e_n) };

        let mut i = tid;
        while i < chunks * e_n {
            // SAFETY: i < chunks · e_n <= ROUTE_HIST.
            unsafe { *hist.add(i) = 0 };
            i += ROUTE_THREADS;
        }
        thread::sync_threads();

        // Counts, per chunk in slot order: per pass, the lowest lane of each
        // group of equal ids adds the group's size to its warp's row, and the
        // refused slots are counted per chunk.
        if wid < chunks {
            let mut refused = 0u32;
            let mut base = lo;
            while base < hi {
                // SAFETY: hi <= n_slots <= ids.len().
                let (id, bad) = unsafe { route_id(ids, base + lane, hi, n_experts, e_n) };
                if bad {
                    // Raised here only: the fill walk reads each slot again.
                    fault.raise(FaultSite::ExpertId);
                }
                refused += warp::ballot(bad).count_ones();
                let peers = warp::match_any_sync(u32::MAX, id);
                if id != NO_EXPERT && peers.trailing_zeros() as usize == lane {
                    // SAFETY: id < e_n, inside this warp's row; one lane per
                    // id, and no other warp touches the row.
                    unsafe { *row.add(id as usize) += peers.count_ones() };
                }
                warp::sync_mask(u32::MAX);
                base += 32;
            }
            if lane == 0 {
                // SAFETY: wid < chunks <= ROUTE_WARPS; one lane per chunk.
                unsafe { *bad_at.add(wid) = refused };
            }
        }
        thread::sync_threads();

        if tid == 0 {
            // The refused slots' first index per chunk: the end of `cols`,
            // chunk after chunk, so they follow every listed slot; and their
            // count for the GEMMs.
            let mut total = 0u32;
            let mut w = 0usize;
            while w < chunks {
                // SAFETY: w < chunks <= ROUTE_WARPS, each written above and
                // published by the barrier; thread 0 alone touches them now.
                total += unsafe { *bad_at.add(w) };
                w += 1;
            }
            let mut run = s_n as u32 - total;
            let mut w = 0usize;
            while w < chunks {
                // SAFETY: w < chunks <= ROUTE_WARPS, inside BAD; warp w's lane
                // 0 wrote the word before the barrier above, and thread 0 alone
                // reads and rewrites these words until the next barrier.
                unsafe {
                    let p = bad_at.add(w);
                    let n = *p;
                    *p = run;
                    run += n;
                }
                w += 1;
            }
            // SAFETY: n_tiles.len() >= 2; one thread writes it.
            unsafe { *n_tiles.get_unchecked_mut(1) = total };
        }

        // Expert `tid`'s count, and each chunk's count of it replaced by the
        // chunks before it: the chunk's first index within the expert.
        let c = if tid < e_n {
            let mut run = 0u32;
            let mut w = 0usize;
            while w < chunks {
                // SAFETY: w < chunks and tid < e_n, inside HIST; column tid
                // is this thread's alone until the barrier below.
                unsafe {
                    let p = hist.add(w * e_n + tid);
                    let n = *p;
                    *p = run;
                    run += n;
                }
                w += 1;
            }
            run
        } else {
            0
        };

        // Exclusive scan over experts of (count | tiles << 16): the low half
        // is the expert's first index into `cols`, the high half its first
        // tile. Neither half's sum reaches 1 << 16 (GEMM_MAX_SLOTS, and
        // gemm_max_tiles's bound), so no carry crosses.
        let tc = c.div_ceil(GEMM_BN as u32);
        let v = c | (tc << 16);
        let mut incl = v;
        let mut o = 1u32;
        while o < 32 {
            let up = warp::shuffle_up(incl, o);
            if lane as u32 >= o {
                incl += up;
            }
            o <<= 1;
        }
        if lane == 31 {
            // SAFETY: wid < 32.
            unsafe { *wt.add(wid) = incl };
        }
        thread::sync_threads();
        if wid == 0 {
            // SAFETY: lane < 32.
            let x = unsafe { *wt.add(lane) };
            let mut ix = x;
            let mut o = 1u32;
            while o < 32 {
                let up = warp::shuffle_up(ix, o);
                if lane as u32 >= o {
                    ix += up;
                }
                o <<= 1;
            }
            warp::sync_mask(u32::MAX);
            // SAFETY: lane < 32; every lane read its slot before the sync.
            unsafe { *wt.add(lane) = ix - x };
        }
        thread::sync_threads();
        // SAFETY: wid < 32, published by the barrier above.
        let excl = incl - v + unsafe { *wt.add(wid) };
        let off = excl & 0xffff;
        let toff = excl >> 16;
        if tid == ROUTE_THREADS - 1 {
            // SAFETY: n_tiles.len() >= 2; one thread writes it.
            unsafe { *n_tiles.get_unchecked_mut(0) = (excl + v) >> 16 };
        }
        if tid < e_n {
            let mut i = 0u32;
            while i < tc {
                let tix = (toff + i) as usize;
                if tix < max_tiles as usize {
                    let len = (c - i * GEMM_BN as u32).min(GEMM_BN as u32);
                    // SAFETY: tix < max_tiles, so both words are inside
                    // tiles; expert tid owns tiles toff..toff + tc.
                    unsafe {
                        *tiles.get_unchecked_mut(2 * tix) = tid as u32;
                        *tiles.get_unchecked_mut(2 * tix + 1) =
                            tile_word(off + i * GEMM_BN as u32, len);
                    }
                } else {
                    // Unreachable by gemm_max_tiles's bound; kept loud.
                    fault.raise(FaultSite::ExpertId);
                }
                i += 1;
            }
            // Each chunk's first index into `cols` for expert tid.
            let mut w = 0usize;
            while w < chunks {
                // SAFETY: w < chunks and tid < e_n, so w · e_n + tid <
                // chunks · e_n <= ROUTE_HIST, inside HIST; column tid is still
                // this thread's alone until the barrier below.
                unsafe { *hist.add(w * e_n + tid) += off };
                w += 1;
            }
        }
        thread::sync_threads();

        // The stable fill: each chunk in slot order, slot s to its expert's
        // next free index in the warp's row; the lanes of one pass that share
        // an id take consecutive indices in lane (= slot) order, and so do
        // the refused ones from the chunk's first refused index.
        if wid < chunks {
            // SAFETY: wid < chunks <= ROUTE_WARPS, published by the barriers
            // since thread 0 wrote it.
            let mut next_bad = unsafe { *bad_at.add(wid) };
            let mut base = lo;
            while base < hi {
                let s = base + lane;
                // SAFETY: hi <= n_slots <= ids.len().
                let (id, bad) = unsafe { route_id(ids, s, hi, n_experts, e_n) };
                let peers = warp::match_any_sync(u32::MAX, id);
                let rank = (peers & warp::lanemask_lt()).count_ones();
                if id != NO_EXPERT {
                    // SAFETY: id < e_n, inside this warp's row; the index is
                    // below the expert's first index plus its count, so below
                    // n_slots <= cols.len(), and each slot writes its own
                    // index.
                    unsafe {
                        let pos = *row.add(id as usize) + rank;
                        *cols.get_unchecked_mut(pos as usize) = s as u32;
                    }
                }
                let bads = warp::ballot(bad);
                if bad {
                    // SAFETY: the chunk's refused slots take the indices from
                    // its first one on, below the next chunk's and all below
                    // s_n <= n_slots <= cols.len(); each slot writes its own.
                    unsafe {
                        let pos = next_bad + (bads & warp::lanemask_lt()).count_ones();
                        *cols.get_unchecked_mut(pos as usize) = s as u32;
                    }
                }
                next_bad += bads.count_ones();
                warp::sync_mask(u32::MAX);
                if id != NO_EXPERT && peers.trailing_zeros() as usize == lane {
                    // SAFETY: id < e_n (route_id gives every other id
                    // NO_EXPERT), inside this warp's row of HIST; one lane per
                    // id, after every lane of the pass has read the old index.
                    unsafe { *row.add(id as usize) += peers.count_ones() };
                }
                warp::sync_mask(u32::MAX);
                base += 32;
            }
        }
    }

    /// The grouped GEMM for a Q4_K stack: `w` is `n_experts · rows` rows of
    /// `36 · n_sb` words (the file's super-blocks), 16-byte aligned and fewer
    /// than 2^30 words (its slabs are staged in 16-byte copies from 32-bit
    /// byte offsets); the activations are the
    /// q6 permutation, `s8` and `d8` of `act_cols` quantized columns; the
    /// route table (`cols`, `tiles`, `n_tiles`) was built for `n_slots`
    /// slots of this stack, and slot `s` reads column `s / slot_div`. Block
    /// `b` is tile `b / row_tiles`, row slab `b % row_tiles`; it writes
    /// `y[s · rows + r]` for its tile's slots and its slab's rows, and the
    /// blocks of tile 0 write NaN over their slab's rows of every slot the
    /// route refused.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            w.len() >= n_experts * rows * 36 * n_sb,
            q6.len() >= act_cols * 128 * half_it,
            s8.len() >= act_cols * 8 * n_sb,
            d8.len() >= act_cols * 2 * n_sb,
            cols.len() >= n_slots,
            tiles.len() >= 2 * max_tiles,
            n_tiles.len() >= 2,
            y.len() >= n_slots * rows
        )
    )]
    pub fn gemm_q4k(
        w: &[u32],
        q6: &[u32],
        s8: &[i32],
        d8: &[f32],
        cols: &[u32],
        tiles: &[u32],
        n_tiles: &[u32],
        n_experts: u32,
        rows: u32,
        n_sb: u32,
        half_it: u32,
        act_cols: u32,
        n_slots: u32,
        max_tiles: u32,
        slot_div: u32,
        row_tiles: u32,
        mut y: DisjointSlice<f32>,
    ) {
        gemm_block!(
            dec: q4k_dec,
            wst: wst_q4k(wt),
            walk: split,
            sb_units: Q4K_SB_WORDS,
            params: (w, q6, s8, d8, cols, tiles, n_tiles, n_experts, rows, n_sb, half_it,
                act_cols, n_slots, max_tiles, slot_div, row_tiles, y),
        );
    }

    /// [`gemm_q4k`] for a Q5_K stack: rows of `44 · n_sb` words.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            w.len() >= n_experts * rows * 44 * n_sb,
            q6.len() >= act_cols * 128 * half_it,
            s8.len() >= act_cols * 8 * n_sb,
            d8.len() >= act_cols * 2 * n_sb,
            cols.len() >= n_slots,
            tiles.len() >= 2 * max_tiles,
            n_tiles.len() >= 2,
            y.len() >= n_slots * rows
        )
    )]
    pub fn gemm_q5k(
        w: &[u32],
        q6: &[u32],
        s8: &[i32],
        d8: &[f32],
        cols: &[u32],
        tiles: &[u32],
        n_tiles: &[u32],
        n_experts: u32,
        rows: u32,
        n_sb: u32,
        half_it: u32,
        act_cols: u32,
        n_slots: u32,
        max_tiles: u32,
        slot_div: u32,
        row_tiles: u32,
        mut y: DisjointSlice<f32>,
    ) {
        gemm_block!(
            dec: q5k_dec,
            wst: wst_direct(),
            walk: plain,
            sb_units: Q5K_SB_WORDS,
            params: (w, q6, s8, d8, cols, tiles, n_tiles, n_experts, rows, n_sb, half_it,
                act_cols, n_slots, max_tiles, slot_div, row_tiles, y),
        );
    }

    /// [`gemm_q4k`] for a Q6_K stack: the stack is the file's byte stream,
    /// `210 · n_sb` bytes a row, as u32 words padded at the end to a whole
    /// word; a super-block may start at 2 mod 4.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            4 * w.len() >= n_experts * rows * 210 * n_sb,
            q6.len() >= act_cols * 128 * half_it,
            s8.len() >= act_cols * 8 * n_sb,
            d8.len() >= act_cols * 2 * n_sb,
            cols.len() >= n_slots,
            tiles.len() >= 2 * max_tiles,
            n_tiles.len() >= 2,
            y.len() >= n_slots * rows
        )
    )]
    pub fn gemm_q6k(
        w: &[u32],
        q6: &[u32],
        s8: &[i32],
        d8: &[f32],
        cols: &[u32],
        tiles: &[u32],
        n_tiles: &[u32],
        n_experts: u32,
        rows: u32,
        n_sb: u32,
        half_it: u32,
        act_cols: u32,
        n_slots: u32,
        max_tiles: u32,
        slot_div: u32,
        row_tiles: u32,
        mut y: DisjointSlice<f32>,
    ) {
        gemm_block!(
            dec: q6k_dec,
            wst: wst_direct(),
            walk: split,
            sb_units: Q6K_SB_BYTES,
            params: (w, q6, s8, d8, cols, tiles, n_tiles, n_experts, rows, n_sb, half_it,
                act_cols, n_slots, max_tiles, slot_div, row_tiles, y),
        );
    }

    /// [`gemm_q6k`] for a Q3_K stack: `110 · n_sb` bytes a row.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            4 * w.len() >= n_experts * rows * 110 * n_sb,
            q6.len() >= act_cols * 128 * half_it,
            s8.len() >= act_cols * 8 * n_sb,
            d8.len() >= act_cols * 2 * n_sb,
            cols.len() >= n_slots,
            tiles.len() >= 2 * max_tiles,
            n_tiles.len() >= 2,
            y.len() >= n_slots * rows
        )
    )]
    pub fn gemm_q3k(
        w: &[u32],
        q6: &[u32],
        s8: &[i32],
        d8: &[f32],
        cols: &[u32],
        tiles: &[u32],
        n_tiles: &[u32],
        n_experts: u32,
        rows: u32,
        n_sb: u32,
        half_it: u32,
        act_cols: u32,
        n_slots: u32,
        max_tiles: u32,
        slot_div: u32,
        row_tiles: u32,
        mut y: DisjointSlice<f32>,
    ) {
        gemm_block!(
            dec: q3k_dec,
            wst: wst_direct(),
            walk: plain,
            sb_units: Q3K_SB_BYTES,
            params: (w, q6, s8, d8, cols, tiles, n_tiles, n_experts, rows, n_sb, half_it,
                act_cols, n_slots, max_tiles, slot_div, row_tiles, y),
        );
    }

    /// SwiGLU of `n_cols` slot columns of `256 · n_sb` values quantized to
    /// q8_1 in one launch, one 32-thread block per 128-value block: lane `l`
    /// of block `(col, b)` takes values `128·b + 4·l .. +3` of column `col` of
    /// `g` and `u`, forms `elem::silu_mul` of each pair, and hands the four
    /// to `q8_1_quant_vals` — the values `elem::swiglu` stores and the bytes
    /// `q3k_quantize_q8_1` writes from them, since both run the same bodies
    /// on the same geometry. A block holding a non-finite value is refused
    /// as the quantizer refuses it (NaN scale, zero codes) and raises
    /// [`FaultSite::QuantColumn`].
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(32)]
    #[launch_contract(
        domain = 1,
        block = (32, 1, 1),
        requires = (
            g.len() >= n_cols * 256 * n_sb,
            u.len() >= n_cols * 256 * n_sb,
            q3.len() >= n_cols * 64 * half_it,
            q4.len() >= n_cols * 256 * quad_it,
            q6.len() >= n_cols * 128 * half_it,
            s8.len() >= n_cols * 8 * n_sb,
            d8.len() >= n_cols * 2 * n_sb
        )
    )]
    pub fn gemm_swiglu_quant(
        g: &[f32],
        u: &[f32],
        n_cols: u32,
        n_sb: u32,
        half_it: u32,
        quad_it: u32,
        mut q3: DisjointSlice<u64>,
        mut q4: DisjointSlice<u32>,
        mut q6: DisjointSlice<u32>,
        mut s8: DisjointSlice<i32>,
        mut d8: DisjointSlice<f32>,
        fault: FaultSink,
    ) {
        let blk = thread::index_1d().get() / 32;
        let n_sb = n_sb as usize;
        let per_col = 2 * n_sb;
        if blk >= n_cols as usize * per_col {
            return; // warp-uniform: one warp per block
        }
        let col = blk / per_col;
        let b = blk - col * per_col;
        let lane = warp::lane_id() as usize;
        let base = col * 256 * n_sb + 128 * b + 4 * lane;
        // SAFETY: base + 3 < (col + 1)·256·n_sb <= n_cols·256·n_sb, inside g
        // and u by the launch contract.
        let (gv, uv) = unsafe {
            (
                [
                    *g.get_unchecked(base),
                    *g.get_unchecked(base + 1),
                    *g.get_unchecked(base + 2),
                    *g.get_unchecked(base + 3),
                ],
                [
                    *u.get_unchecked(base),
                    *u.get_unchecked(base + 1),
                    *u.get_unchecked(base + 2),
                    *u.get_unchecked(base + 3),
                ],
            )
        };
        let v = [
            silu_mul(gv[0], uv[0]),
            silu_mul(gv[1], uv[1]),
            silu_mul(gv[2], uv[2]),
            silu_mul(gv[3], uv[3]),
        ];
        // SAFETY: col < n_cols and b < 2·n_sb by the lines above, the output
        // bounds are the launch contract's, the block is one warp with one
        // `(col, b)`, and `v` holds values 128·b + 4·lane .. +3 of column col.
        let refused = unsafe {
            q8_1_quant_vals(
                v, col, b, n_sb, half_it, quad_it, lane, &mut q3, &mut q4, &mut q6, &mut s8,
                &mut d8,
            )
        };
        if refused && lane == 0 {
            fault.raise(FaultSite::QuantColumn);
        }
    }
}

/// The loaded GEMM module and its enqueue API, each launcher beside its
/// owner's device code (`grouped.rs`, `route.rs`, `swiglu.rs`). Owns no
/// context and no stream — every enqueue takes the engine stream, so
/// launches order with the rest of the step and are capturable.
pub struct GemmKernels {
    pub(super) module: gemm_kernels::LoadedModule,
}

impl GemmKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<GemmKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; every launcher checks its launch contract.
        let module = unsafe { crate::shared_module!(gemm_kernels, ctx)? };
        Ok(GemmKernels { module })
    }
}
