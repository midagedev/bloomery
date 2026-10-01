//! The one device module of the 32-value-block family: the Q8_0 and Q5_1
//! GEMMs, their activations' quantizers, the remapped route and the wide
//! F32 product — every `#[kernel]` entry of the family, so one bundle load
//! serves them all. It is a module of its own, beside `kernels.rs`, so no
//! entry of the K-quant family's module changes with it. Each GEMM entry is
//! one expansion of `gemm32.rs`'s block; the quantizer, route and F32 bodies
//! live with their owners (`act32.rs`, `remap.rs`, `f32tile.rs`) or are
//! written out here.

use super::act32::quant32_group;
use super::f32tile::{STAGE_FLOATS, STAGES, f32_tile_body};
use super::gemm32::{
    B32_ROW, B32_STAGE_W, D32_ROW, D32_STAGE_W, Q5_1_BLOCK_WORDS, Q8_0_BLOCK_BYTES, W32_ROW,
    W32_STAGE_W, q8_0_file_block,
};
use super::grouped::{GEMM_BM, GEMM_NT, GEMM_THREADS};
use super::remap::remap_id;
use super::route::{
    GEMM_MAX_EXPERTS, NO_EXPERT, ROUTE_HIST, ROUTE_THREADS, ROUTE_WARPS, tile_parts, tile_word,
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

// `swiglu_act_quant32`'s contract spells the largest gate·up rule code as a
// literal.
const _: () =
    assert!(crate::kquant::act::ACT_SILU_MUL == 0 && crate::kquant::act::ACT_SWIGLU_CLAMP == 1);

#[cuda_module]
mod gemm32_kernels {
    use super::*;
    use crate::elem::silu_mul;
    use crate::flash::half_bits_to_f32;
    use crate::kquant::act::apply;
    use crate::q5_1_sel::q5_1_codes;
    use cuda_device::async_copy::{
        cp_async_ca_4, cp_async_cg_16, cp_async_commit_group, cp_async_wait_group,
    };
    use cuda_device::float::{fma_rn_f32, mul_rn_f32};
    use cuda_device::shared::cvta_generic_to_shared_u32;
    use cuda_device::vector::U32x4;
    use cuda_device::wmma::{ldmatrix_x4_shared_u32, mma_m16n8k32_s32_s8};

    /// Quantize `n_cols` f32 columns of `32 · blocks` values each into a
    /// `GemmAct32` (`act32.rs`): one 32-thread block per column and group of
    /// four 32-value blocks, lane ℓ taking values `128·g + 4ℓ .. + 3`
    /// ([`quant32_group`]). A block holding a non-finite value is refused
    /// (NaN scale, zero codes and sum) and raises [`FaultSite::QuantColumn`].
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
            x.len() >= n_cols * 32 * blocks,
            2 * steps >= blocks,
            q.len() >= n_cols * 16 * steps,
            d.len() >= n_cols * 2 * steps,
            s.len() >= n_cols * 2 * steps
        )
    )]
    pub fn quantize_gemm32(
        x: &[f32],
        n_cols: u32,
        blocks: u32,
        groups: u32,
        steps: u32,
        mut q: DisjointSlice<u32>,
        mut d: DisjointSlice<f32>,
        mut s: DisjointSlice<i32>,
        fault: FaultSink,
    ) {
        let grp = thread::index_1d().get() / 32;
        let groups = groups as usize;
        if grp >= n_cols as usize * groups {
            return; // warp-uniform: one warp per block
        }
        let col = grp / groups;
        let g = grp - col * groups;
        let lane = warp::lane_id() as usize;
        let blocks = blocks as usize;
        let b = (4 * g + (lane >> 3)).min(blocks - 1);
        let base = col * 32 * blocks + 32 * b + 4 * (lane & 7);
        // SAFETY: b < blocks, so base + 3 < (col + 1)·32·blocks <= x.len()
        // by the launch contract.
        let v = unsafe {
            [
                *x.get_unchecked(base),
                *x.get_unchecked(base + 1),
                *x.get_unchecked(base + 2),
                *x.get_unchecked(base + 3),
            ]
        };
        // SAFETY: the warp enters with one (col, g), col < n_cols and g <
        // groups = ceil(blocks / 4); the planes' bounds are the launch
        // contract's.
        let refused = unsafe {
            quant32_group(
                v,
                col,
                g,
                blocks,
                steps as usize,
                lane,
                &mut q,
                &mut d,
                &mut s,
            )
        };
        if refused && lane & 7 == 0 {
            fault.raise(FaultSite::QuantColumn);
        }
    }

    /// SwiGLU of `n_cols` slot columns quantized into a `GemmAct32` in one
    /// launch: [`quantize_gemm32`]'s geometry over `elem::silu_mul(g, u)` of
    /// each value — the values `elem::swiglu` stores and the bytes
    /// `quantize_gemm32` writes from them, since both run the same bodies.
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
            g.len() >= n_cols * 32 * blocks,
            u.len() >= n_cols * 32 * blocks,
            2 * steps >= blocks,
            q.len() >= n_cols * 16 * steps,
            d.len() >= n_cols * 2 * steps,
            s.len() >= n_cols * 2 * steps
        )
    )]
    pub fn swiglu_quant32(
        g: &[f32],
        u: &[f32],
        n_cols: u32,
        blocks: u32,
        groups: u32,
        steps: u32,
        mut q: DisjointSlice<u32>,
        mut d: DisjointSlice<f32>,
        mut s: DisjointSlice<i32>,
        fault: FaultSink,
    ) {
        let grp = thread::index_1d().get() / 32;
        let groups = groups as usize;
        if grp >= n_cols as usize * groups {
            return; // warp-uniform: one warp per block
        }
        let col = grp / groups;
        let gi = grp - col * groups;
        let lane = warp::lane_id() as usize;
        let blocks = blocks as usize;
        let b = (4 * gi + (lane >> 3)).min(blocks - 1);
        let base = col * 32 * blocks + 32 * b + 4 * (lane & 7);
        // SAFETY: b < blocks, so base + 3 < (col + 1)·32·blocks, inside g and
        // u by the launch contract.
        let v = unsafe {
            [
                silu_mul(*g.get_unchecked(base), *u.get_unchecked(base)),
                silu_mul(*g.get_unchecked(base + 1), *u.get_unchecked(base + 1)),
                silu_mul(*g.get_unchecked(base + 2), *u.get_unchecked(base + 2)),
                silu_mul(*g.get_unchecked(base + 3), *u.get_unchecked(base + 3)),
            ]
        };
        // SAFETY: the warp enters with one (col, gi), col < n_cols and gi <
        // groups = ceil(blocks / 4); the planes' bounds are the launch
        // contract's.
        let refused = unsafe {
            quant32_group(
                v,
                col,
                gi,
                blocks,
                steps as usize,
                lane,
                &mut q,
                &mut d,
                &mut s,
            )
        };
        if refused && lane & 7 == 0 {
            fault.raise(FaultSite::QuantColumn);
        }
    }

    /// [`swiglu_quant32`] over the slots a route left the card: a column
    /// whose place `sel[col]` is `HOST` is the host's — nothing of it is
    /// read, written or refused, so a stale column a reset left raises
    /// nothing; one whose place is below `n_card` is quantized as
    /// `swiglu_quant32` quantizes it; one in `[n_card, HOST)` is no expert
    /// either side serves and raises [`FaultSite::ExpertId`] without a
    /// store.
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
            g.len() >= n_cols * 32 * blocks,
            u.len() >= n_cols * 32 * blocks,
            sel.len() >= n_cols,
            2 * steps >= blocks,
            q.len() >= n_cols * 16 * steps,
            d.len() >= n_cols * 2 * steps,
            s.len() >= n_cols * 2 * steps
        )
    )]
    pub fn swiglu_quant32_sel(
        g: &[f32],
        u: &[f32],
        sel: &[u32],
        n_card: u32,
        n_cols: u32,
        blocks: u32,
        groups: u32,
        steps: u32,
        mut q: DisjointSlice<u32>,
        mut d: DisjointSlice<f32>,
        mut s: DisjointSlice<i32>,
        fault: FaultSink,
    ) {
        let grp = thread::index_1d().get() / 32;
        let groups = groups as usize;
        if grp >= n_cols as usize * groups {
            return; // warp-uniform: one warp per block
        }
        let col = grp / groups;
        let gi = grp - col * groups;
        // SAFETY: col < n_cols <= sel.len() by the launch contract; the
        // column is warp-uniform, so the branch is too.
        let place = unsafe { *sel.get_unchecked(col) };
        if place == crate::hybrid::HOST {
            return;
        }
        if place >= n_card {
            if warp::lane_id() == 0 {
                fault.raise(FaultSite::ExpertId);
            }
            return;
        }
        let lane = warp::lane_id() as usize;
        let blocks = blocks as usize;
        let b = (4 * gi + (lane >> 3)).min(blocks - 1);
        let base = col * 32 * blocks + 32 * b + 4 * (lane & 7);
        // SAFETY: b < blocks, so base + 3 < (col + 1)·32·blocks, inside g and
        // u by the launch contract.
        let v = unsafe {
            [
                silu_mul(*g.get_unchecked(base), *u.get_unchecked(base)),
                silu_mul(*g.get_unchecked(base + 1), *u.get_unchecked(base + 1)),
                silu_mul(*g.get_unchecked(base + 2), *u.get_unchecked(base + 2)),
                silu_mul(*g.get_unchecked(base + 3), *u.get_unchecked(base + 3)),
            ]
        };
        // SAFETY: the warp enters with one (col, gi), col < n_cols and gi <
        // groups = ceil(blocks / 4); the planes' bounds are the launch
        // contract's.
        let refused = unsafe {
            quant32_group(
                v,
                col,
                gi,
                blocks,
                steps as usize,
                lane,
                &mut q,
                &mut d,
                &mut s,
            )
        };
        if refused && lane & 7 == 0 {
            fault.raise(FaultSite::QuantColumn);
        }
    }

    /// The 32-value GEMM for a Q8_0 stack in the q8f32 planes (`gemm32.rs`):
    /// `qs` is `n_experts · rows` rows of `8 · blocks` words, `wd` their
    /// `blocks` f16 scale bits each; the activations are a `GemmAct32` of
    /// `act_cols` columns (`s` is not read); the route table (`cols`,
    /// `tiles`, `n_tiles`) was built for `n_slots` slots of this stack, and
    /// slot `s` reads column `s / slot_div`. Block `b` is tile `b /
    /// row_tiles`, row slab `b % row_tiles`; it writes `y[s · rows + r]` for
    /// its tile's slots and its slab's rows, and the blocks of tile 0 write
    /// NaN over their slab's rows of every slot the route refused.
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
            qs.len() >= n_experts * rows * 8 * blocks,
            wd.len() >= n_experts * rows * blocks,
            2 * steps >= blocks,
            q.len() >= act_cols * 16 * steps,
            d.len() >= act_cols * 2 * steps,
            cols.len() >= n_slots,
            tiles.len() >= 2 * max_tiles,
            n_tiles.len() >= 2,
            y.len() >= n_slots * rows
        )
    )]
    pub fn gemm_q8_0p(
        qs: &[u32],
        wd: &[u16],
        q: &[u32],
        d: &[f32],
        s: &[i32],
        cols: &[u32],
        tiles: &[u32],
        n_tiles: &[u32],
        n_experts: u32,
        rows: u32,
        blocks: u32,
        steps: u32,
        act_cols: u32,
        n_slots: u32,
        max_tiles: u32,
        slot_div: u32,
        row_tiles: u32,
        mut y: DisjointSlice<f32>,
    ) {
        gemm32_block!(
            wt: plane(qs, wd),
            mins: false,
            params: (q, d, s, cols, tiles, n_tiles, n_experts, rows, blocks, steps, act_cols,
                n_slots, max_tiles, slot_div, row_tiles, y),
        );
    }

    /// [`gemm_q8_0p`] for the file's `block_q8_0` stream: `w` is `n_experts
    /// · rows · blocks` blocks of 34 bytes, back to back, as u32 words.
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
            4 * w.len() >= n_experts * rows * 34 * blocks,
            2 * steps >= blocks,
            q.len() >= act_cols * 16 * steps,
            d.len() >= act_cols * 2 * steps,
            cols.len() >= n_slots,
            tiles.len() >= 2 * max_tiles,
            n_tiles.len() >= 2,
            y.len() >= n_slots * rows
        )
    )]
    pub fn gemm_q8_0f(
        w: &[u32],
        q: &[u32],
        d: &[f32],
        s: &[i32],
        cols: &[u32],
        tiles: &[u32],
        n_tiles: &[u32],
        n_experts: u32,
        rows: u32,
        blocks: u32,
        steps: u32,
        act_cols: u32,
        n_slots: u32,
        max_tiles: u32,
        slot_div: u32,
        row_tiles: u32,
        mut y: DisjointSlice<f32>,
    ) {
        gemm32_block!(
            wt: q8f(w),
            mins: false,
            params: (q, d, s, cols, tiles, n_tiles, n_experts, rows, blocks, steps, act_cols,
                n_slots, max_tiles, slot_div, row_tiles, y),
        );
    }

    /// [`gemm_q8_0p`] for the file's `block_q5_1` stream: `w` is `n_experts
    /// · rows · blocks` blocks of six words; the min term reads `s`.
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
            w.len() >= n_experts * rows * 6 * blocks,
            2 * steps >= blocks,
            q.len() >= act_cols * 16 * steps,
            d.len() >= act_cols * 2 * steps,
            s.len() >= act_cols * 2 * steps,
            cols.len() >= n_slots,
            tiles.len() >= 2 * max_tiles,
            n_tiles.len() >= 2,
            y.len() >= n_slots * rows
        )
    )]
    pub fn gemm_q5_1(
        w: &[u32],
        q: &[u32],
        d: &[f32],
        s: &[i32],
        cols: &[u32],
        tiles: &[u32],
        n_tiles: &[u32],
        n_experts: u32,
        rows: u32,
        blocks: u32,
        steps: u32,
        act_cols: u32,
        n_slots: u32,
        max_tiles: u32,
        slot_div: u32,
        row_tiles: u32,
        mut y: DisjointSlice<f32>,
    ) {
        gemm32_block!(
            wt: q51(w),
            mins: true,
            params: (q, d, s, cols, tiles, n_tiles, n_experts, rows, blocks, steps, act_cols,
                n_slots, max_tiles, slot_div, row_tiles, y),
        );
    }

    /// `gemm_route`'s table (`kernels.rs`) over ids mapped through `map`
    /// (`remap.rs`, [`remap_id`]): a slot whose id maps to `HOST` is neither
    /// counted, listed nor refused; an id at or past `n_map` or a map value
    /// at or past `n_experts` that is not `HOST` raises
    /// [`FaultSite::ExpertId`] and is listed as refused, at the end of
    /// `cols`, `n_tiles[1]` their count. The walk is `gemm_route`'s — one
    /// block, the stable counting sort over chunks — with the id read through
    /// the map; so the listed slots and the tiles are the ones `gemm_route`
    /// builds from the mapped ids with the host slots taken out, and
    /// `cols[listed .. n_slots − refused]` is left unwritten.
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
            map.len() >= n_map,
            cols.len() >= n_slots,
            tiles.len() >= 2 * max_tiles,
            n_tiles.len() >= 2
        )
    )]
    pub fn gemm_route_remap(
        ids: &[u32],
        map: &[u32],
        n_slots: u32,
        n_map: u32,
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
        // group of equal mapped ids adds the group's size to its warp's row,
        // and the refused slots are counted per chunk.
        if wid < chunks {
            let mut refused = 0u32;
            let mut base = lo;
            while base < hi {
                // SAFETY: hi <= n_slots <= ids.len(), n_map <= map.len().
                let (id, bad) =
                    unsafe { remap_id(ids, map, base + lane, hi, n_map, n_experts, e_n) };
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
            // chunk after chunk, so they follow every listed slot and the
            // host slots' unwritten gap; and their count for the GEMMs.
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

        // The stable fill: each chunk in slot order, slot s to its mapped
        // expert's next free index in the warp's row; the lanes of one pass
        // that share a mapped id take consecutive indices in lane (= slot)
        // order, and so do the refused ones from the chunk's first refused
        // index. A host slot is written nowhere.
        if wid < chunks {
            // SAFETY: wid < chunks <= ROUTE_WARPS, published by the barriers
            // since thread 0 wrote it.
            let mut next_bad = unsafe { *bad_at.add(wid) };
            let mut base = lo;
            while base < hi {
                let s = base + lane;
                // SAFETY: hi <= n_slots <= ids.len(), n_map <= map.len().
                let (id, bad) = unsafe { remap_id(ids, map, s, hi, n_map, n_experts, e_n) };
                let peers = warp::match_any_sync(u32::MAX, id);
                let rank = (peers & warp::lanemask_lt()).count_ones();
                if id != NO_EXPERT {
                    // SAFETY: id < e_n, inside this warp's row; the index is
                    // below the expert's first index plus its count, so below
                    // the listed count <= n_slots <= cols.len(), and each slot
                    // writes its own index.
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
                    // SAFETY: id < e_n (remap_id gives every other value
                    // NO_EXPERT), inside this warp's row of HIST; one lane per
                    // id, after every lane of the pass has read the old index.
                    unsafe { *row.add(id as usize) += peers.count_ones() };
                }
                warp::sync_mask(u32::MAX);
                base += 32;
            }
        }
    }

    /// The wide F32 product (`f32tile.rs`, [`f32_tile_body`]): `w` an F32
    /// weight of `rows` rows of `k` values, `x` `n` columns of `k`; writes
    /// `y[c · rows + r]`, each value bit for bit `q8f32::f32_gemv`'s for its
    /// row and column. `k` a positive multiple of 64 and `w`, `x` 16-byte
    /// aligned (host-checked).
    #[kernel]
    #[launch_bounds(512)]
    #[launch_contract(
        domain = 1,
        block = (512, 1, 1),
        requires = (
            rows >= 1,
            n >= 1,
            w.len() >= rows * k,
            x.len() >= n * k,
            y.len() >= rows * n
        )
    )]
    pub fn f32_tile_gemm(
        w: &[f32],
        x: &[f32],
        rows: u32,
        k: u32,
        n: u32,
        mut y: DisjointSlice<f32>,
    ) {
        static mut STAGE: SharedArray<f32, { STAGES * STAGE_FLOATS }, 16> = SharedArray::UNINIT;
        // SAFETY: STAGE is this block's own shared allocation of STAGES ·
        // STAGE_FLOATS values, 16-byte aligned; the raw form reaches the
        // `static mut` without a reference.
        let sh = unsafe { SharedArray::as_raw_mut_ptr(&raw mut STAGE) };
        // SAFETY: every thread of the 512-wide block calls it converged; the
        // host sized the grid, checked k and the alignments; the lengths are
        // the launch contract's.
        unsafe { f32_tile_body(w, x, rows, k, n, &mut y, sh) };
    }

    /// [`swiglu_quant32`] under a gate·up rule of the K-quant family
    /// (`kquant::act`): each value is `apply(act, limit, g, u)` — ik's
    /// clamped SwiGLU at `limit` (`ACT_SWIGLU_CLAMP`) or `elem::silu_mul`
    /// (`ACT_SILU_MUL`, then `swiglu_quant32`'s bytes) — quantized by the
    /// same body; the contract admits those two codes alone. A non-finite
    /// `g` or `u` refuses its block and raises [`FaultSite::QuantColumn`]
    /// under either rule: the clamp would carry it to a finite value the
    /// quantizer's test cannot see.
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
            act <= 1,
            g.len() >= n_cols * 32 * blocks,
            u.len() >= n_cols * 32 * blocks,
            2 * steps >= blocks,
            q.len() >= n_cols * 16 * steps,
            d.len() >= n_cols * 2 * steps,
            s.len() >= n_cols * 2 * steps
        )
    )]
    pub fn swiglu_act_quant32(
        g: &[f32],
        u: &[f32],
        act: u32,
        limit: f32,
        n_cols: u32,
        blocks: u32,
        groups: u32,
        steps: u32,
        mut q: DisjointSlice<u32>,
        mut d: DisjointSlice<f32>,
        mut s: DisjointSlice<i32>,
        fault: FaultSink,
    ) {
        let grp = thread::index_1d().get() / 32;
        let groups = groups as usize;
        if grp >= n_cols as usize * groups {
            return; // warp-uniform: one warp per block
        }
        let col = grp / groups;
        let gi = grp - col * groups;
        let lane = warp::lane_id() as usize;
        let blocks = blocks as usize;
        let b = (4 * gi + (lane >> 3)).min(blocks - 1);
        let base = col * 32 * blocks + 32 * b + 4 * (lane & 7);
        // SAFETY: b < blocks, so base + 3 < (col + 1)·32·blocks, inside g and
        // u by the launch contract.
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
        let v = if crate::fault::quad_finite(gv) & crate::fault::quad_finite(uv) {
            [
                apply(act, limit, gv[0], uv[0]),
                apply(act, limit, gv[1], uv[1]),
                apply(act, limit, gv[2], uv[2]),
                apply(act, limit, gv[3], uv[3]),
            ]
        } else {
            [f32::NAN; 4]
        };
        // SAFETY: the warp enters with one (col, gi), col < n_cols and gi <
        // groups = ceil(blocks / 4); the planes' bounds are the launch
        // contract's.
        let refused = unsafe {
            quant32_group(
                v,
                col,
                gi,
                blocks,
                steps as usize,
                lane,
                &mut q,
                &mut d,
                &mut s,
            )
        };
        if refused && lane & 7 == 0 {
            fault.raise(FaultSite::QuantColumn);
        }
    }
}

/// The loaded 32-value family and its enqueue API, each launcher beside its
/// owner's device code (`act32.rs`, `gemm32.rs`, `remap.rs`, `f32tile.rs`).
/// Owns no context and no stream — every enqueue takes the engine stream, so
/// launches order with the rest of the step and are capturable.
pub struct Gemm32Kernels {
    pub(super) module: gemm32_kernels::LoadedModule,
}

impl Gemm32Kernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<Gemm32Kernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; every launcher checks its launch contract.
        let module = unsafe { gemm32_kernels::load(ctx)? };
        Ok(Gemm32Kernels { module })
    }
}
