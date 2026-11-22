//! The looped core's message controller (Inkling 2, llama.cpp PR #32695's
//! `src/models/inkling2.cpp` `control`, calls at its layer walk): the merge of a message mixed
//! from the previous pass's banked blocks into a sublayer's input,
//! `out = (1 − tanh(F·h)) · cur + tanh(G·h) · msg` with `h = D · [rms(cur); rms(msg)]` — the two
//! halves gainless-RMS-normed on their own at `eps`, the whole row down-projected to the gate
//! rank, and the gain `G` and forget `F` rows of the rank writing the hidden row back. The
//! reference computes f32 throughout (the weights are f32 in the file); so does the card.
//!
//! A module is one core layer's controller (the attention, k, v or feed-forward one: its own
//! `down`, `gain` and `forget` rows over its own `cur` and `msg` rows). Every module of every core
//! visit of a pass ≥ 1 runs one (`inkling2.cpp`'s walk: the q/k/v controllers before the input
//! norms, the ffn's after the feed-forward mix). The three attention modules' inputs are ready
//! together after the message mixes, so [`LoopControlKernels::enqueue_attn`] runs them as one
//! pair of launches; the feed-forward's are ready only after the block join and
//! [`LoopControlKernels::enqueue`] runs it alone.
//!
//! Layouts, in the file's own memory order (ggml's `ne[0]`-contiguous; the launch takes each
//! weight as the file lays it, no repack):
//! - `cur` and `msg` `[m][n]`, `y` `[m][n]`, token-major;
//! - `down` `[rank][2n]`: output `r`'s 2n weights contiguous, the input `k` of `[cur; msg]`
//!   fastest — a warp's lanes read consecutive `k`, one dot a warp;
//! - `gain` and `forget` `[n][rank]`: output `j`'s `rank` weights contiguous at `j · rank`, the
//!   rank fastest — a thread's dot reads its own output's row.
//!
//! Geometry: two launches, so the weights' reads spread over the card at a decode row. The
//! down launch (`loop_control_down`) is one block of [`THREADS`] threads per [`DOWN_OUTS`]
//! outputs of one row of one module: the block stages the row's raw `[cur; msg]` into shared
//! memory (`2 · MAX_N` f32, the family's widest hidden), reduces the two sums of squares in the
//! crate's order (thread-strided f32 fused multiply-adds, the warp's by the xor butterfly, the
//! eight warps' in the fixed tree), broadcasts the two `rstd` scales through shared, scales the
//! staged halves in place, and runs its outputs' down-dots from the staged row (a dot a warp over
//! the whole row, the lane-strided f32 FMA order the crate's gemv bodies use) into `h`. Every
//! block of a row repeats the staging and the norm on the same inputs in the same order, so its
//! `rstd` and normed row are bit-equal to its siblings'. The out launch (`loop_control_out`,
//! `loop_control_out3`) is one block per [`OUT_COLS`] outputs of one row of one module: it
//! copies the row's `h` into shared and each of its threads runs the gain and forget dots of
//! its output `j` over the rank ascending, the `tanh` the device's own, and the combine. No
//! block waits on another: the second launch's stream order is the only dependency, and the
//! launches' grids are flat, decoded from the block index (`grp` fastest, then the row, then the
//! module). A dot is never split: a block count changes which blocks run a dot, not its order.
//! The shared totals are `2 · MAX_N + 2 · WARPS + 2` f32 (down) and [`MAX_RANK`] f32 (out).
//!
//! The scratch `h` is `[module][m][rank]` f32, [`scratch_len`] long, written by the down launch
//! and read by the out launch of the same call; the caller owns it and nothing else may use it
//! between the two launches.
//!
//! Numeric contract: the two squares and the down and gain/forget dots accumulate by fused
//! multiply-adds ascending their fast index (the first product rounds alone), each warp's
//! partials by the xor butterfly; the staged normed value `v · rstd` rounds on its own; the
//! combine is `mul_rn((1 − tanh(F·h)), cur) + mul_rn(tanh(G·h), msg)` — the reference's
//! `add(mul(1 − F, cur), mul(G, msg))` (`inkling2.cpp:236`), one rounding a multiply.
//!
//! No silent failure: a sum of squares that is not finite and positive, or an output that stops
//! being finite (a non-finite input, dot or `tanh` argument), raises [`FaultSite::F32Product`]
//! and the value written there is NaN, never a plausible merge.
//!
//! [`LoopControlKernels::enqueue_single_block`] keeps the earlier shape (one fused launch, one
//! block a row) as the gate's bit-identity reference: the split launches' output equals it bit
//! for bit.

use crate::GpuError;
use crate::elem::rms_scale_of_mean_checked;
use crate::fault::{FaultSink, FaultSite};
use crate::launch_u32;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::float::{add_rn_f32, fma_rn_f32, mul_rn_f32};
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread, warp,
};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Threads a block gives one row.
pub const THREADS: usize = 256;
const WARPS: usize = THREADS / 32;
/// The widest hidden the entry stages: the family's largest file's (the Small's 4,608; the
/// looped files' is 2,304). The host refuses a wider one by name.
pub const MAX_N: usize = 4608;
/// The widest gate rank the entry takes: the looped core's (64 both looped files). The host
/// refuses a wider one by name.
pub const MAX_RANK: usize = 64;

/// Down outputs one block of the down launch computes (a warp each; the other warps stage).
pub const DOWN_OUTS: usize = 2;
/// Outputs one block of the out launch computes (a thread each).
pub const OUT_COLS: usize = 64;
/// The most modules one launch pair takes (the three attention modules).
pub const MAX_MODULES: usize = 3;
/// Down-dot loads a lane issues before folding them (the fold stays ascending).
const DOWN_UNROLL: usize = 16;
/// Rank steps a thread loads before folding them (the fold stays ascending).
const OUT_UNROLL: usize = 8;

const _: () = assert!(THREADS == 256 && WARPS == 8);
const _: () = assert!(DOWN_OUTS <= WARPS && OUT_COLS <= THREADS);

/// Floats of the scratch `h` a launch pair of `modules` modules over `m` rows at `rank` needs.
pub const fn scratch_len(modules: usize, m: usize, rank: usize) -> usize {
    modules * m * rank
}

/// The staged raw `[cur; msg]` row.
const HNORM_LEN: usize = 2 * MAX_N;
/// The down-projection's row.
const H_LEN: usize = MAX_RANK;
/// The squares' warp partials: a pair a warp.
const PART_LEN: usize = 2 * WARPS;
/// The two scales.
const RSTD_LEN: usize = 2;

/// The eight warp sums at `ws + 2 · b + at` in the fixed tree `((w0 + w1) + (w2 + w3)) +
/// ((w4 + w5) + (w6 + w7))`.
///
/// # Safety
///
/// `ws` addresses the eight written pairs of `at` (0 cur's squares, 1 msg's).
#[inline(always)]
unsafe fn tree(ws: *const f32, at: usize) -> f32 {
    // SAFETY: the caller's contract: eight pairs at `at`.
    unsafe {
        let w = |b: usize| *ws.add(b * 2 + at);
        ((w(0) + w(1)) + (w(2) + w(3))) + ((w(4) + w(5)) + (w(6) + w(7)))
    }
}

/// One block of the out launch: the gain and forget dots and the combine of output group `grp`
/// of row `t` (module doc), its `h` row at `h_row`. Every thread of the block calls it (it holds
/// a barrier).
///
/// # Safety
///
/// `hs` addresses [`H_LEN`] shared f32; `h_row + rank <= h.len()`; `t · n + n <= cur.len(),
/// msg.len(), y.len()`; `grp · OUT_COLS < n`; `n · rank <= gain.len(), forget.len()`;
/// `1 <= rank <= MAX_RANK`.
#[inline(always)]
#[allow(
    clippy::too_many_arguments,
    reason = "the block's inputs, one argument each (rust-quality R8)"
)]
unsafe fn out_block(
    cur: &[f32],
    msg: &[f32],
    gain: &[f32],
    forget: &[f32],
    h: &[f32],
    hs: *mut f32,
    y: &mut DisjointSlice<f32>,
    n: usize,
    rank: usize,
    t: usize,
    grp: usize,
    h_row: usize,
    fault: FaultSink,
) {
    let tid = thread::threadIdx_x() as usize;
    if tid < rank {
        // SAFETY: h_row + tid < h_row + rank <= h.len(); tid < rank <= H_LEN; this thread alone
        // writes the value.
        unsafe { *hs.add(tid) = *h.get_unchecked(h_row + tid) };
    }
    thread::sync_threads();
    let j = grp * OUT_COLS + tid;
    let mut ok = true;
    if tid < OUT_COLS && j < n {
        let gb = j * rank;
        // SAFETY: gb + d < n · rank <= gain.len() and forget.len() for d < rank; d < rank <=
        // H_LEN, written before the barrier above. The first product rounds alone, then the
        // fused multiply-adds ascend the rank (this order is the gate); an unrolled step's
        // loads are issued before its folds.
        let (mut g, mut f) = unsafe {
            let hv = *hs.add(0);
            (
                mul_rn_f32(*gain.get_unchecked(gb), hv),
                mul_rn_f32(*forget.get_unchecked(gb), hv),
            )
        };
        let mut d = 1usize;
        while d + OUT_UNROLL <= rank {
            // SAFETY: the last index is gb + d + 7 < gb + rank; as above.
            unsafe {
                let gw0 = *gain.get_unchecked(gb + d);
                let fw0 = *forget.get_unchecked(gb + d);
                let gw1 = *gain.get_unchecked(gb + d + 1);
                let fw1 = *forget.get_unchecked(gb + d + 1);
                let gw2 = *gain.get_unchecked(gb + d + 2);
                let fw2 = *forget.get_unchecked(gb + d + 2);
                let gw3 = *gain.get_unchecked(gb + d + 3);
                let fw3 = *forget.get_unchecked(gb + d + 3);
                let gw4 = *gain.get_unchecked(gb + d + 4);
                let fw4 = *forget.get_unchecked(gb + d + 4);
                let gw5 = *gain.get_unchecked(gb + d + 5);
                let fw5 = *forget.get_unchecked(gb + d + 5);
                let gw6 = *gain.get_unchecked(gb + d + 6);
                let fw6 = *forget.get_unchecked(gb + d + 6);
                let gw7 = *gain.get_unchecked(gb + d + 7);
                let fw7 = *forget.get_unchecked(gb + d + 7);
                let hv0 = *hs.add(d);
                let hv1 = *hs.add(d + 1);
                let hv2 = *hs.add(d + 2);
                let hv3 = *hs.add(d + 3);
                let hv4 = *hs.add(d + 4);
                let hv5 = *hs.add(d + 5);
                let hv6 = *hs.add(d + 6);
                let hv7 = *hs.add(d + 7);
                g = fma_rn_f32(gw0, hv0, g);
                f = fma_rn_f32(fw0, hv0, f);
                g = fma_rn_f32(gw1, hv1, g);
                f = fma_rn_f32(fw1, hv1, f);
                g = fma_rn_f32(gw2, hv2, g);
                f = fma_rn_f32(fw2, hv2, f);
                g = fma_rn_f32(gw3, hv3, g);
                f = fma_rn_f32(fw3, hv3, f);
                g = fma_rn_f32(gw4, hv4, g);
                f = fma_rn_f32(fw4, hv4, f);
                g = fma_rn_f32(gw5, hv5, g);
                f = fma_rn_f32(fw5, hv5, f);
                g = fma_rn_f32(gw6, hv6, g);
                f = fma_rn_f32(fw6, hv6, f);
                g = fma_rn_f32(gw7, hv7, g);
                f = fma_rn_f32(fw7, hv7, f);
            }
            d += OUT_UNROLL;
        }
        while d < rank {
            // SAFETY: as above.
            unsafe {
                let hv = *hs.add(d);
                g = fma_rn_f32(*gain.get_unchecked(gb + d), hv, g);
                f = fma_rn_f32(*forget.get_unchecked(gb + d), hv, f);
            }
            d += 1;
        }
        let one_m_f = 1.0 - f.tanh();
        let gm = g.tanh();
        // SAFETY: t · n + j < m · n <= cur.len() and msg.len() by the contract.
        let (cv, mv) = unsafe { (*cur.get_unchecked(t * n + j), *msg.get_unchecked(t * n + j)) };
        let out = add_rn_f32(mul_rn_f32(one_m_f, cv), mul_rn_f32(gm, mv));
        let fin = out.is_finite();
        ok = fin;
        // SAFETY: t · n + j < m · n <= y.len(); this thread alone writes the value.
        unsafe { *y.get_unchecked_mut(t * n + j) = if fin { out } else { f32::NAN } };
    }
    if !ok {
        fault.raise(FaultSite::F32Product);
    }
}

#[cuda_module]
mod loop_control_kernels {
    use super::*;

    /// The controller of one module over `m` rows (module doc), one block a row. `n` and `rank`
    /// are runtime arguments bounded by [`MAX_N`] and [`MAX_RANK`]; the shapes are host-checked.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            cur.len() >= m * n,
            msg.len() >= m * n,
            down.len() >= rank * 2 * n,
            gain.len() >= n * rank,
            forget.len() >= n * rank,
            y.len() >= m * n,
            m >= 1,
            n >= 1,
            n <= 4608,
            rank >= 1,
            rank <= 64
        )
    )]
    pub fn loop_control(
        cur: &[f32],
        msg: &[f32],
        down: &[f32],
        gain: &[f32],
        forget: &[f32],
        n: u32,
        rank: u32,
        m: u32,
        eps: f32,
        fault: FaultSink,
        mut y: DisjointSlice<f32>,
    ) {
        static mut HNORM: SharedArray<f32, HNORM_LEN> = SharedArray::UNINIT;
        static mut H: SharedArray<f32, H_LEN> = SharedArray::UNINIT;
        static mut PART: SharedArray<f32, PART_LEN> = SharedArray::UNINIT;
        static mut RSTD: SharedArray<f32, RSTD_LEN> = SharedArray::UNINIT;

        let t = thread::blockIdx_x() as usize;
        if t >= m as usize {
            return; // block-uniform
        }
        let tid = thread::threadIdx_x() as usize;
        let (n, rank) = (n as usize, rank as usize);
        // SAFETY: the block's own shared allocations; the raw form is the only way to reach them
        // without a reference to a `static mut`. Every access is below its length (`j < 2n <=
        // HNORM_LEN`, `r < rank <= H_LEN`, `tid / 32 < WARPS`) and ordered by `sync_threads`.
        let hn = unsafe { SharedArray::as_raw_mut_ptr(&raw mut HNORM) };
        // SAFETY: as above.
        let h = unsafe { SharedArray::as_raw_mut_ptr(&raw mut H) };
        // SAFETY: as above.
        let ws = unsafe { SharedArray::as_raw_mut_ptr(&raw mut PART) };
        // SAFETY: as above.
        let rs = unsafe { SharedArray::as_raw_mut_ptr(&raw mut RSTD) };

        // Stage the raw row and its two sums of squares.
        let (mut sq_c, mut sq_m) = (0.0f32, 0.0f32);
        let mut j = tid;
        while j < n {
            // SAFETY: t · n + j < m · n <= cur.len() and msg.len() by the contract; this thread
            // alone writes the staged value (j and n + j, one a thread).
            let (c, g) = unsafe { (*cur.get_unchecked(t * n + j), *msg.get_unchecked(t * n + j)) };
            sq_c += c * c;
            sq_m += g * g;
            // SAFETY: j and n + j < 2n <= HNORM_LEN; this thread alone writes them.
            unsafe {
                *hn.add(j) = c;
                *hn.add(n + j) = g;
            }
            j += THREADS;
        }
        let sq_c = warp::reduce_sum_f32(sq_c);
        let sq_m = warp::reduce_sum_f32(sq_m);
        if warp::lane_id() == 0 {
            // SAFETY: tid / 32 < WARPS; one lane a warp writes its pair.
            unsafe {
                *ws.add((tid / 32) * 2) = sq_c;
                *ws.add((tid / 32) * 2 + 1) = sq_m;
            }
        }
        thread::sync_threads();
        if tid == 0 {
            // SAFETY: every pair was written before the barrier above.
            let mean_c = unsafe { tree(ws, 0) } / n as f32;
            // SAFETY: as above.
            let mean_m = unsafe { tree(ws, 1) } / n as f32;
            let rstd_c = rms_scale_of_mean_checked(mean_c, eps, fault, true);
            let rstd_m = rms_scale_of_mean_checked(mean_m, eps, fault, true);
            // SAFETY: thread 0 alone writes RSTD here.
            unsafe {
                *rs.add(0) = rstd_c;
                *rs.add(1) = rstd_m;
            }
        }
        thread::sync_threads();
        // SAFETY: written before the barrier above.
        let (rstd_c, rstd_m) = unsafe { (*rs.add(0), *rs.add(1)) };

        // Scale the staged halves in place: the normed values the down-dots read.
        let mut j = tid;
        while j < n {
            // SAFETY: j < 2n <= HNORM_LEN; this thread alone writes the value.
            unsafe {
                *hn.add(j) = mul_rn_f32(*hn.add(j), rstd_c);
                *hn.add(n + j) = mul_rn_f32(*hn.add(n + j), rstd_m);
            }
            j += THREADS;
        }
        thread::sync_threads();

        // The down-dots: warp `w` takes output `r = w, w + WARPS, …`, lane `l` the row's values
        // `l, l + 32, …`, ascending.
        let mut ok = true;
        let mut r = tid / 32;
        while r < rank {
            let mut acc = 0.0f32;
            let mut k = tid % 32;
            let mut first = true;
            while k < 2 * n {
                // SAFETY: r · 2n + k < rank · 2n <= down.len() by the contract; k < 2n <=
                // HNORM_LEN, written before the barrier above.
                let w = unsafe { *down.get_unchecked(r * 2 * n + k) };
                // SAFETY: as above.
                let v = unsafe { *hn.add(k) };
                acc = if first {
                    first = false;
                    mul_rn_f32(w, v)
                } else {
                    fma_rn_f32(w, v, acc)
                };
                k += 32;
            }
            let acc = warp::reduce_sum_f32(acc);
            if warp::lane_id() == 0 {
                // SAFETY: r < rank <= H_LEN; one lane writes the output.
                unsafe { *h.add(r) = acc };
            }
            r += WARPS;
        }
        thread::sync_threads();

        // The gain and forget rows, the combine: each thread its output value.
        let mut j = tid;
        while j < n {
            let mut g = 0.0f32;
            let mut f = 0.0f32;
            let mut first = true;
            let mut d = 0usize;
            while d < rank {
                // SAFETY: j · rank + d < n · rank <= gain.len() and forget.len() by the contract;
                // d < rank <= H_LEN, written before the barrier above.
                let (gw, fw) = unsafe {
                    (
                        *gain.get_unchecked(j * rank + d),
                        *forget.get_unchecked(j * rank + d),
                    )
                };
                // SAFETY: as above.
                let hv = unsafe { *h.add(d) };
                if first {
                    first = false;
                    g = mul_rn_f32(gw, hv);
                    f = mul_rn_f32(fw, hv);
                } else {
                    g = fma_rn_f32(gw, hv, g);
                    f = fma_rn_f32(fw, hv, f);
                }
                d += 1;
            }
            let one_m_f = 1.0 - f.tanh();
            let gm = g.tanh();
            // SAFETY: t · n + j < m · n <= cur.len() and msg.len() by the contract.
            let (cv, mv) =
                unsafe { (*cur.get_unchecked(t * n + j), *msg.get_unchecked(t * n + j)) };
            let out = add_rn_f32(mul_rn_f32(one_m_f, cv), mul_rn_f32(gm, mv));
            let fin = out.is_finite();
            ok &= fin;
            // SAFETY: t · n + j < m · n <= y.len(); this thread alone writes the value.
            unsafe { *y.get_unchecked_mut(t * n + j) = if fin { out } else { f32::NAN } };
            j += THREADS;
        }
        if !ok {
            fault.raise(FaultSite::F32Product);
        }
    }

    /// The down launch of a pair over `modules` modules and `m` rows (module doc): block `bid`
    /// is output group `bid % groups` of row `(bid / groups) % m` of module `bid / (groups ·
    /// m)`, `groups = ceil(rank / DOWN_OUTS)`. A one-module launch passes module 0's buffers for
    /// all three sets. The shapes are host-checked.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            cur0.len() >= m * n,
            msg0.len() >= m * n,
            down0.len() >= rank * 2 * n,
            cur1.len() >= m * n,
            msg1.len() >= m * n,
            down1.len() >= rank * 2 * n,
            cur2.len() >= m * n,
            msg2.len() >= m * n,
            down2.len() >= rank * 2 * n,
            h.len() >= modules * m * rank,
            modules >= 1,
            modules <= 3,
            m >= 1,
            n >= 1,
            n <= 4608,
            rank >= 1,
            rank <= 64
        )
    )]
    pub fn loop_control_down(
        cur0: &[f32],
        msg0: &[f32],
        down0: &[f32],
        cur1: &[f32],
        msg1: &[f32],
        down1: &[f32],
        cur2: &[f32],
        msg2: &[f32],
        down2: &[f32],
        n: u32,
        rank: u32,
        m: u32,
        modules: u32,
        eps: f32,
        fault: FaultSink,
        mut h: DisjointSlice<f32>,
    ) {
        static mut HNORM: SharedArray<f32, HNORM_LEN> = SharedArray::UNINIT;
        static mut PART: SharedArray<f32, PART_LEN> = SharedArray::UNINIT;
        static mut RSTD: SharedArray<f32, RSTD_LEN> = SharedArray::UNINIT;

        let (n, rank, m) = (n as usize, rank as usize, m as usize);
        let groups = rank.div_ceil(DOWN_OUTS);
        let bid = thread::blockIdx_x() as usize;
        if bid >= groups * m * modules as usize {
            return; // block-uniform
        }
        let grp = bid % groups;
        let t = (bid / groups) % m;
        let mi = bid / (groups * m);
        let (cur, msg, down) = match mi {
            0 => (cur0, msg0, down0),
            1 => (cur1, msg1, down1),
            _ => (cur2, msg2, down2),
        };
        let tid = thread::threadIdx_x() as usize;
        // SAFETY: the block's own shared allocations; the raw form is the only way to reach them
        // without a reference to a `static mut`. Every access is below its length (`j < 2n <=
        // HNORM_LEN`, `tid / 32 < WARPS`) and ordered by `sync_threads`.
        let hn = unsafe { SharedArray::as_raw_mut_ptr(&raw mut HNORM) };
        // SAFETY: as above.
        let ws = unsafe { SharedArray::as_raw_mut_ptr(&raw mut PART) };
        // SAFETY: as above.
        let rs = unsafe { SharedArray::as_raw_mut_ptr(&raw mut RSTD) };

        // Stage the raw row and its two sums of squares (this order is the gate).
        let (mut sq_c, mut sq_m) = (0.0f32, 0.0f32);
        let mut j = tid;
        while j < n {
            // SAFETY: t · n + j < m · n <= cur.len() and msg.len() by the contract; this thread
            // alone writes the staged value (j and n + j, one a thread).
            let (c, g) = unsafe { (*cur.get_unchecked(t * n + j), *msg.get_unchecked(t * n + j)) };
            sq_c += c * c;
            sq_m += g * g;
            // SAFETY: j and n + j < 2n <= HNORM_LEN; this thread alone writes them.
            unsafe {
                *hn.add(j) = c;
                *hn.add(n + j) = g;
            }
            j += THREADS;
        }
        let sq_c = warp::reduce_sum_f32(sq_c);
        let sq_m = warp::reduce_sum_f32(sq_m);
        if warp::lane_id() == 0 {
            // SAFETY: tid / 32 < WARPS; one lane a warp writes its pair.
            unsafe {
                *ws.add((tid / 32) * 2) = sq_c;
                *ws.add((tid / 32) * 2 + 1) = sq_m;
            }
        }
        thread::sync_threads();
        if tid == 0 {
            // SAFETY: every pair was written before the barrier above.
            let mean_c = unsafe { tree(ws, 0) } / n as f32;
            // SAFETY: as above.
            let mean_m = unsafe { tree(ws, 1) } / n as f32;
            let rstd_c = rms_scale_of_mean_checked(mean_c, eps, fault, true);
            let rstd_m = rms_scale_of_mean_checked(mean_m, eps, fault, true);
            // SAFETY: thread 0 alone writes RSTD here.
            unsafe {
                *rs.add(0) = rstd_c;
                *rs.add(1) = rstd_m;
            }
        }
        thread::sync_threads();
        // SAFETY: written before the barrier above.
        let (rstd_c, rstd_m) = unsafe { (*rs.add(0), *rs.add(1)) };

        let mut j = tid;
        while j < n {
            // SAFETY: j < 2n <= HNORM_LEN; this thread alone writes the value.
            unsafe {
                *hn.add(j) = mul_rn_f32(*hn.add(j), rstd_c);
                *hn.add(n + j) = mul_rn_f32(*hn.add(n + j), rstd_m);
            }
            j += THREADS;
        }
        thread::sync_threads();

        // The block's down-dots: warp `w < DOWN_OUTS` takes output `r = grp · DOWN_OUTS + w`,
        // lane `l` the row's values `l, l + 32, …`, ascending (this order is the gate: the
        // first product rounds alone, the loads of an unrolled step are issued before its
        // fused multiply-adds, which stay ascending).
        let w = tid / 32;
        let r = grp * DOWN_OUTS + w;
        if w < DOWN_OUTS && r < rank {
            let base = r * 2 * n;
            let two_n = 2 * n;
            let mut acc = 0.0f32;
            let mut k = tid % 32;
            if k < two_n {
                // SAFETY: r · 2n + k < rank · 2n <= down.len() by the contract; k < 2n <=
                // HNORM_LEN, written before the barrier above.
                acc = unsafe { mul_rn_f32(*down.get_unchecked(base + k), *hn.add(k)) };
                k += 32;
            }
            while k + 32 * (DOWN_UNROLL - 1) < two_n {
                // SAFETY: the last index is base + k + 32 · 15 < base + 2n <= down.len(); the
                // staged indices < 2n <= HNORM_LEN, written before the barrier above.
                unsafe {
                    let w0 = *down.get_unchecked(base + k);
                    let w1 = *down.get_unchecked(base + k + 32);
                    let w2 = *down.get_unchecked(base + k + 64);
                    let w3 = *down.get_unchecked(base + k + 96);
                    let w4 = *down.get_unchecked(base + k + 128);
                    let w5 = *down.get_unchecked(base + k + 160);
                    let w6 = *down.get_unchecked(base + k + 192);
                    let w7 = *down.get_unchecked(base + k + 224);
                    let w8 = *down.get_unchecked(base + k + 256);
                    let w9 = *down.get_unchecked(base + k + 288);
                    let w10 = *down.get_unchecked(base + k + 320);
                    let w11 = *down.get_unchecked(base + k + 352);
                    let w12 = *down.get_unchecked(base + k + 384);
                    let w13 = *down.get_unchecked(base + k + 416);
                    let w14 = *down.get_unchecked(base + k + 448);
                    let w15 = *down.get_unchecked(base + k + 480);
                    let v0 = *hn.add(k);
                    let v1 = *hn.add(k + 32);
                    let v2 = *hn.add(k + 64);
                    let v3 = *hn.add(k + 96);
                    let v4 = *hn.add(k + 128);
                    let v5 = *hn.add(k + 160);
                    let v6 = *hn.add(k + 192);
                    let v7 = *hn.add(k + 224);
                    let v8 = *hn.add(k + 256);
                    let v9 = *hn.add(k + 288);
                    let v10 = *hn.add(k + 320);
                    let v11 = *hn.add(k + 352);
                    let v12 = *hn.add(k + 384);
                    let v13 = *hn.add(k + 416);
                    let v14 = *hn.add(k + 448);
                    let v15 = *hn.add(k + 480);
                    acc = fma_rn_f32(w0, v0, acc);
                    acc = fma_rn_f32(w1, v1, acc);
                    acc = fma_rn_f32(w2, v2, acc);
                    acc = fma_rn_f32(w3, v3, acc);
                    acc = fma_rn_f32(w4, v4, acc);
                    acc = fma_rn_f32(w5, v5, acc);
                    acc = fma_rn_f32(w6, v6, acc);
                    acc = fma_rn_f32(w7, v7, acc);
                    acc = fma_rn_f32(w8, v8, acc);
                    acc = fma_rn_f32(w9, v9, acc);
                    acc = fma_rn_f32(w10, v10, acc);
                    acc = fma_rn_f32(w11, v11, acc);
                    acc = fma_rn_f32(w12, v12, acc);
                    acc = fma_rn_f32(w13, v13, acc);
                    acc = fma_rn_f32(w14, v14, acc);
                    acc = fma_rn_f32(w15, v15, acc);
                }
                k += 32 * DOWN_UNROLL;
            }
            while k < two_n {
                // SAFETY: as the first value.
                acc = unsafe { fma_rn_f32(*down.get_unchecked(base + k), *hn.add(k), acc) };
                k += 32;
            }
            let acc = warp::reduce_sum_f32(acc);
            if warp::lane_id() == 0 {
                // SAFETY: (mi · m + t) · rank + r < modules · m · rank <= h.len() by the
                // contract; one lane writes the output.
                unsafe { *h.get_unchecked_mut((mi * m + t) * rank + r) = acc };
            }
        }
    }

    /// The out launch of one module over `m` rows (module doc): block `bid` is column group
    /// `bid % cols` of row `bid / cols`, `cols = ceil(n / OUT_COLS)`.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            cur.len() >= m * n,
            msg.len() >= m * n,
            gain.len() >= n * rank,
            forget.len() >= n * rank,
            h.len() >= m * rank,
            y.len() >= m * n,
            m >= 1,
            n >= 1,
            n <= 4608,
            rank >= 1,
            rank <= 64
        )
    )]
    pub fn loop_control_out(
        cur: &[f32],
        msg: &[f32],
        gain: &[f32],
        forget: &[f32],
        h: &[f32],
        n: u32,
        rank: u32,
        m: u32,
        fault: FaultSink,
        mut y: DisjointSlice<f32>,
    ) {
        static mut HS: SharedArray<f32, H_LEN> = SharedArray::UNINIT;
        let (n, rank, m) = (n as usize, rank as usize, m as usize);
        let cols = n.div_ceil(OUT_COLS);
        let bid = thread::blockIdx_x() as usize;
        if bid >= cols * m {
            return; // block-uniform
        }
        // SAFETY: the block's own shared allocation; every access is below H_LEN and ordered by
        // `sync_threads` inside `out_block`.
        let hs = unsafe { SharedArray::as_raw_mut_ptr(&raw mut HS) };
        // SAFETY: bid < cols · m, so the row `bid / cols` < m and its `h` row ends within
        // m · rank <= h.len(); the contract bounds every slice.
        unsafe {
            out_block(
                cur,
                msg,
                gain,
                forget,
                h,
                hs,
                &mut y,
                n,
                rank,
                bid / cols,
                bid % cols,
                (bid / cols) * rank,
                fault,
            );
        }
    }

    /// The out launch of the three attention modules (module doc): block `bid` is column group
    /// `bid % cols` of row `(bid / cols) % m` of module `bid / (cols · m)`.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            cur0.len() >= m * n,
            msg0.len() >= m * n,
            gain0.len() >= n * rank,
            forget0.len() >= n * rank,
            y0.len() >= m * n,
            cur1.len() >= m * n,
            msg1.len() >= m * n,
            gain1.len() >= n * rank,
            forget1.len() >= n * rank,
            y1.len() >= m * n,
            cur2.len() >= m * n,
            msg2.len() >= m * n,
            gain2.len() >= n * rank,
            forget2.len() >= n * rank,
            y2.len() >= m * n,
            h.len() >= 3 * m * rank,
            m >= 1,
            n >= 1,
            n <= 4608,
            rank >= 1,
            rank <= 64
        )
    )]
    pub fn loop_control_out3(
        cur0: &[f32],
        msg0: &[f32],
        gain0: &[f32],
        forget0: &[f32],
        cur1: &[f32],
        msg1: &[f32],
        gain1: &[f32],
        forget1: &[f32],
        cur2: &[f32],
        msg2: &[f32],
        gain2: &[f32],
        forget2: &[f32],
        h: &[f32],
        n: u32,
        rank: u32,
        m: u32,
        fault: FaultSink,
        mut y0: DisjointSlice<f32>,
        mut y1: DisjointSlice<f32>,
        mut y2: DisjointSlice<f32>,
    ) {
        static mut HS: SharedArray<f32, H_LEN> = SharedArray::UNINIT;
        let (n, rank, m) = (n as usize, rank as usize, m as usize);
        let cols = n.div_ceil(OUT_COLS);
        let bid = thread::blockIdx_x() as usize;
        if bid >= cols * m * MAX_MODULES {
            return; // block-uniform
        }
        let grp = bid % cols;
        let t = (bid / cols) % m;
        let mi = bid / (cols * m);
        // SAFETY: the block's own shared allocation; every access is below H_LEN and ordered by
        // `sync_threads` inside `out_block`.
        let hs = unsafe { SharedArray::as_raw_mut_ptr(&raw mut HS) };
        let h_row = (mi * m + t) * rank;
        // SAFETY: t < m, mi < 3, grp < cols; the contract bounds every slice and `h` holds the
        // three modules' rows.
        unsafe {
            match mi {
                0 => out_block(
                    cur0, msg0, gain0, forget0, h, hs, &mut y0, n, rank, t, grp, h_row, fault,
                ),
                1 => out_block(
                    cur1, msg1, gain1, forget1, h, hs, &mut y1, n, rank, t, grp, h_row, fault,
                ),
                _ => out_block(
                    cur2, msg2, gain2, forget2, h, hs, &mut y2, n, rank, t, grp, h_row, fault,
                ),
            }
        }
    }
}

/// The loaded module. Owns no stream: each enqueue takes the engine stream.
pub struct LoopControlKernels {
    module: loop_control_kernels::LoadedModule,
}

/// What one controller reads and writes (module doc for the layouts).
pub struct LoopControlArgs<'a> {
    /// The sublayer's input, `m` rows of `n` (the mixed input the message merges into).
    pub cur: &'a DeviceBuffer<f32>,
    /// The message, `m` rows of `n` (its mix's row).
    pub msg: &'a DeviceBuffer<f32>,
    /// The module's down projection, `[rank][2n]` as the file lays it.
    pub down: &'a DeviceBuffer<f32>,
    /// The module's gain rows, `[n][rank]` as the file lays it.
    pub gain: &'a DeviceBuffer<f32>,
    /// The module's forget rows, `[n][rank]` as the file lays it.
    pub forget: &'a DeviceBuffer<f32>,
    pub n: usize,
    pub rank: usize,
    pub m: usize,
    pub eps: f32,
    pub fault: FaultSink,
}

impl LoopControlKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<LoopControlKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for the module above; the
        // launcher checks the launch contract.
        let module = unsafe { crate::shared_module!(loop_control_kernels, ctx)? };
        Ok(LoopControlKernels { module })
    }

    /// Refuse by name a shape or a buffer the launches do not take: no row, width or rank, a
    /// width over [`MAX_N`], a rank over [`MAX_RANK`], a buffer shorter than the launch needs, an
    /// epsilon that is not positive and finite.
    fn check(what: &'static str, a: &LoopControlArgs<'_>, y_len: usize) -> Result<(), GpuError> {
        let LoopControlArgs {
            cur,
            msg,
            down,
            gain,
            forget,
            n,
            rank,
            m,
            eps,
            ..
        } = *a;
        let bad = |detail: String| Err(GpuError::shape(what, detail));
        if n == 0 || rank == 0 || m == 0 {
            return bad(format!("{m} rows of {n} values at rank {rank}"));
        }
        if n > MAX_N {
            return bad(format!("width {n}: the entry stages at most {MAX_N}"));
        }
        if rank > MAX_RANK {
            return bad(format!("rank {rank}: the entry takes at most {MAX_RANK}"));
        }
        if !(eps.is_finite() && eps > 0.0) {
            return bad(format!("epsilon {eps}: a positive finite one"));
        }
        crate::short(
            what,
            &[
                ("cur", cur.len(), m * n),
                ("msg", msg.len(), m * n),
                ("down", down.len(), rank * 2 * n),
                ("gain", gain.len(), n * rank),
                ("forget", forget.len(), n * rank),
                ("y", y_len, m * n),
            ],
        )
    }

    /// Enqueue the controller of `a` into `y` (`m` rows of `n`) as the pair of launches of the
    /// module doc: `ceil(rank / DOWN_OUTS) · m` blocks, then `ceil(n / OUT_COLS) · m`. `h` is
    /// the scratch, [`scratch_len`]`(1, m, rank)` long. A shape the launches do not take is
    /// refused by name with nothing enqueued (see `check`), as is a short `h`. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue(
        &self,
        stream: &CudaStream,
        a: LoopControlArgs<'_>,
        h: &mut DeviceBuffer<f32>,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "loop_control::enqueue";
        Self::check(WHAT, &a, y.len())?;
        let (n, rank, m) = (a.n, a.rank, a.m);
        crate::short(WHAT, &[("h", h.len(), scratch_len(1, m, rank))])?;
        let (n32, rank32, m32) = (
            launch_u32(WHAT, "n", n)?,
            launch_u32(WHAT, "rank", rank)?,
            launch_u32(WHAT, "m", m)?,
        );
        let down_blocks = launch_u32(WHAT, "down blocks", rank.div_ceil(DOWN_OUTS) * m)?;
        let out_blocks = launch_u32(WHAT, "out blocks", n.div_ceil(OUT_COLS) * m)?;
        let prep = self.module.prepare_loop_control_down(LaunchConfig1D::new(
            down_blocks,
            THREADS as u32,
            0,
        ))?;
        self.module.loop_control_down(
            stream, &prep, a.cur, a.msg, a.down, a.cur, a.msg, a.down, a.cur, a.msg, a.down, n32,
            rank32, m32, 1, a.eps, a.fault, h,
        )?;
        let prep = self.module.prepare_loop_control_out(LaunchConfig1D::new(
            out_blocks,
            THREADS as u32,
            0,
        ))?;
        self.module.loop_control_out(
            stream, &prep, a.cur, a.msg, a.gain, a.forget, h, n32, rank32, m32, a.fault, y,
        )?;
        Ok(())
    }

    /// Enqueue the three attention modules `a` (their own weights and rows, `ys` their outputs)
    /// as one pair of launches: `ceil(rank / DOWN_OUTS) · m · 3` blocks, then `ceil(n /
    /// OUT_COLS) · m · 3`. The three share `n`, `rank`, `m`, `eps` and the fault sink of
    /// `a[0]`; a module of another shape is refused by name. `h` is the scratch,
    /// [`scratch_len`]`(3, m, rank)` long. Refusals, asynchrony and capture as
    /// [`LoopControlKernels::enqueue`].
    pub fn enqueue_attn(
        &self,
        stream: &CudaStream,
        a: [LoopControlArgs<'_>; MAX_MODULES],
        h: &mut DeviceBuffer<f32>,
        ys: [&mut DeviceBuffer<f32>; MAX_MODULES],
    ) -> Result<(), GpuError> {
        const WHAT: &str = "loop_control::enqueue_attn";
        let [y0, y1, y2] = ys;
        for (i, (module, y_len)) in a.iter().zip([y0.len(), y1.len(), y2.len()]).enumerate() {
            Self::check(WHAT, module, y_len)?;
            if (module.n, module.rank, module.m) != (a[0].n, a[0].rank, a[0].m)
                || module.eps != a[0].eps
            {
                return Err(GpuError::shape(
                    WHAT,
                    format!(
                        "module {i} is {} rows of {} at rank {} (eps {}), module 0 is {} rows                          of {} at rank {} (eps {}): the modules share one shape",
                        module.m,
                        module.n,
                        module.rank,
                        module.eps,
                        a[0].m,
                        a[0].n,
                        a[0].rank,
                        a[0].eps,
                    ),
                ));
            }
        }
        let (n, rank, m) = (a[0].n, a[0].rank, a[0].m);
        crate::short(WHAT, &[("h", h.len(), scratch_len(MAX_MODULES, m, rank))])?;
        let (n32, rank32, m32) = (
            launch_u32(WHAT, "n", n)?,
            launch_u32(WHAT, "rank", rank)?,
            launch_u32(WHAT, "m", m)?,
        );
        let down_blocks = launch_u32(
            WHAT,
            "down blocks",
            rank.div_ceil(DOWN_OUTS) * m * MAX_MODULES,
        )?;
        let out_blocks = launch_u32(WHAT, "out blocks", n.div_ceil(OUT_COLS) * m * MAX_MODULES)?;
        let prep = self.module.prepare_loop_control_down(LaunchConfig1D::new(
            down_blocks,
            THREADS as u32,
            0,
        ))?;
        self.module.loop_control_down(
            stream,
            &prep,
            a[0].cur,
            a[0].msg,
            a[0].down,
            a[1].cur,
            a[1].msg,
            a[1].down,
            a[2].cur,
            a[2].msg,
            a[2].down,
            n32,
            rank32,
            m32,
            MAX_MODULES as u32,
            a[0].eps,
            a[0].fault,
            h,
        )?;
        let prep = self.module.prepare_loop_control_out3(LaunchConfig1D::new(
            out_blocks,
            THREADS as u32,
            0,
        ))?;
        self.module.loop_control_out3(
            stream,
            &prep,
            a[0].cur,
            a[0].msg,
            a[0].gain,
            a[0].forget,
            a[1].cur,
            a[1].msg,
            a[1].gain,
            a[1].forget,
            a[2].cur,
            a[2].msg,
            a[2].gain,
            a[2].forget,
            h,
            n32,
            rank32,
            m32,
            a[0].fault,
            y0,
            y1,
            y2,
        )?;
        Ok(())
    }

    /// Enqueue the controller of `a` into `y` as the one fused launch, one block of [`THREADS`]
    /// threads a row: the gate's bit-identity reference for [`LoopControlKernels::enqueue`].
    /// Refusals as [`LoopControlKernels::enqueue`].
    pub fn enqueue_single_block(
        &self,
        stream: &CudaStream,
        a: LoopControlArgs<'_>,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "loop_control::enqueue_single_block";
        Self::check(WHAT, &a, y.len())?;
        let LoopControlArgs {
            cur,
            msg,
            down,
            gain,
            forget,
            n,
            rank,
            m,
            eps,
            fault,
        } = a;
        let (n32, rank32, m32) = (
            launch_u32(WHAT, "n", n)?,
            launch_u32(WHAT, "rank", rank)?,
            launch_u32(WHAT, "m", m)?,
        );
        let prep = self
            .module
            .prepare_loop_control(LaunchConfig1D::new(m32, THREADS as u32, 0))?;
        self.module.loop_control(
            stream, &prep, cur, msg, down, gain, forget, n32, rank32, m32, eps, fault, y,
        )?;
        Ok(())
    }
}
