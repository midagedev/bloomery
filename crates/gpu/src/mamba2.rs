//! The Mamba-2 decode step on the card: one launch per layer for one token,
//! the causal conv with its bias and SiLU, the selective scan at state 128,
//! the `D` skip, the `z` gate and the grouped RMS norm, with the conv ring
//! and the scan state read and written in place.
//!
//! The rule, per token at position `p`, head `h` (`n_head` heads of
//! `head_dim` channels, `n_group` groups of `per = n_head / n_group` heads
//! that share one `B` and one `C`, `conv_dim = d_inner + 2·n_group·128`
//! conv channels; the host twin is `model::mamba2`):
//!
//! ```text
//! [x; B; C] = SiLU(conv4(xbc) + bias)       depthwise causal conv over xbc's last 4 inputs
//! dt        = dt_raw + dt_bias              σ = dt            if dt > 20
//!                                              = log1p(exp(dt))  otherwise
//! dA        = exp(σ·A)                      A, D: one float per head
//! s[n]      = s[n]·dA + B[n]·(x·σ)          the scan, state n = 0..128 of a channel
//! y         = Σ_n s[n]·C[n]                 lane l's four n = l + 32j, then the xor butterfly
//! y         = y + x·D                       then y = SiLU(z)·y
//! y         = RMSNorm_gain(y)               per group of d_inner / n_group channels, if a gain is given
//! ```
//!
//! Layouts, all f32:
//! - `xbc` `[conv_dim]` (`x`, then `B` per group, then `C` per group), `z`
//!   `[d_inner]`, `y` `[d_inner]`, channel `c + head·head_dim`; `dt`,
//!   `dt_bias`, `a`, `d` `[n_head]`; the conv weight `[conv_dim][4]` (tap
//!   fastest, tap 0 the oldest input), its bias `[conv_dim]`; the gain
//!   `[d_inner]`;
//! - the scan state `[d_inner][128]`, `n` fastest; updated in place;
//! - the conv ring `[RING_ROWS][conv_dim]`, the ring of the linear-attention
//!   conv (`linear::conv`, [`RING_ROWS`]): the input of position `p` lives in
//!   slot `p mod RING_ROWS`, and the call reads its predecessors at distance
//!   1 to 3 from the slots of `p − 1 .. p − 3` ([`ring_input`]: zero while
//!   `p < d`, the sequence start). The position is a word of the device
//!   buffer `pos` (`pos[pos_at]`), so one captured graph serves every step.
//!
//! Geometry. One block (four warps) takes 16 channels of one head, a warp
//! four of them, so `n_head · ⌈head_dim/16⌉` blocks, group-major (a group's
//! blocks are contiguous). A block first stages its conv outputs in shared
//! memory — the group's 128 `B` and 128 `C` channels (every block of the
//! group computes them again: the conv rows are L2-resident, the scan state
//! is the stream) and its own `x` channels — and writes the ring slots it
//! owns: its `x` channels, and a group's `B` and `C` from the group's first
//! block. A lane holds state `n = lane + 32j` (`j` = 0..4) of each channel
//! and its `B` and `C` values, so a channel is four loads, the update, one
//! xor butterfly and four stores; lane 0 finishes it.
//!
//! The norm needs a whole group's `y`, which no one block holds. Each block
//! writes its channels' `y` (after the gate) to the output, and its partial
//! sum of squares in f64 to the scratch's `part[block]`, in channel order;
//! a fence and one atomic ticket per block follow, and the block that draws
//! the group's last ticket puts the count back to zero (for the next launch
//! or graph replay), sums the group's partials in block order, and rewrites
//! the group's `y` normalized, so the launch is one node and its result does
//! not depend on which block finishes last. With no gain there is no ticket
//! and `y` is the gated value. One node per layer, where conv, scan, mix
//! and norm as separate entries are four.
//!
//! Numeric contract, op for op (the host twin's order is the gate):
//! - conv: `acc = 0`, then `acc = acc + x_j·w_j` for taps 0 to 3, each
//!   product rounded; `+ bias`; [`silu`] (ik's `expf`, [`crate::linear::expf_ik`]),
//!   bit for bit the host's `qdot::swiglu` with a gate of ones;
//! - `σ` and `dA` through `exp` and `ln_1p` of the device library, the one
//!   place the card's bits are not the host's: CUDA's `expf` is within 2
//!   ulp and `log1pf` within 1, the host's libm within about 1;
//! - scan: per element `fl(fl(s·dA) + fl(B·(x·σ)))`, then `acc = acc +
//!   fl(s·C)` over `j` from 0 (no fused multiply-add: every product is a
//!   `mul.rn`); the 32 lanes by the xor butterfly at 16, 8, 4, 2, 1;
//! - `y + fl(x·D)`, then `fl(silu(z)·y)`;
//! - norm: squares rounded to f32 and summed in f64, per block in channel
//!   order and then over blocks in block order (the host's AVX2 sum runs in
//!   eight lanes: the two differ by f64 rounding, which the narrowing of the
//!   mean to f32 absorbs except on an exact rounding tie); `mean = (sum /
//!   width) as f32`, `scale = 1 / sqrt(mean + eps)`, each value
//!   `fl(fl(scale·gain)·y)`.
//!
//! A prompt pass of `U` consecutive tokens (positions `p .. p + U − 1`, `p`
//! the position word) is [`Mamba2Kernels::enqueue_steps`], whose rows are
//! bit for bit `U` steps in order. The step's ring write and norm ticket
//! race across tokens inside one launch, so the pass is up to three entries
//! on one stream:
//! - `mamba2_steps`, the step's grid: per token the conv, whose tap at
//!   distance `k` reads input row `t − k` of the pass when `t ≥ k`, else the
//!   ring slot of position `p + t − k` (the ring is read only); the scan with
//!   the state in registers over the `U` tokens, stored once at the end; `D`,
//!   the gate and the un-normed `y` row. The stage is double-buffered, so a
//!   token is one barrier;
//! - `mamba2_ring_tail`: the inputs of positions `max(p, p + U − RING_ROWS)
//!   .. p + U − 1` into their ring slots, what the steps' last writes leave;
//! - `mamba2_norm_rows` (with a gain): one block per (token, group), the
//!   step's norm arithmetic over the chunks of 16 channels in the step's
//!   block order.
//!
//! No silent failure: state other than 128, a conv of other than 4 taps, a
//! geometry that does not divide, a buffer too short and a non-finite norm
//! epsilon are refused by name at enqueue, before any launch. A non-finite
//! `dt`, bias, `A` or `D` makes `σ` non-finite, so the scan raises
//! [`FaultSite::Mamba2Scan`] instead of writing a finite `y`. A `y` that is
//! not finite — a non-finite input, a state that stopped being finite, a
//! norm whose mean square is not finite — raises the same site, and a norm
//! group holding one normalizes to NaN, never a plausible value.

use crate::fault::{FaultSink, FaultSite};
use crate::linear::{CONV_TAPS, RING_ROWS, ring_input, silu};
use crate::{GpuError, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32};
use cuda_device::float::mul_rn_f32;
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread, threadfence, warp,
};
use cuda_host::cuda_module;
pub use model::mamba2::STATE;
use model::mamba2::{Checked, Geom};
use std::sync::Arc;

/// Warps of a block.
const WARPS: usize = 4;
/// Threads of a block.
const BLOCK: usize = WARPS * 32;
/// Channels one warp walks.
const CH_PER_WARP: usize = 4;
/// Channels of one head a block takes.
const CH_PER_BLOCK: usize = WARPS * CH_PER_WARP;
/// Shared f32s a block stages: its group's `B` and `C`, then its `x`.
const STAGED: usize = 2 * STATE + CH_PER_BLOCK;
/// The pass's stage: two of the step's, token `t` in half `t mod 2`.
const STAGED2: usize = 2 * STAGED;

// The entries' launch attributes spell the block out as a literal, and the
// scan and the conv are written for these widths.
const _: () = assert!(BLOCK == 128 && STATE == 128 && CONV_TAPS == 4 && CH_PER_BLOCK == 16);

#[cuda_module]
mod mamba2_kernels {
    use super::*;

    /// One Mamba-2 decode token (module doc): block `b` is group `b / (per ·
    /// chunks)`, head `b % (per · chunks) / chunks` of it, channels
    /// `16 · (b % chunks) ..` of that head.
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
            per >= 1,
            n_group >= 1,
            head_dim >= 1,
            has_norm <= 1,
            n_head == per * n_group,
            chunks * CH_PER_BLOCK >= head_dim,
            chunks * CH_PER_BLOCK < head_dim + CH_PER_BLOCK,
            xbc.len() >= n_head * head_dim + 2 * n_group * STATE,
            dt.len() >= n_head,
            dt_bias.len() >= n_head,
            a.len() >= n_head,
            d.len() >= n_head,
            z.len() >= n_head * head_dim,
            conv_w.len() >= CONV_TAPS * (n_head * head_dim + 2 * n_group * STATE),
            conv_b.len() >= n_head * head_dim + 2 * n_group * STATE,
            gain.len() >= has_norm * n_head * head_dim,
            pos.len() >= pos_at + 1,
            y.len() >= n_head * head_dim,
            ring.len() >= RING_ROWS * (n_head * head_dim + 2 * n_group * STATE),
            state.len() >= n_head * head_dim * STATE,
            part.len() >= n_head * chunks,
            done.len() >= has_norm * n_group
        )
    )]
    pub fn mamba2_step(
        xbc: &[f32],
        dt: &[f32],
        dt_bias: &[f32],
        z: &[f32],
        a: &[f32],
        d: &[f32],
        conv_w: &[f32],
        conv_b: &[f32],
        gain: &[f32],
        pos: &[u32],
        pos_at: u32,
        eps: f32,
        n_head: u32,
        per: u32,
        n_group: u32,
        head_dim: u32,
        chunks: u32,
        has_norm: u32,
        fault: FaultSink,
        mut y: DisjointSlice<f32>,
        mut ring: DisjointSlice<f32>,
        mut state: DisjointSlice<f32>,
        mut part: DisjointSlice<f64>,
        mut done: DisjointSlice<u32>,
    ) {
        static mut STAGE: SharedArray<f32, STAGED> = SharedArray::UNINIT;
        static mut YS: SharedArray<f32, CH_PER_BLOCK> = SharedArray::UNINIT;
        static mut LAST: SharedArray<u32, 1> = SharedArray::UNINIT;

        let (n_head, per, n_group, hd, chunks) = (
            n_head as usize,
            per as usize,
            n_group as usize,
            head_dim as usize,
            chunks as usize,
        );
        let b = thread::blockIdx_x() as usize;
        if b >= n_head * chunks {
            return; // block-uniform
        }
        let tid = thread::threadIdx_x() as usize;
        let wid = tid / 32;
        let lane = tid % 32;
        let bpg = per * chunks;
        let g = b / bpg;
        let r = b % bpg;
        let h = g * per + r / chunks;
        let chunk = r % chunks;
        let inner = n_head * hd;
        let conv_dim = inner + 2 * n_group * STATE;
        // SAFETY: block-shared arrays of this block, reached without a
        // reference to a `static mut`; the barriers below order their users.
        let (cs, ys, last) = unsafe {
            (
                SharedArray::as_raw_mut_ptr(&raw mut STAGE),
                SharedArray::as_raw_mut_ptr(&raw mut YS),
                SharedArray::as_raw_mut_ptr(&raw mut LAST),
            )
        };
        let (yp, ringp, statep) = (y.as_mut_ptr(), ring.as_mut_ptr(), state.as_mut_ptr());
        // SAFETY: pos_at < pos.len() by the contract.
        let p = unsafe { *pos.get_unchecked(pos_at as usize) } as usize;

        // Stage the conv outputs: item i < 128 is `B` channel i of the
        // group, 128 <= i < 256 its `C` channel i − 128, then the block's
        // channels of `x`. The ring slot of `p` takes each channel's input
        // once: from the block that owns it.
        let c_x = h * hd + chunk * CH_PER_BLOCK;
        #[unroll]
        for round in 0usize..3 {
            let item = tid + BLOCK * round;
            if item < STAGED {
                let (c, live, writes) = if item < STATE {
                    (inner + g * STATE + item, true, r == 0)
                } else if item < 2 * STATE {
                    (
                        inner + n_group * STATE + g * STATE + (item - STATE),
                        true,
                        r == 0,
                    )
                } else {
                    let k = item - 2 * STATE;
                    (c_x + k, chunk * CH_PER_BLOCK + k < hd, true)
                };
                if live {
                    // SAFETY: c < conv_dim <= xbc.len(), conv_b.len() and
                    // CONV_TAPS·c + 3 < conv_w.len() by the contract; the
                    // ring holds RING_ROWS·conv_dim values; the slot of `p`
                    // is none of the three the conv reads, and channel `c`'s
                    // slot is written by this thread alone; item < STAGED.
                    unsafe {
                        let cur = *xbc.get_unchecked(c);
                        let x0 = ring_input(ringp.cast_const(), p, 3, conv_dim, c);
                        let x1 = ring_input(ringp.cast_const(), p, 2, conv_dim, c);
                        let x2 = ring_input(ringp.cast_const(), p, 1, conv_dim, c);
                        let w = 4 * c;
                        let mut acc = 0.0f32;
                        acc += mul_rn_f32(x0, *conv_w.get_unchecked(w));
                        acc += mul_rn_f32(x1, *conv_w.get_unchecked(w + 1));
                        acc += mul_rn_f32(x2, *conv_w.get_unchecked(w + 2));
                        acc += mul_rn_f32(cur, *conv_w.get_unchecked(w + 3));
                        *cs.add(item) = silu(acc + *conv_b.get_unchecked(c));
                        if writes {
                            *ringp.add(p % RING_ROWS * conv_dim + c) = cur;
                        }
                    }
                }
            }
        }
        thread::sync_threads();

        // The head's scalars, in every thread.
        // SAFETY: h < n_head <= dt.len(), dt_bias.len(), a.len(), d.len().
        let (dt_h, bias_h, a_h, d_h) = unsafe {
            (
                *dt.get_unchecked(h),
                *dt_bias.get_unchecked(h),
                *a.get_unchecked(h),
                *d.get_unchecked(h),
            )
        };
        let dtv = dt_h + bias_h;
        // A non-finite parameter becomes a non-finite sigma, so the scan
        // faults instead of writing a finite y (σ of −∞ would be 0).
        let fin = crate::fault::quad_finite([dt_h, bias_h, a_h, d_h]) & dtv.is_finite();
        let sigma = if !fin {
            f32::NAN
        } else if dtv <= 20.0 {
            dtv.exp().ln_1p()
        } else {
            dtv
        };
        let decay = mul_rn_f32(sigma, a_h).exp();

        // SAFETY: lane + 96 + 128 < STAGED; written before the barrier above.
        let (bq, cq) = unsafe {
            (
                [
                    *cs.add(lane),
                    *cs.add(lane + 32),
                    *cs.add(lane + 64),
                    *cs.add(lane + 96),
                ],
                [
                    *cs.add(STATE + lane),
                    *cs.add(STATE + lane + 32),
                    *cs.add(STATE + lane + 64),
                    *cs.add(STATE + lane + 96),
                ],
            )
        };

        // The warp's four channels: their state first, all loads in flight.
        let mut s = [0.0f32; CH_PER_WARP * 4];
        #[unroll]
        for i in 0usize..CH_PER_WARP {
            let ch = chunk * CH_PER_BLOCK + wid * CH_PER_WARP + i;
            if ch < hd {
                let base = (h * hd + ch) * STATE + lane;
                #[unroll]
                for j in 0usize..4 {
                    // SAFETY: (h·hd + ch)·128 + lane + 96 < n_head·hd·128 <=
                    // state.len() by the contract.
                    s[4 * i + j] = unsafe { *statep.add(base + 32 * j) };
                }
            }
        }
        #[unroll]
        for i in 0usize..CH_PER_WARP {
            let k = wid * CH_PER_WARP + i;
            let ch = chunk * CH_PER_BLOCK + k;
            if ch < hd {
                let at = h * hd + ch;
                // SAFETY: 2·STATE + k < STAGED, written before the barrier.
                let x = unsafe { *cs.add(2 * STATE + k) };
                let x_dt = mul_rn_f32(x, sigma);
                let mut acc = 0.0f32;
                #[unroll]
                for j in 0usize..4 {
                    let u = mul_rn_f32(s[4 * i + j], decay) + mul_rn_f32(bq[j], x_dt);
                    s[4 * i + j] = u;
                    acc += mul_rn_f32(u, cq[j]);
                }
                acc += warp::shuffle_xor_f32(acc, 16);
                acc += warp::shuffle_xor_f32(acc, 8);
                acc += warp::shuffle_xor_f32(acc, 4);
                acc += warp::shuffle_xor_f32(acc, 2);
                acc += warp::shuffle_xor_f32(acc, 1);
                // SAFETY: at < inner <= z.len().
                let zv = unsafe { *z.get_unchecked(at) };
                let v = mul_rn_f32(silu(zv), acc + mul_rn_f32(x, d_h));
                let base = at * STATE + lane;
                #[unroll]
                for j in 0usize..4 {
                    // SAFETY: as the loads above; each element is this
                    // lane's own, read before this store.
                    unsafe { *statep.add(base + 32 * j) = s[4 * i + j] };
                }
                if lane == 0 {
                    if !v.is_finite() {
                        fault.raise(FaultSite::Mamba2Scan);
                    }
                    // SAFETY: at < inner <= y.len(); k < CH_PER_BLOCK; lane 0
                    // of this warp is the only writer of both.
                    unsafe {
                        *yp.add(at) = v;
                        *ys.add(k) = v;
                    }
                }
            }
        }
        if has_norm == 0 {
            return;
        }

        // The group's norm. Each block's partial sum of squares, in channel
        // order, then a fence, a barrier and one ticket per block.
        thread::sync_threads();
        let (width, pbase) = (per * hd, g * bpg);
        if tid == 0 {
            let mut sum = 0.0f64;
            let mut k = 0usize;
            while k < CH_PER_BLOCK && chunk * CH_PER_BLOCK + k < hd {
                // SAFETY: k < CH_PER_BLOCK; the lanes 0 wrote it before the
                // barrier above.
                let v = unsafe { *ys.add(k) };
                sum += f64::from(mul_rn_f32(v, v));
                k += 1;
            }
            // SAFETY: b < n_head·chunks <= part.len() by the contract; this
            // block's thread 0 is the only writer.
            unsafe { *part.as_mut_ptr().add(b) = sum };
        }
        threadfence();
        thread::sync_threads();
        if tid == 0 {
            // SAFETY: g < n_group <= done.len() (has_norm = 1); every access
            // to done[g] on the card is atomic.
            let count = unsafe { DeviceAtomicU32::from_ptr(done.as_mut_ptr().add(g)) };
            // The ticket's release orders this block's writes before it; the
            // last block's acquire sees every block's.
            let is_last = count.fetch_add(1, AtomicOrdering::AcqRel) + 1 == bpg as u32;
            if is_last {
                // Every block of the group has drawn its ticket.
                count.store(0, AtomicOrdering::Relaxed);
            }
            // SAFETY: block-shared, one element, read after the barrier below.
            unsafe { *last = u32::from(is_last) };
        }
        thread::sync_threads();
        // SAFETY: written by thread 0 before the barrier above.
        if unsafe { *last } == 0 {
            return;
        }

        // The group's last block: every partial and every `y` of the group
        // is published (volatile loads: this block's L1 never held another
        // block's values, and a volatile load does not ask it).
        let pp = part.as_mut_ptr();
        let mut total = 0.0f64;
        let mut i = 0usize;
        while i < bpg {
            // SAFETY: pbase + i < (g + 1)·bpg <= n_head·chunks <= part.len().
            total += unsafe { core::ptr::read_volatile(pp.add(pbase + i)) };
            i += 1;
        }
        let mean = (total / width as f64) as f32;
        // A non-finite mean gives a NaN scale: each `o` below is NaN and raises.
        let scale = if mean.is_finite() {
            1.0 / (mean + eps).sqrt()
        } else {
            f32::NAN
        };
        let ybase = g * width;
        let mut i = tid;
        while i < width {
            let at = ybase + i;
            // SAFETY: at < (g + 1)·per·hd <= n_head·hd <= y.len() and
            // gain.len() (has_norm = 1) by the contract; value `at` is this
            // thread's alone in this pass, published by its block's fence.
            let o = unsafe {
                let v = core::ptr::read_volatile(yp.add(at));
                mul_rn_f32(mul_rn_f32(scale, *gain.get_unchecked(at)), v)
            };
            if !o.is_finite() {
                fault.raise(FaultSite::Mamba2Scan);
            }
            // SAFETY: as the load above.
            unsafe { *yp.add(at) = o };
            i += BLOCK;
        }
    }

    /// `U` tokens of one layer from the pass's rows (module doc): the step's
    /// grid and, per token, its operations in its order; the ring is read
    /// only, the state is stored once after the last token, and `y` row `t`
    /// is token `t`'s un-normed output.
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
            per >= 1,
            n_group >= 1,
            head_dim >= 1,
            u >= 1,
            n_head == per * n_group,
            chunks * CH_PER_BLOCK >= head_dim,
            chunks * CH_PER_BLOCK < head_dim + CH_PER_BLOCK,
            ld >= z0 + n_head * head_dim,
            ld >= x0 + n_head * head_dim + 2 * n_group * STATE,
            ld >= dt0 + n_head,
            rows.len() >= u * ld,
            dt_bias.len() >= n_head,
            a.len() >= n_head,
            d.len() >= n_head,
            conv_w.len() >= CONV_TAPS * (n_head * head_dim + 2 * n_group * STATE),
            conv_b.len() >= n_head * head_dim + 2 * n_group * STATE,
            pos.len() >= pos_at + 1,
            ring.len() >= RING_ROWS * (n_head * head_dim + 2 * n_group * STATE),
            y.len() >= u * n_head * head_dim,
            state.len() >= n_head * head_dim * STATE
        )
    )]
    pub fn mamba2_steps(
        rows: &[f32],
        dt_bias: &[f32],
        a: &[f32],
        d: &[f32],
        conv_w: &[f32],
        conv_b: &[f32],
        ring: &[f32],
        pos: &[u32],
        pos_at: u32,
        u: u32,
        ld: u32,
        z0: u32,
        x0: u32,
        dt0: u32,
        n_head: u32,
        per: u32,
        n_group: u32,
        head_dim: u32,
        chunks: u32,
        fault: FaultSink,
        mut y: DisjointSlice<f32>,
        mut state: DisjointSlice<f32>,
    ) {
        static mut STAGE2: SharedArray<f32, STAGED2> = SharedArray::UNINIT;

        let (n_head, per, n_group, hd, chunks) = (
            n_head as usize,
            per as usize,
            n_group as usize,
            head_dim as usize,
            chunks as usize,
        );
        let (u, ld, z0, x0, dt0) = (
            u as usize,
            ld as usize,
            z0 as usize,
            x0 as usize,
            dt0 as usize,
        );
        let b = thread::blockIdx_x() as usize;
        if b >= n_head * chunks {
            return; // block-uniform
        }
        let tid = thread::threadIdx_x() as usize;
        let wid = tid / 32;
        let lane = tid % 32;
        let bpg = per * chunks;
        let g = b / bpg;
        let r = b % bpg;
        let h = g * per + r / chunks;
        let chunk = r % chunks;
        let inner = n_head * hd;
        let conv_dim = inner + 2 * n_group * STATE;
        // SAFETY: a block-shared array of this block, reached without a
        // reference to a `static mut`; the barrier of each token orders its
        // users (module doc: token `t` stages into half `t mod 2`).
        let cs = unsafe { SharedArray::as_raw_mut_ptr(&raw mut STAGE2) };
        let (yp, statep, ringp) = (y.as_mut_ptr(), state.as_mut_ptr(), ring.as_ptr());
        // SAFETY: pos_at < pos.len() by the contract.
        let p = unsafe { *pos.get_unchecked(pos_at as usize) } as usize;
        let c_x = h * hd + chunk * CH_PER_BLOCK;

        // The head's scalars that no token moves, in every thread.
        // SAFETY: h < n_head <= dt_bias.len(), a.len(), d.len().
        let (bias_h, a_h, d_h) = unsafe {
            (
                *dt_bias.get_unchecked(h),
                *a.get_unchecked(h),
                *d.get_unchecked(h),
            )
        };

        // The warp's four channels' state, held in registers over the pass.
        let mut s = [0.0f32; CH_PER_WARP * 4];
        #[unroll]
        for i in 0usize..CH_PER_WARP {
            let ch = chunk * CH_PER_BLOCK + wid * CH_PER_WARP + i;
            if ch < hd {
                let base = (h * hd + ch) * STATE + lane;
                #[unroll]
                for j in 0usize..4 {
                    // SAFETY: (h·hd + ch)·128 + lane + 96 < n_head·hd·128 <=
                    // state.len() by the contract.
                    s[4 * i + j] = unsafe { *statep.add(base + 32 * j) };
                }
            }
        }

        let mut t = 0usize;
        while t < u {
            let half = (t % 2) * STAGED;
            let row = t * ld;
            // Stage token t's conv outputs, items as the step's.
            #[unroll]
            for round in 0usize..3 {
                let item = tid + BLOCK * round;
                if item < STAGED {
                    let (c, live) = if item < STATE {
                        (inner + g * STATE + item, true)
                    } else if item < 2 * STATE {
                        (inner + n_group * STATE + g * STATE + (item - STATE), true)
                    } else {
                        let k = item - 2 * STATE;
                        (c_x + k, chunk * CH_PER_BLOCK + k < hd)
                    };
                    if live {
                        // SAFETY: c < conv_dim, x0 + conv_dim <= ld and t < u,
                        // so every row index is < u·ld <= rows.len(); c <
                        // conv_b.len() and CONV_TAPS·c + 3 < conv_w.len(); the
                        // ring holds RING_ROWS·conv_dim values; half + item <
                        // STAGED2, this thread's item alone.
                        unsafe {
                            let cur = *rows.get_unchecked(row + x0 + c);
                            let in0 = pass_input(rows, ringp, p, t, 3, ld, x0, conv_dim, c);
                            let in1 = pass_input(rows, ringp, p, t, 2, ld, x0, conv_dim, c);
                            let in2 = pass_input(rows, ringp, p, t, 1, ld, x0, conv_dim, c);
                            let w = 4 * c;
                            let mut acc = 0.0f32;
                            acc += mul_rn_f32(in0, *conv_w.get_unchecked(w));
                            acc += mul_rn_f32(in1, *conv_w.get_unchecked(w + 1));
                            acc += mul_rn_f32(in2, *conv_w.get_unchecked(w + 2));
                            acc += mul_rn_f32(cur, *conv_w.get_unchecked(w + 3));
                            *cs.add(half + item) = silu(acc + *conv_b.get_unchecked(c));
                        }
                    }
                }
            }
            thread::sync_threads();

            // SAFETY: row + dt0 + h < u·ld <= rows.len(): dt0 + n_head <= ld.
            let dt_h = unsafe { *rows.get_unchecked(row + dt0 + h) };
            let dtv = dt_h + bias_h;
            // As the step: a non-finite parameter becomes a non-finite sigma.
            let fin = crate::fault::quad_finite([dt_h, bias_h, a_h, d_h]) & dtv.is_finite();
            let sigma = if !fin {
                f32::NAN
            } else if dtv <= 20.0 {
                dtv.exp().ln_1p()
            } else {
                dtv
            };
            let decay = mul_rn_f32(sigma, a_h).exp();

            // SAFETY: half + lane + 96 + 128 < STAGED2; written before the
            // barrier above.
            let (bq, cq) = unsafe {
                (
                    [
                        *cs.add(half + lane),
                        *cs.add(half + lane + 32),
                        *cs.add(half + lane + 64),
                        *cs.add(half + lane + 96),
                    ],
                    [
                        *cs.add(half + STATE + lane),
                        *cs.add(half + STATE + lane + 32),
                        *cs.add(half + STATE + lane + 64),
                        *cs.add(half + STATE + lane + 96),
                    ],
                )
            };
            #[unroll]
            for i in 0usize..CH_PER_WARP {
                let k = wid * CH_PER_WARP + i;
                let ch = chunk * CH_PER_BLOCK + k;
                if ch < hd {
                    let at = h * hd + ch;
                    // SAFETY: half + 2·STATE + k < STAGED2, written before
                    // the barrier.
                    let x = unsafe { *cs.add(half + 2 * STATE + k) };
                    let x_dt = mul_rn_f32(x, sigma);
                    let mut acc = 0.0f32;
                    #[unroll]
                    for j in 0usize..4 {
                        let v = mul_rn_f32(s[4 * i + j], decay) + mul_rn_f32(bq[j], x_dt);
                        s[4 * i + j] = v;
                        acc += mul_rn_f32(v, cq[j]);
                    }
                    acc += warp::shuffle_xor_f32(acc, 16);
                    acc += warp::shuffle_xor_f32(acc, 8);
                    acc += warp::shuffle_xor_f32(acc, 4);
                    acc += warp::shuffle_xor_f32(acc, 2);
                    acc += warp::shuffle_xor_f32(acc, 1);
                    // SAFETY: row + z0 + at < u·ld <= rows.len(): z0 + inner
                    // <= ld.
                    let zv = unsafe { *rows.get_unchecked(row + z0 + at) };
                    let v = mul_rn_f32(silu(zv), acc + mul_rn_f32(x, d_h));
                    if lane == 0 {
                        if !v.is_finite() {
                            fault.raise(FaultSite::Mamba2Scan);
                        }
                        // SAFETY: t·inner + at < u·inner <= y.len(); lane 0
                        // of this warp is its only writer.
                        unsafe { *yp.add(t * inner + at) = v };
                    }
                }
            }
            t += 1;
        }

        #[unroll]
        for i in 0usize..CH_PER_WARP {
            let ch = chunk * CH_PER_BLOCK + wid * CH_PER_WARP + i;
            if ch < hd {
                let base = (h * hd + ch) * STATE + lane;
                #[unroll]
                for j in 0usize..4 {
                    // SAFETY: as the loads above; each element is this lane's
                    // own.
                    unsafe { *statep.add(base + 32 * j) = s[4 * i + j] };
                }
            }
        }
    }

    /// The ring after a pass (module doc): thread `i` of the pass's last
    /// `min(U, RING_ROWS)` rows' `conv_dim` inputs writes one channel of one
    /// row into the slot of its position.
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
            u >= 1,
            conv_dim >= 1,
            ld >= x0 + conv_dim,
            rows.len() >= u * ld,
            pos.len() >= pos_at + 1,
            ring.len() >= RING_ROWS * conv_dim
        )
    )]
    pub fn mamba2_ring_tail(
        rows: &[f32],
        pos: &[u32],
        pos_at: u32,
        u: u32,
        ld: u32,
        x0: u32,
        conv_dim: u32,
        mut ring: DisjointSlice<f32>,
    ) {
        let (u, ld, x0, ch) = (u as usize, ld as usize, x0 as usize, conv_dim as usize);
        let tail = if u < RING_ROWS { u } else { RING_ROWS };
        let i = thread::blockIdx_x() as usize * BLOCK + thread::threadIdx_x() as usize;
        if i >= tail * ch {
            return;
        }
        let r = u - tail + i / ch;
        let c = i % ch;
        // SAFETY: pos_at < pos.len() by the contract.
        let p = unsafe { *pos.get_unchecked(pos_at as usize) } as usize;
        // SAFETY: r < u and x0 + c < ld, so r·ld + x0 + c < u·ld <=
        // rows.len(); the slot is < RING_ROWS and c < ch, inside the ring;
        // the tail's at most RING_ROWS consecutive positions own distinct
        // slots, so (slot, c) is this thread's alone.
        unsafe {
            *ring.as_mut_ptr().add((p + r) % RING_ROWS * ch + c) =
                *rows.get_unchecked(r * ld + x0 + c);
        }
    }

    /// The grouped norm of a pass's `y` rows in place (module doc): block `b`
    /// is token `b / n_group`, group `b % n_group`; the step's arithmetic in
    /// the step's order.
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
            per >= 1,
            n_group >= 1,
            head_dim >= 1,
            u >= 1,
            n_head == per * n_group,
            chunks * CH_PER_BLOCK >= head_dim,
            chunks * CH_PER_BLOCK < head_dim + CH_PER_BLOCK,
            gain.len() >= n_head * head_dim,
            y.len() >= u * n_head * head_dim
        )
    )]
    pub fn mamba2_norm_rows(
        gain: &[f32],
        eps: f32,
        u: u32,
        n_head: u32,
        per: u32,
        n_group: u32,
        head_dim: u32,
        chunks: u32,
        fault: FaultSink,
        mut y: DisjointSlice<f32>,
    ) {
        static mut PART: SharedArray<f64, BLOCK> = SharedArray::UNINIT;
        static mut SCALE: SharedArray<f32, 1> = SharedArray::UNINIT;

        let (n_head, per, n_group, hd, chunks) = (
            n_head as usize,
            per as usize,
            n_group as usize,
            head_dim as usize,
            chunks as usize,
        );
        let b = thread::blockIdx_x() as usize;
        if b >= u as usize * n_group {
            return; // block-uniform
        }
        let tid = thread::threadIdx_x() as usize;
        let (t, g) = (b / n_group, b % n_group);
        let (width, bpg) = (per * hd, per * chunks);
        let gbase = g * width;
        let ybase = t * n_head * hd + gbase;
        // SAFETY: block-shared arrays of this block, reached without a
        // reference to a `static mut`; the barriers below order their users.
        let (part, sc) = unsafe {
            (
                SharedArray::as_raw_mut_ptr(&raw mut PART),
                SharedArray::as_raw_mut_ptr(&raw mut SCALE),
            )
        };
        let yp = y.as_mut_ptr();

        // The step's partials: chunk `ci` (head ci / chunks of the group, its
        // 16 channels from 16·(ci mod chunks)) summed in channel order, then
        // the chunks summed in order by thread 0, BLOCK chunks a round.
        let mut total = 0.0f64;
        let mut base = 0usize;
        while base < bpg {
            let ci = base + tid;
            if ci < bpg {
                let c0 = ci % chunks * CH_PER_BLOCK;
                let first = ybase + ci / chunks * hd + c0;
                let mut sum = 0.0f64;
                let mut k = 0usize;
                while k < CH_PER_BLOCK && c0 + k < hd {
                    // SAFETY: first + k < ybase + width <= u·n_head·hd <=
                    // y.len() by the contract.
                    let v = unsafe { *yp.add(first + k) };
                    sum += f64::from(mul_rn_f32(v, v));
                    k += 1;
                }
                // SAFETY: tid < BLOCK; this thread's slot alone this round.
                unsafe { *part.add(tid) = sum };
            }
            thread::sync_threads();
            if tid == 0 {
                let mut i = 0usize;
                while i < BLOCK && base + i < bpg {
                    // SAFETY: i < BLOCK; written before the barrier above.
                    total += unsafe { *part.add(i) };
                    i += 1;
                }
            }
            thread::sync_threads();
            base += BLOCK;
        }
        if tid == 0 {
            let mean = (total / width as f64) as f32;
            // A non-finite mean gives a NaN scale: each `o` below is NaN and
            // raises.
            let scale = if mean.is_finite() {
                1.0 / (mean + eps).sqrt()
            } else {
                f32::NAN
            };
            // SAFETY: block-shared, one element, read after the barrier below.
            unsafe { *sc = scale };
        }
        thread::sync_threads();
        // SAFETY: written by thread 0 before the barrier above.
        let scale = unsafe { *sc };
        let mut i = tid;
        while i < width {
            let at = ybase + i;
            // SAFETY: at < u·n_head·hd <= y.len() and gbase + i < n_head·hd
            // <= gain.len() by the contract; value `at` is this thread's
            // alone.
            let o = unsafe {
                mul_rn_f32(
                    mul_rn_f32(scale, *gain.get_unchecked(gbase + i)),
                    *yp.add(at),
                )
            };
            if !o.is_finite() {
                fault.raise(FaultSite::Mamba2Scan);
            }
            // SAFETY: as the load above.
            unsafe { *yp.add(at) = o };
            i += BLOCK;
        }
    }
}

/// The conv input `k` positions before token `t` of a pass from position `p`
/// (`1 <= k <= 3`), channel `c`: row `t − k` of the pass when `t >= k`, else
/// the ring's (`ring_input` at position `p + t`, zero before the sequence
/// start).
///
/// # Safety
///
/// `c < ch <= ld − x0`, `t` is a row of `rows` (`(t + 1)·ld <= rows.len()`)
/// and `ring` addresses `RING_ROWS · ch` readable f32s.
#[inline(always)]
#[allow(
    clippy::too_many_arguments,
    reason = "a device helper of the pass's conv, inlined into the entry"
)]
unsafe fn pass_input(
    rows: &[f32],
    ring: *const f32,
    p: usize,
    t: usize,
    k: usize,
    ld: usize,
    x0: usize,
    ch: usize,
    c: usize,
) -> f32 {
    if t >= k {
        // SAFETY: (t − k)·ld + x0 + c < (t + 1)·ld <= rows.len().
        unsafe { *rows.get_unchecked((t - k) * ld + x0 + c) }
    } else {
        // SAFETY: the ring and c < ch, by the contract.
        unsafe { ring_input(ring, p + t, k, ch, c) }
    }
}

/// Refuse a conv other than [`CONV_TAPS`] taps, then the host twin's
/// [`model::mamba2::check`]. The host accepts any positive `d_conv`; this
/// kernel runs four.
pub fn check(what: &'static str, g: Geom) -> Result<Checked, GpuError> {
    if g.d_conv != CONV_TAPS {
        return Err(GpuError::shape(
            what,
            format!(
                "d_conv {} is refused: the conv runs {CONV_TAPS} taps",
                g.d_conv
            ),
        ));
    }
    model::mamba2::check(g).map_err(|e| GpuError::shape(what, e.to_string()))
}

/// What the grouped norm's cross-block sum needs between launches: the
/// blocks' partial sums of squares and one ticket count per group, zero at
/// rest (the group's last block puts it back). One scratch serves one launch
/// at a time on a stream; a graph captured with it replays on it.
pub struct Mamba2Scratch {
    part: DeviceBuffer<f64>,
    done: DeviceBuffer<u32>,
}

impl Mamba2Scratch {
    /// A zeroed scratch for `shape`. Load-time only. `shape` is checked again
    /// here, so a conv the host accepted and this kernel refuses never
    /// allocates.
    pub fn new(stream: &CudaStream, shape: Checked) -> Result<Mamba2Scratch, GpuError> {
        let shape = check("mamba2::Mamba2Scratch::new", shape.geom())?;
        let g = shape.geom();
        Ok(Mamba2Scratch {
            part: DeviceBuffer::zeroed(
                stream,
                g.n_heads * shape.head_dim().div_ceil(CH_PER_BLOCK),
            )?,
            done: DeviceBuffer::zeroed(stream, g.n_groups)?,
        })
    }
}

/// [`Mamba2Kernels::enqueue_step`]'s arguments: the layer's widths, the
/// token's conv input `xbc`, raw `dt` and `z` (the in-projection's split),
/// the layer's `dt_bias`, `A`, `D`, conv weight and bias and the norm's gain
/// (`None`: no norm), the norm's epsilon, the position word (`pos[pos_at]`),
/// the sink a non-finite `y` raises on, and the outputs `y`, the conv ring
/// and the scan state (the last two read and written in place).
pub struct StepArgs<'a> {
    pub shape: Checked,
    pub xbc: &'a DeviceBuffer<f32>,
    pub dt: &'a DeviceBuffer<f32>,
    pub dt_bias: &'a DeviceBuffer<f32>,
    pub z: &'a DeviceBuffer<f32>,
    pub a: &'a DeviceBuffer<f32>,
    pub d: &'a DeviceBuffer<f32>,
    pub conv_w: &'a DeviceBuffer<f32>,
    pub conv_b: &'a DeviceBuffer<f32>,
    pub norm: Option<&'a DeviceBuffer<f32>>,
    pub eps: f32,
    pub pos: &'a DeviceBuffer<u32>,
    pub pos_at: usize,
    pub fault: FaultSink,
    pub y: &'a mut DeviceBuffer<f32>,
    pub ring: &'a mut DeviceBuffer<f32>,
    pub state: &'a mut DeviceBuffer<f32>,
    pub scratch: &'a mut Mamba2Scratch,
}

/// [`Mamba2Kernels::enqueue_steps`]'s arguments: [`StepArgs`] for a pass of
/// `u` tokens. The inputs are `u` token-major rows of `rows`, row stride
/// `ld`, each holding the token's `z` at column `z0`, its conv input `xbc` at
/// `x0` and its raw `dt` at `dt0` (the in-projection's output as it lands);
/// `y` takes `u` rows of `d_inner`. The pass needs no scratch.
pub struct StepsArgs<'a> {
    pub shape: Checked,
    pub u: usize,
    pub rows: &'a DeviceBuffer<f32>,
    pub ld: usize,
    pub z0: usize,
    pub x0: usize,
    pub dt0: usize,
    pub dt_bias: &'a DeviceBuffer<f32>,
    pub a: &'a DeviceBuffer<f32>,
    pub d: &'a DeviceBuffer<f32>,
    pub conv_w: &'a DeviceBuffer<f32>,
    pub conv_b: &'a DeviceBuffer<f32>,
    pub norm: Option<&'a DeviceBuffer<f32>>,
    pub eps: f32,
    pub pos: &'a DeviceBuffer<u32>,
    pub pos_at: usize,
    pub fault: FaultSink,
    pub y: &'a mut DeviceBuffer<f32>,
    pub ring: &'a mut DeviceBuffer<f32>,
    pub state: &'a mut DeviceBuffer<f32>,
}

/// The loaded module. Owns no stream: each enqueue takes the engine stream.
pub struct Mamba2Kernels {
    module: mamba2_kernels::LoadedModule,
}

impl Mamba2Kernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<Mamba2Kernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; the launcher checks its launch contract.
        let module = unsafe { crate::shared_module!(mamba2_kernels, ctx)? };
        Ok(Mamba2Kernels { module })
    }

    /// Enqueue one decode token of one layer ([`StepArgs`], module doc):
    /// `n_head · ⌈head_dim / 16⌉` blocks of 128 threads, the ring and the
    /// state read and written in place. Refuses a state other than 128, a
    /// conv other than 4 taps, widths that do not divide, a buffer too
    /// short and a non-finite norm epsilon by name, launching nothing. One
    /// launch. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_step(&self, stream: &CudaStream, args: StepArgs<'_>) -> Result<(), GpuError> {
        let what = "mamba2::enqueue_step";
        let StepArgs {
            shape,
            xbc,
            dt,
            dt_bias,
            z,
            a: a_log,
            d,
            conv_w,
            conv_b,
            norm,
            eps,
            pos,
            pos_at,
            fault,
            y,
            ring,
            state,
            scratch,
        } = args;
        let shape = check(what, shape.geom())?;
        // +∞ epsilon makes the norm's scale 0, a finite y and no fault.
        if norm.is_some() && !eps.is_finite() {
            return Err(GpuError::shape(what, "norm eps must be finite"));
        }
        let g = shape.geom();
        let (inner, heads, groups) = (g.d_inner, g.n_heads, g.n_groups);
        let conv_dim = shape.conv_dim();
        let head_dim = shape.head_dim();
        let state_len = inner * STATE;
        let ring_len = RING_ROWS * conv_dim;
        let blocks = heads * head_dim.div_ceil(CH_PER_BLOCK);
        let lens = [
            ("xbc", xbc.len(), conv_dim),
            ("dt", dt.len(), heads),
            ("dt_bias", dt_bias.len(), heads),
            ("z", z.len(), inner),
            ("a", a_log.len(), heads),
            ("d", d.len(), heads),
            ("conv_w", conv_w.len(), CONV_TAPS * conv_dim),
            ("conv_b", conv_b.len(), conv_dim),
            ("norm", norm.map_or(inner, |n| n.len()), inner),
            ("pos", pos.len(), pos_at + 1),
            ("y", y.len(), inner),
            ("ring", ring.len(), ring_len),
            ("state", state.len(), state_len),
            ("scratch.part", scratch.part.len(), blocks),
            ("scratch.done", scratch.done.len(), groups),
        ];
        if let Some((name, got, need)) = lens.iter().find(|(_, got, need)| got < need) {
            return Err(GpuError::shape(
                what,
                format!("{name}.len() {got} < {need}"),
            ));
        }
        let chunks = head_dim.div_ceil(CH_PER_BLOCK);
        let prep = self.module.prepare_mamba2_step(LaunchConfig1D::new(
            launch_u32(what, "blocks", blocks)?,
            BLOCK as u32,
            0,
        ))?;
        self.module.mamba2_step(
            stream,
            &prep,
            xbc,
            dt,
            dt_bias,
            z,
            a_log,
            d,
            conv_w,
            conv_b,
            // With no norm the kernel never reads its gain: `d` stands in.
            norm.unwrap_or(d),
            pos,
            launch_u32(what, "pos_at", pos_at)?,
            eps,
            launch_u32(what, "n_head", heads)?,
            launch_u32(what, "per", heads / groups)?,
            launch_u32(what, "n_group", groups)?,
            launch_u32(what, "head_dim", head_dim)?,
            launch_u32(what, "chunks", chunks)?,
            u32::from(norm.is_some()),
            fault,
            y,
            ring,
            state,
            &mut scratch.part,
            &mut scratch.done,
        )?;
        Ok(())
    }

    /// Enqueue a prompt pass of `u` tokens of one layer ([`StepsArgs`],
    /// module doc): `mamba2_steps`, `mamba2_ring_tail` and, with a gain,
    /// `mamba2_norm_rows`, in that order on `stream`. Every `y` row, the ring
    /// and the state after it are those of `u` [`Self::enqueue_step`] calls
    /// on the same rows at positions `p ..`. Refuses what the step refuses,
    /// and `u` = 0, a column window past `ld`, rows shorter than `u · ld` and
    /// a `y` shorter than `u` rows, by name, launching nothing. Two or three
    /// launches. Asynchronous, allocation-free, capturable.
    pub fn enqueue_steps(&self, stream: &CudaStream, args: StepsArgs<'_>) -> Result<(), GpuError> {
        let what = "mamba2::enqueue_steps";
        let StepsArgs {
            shape,
            u,
            rows,
            ld,
            z0,
            x0,
            dt0,
            dt_bias,
            a: a_log,
            d,
            conv_w,
            conv_b,
            norm,
            eps,
            pos,
            pos_at,
            fault,
            y,
            ring,
            state,
        } = args;
        let shape = check(what, shape.geom())?;
        if norm.is_some() && !eps.is_finite() {
            return Err(GpuError::shape(what, "norm eps must be finite"));
        }
        if u == 0 {
            return Err(GpuError::shape(
                what,
                "u 0 is refused: a pass holds a token",
            ));
        }
        let g = shape.geom();
        let (inner, heads, groups) = (g.d_inner, g.n_heads, g.n_groups);
        let conv_dim = shape.conv_dim();
        let head_dim = shape.head_dim();
        let windows = [("z", z0, inner), ("xbc", x0, conv_dim), ("dt", dt0, heads)];
        for (name, at, width) in windows {
            if at.saturating_add(width) > ld {
                return Err(GpuError::shape(
                    what,
                    format!("the {name} window {at} + {width} is past the row stride ld {ld}"),
                ));
            }
        }
        let lens = [
            ("rows", rows.len(), u.saturating_mul(ld)),
            ("dt_bias", dt_bias.len(), heads),
            ("a", a_log.len(), heads),
            ("d", d.len(), heads),
            ("conv_w", conv_w.len(), CONV_TAPS * conv_dim),
            ("conv_b", conv_b.len(), conv_dim),
            ("norm", norm.map_or(inner, |n| n.len()), inner),
            ("pos", pos.len(), pos_at + 1),
            ("y", y.len(), u.saturating_mul(inner)),
            ("ring", ring.len(), RING_ROWS * conv_dim),
            ("state", state.len(), inner * STATE),
        ];
        if let Some((name, got, need)) = lens.iter().find(|(_, got, need)| got < need) {
            return Err(GpuError::shape(
                what,
                format!("{name}.len() {got} < {need}"),
            ));
        }
        let chunks = head_dim.div_ceil(CH_PER_BLOCK);
        let blocks = heads * chunks;
        let tail_blocks = (u.min(RING_ROWS) * conv_dim).div_ceil(BLOCK);
        let (u32_u, u32_ld, u32_pos_at) = (
            launch_u32(what, "u", u)?,
            launch_u32(what, "ld", ld)?,
            launch_u32(what, "pos_at", pos_at)?,
        );
        let (u32_heads, u32_per, u32_groups, u32_hd, u32_chunks) = (
            launch_u32(what, "n_head", heads)?,
            launch_u32(what, "per", heads / groups)?,
            launch_u32(what, "n_group", groups)?,
            launch_u32(what, "head_dim", head_dim)?,
            launch_u32(what, "chunks", chunks)?,
        );
        let u32_x0 = launch_u32(what, "x0", x0)?;
        let steps = self.module.prepare_mamba2_steps(LaunchConfig1D::new(
            launch_u32(what, "blocks", blocks)?,
            BLOCK as u32,
            0,
        ))?;
        let tail = self.module.prepare_mamba2_ring_tail(LaunchConfig1D::new(
            launch_u32(what, "tail blocks", tail_blocks)?,
            BLOCK as u32,
            0,
        ))?;
        let rows_blocks = launch_u32(what, "norm blocks", u.saturating_mul(groups))?;
        let normed = match norm {
            Some(_) => Some(self.module.prepare_mamba2_norm_rows(LaunchConfig1D::new(
                rows_blocks,
                BLOCK as u32,
                0,
            ))?),
            None => None,
        };
        self.module.mamba2_steps(
            stream,
            &steps,
            rows,
            dt_bias,
            a_log,
            d,
            conv_w,
            conv_b,
            &*ring,
            pos,
            u32_pos_at,
            u32_u,
            u32_ld,
            launch_u32(what, "z0", z0)?,
            u32_x0,
            launch_u32(what, "dt0", dt0)?,
            u32_heads,
            u32_per,
            u32_groups,
            u32_hd,
            u32_chunks,
            fault,
            &mut *y,
            state,
        )?;
        self.module.mamba2_ring_tail(
            stream,
            &tail,
            rows,
            pos,
            u32_pos_at,
            u32_u,
            u32_ld,
            u32_x0,
            launch_u32(what, "conv_dim", conv_dim)?,
            ring,
        )?;
        if let (Some(gain), Some(prep)) = (norm, normed) {
            self.module.mamba2_norm_rows(
                stream, &prep, gain, eps, u32_u, u32_heads, u32_per, u32_groups, u32_hd,
                u32_chunks, fault, y,
            )?;
        }
        Ok(())
    }
}
