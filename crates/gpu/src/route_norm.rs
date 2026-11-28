//! The feed-forward norm and the router of one decode token in one launch: `rms_norm` of the row,
//! its q8_1 form for the card leg's gate·up (`q3k_quantize_q8_1`), and the biased sigmoid router
//! at 128 experts and the top 8 over the normed row (`sigmoid_128_8_router`) — each output bit for
//! bit what the three launches write.
//!
//! The router's grid is [`BLOCKS`] blocks, one warp an expert row in a block's first
//! [`ROWS_PER_BLOCK`] warps, and every block reads the whole activation row. Each block therefore
//! norms the row itself, as `norm_proj` and `norm_quant` do: thread `tid` sums the squares of
//! values `tid, tid + 256, …` (`rms_partial_sq`'s order), the warp butterfly, the fixed eight-warp
//! tree, `rms_scale_checked`, then each value `(scale · gain) · x` into shared memory — the
//! expression and order `rms_norm` stores. No grid barrier stands between the norm and the row
//! walk: every block's scale is the same sum over the same values in the same order, so the
//! normed row, and with it every logit, is `rms_norm`'s bits whichever block holds it.
//!
//! After the barrier the block's work divides by warp:
//!
//! - warps 0 to 3 walk the block's four expert rows over the shared normed row
//!   (`f32_lane_partial_1col_w32`, the router's own lane walk), write the logit and its score, and
//!   then, as in `router_body`, every block draws a ticket and the block that draws the last one
//!   publishes the scores and runs the selection (`publish_n`, `select_n`) over warp 0;
//! - warps 4 to 7 take the quantizer's 128-value blocks (`q8_1_quant_vals`, the body of
//!   `q3k_quantize_q8_1`) from the shared row, block `blockIdx + BLOCKS · (warp − 4)`, a block
//!   wholly one warp's, so the stored bytes do not depend on which block or warp computes them;
//! - block `b` also stores values `b · 256 ..` of the normed row to `y`, the f32 consumers' copy.
//!
//! A row whose sum of squares is not finite and positive raises the norm's fault from block 0 and
//! every value is NaN (the scale is NaN, as `rms_norm` leaves it); a 128-value block holding a
//! non-finite value has no q8_1 form and raises [`FaultSite::QuantColumn`], the standalone
//! quantizer's own site, and the router's non-finite scores raise as `router_body`'s do, so the
//! fault word is the three launches' union.

use crate::GpuError;
use crate::elem::{RMS_THREADS, RMS_WARPS, rms_partial_sq, rms_scale_checked, rms_warp_tree};
use crate::fault::{FaultSink, FaultSite};
use crate::launch_u32;
use crate::q8_1_quant_vals;
use crate::q8f32::f32_lane_partial_1col_w32;
use crate::route_body::{ROUTER_THREADS, ROWS_PER_BLOCK, publish_n, select_n, sigmoid_128_8};
use crate::route_core::{Sigmoid, router_score};
use crate::tensor::{DeviceTensor, Q8Act};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32};
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread, threadfence, warp,
};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Threads a block: `rms_norm`'s block, eight warps.
pub const THREADS: usize = RMS_THREADS;
/// Experts the entry routes and the slots it picks: [`sigmoid_128_8`]'s.
const E: usize = sigmoid_128_8::N_EXPERT;
const U: usize = sigmoid_128_8::N_USED;
/// Blocks a launch: [`ROWS_PER_BLOCK`] expert rows each.
pub const BLOCKS: usize = E / ROWS_PER_BLOCK;
/// The widest row the shared normed row holds.
pub const MAX_K: usize = 3072;
/// Quantizer warps a block: the warps past the router's row warps.
const QUANT_WARPS: usize = THREADS / 32 - ROWS_PER_BLOCK;
// The entry's `launch_bounds` and `launch_contract` spell the shapes out as literals.
const _: () = assert!(
    THREADS == 256 && RMS_WARPS == 8 && E == 128 && U == 8 && BLOCKS == 32 && ROWS_PER_BLOCK == 4
);
const _: () = assert!(ROUTER_THREADS == 128 && QUANT_WARPS == 4 && MAX_K == 3072);

#[cuda_module]
mod route_norm_kernels {
    use super::*;

    /// The norm, the q8_1 form and the router (module doc) of one token: `x` the row of `k` f32,
    /// `gain` the norm's gain, `w` the router's [`E`] rows of `k` f32, `bias` the selection bias.
    /// Writes the normed row to `y`, its q8_1 planes to `q3`, `q4`, `q6`, `s8` and `d8` (a
    /// one-column `Q8Act` of `n_sb = k / 256` super-blocks, `half_it` and `quad_it` its group
    /// counts), the logit and the score of every expert to `logits` and `probs`, and the picks to
    /// `ids` and `weights` in rank order. `done[0]` is the block ticket count: zero before the
    /// launch, zero again after it. Grid [`BLOCKS`] blocks of [`THREADS`].
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
            k >= 256,
            k <= 3072,
            n_expert == 128,
            w.len() >= n_expert * k,
            x.len() >= k,
            gain.len() >= k,
            bias.len() >= n_expert,
            q3.len() >= 64 * half_it,
            q4.len() >= 256 * quad_it,
            q6.len() >= 128 * half_it,
            s8.len() >= 8 * n_sb,
            d8.len() >= 2 * n_sb,
            y.len() >= k,
            logits.len() >= n_expert,
            probs.len() >= n_expert,
            ids.len() >= 8,
            weights.len() >= 8,
            done.len() >= 1
        )
    )]
    pub fn sigmoid_128_8_router_norm(
        w: &[f32],
        x: &[f32],
        gain: &[f32],
        bias: &[f32],
        eps: f32,
        k: u32,
        n_expert: u32,
        scale: f32,
        n_sb: u32,
        half_it: u32,
        quad_it: u32,
        mut q3: DisjointSlice<u64>,
        mut q4: DisjointSlice<u32>,
        mut q6: DisjointSlice<u32>,
        mut s8: DisjointSlice<i32>,
        mut d8: DisjointSlice<f32>,
        mut y: DisjointSlice<f32>,
        mut logits: DisjointSlice<f32>,
        mut probs: DisjointSlice<f32>,
        mut ids: DisjointSlice<u32>,
        mut weights: DisjointSlice<f32>,
        mut done: DisjointSlice<u32>,
        fault: FaultSink,
    ) {
        static mut XN: SharedArray<f32, MAX_K, 16> = SharedArray::UNINIT;
        static mut WSUM: SharedArray<f32, RMS_WARPS> = SharedArray::UNINIT;
        static mut SCORE: SharedArray<f32, E> = SharedArray::UNINIT;
        static mut SEL_V: SharedArray<f32, E> = SharedArray::UNINIT;
        static mut LAST: SharedArray<u32, 1> = SharedArray::UNINIT;
        static mut SLOT_P: SharedArray<f32, U> = SharedArray::UNINIT;

        let blk = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        let lane = warp::lane_id() as usize;
        let wi = tid / 32;
        let k_us = k as usize;
        // SAFETY: XN and WSUM are this block's own shared arrays of MAX_K and RMS_WARPS entries;
        // the raw form is the only way to reach a `static mut` without a reference. Every access
        // is below its length (k <= MAX_K by the contract, wi < RMS_WARPS) and ordered by
        // `sync_threads`.
        let (xn, ws) = unsafe {
            (
                SharedArray::as_raw_mut_ptr(&raw mut XN),
                SharedArray::as_raw_mut_ptr(&raw mut WSUM),
            )
        };

        // The norm, as `norm_proj` takes it: the sum of squares, then each value into the shared
        // row.
        // SAFETY: x.len() >= k by the contract, base 0, tid < RMS_THREADS (the block's width).
        let part = warp::reduce_sum_f32(unsafe { rms_partial_sq(x, 0, k_us, tid) });
        if lane == 0 {
            // SAFETY: wi < RMS_WARPS; one lane a warp writes its slot.
            unsafe { *ws.add(wi) = part };
        }
        thread::sync_threads();
        // SAFETY: every slot was written before the barrier above.
        let sums = unsafe {
            [
                *ws.add(0),
                *ws.add(1),
                *ws.add(2),
                *ws.add(3),
                *ws.add(4),
                *ws.add(5),
                *ws.add(6),
                *ws.add(7),
            ]
        };
        // Block 0 raises: every block holds the same scale, and the fault word takes one site.
        let norm = rms_scale_checked(rms_warp_tree(sums), k, eps, fault, tid == 0 && blk == 0);
        let mut j = tid;
        while j < k_us {
            // SAFETY: j < k <= x.len(), gain.len() and MAX_K by the contract; this thread alone
            // writes value j.
            unsafe { *xn.add(j) = (norm * *gain.get_unchecked(j)) * *x.get_unchecked(j) };
            j += THREADS;
        }
        thread::sync_threads();

        // The f32 row for the consumers beside the router: block `blk` stores values
        // `blk · THREADS + i · BLOCKS · THREADS`.
        let mut j = blk * THREADS + tid;
        while j < k_us {
            // SAFETY: j < k <= y.len() by the contract and the shared row holds k values past
            // the barrier above; (blk, tid) alone writes value j.
            unsafe { *y.get_unchecked_mut(j) = *xn.add(j) };
            j += BLOCKS * THREADS;
        }

        if wi >= ROWS_PER_BLOCK {
            // The quantizer: a 128-value block is one warp's, lane `lane` its values
            // `4 · lane ..`, as `q3k_quantize_q8_1` takes it.
            let blocks = k_us / 128;
            let n_sb = n_sb as usize;
            let mut b = blk + BLOCKS * (wi - ROWS_PER_BLOCK);
            while b < blocks {
                let at = 128 * b + 4 * lane;
                // SAFETY: at + 3 < 128 · blocks <= k (k a multiple of 256 by the launcher), and
                // the shared row holds k values past the barrier above.
                let v = unsafe {
                    [
                        *xn.add(at),
                        *xn.add(at + 1),
                        *xn.add(at + 2),
                        *xn.add(at + 3),
                    ]
                };
                // SAFETY: column 0, b < 2 · n_sb (n_sb = k / 256, host-checked), the planes' sizes
                // are the contract's (half_it, quad_it the launcher's ceil(n_sb / 2), ceil(n_sb /
                // 4)), the whole warp enters with the same b, `v` holds values 128 · b + 4 · lane
                // ..+3, and no other warp of the grid enters with that b.
                let refused = unsafe {
                    q8_1_quant_vals(
                        v, 0, b, n_sb, half_it, quad_it, lane, &mut q3, &mut q4, &mut q6, &mut s8,
                        &mut d8,
                    )
                };
                if refused && lane == 0 {
                    fault.raise(FaultSite::QuantColumn);
                }
                b += BLOCKS * QUANT_WARPS;
            }
        } else {
            // The router's rows, one a warp over the shared normed row (`row` is warp-uniform).
            let row = blk * ROWS_PER_BLOCK + wi;
            if row < n_expert as usize {
                // SAFETY: the shared row holds k values past the barrier above and nothing writes
                // it from here on.
                let xs: &[f32] = unsafe { core::slice::from_raw_parts(xn.cast_const(), k_us) };
                // SAFETY: row < n_expert, so w holds the row's k values by the contract; xs holds
                // k values, k a multiple of 32 (host-checked) and lane < 32.
                let partial = unsafe { f32_lane_partial_1col_w32(w, xs, k, row, lane) };
                let logit = warp::reduce_sum_f32(partial);
                if lane == 0 {
                    // SAFETY: row < n_expert <= logits.len() and probs.len() by the contract; lane
                    // 0 of the row's warp is each slot's only writer.
                    unsafe {
                        *logits.get_unchecked_mut(row) = logit;
                        *probs.get_unchecked_mut(row) = router_score::<Sigmoid>(logit);
                    }
                }
            }
        }
        threadfence();
        thread::sync_threads();

        // The block that draws the last ticket selects, as `router_body`.
        let ticket = done.as_mut_ptr();
        // SAFETY: LAST is this block's own shared array of one entry.
        let last = unsafe { SharedArray::as_raw_mut_ptr(&raw mut LAST) };
        if tid == 0 {
            // SAFETY: `ticket` is done[0], inside `done` by the contract; every access to it is
            // atomic.
            let count = unsafe { DeviceAtomicU32::from_ptr(ticket) };
            let is_last = count.fetch_add(1, AtomicOrdering::AcqRel) + 1 == thread::gridDim_x();
            // SAFETY: block-shared, one element, written before the barrier that publishes it.
            unsafe { *last = u32::from(is_last) };
        }
        thread::sync_threads();
        // SAFETY: block-shared, one element, written by thread 0 before the barrier above.
        if unsafe { *last } == 0 {
            return; // block-uniform
        }
        // SAFETY: SCORE, SEL_V and SLOT_P are this block's own shared arrays of E, E and U
        // entries.
        let (score, sel_v, slot_p) = unsafe {
            (
                SharedArray::as_raw_mut_ptr(&raw mut SCORE),
                SharedArray::as_raw_mut_ptr(&raw mut SEL_V),
                SharedArray::as_raw_mut_ptr(&raw mut SLOT_P),
            )
        };
        if tid < ROUTER_THREADS {
            // SAFETY: probs holds E scores from 0 and bias E entries by the contract; score and
            // sel_v are the block's shared arrays of E; the barrier below publishes them. The
            // publish strides by ROUTER_THREADS, so only the router's first 128 threads call it.
            unsafe { publish_n::<E>(probs.as_mut_ptr(), Some(bias), 0, score, sel_v, tid, fault) };
        }
        thread::sync_threads();
        if tid < 32 {
            // SAFETY: the barrier above published score and sel_v; ids and weights hold U
            // entries from 0 by the contract; warp 0 enters converged.
            unsafe {
                select_n::<E, U, true, false>(
                    score,
                    sel_v,
                    slot_p,
                    lane,
                    scale,
                    fault,
                    &mut ids,
                    &mut weights,
                    0,
                )
            };
            if lane == 0 {
                // SAFETY: `ticket` is done[0], inside `done` by the contract; every block has
                // drawn its ticket by now.
                let count = unsafe { DeviceAtomicU32::from_ptr(ticket) };
                count.store(0, AtomicOrdering::Relaxed);
            }
        }
    }
}

/// The loaded module. Owns no stream: each enqueue takes the engine stream.
pub struct RouteNormKernels {
    module: route_norm_kernels::LoadedModule,
}

/// What one launch reads.
pub struct RouteNormArgs<'a> {
    /// The router's [`E`] rows of `k` f32.
    pub w: &'a DeviceTensor<f32>,
    /// The row to norm, `k` values.
    pub x: &'a DeviceBuffer<f32>,
    /// The norm's gain, `k` values.
    pub gain: &'a DeviceBuffer<f32>,
    /// The selection bias, [`E`] values.
    pub bias: &'a DeviceBuffer<f32>,
    pub eps: f32,
    /// The file's `expert_weights_scale`.
    pub scale: f32,
    pub fault: FaultSink,
}

/// What one launch writes: the normed row (f32 and q8_1), the router's results and the block
/// ticket, zero between launches.
pub struct RouteNormOut<'a> {
    pub y: &'a mut DeviceBuffer<f32>,
    pub act: &'a mut Q8Act,
    pub route: &'a mut sigmoid_128_8::RouterOut,
    pub ticket: &'a mut DeviceBuffer<u32>,
}

impl RouteNormKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<RouteNormKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for the module above; the
        // launcher checks the launch contract.
        let module = unsafe { crate::shared_module!(route_norm_kernels, ctx)? };
        Ok(RouteNormKernels { module })
    }

    /// Enqueue the norm, the q8_1 form and the router of one token (module doc). Refused by name,
    /// nothing enqueued: a weight that is not [`E`] rows of a multiple of 256 up to [`MAX_K`]
    /// values, an activation plane of another width or more than one column, a short buffer, a
    /// non-finite scale. Asynchronous, allocation-free, capturable.
    pub fn enqueue(
        &self,
        stream: &CudaStream,
        a: RouteNormArgs<'_>,
        o: RouteNormOut<'_>,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "route_norm::enqueue";
        let RouteNormArgs {
            w,
            x,
            gain,
            bias,
            eps,
            scale,
            fault,
        } = a;
        let RouteNormOut {
            y,
            act,
            route,
            ticket,
        } = o;
        let k = w.cols();
        if w.rows() != E || k == 0 || !k.is_multiple_of(256) || k > MAX_K {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "router weight is {} x {k}, want {E} rows of a multiple of 256 up to {MAX_K}",
                    w.rows()
                ),
            ));
        }
        if act.m() != 1 || act.k() != k {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "activation planes are {} columns of {}, want one of {k}",
                    act.m(),
                    act.k()
                ),
            ));
        }
        if !scale.is_finite() || !eps.is_finite() {
            return Err(GpuError::shape(
                WHAT,
                format!("scale {scale} and eps {eps} must be finite"),
            ));
        }
        crate::short(
            WHAT,
            &[
                ("x", x.len(), k),
                ("gain", gain.len(), k),
                ("y", y.len(), k),
                ("bias", bias.len(), E),
                ("logits", route.logits.len(), E),
                ("probs", route.probs.len(), E),
                ("ids", route.ids.len(), U),
                ("weights", route.weights.len(), U),
                ("ticket", ticket.len(), 1),
            ],
        )?;
        let n_sb = act.n_sb();
        let k_u = launch_u32(WHAT, "k", k)?;
        let n_expert = launch_u32(WHAT, "n_expert", E)?;
        let n_sb = launch_u32(WHAT, "n_sb", n_sb)?;
        let prep = self
            .module
            .prepare_sigmoid_128_8_router_norm(LaunchConfig1D::new(
                launch_u32(WHAT, "grid", BLOCKS)?,
                launch_u32(WHAT, "block", THREADS)?,
                0,
            ))?;
        self.module.sigmoid_128_8_router_norm(
            stream,
            &prep,
            w.buf(),
            x,
            gain,
            bias,
            eps,
            k_u,
            n_expert,
            scale,
            n_sb,
            n_sb.div_ceil(2),
            n_sb.div_ceil(4),
            &mut act.q3,
            &mut act.q4,
            &mut act.q6,
            &mut act.s8,
            &mut act.d8,
            y,
            &mut route.logits,
            &mut route.probs,
            &mut route.ids,
            &mut route.weights,
            ticket,
            fault,
        )?;
        Ok(())
    }
}

/// A [`Q8Act`]'s five planes on the host as stored, `d8` as bits: what a gate compares between two
/// launches that fill one.
#[derive(Debug, PartialEq, Eq)]
pub struct ActPlanes {
    pub q3: Vec<u64>,
    pub q4: Vec<u32>,
    pub q6: Vec<u32>,
    pub s8: Vec<i32>,
    pub d8: Vec<u32>,
}

/// Read `act`'s planes back after `stream`'s queued work. Not capturable.
pub fn act_planes(stream: &CudaStream, act: &Q8Act) -> Result<ActPlanes, GpuError> {
    stream.synchronize()?;
    Ok(ActPlanes {
        q3: act.q3.to_host_vec(stream)?,
        q4: act.q4.to_host_vec(stream)?,
        q6: act.q6.to_host_vec(stream)?,
        s8: act.s8.to_host_vec(stream)?,
        d8: act
            .d8
            .to_host_vec(stream)?
            .into_iter()
            .map(f32::to_bits)
            .collect(),
    })
}
