//! Expert routing: each expert scores √softplus(logit); a per-expert bias is
//! added for the selection only; the top 6 of 384 experts are kept, and their
//! unbiased scores are renormalized to sum 1 and scaled by 1.5.
//!
//! One launch per token, the decode shape: the router's logit gemv with the
//! rest of the chain fused into it. The gemv is `q8f32::f32_gemv`'s m = 1 row
//! body — one warp per expert row, `f32_lane_partial_1col` and the fixed
//! butterfly — so the logits are bit for bit that kernel's. The warp that owns
//! a row also scores it ([`sqrt_softplus`]). The selection needs every
//! expert's score, so it runs in the block that finishes last: each block
//! publishes its rows (a fence, then one atomic ticket per block), and the
//! block that draws the last ticket reads all scores back, adds the selection
//! bias, picks the top six, writes the weights, and puts the ticket count back
//! to zero for the next launch or graph replay.
//!
//! Numeric contract — ik's CPU rule for this architecture, node for node
//! (`SQRT_SOFTPLUS`, `ADD`, `ARGSORT`, `GET_ROWS`, `SUM_ROWS`, `DIV`,
//! `SCALE`), except where named:
//! - score `sqrt(x > 20 ? x : ln(1 + e^x))`. The device's `expf`/`logf` are
//!   CUDA's, a few ulp from the host libm's, and the compiler fuses
//!   `1 + expf(x)` into `expf`'s last step, so that sum rounds once;
//! - selection value `score + bias`; top six by (value descending, index
//!   descending) — ik sorts `(value, index)` pairs with `std::greater`, so an
//!   equal value goes to the LARGER expert id;
//! - weights: the six unbiased scores in slot order summed in f64, the sum
//!   narrowed to f32 and guarded ([`renorm_divisor`]: ik divides by the bare
//!   sum, which is 0 when every selected score underflowed, and the guard
//!   leaves every other sum's bits alone), each score divided by it in f32,
//!   then multiplied by `expert_weights_scale` in f32.

use bloomery_gpu::q8f32::f32_lane_partial_1col;
/// ik's expert score in f32; the gates' host side simulates the device with it.
pub use bloomery_gpu::route_core::sqrt_softplus;
use bloomery_gpu::route_core::{renorm_divisor, take};
use bloomery_gpu::{DeviceTensor, GpuError, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32};
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread, threadfence, warp,
};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Experts the router scores (this model's `expert_count`): the width of the
/// selecting block's shared arrays, so a compile-time shape.
pub const N_EXPERT: usize = 384;

/// Experts each token routes to (`expert_used_count`).
pub const N_USED: usize = 6;

/// Threads per router block, eight warps, one expert row per warp.
const ROUTER_THREADS: usize = 256;
const ROUTER_THREADS_U32: u32 = ROUTER_THREADS as u32;
const ROWS_PER_BLOCK: usize = ROUTER_THREADS / 32;

/// Scores each lane of the selecting warp owns: experts `lane + 32 j`.
const PER_LANE: usize = N_EXPERT / 32;

const _: () = assert!(ROUTER_THREADS_U32 as usize == ROUTER_THREADS);
const _: () = assert!(N_EXPERT.is_multiple_of(32) && N_EXPERT.is_multiple_of(ROWS_PER_BLOCK));
// The selecting lane keeps its taken entries as bits of one u32.
const _: () = assert!(PER_LANE <= 32);

#[cuda_module]
mod router_kernels {
    use super::*;

    /// The router for one token: `n_expert` (= [`N_EXPERT`]) rows of `k`
    /// f32 in `w`, the token's `k` activations in `x`, the selection bias in
    /// `bias`. Writes `logits[e]` and `probs[e]` (the score) for every
    /// expert, and `ids[s]`/`weights[s]` for the [`N_USED`] slots in rank
    /// order. `done[0]` is the block ticket count: zero before the launch,
    /// zero again after it.
    ///
    /// Grid `n_expert / 8` blocks of [`ROUTER_THREADS`]; warp `r % 8` of
    /// block `r / 8` owns row `r`. After its rows every thread fences and the
    /// block meets at a barrier, so the rows are visible device-wide before
    /// thread 0 draws the block's ticket. The block that draws the last
    /// ticket copies every score into shared memory (coherent loads — its L1
    /// never held those lines, and a volatile load does not ask it), adds the
    /// bias, and its warp 0 runs six rounds of a butterfly argmax under
    /// [`take`]`::<true>` (ties to the larger id): lane `L` scans experts
    /// `L + 32 j` that no earlier round took, the butterfly merges the lanes,
    /// and the winning lane marks its entry taken. Lane 0 writes the ids,
    /// then the weights from the six scores, and puts `done[0]` back to zero.
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
            n_expert == 384,
            w.len() >= n_expert * k,
            x.len() >= k,
            bias.len() >= n_expert,
            logits.len() >= n_expert,
            probs.len() >= n_expert,
            ids.len() >= 6,
            weights.len() >= 6,
            done.len() >= 1
        )
    )]
    pub fn ds41_router(
        w: &[f32],
        x: &[f32],
        bias: &[f32],
        n_expert: u32,
        k: u32,
        scale: f32,
        mut logits: DisjointSlice<f32>,
        mut probs: DisjointSlice<f32>,
        mut ids: DisjointSlice<u32>,
        mut weights: DisjointSlice<f32>,
        mut done: DisjointSlice<u32>,
    ) {
        static mut SCORE: SharedArray<f32, N_EXPERT> = SharedArray::UNINIT;
        static mut SEL_V: SharedArray<f32, N_EXPERT> = SharedArray::UNINIT;
        static mut LAST: SharedArray<u32, 1> = SharedArray::UNINIT;
        static mut SLOT_P: SharedArray<f32, N_USED> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let lane = warp::lane_id() as usize;
        let row = thread::blockIdx_x() as usize * ROWS_PER_BLOCK + tid / 32;
        if row < n_expert as usize {
            // `f32_gemv`'s m = 1 row: the same body and the same tree.
            // SAFETY: row < n_expert, so w.len() >= n_expert * k >= (row + 1) *
            // k, and x.len() >= k, by the launch contract; the launcher
            // passes k a positive multiple of 32; lane < 32.
            let partial = unsafe { f32_lane_partial_1col(w, x, k, row, lane) };
            let logit = warp::reduce_sum_f32(partial);
            if lane == 0 {
                // SAFETY: row < n_expert <= logits.len() by the launch
                // contract; lane 0 of the row's warp is the slot's only writer.
                unsafe { *logits.get_unchecked_mut(row) = logit };
                let score = sqrt_softplus(logit);
                // SAFETY: row < n_expert <= probs.len() by the launch
                // contract; lane 0 of the row's warp is the slot's only writer.
                unsafe { *probs.get_unchecked_mut(row) = score };
            }
        }
        threadfence();
        thread::sync_threads();

        // SAFETY: block-shared, one element; the raw form reaches the
        // `static mut` without a reference.
        let last = unsafe { SharedArray::as_raw_mut_ptr(&raw mut LAST) };
        let ticket = done.as_mut_ptr();
        if tid == 0 {
            // SAFETY: `ticket` is done[0], inside `done` by the launch
            // contract; every access to it is atomic.
            let count = unsafe { DeviceAtomicU32::from_ptr(ticket) };
            let is_last = count.fetch_add(1, AtomicOrdering::AcqRel) + 1 == thread::gridDim_x();
            // SAFETY: block-shared, one element, written before the barrier
            // that publishes it.
            unsafe { *last = u32::from(is_last) };
        }
        thread::sync_threads();
        // SAFETY: block-shared, one element, written by thread 0 before the
        // barrier above.
        if unsafe { *last } == 0 {
            return;
        }

        // The last block: every block's rows are published.
        // SAFETY: block-shared, N_EXPERT entries; the raw form reaches the
        // `static mut` without a reference.
        let score = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SCORE) };
        // SAFETY: block-shared, N_EXPERT entries; the raw form reaches the
        // `static mut` without a reference.
        let sel_v = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SEL_V) };
        // SAFETY: block-shared, N_USED entries; the raw form reaches the
        // `static mut` without a reference.
        let slot_p = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SLOT_P) };
        let probs_ptr = probs.as_mut_ptr();
        let mut e = tid;
        while e < N_EXPERT {
            // SAFETY: e < N_EXPERT = n_expert <= probs.len() by the launch
            // contract; the volatile load reads the row where its block left it.
            let p = unsafe { core::ptr::read_volatile(probs_ptr.add(e)) };
            // SAFETY: e < N_EXPERT = n_expert <= bias.len() by the launch
            // contract.
            let b = unsafe { *bias.get_unchecked(e) };
            // SAFETY: block-shared, e < N_EXPERT; thread `tid` writes entries
            // tid + 256·i only, before the barrier that publishes them.
            unsafe { *score.add(e) = p };
            // SAFETY: block-shared, e < N_EXPERT; thread `tid` writes entries
            // tid + 256·i only, before the barrier that publishes them.
            unsafe { *sel_v.add(e) = p + b };
            e += ROUTER_THREADS;
        }
        thread::sync_threads();

        if tid < 32 {
            let mut taken = 0u32;
            let mut s = 0usize;
            while s < N_USED {
                let mut bv = f32::NEG_INFINITY;
                let mut bi = 0u32;
                let mut j = 0usize;
                while j < PER_LANE {
                    if (taken >> j) & 1 == 0 {
                        let ej = lane + 32 * j;
                        // SAFETY: block-shared, ej < 32·PER_LANE = N_EXPERT,
                        // written before the barrier above.
                        let v = unsafe { *sel_v.add(ej) };
                        if take::<true>(v, ej as u32, bv, bi) {
                            bv = v;
                            bi = ej as u32;
                        }
                    }
                    j += 1;
                }
                let mut off = 16u32;
                while off > 0 {
                    let (ov, oi) = (warp::shuffle_xor_f32(bv, off), warp::shuffle_xor(bi, off));
                    if take::<true>(ov, oi, bv, bi) {
                        bv = ov;
                        bi = oi;
                    }
                    off >>= 1;
                }
                // Every lane leaves the butterfly with the same winner; its
                // owner marks it taken.
                if bi as usize % 32 == lane {
                    taken |= 1 << (bi as usize / 32);
                }
                if lane == 0 {
                    // SAFETY: block-shared, bi < N_EXPERT (every candidate is
                    // one of a lane's own entries), written before the barrier.
                    let p = unsafe { *score.add(bi as usize) };
                    // SAFETY: block-shared, s < N_USED; lane 0 alone writes and
                    // reads these entries.
                    unsafe { *slot_p.add(s) = p };
                    // SAFETY: s < N_USED <= ids.len() by the launch contract;
                    // lane 0 is the only writer.
                    unsafe { *ids.get_unchecked_mut(s) = bi };
                }
                s += 1;
            }
            if lane == 0 {
                let mut sum = 0.0f64;
                let mut s = 0usize;
                while s < N_USED {
                    // SAFETY: block-shared, s < N_USED, this lane's own write.
                    sum += f64::from(unsafe { *slot_p.add(s) });
                    s += 1;
                }
                let sum = renorm_divisor(sum as f32);
                let mut s = 0usize;
                while s < N_USED {
                    // SAFETY: block-shared, s < N_USED, this lane's own write.
                    let p = unsafe { *slot_p.add(s) };
                    // SAFETY: s < N_USED <= weights.len() by the launch
                    // contract; lane 0 is the only writer.
                    unsafe { *weights.get_unchecked_mut(s) = p / sum * scale };
                    s += 1;
                }
                // SAFETY: `ticket` is done[0], inside `done` by the launch
                // contract; every block has drawn its ticket by now.
                let count = unsafe { DeviceAtomicU32::from_ptr(ticket) };
                count.store(0, AtomicOrdering::Relaxed);
            }
        }
    }
}

/// Where one router launch leaves its results, allocated once and reused by
/// every launch (and by a captured graph's replays): per expert the logit and
/// the score, per slot the expert id and the weight, and the block ticket
/// count the launch returns to zero. The count serves one launch at a time,
/// so launches that share a `RouterOut` must be ordered on one stream.
pub struct RouterOut {
    pub logits: DeviceBuffer<f32>,
    pub probs: DeviceBuffer<f32>,
    pub ids: DeviceBuffer<u32>,
    pub weights: DeviceBuffer<f32>,
    done: DeviceBuffer<u32>,
}

impl RouterOut {
    /// Allocate the buffers, the ticket count zeroed. Load-time only.
    pub fn new(stream: &CudaStream) -> Result<RouterOut, GpuError> {
        Ok(RouterOut {
            logits: DeviceBuffer::zeroed(stream, N_EXPERT)?,
            probs: DeviceBuffer::zeroed(stream, N_EXPERT)?,
            ids: DeviceBuffer::zeroed(stream, N_USED)?,
            weights: DeviceBuffer::zeroed(stream, N_USED)?,
            done: DeviceBuffer::zeroed(stream, 1)?,
        })
    }

    /// The ticket count as it stands: zero between launches.
    pub fn tickets(&self, stream: &CudaStream) -> Result<u32, GpuError> {
        Ok(self.done.to_host_vec(stream)?[0])
    }
}

/// The loaded router module. Owns no context and no stream — every enqueue
/// takes the engine stream, so launches order with the rest of the step and
/// are capturable.
pub struct RouterKernels {
    module: router_kernels::LoadedModule,
}

impl RouterKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<RouterKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; the launcher checks its launch contract.
        let module = unsafe { router_kernels::load(ctx)? };
        Ok(RouterKernels { module })
    }

    /// Enqueue the router for one token: `w` the router weight as f32
    /// ([`N_EXPERT`] rows of `k`, `k` a positive multiple of 32), `x` the
    /// token's `k` activations, `bias` the [`N_EXPERT`] selection biases,
    /// `scale` the file's `expert_weights_scale`; results into `out`.
    /// Asynchronous, allocation-free, capturable.
    pub fn enqueue_router(
        &self,
        stream: &CudaStream,
        w: &DeviceTensor<f32>,
        x: &DeviceBuffer<f32>,
        bias: &DeviceBuffer<f32>,
        scale: f32,
        out: &mut RouterOut,
    ) -> Result<(), GpuError> {
        let what = "enqueue_router";
        let shape = |detail: String| GpuError::Shape { what, detail };
        let k = w.cols();
        if w.rows() != N_EXPERT || k == 0 || !k.is_multiple_of(32) {
            return Err(shape(format!(
                "router weight is {} x {k}, want {N_EXPERT} rows of a positive multiple of 32",
                w.rows()
            )));
        }
        if x.len() < k || bias.len() < N_EXPERT {
            return Err(shape(format!(
                "x.len() {} < k {k} or bias.len() {} < {N_EXPERT}",
                x.len(),
                bias.len()
            )));
        }
        let k = launch_u32(what, "k", k)?;
        let n_expert = launch_u32(what, "n_expert", N_EXPERT)?;
        let grid = launch_u32(what, "grid", N_EXPERT / ROWS_PER_BLOCK)?;
        let prep =
            self.module
                .prepare_ds41_router(LaunchConfig1D::new(grid, ROUTER_THREADS_U32, 0))?;
        self.module.ds41_router(
            stream,
            &prep,
            w.buf(),
            x,
            bias,
            n_expert,
            k,
            scale,
            &mut out.logits,
            &mut out.probs,
            &mut out.ids,
            &mut out.weights,
            &mut out.done,
        )?;
        Ok(())
    }
}
