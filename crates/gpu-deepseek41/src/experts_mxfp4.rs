//! The DSpark draft's router: 128 experts, 3 used. Its MXFP4 experts are
//! `bloomery_gpu::mxfp4_sel`'s, the common crate's, which the draft calls
//! with this router's ids and weights as the route plan.
//!
//! `dflash_router`: `ds41_router`'s body with this model's constants — the
//! `f32_gemv` m = 1 row per expert in its accumulation order (the lane walk
//! `f32_lane_partial_1col_w32`), [`sqrt_softplus`], the selection bias,
//! three rounds of the butterfly argmax under `take::<true>` (an equal
//! value goes to the larger id), the weights from the unbiased scores. One
//! token per launch; token `tok` reads `x[tok·k ..]` and writes its logits,
//! scores, ids and weights at `tok`'s rows of a [`DraftRouterOut`], so `m`
//! launches fill one plan.

use bloomery_gpu::q8f32::f32_lane_partial_1col_w32;
/// ik's expert score in f32; the gate's host side simulates the device with it.
pub use bloomery_gpu::route_core::sqrt_softplus;
use bloomery_gpu::route_core::{renorm_divisor, take};
use bloomery_gpu::{DeviceTensor, FaultSink, FaultSite, GpuError, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32};
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread, threadfence, warp,
};
use cuda_host::cuda_module;
use model::arch::models::shape::{self, RouterBody, RouterInst, rules};
use std::sync::Arc;

/// Experts the draft's router scores: the width of the selecting block's
/// shared arrays.
pub const N_EXPERT: usize = 128;

/// Experts each token routes to: compiled into the router, the expert
/// kernels' route layout and the draft's buffers.
pub use bloomery_gpu::mxfp4_sel::N_USED;

/// The router's instance row: the width and the one pick count above, the
/// rule √softplus with a selection bias, `norm` a launch argument. A draft's
/// shape selects it through `models::shape::select_router`.
pub const ROUTER_ROW: RouterInst = shape::router_row(RouterBody::Dflash, PER_LANE as u32);
const _: () = assert!(shape::router_row_is(
    ROUTER_ROW,
    rules::BIASED_SQRT_SOFTPLUS,
    true,
    (N_USED as u32, N_USED as u32)
));

/// Threads per router block, four warps, one expert row per warp.
const ROUTER_THREADS: usize = 128;
const ROUTER_THREADS_U32: u32 = ROUTER_THREADS as u32;
const ROWS_PER_BLOCK: usize = ROUTER_THREADS / 32;

/// Scores each lane of the selecting warp owns: experts `lane + 32 j`.
const PER_LANE: usize = N_EXPERT / 32;

const _: () = assert!(ROUTER_THREADS_U32 as usize == ROUTER_THREADS);
// The router entry's `launch_bounds` and `launch_contract` block are literals.
const _: () = assert!(ROUTER_THREADS == 128);
const _: () = assert!(N_EXPERT.is_multiple_of(32) && N_EXPERT.is_multiple_of(ROWS_PER_BLOCK));
// The selecting lane keeps its taken entries as bits of one u32.
const _: () = assert!(PER_LANE <= 32);

#[cuda_module]
mod dflash_kernels {
    use super::*;

    /// The draft's router for token `tok`: `n_expert` (=
    /// [`N_EXPERT`]) rows of `k` f32 in `w`, the token's `k` activations at
    /// `x[tok·k ..]`, the selection bias in `bias`. Writes `logits` and
    /// `probs` (the score) at `tok · N_EXPERT + e` for every expert, and
    /// `ids`/`weights` at `tok · N_USED + s` for the slots in rank order;
    /// the weights are `score / sum · scale` with `norm != 0`, the sum
    /// guarded by [`renorm_divisor`], and `score · scale` without.
    /// `done[0]` is the block ticket count: zero before the
    /// launch, zero again after it. The body is `ds41_router`'s (its doc
    /// states the ticket protocol and the selection).
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
            n_expert == 128,
            w.len() >= n_expert * k,
            x.len() >= tok * k + k,
            bias.len() >= n_expert,
            logits.len() >= tok * n_expert + n_expert,
            probs.len() >= tok * n_expert + n_expert,
            ids.len() >= 3 * tok + 3,
            weights.len() >= 3 * tok + 3,
            done.len() >= 1
        )
    )]
    pub fn dflash_router(
        w: &[f32],
        x: &[f32],
        bias: &[f32],
        n_expert: u32,
        k: u32,
        tok: u32,
        scale: f32,
        norm: u32,
        mut logits: DisjointSlice<f32>,
        mut probs: DisjointSlice<f32>,
        mut ids: DisjointSlice<u32>,
        mut weights: DisjointSlice<f32>,
        mut done: DisjointSlice<u32>,
        fault: FaultSink,
    ) {
        static mut SCORE: SharedArray<f32, N_EXPERT> = SharedArray::UNINIT;
        static mut SEL_V: SharedArray<f32, N_EXPERT> = SharedArray::UNINIT;
        static mut LAST: SharedArray<u32, 1> = SharedArray::UNINIT;
        static mut SLOT_P: SharedArray<f32, N_USED> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let lane = warp::lane_id() as usize;
        let row = thread::blockIdx_x() as usize * ROWS_PER_BLOCK + tid / 32;
        let e0 = tok as usize * N_EXPERT;
        let s0 = tok as usize * N_USED;
        if row < n_expert as usize {
            let x0 = tok as usize * k as usize;
            // SAFETY: x.len() >= tok·k + k = x0 + k by the launch contract.
            let xt = unsafe { x.get_unchecked(x0..x0 + k as usize) };
            // `f32_gemv`'s m = 1 row: the same sum order and the same tree.
            // SAFETY: row < n_expert, so w.len() >= n_expert * k >= (row + 1) *
            // k, and xt.len() = k, by the launch contract; the launcher
            // passes k a positive multiple of 32; lane < 32.
            let partial = unsafe { f32_lane_partial_1col_w32(w, xt, k, row, lane) };
            let logit = warp::reduce_sum_f32(partial);
            if lane == 0 {
                // SAFETY: e0 + row < tok·n_expert + n_expert <= logits.len()
                // by the launch contract; lane 0 of the row's warp is the only writer.
                unsafe { *logits.get_unchecked_mut(e0 + row) = logit };
                let score = sqrt_softplus(logit);
                // SAFETY: e0 + row < tok·n_expert + n_expert <= probs.len()
                // by the launch contract; lane 0 of the row's warp is the only writer.
                unsafe { *probs.get_unchecked_mut(e0 + row) = score };
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
            // SAFETY: e0 + e < tok·N_EXPERT + N_EXPERT <= probs.len() by the
            // launch contract; the volatile load reads the row where its block left it.
            let p = unsafe { core::ptr::read_volatile(probs_ptr.add(e0 + e)) };
            // SAFETY: e < N_EXPERT = n_expert <= bias.len() by the launch
            // contract.
            let b = unsafe { *bias.get_unchecked(e) };
            // SAFETY: block-shared, e < N_EXPERT; thread `tid` writes entries
            // tid + ROUTER_THREADS·i only, before the barrier that publishes
            // them.
            unsafe { *score.add(e) = p };
            // SAFETY: block-shared, e < N_EXPERT; thread `tid` writes entries
            // tid + ROUTER_THREADS·i only, before the barrier that publishes
            // them.
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
                    // The winner of every round is warp-uniform; a winner
                    // that is not finite means fewer finite candidates than
                    // slots (`crate::router`'s rule), and lane 0 raises.
                    if !bv.is_finite() {
                        fault.raise(FaultSite::Router);
                    }
                    // SAFETY: block-shared, bi < N_EXPERT (every candidate is
                    // one of a lane's own entries), written before the barrier.
                    let p = unsafe { *score.add(bi as usize) };
                    // SAFETY: block-shared, s < N_USED; lane 0 alone writes and
                    // reads these entries.
                    unsafe { *slot_p.add(s) = p };
                    // SAFETY: s0 + s < 3·tok + 3 <= ids.len() by the launch
                    // contract; lane 0 is the only writer.
                    unsafe { *ids.get_unchecked_mut(s0 + s) = bi };
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
                    let v = if norm != 0 { p / sum } else { p };
                    let wt = v * scale;
                    if !wt.is_finite() {
                        fault.raise(FaultSite::Router);
                    }
                    // SAFETY: s0 + s < 3·tok + 3 <= weights.len() by the launch
                    // contract; lane 0 is the only writer.
                    unsafe { *weights.get_unchecked_mut(s0 + s) = wt };
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

/// Where router launches leave their results for up to `n_tok` tokens,
/// allocated once: per token the logits and scores of every expert (`tok ·
/// N_EXPERT + e`), the ids and weights of its slots (`tok · N_USED + s`),
/// and the block ticket count each launch returns to zero. Launches that
/// share one are ordered on one stream.
pub struct DraftRouterOut {
    pub logits: DeviceBuffer<f32>,
    pub probs: DeviceBuffer<f32>,
    pub ids: DeviceBuffer<u32>,
    pub weights: DeviceBuffer<f32>,
    done: DeviceBuffer<u32>,
    n_tok: usize,
}

impl DraftRouterOut {
    /// Allocate for `n_tok` tokens, the ticket count zeroed. Load-time only.
    pub fn new(stream: &CudaStream, n_tok: usize) -> Result<DraftRouterOut, GpuError> {
        if n_tok == 0 {
            return Err(GpuError::Shape {
                what: "DraftRouterOut::new",
                detail: "zero tokens".into(),
            });
        }
        Ok(DraftRouterOut {
            logits: DeviceBuffer::zeroed(stream, n_tok * N_EXPERT)?,
            probs: DeviceBuffer::zeroed(stream, n_tok * N_EXPERT)?,
            ids: DeviceBuffer::zeroed(stream, n_tok * N_USED)?,
            weights: DeviceBuffer::zeroed(stream, n_tok * N_USED)?,
            done: DeviceBuffer::zeroed(stream, 1)?,
            n_tok,
        })
    }

    /// The ticket count as it stands: zero between launches.
    pub fn tickets(&self, stream: &CudaStream) -> Result<u32, GpuError> {
        Ok(self.done.to_host_vec(stream)?[0])
    }
}

/// One router launch ([`DraftRouterKernels::enqueue_router`]).
pub struct RouterArgs<'a> {
    /// `ffn_gate_inp` as f32: [`N_EXPERT`] rows of `k`, `k` a positive
    /// multiple of 32.
    pub w: &'a DeviceTensor<f32>,
    /// At least `(tok + 1) · k` activations; the token's are `x[tok · k ..]`.
    pub x: &'a DeviceBuffer<f32>,
    /// `exp_probs_b`: [`N_EXPERT`] selection biases.
    pub bias: &'a DeviceBuffer<f32>,
    /// `expert_weights_scale`.
    pub scale: f32,
    /// `expert_weights_norm`.
    pub norm: bool,
    /// The token's row in the output.
    pub tok: usize,
    /// Where a selection the router cannot make is raised.
    pub fault: FaultSink,
}
/// The loaded draft router module. Owns no context and no stream — every
/// enqueue takes the engine stream, so launches order with the rest of the
/// step and are capturable.
pub struct DraftRouterKernels {
    module: dflash_kernels::LoadedModule,
}

impl DraftRouterKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<DraftRouterKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; every launcher checks its launch contract.
        let module = unsafe { bloomery_gpu::shared_module!(dflash_kernels, ctx)? };
        Ok(DraftRouterKernels { module })
    }

    /// Enqueue the router for token `a.tok`; results into `out` at that
    /// token's rows. Asynchronous, allocation-free, capturable.
    pub fn enqueue_router(
        &self,
        stream: &CudaStream,
        a: &RouterArgs<'_>,
        out: &mut DraftRouterOut,
    ) -> Result<(), GpuError> {
        let what = "DraftRouterKernels::enqueue_router";
        let shape = |detail: String| GpuError::Shape { what, detail };
        let k = a.w.cols();
        if a.w.rows() != N_EXPERT || k == 0 || !k.is_multiple_of(32) {
            return Err(shape(format!(
                "router weight is {} x {k}, want {N_EXPERT} rows of a positive multiple of 32",
                a.w.rows()
            )));
        }
        if a.tok >= out.n_tok || a.x.len() < (a.tok + 1) * k || a.bias.len() < N_EXPERT {
            return Err(shape(format!(
                "tok {} of {} tokens: x.len() {} (need {}), bias.len() {} (need {N_EXPERT})",
                a.tok,
                out.n_tok,
                a.x.len(),
                (a.tok + 1) * k,
                a.bias.len()
            )));
        }
        let grid = launch_u32(what, "grid", N_EXPERT / ROWS_PER_BLOCK)?;
        let prep =
            self.module
                .prepare_dflash_router(LaunchConfig1D::new(grid, ROUTER_THREADS_U32, 0))?;
        self.module.dflash_router(
            stream,
            &prep,
            a.w.buf(),
            a.x,
            a.bias,
            launch_u32(what, "n_expert", N_EXPERT)?,
            launch_u32(what, "k", k)?,
            launch_u32(what, "tok", a.tok)?,
            a.scale,
            u32::from(a.norm),
            &mut out.logits,
            &mut out.probs,
            &mut out.ids,
            &mut out.weights,
            &mut out.done,
            a.fault,
        )?;
        Ok(())
    }
}
