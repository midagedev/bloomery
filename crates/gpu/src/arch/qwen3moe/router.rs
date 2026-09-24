//! The qwen3moe router: softmax over the 128 experts' logits, the top 8 by
//! probability, and their weights renormalized to sum to one
//! (`norm_topk_prob`). One token per launch — the decode shape.
//!
//! The body is `crate::router::router_topk`'s m = 1 path with this model's
//! constants; on this cuda-oxide pin a function boundary around the
//! selection loop changes the kernel's PTX, so it is copied rather than
//! shared, and only the leaf tie rule comes from `route_core`.
//!
//! Numeric contract, op for op:
//! - the max of the 128 logits by a serial ascending `f32::max` fold;
//! - `exp(logit − max)` in f32, the sum of the exps accumulated in f64 in
//!   ascending expert order, each probability the f32 divide by `sum as f32`;
//! - the top 8 by descending probability with ties toward the smaller
//!   expert id;
//! - the chosen probabilities summed in f64 in slot order, the sum rounded
//!   once to f32, each weight the f32 divide of its probability by it.

use crate::GpuError;
use crate::route_core::take;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread, warp,
};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Experts the router softmaxes over (`expert_count`).
pub const N_EXPERT: usize = 128;

/// Experts each token routes to (`expert_used_count`).
pub const N_USED: usize = 8;

/// Threads per launch: one warp, lane 0 running the serial passes.
const ROUTER_THREADS: u32 = 32;

/// Probabilities each lane of the selecting warp owns: experts `lane + 32 j`.
const PER_LANE: usize = N_EXPERT / 32;
// The selecting lane keeps its taken entries as bits of one u32.
const _: () = assert!(N_EXPERT.is_multiple_of(32) && PER_LANE <= 32);

#[cuda_module]
mod qwen3moe_router_kernels {
    use super::*;

    /// The router for one token: its [`N_EXPERT`] logits in `x` → every
    /// probability in `probs`, the ids in rank order in `ids` and their
    /// renormalized weights in `weights`. One block of 32 threads: lane 0
    /// runs the three serial passes into shared memory (and `probs`), then
    /// the warp selects: lane `L` owns experts `L + 32 j`, eight rounds of a
    /// butterfly argmax under [`take`]`::<false>` (the order the serial
    /// scan's strict `>` over ascending ids realizes, seed `(−inf, 0)`), the
    /// winning lane marking its entry taken. Lane 0 writes the ids, then the
    /// weights.
    #[kernel]
    #[launch_bounds(32)]
    #[launch_contract(
        domain = 1,
        block = (32, 1, 1),
        requires = (
            x.len() >= 128,
            probs.len() >= 128,
            ids.len() >= 8,
            weights.len() >= 8
        )
    )]
    pub fn qwen3moe_router(
        x: &[f32],
        mut probs: DisjointSlice<f32>,
        mut ids: DisjointSlice<u32>,
        mut weights: DisjointSlice<f32>,
    ) {
        static mut P: SharedArray<f32, N_EXPERT> = SharedArray::UNINIT;
        static mut SLOT_P: SharedArray<f32, N_USED> = SharedArray::UNINIT;

        let t = thread::index_1d().get();
        // SAFETY: P is this block's own shared allocation; the raw form is
        // the only way to reach it without a reference to a `static mut`.
        // Every index below is an expert id < N_EXPERT, and lane 0's writes
        // precede every other lane's reads by `sync_threads`.
        let p = unsafe { SharedArray::as_raw_mut_ptr(&raw mut P) };
        // SAFETY: block-shared, N_USED entries; lane 0 alone writes and
        // reads them.
        let slot_p = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SLOT_P) };
        if t == 0 {
            let mut mx = f32::NEG_INFINITY;
            let mut e = 0usize;
            while e < N_EXPERT {
                // SAFETY: e < 128 <= x.len() by the launch contract.
                let le = unsafe { *x.get_unchecked(e) };
                mx = mx.max(le);
                e += 1;
            }
            let mut sum = 0.0f64;
            let mut e = 0usize;
            while e < N_EXPERT {
                // SAFETY: e < 128 <= x.len(); p.add(e) is inside P.
                let ex = unsafe {
                    let ex = (*x.get_unchecked(e) - mx).exp();
                    *p.add(e) = ex;
                    ex
                };
                sum += f64::from(ex);
                e += 1;
            }
            let inv = sum as f32;
            let mut e = 0usize;
            while e < N_EXPERT {
                // SAFETY: e < 128 — inside P, and inside probs (launch
                // contract).
                unsafe {
                    let v = *p.add(e) / inv;
                    *p.add(e) = v;
                    *probs.get_unchecked_mut(e) = v;
                }
                e += 1;
            }
        }
        thread::sync_threads();

        let lane = warp::lane_id() as usize;
        let mut taken = 0u32;
        let mut s = 0usize;
        while s < N_USED {
            let mut bv = f32::NEG_INFINITY;
            let mut bi = 0u32;
            let mut j = 0usize;
            while j < PER_LANE {
                if (taken >> j) & 1 == 0 {
                    let ej = lane + 32 * j;
                    // SAFETY: ej < 32·PER_LANE = N_EXPERT, inside P, written
                    // above and visible past the barrier.
                    let v = unsafe { *p.add(ej) };
                    if take::<false>(v, ej as u32, bv, bi) {
                        bv = v;
                        bi = ej as u32;
                    }
                }
                j += 1;
            }
            let mut off = 16u32;
            while off > 0 {
                let (ov, oi) = (warp::shuffle_xor_f32(bv, off), warp::shuffle_xor(bi, off));
                if take::<false>(ov, oi, bv, bi) {
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
            if t == 0 {
                // SAFETY: bi < N_EXPERT — inside P; s < N_USED, inside
                // SLOT_P and ids (launch contract). The probability is read
                // from the winner's slot rather than from the comparison, so
                // one no comparison won is carried through unchanged.
                unsafe {
                    *slot_p.add(s) = *p.add(bi as usize);
                    *ids.get_unchecked_mut(s) = bi;
                }
            }
            s += 1;
        }
        if t == 0 {
            let mut sum = 0.0f64;
            let mut s = 0usize;
            while s < N_USED {
                // SAFETY: s < N_USED, this lane's own write.
                sum += f64::from(unsafe { *slot_p.add(s) });
                s += 1;
            }
            let sum = sum as f32;
            let mut s = 0usize;
            while s < N_USED {
                // SAFETY: s < N_USED, inside SLOT_P and weights (launch
                // contract); lane 0 is the only writer.
                unsafe { *weights.get_unchecked_mut(s) = *slot_p.add(s) / sum };
                s += 1;
            }
        }
    }
}

/// Where one router launch leaves its results, allocated once and reused by
/// every launch and replay.
pub struct RouterOut {
    pub probs: DeviceBuffer<f32>,
    pub ids: DeviceBuffer<u32>,
    pub weights: DeviceBuffer<f32>,
}

impl RouterOut {
    /// Allocate the buffers. Load-time only.
    pub fn new(stream: &CudaStream) -> Result<RouterOut, GpuError> {
        Ok(RouterOut {
            probs: DeviceBuffer::zeroed(stream, N_EXPERT)?,
            ids: DeviceBuffer::zeroed(stream, N_USED)?,
            weights: DeviceBuffer::zeroed(stream, N_USED)?,
        })
    }
}

/// The loaded router module. Owns no stream: each enqueue takes the engine
/// stream, so its launch orders with the rest of the step and capture.
pub struct RouterKernels {
    module: qwen3moe_router_kernels::LoadedModule,
}

impl RouterKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<RouterKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; the launcher checks its launch contract.
        let module = unsafe { qwen3moe_router_kernels::load(ctx)? };
        Ok(RouterKernels { module })
    }

    /// Enqueue the router of one token over its `x` ([`N_EXPERT`] logits)
    /// into `out`. The decode shape only: a batch of tokens is a launch per
    /// token. Asynchronous, allocation-free, capturable.
    pub fn enqueue(
        &self,
        stream: &CudaStream,
        x: &DeviceBuffer<f32>,
        out: &mut RouterOut,
    ) -> Result<(), GpuError> {
        let what = "qwen3moe::router::enqueue";
        if x.len() < N_EXPERT
            || out.probs.len() < N_EXPERT
            || out.ids.len() < N_USED
            || out.weights.len() < N_USED
        {
            return Err(GpuError::shape(
                what,
                format!(
                    "x.len() {} / probs.len() {} need {N_EXPERT}, ids.len() {} / weights.len() \
                     {} need {N_USED}",
                    x.len(),
                    out.probs.len(),
                    out.ids.len(),
                    out.weights.len()
                ),
            ));
        }
        let prep =
            self.module
                .prepare_qwen3moe_router(LaunchConfig1D::new(1, ROUTER_THREADS, 0))?;
        self.module.qwen3moe_router(
            stream,
            &prep,
            x,
            &mut out.probs,
            &mut out.ids,
            &mut out.weights,
        )?;
        Ok(())
    }
}
