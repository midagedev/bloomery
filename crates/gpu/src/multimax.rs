//! MultiMax: a head's learned piecewise term on its logits, in place between the
//! lm_head's projection and the argmax —
//! `l + t₀·relu(r₀ − l) + t₁·relu(l − r₁) + t₂·relu(r₂ − l)² + t₃·relu(l − r₃)²`,
//! four range edges `r` and four weights `t` of the file (llama.cpp's
//! `build_multimax`; ERNIE 5.5's `output_multimax_ranges` and `output_multimax_ts`).
//! Every product and sum is rounded on its own, as the graph's MUL and ADD nodes
//! round them — never a fused multiply-add — and `relu` is the reference's: a value
//! above zero or not a number stays, the rest is `0.0`. A head enqueues it through
//! `Head::enqueue_with_epilogue`, so the argmax and the sampling read the MultiMax
//! logits.
//!
//! No silent failure: a logit that is not finite raises [`FaultSite::Logit`] from the
//! thread that holds it and keeps a non-finite output (the finite argmax behind an
//! f32-activation head refuses it again). A masked logit of −inf does not stay −inf
//! here — the term `t₀·relu(r₀ − (−inf))` is infinite — so a mask is applied after
//! the epilogue, never before it.
//!
//! One entry, [`multimax_kernels::multimax_rows`]: a thread a logit, over `n` values.

use crate::fault::{FaultSink, FaultSite};
use crate::{GpuError, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::float::{add_rn_f32, mul_rn_f32};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;
use std::sync::Arc;

/// The values a head's MultiMax holds: four range edges and four weights.
pub const MULTIMAX: usize = 4;
/// Threads of a block.
const THREADS: usize = 256;
const THREADS_U32: u32 = THREADS as u32;
const _: () = assert!(THREADS_U32 as usize == THREADS);

// ------------------------------------------------------------------ cores

/// The reference's `relu`: `z` where it is above zero or not a number, else `0.0`
/// (a negative zero becomes positive).
#[inline(always)]
fn relu(z: f32) -> f32 {
    if z.is_nan() | (z > 0.0) { z } else { 0.0 }
}

/// One logit's MultiMax, every product and sum rounded on its own and added in the
/// graph's order: the below-`r₀` and above-`r₁` terms, then the two squares. The
/// subtractions have no product to contract, so the plain operator is the one
/// rounded op; every product and sum that follows a product is spelled `_rn`.
#[inline(always)]
fn multimax_elem(l: f32, r: &[f32; MULTIMAX], t: &[f32; MULTIMAX]) -> f32 {
    let a = relu(r[0] - l);
    let b = relu(l - r[1]);
    let c = relu(r[2] - l);
    let d = relu(l - r[3]);
    let y = add_rn_f32(l, mul_rn_f32(t[0], a));
    let y = add_rn_f32(y, mul_rn_f32(t[1], b));
    let y = add_rn_f32(y, mul_rn_f32(t[2], mul_rn_f32(c, c)));
    add_rn_f32(y, mul_rn_f32(t[3], mul_rn_f32(d, d)))
}

// ---------------------------------------------------------------- kernels

#[cuda_module]
mod multimax_kernels {
    use super::*;

    /// `y[i] = multimax(y[i])` over the first `n` values, in place: `r` the four
    /// range edges, `t` the four weights. A logit that is not finite raises
    /// [`FaultSite::Logit`] on `fault` and keeps a non-finite output.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (y.len() >= n, r.len() >= 4, t.len() >= 4)
    )]
    pub fn multimax_rows(
        mut y: DisjointSlice<f32>,
        r: &[f32],
        t: &[f32],
        n: u32,
        fault: FaultSink,
    ) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        // SAFETY: i < n <= y.len() by the launch contract; thread i owns logit i,
        // the only access it makes before the store below.
        let l = unsafe { *y.get_unchecked_mut(i) };
        if !l.is_finite() {
            fault.raise(FaultSite::Logit);
        }
        // SAFETY: r.len() >= 4 and t.len() >= 4 by the launch contract; the four
        // values of each are this kernel's to read.
        let (r, t) = unsafe {
            (
                [
                    *r.get_unchecked(0),
                    *r.get_unchecked(1),
                    *r.get_unchecked(2),
                    *r.get_unchecked(3),
                ],
                [
                    *t.get_unchecked(0),
                    *t.get_unchecked(1),
                    *t.get_unchecked(2),
                    *t.get_unchecked(3),
                ],
            )
        };
        // SAFETY: i < n <= y.len(); one thread a logit.
        unsafe {
            *y.get_unchecked_mut(i) = multimax_elem(l, &r, &t);
        }
    }
}

// ------------------------------------------------------------------ host

/// The inputs of one `multimax_rows` launch: the first `n` of `y`'s logits in place,
/// the four range edges and the four weights, and where a non-finite logit is raised.
pub struct MultiMaxArgs<'a> {
    /// The logits, [`MULTIMAX`] terms applied in place.
    pub y: &'a mut DeviceBuffer<f32>,
    /// How many of `y`'s values are logits.
    pub n: usize,
    /// The four range edges.
    pub ranges: &'a DeviceBuffer<f32>,
    /// The four weights.
    pub ts: &'a DeviceBuffer<f32>,
    /// The launch's fault sink.
    pub fault: FaultSink,
}

/// The loaded MultiMax module. Owns no stream.
pub struct MultiMaxKernels {
    module: multimax_kernels::LoadedModule,
}

impl MultiMaxKernels {
    /// Load this module's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<MultiMaxKernels, GpuError> {
        // SAFETY: this crate owns the embedded device bundle produced for the
        // module above; the launcher checks its launch contract.
        let module = unsafe { crate::shared_module!(multimax_kernels, ctx)? };
        Ok(MultiMaxKernels { module })
    }

    /// Enqueue `multimax_rows` (the module doc): `a.y[..a.n]`'s MultiMax in place by
    /// `a.ranges` and `a.ts`. Asynchronous, allocation-free, capturable.
    pub fn enqueue(&self, stream: &CudaStream, a: MultiMaxArgs<'_>) -> Result<(), GpuError> {
        const WHAT: &str = "MultiMaxKernels::enqueue";
        let MultiMaxArgs {
            y,
            n,
            ranges,
            ts,
            fault,
        } = a;
        if n == 0 || y.len() < n || ranges.len() < MULTIMAX || ts.len() < MULTIMAX {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!(
                    "n = {n} (at least 1), y.len() {}, ranges.len() {} / ts.len() {} (need {MULTIMAX})",
                    y.len(),
                    ranges.len(),
                    ts.len()
                ),
            });
        }
        let grid = launch_u32(WHAT, "grid", n.div_ceil(THREADS))?;
        let n = launch_u32(WHAT, "n", n)?;
        let prep = self
            .module
            .prepare_multimax_rows(LaunchConfig1D::new(grid, THREADS_U32, 0))?;
        self.module
            .multimax_rows(stream, &prep, y, ranges, ts, n, fault)?;
        Ok(())
    }
}
