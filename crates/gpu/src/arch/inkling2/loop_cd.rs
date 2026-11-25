//! LoopCD-Hidden at the loop→coda boundary (arXiv 2610.02185, the hidden-state form;
//! `BLOOMERY_INK2_LOOPCD`): the stream past the looped core's last pass extrapolated from the
//! core's output after an earlier pass, `h' = h + ω (h − h_b)` — `h` the stream as the last pass
//! leaves it, `h_b` the stream as pass `b` left it — over `m` rows of `n` values, in place on `h`.
//!
//! One entry, `loop_cd_apply`: one thread a value, `ceil(m · n / THREADS)` blocks. The body
//! (`super`) copies `h_b` out at the end of the reference pass and launches the entry once, before
//! the first visit past the core reads the stream; a step, a verify's rows and a prompt pass launch
//! the same entry over their rows.
//!
//! Numeric contract: the difference `h − h_b` rounds on its own, and the product with ω and the
//! sum with `h` round together, one fused multiply-add — `fma_rn_f32(ω, h − h_b, h)`, two
//! roundings. This order is the gate. A value that is not finite makes its output not finite, the
//! rule's own result (nothing is clamped); the entry raises no fault of its own, as the elementwise
//! add does not.

use crate::GpuError;
use crate::launch_u32;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::float::{add_rn_f32, fma_rn_f32};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Threads a block takes; the entry's `launch_bounds` spells the same count.
pub(super) const THREADS: usize = 256;
const _: () = assert!(THREADS == 256);

#[cuda_module]
mod loop_cd_kernels {
    use super::*;

    /// `y[i] = ω · (y[i] − h[i]) + y[i]` over the first `n` values, one thread a value (the module
    /// doc's contract). `n` is a runtime argument; the lengths are host-checked.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (h.len() >= n, y.len() >= n, n >= 1)
    )]
    pub fn loop_cd_apply(h: &[f32], n: u32, omega: f32, mut y: DisjointSlice<f32>) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        // SAFETY: i < n <= h.len() and y.len() by the launch contract; thread i alone reads h[i]
        // and writes y[i]. The negation is exact, so the sum is the rounded difference.
        unsafe {
            let yv = *y.get_unchecked_mut(i);
            let d = add_rn_f32(yv, -*h.get_unchecked(i));
            *y.get_unchecked_mut(i) = fma_rn_f32(omega, d, yv);
        }
    }
}

/// The loaded module. Owns no stream: the launch takes the engine stream.
pub(in crate::arch::inkling2) struct LoopCdKernels {
    module: loop_cd_kernels::LoadedModule,
}

/// What the boundary step reads and writes (the module doc).
pub(in crate::arch::inkling2) struct LoopCdArgs<'a> {
    /// The stream saved at the end of the reference pass: `m` rows of `n`.
    pub h_b: &'a DeviceBuffer<f32>,
    /// ω: finite, not negative.
    pub omega: f32,
    /// Values a row.
    pub n: usize,
    /// Rows.
    pub m: usize,
}

impl LoopCdKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub(in crate::arch::inkling2) fn load(
        ctx: &Arc<CudaContext>,
    ) -> Result<LoopCdKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for the module above; the
        // launcher checks the launch contract.
        let module = unsafe { crate::shared_module!(loop_cd_kernels, ctx)? };
        Ok(LoopCdKernels { module })
    }

    /// Enqueue the boundary step over `a` into `y`, in place (the module doc): one launch of
    /// `ceil(m · n / THREADS)` blocks. Refused by name, nothing enqueued: no row or no width, an ω
    /// that is negative or not finite, a `h_b` or a `y` shorter than the rows. Asynchronous,
    /// allocation-free, capturable.
    pub(in crate::arch::inkling2) fn enqueue(
        &self,
        stream: &CudaStream,
        a: LoopCdArgs<'_>,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "loop_cd::enqueue";
        let bad = |detail: String| Err(GpuError::shape(WHAT, detail));
        if a.m == 0 || a.n == 0 {
            return bad(format!("{} rows of {} values", a.m, a.n));
        }
        if !a.omega.is_finite() || a.omega < 0.0 {
            return bad(format!("omega {}: a finite one, not negative", a.omega));
        }
        let Some(len) = a.m.checked_mul(a.n) else {
            return bad(format!("{} rows of {} values", a.m, a.n));
        };
        crate::short(WHAT, &[("h_b", a.h_b.len(), len), ("y", y.len(), len)])?;
        let n = launch_u32(WHAT, "values", len)?;
        let prep = self.module.prepare_loop_cd_apply(LaunchConfig1D::new(
            n.div_ceil(THREADS as u32),
            THREADS as u32,
            0,
        ))?;
        self.module
            .loop_cd_apply(stream, &prep, a.h_b, n, a.omega, y)?;
        Ok(())
    }
}
