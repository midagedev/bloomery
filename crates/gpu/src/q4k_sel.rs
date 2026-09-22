//! Q4_K gemv over expert slots selected on the device: the MoE down
//! projection of a Q4_K expert stack in the decode shape, all `n_slots`
//! routed experts in one launch. The ids are read from a device buffer, so
//! the launch lives inside a captured graph whose replay consumes whatever
//! the router last wrote there. Down is the per-slot-column shape: slot `s`
//! dots the rows of expert `sel[s]` with activation column `s` (that
//! expert's own swiglu output), as `q5_0_gemv_sel` does for Q5_0 — where
//! `q3k_gemv_sel` (gate/up) shares one column across slots.
//!
//! The kernel owns no arithmetic: its per-row body is `q4k_gemv`'s m = 1
//! path verbatim (`cores::q4k_row_dot_1col`, the warp reduction, the lane-0
//! store), with `col0 = slot`. So slot `s` is bit-identical to `q4k_gemv`
//! run on an upload of expert `sel[s]`'s rows alone against a one-column
//! activation quantized from input column `s`; `gate_q4k_sel` pins that.
//! Q4_K rows are `36 * n_sb` u32 words, whole words for any `n_sb`, so an
//! odd super-block count needs no load-time repack (unlike Q3_K): the core's
//! guarded tail runs the partial last four-super-block iteration.

use crate::GpuError;
use crate::cores::q4k_row_dot_1col;
use crate::launch_u32;
use crate::tensor::{DeviceTensor, Q8Act};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread, warp};
use cuda_host::cuda_module;
use std::sync::Arc;

#[cuda_module]
mod q4k_sel_kernels {
    use super::*;

    /// Q4_K gemv over expert slots selected on the device (the MoE down
    /// projection, decode shape): one launch computes `n_slots` experts of
    /// the resident flat stack — `n_experts * rows_per_expert` rows of
    /// `36 * n_sb` words — where slot s reads activation column `s` of the
    /// q8_1 scratch (each expert's down input differs) and writes
    /// `y[s*rows_per_expert + r]`. Thread geometry as `q4k_gemv` (one warp
    /// per output row, eight rows per 256-thread block); thread row
    /// `n = slot * rows_per_expert + r` stores `y[n]` and reads weight row
    /// `sel[slot] * rows_per_expert + r`. The per-row body is
    /// `cores::q4k_row_dot_1col` with col0 = slot — the m = 1 path of
    /// `q4k_gemv`, same loads, same accumulation order, same reduction.
    ///
    /// An id >= n_experts cannot be rejected by the host contract (it
    /// lives in device memory): the slot's warps return before their first
    /// weight load — warp-uniform, no divergent branch — leaving that slot
    /// of `y` untouched and every other slot unaffected.
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
            w.len() >= n_experts * rows_per_expert * 36 * n_sb,
            q.len() >= n_slots * 256 * iters,
            s8.len() >= n_slots * 8 * n_sb,
            d8.len() >= n_slots * 2 * n_sb,
            sel.len() >= n_slots,
            y.len() >= n_slots * rows_per_expert
        )
    )]
    pub fn q4k_gemv_sel(
        w: &[u32],
        q: &[u32],
        s8: &[i32],
        d8: &[f32],
        sel: &[u32],
        n_experts: u32,
        rows_per_expert: u32,
        n_slots: u32,
        n_sb: u32,
        iters: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let t = thread::index_1d().get() % 256;
        let row = (thread::index_1d().get() / 256) * 8 + t / 32;
        if row >= n_slots as usize * rows_per_expert as usize {
            return;
        }
        let slot = row / rows_per_expert as usize;
        // SAFETY: slot < n_slots <= sel.len() by the launch contract; the
        // load is warp-uniform (all 32 lanes of the warp share `row`, hence
        // `slot`), so the out-of-range return below never diverges a warp.
        let id = unsafe { *sel.get_unchecked(slot) } as usize;
        if id >= n_experts as usize {
            return;
        }
        let row_abs = id * rows_per_expert as usize + row % rows_per_expert as usize;
        let lane = warp::lane_id() as usize;
        // The core's caller contract, from the launch contract: row_abs <
        // n_experts * rows_per_expert rows of `w`, column slot < n_slots of
        // `q`/`s8`/`d8`, iters = ceil(n_sb/4) from the host, and all 32
        // lanes of the warp are here (both returns above are warp-uniform).
        let f0 = q4k_row_dot_1col(w, q, s8, d8, n_sb as usize, iters, row_abs, slot, lane);
        let s0 = warp::reduce_sum_f32(f0);
        if lane == 0 {
            // SAFETY: row < n_slots*rows_per_expert <= y.len() by the launch
            // contract; only lane 0 of the warp writes y[row].
            unsafe {
                *y.get_unchecked_mut(row) = s0;
            }
        }
    }
}

/// The loaded Q4_K expert-select module and its enqueue API. Owns no
/// context and no stream — every enqueue takes the engine stream
/// (`Gpu::stream()`), so launches order with the rest of the step and are
/// capturable.
pub struct Q4kSelKernels {
    module: q4k_sel_kernels::LoadedModule,
}

impl Q4kSelKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<Q4kSelKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; every launcher checks its launch contract.
        let module = unsafe { q4k_sel_kernels::load(ctx)? };
        Ok(Q4kSelKernels { module })
    }

    /// Enqueue the device-indirect MoE down projection for a Q4_K expert
    /// stack: one launch computes `n_slots` experts, slot s writing
    /// `y[s*rows_per_expert .. (s+1)*rows_per_expert]` as
    /// `w[sel[s]*rows_per_expert .. +rows_per_expert] · act` column `s`.
    /// `act` holds exactly `n_slots` quantized columns, one per slot — a
    /// shared-input (m = 1) `act` is a misuse here. `w` is the full resident
    /// stack in `Gpu::enqueue_gemv_q4k`'s row format (`36 * n_sb` u32 words
    /// per row), `w.rows()` a positive multiple of `rows_per_expert`
    /// (`n_experts = w.rows() / rows_per_expert`). `sel` is a device buffer
    /// of at least `n_slots` ids read by the kernel per launch, so a
    /// captured graph replay picks up new ids written between replays; an
    /// id >= n_experts leaves that slot of `y` untouched. Asynchronous,
    /// allocation-free, capturable.
    #[allow(
        clippy::too_many_arguments,
        reason = "host launcher; folding these into a *Args struct is the R8 round"
    )]
    pub fn enqueue_gemv_q4k_sel(
        &self,
        stream: &CudaStream,
        w: &DeviceTensor<u32>,
        act: &Q8Act,
        sel: &DeviceBuffer<u32>,
        n_slots: usize,
        rows_per_expert: usize,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "enqueue_gemv_q4k_sel";
        let n_sb = act.n_sb();
        if act.m() != n_slots {
            return Err(GpuError::shape(
                what,
                format!(
                    "one activation column per slot: act.m() = {} must equal \
                 n_slots = {n_slots}",
                    act.m()
                ),
            ));
        }
        if w.cols() != 36 * n_sb {
            return Err(GpuError::shape(
                what,
                format!(
                    "Q4_K rows are 36*{n_sb} = {} words at K={}, got {}",
                    36 * n_sb,
                    act.k(),
                    w.cols()
                ),
            ));
        }
        if rows_per_expert == 0 || !w.rows().is_multiple_of(rows_per_expert) {
            return Err(GpuError::shape(
                what,
                format!(
                    "w.rows() {} must be a positive multiple of \
                 rows_per_expert {rows_per_expert}",
                    w.rows()
                ),
            ));
        }
        if n_slots == 0 || sel.len() < n_slots {
            return Err(GpuError::shape(
                what,
                format!(
                    "need n_slots >= 1 and sel.len() >= n_slots, got \
                 n_slots {n_slots} sel.len() {}",
                    sel.len()
                ),
            ));
        }
        if y.len() < n_slots * rows_per_expert {
            return Err(GpuError::shape(
                what,
                format!(
                    "y.len() {} < n_slots*rows_per_expert = {}",
                    y.len(),
                    n_slots * rows_per_expert
                ),
            ));
        }
        let grid = launch_u32(what, "grid", (n_slots * rows_per_expert).div_ceil(8))?;
        let n_experts = launch_u32(what, "n_experts", w.rows() / rows_per_expert)?;
        let rows_per_expert = launch_u32(what, "rows_per_expert", rows_per_expert)?;
        let n_slots = launch_u32(what, "n_slots", n_slots)?;
        let n_sb = launch_u32(what, "n_sb", n_sb)?;
        let prep = self
            .module
            .prepare_q4k_gemv_sel(LaunchConfig1D::new(grid, 256, 0))?;
        self.module.q4k_gemv_sel(
            stream,
            &prep,
            w.buf(),
            &act.q4,
            &act.s8,
            &act.d8,
            sel,
            n_experts,
            rows_per_expert,
            n_slots,
            n_sb,
            n_sb.div_ceil(4),
            y,
        )?;
        Ok(())
    }
}
