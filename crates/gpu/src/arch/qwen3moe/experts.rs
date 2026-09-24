//! The qwen3moe routed experts' two launches beside the down `_sel`: the
//! selected experts' gate·up·SwiGLU over Q4_K stacks in one launch, and the
//! combine of the down outputs with the router weights and the residual —
//! this model has no shared expert, so the combine is the weighted sum plus
//! the residual and nothing else.
//!
//! The gate·up kernel is `moe_fused::expert_gate_up_swiglu_q3k`'s shape over
//! Q4_K rows: one warp per output row, thread row `n = slot · rows_per_expert
//! + r` reading weight row `sel[slot] · rows_per_expert + r` of both stacks,
//! each dotted against the one quantized activation column by
//! `cores::q4k_row_dot_1col` (the m = 1 body of `q4k_gemv`), reduced by the
//! fixed warp tree, then `elem::silu_mul`. So slot `s` of `h` is bit for bit
//! `silu(q4k_gemv(gate_e)) · q4k_gemv(up_e)` for `e = sel[s]`.
//!
//! The combine, one thread per output value `d`: `y[d] =
//! elem::weighted_expert_sum(down, w, d) + resid[d]` — the slot-ascending
//! weighted sum, then one add, the grouping of the reference's
//! `routed_out + ffn_inp`.

use crate::GpuError;
use crate::cores::q4k_row_dot_1col;
use crate::elem::{silu_mul, weighted_expert_sum};
use crate::launch_u32;
use crate::tensor::{DeviceTensor, Q8Act};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread, warp};
use cuda_host::cuda_module;
use std::sync::Arc;

#[cuda_module]
mod qwen3moe_expert_kernels {
    use super::*;

    /// The routed experts' gate·up·SwiGLU in one launch (module doc). An id
    /// `>= n_experts` cannot be refused by the host (it lives in device
    /// memory): the slot's warps return before their first weight load —
    /// warp-uniform — leaving that slot of `h` untouched.
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
            wg.len() >= n_experts * rows_per_expert * 36 * n_sb,
            wu.len() >= n_experts * rows_per_expert * 36 * n_sb,
            q.len() >= 256 * iters,
            s8.len() >= 8 * n_sb,
            d8.len() >= 2 * n_sb,
            sel.len() >= n_slots,
            h.len() >= n_slots * rows_per_expert
        )
    )]
    pub fn qwen3moe_gate_up_swiglu_q4k(
        wg: &[u32],
        wu: &[u32],
        q: &[u32],
        s8: &[i32],
        d8: &[f32],
        sel: &[u32],
        n_experts: u32,
        rows_per_expert: u32,
        n_slots: u32,
        n_sb: u32,
        iters: u32,
        mut h: DisjointSlice<f32>,
    ) {
        let t = thread::index_1d().get() % 256;
        let row = (thread::index_1d().get() / 256) * 8 + t / 32;
        if row >= n_slots as usize * rows_per_expert as usize {
            return;
        }
        let slot = row / rows_per_expert as usize;
        // SAFETY: slot < n_slots <= sel.len() by the launch contract; the
        // load is warp-uniform (the warp's 32 lanes share `row`), so the
        // return below never diverges a warp.
        let id = unsafe { *sel.get_unchecked(slot) } as usize;
        if id >= n_experts as usize {
            return;
        }
        let row_abs = id * rows_per_expert as usize + row % rows_per_expert as usize;
        let lane = warp::lane_id() as usize;
        // The core's caller contract, from the launch contract: row_abs <
        // n_experts · rows_per_expert rows of both stacks, column 0 of the
        // one-column activation, iters = ceil(n_sb/4) from the host, and all
        // 32 lanes of the warp are here (both returns are warp-uniform).
        let fg = q4k_row_dot_1col(wg, q, s8, d8, n_sb as usize, iters, row_abs, 0, lane);
        let fu = q4k_row_dot_1col(wu, q, s8, d8, n_sb as usize, iters, row_abs, 0, lane);
        let g = warp::reduce_sum_f32(fg);
        let u = warp::reduce_sum_f32(fu);
        if lane == 0 {
            // SAFETY: row < n_slots · rows_per_expert <= h.len() by the
            // launch contract; only lane 0 of the warp writes h[row].
            unsafe {
                *h.get_unchecked_mut(row) = silu_mul(g, u);
            }
        }
    }

    /// The combine, one thread per output value (module doc). `down` is the
    /// down `_sel` output (slot-major, `n_slots · rows`), `w` the router's
    /// per-slot weights.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            down.len() >= rows * n_slots,
            w.len() >= n_slots,
            resid.len() >= rows,
            y.len() >= rows
        )
    )]
    pub fn qwen3moe_combine(
        down: &[f32],
        w: &[f32],
        resid: &[f32],
        rows: u32,
        n_slots: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let d = thread::index_1d().get();
        if d >= rows as usize {
            return;
        }
        let acc = weighted_expert_sum(down, w, rows, n_slots, 0, d);
        // SAFETY: d < rows bounds the resid read and the y store by the
        // launch contract.
        unsafe {
            let r = *resid.get_unchecked(d);
            *y.get_unchecked_mut(d) = acc + r;
        }
    }
}

/// [`ExpertKernels::enqueue_gate_up`]'s arguments: both resident Q4_K stacks
/// (`36 · n_sb` words per row, the same positive multiple of
/// `rows_per_expert` rows each), the one-column quantized input, the device
/// ids, the slot count and the slot-major output.
pub struct GateUpArgs<'a> {
    pub wg: &'a DeviceTensor<u32>,
    pub wu: &'a DeviceTensor<u32>,
    pub act: &'a Q8Act,
    pub sel: &'a DeviceBuffer<u32>,
    pub n_slots: usize,
    pub rows_per_expert: usize,
    pub h: &'a mut DeviceBuffer<f32>,
}

/// The loaded module. Owns no stream: each enqueue takes the engine stream.
pub struct ExpertKernels {
    module: qwen3moe_expert_kernels::LoadedModule,
}

impl ExpertKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<ExpertKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; every launcher checks its launch contract.
        let module = unsafe { qwen3moe_expert_kernels::load(ctx)? };
        Ok(ExpertKernels { module })
    }

    /// Enqueue the selected experts' gate·up·SwiGLU: slot `s` writes
    /// `h[s · rows_per_expert ..][..rows_per_expert]` as `silu(gate row) ·
    /// up row` of expert `sel[s]`, every slot dotting the one quantized
    /// column of `act`. Asynchronous, allocation-free, capturable.
    pub fn enqueue_gate_up(&self, stream: &CudaStream, a: GateUpArgs<'_>) -> Result<(), GpuError> {
        let what = "qwen3moe::enqueue_gate_up";
        let GateUpArgs {
            wg,
            wu,
            act,
            sel,
            n_slots,
            rows_per_expert,
            h,
        } = a;
        let n_sb = act.n_sb();
        if act.m() != 1 {
            return Err(GpuError::shape(
                what,
                format!("one shared input column, got act.m() = {}", act.m()),
            ));
        }
        if wg.cols() != 36 * n_sb || wu.cols() != 36 * n_sb {
            return Err(GpuError::shape(
                what,
                format!(
                    "Q4_K rows are 36*{n_sb} words at K={}, got gate {}x{}, up {}x{}",
                    act.k(),
                    wg.rows(),
                    wg.cols(),
                    wu.rows(),
                    wu.cols()
                ),
            ));
        }
        if rows_per_expert == 0
            || wg.rows() != wu.rows()
            || !wg.rows().is_multiple_of(rows_per_expert)
        {
            return Err(GpuError::shape(
                what,
                format!(
                    "gate rows {} and up rows {} must be the same positive multiple of \
                     rows_per_expert {rows_per_expert}",
                    wg.rows(),
                    wu.rows()
                ),
            ));
        }
        if n_slots == 0 || sel.len() < n_slots || h.len() < n_slots * rows_per_expert {
            return Err(GpuError::shape(
                what,
                format!(
                    "n_slots {n_slots}, sel.len() {}, h.len() {} (need n_slots·rows_per_expert = {})",
                    sel.len(),
                    h.len(),
                    n_slots * rows_per_expert
                ),
            ));
        }
        let grid = launch_u32(what, "grid", (n_slots * rows_per_expert).div_ceil(8))?;
        let n_experts = launch_u32(what, "n_experts", wg.rows() / rows_per_expert)?;
        let rows_per_expert = launch_u32(what, "rows_per_expert", rows_per_expert)?;
        let n_slots = launch_u32(what, "n_slots", n_slots)?;
        let n_sb = launch_u32(what, "n_sb", n_sb)?;
        let prep = self
            .module
            .prepare_qwen3moe_gate_up_swiglu_q4k(LaunchConfig1D::new(grid, 256, 0))?;
        self.module.qwen3moe_gate_up_swiglu_q4k(
            stream,
            &prep,
            wg.buf(),
            wu.buf(),
            &act.q4,
            &act.s8,
            &act.d8,
            sel,
            n_experts,
            rows_per_expert,
            n_slots,
            n_sb,
            n_sb.div_ceil(4),
            h,
        )?;
        Ok(())
    }

    /// Enqueue `y[d] = Σ_s w[s] · down[s · rows + d] + resid[d]` over `rows`
    /// values and `n_slots` slots (module doc). Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_combine(
        &self,
        stream: &CudaStream,
        down: &DeviceBuffer<f32>,
        w: &DeviceBuffer<f32>,
        resid: &DeviceBuffer<f32>,
        n_slots: usize,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "qwen3moe::enqueue_combine";
        let rows = resid.len();
        if rows == 0
            || n_slots == 0
            || down.len() < rows * n_slots
            || w.len() < n_slots
            || y.len() < rows
        {
            return Err(GpuError::shape(
                what,
                format!(
                    "rows {rows} (resid), n_slots {n_slots}: down.len() {} w.len() {} y.len() {}",
                    down.len(),
                    w.len(),
                    y.len()
                ),
            ));
        }
        let rows = launch_u32(what, "rows", rows)?;
        let n_slots = launch_u32(what, "n_slots", n_slots)?;
        let prep = self.module.prepare_qwen3moe_combine(LaunchConfig1D::new(
            rows.div_ceil(256),
            256,
            0,
        ))?;
        self.module
            .qwen3moe_combine(stream, &prep, down, w, resid, rows, n_slots, y)?;
        Ok(())
    }
}
