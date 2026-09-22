//! MoE block fused kernels (P8 prep): the six routed experts' gate·up·swiglu
//! in one launch (`sel`-indirect rows, the shape `q3k_gemv_sel` addresses),
//! and the combine of the six down outputs with the router weights, the
//! shared-expert output and the residual in one launch. Bit-identical to the
//! per-op path they replace; the gate is `gate_moe_fused`.

use crate::GpuError;
use crate::cores::q3k_row_dot;
use crate::elem::{silu_mul, weighted_expert_sum};
use crate::tensor::{DeviceTensor, Q8Act};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread, warp};
use cuda_host::cuda_module;
use std::sync::Arc;

// Fusion rule (the same one the dense-block fusion applies): merge launches
// wherever there is NO cross-thread dependency, keep a launch boundary where
// there is one. The op path's eight launches after the shared input
// quantization:
//   gate gemv_sel -> up gemv_sel -> swiglu -> 32-value quantize
//   -> down gemv_sel -> weighted_sum -> add(shexp) -> add(resid)
// The two boundaries that stay: the 32-value quantization of the six
// intermediates needs ALL n_slots * rows_per_expert values (one 32-value
// block spans 32 rows = 32 warps = 4 CUDA blocks), and the combine needs all
// n_slots down outputs. Everything else merges:
//   [expert gate·up·swiglu, sel-indirect] [32-value quantize]
//   [down gemv_sel] [combine: weighted sum + shexp + residual]
// The arithmetic bodies are the cores the op-path kernels run
// (`cores::q3k_row_dot` twice, `elem::silu_mul`, `elem::weighted_expert_sum`
// plus two plain adds), so the gate's contract with the op path is bit
// identity, not a band. m = 1 (decode) only.

#[cuda_module]
mod moe_fused_kernels {
    use super::*;

    /// The routed experts' gate·up·swiglu in ONE launch: one warp per output
    /// row, thread row `n = slot * rows_per_expert + row_in_expert` reading
    /// weight row `sel[slot] * rows_per_expert + row_in_expert` of BOTH
    /// stacks (the `q3k_gemv_sel` addressing, same guards) —
    /// `cores::q3k_row_dot` twice against the ONE quantized activation
    /// column (gate and up read the same input, m = 1), each reduced with
    /// the same fixed warp tree, then `elem::silu_mul` on the two row dots,
    /// lane 0 storing `h[row]`. A row's output depends only on its own two
    /// weight rows and the shared quantized input, so the op path's three
    /// launches (gate gemv, up gemv, swiglu) close in the warp that owns the
    /// row. `h` is slot-major `[n_slots * rows_per_expert]` — exactly the `m
    /// = n_slots` columns of `k = rows_per_expert` f32 that `Q8Blocks32`'s
    /// quantizer consumes (column s = `h[s*rpe .. (s+1)*rpe]`), and the
    /// layout the down `_sel` kernel's per-slot column addressing expects.
    ///
    /// An id >= n_experts cannot be rejected by the host contract (it lives
    /// in device memory): the slot's warps return before their first load —
    /// warp-uniform, no divergent branch — leaving that slot of `h`
    /// untouched and every other slot unaffected, exactly as `q3k_gemv_sel`
    /// leaves its `y`.
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
            4 * wg.len() >= n_experts * rows_per_expert * 110 * n_sb,
            4 * wu.len() >= n_experts * rows_per_expert * 110 * n_sb,
            q.len() >= 64 * iters,
            d8.len() >= 2 * n_sb,
            sel.len() >= n_slots,
            h.len() >= n_slots * rows_per_expert
        )
    )]
    pub fn expert_gate_up_swiglu_q3k(
        wg: &[u32],
        wu: &[u32],
        q: &[u64],
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
        // load is warp-uniform (all 32 lanes of the warp share `row`, hence
        // `slot`), so the out-of-range return below never diverges a warp.
        let id = unsafe { *sel.get_unchecked(slot) } as usize;
        if id >= n_experts as usize {
            return;
        }
        let row_abs = id * rows_per_expert as usize + row % rows_per_expert as usize;
        let lane = warp::lane_id() as usize;
        // The m = 1 per-row body twice, shared with `q3k_gemv`,
        // `q3k_gemv_sel` and the dense-block fused kernel.
        let fg = q3k_row_dot(wg, q, d8, n_sb as usize, iters, row_abs, 0, 1, lane);
        let fu = q3k_row_dot(wu, q, d8, n_sb as usize, iters, row_abs, 0, 1, lane);
        // Each op-path gemv reduces its lane partials with the fixed warp
        // tree; swiglu then combines the two row dots.
        let g = warp::reduce_sum_f32(fg[0]);
        let u = warp::reduce_sum_f32(fu[0]);
        if lane == 0 {
            // SAFETY: row < n_slots*rows_per_expert <= h.len() by the launch
            // contract; only lane 0 of the warp writes h[row].
            unsafe {
                *h.get_unchecked_mut(row) = silu_mul(g, u);
            }
        }
    }

    /// The MoE combine, one thread per output value `d`:
    /// `y[d] = (Σ_e w[e] · down[e*rows + d] + shexp[d]) + resid[d]`. The sum
    /// is `elem::weighted_expert_sum` verbatim (its serial e-loop IS the op
    /// path's summation order), and the two adds are parenthesised exactly as
    /// the op path's two `elem::add` launches group them (`ffn_out =
    /// ffn_moe_out + ffn_shexp`, then `l_out = ffn_out + ffn_inp`): f32
    /// addition is commutative but not associative, so the grouping is
    /// load-bearing. `down` is the `q5_0_gemv_sel` output layout
    /// (slot-major, `n_slots * rows`), `w` the router's per-slot weights,
    /// m = 1.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            down.len() >= rows * n_slots,
            w.len() >= n_slots,
            shexp.len() >= rows,
            resid.len() >= rows,
            y.len() >= rows
        )
    )]
    pub fn moe_combine(
        down: &[f32],
        w: &[f32],
        shexp: &[f32],
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
        // SAFETY: d < rows bounds the shexp/resid reads and the y store by
        // the launch contract.
        unsafe {
            let s = *shexp.get_unchecked(d);
            let r = *resid.get_unchecked(d);
            *y.get_unchecked_mut(d) = (acc + s) + r;
        }
    }
}

/// The loaded MoE fused device module and its enqueue API. Owns no context
/// and no stream — every enqueue takes the engine stream (`Gpu::stream()`),
/// so launches order with the rest of the step and are capturable.
pub struct MoeFusedKernels {
    module: moe_fused_kernels::LoadedModule,
}

impl MoeFusedKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<MoeFusedKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; every launcher checks its launch contract.
        let module = unsafe { moe_fused_kernels::load(ctx)? };
        Ok(MoeFusedKernels { module })
    }

    /// Enqueue the routed experts' gate·up·swiglu for the m = 1 decode
    /// shape: one launch computes `n_slots` experts of BOTH Q3_K stacks —
    /// `wg`/`wu` the full resident flat stacks (rows of `110 * n_sb / 4` u32
    /// words, even n_sb, `w.rows()` the same positive multiple of
    /// `rows_per_expert` for both, `n_experts = wg.rows()/rows_per_expert`,
    /// as `Gpu::enqueue_gemv_q3k_sel` takes) — slot s writing
    /// `h[s*rows_per_expert .. (s+1)*rows_per_expert]` as
    /// `silu(gate_row(sel[s])) · (up_row(sel[s]))`, every slot dotting the
    /// ONE quantized column of `act`. `h` holds `n_slots * rows_per_expert`
    /// f32 slot-major — the exact span `Q5Kernels::enqueue_quantize_q8`
    /// consumes at m = n_slots, k = rows_per_expert. `sel` is a device
    /// buffer of at least `n_slots` ids read by the kernel per launch, so a
    /// captured graph replay picks up new ids written between replays; an
    /// id >= n_experts leaves that slot of `h` untouched. Asynchronous,
    /// allocation-free, capturable.
    #[allow(
        clippy::too_many_arguments,
        reason = "host launcher; folding these into a *Args struct is the R8 round"
    )]
    pub fn enqueue_expert_gate_up_swiglu(
        &self,
        stream: &CudaStream,
        wg: &DeviceTensor<u32>,
        wu: &DeviceTensor<u32>,
        act: &Q8Act,
        sel: &DeviceBuffer<u32>,
        n_slots: usize,
        rows_per_expert: usize,
        h: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let n_sb = act.n_sb();
        if act.m() != 1 {
            return Err(format!(
                "enqueue_expert_gate_up_swiglu: m = 1 only (the shared expert input), got \
                 act.m() = {}",
                act.m()
            )
            .into());
        }
        if !n_sb.is_multiple_of(2) {
            return Err(format!(
                "enqueue_expert_gate_up_swiglu: odd super-block count {n_sb} (K={}) leaves \
                 rows unaligned; repack rows at load time",
                act.k()
            )
            .into());
        }
        if wg.cols() != 110 * n_sb / 4 || wu.cols() != 110 * n_sb / 4 {
            return Err(format!(
                "enqueue_expert_gate_up_swiglu: Q3_K rows are 110*{n_sb}/4 = {} words at K={}, \
                 got gate {} x {}, up {} x {}",
                110 * n_sb / 4,
                act.k(),
                wg.rows(),
                wg.cols(),
                wu.rows(),
                wu.cols()
            )
            .into());
        }
        if rows_per_expert == 0
            || !wg.rows().is_multiple_of(rows_per_expert)
            || !wu.rows().is_multiple_of(rows_per_expert)
            || wg.rows() != wu.rows()
        {
            return Err(format!(
                "enqueue_expert_gate_up_swiglu: gate rows {} and up rows {} must be the same \
                 positive multiple of rows_per_expert {rows_per_expert}",
                wg.rows(),
                wu.rows()
            )
            .into());
        }
        if n_slots == 0 || sel.len() < n_slots {
            return Err(format!(
                "enqueue_expert_gate_up_swiglu: need n_slots >= 1 and sel.len() >= n_slots, \
                 got n_slots {n_slots} sel.len() {}",
                sel.len()
            )
            .into());
        }
        if h.len() < n_slots * rows_per_expert {
            return Err(format!(
                "enqueue_expert_gate_up_swiglu: h.len() {} < n_slots*rows_per_expert = {}",
                h.len(),
                n_slots * rows_per_expert
            )
            .into());
        }
        let n_experts = wg.rows() / rows_per_expert;
        let prep = self
            .module
            .prepare_expert_gate_up_swiglu_q3k(LaunchConfig1D::new(
                (n_slots * rows_per_expert).div_ceil(8) as u32,
                256,
                0,
            ))?;
        self.module.expert_gate_up_swiglu_q3k(
            stream,
            &prep,
            wg.buf(),
            wu.buf(),
            &act.q3,
            &act.d8,
            sel,
            n_experts as u32,
            rows_per_expert as u32,
            n_slots as u32,
            n_sb as u32,
            n_sb.div_ceil(2) as u32,
            h,
        )?;
        Ok(())
    }

    /// Enqueue the MoE combine — `elem::enqueue_weighted_sum` followed by
    /// the two `elem::enqueue_add` launches (moe+shexp, then +resid) in one
    /// launch, bit for bit: `y[d] = (Σ_e w[e]·down[e*rows + d] + shexp[d]) +
    /// resid[d]`. `down` holds `n_slots * rows` f32 in the `q5_0_gemv_sel`
    /// output layout (slot-major), `w` the router's `n_slots` per-slot
    /// weights, `shexp`/`resid`/`y` `rows` f32 each. Asynchronous,
    /// allocation-free, capturable.
    #[allow(
        clippy::too_many_arguments,
        reason = "host launcher; folding these into a *Args struct is the R8 round"
    )]
    pub fn enqueue_moe_combine(
        &self,
        stream: &CudaStream,
        down: &DeviceBuffer<f32>,
        w: &DeviceBuffer<f32>,
        shexp: &DeviceBuffer<f32>,
        resid: &DeviceBuffer<f32>,
        rows: usize,
        n_slots: usize,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        if rows == 0 || n_slots == 0 {
            return Err(format!(
                "enqueue_moe_combine: need rows/n_slots >= 1, got {rows}/{n_slots}"
            )
            .into());
        }
        if down.len() < rows * n_slots || w.len() < n_slots {
            return Err(format!(
                "enqueue_moe_combine: down.len() {} (need rows*n_slots = {}), w.len() {} (need \
                 {n_slots})",
                down.len(),
                rows * n_slots,
                w.len()
            )
            .into());
        }
        if shexp.len() < rows || resid.len() < rows || y.len() < rows {
            return Err(format!(
                "enqueue_moe_combine: shexp.len() {} / resid.len() {} / y.len() {} vs rows \
                 {rows}",
                shexp.len(),
                resid.len(),
                y.len()
            )
            .into());
        }
        let prep = self.module.prepare_moe_combine(LaunchConfig1D::new(
            rows.div_ceil(256) as u32,
            256,
            0,
        ))?;
        self.module.moe_combine(
            stream,
            &prep,
            down,
            w,
            shexp,
            resid,
            rows as u32,
            n_slots as u32,
            y,
        )?;
        Ok(())
    }
}
