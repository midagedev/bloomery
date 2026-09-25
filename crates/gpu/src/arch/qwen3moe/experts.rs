//! The qwen3moe routed experts' two launches beside the down `_sel`: the
//! selected experts' gate·up·SwiGLU over Q4_K stacks in one launch, and the
//! combine of the down outputs with the router weights and the residual —
//! this model has no shared expert, so the combine is the weighted sum plus
//! the residual and nothing else.
//!
//! The gate·up kernel is `moe_fused::expert_gate_up_swiglu_q3k`'s shape over
//! Q4_K rows: one warp per output row, thread row
//! `n = slot · rows_per_expert + r` reading weight row
//! `sel[slot] · rows_per_expert + r` of both stacks, each dotted against its
//! slot's quantized activation column by
//! `cores::q4k_row_dot_1col` (the m = 1 body of `q4k_gemv`, which picks the
//! column by its base offset), reduced by the fixed warp tree, then
//! `elem::silu_mul`. Slot `s` reads column `s / slots_per_col`: one column
//! for every slot of a token, a token's run of slots per column over several
//! tokens. So slot `s` of `h` is bit for bit `silu(q4k_gemv(gate_e)) ·
//! q4k_gemv(up_e)` for `e = sel[s]` on that column alone.
//!
//! An id at or past the stack's expert count has no rows to read: this chain
//! has no host tier, so no id past the stack is a contract here (not even
//! [`crate::hybrid::HOST`]). The slot's warps raise [`FaultSite::ExpertId`]
//! and write NaN into its rows of `h` before their first weight load, so no
//! stale row of an earlier launch passes for the slot's output.
//!
//! The combine, one thread per output value `d` of token `t`: `y[t · rows +
//! d] = elem::weighted_expert_sum(down, w, t, d) + resid[t · rows + d]` — the
//! slot-ascending weighted sum, then one add, the grouping of the
//! reference's `routed_out + ffn_inp`.

use crate::GpuError;
use crate::cores::q4k_row_dot_1col;
use crate::elem::{silu_mul, weighted_expert_sum};
use crate::fault::{FaultSink, FaultSite};
use crate::launch_u32;
use crate::tensor::{DeviceTensor, Q8Act};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread, warp};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Output rows of one gate·up block: eight warps, a row each.
const ROWS_PER_BLOCK: usize = 8;

/// Row `row` of the gate·up·SwiGLU (module doc): slot `row / rows_per_expert`
/// against column `slot / slots_per_col`, `h[row] = silu(g) · u`. A slot whose
/// id is past the stack raises [`FaultSite::ExpertId`] on `fault` and gets
/// `h[row] = NaN`, with no weight load.
///
/// # Safety
///
/// All 32 lanes of one warp call it with the same `row`, `row < n_slots ·
/// rows_per_expert` and `slot / slots_per_col < m_cols` for the entry's
/// launch contract (whose bounds on the stacks, the activation, `sel` and `h`
/// hold), `iters = ceil(n_sb/4)`, and lane 0 of this warp is `h[row]`'s only
/// writer.
#[allow(
    clippy::too_many_arguments,
    reason = "device core: it is handed a kernel entry's flat arguments (rust-quality R8)"
)]
#[inline(always)]
unsafe fn gate_up_row(
    wg: &[u32],
    wu: &[u32],
    q: &[u32],
    s8: &[i32],
    d8: &[f32],
    sel: &[u32],
    n_experts: u32,
    rows_per_expert: u32,
    slots_per_col: u32,
    n_sb: u32,
    iters: u32,
    row: usize,
    lane: usize,
    fault: FaultSink,
    h: &mut DisjointSlice<f32>,
) {
    let slot = row / rows_per_expert as usize;
    // SAFETY: slot < n_slots <= sel.len() by the caller's contract; the load
    // is warp-uniform (the warp's 32 lanes share `row`), so the return below
    // never diverges a warp.
    let id = unsafe { *sel.get_unchecked(slot) } as usize;
    if id >= n_experts as usize {
        if lane == 0 {
            fault.raise(FaultSite::ExpertId);
            // SAFETY: row < n_slots · rows_per_expert <= h.len() by the
            // caller's contract; only lane 0 of the warp writes h[row].
            unsafe { *h.get_unchecked_mut(row) = f32::NAN };
        }
        return;
    }
    let row_abs = id * rows_per_expert as usize + row % rows_per_expert as usize;
    let col = slot / slots_per_col as usize;
    // The core's caller contract, from this fn's: row_abs < n_experts ·
    // rows_per_expert rows of both stacks, column col < m_cols, iters =
    // ceil(n_sb/4), and all 32 lanes of the warp are here (the return above
    // is warp-uniform).
    let fg = q4k_row_dot_1col(wg, q, s8, d8, n_sb as usize, iters, row_abs, col, lane);
    let fu = q4k_row_dot_1col(wu, q, s8, d8, n_sb as usize, iters, row_abs, col, lane);
    let g = warp::reduce_sum_f32(fg);
    let u = warp::reduce_sum_f32(fu);
    if lane == 0 {
        // SAFETY: row < n_slots · rows_per_expert <= h.len() by the caller's
        // contract; only lane 0 of the warp writes h[row].
        unsafe {
            *h.get_unchecked_mut(row) = silu_mul(g, u);
        }
    }
}

#[cuda_module]
mod qwen3moe_expert_kernels {
    use super::*;

    /// The routed experts' gate·up·SwiGLU in one launch (module doc): slot
    /// `s` against column `s / slots_per_col` of the `m_cols`-column
    /// activation. An id `>= n_experts` cannot be refused by the host (it
    /// lives in device memory): the slot's warps raise
    /// [`FaultSite::ExpertId`] on `fault`, write NaN into the slot's rows of
    /// `h` and return before their first weight load — warp-uniform.
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
            slots_per_col >= 1,
            n_slots <= m_cols * slots_per_col,
            q.len() >= m_cols * 256 * iters,
            s8.len() >= m_cols * 8 * n_sb,
            d8.len() >= m_cols * 2 * n_sb,
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
        m_cols: u32,
        slots_per_col: u32,
        n_sb: u32,
        iters: u32,
        fault: FaultSink,
        mut h: DisjointSlice<f32>,
    ) {
        let t = thread::index_1d().get() % 256;
        let row = (thread::index_1d().get() / 256) * 8 + t / 32;
        if row >= n_slots as usize * rows_per_expert as usize {
            return;
        }
        // m_cols only bounds the activation in the launch contract.
        let _ = m_cols;
        // SAFETY: the guard above is warp-uniform (the warp's lanes share
        // `row`) and keeps row < n_slots · rows_per_expert; slot / slots_per_col
        // < m_cols because slot < n_slots <= m_cols · slots_per_col; the host
        // passes iters = ceil(n_sb/4); the rest is the launch contract; each
        // row is one warp's.
        unsafe {
            gate_up_row(
                wg,
                wu,
                q,
                s8,
                d8,
                sel,
                n_experts,
                rows_per_expert,
                slots_per_col,
                n_sb,
                iters,
                row,
                warp::lane_id() as usize,
                fault,
                &mut h,
            );
        }
    }

    /// The combine of `m` tokens, one thread per output value (module doc).
    /// `down` is the down `_sel` output (per token slot-major, `m · n_slots
    /// · rows`), `w` the router's per-slot weights (`m · n_slots`), `resid`
    /// and `y` token-major.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            down.len() >= rows * n_slots * m,
            w.len() >= n_slots * m,
            resid.len() >= rows * m,
            y.len() >= rows * m
        )
    )]
    pub fn qwen3moe_combine(
        down: &[f32],
        w: &[f32],
        resid: &[f32],
        rows: u32,
        n_slots: u32,
        m: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        let rows_u = rows as usize;
        if i >= rows_u * m as usize {
            return;
        }
        let t = i / rows_u;
        let acc = weighted_expert_sum(down, w, rows, n_slots, t, i - t * rows_u);
        // SAFETY: i < rows·m bounds the resid read and the y store by the
        // launch contract.
        unsafe {
            let r = *resid.get_unchecked(i);
            *y.get_unchecked_mut(i) = acc + r;
        }
    }
}

/// [`ExpertKernels::enqueue_gate_up`]'s arguments: both resident Q4_K stacks
/// (`36 · n_sb` words per row, the same positive multiple of
/// `rows_per_expert` rows each), the one-column quantized input, the device
/// ids, the slot count, the sink an id past the stack raises on, and the
/// slot-major output.
pub struct GateUpArgs<'a> {
    pub wg: &'a DeviceTensor<u32>,
    pub wu: &'a DeviceTensor<u32>,
    pub act: &'a Q8Act,
    pub sel: &'a DeviceBuffer<u32>,
    pub n_slots: usize,
    pub rows_per_expert: usize,
    pub fault: FaultSink,
    pub h: &'a mut DeviceBuffer<f32>,
}

/// [`ExpertKernels::enqueue_combine_tokens`]'s arguments: the down outputs
/// (per token slot-major), the router weights, the residual and the output
/// (token-major), `rows` values per token, `n_slots` slots per token, `m`
/// tokens.
pub struct CombineArgs<'a> {
    pub down: &'a DeviceBuffer<f32>,
    pub w: &'a DeviceBuffer<f32>,
    pub resid: &'a DeviceBuffer<f32>,
    pub rows: usize,
    pub n_slots: usize,
    pub m: usize,
    pub y: &'a mut DeviceBuffer<f32>,
}

/// A gate·up launch's checked shape, as the kernels take it.
struct GateUpShape {
    grid: u32,
    n_experts: u32,
    rows_per_expert: u32,
    slots_per_col: u32,
    n_slots: u32,
    m: u32,
    n_sb: u32,
}

/// Err unless `a` is a launchable gate·up (module doc, [`GateUpArgs`]): the
/// slots split over the activation's columns, both stacks Q4_K rows of the
/// activation's K and the same positive multiple of `rows_per_expert`, and
/// `sel` and `h` room for every slot.
fn gate_up_shape(what: &'static str, a: &GateUpArgs<'_>) -> Result<GateUpShape, GpuError> {
    let GateUpArgs {
        wg,
        wu,
        act,
        sel,
        n_slots,
        rows_per_expert,
        h,
        ..
    } = a;
    let (n_slots, rows_per_expert) = (*n_slots, *rows_per_expert);
    let (n_sb, m) = (act.n_sb(), act.m());
    if n_slots == 0 || !n_slots.is_multiple_of(m) {
        return Err(GpuError::shape(
            what,
            format!("{n_slots} slots do not split over act.m() = {m} columns"),
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
    if rows_per_expert == 0 || wg.rows() != wu.rows() || !wg.rows().is_multiple_of(rows_per_expert)
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
    if sel.len() < n_slots || h.len() < n_slots * rows_per_expert {
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
    Ok(GateUpShape {
        grid: launch_u32(
            what,
            "grid",
            (n_slots * rows_per_expert).div_ceil(ROWS_PER_BLOCK),
        )?,
        n_experts: launch_u32(what, "n_experts", wg.rows() / rows_per_expert)?,
        rows_per_expert: launch_u32(what, "rows_per_expert", rows_per_expert)?,
        slots_per_col: launch_u32(what, "slots_per_col", n_slots / m)?,
        n_slots: launch_u32(what, "n_slots", n_slots)?,
        m: launch_u32(what, "m", m)?,
        n_sb: launch_u32(what, "n_sb", n_sb)?,
    })
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
    /// up row` of expert `sel[s]`, dotting column `s / (n_slots / act.m())`
    /// of `act` — its one column, or at `m` columns each token's
    /// `n_slots / m` slots its own. A slot whose id is past the stack
    /// raises [`FaultSite::ExpertId`] on `a.fault` and its rows of `h` are
    /// NaN. Asynchronous, allocation-free, capturable.
    pub fn enqueue_gate_up(&self, stream: &CudaStream, a: GateUpArgs<'_>) -> Result<(), GpuError> {
        let what = "qwen3moe::enqueue_gate_up";
        let g = gate_up_shape(what, &a)?;
        let GateUpArgs {
            wg,
            wu,
            act,
            sel,
            fault,
            h,
            ..
        } = a;
        let prep = self
            .module
            .prepare_qwen3moe_gate_up_swiglu_q4k(LaunchConfig1D::new(g.grid, 256, 0))?;
        self.module.qwen3moe_gate_up_swiglu_q4k(
            stream,
            &prep,
            wg.buf(),
            wu.buf(),
            &act.q4,
            &act.s8,
            &act.d8,
            sel,
            g.n_experts,
            g.rows_per_expert,
            g.n_slots,
            g.m,
            g.slots_per_col,
            g.n_sb,
            g.n_sb.div_ceil(4),
            fault,
            h,
        )?;
        Ok(())
    }

    /// Enqueue `y[d] = Σ_s w[s] · down[s · rows + d] + resid[d]` over `rows
    /// = resid.len()` values and `n_slots` slots (module doc): one token.
    /// Asynchronous, allocation-free, capturable.
    pub fn enqueue_combine(
        &self,
        stream: &CudaStream,
        down: &DeviceBuffer<f32>,
        w: &DeviceBuffer<f32>,
        resid: &DeviceBuffer<f32>,
        n_slots: usize,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        self.enqueue_combine_tokens(
            stream,
            CombineArgs {
                down,
                w,
                resid,
                rows: resid.len(),
                n_slots,
                m: 1,
                y,
            },
        )
    }

    /// Enqueue the combine of `a.m` tokens (module doc): token `t`'s `rows`
    /// values from its `n_slots` down rows and weights. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_combine_tokens(
        &self,
        stream: &CudaStream,
        a: CombineArgs<'_>,
    ) -> Result<(), GpuError> {
        let what = "qwen3moe::enqueue_combine";
        let CombineArgs {
            down,
            w,
            resid,
            rows,
            n_slots,
            m,
            y,
        } = a;
        if rows == 0
            || n_slots == 0
            || m == 0
            || down.len() < rows * n_slots * m
            || w.len() < n_slots * m
            || resid.len() < rows * m
            || y.len() < rows * m
        {
            return Err(GpuError::shape(
                what,
                format!(
                    "rows {rows}, n_slots {n_slots}, m {m}: down.len() {} w.len() {} resid.len() \
                     {} y.len() {}",
                    down.len(),
                    w.len(),
                    resid.len(),
                    y.len()
                ),
            ));
        }
        let grid = launch_u32(what, "grid", (rows * m).div_ceil(256))?;
        let rows = launch_u32(what, "rows", rows)?;
        let n_slots = launch_u32(what, "n_slots", n_slots)?;
        let m = launch_u32(what, "m", m)?;
        let prep = self
            .module
            .prepare_qwen3moe_combine(LaunchConfig1D::new(grid, 256, 0))?;
        self.module
            .qwen3moe_combine(stream, &prep, down, w, resid, rows, n_slots, m, y)?;
        Ok(())
    }
}
