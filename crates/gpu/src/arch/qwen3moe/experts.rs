//! The qwen3moe routed experts' two launches beside the down `_sel`: the
//! selected experts' gate·up·SwiGLU over Q4_K stacks in one launch (at one
//! token `qwen3moe_gate_up_swiglu_quant_q4k`, which also writes the down's
//! q8_1 input, so its quantizer launch goes), and the combine of the down
//! outputs with the router weights and the residual — this model has no
//! shared expert, so the combine is the weighted sum plus the residual and
//! nothing else.
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
//! The combine, one thread per output value `d` of token `t`: `y[t · rows +
//! d] = elem::weighted_expert_sum(down, w, t, d) + resid[t · rows + d]` — the
//! slot-ascending weighted sum, then one add, the grouping of the
//! reference's `routed_out + ffn_inp`.

use crate::GpuError;
use crate::cores::q4k_row_dot_1col;
use crate::elem::{silu_mul, weighted_expert_sum};
use crate::fault::{FaultSink, FaultSite, quad_finite};
use crate::launch_u32;
use crate::q8_1_quant_vals;
use crate::tensor::{DeviceTensor, Q8Act};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32};
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread, threadfence, warp,
};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Output rows of one gate·up block: eight warps, a row each.
const ROWS_PER_BLOCK: usize = 8;
/// Values of one q8_1 block, and so rows of `h` one quantized group spans.
const GROUP_ROWS: usize = 128;
/// Gate·up blocks whose rows make up one q8_1 group: the tickets a group
/// draws before its values are whole.
const BLOCKS_PER_GROUP: u32 = (GROUP_ROWS / ROWS_PER_BLOCK) as u32;
const _: () = assert!(BLOCKS_PER_GROUP as usize * ROWS_PER_BLOCK == GROUP_ROWS);

/// Row `row` of the gate·up·SwiGLU (module doc): slot `row / rows_per_expert`
/// against column `slot / slots_per_col`, `h[row] = silu(g) · u`. A slot whose
/// id is past the stack is left untouched: its warp stores nothing.
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
    h: &mut DisjointSlice<f32>,
) {
    let slot = row / rows_per_expert as usize;
    // SAFETY: slot < n_slots <= sel.len() by the caller's contract; the load
    // is warp-uniform (the warp's 32 lanes share `row`), so the return below
    // never diverges a warp.
    let id = unsafe { *sel.get_unchecked(slot) } as usize;
    if id >= n_experts as usize {
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
    /// lives in device memory): the slot's warps return before their first
    /// weight load — warp-uniform — leaving that slot of `h` untouched.
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
                &mut h,
            );
        }
    }

    /// The routed experts' gate·up·SwiGLU of one token with the q8_1 of its
    /// output in the same launch: every row as `qwen3moe_gate_up_swiglu_q4k`
    /// at one column ([`gate_up_row`]), then the quantizer of the down's
    /// input. A 128-value q8_1 block of `h` is 128 rows = [`BLOCKS_PER_GROUP`]
    /// blocks of this grid (`rows_per_expert` a multiple of 128, so a group
    /// never straddles two slots): each block publishes its rows (a fence,
    /// then one ticket on its group's count `done[g]`), and the block that
    /// draws a group's last ticket puts the count back to zero and quantizes
    /// the group with warp 0 — its values read back volatile, the fault raise
    /// of `q8_1_quant_block` ([`FaultSite::QuantColumn`]), then
    /// `q8_1_quant_vals`. So `act_h` holds the bytes `q3k_quantize_q8_1`
    /// writes from `h` after the gate·up launch: group `g` is column `g /
    /// (2·h_n_sb)`, block `g mod 2·h_n_sb` of the `n_slots`-column activation.
    ///
    /// Five blocks per SM, the unfused entry's occupancy: the quantizer tail
    /// runs after the rows, so the bound caps registers the tail would
    /// otherwise add to the whole launch.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256, 5)]
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
            h.len() >= n_slots * rows_per_expert,
            rows_per_expert == 256 * h_n_sb,
            hq3.len() >= n_slots * 64 * h_half_it,
            hq4.len() >= n_slots * 256 * h_quad_it,
            hq6.len() >= n_slots * 128 * h_half_it,
            hs8.len() >= n_slots * 8 * h_n_sb,
            hd8.len() >= n_slots * 2 * h_n_sb,
            128 * done.len() >= n_slots * rows_per_expert
        )
    )]
    pub fn qwen3moe_gate_up_swiglu_quant_q4k(
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
        h_n_sb: u32,
        h_half_it: u32,
        h_quad_it: u32,
        mut h: DisjointSlice<f32>,
        mut hq3: DisjointSlice<u64>,
        mut hq4: DisjointSlice<u32>,
        mut hq6: DisjointSlice<u32>,
        mut hs8: DisjointSlice<i32>,
        mut hd8: DisjointSlice<f32>,
        mut done: DisjointSlice<u32>,
        fault: FaultSink,
    ) {
        static mut LAST: SharedArray<u32, 1> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let wi = tid / 32;
        let lane = warp::lane_id() as usize;
        let blk = thread::blockIdx_x() as usize;
        let row = blk * ROWS_PER_BLOCK + wi;
        // The guard is warp-uniform; a warp past the rows still meets the
        // barriers and its block the ticket.
        if row < n_slots as usize * rows_per_expert as usize {
            // SAFETY: as in `qwen3moe_gate_up_swiglu_q4k` with one column:
            // slots_per_col = n_slots puts every slot on column 0, the host
            // passes iters = ceil(n_sb/4), the bounds are the launch
            // contract's, and each row is one warp's.
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
                    n_slots,
                    n_sb,
                    iters,
                    row,
                    lane,
                    &mut h,
                );
            }
        }
        threadfence();
        thread::sync_threads();

        let group = blk / BLOCKS_PER_GROUP as usize;
        // SAFETY: block-shared, one element; the raw form reaches the
        // `static mut` without a reference.
        let last = unsafe { SharedArray::as_raw_mut_ptr(&raw mut LAST) };
        if tid == 0 {
            // SAFETY: the grid is n_slots·rows_per_expert / 8 blocks (host),
            // so group < n_slots·rows_per_expert / 128 <= done.len() (launch
            // contract); every access to a count is atomic.
            let count = unsafe { DeviceAtomicU32::from_ptr(done.as_mut_ptr().add(group)) };
            let is_last = count.fetch_add(1, AtomicOrdering::AcqRel) + 1 == BLOCKS_PER_GROUP;
            if is_last {
                // Every block of the group has drawn its ticket.
                count.store(0, AtomicOrdering::Relaxed);
            }
            // SAFETY: block-shared, one element, written before the barrier
            // that publishes it.
            unsafe { *last = u32::from(is_last) };
        }
        thread::sync_threads();
        // SAFETY: block-shared, one element, written by thread 0 before the
        // barrier above.
        if unsafe { *last } == 0 || wi != 0 {
            return;
        }

        // Warp 0 of the group's last block: every row of the group is
        // published. Volatile loads: this block's L1 never held the other
        // blocks' rows, and a volatile load does not ask it.
        let base = GROUP_ROWS * group + 4 * lane;
        let hp = h.as_mut_ptr();
        // SAFETY: base + 3 < 128·(group + 1) <= n_slots·rows_per_expert <=
        // h.len() by the launch contract.
        let v = unsafe {
            [
                core::ptr::read_volatile(hp.add(base)),
                core::ptr::read_volatile(hp.add(base + 1)),
                core::ptr::read_volatile(hp.add(base + 2)),
                core::ptr::read_volatile(hp.add(base + 3)),
            ]
        };
        if !quad_finite(v) {
            fault.raise(FaultSite::QuantColumn);
        }
        let per_col = 2 * h_n_sb as usize;
        // SAFETY: rows_per_expert = 256·h_n_sb (launch contract), so group g
        // is block g mod 2·h_n_sb < 2·h_n_sb of column g / (2·h_n_sb) <
        // n_slots, and `v` holds that block's values 128·b + 4·lane .. +3 of
        // the column (h is slot-major, one column per slot); the output
        // bounds are the launch contract's, and the whole warp is here with
        // the same group.
        unsafe {
            q8_1_quant_vals(
                v,
                group / per_col,
                group % per_col,
                h_n_sb as usize,
                h_half_it,
                h_quad_it,
                lane,
                &mut hq3,
                &mut hq4,
                &mut hq6,
                &mut hs8,
                &mut hd8,
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

/// The fused gate·up's per-group ticket counts
/// ([`ExpertKernels::enqueue_gate_up_quant`]): one per 128-value q8_1 group
/// of `h`, zero before a launch and zero again after it. The counts serve one
/// launch at a time, so launches sharing them must be ordered on one stream.
pub struct GroupTickets {
    done: DeviceBuffer<u32>,
}

impl GroupTickets {
    /// `groups` counts at zero. Load-time only.
    pub fn new(stream: &CudaStream, groups: usize) -> Result<GroupTickets, GpuError> {
        if groups == 0 {
            return Err(GpuError::shape(
                "qwen3moe::GroupTickets::new",
                "zero groups",
            ));
        }
        Ok(GroupTickets {
            done: DeviceBuffer::zeroed(stream, groups)?,
        })
    }

    /// Whether every count stands at zero, as it must between launches.
    /// Blocking read; gate use.
    pub fn at_zero(&self, stream: &CudaStream) -> Result<bool, GpuError> {
        Ok(self.done.to_host_vec(stream)?.iter().all(|&c| c == 0))
    }

    /// Device bytes of the counts.
    pub fn bytes(&self) -> usize {
        self.done.num_bytes()
    }
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
    /// `n_slots / m` slots its own. Asynchronous, allocation-free,
    /// capturable.
    pub fn enqueue_gate_up(&self, stream: &CudaStream, a: GateUpArgs<'_>) -> Result<(), GpuError> {
        let what = "qwen3moe::enqueue_gate_up";
        let g = gate_up_shape(what, &a)?;
        let GateUpArgs {
            wg,
            wu,
            act,
            sel,
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
            h,
        )?;
        Ok(())
    }

    /// Enqueue one token's gate·up·SwiGLU as [`ExpertKernels::enqueue_gate_up`]
    /// does at one column, with the q8_1 of `h` in the same launch
    /// (`qwen3moe_gate_up_swiglu_quant_q4k`): `act_h` — `n_slots` columns of
    /// `rows_per_expert` values, one per slot — takes the bytes
    /// `Gpu::enqueue_quantize_q8_1` writes from `h`, and `fault` the
    /// quantizer's raise. `rows_per_expert` must be a multiple of 128 (a
    /// q8_1 group never straddles two slots) and `tickets` hold a count per
    /// group. Asynchronous, allocation-free, capturable.
    pub fn enqueue_gate_up_quant(
        &self,
        stream: &CudaStream,
        a: GateUpArgs<'_>,
        act_h: &mut Q8Act,
        tickets: &mut GroupTickets,
        fault: FaultSink,
    ) -> Result<(), GpuError> {
        let what = "qwen3moe::enqueue_gate_up_quant";
        let g = gate_up_shape(what, &a)?;
        let GateUpArgs {
            wg,
            wu,
            act,
            sel,
            n_slots,
            rows_per_expert,
            h,
        } = a;
        let groups = n_slots * rows_per_expert / GROUP_ROWS;
        if act.m() != 1
            || !rows_per_expert.is_multiple_of(GROUP_ROWS)
            || act_h.m() != n_slots
            || act_h.k() != rows_per_expert
            || tickets.done.len() < groups
        {
            return Err(GpuError::shape(
                what,
                format!(
                    "one activation column (got {}), rows_per_expert {rows_per_expert} a multiple of \
                     {GROUP_ROWS}, act_h {} x {} for {n_slots} slots of {rows_per_expert}, {} ticket \
                     counts for {groups} groups",
                    act.m(),
                    act_h.m(),
                    act_h.k(),
                    tickets.done.len()
                ),
            ));
        }
        let h_n_sb = act_h.n_sb();
        let h_n_sb = launch_u32(what, "h_n_sb", h_n_sb)?;
        let prep = self
            .module
            .prepare_qwen3moe_gate_up_swiglu_quant_q4k(LaunchConfig1D::new(g.grid, 256, 0))?;
        self.module.qwen3moe_gate_up_swiglu_quant_q4k(
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
            g.n_sb,
            g.n_sb.div_ceil(4),
            h_n_sb,
            h_n_sb.div_ceil(2),
            h_n_sb.div_ceil(4),
            h,
            &mut act_h.q3,
            &mut act_h.q4,
            &mut act_h.q6,
            &mut act_h.s8,
            &mut act_h.d8,
            &mut tickets.done,
            fault,
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
