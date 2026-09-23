//! Expert FFNs: SwiGLU with its ±10 clamp, for the routed experts and for
//! the shared expert, whose weights are q8_0.
//!
//! Two kernels, the decode shape (one token per launch):
//! - `ds41_expert_gate_up`: the routed experts' gate·up·SwiGLU in one launch
//!   — the addressing of `moe_fused`'s `expert_gate_up_swiglu_q3k` (a warp
//!   per output row, weight row `sel[slot]·rows + row` of both stacks),
//!   `cores::q3k_row_dot` twice against the one q8_1 activation column, each
//!   reduced by the warp tree — the dots are bit for bit `q3k_gemv_sel`'s —
//!   then [`swiglu_clamp`]. The down projection after it is `q4k_sel`'s gemv
//!   over the q8_1 of `h`.
//! - `ds41_shexp_gate_up`: the shared expert's gate·up·SwiGLU, q8_0 weights
//!   against f32 activations: `q8f32::q8_0_lane_partial_1col` twice with the
//!   warp tree (each dot bit for bit `q8_0_gemv`'s at m = 1), then
//!   [`swiglu_clamp`]. The down projection after it is `q8_0_gemv`.
//!
//! Numeric contract, where it is ik's CPU rule op for op (the gate holds each
//! kernel to this module's host functions, bit for bit):
//! - SwiGLU: `min(silu(g), L) · clamp(u, -L, L)` with `L` the layer's
//!   `swiglu_clamp_exp` (routed) or `swiglu_clamp_shexp` (shared) value, and
//!   no clamp when `L <= 1e-6` — ik's `mul_mat_up_gate_NxM`, which clamps
//!   `silu(g)`, not `g`. `silu(x) = x / (1 + e^(0 - x))` with the
//!   exponential `expf_ik`, ik's AVX2 `v_expf`, which is what its CPU build
//!   runs on every row of these shapes.
//!
//! The dots are ours, not ik's: q8_1 activations per 128 values for the
//! q3_K stacks, f32 activations for q8_0, where ik quantizes to q8_K and
//! q8_2 — the gate derives its band against ik's dump from that difference.

use bloomery_gpu::cores::q3k_row_dot;
use bloomery_gpu::q8f32::q8_0_lane_partial_1col;
use bloomery_gpu::weights::DevWeight;
use bloomery_gpu::{DeviceTensor, GpuError, Q8Act, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread, warp};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Threads per block of every kernel here: eight warps.
const BLOCK: u32 = 256;

/// ik's AVX2 `v_expf` (`iqk_utils.h`), one lane, op for op: `x` split as
/// `n·ln2 + b` by the `0x1.8p23` shift (two fused multiply-adds for `b`), a
/// degree-5 polynomial in `b` by fused multiply-adds, and the scale `2^n` put
/// into the exponent bits — with ik's two escape paths for `|n| > 126` (split
/// scale) and `|n| > 192` (overflow to infinity, underflow to zero). Every
/// product that meets an add is an explicit `mul_add`, so the device and the
/// host round the same operations; the two plain products of the escape paths
/// scale by a power of two, exact unless the result leaves the normal range,
/// where adding 1 rounds to the same value fused or not.
#[inline(always)]
pub(crate) fn expf_ik(x: f32) -> f32 {
    const SHIFT: f32 = f32::from_bits(0x4b40_0000); // 0x1.8p23
    const LOG2E: f32 = f32::from_bits(0x3fb8_aa3b); // 0x1.715476p+0
    const LN2_HI: f32 = f32::from_bits(0x3f31_7200); // 0x1.62e4p-1
    const LN2_LO: f32 = f32::from_bits(0x35bf_be8e); // 0x1.7f7d1cp-20
    const C0: f32 = f32::from_bits(0x3f7f_fff6); // 0x1.ffffecp-1
    const C1: f32 = f32::from_bits(0x3eff_fedb); // 0x1.fffdb6p-2
    const C2: f32 = f32::from_bits(0x3e2a_af33); // 0x1.555e66p-3
    const C3: f32 = f32::from_bits(0x3d2b_9f17); // 0x1.573e2ep-5
    const C4: f32 = f32::from_bits(0x3c07_2010); // 0x1.0e4020p-7
    let z = x.mul_add(LOG2E, SHIFT);
    let n = z - SHIFT;
    let b = (-n).mul_add(LN2_LO, (-n).mul_add(LN2_HI, x));
    let e = z.to_bits() << 23;
    let k = f32::from_bits(e.wrapping_add(0x3f80_0000));
    let u = b * b;
    let j = C4
        .mul_add(b, C3)
        .mul_add(u, C2.mul_add(b, C1))
        .mul_add(u, C0 * b);
    // An ordered compare, as ik's: a NaN `n` takes the main path.
    if n.abs() > 126.0 {
        let g: u32 = if n <= 0.0 { 0x8200_0000 } else { 0 };
        let s1 = f32::from_bits(g.wrapping_add(0x7f00_0000));
        let s2 = f32::from_bits(e.wrapping_sub(g));
        return if n.abs() > 192.0 {
            s1 * s1
        } else {
            s2.mul_add(j, s2) * s1
        };
    }
    j.mul_add(k, k)
}

/// ik's AVX2 `v_silu`: `x / (1 + expf_ik(0 - x))`.
#[inline(always)]
pub fn silu_ik(x: f32) -> f32 {
    x / (1.0 + expf_ik(0.0 - x))
}

/// ik's clamped SwiGLU for one row: `min(silu(g), limit) · max(-limit,
/// min(limit, u))`, or `silu(g) · u` when `limit <= 1e-6`, each `min`/`max`
/// spelled as the `std::min`/`std::max` comparison it compiles from (a NaN
/// `silu(g)` passes the first; a NaN `u` becomes `limit`).
#[inline(always)]
pub fn swiglu_clamp(g: f32, u: f32, limit: f32) -> f32 {
    let mut s = silu_ik(g);
    let mut uc = u;
    if limit > 1e-6 {
        s = if limit < s { limit } else { s };
        uc = if u < limit { u } else { limit };
        uc = if -limit < uc { uc } else { -limit };
    }
    uc * s
}

#[cuda_module]
mod experts_kernels {
    use super::*;

    /// The routed experts' gate·up·SwiGLU for one token: `n_slots` experts
    /// of both Q3_K stacks in one launch. Thread row `n = slot ·
    /// rows_per_expert + r` (one warp per row, 8 rows per block) reads weight
    /// row `sel[slot] · rows_per_expert + r` of `wg` and `wu`, dots both
    /// against the one q8_1 column (`q`, `d8`), reduces each with the warp
    /// tree and stores `h[n] = swiglu_clamp(g, u, limit)` from lane 0. `h`
    /// is slot-major, the `n_slots` columns of `rows_per_expert` values the
    /// down projection's q8_1 quantization reads. An id `>= n_experts` (an
    /// expert not resident on this card) returns before any load — the test
    /// is warp-uniform — and leaves its slot of `h` as it was.
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
    pub fn ds41_expert_gate_up(
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
        limit: f32,
        mut h: DisjointSlice<f32>,
    ) {
        let t = thread::index_1d().get() % 256;
        let row = (thread::index_1d().get() / 256) * 8 + t / 32;
        if row >= n_slots as usize * rows_per_expert as usize {
            return;
        }
        let slot = row / rows_per_expert as usize;
        // SAFETY: slot < n_slots <= sel.len() by the launch contract. All 32
        // lanes of the warp share `row`, hence `slot` and `id`: the return
        // below is warp-uniform.
        let id = unsafe { *sel.get_unchecked(slot) } as usize;
        if id >= n_experts as usize {
            return;
        }
        let row_abs = id * rows_per_expert as usize + row % rows_per_expert as usize;
        let lane = warp::lane_id() as usize;
        let fg = q3k_row_dot(wg, q, d8, n_sb as usize, iters, row_abs, 0, 1, lane);
        let fu = q3k_row_dot(wu, q, d8, n_sb as usize, iters, row_abs, 0, 1, lane);
        let g = warp::reduce_sum_f32(fg[0]);
        let u = warp::reduce_sum_f32(fu[0]);
        if lane == 0 {
            let v = swiglu_clamp(g, u, limit);
            // SAFETY: row < n_slots * rows_per_expert <= h.len() by the
            // launch contract; lane 0 of the row's warp is its only writer.
            unsafe { *h.get_unchecked_mut(row) = v };
        }
    }

    /// The shared expert's gate·up·SwiGLU for one token: row `r` (one warp
    /// per row, 8 rows per block) dots gate row `r` and up row `r` — q8_0 in
    /// the q8f32 two-plane layout, `k` values each — against the `k` f32
    /// activations `x`, reduces each with the warp tree and stores `h[r] =
    /// swiglu_clamp(g, u, limit)` from lane 0.
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
            4 * qs_g.len() >= n_rows * k,
            32 * d_g.len() >= n_rows * k,
            4 * qs_u.len() >= n_rows * k,
            32 * d_u.len() >= n_rows * k,
            x.len() >= k,
            h.len() >= n_rows
        )
    )]
    pub fn ds41_shexp_gate_up(
        qs_g: &[u32],
        d_g: &[u16],
        qs_u: &[u32],
        d_u: &[u16],
        x: &[f32],
        n_rows: u32,
        k: u32,
        limit: f32,
        mut h: DisjointSlice<f32>,
    ) {
        let t = thread::index_1d().get() % 256;
        let row = (thread::index_1d().get() / 256) * 8 + t / 32;
        if row >= n_rows as usize {
            return;
        }
        let lane = warp::lane_id() as usize;
        // SAFETY: row < n_rows puts the row's words and scales inside qs_g and
        // d_g (the contract's 4·qs_g.len() and 32·d_g.len() bounds), x.len()
        // >= k with x0 = 0, the launcher passes k a positive multiple of 32,
        // and lane < 32.
        let fg = unsafe { q8_0_lane_partial_1col(qs_g, d_g, x, k, row, 0, lane) };
        let g = warp::reduce_sum_f32(fg);
        // SAFETY: the same bounds for qs_u and d_u.
        let fu = unsafe { q8_0_lane_partial_1col(qs_u, d_u, x, k, row, 0, lane) };
        let u = warp::reduce_sum_f32(fu);
        if lane == 0 {
            let v = swiglu_clamp(g, u, limit);
            // SAFETY: row < n_rows <= h.len() by the launch contract; lane 0
            // of the row's warp is its only writer.
            unsafe { *h.get_unchecked_mut(row) = v };
        }
    }
}

/// One token's routed gate·up·SwiGLU launch
/// ([`ExpertKernels::enqueue_expert_gate_up`]).
pub struct ExpertGateUp<'a> {
    /// The gate stack resident on this card: `n_experts · rows_per_expert`
    /// Q3_K rows of `110 · n_sb / 4` words, experts in id order from 0.
    pub wg: &'a DeviceTensor<u32>,
    /// The up stack, the same shape as `wg`.
    pub wu: &'a DeviceTensor<u32>,
    /// The token's activations quantized to q8_1: one column, even `n_sb`.
    pub act: &'a Q8Act,
    /// At least `n_slots` expert ids, read on the device at each launch (a
    /// replayed graph picks up ids written between replays).
    pub sel: &'a DeviceBuffer<u32>,
    pub n_slots: usize,
    pub rows_per_expert: usize,
    /// The layer's `swiglu_clamp_exp` value.
    pub limit: f32,
}

/// The loaded expert module. Owns no context and no stream — every enqueue
/// takes the engine stream, so launches order with the rest of the step and
/// are capturable.
pub struct ExpertKernels {
    module: experts_kernels::LoadedModule,
}

impl ExpertKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<ExpertKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; every launcher checks its launch contract.
        let module = unsafe { experts_kernels::load(ctx)? };
        Ok(ExpertKernels { module })
    }

    /// Enqueue the routed experts' gate·up·SwiGLU for one token: `h` takes
    /// `n_slots · rows_per_expert` f32, slot-major; a slot whose id is not
    /// below `wg.rows() / rows_per_expert` keeps what it held. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_expert_gate_up(
        &self,
        stream: &CudaStream,
        a: &ExpertGateUp<'_>,
        h: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "enqueue_expert_gate_up";
        let shape = |detail: String| GpuError::Shape { what, detail };
        let (wg, wu, rpe, n_slots) = (a.wg, a.wu, a.rows_per_expert, a.n_slots);
        let n_sb = a.act.n_sb();
        if a.act.m() != 1 || !n_sb.is_multiple_of(2) {
            return Err(shape(format!(
                "one q8_1 column of an even super-block count, got m = {} n_sb = {n_sb}",
                a.act.m()
            )));
        }
        let words = 110 * n_sb / 4;
        if wg.cols() != words || wu.cols() != words || wg.rows() != wu.rows() {
            return Err(shape(format!(
                "Q3_K rows are {words} words at n_sb {n_sb}, got gate {} x {}, up {} x {}",
                wg.rows(),
                wg.cols(),
                wu.rows(),
                wu.cols()
            )));
        }
        if rpe == 0 || !wg.rows().is_multiple_of(rpe) {
            return Err(shape(format!(
                "stack rows {} are not a positive multiple of rows_per_expert {rpe}",
                wg.rows()
            )));
        }
        if n_slots == 0 || a.sel.len() < n_slots || h.len() < n_slots * rpe {
            return Err(shape(format!(
                "n_slots {n_slots} needs sel.len() {} >= n_slots >= 1 and h.len() {} >= {}",
                a.sel.len(),
                h.len(),
                n_slots * rpe
            )));
        }
        let grid = launch_u32(what, "grid", (n_slots * rpe).div_ceil(8))?;
        let n_experts = launch_u32(what, "n_experts", wg.rows() / rpe)?;
        let rpe = launch_u32(what, "rows_per_expert", rpe)?;
        let n_slots = launch_u32(what, "n_slots", n_slots)?;
        let n_sb = launch_u32(what, "n_sb", n_sb)?;
        let prep = self
            .module
            .prepare_ds41_expert_gate_up(LaunchConfig1D::new(grid, BLOCK, 0))?;
        self.module.ds41_expert_gate_up(
            stream,
            &prep,
            wg.buf(),
            wu.buf(),
            a.act.q3(),
            a.act.d8(),
            a.sel,
            n_experts,
            rpe,
            n_slots,
            n_sb,
            n_sb.div_ceil(2),
            a.limit,
            h,
        )?;
        Ok(())
    }

    /// Enqueue the shared expert's gate·up·SwiGLU for one token: `gate` and
    /// `up` the layer's q8_0 file tensors (the same rows × k), `x` the
    /// token's `k` f32 activations, `limit` the layer's `swiglu_clamp_shexp`
    /// value; `h` takes one f32 per row. Asynchronous, allocation-free,
    /// capturable.
    pub fn enqueue_shexp_gate_up(
        &self,
        stream: &CudaStream,
        gate: &DevWeight,
        up: &DevWeight,
        x: &DeviceBuffer<f32>,
        limit: f32,
        h: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "enqueue_shexp_gate_up";
        let shape = |detail: String| GpuError::Shape { what, detail };
        let (
            DevWeight::Q8_0 {
                qs: qs_g,
                d: d_g,
                k,
            },
            DevWeight::Q8_0 {
                qs: qs_u,
                d: d_u,
                k: k_u,
            },
        ) = (gate, up)
        else {
            return Err(shape("gate and up must both be q8_0 file tensors".into()));
        };
        let (n_rows, k) = (d_g.rows(), *k);
        if *k_u != k
            || d_u.rows() != n_rows
            || k == 0
            || !k.is_multiple_of(32)
            || d_g.cols() * 32 != k
            || d_u.cols() * 32 != k
            || qs_g.rows() != n_rows
            || qs_u.rows() != n_rows
            || qs_g.cols() * 4 != k
            || qs_u.cols() * 4 != k
        {
            return Err(shape(format!(
                "gate {} rows x {k} and up {} rows x {k_u} must match, with k/4 code words \
                 and k/32 scales per row",
                d_g.rows(),
                d_u.rows()
            )));
        }
        if n_rows == 0 || x.len() < k || h.len() < n_rows {
            return Err(shape(format!(
                "{n_rows} rows need x.len() {} >= {k} and h.len() {} >= {n_rows}",
                x.len(),
                h.len()
            )));
        }
        let grid = launch_u32(what, "grid", n_rows.div_ceil(8))?;
        let n_rows = launch_u32(what, "n_rows", n_rows)?;
        let k = launch_u32(what, "k", k)?;
        let prep = self
            .module
            .prepare_ds41_shexp_gate_up(LaunchConfig1D::new(grid, BLOCK, 0))?;
        self.module.ds41_shexp_gate_up(
            stream,
            &prep,
            qs_g.buf(),
            d_g.buf(),
            qs_u.buf(),
            d_u.buf(),
            x,
            n_rows,
            k,
            limit,
            h,
        )?;
        Ok(())
    }
}
