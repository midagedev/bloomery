//! Hy4-flash's engram delta: the looked-up rows, projected into a key per
//! stream and one value, add a gated per-stream delta to the four residual
//! streams before the layer's attention (llama.cpp `build_engram`, deltas 1
//! to 4 and the whole-token mask). The gate itself is
//! [`engram_gate::row_3072`], the V4.1 rule at this model's width.
//!
//! The chain, per engram layer, in the order the launches run:
//! 1. [`EngramDeltaKernels::enqueue_rows_norm`] (delta 1): each of the
//!    `cols` looked-up rows of `width` values, RMS-normed on its own,
//!    scaled by its column's `lookup_norm` row and by `valid`.
//! 2. `engram_kv`, the dense Q8_0 gemv (`[cols·width → 5·n]`), then
//!    [`EngramDeltaKernels::enqueue_kv_bias`] (delta 2): `kv += engram_kv_b`
//!    over each token's `(hc + 1)·n` values — the `hc` keys, then the value.
//! 3. `gate.enqueue_key_norm` and `gate.enqueue_gate` (the gate): `kv`
//!    alone feeds the first, so steps 1 to 3a depend on the token ids only
//!    and can run before the previous layer finishes; the second reads the
//!    streams.
//! 4. [`EngramDeltaKernels::enqueue_delta`] (deltas 3 and 4, the mask): the
//!    streams plus the masked delta.
//!
//! Layouts, token-major: the rows `[cols·width, m]`, `kv` `[(hc+1)·n, m]` (keys
//! then value), the streams `x` and `out` `[n, hc, m]`, `gate` `[hc, m]`,
//! `valid` `[cols, m]`.
//!
//! The numeric rule each entry holds, op for op the oracle's graph with every
//! f32 product and sum an explicit round-to-nearest intrinsic (the compiler
//! contracts a plain `a·b + c` into a fused multiply-add):
//! - `engram_rows_norm`: per row, a lane's `fma(v, v, acc)` over
//!   `i = lane, lane + 32, …` from 0, the warp butterfly, `scale = 1 /
//!   sqrt(sum / width + eps)` (`elem::rms_scale_checked`), then
//!   `((v·scale)·lookup[col·width + i])·valid`.
//! - `engram_kv_bias`: `kv[i] = kv[i] + bias[i mod width]`.
//! - `engram_delta_apply`: per (token, stream) block, the value's scale as
//!   `elem::rms_norm`'s tree over the row (`engram_gate::block_scale_n`),
//!   `signal = (v·scale)·gamma`, `delta = (signal·(gate + hc_scale)) ·
//!   any`, `out = x + delta`, with `any` = 1 when the token's `valid` values
//!   sum above 0 and 0 otherwise.

use crate::elem::{RMS_THREADS, RMS_WARPS, rms_scale_checked};
pub use crate::engram_gate::{GateOnlyArgs, KeyNormArgs};
use crate::engram_gate::{block_scale_n, group, row_3072, store_group, sum_sq};
use crate::{FaultSink, GpuError, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::float::{add_rn_f32, fma_rn_f32, mul_rn_f32};
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread, warp,
};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Residual streams (`hc_mult`).
pub const HC: usize = 4;
/// Values in one stream row: Hy4-flash's `n_embd`.
pub const ROW: usize = row_3072::ROW;
/// The oracle's `HY_V4_FLASH_ENGRAM_EPS`: the epsilon of every norm in the
/// engram path.
pub const EPS: f32 = 1.1920929e-7;
/// Warps of the rows kernel's block: one looked-up row a warp.
const ROWS_WARPS: usize = 4;
const ROWS_BLOCK: u32 = 128;
const _: () = assert!(ROWS_BLOCK as usize == 32 * ROWS_WARPS);
/// The delta kernel's block width, `elem::RMS_THREADS`.
const BLOCK: u32 = 256;
const _: () = assert!(BLOCK as usize == RMS_THREADS && HC == 4);
const _: () = assert!(ROW == 3 * 4 * 256);

// ---------------------------------------------------------------- kernels

#[cuda_module]
mod engram_delta_kernels {
    use super::*;

    /// Delta 1: warp `w` of block `b` normalizes looked-up row `r = 4·b + w`
    /// (token `r / cols`, column `r % cols`) — `rows[r·width ..]` — and
    /// writes it, scaled by `lookup[(r % cols)·width ..]` and `valid[r]`, to
    /// `out[r·width ..]`. The row guard is warp-uniform, so no warp
    /// collective is skipped.
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
            rows.len() >= m * cols * width,
            lookup.len() >= cols * width,
            valid.len() >= m * cols,
            out.len() >= m * cols * width
        )
    )]
    pub fn engram_rows_norm(
        rows: &[f32],
        lookup: &[f32],
        valid: &[f32],
        eps: f32,
        width: u32,
        cols: u32,
        m: u32,
        fault: FaultSink,
        mut out: DisjointSlice<f32>,
    ) {
        let lane = warp::lane_id() as usize;
        let r = thread::blockIdx_x() as usize * ROWS_WARPS + thread::threadIdx_x() as usize / 32;
        let (width, cols) = (width as usize, cols as usize);
        if r >= m as usize * cols {
            return;
        }
        let base = r * width;
        let mut acc = 0.0f32;
        let mut i = lane;
        while i < width {
            // SAFETY: r < m·cols and i < width, so base + i < m·cols·width <=
            // rows.len() by the launch contract.
            let v = unsafe { *rows.get_unchecked(base + i) };
            acc = fma_rn_f32(v, v, acc);
            i += 32;
        }
        let sum = warp::reduce_sum_f32(acc);
        let scale = rms_scale_checked(sum, width as u32, eps, fault, lane == 0);
        let gain_base = (r % cols) * width;
        // SAFETY: r < m·cols <= valid.len() by the launch contract.
        let keep = unsafe { *valid.get_unchecked(r) };
        let mut i = lane;
        while i < width {
            // SAFETY: base + i < m·cols·width <= rows.len() and out.len(),
            // col < cols so gain_base + i < cols·width <= lookup.len(), all by
            // the launch contract; the position is this lane's alone.
            unsafe {
                let v = *rows.get_unchecked(base + i);
                let g = *lookup.get_unchecked(gain_base + i);
                *out.get_unchecked_mut(base + i) =
                    mul_rn_f32(mul_rn_f32(mul_rn_f32(v, scale), g), keep);
            }
            i += 32;
        }
    }

    /// Delta 2's bias: `kv[i] = kv[i] + bias[i mod width]` over `total`
    /// values, in place.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (kv.len() >= total, bias.len() >= width)
    )]
    pub fn engram_kv_bias(bias: &[f32], width: u32, total: u32, mut kv: DisjointSlice<f32>) {
        let i = thread::index_1d().get();
        if i >= total as usize {
            return;
        }
        // SAFETY: i < total <= kv.len() and i % width < width <= bias.len() by
        // the launch contract; thread i owns entry i.
        unsafe {
            let b = *bias.get_unchecked(i % width as usize);
            let v = kv.get_unchecked_mut(i);
            *v = add_rn_f32(*v, b);
        }
    }

    /// Deltas 3 and 4 and the mask: block `b` is stream `s = b % 4` of token
    /// `t = b / 4`. It scales token `t`'s value row (`kv[t·5·ROW + 4·ROW ..]`)
    /// by its norm and `gamma`, the result by `gate[b] + hc_scale[s·ROW ..]`
    /// and the token's `any`, and writes `out[b·ROW + d] = x[b·ROW + d] +
    /// delta`. The block guard is uniform, so no barrier is skipped.
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
            x.len() >= m * 4 * 3072,
            kv.len() >= m * 5 * 3072,
            gate.len() >= m * 4,
            gamma.len() >= 3072,
            hc_scale.len() >= 4 * 3072,
            valid.len() >= m * cols,
            out.len() >= m * 4 * 3072
        )
    )]
    pub fn engram_delta_apply(
        x: &[f32],
        kv: &[f32],
        gate: &[f32],
        gamma: &[f32],
        hc_scale: &[f32],
        valid: &[f32],
        eps: f32,
        cols: u32,
        m: u32,
        fault: FaultSink,
        mut out: DisjointSlice<f32>,
    ) {
        static mut WSUM: SharedArray<f32, RMS_WARPS> = SharedArray::UNINIT;

        let b = thread::blockIdx_x() as usize;
        if b >= HC * m as usize {
            return;
        }
        let (t, s) = (b / HC, b % HC);
        let tid = thread::threadIdx_x() as usize;
        let (xb, vb, hb) = (b * ROW, (t * (HC + 1) + HC) * ROW, s * ROW);
        // SAFETY: b < 4·m, so xb + ROW <= m·4·ROW bounds x; t < m, so vb +
        // ROW <= m·5·ROW bounds kv; s < 4, so hb + ROW <= 4·ROW bounds
        // hc_scale; gamma holds ROW values — all by the launch contract.
        // tid < RMS_THREADS; every group index is below 3 = ROW / (4·256).
        let (v, g, h, xs) = unsafe {
            (
                [
                    group(kv, vb, tid, 0),
                    group(kv, vb, tid, 1),
                    group(kv, vb, tid, 2),
                ],
                [
                    group(gamma, 0, tid, 0),
                    group(gamma, 0, tid, 1),
                    group(gamma, 0, tid, 2),
                ],
                [
                    group(hc_scale, hb, tid, 0),
                    group(hc_scale, hb, tid, 1),
                    group(hc_scale, hb, tid, 2),
                ],
                [
                    group(x, xb, tid, 0),
                    group(x, xb, tid, 1),
                    group(x, xb, tid, 2),
                ],
            )
        };
        let part = sum_sq(sum_sq(sum_sq(0.0, v[0]), v[1]), v[2]);
        // SAFETY: WSUM is this block's own shared allocation, reached only
        // through `block_scale_n`; every thread arrives.
        let scale = unsafe {
            block_scale_n(
                part,
                tid,
                SharedArray::as_raw_mut_ptr(&raw mut WSUM),
                ROW as u32,
                eps,
                fault,
            )
        };
        // SAFETY: b < 4·m, so b < gate.len() by the launch contract.
        let gate_b = unsafe { *gate.get_unchecked(b) };
        let mut seen = 0.0f32;
        let mut c = 0usize;
        while c < cols as usize {
            // SAFETY: t < m and c < cols, so t·cols + c < m·cols <= valid.len()
            // by the launch contract.
            seen = add_rn_f32(seen, unsafe { *valid.get_unchecked(t * cols as usize + c) });
            c += 1;
        }
        let any = if seen > 0.0 { 1.0f32 } else { 0.0 };
        // SAFETY: b < 4·m, so xb + ROW <= m·4·ROW <= out.len() by the launch
        // contract; tid < RMS_THREADS, every group index is below 3, and row
        // b's positions of this thread are written by this thread alone.
        unsafe {
            store_group(
                &mut out,
                xb,
                tid,
                0,
                apply(xs[0], v[0], g[0], h[0], (scale, gate_b, any)),
            );
            store_group(
                &mut out,
                xb,
                tid,
                1,
                apply(xs[1], v[1], g[1], h[1], (scale, gate_b, any)),
            );
            store_group(
                &mut out,
                xb,
                tid,
                2,
                apply(xs[2], v[2], g[2], h[2], (scale, gate_b, any)),
            );
        }
    }
}

/// One group of the update: `x + (((v·scale)·gamma)·(gate + hc_scale))·any`
/// per value, each product and sum rounded on its own.
#[inline(always)]
fn apply(
    x: [f32; 4],
    v: [f32; 4],
    gamma: [f32; 4],
    hc_scale: [f32; 4],
    (scale, gate, any): (f32, f32, f32),
) -> [f32; 4] {
    [
        lane(x[0], v[0], gamma[0], hc_scale[0], (scale, gate, any)),
        lane(x[1], v[1], gamma[1], hc_scale[1], (scale, gate, any)),
        lane(x[2], v[2], gamma[2], hc_scale[2], (scale, gate, any)),
        lane(x[3], v[3], gamma[3], hc_scale[3], (scale, gate, any)),
    ]
}

/// One value of the update, [`apply`]'s rule.
#[inline(always)]
fn lane(x: f32, v: f32, gamma: f32, hc_scale: f32, (scale, gate, any): (f32, f32, f32)) -> f32 {
    let signal = mul_rn_f32(mul_rn_f32(v, scale), gamma);
    let delta = mul_rn_f32(signal, add_rn_f32(gate, hc_scale));
    add_rn_f32(x, mul_rn_f32(delta, any))
}

// -------------------------------------------------------------- launchers

/// [`EngramDeltaKernels::enqueue_rows_norm`]'s arguments: `m` tokens' looked-up
/// `rows` (`cols` rows of `width`, token-major), the per-column norm weights
/// `lookup` (`cols·width`), `valid` (`m·cols` 0/1 values), `eps`, and the
/// normed `out` in `rows`' layout.
pub struct RowsArgs<'a> {
    pub rows: &'a DeviceBuffer<f32>,
    pub lookup: &'a DeviceBuffer<f32>,
    pub valid: &'a DeviceBuffer<f32>,
    pub eps: f32,
    pub width: usize,
    pub cols: usize,
    pub m: usize,
    pub out: &'a mut DeviceBuffer<f32>,
    /// Where a row whose squares leave the norm no scale is raised.
    pub fault: FaultSink,
}

/// [`EngramDeltaKernels::enqueue_delta`]'s arguments: the streams `x`
/// (`m·4·ROW`), `kv` (`m·5·ROW`, biased), the `gate` the gate entry wrote
/// (`m·4`), `gamma` (`engram_pre_gamma`, `ROW`), `hc_scale`
/// (`engram_hc_scale`, `4·ROW`), `valid` (`m·cols`), and `out` in `x`'s
/// layout (another buffer than `x`).
pub struct DeltaArgs<'a> {
    pub x: &'a DeviceBuffer<f32>,
    pub kv: &'a DeviceBuffer<f32>,
    pub gate: &'a DeviceBuffer<f32>,
    pub gamma: &'a DeviceBuffer<f32>,
    pub hc_scale: &'a DeviceBuffer<f32>,
    pub valid: &'a DeviceBuffer<f32>,
    pub eps: f32,
    pub cols: usize,
    pub m: usize,
    pub out: &'a mut DeviceBuffer<f32>,
    /// Where a value row whose squares leave the norm no scale is raised.
    pub fault: FaultSink,
}

/// The loaded engram-delta module and the gate instance it composes with.
/// Owns no stream: each enqueue takes the engine stream, so its launches
/// order with the rest of the step and capture.
pub struct EngramDeltaKernels {
    module: engram_delta_kernels::LoadedModule,
    /// The key and gate kernels at [`ROW`].
    pub gate: row_3072::Kernels,
}

impl EngramDeltaKernels {
    /// Load this file's device bundle and the gate instance's into `ctx`.
    /// Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<EngramDeltaKernels, GpuError> {
        // SAFETY: this crate owns the embedded device bundle produced for the
        // module above; the launchers check its launch contracts.
        let module = unsafe { crate::shared_module!(engram_delta_kernels, ctx)? };
        Ok(EngramDeltaKernels {
            module,
            gate: row_3072::Kernels::load(ctx)?,
        })
    }

    /// Enqueue delta 1 over `a.m` tokens. Asynchronous, allocation-free,
    /// capturable.
    pub fn enqueue_rows_norm(&self, stream: &CudaStream, a: RowsArgs<'_>) -> Result<(), GpuError> {
        const WHAT: &str = "engram_delta::enqueue_rows_norm";
        if a.width == 0 || a.cols == 0 || a.m == 0 {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!(
                    "need width, cols and m >= 1, got width={} cols={} m={}",
                    a.width, a.cols, a.m
                ),
            });
        }
        let rows = a.m * a.cols;
        need(WHAT, "rows", a.rows.len(), rows * a.width)?;
        need(WHAT, "lookup", a.lookup.len(), a.cols * a.width)?;
        need(WHAT, "valid", a.valid.len(), rows)?;
        need(WHAT, "out", a.out.len(), rows * a.width)?;
        let blocks = launch_u32(WHAT, "rows / 4", rows.div_ceil(ROWS_WARPS))?;
        let width = launch_u32(WHAT, "width", a.width)?;
        let cols = launch_u32(WHAT, "cols", a.cols)?;
        let m = launch_u32(WHAT, "m", a.m)?;
        let prep = self
            .module
            .prepare_engram_rows_norm(LaunchConfig1D::new(blocks, ROWS_BLOCK, 0))?;
        self.module.engram_rows_norm(
            stream, &prep, a.rows, a.lookup, a.valid, a.eps, width, cols, m, a.fault, a.out,
        )?;
        Ok(())
    }

    /// Enqueue delta 2's bias over `m` tokens' `kv` (`(HC + 1)·ROW` values
    /// each), in place. Asynchronous, allocation-free, capturable.
    pub fn enqueue_kv_bias(
        &self,
        stream: &CudaStream,
        bias: &DeviceBuffer<f32>,
        m: usize,
        kv: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "engram_delta::enqueue_kv_bias";
        let width = (HC + 1) * ROW;
        if m == 0 {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: "need m >= 1".into(),
            });
        }
        need(WHAT, "bias", bias.len(), width)?;
        need(WHAT, "kv", kv.len(), m * width)?;
        let total = launch_u32(WHAT, "m·5·ROW", m * width)?;
        let width = launch_u32(WHAT, "5·ROW", width)?;
        let prep = self.module.prepare_engram_kv_bias(LaunchConfig1D::new(
            total.div_ceil(BLOCK),
            BLOCK,
            0,
        ))?;
        self.module
            .engram_kv_bias(stream, &prep, bias, width, total, kv)?;
        Ok(())
    }

    /// Enqueue deltas 3 and 4 and the mask over `a.m` tokens. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_delta(&self, stream: &CudaStream, a: DeltaArgs<'_>) -> Result<(), GpuError> {
        const WHAT: &str = "engram_delta::enqueue_delta";
        if a.m == 0 || a.cols == 0 {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!("need m and cols >= 1, got m={} cols={}", a.m, a.cols),
            });
        }
        let blocks = HC * a.m;
        need(WHAT, "x", a.x.len(), blocks * ROW)?;
        need(WHAT, "kv", a.kv.len(), a.m * (HC + 1) * ROW)?;
        need(WHAT, "gate", a.gate.len(), blocks)?;
        need(WHAT, "gamma", a.gamma.len(), ROW)?;
        need(WHAT, "hc_scale", a.hc_scale.len(), HC * ROW)?;
        need(WHAT, "valid", a.valid.len(), a.m * a.cols)?;
        need(WHAT, "out", a.out.len(), blocks * ROW)?;
        let grid = launch_u32(WHAT, "4·m", blocks)?;
        let cols = launch_u32(WHAT, "cols", a.cols)?;
        let m = launch_u32(WHAT, "m", a.m)?;
        let prep = self
            .module
            .prepare_engram_delta_apply(LaunchConfig1D::new(grid, BLOCK, 0))?;
        self.module.engram_delta_apply(
            stream, &prep, a.x, a.kv, a.gate, a.gamma, a.hc_scale, a.valid, a.eps, cols, m,
            a.fault, a.out,
        )?;
        Ok(())
    }
}

/// Refuse a buffer `name` of `len` elements shorter than `want`.
fn need(what: &'static str, name: &str, len: usize, want: usize) -> Result<(), GpuError> {
    if len < want {
        return Err(GpuError::Shape {
            what,
            detail: format!("{name}.len() {len} < {want}"),
        });
    }
    Ok(())
}
