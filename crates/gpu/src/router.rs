//! P6: the MoE router and the expert offset table. Routing stays on the
//! device because the decode step is one captured CUDA graph: a selected
//! expert id cannot be a host-side launch argument, so the router's ids live
//! in device memory and `expert_table` turns them into the per-slot row0
//! offsets the expert matmul launches read from a device buffer.
//!
//! Numeric contract — the CPU engine's routing, mirrored op for op
//! (`model::moe::route_inner`): the max of the token's 64 logits by a serial
//! ascending `f32::max` fold, `exp` in f32, the sum of exps accumulated in
//! f64 in ascending expert order, each prob divided in f32 by `sum as f32`;
//! top-6 by descending probability with ties toward the smaller expert id;
//! weights are the chosen experts' probs times `scale` (the file's
//! `expert_weights_scale`; 1.0 in this model, an exact multiply no-op). One
//! thread computes one token serially, so the summation and selection orders
//! are fixed by construction. Logits arrive in the f32 gemv output layout
//! `x[e * m + t]` (expert e of token t) — the router gemv's `y` as produced.

use crate::GpuError;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Experts the router softmaxes over (this model's `expert_count`); the
/// kernels' block structure, a launch-time constant.
pub const N_EXPERT: usize = 64;

/// Experts each token routes to (this model's `expert_used_count`).
pub const N_USED: usize = 6;

// --------------------------------------------------------------- cores

/// The softmax over token `t`'s 64 router logits: `x[e * m + t]` in, the
/// prob of expert e at `out[e]`, in the module doc's numeric order (serial
/// ascending max fold, f32 `exp`, f64 sum accumulated ascending, f32 divide
/// by `sum as f32`).
///
/// Caller contract: `x.len() >= 64 * m`, `t < m`, and the token's 64 logits
/// finite.
#[inline(always)]
pub fn softmax64(x: &[f32], m: u32, t: usize) -> [f32; 64] {
    let m = m as usize;
    let mut v = [0.0f32; 64];
    let mut mx = f32::NEG_INFINITY;
    let mut e = 0usize;
    while e < 64 {
        // SAFETY: e < 64 and t < m, so e*m + t < 64*m <= x.len() by the
        // caller contract.
        let le = unsafe { *x.get_unchecked(e * m + t) };
        v[e] = le;
        mx = mx.max(le);
        e += 1;
    }
    let mut sum = 0.0f64;
    let mut e = 0usize;
    while e < 64 {
        let ex = (v[e] - mx).exp();
        v[e] = ex;
        sum += f64::from(ex);
        e += 1;
    }
    let inv = sum as f32;
    let mut e = 0usize;
    while e < 64 {
        v[e] /= inv;
        e += 1;
    }
    v
}

/// The top-6 of one token's 64 softmax probs: descending probability, ties
/// toward the smaller expert id — the order the CPU engine's argsort states
/// (`p[b].cmp(p[a]).then(a.cmp(b))`, a total order over distinct ids).
/// Strict `>` over an ascending scan is that same order: the first of equal
/// probs wins, and taken ids are masked out. Returns the 6 expert ids and
/// their weights `probs[id] * scale`.
#[inline(always)]
pub fn top6(probs: &[f32; 64], scale: f32) -> ([u32; 6], [f32; 6]) {
    let mut ids = [0u32; 6];
    let mut ws = [0.0f32; 6];
    let mut taken = 0u64;
    let mut s = 0usize;
    while s < 6 {
        let mut best = 0u32;
        let mut best_p = f32::NEG_INFINITY;
        let mut e = 0u32;
        while e < 64 {
            if (taken >> e) & 1 == 0 && probs[e as usize] > best_p {
                best = e;
                best_p = probs[e as usize];
            }
            e += 1;
        }
        taken |= 1u64 << best;
        ids[s] = best;
        ws[s] = probs[best as usize] * scale;
        s += 1;
    }
    (ids, ws)
}

/// Expert `id`'s first row in a flat stacked expert tensor with `rows` rows
/// per expert: the row0 a gemv launch's row addressing starts from. Exact in
/// u32 for this model's stacks (63 * 2048 is the largest product).
#[inline(always)]
pub fn expert_row0(id: u32, rows: u32) -> u32 {
    id * rows
}

// -------------------------------------------------------------- kernels

#[cuda_module]
mod router_kernels {
    use super::*;

    /// The router: `m` (1..=8) tokens' logits in the layout `x[e*m + t]` ->
    /// probs in the same layout, ids `ids[s*m + t]` and weights
    /// `weights[s*m + t]` (slot s = the token's rank by descending
    /// probability). One 32-thread block, thread `t` owning token `t` whole;
    /// every store is this thread's own token's slot.
    #[kernel]
    #[launch_bounds(32)]
    #[launch_contract(
        domain = 1,
        block = (32, 1, 1),
        requires = (
            x.len() >= 64 * m,
            probs.len() >= 64 * m,
            ids.len() >= 6 * m,
            weights.len() >= 6 * m
        )
    )]
    pub fn router_topk(
        x: &[f32],
        m: u32,
        scale: f32,
        mut probs: DisjointSlice<f32>,
        mut ids: DisjointSlice<u32>,
        mut weights: DisjointSlice<f32>,
    ) {
        let t = thread::index_1d().get();
        if t >= m as usize {
            return;
        }
        let mi = m as usize;
        let p = softmax64(x, m, t);
        let (top, w) = top6(&p, scale);
        let mut e = 0usize;
        while e < 64 {
            // SAFETY: e < 64 and t < m, so e*m + t < 64*m <= probs.len() by
            // the launch contract.
            unsafe {
                *probs.get_unchecked_mut(e * mi + t) = p[e];
            }
            e += 1;
        }
        let mut s = 0usize;
        while s < 6 {
            // SAFETY: s < 6 and t < m, so s*m + t < 6*m <= ids.len() and
            // weights.len() by the launch contract.
            unsafe {
                *ids.get_unchecked_mut(s * mi + t) = top[s];
                *weights.get_unchecked_mut(s * mi + t) = w[s];
            }
            s += 1;
        }
    }

    /// The decode (m = 1) expert offset table: from the router's 6 expert
    /// ids, the row0 of each slot's expert in the three stacked expert
    /// tensors — `row0_gate[s] = row0_up[s] = ids[s] * rows_gu` (the gate
    /// and up stacks, 1408 rows of K = 2048 per expert) and
    /// `row0_down[s] = ids[s] * rows_dn` (the down stack, 2048 rows of
    /// K = 1408). The ids are read from device memory, so the table tracks
    /// the router inside a captured graph; `rows_gu`/`rows_dn` are
    /// load-time constants and may stay launch scalars.
    #[kernel]
    #[launch_bounds(32)]
    #[launch_contract(
        domain = 1,
        block = (32, 1, 1),
        requires = (
            ids.len() >= 6,
            row0_gate.len() >= 6,
            row0_up.len() >= 6,
            row0_down.len() >= 6
        )
    )]
    pub fn expert_table(
        ids: &[u32],
        rows_gu: u32,
        rows_dn: u32,
        mut row0_gate: DisjointSlice<u32>,
        mut row0_up: DisjointSlice<u32>,
        mut row0_down: DisjointSlice<u32>,
    ) {
        let s = thread::index_1d().get();
        if s >= 6 {
            return;
        }
        // SAFETY: s < 6 <= ids.len() by the launch contract.
        let e = unsafe { *ids.get_unchecked(s) };
        // SAFETY: s < 6 <= every output's length by the launch contract.
        unsafe {
            *row0_gate.get_unchecked_mut(s) = expert_row0(e, rows_gu);
            *row0_up.get_unchecked_mut(s) = expert_row0(e, rows_gu);
            *row0_down.get_unchecked_mut(s) = expert_row0(e, rows_dn);
        }
    }
}

/// The loaded P6 device module: `router_topk` and `expert_table`. Owns no
/// context and no stream — every enqueue takes the engine stream
/// (`Gpu::stream()`), so launches order with the rest of the step and are
/// capturable.
pub struct RouterKernels {
    module: router_kernels::LoadedModule,
}

impl RouterKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<RouterKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; every launcher checks its launch contract.
        let module = unsafe { router_kernels::load(ctx)? };
        Ok(RouterKernels { module })
    }

    /// Enqueue the router for `m` (1..=8) tokens. `x` holds `64 * m` logits
    /// in the f32 gemv layout `x[e*m + t]`; `probs` takes `64 * m` f32 in
    /// the same layout; `ids` takes `6 * m` u32 and `weights` `6 * m` f32,
    /// both slot-major (`ids[s*m + t]`, slot s = rank by descending
    /// probability, ties to the smaller id). `scale` is the file's
    /// `expert_weights_scale` (1.0 in this model). Asynchronous,
    /// allocation-free, capturable.
    #[allow(clippy::too_many_arguments)]
    pub fn enqueue_router_topk(
        &self,
        stream: &CudaStream,
        x: &DeviceBuffer<f32>,
        m: usize,
        scale: f32,
        probs: &mut DeviceBuffer<f32>,
        ids: &mut DeviceBuffer<u32>,
        weights: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        if !(1..=8).contains(&m) {
            return Err(format!("enqueue_router_topk: need 1 <= m <= 8, got m={m}").into());
        }
        if x.len() < N_EXPERT * m {
            return Err(format!(
                "enqueue_router_topk: x.len() {} < 64*m = {}",
                x.len(),
                N_EXPERT * m
            )
            .into());
        }
        if probs.len() < N_EXPERT * m {
            return Err(format!(
                "enqueue_router_topk: probs.len() {} < 64*m = {}",
                probs.len(),
                N_EXPERT * m
            )
            .into());
        }
        if ids.len() < N_USED * m || weights.len() < N_USED * m {
            return Err(format!(
                "enqueue_router_topk: ids.len() {} / weights.len() {} < 6*m = {}",
                ids.len(),
                weights.len(),
                N_USED * m
            )
            .into());
        }
        let prep = self
            .module
            .prepare_router_topk(LaunchConfig1D::new(1, 32, 0))?;
        self.module
            .router_topk(stream, &prep, x, m as u32, scale, probs, ids, weights)?;
        Ok(())
    }

    /// Enqueue the decode (m = 1) expert offset table: from `ids` (the
    /// router's 6 per-slot expert ids, device-resident) write `row0_gate`
    /// and `row0_up` (`ids[s] * rows_gu`) and `row0_down`
    /// (`ids[s] * rows_dn`), 6 u32 each. `rows_gu` is the gate/up stacks'
    /// rows per expert (1408, K = 2048), `rows_dn` the down stack's (2048,
    /// K = 1408). Asynchronous, allocation-free, capturable.
    #[allow(clippy::too_many_arguments)]
    pub fn enqueue_expert_table(
        &self,
        stream: &CudaStream,
        ids: &DeviceBuffer<u32>,
        rows_gu: usize,
        rows_dn: usize,
        row0_gate: &mut DeviceBuffer<u32>,
        row0_up: &mut DeviceBuffer<u32>,
        row0_down: &mut DeviceBuffer<u32>,
    ) -> Result<(), GpuError> {
        if rows_gu == 0
            || rows_gu > u32::MAX as usize
            || rows_dn == 0
            || rows_dn > u32::MAX as usize
        {
            return Err(format!(
                "enqueue_expert_table: rows per expert must be 1..=u32::MAX, got gu={rows_gu} dn={rows_dn}"
            )
            .into());
        }
        if ids.len() < N_USED {
            return Err(format!(
                "enqueue_expert_table: ids.len() {} < 6 (decode m=1 shape)",
                ids.len()
            )
            .into());
        }
        if row0_gate.len() < N_USED || row0_up.len() < N_USED || row0_down.len() < N_USED {
            return Err(format!(
                "enqueue_expert_table: row0 buffers need 6 u32 each, got {} / {} / {}",
                row0_gate.len(),
                row0_up.len(),
                row0_down.len()
            )
            .into());
        }
        let prep = self
            .module
            .prepare_expert_table(LaunchConfig1D::new(1, 32, 0))?;
        self.module.expert_table(
            stream,
            &prep,
            ids,
            rows_gu as u32,
            rows_dn as u32,
            row0_gate,
            row0_up,
            row0_down,
        )?;
        Ok(())
    }
}
