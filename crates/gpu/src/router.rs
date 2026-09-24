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
use crate::launch_u32;
use crate::route_core::take;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread, warp,
};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Experts the router softmaxes over (this model's `expert_count`); the
/// kernels' block structure, a launch-time constant.
pub const N_EXPERT: usize = 64;

/// Experts each token routes to (this model's `expert_used_count`).
pub const N_USED: usize = 6;

/// Threads both router kernels launch with, and the width their device code
/// is compiled for. One thread owns one token whole — the f64 sum of the 64
/// exps is contracted to ascending expert order, so the token cannot be
/// split across lanes — and the router drives `m` in 1..=8, so one warp
/// covers every shape. `gate_p6`'s `router_shape` asserts the compiled
/// `.reqntid` against this.
pub const ROUTER_THREADS: usize = 32;
/// [`ROUTER_THREADS`] as the `u32` block width a launch takes.
const ROUTER_THREADS_U32: u32 = ROUTER_THREADS as u32;
const _: () = assert!(ROUTER_THREADS_U32 as usize == ROUTER_THREADS);

// --------------------------------------------------------------- cores

/// Expert `id`'s first row in a flat stacked expert tensor with `rows` rows
/// per expert: the row0 a gemv launch's row addressing starts from. Exact in
/// u32 for this model's stacks (63 * 2048 is the largest product).
#[inline(always)]
pub(crate) fn expert_row0(id: u32, rows: u32) -> u32 {
    id * rows
}

// -------------------------------------------------------------- kernels

#[cuda_module]
mod router_kernels {
    use super::*;

    /// The router: `m` (1..=8) tokens' logits in the layout `x[e*m + t]` ->
    /// probs in the same layout, ids `ids[s*m + t]` and weights
    /// `weights[s*m + t]` (slot s = the token's rank by descending
    /// probability). One [`ROUTER_THREADS`] block, thread `t` owning token
    /// `t` whole; every store is this thread's own token's slot.
    ///
    /// The numeric order is the module doc's, op for op. It is carried in
    /// four passes over the token's own slots of `probs` rather than in a
    /// 64-element per-thread array: the selection reads `probs[best]` at a
    /// data-dependent index, which puts such an array in local memory
    /// whatever the loops unroll to, and every read of it then costs a
    /// round trip. Passes 2 and 3 park the exps and then the quotients in
    /// the slots they have to be written to anyway, and the selection reads
    /// them back — the same thread's own stores, in program order, so no
    /// barrier and no collective is involved. `gate_p6`'s `router_shape`
    /// asserts the compiled entry carries no depot.
    ///
    /// The width is one thread per token by contract, not by convenience:
    /// the f64 sum of the 64 exps is specified in ascending expert order,
    /// and a tree over lanes is a different sum. `m <= 8`, so one warp
    /// covers every shape the router is launched at.
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
        // The token's 64 probabilities for the decode path below: lane 0
        // writes them in the contract's order, the whole warp reads them past
        // the barrier. Shared, not a per-thread array — an array a
        // data-dependent id selects is served from a local depot whatever the
        // loops unroll to, and that round trip is what this kernel's shape
        // exists to keep out.
        static mut P: SharedArray<f32, N_EXPERT> = SharedArray::UNINIT;

        let t = thread::index_1d().get();
        if m == 1 {
            // SAFETY: P is this block's own shared allocation; the raw form
            // is the only way to reach it without a reference to a
            // `static mut`. Every index below is an expert id < N_EXPERT, and
            // lane 0's writes precede every other lane's reads by
            // `sync_threads`.
            let p = unsafe { SharedArray::as_raw_mut_ptr(&raw mut P) };
            if t == 0 {
                // Passes 1-3, the module doc's orders op for op — the serial
                // ascending `f32::max` fold, `exp` in f32, the sum in f64 in
                // ascending expert order, the f32 divide by `sum as f32`. The
                // quotients land in shared and in `probs`, which its real
                // consumers read.
                let mut mx = f32::NEG_INFINITY;
                let mut e = 0usize;
                while e < N_EXPERT {
                    // SAFETY: e < 64 = 64*m <= x.len() by the launch contract.
                    let le = unsafe { *x.get_unchecked(e) };
                    mx = mx.max(le);
                    e += 1;
                }
                let mut sum = 0.0f64;
                let mut e = 0usize;
                while e < N_EXPERT {
                    // SAFETY: e < 64 <= x.len(); p.add(e) is inside P.
                    let ex = unsafe {
                        let ex = (*x.get_unchecked(e) - mx).exp();
                        *p.add(e) = ex;
                        ex
                    };
                    sum += f64::from(ex);
                    e += 1;
                }
                let inv = sum as f32;
                let mut e = 0usize;
                while e < N_EXPERT {
                    // SAFETY: e < 64 — inside P, and inside probs, whose
                    // length is >= 64*m = 64 by the launch contract.
                    unsafe {
                        let v = *p.add(e) / inv;
                        *p.add(e) = v;
                        *probs.get_unchecked_mut(e) = v;
                    }
                    e += 1;
                }
            }
            thread::sync_threads();
            // Pass 4 over the warp. Lane L owns experts L and L+32, read once
            // — the six selections differ only by the `taken` mask.
            // `take::<false>` is the total order (probability descending, id
            // ascending) that the serial scan's strict `>` over ascending
            // experts realizes, so no regrouping can move a tie, and its
            // `(-inf, 0)` seed is the serial scan's seed, so a token no
            // comparison ever wins answers expert 0 on both paths. A NaN
            // probability never enters the reduction: `take::<false>` takes a
            // candidate only on `>` or `==`, both false for NaN, so no lane's
            // running best is ever NaN and the merge only ever sees numbers.
            let lane = warp::lane_id() as usize;
            let (e0, e1) = (lane, lane + 32);
            // SAFETY: e0 < 32 and e1 < 64 — inside P, written above and
            // visible past the barrier.
            let (p0, p1) = unsafe { (*p.add(e0), *p.add(e1)) };
            let mut taken = 0u64;
            let mut s = 0usize;
            while s < N_USED {
                let mut bv = f32::NEG_INFINITY;
                let mut bi = 0u32;
                if (taken >> e0) & 1 == 0 && take::<false>(p0, e0 as u32, bv, bi) {
                    bv = p0;
                    bi = e0 as u32;
                }
                if (taken >> e1) & 1 == 0 && take::<false>(p1, e1 as u32, bv, bi) {
                    bv = p1;
                    bi = e1 as u32;
                }
                let mut off = 16u32;
                while off > 0 {
                    let (ov, oi) = (warp::shuffle_xor_f32(bv, off), warp::shuffle_xor(bi, off));
                    if take::<false>(ov, oi, bv, bi) {
                        bv = ov;
                        bi = oi;
                    }
                    off >>= 1;
                }
                // Every lane leaves the butterfly with the same winner, so
                // every lane masks the same expert out for the next slot.
                taken |= 1u64 << bi;
                if t == 0 {
                    // SAFETY: bi < 64 — inside P; s < 6 <= ids.len() and
                    // weights.len() by the launch contract. The weight is
                    // read from the winner's slot rather than from the
                    // comparison, so a prob no comparison won is carried
                    // through unchanged.
                    unsafe {
                        let w = *p.add(bi as usize) * scale;
                        *ids.get_unchecked_mut(s) = bi;
                        *weights.get_unchecked_mut(s) = w;
                    }
                }
                s += 1;
            }
            return;
        }
        if t >= m as usize {
            return;
        }
        let mi = m as usize;

        // Pass 1: the largest of the token's 64 logits, by a serial
        // ascending `f32::max` fold.
        let mut mx = f32::NEG_INFINITY;
        let mut e = 0usize;
        while e < 64 {
            // SAFETY: e < 64 and t < m, so e*mi + t < 64*m <= x.len() by the
            // launch contract.
            let le = unsafe { *x.get_unchecked(e * mi + t) };
            mx = mx.max(le);
            e += 1;
        }

        // Pass 2: `exp` in f32 into the token's own prob slots, the sum
        // accumulated in f64 in ascending expert order.
        let mut sum = 0.0f64;
        let mut e = 0usize;
        while e < 64 {
            // SAFETY: e < 64 and t < m, so e*mi + t is inside both x.len()
            // and probs.len() (>= 64*m) by the launch contract.
            let ex = unsafe {
                let ex = (*x.get_unchecked(e * mi + t) - mx).exp();
                *probs.get_unchecked_mut(e * mi + t) = ex;
                ex
            };
            sum += f64::from(ex);
            e += 1;
        }

        // Pass 3: the f32 divide by `sum as f32`, in place.
        let inv = sum as f32;
        let mut e = 0usize;
        while e < 64 {
            // SAFETY: e < 64 and t < m, so e*mi + t < 64*m <= probs.len()
            // by the launch contract — this thread's own slot.
            unsafe {
                *probs.get_unchecked_mut(e * mi + t) /= inv;
            }
            e += 1;
        }

        // Pass 4: the top-6 by descending probability with ties toward the
        // smaller expert id. Strict `>` over an ascending scan is that
        // order — the first of equal probs wins — and taken ids are masked
        // out; the weight is the winner's prob times `scale`, read from the
        // slot rather than from the comparison so a prob no comparison ever
        // won (an all-NaN token) is carried through unchanged.
        let mut taken = 0u64;
        let mut s = 0usize;
        while s < 6 {
            let mut best = 0u32;
            let mut best_p = f32::NEG_INFINITY;
            let mut e = 0u32;
            while e < 64 {
                if (taken >> e) & 1 == 0 {
                    // SAFETY: e < 64 and t < m — this thread's own slot.
                    let pv = unsafe { *probs.get_unchecked_mut(e as usize * mi + t) };
                    if pv > best_p {
                        best = e;
                        best_p = pv;
                    }
                }
                e += 1;
            }
            taken |= 1u64 << best;
            // SAFETY: best < 64 and t < m — this thread's own prob slot.
            let w = unsafe { *probs.get_unchecked_mut(best as usize * mi + t) } * scale;
            // SAFETY: s < 6 and t < m, so s*mi + t < 6*m <= ids.len() and
            // weights.len() by the launch contract.
            unsafe {
                *ids.get_unchecked_mut(s * mi + t) = best;
                *weights.get_unchecked_mut(s * mi + t) = w;
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
    #[allow(
        clippy::too_many_arguments,
        reason = "host launcher; folding these into a *Args struct is the R8 round"
    )]
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
            return Err(GpuError::shape(
                "enqueue_router_topk",
                format!("need 1 <= m <= 8, got m={m}"),
            ));
        }
        if x.len() < N_EXPERT * m {
            return Err(GpuError::shape(
                "enqueue_router_topk",
                format!("x.len() {} < 64*m = {}", x.len(), N_EXPERT * m),
            ));
        }
        if probs.len() < N_EXPERT * m {
            return Err(GpuError::shape(
                "enqueue_router_topk",
                format!("probs.len() {} < 64*m = {}", probs.len(), N_EXPERT * m),
            ));
        }
        if ids.len() < N_USED * m || weights.len() < N_USED * m {
            return Err(GpuError::shape(
                "enqueue_router_topk",
                format!(
                    "ids.len() {} / weights.len() {} < 6*m = {}",
                    ids.len(),
                    weights.len(),
                    N_USED * m
                ),
            ));
        }
        let m = launch_u32("enqueue_router_topk", "m", m)?;
        let prep =
            self.module
                .prepare_router_topk(LaunchConfig1D::new(1, ROUTER_THREADS_U32, 0))?;
        self.module
            .router_topk(stream, &prep, x, m, scale, probs, ids, weights)?;
        Ok(())
    }

    /// Enqueue the decode (m = 1) expert offset table: from `ids` (the
    /// router's 6 per-slot expert ids, device-resident) write `row0_gate`
    /// and `row0_up` (`ids[s] * rows_gu`) and `row0_down`
    /// (`ids[s] * rows_dn`), 6 u32 each. `rows_gu` is the gate/up stacks'
    /// rows per expert (1408, K = 2048), `rows_dn` the down stack's (2048,
    /// K = 1408). Asynchronous, allocation-free, capturable.
    #[allow(
        clippy::too_many_arguments,
        reason = "host launcher; folding these into a *Args struct is the R8 round"
    )]
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
        let (rows_gu, rows_dn) = match (u32::try_from(rows_gu), u32::try_from(rows_dn)) {
            (Ok(gu), Ok(dn)) if gu > 0 && dn > 0 => (gu, dn),
            _ => {
                return Err(GpuError::shape(
                    "enqueue_expert_table",
                    format!("rows per expert must be 1..=u32::MAX, got gu={rows_gu} dn={rows_dn}"),
                ));
            }
        };
        if ids.len() < N_USED {
            return Err(GpuError::shape(
                "enqueue_expert_table",
                format!("ids.len() {} < 6 (decode m=1 shape)", ids.len()),
            ));
        }
        if row0_gate.len() < N_USED || row0_up.len() < N_USED || row0_down.len() < N_USED {
            return Err(GpuError::shape(
                "enqueue_expert_table",
                format!(
                    "row0 buffers need 6 u32 each, got {} / {} / {}",
                    row0_gate.len(),
                    row0_up.len(),
                    row0_down.len()
                ),
            ));
        }
        let prep =
            self.module
                .prepare_expert_table(LaunchConfig1D::new(1, ROUTER_THREADS_U32, 0))?;
        self.module.expert_table(
            stream, &prep, ids, rows_gu, rows_dn, row0_gate, row0_up, row0_down,
        )?;
        Ok(())
    }
}
