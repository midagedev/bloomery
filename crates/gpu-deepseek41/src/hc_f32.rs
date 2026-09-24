//! HC_PRE with F32 weights: the DSpark draft's hyper-connection head, whose
//! `hc_{attn,ffn}_fn` are F32 `[K, 24]` rather than the target's q3_K. The
//! layouts are [`crate::hc`]'s: a token's streams are `K = 4 * n_embd` f32
//! read flat, its result [`HC_MIX`] f32 (`pre[4]`, `post[4]`, `comb[16]`).
//!
//! The numeric rule `ds41_hc_pre_f32` holds, and the gate transcribes on the
//! host. One block per token, one launch; no value crosses blocks, so the
//! launch needs no scratch and no ticket.
//! 1. The activations stay f32: no quantization. The RMS scale multiplies the
//!    gemv result, as in `ds41_hc_pre`.
//! 2. Row `r` (warp `r % 8` owns rows `w`, `w + 8`, `w + 16`): `f32_gemv`'s
//!    order at one column — lane `L` sums `w[r*K + 32*it + L] * x[32*it + L]`
//!    by `mul_add` from 0, `it` ascending
//!    (`bloomery_gpu::q8f32::f32_lane_partial_1col`), then
//!    `warp::reduce_sum_f32` (lane `l` adds lane `l ^ s`, s = 16, 8, 4, 2, 1).
//! 3. The squares on warp 0: lane `L` sums `x[32*it + L]` squared by
//!    `v.mul_add(v, acc)` from 0, `it` ascending, then `warp::reduce_sum_f32`.
//! 4. `scale = 1 / sqrt(squares / K + rms_eps)`, `mix = raw * scale`.
//! 5. HC_PRE on warp 0 ([`crate::hc`]'s `hc_pre_lane`).

use crate::hc::{HC_MIX, hc_pre_lane};
use bloomery_gpu::q8f32::f32_lane_partial_1col;
use bloomery_gpu::{DeviceTensor, GpuError, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread, warp,
};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Threads of a `ds41_hc_pre_f32` block: eight warps, three rows each.
const HC_F32_THREADS: u32 = 256;
const HC_F32_WARPS: usize = HC_F32_THREADS as usize / 32;
const _: () = assert!(3 * HC_F32_WARPS == HC_MIX);

/// The sum of squares of one lane's share of a token's `k` values (step 3):
/// `x[32*it + lane]`, `it` ascending, one fused multiply-add each from 0.
///
/// SAFETY: `x.len() >= k` and `lane < 32`.
#[inline(always)]
unsafe fn lane_squares(x: &[f32], k: usize, lane: usize) -> f32 {
    let mut acc = 0.0f32;
    let mut i = lane;
    while i < k & !31 {
        // SAFETY: i < k <= x.len() by this fn's contract.
        let v = unsafe { *x.get_unchecked(i) };
        acc = v.mul_add(v, acc);
        i += 32;
    }
    acc
}

#[cuda_module]
mod hc_f32_kernels {
    use super::*;

    /// RMS + F32 gemv + HC_PRE (the module doc's rule), block `t` for token
    /// `t` of `m`. `k` is a positive multiple of 32.
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
            w.len() >= 24 * k,
            x.len() >= k * m,
            scale.len() >= 3,
            base.len() >= 24,
            mixes.len() >= 24 * m,
            hc.len() >= 24 * m,
            k >= 32,
            m >= 1
        )
    )]
    pub fn ds41_hc_pre_f32(
        w: &[f32],
        x: &[f32],
        scale: &[f32],
        base: &[f32],
        k: u32,
        m: u32,
        rms_eps: f32,
        hc_eps: f32,
        iters: u32,
        mut mixes: DisjointSlice<f32>,
        mut hc: DisjointSlice<f32>,
    ) {
        static mut MIX: SharedArray<f32, HC_MIX> = SharedArray::UNINIT;
        static mut SCL: SharedArray<f32, 1> = SharedArray::UNINIT;

        let t = thread::blockIdx_x() as usize;
        if t >= m as usize {
            return;
        }
        let tid = thread::threadIdx_x() as usize;
        let lane = warp::lane_id() as usize;
        let wp = tid / 32;
        let kk = k as usize;
        // SAFETY: (t + 1) * k <= k * m <= x.len() by the launch contract.
        let xt = unsafe { x.get_unchecked(t * kk..(t + 1) * kk) };
        // SAFETY: MIX and SCL are this block's own shared allocations; each
        // slot is written by one lane before the barrier that publishes it.
        let (mix_s, scl_s) = unsafe {
            (
                SharedArray::as_raw_mut_ptr(&raw mut MIX),
                SharedArray::as_raw_mut_ptr(&raw mut SCL),
            )
        };

        // Rows wp, wp + 8, wp + 16 < 24, so w.len() >= 24k >= (row + 1) * k;
        // xt.len() = k, k a multiple of 32 (host-checked), lane < 32: the
        // callee's contract, for each of the three calls.
        // SAFETY: the contract above, row wp.
        let pa = unsafe { f32_lane_partial_1col(w, xt, k, wp, lane) };
        // SAFETY: the contract above, row wp + 8.
        let pb = unsafe { f32_lane_partial_1col(w, xt, k, wp + 8, lane) };
        // SAFETY: the contract above, row wp + 16.
        let pc = unsafe { f32_lane_partial_1col(w, xt, k, wp + 16, lane) };
        let s_a = warp::reduce_sum_f32(pa);
        let s_b = warp::reduce_sum_f32(pb);
        let s_c = warp::reduce_sum_f32(pc);
        if lane == 0 {
            // SAFETY: wp + 16 < 24; lane 0 of warp wp alone writes these rows.
            unsafe {
                *mix_s.add(wp) = s_a;
                *mix_s.add(wp + 8) = s_b;
                *mix_s.add(wp + 16) = s_c;
            }
        }
        if wp == 0 {
            // SAFETY: xt.len() = k, lane < 32.
            let sq = warp::reduce_sum_f32(unsafe { lane_squares(xt, kk, lane) });
            if lane == 0 {
                // SAFETY: slot 0, written by lane 0 of warp 0 alone.
                unsafe {
                    *scl_s = 1.0 / (sq / k as f32 + rms_eps).sqrt();
                }
            }
        }
        thread::sync_threads();
        if wp != 0 {
            return;
        }
        // SAFETY: scale.len() >= 3 by the launch contract.
        let sc = unsafe {
            [
                *scale.get_unchecked(0),
                *scale.get_unchecked(1),
                *scale.get_unchecked(2),
            ]
        };
        let (mv, bv) = if lane < HC_MIX {
            // SAFETY: lane < 24: a slot written before the barrier, and
            // base.len() >= 24.
            unsafe { (*mix_s.add(lane) * *scl_s, *base.get_unchecked(lane)) }
        } else {
            (0.0, 0.0)
        };
        let y = hc_pre_lane(mv, lane as u32, sc, bv, hc_eps, iters);
        if lane < HC_MIX {
            // SAFETY: t*24 + lane < 24m <= mixes.len(), hc.len(); one lane
            // per slot.
            unsafe {
                *mixes.get_unchecked_mut(t * HC_MIX + lane) = mv;
                *hc.get_unchecked_mut(t * HC_MIX + lane) = y;
            }
        }
    }
}

/// One sub-layer's F32 HC_PRE parameters as the draft file holds them.
pub struct HcF32Params<'a> {
    /// `hc_{attn,ffn}_fn`: [`HC_MIX`] rows of K f32.
    pub w: &'a DeviceTensor<f32>,
    /// `hc_{attn,ffn}_scale`: the three affine scales (pre, post, comb).
    pub scale: &'a DeviceBuffer<f32>,
    /// `hc_{attn,ffn}_base`: the [`HC_MIX`] affine offsets.
    pub base: &'a DeviceBuffer<f32>,
    /// `hyper_connection.epsilon`.
    pub eps: f32,
    /// `hyper_connection.sinkhorn_iterations`.
    pub iters: u32,
}

/// The inputs of one `ds41_hc_pre_f32` launch.
pub struct HcF32Args<'a> {
    /// The sub-layer's HC_PRE parameters.
    pub params: &'a HcF32Params<'a>,
    /// The streams the sub-layer reads: `tokens` tokens of K values.
    pub x: &'a DeviceBuffer<f32>,
    /// Tokens, at least one.
    pub tokens: usize,
    /// `attention.layer_norm_rms_epsilon`.
    pub rms_eps: f32,
}

/// The loaded F32 HC_PRE module. Owns no stream.
pub struct HcF32Kernels {
    module: hc_f32_kernels::LoadedModule,
}

impl HcF32Kernels {
    /// Load this module's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<HcF32Kernels, GpuError> {
        // SAFETY: this crate owns the embedded device bundle produced for the
        // module above; each launcher checks its launch contract.
        let module = unsafe { hc_f32_kernels::load(ctx)? };
        Ok(HcF32Kernels { module })
    }

    /// Enqueue RMS + F32 gemv + HC_PRE (`ds41_hc_pre_f32`) for `a.tokens`
    /// tokens of `a.x` (K = `a.params.w.cols()` values each): `mixes` takes the scaled
    /// mixes and `hc` the HC_PRE result, [`HC_MIX`] per token. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_pre(
        &self,
        stream: &CudaStream,
        a: &HcF32Args<'_>,
        mixes: &mut DeviceBuffer<f32>,
        hc: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "HcF32Kernels::enqueue_pre";
        let (p, x) = (a.params, a.x);
        let (k, m) = (p.w.cols(), a.tokens);
        if p.w.rows() != HC_MIX
            || k == 0
            || !k.is_multiple_of(32)
            || p.scale.len() < 3
            || p.base.len() < HC_MIX
            || p.iters == 0
            || m == 0
            || x.len() < k * m
            || mixes.len() < HC_MIX * m
            || hc.len() < HC_MIX * m
        {
            return Err(GpuError::Shape {
                what,
                detail: format!(
                    "w {}x{k} (need {HC_MIX} rows, k a positive multiple of 32), scale {} base {} iters {}, \
                     tokens {m}, x {} (need {}), mixes {} / hc {} (need {})",
                    p.w.rows(),
                    p.scale.len(),
                    p.base.len(),
                    p.iters,
                    x.len(),
                    k * m,
                    mixes.len(),
                    hc.len(),
                    HC_MIX * m
                ),
            });
        }
        let (k, m) = (launch_u32(what, "k", k)?, launch_u32(what, "tokens", m)?);
        let prep =
            self.module
                .prepare_ds41_hc_pre_f32(LaunchConfig1D::new(m, HC_F32_THREADS, 0))?;
        self.module.ds41_hc_pre_f32(
            stream,
            &prep,
            p.w.buf(),
            x,
            p.scale,
            p.base,
            k,
            m,
            a.rms_eps,
            p.eps,
            p.iters,
            mixes,
            hc,
        )?;
        Ok(())
    }
}
