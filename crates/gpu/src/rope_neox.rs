//! One launch per layer for the query and key heads of a grouped-query
//! attention layer with a per-head norm and a NEOX rope over the whole head:
//! each head's RMS norm by its own gain vector (`attn_q_norm` /
//! `attn_k_norm`), the NEOX turn of pair `(i, i + HEAD/2)` by the token's
//! table pair `i`, and for the key heads the append of the turned key and of
//! the value head to the layer's f16 cache planes.
//!
//! The planes are head-major, `[n_kv][ctx][HEAD]` u16: key head `h` of the
//! token at position `p` is row `h·ctx + p`, so a flash block that walks one
//! key head reads its rows contiguously.
//!
//! Numeric contract, per head of [`HEAD`] values (thread `t` owns values `t`
//! and `t + HEAD/2`, which are also NEOX pair `t`):
//! - each value squared in f32, the two squares added in f64, the 32 lanes
//!   of a warp by the xor butterfly (16, 8, 4, 2, 1) in f64, the two warps'
//!   sums added in f64 (warp 0 first);
//! - the mean `(sum / HEAD) as f32` (an f64 divide, rounded once to f32),
//!   the scale `1 / sqrt(mean + eps)` in f32, each op rounded on its own;
//! - the normalized value `(scale · gain) · x`;
//! - the turn through [`neox_pair`]: `y0 = fma(x0, c, −(x1·s))`,
//!   `y1 = fma(x0, s, x1·c)`, the inner products rounded on their own —
//!   the rounding ik's compiled CPU NEOX loop produces (its source writes
//!   `x0·c − x1·s` and `x0·s + x1·c`), so on the same normalized values the
//!   turn is ik's bit for bit;
//! - the cache rows rounded once to f16, to nearest even.
//!
//! ik's CPU norm sums the same f32 squares serially in f64; the two f64 sums
//! of 128 terms differ by far less than one f32 ulp of the mean, so the
//! rounded means agree unless the exact mean sits within that distance of an
//! f32 rounding boundary. The gates print how many values agree bit for bit.

use crate::GpuError;
use crate::flash::f32_to_f16_bits;
use crate::launch_u32;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::float::{fma_rn_f32, mul_rn_f32};
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread, warp,
};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Values per head, and twice the block width: thread `t` owns a head's
/// values `t` and `t + HEAD/2`.
pub const HEAD: usize = 128;

/// Threads per block, one per NEOX pair.
const THREADS: usize = HEAD / 2;
const THREADS_U32: u32 = THREADS as u32;
const _: () = assert!(THREADS_U32 as usize == THREADS && THREADS == 64);

/// NEOX pair `(x0, x1)` turned by `(c, s)` in the module's form: each inner
/// product rounded on its own (`mul.rn`, which the compiler never contracts),
/// then one fused multiply-add.
#[inline(always)]
pub fn neox_pair(x0: f32, x1: f32, c: f32, s: f32) -> (f32, f32) {
    (
        fma_rn_f32(x0, c, -mul_rn_f32(x1, s)),
        fma_rn_f32(x0, s, mul_rn_f32(x1, c)),
    )
}

#[cuda_module]
mod rope_neox_kernels {
    use super::*;

    /// The per-head norm, NEOX turn and cache append of `m` tokens. Block
    /// `b = t·(n_head + n_kv) + h`: for `h < n_head` query head `h` of token
    /// `t` is normalized by `gq`, turned, and written back in place in `q`;
    /// for `h = n_head + j` key head `j` is normalized by `gk`, turned,
    /// written back in place in `k` and rounded to f16 into `cache_k` row
    /// `j·ctx + pos[t]`, and value head `j` of `v` is rounded into `cache_v`
    /// at the same row. The table of token `t` is `cs[t·HEAD ..]`, `[cos_0,
    /// sin_0, …]` for the 64 pairs. The branch is on the block index, so no
    /// warp splits across it; `pos[t] < ctx` is host-checked where the host
    /// can see it and bounds-checked here where it cannot (a position read
    /// from device memory past the planes skips the append, never writes
    /// outside them).
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(64)]
    #[launch_contract(
        domain = 1,
        block = (64, 1, 1),
        requires = (
            q.len() >= m * n_head * 128,
            k.len() >= m * n_kv * 128,
            v.len() >= m * n_kv * 128,
            gq.len() >= 128,
            gk.len() >= 128,
            cs.len() >= m * 128,
            pos.len() >= m,
            cache_k.len() >= n_kv * ctx * 128,
            cache_v.len() >= n_kv * ctx * 128
        )
    )]
    pub fn head_norm_neox_append(
        gq: &[f32],
        gk: &[f32],
        cs: &[f32],
        pos: &[u32],
        v: &[f32],
        eps: f32,
        n_head: u32,
        n_kv: u32,
        ctx: u32,
        m: u32,
        mut q: DisjointSlice<f32>,
        mut k: DisjointSlice<f32>,
        mut cache_k: DisjointSlice<u16>,
        mut cache_v: DisjointSlice<u16>,
    ) {
        static mut WSUM: SharedArray<f64, 2> = SharedArray::UNINIT;

        let heads = (n_head + n_kv) as usize;
        let b = thread::blockIdx_x() as usize;
        if b >= m as usize * heads {
            return; // block-uniform
        }
        let t = b / heads;
        let h = b - t * heads;
        let tid = thread::threadIdx_x() as usize;
        let is_q = h < n_head as usize;
        let kh = h.wrapping_sub(n_head as usize);
        let base = if is_q {
            (t * n_head as usize + h) * HEAD
        } else {
            (t * n_kv as usize + kh) * HEAD
        };

        // SAFETY: base + tid + 64 < (t·n + h + 1)·128 <= m·n·128, inside the
        // head's buffer by the launch contract; one thread per value pair.
        let (x0, x1) = unsafe {
            if is_q {
                (
                    *q.get_unchecked_mut(base + tid),
                    *q.get_unchecked_mut(base + tid + THREADS),
                )
            } else {
                (
                    *k.get_unchecked_mut(base + tid),
                    *k.get_unchecked_mut(base + tid + THREADS),
                )
            }
        };
        let mut acc = f64::from(x0 * x0) + f64::from(x1 * x1);
        acc += warp::shuffle_xor_f64(acc, 16);
        acc += warp::shuffle_xor_f64(acc, 8);
        acc += warp::shuffle_xor_f64(acc, 4);
        acc += warp::shuffle_xor_f64(acc, 2);
        acc += warp::shuffle_xor_f64(acc, 1);
        // SAFETY: WSUM is this block's own shared allocation; the raw form is
        // the only way to reach it without a reference to a `static mut`.
        // Two slots, one per warp, written before the barrier that publishes
        // them.
        let ws = unsafe { SharedArray::as_raw_mut_ptr(&raw mut WSUM) };
        if warp::lane_id() == 0 {
            // SAFETY: tid / 32 < 2; one lane per warp writes its slot.
            unsafe { *ws.add(tid / 32) = acc };
        }
        thread::sync_threads();
        // SAFETY: both slots were written before the barrier above.
        let sum = unsafe { *ws.add(0) + *ws.add(1) };
        let mean = (sum / HEAD as f64) as f32;
        let scale = 1.0 / (mean + eps).sqrt();

        // SAFETY: tid + 64 < 128 <= the gain's length; 2·tid + 1 < 128, so
        // the table pair is inside token t's 128 values, t < m.
        let (g0, g1, c, s) = unsafe {
            let g = if is_q { gq } else { gk };
            (
                *g.get_unchecked(tid),
                *g.get_unchecked(tid + THREADS),
                *cs.get_unchecked(t * HEAD + 2 * tid),
                *cs.get_unchecked(t * HEAD + 2 * tid + 1),
            )
        };
        let n0 = (scale * g0) * x0;
        let n1 = (scale * g1) * x1;
        let (y0, y1) = neox_pair(n0, n1, c, s);

        if is_q {
            // SAFETY: the positions read above; this thread owns them.
            unsafe {
                *q.get_unchecked_mut(base + tid) = y0;
                *q.get_unchecked_mut(base + tid + THREADS) = y1;
            }
            return;
        }
        // SAFETY: the positions read above; this thread owns them.
        unsafe {
            *k.get_unchecked_mut(base + tid) = y0;
            *k.get_unchecked_mut(base + tid + THREADS) = y1;
        }
        // SAFETY: t < m <= pos.len() by the launch contract.
        let p = unsafe { *pos.get_unchecked(t) } as usize;
        if p >= ctx as usize {
            return;
        }
        let row = (kh * ctx as usize + p) * HEAD;
        // SAFETY: kh < n_kv and p < ctx, so row + 127 < n_kv·ctx·128 <= the
        // planes' lengths; base + tid + 64 < m·n_kv·128 <= v.len(). One
        // thread per pair of the row; the tokens of one launch hold distinct
        // positions (host-checked), so no two blocks write one row.
        unsafe {
            *cache_k.get_unchecked_mut(row + tid) = f32_to_f16_bits(y0);
            *cache_k.get_unchecked_mut(row + tid + THREADS) = f32_to_f16_bits(y1);
            *cache_v.get_unchecked_mut(row + tid) = f32_to_f16_bits(*v.get_unchecked(base + tid));
            *cache_v.get_unchecked_mut(row + tid + THREADS) =
                f32_to_f16_bits(*v.get_unchecked(base + tid + THREADS));
        }
    }
}

/// [`RopeNeoxKernels::enqueue_head_norm_neox_append`]'s arguments: `m`
/// tokens of `n_head` query heads in `q` and `n_kv` key and value heads in
/// `k` and `v` (token-major, head after head, [`HEAD`] values each), the
/// gains, the `m` tables (`RopeTable::push` over a [`HEAD`]-wide spec) and
/// positions on the device, and the layer's two cache planes of
/// `n_kv · ctx` rows of [`HEAD`] f16.
pub struct NeoxArgs<'a> {
    pub q: &'a mut DeviceBuffer<f32>,
    pub k: &'a mut DeviceBuffer<f32>,
    pub v: &'a DeviceBuffer<f32>,
    pub gq: &'a DeviceBuffer<f32>,
    pub gk: &'a DeviceBuffer<f32>,
    pub cs: &'a DeviceBuffer<f32>,
    pub pos: &'a DeviceBuffer<u32>,
    pub eps: f32,
    pub n_head: usize,
    pub n_kv: usize,
    pub ctx: usize,
    pub m: usize,
    pub cache_k: &'a mut DeviceBuffer<u16>,
    pub cache_v: &'a mut DeviceBuffer<u16>,
}

/// The loaded module. Owns no stream: each enqueue takes the engine stream.
pub struct RopeNeoxKernels {
    module: rope_neox_kernels::LoadedModule,
}

impl RopeNeoxKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<RopeNeoxKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; the launcher checks its launch contract.
        let module = unsafe { rope_neox_kernels::load(ctx)? };
        Ok(RopeNeoxKernels { module })
    }

    /// Enqueue the norm, turn and append of `args.m` tokens: one block per
    /// (token, head), `m·(n_head + n_kv)` blocks of 64 threads. The tokens
    /// of one launch must hold distinct positions below `ctx` (the caller's
    /// positions are consecutive; the kernel skips a position `>= ctx`).
    /// Asynchronous, allocation-free, capturable.
    pub fn enqueue_head_norm_neox_append(
        &self,
        stream: &CudaStream,
        args: NeoxArgs<'_>,
    ) -> Result<(), GpuError> {
        let what = "enqueue_head_norm_neox_append";
        let NeoxArgs {
            q,
            k,
            v,
            gq,
            gk,
            cs,
            pos,
            eps,
            n_head,
            n_kv,
            ctx,
            m,
            cache_k,
            cache_v,
        } = args;
        if n_head == 0 || n_kv == 0 || m == 0 || ctx == 0 || m > ctx {
            return Err(GpuError::shape(
                what,
                format!(
                    "need n_head, n_kv, ctx >= 1 and 1 <= m <= ctx, got n_head={n_head} \
                     n_kv={n_kv} ctx={ctx} m={m}"
                ),
            ));
        }
        let plane = n_kv * ctx * HEAD;
        let lens = [
            ("q", q.len(), m * n_head * HEAD),
            ("k", k.len(), m * n_kv * HEAD),
            ("v", v.len(), m * n_kv * HEAD),
            ("gq", gq.len(), HEAD),
            ("gk", gk.len(), HEAD),
            ("cs", cs.len(), m * HEAD),
            ("pos", pos.len(), m),
            ("cache_k", cache_k.len(), plane),
            ("cache_v", cache_v.len(), plane),
        ];
        if let Some((name, got, need)) = lens.iter().find(|(_, got, need)| got < need) {
            return Err(GpuError::shape(
                what,
                format!("{name}.len() {got} < {need}"),
            ));
        }
        let grid = launch_u32(what, "grid", m * (n_head + n_kv))?;
        let n_head = launch_u32(what, "n_head", n_head)?;
        let n_kv = launch_u32(what, "n_kv", n_kv)?;
        let ctx = launch_u32(what, "ctx", ctx)?;
        let m = launch_u32(what, "m", m)?;
        let prep = self
            .module
            .prepare_head_norm_neox_append(LaunchConfig1D::new(grid, THREADS_U32, 0))?;
        self.module.head_norm_neox_append(
            stream, &prep, gq, gk, cs, pos, v, eps, n_head, n_kv, ctx, m, q, k, cache_k, cache_v,
        )?;
        Ok(())
    }
}
