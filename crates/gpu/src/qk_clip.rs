//! The projection epilogues of a rel-bias attention visit, and the f16
//! append of its key and value: one launch a visit covers the query rows,
//! the key rows, the value rows and the relative rows.
//!
//! The rows a token brings — `n_head` query rows of [`HEAD`] values,
//! `n_kv` key rows, `n_kv` value rows, `n_head` relative rows of [`REL`]
//! values — are the reference's epilogue order, one warp a row, eight warps
//! a block:
//!
//! - a query row takes its head's per-channel scale (row `h` of the file's
//!   `[n_head][128]` `attn_q_clip`), is clipped to an L2 norm of
//!   at most `qk_thresh`, then carries the row's `tau`;
//! - a key row takes no scale and clips at the same threshold; a value row
//!   takes a gainless RMS norm at `v_eps` and no clip; both append f16 into
//!   this visit's cache planes, head-major `[n_kv][ctx][HEAD]` u16 as
//!   `qk_rms` writes them;
//! - a relative row takes its head's per-channel scale (row `h` of the file's
//!   `[n_head][16]` `attn_rel_clip`), clips at `r_thresh`, then
//!   carries `tau`. A windowed visit passes `tau` 1.0 and the multiply is
//!   exact; a global visit passes the log-scaled position factor.
//!
//! The clip is the reference's: each value scaled, squared in f32, the
//! squares summed in f64 (lane order ascending, then the warp by the xor
//! butterfly — this order is the gate), the sum cast to f32 and rooted, the
//! root biased by the fixed 1e-6, the factor `threshold / (root + 1e-6)`
//! clamped to `[0, 1]`, one multiply back. The value row's RMS is
//! `qk_rms`'s: the same f64 square sum, the mean, then
//! `rms_scale_of_mean_checked`.
//!
//! A non-finite square sum (a non-finite input, or a square past f32)
//! raises [`FaultSite::CacheValue`] and the row is NaN; a value past f16's
//! range raises it at the append; a position at or past `ctx` raises
//! [`FaultSite::CachePos`], leaves that token's key and value rows NaN and
//! appends nothing.

use crate::GpuError;
use crate::elem::rms_scale_of_mean_checked;
use crate::fault::{FaultSink, FaultSite};
use crate::flash::cache_f16_pair;
use crate::launch_u32;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread, warp};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Values per query, key or value head.
pub const HEAD: usize = 128;
/// Values per relative row.
pub const REL: usize = 16;
const WARPS: usize = 8;
const THREADS: usize = WARPS * 32;
const THREADS_U32: u32 = THREADS as u32;
/// Values a lane owns of a 128-wide row: `lane`, `lane + 32`, `+ 64`, `+ 96`.
const STRIDE: usize = 32;
const _: () = assert!(THREADS == 256 && HEAD == 4 * STRIDE && REL <= STRIDE);
/// The fixed bias the reference's clip adds to the rooted norm.
const CLIP_BIAS: f32 = 1e-6;

#[cuda_module]
mod qk_clip_kernels {
    use super::*;

    /// The epilogue of `m` tokens. Block `b`, warp `w` is the block's warp
    /// `8·b + w` of the launch's rows — `2·(n_head + n_kv)` a token, the
    /// query heads first, then the key heads, the value heads, the relative
    /// heads; a warp past the last row of the last token does nothing.
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
            gq.len() >= n_head * 128,
            gr.len() >= n_head * 16,
            tau.len() >= m,
            pos.len() >= m,
            n_head >= 1,
            n_kv >= 1,
            ctx >= 1,
            q.len() >= m * n_head * 128,
            k.len() >= m * n_kv * 128,
            v.len() >= m * n_kv * 128,
            r.len() >= m * n_head * 16,
            cache_k.len() >= n_kv * ctx * 128,
            cache_v.len() >= n_kv * ctx * 128
        )
    )]
    pub fn qk_clip_append(
        gq: &[f32],
        gr: &[f32],
        tau: &[f32],
        pos: &[u32],
        v_eps: f32,
        qk_thresh: f32,
        r_thresh: f32,
        n_head: u32,
        n_kv: u32,
        ctx: u32,
        m: u32,
        fault: FaultSink,
        mut q: DisjointSlice<f32>,
        mut k: DisjointSlice<f32>,
        mut v: DisjointSlice<f32>,
        mut r: DisjointSlice<f32>,
        mut cache_k: DisjointSlice<u16>,
        mut cache_v: DisjointSlice<u16>,
    ) {
        let heads = n_head as usize;
        let kv = n_kv as usize;
        let rows = m as usize * 2 * (heads + kv);
        let b = thread::blockIdx_x() as usize;
        let gw = b * WARPS + thread::threadIdx_x() as usize / 32;
        if gw >= rows {
            return; // block-uniform
        }
        let t = gw / (2 * (heads + kv));
        let j = gw - t * 2 * (heads + kv);
        let lane = warp::lane_id() as usize;
        // SAFETY: t < m <= tau.len() and pos.len() by the launch contract.
        let (tau_t, p) = unsafe { (*tau.get_unchecked(t), *pos.get_unchecked(t) as usize) };

        // The query heads, then the key heads. The two rows are separate
        // launch parameters, so each touch names its own: `is_q` is
        // warp-uniform and its branch is free.
        if j < heads + kv {
            let (is_q, h) = if j < heads {
                (true, j)
            } else {
                (false, j - heads)
            };
            let base = (t * if is_q { heads } else { kv } + h) * HEAD;
            if !is_q && p >= ctx as usize {
                if lane == 0 {
                    fault.raise(FaultSite::CachePos);
                }
                // SAFETY: the positions read above. This lane owns them.
                unsafe { nan_k_row(&mut k, base, lane) };
                return; // warp-uniform: p is the token's
            }
            // A query row's values take the channel scale first; the dot's
            // order is the gate: squares in f32, lane order ascending, the
            // warp by the xor butterfly, all in f64.
            let mut x = [0.0f32; 4];
            let mut acc = 0.0f64;
            let mut i = 0usize;
            while i < 4 {
                // SAFETY: base + lane + i·32 is inside the head's buffer by
                // the launch contract. One lane owns four values.
                let c = lane + i * STRIDE;
                // SAFETY: base + c is inside the head's own buffer (q for a
                // query head, k for a key head) by the launch contract. One
                // lane owns four values.
                let xc = unsafe {
                    if is_q {
                        *q.get_unchecked_mut(base + c)
                    } else {
                        *k.get_unchecked_mut(base + c)
                    }
                };
                let yc = if is_q {
                    // SAFETY: h·128 + c < n_head·128 <= gq.len() by the launch
                    // contract.
                    xc * unsafe { *gq.get_unchecked(h * HEAD + c) }
                } else {
                    xc
                };
                x[i] = yc;
                acc += f64::from(yc * yc);
                i += 1;
            }
            let root = (warp_butterfly(acc) as f32).sqrt();
            let (f, ok) = clip_factor(root, qk_thresh);
            if !ok && lane == 0 {
                fault.raise(FaultSite::CacheValue);
            }
            let mut i = 0usize;
            while i < 4 {
                let out = if !ok {
                    f32::NAN
                } else if is_q {
                    x[i] * f * tau_t
                } else {
                    x[i] * f
                };
                // SAFETY: the positions read above. This lane owns them.
                unsafe {
                    if is_q {
                        *q.get_unchecked_mut(base + lane + i * STRIDE) = out;
                    } else {
                        *k.get_unchecked_mut(base + lane + i * STRIDE) = out;
                    }
                }
                i += 1;
            }
            if is_q {
                return; // warp-uniform: the query rows append nothing
            }
            if !ok {
                return; // warp-uniform: the root is the warp's
            }
            // SAFETY: h < n_kv and p < ctx, so the row is inside both
            // planes. The tokens of one launch hold distinct positions, so
            // no two blocks write one row.
            unsafe {
                let at = (h * ctx as usize + p) * HEAD + lane;
                let (a, b) = cache_f16_pair(x[0] * f, x[1] * f, fault);
                let (c, d) = cache_f16_pair(x[2] * f, x[3] * f, fault);
                *cache_k.get_unchecked_mut(at) = a;
                *cache_k.get_unchecked_mut(at + STRIDE) = b;
                *cache_k.get_unchecked_mut(at + 2 * STRIDE) = c;
                *cache_k.get_unchecked_mut(at + 3 * STRIDE) = d;
            }
            return;
        }

        // The value heads.
        if j < heads + 2 * kv {
            let h = j - heads - kv;
            let base = (t * kv + h) * HEAD;
            if p >= ctx as usize {
                if lane == 0 {
                    fault.raise(FaultSite::CachePos);
                }
                // SAFETY: the positions read above. This lane owns them.
                unsafe { nan_k_row(&mut v, base, lane) };
                return; // warp-uniform: p is the token's
            }
            let mut x = [0.0f32; 4];
            let mut acc = 0.0f64;
            let mut i = 0usize;
            while i < 4 {
                // SAFETY: base + lane + i·32 is inside v by the launch
                // contract. One lane owns four values.
                let c = lane + i * STRIDE;
                // SAFETY: base + c is inside v by the launch contract. One
                // lane owns four values.
                let xc = unsafe { *v.get_unchecked_mut(base + c) };
                x[i] = xc;
                acc += f64::from(xc * xc);
                i += 1;
            }
            let mean = (warp_butterfly(acc) / HEAD as f64) as f32;
            let scale = rms_scale_of_mean_checked(mean, v_eps, fault, lane == 0);
            let mut i = 0usize;
            while i < 4 {
                x[i] *= scale;
                i += 1;
            }
            // SAFETY: the positions read above. This lane owns them.
            unsafe {
                let c = lane;
                *v.get_unchecked_mut(base + c) = x[0];
                *v.get_unchecked_mut(base + c + STRIDE) = x[1];
                *v.get_unchecked_mut(base + c + 2 * STRIDE) = x[2];
                *v.get_unchecked_mut(base + c + 3 * STRIDE) = x[3];
                // h < n_kv and p < ctx, so the row is inside the plane.
                let at = (h * ctx as usize + p) * HEAD + lane;
                let (a, b) = cache_f16_pair(x[0], x[1], fault);
                let (c, d) = cache_f16_pair(x[2], x[3], fault);
                *cache_v.get_unchecked_mut(at) = a;
                *cache_v.get_unchecked_mut(at + STRIDE) = b;
                *cache_v.get_unchecked_mut(at + 2 * STRIDE) = c;
                *cache_v.get_unchecked_mut(at + 3 * STRIDE) = d;
            }
            return;
        }

        // The relative heads.
        let h = j - heads - 2 * kv;
        let base = (t * heads + h) * REL;
        // SAFETY: base + lane is inside r by the launch contract when
        // lane < REL; the lanes past the row read nothing and hold zero.
        let xc = unsafe {
            if lane < REL {
                *r.get_unchecked_mut(base + lane)
            } else {
                0.0
            }
        };
        let g = if lane < REL {
            // SAFETY: h·REL + lane < n_head·REL <= gr.len() by the launch
            // contract; the lanes past the row read nothing.
            unsafe { *gr.get_unchecked(h * REL + lane) }
        } else {
            0.0
        };
        let yc = if lane < REL { xc * g } else { 0.0 };
        let acc = warp_butterfly(if lane < REL { f64::from(yc * yc) } else { 0.0 });
        let root = (acc as f32).sqrt();
        let (f, ok) = clip_factor(root, r_thresh);
        let out = if ok { yc * f * tau_t } else { f32::NAN };
        if !ok && lane == 0 {
            fault.raise(FaultSite::CacheValue);
        }
        if lane < REL {
            // SAFETY: the positions read above. This lane owns its value.
            unsafe { *r.get_unchecked_mut(base + lane) = out };
        }
    }

    /// The f64 xor butterfly over the warp: the five shuffles, halves down.
    /// All 32 lanes of the warp call it together.
    #[inline(always)]
    fn warp_butterfly(acc: f64) -> f64 {
        let mut a = acc;
        a += warp::shuffle_xor_f64(a, 16);
        a += warp::shuffle_xor_f64(a, 8);
        a += warp::shuffle_xor_f64(a, 4);
        a += warp::shuffle_xor_f64(a, 2);
        a += warp::shuffle_xor_f64(a, 1);
        a
    }

    /// The clip's factor: `threshold / (root + 1e-6)` clamped to `[0, 1]`,
    /// with `ok` false when the root the warp shares is not finite — the
    /// row has no defined clip.
    #[inline(always)]
    fn clip_factor(root: f32, threshold: f32) -> (f32, bool) {
        if root.is_finite() {
            ((threshold / (root + CLIP_BIAS)).clamp(0.0, 1.0), true)
        } else {
            (f32::NAN, false)
        }
    }

    /// Write NaN over the lane's four values of one 128-wide key row.
    ///
    /// SAFETY: `base + lane + i·32` is inside `row` for the lane's four
    /// values; this lane owns them.
    #[inline(always)]
    unsafe fn nan_k_row(row: &mut DisjointSlice<f32>, base: usize, lane: usize) {
        let mut i = 0usize;
        while i < 4 {
            // SAFETY: the caller's positions. This lane owns them.
            unsafe { *row.get_unchecked_mut(base + lane + i * STRIDE) = f32::NAN };
            i += 1;
        }
    }
}

/// Arguments of one epilogue launch. `gq` and `gr` are the per-head,
/// per-channel scales of the query and relative rows (`n_head · 128` and
/// `n_head · 16` values, head-major, as the file stores `attn_q_clip` and
/// `attn_rel_clip`), `tau` one factor a token (1.0 on
/// a windowed visit), `pos` the append position a token, `v_eps` the value
/// row's RMS epsilon, `qk_thresh` the query and key clip's threshold,
/// `r_thresh` the relative rows'.
pub struct QkClipArgs<'a> {
    pub q: &'a mut DeviceBuffer<f32>,
    pub k: &'a mut DeviceBuffer<f32>,
    pub v: &'a mut DeviceBuffer<f32>,
    pub r: &'a mut DeviceBuffer<f32>,
    pub gq: &'a DeviceBuffer<f32>,
    pub gr: &'a DeviceBuffer<f32>,
    pub tau: &'a DeviceBuffer<f32>,
    pub pos: &'a DeviceBuffer<u32>,
    pub v_eps: f32,
    pub qk_thresh: f32,
    pub r_thresh: f32,
    pub n_head: usize,
    pub n_kv: usize,
    pub ctx: usize,
    pub m: usize,
    pub fault: FaultSink,
    pub cache_k: &'a mut DeviceBuffer<u16>,
    pub cache_v: &'a mut DeviceBuffer<u16>,
}

/// The loaded module. Owns no stream: each enqueue takes the engine stream.
pub struct QkClipKernels {
    module: qk_clip_kernels::LoadedModule,
}

impl QkClipKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<QkClipKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; the launcher checks its launch contract.
        let module = unsafe { crate::shared_module!(qk_clip_kernels, ctx)? };
        Ok(QkClipKernels { module })
    }

    /// Enqueue the epilogue of `args.m` tokens. A zero count, `m > ctx`, a
    /// non-finite or non-positive threshold or epsilon, or a short buffer
    /// is refused by name before the launch.
    pub fn enqueue(&self, stream: &CudaStream, args: QkClipArgs<'_>) -> Result<(), GpuError> {
        let what = "qk_clip::enqueue";
        let QkClipArgs {
            q,
            k,
            v,
            r,
            gq,
            gr,
            tau,
            pos,
            v_eps,
            qk_thresh,
            r_thresh,
            n_head,
            n_kv,
            ctx,
            m,
            fault,
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
        for (name, t) in [("qk_thresh", qk_thresh), ("r_thresh", r_thresh)] {
            if !t.is_finite() || t <= 0.0 {
                return Err(GpuError::shape(
                    what,
                    format!("the clip thresholds must be finite and positive; {name}={t}"),
                ));
            }
        }
        if !v_eps.is_finite() || v_eps < 0.0 {
            return Err(GpuError::shape(
                what,
                format!("the value rows' eps must be finite and non-negative, got {v_eps}"),
            ));
        }
        let plane = n_kv * ctx * HEAD;
        let lens = [
            ("q", q.len(), m * n_head * HEAD),
            ("k", k.len(), m * n_kv * HEAD),
            ("v", v.len(), m * n_kv * HEAD),
            ("r", r.len(), m * n_head * REL),
            ("gq", gq.len(), n_head * HEAD),
            ("gr", gr.len(), n_head * REL),
            ("tau", tau.len(), m),
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
        let rows = 2 * (n_head + n_kv);
        let grid = launch_u32(what, "grid", m * rows.div_ceil(WARPS))?;
        let n_head = launch_u32(what, "n_head", n_head)?;
        let n_kv = launch_u32(what, "n_kv", n_kv)?;
        let ctx = launch_u32(what, "ctx", ctx)?;
        let m = launch_u32(what, "m", m)?;
        let prep = self
            .module
            .prepare_qk_clip_append(LaunchConfig1D::new(grid, THREADS_U32, 0))?;
        self.module.qk_clip_append(
            stream, &prep, gq, gr, tau, pos, v_eps, qk_thresh, r_thresh, n_head, n_kv, ctx, m,
            fault, q, k, v, r, cache_k, cache_v,
        )?;
        Ok(())
    }
}
