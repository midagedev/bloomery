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
//!
//! The decode step's form ([`QkClipKernels::enqueue_conv`], one token) takes the k and v rows'
//! input from the residual short conv itself: the warp of a key or value row computes the conv's
//! four channels of each of its lanes ([`crate::linear::conv_res`]'s rule, op for op), stores
//! the conv ring's slot, and feeds the values to the epilogue above, so the pair of conv launches
//! and the epilogue launch are one. The query and relative rows are the same code as the rows
//! form's.

use crate::GpuError;
use crate::elem::rms_scale_of_mean_checked;
use crate::fault::{FaultSink, FaultSite};
use crate::flash::cache_f16_pair;
use crate::launch_u32;
use crate::linear::ring_input;
use chain::linear::Ring;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::float::{add_rn_f32, fma_rn_f32, mul_rn_f32};
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
/// The conv's taps the fused entry is built for (`attention.shortconv_l_cache`): its tap loops
/// are unrolled over registers, so another width is refused by name and the caller keeps the
/// conv and epilogue launches apart.
pub const CONV_TAPS: usize = 4;
// Both rows of the conv's channels a lane owns are the epilogue's four values a lane.
const _: () = assert!(CONV_TAPS == 4);

/// The k and v convs the fused entry runs inside the key and value rows (module doc): per conv
/// its input rows `x`, filters `w` (`[d][CONV_TAPS]`) and dynamic taps `dyn_` (`[heads][CONV_TAPS]`,
/// lag 0 the current input), the ring `ring` (`rows · ch` values; the k conv's channels from
/// `base[0]`, the v conv's from `base[1]`) and the position `pos[0]` of the token in it.
#[derive(Clone, Copy)]
struct ConvIn<'a> {
    x: [&'a [f32]; 2],
    w: [&'a [f32]; 2],
    dyn_: [&'a [f32]; 2],
    pos: &'a [u32],
    ring: *mut f32,
    rows: usize,
    ch: usize,
    base: [usize; 2],
    dyn_heads: usize,
    scale: f32,
}

/// The residual short conv of the token's four channels `row + lane + 32 i` of conv `which` (0 the
/// k conv, 1 the v conv), `row = h · HEAD` the key head's first channel: the rule of
/// [`crate::linear::conv_res`] at one token, each channel's taps and residual in that entry's
/// order. Stores the token's inputs into the ring's slot after every read and raises
/// [`FaultSite::LinearConv`] for an output that is not finite, which is written as NaN.
///
/// # Safety
///
/// `row + 128 <= d` for the conv's `d` channels, the buffers hold what [`ConvIn`] says
/// (`x.len() >= d`, `w.len() >= d · CONV_TAPS`, `dyn_.len() >= heads · CONV_TAPS`, `pos.len() >= 1`,
/// `rows >= CONV_TAPS`, `base + d <= ch`), and no other thread touches this warp's channels of
/// the ring while it runs.
#[inline(always)]
unsafe fn conv_lane(
    c: &ConvIn<'_>,
    which: usize,
    row: usize,
    lane: usize,
    fault: FaultSink,
) -> [f32; 4] {
    let (x, w, dyn_, base) = (c.x[which], c.w[which], c.dyn_[which], c.base[which]);
    // SAFETY: pos.len() >= 1 by the fn's contract.
    let p = unsafe { *c.pos.get_unchecked(0) } as usize;
    let reach = CONV_TAPS - 1;
    // Every read first, then the arithmetic, then the ring stores: the stores cannot move above a
    // read of the ring, and the reads are issued together.
    let mut hist = [[0.0f32; CONV_TAPS]; 4];
    let mut tap = [[0.0f32; CONV_TAPS]; 4];
    let mut i = 0usize;
    while i < 4 {
        let cc = row + lane + i * STRIDE;
        // SAFETY: cc < d <= x.len(); the ring row of a position `back` before is the ring's, c
        // < ch and rows > back >= 1 by the contract.
        unsafe {
            hist[i][0] = *x.get_unchecked(cc);
            let mut back = 1usize;
            while back < CONV_TAPS {
                hist[i][back] = ring_input(c.ring.cast_const(), c.rows, p, back, c.ch, base + cc);
                back += 1;
            }
            let mut k = 0usize;
            while k < CONV_TAPS {
                tap[i][k] = *w.get_unchecked(cc * CONV_TAPS + k);
                k += 1;
            }
        }
        i += 1;
    }
    let mut lag_tap = [0.0f32; CONV_TAPS];
    if c.dyn_heads > 0 {
        let head = row / HEAD;
        let mut lag = 0usize;
        while lag < CONV_TAPS {
            // SAFETY: head < heads, so head · CONV_TAPS + lag < dyn_.len().
            lag_tap[lag] = unsafe { *dyn_.get_unchecked(head * CONV_TAPS + lag) };
            lag += 1;
        }
    }
    let mut out = [0.0f32; 4];
    let mut ok = true;
    let mut i = 0usize;
    while i < 4 {
        // The static taps, the oldest first: tap k pairs the input `reach − k` positions back.
        let mut acc = mul_rn_f32(hist[i][reach], tap[i][0]);
        let mut k = 1usize;
        while k < CONV_TAPS {
            acc = fma_rn_f32(tap[i][k], hist[i][reach - k], acc);
            k += 1;
        }
        // The dynamic taps, the current input first.
        if c.dyn_heads > 0 {
            let mut lag = 0usize;
            while lag < CONV_TAPS {
                acc = fma_rn_f32(lag_tap[lag], hist[i][lag], acc);
                lag += 1;
            }
        }
        let y = mul_rn_f32(add_rn_f32(hist[i][0], acc), c.scale);
        let fin = y.is_finite();
        ok &= fin;
        out[i] = if fin { y } else { f32::NAN };
        i += 1;
    }
    // The token's inputs enter their slot of the ring, after the reads above.
    let slot = p % c.rows;
    let mut i = 0usize;
    while i < 4 {
        let cc = row + lane + i * STRIDE;
        // SAFETY: slot < rows and base + cc < ch, so the index is inside the ring; this thread
        // owns the channel.
        unsafe { *c.ring.add(slot * c.ch + base + cc) = hist[i][0] };
        i += 1;
    }
    if !ok {
        fault.raise(FaultSite::LinearConv);
    }
    out
}

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
        epilogue_rows(
            (gq, gr, tau, pos),
            (v_eps, qk_thresh, r_thresh),
            (n_head as usize, n_kv as usize, ctx as usize, m as usize),
            fault,
            (&mut q, &mut k, &mut v, &mut r),
            (&mut cache_k, &mut cache_v),
            None,
        );
    }

    /// [`qk_clip_append`] of one token whose key and value rows come from the residual short
    /// conv ([`crate::linear::conv_res`]'s `conv_res` of the k and v channels), run by the row's
    /// own warp (module doc): `kx` and `vx` the convs' inputs, `kw` and `vw` their filters,
    /// `kd` and `vd` the dynamic taps (`dyn_heads` heads, 0 for a static conv), `cpos` the
    /// token's position in the conv ring `ring` (`rows` slots of `ch` channels, the k conv's from
    /// `base_k`, the v conv's from `base_v`), `pos` its append position. The ring is stored and
    /// the key and value rows hold the epilogue's values as the pair of launches leaves them.
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
            tau.len() >= 1,
            pos.len() >= 1,
            cpos.len() >= 1,
            n_head >= 1,
            n_kv >= 1,
            ctx >= 1,
            rows >= 4,
            q.len() >= n_head * 128,
            k.len() >= n_kv * 128,
            v.len() >= n_kv * 128,
            r.len() >= n_head * 16,
            kx.len() >= n_kv * 128,
            vx.len() >= n_kv * 128,
            kw.len() >= n_kv * 512,
            vw.len() >= n_kv * 512,
            kd.len() >= dyn_heads * 4,
            vd.len() >= dyn_heads * 4,
            base_k + n_kv * 128 <= ch,
            base_v + n_kv * 128 <= ch,
            ring.len() >= rows * ch,
            cache_k.len() >= n_kv * ctx * 128,
            cache_v.len() >= n_kv * ctx * 128
        )
    )]
    pub fn qk_clip_conv_append(
        gq: &[f32],
        gr: &[f32],
        tau: &[f32],
        pos: &[u32],
        kx: &[f32],
        vx: &[f32],
        kw: &[f32],
        vw: &[f32],
        kd: &[f32],
        vd: &[f32],
        cpos: &[u32],
        v_eps: f32,
        qk_thresh: f32,
        r_thresh: f32,
        scale: f32,
        n_head: u32,
        n_kv: u32,
        ctx: u32,
        rows: u32,
        ch: u32,
        base_k: u32,
        base_v: u32,
        dyn_heads: u32,
        fault: FaultSink,
        mut q: DisjointSlice<f32>,
        mut k: DisjointSlice<f32>,
        mut v: DisjointSlice<f32>,
        mut r: DisjointSlice<f32>,
        mut cache_k: DisjointSlice<u16>,
        mut cache_v: DisjointSlice<u16>,
        mut ring: DisjointSlice<f32>,
    ) {
        let conv = ConvIn {
            x: [kx, vx],
            w: [kw, vw],
            dyn_: [kd, vd],
            pos: cpos,
            ring: ring.as_mut_ptr(),
            rows: rows as usize,
            ch: ch as usize,
            base: [base_k as usize, base_v as usize],
            dyn_heads: dyn_heads as usize,
            scale,
        };
        epilogue_rows(
            (gq, gr, tau, pos),
            (v_eps, qk_thresh, r_thresh),
            (n_head as usize, n_kv as usize, ctx as usize, 1),
            fault,
            (&mut q, &mut k, &mut v, &mut r),
            (&mut cache_k, &mut cache_v),
            Some(&conv),
        );
    }

    /// The epilogue's rows of the launch (`m` tokens, `conv` the k and v convs the key and
    /// value rows run first, only for `m = 1`): one warp a row.
    #[allow(
        clippy::too_many_lines,
        reason = "one body for the four row kinds, as the entry always had"
    )]
    #[inline(always)]
    fn epilogue_rows(
        (gq, gr, tau, pos): (&[f32], &[f32], &[f32], &[u32]),
        (v_eps, qk_thresh, r_thresh): (f32, f32, f32),
        (heads, kv, ctx, m): (usize, usize, usize, usize),
        fault: FaultSink,
        (q, k, v, r): (
            &mut DisjointSlice<f32>,
            &mut DisjointSlice<f32>,
            &mut DisjointSlice<f32>,
            &mut DisjointSlice<f32>,
        ),
        (cache_k, cache_v): (&mut DisjointSlice<u16>, &mut DisjointSlice<u16>),
        conv: Option<&ConvIn<'_>>,
    ) {
        let rows = m * 2 * (heads + kv);
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
            // The key row's input from the conv, when the launch runs it (the conv's ring is
            // stored whatever the append position).
            let conv_x = match conv {
                // SAFETY: the launch's contract (`m` is 1: the row is the head's alone).
                Some(c) if !is_q => Some(unsafe { conv_lane(c, 0, h * HEAD, lane, fault) }),
                _ => None,
            };
            if !is_q && p >= ctx {
                if lane == 0 {
                    fault.raise(FaultSite::CachePos);
                }
                // SAFETY: the positions read above. This lane owns them.
                unsafe { nan_k_row(k, base, lane) };
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
                let xc = match conv_x {
                    Some(cx) => cx[i],
                    // SAFETY: base + c is inside the head's own buffer (q for a
                    // query head, k for a key head) by the launch contract. One
                    // lane owns four values.
                    None => unsafe {
                        if is_q {
                            *q.get_unchecked_mut(base + c)
                        } else {
                            *k.get_unchecked_mut(base + c)
                        }
                    },
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
                let at = (h * ctx + p) * HEAD + lane;
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
            #[allow(
                clippy::manual_map,
                reason = "a kernel body takes no closure: the arm is the call"
            )]
            let conv_x = match conv {
                // SAFETY: as the key rows'.
                Some(c) => Some(unsafe { conv_lane(c, 1, h * HEAD, lane, fault) }),
                None => None,
            };
            if p >= ctx {
                if lane == 0 {
                    fault.raise(FaultSite::CachePos);
                }
                // SAFETY: the positions read above. This lane owns them.
                unsafe { nan_k_row(v, base, lane) };
                return; // warp-uniform: p is the token's
            }
            let mut x = [0.0f32; 4];
            let mut acc = 0.0f64;
            let mut i = 0usize;
            while i < 4 {
                // SAFETY: base + lane + i·32 is inside v by the launch
                // contract. One lane owns four values.
                let c = lane + i * STRIDE;
                let xc = match conv_x {
                    Some(cx) => cx[i],
                    // SAFETY: base + c is inside v by the launch contract. One
                    // lane owns four values.
                    None => unsafe { *v.get_unchecked_mut(base + c) },
                };
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
                let at = (h * ctx + p) * HEAD + lane;
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

/// The k and v convs [`QkClipKernels::enqueue_conv`] runs inside the epilogue, each pair of
/// buffers the k conv's, then the v conv's: the input rows `x` (`[n_kv · HEAD]`, the
/// projection's raw output), the filters `weights` (`[n_kv · HEAD][CONV_TAPS]`, the oldest tap
/// first), the dynamic taps `dynamic` (`[heads][CONV_TAPS]`, lag 0 the current input, empty
/// where `heads` is 0), the token's position `pos` in the ring (the words the conv reads) and the
/// ring `state` (`[ring.rows()][ch]`) the convs' channels sit in at `base`, read and written in
/// place; `scale` is the convs' output scale.
pub struct QkClipConv<'a> {
    pub ring: Ring,
    /// The ring's row width: the visit's whole channel width.
    pub ch: usize,
    /// The k and v convs' first channel in the ring's rows.
    pub base: [usize; 2],
    /// The dynamic taps' heads: the key-value head count, 0 for static convs.
    pub heads: usize,
    pub scale: f32,
    pub x: [&'a DeviceBuffer<f32>; 2],
    pub weights: [&'a DeviceBuffer<f32>; 2],
    pub dynamic: [&'a DeviceBuffer<f32>; 2],
    pub pos: &'a DeviceBuffer<u32>,
    pub state: &'a mut DeviceBuffer<f32>,
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
        check(what, &args)?;
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

    /// [`QkClipKernels::enqueue`] of one token with the k and v convs run inside it (module doc):
    /// `args.k` and `args.v` are the epilogue's outputs, the convs' inputs are `conv.x`. Refuses by
    /// name a count other than one token, a conv of other taps than [`CONV_TAPS`], dynamic heads
    /// that are neither none nor the key-value heads, a channel range outside the ring's rows, a
    /// scale that is not finite and a buffer shorter than the launch needs, besides
    /// [`QkClipKernels::enqueue`]'s refusals.
    pub fn enqueue_conv(
        &self,
        stream: &CudaStream,
        args: QkClipArgs<'_>,
        conv: QkClipConv<'_>,
    ) -> Result<(), GpuError> {
        let what = "qk_clip::enqueue_conv";
        check(what, &args)?;
        let QkClipConv {
            ring,
            ch,
            base,
            heads,
            scale,
            x,
            weights,
            dynamic,
            pos: cpos,
            state,
        } = conv;
        let (taps, rows) = (ring.taps(), ring.rows());
        if args.m != 1 {
            return Err(GpuError::shape(
                what,
                format!("one token a launch, got m={}", args.m),
            ));
        }
        if taps != CONV_TAPS {
            return Err(GpuError::shape(
                what,
                format!("the fused conv is built for {CONV_TAPS} taps, got {taps}"),
            ));
        }
        if heads != 0 && heads != args.n_kv {
            return Err(GpuError::shape(
                what,
                format!(
                    "dynamic heads {heads}: none, or the {} key-value heads",
                    args.n_kv
                ),
            ));
        }
        if !scale.is_finite() {
            return Err(GpuError::shape(
                what,
                format!("scale {scale}: a finite one"),
            ));
        }
        let d = args.n_kv * HEAD;
        if let Some(b) = base.iter().find(|b| **b + d > ch) {
            return Err(GpuError::shape(
                what,
                format!("channels {b}..{} of a ring row of {ch}: inside it", *b + d),
            ));
        }
        crate::short(
            what,
            &[
                ("kx", x[0].len(), d),
                ("vx", x[1].len(), d),
                ("kw", weights[0].len(), d * taps),
                ("vw", weights[1].len(), d * taps),
                ("kd", dynamic[0].len(), heads * taps),
                ("vd", dynamic[1].len(), heads * taps),
                ("conv pos", cpos.len(), 1),
                ("state", state.len(), rows * ch),
            ],
        )?;
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
            fault,
            cache_k,
            cache_v,
            ..
        } = args;
        let grid = launch_u32(what, "grid", (2 * (n_head + n_kv)).div_ceil(WARPS))?;
        let n_head = launch_u32(what, "n_head", n_head)?;
        let n_kv = launch_u32(what, "n_kv", n_kv)?;
        let ctx = launch_u32(what, "ctx", ctx)?;
        let rows = launch_u32(what, "rows", rows)?;
        let ch = launch_u32(what, "ch", ch)?;
        let (base_k, base_v) = (
            launch_u32(what, "base_k", base[0])?,
            launch_u32(what, "base_v", base[1])?,
        );
        let heads = launch_u32(what, "heads", heads)?;
        let prep = self
            .module
            .prepare_qk_clip_conv_append(LaunchConfig1D::new(grid, THREADS_U32, 0))?;
        self.module.qk_clip_conv_append(
            stream, &prep, gq, gr, tau, pos, x[0], x[1], weights[0], weights[1], dynamic[0],
            dynamic[1], cpos, v_eps, qk_thresh, r_thresh, scale, n_head, n_kv, ctx, rows, ch,
            base_k, base_v, heads, fault, q, k, v, r, cache_k, cache_v, state,
        )?;
        Ok(())
    }
}

/// The refusals both launch forms share: a count, a threshold or epsilon, a buffer.
fn check(what: &'static str, args: &QkClipArgs<'_>) -> Result<(), GpuError> {
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
        cache_k,
        cache_v,
        ..
    } = args;
    let (n_head, n_kv, ctx, m) = (*n_head, *n_kv, *ctx, *m);
    if n_head == 0 || n_kv == 0 || m == 0 || ctx == 0 || m > ctx {
        return Err(GpuError::shape(
            what,
            format!(
                "need n_head, n_kv, ctx >= 1 and 1 <= m <= ctx, got n_head={n_head} \
                 n_kv={n_kv} ctx={ctx} m={m}"
            ),
        ));
    }
    for (name, t) in [("qk_thresh", *qk_thresh), ("r_thresh", *r_thresh)] {
        if !t.is_finite() || t <= 0.0 {
            return Err(GpuError::shape(
                what,
                format!("the clip thresholds must be finite and positive; {name}={t}"),
            ));
        }
    }
    if !v_eps.is_finite() || *v_eps < 0.0 {
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
    Ok(())
}
