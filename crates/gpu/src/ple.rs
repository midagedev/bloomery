//! A PLE site: a per-layer n-gram embedding added to every hyper-connection
//! stream (Qwen3.8-Flash-Next's layer 1). The host hashes each token's last
//! n-grams into table rows (`engram::Hash::ple_rows_into`), decodes them
//! (`gguf::quant::dequant_row`, IQ4_NL) into `e`, `ROW` values a token, and
//! two Q8_0 gemvs make `key = ple_key·e` (`hc` rows of [`ROW`] a token) and
//! `value = ple_value·e` ([`ROW`] a token). Two launches follow:
//! - [`PleKernels::enqueue_gate`], one block per (token, stream): the
//!   stream's key and the stream itself normalized and scaled by their gains,
//!   their dot, the gate `σ(sgn(s)·√max(|s|, 1e-6))` of `s = dot/√ROW`, the
//!   gated value `gv = value·gate`, and `gv` normalized and scaled by the conv
//!   gain (`ngv`);
//! - [`PleKernels::enqueue_conv`], one thread per (unit, channel) over the
//!   `hc·ROW` channels: the causal depthwise conv of [`TAPS`] taps
//!   [`DILATION`] positions apart over `ngv`, SiLU, and the stream update
//!   `out = x + (gv + conv)`, with the conv ring's new slots.
//!
//! Layouts, token-major: `key`, the streams `x` and `out`, `gv` and `ngv`
//! `[ROW, hc, m]` (channel `c = s·ROW + d`); `value` `[ROW, m]`; the three
//! gains `[ROW, hc]` in f32; `gate` `[hc, m]`; the conv weight `[TAPS, hc·ROW]`
//! (tap `k` of channel `c` at `TAPS·c + k`); the ring `[RING_ROWS][hc·ROW]`.
//!
//! Positions: token `t` is position `pos[t]`, the words the embedding launch
//! writes, `pos[t] = pos[0] + t`. Tap `k` of token `t` reads `ngv` of the
//! position `d = (TAPS − 1 − k)·DILATION` back: the call's row `t − d` when
//! `d <= t`; otherwise zero when `pos[t] < d` (the sequence start, whatever
//! the ring holds), else ring slot `(pos[t] − d) mod RING_ROWS`. The call
//! writes `ngv` of each of its last `min(m, RING_ROWS)` tokens into slot
//! `pos[t] mod RING_ROWS`. The ring keeps [`HIST`] positions plus a pass of
//! [`PASS_ROWS`], so a call from any position of the last pass reads its
//! predecessors — a rollback needs no copy. Unit 0 of a channel is tokens
//! `0 .. min(m, HIST)`, every token that can read the ring, and it writes the
//! ring after its reads; unit `u >= 1` is token `u + HIST − 1`, whose taps
//! are all the call's rows. So the ring's channel `c` is touched by one
//! thread, and a column's arithmetic does not depend on `m`. `m` is any count
//! — a decode step, a verify pass, or a whole ubatch of `T` rows in one launch,
//! whose rows read the batch's own `ngv` inside it and the ring only before
//! its first position, and which leave the ring holding its last
//! [`RING_ROWS`] positions.
//!
//! Numeric contract (the host rules [`gate_host`] and [`conv_host`] are this
//! list):
//! - norm of a row (key, stream, gated value): per thread its values `tid +
//!   RMS_THREADS·c`, `c` ascending, squared by fused multiply-adds, then the
//!   warp butterfly, `rms_warp_tree` and `rms_scale`; then `(v·scale)·gain`,
//!   two roundings (ggml's RMS_NORM, then its MUL);
//! - dot: each product `kn·qn` rounded to f32, summed in f64 per thread in
//!   `c` order, then warp 0 lane `l` adds slots `l, l + 32, …, l + 224` in
//!   order and the lanes meet in the f64 butterfly, rounded once;
//! - `s = dot·(1/√ROW)`, `m = sgn(s)·√max(|s|, 1e-6)` (ggml's `sgn` and
//!   `clamp`), `gate = 1/(1 + e^(−m))` with the exponential in f64 rounded
//!   once — `engram_gate`'s rule;
//! - `gv = value·gate`, one rounding;
//! - conv: each tap's product rounded, the four summed oldest first, each sum
//!   rounded (ggml's MUL per tap and ADD chain), then [`silu`];
//! - `out = x + (gv + conv)`, two roundings.
//!
//! Every f32 product and sum the contract names is an explicit
//! round-to-nearest intrinsic on the device: the compiler contracts a plain
//! `a·b + c` into a fused multiply-add.
//!
//! No silent failure: a key or stream whose sum of squares is not finite,
//! a dot that is not finite, a gated value whose sum of squares is not
//! finite (the gate's block), a conv sum or an output that is not finite, or
//! a position other than `pos[0] + t` (the conv) raises [`FaultSite::Ple`],
//! and the values it reaches are NaN, never the zero a vanishing scale or
//! the ½ a NaN dot's `sgn` would give.

use crate::GpuError;
use crate::elem::{RMS_THREADS, RMS_WARPS, rms_scale, rms_warp_tree};
use crate::fault::{FaultSink, FaultSite};
use crate::launch_u32;
use crate::linear::{PASS_ROWS, butterfly_f32, butterfly_f64, silu};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::float::{add_rn_f32, div_rn_f32, fma_rn_f32, mul_rn_f32, sqrt_rn_f32};
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread, warp,
};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Values in one stream row: Qwen3.8's `n_embd`, and so the rows the PLE
/// table's heads fill together. The gate holds a row per block in registers,
/// so the width is fixed here and the launchers refuse any other.
pub const ROW: usize = 2560;
/// Values of a row one gate thread holds.
pub const PER_THREAD: usize = ROW / RMS_THREADS;
/// Values a thread loads together; five groups make its share of a row.
const GROUP: usize = 2;
const _: () = assert!(PER_THREAD * RMS_THREADS == ROW && PER_THREAD == 5 * GROUP);
/// The gate's block width, [`RMS_THREADS`]: the norm is `elem::rms_norm`'s
/// tree, which is written for that many warps.
const BLOCK: u32 = 256;
const _: () = assert!(BLOCK as usize == RMS_THREADS && RMS_WARPS == 8);
/// ggml's `clamp` floor under `|s|` before the square root.
pub const CLAMP_MIN: f32 = 1e-6;
/// Conv taps (`ple.conv_kernel`).
pub const TAPS: usize = 4;
/// Positions between two taps: the n-gram size (`ple.ngram_size`), the
/// architecture's rule.
pub const DILATION: usize = 3;
/// Positions the conv reaches back.
pub const HIST: usize = (TAPS - 1) * DILATION;
/// Slots of the conv ring: the reach plus the widest pass a later call may
/// roll back into.
pub const RING_ROWS: usize = HIST + PASS_ROWS;
/// Threads per block of the conv: one channel each.
const CONV_BLOCK: u32 = 128;
const _: () = assert!(TAPS == 4 && DILATION == 3 && HIST == 9 && RING_ROWS == 17);

/// The conv's units of a call of `m` tokens per channel: unit 0 takes the
/// first `min(m, HIST)` tokens, every later token a unit of its own.
#[inline(always)]
#[must_use]
pub fn units(m: usize) -> usize {
    m.max(HIST) - (HIST - 1)
}

// ------------------------------------------------------------ gate cores

/// Group `i` of thread `tid`'s share of the row at `base`: the values at
/// `base + tid + RMS_THREADS·(GROUP·i + j)`, `j` ascending.
///
/// # Safety
///
/// `base + ROW <= x.len()`, `tid < RMS_THREADS` and `i < 5`.
#[inline(always)]
unsafe fn pair(x: &[f32], base: usize, tid: usize, i: usize) -> [f32; GROUP] {
    let b = base + tid + RMS_THREADS * GROUP * i;
    // SAFETY: the largest index is base + tid + RMS_THREADS·(GROUP·i + 1) <
    // base + RMS_THREADS·PER_THREAD = base + ROW <= x.len() by this
    // function's contract.
    unsafe { [*x.get_unchecked(b), *x.get_unchecked(b + RMS_THREADS)] }
}

/// Store group `i` of thread `tid`'s share of the row at `base` — the
/// positions [`pair`] reads.
///
/// # Safety
///
/// `base + ROW <= y.len()`, `tid < RMS_THREADS`, `i < 5`, and no other thread
/// writes these positions.
#[inline(always)]
unsafe fn store_pair(
    y: &mut DisjointSlice<f32>,
    base: usize,
    tid: usize,
    i: usize,
    v: [f32; GROUP],
) {
    let b = base + tid + RMS_THREADS * GROUP * i;
    // SAFETY: the largest index is base + tid + RMS_THREADS·(GROUP·i + 1) <
    // base + ROW <= y.len(), and the positions are this thread's alone, by
    // this function's contract.
    unsafe {
        *y.get_unchecked_mut(b) = v[0];
        *y.get_unchecked_mut(b + RMS_THREADS) = v[1];
    }
}

/// `acc` plus the squares of a group in order, one fused multiply-add each.
#[inline(always)]
fn sum_sq(acc: f32, v: [f32; GROUP]) -> f32 {
    let acc = fma_rn_f32(v[0], v[0], acc);
    fma_rn_f32(v[1], v[1], acc)
}

/// `(v·scale)·gain` per lane, each product rounded.
#[inline(always)]
fn norm_gain(v: [f32; GROUP], scale: f32, gain: [f32; GROUP]) -> [f32; GROUP] {
    [
        mul_rn_f32(mul_rn_f32(v[0], scale), gain[0]),
        mul_rn_f32(mul_rn_f32(v[1], scale), gain[1]),
    ]
}

/// `acc` plus the products `kn·qn` of a group in order, each rounded to f32
/// before it is widened.
#[inline(always)]
fn dot_pair(acc: f64, kn: [f32; GROUP], qn: [f32; GROUP]) -> f64 {
    let acc = acc + f64::from(mul_rn_f32(kn[0], qn[0]));
    acc + f64::from(mul_rn_f32(kn[1], qn[1]))
}

/// `v·g` per lane, rounded.
#[inline(always)]
fn scaled(v: [f32; GROUP], g: f32) -> [f32; GROUP] {
    [mul_rn_f32(v[0], g), mul_rn_f32(v[1], g)]
}

/// ggml's `sgn`: 1, −1, or 0 for zero and NaN.
#[inline(always)]
fn ggml_sgn(v: f32) -> f32 {
    if v > 0.0 {
        1.0
    } else if v < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// The gate of one stream from its f64 dot: `s = dot·inv_sqrt_n`, `m =
/// sgn(s)·√(clamp(|s|, CLAMP_MIN, ∞))` with ggml's `clamp` spelled as its C
/// macros (`MAX(MIN(v, hi), lo)`), then `1/(1 + e^(−m))`, the exponential in
/// f64 rounded once.
#[inline(always)]
fn gate_of(dot: f64, inv_sqrt_n: f32) -> f32 {
    let s = mul_rn_f32(dot as f32, inv_sqrt_n);
    let a = s.abs();
    let lo = if a < f32::INFINITY { a } else { f32::INFINITY };
    let c = if lo > CLAMP_MIN { lo } else { CLAMP_MIN };
    let m = mul_rn_f32(ggml_sgn(s), sqrt_rn_f32(c));
    let e = f64::from(-m).exp() as f32;
    div_rn_f32(1.0, add_rn_f32(1.0, e))
}

/// A warp's eight slots of the block-shared array at `ws`.
///
/// # Safety
///
/// `ws` points at eight block-shared f32s, every one written before the
/// barrier that precedes this read.
#[inline(always)]
unsafe fn slots(ws: *const f32) -> [f32; RMS_WARPS] {
    // SAFETY: eight slots from `ws` by this function's contract.
    unsafe {
        [
            *ws.add(0),
            *ws.add(1),
            *ws.add(2),
            *ws.add(3),
            *ws.add(4),
            *ws.add(5),
            *ws.add(6),
            *ws.add(7),
        ]
    }
}

/// The block's sums of the per-thread partials `a` and `b`: each warp's
/// butterfly into its slot of `ws` (`a`'s eight slots, then `b`'s), the
/// barrier, then `rms_warp_tree` of each in every thread. Every thread of
/// the block calls this.
///
/// # Safety
///
/// `ws` points at this block's 16-slot shared array, and no other access to
/// it is in flight.
#[inline(always)]
unsafe fn block_sums2(a: f32, b: f32, tid: usize, ws: *mut f32) -> (f32, f32) {
    let wa = warp::reduce_sum_f32(a);
    let wb = warp::reduce_sum_f32(b);
    if warp::lane_id() == 0 {
        // SAFETY: block-shared, 2·RMS_WARPS slots, RMS_WARPS == blockDim.x /
        // 32; one lane per warp writes its own two slots before the barrier
        // that publishes them.
        unsafe {
            *ws.add(tid / 32) = wa;
            *ws.add(RMS_WARPS + tid / 32) = wb;
        }
    }
    thread::sync_threads();
    // SAFETY: both halves written before the barrier above.
    unsafe {
        (
            rms_warp_tree(slots(ws)),
            rms_warp_tree(slots(ws.add(RMS_WARPS))),
        )
    }
}

/// [`block_sums2`] of one partial, over an eight-slot array.
///
/// # Safety
///
/// `ws` points at this block's [`RMS_WARPS`]-slot shared array, and no
/// other access to it is in flight.
#[inline(always)]
unsafe fn block_sum(a: f32, tid: usize, ws: *mut f32) -> f32 {
    let wa = warp::reduce_sum_f32(a);
    if warp::lane_id() == 0 {
        // SAFETY: block-shared, one lane per warp writes its own slot
        // before the barrier that publishes it.
        unsafe {
            *ws.add(tid / 32) = wa;
        }
    }
    thread::sync_threads();
    // SAFETY: every slot written before the barrier above.
    rms_warp_tree(unsafe { slots(ws) })
}

// ------------------------------------------------------------ conv core

/// Tap input of token `t` (position `p`) at distance `d` on channel `c`:
/// the call's row `t − d`, zero before the sequence start, else the ring.
///
/// # Safety
///
/// `t < m`; `ngv` holds `m·ch` values and `ring` `RING_ROWS·ch`; `c < ch`;
/// the ring is read only by unit 0 of this channel, before its writes.
#[inline(always)]
unsafe fn tap(
    ngv: &[f32],
    ring: *const f32,
    t: usize,
    p: usize,
    d: usize,
    c: usize,
    ch: usize,
) -> f32 {
    if d <= t {
        // SAFETY: t − d < m, so (t − d)·ch + c < m·ch <= ngv.len().
        unsafe { *ngv.get_unchecked((t - d) * ch + c) }
    } else if p < d {
        0.0
    } else {
        // SAFETY: the slot is below RING_ROWS, so the index is inside the
        // ring's RING_ROWS·ch values.
        unsafe { *ring.add((p - d) % RING_ROWS * ch + c) }
    }
}

// ---------------------------------------------------------------- kernels

#[cuda_module]
mod ple_kernels {
    use super::*;

    /// Block `b` is stream `s = b % hc` of token `t = b / hc`: it
    /// normalizes key row `key[b·ROW ..]` and stream `x[b·ROW ..]` (gains
    /// `gk`, `gq` row `s`), dots them in f64, turns the dot into the gate
    /// ([`gate_of`]) — stored at `gate[b]` — writes `gv[b·ROW + d] =
    /// value[t·ROW + d]·gate` and `ngv`, `gv` normalized and scaled by `gc`
    /// row `s`. The block guard is uniform, so no barrier and no warp
    /// collective is skipped.
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
            key.len() >= m * hc * 2560,
            value.len() >= m * 2560,
            x.len() >= m * hc * 2560,
            gk.len() >= hc * 2560,
            gq.len() >= hc * 2560,
            gc.len() >= hc * 2560,
            gv.len() >= m * hc * 2560,
            ngv.len() >= m * hc * 2560,
            gate.len() >= m * hc
        )
    )]
    pub fn ple_gate(
        key: &[f32],
        value: &[f32],
        x: &[f32],
        gk: &[f32],
        gq: &[f32],
        gc: &[f32],
        eps: f32,
        inv_sqrt_n: f32,
        hc: u32,
        m: u32,
        fault: FaultSink,
        mut gv: DisjointSlice<f32>,
        mut ngv: DisjointSlice<f32>,
        mut gate: DisjointSlice<f32>,
    ) {
        static mut WSUM: SharedArray<f32, { 2 * RMS_WARPS }> = SharedArray::UNINIT;
        static mut WSUM_G: SharedArray<f32, RMS_WARPS> = SharedArray::UNINIT;
        static mut DOT: SharedArray<f64, RMS_THREADS> = SharedArray::UNINIT;
        static mut GATE: SharedArray<f32, 1> = SharedArray::UNINIT;

        let b = thread::blockIdx_x() as usize;
        let hc = hc as usize;
        if b >= hc * m as usize {
            return;
        }
        let (t, s) = (b / hc, b % hc);
        let tid = thread::threadIdx_x() as usize;
        let rb = b * ROW;
        let vb = t * ROW;
        let gb = s * ROW;
        // SAFETY: b < hc·m, so rb + ROW <= m·hc·ROW bounds key and x; s < hc,
        // so gb + ROW <= hc·ROW bounds gk and gq — by the launch contract.
        // tid < RMS_THREADS, every group index is below 5.
        let (k0, k1, k2, k3, k4, x0, x1, x2, x3, x4) = unsafe {
            (
                pair(key, rb, tid, 0),
                pair(key, rb, tid, 1),
                pair(key, rb, tid, 2),
                pair(key, rb, tid, 3),
                pair(key, rb, tid, 4),
                pair(x, rb, tid, 0),
                pair(x, rb, tid, 1),
                pair(x, rb, tid, 2),
                pair(x, rb, tid, 3),
                pair(x, rb, tid, 4),
            )
        };
        // SAFETY: as above for gk and gq; t < m, so vb + ROW <= m·ROW bounds
        // value (launch contract).
        let (a0, a1, a2, a3, a4, q0, q1, q2, q3, q4) = unsafe {
            (
                pair(gk, gb, tid, 0),
                pair(gk, gb, tid, 1),
                pair(gk, gb, tid, 2),
                pair(gk, gb, tid, 3),
                pair(gk, gb, tid, 4),
                pair(gq, gb, tid, 0),
                pair(gq, gb, tid, 1),
                pair(gq, gb, tid, 2),
                pair(gq, gb, tid, 3),
                pair(gq, gb, tid, 4),
            )
        };
        // SAFETY: t < m, so vb + ROW <= m·ROW <= value.len() (launch
        // contract); tid < RMS_THREADS, every group index is below 5.
        let (v0, v1, v2, v3, v4) = unsafe {
            (
                pair(value, vb, tid, 0),
                pair(value, vb, tid, 1),
                pair(value, vb, tid, 2),
                pair(value, vb, tid, 3),
                pair(value, vb, tid, 4),
            )
        };

        let pk = sum_sq(sum_sq(sum_sq(sum_sq(sum_sq(0.0, k0), k1), k2), k3), k4);
        let px = sum_sq(sum_sq(sum_sq(sum_sq(sum_sq(0.0, x0), x1), x2), x3), x4);
        // SAFETY: WSUM is this block's own shared allocation, reached only
        // through `block_sums2`.
        let (sk, sx) =
            unsafe { block_sums2(pk, px, tid, SharedArray::as_raw_mut_ptr(&raw mut WSUM)) };
        let norms_ok = sk.is_finite() & sx.is_finite();
        let sck = rms_scale(sk, ROW as u32, eps);
        let scx = rms_scale(sx, ROW as u32, eps);

        let d = dot_pair(0.0, norm_gain(k0, sck, a0), norm_gain(x0, scx, q0));
        let d = dot_pair(d, norm_gain(k1, sck, a1), norm_gain(x1, scx, q1));
        let d = dot_pair(d, norm_gain(k2, sck, a2), norm_gain(x2, scx, q2));
        let d = dot_pair(d, norm_gain(k3, sck, a3), norm_gain(x3, scx, q3));
        let d = dot_pair(d, norm_gain(k4, sck, a4), norm_gain(x4, scx, q4));
        // SAFETY: DOT and GATE are this block's own shared allocations; the
        // raw form is the only way to reach them without a reference to a
        // `static mut`. Every access below is inside their lengths and
        // ordered by `sync_threads`.
        let (dots, gs) = unsafe {
            (
                SharedArray::as_raw_mut_ptr(&raw mut DOT),
                SharedArray::as_raw_mut_ptr(&raw mut GATE),
            )
        };
        // SAFETY: block-shared, RMS_THREADS == blockDim.x, each thread writes
        // its own slot before the barrier that publishes it.
        unsafe {
            *dots.add(tid) = d;
        }
        thread::sync_threads();
        if tid < 32 {
            // Warp 0: lane `l` sums slots l, l + 32, …, l + 224 in order, then
            // the lanes meet in the xor butterfly (16, 8, 4, 2, 1).
            // SAFETY: block-shared, tid + 224 < RMS_THREADS == blockDim.x,
            // every slot written before the barrier above.
            let (d0, d1, d2, d3, d4, d5, d6, d7) = unsafe {
                (
                    *dots.add(tid),
                    *dots.add(tid + 32),
                    *dots.add(tid + 64),
                    *dots.add(tid + 96),
                    *dots.add(tid + 128),
                    *dots.add(tid + 160),
                    *dots.add(tid + 192),
                    *dots.add(tid + 224),
                )
            };
            let mut acc = ((((((d0 + d1) + d2) + d3) + d4) + d5) + d6) + d7;
            acc += warp::shuffle_xor_f64(acc, 16);
            acc += warp::shuffle_xor_f64(acc, 8);
            acc += warp::shuffle_xor_f64(acc, 4);
            acc += warp::shuffle_xor_f64(acc, 2);
            acc += warp::shuffle_xor_f64(acc, 1);
            if tid == 0 {
                let g = if norms_ok & acc.is_finite() {
                    gate_of(acc, inv_sqrt_n)
                } else {
                    f32::NAN
                };
                // SAFETY: block-shared, one slot, written by thread 0 alone
                // before the barrier that publishes it.
                unsafe {
                    *gs = g;
                }
                // SAFETY: b < hc·m <= gate.len() by the launch contract; only
                // thread 0 of block b writes gate[b].
                unsafe {
                    *gate.get_unchecked_mut(b) = g;
                }
            }
        }
        thread::sync_threads();
        // SAFETY: block-shared, one slot, written before the barrier above.
        let g = unsafe { *gs };

        let (w0, w1, w2, w3, w4) = (
            scaled(v0, g),
            scaled(v1, g),
            scaled(v2, g),
            scaled(v3, g),
            scaled(v4, g),
        );
        let pg = sum_sq(sum_sq(sum_sq(sum_sq(sum_sq(0.0, w0), w1), w2), w3), w4);
        // SAFETY: WSUM_G is this block's own shared allocation, reached only
        // through `block_sum`.
        let sg = unsafe { block_sum(pg, tid, SharedArray::as_raw_mut_ptr(&raw mut WSUM_G)) };
        let scg = if sg.is_finite() {
            rms_scale(sg, ROW as u32, eps)
        } else {
            f32::NAN
        };
        // SAFETY: s < hc, so gb + ROW <= hc·ROW <= gc.len() (launch
        // contract); tid < RMS_THREADS, every group index is below 5.
        let (c0, c1, c2, c3, c4) = unsafe {
            (
                pair(gc, gb, tid, 0),
                pair(gc, gb, tid, 1),
                pair(gc, gb, tid, 2),
                pair(gc, gb, tid, 3),
                pair(gc, gb, tid, 4),
            )
        };
        // SAFETY: b < hc·m, so rb + ROW <= m·hc·ROW bounds gv and ngv (launch
        // contract); tid < RMS_THREADS, every group index is below 5, and row
        // b's positions of this thread are written by this thread alone.
        unsafe {
            store_pair(&mut gv, rb, tid, 0, w0);
            store_pair(&mut gv, rb, tid, 1, w1);
            store_pair(&mut gv, rb, tid, 2, w2);
            store_pair(&mut gv, rb, tid, 3, w3);
            store_pair(&mut gv, rb, tid, 4, w4);
            store_pair(&mut ngv, rb, tid, 0, norm_gain(w0, scg, c0));
            store_pair(&mut ngv, rb, tid, 1, norm_gain(w1, scg, c1));
            store_pair(&mut ngv, rb, tid, 2, norm_gain(w2, scg, c2));
            store_pair(&mut ngv, rb, tid, 3, norm_gain(w3, scg, c3));
            store_pair(&mut ngv, rb, tid, 4, norm_gain(w4, scg, c4));
        }
        if tid == 0 && !(g.is_finite() & sg.is_finite()) {
            fault.raise(FaultSite::Ple);
        }
    }

    /// Thread `(u, c)` — unit `u`, channel `c` of `hc·ROW` — runs its unit's
    /// tokens (module doc): the conv of `ngv` over the taps, [`silu`], and
    /// `out[t·ch + c] = x[t·ch + c] + (gv[t·ch + c] + conv)`; unit 0
    /// afterwards writes `ngv` of the call's last `min(m, RING_ROWS)` tokens
    /// into the ring. A token whose conv sum or output is not finite, or
    /// whose position is not `pos[0] + t`, gets NaN and raises.
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
            m >= 1,
            ngv.len() >= m * hc * 2560,
            gv.len() >= m * hc * 2560,
            x.len() >= m * hc * 2560,
            w.len() >= 4 * hc * 2560,
            pos.len() >= m,
            out.len() >= m * hc * 2560,
            ring.len() >= 17 * hc * 2560
        )
    )]
    pub fn ple_conv(
        ngv: &[f32],
        gv: &[f32],
        x: &[f32],
        w: &[f32],
        pos: &[u32],
        hc: u32,
        m: u32,
        fault: FaultSink,
        mut out: DisjointSlice<f32>,
        mut ring: DisjointSlice<f32>,
    ) {
        let ch = hc as usize * ROW;
        let m = m as usize;
        let gid =
            thread::blockIdx_x() as usize * CONV_BLOCK as usize + thread::threadIdx_x() as usize;
        if gid >= units(m) * ch {
            return;
        }
        let u = gid / ch;
        let c = gid - u * ch;
        let (t0, t1) = if u == 0 {
            (0, m.min(HIST))
        } else {
            (u + HIST - 1, u + HIST)
        };
        // SAFETY: c < ch, so TAPS·c + 3 < TAPS·ch <= w.len() (launch
        // contract).
        let (w0, w1, w2, w3) = unsafe {
            (
                *w.get_unchecked(TAPS * c),
                *w.get_unchecked(TAPS * c + 1),
                *w.get_unchecked(TAPS * c + 2),
                *w.get_unchecked(TAPS * c + 3),
            )
        };
        let ring = ring.as_mut_ptr();
        let outp = out.as_mut_ptr();
        // SAFETY: pos holds m >= 1 words by the contract.
        let p0 = u64::from(unsafe { *pos.get_unchecked(0) });
        let mut ok = true;
        let mut t = t0;
        while t < t1 {
            // SAFETY: t < m <= pos.len().
            let pw = unsafe { *pos.get_unchecked(t) };
            let p = pw as usize;
            // SAFETY: t < m, c < ch; ngv holds m·ch values and the ring
            // RING_ROWS·ch; the ring is read only when d > t, which holds only
            // for t < HIST — unit 0's tokens — before unit 0's writes below.
            let (i0, i1, i2) = unsafe {
                (
                    tap(ngv, ring, t, p, 3 * DILATION, c, ch),
                    tap(ngv, ring, t, p, 2 * DILATION, c, ch),
                    tap(ngv, ring, t, p, DILATION, c, ch),
                )
            };
            let at = t * ch + c;
            // SAFETY: at < m·ch, inside ngv, gv and x by the launch contract.
            let (i3, gvv, xv) = unsafe {
                (
                    *ngv.get_unchecked(at),
                    *gv.get_unchecked(at),
                    *x.get_unchecked(at),
                )
            };
            let acc = mul_rn_f32(i0, w0);
            let acc = add_rn_f32(acc, mul_rn_f32(i1, w1));
            let acc = add_rn_f32(acc, mul_rn_f32(i2, w2));
            let acc = add_rn_f32(acc, mul_rn_f32(i3, w3));
            let o = add_rn_f32(xv, add_rn_f32(gvv, silu(acc)));
            let good = (u64::from(pw) == p0 + t as u64) & acc.is_finite() & o.is_finite();
            // SAFETY: at < m·ch <= out.len(); (t, c) is this thread's alone.
            unsafe {
                *outp.add(at) = if good { o } else { f32::NAN };
            }
            ok &= good;
            t += 1;
        }
        if u == 0 {
            // `ngv` of the last min(m, RING_ROWS) tokens into their slots,
            // after every read of the ring above.
            let mut t = m.saturating_sub(RING_ROWS);
            while t < m {
                // SAFETY: t < m <= pos.len().
                let slot = unsafe { *pos.get_unchecked(t) } as usize % RING_ROWS;
                // SAFETY: slot < RING_ROWS and c < ch: inside the ring
                // (RING_ROWS·ch values) and row t of ngv (m·ch values);
                // channel c of the ring is this thread's alone.
                unsafe { *ring.add(slot * ch + c) = *ngv.get_unchecked(t * ch + c) };
                t += 1;
            }
        }
        if !ok {
            fault.raise(FaultSite::Ple);
        }
    }
}

// -------------------------------------------------------------- launchers

/// [`PleKernels::enqueue_gate`]'s arguments: `m` tokens' keys `key`
/// (`hc·ROW` each) and values `value` (`ROW` each), the streams `x` (`hc·ROW`
/// each), the three gains (`hc·ROW` f32 each, stream-major), and the
/// outputs: `gv` and `ngv` in `x`'s layout, and `gate`, the `m·hc` gates.
pub struct PleGateArgs<'a> {
    pub key: &'a DeviceBuffer<f32>,
    pub value: &'a DeviceBuffer<f32>,
    pub x: &'a DeviceBuffer<f32>,
    pub gain_key: &'a DeviceBuffer<f32>,
    pub gain_query: &'a DeviceBuffer<f32>,
    pub gain_conv: &'a DeviceBuffer<f32>,
    pub eps: f32,
    pub hc: usize,
    pub m: usize,
    pub fault: FaultSink,
    pub gv: &'a mut DeviceBuffer<f32>,
    pub ngv: &'a mut DeviceBuffer<f32>,
    pub gate: &'a mut DeviceBuffer<f32>,
}

/// [`PleKernels::enqueue_conv`]'s arguments: the gate's `ngv` and `gv`, the
/// streams `x`, the conv weight `w` (`TAPS·hc·ROW`, tap-fastest as the file
/// stores `ple_conv1d`), the call's position words `pos` (`m`), the conv's
/// taps and dilation as the model states them (refused unless [`TAPS`] and
/// [`DILATION`]), and the outputs: `out` in `x`'s layout, and the ring
/// ([`ring_len`] values, read and written in place).
pub struct PleConvArgs<'a> {
    pub ngv: &'a DeviceBuffer<f32>,
    pub gv: &'a DeviceBuffer<f32>,
    pub x: &'a DeviceBuffer<f32>,
    pub w: &'a DeviceBuffer<f32>,
    pub pos: &'a DeviceBuffer<u32>,
    pub taps: usize,
    pub dilation: usize,
    pub hc: usize,
    pub m: usize,
    pub fault: FaultSink,
    pub out: &'a mut DeviceBuffer<f32>,
    pub ring: &'a mut DeviceBuffer<f32>,
}

/// `1/√ROW` as ik computes the score's scale: `1.0f / sqrtf((float) n_embd)`.
#[must_use]
pub fn inv_sqrt_row() -> f32 {
    1.0 / (ROW as f32).sqrt()
}

/// f32s of one sequence's conv ring for `hc` streams: `RING_ROWS·hc·ROW`.
#[must_use]
pub fn ring_len(hc: usize) -> usize {
    RING_ROWS * hc * ROW
}

/// The loaded PLE module. Owns no stream: each enqueue takes the engine
/// stream, so its launches order with the rest of the step and capture.
pub struct PleKernels {
    module: ple_kernels::LoadedModule,
}

impl PleKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<PleKernels, GpuError> {
        // SAFETY: this crate owns the embedded device bundle produced for the
        // module above; the launchers check its launch contracts.
        let module = unsafe { crate::shared_module!(ple_kernels, ctx)? };
        Ok(PleKernels { module })
    }

    /// Enqueue the gate: `hc·m` blocks of [`BLOCK`] threads, per the module
    /// contract. Asynchronous, allocation-free, capturable.
    pub fn enqueue_gate(&self, stream: &CudaStream, a: PleGateArgs<'_>) -> Result<(), GpuError> {
        const WHAT: &str = "ple enqueue_gate";
        let blocks = check_rows(WHAT, a.hc, a.m)?;
        need(WHAT, "key", a.key.len(), blocks * ROW)?;
        need(WHAT, "value", a.value.len(), a.m * ROW)?;
        need(WHAT, "x", a.x.len(), blocks * ROW)?;
        need(WHAT, "gain_key", a.gain_key.len(), a.hc * ROW)?;
        need(WHAT, "gain_query", a.gain_query.len(), a.hc * ROW)?;
        need(WHAT, "gain_conv", a.gain_conv.len(), a.hc * ROW)?;
        need(WHAT, "gv", a.gv.len(), blocks * ROW)?;
        need(WHAT, "ngv", a.ngv.len(), blocks * ROW)?;
        need(WHAT, "gate", a.gate.len(), blocks)?;
        let grid = launch_u32(WHAT, "hc·m", blocks)?;
        let hc = launch_u32(WHAT, "hc", a.hc)?;
        let m = launch_u32(WHAT, "m", a.m)?;
        let prep = self
            .module
            .prepare_ple_gate(LaunchConfig1D::new(grid, BLOCK, 0))?;
        self.module.ple_gate(
            stream,
            &prep,
            a.key,
            a.value,
            a.x,
            a.gain_key,
            a.gain_query,
            a.gain_conv,
            a.eps,
            inv_sqrt_row(),
            hc,
            m,
            a.fault,
            a.gv,
            a.ngv,
            a.gate,
        )?;
        Ok(())
    }

    /// Enqueue the conv and the stream update: `⌈units(m)·hc·ROW / 128⌉`
    /// blocks of 128 threads, the ring read and written in place.
    /// Asynchronous, allocation-free, capturable.
    pub fn enqueue_conv(&self, stream: &CudaStream, a: PleConvArgs<'_>) -> Result<(), GpuError> {
        const WHAT: &str = "ple enqueue_conv";
        if (a.taps, a.dilation) != (TAPS, DILATION) {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "the conv is built for {TAPS} taps {DILATION} positions apart; the model \
                     states {} taps {} apart",
                    a.taps, a.dilation
                ),
            ));
        }
        let blocks = check_rows(WHAT, a.hc, a.m)?;
        let ch = a.hc * ROW;
        need(WHAT, "ngv", a.ngv.len(), blocks * ROW)?;
        need(WHAT, "gv", a.gv.len(), blocks * ROW)?;
        need(WHAT, "x", a.x.len(), blocks * ROW)?;
        need(WHAT, "w", a.w.len(), TAPS * ch)?;
        need(WHAT, "pos", a.pos.len(), a.m)?;
        need(WHAT, "out", a.out.len(), blocks * ROW)?;
        need(WHAT, "ring", a.ring.len(), ring_len(a.hc))?;
        let threads = units(a.m).checked_mul(ch).ok_or_else(|| {
            GpuError::shape(WHAT, format!("units·channels overflows at m={}", a.m))
        })?;
        let grid = launch_u32(WHAT, "grid", threads.div_ceil(CONV_BLOCK as usize))?;
        let hc = launch_u32(WHAT, "hc", a.hc)?;
        let m = launch_u32(WHAT, "m", a.m)?;
        let prep = self
            .module
            .prepare_ple_conv(LaunchConfig1D::new(grid, CONV_BLOCK, 0))?;
        self.module.ple_conv(
            stream, &prep, a.ngv, a.gv, a.x, a.w, a.pos, hc, m, a.fault, a.out, a.ring,
        )?;
        Ok(())
    }

    /// Zero a sequence's conv ring (a model's reset). The conv reads zero
    /// before the sequence start whatever the ring holds; this keeps a
    /// ring's content a function of the current sequence alone. Asynchronous
    /// on `stream`, capturable as a memset node.
    pub fn reset_ring(
        &self,
        stream: &CudaStream,
        ring: &mut DeviceBuffer<f32>,
        hc: usize,
    ) -> Result<(), GpuError> {
        need("ple reset_ring", "ring", ring.len(), ring_len(hc))?;
        ring.zero_async(stream)?;
        Ok(())
    }
}

/// The launch's block count `hc·m`, refusing an empty launch.
fn check_rows(what: &'static str, hc: usize, m: usize) -> Result<usize, GpuError> {
    if hc == 0 || m == 0 {
        return Err(GpuError::shape(
            what,
            format!("need hc >= 1 and m >= 1, got hc={hc} m={m}"),
        ));
    }
    hc.checked_mul(m)
        .ok_or_else(|| GpuError::shape(what, format!("hc·m = {hc}·{m} overflows")))
}

/// Refuse a buffer `name` of `len` elements shorter than `want`.
fn need(what: &'static str, name: &str, len: usize, want: usize) -> Result<(), GpuError> {
    if len < want {
        return Err(GpuError::shape(
            what,
            format!("{name}.len() {len} < {want}"),
        ));
    }
    Ok(())
}

// ------------------------------------------------------------ host rules

/// The gate's outputs, host side: `gv` and `ngv` `[ROW, hc, m]`, `gate`
/// `[hc, m]`.
#[derive(Clone, Debug, PartialEq)]
pub struct GateOut {
    pub gv: Vec<f32>,
    pub ngv: Vec<f32>,
    pub gate: Vec<f32>,
}

/// Thread `tid`'s values of a row, `tid + RMS_THREADS·c` for `c` ascending.
fn thread_values(tid: usize) -> impl Iterator<Item = usize> {
    (0..PER_THREAD).map(move |c| tid + RMS_THREADS * c)
}

/// The block sum of squares of `row`, the kernels' tree: per thread its
/// values' squares by fused multiply-adds in order, the butterfly per warp,
/// `rms_warp_tree`.
fn host_sum_sq(row: &[f32]) -> f32 {
    let part: Vec<f32> = (0..RMS_THREADS)
        .map(|tid| thread_values(tid).fold(0.0f32, |acc, i| row[i].mul_add(row[i], acc)))
        .collect();
    let mut sums = [0.0f32; RMS_WARPS];
    for (s, lanes) in sums.iter_mut().zip(part.as_chunks::<32>().0) {
        *s = butterfly_f32(lanes);
    }
    rms_warp_tree(sums)
}

/// [`gate_of`] on the host, op for op.
fn host_gate_of(dot: f64) -> f32 {
    let s = (dot as f32) * inv_sqrt_row();
    let a = s.abs();
    let lo = if a < f32::INFINITY { a } else { f32::INFINITY };
    let c = if lo > CLAMP_MIN { lo } else { CLAMP_MIN };
    let sgn = if s > 0.0 {
        1.0
    } else if s < 0.0 {
        -1.0
    } else {
        0.0
    };
    let m = sgn * c.sqrt();
    let e = f64::from(-m).exp() as f32;
    1.0 / (1.0 + e)
}

/// The host rule of [`PleKernels::enqueue_gate`]: the module doc's numeric
/// contract, op for op, on host slices in [`PleGateArgs`]'s layouts.
#[allow(
    clippy::too_many_arguments,
    reason = "the launch's inputs, as host slices"
)]
#[must_use]
pub fn gate_host(
    key: &[f32],
    value: &[f32],
    x: &[f32],
    gk: &[f32],
    gq: &[f32],
    gc: &[f32],
    eps: f32,
    hc: usize,
    m: usize,
) -> GateOut {
    let mut out = GateOut {
        gv: vec![0.0; m * hc * ROW],
        ngv: vec![0.0; m * hc * ROW],
        gate: vec![0.0; m * hc],
    };
    for b in 0..m * hc {
        let (t, s) = (b / hc, b % hc);
        let kr = &key[b * ROW..][..ROW];
        let xr = &x[b * ROW..][..ROW];
        let vr = &value[t * ROW..][..ROW];
        let (gkr, gqr, gcr) = (
            &gk[s * ROW..][..ROW],
            &gq[s * ROW..][..ROW],
            &gc[s * ROW..][..ROW],
        );
        let (sk, sx) = (host_sum_sq(kr), host_sum_sq(xr));
        let (sck, scx) = (
            rms_scale(sk, ROW as u32, eps),
            rms_scale(sx, ROW as u32, eps),
        );
        let dots: Vec<f64> = (0..RMS_THREADS)
            .map(|tid| {
                thread_values(tid).fold(0.0f64, |acc, i| {
                    let kn = (kr[i] * sck) * gkr[i];
                    let qn = (xr[i] * scx) * gqr[i];
                    acc + f64::from(kn * qn)
                })
            })
            .collect();
        let mut lanes = [0.0f64; 32];
        for (l, v) in lanes.iter_mut().enumerate() {
            *v = (1..RMS_WARPS).fold(dots[l], |acc, wp| acc + dots[l + 32 * wp]);
        }
        let dot = butterfly_f64(&lanes);
        let g = if sk.is_finite() && sx.is_finite() && dot.is_finite() {
            host_gate_of(dot)
        } else {
            f32::NAN
        };
        out.gate[b] = g;
        let gvr = &mut out.gv[b * ROW..][..ROW];
        for (o, &v) in gvr.iter_mut().zip(vr) {
            *o = v * g;
        }
        let sg = host_sum_sq(gvr);
        let scg = if sg.is_finite() {
            rms_scale(sg, ROW as u32, eps)
        } else {
            f32::NAN
        };
        for ((o, &v), &gn) in out.ngv[b * ROW..][..ROW].iter_mut().zip(&*gvr).zip(gcr) {
            *o = (v * scg) * gn;
        }
    }
    out
}

/// The conv's outputs, host side: `out` `[ROW, hc, m]` and the ring after
/// the call.
#[derive(Clone, Debug, PartialEq)]
pub struct ConvOut {
    pub out: Vec<f32>,
    pub ring: Vec<f32>,
}

/// The host rule of [`PleKernels::enqueue_conv`]: the module doc's conv,
/// position and ring rule op for op, on host slices in [`PleConvArgs`]'s
/// layouts; `ring` is the ring before the call.
#[allow(
    clippy::too_many_arguments,
    reason = "the launch's inputs, as host slices"
)]
#[must_use]
pub fn conv_host(
    ngv: &[f32],
    gv: &[f32],
    x: &[f32],
    w: &[f32],
    pos: &[u32],
    ring: &[f32],
    hc: usize,
    m: usize,
) -> ConvOut {
    let ch = hc * ROW;
    let mut out = vec![0.0f32; m * ch];
    for t in 0..m {
        let p = pos[t] as usize;
        let contiguous = u64::from(pos[t]) == u64::from(pos[0]) + t as u64;
        for c in 0..ch {
            let input = |d: usize| {
                if d <= t {
                    ngv[(t - d) * ch + c]
                } else if p < d {
                    0.0
                } else {
                    ring[(p - d) % RING_ROWS * ch + c]
                }
            };
            let mut acc = input(HIST) * w[TAPS * c];
            for k in 1..TAPS {
                acc += input((TAPS - 1 - k) * DILATION) * w[TAPS * c + k];
            }
            let at = t * ch + c;
            let o = x[at] + (gv[at] + silu(acc));
            out[at] = if contiguous && acc.is_finite() && o.is_finite() {
                o
            } else {
                f32::NAN
            };
        }
    }
    let mut ring = ring[..ring_len(hc)].to_vec();
    for t in m.saturating_sub(RING_ROWS)..m {
        let slot = pos[t] as usize % RING_ROWS;
        ring[slot * ch..(slot + 1) * ch].copy_from_slice(&ngv[t * ch..(t + 1) * ch]);
    }
    ConvOut { out, ring }
}
