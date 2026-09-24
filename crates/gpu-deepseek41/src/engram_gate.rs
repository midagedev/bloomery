//! The engram gate: the looked-up rows, projected into a key per stream copy
//! and one shared value, gate the residual update — gate = sigmoid of the
//! signed square root of the query–key dot, and h += gate ⊗ value.
//!
//! `engram_wkv` (the dense gemv of its type, [`crate::dense`]) turns a
//! token's looked-up rows into `kv`: `hc` keys, one per stream, then the
//! value the streams share, each [`ROW`] values. Two launches follow it, cut by what they wait for:
//! - [`EngramGateKernels::enqueue_key_norm`] reads `kv` alone. The rows depend
//!   on the token id and nothing else, so this launch and the gemv can run
//!   before the previous layer has finished;
//! - [`EngramGateKernels::enqueue_gate`] reads the previous layer's streams:
//!   their norm, the per-stream dot with the normalized key, the gate, and
//!   the update `out = x + value·gate`.
//!
//! One [`RMS_THREADS`] block per (token, stream). A block holds its row in
//! registers, [`PER_THREAD`] values a thread at `tid + RMS_THREADS·c` for `c`
//! ascending, and issues every load of a launch before the first use.
//!
//! Layouts, token-major: `kv` `[ROW·(hc+1), m]`, the streams `x` and `out`
//! and the normalized key `kn` `[ROW, hc, m]`, the gains `[ROW, hc]` in f32,
//! `gate` `[hc, m]`.
//!
//! Numeric contract: ik's CPU graph (`ds4_build_engram`) op for op, except
//! the norm's sum of squares.
//! - Norm: `elem::rms_norm`'s tree — per thread its values in `c` order, one
//!   fused multiply-add each, then the warp butterfly, `rms_warp_tree` and
//!   `rms_scale` — where ik sums the squares in f64. Then `(v·scale)·gain`,
//!   two roundings: ik's RMS_NORM followed by its MUL.
//! - Dot: each product `kn·qn` rounded to f32, summed in f64 (per thread in
//!   `c` order, then the block's fixed tree) and rounded once. ik's SUM_ROWS
//!   sums in f64 serially; two sums that close to the exact one round to the
//!   same f32 except within an f64 ulp of a tie.
//! - `s = dot·(1/√ROW)`, `m = sgn(s)·√max(|s|, 1e-6)` (ggml's `sgn`, and its
//!   `clamp` as the C macros), `gate = 1/(1 + e^(−m))` with the exponential
//!   taken in f64 and rounded once — the correctly rounded `expf` except
//!   within an f64 ulp of a tie.
//! - `out = x + value·gate`.
//!
//! Every f32 product and sum whose rounding the contract names is an
//! explicit round-to-nearest intrinsic: the compiler contracts a plain
//! `a·b + c` into a fused multiply-add, and fuses `f64::from(a·b) + c` into
//! an f64 one.

use bloomery_gpu::elem::{RMS_THREADS, RMS_WARPS, rms_scale, rms_warp_tree};
use bloomery_gpu::{GpuError, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::float::{add_rn_f32, div_rn_f32, fma_rn_f32, mul_rn_f32, sqrt_rn_f32};
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread, warp,
};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Values in one stream row: V4.1's `n_embd`. The kernels hold a row per
/// block in registers, so the width is fixed here and the launchers refuse
/// any other.
pub const ROW: usize = 5120;
/// Values of a row one thread holds.
pub const PER_THREAD: usize = ROW / RMS_THREADS;
const _: () = assert!(PER_THREAD * RMS_THREADS == ROW && PER_THREAD == 5 * GROUP);
/// Values a thread loads together; five groups make its share of a row.
const GROUP: usize = 4;
/// The launch's block width, [`RMS_THREADS`]: the norm is `elem::rms_norm`'s
/// tree, which is written for that many warps.
const BLOCK: u32 = 256;
const _: () = assert!(BLOCK as usize == RMS_THREADS && RMS_WARPS == 8);
/// ggml's `clamp` floor under `|s|` before the square root.
pub const CLAMP_MIN: f32 = 1e-6;

// ------------------------------------------------------------------ cores

/// Group `i` of thread `tid`'s share of the row at `base`: the values at
/// `base + tid + RMS_THREADS·(GROUP·i + j)`, `j` ascending.
///
/// # Safety
///
/// `base + ROW <= x.len()`, `tid < RMS_THREADS` and `i < 5`.
#[inline(always)]
unsafe fn group(x: &[f32], base: usize, tid: usize, i: usize) -> [f32; GROUP] {
    let b = base + tid + RMS_THREADS * GROUP * i;
    // SAFETY: the largest index is base + tid + RMS_THREADS·(GROUP·i + 3) <
    // base + RMS_THREADS·PER_THREAD = base + ROW <= x.len() by this
    // function's contract.
    unsafe {
        [
            *x.get_unchecked(b),
            *x.get_unchecked(b + RMS_THREADS),
            *x.get_unchecked(b + 2 * RMS_THREADS),
            *x.get_unchecked(b + 3 * RMS_THREADS),
        ]
    }
}

/// Store group `i` of thread `tid`'s share of the row at `base` — the
/// positions [`group`] reads.
///
/// # Safety
///
/// `base + ROW <= y.len()`, `tid < RMS_THREADS`, `i < 5`, and no other thread
/// writes these positions.
#[inline(always)]
unsafe fn store_group(
    y: &mut DisjointSlice<f32>,
    base: usize,
    tid: usize,
    i: usize,
    v: [f32; GROUP],
) {
    let b = base + tid + RMS_THREADS * GROUP * i;
    // SAFETY: the largest index is base + tid + RMS_THREADS·(GROUP·i + 3) <
    // base + ROW <= y.len(), and the positions are this thread's alone, by
    // this function's contract.
    unsafe {
        *y.get_unchecked_mut(b) = v[0];
        *y.get_unchecked_mut(b + RMS_THREADS) = v[1];
        *y.get_unchecked_mut(b + 2 * RMS_THREADS) = v[2];
        *y.get_unchecked_mut(b + 3 * RMS_THREADS) = v[3];
    }
}

/// `acc` plus the squares of a group in order, one fused multiply-add each.
#[inline(always)]
fn sum_sq(acc: f32, v: [f32; GROUP]) -> f32 {
    let acc = fma_rn_f32(v[0], v[0], acc);
    let acc = fma_rn_f32(v[1], v[1], acc);
    let acc = fma_rn_f32(v[2], v[2], acc);
    fma_rn_f32(v[3], v[3], acc)
}

/// `(v·scale)·gain` per lane, each product rounded.
#[inline(always)]
fn norm_gain(v: [f32; GROUP], scale: f32, gain: [f32; GROUP]) -> [f32; GROUP] {
    [
        mul_rn_f32(mul_rn_f32(v[0], scale), gain[0]),
        mul_rn_f32(mul_rn_f32(v[1], scale), gain[1]),
        mul_rn_f32(mul_rn_f32(v[2], scale), gain[2]),
        mul_rn_f32(mul_rn_f32(v[3], scale), gain[3]),
    ]
}

/// `acc` plus the products `kn·qn` of a group in order, each rounded to f32
/// before it is widened, with `qn = (x·scale)·gain`.
#[inline(always)]
fn dot_group(acc: f64, kn: [f32; GROUP], x: [f32; GROUP], scale: f32, gain: [f32; GROUP]) -> f64 {
    let qn = norm_gain(x, scale, gain);
    let acc = acc + f64::from(mul_rn_f32(kn[0], qn[0]));
    let acc = acc + f64::from(mul_rn_f32(kn[1], qn[1]));
    let acc = acc + f64::from(mul_rn_f32(kn[2], qn[2]));
    acc + f64::from(mul_rn_f32(kn[3], qn[3]))
}

/// `x + value·gate` per lane, product and sum rounded on their own.
#[inline(always)]
fn update(x: [f32; GROUP], value: [f32; GROUP], gate: f32) -> [f32; GROUP] {
    [
        add_rn_f32(x[0], mul_rn_f32(value[0], gate)),
        add_rn_f32(x[1], mul_rn_f32(value[1], gate)),
        add_rn_f32(x[2], mul_rn_f32(value[2], gate)),
        add_rn_f32(x[3], mul_rn_f32(value[3], gate)),
    ]
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

/// The row's norm scale from each thread's partial sum of squares: the warp
/// butterfly, one slot per warp in `wsum`, the barrier, then `rms_warp_tree`
/// and `rms_scale` in every thread. Every thread of the block calls this.
///
/// # Safety
///
/// `wsum` points at this block's [`RMS_WARPS`]-slot shared array, and no
/// other access to it is in flight.
#[inline(always)]
unsafe fn block_scale(part: f32, tid: usize, wsum: *mut f32, eps: f32) -> f32 {
    let w = warp::reduce_sum_f32(part);
    if warp::lane_id() == 0 {
        // SAFETY: block-shared, RMS_WARPS == blockDim.x / 32, one lane per
        // warp writes its own slot tid / 32 before the barrier that publishes
        // it.
        unsafe {
            *wsum.add(tid / 32) = w;
        }
    }
    thread::sync_threads();
    // SAFETY: block-shared, RMS_WARPS slots, every one written before the
    // barrier above.
    let sums = unsafe {
        [
            *wsum.add(0),
            *wsum.add(1),
            *wsum.add(2),
            *wsum.add(3),
            *wsum.add(4),
            *wsum.add(5),
            *wsum.add(6),
            *wsum.add(7),
        ]
    };
    rms_scale(rms_warp_tree(sums), ROW as u32, eps)
}

// ---------------------------------------------------------------- kernels

#[cuda_module]
mod engram_gate_kernels {
    use super::*;

    /// The key side: block `b` normalizes key `s = b % hc` of token `t = b /
    /// hc` — `kv[t·(hc+1)·ROW + s·ROW ..]` — and scales it by gain row `s`,
    /// into `kn[b·ROW ..]`. The block guard is uniform, so no barrier and no
    /// warp collective is skipped.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            kv.len() >= m * (hc + 1) * 5120,
            gain.len() >= hc * 5120,
            kn.len() >= m * hc * 5120
        )
    )]
    pub fn ds41_engram_key_norm(
        kv: &[f32],
        gain: &[f32],
        eps: f32,
        hc: u32,
        m: u32,
        mut kn: DisjointSlice<f32>,
    ) {
        static mut WSUM: SharedArray<f32, RMS_WARPS> = SharedArray::UNINIT;

        let b = thread::blockIdx_x() as usize;
        let hc = hc as usize;
        if b >= hc * m as usize {
            return;
        }
        let (t, s) = (b / hc, b % hc);
        let tid = thread::threadIdx_x() as usize;
        let kb = (t * (hc + 1) + s) * ROW;
        let gb = s * ROW;
        // SAFETY: t < m and s < hc, so kb + ROW <= m·(hc+1)·ROW <= kv.len()
        // and gb + ROW <= hc·ROW <= gain.len() by the launch contract; tid <
        // RMS_THREADS; every group index is below 5.
        let (k0, k1, k2, k3, k4, g0, g1, g2, g3, g4) = unsafe {
            (
                group(kv, kb, tid, 0),
                group(kv, kb, tid, 1),
                group(kv, kb, tid, 2),
                group(kv, kb, tid, 3),
                group(kv, kb, tid, 4),
                group(gain, gb, tid, 0),
                group(gain, gb, tid, 1),
                group(gain, gb, tid, 2),
                group(gain, gb, tid, 3),
                group(gain, gb, tid, 4),
            )
        };
        let part = sum_sq(sum_sq(sum_sq(sum_sq(sum_sq(0.0, k0), k1), k2), k3), k4);
        // SAFETY: WSUM is this block's own shared allocation, reached only
        // through `block_scale`.
        let scale =
            unsafe { block_scale(part, tid, SharedArray::as_raw_mut_ptr(&raw mut WSUM), eps) };
        let ob = b * ROW;
        // SAFETY: b < hc·m, so ob + ROW <= m·hc·ROW <= kn.len() by the launch
        // contract; tid < RMS_THREADS, every group index is below 5, and row
        // b's positions of this thread are written by this thread alone.
        unsafe {
            store_group(&mut kn, ob, tid, 0, norm_gain(k0, scale, g0));
            store_group(&mut kn, ob, tid, 1, norm_gain(k1, scale, g1));
            store_group(&mut kn, ob, tid, 2, norm_gain(k2, scale, g2));
            store_group(&mut kn, ob, tid, 3, norm_gain(k3, scale, g3));
            store_group(&mut kn, ob, tid, 4, norm_gain(k4, scale, g4));
        }
    }

    /// The query side and the update: block `b` is stream `s = b % hc` of
    /// token `t = b / hc`. It normalizes `x[b·ROW ..]` and scales it by gain
    /// row `s`, dots it with `kn[b·ROW ..]` in f64, turns the dot into the
    /// gate ([`gate_of`]) — stored at `gate[b]` — and writes `out[b·ROW + d]
    /// = x[b·ROW + d] + value[d]·gate`, the value at `kv[t·(hc+1)·ROW +
    /// hc·ROW ..]`. The block guard is uniform, so no barrier and no warp
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
            x.len() >= m * hc * 5120,
            kn.len() >= m * hc * 5120,
            kv.len() >= m * (hc + 1) * 5120,
            gain.len() >= hc * 5120,
            out.len() >= m * hc * 5120,
            gate.len() >= m * hc
        )
    )]
    pub fn ds41_engram_gate(
        x: &[f32],
        kn: &[f32],
        kv: &[f32],
        gain: &[f32],
        eps: f32,
        inv_sqrt_n: f32,
        hc: u32,
        m: u32,
        mut out: DisjointSlice<f32>,
        mut gate: DisjointSlice<f32>,
    ) {
        static mut WSUM: SharedArray<f32, RMS_WARPS> = SharedArray::UNINIT;
        static mut DOT: SharedArray<f64, RMS_THREADS> = SharedArray::UNINIT;
        static mut GATE: SharedArray<f32, 1> = SharedArray::UNINIT;

        let b = thread::blockIdx_x() as usize;
        let hc = hc as usize;
        if b >= hc * m as usize {
            return;
        }
        let (t, s) = (b / hc, b % hc);
        let tid = thread::threadIdx_x() as usize;
        let xb = b * ROW;
        let vb = (t * (hc + 1) + hc) * ROW;
        let gb = s * ROW;
        // SAFETY: b < hc·m, so xb + ROW <= m·hc·ROW bounds x and kn; t < m, so
        // vb + ROW <= m·(hc+1)·ROW bounds kv; s < hc, so gb + ROW <= hc·ROW
        // bounds gain — all by the launch contract. tid < RMS_THREADS, every
        // group index is below 5.
        let (x0, x1, x2, x3, x4, n0, n1, n2, n3, n4) = unsafe {
            (
                group(x, xb, tid, 0),
                group(x, xb, tid, 1),
                group(x, xb, tid, 2),
                group(x, xb, tid, 3),
                group(x, xb, tid, 4),
                group(kn, xb, tid, 0),
                group(kn, xb, tid, 1),
                group(kn, xb, tid, 2),
                group(kn, xb, tid, 3),
                group(kn, xb, tid, 4),
            )
        };
        // SAFETY: t < m, so vb + ROW <= m·(hc+1)·ROW bounds kv; s < hc, so
        // gb + ROW <= hc·ROW bounds gain (launch contract); tid < RMS_THREADS,
        // every group index is below 5.
        let (v0, v1, v2, v3, v4, g0, g1, g2, g3, g4) = unsafe {
            (
                group(kv, vb, tid, 0),
                group(kv, vb, tid, 1),
                group(kv, vb, tid, 2),
                group(kv, vb, tid, 3),
                group(kv, vb, tid, 4),
                group(gain, gb, tid, 0),
                group(gain, gb, tid, 1),
                group(gain, gb, tid, 2),
                group(gain, gb, tid, 3),
                group(gain, gb, tid, 4),
            )
        };

        let part = sum_sq(sum_sq(sum_sq(sum_sq(sum_sq(0.0, x0), x1), x2), x3), x4);
        // SAFETY: WSUM is this block's own shared allocation, reached only
        // through `block_scale`.
        let scale =
            unsafe { block_scale(part, tid, SharedArray::as_raw_mut_ptr(&raw mut WSUM), eps) };

        let d = dot_group(0.0, n0, x0, scale, g0);
        let d = dot_group(d, n1, x1, scale, g1);
        let d = dot_group(d, n2, x2, scale, g2);
        let d = dot_group(d, n3, x3, scale, g3);
        let d = dot_group(d, n4, x4, scale, g4);
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
                let g = gate_of(acc, inv_sqrt_n);
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

        // SAFETY: b < hc·m, so xb + ROW <= m·hc·ROW <= out.len() by the
        // launch contract; tid < RMS_THREADS, every group index is below 5,
        // and row b's positions of this thread are written by this thread
        // alone.
        unsafe {
            store_group(&mut out, xb, tid, 0, update(x0, v0, g));
            store_group(&mut out, xb, tid, 1, update(x1, v1, g));
            store_group(&mut out, xb, tid, 2, update(x2, v2, g));
            store_group(&mut out, xb, tid, 3, update(x3, v3, g));
            store_group(&mut out, xb, tid, 4, update(x4, v4, g));
        }
    }
}

// -------------------------------------------------------------- launchers

/// [`EngramGateKernels::enqueue_key_norm`]'s arguments: `m` tokens' `kv`
/// (`(hc+1)·ROW` values each, token-major), the key gains `gain` (`hc·ROW`
/// f32, stream-major), and `kn` for the `m·hc` normalized keys.
pub struct KeyNormArgs<'a> {
    pub kv: &'a DeviceBuffer<f32>,
    pub gain: &'a DeviceBuffer<f32>,
    pub eps: f32,
    pub hc: usize,
    pub m: usize,
    pub kn: &'a mut DeviceBuffer<f32>,
}

/// [`EngramGateKernels::enqueue_gate`]'s arguments: the streams `x` (`m·hc`
/// rows of ROW, token-major), the normalized keys `kn` from
/// [`EngramGateKernels::enqueue_key_norm`], the same `kv` (its value rows),
/// the query gains `gain`, and the outputs: `out`, the updated streams in
/// `x`'s layout, and `gate`, the `m·hc` gates.
pub struct GateArgs<'a> {
    pub x: &'a DeviceBuffer<f32>,
    pub kn: &'a DeviceBuffer<f32>,
    pub kv: &'a DeviceBuffer<f32>,
    pub gain: &'a DeviceBuffer<f32>,
    pub eps: f32,
    pub hc: usize,
    pub m: usize,
    pub out: &'a mut DeviceBuffer<f32>,
    pub gate: &'a mut DeviceBuffer<f32>,
}

/// `1/√ROW` as ik computes the score's scale: `1.0f / sqrtf((float) n_embd)`.
#[must_use]
pub fn inv_sqrt_row() -> f32 {
    1.0 / (ROW as f32).sqrt()
}

/// The loaded engram gate module. Owns no stream: each enqueue takes the
/// engine stream, so its launches order with the rest of the step and
/// capture.
pub struct EngramGateKernels {
    module: engram_gate_kernels::LoadedModule,
}

impl EngramGateKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<EngramGateKernels, GpuError> {
        // SAFETY: this crate owns the embedded device bundle produced for the
        // module above; the launchers check its launch contracts.
        let module = unsafe { engram_gate_kernels::load(ctx)? };
        Ok(EngramGateKernels { module })
    }

    /// Enqueue the key side: for key `s` of token `t`,
    /// `a.kn[(t·hc + s)·ROW + d] = (key·scale)·gain[s·ROW + d]`, its norm per
    /// the module contract. Asynchronous, allocation-free, capturable.
    pub fn enqueue_key_norm(
        &self,
        stream: &CudaStream,
        a: KeyNormArgs<'_>,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "enqueue_key_norm";
        let blocks = check_rows(WHAT, a.hc, a.m)?;
        need(WHAT, "kv", a.kv.len(), a.m * (a.hc + 1) * ROW)?;
        need(WHAT, "gain", a.gain.len(), a.hc * ROW)?;
        need(WHAT, "kn", a.kn.len(), blocks * ROW)?;
        let grid = launch_u32(WHAT, "hc·m", blocks)?;
        let hc = launch_u32(WHAT, "hc", a.hc)?;
        let m = launch_u32(WHAT, "m", a.m)?;
        let prep = self
            .module
            .prepare_ds41_engram_key_norm(LaunchConfig1D::new(grid, BLOCK, 0))?;
        self.module
            .ds41_engram_key_norm(stream, &prep, a.kv, a.gain, a.eps, hc, m, a.kn)?;
        Ok(())
    }

    /// Enqueue the query side and the update: per stream `s` of token `t`,
    /// the gate from the dot of `x`'s normalized row with `kn`'s, into
    /// `a.gate[t·hc + s]`, and `a.out = x + value·gate` over the row.
    /// Asynchronous, allocation-free, capturable.
    pub fn enqueue_gate(&self, stream: &CudaStream, a: GateArgs<'_>) -> Result<(), GpuError> {
        const WHAT: &str = "enqueue_gate";
        let blocks = check_rows(WHAT, a.hc, a.m)?;
        need(WHAT, "x", a.x.len(), blocks * ROW)?;
        need(WHAT, "kn", a.kn.len(), blocks * ROW)?;
        need(WHAT, "kv", a.kv.len(), a.m * (a.hc + 1) * ROW)?;
        need(WHAT, "gain", a.gain.len(), a.hc * ROW)?;
        need(WHAT, "out", a.out.len(), blocks * ROW)?;
        need(WHAT, "gate", a.gate.len(), blocks)?;
        let grid = launch_u32(WHAT, "hc·m", blocks)?;
        let hc = launch_u32(WHAT, "hc", a.hc)?;
        let m = launch_u32(WHAT, "m", a.m)?;
        let prep = self
            .module
            .prepare_ds41_engram_gate(LaunchConfig1D::new(grid, BLOCK, 0))?;
        self.module.ds41_engram_gate(
            stream,
            &prep,
            a.x,
            a.kn,
            a.kv,
            a.gain,
            a.eps,
            inv_sqrt_row(),
            hc,
            m,
            a.out,
            a.gate,
        )?;
        Ok(())
    }
}

/// The launch's block count `hc·m`, refusing an empty launch.
fn check_rows(what: &'static str, hc: usize, m: usize) -> Result<usize, GpuError> {
    if hc == 0 || m == 0 {
        return Err(GpuError::Shape {
            what,
            detail: format!("need hc >= 1 and m >= 1, got hc={hc} m={m}"),
        });
    }
    hc.checked_mul(m).ok_or_else(|| GpuError::Shape {
        what,
        detail: format!("hc·m = {hc}·{m} overflows"),
    })
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
