//! The input norms and the projections of one attention input in one launch: up to three
//! input rows, each RMS-normed with its gain, and eight one-row projections of them — five over
//! Q8_0 planes and three over F32 rows — each output bit for bit what `rms_norm` followed by the
//! site's own gemv (`q8_0_gemv`, `f32_gemv`) writes.
//!
//! The sites, in block order: the Q8_0 `q`, `k`, `v`, `rel` and `gate` and the F32 `sink`, `kt`
//! and `vt` (a site of no row is absent). `q`, `rel`, `gate` and `sink` read input 0, `k` and
//! `kt` input `ik`, `v` and `vt` input `iv` (`ik`, `iv` below 3). Every site's rows are a multiple
//! of [`ROWS`], so a block's rows are one site's.
//!
//! A block of [`THREADS`] threads first norms its site's input into shared memory as `rms_norm`
//! does: thread `tid` sums the squares of values `tid, tid + 256, …` (`acc += v · v` in that
//! order), the warp butterfly, the fixed eight-warp tree, `rms_scale_checked`, and each value
//! `(scale · gain) · x`. Then each warp walks one row with the site's own gemv lane walk
//! ([`q8_0_lane_partial_1col`], [`f32_lane_partial_1col`]) over the normed row and the warp
//! butterfly, and lane 0 writes the sum. Every block of a site norms the same input in the same
//! order, so every block's normed row is `rms_norm`'s bits, and every output the site's gemv's.
//!
//! No silent failure: a sum of squares that is not finite and positive raises
//! [`crate::fault::FaultSite`]'s norm site through `rms_scale_checked` and the normed row, so
//! every output of that input, is NaN.

use crate::elem::{RMS_THREADS, RMS_WARPS, rms_scale_checked, rms_warp_tree};
use crate::fault::FaultSink;
use crate::q8f32::{f32_lane_partial_1col, q8_0_lane_partial_1col};
use crate::{DeviceTensor, GpuError, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread, warp,
};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Threads a block: `rms_norm`'s block, eight warps.
pub const THREADS: usize = 256;
const WARPS: usize = THREADS / 32;
/// Rows a block: one a warp.
pub const ROWS: usize = WARPS;
/// The widest input row a launch norms (its shared row).
pub const MAX_K: usize = 3072;
/// The sites a launch projects: five Q8_0, three F32.
pub const SITES: usize = 8;
const _: () = assert!(THREADS == RMS_THREADS && WARPS == RMS_WARPS && WARPS == 8);
// The entry's `launch_bounds` and `launch_contract` block are literals.
const _: () = assert!(THREADS == 256 && MAX_K == 3072);

/// One thread's sum of squares over `x[..k]`'s values `tid, tid + 256, …`, `acc += v · v` in
/// that order — `rms_partial_sq`'s sum — nine strides' loads at a time.
///
/// # Safety
///
/// `x.len() >= k`.
#[inline(always)]
unsafe fn sum_sq(x: &[f32], k: usize, tid: usize) -> f32 {
    let mut acc = 0.0f32;
    let mut it = tid;
    while it + 8 * THREADS < k {
        // SAFETY: it + 8 · THREADS < k <= x.len() (the loop condition, this fn's contract).
        let v = unsafe {
            [
                *x.get_unchecked(it),
                *x.get_unchecked(it + THREADS),
                *x.get_unchecked(it + 2 * THREADS),
                *x.get_unchecked(it + 3 * THREADS),
                *x.get_unchecked(it + 4 * THREADS),
                *x.get_unchecked(it + 5 * THREADS),
                *x.get_unchecked(it + 6 * THREADS),
                *x.get_unchecked(it + 7 * THREADS),
                *x.get_unchecked(it + 8 * THREADS),
            ]
        };
        acc += v[0] * v[0];
        acc += v[1] * v[1];
        acc += v[2] * v[2];
        acc += v[3] * v[3];
        acc += v[4] * v[4];
        acc += v[5] * v[5];
        acc += v[6] * v[6];
        acc += v[7] * v[7];
        acc += v[8] * v[8];
        it += 9 * THREADS;
    }
    while it < k {
        // SAFETY: it < k <= x.len().
        let v = unsafe { *x.get_unchecked(it) };
        acc += v * v;
        it += THREADS;
    }
    acc
}

#[cuda_module]
mod norm_proj_kernels {
    use super::*;

    /// The norms and the eight projections (module doc): block `b` is rows `8·(b − first) ..`
    /// of the site whose blocks hold `b`, `first` that site's first block.
    #[allow(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        reason = "kernel entry: the device ABI takes the eight sites' planes and outputs flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            k >= 32,
            k <= 3072,
            x0.len() >= k,
            x1.len() >= k,
            x2.len() >= k,
            g0.len() >= k,
            g1.len() >= k,
            g2.len() >= k,
            ik < 3,
            iv < 3,
            4 * qs_q.len() >= rq * k,
            32 * d_q.len() >= rq * k,
            4 * qs_k.len() >= rk * k,
            32 * d_k.len() >= rk * k,
            4 * qs_v.len() >= rv * k,
            32 * d_v.len() >= rv * k,
            4 * qs_r.len() >= rr * k,
            32 * d_r.len() >= rr * k,
            4 * qs_g.len() >= rg * k,
            32 * d_g.len() >= rg * k,
            w_s.len() >= rs * k,
            w_kt.len() >= rkt * k,
            w_vt.len() >= rvt * k,
            y_q.len() >= rq,
            y_k.len() >= rk,
            y_v.len() >= rv,
            y_r.len() >= rr,
            y_g.len() >= rg,
            y_s.len() >= rs,
            y_kt.len() >= rkt,
            y_vt.len() >= rvt
        )
    )]
    pub fn norm_proj_q8_f32(
        x0: &[f32],
        x1: &[f32],
        x2: &[f32],
        g0: &[f32],
        g1: &[f32],
        g2: &[f32],
        ik: u32,
        iv: u32,
        eps: f32,
        k: u32,
        fault: FaultSink,
        qs_q: &[u32],
        d_q: &[u16],
        qs_k: &[u32],
        d_k: &[u16],
        qs_v: &[u32],
        d_v: &[u16],
        qs_r: &[u32],
        d_r: &[u16],
        qs_g: &[u32],
        d_g: &[u16],
        w_s: &[f32],
        w_kt: &[f32],
        w_vt: &[f32],
        rq: u32,
        rk: u32,
        rv: u32,
        rr: u32,
        rg: u32,
        rs: u32,
        rkt: u32,
        rvt: u32,
        mut y_q: DisjointSlice<f32>,
        mut y_k: DisjointSlice<f32>,
        mut y_v: DisjointSlice<f32>,
        mut y_r: DisjointSlice<f32>,
        mut y_g: DisjointSlice<f32>,
        mut y_s: DisjointSlice<f32>,
        mut y_kt: DisjointSlice<f32>,
        mut y_vt: DisjointSlice<f32>,
    ) {
        static mut XN: SharedArray<f32, MAX_K, 16> = SharedArray::UNINIT;
        static mut WSUM: SharedArray<f32, RMS_WARPS> = SharedArray::UNINIT;

        let blk = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        let k_us = k as usize;
        // The site whose blocks hold this block, and the block's first row in it.
        let rows = [rq, rk, rv, rr, rg, rs, rkt, rvt];
        let mut site = 0usize;
        let mut first = 0usize;
        while site < SITES && blk >= first + rows[site] as usize / ROWS {
            first += rows[site] as usize / ROWS;
            site += 1;
        }
        if site >= SITES {
            return; // block-uniform: past the last site's blocks
        }
        let input = match site {
            1 | 6 => ik,
            2 | 7 => iv,
            _ => 0,
        };
        let (x, g) = match input {
            0 => (x0, g0),
            1 => (x1, g1),
            _ => (x2, g2),
        };
        // SAFETY: XN and WSUM are this block's own shared allocations; the raw form is the only
        // way to reach them without a reference to a `static mut`. Every access is below its
        // length (k <= MAX_K by the contract, tid / 32 < RMS_WARPS) and ordered by
        // `sync_threads`.
        let xn = unsafe { SharedArray::as_raw_mut_ptr(&raw mut XN) };
        // SAFETY: as above.
        let ws = unsafe { SharedArray::as_raw_mut_ptr(&raw mut WSUM) };
        // SAFETY: x.len() >= k by the contract.
        let part = warp::reduce_sum_f32(unsafe { sum_sq(x, k_us, tid) });
        if warp::lane_id() == 0 {
            // SAFETY: tid / 32 < RMS_WARPS; one lane a warp writes its slot.
            unsafe { *ws.add(tid / 32) = part };
        }
        thread::sync_threads();
        // SAFETY: every slot was written before the barrier above.
        let sums = unsafe {
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
        };
        let scale = rms_scale_checked(rms_warp_tree(sums), k, eps, fault, tid == 0);
        let mut j = tid;
        while j < k_us {
            // SAFETY: j < k <= x.len(), g.len() and MAX_K; this thread alone writes value j.
            unsafe { *xn.add(j) = (scale * *g.get_unchecked(j)) * *x.get_unchecked(j) };
            j += THREADS;
        }
        thread::sync_threads();
        // SAFETY: the k values of XN were written before the barrier above; nothing writes them
        // past it.
        let xs: &[f32] = unsafe { core::slice::from_raw_parts(xn.cast_const(), k_us) };
        let lane = warp::lane_id() as usize;
        let row = (blk - first) * ROWS + tid / 32;
        // Each arm: the site's gemv lane walk of `row` (row < the site's rows: the block's rows
        // are the site's, ROWS | rows), the butterfly, lane 0's store into the site's output.
        // SAFETY (every arm): row < the site's rows puts the row's words and scales (or values)
        // inside its planes by the contract, xs holds k values, k a multiple of 32 (host-checked)
        // and lane < 32; y has the site's rows by the contract, and only lane 0 of the row's warp
        // writes slot row.
        let s = match site {
            // SAFETY: as above.
            0 => warp::reduce_sum_f32(unsafe {
                q8_0_lane_partial_1col(qs_q, d_q, xs, k, row, 0, lane)
            }),
            // SAFETY: as above.
            1 => warp::reduce_sum_f32(unsafe {
                q8_0_lane_partial_1col(qs_k, d_k, xs, k, row, 0, lane)
            }),
            // SAFETY: as above.
            2 => warp::reduce_sum_f32(unsafe {
                q8_0_lane_partial_1col(qs_v, d_v, xs, k, row, 0, lane)
            }),
            // SAFETY: as above.
            3 => warp::reduce_sum_f32(unsafe {
                q8_0_lane_partial_1col(qs_r, d_r, xs, k, row, 0, lane)
            }),
            // SAFETY: as above.
            4 => warp::reduce_sum_f32(unsafe {
                q8_0_lane_partial_1col(qs_g, d_g, xs, k, row, 0, lane)
            }),
            // SAFETY: as above.
            5 => warp::reduce_sum_f32(unsafe { f32_lane_partial_1col(w_s, xs, k, row, lane) }),
            // SAFETY: as above.
            6 => warp::reduce_sum_f32(unsafe { f32_lane_partial_1col(w_kt, xs, k, row, lane) }),
            // SAFETY: as above.
            _ => warp::reduce_sum_f32(unsafe { f32_lane_partial_1col(w_vt, xs, k, row, lane) }),
        };
        if lane == 0 {
            // SAFETY: as above.
            unsafe {
                match site {
                    0 => *y_q.get_unchecked_mut(row) = s,
                    1 => *y_k.get_unchecked_mut(row) = s,
                    2 => *y_v.get_unchecked_mut(row) = s,
                    3 => *y_r.get_unchecked_mut(row) = s,
                    4 => *y_g.get_unchecked_mut(row) = s,
                    5 => *y_s.get_unchecked_mut(row) = s,
                    6 => *y_kt.get_unchecked_mut(row) = s,
                    _ => *y_vt.get_unchecked_mut(row) = s,
                }
            }
        }
    }
}

/// The loaded module. Owns no stream: each enqueue takes the engine stream.
pub struct NormProjKernels {
    module: norm_proj_kernels::LoadedModule,
}

/// A Q8_0 site's planes and its output.
pub struct Q8Site<'a> {
    pub qs: &'a DeviceTensor<u32>,
    pub d: &'a DeviceTensor<u16>,
    pub y: &'a mut DeviceBuffer<f32>,
}

/// An F32 site's rows and its output.
pub struct F32Site<'a> {
    pub w: &'a DeviceTensor<f32>,
    pub y: &'a mut DeviceBuffer<f32>,
}

/// One launch's inputs and sites (module doc).
pub struct NormProjArgs<'a> {
    /// The input rows and their gains: one, or three (`ik`, `iv` then name rows 1 and 2).
    pub x: [&'a DeviceBuffer<f32>; 3],
    pub gain: [&'a DeviceBuffer<f32>; 3],
    /// The inputs `k`, `kt` and `v`, `vt` read.
    pub ik: usize,
    pub iv: usize,
    pub eps: f32,
    /// The input width.
    pub k: usize,
    pub fault: FaultSink,
    /// `q`, `k`, `v`, `rel`, `gate`.
    pub q8: [Q8Site<'a>; 5],
    /// `sink`, then the two conv-tap sites: run where `taps` holds, else of no row (their
    /// rows are not read and their outputs not written).
    pub sink: F32Site<'a>,
    pub kt: F32Site<'a>,
    pub vt: F32Site<'a>,
    pub taps: bool,
}

impl NormProjKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<NormProjKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for the module above;
        // the launcher checks its launch contract.
        let module = unsafe { crate::shared_module!(norm_proj_kernels, ctx)? };
        Ok(NormProjKernels { module })
    }

    /// Enqueue the norms and the projections of `a` (module doc): `Σ rows / 8` blocks of 256
    /// threads. A width that is not a positive multiple of 32 or past [`MAX_K`], a site whose
    /// planes are not `k` wide or whose rows are not a multiple of [`ROWS`], an output shorter
    /// than its rows and an input index past 2 are refused by name. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue(&self, stream: &CudaStream, a: NormProjArgs<'_>) -> Result<(), GpuError> {
        const WHAT: &str = "norm_proj::enqueue";
        let bad = |detail: String| Err(GpuError::shape(WHAT, detail));
        let k = a.k;
        if k == 0 || !k.is_multiple_of(32) || k > MAX_K {
            return bad(format!(
                "an input of {k} values: a positive multiple of 32 up to {MAX_K}"
            ));
        }
        if a.ik > 2 || a.iv > 2 {
            return bad(format!("inputs {} and {}: three rows at most", a.ik, a.iv));
        }
        for (i, (x, g)) in a.x.iter().zip(&a.gain).enumerate() {
            if x.len() < k || g.len() < k {
                return bad(format!(
                    "input {i}: {} values and a gain of {}, for {k}",
                    x.len(),
                    g.len()
                ));
            }
        }
        let rows_of = |name: &str, rows: usize, y: usize| -> Result<usize, GpuError> {
            if !rows.is_multiple_of(ROWS) || y < rows {
                return Err(GpuError::shape(
                    WHAT,
                    format!("{name}: {rows} rows (a multiple of {ROWS}) into an output of {y}"),
                ));
            }
            Ok(rows)
        };
        const Q8_NAMES: [&str; 5] = ["q", "k", "v", "rel", "gate"];
        let mut q8_rows = [0usize; 5];
        for (i, s) in a.q8.iter().enumerate() {
            if s.qs.cols() * 4 != k || s.d.cols() * 32 != k || s.qs.rows() != s.d.rows() {
                return bad(format!(
                    "{}: Q8_0 planes of {} words and {} scales a row over {} rows, for {k} values",
                    Q8_NAMES[i],
                    s.qs.cols(),
                    s.d.cols(),
                    s.qs.rows()
                ));
            }
            q8_rows[i] = rows_of(Q8_NAMES[i], s.qs.rows(), s.y.len())?;
        }
        let f32_rows = |name: &str, s: &F32Site<'_>| -> Result<usize, GpuError> {
            if s.w.cols() != k {
                return Err(GpuError::shape(
                    WHAT,
                    format!("{name}: F32 rows of {} values, for {k}", s.w.cols()),
                ));
            }
            rows_of(name, s.w.rows(), s.y.len())
        };
        let rs = f32_rows("sink", &a.sink)?;
        let (rkt, rvt) = if a.taps {
            (f32_rows("kt", &a.kt)?, f32_rows("vt", &a.vt)?)
        } else {
            (0, 0)
        };
        let blocks: usize = (q8_rows.iter().sum::<usize>() + rs + rkt + rvt) / ROWS;
        let u = |name: &'static str, v: usize| launch_u32(WHAT, name, v);
        let NormProjArgs {
            x,
            gain,
            ik,
            iv,
            eps,
            fault,
            q8,
            sink,
            kt,
            vt,
            ..
        } = a;
        let [sq, sk, sv, sr, sg] = q8;
        let (
            F32Site { w: w_s, y: y_s },
            F32Site { w: w_kt, y: y_kt },
            F32Site { w: w_vt, y: y_vt },
        ) = (sink, kt, vt);
        let prep = self.module.prepare_norm_proj_q8_f32(LaunchConfig1D::new(
            u("grid", blocks)?,
            THREADS as u32,
            0,
        ))?;
        self.module.norm_proj_q8_f32(
            stream,
            &prep,
            x[0],
            x[1],
            x[2],
            gain[0],
            gain[1],
            gain[2],
            u("ik", ik)?,
            u("iv", iv)?,
            eps,
            u("k", k)?,
            fault,
            sq.qs.buf(),
            sq.d.buf(),
            sk.qs.buf(),
            sk.d.buf(),
            sv.qs.buf(),
            sv.d.buf(),
            sr.qs.buf(),
            sr.d.buf(),
            sg.qs.buf(),
            sg.d.buf(),
            w_s.buf(),
            w_kt.buf(),
            w_vt.buf(),
            u("rows", q8_rows[0])?,
            u("rows", q8_rows[1])?,
            u("rows", q8_rows[2])?,
            u("rows", q8_rows[3])?,
            u("rows", q8_rows[4])?,
            u("rows", rs)?,
            u("rows", rkt)?,
            u("rows", rvt)?,
            sq.y,
            sk.y,
            sv.y,
            sr.y,
            sg.y,
            y_s,
            y_kt,
            y_vt,
        )?;
        Ok(())
    }
}
