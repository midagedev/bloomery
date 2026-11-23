//! [`crate::norm_proj`]'s launch over BF16 projections: the input norms and the eight
//! projections of one attention input, the five sites `norm_proj_q8_f32` runs over Q8_0 planes
//! (`q`, `k`, `v`, `rel`, `gate`) here over a BF16 weight's words (`Bf16Raw`, two values a
//! word, `k / 2` words a row) and the three F32 sites (`sink`, `kt`, `vt`) as there. Each
//! output is bit for bit what `rms_norm` followed by the site's own gemv (`bf16_gemv`,
//! `f32_gemv`) writes. Two entries:
//!
//! - `norm_proj_bf16_f32`: one input row, `norm_proj_q8_f32`'s block, shared row and order,
//!   each warp walking its row with the gemv's one-column lane walk
//!   ([`f32_lane_partial_1col`] over [`Bf16Words`] or the F32 row).
//! - `norm_proj_bf16_f32_cols`: `m` (1..=[`MAX_COLS`]) input rows of each input, token after
//!   token, each normed as the one-row entry norms its row into a shared block of `m` rows
//!   (`m · k` at most [`COLS_VALUES`]), each warp walking its row with the gemv's `m`-column
//!   lane walk ([`f32_lane_partials`], [`gemv_lane_sums`]); token `c`'s output of row `r`
//!   lands at `y[c · rows + r]`.
//!
//! The site order, the inputs `ik`, `iv` and the block-to-site rule are `norm_proj`'s. No
//! silent failure: a sum of squares that is not finite and positive raises the norm site through
//! `rms_scale_checked` and the normed row, so every output of that input, is NaN.

use crate::elem::{RMS_THREADS, RMS_WARPS, rms_scale_checked, rms_warp_tree};
use crate::fault::FaultSink;
use crate::q8f32::{Bf16Words, f32_lane_partial_1col, f32_lane_partials, gemv_lane_sums};
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
/// The widest input row either entry norms.
pub const MAX_K: usize = 3072;
/// The most tokens the `_cols` entry norms and projects in one launch.
pub const MAX_COLS: usize = 4;
/// The `_cols` entry's shared rows: `m · k` normed values at most.
pub const COLS_VALUES: usize = 11264;
/// The sites a launch projects: five BF16, three F32.
pub const SITES: usize = 8;
const _: () = assert!(THREADS == RMS_THREADS && WARPS == RMS_WARPS && WARPS == 8);
// The entries' `launch_bounds` and `launch_contract` spell these as literals.
const _: () = assert!(THREADS == 256 && MAX_K == 3072 && MAX_COLS == 4 && COLS_VALUES == 11264);

/// One thread's sum of squares over `x[..k]`'s values `tid, tid + 256, …`, `acc += v · v` in
/// that order — `rms_partial_sq`'s sum, as `norm_proj`'s.
///
/// # Safety
///
/// `x.len() >= k`.
#[inline(always)]
unsafe fn sum_sq(x: &[f32], k: usize, tid: usize) -> f32 {
    let mut acc = 0.0f32;
    let mut it = tid;
    while it < k {
        // SAFETY: it < k <= x.len().
        let v = unsafe { *x.get_unchecked(it) };
        acc += v * v;
        it += THREADS;
    }
    acc
}

/// The site whose blocks hold block `blk` (`SITES` past the last), its first block, and the
/// input it reads: `norm_proj`'s block-to-site rule.
#[inline(always)]
fn site_of(blk: usize, rows: [u32; SITES], ik: u32, iv: u32) -> (usize, usize, u32) {
    let mut site = 0usize;
    let mut first = 0usize;
    while site < SITES && blk >= first + rows[site] as usize / ROWS {
        first += rows[site] as usize / ROWS;
        site += 1;
    }
    let input = match site {
        1 | 6 => ik,
        2 | 7 => iv,
        _ => 0,
    };
    (site, first, input)
}

#[cuda_module]
mod norm_proj_bf16_kernels {
    use super::*;

    /// The one-row norms and eight projections (module doc): block `b` is rows `8·(b − first)
    /// ..` of the site whose blocks hold `b`.
    #[allow(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        reason = "kernel entry: the device ABI takes the eight sites' rows and outputs flat (rust-quality R8)"
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
            2 * w_q.len() >= rq * k,
            2 * w_k.len() >= rk * k,
            2 * w_v.len() >= rv * k,
            2 * w_r.len() >= rr * k,
            2 * w_g.len() >= rg * k,
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
    pub fn norm_proj_bf16_f32(
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
        w_q: &[u32],
        w_k: &[u32],
        w_v: &[u32],
        w_r: &[u32],
        w_g: &[u32],
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
        let (site, first, input) = site_of(blk, [rq, rk, rv, rr, rg, rs, rkt, rvt], ik, iv);
        if site >= SITES {
            return; // block-uniform: past the last site's blocks
        }
        let (x, g) = match input {
            0 => (x0, g0),
            1 => (x1, g1),
            _ => (x2, g2),
        };
        // SAFETY: XN and WSUM are this block's own shared allocations, reached raw; every access
        // is below its length (k <= MAX_K by the contract, tid / 32 < RMS_WARPS) and ordered by
        // `sync_threads`.
        let (xn, ws) = unsafe {
            (
                SharedArray::as_raw_mut_ptr(&raw mut XN),
                SharedArray::as_raw_mut_ptr(&raw mut WSUM),
            )
        };
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
        // SAFETY: (every arm) row < the site's rows (the block's rows are the site's, ROWS |
        // rows) puts the row's values inside its words or rows by the contract, xs holds k values,
        // k a multiple of 32 (host-checked) and lane < 32.
        let s = warp::reduce_sum_f32(unsafe {
            match site {
                0 => f32_lane_partial_1col(Bf16Words(w_q), xs, k, row, lane),
                1 => f32_lane_partial_1col(Bf16Words(w_k), xs, k, row, lane),
                2 => f32_lane_partial_1col(Bf16Words(w_v), xs, k, row, lane),
                3 => f32_lane_partial_1col(Bf16Words(w_r), xs, k, row, lane),
                4 => f32_lane_partial_1col(Bf16Words(w_g), xs, k, row, lane),
                5 => f32_lane_partial_1col(w_s, xs, k, row, lane),
                6 => f32_lane_partial_1col(w_kt, xs, k, row, lane),
                _ => f32_lane_partial_1col(w_vt, xs, k, row, lane),
            }
        });
        if lane == 0 {
            // SAFETY: y has the site's rows by the contract, and only lane 0 of the row's warp
            // writes slot row.
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

    /// The `m`-token norms and eight projections (module doc): block `b` is rows `8·(b −
    /// first) ..` of the site whose blocks hold `b`, for every token.
    #[allow(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        reason = "kernel entry: the device ABI takes the eight sites' rows and outputs flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            k >= 32,
            k <= 3072,
            m >= 1,
            m <= 4,
            m * k <= 11264,
            x0.len() >= m * k,
            x1.len() >= m * k,
            x2.len() >= m * k,
            g0.len() >= k,
            g1.len() >= k,
            g2.len() >= k,
            ik < 3,
            iv < 3,
            2 * w_q.len() >= rq * k,
            2 * w_k.len() >= rk * k,
            2 * w_v.len() >= rv * k,
            2 * w_r.len() >= rr * k,
            2 * w_g.len() >= rg * k,
            w_s.len() >= rs * k,
            w_kt.len() >= rkt * k,
            w_vt.len() >= rvt * k,
            y_q.len() >= m * rq,
            y_k.len() >= m * rk,
            y_v.len() >= m * rv,
            y_r.len() >= m * rr,
            y_g.len() >= m * rg,
            y_s.len() >= m * rs,
            y_kt.len() >= m * rkt,
            y_vt.len() >= m * rvt
        )
    )]
    pub fn norm_proj_bf16_f32_cols(
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
        m: u32,
        fault: FaultSink,
        w_q: &[u32],
        w_k: &[u32],
        w_v: &[u32],
        w_r: &[u32],
        w_g: &[u32],
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
        static mut XN: SharedArray<f32, COLS_VALUES, 16> = SharedArray::UNINIT;
        static mut WSUM: SharedArray<f32, { MAX_COLS * RMS_WARPS }> = SharedArray::UNINIT;

        let blk = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        let (k_us, m_us) = (k as usize, m as usize);
        let rows = [rq, rk, rv, rr, rg, rs, rkt, rvt];
        let (site, first, input) = site_of(blk, rows, ik, iv);
        if site >= SITES {
            return; // block-uniform: past the last site's blocks
        }
        let (x, g) = match input {
            0 => (x0, g0),
            1 => (x1, g1),
            _ => (x2, g2),
        };
        // SAFETY: XN and WSUM are this block's own shared allocations, reached raw; every access
        // is below its length (m · k <= COLS_VALUES and m <= MAX_COLS by the contract, tid / 32 <
        // RMS_WARPS) and ordered by `sync_threads`.
        let (xn, ws) = unsafe {
            (
                SharedArray::as_raw_mut_ptr(&raw mut XN),
                SharedArray::as_raw_mut_ptr(&raw mut WSUM),
            )
        };
        let mut c = 0usize;
        while c < m_us {
            // SAFETY: x.len() >= m · k by the contract, so token c's k values follow c · k.
            let part =
                warp::reduce_sum_f32(unsafe { sum_sq(x.get_unchecked(c * k_us..), k_us, tid) });
            if warp::lane_id() == 0 {
                // SAFETY: c · RMS_WARPS + tid / 32 < MAX_COLS · RMS_WARPS; one lane a warp
                // writes its slot.
                unsafe { *ws.add(c * RMS_WARPS + tid / 32) = part };
            }
            c += 1;
        }
        thread::sync_threads();
        let mut c = 0usize;
        while c < m_us {
            // SAFETY: every slot of token c was written before the barrier above.
            let sums = unsafe {
                let w = ws.add(c * RMS_WARPS);
                [
                    *w.add(0),
                    *w.add(1),
                    *w.add(2),
                    *w.add(3),
                    *w.add(4),
                    *w.add(5),
                    *w.add(6),
                    *w.add(7),
                ]
            };
            let scale = rms_scale_checked(rms_warp_tree(sums), k, eps, fault, tid == 0);
            let mut j = tid;
            while j < k_us {
                // SAFETY: c · k + j < m · k <= x.len() and COLS_VALUES, j < k <= g.len(); this
                // thread alone writes value c · k + j.
                unsafe {
                    *xn.add(c * k_us + j) =
                        (scale * *g.get_unchecked(j)) * *x.get_unchecked(c * k_us + j);
                }
                j += THREADS;
            }
            c += 1;
        }
        thread::sync_threads();
        // SAFETY: the m · k values of XN were written before the barrier above; nothing writes
        // them past it.
        let xs: &[f32] = unsafe { core::slice::from_raw_parts(xn.cast_const(), m_us * k_us) };
        let lane = warp::lane_id() as usize;
        let row = (blk - first) * ROWS + tid / 32;
        // SAFETY: (every arm) row < the site's rows puts the row's values inside its words or
        // rows by the contract, xs holds m · k values, k a multiple of 32 (host-checked), m in
        // 1..=4 and lane < 32.
        let partials = unsafe {
            match site {
                0 => f32_lane_partials(Bf16Words(w_q), xs, k, row, m, lane),
                1 => f32_lane_partials(Bf16Words(w_k), xs, k, row, m, lane),
                2 => f32_lane_partials(Bf16Words(w_v), xs, k, row, m, lane),
                3 => f32_lane_partials(Bf16Words(w_r), xs, k, row, m, lane),
                4 => f32_lane_partials(Bf16Words(w_g), xs, k, row, m, lane),
                5 => f32_lane_partials(w_s, xs, k, row, m, lane),
                6 => f32_lane_partials(w_kt, xs, k, row, m, lane),
                _ => f32_lane_partials(w_vt, xs, k, row, m, lane),
            }
        };
        let sums = gemv_lane_sums(partials, m);
        if lane == 0 {
            // The site's rows by a constant index, so `rows` stays in registers.
            let n = match site {
                0 => rq,
                1 => rk,
                2 => rv,
                3 => rr,
                4 => rg,
                5 => rs,
                6 => rkt,
                _ => rvt,
            } as usize;
            let mut c = 0usize;
            while c < m_us {
                let at = c * n + row;
                // Token c's sum by a constant index, so `sums` stays in registers.
                let v = match c {
                    0 => sums[0],
                    1 => sums[1],
                    2 => sums[2],
                    _ => sums[3],
                };
                // SAFETY: at < m · rows <= y.len() by the contract; only lane 0 of the row's
                // warp writes token c's slot of row.
                unsafe {
                    match site {
                        0 => *y_q.get_unchecked_mut(at) = v,
                        1 => *y_k.get_unchecked_mut(at) = v,
                        2 => *y_v.get_unchecked_mut(at) = v,
                        3 => *y_r.get_unchecked_mut(at) = v,
                        4 => *y_g.get_unchecked_mut(at) = v,
                        5 => *y_s.get_unchecked_mut(at) = v,
                        6 => *y_kt.get_unchecked_mut(at) = v,
                        _ => *y_vt.get_unchecked_mut(at) = v,
                    }
                }
                c += 1;
            }
        }
    }
}

/// The loaded module. Owns no stream: each enqueue takes the engine stream.
pub struct NormProjBf16Kernels {
    module: norm_proj_bf16_kernels::LoadedModule,
}

/// A BF16 site's words (`Bf16Raw`: rows × k/2) and its output.
pub struct Bf16Site<'a> {
    pub w: &'a DeviceTensor<u32>,
    pub y: &'a mut DeviceBuffer<f32>,
}

/// One launch's inputs and sites (module doc): [`crate::norm_proj::NormProjArgs`] with the five
/// projection sites in BF16 words, and `m` tokens of each input.
pub struct NormProjBf16Args<'a> {
    /// The input rows (`m · k` values each, token after token) and their gains: one, or three
    /// (`ik`, `iv` then name rows 1 and 2).
    pub x: [&'a DeviceBuffer<f32>; 3],
    pub gain: [&'a DeviceBuffer<f32>; 3],
    pub ik: usize,
    pub iv: usize,
    pub eps: f32,
    /// The input width.
    pub k: usize,
    /// Tokens: 1 runs the one-row entry, 2..=[`MAX_COLS`] the `_cols` entry.
    pub m: usize,
    pub fault: FaultSink,
    /// `q`, `k`, `v`, `rel`, `gate`.
    pub bf16: [Bf16Site<'a>; 5],
    /// `sink`, then the two conv-tap sites: run where `taps` holds, else of no row.
    pub sink: crate::norm_proj::F32Site<'a>,
    pub kt: crate::norm_proj::F32Site<'a>,
    pub vt: crate::norm_proj::F32Site<'a>,
    pub taps: bool,
}

impl NormProjBf16Kernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<NormProjBf16Kernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for the module above;
        // the launcher checks its launch contract.
        let module = unsafe { crate::shared_module!(norm_proj_bf16_kernels, ctx)? };
        Ok(NormProjBf16Kernels { module })
    }

    /// Enqueue the norms and the projections of `a` (module doc): `Σ rows / 8` blocks of 256
    /// threads, the one-row entry at `m` 1 and the `_cols` entry above. A width that is not a
    /// positive multiple of 32 or past [`MAX_K`], `m` outside `1..=`[`MAX_COLS`] or `m · k`
    /// past [`COLS_VALUES`], an input shorter than `m · k`, a site whose rows are not `k` wide
    /// or not a multiple of [`ROWS`], an output shorter than `m` times its rows and an input
    /// index past 2 are refused by name. Asynchronous, allocation-free, capturable.
    pub fn enqueue(&self, stream: &CudaStream, a: NormProjBf16Args<'_>) -> Result<(), GpuError> {
        const WHAT: &str = "norm_proj_bf16::enqueue";
        let bad = |detail: String| Err(GpuError::shape(WHAT, detail));
        let (k, m) = (a.k, a.m);
        if k == 0 || !k.is_multiple_of(32) || k > MAX_K {
            return bad(format!(
                "an input of {k} values: a positive multiple of 32 up to {MAX_K}"
            ));
        }
        if m == 0 || m > MAX_COLS || (m > 1 && m * k > COLS_VALUES) {
            return bad(format!(
                "{m} tokens of {k} values: 1..={MAX_COLS} tokens of at most {COLS_VALUES} values"
            ));
        }
        if a.ik > 2 || a.iv > 2 {
            return bad(format!("inputs {} and {}: three rows at most", a.ik, a.iv));
        }
        for (i, (x, g)) in a.x.iter().zip(&a.gain).enumerate() {
            if x.len() < m * k || g.len() < k {
                return bad(format!(
                    "input {i}: {} values and a gain of {}, for {m} tokens of {k}",
                    x.len(),
                    g.len()
                ));
            }
        }
        let rows_of = |name: &str, rows: usize, y: usize| -> Result<usize, GpuError> {
            if !rows.is_multiple_of(ROWS) || y < m * rows {
                return Err(GpuError::shape(
                    WHAT,
                    format!(
                        "{name}: {rows} rows (a multiple of {ROWS}) of {m} tokens into an output \
                         of {y}"
                    ),
                ));
            }
            Ok(rows)
        };
        const NAMES: [&str; 5] = ["q", "k", "v", "rel", "gate"];
        let mut bf_rows = [0usize; 5];
        for (i, s) in a.bf16.iter().enumerate() {
            if 2 * s.w.cols() != k {
                return bad(format!(
                    "{}: BF16 rows of {} words, for {k} values",
                    NAMES[i],
                    s.w.cols()
                ));
            }
            bf_rows[i] = rows_of(NAMES[i], s.w.rows(), s.y.len())?;
        }
        let f32_rows = |name: &str, s: &crate::norm_proj::F32Site<'_>| -> Result<usize, GpuError> {
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
        let blocks: usize = (bf_rows.iter().sum::<usize>() + rs + rkt + rvt) / ROWS;
        let u = |name: &'static str, v: usize| launch_u32(WHAT, name, v);
        let NormProjBf16Args {
            x,
            gain,
            ik,
            iv,
            eps,
            fault,
            bf16,
            sink,
            kt,
            vt,
            ..
        } = a;
        let [sq, sk, sv, sr, sg] = bf16;
        let (
            crate::norm_proj::F32Site { w: w_s, y: y_s },
            crate::norm_proj::F32Site { w: w_kt, y: y_kt },
            crate::norm_proj::F32Site { w: w_vt, y: y_vt },
        ) = (sink, kt, vt);
        let cfg = LaunchConfig1D::new(u("grid", blocks)?, THREADS as u32, 0);
        macro_rules! launch {
            ($prepare:ident, $entry:ident, $($m:expr)?) => {{
                let prep = self.module.$prepare(cfg)?;
                self.module.$entry(
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
                    $($m,)?
                    fault,
                    sq.w.buf(),
                    sk.w.buf(),
                    sv.w.buf(),
                    sr.w.buf(),
                    sg.w.buf(),
                    w_s.buf(),
                    w_kt.buf(),
                    w_vt.buf(),
                    u("rows", bf_rows[0])?,
                    u("rows", bf_rows[1])?,
                    u("rows", bf_rows[2])?,
                    u("rows", bf_rows[3])?,
                    u("rows", bf_rows[4])?,
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
            }};
        }
        if m == 1 {
            launch!(prepare_norm_proj_bf16_f32, norm_proj_bf16_f32,);
        } else {
            launch!(
                prepare_norm_proj_bf16_f32_cols,
                norm_proj_bf16_f32_cols,
                u("m", m)?
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Gpu;
    use crate::norm_proj::F32Site;

    /// Next value of a xorshift64 stream.
    fn next(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    /// A value in [-1, 1) of the stream.
    fn unit(state: &mut u64) -> f32 {
        ((next(state) >> 33) % 2001) as f32 / 1000.0 - 1.0
    }

    /// The BF16 value of half `odd` of a word, widened.
    fn widen(w: u32, odd: bool) -> f32 {
        f32::from_bits(if odd { w & 0xffff_0000 } else { w << 16 })
    }

    /// Both entries against the unfused path and the host: three inputs of K 2304 (`ik` 1,
    /// `iv` 2), five BF16 sites and three F32 sites of 16 to 40 rows, at m = 1 (the one-row
    /// entry) and m = 2, 3, 4 (`_cols`). Every output is `rms_norm` then `bf16_gemv` or
    /// `f32_gemv` over the same tokens bit for bit, and within `2^-16` of its row's magnitude
    /// of the host's f64 norm and dot over the widened BF16 values.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it"]
    fn hw_norm_proj_bf16_is_the_norm_then_the_gemv() {
        const K: usize = 2304;
        const EPS: f32 = 1e-6;
        const ROWS_OF: [usize; SITES] = [40, 16, 24, 32, 16, 8, 16, 24];
        const INPUT: [usize; SITES] = [0, 1, 2, 0, 0, 0, 1, 2];
        let gpu = Gpu::new().expect("card");
        let stream = gpu.stream();
        let kernels = NormProjBf16Kernels::load(gpu.context()).expect("the module");
        let mut state = 0x4242u64;
        let gains: Vec<Vec<f32>> = (0..3)
            .map(|_| (0..K).map(|_| 0.5 + unit(&mut state).abs()).collect())
            .collect();
        let words: Vec<Vec<u32>> = (0..5)
            .map(|s| {
                (0..ROWS_OF[s] * K / 2)
                    .map(|_| {
                        let half = |st: &mut u64| (unit(st).to_bits() >> 16) & 0xffff;
                        half(&mut state) | (half(&mut state) << 16)
                    })
                    .collect()
            })
            .collect();
        let f32s: Vec<Vec<f32>> = (5..SITES)
            .map(|s| (0..ROWS_OF[s] * K).map(|_| unit(&mut state)).collect())
            .collect();
        let g_dev: Vec<DeviceBuffer<f32>> = gains
            .iter()
            .map(|g| DeviceBuffer::from_host(stream, g).expect("gain"))
            .collect();
        let w_bf: Vec<DeviceTensor<u32>> = (0..5)
            .map(|s| DeviceTensor::upload(stream, &words[s], ROWS_OF[s], K / 2).expect("words"))
            .collect();
        let w_f: Vec<DeviceTensor<f32>> = (0..3)
            .map(|s| DeviceTensor::upload(stream, &f32s[s], ROWS_OF[5 + s], K).expect("rows"))
            .collect();
        let mut failures = Vec::new();
        for m in 1..=MAX_COLS {
            let xs: Vec<Vec<f32>> = (0..3)
                .map(|_| (0..m * K).map(|_| 3.0 * unit(&mut state)).collect())
                .collect();
            let x_dev: Vec<DeviceBuffer<f32>> = xs
                .iter()
                .map(|x| DeviceBuffer::from_host(stream, x).expect("x"))
                .collect();
            let mut ys: Vec<DeviceBuffer<f32>> = (0..SITES)
                .map(|s| DeviceBuffer::zeroed(stream, m * ROWS_OF[s]).expect("y"))
                .collect();
            {
                let [y0, y1, y2, y3, y4, y5, y6, y7] = &mut ys[..] else {
                    unreachable!("eight sites")
                };
                kernels
                    .enqueue(
                        stream,
                        NormProjBf16Args {
                            x: [&x_dev[0], &x_dev[1], &x_dev[2]],
                            gain: [&g_dev[0], &g_dev[1], &g_dev[2]],
                            ik: 1,
                            iv: 2,
                            eps: EPS,
                            k: K,
                            m,
                            fault: gpu.unlabelled_sink(),
                            bf16: [
                                Bf16Site { w: &w_bf[0], y: y0 },
                                Bf16Site { w: &w_bf[1], y: y1 },
                                Bf16Site { w: &w_bf[2], y: y2 },
                                Bf16Site { w: &w_bf[3], y: y3 },
                                Bf16Site { w: &w_bf[4], y: y4 },
                            ],
                            sink: F32Site { w: &w_f[0], y: y5 },
                            kt: F32Site { w: &w_f[1], y: y6 },
                            vt: F32Site { w: &w_f[2], y: y7 },
                            taps: true,
                        },
                    )
                    .expect("the fused launch");
            }
            assert_eq!(gpu.take_fault().expect("fault word"), None);
            for s in 0..SITES {
                let i = INPUT[s];
                let mut normed = DeviceBuffer::zeroed(stream, m * K).expect("normed");
                gpu.elem()
                    .enqueue_rms_norm(stream, &x_dev[i], &g_dev[i], EPS, K, m, &mut normed)
                    .expect("rms_norm");
                let mut want = DeviceBuffer::zeroed(stream, m * ROWS_OF[s]).expect("want");
                if s < 5 {
                    gpu.q8f32()
                        .enqueue_bf16_gemv(stream, &w_bf[s], &normed, m, &mut want)
                        .expect("bf16_gemv");
                } else {
                    gpu.q8f32()
                        .enqueue_f32_gemv(stream, &w_f[s - 5], &normed, m, &mut want)
                        .expect("f32_gemv");
                }
                let (got, want) = (
                    ys[s].to_host_vec(stream).expect("got"),
                    want.to_host_vec(stream).expect("want"),
                );
                for c in 0..m {
                    let x = &xs[i][c * K..][..K];
                    let ms = x.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / K as f64;
                    let scale = 1.0 / (ms + f64::from(EPS)).sqrt();
                    for r in 0..ROWS_OF[s] {
                        // The fused entry is token-major, the gemv row-major.
                        let (g, w) = (got[c * ROWS_OF[s] + r], want[r * m + c]);
                        let (mut dot, mut mag) = (0.0f64, 0.0f64);
                        for j in 0..K {
                            let wv = if s < 5 {
                                widen(words[s][(r * K + j) / 2], j % 2 == 1)
                            } else {
                                f32s[s - 5][r * K + j]
                            };
                            let xn = scale * f64::from(gains[i][j]) * f64::from(x[j]);
                            dot += f64::from(wv) * xn;
                            mag += (f64::from(wv) * xn).abs();
                        }
                        if g.to_bits() != w.to_bits()
                            || (f64::from(g) - dot).abs() > mag * f64::from(2.0f32.powi(-16))
                        {
                            failures.push(format!(
                                "m {m} site {s} token {c} row {r}: {g} against the gemv's {w} \
                                 and f64 {dot} (magnitude {mag})"
                            ));
                        }
                    }
                }
            }
        }
        println!(
            "norm_proj_bf16: {} failures; first: {:?}",
            failures.len(),
            failures.first()
        );
        assert!(failures.is_empty(), "\n{}", failures.join("\n"));
    }
}
