//! Inkling 2's normed block-attention mix ([`crate::attn_res`]'s `attn_res_mix_normed`, one row)
//! spread over the card: the same rule and the same bits, in two launches, and up to two
//! independent mixes in each of them.
//!
//! `attn_res_mix_normed` gives a row one block, which walks the row's vectors one after another:
//! every vector's sums wait on the one before. Here the sums of a vector are a block of their own:
//!
//! 1. `attn_res_step_scores`: one block a vector of each mix. Thread `tid` takes the vector's
//!    values `tid, tid + 256, …` and keeps `attn_res`'s partials (a dot a query row and the sum of
//!    squares, each `acc += a · b` in that order); the warp butterfly and the fixed eight-warp tree
//!    `((w0 + w1) + (w2 + w3)) + ((w4 + w5) + (w6 + w7))` give the sums; thread 0 writes the
//!    vector's RMS scale and its scores `(rstd · dot) · scale` into the mix's statistics. The block
//!    that draws a mix's last ticket runs the softmax of each query row over the vectors in vector
//!    order (a thread a query row) and writes the probabilities over the scores, then puts the
//!    ticket count back to zero.
//! 2. `attn_res_step_out`: one thread a value of each mix's output, accumulating
//!    `p_i · (v_i · rstd_i)` by fused multiply-adds in vector order, the stream last.
//!
//! Every operation is `attn_res_mix_normed`'s in its order, so each output value is that entry's
//! bits (the module doc of [`crate::attn_res`] is the numeric contract). Only the blocks that
//! compute them differ; the loads of a thread's values are batched three strides at a time, the
//! sums unchanged.
//!
//! Layout as [`crate::attn_res`]'s at one row (`m = 1`): bank `i` is `banks[i · stride ..][..n]`,
//! the stream `cur[..n]`, the output `y[q · n ..][..n]` for query row `q`. The statistics are
//! [`JOB_STATS`] f32s a mix: the scale of vector `i` at `i`, the score (then the probability) of
//! query row `q` and vector `i` at `MAX_VECS + q · MAX_VECS + i`; the tickets one `u32` a mix,
//! zero between launches.
//!
//! The same two entries take `m` rows ([`AttnResStepKernels::enqueue_rows`]): the grid is the
//! one-row grid a row, the row the fastest index, and a block is its row's block of the one-row
//! launch on that row's values ([`crate::attn_res`]'s layout: bank `i`'s row `t` at
//! `banks[i · stride + t · n ..][..n]` with `stride >= m · n`, the stream's row `t` at
//! `cur[t · n ..]`, the output value `j` of query row `q` and row `t` at
//! `y[t · nq · n + q · n + j]`). The statistics and the tickets are the one-row ones a row
//! (`row · JOBS` apart). A mix's tickets count its own row's blocks, so a row's softmax waits on
//! that row's vectors alone, and every row is the one-row launch's bits on its values.
//!
//! No silent failure, as [`crate::attn_res`]'s entry: a vector whose sum of squares is not finite
//! and positive raises [`FaultSite::F32Product`] through its scale, a score that is not finite
//! raises it in the softmax, and that query row's probabilities, so its output, are NaN.

use crate::GpuError;
use crate::attn_res::{AttnResNormedArgs, MAX_NQ, MAX_VECS, THREADS};
use crate::elem::rms_scale_of_mean_checked;
use crate::fault::{FaultSink, FaultSite};
use crate::launch_u32;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32};
use cuda_device::float::mul_rn_f32;
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread, threadfence, warp,
};
use cuda_host::cuda_module;
use std::sync::Arc;

const WARPS: usize = THREADS / 32;
/// f32s of one mix's statistics: a scale a vector, then a score a (query row, vector).
pub const JOB_STATS: usize = MAX_VECS * (1 + MAX_NQ);
/// The mixes one launch pair carries.
pub const JOBS: usize = 2;
/// f32s of the scores block's warp sums: a dot a query row and the squares, a slot a warp.
const PART_LEN: usize = (MAX_NQ + 1) * WARPS;
// The entries' `launch_contract` spells JOBS · JOB_STATS as the literal 160.
const _: () = assert!(THREADS == 256 && WARPS == 8 && MAX_NQ == 4 && JOBS * JOB_STATS == 160);

/// One thread's partial sums over the vector `src[base ..][..n]`'s values `tid, tid + THREADS, …`:
/// a dot with each of the four query rows and the sum of squares, each `acc += a · b` in that
/// order (`attn_res`'s partials, term for term). A query row at or past `nq` reads row `nq − 1`;
/// its sums are not read.
///
/// # Safety
///
/// `src` holds `n` values from `base`, `queries` holds `nq · n`, `1 <= nq <= MAX_NQ`.
#[inline(always)]
unsafe fn partials(
    (src, base): (&[f32], usize),
    queries: &[f32],
    nq: usize,
    n: usize,
    tid: usize,
) -> ([f32; MAX_NQ], f32) {
    let row = |q: usize| q.min(nq - 1) * n;
    let (r1, r2, r3) = (row(1), row(2), row(3));
    let (mut d0, mut d1, mut d2, mut d3, mut sq) = (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let mut j = tid;
    // SAFETY (every read below): j < n puts `base + j` inside the vector and `r + j` inside the
    // query rows (r <= (nq − 1) · n) by this fn's contract.
    while j + 2 * THREADS < n {
        let (ja, jb, jc) = (j, j + THREADS, j + 2 * THREADS);
        // SAFETY: as above.
        let (va, vb, vc) = unsafe {
            (
                *src.get_unchecked(base + ja),
                *src.get_unchecked(base + jb),
                *src.get_unchecked(base + jc),
            )
        };
        // SAFETY: as above.
        let qa = unsafe {
            [
                *queries.get_unchecked(ja),
                *queries.get_unchecked(r1 + ja),
                *queries.get_unchecked(r2 + ja),
                *queries.get_unchecked(r3 + ja),
            ]
        };
        // SAFETY: as above.
        let qb = unsafe {
            [
                *queries.get_unchecked(jb),
                *queries.get_unchecked(r1 + jb),
                *queries.get_unchecked(r2 + jb),
                *queries.get_unchecked(r3 + jb),
            ]
        };
        // SAFETY: as above.
        let qc = unsafe {
            [
                *queries.get_unchecked(jc),
                *queries.get_unchecked(r1 + jc),
                *queries.get_unchecked(r2 + jc),
                *queries.get_unchecked(r3 + jc),
            ]
        };
        d0 += va * qa[0];
        d1 += va * qa[1];
        d2 += va * qa[2];
        d3 += va * qa[3];
        sq += va * va;
        d0 += vb * qb[0];
        d1 += vb * qb[1];
        d2 += vb * qb[2];
        d3 += vb * qb[3];
        sq += vb * vb;
        d0 += vc * qc[0];
        d1 += vc * qc[1];
        d2 += vc * qc[2];
        d3 += vc * qc[3];
        sq += vc * vc;
        j += 3 * THREADS;
    }
    while j < n {
        // SAFETY: as above.
        let v = unsafe { *src.get_unchecked(base + j) };
        // SAFETY: as above.
        let q = unsafe {
            [
                *queries.get_unchecked(j),
                *queries.get_unchecked(r1 + j),
                *queries.get_unchecked(r2 + j),
                *queries.get_unchecked(r3 + j),
            ]
        };
        d0 += v * q[0];
        d1 += v * q[1];
        d2 += v * q[2];
        d3 += v * q[3];
        sq += v * v;
        j += THREADS;
    }
    ([d0, d1, d2, d3], sq)
}

/// The eight warp sums of slot `at` (a query row's dot, or [`MAX_NQ`] the squares) in the fixed
/// tree of the module doc.
///
/// # Safety
///
/// `ws` addresses [`PART_LEN`] written slots, `at <= MAX_NQ`.
#[inline(always)]
unsafe fn tree(ws: *const f32, at: usize) -> f32 {
    // SAFETY: at · WARPS + b < PART_LEN by the caller's contract.
    unsafe {
        let w = |b: usize| *ws.add(at * WARPS + b);
        ((w(0) + w(1)) + (w(2) + w(3))) + ((w(4) + w(5)) + (w(6) + w(7)))
    }
}

/// One mix's inputs as a kernel reads them.
#[derive(Clone, Copy)]
struct Job<'a> {
    banks: &'a [f32],
    cur: &'a [f32],
    nb: usize,
    stride: usize,
}

#[cuda_module]
mod attn_res_step_kernels {
    use super::*;

    /// The sums, scales and scores of every vector of up to two mixes (module doc, launch 1) over
    /// `m` rows: block `b` is vector `b / m` (mix a's `0 .. nb_a`, then mix b's `nb_b + 1`, none
    /// at `jobs = 1`) of row `b % m`. The mix's last block of a row writes that row's
    /// probabilities and zeroes its ticket.
    #[allow(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        reason = "kernel entry: the device ABI takes the two mixes' arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            banks_a.len() >= nb_a * stride_a,
            stride_a >= m * n,
            cur_a.len() >= m * n,
            queries_a.len() >= nq_a * n,
            banks_b.len() >= nb_b * stride_b,
            stride_b >= m * n,
            cur_b.len() >= m * n,
            queries_b.len() >= nq_b * n,
            nq_a >= 1,
            nq_a <= 4,
            nq_b >= 1,
            nq_b <= 4,
            nb_a < 16,
            nb_b < 16,
            jobs >= 1,
            jobs <= 2,
            m >= 1,
            stats.len() >= 160 * m,
            done.len() >= 2 * m
        )
    )]
    pub fn attn_res_step_scores(
        banks_a: &[f32],
        cur_a: &[f32],
        queries_a: &[f32],
        nb_a: u32,
        nq_a: u32,
        stride_a: u32,
        fault_a: FaultSink,
        banks_b: &[f32],
        cur_b: &[f32],
        queries_b: &[f32],
        nb_b: u32,
        nq_b: u32,
        stride_b: u32,
        fault_b: FaultSink,
        jobs: u32,
        scale: f32,
        eps: f32,
        n: u32,
        m: u32,
        mut stats: DisjointSlice<f32>,
        mut done: DisjointSlice<u32>,
    ) {
        static mut PART: SharedArray<f32, PART_LEN> = SharedArray::UNINIT;
        static mut LAST: SharedArray<u32, 1> = SharedArray::UNINIT;

        let blk = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        let (n, m) = (n as usize, m as usize);
        let va = nb_a as usize + 1;
        let (vec, t) = (blk / m, blk % m);
        let (job, i) = if vec < va {
            (0usize, vec)
        } else {
            (1usize, vec - va)
        };
        if job >= jobs as usize || (job == 1 && i > nb_b as usize) {
            return; // block-uniform: a block past the mixes' vectors
        }
        let (src, queries, nq, fault, vecs) = if job == 0 {
            let j = Job {
                banks: banks_a,
                cur: cur_a,
                nb: nb_a as usize,
                stride: stride_a as usize,
            };
            (j, queries_a, nq_a as usize, fault_a, va)
        } else {
            let j = Job {
                banks: banks_b,
                cur: cur_b,
                nb: nb_b as usize,
                stride: stride_b as usize,
            };
            (j, queries_b, nq_b as usize, fault_b, nb_b as usize + 1)
        };
        // SAFETY: PART and LAST are this block's own shared allocations; the raw form is the only
        // way to reach them without a reference to a `static mut`. Every access is below its
        // length and ordered by `sync_threads`.
        let ws = unsafe { SharedArray::as_raw_mut_ptr(&raw mut PART) };
        // SAFETY: as above.
        let last = unsafe { SharedArray::as_raw_mut_ptr(&raw mut LAST) };
        // SAFETY: bank i's row t ends below (i + 1) · stride <= nb · stride <= banks.len()
        // (stride >= m · n, the contract), the stream's row t below (t + 1) · n <= m · n <=
        // cur.len(), the query rows below nq · n <= queries.len(), by the contract.
        let (dot, sq) = unsafe {
            if i < src.nb {
                partials((src.banks, i * src.stride + t * n), queries, nq, n, tid)
            } else {
                partials((src.cur, t * n), queries, nq, n, tid)
            }
        };
        let dot = [
            warp::reduce_sum_f32(dot[0]),
            warp::reduce_sum_f32(dot[1]),
            warp::reduce_sum_f32(dot[2]),
            warp::reduce_sum_f32(dot[3]),
        ];
        let sq = warp::reduce_sum_f32(sq);
        if warp::lane_id() == 0 {
            let w = tid / 32;
            // SAFETY: q · WARPS + w < PART_LEN for q <= MAX_NQ, w < WARPS; one lane a warp writes
            // its slots.
            unsafe {
                *ws.add(w) = dot[0];
                *ws.add(WARPS + w) = dot[1];
                *ws.add(2 * WARPS + w) = dot[2];
                *ws.add(3 * WARPS + w) = dot[3];
                *ws.add(MAX_NQ * WARPS + w) = sq;
            }
        }
        thread::sync_threads();
        let at = (t * JOBS + job) * JOB_STATS;
        if tid == 0 {
            // SAFETY: every slot was written before the barrier above.
            let sq = unsafe { tree(ws, MAX_NQ) };
            let rstd = rms_scale_of_mean_checked(sq / n as f32, eps, fault, true);
            // SAFETY: at + i < m · JOBS · JOB_STATS <= stats.len() (t < m by the grid, job < 2,
            // i < vecs <= MAX_VECS); this block's thread 0 alone writes vector i's slots of its
            // row's mix.
            unsafe { *stats.get_unchecked_mut(at + i) = rstd };
            let mut q = 0usize;
            while q < nq {
                // SAFETY: as above, at query q's dot slot.
                let d = unsafe { tree(ws, q) };
                // The dot rounds, then the scale; this order is the gate.
                let s = mul_rn_f32(mul_rn_f32(rstd, d), scale);
                // SAFETY: MAX_VECS + q · MAX_VECS + i < JOB_STATS; as above.
                unsafe { *stats.get_unchecked_mut(at + MAX_VECS + q * MAX_VECS + i) = s };
                q += 1;
            }
        }
        threadfence();
        thread::sync_threads();
        // SAFETY: done holds JOBS tickets a row, 2 · m by the contract, and t · JOBS + job is
        // inside them (t < m by the grid, job < 2); every access to one is atomic.
        let ticket = unsafe { done.as_mut_ptr().add(t * JOBS + job) };
        if tid == 0 {
            // SAFETY: as above.
            let count = unsafe { DeviceAtomicU32::from_ptr(ticket) };
            let is_last = count.fetch_add(1, AtomicOrdering::AcqRel) + 1 == vecs as u32;
            // SAFETY: block-shared, one element, written before the barrier that publishes it.
            unsafe { *last = u32::from(is_last) };
        }
        thread::sync_threads();
        // SAFETY: written by thread 0 before the barrier above.
        if unsafe { *last } == 0 {
            return;
        }
        // The row's mix's last block: every vector's scale and scores of the row are published.
        let pr = stats.as_mut_ptr();
        if tid < nq {
            let q = tid;
            let row = at + MAX_VECS + q * MAX_VECS;
            let mut finite = true;
            let mut mx = f32::NEG_INFINITY;
            let mut v = 0usize;
            while v < vecs {
                // SAFETY: row + v < at + JOB_STATS <= m · JOBS · JOB_STATS <= stats.len(); this
                // thread alone touches query q's row.
                let s = unsafe { *pr.add(row + v) };
                finite &= s.is_finite();
                if s > mx {
                    mx = s;
                }
                v += 1;
            }
            let mut total = 0.0f32;
            let mut v = 0usize;
            while v < vecs {
                // SAFETY: as above.
                let e = unsafe { (*pr.add(row + v) - mx).exp() };
                total += e;
                // SAFETY: as above.
                unsafe { *pr.add(row + v) = e };
                v += 1;
            }
            if !finite {
                fault.raise(FaultSite::F32Product);
            }
            let mut v = 0usize;
            while v < vecs {
                // SAFETY: as above.
                unsafe {
                    *pr.add(row + v) = if finite {
                        *pr.add(row + v) / total
                    } else {
                        f32::NAN
                    }
                };
                v += 1;
            }
        }
        if tid == 0 {
            // SAFETY: as above; every block of the mix has drawn its ticket by now.
            let count = unsafe { DeviceAtomicU32::from_ptr(ticket) };
            count.store(0, AtomicOrdering::Relaxed);
        }
    }

    /// Every output value of up to two mixes (module doc, launch 2) over `m` rows: block `b` is
    /// mix `b / (m · per)`'s value chunk `(b / m) % per` of row `b % m`, `per = ⌈n / 256⌉`;
    /// thread `tid` of a block value `chunk · 256 + tid` of its row.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the two mixes' arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            banks_a.len() >= nb_a * stride_a,
            stride_a >= m * n,
            cur_a.len() >= m * n,
            banks_b.len() >= nb_b * stride_b,
            stride_b >= m * n,
            cur_b.len() >= m * n,
            nq_a >= 1,
            nq_a <= 4,
            nq_b >= 1,
            nq_b <= 4,
            nb_a < 16,
            nb_b < 16,
            jobs >= 1,
            jobs <= 2,
            m >= 1,
            stats.len() >= 160 * m,
            y_a.len() >= m * nq_a * n,
            y_b.len() >= m * nq_b * n
        )
    )]
    pub fn attn_res_step_out(
        banks_a: &[f32],
        cur_a: &[f32],
        nb_a: u32,
        nq_a: u32,
        stride_a: u32,
        banks_b: &[f32],
        cur_b: &[f32],
        nb_b: u32,
        nq_b: u32,
        stride_b: u32,
        jobs: u32,
        n: u32,
        m: u32,
        stats: &[f32],
        mut y_a: DisjointSlice<f32>,
        mut y_b: DisjointSlice<f32>,
    ) {
        static mut ST: SharedArray<f32, JOB_STATS> = SharedArray::UNINIT;

        let blk = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        let (n, m) = (n as usize, m as usize);
        let per = n.div_ceil(THREADS);
        let (mix, t) = (blk / m, blk % m);
        let job = mix / per;
        if job >= jobs as usize {
            return; // block-uniform
        }
        let (src, nq) = if job == 0 {
            (
                Job {
                    banks: banks_a,
                    cur: cur_a,
                    nb: nb_a as usize,
                    stride: stride_a as usize,
                },
                nq_a as usize,
            )
        } else {
            (
                Job {
                    banks: banks_b,
                    cur: cur_b,
                    nb: nb_b as usize,
                    stride: stride_b as usize,
                },
                nq_b as usize,
            )
        };
        let at = (t * JOBS + job) * JOB_STATS;
        // SAFETY: ST is this block's own shared allocation; the raw form is the only way to reach
        // it without a reference to a `static mut`. Every access is below JOB_STATS and ordered
        // by `sync_threads`.
        let st = unsafe { SharedArray::as_raw_mut_ptr(&raw mut ST) };
        if tid < JOB_STATS {
            // SAFETY: at + tid < m · JOBS · JOB_STATS <= stats.len() (t < m by the grid, job < 2);
            // tid < JOB_STATS.
            unsafe { *st.add(tid) = *stats.get_unchecked(at + tid) };
        }
        thread::sync_threads();
        let j = (mix - job * per) * THREADS + tid;
        if j >= n {
            return;
        }
        let p = |q: usize, v: usize| -> f32 {
            // SAFETY: MAX_VECS + q · MAX_VECS + v < JOB_STATS for q < MAX_NQ, v < MAX_VECS;
            // loaded before the barrier above.
            unsafe { *st.add(MAX_VECS + q.min(nq - 1) * MAX_VECS + v) }
        };
        let (mut a0, mut a1, mut a2, mut a3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        let mut v = 0usize;
        while v < src.nb {
            // Four banks' values loaded ahead of their sums; a bank past the last reads bank 0
            // and is not summed.
            let bank = |b: usize| if b < src.nb { b } else { 0 };
            // SAFETY: bank b < nb's row t ends below b · stride + m · n <= (b + 1) · stride <=
            // nb · stride <= banks.len() (stride >= m · n, the contract) and j < n.
            let w = unsafe {
                [
                    *src.banks.get_unchecked(bank(v) * src.stride + t * n + j),
                    *src.banks
                        .get_unchecked(bank(v + 1) * src.stride + t * n + j),
                    *src.banks
                        .get_unchecked(bank(v + 2) * src.stride + t * n + j),
                    *src.banks
                        .get_unchecked(bank(v + 3) * src.stride + t * n + j),
                ]
            };
            let mut k = 0usize;
            while k < 4 && v + k < src.nb {
                // SAFETY: v + k < nb < MAX_VECS; loaded before the barrier above.
                let r = unsafe { *st.add(v + k) };
                let nv = mul_rn_f32(w[k], r);
                a0 += p(0, v + k) * nv;
                a1 += p(1, v + k) * nv;
                a2 += p(2, v + k) * nv;
                a3 += p(3, v + k) * nv;
                k += 1;
            }
            v += 4;
        }
        // SAFETY: t · n + j < m · n <= cur.len() by the contract; nb < MAX_VECS, loaded before the
        // barrier above.
        let (r, c) = unsafe { (*st.add(src.nb), *src.cur.get_unchecked(t * n + j)) };
        let nv = mul_rn_f32(c, r);
        a0 += p(0, src.nb) * nv;
        a1 += p(1, src.nb) * nv;
        a2 += p(2, src.nb) * nv;
        a3 += p(3, src.nb) * nv;
        let acc = [a0, a1, a2, a3];
        let mut q = 0usize;
        while q < nq {
            // SAFETY: t · nq · n + q · n + j < m · nq · n <= the mix's output length by the
            // contract; this thread alone writes the value.
            unsafe {
                if job == 0 {
                    *y_a.get_unchecked_mut(t * nq * n + q * n + j) = acc[q];
                } else {
                    *y_b.get_unchecked_mut(t * nq * n + q * n + j) = acc[q];
                }
            }
            q += 1;
        }
    }
}

/// The loaded module. Owns no stream: each enqueue takes the engine stream.
pub struct AttnResStepKernels {
    module: attn_res_step_kernels::LoadedModule,
}

/// The device words the launch pair passes between its launches: the [`JOBS`] mixes' statistics
/// and their tickets, a row of the pair's rows. One a stream: a pair's second launch reads what its
/// first wrote, and the next pair on the stream starts after it.
pub struct MixStepScratch {
    stats: DeviceBuffer<f32>,
    done: DeviceBuffer<u32>,
    /// The stand-in output of a pair that carries one mix: its rows of `n`.
    spare: DeviceBuffer<f32>,
}

impl MixStepScratch {
    /// The statistics and the tickets (zero) of one row's mixes `n` wide: the step's.
    pub fn new(stream: &CudaStream, n: usize) -> Result<MixStepScratch, GpuError> {
        Self::rows(stream, (n, 1))
    }

    /// The statistics and the tickets (zero) of `rows` rows' mixes `n` wide, and the stand-in
    /// output of a pair that carries one mix, `rows` rows of `n`.
    pub fn rows(
        stream: &CudaStream,
        (n, rows): (usize, usize),
    ) -> Result<MixStepScratch, GpuError> {
        Ok(MixStepScratch {
            stats: DeviceBuffer::zeroed(stream, rows * JOBS * JOB_STATS)?,
            done: DeviceBuffer::zeroed(stream, rows * JOBS)?,
            spare: DeviceBuffer::zeroed(stream, (rows * n).max(1))?,
        })
    }

    /// The bytes it holds on the card.
    pub fn bytes(&self) -> usize {
        self.stats.num_bytes() + self.done.num_bytes() + self.spare.num_bytes()
    }
}

/// One mix's checks at its `m` rows, [`crate::attn_res::AttnResKernels::enqueue_normed`]'s;
/// refused by name as `what`'s.
fn check_shape(
    what: &'static str,
    a: &AttnResNormedArgs<'_>,
    y_len: usize,
) -> Result<(), GpuError> {
    let bad = |detail: String| Err(GpuError::shape(what, detail));
    if a.m == 0 {
        return bad("a mix of no row".into());
    }
    if a.n == 0 {
        return bad("a row of no value".into());
    }
    if a.nq == 0 || a.nq > MAX_NQ {
        return bad(format!(
            "{} query rows: the launch mixes at most {MAX_NQ} in one",
            a.nq
        ));
    }
    if a.n_banks + 1 > MAX_VECS {
        return bad(format!(
            "{} banks and the stream: the launch mixes at most {MAX_VECS} vectors",
            a.n_banks
        ));
    }
    if a.stride < a.m * a.n {
        return bad(format!(
            "a stride of {} between banks under the {} rows' {}",
            a.stride, a.m, a.n
        ));
    }
    if !(a.eps.is_finite() && a.eps > 0.0) {
        return bad(format!("epsilon {}: a positive finite one", a.eps));
    }
    if !a.scale.is_finite() {
        return bad(format!("scale {}: a finite one", a.scale));
    }
    if a.banks.len() < a.n_banks * a.stride
        || a.cur.len() < a.m * a.n
        || a.queries.len() < a.nq * a.n
        || y_len < a.m * a.nq * a.n
    {
        return bad(format!(
            "{} banks of stride {} in {} values, the stream {} of {} · {}, the queries {} of {} · \
             {}, the output {y_len} of {} · {} · {}",
            a.n_banks,
            a.stride,
            a.banks.len(),
            a.cur.len(),
            a.m,
            a.n,
            a.queries.len(),
            a.nq,
            a.n,
            a.m,
            a.nq,
            a.n
        ));
    }
    Ok(())
}

/// A mix's checks at the step's one row; refused by name as `what`'s.
fn check(what: &'static str, a: &AttnResNormedArgs<'_>, y_len: usize) -> Result<(), GpuError> {
    if a.m != 1 {
        return Err(GpuError::shape(
            what,
            format!(
                "{} rows: the step's mix takes one row (a pass of rows is enqueue_rows')",
                a.m
            ),
        ));
    }
    check_shape(what, a, y_len)
}

impl AttnResStepKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<AttnResStepKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for the module above;
        // the launcher checks its launch contract.
        let module = unsafe { crate::shared_module!(attn_res_step_kernels, ctx)? };
        Ok(AttnResStepKernels { module })
    }

    /// Enqueue mix `a` into `y_a` and, when given, the independent mix `b` into its output, each
    /// bit for bit [`crate::attn_res::AttnResKernels::enqueue_normed`]'s at one row: two launches
    /// (module doc). The two mixes read nothing the other writes. Their checks are
    /// `enqueue_normed`'s, a pass of rows refused by name; the two scales and epsilons must agree.
    /// Asynchronous, allocation-free, capturable.
    pub fn enqueue(
        &self,
        stream: &CudaStream,
        (a, y_a): (AttnResNormedArgs<'_>, &mut DeviceBuffer<f32>),
        b: Option<(AttnResNormedArgs<'_>, &mut DeviceBuffer<f32>)>,
        scratch: &mut MixStepScratch,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "attn_res_step::enqueue";
        check(WHAT, &a, y_a.len())?;
        Self::agreed(WHAT, &a, b.as_ref().map(|(b, y_b)| (b, y_b.len())))?;
        self.pair(stream, WHAT, a, y_a, b, scratch)
    }

    /// [`AttnResStepKernels::enqueue`] over the `a.m` rows of `a` and `b` (module doc): every row
    /// of each output is `enqueue_normed`'s bits at that row, two launches over the one-row grids
    /// a row. The banks are `a.stride` apart, the stream, the outputs and the scratch `a.m` rows
    /// ([`MixStepScratch::rows`]). Refusals as `enqueue`'s beside a short scratch; asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_rows(
        &self,
        stream: &CudaStream,
        (a, y_a): (AttnResNormedArgs<'_>, &mut DeviceBuffer<f32>),
        b: Option<(AttnResNormedArgs<'_>, &mut DeviceBuffer<f32>)>,
        scratch: &mut MixStepScratch,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "attn_res_step::enqueue_rows";
        check_shape(WHAT, &a, y_a.len())?;
        Self::agreed(WHAT, &a, b.as_ref().map(|(b, y_b)| (b, y_b.len())))?;
        self.pair(stream, WHAT, a, y_a, b, scratch)
    }

    /// The second mix's checks, and that the two mixes take one width, scale and epsilon.
    fn agreed(
        what: &'static str,
        a: &AttnResNormedArgs<'_>,
        b: Option<(&AttnResNormedArgs<'_>, usize)>,
    ) -> Result<(), GpuError> {
        let Some((b, y_len)) = b else {
            return Ok(());
        };
        check_shape(what, b, y_len)?;
        if b.n != a.n
            || b.scale.to_bits() != a.scale.to_bits()
            || b.eps.to_bits() != a.eps.to_bits()
            || b.m != a.m
        {
            return Err(GpuError::shape(
                what,
                format!(
                    "two mixes of one launch take one width, scale, epsilon and row count: {} / {} \
                     / {} / {} and {} / {} / {} / {}",
                    a.n, a.scale, a.eps, a.m, b.n, b.scale, b.eps, b.m
                ),
            ));
        }
        Ok(())
    }

    /// The two launches of the pair over the checked jobs (module doc): the scores of every vector
    /// of every row, then the outputs. One mix gives the second no work: its arguments are the
    /// first's, its output the spare, and no block reads them.
    fn pair(
        &self,
        stream: &CudaStream,
        what: &'static str,
        a: AttnResNormedArgs<'_>,
        y_a: &mut DeviceBuffer<f32>,
        b: Option<(AttnResNormedArgs<'_>, &mut DeviceBuffer<f32>)>,
        scratch: &mut MixStepScratch,
    ) -> Result<(), GpuError> {
        let m = a.m;
        crate::short(
            what,
            &[
                ("stats", scratch.stats.len(), m * JOBS * JOB_STATS),
                ("done", scratch.done.len(), m * JOBS),
            ],
        )?;
        let u = |name: &'static str, v: usize| launch_u32(what, name, v);
        let jobs = if b.is_some() { 2 } else { 1 };
        let (nb_a, nq_a, stride_a) = (
            u("n_banks", a.n_banks)?,
            u("nq", a.nq)?,
            u("stride", a.stride)?,
        );
        let n = u("n", a.n)?;
        let m32 = u("rows", m)?;
        let (bb, y_b): (AttnResNormedArgs<'_>, &mut DeviceBuffer<f32>) = match b {
            Some((b, y_b)) => (b, y_b),
            None => (
                AttnResNormedArgs {
                    nq: 1,
                    n_banks: 0,
                    ..a
                },
                &mut scratch.spare,
            ),
        };
        let need = bb.nq * m * n as usize;
        if y_b.len() < need {
            return Err(GpuError::shape(
                what,
                format!(
                    "the spare rows hold {} values, the mixes are {} wide over {m} rows",
                    y_b.len(),
                    n
                ),
            ));
        }
        let (nb_b, nq_b, stride_b) = (
            u("n_banks", bb.n_banks)?,
            u("nq", bb.nq)?,
            u("stride", bb.stride)?,
        );
        let vecs = a.n_banks + 1 + if jobs == 2 { bb.n_banks + 1 } else { 0 };
        let jobs32 = u("jobs", jobs)?;
        let prep = self
            .module
            .prepare_attn_res_step_scores(LaunchConfig1D::new(
                u("grid", vecs * m)?,
                THREADS as u32,
                0,
            ))?;
        self.module.attn_res_step_scores(
            stream,
            &prep,
            a.banks,
            a.cur,
            a.queries,
            nb_a,
            nq_a,
            stride_a,
            a.fault,
            bb.banks,
            bb.cur,
            bb.queries,
            nb_b,
            nq_b,
            stride_b,
            bb.fault,
            jobs32,
            a.scale,
            a.eps,
            n,
            m32,
            &mut scratch.stats,
            &mut scratch.done,
        )?;
        let per = a.n.div_ceil(THREADS);
        let prep = self.module.prepare_attn_res_step_out(LaunchConfig1D::new(
            u("grid", jobs * per * m)?,
            THREADS as u32,
            0,
        ))?;
        self.module.attn_res_step_out(
            stream,
            &prep,
            a.banks,
            a.cur,
            nb_a,
            nq_a,
            stride_a,
            bb.banks,
            bb.cur,
            nb_b,
            nq_b,
            stride_b,
            jobs32,
            n,
            m32,
            &scratch.stats,
            y_a,
            y_b,
        )?;
        Ok(())
    }
}
