//! The cross-layer residual's mix (AttnRes, Kimi K3 `src/models/kimi-k3.cpp` `res_mix`): the vector
//! a sublayer reads is a softmax-weighted sum of the banked layer inputs and the running stream.
//!
//! For a row `t`, the `V = n_banks + 1` vectors are the bank rows `bank_i[t]` (`i < n_banks`) and
//! the stream row `cur[t]`. Each vector `v` is scored by `s = Σ_j (v_j · rstd(v)) · score_j`, where
//! `rstd(v) = 1 / sqrt(mean(v²) + eps)` (an RMS norm without gain, then the layer's score vector),
//! the scores go through one softmax over the `V` vectors, and the output is `Σ_i p_i · v_i` over
//! the RAW vectors, the banks in order and the stream last. The kernel is one launch for the mix:
//! one block of [`THREADS`] threads a row, which reads every vector once for its two sums, then once
//! more for the weighted sum.
//!
//! Layout: bank `i`'s row `t` is `banks[i · stride + t · n ..][..n]`, the stream's and the output's
//! row `t` is `cur[t · n ..]` and `y[t · n ..]`; `stride >= m · n`, so a pass's rows are the step's
//! row `t` by bits (a block never reads another row).
//!
//! Numeric contract (the host rule in the gate is this list): thread `tid` takes values `tid,
//! tid + 256, …` of a vector, its two sums (`Σ v·score`, `Σ v²`) accumulated in f32 by fused
//! multiply-adds in that order; each warp's sums by the xor butterfly, the eight warps' in the fixed
//! tree `((w0 + w1) + (w2 + w3)) + ((w4 + w5) + (w6 + w7))`; the score is `rstd · dot` with `rstd`
//! the shared RMS scale ([`rms_scale_of_mean_checked`]); the softmax is `exp(s − max)` in f32,
//! summed in vector order, each divided by the sum; the output accumulates `p_i · v_i` by fused
//! multiply-adds in vector order.
//!
//! No silent failure: a vector whose sum of squares is not finite and positive, or a score that is
//! not finite, raises [`FaultSite::F32Product`] and the row's output is NaN (every probability is
//! NaN), never a plausible mix.

use crate::GpuError;
use crate::elem::rms_scale_of_mean_checked;
use crate::fault::{FaultSink, FaultSite};
use crate::launch_u32;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread, warp,
};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Threads a block gives one row.
pub const THREADS: usize = 256;
const WARPS: usize = THREADS / 32;
/// The most vectors one mix reads: the banks and the stream.
pub const MAX_VECS: usize = 16;
/// f32s of the partial sums: a dot and a sum of squares for each (vector, warp).
const PART_LEN: usize = 2 * WARPS * MAX_VECS;

const _: () = assert!(THREADS == 256 && WARPS == 8);

/// One thread's two partial sums over a vector's values `tid, tid + THREADS, …`: `Σ v · score` and
/// `Σ v²`, each accumulated by fused multiply-adds in that order.
///
/// # Safety
///
/// `src` holds `n` values from `base`, and `score` holds `n`.
#[inline(always)]
unsafe fn partials(
    (src, base): (&[f32], usize),
    score: &[f32],
    n: usize,
    tid: usize,
) -> (f32, f32) {
    let mut dot = 0.0f32;
    let mut sq = 0.0f32;
    let mut j = tid;
    while j < n {
        // SAFETY: j < n, inside the vector's span and the scores' by the caller's contract.
        let (v, w) = unsafe { (*src.get_unchecked(base + j), *score.get_unchecked(j)) };
        dot += v * w;
        sq += v * v;
        j += THREADS;
    }
    (dot, sq)
}

/// The eight warp sums at `ws + at, ws + at + 2, …` in the fixed tree of the module doc.
///
/// # Safety
///
/// `ws` addresses the eight written slots of a vector's pair `at` (0 the dots, 1 the squares).
#[inline(always)]
unsafe fn tree(ws: *const f32, i: usize, at: usize) -> f32 {
    // SAFETY: the caller's contract: eight slots of vector i, two f32s apart.
    unsafe {
        let w = |b: usize| *ws.add((i * WARPS + b) * 2 + at);
        ((w(0) + w(1)) + (w(2) + w(3))) + ((w(4) + w(5)) + (w(6) + w(7)))
    }
}

#[cuda_module]
mod attn_res_kernels {
    use super::*;

    /// `y[t] = Σ_i p_i · v_i` over the vectors of row `t` (module doc), one block a row. `n_banks +
    /// 1 <= MAX_VECS`, `m · n <= stride` and the launch's block are host-checked.
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
            banks.len() >= n_banks * stride,
            cur.len() >= m * n,
            score.len() >= n,
            y.len() >= m * n
        )
    )]
    pub fn attn_res_mix(
        banks: &[f32],
        cur: &[f32],
        score: &[f32],
        eps: f32,
        n: u32,
        m: u32,
        n_banks: u32,
        stride: u32,
        fault: FaultSink,
        mut y: DisjointSlice<f32>,
    ) {
        static mut PART: SharedArray<f32, PART_LEN> = SharedArray::UNINIT;
        static mut PROB: SharedArray<f32, MAX_VECS> = SharedArray::UNINIT;

        let t = thread::blockIdx_x() as usize;
        if t >= m as usize {
            return; // block-uniform
        }
        let tid = thread::threadIdx_x() as usize;
        let (n, nb, stride) = (n as usize, n_banks as usize, stride as usize);
        let vecs = nb + 1;
        // SAFETY: PART and PROB are this block's own shared allocations; the raw form is the only
        // way to reach them without a reference to a `static mut`. Every access is below their
        // lengths (`i < vecs <= MAX_VECS`, `tid / 32 < WARPS`) and ordered by `sync_threads`.
        let ws = unsafe { SharedArray::as_raw_mut_ptr(&raw mut PART) };
        // SAFETY: as above.
        let pr = unsafe { SharedArray::as_raw_mut_ptr(&raw mut PROB) };
        let mut i = 0usize;
        while i < vecs {
            // SAFETY: bank i's row t lies below (i + 1) · stride <= n_banks · stride <=
            // banks.len() (`m · n <= stride`, host-checked), the stream's row below m · n <=
            // cur.len(), and score.len() >= n by the launch contract.
            let (dot, sq) = unsafe {
                if i < nb {
                    partials((banks, i * stride + t * n), score, n, tid)
                } else {
                    partials((cur, t * n), score, n, tid)
                }
            };
            let dot = warp::reduce_sum_f32(dot);
            let sq = warp::reduce_sum_f32(sq);
            if warp::lane_id() == 0 {
                // SAFETY: (i · WARPS + tid / 32) · 2 + 1 < 2 · WARPS · MAX_VECS = PART_LEN; one
                // lane a warp writes its two slots.
                unsafe {
                    *ws.add((i * WARPS + tid / 32) * 2) = dot;
                    *ws.add((i * WARPS + tid / 32) * 2 + 1) = sq;
                }
            }
            i += 1;
        }
        thread::sync_threads();
        if tid == 0 {
            let mut finite = true;
            let mut mx = f32::NEG_INFINITY;
            let mut i = 0usize;
            while i < vecs {
                // SAFETY: every slot of vector i was written before the barrier above.
                let (dot, sq) = unsafe { (tree(ws, i, 0), tree(ws, i, 1)) };
                let rstd = rms_scale_of_mean_checked(sq / n as f32, eps, fault, true);
                let s = rstd * dot;
                finite &= s.is_finite();
                if s > mx {
                    mx = s;
                }
                // SAFETY: i < vecs <= MAX_VECS; thread 0 alone writes PROB here.
                unsafe { *pr.add(i) = s };
                i += 1;
            }
            let mut total = 0.0f32;
            let mut i = 0usize;
            while i < vecs {
                // SAFETY: as above.
                let e = unsafe { (*pr.add(i) - mx).exp() };
                total += e;
                // SAFETY: as above.
                unsafe { *pr.add(i) = e };
                i += 1;
            }
            if !finite {
                fault.raise(FaultSite::F32Product);
            }
            let mut i = 0usize;
            while i < vecs {
                // SAFETY: as above.
                unsafe { *pr.add(i) = if finite { *pr.add(i) / total } else { f32::NAN } };
                i += 1;
            }
        }
        thread::sync_threads();
        let mut j = tid;
        while j < n {
            let mut acc = 0.0f32;
            let mut i = 0usize;
            while i < nb {
                // SAFETY: i < nb <= MAX_VECS, and j < n puts bank i's row t inside banks as in the
                // first pass; the probabilities were written before the barrier above.
                let (p, v) = unsafe { (*pr.add(i), *banks.get_unchecked(i * stride + t * n + j)) };
                acc += p * v;
                i += 1;
            }
            // SAFETY: nb < MAX_VECS; t · n + j < m · n <= cur.len() and y.len().
            let (p, c) = unsafe { (*pr.add(nb), *cur.get_unchecked(t * n + j)) };
            acc += p * c;
            // SAFETY: t · n + j < m · n <= y.len(); this thread alone writes the value.
            unsafe { *y.get_unchecked_mut(t * n + j) = acc };
            j += THREADS;
        }
    }
}

/// The loaded module. Owns no stream: each enqueue takes the engine stream.
pub struct AttnResKernels {
    module: attn_res_kernels::LoadedModule,
}

/// What one mix reads and writes (module doc for the layout).
pub struct AttnResArgs<'a> {
    /// `n_banks` banks, bank `i`'s row `t` at `i · stride + t · n`.
    pub banks: &'a DeviceBuffer<f32>,
    pub n_banks: usize,
    /// The values between two banks: at least `m · n`.
    pub stride: usize,
    /// The running stream, `m` rows of `n`.
    pub cur: &'a DeviceBuffer<f32>,
    /// The layer's score vector, `n` values.
    pub score: &'a DeviceBuffer<f32>,
    pub eps: f32,
    pub n: usize,
    pub m: usize,
    pub fault: FaultSink,
}

impl AttnResKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<AttnResKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for the module above; the
        // launcher checks the launch contract.
        let module = unsafe { crate::shared_module!(attn_res_kernels, ctx)? };
        Ok(AttnResKernels { module })
    }

    /// Enqueue the mix of `a` into `y` (`m` rows of `n`). A shape the launch does not take is
    /// refused by name with nothing enqueued: no row or width, more vectors than [`MAX_VECS`], a
    /// stride under a pass's rows, a buffer shorter than its span, an epsilon that is not positive
    /// and finite. Asynchronous, allocation-free, capturable.
    pub fn enqueue(
        &self,
        stream: &CudaStream,
        a: AttnResArgs<'_>,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "attn_res::enqueue";
        let AttnResArgs {
            banks,
            n_banks,
            stride,
            cur,
            score,
            eps,
            n,
            m,
            fault,
        } = a;
        let bad = |detail: String| Err(GpuError::shape(WHAT, detail));
        if n == 0 || m == 0 {
            return bad(format!("{m} rows of {n} values"));
        }
        if n_banks + 1 > MAX_VECS {
            return bad(format!(
                "{n_banks} banks and the stream: the launch mixes at most {MAX_VECS} vectors"
            ));
        }
        if n_banks > 0 && stride < m * n {
            return bad(format!(
                "a stride of {stride} between banks under the pass's {m} rows of {n}"
            ));
        }
        if !(eps.is_finite() && eps > 0.0) {
            return bad(format!("epsilon {eps}: a positive finite one"));
        }
        if banks.len() < n_banks * stride || cur.len() < m * n || score.len() < n || y.len() < m * n
        {
            return bad(format!(
                "{n_banks} banks of stride {stride} in {} values, the stream {} of {}, the scores \
                 {} of {n}, the output {} of {}",
                banks.len(),
                cur.len(),
                m * n,
                score.len(),
                y.len(),
                m * n
            ));
        }
        let (n32, m32) = (launch_u32(WHAT, "n", n)?, launch_u32(WHAT, "m", m)?);
        let (nb32, stride32) = (
            launch_u32(WHAT, "n_banks", n_banks)?,
            launch_u32(WHAT, "stride", stride)?,
        );
        let prep = self
            .module
            .prepare_attn_res_mix(LaunchConfig1D::new(m32, THREADS as u32, 0))?;
        self.module.attn_res_mix(
            stream, &prep, banks, cur, score, eps, n32, m32, nb32, stride32, fault, y,
        )?;
        Ok(())
    }
}
