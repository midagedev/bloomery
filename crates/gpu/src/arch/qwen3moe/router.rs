//! The qwen3moe router: softmax over the 128 experts' logits, the top 8 by
//! probability, and their weights renormalized to sum to one
//! (`norm_topk_prob`).
//!
//! Two entries share one warp-level routing body, [`route_warp`]:
//! `qwen3moe_router_fused`, the engine's — the router's logit gemv with the
//! routing fused into it, one launch for up to [`MAX_TOKENS`] tokens — and
//! `qwen3moe_router`, the routing alone over one token's given logits. The
//! fused gemv is `q8f32::f32_gemv`'s row body — one warp per expert row,
//! `f32_lane_partials` and one fixed warp tree per column — so every logit is
//! bit for bit that kernel's. The routing needs all of a token's logits, so
//! it runs in the block that finishes last: each block publishes its rows (a
//! fence, then one atomic ticket per block), and the block that draws the
//! last ticket puts the count back to zero for the next launch or graph
//! replay and routes token `t` with its warp `t`.
//!
//! Numeric contract, op for op:
//! - the max of the 128 logits by `f32::max` (exact, so its order is free);
//! - `exp(logit − max)` in f32; the sum of the exps in f64, each lane's four
//!   (experts `lane + 32 j`, ascending `j`) and then a five-step xor
//!   butterfly over the lanes; each probability the f32 divide by `sum as
//!   f32`;
//! - the top 8 by descending probability with ties toward the smaller
//!   expert id;
//! - the chosen probabilities summed in f64 in slot order, the sum rounded
//!   once to f32, each weight the f32 divide of its probability by it.

use crate::q8f32::{f32_lane_partials, gemv_lane_sums};
use crate::route_core::take;
use crate::tensor::DeviceTensor;
use crate::{GpuError, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32};
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread, threadfence, warp,
};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Experts the router softmaxes over (`expert_count`).
pub const N_EXPERT: usize = 128;

/// Experts each token routes to (`expert_used_count`).
pub const N_USED: usize = 8;

/// Tokens one fused launch routes: the f32 gemv's column bound, and one
/// warp of the routing block per token.
pub const MAX_TOKENS: usize = 8;

/// Threads per routing-only launch: one warp.
const ROUTER_THREADS: u32 = 32;

/// Threads per fused block: eight warps, one expert row each.
const FUSED_THREADS: usize = 256;
const FUSED_THREADS_U32: u32 = FUSED_THREADS as u32;
const ROWS_PER_BLOCK: usize = FUSED_THREADS / 32;

/// Probabilities each lane of the routing warp owns: experts `lane + 32 j`.
const PER_LANE: usize = N_EXPERT / 32;

/// The fused block's shared scratch: one routing warp's entries per token.
const P_LEN: usize = MAX_TOKENS * N_EXPERT;
const SLOT_LEN: usize = MAX_TOKENS * N_USED;

const _: () = assert!(FUSED_THREADS_U32 as usize == FUSED_THREADS);
const _: () = assert!(N_EXPERT.is_multiple_of(ROWS_PER_BLOCK));
// The routing warp's lanes hold four logits each, as named scalars.
const _: () = assert!(PER_LANE == 4);
// The routing block has a warp for every token.
const _: () = assert!(MAX_TOKENS <= ROWS_PER_BLOCK);

/// One token's routing by one warp, the module doc's contract: lane `L`
/// brings the token's logits of experts `L + 32 j` in `v`; `p` and `slot_p`
/// are the token's [`N_EXPERT`] and [`N_USED`] shared entries. Writes
/// `probs[t·128 + e]` for every expert, then `ids[t·8 + s]` and
/// `weights[t·8 + s]` in rank order. The selection runs eight rounds of a
/// butterfly argmax under [`take`]`::<false>` (the order the serial scan's
/// strict `>` over ascending ids realizes, seed `(−inf, 0)`), the winning
/// lane marking its entry taken; lane 0 writes the ids, then the weights.
///
/// # Safety
///
/// All 32 lanes of the warp call it, converged, with the same `p`, `slot_p`
/// and `t`; `p` and `slot_p` point to [`N_EXPERT`] and [`N_USED`] entries of
/// the block's shared memory that no other warp touches; `probs.len() >= (t +
/// 1)·128`, and `ids.len()`, `weights.len() >= (t + 1)·8`, those slots
/// written by this warp alone.
#[inline(always)]
unsafe fn route_warp(
    v: (f32, f32, f32, f32),
    p: *mut f32,
    slot_p: *mut f32,
    t: usize,
    probs: &mut DisjointSlice<f32>,
    ids: &mut DisjointSlice<u32>,
    weights: &mut DisjointSlice<f32>,
) {
    let lane = warp::lane_id() as usize;
    let (v0, v1, v2, v3) = v;
    let mut mx = f32::NEG_INFINITY.max(v0).max(v1).max(v2).max(v3);
    let mut off = 16u32;
    while off > 0 {
        mx = mx.max(warp::shuffle_xor_f32(mx, off));
        off >>= 1;
    }
    let (e0, e1, e2, e3) = (
        (v0 - mx).exp(),
        (v1 - mx).exp(),
        (v2 - mx).exp(),
        (v3 - mx).exp(),
    );
    let mut sum = f64::from(e0) + f64::from(e1) + f64::from(e2) + f64::from(e3);
    let mut off = 16u32;
    while off > 0 {
        // Both lanes of a pair add the same two values, so every lane leaves
        // with the same sum.
        sum += warp::shuffle_xor_f64(sum, off);
        off >>= 1;
    }
    let inv = sum as f32;
    let (p0, p1, p2, p3) = (e0 / inv, e1 / inv, e2 / inv, e3 / inv);
    let pb = t * N_EXPERT + lane;
    // SAFETY: lane + 96 < N_EXPERT — inside the token's shared entries and
    // its probs slots (this fn's contract); entries `lane + 32 j` are this
    // lane's own.
    unsafe {
        *p.add(lane) = p0;
        *p.add(lane + 32) = p1;
        *p.add(lane + 64) = p2;
        *p.add(lane + 96) = p3;
        *probs.get_unchecked_mut(pb) = p0;
        *probs.get_unchecked_mut(pb + 32) = p1;
        *probs.get_unchecked_mut(pb + 64) = p2;
        *probs.get_unchecked_mut(pb + 96) = p3;
    }
    // Lane 0 reads each round's winner, whichever lane wrote it.
    warp::sync_mask(u32::MAX);

    let mut taken = 0u32;
    let mut s = 0usize;
    while s < N_USED {
        let mut bv = f32::NEG_INFINITY;
        let mut bi = 0u32;
        let mut j = 0usize;
        while j < PER_LANE {
            if (taken >> j) & 1 == 0 {
                let ej = lane + 32 * j;
                // SAFETY: ej < 32·PER_LANE = N_EXPERT, this lane's own entry.
                let v = unsafe { *p.add(ej) };
                if take::<false>(v, ej as u32, bv, bi) {
                    bv = v;
                    bi = ej as u32;
                }
            }
            j += 1;
        }
        let mut off = 16u32;
        while off > 0 {
            let (ov, oi) = (warp::shuffle_xor_f32(bv, off), warp::shuffle_xor(bi, off));
            if take::<false>(ov, oi, bv, bi) {
                bv = ov;
                bi = oi;
            }
            off >>= 1;
        }
        // Every lane leaves the butterfly with the same winner; its owner
        // marks it taken.
        if bi as usize % 32 == lane {
            taken |= 1 << (bi as usize / 32);
        }
        if lane == 0 {
            // SAFETY: bi < N_EXPERT — inside the token's entries, published
            // by the warp sync above; s < N_USED, inside `slot_p` and the
            // token's ids slots (this fn's contract). The probability is read
            // from the winner's entry rather than from the comparison, so one
            // no comparison won is carried through unchanged.
            unsafe {
                *slot_p.add(s) = *p.add(bi as usize);
                *ids.get_unchecked_mut(t * N_USED + s) = bi;
            }
        }
        s += 1;
    }
    if lane == 0 {
        let mut sum = 0.0f64;
        let mut s = 0usize;
        while s < N_USED {
            // SAFETY: s < N_USED, this lane's own write.
            sum += f64::from(unsafe { *slot_p.add(s) });
            s += 1;
        }
        let sum = sum as f32;
        let mut s = 0usize;
        while s < N_USED {
            // SAFETY: s < N_USED, inside `slot_p` and the token's weights
            // slots (this fn's contract); lane 0 is the only writer.
            unsafe { *weights.get_unchecked_mut(t * N_USED + s) = *slot_p.add(s) / sum };
            s += 1;
        }
    }
}

/// Lane 0's stores of one expert row's `m` column logits, token-major:
/// `y[c · rows + r] = sums[c]`. One guarded store per column with a constant
/// index: a loop over `c` would index `sums` at run time and put it in a
/// local depot.
///
/// # Safety
///
/// `1 <= m <= 8`, `r < rows`, `m · rows <= y.len()`, and no other thread
/// writes those slots.
#[inline(always)]
unsafe fn store_cols(y: &mut DisjointSlice<f32>, r: usize, rows: usize, m: usize, sums: &[f32; 8]) {
    let [s0, s1, s2, s3, s4, s5, s6, s7] = *sums;
    // SAFETY: each store is guarded by c < m, so its slot c·rows + r is one
    // this fn's contract puts inside y and gives to this thread alone.
    unsafe {
        *y.get_unchecked_mut(r) = s0;
        if m > 1 {
            *y.get_unchecked_mut(rows + r) = s1;
        }
        if m > 2 {
            *y.get_unchecked_mut(2 * rows + r) = s2;
        }
        if m > 3 {
            *y.get_unchecked_mut(3 * rows + r) = s3;
        }
        if m > 4 {
            *y.get_unchecked_mut(4 * rows + r) = s4;
        }
        if m > 5 {
            *y.get_unchecked_mut(5 * rows + r) = s5;
        }
        if m > 6 {
            *y.get_unchecked_mut(6 * rows + r) = s6;
        }
        if m > 7 {
            *y.get_unchecked_mut(7 * rows + r) = s7;
        }
    }
}

#[cuda_module]
mod qwen3moe_router_kernels {
    use super::*;

    /// The routing alone for one token: its [`N_EXPERT`] logits in `x` →
    /// every probability in `probs`, the ids in rank order in `ids` and
    /// their renormalized weights in `weights`. One block of one warp,
    /// [`route_warp`].
    #[kernel]
    #[launch_bounds(32)]
    #[launch_contract(
        domain = 1,
        block = (32, 1, 1),
        requires = (
            x.len() >= 128,
            probs.len() >= 128,
            ids.len() >= 8,
            weights.len() >= 8
        )
    )]
    pub fn qwen3moe_router(
        x: &[f32],
        mut probs: DisjointSlice<f32>,
        mut ids: DisjointSlice<u32>,
        mut weights: DisjointSlice<f32>,
    ) {
        static mut P: SharedArray<f32, N_EXPERT> = SharedArray::UNINIT;
        static mut SLOT_P: SharedArray<f32, N_USED> = SharedArray::UNINIT;

        let lane = warp::lane_id() as usize;
        // SAFETY: block-shared, N_EXPERT entries; the raw form reaches the
        // `static mut` without a reference.
        let p = unsafe { SharedArray::as_raw_mut_ptr(&raw mut P) };
        // SAFETY: block-shared, N_USED entries; the raw form reaches the
        // `static mut` without a reference.
        let slot_p = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SLOT_P) };
        // SAFETY: lane + 96 < 128 <= x.len() by the launch contract.
        let v = unsafe {
            (
                *x.get_unchecked(lane),
                *x.get_unchecked(lane + 32),
                *x.get_unchecked(lane + 64),
                *x.get_unchecked(lane + 96),
            )
        };
        // SAFETY: the block is this one warp, converged here; P and SLOT_P
        // are its own; token 0's slots are inside probs, ids and weights by
        // the launch contract.
        unsafe { route_warp(v, p, slot_p, 0, &mut probs, &mut ids, &mut weights) };
    }

    /// The router of `m_cols` tokens: `w` the router weight ([`N_EXPERT`]
    /// rows of `k` f32), `x` the tokens' normed activations (`m_cols`
    /// columns of `k`). Writes `logits[t·128 + e]`, `probs[t·128 + e]`,
    /// `ids[t·8 + s]` and `weights[t·8 + s]`. `done[0]` is the block ticket
    /// count: zero before the launch, zero again after it.
    ///
    /// Grid `N_EXPERT / 8` blocks of 256; warp `r % 8` of block `r / 8` owns
    /// row `r` — `f32_gemv`'s geometry and row body. After its rows every
    /// thread fences and the block meets at a barrier, so the rows are
    /// visible device-wide before thread 0 draws the block's ticket. The
    /// block that draws the last ticket returns the count to zero, and its
    /// warp `t < m_cols` reads token `t`'s logits back (volatile loads: this
    /// block's L1 never held other blocks' rows, and a volatile load does not
    /// ask it) and runs [`route_warp`].
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
            m_cols >= 1,
            m_cols <= 8,
            w.len() >= 128 * k,
            x.len() >= m_cols * k,
            logits.len() >= 128 * m_cols,
            probs.len() >= 128 * m_cols,
            ids.len() >= 8 * m_cols,
            weights.len() >= 8 * m_cols,
            done.len() >= 1
        )
    )]
    pub fn qwen3moe_router_fused(
        w: &[f32],
        x: &[f32],
        k: u32,
        m_cols: u32,
        mut logits: DisjointSlice<f32>,
        mut probs: DisjointSlice<f32>,
        mut ids: DisjointSlice<u32>,
        mut weights: DisjointSlice<f32>,
        mut done: DisjointSlice<u32>,
    ) {
        static mut P: SharedArray<f32, P_LEN> = SharedArray::UNINIT;
        static mut SLOT_P: SharedArray<f32, SLOT_LEN> = SharedArray::UNINIT;
        static mut LAST: SharedArray<u32, 1> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let lane = warp::lane_id() as usize;
        let wi = tid / 32;
        let m = m_cols as usize;
        // The row guard is warp-uniform, so the warp tree sees a full warp;
        // a thread past the rows still meets the barrier and the ticket.
        let row = thread::blockIdx_x() as usize * ROWS_PER_BLOCK + wi;
        if row < N_EXPERT {
            let sums = gemv_lane_sums(f32_lane_partials(w, x, k, row, m_cols, lane), m_cols);
            if lane == 0 {
                // SAFETY: row < N_EXPERT and 1 <= m <= 8, so the slots c·128 +
                // row (c < m) lie inside logits (launch contract); lane 0 of
                // the row's warp is their only writer.
                unsafe { store_cols(&mut logits, row, N_EXPERT, m, &sums) };
            }
        }
        threadfence();
        thread::sync_threads();

        // SAFETY: block-shared, one element; the raw form reaches the
        // `static mut` without a reference.
        let last = unsafe { SharedArray::as_raw_mut_ptr(&raw mut LAST) };
        let ticket = done.as_mut_ptr();
        if tid == 0 {
            // SAFETY: `ticket` is done[0], inside `done` by the launch
            // contract; every access to it is atomic.
            let count = unsafe { DeviceAtomicU32::from_ptr(ticket) };
            let is_last = count.fetch_add(1, AtomicOrdering::AcqRel) + 1 == thread::gridDim_x();
            if is_last {
                // Every block has drawn its ticket.
                count.store(0, AtomicOrdering::Relaxed);
            }
            // SAFETY: block-shared, one element, written before the barrier
            // that publishes it.
            unsafe { *last = u32::from(is_last) };
        }
        thread::sync_threads();
        // SAFETY: block-shared, one element, written by thread 0 before the
        // barrier above.
        if unsafe { *last } == 0 || wi >= m {
            return;
        }

        // The last block, warp `wi` routing token `wi`: every block's rows
        // are published.
        let base = wi * N_EXPERT;
        let lp = logits.as_mut_ptr();
        // SAFETY: base + lane + 96 < m·128 <= logits.len() (launch contract).
        let v = unsafe {
            (
                core::ptr::read_volatile(lp.add(base + lane)),
                core::ptr::read_volatile(lp.add(base + lane + 32)),
                core::ptr::read_volatile(lp.add(base + lane + 64)),
                core::ptr::read_volatile(lp.add(base + lane + 96)),
            )
        };
        // SAFETY: block-shared, P_LEN = MAX_TOKENS·N_EXPERT entries and wi <
        // m <= MAX_TOKENS: the token's N_EXPERT entries; the raw form reaches
        // the `static mut` without a reference.
        let p = unsafe { SharedArray::as_raw_mut_ptr(&raw mut P).add(base) };
        // SAFETY: block-shared, SLOT_LEN = MAX_TOKENS·N_USED entries: the
        // token's N_USED entries, as above.
        let slot_p = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SLOT_P).add(wi * N_USED) };
        // SAFETY: the whole warp is here (`wi` is warp-uniform), converged
        // past the barrier; `p` and `slot_p` are token wi's alone; its slots
        // are inside probs, ids and weights by the launch contract (wi < m).
        unsafe { route_warp(v, p, slot_p, wi, &mut probs, &mut ids, &mut weights) };
    }
}

/// Where one router launch leaves its results, allocated once and reused by
/// every launch and replay: per token the logits (the fused launch's) and
/// the probabilities, per slot the expert id and the weight, and the fused
/// launch's block ticket count, which it returns to zero. The count serves
/// one launch at a time, so launches that share a `RouterOut` must be
/// ordered on one stream.
pub struct RouterOut {
    pub logits: DeviceBuffer<f32>,
    pub probs: DeviceBuffer<f32>,
    pub ids: DeviceBuffer<u32>,
    pub weights: DeviceBuffer<f32>,
    done: DeviceBuffer<u32>,
    tokens: usize,
}

impl RouterOut {
    /// Allocate the buffers for one token, the ticket count zeroed.
    /// Load-time only.
    pub fn new(stream: &CudaStream) -> Result<RouterOut, GpuError> {
        RouterOut::with_tokens(stream, 1)
    }

    /// Allocate the buffers for up to `tokens` (1..=[`MAX_TOKENS`]) tokens
    /// per launch. Load-time only.
    pub fn with_tokens(stream: &CudaStream, tokens: usize) -> Result<RouterOut, GpuError> {
        if !(1..=MAX_TOKENS).contains(&tokens) {
            return Err(GpuError::shape(
                "qwen3moe::router::RouterOut::with_tokens",
                format!("{tokens} tokens, want 1..={MAX_TOKENS}"),
            ));
        }
        Ok(RouterOut {
            logits: DeviceBuffer::zeroed(stream, tokens * N_EXPERT)?,
            probs: DeviceBuffer::zeroed(stream, tokens * N_EXPERT)?,
            ids: DeviceBuffer::zeroed(stream, tokens * N_USED)?,
            weights: DeviceBuffer::zeroed(stream, tokens * N_USED)?,
            done: DeviceBuffer::zeroed(stream, 1)?,
            tokens,
        })
    }

    /// The tokens one launch into these buffers may route.
    pub fn tokens(&self) -> usize {
        self.tokens
    }

    /// The ticket count as it stands: zero between launches.
    pub fn tickets(&self, stream: &CudaStream) -> Result<u32, GpuError> {
        Ok(self.done.to_host_vec(stream)?[0])
    }

    /// Device bytes of the buffers.
    pub fn bytes(&self) -> usize {
        self.logits.num_bytes()
            + self.probs.num_bytes()
            + self.ids.num_bytes()
            + self.weights.num_bytes()
            + self.done.num_bytes()
    }
}

/// The loaded router module. Owns no stream: each enqueue takes the engine
/// stream, so its launch orders with the rest of the step and capture.
pub struct RouterKernels {
    module: qwen3moe_router_kernels::LoadedModule,
}

impl RouterKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<RouterKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; the launcher checks its launch contract.
        let module = unsafe { qwen3moe_router_kernels::load(ctx)? };
        Ok(RouterKernels { module })
    }

    /// Enqueue the routing alone of one token over its `x` ([`N_EXPERT`]
    /// logits) into `out`'s first token. Asynchronous, allocation-free,
    /// capturable.
    pub fn enqueue(
        &self,
        stream: &CudaStream,
        x: &DeviceBuffer<f32>,
        out: &mut RouterOut,
    ) -> Result<(), GpuError> {
        if x.len() < N_EXPERT {
            return Err(GpuError::shape(
                "qwen3moe::router::enqueue",
                format!("x.len() {} needs {N_EXPERT}", x.len()),
            ));
        }
        let prep =
            self.module
                .prepare_qwen3moe_router(LaunchConfig1D::new(1, ROUTER_THREADS, 0))?;
        self.module.qwen3moe_router(
            stream,
            &prep,
            x,
            &mut out.probs,
            &mut out.ids,
            &mut out.weights,
        )?;
        Ok(())
    }

    /// Enqueue the router of `m` (1..=`out.tokens()`) tokens: `w` the
    /// router weight as f32 ([`N_EXPERT`] rows of `k`, `k` a positive
    /// multiple of 32), `x` the tokens' `m` columns of `k` activations;
    /// results into `out`. Asynchronous, allocation-free, capturable.
    pub fn enqueue_fused(
        &self,
        stream: &CudaStream,
        w: &DeviceTensor<f32>,
        x: &DeviceBuffer<f32>,
        m: usize,
        out: &mut RouterOut,
    ) -> Result<(), GpuError> {
        let what = "qwen3moe::router::enqueue_fused";
        let k = w.cols();
        if w.rows() != N_EXPERT || k == 0 || !k.is_multiple_of(32) {
            return Err(GpuError::shape(
                what,
                format!(
                    "router weight is {} x {k}, want {N_EXPERT} rows of a positive multiple of 32",
                    w.rows()
                ),
            ));
        }
        if !(1..=out.tokens).contains(&m) || x.len() < m * k {
            return Err(GpuError::shape(
                what,
                format!(
                    "{m} tokens into buffers for {}, x.len() {} for {m} x {k}",
                    out.tokens,
                    x.len()
                ),
            ));
        }
        let k = launch_u32(what, "k", k)?;
        let m = launch_u32(what, "m", m)?;
        let grid = launch_u32(what, "grid", N_EXPERT / ROWS_PER_BLOCK)?;
        let prep = self
            .module
            .prepare_qwen3moe_router_fused(LaunchConfig1D::new(grid, FUSED_THREADS_U32, 0))?;
        self.module.qwen3moe_router_fused(
            stream,
            &prep,
            w.buf(),
            x,
            k,
            m,
            &mut out.logits,
            &mut out.probs,
            &mut out.ids,
            &mut out.weights,
            &mut out.done,
        )?;
        Ok(())
    }
}
