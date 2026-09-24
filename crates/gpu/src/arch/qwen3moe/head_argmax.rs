//! The output head's Q6_K projection with the argmax folded into its
//! epilogue: one launch where the shared head runs `q6k_gemv` and then a
//! one-block `argmax_fault` over the whole vocabulary.
//!
//! Every row is `q6k_gemv`'s one-column body (the byte windows with the
//! 16-bit funnel, the dp4a chain, `f0 += a · (d8 · d · sc)`, the warp tree),
//! eight rows per 256-thread block, and lane 0 still stores the logit, so the
//! logits are bit for bit that kernel's. Then each block takes the best of
//! its eight rows and publishes it as one packed key ([`argmax_key`]) by a
//! device-wide atomic max, then draws one ticket; the block that draws the
//! last ticket writes the winner's index and the fault word into the
//! readback pair — `argmax_fault`'s layout, `out[0]` the token, `out[1]` the
//! word — and puts the key and the ticket count back to their seeds for the
//! next launch or graph replay.
//!
//! The token is `argmax_fault`'s bit for bit whatever order the blocks
//! finish in: [`argmax_take`] is a total order on (value, index) over the
//! values that are not NaN, a NaN is never taken (both of its comparisons are
//! false), and the seed `(−inf, 0)` is every reduction's start — so any tree
//! over the same candidates returns the same index, and the packed key's
//! integer order is that order.

use crate::GpuError;
use crate::cores::{funnel16, half_to_f32, q6k_chain, q6k_dequant, q6k_sub_scale};
use crate::elem::argmax_take;
use crate::fault::FaultSink;
use crate::launch_u32;
use crate::tensor::{DeviceTensor, Q8Act};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32, DeviceAtomicU64};
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread, warp,
};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Threads per block: eight warps, one vocabulary row each.
const THREADS: u32 = 256;
/// Rows per block.
const ROWS_PER_BLOCK: usize = THREADS as usize / 32;

/// The packed key of candidate `(v, i)`: the value's bits in an order whose
/// unsigned comparison is `f32`'s (`−0.0` folded onto `+0.0`, which `f32`
/// calls equal), then the complement of the index, so the larger key is the
/// larger value and, at an equal value, the lower index — [`argmax_take`]'s
/// order. `v` is never NaN: the only callers pass a value [`argmax_take`]
/// took, or the seed. Public so a gate can check the order on the host.
#[inline(always)]
#[must_use]
pub const fn argmax_key(v: f32, i: u32) -> u64 {
    let b = if v == 0.0 { 0 } else { v.to_bits() };
    let o = if b & 0x8000_0000 != 0 {
        !b
    } else {
        b | 0x8000_0000
    };
    ((o as u64) << 32) | (!i as u64)
}

/// The key of the seed `(−inf, 0)`: what the key holds between launches.
const KEY_SEED: u64 = argmax_key(f32::NEG_INFINITY, 0);

#[cuda_module]
mod qwen3moe_head_kernels {
    use super::*;

    /// `y = w · act` for a Q6_K weight of `n_rows` rows against one q8_1
    /// column, `q6k_gemv`'s geometry and one-column body, and the argmax of
    /// `y` (ties to the lower index) with the fault word into `out[0..2]`
    /// (module doc). `key[0]` and `done[0]` are the block reduction's key and
    /// ticket count: [`KEY_SEED`] and zero before the launch, the same again
    /// after it.
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
            4 * w.len() >= n_rows * 210 * n_sb,
            q.len() >= 128 * iters,
            d8.len() >= 2 * n_sb,
            y.len() >= n_rows,
            key.len() >= 1,
            done.len() >= 1,
            out.len() >= 2
        )
    )]
    pub fn qwen3moe_head_q6k_argmax(
        w: &[u32],
        q: &[u32],
        d8: &[f32],
        n_rows: u32,
        n_sb: u32,
        iters: u32,
        fault: FaultSink,
        mut y: DisjointSlice<f32>,
        mut key: DisjointSlice<u64>,
        mut done: DisjointSlice<u32>,
        mut out: DisjointSlice<u32>,
    ) {
        static mut BEST_V: SharedArray<f32, ROWS_PER_BLOCK> = SharedArray::UNINIT;
        static mut BEST_I: SharedArray<u32, ROWS_PER_BLOCK> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let wi = tid / 32;
        let lane = warp::lane_id() as usize;
        let row = thread::blockIdx_x() as usize * ROWS_PER_BLOCK + wi;
        // A warp past the rows keeps the seed and still meets the barrier.
        let mut cand_v = f32::NEG_INFINITY;
        let mut cand_i = 0u32;
        if row < n_rows as usize {
            let n_sb = n_sb as usize;
            let row_bytes = 210 * n_sb;
            let w16 = lane & 15;
            let half = lane >> 4;
            let qlok = 16 * (w16 & 3);
            let qhok = 16 * (w16 & 1);
            let nib_sh = ((w16 >> 2) & 1) as u32 * 4;
            let hib_sh = 2 * ((w16 >> 1) as u32 & 3);

            let mut f0 = 0.0f32;
            let mut it: u32 = 0;
            while it < iters {
                let sbp = ((it << 1) | half as u32) as usize;
                if sbp < n_sb {
                    let base = row * row_bytes + sbp * 210;
                    let par = (base >> 1) & 1;

                    let lk = (base + 64 * (w16 >> 3) + qlok) >> 2;
                    // SAFETY: row < n_rows, so the window ends inside the
                    // row's words (launch contract: 4·w.len() covers every
                    // row's 210·n_sb bytes; the floored word index of a
                    // window stays inside the row's words, as in q6k_gemv).
                    let (l0, l1, l2, l3, l4) = unsafe {
                        (
                            *w.get_unchecked(lk),
                            *w.get_unchecked(lk + 1),
                            *w.get_unchecked(lk + 2),
                            *w.get_unchecked(lk + 3),
                            *w.get_unchecked(lk + 4),
                        )
                    };
                    let ql0 = if par == 0 { l0 } else { funnel16(l0, l1) };
                    let ql1 = if par == 0 { l1 } else { funnel16(l1, l2) };
                    let ql2 = if par == 0 { l2 } else { funnel16(l2, l3) };
                    let ql3 = if par == 0 { l3 } else { funnel16(l3, l4) };

                    let hk = (base + 128 + 32 * (w16 >> 3) + qhok) >> 2;
                    // SAFETY: the same row bounds as the ql window.
                    let (h0, h1, h2, h3, h4) = unsafe {
                        (
                            *w.get_unchecked(hk),
                            *w.get_unchecked(hk + 1),
                            *w.get_unchecked(hk + 2),
                            *w.get_unchecked(hk + 3),
                            *w.get_unchecked(hk + 4),
                        )
                    };
                    let qh0 = if par == 0 { h0 } else { funnel16(h0, h1) };
                    let qh1 = if par == 0 { h1 } else { funnel16(h1, h2) };
                    let qh2 = if par == 0 { h2 } else { funnel16(h2, h3) };
                    let qh3 = if par == 0 { h3 } else { funnel16(h3, h4) };

                    let ak = (base + 192) >> 2;
                    // SAFETY: ak + 4 is at most the final word of the last
                    // row's last super-block; the launch contract bounds
                    // 4·w.len().
                    let (a0, a1, a2, a3, a4) = unsafe {
                        (
                            *w.get_unchecked(ak),
                            *w.get_unchecked(ak + 1),
                            *w.get_unchecked(ak + 2),
                            *w.get_unchecked(ak + 3),
                            *w.get_unchecked(ak + 4),
                        )
                    };
                    let (sw0, sw1, sw2, sw3, d_bits) = if par == 0 {
                        (a0, a1, a2, a3, (a4 & 0xffff) as u16)
                    } else {
                        (
                            funnel16(a0, a1),
                            funnel16(a1, a2),
                            funnel16(a2, a3),
                            funnel16(a3, a4),
                            (a4 >> 16) as u16,
                        )
                    };
                    let sc = q6k_sub_scale(&[sw0, sw1, sw2, sw3], w16);
                    let drow = half_to_f32(d_bits);
                    let vi = [
                        q6k_dequant(ql0, qh0, nib_sh, hib_sh),
                        q6k_dequant(ql1, qh1, nib_sh, hib_sh),
                        q6k_dequant(ql2, qh2, nib_sh, hib_sh),
                        q6k_dequant(ql3, qh3, nib_sh, hib_sh),
                    ];
                    let qb = 128 * it as usize + lane;
                    let d8b = 2 * sbp + (w16 >> 3);
                    // SAFETY: qb + 99 < 128·iters <= q.len() — this lane's
                    // four q8 words are inside the column; d8b < 2·n_sb <=
                    // d8.len() by the sbp guard (launch contract).
                    let (q0, q1, q2, q3, e0) = unsafe {
                        (
                            *q.get_unchecked(qb),
                            *q.get_unchecked(qb + 32),
                            *q.get_unchecked(qb + 64),
                            *q.get_unchecked(qb + 96),
                            *d8.get_unchecked(d8b),
                        )
                    };
                    let a = q6k_chain(&vi, &[q0, q1, q2, q3]);
                    f0 += (a as f32) * (e0 * drow * sc as f32);
                }
                it += 1;
            }
            let s0 = warp::reduce_sum_f32(f0);
            if lane == 0 {
                // SAFETY: row < n_rows <= y.len() by the launch contract;
                // only lane 0 of the row's warp writes y[row].
                unsafe {
                    *y.get_unchecked_mut(row) = s0;
                }
            }
            cand_v = s0;
            // row < n_rows, a u32: the cast is exact.
            cand_i = row as u32;
        }

        // SAFETY: both arrays are this block's own shared allocations; the
        // raw form reaches the `static mut` without a reference.
        let (bv, bi) = unsafe {
            (
                SharedArray::as_raw_mut_ptr(&raw mut BEST_V),
                SharedArray::as_raw_mut_ptr(&raw mut BEST_I),
            )
        };
        if lane == 0 {
            // SAFETY: wi < ROWS_PER_BLOCK; lane 0 of each warp writes its
            // own slot pair.
            unsafe {
                *bv.add(wi) = cand_v;
                *bi.add(wi) = cand_i;
            }
        }
        thread::sync_threads();
        if tid != 0 {
            return;
        }
        let (mut fv, mut fi) = (f32::NEG_INFINITY, 0u32);
        let mut s = 0usize;
        while s < ROWS_PER_BLOCK {
            // SAFETY: s < ROWS_PER_BLOCK, written above, past the barrier.
            let (cv, ci) = unsafe { (*bv.add(s), *bi.add(s)) };
            if argmax_take(cv, ci, fv, fi) {
                fv = cv;
                fi = ci;
            }
            s += 1;
        }
        // SAFETY: key[0] and done[0] are inside their buffers by the launch
        // contract; every access to either on the card is atomic.
        let (k, count) = unsafe {
            (
                DeviceAtomicU64::from_ptr(key.as_mut_ptr()),
                DeviceAtomicU32::from_ptr(done.as_mut_ptr()),
            )
        };
        k.fetch_max(argmax_key(fv, fi), AtomicOrdering::Relaxed);
        // The ticket's release orders this block's max before it; the last
        // block's acquire sees every block's.
        if count.fetch_add(1, AtomicOrdering::AcqRel) + 1 != thread::gridDim_x() {
            return;
        }
        let best = k.load(AtomicOrdering::Relaxed);
        k.store(KEY_SEED, AtomicOrdering::Relaxed);
        count.store(0, AtomicOrdering::Relaxed);
        let word = fault.read();
        // SAFETY: out.len() >= 2 by the launch contract; the last block's
        // thread 0 alone writes it.
        unsafe {
            *out.get_unchecked_mut(0) = !(best as u32);
            *out.get_unchecked_mut(1) = word;
        }
    }
}

/// The block reduction's device state: the packed key and the ticket count,
/// allocated at their seeds once and returned to them by every launch. One
/// launch at a time, so launches sharing it must be ordered on one stream.
pub struct HeadArgmaxState {
    key: DeviceBuffer<u64>,
    done: DeviceBuffer<u32>,
}

impl HeadArgmaxState {
    /// The key at [`KEY_SEED`] and the count at zero. Load-time only.
    pub fn new(stream: &CudaStream) -> Result<HeadArgmaxState, GpuError> {
        Ok(HeadArgmaxState {
            key: DeviceBuffer::from_host(stream, &[KEY_SEED])?,
            done: DeviceBuffer::zeroed(stream, 1)?,
        })
    }

    /// Whether the key and the count stand at their seeds, as they must
    /// between launches. Blocking read; gate use.
    pub fn at_seed(&self, stream: &CudaStream) -> Result<bool, GpuError> {
        Ok(self.key.to_host_vec(stream)?[0] == KEY_SEED && self.done.to_host_vec(stream)?[0] == 0)
    }

    /// Device bytes of the state.
    pub fn bytes(&self) -> usize {
        self.key.num_bytes() + self.done.num_bytes()
    }
}

/// The loaded head module. Owns no stream: each enqueue takes the engine
/// stream.
pub struct HeadArgmaxKernels {
    module: qwen3moe_head_kernels::LoadedModule,
}

impl HeadArgmaxKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<HeadArgmaxKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; the launcher checks its launch contract.
        let module = unsafe { qwen3moe_head_kernels::load(ctx)? };
        Ok(HeadArgmaxKernels { module })
    }

    /// Enqueue `logits = w · act` for the Q6_K head weight `w` (the word
    /// plane `enqueue_gemv_q6k` takes: `210 · n_sb / 4` words per row, `n_sb`
    /// even) against one q8_1 column, and the argmax of the logits (ties to
    /// the lower index) into `out[0]` with the fault word `fault` reads into
    /// `out[1]` — `enqueue_argmax_fault`'s readback. Asynchronous,
    /// allocation-free, capturable.
    #[allow(
        clippy::too_many_arguments,
        reason = "host launcher over the head's five buffers, the shape of the kernel's arguments"
    )]
    pub fn enqueue(
        &self,
        stream: &CudaStream,
        w: &DeviceTensor<u32>,
        act: &Q8Act,
        fault: FaultSink,
        logits: &mut DeviceBuffer<f32>,
        state: &mut HeadArgmaxState,
        out: &mut DeviceBuffer<u32>,
    ) -> Result<(), GpuError> {
        let what = "qwen3moe::head_argmax::enqueue";
        let (n_rows, n_sb) = (w.rows(), act.n_sb());
        if act.m() != 1 {
            return Err(GpuError::shape(
                what,
                format!("one activation column, got {}", act.m()),
            ));
        }
        if !n_sb.is_multiple_of(2) || w.cols() != 210 * n_sb / 4 {
            return Err(GpuError::shape(
                what,
                format!(
                    "Q6_K rows of {} words at K={}, want 210*{n_sb}/4 with an even super-block count",
                    w.cols(),
                    act.k()
                ),
            ));
        }
        if n_rows == 0 || logits.len() < n_rows || out.len() < 2 {
            return Err(GpuError::shape(
                what,
                format!(
                    "{n_rows} rows into {} logits and a readback of {}, want >= 1 rows, {n_rows} logits and 2",
                    logits.len(),
                    out.len()
                ),
            ));
        }
        let grid = launch_u32(what, "grid", n_rows.div_ceil(ROWS_PER_BLOCK))?;
        let n_rows = launch_u32(what, "n_rows", n_rows)?;
        let n_sb = launch_u32(what, "n_sb", n_sb)?;
        let prep = self
            .module
            .prepare_qwen3moe_head_q6k_argmax(LaunchConfig1D::new(grid, THREADS, 0))?;
        self.module.qwen3moe_head_q6k_argmax(
            stream,
            &prep,
            w.buf(),
            &act.q6,
            &act.d8,
            n_rows,
            n_sb,
            n_sb.div_ceil(2),
            fault,
            logits,
            &mut state.key,
            &mut state.done,
            out,
        )?;
        Ok(())
    }
}
