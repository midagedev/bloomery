//! Qwen3.8's MTP draft layer's two kernels of its own: the draft's input
//! pack and its head's argmax with the row map and the draft's probability.
//! Every other launch of the draft's program is an entry the target's chain
//! already has, called with the draft's weights and shapes
//! (`arch::qwen3moe::mtp38`).
//!
//! - [`mtp_kernels::mtp_input`] — one block a row: the two RMS norms of
//!   ik's MTP graph (`build_qwen4exp.cpp`, `is_mtp`), the embedding row `e`
//!   (`hidden` values) over itself and the target's hidden row `h` (the
//!   four streams, `4 · hidden` values) over all of it at once, each value
//!   `(x · r) · γ` with `r = 1/√(Σx²/n + eps)` as `ggml_rms_norm` then
//!   `ggml_mul`, packed stream by stream as `[e_n | h_n,s]` — row `t`, stream
//!   `s` at `(t·4 + s) · 2·hidden`, the `eh_proj` gemv's `4·m` columns. A
//!   thread takes values `t, t + 256, …` of a row by fused multiply-adds,
//!   each warp's by the xor butterfly, the eight warps' in order from warp
//!   0 — the hyper-connection norm's order (`crate::hc_gated`).
//! - [`mtp_kernels::argmax_p_rows_fault`] — one block a row of the head's
//!   logits in the gemvs' layout (`x[i·m + c]`, `n` rows of the head): the
//!   argmax walk of `crate::elem`'s argmax kernels (ties to the lower row,
//!   [`crate::elem`]'s `argmax_take`), then `p = 1/Σ exp(x_i − max)` over the
//!   same rows — the draft's largest probability among the head's rows, each
//!   thread's share in increasing `i`, the butterfly, the warp slots in
//!   order. Row `i` names token `map[i]` when the head is a row list
//!   (`mapped`), `i` itself otherwise. Block `c` writes the token to
//!   `out[c]` and `p`'s bits to `out[m + c]`; the block that draws the last
//!   ticket copies the fault word and its site mask to `out[2m]` and
//!   `out[2m + 1]` and puts the count back to zero.
//!
//! No silent failure: a row whose sums of squares are not finite raises
//! [`FaultSite::HcMix`] (`h` is the target's hyper-connection streams, `e`
//! beside it); a logit that is not finite raises [`FaultSite::Logit`], and a
//! mapped token at or past the vocabulary [`FaultSite::TokenId`]. Every
//! value is written as computed.

use crate::elem::argmax_take;
use crate::fault::{FaultSink, FaultSite};
use crate::{GpuError, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32};
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread, warp,
};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Streams of the target's hidden row.
pub const STREAMS: usize = 4;
/// Threads of an input block, and its warps.
const INPUT_THREADS: usize = 256;
const INPUT_WARPS: usize = INPUT_THREADS / 32;
/// Threads of an argmax block, and its warps.
const ARGMAX_THREADS: usize = 1024;
const ARGMAX_WARPS: usize = ARGMAX_THREADS / 32;
// The entries' launch attributes spell the blocks out as literals.
const _: () = assert!(INPUT_THREADS == 256 && ARGMAX_THREADS == 1024 && STREAMS == 4);

/// The block-wide sum of `v` over an [`INPUT_THREADS`] block: each warp's by
/// the butterfly, then the warps' in order from warp 0. Every thread calls
/// it and gets the same value (two barriers).
///
/// # Safety
/// `ws` points at [`INPUT_WARPS`] f32 of this block's shared memory that
/// nothing else uses across the call.
#[inline(always)]
unsafe fn input_sum(v: f32, tid: usize, ws: *mut f32) -> f32 {
    let s = warp::reduce_sum_f32(v);
    if tid.is_multiple_of(32) {
        // SAFETY: tid / 32 < INPUT_WARPS; lane 0 of each warp owns its slot.
        unsafe { *ws.add(tid / 32) = s };
    }
    thread::sync_threads();
    // SAFETY: every slot was written before the barrier above.
    let mut sum = unsafe { *ws };
    for w in 1..INPUT_WARPS {
        thread::__unroll_config::<0>();
        // SAFETY: w < INPUT_WARPS.
        sum += unsafe { *ws.add(w) };
    }
    thread::sync_threads();
    sum
}

#[cuda_module]
mod mtp_kernels {
    use super::*;

    /// The draft's input pack (module doc): block `t` is row `t`.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            e.len() >= m * hidden,
            h.len() >= m * 4 * hidden,
            enorm.len() >= hidden,
            hnorm.len() >= 4 * hidden,
            out.len() >= m * 8 * hidden
        )
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    pub fn mtp_input(
        e: &[f32],
        h: &[f32],
        enorm: &[f32],
        hnorm: &[f32],
        hidden: u32,
        eps: f32,
        m: u32,
        fault: FaultSink,
        mut out: DisjointSlice<f32>,
    ) {
        static mut WS: SharedArray<f32, INPUT_WARPS> = SharedArray::UNINIT;
        let t = thread::blockIdx_x() as usize;
        if t >= m as usize {
            return; // block-uniform
        }
        let tid = thread::threadIdx_x() as usize;
        let hid = hidden as usize;
        let wide = STREAMS * hid;
        // SAFETY: this block's own shared allocation, reached without a
        // reference; `input_sum` is its only user.
        let ws = unsafe { SharedArray::as_raw_mut_ptr(&raw mut WS) };
        let mut acc = 0.0f32;
        let mut i = tid;
        while i < hid {
            // SAFETY: t·hidden + i < m·hidden <= e.len().
            let x = unsafe { *e.get_unchecked(t * hid + i) };
            acc = f32::mul_add(x, x, acc);
            i += INPUT_THREADS;
        }
        // SAFETY: `ws` is INPUT_WARPS f32 of this block's shared memory.
        let se = unsafe { input_sum(acc, tid, ws) };
        let mut acc = 0.0f32;
        let mut j = tid;
        while j < wide {
            // SAFETY: t·4·hidden + j < m·4·hidden <= h.len().
            let x = unsafe { *h.get_unchecked(t * wide + j) };
            acc = f32::mul_add(x, x, acc);
            j += INPUT_THREADS;
        }
        // SAFETY: as above; `input_sum`'s closing barrier ends the first use.
        let sh = unsafe { input_sum(acc, tid, ws) };
        if tid == 0 && !(se.is_finite() && sh.is_finite()) {
            fault.raise(FaultSite::HcMix);
        }
        let re = 1.0 / (se / hidden as f32 + eps).sqrt();
        let rh = 1.0 / (sh / wide as f32 + eps).sqrt();
        let pitch = 2 * hid;
        let mut i = tid;
        while i < hid {
            // SAFETY: t·hidden + i < e.len() and i < hidden <= enorm.len(); the
            // four slots (t·4 + s)·2·hidden + i, s < 4, lie below m·8·hidden
            // <= out.len(), and value i of row t is this thread's alone.
            unsafe {
                let v = (*e.get_unchecked(t * hid + i) * re) * *enorm.get_unchecked(i);
                let base = t * STREAMS * pitch + i;
                *out.get_unchecked_mut(base) = v;
                *out.get_unchecked_mut(base + pitch) = v;
                *out.get_unchecked_mut(base + 2 * pitch) = v;
                *out.get_unchecked_mut(base + 3 * pitch) = v;
            }
            i += INPUT_THREADS;
        }
        let mut j = tid;
        while j < wide {
            let (s, i) = (j / hid, j % hid);
            // SAFETY: t·4·hidden + j < h.len() and j < 4·hidden <=
            // hnorm.len(); (t·4 + s)·2·hidden + hidden + i < m·8·hidden <=
            // out.len(), and value j of row t is this thread's alone.
            unsafe {
                let v = (*h.get_unchecked(t * wide + j) * rh) * *hnorm.get_unchecked(j);
                *out.get_unchecked_mut((t * STREAMS + s) * pitch + hid + i) = v;
            }
            j += INPUT_THREADS;
        }
    }

    /// The draft head's argmax, row map and probability (module doc): block
    /// `c` is row `c`.
    #[kernel]
    #[launch_bounds(1024)]
    #[launch_contract(
        domain = 1,
        block = (1024, 1, 1),
        requires = (
            x.len() >= n * m,
            mapped <= 1,
            map.len() >= mapped * n,
            out.len() >= 2 * m + 2,
            done.len() >= 1
        )
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    pub fn argmax_p_rows_fault(
        x: &[f32],
        n: u32,
        m: u32,
        map: &[u32],
        mapped: u32,
        vocab: u32,
        fault: FaultSink,
        mut out: DisjointSlice<u32>,
        mut done: DisjointSlice<u32>,
    ) {
        static mut BEST_V: SharedArray<f32, ARGMAX_WARPS> = SharedArray::UNINIT;
        static mut BEST_I: SharedArray<u32, ARGMAX_WARPS> = SharedArray::UNINIT;
        static mut BAD: SharedArray<u32, ARGMAX_WARPS> = SharedArray::UNINIT;
        static mut SUMS: SharedArray<f32, ARGMAX_WARPS> = SharedArray::UNINIT;
        static mut TOP: SharedArray<f32, 1> = SharedArray::UNINIT;

        let c = thread::blockIdx_x();
        if c >= m {
            return; // block-uniform
        }
        let tid = thread::threadIdx_x();
        // SAFETY: all five arrays are this block's own shared allocations;
        // the raw form is the only way to reach them without a reference to
        // a `static mut`.
        let (bv, bi, bw, bs, top) = unsafe {
            (
                SharedArray::as_raw_mut_ptr(&raw mut BEST_V),
                SharedArray::as_raw_mut_ptr(&raw mut BEST_I),
                SharedArray::as_raw_mut_ptr(&raw mut BAD),
                SharedArray::as_raw_mut_ptr(&raw mut SUMS),
                SharedArray::as_raw_mut_ptr(&raw mut TOP),
            )
        };
        let mut finite = true;
        let mut best_v = f32::NEG_INFINITY;
        let mut best_i = 0u32;
        let mut i = tid;
        while i < n {
            // SAFETY: i < n and c < m, so i·m + c < n·m <= x.len().
            let v = unsafe { *x.get_unchecked((i * m + c) as usize) };
            finite &= v.is_finite();
            if argmax_take(v, i, best_v, best_i) {
                best_v = v;
                best_i = i;
            }
            i += ARGMAX_THREADS as u32;
        }
        let bad = warp::ballot(!finite);
        let mut off = 16u32;
        while off > 0 {
            let (ov, oi) = (
                warp::shuffle_xor_f32(best_v, off),
                warp::shuffle_xor(best_i, off),
            );
            if argmax_take(ov, oi, best_v, best_i) {
                best_v = ov;
                best_i = oi;
            }
            off >>= 1;
        }
        if warp::lane_id() == 0 {
            // SAFETY: tid / 32 < ARGMAX_WARPS; one lane per warp writes its
            // own slots, which thread 0 reads past the barrier below.
            unsafe {
                *bv.add(tid as usize / 32) = best_v;
                *bi.add(tid as usize / 32) = best_i;
                *bw.add(tid as usize / 32) = bad;
            }
        }
        thread::sync_threads();
        if tid == 0 {
            // SAFETY: every slot was written above and is visible past the
            // barrier; the walk stays below ARGMAX_WARPS.
            let (mut fv, mut fi) = unsafe { (*bv.add(0), *bi.add(0)) };
            let mut w = 1usize;
            while w < ARGMAX_WARPS {
                // SAFETY: w < ARGMAX_WARPS, written above, past the barrier.
                let (cv, ci) = unsafe { (*bv.add(w), *bi.add(w)) };
                if argmax_take(cv, ci, fv, fi) {
                    fv = cv;
                    fi = ci;
                }
                w += 1;
            }
            // SAFETY: slot 0 of this block's shared arrays; thread 0 alone
            // writes them, the others read past the barrier below.
            unsafe {
                *top.add(0) = fv;
                *bi.add(0) = fi;
            }
        }
        thread::sync_threads();
        // SAFETY: written by thread 0 before the barrier above.
        let (fv, fi) = unsafe { (*top.add(0), *bi.add(0)) };
        let mut z = 0.0f32;
        let mut i = tid;
        while i < n {
            // SAFETY: as the walk above.
            let v = unsafe { *x.get_unchecked((i * m + c) as usize) };
            z += (v - fv).exp();
            i += ARGMAX_THREADS as u32;
        }
        let zs = warp::reduce_sum_f32(z);
        if warp::lane_id() == 0 {
            // SAFETY: tid / 32 < ARGMAX_WARPS; lane 0 of each warp owns its slot.
            unsafe { *bs.add(tid as usize / 32) = zs };
        }
        thread::sync_threads();
        if tid != 0 {
            return;
        }
        let mut total = 0.0f32;
        let mut any = 0u32;
        let mut w = 0usize;
        while w < ARGMAX_WARPS {
            // SAFETY: w < ARGMAX_WARPS, written above, past the barrier.
            unsafe {
                total += *bs.add(w);
                any |= *bw.add(w);
            }
            w += 1;
        }
        if any != 0 {
            fault.raise(FaultSite::Logit);
        }
        let token = if mapped == 1 {
            // SAFETY: fi < n (a walked row, or 0 with n >= 1 by the host) and
            // mapped = 1 puts n <= map.len() by the launch contract.
            unsafe { *map.get_unchecked(fi as usize) }
        } else {
            fi
        };
        if token >= vocab {
            fault.raise(FaultSite::TokenId);
        }
        // SAFETY: c < m, so c and m + c lie below 2m + 2 <= out.len(); block
        // c's thread 0 alone writes them.
        unsafe {
            *out.get_unchecked_mut(c as usize) = token;
            *out.get_unchecked_mut((m + c) as usize) = (1.0 / total).to_bits();
        }
        // SAFETY: done.len() >= 1 by the launch contract; every access to
        // done[0] on the card is atomic.
        let count = unsafe { DeviceAtomicU32::from_ptr(done.as_mut_ptr()) };
        // The ticket's release orders this block's raises before it; the last
        // block's acquire sees every block's.
        if count.fetch_add(1, AtomicOrdering::AcqRel) + 1 != m {
            return;
        }
        count.store(0, AtomicOrdering::Relaxed);
        let word = fault.read();
        let sites = fault.read_sites(word);
        // SAFETY: 2m + 1 < out.len() by the launch contract; the last block's
        // thread 0 alone writes both.
        unsafe {
            *out.get_unchecked_mut(2 * m as usize) = word;
            *out.get_unchecked_mut(2 * m as usize + 1) = sites;
        }
    }
}

/// [`MtpKernels::enqueue_mtp_input`]'s arguments: `m` embedding rows of
/// `hidden` values and `m` hidden rows of four streams, the two gains, the
/// norms' epsilon, the sink a non-finite row raises on, and the pack, `m ·
/// 4 · 2·hidden` values.
pub struct MtpInputArgs<'a> {
    pub e: &'a DeviceBuffer<f32>,
    pub h: &'a DeviceBuffer<f32>,
    pub enorm: &'a DeviceBuffer<f32>,
    pub hnorm: &'a DeviceBuffer<f32>,
    pub hidden: usize,
    pub eps: f32,
    pub m: usize,
    pub fault: FaultSink,
    pub out: &'a mut DeviceBuffer<f32>,
}

/// [`MtpKernels::enqueue_argmax_p_rows_fault`]'s arguments: `m` rows of `n`
/// logits in the gemvs' layout, the row → token map when the head is a row
/// list (`None`: row `i` is token `i`) with a stand-in word the entry
/// takes in its place, the vocabulary a token must lie below, the sink, the
/// readback (`2m + 2` words) and the ticket count (zero between launches).
pub struct ArgmaxPArgs<'a> {
    pub x: &'a DeviceBuffer<f32>,
    pub n: usize,
    pub m: usize,
    pub map: Option<&'a DeviceBuffer<u32>>,
    pub no_map: &'a DeviceBuffer<u32>,
    pub vocab: usize,
    pub fault: FaultSink,
    pub out: &'a mut DeviceBuffer<u32>,
    pub done: &'a mut DeviceBuffer<u32>,
}

/// The loaded module. Owns no stream: each enqueue takes the engine stream.
pub struct MtpKernels {
    module: mtp_kernels::LoadedModule,
}

impl MtpKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<MtpKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; the launchers check its launch contracts.
        let module = unsafe { mtp_kernels::load(ctx)? };
        Ok(MtpKernels { module })
    }

    /// Enqueue the draft's input pack over `m` rows ([`MtpInputArgs`],
    /// module doc). One launch. Asynchronous, allocation-free, capturable.
    pub fn enqueue_mtp_input(
        &self,
        stream: &CudaStream,
        a: MtpInputArgs<'_>,
    ) -> Result<(), GpuError> {
        let what = "mtp::enqueue_mtp_input";
        let (hid, m) = (a.hidden, a.m);
        let lens = [
            ("e", a.e.len(), m * hid),
            ("h", a.h.len(), m * STREAMS * hid),
            ("enorm", a.enorm.len(), hid),
            ("hnorm", a.hnorm.len(), STREAMS * hid),
            ("out", a.out.len(), m * 2 * STREAMS * hid),
        ];
        if m == 0 || hid == 0 {
            return Err(GpuError::shape(
                what,
                format!("{m} rows of {hid} values: at least one of each"),
            ));
        }
        if let Some((name, got, need)) = lens.iter().find(|(_, got, need)| got < need) {
            return Err(GpuError::shape(
                what,
                format!("{name}.len() {got} < {need}"),
            ));
        }
        let prep = self.module.prepare_mtp_input(LaunchConfig1D::new(
            launch_u32(what, "m", m)?,
            INPUT_THREADS as u32,
            0,
        ))?;
        self.module.mtp_input(
            stream,
            &prep,
            a.e,
            a.h,
            a.enorm,
            a.hnorm,
            launch_u32(what, "hidden", hid)?,
            a.eps,
            launch_u32(what, "m", m)?,
            a.fault,
            a.out,
        )?;
        Ok(())
    }

    /// Enqueue the draft head's argmax over `m` rows of `n` logits
    /// ([`ArgmaxPArgs`], module doc). A map shorter than the head's rows is
    /// refused by name. One launch. Asynchronous, allocation-free,
    /// capturable.
    pub fn enqueue_argmax_p_rows_fault(
        &self,
        stream: &CudaStream,
        a: ArgmaxPArgs<'_>,
    ) -> Result<(), GpuError> {
        let what = "mtp::enqueue_argmax_p_rows_fault";
        let (n, m) = (a.n, a.m);
        if n == 0
            || m == 0
            || a.x.len() < n * m
            || a.out.len() < 2 * m + 2
            || a.done.is_empty()
            || a.map.is_some_and(|map| map.len() < n)
            || a.no_map.is_empty()
        {
            return Err(GpuError::shape(
                what,
                format!(
                    "{m} rows of {n} logits (at least one each) in x.len() {}, out.len() {} (need \
                     2m+2), done.len() {} (need 1), a map of {:?} words (at least n), a stand-in \
                     of {}",
                    a.x.len(),
                    a.out.len(),
                    a.done.len(),
                    a.map.map(DeviceBuffer::len),
                    a.no_map.len()
                ),
            ));
        }
        // The walk indexes i·m + c in u32.
        launch_u32(what, "n*m", n * m)?;
        let prep = self
            .module
            .prepare_argmax_p_rows_fault(LaunchConfig1D::new(
                launch_u32(what, "m", m)?,
                ARGMAX_THREADS as u32,
                0,
            ))?;
        self.module.argmax_p_rows_fault(
            stream,
            &prep,
            a.x,
            launch_u32(what, "n", n)?,
            launch_u32(what, "m", m)?,
            a.map.unwrap_or(a.no_map),
            u32::from(a.map.is_some()),
            launch_u32(what, "vocab", a.vocab)?,
            a.fault,
            a.out,
            a.done,
        )?;
        Ok(())
    }
}
