//! The looped core's message controller (Inkling 2, llama.cpp PR #32695's
//! `src/models/inkling2.cpp` `control`, calls at its layer walk): the merge of a message mixed
//! from the previous pass's banked blocks into a sublayer's input,
//! `out = (1 − tanh(F·h)) · cur + tanh(G·h) · msg` with `h = D · [rms(cur); rms(msg)]` — the two
//! halves gainless-RMS-normed on their own at `eps`, the whole row down-projected to the gate
//! rank, and the gain `G` and forget `F` rows of the rank writing the hidden row back. The
//! reference computes f32 throughout (the weights are f32 in the file); so does the card.
//!
//! One launch covers one module of a core layer (the attention, k, v or feed-forward one: its
//! own `down`, `gain` and `forget` rows over its own `cur` and `msg` rows). A visit enqueues it
//! once a module — the three attention modules' inputs are ready together after the message
//! mixes, the feed-forward's only after the block join, so the four never share a launch — and
//! every module of every core visit of a pass ≥ 1 runs it (`inkling2.cpp`'s walk: the q/k/v
//! controllers before the input norms, the ffn's after the feed-forward mix).
//!
//! Layouts, in the file's own memory order (ggml's `ne[0]`-contiguous; the launch takes each
//! weight as the file lays it, no repack):
//! - `cur` and `msg` `[m][n]`, `y` `[m][n]`, token-major;
//! - `down` `[rank][2n]`: output `r`'s 2n weights contiguous, the input `k` of `[cur; msg]`
//!   fastest — a warp's lanes read consecutive `k`, one dot a warp;
//! - `gain` and `forget` `[n][rank]`: output `j`'s `rank` weights contiguous at `j · rank`, the
//!   rank fastest — a thread's dot reads its own output's row.
//!
//! Geometry: one block of [`THREADS`] threads a row, `m` blocks; no grid-wide sync. The row's
//! raw `[cur; msg]` is staged into shared memory once ([`HNORM`], `2 · MAX_N` f32, the family's
//! widest hidden), the two sums of squares reduced in the crate's order (thread-strided f32
//! fused multiply-adds, the warp's by the xor butterfly, the eight warps' in the fixed tree),
//! the two `rstd` scales broadcast through shared, the staged halves scaled in place, and the
//! `rank` down-dots read the staged row from shared (a dot a warp over the whole row, the
//! lane-strided f32 FMA order the crate's gemv bodies use). The gain and forget dots run
//! value-parallel, each thread its output `j` over the rank ascending from the shared `h`; the
//! `tanh` is the device's own. Registers stay near the staging's; the shared total is
//! `2 · MAX_N + MAX_RANK + 2 · WARPS + 2` f32.
//!
//! Numeric contract: the two squares and the down and gain/forget dots accumulate by fused
//! multiply-adds ascending their fast index (the first product rounds alone), each warp's
//! partials by the xor butterfly; the staged normed value `v · rstd` rounds on its own; the
//! combine is `mul_rn((1 − tanh(F·h)), cur) + mul_rn(tanh(G·h), msg)` — the reference's
//! `add(mul(1 − F, cur), mul(G, msg))` (`inkling2.cpp:236`), one rounding a multiply.
//!
//! No silent failure: a sum of squares that is not finite and positive, or an output that stops
//! being finite (a non-finite input, dot or `tanh` argument), raises [`FaultSite::F32Product`]
//! and the value written there is NaN, never a plausible merge.

use crate::GpuError;
use crate::elem::rms_scale_of_mean_checked;
use crate::fault::{FaultSink, FaultSite};
use crate::launch_u32;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::float::{add_rn_f32, fma_rn_f32, mul_rn_f32};
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread, warp,
};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Threads a block gives one row.
pub const THREADS: usize = 256;
const WARPS: usize = THREADS / 32;
/// The widest hidden the entry stages: the family's largest file's (the Small's 4,608; the
/// looped files' is 2,304). The host refuses a wider one by name.
pub const MAX_N: usize = 4608;
/// The widest gate rank the entry takes: the looped core's (64 both looped files). The host
/// refuses a wider one by name.
pub const MAX_RANK: usize = 64;

const _: () = assert!(THREADS == 256 && WARPS == 8);

/// The staged raw `[cur; msg]` row.
const HNORM_LEN: usize = 2 * MAX_N;
/// The down-projection's row.
const H_LEN: usize = MAX_RANK;
/// The squares' warp partials: a pair a warp.
const PART_LEN: usize = 2 * WARPS;
/// The two scales.
const RSTD_LEN: usize = 2;

/// The eight warp sums at `ws + 2 · b + at` in the fixed tree `((w0 + w1) + (w2 + w3)) +
/// ((w4 + w5) + (w6 + w7))`.
///
/// # Safety
///
/// `ws` addresses the eight written pairs of `at` (0 cur's squares, 1 msg's).
#[inline(always)]
unsafe fn tree(ws: *const f32, at: usize) -> f32 {
    // SAFETY: the caller's contract: eight pairs at `at`.
    unsafe {
        let w = |b: usize| *ws.add(b * 2 + at);
        ((w(0) + w(1)) + (w(2) + w(3))) + ((w(4) + w(5)) + (w(6) + w(7)))
    }
}

#[cuda_module]
mod loop_control_kernels {
    use super::*;

    /// The controller of one module over `m` rows (module doc), one block a row. `n` and `rank`
    /// are runtime arguments bounded by [`MAX_N`] and [`MAX_RANK`]; the shapes are host-checked.
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
            cur.len() >= m * n,
            msg.len() >= m * n,
            down.len() >= rank * 2 * n,
            gain.len() >= n * rank,
            forget.len() >= n * rank,
            y.len() >= m * n,
            m >= 1,
            n >= 1,
            n <= 4608,
            rank >= 1,
            rank <= 64
        )
    )]
    pub fn loop_control(
        cur: &[f32],
        msg: &[f32],
        down: &[f32],
        gain: &[f32],
        forget: &[f32],
        n: u32,
        rank: u32,
        m: u32,
        eps: f32,
        fault: FaultSink,
        mut y: DisjointSlice<f32>,
    ) {
        static mut HNORM: SharedArray<f32, HNORM_LEN> = SharedArray::UNINIT;
        static mut H: SharedArray<f32, H_LEN> = SharedArray::UNINIT;
        static mut PART: SharedArray<f32, PART_LEN> = SharedArray::UNINIT;
        static mut RSTD: SharedArray<f32, RSTD_LEN> = SharedArray::UNINIT;

        let t = thread::blockIdx_x() as usize;
        if t >= m as usize {
            return; // block-uniform
        }
        let tid = thread::threadIdx_x() as usize;
        let (n, rank) = (n as usize, rank as usize);
        // SAFETY: the block's own shared allocations; the raw form is the only way to reach them
        // without a reference to a `static mut`. Every access is below its length (`j < 2n <=
        // HNORM_LEN`, `r < rank <= H_LEN`, `tid / 32 < WARPS`) and ordered by `sync_threads`.
        let hn = unsafe { SharedArray::as_raw_mut_ptr(&raw mut HNORM) };
        // SAFETY: as above.
        let h = unsafe { SharedArray::as_raw_mut_ptr(&raw mut H) };
        // SAFETY: as above.
        let ws = unsafe { SharedArray::as_raw_mut_ptr(&raw mut PART) };
        // SAFETY: as above.
        let rs = unsafe { SharedArray::as_raw_mut_ptr(&raw mut RSTD) };

        // Stage the raw row and its two sums of squares.
        let (mut sq_c, mut sq_m) = (0.0f32, 0.0f32);
        let mut j = tid;
        while j < n {
            // SAFETY: t · n + j < m · n <= cur.len() and msg.len() by the contract; this thread
            // alone writes the staged value (j and n + j, one a thread).
            let (c, g) = unsafe { (*cur.get_unchecked(t * n + j), *msg.get_unchecked(t * n + j)) };
            sq_c += c * c;
            sq_m += g * g;
            // SAFETY: j and n + j < 2n <= HNORM_LEN; this thread alone writes them.
            unsafe {
                *hn.add(j) = c;
                *hn.add(n + j) = g;
            }
            j += THREADS;
        }
        let sq_c = warp::reduce_sum_f32(sq_c);
        let sq_m = warp::reduce_sum_f32(sq_m);
        if warp::lane_id() == 0 {
            // SAFETY: tid / 32 < WARPS; one lane a warp writes its pair.
            unsafe {
                *ws.add((tid / 32) * 2) = sq_c;
                *ws.add((tid / 32) * 2 + 1) = sq_m;
            }
        }
        thread::sync_threads();
        if tid == 0 {
            // SAFETY: every pair was written before the barrier above.
            let mean_c = unsafe { tree(ws, 0) } / n as f32;
            // SAFETY: as above.
            let mean_m = unsafe { tree(ws, 1) } / n as f32;
            let rstd_c = rms_scale_of_mean_checked(mean_c, eps, fault, true);
            let rstd_m = rms_scale_of_mean_checked(mean_m, eps, fault, true);
            // SAFETY: thread 0 alone writes RSTD here.
            unsafe {
                *rs.add(0) = rstd_c;
                *rs.add(1) = rstd_m;
            }
        }
        thread::sync_threads();
        // SAFETY: written before the barrier above.
        let (rstd_c, rstd_m) = unsafe { (*rs.add(0), *rs.add(1)) };

        // Scale the staged halves in place: the normed values the down-dots read.
        let mut j = tid;
        while j < n {
            // SAFETY: j < 2n <= HNORM_LEN; this thread alone writes the value.
            unsafe {
                *hn.add(j) = mul_rn_f32(*hn.add(j), rstd_c);
                *hn.add(n + j) = mul_rn_f32(*hn.add(n + j), rstd_m);
            }
            j += THREADS;
        }
        thread::sync_threads();

        // The down-dots: warp `w` takes output `r = w, w + WARPS, …`, lane `l` the row's values
        // `l, l + 32, …`, ascending.
        let mut ok = true;
        let mut r = tid / 32;
        while r < rank {
            let mut acc = 0.0f32;
            let mut k = tid % 32;
            let mut first = true;
            while k < 2 * n {
                // SAFETY: r · 2n + k < rank · 2n <= down.len() by the contract; k < 2n <=
                // HNORM_LEN, written before the barrier above.
                let w = unsafe { *down.get_unchecked(r * 2 * n + k) };
                // SAFETY: as above.
                let v = unsafe { *hn.add(k) };
                acc = if first {
                    first = false;
                    mul_rn_f32(w, v)
                } else {
                    fma_rn_f32(w, v, acc)
                };
                k += 32;
            }
            let acc = warp::reduce_sum_f32(acc);
            if warp::lane_id() == 0 {
                // SAFETY: r < rank <= H_LEN; one lane writes the output.
                unsafe { *h.add(r) = acc };
            }
            r += WARPS;
        }
        thread::sync_threads();

        // The gain and forget rows, the combine: each thread its output value.
        let mut j = tid;
        while j < n {
            let mut g = 0.0f32;
            let mut f = 0.0f32;
            let mut first = true;
            let mut d = 0usize;
            while d < rank {
                // SAFETY: j · rank + d < n · rank <= gain.len() and forget.len() by the contract;
                // d < rank <= H_LEN, written before the barrier above.
                let (gw, fw) = unsafe {
                    (
                        *gain.get_unchecked(j * rank + d),
                        *forget.get_unchecked(j * rank + d),
                    )
                };
                // SAFETY: as above.
                let hv = unsafe { *h.add(d) };
                if first {
                    first = false;
                    g = mul_rn_f32(gw, hv);
                    f = mul_rn_f32(fw, hv);
                } else {
                    g = fma_rn_f32(gw, hv, g);
                    f = fma_rn_f32(fw, hv, f);
                }
                d += 1;
            }
            let one_m_f = 1.0 - f.tanh();
            let gm = g.tanh();
            // SAFETY: t · n + j < m · n <= cur.len() and msg.len() by the contract.
            let (cv, mv) =
                unsafe { (*cur.get_unchecked(t * n + j), *msg.get_unchecked(t * n + j)) };
            let out = add_rn_f32(mul_rn_f32(one_m_f, cv), mul_rn_f32(gm, mv));
            let fin = out.is_finite();
            ok &= fin;
            // SAFETY: t · n + j < m · n <= y.len(); this thread alone writes the value.
            unsafe { *y.get_unchecked_mut(t * n + j) = if fin { out } else { f32::NAN } };
            j += THREADS;
        }
        if !ok {
            fault.raise(FaultSite::F32Product);
        }
    }
}

/// The loaded module. Owns no stream: each enqueue takes the engine stream.
pub struct LoopControlKernels {
    module: loop_control_kernels::LoadedModule,
}

/// What one controller reads and writes (module doc for the layouts).
pub struct LoopControlArgs<'a> {
    /// The sublayer's input, `m` rows of `n` (the mixed input the message merges into).
    pub cur: &'a DeviceBuffer<f32>,
    /// The message, `m` rows of `n` (its mix's row).
    pub msg: &'a DeviceBuffer<f32>,
    /// The module's down projection, `[rank][2n]` as the file lays it.
    pub down: &'a DeviceBuffer<f32>,
    /// The module's gain rows, `[n][rank]` as the file lays it.
    pub gain: &'a DeviceBuffer<f32>,
    /// The module's forget rows, `[n][rank]` as the file lays it.
    pub forget: &'a DeviceBuffer<f32>,
    pub n: usize,
    pub rank: usize,
    pub m: usize,
    pub eps: f32,
    pub fault: FaultSink,
}

impl LoopControlKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<LoopControlKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for the module above; the
        // launcher checks the launch contract.
        let module = unsafe { crate::shared_module!(loop_control_kernels, ctx)? };
        Ok(LoopControlKernels { module })
    }

    /// Enqueue the controller of `a` into `y` (`m` rows of `n`): `m` blocks of [`THREADS`]
    /// threads. A shape the launch does not take is refused by name with nothing enqueued: no
    /// row, width or rank, a width over [`MAX_N`], a rank over [`MAX_RANK`], a buffer shorter
    /// than the launch needs, an epsilon that is not positive and finite. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue(
        &self,
        stream: &CudaStream,
        a: LoopControlArgs<'_>,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "loop_control::enqueue";
        let LoopControlArgs {
            cur,
            msg,
            down,
            gain,
            forget,
            n,
            rank,
            m,
            eps,
            fault,
        } = a;
        let bad = |detail: String| Err(GpuError::shape(WHAT, detail));
        if n == 0 || rank == 0 || m == 0 {
            return bad(format!("{m} rows of {n} values at rank {rank}"));
        }
        if n > MAX_N {
            return bad(format!("width {n}: the entry stages at most {MAX_N}"));
        }
        if rank > MAX_RANK {
            return bad(format!("rank {rank}: the entry takes at most {MAX_RANK}"));
        }
        if !(eps.is_finite() && eps > 0.0) {
            return bad(format!("epsilon {eps}: a positive finite one"));
        }
        crate::short(
            WHAT,
            &[
                ("cur", cur.len(), m * n),
                ("msg", msg.len(), m * n),
                ("down", down.len(), rank * 2 * n),
                ("gain", gain.len(), n * rank),
                ("forget", forget.len(), n * rank),
                ("y", y.len(), m * n),
            ],
        )?;
        let (n32, rank32, m32) = (
            launch_u32(WHAT, "n", n)?,
            launch_u32(WHAT, "rank", rank)?,
            launch_u32(WHAT, "m", m)?,
        );
        let prep = self
            .module
            .prepare_loop_control(LaunchConfig1D::new(m32, THREADS as u32, 0))?;
        self.module.loop_control(
            stream, &prep, cur, msg, down, gain, forget, n32, rank32, m32, eps, fault, y,
        )?;
        Ok(())
    }
}
