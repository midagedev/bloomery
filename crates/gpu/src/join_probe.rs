//! The kernels of the GPU ↔ host join probe (`bench_join` in gpu-gates): a
//! producer that stands in for the router, a timed spin that stands in for
//! the GPU work overlapping the host experts, and a consumer that checks the
//! host's result against the sequence number its replay is due.
//!
//! The sequence number is device state: `join_produce` advances a device
//! counter once per launch, so every replay of a captured graph is a new
//! token without a host write between replays. The values the probe compares
//! are integers below 2^24 carried in f32, so every one is exact, and a value
//! left over from another replay decodes to that replay's number
//! ([`result_seq`]).
//!
//! Every kernel stamps `%globaltimer` into the ring slot of its sequence
//! number, `stamps[(seq % ring) * N_STAMP + ST_*]`: a batch of back-to-back
//! replays keeps one device timeline per replay, read back once at the end.

use crate::GpuError;
use crate::launch_u32;
use cuda_core::sys::CUdeviceptr;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32, DeviceAtomicU64, SystemAtomicU32};
use cuda_device::{
    DisjointSlice, SharedArray, debug, kernel, launch_bounds, launch_contract, thread,
};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Values in one token's activation vector (DeepSeek-V4.1's hidden width).
pub const HIDDEN: usize = 5120;
/// Routed experts per token: the router result's ids and weights, and the
/// expert outputs in the host result.
pub const N_SLOTS: usize = 6;
/// Words ahead of the activation in the producer's output: the sequence
/// number, `N_SLOTS` ids, `N_SLOTS` weights as f32 bits, zero padding.
pub const HEADER: usize = 16;
/// Words the producer writes: the header, then `HIDDEN` activation values
/// as f32 bits. The launch contract of `join_produce` carries this literal.
pub const PAYLOAD_WORDS: usize = HEADER + HIDDEN;
/// f32 values of the host result: `N_SLOTS` expert outputs of `HIDDEN`.
pub const RESULT_LEN: usize = N_SLOTS * HIDDEN;
const _: () = assert!(PAYLOAD_WORDS == 5136 && 2 * N_SLOTS < HEADER);

/// Stamp words per replay slot, and what each one holds (`%globaltimer`
/// nanoseconds, except `ST_TIMER_STEP`).
pub const N_STAMP: usize = 8;
/// The producer's thread 0 at entry.
pub const ST_P_START: usize = 0;
/// The producer's thread 0 after its block's last store.
pub const ST_P_END: usize = 1;
/// The spin's start.
pub const ST_G_START: usize = 2;
/// The spin's end, just before it publishes its done word.
pub const ST_G_END: usize = 3;
/// The earliest consumer block's entry (a min over blocks).
pub const ST_C_START: usize = 4;
/// The latest consumer block's exit after its compares (a max over blocks).
pub const ST_C_END: usize = 5;
/// The smallest nonzero `%globaltimer` step the spin observed; 0 when it
/// did not spin.
pub const ST_TIMER_STEP: usize = 6;
const _: () = assert!(N_STAMP == 8 && ST_TIMER_STEP < N_STAMP);

/// Report words: the producer zeroes the per-replay ones, the consumer
/// fills them.
pub const N_REPORT: usize = 4;
/// Result values of this replay that were not the ones due.
pub const REP_MISMATCH: usize = 0;
/// `1 + (seq mod 2048)` of the largest sequence number a mismatching value
/// decodes to; 0 when every value matched.
pub const REP_STALE: usize = 1;
/// The sequence number this replay's consumer compared against.
pub const REP_DUE: usize = 2;
/// Mismatches over every replay since the host last cleared the buffer.
pub const REP_TOTAL: usize = 3;
const _: () = assert!(N_REPORT == 4 && REP_TOTAL < N_REPORT);

/// Threads of the producer's one block, and of each consumer block. The
/// launch contracts and `launch_bounds` carry the literal.
pub const PRODUCE_THREADS: usize = 256;
/// Threads per consumer block.
pub const CONSUME_THREADS: usize = 256;
/// Threads of the spin's one block; only thread 0 works.
pub const SPIN_THREADS: usize = 32;
const PRODUCE_THREADS_U32: u32 = PRODUCE_THREADS as u32;
const CONSUME_THREADS_U32: u32 = CONSUME_THREADS as u32;
const SPIN_THREADS_U32: u32 = SPIN_THREADS as u32;
const _: () = assert!(
    PRODUCE_THREADS_U32 as usize == PRODUCE_THREADS
        && CONSUME_THREADS_U32 as usize == CONSUME_THREADS
        && SPIN_THREADS_U32 as usize == SPIN_THREADS
        && RESULT_LEN.is_multiple_of(CONSUME_THREADS)
);

/// The bits of the sequence number a probe value carries.
const SEQ_BITS: u32 = 0x7ff;

/// Activation value `j` of token `seq`: `(seq mod 2048) * 4096 + (j mod
/// 4096)`, an integer below 2^23 and exact in f32.
#[inline(always)]
#[must_use]
pub fn x_value(seq: u32, j: u32) -> f32 {
    ((seq & SEQ_BITS) * 4096 + (j & 0xfff)) as f32
}

/// Value `j` of expert output `e` in token `seq`'s host result: the
/// activation value plus `e`. Below 2^24 for `e < N_SLOTS`, so exact in f32,
/// and exactly what the host gets by adding `e` to the activation it read.
#[inline(always)]
#[must_use]
pub fn result_value(seq: u32, e: u32, j: u32) -> f32 {
    ((seq & SEQ_BITS) * 4096 + (j & 0xfff) + e) as f32
}

/// The `seq mod 2048` a result value `v` at `(e, j)` was computed for — the
/// inverse of [`result_value`], so a stale value names its replay.
#[inline(always)]
#[must_use]
pub fn result_seq(v: f32, e: u32, j: u32) -> u32 {
    ((v as u32).wrapping_sub((j & 0xfff) + e) >> 12) & SEQ_BITS
}

/// Header word `i` of token `seq`: the sequence number, then six router ids,
/// then six router weights as f32 bits, then zeros. The ids and weights are
/// placeholders of the router result's size; nothing reads them back.
#[inline(always)]
#[must_use]
pub fn header_word(seq: u32, i: usize) -> u32 {
    if i == 0 {
        seq
    } else if i <= N_SLOTS {
        (seq.wrapping_mul(7) + 61 * i as u32) % 384
    } else if i <= 2 * N_SLOTS {
        (1.0f32 / (i - N_SLOTS) as f32).to_bits()
    } else {
        0
    }
}

#[cuda_module]
mod join_kernels {
    use super::*;

    /// The producer: advances the device sequence number `seq[0]` by one and
    /// writes the payload the host reads — header words 0..16
    /// ([`header_word`]), then `x_value(seq, j)` for the `HIDDEN` activation
    /// values as f32 bits. It also opens the replay's stamp slot (its own
    /// start and end, the consumer's min and max seeds) and zeroes the
    /// per-replay report words. One block; thread 0 owns the counter, the
    /// stamps and the report, and the block reads the new number from shared
    /// memory past the barrier.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            seq.len() >= 1,
            out.len() >= 5136,
            report.len() >= 4,
            stamps.len() >= 8 * ring,
            ring >= 1
        )
    )]
    pub fn join_produce(
        ring: u32,
        mut seq: DisjointSlice<u32>,
        mut out: DisjointSlice<u32>,
        mut report: DisjointSlice<u32>,
        mut stamps: DisjointSlice<u64>,
    ) {
        static mut SEQ: SharedArray<u32, 1> = SharedArray::UNINIT;
        let tid = thread::threadIdx_x() as usize;
        // SAFETY: SEQ is this block's own shared allocation; the raw form is
        // the only way to reach it without a reference to a `static mut`.
        let sp = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SEQ) };
        if tid == 0 {
            let t0 = debug::globaltimer();
            // SAFETY: seq.len() >= 1 by the launch contract, and thread 0 of
            // the one block is the only thread that touches it.
            let s = unsafe {
                let slot = seq.get_unchecked_mut(0);
                *slot = slot.wrapping_add(1);
                *slot
            };
            let base = (s % ring) as usize * N_STAMP;
            // SAFETY: base + N_STAMP <= ring * 8 <= stamps.len() and every
            // report index is below 4 <= report.len(), by the launch
            // contract; sp is SEQ, read by the block only past the barrier.
            unsafe {
                *sp = s;
                *stamps.get_unchecked_mut(base + ST_P_START) = t0;
                *stamps.get_unchecked_mut(base + ST_C_START) = u64::MAX;
                *stamps.get_unchecked_mut(base + ST_C_END) = 0;
                *report.get_unchecked_mut(REP_MISMATCH) = 0;
                *report.get_unchecked_mut(REP_STALE) = 0;
                *report.get_unchecked_mut(REP_DUE) = s;
            }
        }
        thread::sync_threads();
        // SAFETY: SEQ[0] was written by thread 0 before the barrier.
        let s = unsafe { *sp };
        if tid < HEADER {
            // SAFETY: tid < 16 <= out.len(); word tid is this thread's own.
            unsafe {
                *out.get_unchecked_mut(tid) = header_word(s, tid);
            }
        }
        let mut j = tid;
        while j < HIDDEN {
            // SAFETY: HEADER + j < 5136 <= out.len() by the launch contract;
            // the stride gives every word one thread.
            unsafe {
                *out.get_unchecked_mut(HEADER + j) = x_value(s, j as u32).to_bits();
            }
            j += PRODUCE_THREADS;
        }
        thread::sync_threads();
        if tid == 0 {
            let base = (s % ring) as usize * N_STAMP;
            // SAFETY: base + ST_P_END < ring * 8 <= stamps.len().
            unsafe {
                *stamps.get_unchecked_mut(base + ST_P_END) = debug::globaltimer();
            }
        }
    }

    /// The overlapped GPU work's stand-in: thread 0 spins on `%globaltimer`
    /// for `params[0]` ns, stamps its start, its end and the smallest timer
    /// step it saw, then publishes the replay's sequence number to `done`
    /// with a system-scope release store — the host reads that word to learn
    /// that the GPU work finished while the host was still on its part.
    ///
    /// # Safety
    ///
    /// `done` is the device address of a `u32` in host-mapped pinned memory
    /// that stays allocated for as long as any launch of this kernel, or any
    /// replay of a graph that captured one, can run.
    #[kernel]
    #[launch_bounds(32)]
    #[launch_contract(
        domain = 1,
        block = (32, 1, 1),
        requires = (
            seq.len() >= 1,
            params.len() >= 1,
            stamps.len() >= 8 * ring,
            ring >= 1
        )
    )]
    pub unsafe fn join_spin(
        ring: u32,
        seq: &[u32],
        params: &[u64],
        mut stamps: DisjointSlice<u64>,
        done: *mut u32,
    ) {
        if thread::threadIdx_x() != 0 {
            return;
        }
        // SAFETY: seq.len() >= 1 and params.len() >= 1 by the launch contract.
        let (s, ns) = unsafe { (*seq.get_unchecked(0), *params.get_unchecked(0)) };
        let t0 = debug::globaltimer();
        let mut now = t0;
        let mut step = u64::MAX;
        while now.wrapping_sub(t0) < ns {
            let t = debug::globaltimer();
            if t > now && t - now < step {
                step = t - now;
            }
            now = t;
        }
        let t1 = debug::globaltimer();
        let base = (s % ring) as usize * N_STAMP;
        // SAFETY: base + N_STAMP <= ring * 8 <= stamps.len() by the launch
        // contract; thread 0 of the one block is the only writer.
        unsafe {
            *stamps.get_unchecked_mut(base + ST_G_START) = t0;
            *stamps.get_unchecked_mut(base + ST_G_END) = t1;
            *stamps.get_unchecked_mut(base + ST_TIMER_STEP) =
                if step == u64::MAX { 0 } else { step };
        }
        // SAFETY: `done` addresses a live host-mapped u32 by this kernel's
        // contract; the store is atomic at system scope, so the host's atomic
        // load of the same word is not a data race.
        unsafe { SystemAtomicU32::from_ptr(done).store(s, AtomicOrdering::Release) };
    }

    /// The consumer: every thread compares one value of the host's result —
    /// a volatile load from `res`, which is either the device copy the
    /// graph's H2D wrote or the host-mapped result itself — with
    /// [`result_value`] of the sequence number this replay is due.
    /// Mismatches count into `report[REP_MISMATCH]` and `report[REP_TOTAL]`,
    /// and `report[REP_STALE]` keeps the largest `1 + (seq mod 2048)` a
    /// mismatching value decodes to. Each block's thread 0 folds its entry
    /// and its exit (after the block's compares) into the replay's
    /// `ST_C_START` (min) and `ST_C_END` (max).
    ///
    /// # Safety
    ///
    /// `res` addresses `RESULT_LEN` readable f32; `report` addresses
    /// `N_REPORT` u32 and `stamps` `ring * N_STAMP` u64 of device memory;
    /// all stay allocated while a launch, or a replay of a graph that
    /// captured one, can run.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), requires = (seq.len() >= 1, ring >= 1))]
    pub unsafe fn join_consume(
        ring: u32,
        seq: &[u32],
        res: *const f32,
        report: *mut u32,
        stamps: *mut u64,
    ) {
        let tid = thread::threadIdx_x();
        let k = thread::index_1d().get();
        // SAFETY: seq.len() >= 1 by the launch contract.
        let s = unsafe { *seq.get_unchecked(0) };
        let base = (s % ring) as usize * N_STAMP;
        if tid == 0 {
            let t = debug::globaltimer();
            // SAFETY: base + ST_C_START < ring * N_STAMP, inside `stamps` by
            // this kernel's contract; the word is only ever reached
            // atomically while consumer blocks run.
            unsafe {
                DeviceAtomicU64::from_ptr(stamps.add(base + ST_C_START))
                    .fetch_min(t, AtomicOrdering::Relaxed)
            };
        }
        if k < RESULT_LEN {
            // SAFETY: k < RESULT_LEN, inside `res` by this kernel's contract.
            // Volatile: the value must come from memory on every replay,
            // never from a cache line an earlier replay left.
            let v = unsafe { core::ptr::read_volatile(res.add(k)) };
            let (e, j) = ((k / HIDDEN) as u32, (k % HIDDEN) as u32);
            if v.to_bits() != result_value(s, e, j).to_bits() {
                // SAFETY: the three indices are below N_REPORT, inside
                // `report` by this kernel's contract; consumer threads reach
                // these words only atomically.
                unsafe {
                    DeviceAtomicU32::from_ptr(report.add(REP_MISMATCH))
                        .fetch_add(1, AtomicOrdering::Relaxed);
                    DeviceAtomicU32::from_ptr(report.add(REP_TOTAL))
                        .fetch_add(1, AtomicOrdering::Relaxed);
                    DeviceAtomicU32::from_ptr(report.add(REP_STALE))
                        .fetch_max(result_seq(v, e, j) + 1, AtomicOrdering::Relaxed);
                }
            }
        }
        thread::sync_threads();
        if tid == 0 {
            let t = debug::globaltimer();
            // SAFETY: as for ST_C_START above.
            unsafe {
                DeviceAtomicU64::from_ptr(stamps.add(base + ST_C_END))
                    .fetch_max(t, AtomicOrdering::Relaxed)
            };
        }
    }
}

/// The loaded join-probe module. Owns no stream; every enqueue takes the
/// caller's, so the launches order with the join operations around them and
/// are capturable.
pub struct JoinProbe {
    module: join_kernels::LoadedModule,
}

impl JoinProbe {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<JoinProbe, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; every launcher checks its launch contract.
        let module = unsafe { join_kernels::load(ctx)? };
        Ok(JoinProbe { module })
    }

    /// Enqueue the producer: advance `seq[0]`, write `PAYLOAD_WORDS` words
    /// into `out`, zero the per-replay words of `report` and open the stamp
    /// slot of the new number in `stamps` (`ring * N_STAMP` u64).
    /// Asynchronous, allocation-free, capturable.
    pub fn enqueue_produce(
        &self,
        stream: &CudaStream,
        ring: usize,
        seq: &mut DeviceBuffer<u32>,
        out: &mut DeviceBuffer<u32>,
        report: &mut DeviceBuffer<u32>,
        stamps: &mut DeviceBuffer<u64>,
    ) -> Result<(), GpuError> {
        let ring = launch_u32("enqueue_produce", "ring", ring)?;
        let prep =
            self.module
                .prepare_join_produce(LaunchConfig1D::new(1, PRODUCE_THREADS_U32, 0))?;
        self.module
            .join_produce(stream, &prep, ring, seq, out, report, stamps)?;
        Ok(())
    }

    /// Enqueue the spin: `params[0]` ns of `%globaltimer`, then the replay's
    /// sequence number stored to `done`. Asynchronous, allocation-free,
    /// capturable.
    ///
    /// # Safety
    ///
    /// `done` is the device address of a `u32` in host-mapped pinned memory
    /// that outlives every launch and every replay of a graph this enqueue
    /// is captured into.
    pub unsafe fn enqueue_spin(
        &self,
        stream: &CudaStream,
        ring: usize,
        seq: &DeviceBuffer<u32>,
        params: &DeviceBuffer<u64>,
        stamps: &mut DeviceBuffer<u64>,
        done: CUdeviceptr,
    ) -> Result<(), GpuError> {
        let ring = launch_u32("enqueue_spin", "ring", ring)?;
        let prep = self
            .module
            .prepare_join_spin(LaunchConfig1D::new(1, SPIN_THREADS_U32, 0))?;
        // SAFETY: the caller guarantees `done`; the launch contract checks
        // every slice against `ring`.
        unsafe {
            self.module
                .join_spin(stream, &prep, ring, seq, params, stamps, done as *mut u32)?;
        }
        Ok(())
    }

    /// Enqueue the consumer over `RESULT_LEN` values at `res`, counting into
    /// `report` (`N_REPORT` u32) and folding its stamps into `stamps` (`ring
    /// * N_STAMP` u64). Asynchronous, allocation-free, capturable.
    ///
    /// # Safety
    ///
    /// `res` is the device address of `RESULT_LEN` f32 — device memory or
    /// host-mapped pinned memory — that outlives every launch and every
    /// replay of a graph this enqueue is captured into.
    pub unsafe fn enqueue_consume(
        &self,
        stream: &CudaStream,
        ring: usize,
        seq: &DeviceBuffer<u32>,
        res: CUdeviceptr,
        report: &mut DeviceBuffer<u32>,
        stamps: &mut DeviceBuffer<u64>,
    ) -> Result<(), GpuError> {
        if report.len() < N_REPORT || stamps.len() < ring * N_STAMP || ring == 0 {
            return Err(GpuError::shape(
                "enqueue_consume",
                format!(
                    "need report >= {N_REPORT} and stamps >= ring*{N_STAMP} with ring >= 1, got \
                     report {} stamps {} ring {ring}",
                    report.len(),
                    stamps.len()
                ),
            ));
        }
        let ring = launch_u32("enqueue_consume", "ring", ring)?;
        let blocks = launch_u32("enqueue_consume", "blocks", RESULT_LEN / CONSUME_THREADS)?;
        let prep = self.module.prepare_join_consume(LaunchConfig1D::new(
            blocks,
            CONSUME_THREADS_U32,
            0,
        ))?;
        // SAFETY: the caller guarantees `res`; `report` and `stamps` are live
        // device buffers checked above to hold what the kernel addresses, and
        // like every buffer a captured launch names they must outlive the
        // graph (`Graph`'s own contract).
        unsafe {
            self.module.join_consume(
                stream,
                &prep,
                ring,
                seq,
                res as *const f32,
                report.cu_deviceptr() as *mut u32,
                stamps.cu_deviceptr() as *mut u64,
            )?;
        }
        Ok(())
    }
}
