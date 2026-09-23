//! Bench for the GPU ↔ host join inside a captured CUDA graph: the boundary
//! a hybrid MoE layer crosses when the host computes its share of the routed
//! experts (`docs/v41-placement.md` §5, `docs/research/hybrid-engines.md` R1
//! and R2). It is not a gate; its verdicts guard its own measurement.
//!
//! Each mechanism is captured into one graph shaped like a layer's boundary:
//!
//! ```text
//! P ─ D2H ─ out ─ G ─ back ─ [H2D] ─ C
//! ```
//!
//! `join_produce` (P) writes a token's router result and activation vector
//! (`bloomery_gpu::join_probe`); the D2H lands them in pinned host memory;
//! the join out tells the host; `join_spin` (G) stands in for the GPU work
//! that overlaps the host part (the shared expert); the join back waits for
//! the host's result; the result reaches the consumer through an H2D copy
//! (arm `h2d`) or by the consumer reading the host-mapped result in place
//! (arm `mapped`); and `join_consume` (C) checks every value against the
//! token it is due.
//!
//! The mechanisms, as join out / join back:
//!
//! - `hostfn`: two `cuLaunchHostFunc` host nodes, blocking dispatch. The
//!   submit node hands the token to the host worker and returns; the wait
//!   node returns once the worker is done (KTransformers' shape).
//! - `hostfn1`: one blocking host node that hands over and waits in the same
//!   call, with G after it: a node that holds the stream while the host works.
//! - `hostfn_spin`: `hostfn` through `cuLaunchHostFunc_v2` with
//!   `CU_HOST_TASK_SPINWAIT`.
//! - `memop`: `cuStreamWriteValue32` of 1 to a host-mapped flag that the host
//!   clears; back is one batch-memop node — wait for the host's flag == 1,
//!   then write it back to 0. A captured memop carries the same value on
//!   every replay, so the token's number travels in the payload and a flag
//!   has to be a toggle its reader clears.
//! - `memop_red`: the stream-memop atomic reduction (CUDA 13.1), the one
//!   memop that yields a new value from a fixed node. Out is a system barrier
//!   (it orders the D2H before the signal) and ADD 1 to a host-mapped
//!   generation word; `--workers` host threads each wait for the next
//!   generation, compute their experts and add 1 to a counter; back waits for
//!   the counter >= workers and consumes it with ADD -workers.
//!
//! Before any capture, each mechanism's special operation runs once outside
//! capture against a scratch word and its effect is read back, so "the API
//! refuses" and "capture refuses" stay apart. A capture that refuses is
//! printed as a finding and the mechanism is skipped, never replaced.
//!
//! `--check`: per mechanism and arm, `--replays` replays with a host part of
//! `--host-work-us` (default 1000) and a GPU part of `--gpu-work-us`
//! (default 20), each replay waited for on its own. It passes when the
//! consumer read exactly the host's values of its own replay on every
//! replay and the host found every payload fresh. The ordering proof rides
//! along: whether G had published done(seq) before the host signalled back,
//! per replay, printed next to what the stream semantics predict (`hostfn1`
//! is the one that should not overlap).
//!
//! `--time` (lead-only, under `tools/ref/time-gate.sh`): the check first,
//! refusing to time if it fails, then `--rounds` rounds with the order of the
//! cells (mechanism × arm) rotated every round. Per cell: a zero-work batch
//! (G = H = 0) and one batch per `--grid` point, `--batch` back-to-back
//! replays each. The join is read from the device's `%globaltimer`, P's end
//! to C's start and to C's end: both ends sit on one clock, where a replay's
//! wall time also carries the launch and the host's wait. A replay whose
//! stamps are out of timeline order, or span more than a replay may take, is
//! counted as `bad_slots` and left out of the statistics: `C_START` is a min
//! over every consumer block's `%globaltimer` read, so one read that comes
//! back wrong is enough. CPU time per thread comes from
//! `/proc/self/task/*/schedstat`, so the driver's own threads are counted too.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("bench_join: built without the `gpu` feature; see `just bench-gpu-join-check`.");
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("bench_join", bench::run())
}

#[cfg(feature = "gpu")]
mod bench {
    use bloomery_gpu::join_probe::{self as jp, JoinProbe};
    use bloomery_gpu::{Gpu, GpuError, Graph, NodeInfo};
    use bloomery_gpu_gates::{GateError, checks_failed, verdict};
    use cuda_core::{CudaStream, DeviceBuffer, DriverError, sys};
    use std::ffi::c_void;
    use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
    use std::sync::{Mutex, MutexGuard};
    use std::time::{Duration, Instant};

    /// Stamp ring slots; a batch launches at most this many replays.
    const RING: usize = 256;
    /// Replays at the head of a timed batch left out of its statistics.
    const WARM: usize = 2;
    /// Words between two flags of the flag page: a 64-byte line each, so a
    /// thread spinning on one flag never shares a line with another flag.
    const FLAG_STRIDE: usize = 16;
    /// memop: the GPU writes 1, the host clears it.
    const FLAG_OUT: usize = 0;
    /// memop: the host writes 1, the GPU writes it back to 0.
    const FLAG_BACK: usize = 1;
    /// memop_red: the GPU adds 1 per replay.
    const GEN_OUT: usize = 2;
    /// memop_red: each worker adds 1, the GPU subtracts the worker count.
    const CNT_BACK: usize = 3;
    /// The spin publishes its replay's sequence number here.
    const G_DONE: usize = 4;
    /// The eager probes' target.
    const SCRATCH: usize = 5;
    const N_FLAGS: usize = 6;
    /// How long one checked replay, or one timed batch past its own
    /// expected length, may take before the harness releases it.
    const DEADLINE: Duration = Duration::from_secs(2);

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum Mech {
        HostFn,
        HostFnOne,
        HostFnSpin,
        MemOp,
        MemOpRed,
    }

    /// Every mechanism, the one most likely to fault the context last: a
    /// sticky fault would take the verdicts of everything after it.
    const MECHS: [Mech; 5] = [
        Mech::HostFn,
        Mech::HostFnOne,
        Mech::HostFnSpin,
        Mech::MemOp,
        Mech::MemOpRed,
    ];

    impl Mech {
        fn name(self) -> &'static str {
            match self {
                Mech::HostFn => "hostfn",
                Mech::HostFnOne => "hostfn1",
                Mech::HostFnSpin => "hostfn_spin",
                Mech::MemOp => "memop",
                Mech::MemOpRed => "memop_red",
            }
        }

        /// Whether G can run while the host works, by the stream semantics:
        /// nothing between the join out and G waits for the host, except in
        /// `hostfn1`, whose one node returns only when the host is done.
        fn overlaps(self) -> bool {
            self != Mech::HostFnOne
        }

        /// Host threads the mechanism runs with.
        fn workers(self, n: usize) -> usize {
            if self == Mech::MemOpRed { n } else { 1 }
        }

        fn from_name(s: &str) -> Option<Mech> {
            MECHS.into_iter().find(|m| m.name() == s)
        }
    }

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum Arm {
        /// The graph copies the host result to a device buffer; C reads that.
        H2d,
        /// C reads the host-mapped result in place.
        Mapped,
    }

    const ARMS: [Arm; 2] = [Arm::H2d, Arm::Mapped];

    impl Arm {
        fn name(self) -> &'static str {
            match self {
                Arm::H2d => "h2d",
                Arm::Mapped => "mapped",
            }
        }
    }

    /// `GpuError::Driver` for a raw driver call that did not succeed.
    fn drv(rc: sys::CUresult, what: &'static str) -> Result<(), GpuError> {
        if rc == sys::cudaError_enum_CUDA_SUCCESS {
            Ok(())
        } else {
            Err(GpuError::Driver {
                op: Some(what),
                source: DriverError(rc),
            })
        }
    }

    /// Pinned host memory mapped into the device's address space
    /// (`cuMemHostAlloc` with `DEVICEMAP`): the host reaches it at `host`,
    /// kernels and stream memory operations at `dev`. Zeroed at allocation.
    struct HostMem {
        host: *mut u8,
        dev: sys::CUdeviceptr,
        bytes: usize,
    }

    impl HostMem {
        fn new(gpu: &Gpu, bytes: usize) -> Result<HostMem, GateError> {
            gpu.context().bind_to_thread()?;
            let mut host: *mut c_void = std::ptr::null_mut();
            // SAFETY: the context is current on this thread (bound above) and
            // `host` is a live local the call writes.
            let rc =
                unsafe { sys::cuMemHostAlloc(&mut host, bytes, sys::CU_MEMHOSTALLOC_DEVICEMAP) };
            drv(rc, "cuMemHostAlloc")?;
            let mut dev: sys::CUdeviceptr = 0;
            // SAFETY: `host` is the mapped allocation just made; flags must be 0.
            let rc = unsafe { sys::cuMemHostGetDevicePointer_v2(&mut dev, host, 0) };
            if let Err(e) = drv(rc, "cuMemHostGetDevicePointer_v2") {
                // SAFETY: `host` came from cuMemHostAlloc and is freed once, here.
                unsafe { sys::cuMemFreeHost(host) };
                return Err(e.into());
            }
            // SAFETY: the allocation holds `bytes` writable bytes and nothing
            // else references it yet.
            unsafe { std::ptr::write_bytes(host.cast::<u8>(), 0, bytes) };
            Ok(HostMem {
                host: host.cast(),
                dev,
                bytes,
            })
        }

        /// The device address of byte `off`.
        fn dev_at(&self, off: usize) -> sys::CUdeviceptr {
            assert!(
                off < self.bytes,
                "offset {off} outside {} mapped bytes",
                self.bytes
            );
            self.dev + off as u64
        }
    }

    impl Drop for HostMem {
        fn drop(&mut self) {
            // SAFETY: `host` came from cuMemHostAlloc and is freed once, here,
            // after every graph that names it (declared later, dropped first).
            // A failure on the drop path is unreportable and ignored.
            unsafe { sys::cuMemFreeHost(self.host.cast()) };
        }
    }

    /// One worker's pass over one replay.
    #[derive(Clone, Copy)]
    struct Rec {
        seq: u32,
        worker: usize,
        /// The payload's header and every activation value were the ones due.
        payload_ok: bool,
        /// G's done word held this replay's number before the host signalled.
        overlap: bool,
        /// Host-clock ns since `Shared::base`: join out seen, result written,
        /// join back signalled.
        seen: u64,
        written: u64,
        signaled: u64,
    }

    /// What the host side of a join shares: the host functions the graphs
    /// call, the workers, and the harness. It lives in a `Box` that outlives
    /// every graph that captured its address.
    struct Shared {
        /// Host views of the D2H'd payload, the result, and the flag page.
        payload: *const u32,
        result: *mut f32,
        flags: *const AtomicU32,
        /// hostfn*: the sequence number the submit node handed over, and the
        /// one the worker finished.
        go: AtomicU32,
        done: AtomicU32,
        /// Ends the workers and releases a waiting host function.
        stop: AtomicBool,
        /// Check mode: the workers verify the payload.
        check: AtomicBool,
        /// The host part per replay, ns.
        host_ns: AtomicU64,
        /// The last thread a host function ran on.
        hostfn_tid: AtomicU32,
        worker_tids: Mutex<Vec<u32>>,
        recs: Mutex<Vec<Rec>>,
        base: Instant,
    }

    // SAFETY: the three pointers address host-mapped allocations owned by the
    // `Rig` that owns this struct; they outlive every thread and graph that
    // reaches them. Flag words are only touched through atomics, and the
    // payload and result are ordered by the flag protocol: a reader looks at
    // them only after an acquire load of the flag their writer released.
    unsafe impl Sync for Shared {}

    thread_local! {
        /// This thread's kernel id, read once.
        static TID: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    }

    /// The calling thread's kernel id (`/proc/thread-self` names it).
    fn read_tid() -> u32 {
        std::fs::read_link("/proc/thread-self")
            .ok()
            .and_then(|p| p.file_name()?.to_str()?.parse().ok())
            .unwrap_or(0)
    }

    fn tid() -> u32 {
        TID.with(|t| {
            if t.get() == 0 {
                t.set(read_tid());
            }
            t.get()
        })
    }

    fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
        m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn spin_for(ns: u64) {
        let t0 = Instant::now();
        let d = Duration::from_nanos(ns);
        while t0.elapsed() < d {
            std::hint::spin_loop();
        }
    }

    impl Shared {
        fn flag(&self, i: usize) -> &AtomicU32 {
            assert!(i < N_FLAGS, "flag {i} outside the page");
            // SAFETY: the page holds N_FLAGS * FLAG_STRIDE words, 4-aligned;
            // AtomicU32 has u32's layout and every host access to a flag word
            // goes through this view.
            unsafe { &*self.flags.add(i * FLAG_STRIDE) }
        }

        fn since(&self, t: Instant) -> u64 {
            u64::try_from(t.duration_since(self.base).as_nanos()).unwrap_or(u64::MAX)
        }

        /// The sequence number in the payload header.
        fn header_seq(&self) -> u32 {
            // SAFETY: word 0 of the live payload allocation; the caller has
            // acquired the flag the D2H's completion was released through.
            unsafe { self.payload.read_volatile() }
        }

        /// The payload carries token `s`: the header's number and every
        /// activation value.
        fn payload_fresh(&self, s: u32) -> bool {
            self.header_seq() == s
                && (0..jp::HIDDEN).all(|j| {
                    // SAFETY: HEADER + j < PAYLOAD_WORDS, inside the payload.
                    let w = unsafe { self.payload.add(jp::HEADER + j).read() };
                    w == jp::x_value(s, j as u32).to_bits()
                })
        }

        /// Write worker `w` of `n`'s expert outputs: `x[j] + e` for every
        /// expert `e ≡ w (mod n)`, from the activation the D2H delivered.
        fn write_result(&self, w: usize, n: usize) {
            for e in (w..jp::N_SLOTS).step_by(n) {
                let add = e as f32;
                for j in 0..jp::HIDDEN {
                    // SAFETY: HEADER + j < PAYLOAD_WORDS and e * HIDDEN + j <
                    // RESULT_LEN; worker w alone writes expert e's outputs.
                    unsafe {
                        let x = f32::from_bits(self.payload.add(jp::HEADER + j).read());
                        self.result.add(e * jp::HIDDEN + j).write(x + add);
                    }
                }
            }
        }

        /// Wait for the join out; `None` once `stop` is set.
        fn wait_out(&self, mech: Mech, last: &mut u32) -> Option<u32> {
            loop {
                if self.stop.load(Ordering::Relaxed) {
                    return None;
                }
                match mech {
                    Mech::MemOp => {
                        if self.flag(FLAG_OUT).load(Ordering::Acquire) == 1 {
                            self.flag(FLAG_OUT).store(0, Ordering::Relaxed);
                            return Some(self.header_seq());
                        }
                    }
                    Mech::MemOpRed => {
                        let g = self.flag(GEN_OUT).load(Ordering::Acquire);
                        if g != *last {
                            *last = g;
                            return Some(self.header_seq());
                        }
                    }
                    Mech::HostFn | Mech::HostFnOne | Mech::HostFnSpin => {
                        let s = self.go.load(Ordering::Acquire);
                        if s != *last {
                            *last = s;
                            return Some(s);
                        }
                    }
                }
                std::hint::spin_loop();
            }
        }

        fn signal_back(&self, mech: Mech, s: u32) {
            match mech {
                Mech::MemOp => self.flag(FLAG_BACK).store(1, Ordering::Release),
                Mech::MemOpRed => {
                    self.flag(CNT_BACK).fetch_add(1, Ordering::Release);
                }
                Mech::HostFn | Mech::HostFnOne | Mech::HostFnSpin => {
                    self.done.store(s, Ordering::Release);
                }
            }
        }

        /// The submit half of a host-function join: hand the payload's token
        /// to the worker.
        fn submit(&self) {
            self.hostfn_tid.store(tid(), Ordering::Relaxed);
            self.go.store(self.header_seq(), Ordering::Release);
        }

        /// The wait half: return once the worker finished the token handed
        /// over, or once the harness stops.
        fn wait_done(&self) {
            let s = self.go.load(Ordering::Acquire);
            while self.done.load(Ordering::Acquire) != s && !self.stop.load(Ordering::Relaxed) {
                std::hint::spin_loop();
            }
        }

        /// One host worker: `w` of `n`, until `stop`.
        fn worker(&self, mech: Mech, w: usize, n: usize) {
            lock(&self.worker_tids).push(tid());
            let mut last = match mech {
                Mech::MemOpRed => self.flag(GEN_OUT).load(Ordering::Acquire),
                Mech::MemOp => 0,
                Mech::HostFn | Mech::HostFnOne | Mech::HostFnSpin => {
                    self.go.load(Ordering::Acquire)
                }
            };
            while let Some(s) = self.wait_out(mech, &mut last) {
                let seen = Instant::now();
                let payload_ok = !self.check.load(Ordering::Relaxed) || self.payload_fresh(s);
                spin_for(self.host_ns.load(Ordering::Relaxed));
                self.write_result(w, n);
                let written = Instant::now();
                let overlap = self.flag(G_DONE).load(Ordering::Acquire) == s;
                self.signal_back(mech, s);
                let signaled = Instant::now();
                let rec = Rec {
                    seq: s,
                    worker: w,
                    payload_ok,
                    overlap,
                    seen: self.since(seen),
                    written: self.since(written),
                    signaled: self.since(signaled),
                };
                lock(&self.recs).push(rec);
            }
        }

        /// Records of token `s` so far.
        fn recs_of(&self, s: u32) -> usize {
            lock(&self.recs).iter().filter(|r| r.seq == s).count()
        }
    }

    /// Submit node.
    ///
    /// # Safety
    ///
    /// `user` is the `Shared` the graph was captured with; it outlives every
    /// replay.
    unsafe extern "C" fn submit_fn(user: *mut c_void) {
        // SAFETY: the capture passed a `Shared` that outlives every replay.
        let sh = unsafe { &*user.cast::<Shared>() };
        sh.submit();
    }

    /// Wait node.
    ///
    /// # Safety
    ///
    /// As [`submit_fn`].
    unsafe extern "C" fn wait_fn(user: *mut c_void) {
        // SAFETY: the capture passed a `Shared` that outlives every replay.
        let sh = unsafe { &*user.cast::<Shared>() };
        sh.wait_done();
    }

    /// The one node of `hostfn1`: submit, then wait, in one call.
    ///
    /// # Safety
    ///
    /// As [`submit_fn`].
    unsafe extern "C" fn hold_fn(user: *mut c_void) {
        // SAFETY: the capture passed a `Shared` that outlives every replay.
        let sh = unsafe { &*user.cast::<Shared>() };
        sh.submit();
        sh.wait_done();
    }

    /// The eager probe's host function: count one call.
    ///
    /// # Safety
    ///
    /// `user` is an `AtomicU32` that outlives the stream synchronize after
    /// the launch.
    unsafe extern "C" fn count_fn(user: *mut c_void) {
        // SAFETY: the probe passes an AtomicU32 alive past its synchronize.
        let hits = unsafe { &*user.cast::<AtomicU32>() };
        hits.fetch_add(1, Ordering::Relaxed);
    }

    /// An all-zero batch operation (every member of the union is integers).
    fn op_zero() -> sys::CUstreamBatchMemOpParams {
        // SAFETY: the union's members are plain C structs of integers, so
        // all-zero bytes are a valid value.
        unsafe { std::mem::zeroed() }
    }

    fn op_wait(addr: sys::CUdeviceptr, value: u32, flags: u32) -> sys::CUstreamBatchMemOpParams {
        let mut p = op_zero();
        p.waitValue = sys::CUstreamBatchMemOpParams_union_CUstreamMemOpWaitValueParams_st {
            operation: sys::CUstreamBatchMemOpType_enum_CU_STREAM_MEM_OP_WAIT_VALUE_32,
            address: addr,
            __bindgen_anon_1:
                sys::CUstreamBatchMemOpParams_union_CUstreamMemOpWaitValueParams_st__bindgen_ty_1 {
                    value,
                },
            flags,
            alias: 0,
        };
        p
    }

    fn op_write(addr: sys::CUdeviceptr, value: u32) -> sys::CUstreamBatchMemOpParams {
        let mut p = op_zero();
        p.writeValue = sys::CUstreamBatchMemOpParams_union_CUstreamMemOpWriteValueParams_st {
            operation: sys::CUstreamBatchMemOpType_enum_CU_STREAM_MEM_OP_WRITE_VALUE_32,
            address: addr,
            __bindgen_anon_1:
                sys::CUstreamBatchMemOpParams_union_CUstreamMemOpWriteValueParams_st__bindgen_ty_1 {
                    value,
                },
            flags: sys::CUstreamWriteValue_flags_enum_CU_STREAM_WRITE_VALUE_DEFAULT,
            alias: 0,
        };
        p
    }

    fn op_barrier_sys() -> sys::CUstreamBatchMemOpParams {
        let mut p = op_zero();
        p.memoryBarrier = sys::CUstreamBatchMemOpParams_union_CUstreamMemOpMemoryBarrierParams_st {
            operation: sys::CUstreamBatchMemOpType_enum_CU_STREAM_MEM_OP_BARRIER,
            flags: sys::CUstreamMemoryBarrier_flags_enum_CU_STREAM_MEMORY_BARRIER_TYPE_SYS,
        };
        p
    }

    /// Atomic reduction `*addr += value` on a u32 (wrapping).
    fn op_add(addr: sys::CUdeviceptr, value: u32) -> sys::CUstreamBatchMemOpParams {
        let mut p = op_zero();
        p.atomicReduction =
            sys::CUstreamBatchMemOpParams_union_CUstreamMemOpAtomicReductionParams_st {
                operation: sys::CUstreamBatchMemOpType_enum_CU_STREAM_MEM_OP_ATOMIC_REDUCTION,
                flags: 0,
                reductionOp:
                    sys::CUstreamAtomicReductionOpType_enum_CU_STREAM_ATOMIC_REDUCTION_OP_ADD,
                dataType:
                    sys::CUstreamAtomicReductionDataType_enum_CU_STREAM_ATOMIC_REDUCTION_UNSIGNED_32,
                address: addr,
                value: u64::from(value),
                alias: 0,
            };
        p
    }

    /// Enqueue `ops` as one batch of stream memory operations (one graph
    /// node when captured).
    fn mem_batch(
        hs: sys::CUstream,
        ops: &mut [sys::CUstreamBatchMemOpParams],
        what: &'static str,
    ) -> Result<(), GpuError> {
        let n = u32::try_from(ops.len()).expect("a batch holds a handful of operations");
        // SAFETY: `ops` is a live array of `n` initialized operations; the
        // driver reads it during the call (a capture copies it into the node);
        // flags must be 0.
        let rc = unsafe { sys::cuStreamBatchMemOp_v2(hs, n, ops.as_mut_ptr(), 0) };
        drv(rc, what)
    }

    /// The device buffers the graphs name.
    struct Dev {
        seq: DeviceBuffer<u32>,
        out: DeviceBuffer<u32>,
        res: DeviceBuffer<f32>,
        report: DeviceBuffer<u32>,
        stamps: DeviceBuffer<u64>,
        /// `params[0]`: G's spin, ns.
        params: DeviceBuffer<u64>,
    }

    /// Everything the graphs capture addresses of. The graphs themselves are
    /// declared after the rig in `run`, so they drop first.
    struct Rig<'g> {
        gpu: &'g Gpu,
        probe: JoinProbe,
        dev: Dev,
        payload: HostMem,
        result: HostMem,
        flags: HostMem,
        sh: Box<Shared>,
        /// `memop_red`'s host threads.
        workers: usize,
    }

    impl<'g> Rig<'g> {
        fn new(gpu: &'g Gpu, workers: usize) -> Result<Rig<'g>, GateError> {
            let s = gpu.stream();
            let probe = JoinProbe::load(gpu.context())?;
            let dev = Dev {
                seq: DeviceBuffer::zeroed(s, 1)?,
                out: DeviceBuffer::zeroed(s, jp::PAYLOAD_WORDS)?,
                res: DeviceBuffer::zeroed(s, jp::RESULT_LEN)?,
                report: DeviceBuffer::zeroed(s, jp::N_REPORT)?,
                stamps: DeviceBuffer::zeroed(s, RING * jp::N_STAMP)?,
                params: DeviceBuffer::zeroed(s, 1)?,
            };
            let payload = HostMem::new(gpu, 4 * jp::PAYLOAD_WORDS)?;
            let result = HostMem::new(gpu, 4 * jp::RESULT_LEN)?;
            let flags = HostMem::new(gpu, 4 * N_FLAGS * FLAG_STRIDE)?;
            let res = result.host.cast::<f32>();
            // Token 0's values, so a read before any host write names token 0.
            for k in 0..jp::RESULT_LEN {
                let v = jp::result_value(0, (k / jp::HIDDEN) as u32, (k % jp::HIDDEN) as u32);
                // SAFETY: k < RESULT_LEN, inside the result allocation, which
                // no graph or thread touches yet.
                unsafe { res.add(k).write(v) };
            }
            let sh = Box::new(Shared {
                payload: payload.host.cast::<u32>(),
                result: res,
                flags: flags.host.cast::<AtomicU32>(),
                go: AtomicU32::new(0),
                done: AtomicU32::new(0),
                stop: AtomicBool::new(false),
                check: AtomicBool::new(true),
                host_ns: AtomicU64::new(0),
                hostfn_tid: AtomicU32::new(0),
                worker_tids: Mutex::new(Vec::new()),
                recs: Mutex::new(Vec::new()),
                base: Instant::now(),
            });
            Ok(Rig {
                gpu,
                probe,
                dev,
                payload,
                result,
                flags,
                sh,
                workers,
            })
        }

        /// The device address of flag `i`.
        fn flag_dev(&self, i: usize) -> sys::CUdeviceptr {
            self.flags.dev_at(4 * i * FLAG_STRIDE)
        }

        /// Set G's spin and the host part for the next replays; the GPU is
        /// idle between batches, so the synchronous copy lands before them.
        fn set_work(&mut self, g_ns: u64, h_ns: u64) -> Result<(), GateError> {
            self.dev.params.copy_from_host(self.gpu.stream(), &[g_ns])?;
            self.sh.host_ns.store(h_ns, Ordering::Relaxed);
            Ok(())
        }

        /// A clean start for a cell: flags down, no records, no stop.
        fn open_cell(&mut self, check: bool) -> Result<(), GateError> {
            for f in [FLAG_OUT, FLAG_BACK, CNT_BACK, G_DONE] {
                self.sh.flag(f).store(0, Ordering::Relaxed);
            }
            let go = self.sh.go.load(Ordering::Relaxed);
            self.sh.done.store(go, Ordering::Relaxed);
            self.sh.stop.store(false, Ordering::Relaxed);
            self.sh.check.store(check, Ordering::Relaxed);
            lock(&self.sh.recs).clear();
            lock(&self.sh.worker_tids).clear();
            self.dev
                .report
                .copy_from_host(self.gpu.stream(), &[0; jp::N_REPORT])?;
            Ok(())
        }
    }

    /// Release a stuck replay: stop the workers, let a waiting host function
    /// return, and satisfy the memop waits.
    fn rescue(sh: &Shared, mech: Mech, n: usize) {
        sh.stop.store(true, Ordering::Release);
        match mech {
            Mech::MemOp => sh.flag(FLAG_BACK).store(1, Ordering::Release),
            Mech::MemOpRed => sh.flag(CNT_BACK).store(n as u32, Ordering::Release),
            Mech::HostFn | Mech::HostFnOne | Mech::HostFnSpin => {}
        }
    }

    /// How often a checked replay's wait asks the stream.
    const POLL_CHECK: Duration = Duration::from_micros(20);
    /// How often a timed batch's wait asks: the numbers come from the device
    /// clock, and fewer queries leave the driver's own threads alone.
    const POLL_TIME: Duration = Duration::from_micros(500);

    /// Poll `stream` every `poll` until its work is done. Past `deadline`,
    /// `rescue` runs once and the wait starts over; `Ok(false)` says it had
    /// to. A stream still busy a deadline after the release ends the process
    /// on the spot, naming `what`: tearing the rig down would block on that
    /// stream while this process holds the gate lock.
    fn wait_stream(
        stream: &CudaStream,
        deadline: Duration,
        poll: Duration,
        what: &str,
        rescue: &dyn Fn(),
    ) -> Result<bool, GateError> {
        let mut t0 = Instant::now();
        let mut rescued = false;
        loop {
            if stream.query()? {
                return Ok(!rescued);
            }
            if t0.elapsed() > deadline {
                if rescued {
                    eprintln!(
                        "FAIL: bench_join: {what}: the stream is still busy {deadline:?} after the \
                         release; exiting without teardown"
                    );
                    std::process::exit(1);
                }
                rescue();
                rescued = true;
                t0 = Instant::now();
            }
            std::thread::sleep(poll);
        }
    }

    /// Wait until `n` workers have recorded token `s`.
    fn wait_recs(sh: &Shared, s: u32, n: usize) -> bool {
        let t0 = Instant::now();
        while sh.recs_of(s) < n {
            if t0.elapsed() > DEADLINE {
                return false;
            }
            std::thread::sleep(Duration::from_micros(20));
        }
        true
    }

    /// Wait until the cell's `n` workers have registered.
    fn wait_registered(sh: &Shared, n: usize) -> Result<(), GateError> {
        let t0 = Instant::now();
        while lock(&sh.worker_tids).len() < n {
            if t0.elapsed() > DEADLINE {
                return Err("bench_join: workers did not start".into());
            }
            std::thread::sleep(Duration::from_micros(50));
        }
        Ok(())
    }

    /// The device addresses and counts a join enqueues, fixed per rig.
    #[derive(Clone, Copy)]
    struct JoinAddrs {
        flag_out: sys::CUdeviceptr,
        flag_back: sys::CUdeviceptr,
        gen_out: sys::CUdeviceptr,
        cnt_back: sys::CUdeviceptr,
        /// `memop_red`'s workers, the count its join back waits for.
        workers: u32,
    }

    /// The join out of `mech`, enqueued on `hs`.
    fn join_out(
        hs: sys::CUstream,
        mech: Mech,
        a: JoinAddrs,
        user: *mut c_void,
    ) -> Result<(), GpuError> {
        match mech {
            Mech::HostFn => {
                // SAFETY: `user` is the rig's `Shared`, alive past every replay.
                let rc = unsafe { sys::cuLaunchHostFunc(hs, Some(submit_fn), user) };
                drv(rc, "cuLaunchHostFunc")
            }
            Mech::HostFnOne => {
                // SAFETY: as above.
                let rc = unsafe { sys::cuLaunchHostFunc(hs, Some(hold_fn), user) };
                drv(rc, "cuLaunchHostFunc")
            }
            Mech::HostFnSpin => {
                // SAFETY: as above.
                let rc = unsafe {
                    sys::cuLaunchHostFunc_v2(hs, Some(submit_fn), user, sys::CU_HOST_TASK_SPINWAIT)
                };
                drv(rc, "cuLaunchHostFunc_v2")
            }
            Mech::MemOp => {
                // SAFETY: the flag page is host-mapped and outlives the graph.
                let rc = unsafe { sys::cuStreamWriteValue32_v2(hs, a.flag_out, 1, 0) };
                drv(rc, "cuStreamWriteValue32_v2")
            }
            Mech::MemOpRed => mem_batch(
                hs,
                &mut [op_barrier_sys(), op_add(a.gen_out, 1)],
                "cuStreamBatchMemOp_v2 (barrier, atomic add 1)",
            ),
        }
    }

    /// The join back of `mech`, enqueued on `hs`.
    fn join_back(
        hs: sys::CUstream,
        mech: Mech,
        a: JoinAddrs,
        user: *mut c_void,
    ) -> Result<(), GpuError> {
        match mech {
            Mech::HostFn => {
                // SAFETY: `user` is the rig's `Shared`, alive past every replay.
                let rc = unsafe { sys::cuLaunchHostFunc(hs, Some(wait_fn), user) };
                drv(rc, "cuLaunchHostFunc")
            }
            Mech::HostFnOne => Ok(()),
            Mech::HostFnSpin => {
                // SAFETY: as above.
                let rc = unsafe {
                    sys::cuLaunchHostFunc_v2(hs, Some(wait_fn), user, sys::CU_HOST_TASK_SPINWAIT)
                };
                drv(rc, "cuLaunchHostFunc_v2")
            }
            Mech::MemOp => mem_batch(
                hs,
                &mut [
                    op_wait(
                        a.flag_back,
                        1,
                        sys::CUstreamWaitValue_flags_enum_CU_STREAM_WAIT_VALUE_EQ,
                    ),
                    op_write(a.flag_back, 0),
                ],
                "cuStreamBatchMemOp_v2 (wait == 1, write 0)",
            ),
            Mech::MemOpRed => mem_batch(
                hs,
                &mut [
                    op_wait(
                        a.cnt_back,
                        a.workers,
                        sys::CUstreamWaitValue_flags_enum_CU_STREAM_WAIT_VALUE_GEQ,
                    ),
                    op_add(a.cnt_back, 0u32.wrapping_sub(a.workers)),
                ],
                "cuStreamBatchMemOp_v2 (wait >= workers, atomic add -workers)",
            ),
        }
    }

    /// Capture `mech` × `arm` into one graph: P, D2H, out, G, back, [H2D], C.
    fn capture(rig: &mut Rig, mech: Mech, arm: Arm) -> Result<Graph, GpuError> {
        let a = JoinAddrs {
            flag_out: rig.flag_dev(FLAG_OUT),
            flag_back: rig.flag_dev(FLAG_BACK),
            gen_out: rig.flag_dev(GEN_OUT),
            cnt_back: rig.flag_dev(CNT_BACK),
            workers: u32::try_from(rig.workers).expect("at most N_SLOTS workers"),
        };
        let g_done = rig.flag_dev(G_DONE);
        let user = (&raw const *rig.sh).cast_mut().cast::<c_void>();
        let Rig {
            gpu,
            probe,
            dev,
            payload,
            result,
            ..
        } = rig;
        let res = match arm {
            Arm::H2d => dev.res.cu_deviceptr(),
            Arm::Mapped => result.dev,
        };
        gpu.capture(|s| {
            let hs = s.cu_stream();
            probe.enqueue_produce(
                s,
                RING,
                &mut dev.seq,
                &mut dev.out,
                &mut dev.report,
                &mut dev.stamps,
            )?;
            // SAFETY: the payload is a live pinned allocation of PAYLOAD_WORDS
            // words that outlives the graph; `out` holds as many.
            let rc = unsafe {
                sys::cuMemcpyDtoHAsync_v2(
                    payload.host.cast(),
                    dev.out.cu_deviceptr(),
                    4 * jp::PAYLOAD_WORDS,
                    hs,
                )
            };
            drv(rc, "cuMemcpyDtoHAsync_v2")?;
            join_out(hs, mech, a, user)?;
            // SAFETY: `g_done` is a word of the host-mapped flag page, which
            // outlives the graph.
            unsafe { probe.enqueue_spin(s, RING, &dev.seq, &dev.params, &mut dev.stamps, g_done)? };
            join_back(hs, mech, a, user)?;
            if arm == Arm::H2d {
                // SAFETY: the result is a live pinned allocation of RESULT_LEN
                // f32 that outlives the graph; `res` holds as many.
                let rc = unsafe {
                    sys::cuMemcpyHtoDAsync_v2(
                        dev.res.cu_deviceptr(),
                        result.host.cast_const().cast(),
                        4 * jp::RESULT_LEN,
                        hs,
                    )
                };
                drv(rc, "cuMemcpyHtoDAsync_v2")?;
            }
            // SAFETY: `res` is the device buffer `dev.res` or the host-mapped
            // result, RESULT_LEN f32 either way, and both outlive the graph.
            unsafe {
                probe.enqueue_consume(s, RING, &dev.seq, res, &mut dev.report, &mut dev.stamps)
            }
        })
    }

    /// Run `mech`'s special operation once outside capture against the
    /// scratch word and read its effect back. The inner `Err` is a refusal
    /// or a wrong effect, a finding the run goes on past; the outer one is a
    /// stream that did not come back, which ends the run.
    fn eager_probe(rig: &Rig, mech: Mech) -> Result<Result<String, String>, GateError> {
        let stream = rig.gpu.stream();
        let hs = stream.cu_stream();
        let scratch = rig.sh.flag(SCRATCH);
        let addr = rig.flag_dev(SCRATCH);
        match mech {
            Mech::HostFn | Mech::HostFnOne | Mech::HostFnSpin => {
                let hits = AtomicU32::new(0);
                let user = (&raw const hits).cast_mut().cast::<c_void>();
                let (rc, what) = if mech == Mech::HostFnSpin {
                    // SAFETY: count_fn touches only `hits`, which outlives the
                    // synchronize below.
                    let rc = unsafe {
                        sys::cuLaunchHostFunc_v2(
                            hs,
                            Some(count_fn),
                            user,
                            sys::CU_HOST_TASK_SPINWAIT,
                        )
                    };
                    (rc, "cuLaunchHostFunc_v2 (spinwait)")
                } else {
                    // SAFETY: as above.
                    let rc = unsafe { sys::cuLaunchHostFunc(hs, Some(count_fn), user) };
                    (rc, "cuLaunchHostFunc")
                };
                if let Err(e) = drv(rc, what) {
                    return Ok(Err(e.to_string()));
                }
                stream.synchronize()?;
                Ok(match hits.load(Ordering::Relaxed) {
                    1 => Ok(format!("op=\"{what}\" the host function ran once")),
                    n => Err(format!("{what}: the host function ran {n} times")),
                })
            }
            Mech::MemOp => {
                scratch.store(0, Ordering::Release);
                // SAFETY: the scratch word is host-mapped and outlives the
                // synchronize.
                let rc = unsafe { sys::cuStreamWriteValue32_v2(hs, addr, 0xa5a5, 0) };
                if let Err(e) = drv(rc, "cuStreamWriteValue32_v2") {
                    return Ok(Err(e.to_string()));
                }
                stream.synchronize()?;
                let wrote = scratch.load(Ordering::Acquire);
                scratch.store(0, Ordering::Release);
                if let Err(e) = mem_batch(
                    hs,
                    &mut [
                        op_wait(
                            addr,
                            7,
                            sys::CUstreamWaitValue_flags_enum_CU_STREAM_WAIT_VALUE_EQ,
                        ),
                        op_write(addr, 0),
                    ],
                    "cuStreamBatchMemOp_v2 (wait == 7, write 0)",
                ) {
                    return Ok(Err(e.to_string()));
                }
                let held = hold(stream)?;
                scratch.store(7, Ordering::Release);
                let passed = wait_stream(stream, DEADLINE, POLL_CHECK, "eager memop", &|| {
                    scratch.store(7, Ordering::Release);
                })?;
                let reset = scratch.load(Ordering::Acquire);
                Ok(if wrote == 0xa5a5 && held && passed && reset == 0 {
                    Ok(format!(
                        "op=\"write 0xa5a5; batch(wait == 7, write 0)\" read {wrote:#x}; the wait \
                         held {HOLD:?} until the host stored 7, then read {reset}"
                    ))
                } else {
                    Err(format!(
                        "write 0xa5a5 read {wrote:#x}; batch(wait == 7, write 0) held={held} \
                         released_by_host={passed} read {reset}"
                    ))
                })
            }
            Mech::MemOpRed => {
                scratch.store(5, Ordering::Release);
                if let Err(e) = mem_batch(
                    hs,
                    &mut [op_barrier_sys(), op_add(addr, 1)],
                    "cuStreamBatchMemOp_v2 (barrier, atomic add 1)",
                ) {
                    return Ok(Err(e.to_string()));
                }
                stream.synchronize()?;
                let up = scratch.load(Ordering::Acquire);
                if let Err(e) = mem_batch(
                    hs,
                    &mut [
                        op_wait(
                            addr,
                            7,
                            sys::CUstreamWaitValue_flags_enum_CU_STREAM_WAIT_VALUE_GEQ,
                        ),
                        op_add(addr, 0u32.wrapping_sub(7)),
                    ],
                    "cuStreamBatchMemOp_v2 (wait >= 7, atomic add -7)",
                ) {
                    return Ok(Err(e.to_string()));
                }
                let held = hold(stream)?;
                scratch.fetch_add(1, Ordering::Release);
                let passed = wait_stream(stream, DEADLINE, POLL_CHECK, "eager memop_red", &|| {
                    scratch.store(7, Ordering::Release);
                })?;
                let down = scratch.load(Ordering::Acquire);
                Ok(if up == 6 && held && passed && down == 0 {
                    Ok(format!(
                        "op=\"batch(barrier sys, atomic add 1); batch(wait >= 7, atomic add -7)\" \
                         5 -> {up}; the wait held {HOLD:?} until the host added 1, then read {down}"
                    ))
                } else {
                    Err(format!(
                        "atomic add on host-mapped memory: 5 +1 -> {up}; batch(wait >= 7, atomic \
                         add -7) held={held} released_by_host={passed} read {down}"
                    ))
                })
            }
        }
    }

    /// How long an eager wait must hold before the host releases it.
    const HOLD: Duration = Duration::from_millis(1);

    /// Give an enqueued wait `HOLD` to pass on its own, then say whether it
    /// still holds the stream: a wait that passed before the host wrote the
    /// value it waits for is not waiting.
    fn hold(stream: &CudaStream) -> Result<bool, GateError> {
        std::thread::sleep(HOLD);
        Ok(!stream.query()?)
    }

    /// One captured mechanism × arm.
    struct Cell {
        mech: Mech,
        arm: Arm,
        graph: Graph,
    }

    impl Cell {
        fn label(&self) -> String {
            format!("{}/{}", self.mech.name(), self.arm.name())
        }
    }

    /// The driver's names for the node types a join graph can hold.
    const NODE_KINDS: [(sys::CUgraphNodeType, &str); 7] = [
        (
            sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_KERNEL,
            "kernel",
        ),
        (
            sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_MEMCPY,
            "memcpy",
        ),
        (
            sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_MEMSET,
            "memset",
        ),
        (sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_HOST, "host"),
        (sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_EMPTY, "empty"),
        (
            sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_WAIT_EVENT,
            "wait_event",
        ),
        (
            sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_BATCH_MEM_OP,
            "batch_mem_op",
        ),
    ];

    /// `kernel,memcpy,host(spinwait),…` for a node list.
    fn kinds(nodes: &[NodeInfo]) -> String {
        nodes
            .iter()
            .map(|n| {
                let kind = NODE_KINDS
                    .iter()
                    .find(|k| k.0 == n.kind)
                    .map_or_else(|| format!("type{}", n.kind), |k| k.1.to_string());
                match n.host_sync {
                    Some(sys::CU_HOST_TASK_BLOCKING) => format!("{kind}(blocking)"),
                    Some(sys::CU_HOST_TASK_SPINWAIT) => format!("{kind}(spinwait)"),
                    Some(m) => format!("{kind}(sync{m})"),
                    None => kind,
                }
            })
            .collect::<Vec<_>>()
            .join(",")
    }

    /// One checked replay: what the consumer reported for it.
    struct Row {
        due: u32,
        mismatch: u32,
        stale: u32,
    }

    /// `--replays` single replays of one cell, each waited for, then the
    /// verdict line. `Ok(false)` is a failed check.
    fn check_cell(rig: &mut Rig, cell: &Cell, o: &Opts) -> Result<bool, GateError> {
        let (n, label) = (cell.mech.workers(rig.workers), cell.label());
        rig.set_work(o.gpu_us * 1000, o.host_us * 1000)?;
        rig.open_cell(true)?;
        let rig: &Rig = rig;
        let (sh, stream, mech) = (&*rig.sh, rig.gpu.stream(), cell.mech);
        let mut rows = Vec::with_capacity(o.replays);
        let mut timeout: Option<String> = None;
        std::thread::scope(|sc| -> Result<(), GateError> {
            for w in 0..n {
                sc.spawn(move || sh.worker(mech, w, n));
            }
            let run = (|| -> Result<(), GateError> {
                wait_registered(sh, n)?;
                for r in 0..o.replays {
                    cell.graph.launch(stream)?;
                    if !wait_stream(stream, DEADLINE, POLL_CHECK, &label, &|| {
                        rescue(sh, mech, n)
                    })? {
                        timeout = Some(format!("replay {r}: released after {DEADLINE:?}"));
                        break;
                    }
                    let rep = rig.dev.report.to_host_vec(stream)?;
                    let due = rep[jp::REP_DUE];
                    if !wait_recs(sh, due, n) {
                        timeout = Some(format!("replay {r}: the host never recorded token {due}"));
                        break;
                    }
                    rows.push(Row {
                        due,
                        mismatch: rep[jp::REP_MISMATCH],
                        stale: rep[jp::REP_STALE],
                    });
                }
                Ok(())
            })();
            sh.stop.store(true, Ordering::Release);
            run
        })?;
        let recs = lock(&sh.recs).clone();
        let (mut payload_bad, mut overlap_yes) = (0usize, 0usize);
        for row in &rows {
            let mine: Vec<&Rec> = recs.iter().filter(|x| x.seq == row.due).collect();
            if mine.len() != n || mine.iter().any(|x| !x.payload_ok) {
                payload_bad += 1;
            }
            if mine.len() == n && mine.iter().all(|x| x.overlap) {
                overlap_yes += 1;
            }
        }
        let gaps = (0..n)
            .filter(|&w| {
                let seqs: Vec<u32> = recs
                    .iter()
                    .filter(|x| x.worker == w)
                    .map(|x| x.seq)
                    .collect();
                seqs.windows(2).any(|p| p[1] != p[0].wrapping_add(1))
            })
            .count();
        let stale: Vec<(usize, &Row)> = rows
            .iter()
            .enumerate()
            .filter(|(_, r)| r.mismatch != 0)
            .collect();
        let pass = timeout.is_none()
            && rows.len() == o.replays
            && stale.is_empty()
            && payload_bad == 0
            && gaps == 0;
        let overlap = match overlap_yes {
            0 => "no",
            k if k == rows.len() => "yes",
            _ => "mixed",
        };
        let mut detail = String::new();
        if let Some((i, r)) = stale.first() {
            detail += &format!(
                " first_stale=\"replay {i}: the consumer read token {} (mod 2048) where {} was due, \
                 {} of {} values\"",
                r.stale.wrapping_sub(1),
                r.due & 0x7ff,
                r.mismatch,
                jp::RESULT_LEN
            );
        }
        if let Some(t) = &timeout {
            detail += &format!(" timeout=\"{t}\"");
        }
        println!(
            "check mech={} arm={} replays={}/{} gpu_work_us={} host_work_us={} workers={n} \
             consumer_stale={} payload_stale={payload_bad} worker_gaps={gaps} \
             overlap={overlap}({overlap_yes}/{}) expect_overlap={} {}{detail}",
            cell.mech.name(),
            cell.arm.name(),
            rows.len(),
            o.replays,
            o.gpu_us,
            o.host_us,
            stale.len(),
            rows.len(),
            if cell.mech.overlaps() { "yes" } else { "no" },
            verdict(pass),
        );
        Ok(pass)
    }

    /// `err` prefixed with the cell (or mechanism) it came out of.
    fn in_cell(label: &str, err: &GateError) -> GateError {
        let msg = err.to_string();
        format!(
            "bench_join: {label}: {}",
            msg.trim_start_matches("bench_join: ")
        )
        .into()
    }

    /// Check every cell; the labels of the ones that failed.
    fn check_all(rig: &mut Rig, cells: &[Cell], o: &Opts) -> Result<Vec<String>, GateError> {
        let mut failed = Vec::new();
        for c in cells {
            if !check_cell(rig, c, o).map_err(|e| in_cell(&c.label(), &e))? {
                failed.push(c.label());
            }
        }
        let stamps = rig.dev.stamps.to_host_vec(rig.gpu.stream())?;
        let step = (0..RING)
            .map(|i| stamps[i * jp::N_STAMP + jp::ST_TIMER_STEP])
            .filter(|&v| v > 0)
            .min();
        println!(
            "check clock=globaltimer smallest_step_ns={}",
            step.map_or_else(|| "n/a".to_string(), |v| v.to_string())
        );
        Ok(failed)
    }

    /// CPU ns per thread of this process (`/proc/self/task/*/schedstat`).
    fn cpu_by_thread() -> Vec<(u32, u64)> {
        let Ok(dir) = std::fs::read_dir("/proc/self/task") else {
            return Vec::new();
        };
        dir.filter_map(|e| {
            let e = e.ok()?;
            let tid = e.file_name().to_str()?.parse().ok()?;
            let s = std::fs::read_to_string(e.path().join("schedstat")).ok()?;
            let ns = s.split_whitespace().next()?.parse().ok()?;
            Some((tid, ns))
        })
        .collect()
    }

    /// Who burned the CPU between two snapshots, µs: [the harness thread,
    /// the workers, the thread the host functions ran on, every other thread
    /// (the driver's)].
    fn cpu_split(before: &[(u32, u64)], after: &[(u32, u64)], main: u32, sh: &Shared) -> [f64; 4] {
        let workers = lock(&sh.worker_tids).clone();
        let hostfn = sh.hostfn_tid.load(Ordering::Relaxed);
        let mut out = [0.0; 4];
        for &(t, ns) in after {
            let was = before.iter().find(|b| b.0 == t).map_or(0, |b| b.1);
            let class = if t == main {
                0
            } else if workers.contains(&t) {
                1
            } else if hostfn != 0 && t == hostfn {
                2
            } else {
                3
            };
            out[class] += ns.saturating_sub(was) as f64 / 1e3;
        }
        out
    }

    /// One (cell, grid point)'s timings over every round.
    #[derive(Default)]
    struct Acc {
        p_end_to_c_start: Vec<f64>,
        p_end_to_c_end: Vec<f64>,
        p_end_to_g_start: Vec<f64>,
        g_end_to_c_start: Vec<f64>,
        span: Vec<f64>,
        host_leg: Vec<f64>,
        host_signal: Vec<f64>,
        wall_us: f64,
        replays: usize,
        cpu_us: [f64; 4],
        /// The smallest nonzero `%globaltimer` step a spin saw, ns; 0 when
        /// no spin ran.
        clock_step_ns: u64,
        /// Stamp slots out of timeline order (P, G, C) or longer than a
        /// replay may take, left out of every statistic, and the first one's
        /// words. A `%globaltimer` read that came back wrong lands here.
        bad_slots: usize,
        first_bad: Option<String>,
    }

    /// What a timed batch runs against.
    struct Batch<'a> {
        gpu: &'a Gpu,
        dev: &'a mut Dev,
        sh: &'a Shared,
        cell: &'a Cell,
        workers: usize,
        main_tid: u32,
    }

    impl Batch<'_> {
        /// `k` back-to-back replays at G = `g_us`, H = `h_us`, folded into `acc`.
        fn run(&mut self, k: usize, g_us: u64, h_us: u64, acc: &mut Acc) -> Result<(), GateError> {
            let (sh, mech, n) = (self.sh, self.cell.mech, self.workers);
            let stream = self.gpu.stream();
            self.dev.params.copy_from_host(stream, &[g_us * 1000])?;
            sh.host_ns.store(h_us * 1000, Ordering::Relaxed);
            let s0 = self.dev.seq.to_host_vec(stream)?[0];
            let recs0 = lock(&sh.recs).len();
            let cpu0 = cpu_by_thread();
            let t0 = Instant::now();
            for _ in 0..k {
                self.cell.graph.launch(stream)?;
            }
            let budget = DEADLINE + Duration::from_micros((g_us + h_us + 1000) * k as u64);
            if !wait_stream(stream, budget, POLL_TIME, &self.cell.label(), &|| {
                rescue(sh, mech, n);
            })? {
                return Err(format!(
                    "bench_join: {} batch released after {budget:?}",
                    self.cell.label()
                )
                .into());
            }
            let wall = t0.elapsed();
            let last = s0.wrapping_add(k as u32);
            if !wait_recs(sh, last, n) {
                return Err(format!(
                    "bench_join: {} host never recorded token {last}",
                    self.cell.label()
                )
                .into());
            }
            let cpu1 = cpu_by_thread();
            let stamps = self.dev.stamps.to_host_vec(stream)?;
            for i in WARM..k {
                let s = s0.wrapping_add(1 + i as u32);
                let b = s as usize % RING * jp::N_STAMP;
                let st = &stamps[b..b + jp::N_STAMP];
                let chain = [
                    jp::ST_P_START,
                    jp::ST_P_END,
                    jp::ST_G_START,
                    jp::ST_G_END,
                    jp::ST_C_START,
                    jp::ST_C_END,
                ];
                let too_long =
                    st[jp::ST_C_END].wrapping_sub(st[jp::ST_P_START]) > DEADLINE.as_nanos() as u64;
                if too_long || chain.windows(2).any(|w| st[w[0]] > st[w[1]]) {
                    acc.bad_slots += 1;
                    if acc.first_bad.is_none() {
                        let p = (s.wrapping_sub(1)) as usize % RING * jp::N_STAMP;
                        acc.first_bad = Some(format!(
                            "token {s} (batch replay {i} of {k}): {:?}; token {}: {:?}",
                            st,
                            s.wrapping_sub(1),
                            &stamps[p..p + jp::N_STAMP]
                        ));
                    }
                    continue;
                }
                let us = |from: usize, to: usize| st[to].wrapping_sub(st[from]) as i64 as f64 / 1e3;
                acc.p_end_to_c_start.push(us(jp::ST_P_END, jp::ST_C_START));
                acc.p_end_to_c_end.push(us(jp::ST_P_END, jp::ST_C_END));
                acc.p_end_to_g_start.push(us(jp::ST_P_END, jp::ST_G_START));
                acc.g_end_to_c_start.push(us(jp::ST_G_END, jp::ST_C_START));
                acc.span.push(us(jp::ST_P_START, jp::ST_C_END));
                let step = st[jp::ST_TIMER_STEP];
                if step > 0 && (acc.clock_step_ns == 0 || step < acc.clock_step_ns) {
                    acc.clock_step_ns = step;
                }
            }
            let first = s0.wrapping_add(1 + WARM as u32);
            for r in &lock(&sh.recs)[recs0..] {
                if r.seq >= first && r.seq <= last {
                    acc.host_leg
                        .push(r.signaled.saturating_sub(r.seen) as f64 / 1e3);
                    acc.host_signal
                        .push(r.signaled.saturating_sub(r.written) as f64 / 1e3);
                }
            }
            acc.wall_us += wall.as_secs_f64() * 1e6;
            acc.replays += k;
            for (a, d) in acc
                .cpu_us
                .iter_mut()
                .zip(cpu_split(&cpu0, &cpu1, self.main_tid, sh))
            {
                *a += d;
            }
            Ok(())
        }
    }

    /// One round of one cell: the zero-work batch, then every grid point.
    fn time_cell(
        rig: &mut Rig,
        cell: &Cell,
        o: &Opts,
        accs: &mut [Acc],
        main_tid: u32,
    ) -> Result<(), GateError> {
        let (mech, n) = (cell.mech, cell.mech.workers(rig.workers));
        rig.open_cell(false)?;
        let sh: &Shared = &rig.sh;
        let mut batch = Batch {
            gpu: rig.gpu,
            dev: &mut rig.dev,
            sh,
            cell,
            workers: n,
            main_tid,
        };
        std::thread::scope(|sc| -> Result<(), GateError> {
            for w in 0..n {
                sc.spawn(move || sh.worker(mech, w, n));
            }
            let run = (|| -> Result<(), GateError> {
                wait_registered(sh, n)?;
                for (acc, &(g, h)) in accs.iter_mut().zip(&o.points()) {
                    batch.run(o.batch, g, h, acc)?;
                }
                Ok(())
            })();
            sh.stop.store(true, Ordering::Release);
            run
        })
    }

    /// `name_med=… name_mean=… name_min=… name_max=…` over `v`, µs.
    fn stat(name: &str, v: &mut [f64]) -> String {
        if v.is_empty() {
            return format!("{name}=n/a");
        }
        v.sort_by(f64::total_cmp);
        let med = v[v.len() / 2];
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        format!(
            "{name}_med={med:.2} {name}_mean={mean:.2} {name}_min={:.2} {name}_max={:.2}",
            v[0],
            v[v.len() - 1]
        )
    }

    fn median(v: &mut [f64]) -> f64 {
        v.sort_by(f64::total_cmp);
        v.get(v.len() / 2).copied().unwrap_or(f64::NAN)
    }

    /// Every round of every cell, the order rotated per round, then one line
    /// per cell and grid point.
    fn time_all(
        rig: &mut Rig,
        cells: &[Cell],
        o: &Opts,
        card: &str,
        main_tid: u32,
    ) -> Result<(), GateError> {
        let points = o.points();
        let mut accs: Vec<Vec<Acc>> = cells
            .iter()
            .map(|_| points.iter().map(|_| Acc::default()).collect())
            .collect();
        println!(
            "time card={card} rounds={} batch={} warm={WARM} points={} clock=globaltimer \
             join=\"P end to C start/end on the device clock: both ends on one clock, where a \
             replay's wall time also carries the launch and the harness's wait; one tick is \
             clock_step_ns, so a single sample is quantized to it\"",
            o.rounds,
            o.batch,
            points
                .iter()
                .map(|(g, h)| format!("{g}:{h}"))
                .collect::<Vec<_>>()
                .join(","),
        );
        for r in 0..o.rounds {
            let order: Vec<usize> = (0..cells.len()).map(|i| (i + r) % cells.len()).collect();
            let names: Vec<String> = order.iter().map(|&i| cells[i].label()).collect();
            println!("time round={r} order={}", names.join(","));
            for &i in &order {
                time_cell(rig, &cells[i], o, &mut accs[i], main_tid)
                    .map_err(|e| in_cell(&cells[i].label(), &e))?;
            }
        }
        for (cell, acc) in cells.iter().zip(accs.iter_mut()) {
            let n = cell.mech.workers(rig.workers);
            let zero_span = median(&mut acc[0].span);
            for (a, &(g, h)) in acc.iter_mut().zip(&points) {
                let per = |x: f64| x / a.replays.max(1) as f64;
                let mut line = format!(
                    "time card={card} mech={} arm={} gpu_work_us={g} host_work_us={h} workers={n} \
                     rounds={} samples={} clock_step_ns={} {} {} {} {} {} {} {} \
                     wall_us_per_replay={:.2} cpu_us_per_replay main={:.1} workers={:.1} \
                     hostfn_thread={:.1} other={:.1}",
                    cell.mech.name(),
                    cell.arm.name(),
                    o.rounds,
                    a.span.len(),
                    match a.clock_step_ns {
                        0 => "n/a".to_string(),
                        v => v.to_string(),
                    },
                    stat("p_end_to_c_start_us", &mut a.p_end_to_c_start),
                    stat("p_end_to_c_end_us", &mut a.p_end_to_c_end),
                    stat("p_end_to_g_start_us", &mut a.p_end_to_g_start),
                    stat("g_end_to_c_start_us", &mut a.g_end_to_c_start),
                    stat("span_us", &mut a.span),
                    stat("host_leg_us", &mut a.host_leg),
                    stat("host_signal_us", &mut a.host_signal),
                    per(a.wall_us),
                    per(a.cpu_us[0]),
                    per(a.cpu_us[1]),
                    per(a.cpu_us[2]),
                    per(a.cpu_us[3]),
                );
                line += &format!(" bad_slots={}", a.bad_slots);
                if let Some(b) = &a.first_bad {
                    line += &format!(" first_bad=\"{b}\"");
                }
                if g > 0 || h > 0 {
                    let span = median(&mut a.span);
                    let (gf, hf) = (g as f64, h as f64);
                    let hidden = (zero_span + gf + hf - span) / gf.min(hf).max(1.0);
                    line += &format!(
                        " zero_span_us={zero_span:.2} ideal_us={:.2} serial_us={:.2} hidden_fraction={hidden:.3}",
                        zero_span + gf.max(hf),
                        zero_span + gf + hf,
                    );
                }
                println!("{line}");
            }
        }
        Ok(())
    }

    /// The command line.
    struct Opts {
        time: bool,
        gpu_us: u64,
        host_us: u64,
        replays: usize,
        rounds: usize,
        batch: usize,
        grid: Vec<(u64, u64)>,
        workers: usize,
        only: Vec<Mech>,
    }

    fn value<'a>(it: &mut std::slice::Iter<'a, String>, flag: &str) -> Result<&'a str, GateError> {
        it.next()
            .map(String::as_str)
            .ok_or_else(|| format!("bench_join: {flag} needs a value").into())
    }

    fn number<T: std::str::FromStr>(v: &str, flag: &str) -> Result<T, GateError> {
        v.parse()
            .map_err(|_| format!("bench_join: {flag} takes a number, got {v:?}").into())
    }

    impl Opts {
        fn parse() -> Result<Opts, GateError> {
            let args: Vec<String> = std::env::args().skip(1).collect();
            let mut o = Opts {
                time: false,
                gpu_us: 20,
                host_us: 1000,
                replays: 32,
                rounds: 5,
                batch: 64,
                grid: vec![(200, 600), (600, 600), (1000, 600)],
                workers: jp::N_SLOTS,
                only: MECHS.to_vec(),
            };
            let mut mode: Option<bool> = None;
            let mut it = args.iter();
            while let Some(a) = it.next() {
                let a = a.as_str();
                match a {
                    "--check" | "--time" => {
                        if mode.replace(a == "--time").is_some() {
                            return Err("bench_join: give one of --check, --time".into());
                        }
                    }
                    "--gpu-work-us" => o.gpu_us = number(value(&mut it, a)?, a)?,
                    "--host-work-us" => o.host_us = number(value(&mut it, a)?, a)?,
                    "--replays" => o.replays = number(value(&mut it, a)?, a)?,
                    "--rounds" => o.rounds = number(value(&mut it, a)?, a)?,
                    "--batch" => o.batch = number(value(&mut it, a)?, a)?,
                    "--workers" => o.workers = number(value(&mut it, a)?, a)?,
                    "--grid" => {
                        o.grid = value(&mut it, a)?
                            .split(',')
                            .map(|p| {
                                let (g, h) = p.split_once(':').ok_or_else(|| {
                                    format!("bench_join: --grid points are G:H, got {p:?}")
                                })?;
                                Ok((number(g, a)?, number(h, a)?))
                            })
                            .collect::<Result<_, GateError>>()?;
                    }
                    "--only" => {
                        o.only = value(&mut it, a)?
                            .split(',')
                            .map(|m| {
                                Mech::from_name(m)
                                    .ok_or_else(|| format!("bench_join: no mechanism {m:?}").into())
                            })
                            .collect::<Result<_, GateError>>()?;
                    }
                    other => return Err(format!("bench_join: unknown argument {other:?}").into()),
                }
            }
            o.time = mode.ok_or("bench_join: want one of --check, --time")?;
            if !(1..=jp::N_SLOTS).contains(&o.workers)
                || !(WARM + 1..=RING).contains(&o.batch)
                || o.replays == 0
                || o.rounds == 0
                || o.only.is_empty()
                || o.grid.iter().any(|&(g, h)| g > 100_000 || h > 100_000)
            {
                return Err(format!(
                    "bench_join: need 1 <= workers <= {}, {} <= batch <= {RING}, replays and rounds \
                     >= 1, a mechanism, grid points <= 100000 us",
                    jp::N_SLOTS,
                    WARM + 1
                )
                .into());
            }
            Ok(o)
        }

        /// The zero-work point, then the grid.
        fn points(&self) -> Vec<(u64, u64)> {
            std::iter::once((0, 0))
                .chain(self.grid.iter().copied())
                .collect()
        }
    }

    fn attr(gpu: &Gpu, a: sys::CUdevice_attribute) -> Result<i32, GateError> {
        let mut v = 0;
        // SAFETY: `v` is a live local the call writes; the device handle is
        // the one the context was made on.
        let rc = unsafe { sys::cuDeviceGetAttribute(&mut v, a, gpu.context().cu_device()) };
        drv(rc, "cuDeviceGetAttribute")?;
        Ok(v)
    }

    /// The card, the driver and the attributes the mechanisms depend on.
    fn header(gpu: &Gpu, card: &str, o: &Opts) -> Result<(), GateError> {
        let mut version = 0;
        // SAFETY: `version` is a live local the call writes.
        let rc = unsafe { sys::cuDriverGetVersion(&mut version) };
        drv(rc, "cuDriverGetVersion")?;
        println!(
            "bench_join card={card} sms={} driver_api={version} headers={} mode={} \
             mechanisms={}",
            gpu.context().multiprocessor_count()?,
            sys::CUDA_VERSION,
            if o.time { "time" } else { "check" },
            o.only
                .iter()
                .map(|m| m.name())
                .collect::<Vec<_>>()
                .join(","),
        );
        let attrs = [
            (
                "stream_mem_ops_64bit",
                sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS,
            ),
            (
                "wait_value_nor",
                sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR,
            ),
            (
                "atomic_reduction",
                sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_ATOMIC_REDUCTION_SUPPORTED,
            ),
            (
                "host_native_atomic",
                sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED,
            ),
            (
                "unified_addressing",
                sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING,
            ),
            (
                "can_map_host_memory",
                sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY,
            ),
            (
                "can_flush_remote_writes",
                sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES,
            ),
        ];
        let vals = attrs
            .iter()
            .map(|&(name, a)| Ok(format!("{name}={}", attr(gpu, a)?)))
            .collect::<Result<Vec<_>, GateError>>()?;
        println!("bench_join attrs {}", vals.join(" "));
        println!(
            "bench_join shape hidden={} slots={} payload_bytes={} result_bytes={} ring={RING} \
             memop_red_workers={}",
            jp::HIDDEN,
            jp::N_SLOTS,
            4 * jp::PAYLOAD_WORDS,
            4 * jp::RESULT_LEN,
            o.workers
        );
        Ok(())
    }

    pub fn run() -> Result<(), GateError> {
        let o = Opts::parse()?;
        let gpu = Gpu::new()?;
        let card = gpu.context().device_name()?;
        header(&gpu, &card, &o)?;
        let main_tid = tid();
        let mut rig = Rig::new(&gpu, o.workers)?;
        let (mut eager_ok, mut eager_refused) = (Vec::new(), Vec::new());
        for &m in &o.only {
            match eager_probe(&rig, m).map_err(|e| in_cell(m.name(), &e))? {
                Ok(what) => {
                    println!("eager mech={} {what} ok", m.name());
                    eager_ok.push(m);
                }
                Err(e) => {
                    println!("eager mech={} refused error=\"{e}\"", m.name());
                    eager_refused.push(m.name());
                }
            }
        }
        // Declared after `rig`: every graph drops before the buffers and the
        // `Shared` whose addresses it captured.
        let mut cells: Vec<Cell> = Vec::new();
        let mut refused = Vec::new();
        for &m in &o.only {
            for arm in ARMS {
                let label = format!("{}/{}", m.name(), arm.name());
                let graph = match capture(&mut rig, m, arm) {
                    Ok(graph) => graph,
                    Err(e) => {
                        println!(
                            "capture mech={} arm={} refused error=\"{e}\"",
                            m.name(),
                            arm.name()
                        );
                        refused.push(label);
                        continue;
                    }
                };
                let kinds = match graph.nodes() {
                    Ok(nodes) => kinds(&nodes),
                    Err(e) => format!("unavailable error=\"{e}\""),
                };
                println!(
                    "capture mech={} arm={} accepted nodes={} kinds={kinds}",
                    m.name(),
                    arm.name(),
                    graph.node_count(),
                );
                if eager_ok.contains(&m) {
                    cells.push(Cell {
                        mech: m,
                        arm,
                        graph,
                    });
                } else {
                    println!(
                        "capture mech={} arm={} not replayed: its eager probe refused",
                        m.name(),
                        arm.name()
                    );
                }
            }
        }
        if cells.is_empty() {
            return Err("bench_join: no mechanism both passed its eager probe and captured".into());
        }
        let failed = check_all(&mut rig, &cells, &o)?;
        if !failed.is_empty() {
            eprintln!("FAIL: bench_join check failed for: {}", failed.join(", "));
            return Err(checks_failed());
        }
        println!(
            "PASSED: bench_join check — {} cells, every replay's consumer read that replay's own \
             host values; eager refused: [{}]; capture refused: [{}]",
            cells.len(),
            eager_refused.join(", "),
            refused.join(", ")
        );
        if o.time {
            time_all(&mut rig, &cells, &o, &card, main_tid)?;
        }
        Ok(())
    }
}
