//! The device contract's clauses on the card (`bloomery_device::conformance`):
//! the clauses Metal's gate runs, over `bloomery_gpu`'s `Device` and this
//! bin's kernels. One clause per line, `PASS`, `SKIP` or `FAIL`; the run's
//! exit code is 1 when any clause failed.
//!
//! The kernels: a copy of f32s between two windows, a long kernel that spins
//! on the global timer and then writes 1, and a raise of many lanes at once
//! that goes through the engine's `FaultSink` for every layer the word
//! carries and, for a layer past the last, lowers the first word alone, as
//! the contract's host model does. The hold a stream waits on is a host flag
//! (`HostFlags`), the engine's own.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    match gate::run() {
        Ok(code) => code,
        Err(e) => bloomery_gpu_gates::exit_with("gate_gpu_device", Err(e)),
    }
}

#[cfg(feature = "gpu")]
mod gate {
    use core::sync::atomic::Ordering;
    use std::cell::RefCell;
    use std::process::ExitCode;
    use std::sync::Arc;

    use bloomery_device::conformance::{self, Harness, Hold, Report};
    use bloomery_device::{
        Buffer, Device, DeviceError, Elems, FAULT_NONE, FaultSite, LAYER_NONE, Shared, Window,
        WindowMut, check_len,
    };
    use bloomery_gpu::device::{GpuCensus, GpuShared, GpuStream};
    use bloomery_gpu::{FaultSink, Gpu, GpuError, HostFlags};
    use bloomery_gpu_gates::GateError;
    use cuda_core::{CudaStream, LaunchConfig1D};
    use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32};
    use cuda_device::{DisjointSlice, debug, kernel, launch_bounds, launch_contract, thread};
    use cuda_host::cuda_module;

    /// How long [`device_kernels::long`] spins: far longer than the host
    /// takes to come back from a call, and short of any watchdog.
    const LONG_NS: u64 = 200_000_000;

    /// The threads of a copy's block.
    const COPY_THREADS: u32 = 128;

    /// The threads of a raise's block.
    const RAISE_THREADS: u32 = 64;

    #[cuda_module]
    mod device_kernels {
        use super::*;

        /// `dst[i] = src[i]` for every `i < n`, one element a thread.
        #[kernel]
        #[launch_bounds(128)]
        #[launch_contract(
            domain = 1,
            block = (128, 1, 1),
            requires = (src.len() >= n, dst.len() >= n)
        )]
        pub fn copy_f32(src: &[f32], mut dst: DisjointSlice<f32>, n: u32) {
            let idx = thread::index_1d();
            let i = idx.get();
            if i >= n as usize {
                return;
            }
            if let (Some(&v), Some(out)) = (src.get(i), dst.get_mut(idx)) {
                *out = v;
            }
        }

        /// Thread 0 spins until `ns` of the global timer have passed, then
        /// writes 1 to `out[0]`.
        #[kernel]
        #[launch_bounds(32)]
        #[launch_contract(domain = 1, block = (32, 1, 1), requires = (out.len() >= 1))]
        pub fn long(mut out: DisjointSlice<u32>, ns: u64) {
            let idx = thread::index_1d();
            if idx.get() != 0 {
                return;
            }
            let start = debug::globaltimer();
            while debug::globaltimer().wrapping_sub(start) < ns {}
            if let Some(o) = out.get_mut(idx) {
                *o = 1;
            }
        }

        /// Thread `i < n` raises site `codes[i]` at layer `layers[i]` into
        /// the fault allocation at device address `word`: through a
        /// [`FaultSink`] for a layer the word carries, and for a layer past
        /// [`LAYER_NONE`] by lowering the first word alone. A code no site
        /// has raises nothing; the host passes only sites.
        #[kernel]
        #[launch_bounds(64)]
        #[launch_contract(
            domain = 1,
            block = (64, 1, 1),
            requires = (layers.len() >= n, codes.len() >= n)
        )]
        pub fn raise_lanes(word: u64, layers: &[u32], codes: &[u32], n: u32) {
            let i = thread::index_1d().get();
            if i >= n as usize {
                return;
            }
            let (Some(&layer), Some(&code)) = (layers.get(i), codes.get(i)) else {
                return;
            };
            let Some(site) = FaultSite::from_code(code) else {
                return;
            };
            let word = word as *mut u32;
            if layer <= LAYER_NONE {
                // SAFETY: `word` is a `Gpu`'s fault allocation, FAULT_WORDS
                // live u32s of this context that outlive the launch (the
                // host passes its own `Gpu`'s word and waits for the launch
                // before the `Gpu` goes), and `layer <= LAYER_NONE`.
                unsafe { FaultSink::new(word, layer) }.raise(site);
            } else {
                // SAFETY: as above: word 0 is a live, aligned u32 of the
                // allocation, which the card reaches only atomically.
                let first = unsafe { DeviceAtomicU32::from_ptr(word) };
                first.fetch_min((layer << 8) | code, AtomicOrdering::Relaxed);
            }
        }
    }

    /// A failure of the engine crate, as the contract's error.
    fn cuda(e: impl Into<GpuError>) -> DeviceError {
        DeviceError::from(e.into())
    }

    /// The card and this bin's kernels, lent to the contract's clauses.
    struct CudaHarness {
        gpu: Gpu,
        kernels: device_kernels::LoadedModule,
        /// The raises' lane buffers, kept until the run ends: freeing one
        /// waits for the context's work, which would order a raise before the
        /// read a clause checks waits for it.
        lanes: RefCell<Vec<GpuShared<u32>>>,
    }

    /// A host flag the stream waits on: the hold opens when the host raises
    /// it. Its drop raises it and waits for the stream, so the flag's page
    /// outlives the wait the card polls it with; a stream that does not come
    /// back leaves the page alive.
    struct CudaHold {
        flags: std::mem::ManuallyDrop<HostFlags>,
        stream: Arc<CudaStream>,
    }

    impl Hold for CudaHold {
        fn open(&self) -> Result<(), DeviceError> {
            self.flags.raise(0).map_err(cuda)
        }
    }

    impl Drop for CudaHold {
        fn drop(&mut self) {
            if self.flags.raise(0).is_ok() && self.stream.synchronize().is_ok() {
                // SAFETY: taken once, here; the stream that waited on the
                // flag is drained, so no card wait reads the page again.
                unsafe { std::mem::ManuallyDrop::drop(&mut self.flags) };
            }
        }
    }

    impl Harness for CudaHarness {
        type Device = Gpu;
        type Hold = CudaHold;

        fn device(&self) -> &Gpu {
            &self.gpu
        }

        fn copy_f32(
            &self,
            s: &GpuStream,
            from: Window<'_, f32, Gpu>,
            to: WindowMut<'_, f32, Gpu>,
        ) -> Result<(), DeviceError> {
            check_len("copy", from.len().get(), to.len())?;
            let n = u32::try_from(from.len().get()).map_err(|_| DeviceError::Range {
                what: "copy",
                detail: format!("{} elements pass the kernel's u32", from.len().get()),
            })?;
            let (src, mut dst) = (self.gpu.bind(from)?, self.gpu.bind_mut(to)?);
            let cfg = LaunchConfig1D::new(n.div_ceil(COPY_THREADS).max(1), COPY_THREADS, 0);
            let prep = self.kernels.prepare_copy_f32(cfg).map_err(cuda)?;
            self.kernels
                .copy_f32(s.cuda(), &prep, &src, &mut dst, n)
                .map_err(cuda)
        }

        fn long(&self, s: &GpuStream, out: WindowMut<'_, u32, Gpu>) -> Result<(), DeviceError> {
            let mut out = self.gpu.bind_mut(out)?;
            let prep = self
                .kernels
                .prepare_long(LaunchConfig1D::new(1, 32, 0))
                .map_err(cuda)?;
            self.kernels
                .long(s.cuda(), &prep, &mut out, LONG_NS)
                .map_err(cuda)
        }

        fn raise(&self, s: &GpuStream, raises: &[(u32, FaultSite)]) -> Result<(), DeviceError> {
            let n = u32::try_from(raises.len()).map_err(|_| DeviceError::Range {
                what: "raise",
                detail: format!("{} lanes pass the kernel's u32", raises.len()),
            })?;
            let layers = self.gpu.shared::<u32>(Elems(raises.len()))?;
            let codes = self.gpu.shared::<u32>(Elems(raises.len()))?;
            for (i, &(layer, site)) in raises.iter().enumerate() {
                layers.store(i, layer, Ordering::Relaxed)?;
                codes.store(i, site as u32, Ordering::Relaxed)?;
            }
            {
                let (l, c) = (
                    self.gpu.bind(layers.whole()?)?,
                    self.gpu.bind(codes.whole()?)?,
                );
                let cfg = LaunchConfig1D::new(n.div_ceil(RAISE_THREADS).max(1), RAISE_THREADS, 0);
                let prep = self.kernels.prepare_raise_lanes(cfg).map_err(cuda)?;
                self.kernels
                    .raise_lanes(
                        s.cuda(),
                        &prep,
                        self.gpu.fault_word().cu_deviceptr(),
                        &l,
                        &c,
                        n,
                    )
                    .map_err(cuda)?;
            }
            self.lanes.borrow_mut().extend([layers, codes]);
            Ok(())
        }

        fn fault_words(&self, s: &GpuStream) -> Result<Vec<u32>, DeviceError> {
            self.gpu.fault_words(s)
        }

        fn hold(&self, s: &GpuStream) -> Result<CudaHold, DeviceError> {
            let flags = HostFlags::new(self.gpu.context(), 1).map_err(cuda)?;
            flags.enqueue_wait(s.cuda(), 0).map_err(cuda)?;
            Ok(CudaHold {
                flags: std::mem::ManuallyDrop::new(flags),
                stream: Arc::clone(s.cuda()),
            })
        }

        fn other_stream(&self) -> Result<GpuStream, DeviceError> {
            self.gpu.new_stream()
        }
    }

    pub fn run() -> Result<ExitCode, GateError> {
        let gpu = Gpu::new()?;
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above, loaded once into the card's context before any
        // launch; every launcher checks its launch contract.
        let kernels = unsafe { bloomery_gpu::shared_module!(device_kernels, gpu.context())? };
        let info = Device::info(&gpu)?;
        println!(
            "gate_gpu_device: device={} ordinal={} backend={} fault_none={FAULT_NONE:#x}",
            info.name,
            info.ordinal,
            <Gpu as Device>::BACKEND
        );
        let harness = CudaHarness {
            gpu,
            kernels,
            lanes: RefCell::new(Vec::new()),
        };
        let mut report = Report::new();
        conformance::run::<_, GpuCensus>(&harness, &mut report);
        Ok(report.finish("gate_gpu_device"))
    }
}
