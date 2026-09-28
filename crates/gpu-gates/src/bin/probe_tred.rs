//! Cross-card exchange probe: how long one card takes to hand a row to the
//! other through mapped host memory, inside captured graphs — the reduction a
//! tensor split across the A6000 and the 3090 would pay twice a layer. The
//! cards have no peer path, so the row crosses host memory both ways.
//!
//! Two graphs, one a card, captured on each card's engine stream, run
//! `k` ping-pongs. Ping-pong `i`, card A: (copy arm) its payload into the
//! mapped page, a system barrier and flag A := `i`; then a wait until flag
//! B >= `i`, (copy arm) and B's payload into its own buffer. Card B mirrors
//! it: wait for flag A >= `i`, (copy arm) A's payload in, its own out, a
//! barrier, flag B := `i`. The waits are stream memory operations on the
//! mapped words, the device-side poll the host tier's join already uses — not
//! an SM spinning on the word. The `flag` arm is the same without the copies:
//! the signalling floor. The host zeroes both flags before each launch pair,
//! launches B's graph, then A's, and times from A's launch to both streams'
//! end; the round trip is that over `k`, one direction its half.
//!
//! `probe_tred --check`: both arms at [`CHECK_K`] ping-pongs; both flags must
//! end at the count, and in the copy arm each card's received buffer must equal
//! the other card's payload word for word. Red on any. Also prints whether the
//! cards can reach each other's memory (`cuDeviceCanAccessPeer`, both ways).
//!
//! `probe_tred --time` (under `tools/ref/probe-twocard.sh`: the lease, both
//! cards in view, the A6000 first): the check, then per arm one warm launch
//! pair and [`REPS`] timed ones at [`K`] ping-pongs; the round trip's minimum,
//! median and maximum over the pairs, and the median's half. Every line is a
//! record kind of `bloomery_gpu_gates::record` (`--records-schema`).

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("probe_tred: built without the `gpu` feature; see `just probe-gpu-tred`.");
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("probe_tred", probe::run())
}

#[cfg(feature = "gpu")]
mod probe {
    use std::ffi::c_void;
    use std::ptr;
    use std::time::Instant;

    use bloomery_gpu::{Gpu, GpuError, Graph};
    use bloomery_gpu_gates::record::{self, Record};
    use bloomery_gpu_gates::{GateError, checks_failed};
    use cuda_core::{DeviceBuffer, DriverError, sys};

    /// Ping-pongs per timed launch pair.
    const K: usize = 1024;
    /// Ping-pongs per checked launch pair.
    const CHECK_K: usize = 16;
    /// Timed launch pairs per arm.
    const REPS: usize = 7;
    /// Bytes each direction carries in the copy arm: one f32 row of 4,096.
    const PAYLOAD: usize = 16384;
    /// Byte offsets in the mapped page: each flag on a line of its own, then
    /// A's payload and B's.
    const FLAG_A: usize = 0;
    const FLAG_B: usize = 64;
    const DATA_A: usize = 4096;
    const DATA_B: usize = DATA_A + PAYLOAD;
    const MAPPED: usize = DATA_B + PAYLOAD;
    /// Card A and card B, by the names the plans use.
    const CARD_A: &str = "A6000";
    const CARD_B: &str = "3090";

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Arm {
        Flag,
        Copy,
    }

    impl Arm {
        fn name(self) -> &'static str {
            match self {
                Arm::Flag => "flag",
                Arm::Copy => "copy",
            }
        }
    }

    fn cu(rc: sys::CUresult, what: &'static str) -> Result<(), GpuError> {
        if rc == sys::cudaError_enum_CUDA_SUCCESS {
            Ok(())
        } else {
            Err(GpuError::Driver {
                op: Some(what),
                source: DriverError(rc),
            })
        }
    }

    /// Mapped, portable pinned host memory: the host reaches it at `host`,
    /// each card at `dev[card]`. Freed on drop.
    struct Page {
        host: *mut u8,
        dev: [sys::CUdeviceptr; 2],
    }

    impl Page {
        fn new(cards: [&Gpu; 2]) -> Result<Page, GpuError> {
            cards[0].context().bind_to_thread()?;
            let mut host: *mut c_void = ptr::null_mut();
            // SAFETY: card A's context is current (bound above) and `host` is
            // a live local the call writes.
            let rc = unsafe {
                sys::cuMemHostAlloc(
                    &mut host,
                    MAPPED,
                    sys::CU_MEMHOSTALLOC_PORTABLE | sys::CU_MEMHOSTALLOC_DEVICEMAP,
                )
            };
            cu(rc, "cuMemHostAlloc")?;
            let mut page = Page {
                host: host.cast(),
                dev: [0; 2],
            };
            // SAFETY: the allocation holds MAPPED writable bytes, referenced
            // by nothing else yet.
            unsafe { ptr::write_bytes(page.host, 0, MAPPED) };
            for (c, gpu) in cards.iter().enumerate() {
                gpu.context().bind_to_thread()?;
                // SAFETY: `host` is the portable mapped allocation above, which
                // every context of the process may map; the flags must be 0.
                let rc = unsafe { sys::cuMemHostGetDevicePointer_v2(&mut page.dev[c], host, 0) };
                cu(rc, "cuMemHostGetDevicePointer_v2")?;
            }
            Ok(page)
        }

        fn flag(&self, off: usize) -> u32 {
            // SAFETY: `off` is FLAG_A or FLAG_B, a 4-aligned word inside the
            // allocation; the cards write it by DMA, so the read is volatile.
            unsafe { ptr::read_volatile(self.host.add(off).cast::<u32>()) }
        }

        fn clear_flags(&self) {
            for off in [FLAG_A, FLAG_B] {
                // SAFETY: as `flag`; both streams are idle when this runs.
                unsafe { ptr::write_volatile(self.host.add(off).cast::<u32>(), 0) };
            }
        }
    }

    impl Drop for Page {
        fn drop(&mut self) {
            // SAFETY: `host` came from cuMemHostAlloc and is freed once, after
            // the graphs that name it (they are dropped before the page).
            unsafe { sys::cuMemFreeHost(self.host.cast()) };
        }
    }

    fn op_zero() -> sys::CUstreamBatchMemOpParams {
        // SAFETY: every member of the union is a plain C struct of integers.
        unsafe { std::mem::zeroed() }
    }

    fn op_wait_geq(addr: sys::CUdeviceptr, value: u32) -> sys::CUstreamBatchMemOpParams {
        let mut p = op_zero();
        p.waitValue = sys::CUstreamBatchMemOpParams_union_CUstreamMemOpWaitValueParams_st {
            operation: sys::CUstreamBatchMemOpType_enum_CU_STREAM_MEM_OP_WAIT_VALUE_32,
            address: addr,
            __bindgen_anon_1:
                sys::CUstreamBatchMemOpParams_union_CUstreamMemOpWaitValueParams_st__bindgen_ty_1 {
                    value,
                },
            flags: sys::CUstreamWaitValue_flags_enum_CU_STREAM_WAIT_VALUE_GEQ,
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

    fn mem_batch(
        stream: &cuda_core::CudaStream,
        ops: &mut [sys::CUstreamBatchMemOpParams],
    ) -> Result<(), GpuError> {
        let n = u32::try_from(ops.len()).expect("a batch of a few operations");
        // SAFETY: `ops` is a live array of `n` initialized operations the
        // driver reads during the call (a capture copies them into the node).
        let rc = unsafe { sys::cuStreamBatchMemOp_v2(stream.cu_stream(), n, ops.as_mut_ptr(), 0) };
        cu(rc, "cuStreamBatchMemOp_v2")
    }

    /// One card's device side: its payload out and the buffer the other
    /// card's payload lands in.
    struct Side {
        out: DeviceBuffer<u32>,
        inb: DeviceBuffer<u32>,
    }

    impl Side {
        fn new(gpu: &Gpu, tag: u32) -> Result<Side, GpuError> {
            gpu.context().bind_to_thread()?;
            let words = PAYLOAD / 4;
            let pattern: Vec<u32> = (0..words as u32)
                .map(|i| tag ^ i.wrapping_mul(2_654_435_761))
                .collect();
            Ok(Side {
                out: DeviceBuffer::from_host(gpu.stream(), &pattern)?,
                inb: DeviceBuffer::zeroed(gpu.stream(), words)?,
            })
        }
    }

    /// Card `me`'s graph of `k` ping-pongs. A (`me` 0) signals first, B waits
    /// first.
    fn capture(
        gpu: &Gpu,
        me: usize,
        page: &Page,
        side: &Side,
        arm: Arm,
        k: usize,
    ) -> Result<Graph, GpuError> {
        gpu.context().bind_to_thread()?;
        let dev = page.dev[me];
        let (my_flag, their_flag, my_data, their_data) = if me == 0 {
            (FLAG_A, FLAG_B, DATA_A, DATA_B)
        } else {
            (FLAG_B, FLAG_A, DATA_B, DATA_A)
        };
        let send = |s: &cuda_core::CudaStream, i: u32| -> Result<(), GpuError> {
            if arm == Arm::Copy {
                // SAFETY: the page holds PAYLOAD bytes at `my_data` and the
                // buffer PAYLOAD bytes; both outlive every launch of the graph.
                let rc = unsafe {
                    sys::cuMemcpyDtoHAsync_v2(
                        page.host.add(my_data).cast(),
                        side.out.cu_deviceptr(),
                        PAYLOAD,
                        s.cu_stream(),
                    )
                };
                cu(rc, "cuMemcpyDtoHAsync_v2 (payload out)")?;
            }
            mem_batch(
                s,
                &mut [op_barrier_sys(), op_write(dev + my_flag as u64, i)],
            )
        };
        let receive = |s: &cuda_core::CudaStream, i: u32| -> Result<(), GpuError> {
            mem_batch(s, &mut [op_wait_geq(dev + their_flag as u64, i)])?;
            if arm == Arm::Copy {
                // SAFETY: as in `send`, the other card's span of the page.
                let rc = unsafe {
                    sys::cuMemcpyHtoDAsync_v2(
                        side.inb.cu_deviceptr(),
                        page.host.add(their_data).cast_const().cast(),
                        PAYLOAD,
                        s.cu_stream(),
                    )
                };
                cu(rc, "cuMemcpyHtoDAsync_v2 (payload in)")?;
            }
            Ok(())
        };
        gpu.capture(|s| {
            for i in 1..=u32::try_from(k).expect("k fits u32") {
                if me == 0 {
                    send(s, i)?;
                    receive(s, i)?;
                } else {
                    receive(s, i)?;
                    send(s, i)?;
                }
            }
            Ok(())
        })
    }

    /// One launch pair: flags zeroed, B launched, then A; microseconds from
    /// A's launch to both streams' end.
    fn pair(cards: [&Gpu; 2], page: &Page, graphs: &[Graph; 2]) -> Result<f64, GateError> {
        page.clear_flags();
        cards[1].context().bind_to_thread()?;
        graphs[1].launch(cards[1].stream())?;
        cards[0].context().bind_to_thread()?;
        let t0 = Instant::now();
        graphs[0].launch(cards[0].stream())?;
        cards[0].stream().synchronize()?;
        cards[1].context().bind_to_thread()?;
        cards[1].stream().synchronize()?;
        Ok(t0.elapsed().as_secs_f64() * 1e6)
    }

    fn peer(cards: [&Gpu; 2]) -> Result<[bool; 2], GpuError> {
        let mut out = [false; 2];
        for (i, o) in out.iter_mut().enumerate() {
            let mut can = 0i32;
            // SAFETY: both handles are the live devices of the two contexts;
            // `can` is a local the call writes.
            let rc = unsafe {
                sys::cuDeviceCanAccessPeer(
                    &mut can,
                    cards[i].context().cu_device(),
                    cards[1 - i].context().cu_device(),
                )
            };
            cu(rc, "cuDeviceCanAccessPeer")?;
            *o = can != 0;
        }
        Ok(out)
    }

    /// One arm at `k` ping-pongs: its graphs, one launch pair, then the check;
    /// `true` when both flags ended at `k` and (copy) each card holds the
    /// other's payload.
    fn check_arm(
        cards: [&Gpu; 2],
        page: &Page,
        sides: &mut [Side; 2],
        arm: Arm,
        k: usize,
    ) -> Result<bool, GateError> {
        for (c, s) in sides.iter_mut().enumerate() {
            cards[c].context().bind_to_thread()?;
            s.inb
                .copy_from_host(cards[c].stream(), &vec![0u32; PAYLOAD / 4])?;
        }
        let graphs = [
            capture(cards[0], 0, page, &sides[0], arm, k)?,
            capture(cards[1], 1, page, &sides[1], arm, k)?,
        ];
        pair(cards, page, &graphs)?;
        let want = u32::try_from(k).expect("k fits u32");
        let flags_ok = page.flag(FLAG_A) == want && page.flag(FLAG_B) == want;
        let mut data_ok = true;
        if arm == Arm::Copy {
            for c in 0..2 {
                cards[c].context().bind_to_thread()?;
                let got = sides[c].inb.to_host_vec(cards[c].stream())?;
                cards[1 - c].context().bind_to_thread()?;
                let sent = sides[1 - c].out.to_host_vec(cards[1 - c].stream())?;
                data_ok &= got == sent;
            }
        }
        let payload = if arm == Arm::Copy {
            format!(" payload_matches={data_ok}")
        } else {
            String::new()
        };
        Record::new(&record::PROBE_CHECK)
            .w("probe", "probe_tred")
            .w("case", arm.name())
            .w(
                "saw",
                format!(
                    "k={k} flag_a={} flag_b={}{payload}",
                    page.flag(FLAG_A),
                    page.flag(FLAG_B)
                ),
            )
            .w("pass", flags_ok && data_ok)
            .print();
        Ok(flags_ok && data_ok)
    }

    pub fn run() -> Result<(), GateError> {
        record::at_main("probe_tred", record::PROBE_TRED_BIN);
        let args: Vec<String> = std::env::args().skip(1).collect();
        let time = match args
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
            .as_slice()
        {
            ["--check"] => false,
            ["--time"] => true,
            _ => return Err(format!("want exactly one of --check, --time; got {args:?}").into()),
        };
        let a = Gpu::for_card(CARD_A)?;
        let b = Gpu::for_card(CARD_B)?;
        let cards = [&a, &b];
        let [a_to_b, b_to_a] = peer(cards)?;
        Record::new(&record::PROBE_PEER)
            .w("a", a.device_name()?)
            .w("b", b.device_name()?)
            .w("a_to_b", a_to_b)
            .w("b_to_a", b_to_a)
            .print();
        let page = Page::new(cards)?;
        let mut sides = [Side::new(&a, 0xA000_0000)?, Side::new(&b, 0xB000_0000)?];
        let mut ok = true;
        for arm in [Arm::Flag, Arm::Copy] {
            ok &= check_arm(cards, &page, &mut sides, arm, CHECK_K)?;
        }
        Record::new(&record::PROBE_CHECK)
            .w("probe", "probe_tred")
            .w("case", "all")
            .w(
                "saw",
                "both flags end at the count; each card holds the other's payload",
            )
            .w("pass", ok)
            .print();
        if !ok {
            return Err(checks_failed());
        }
        if !time {
            return Ok(());
        }
        for arm in [Arm::Flag, Arm::Copy] {
            let graphs = [
                capture(&a, 0, &page, &sides[0], arm, K)?,
                capture(&b, 1, &page, &sides[1], arm, K)?,
            ];
            pair(cards, &page, &graphs)?;
            let mut rt: Vec<f64> = Vec::with_capacity(REPS);
            for _ in 0..REPS {
                rt.push(pair(cards, &page, &graphs)? / K as f64);
                let want = u32::try_from(K).expect("K fits u32");
                if page.flag(FLAG_A) != want || page.flag(FLAG_B) != want {
                    return Err(format!(
                        "arm {}: the flags ended at {} and {}, not {K}: a wait was satisfied early",
                        arm.name(),
                        page.flag(FLAG_A),
                        page.flag(FLAG_B)
                    )
                    .into());
                }
            }
            rt.sort_by(f64::total_cmp);
            let median = rt[REPS / 2];
            Record::new(&record::PROBE_TRED)
                .w("arm", arm.name())
                .u("k", K)
                .u("payload", if arm == Arm::Copy { PAYLOAD } else { 0 })
                .u("reps", REPS)
                .f("rt_us_min", rt[0])
                .f("rt_us_median", median)
                .f("rt_us_max", rt[REPS - 1])
                .f("one_way_us", median / 2.0)
                .print();
        }
        Ok(())
    }
}
