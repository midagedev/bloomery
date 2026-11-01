//! The Metal device gate's clauses (macOS only).
//!
//! One clause per line, `PASS`, `SKIP` or `FAIL`, and the run's exit code is
//! 1 when any clause failed, 78 when this machine cannot run the backend (a
//! macOS below the floor, no Metal device), 0 otherwise. The contract's
//! clauses are `bloomery_device::conformance`'s, the ones every backend runs,
//! over this backend's kernels ([`MetalHarness`]); this backend's own follow:
//!
//! 1. **open**: the device opens, and the library path it came from is named.
//! 2. the contract's clauses, `zeroed_upload_roundtrip` to `census`.
//! 3. **encoder_refusals**: a binding slot past the last, a value of no bytes
//!    or too many, a dispatch of no groups, no pipeline or a too large group
//!    are refused by name.
//! 4. **sync_bound_named**: a sync and an event wait that outlast their bound
//!    are refused by name, and a host event is done only once signalled.
//! 5. **sync_two_waiters**: two threads waiting on one held stream at once,
//!    neither's wait ends before the host event opens, and both return `Ok`
//!    after it.
//! 6. **event_values_rise**: the events one stream records rise in value.
//! 7. **arith_matches_host**: a fused multiply-add and a multiply and add
//!    rounded apart are the host's bits.
//! 8. **metallib_equals_source**: the metallib's output and the source's
//!    output are the same bits, when both paths exist.
//! 9. **census_driver**: the census names an integrated device whose total is
//!    the driver's working set and no more than the machine's memory.
//! 10. **macos_floor**: a macOS below 26.2 is refused by name.
//! 11. **concurrent_barrier**: one concurrent encoder runs `spin_until_set`
//!     and then `set_flag`. With no barrier, a GPU that overlaps the two
//!     writes 2 (the flag arrived during the spin). With a memory barrier
//!     between them, the flag is stored only after the spin has written 1.
//!     No overlap, and the clause skips: a barrier cannot be shown to matter.
//! 12. **parked_goes_stale**: a token recording parked off the queue comes
//!     back while the stream's boundary is the value its wait names; once a
//!     signalled buffer has moved the boundary, or on another stream, it does
//!     not.
//! 13. **wide_heads_compile**: the source of an instance made at open for 64
//!     query heads over 8 key heads of 128 values (more query heads than the
//!     flash's residency hold covers) compiles, and its flash pipelines build.
//!
//! `--drop-barrier` is this gate's seam: the ordered arm encodes no memory
//! barrier. Nothing else reads it. `--help` lists the arguments.

use core::sync::atomic::Ordering;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use bloomery_device::conformance::{
    self, Clause, Harness, Hold, Report, Verdict, expect_range, why,
};
use bloomery_device::{
    Buffer, Census, Device, DeviceError, Elems, Event, FaultSite, Shared, Stream, Window,
    WindowMut, check_len,
};
use objc2_foundation::NSProcessInfo;
use objc2_metal::MTLDevice;

use bloomery_metal::buffer::{MetalBuffer, MetalShared};
use bloomery_metal::census::MetalCensus;
use bloomery_metal::device::{MetalDevice, require_macos};
use bloomery_metal::encode::{Binding, MAX_BINDINGS, MAX_BYTES_BINDING};
use bloomery_metal::instance::{HOST, QWEN3MOE_A3B, open_key};
use bloomery_metal::library::{Library, Pipeline, sources_for};
use bloomery_metal::stream::{HostEvent, MetalStream, Parked};
use models::shape::gqa_row;

/// How many times the long kernels read their flag: long enough that the host
/// is back from a call before the kernel is over, and short of the GPU
/// watchdog.
const SPIN: u32 = 1 << 22;

/// What `spin_until_set` writes when its loads run out.
const SPIN_RAN_OUT: u32 = 1;

/// What `spin_until_set` writes when it saw the flag.
const SPIN_SAW_FLAG: u32 = 2;

/// How long the host waits for the barrier clause's command buffer.
///
/// The spin writes and returns after [`SPIN`] loads. The long kernel uses
/// that same cap and returns inside this gate, so the wait is not what ends
/// the spin: a buffer that is still running is a named failure.
const BARRIER_WAIT: Duration = Duration::from_secs(30);

/// The largest group a gate dispatch asks for.
const GROUP: usize = 64;

/// Encode one dispatch of `pipeline` over `bindings` (slot = position) and
/// submit it; it does not wait.
fn launch(
    stream: &MetalStream,
    pipeline: &Pipeline,
    bindings: &[Binding<'_>],
    threads: usize,
    group: usize,
) -> Result<(), DeviceError> {
    let mut recording = stream.begin()?;
    {
        let mut encoder = recording.compute_encoder()?;
        encoder.set_pipeline(pipeline);
        for (slot, binding) in bindings.iter().enumerate() {
            encoder.set_binding(slot, *binding)?;
        }
        encoder.dispatch_groups(threads.div_ceil(group), 1, group)?;
        encoder.end();
    }
    stream.submit(recording)
}

/// A count as the kernels read it; a count past a u32 is refused by name.
fn count_bytes(count: usize) -> Result<[u8; 4], DeviceError> {
    u32::try_from(count)
        .map(u32::to_ne_bytes)
        .map_err(|_| DeviceError::Range {
            what: "count",
            detail: format!("{count} does not fit the kernels' u32"),
        })
}

/// This backend's kernels, lent to the contract's clauses.
struct MetalHarness {
    device: MetalDevice,
    copy: Pipeline,
    spin: Pipeline,
    raise_one: Pipeline,
    raise_lanes: Pipeline,
    /// The flag [`Harness::long`]'s spin reads, which nothing sets: the spin
    /// runs to its end.
    never: MetalShared<u32>,
}

impl MetalHarness {
    fn new(device: MetalDevice) -> Result<Self, DeviceError> {
        Ok(Self {
            copy: device.pipeline("copy_f32")?,
            spin: device.pipeline("spin_until_set")?,
            raise_one: device.pipeline("raise_fault")?,
            raise_lanes: device.pipeline("raise_fault_lanes")?,
            never: device.shared::<u32>(Elems(1))?,
            device,
        })
    }
}

/// A host event a stream waits on: the hold opens when the host signals it.
struct MetalHold(HostEvent);

impl Hold for MetalHold {
    fn open(&self) -> Result<(), DeviceError> {
        self.0.signal();
        Ok(())
    }
}

impl Drop for MetalHold {
    fn drop(&mut self) {
        self.0.signal();
    }
}

impl Harness for MetalHarness {
    type Device = MetalDevice;
    type Hold = MetalHold;

    fn device(&self) -> &MetalDevice {
        &self.device
    }

    fn held_bytes(&self) -> Result<u64, DeviceError> {
        self.device.allocated_bytes()
    }

    fn copy_f32(
        &self,
        s: &MetalStream,
        from: Window<'_, f32, MetalDevice>,
        to: WindowMut<'_, f32, MetalDevice>,
    ) -> Result<(), DeviceError> {
        check_len("copy", from.len().get(), to.len())?;
        let n = from.len().get();
        let count = count_bytes(n)?;
        let bindings = [
            Binding::of_mut(&to)?,
            Binding::of(&from)?,
            Binding::bytes(&count)?,
        ];
        launch(s, &self.copy, &bindings, n, n.min(GROUP))
    }

    fn long(
        &self,
        s: &MetalStream,
        out: WindowMut<'_, u32, MetalDevice>,
    ) -> Result<(), DeviceError> {
        let flag = self.never.whole()?;
        let limit = SPIN.to_ne_bytes();
        let bindings = [
            Binding::of(&flag)?,
            Binding::of_mut(&out)?,
            Binding::bytes(&limit)?,
        ];
        launch(s, &self.spin, &bindings, 1, 1)
    }

    fn raise(&self, s: &MetalStream, raises: &[(u32, FaultSite)]) -> Result<(), DeviceError> {
        let word = self.device.fault_word().binding();
        if let [(layer, site)] = raises {
            let (layer, site) = (layer.to_ne_bytes(), (*site as u32).to_ne_bytes());
            let bindings = [word, Binding::bytes(&layer)?, Binding::bytes(&site)?];
            return launch(s, &self.raise_one, &bindings, 1, 1);
        }
        let layers: Vec<u8> = raises.iter().flat_map(|(l, _)| l.to_ne_bytes()).collect();
        let sites: Vec<u8> = raises
            .iter()
            .flat_map(|(_, s)| (*s as u32).to_ne_bytes())
            .collect();
        let count = count_bytes(raises.len())?;
        let bindings = [
            word,
            Binding::bytes(&layers)?,
            Binding::bytes(&sites)?,
            Binding::bytes(&count)?,
        ];
        launch(
            s,
            &self.raise_lanes,
            &bindings,
            raises.len(),
            raises.len().min(GROUP),
        )
    }

    fn fault_words(&self, s: &MetalStream) -> Result<Vec<u32>, DeviceError> {
        self.device.fault_word().words(s)
    }

    fn hold(&self, s: &MetalStream) -> Result<MetalHold, DeviceError> {
        let gate = HostEvent::new(&self.device)?;
        s.wait(gate.event())?;
        Ok(MetalHold(gate))
    }

    fn other_stream(&self) -> Result<MetalStream, DeviceError> {
        self.device.new_stream()
    }
}

fn encoder_refusals(device: &MetalDevice) -> Clause {
    let stream = device.stream();
    let pipeline = device.pipeline("copy_f32").map_err(why)?;
    let buffer = device.zeroed::<f32>(stream, Elems(4)).map_err(why)?;
    let window = buffer.window(Elems(0), Elems(4)).map_err(why)?;
    let binding = Binding::of(&window).map_err(why)?;

    expect_range(Binding::bytes(&[]), "binding", "0 bytes")?;
    expect_range(
        Binding::bytes(&[0u8; MAX_BYTES_BINDING + 1]),
        "binding",
        "4097 bytes",
    )?;
    Binding::bytes(&[0u8; MAX_BYTES_BINDING]).map_err(why)?;

    let mut recording = stream.begin().map_err(why)?;
    {
        let mut encoder = recording.compute_encoder().map_err(why)?;
        expect_range(encoder.dispatch_groups(1, 1, 4), "dispatch", "no pipeline")?;
        encoder.set_pipeline(&pipeline);
        expect_range(
            encoder.set_binding(MAX_BINDINGS, binding),
            "binding",
            "index 31",
        )?;
        encoder
            .set_binding(MAX_BINDINGS - 1, binding)
            .map_err(why)?;
        expect_range(encoder.dispatch_groups(0, 1, 4), "dispatch", "0 by 1")?;
        expect_range(encoder.dispatch_groups(1, 1, 0), "dispatch", "groups of 0")?;
        let too_many = pipeline.max_threads_per_group() + 1;
        expect_range(
            encoder.dispatch_groups(1, 1, too_many),
            "dispatch",
            "passes the pipeline's",
        )?;
        expect_range(encoder.dispatch_groups(1, 0, 4), "dispatch", "by 0")?;
        expect_range(
            encoder.dispatch_groups(usize::MAX, 1, 2),
            "dispatch",
            "wrap",
        )?;
        encoder.end();
    }
    stream.submit(recording).map_err(why)?;
    stream.sync().map_err(why)?;
    Ok(Verdict::Pass)
}

/// A refusal is a backend failure of `what` whose text holds `needle`.
fn expect_backend(result: Result<(), DeviceError>, what: &str, needle: &str) -> Result<(), String> {
    match result {
        Err(DeviceError::Backend { what: got, source }) if got == what => {
            let text = source.to_string();
            if text.contains(needle) {
                Ok(())
            } else {
                Err(format!("the {what} refusal does not name {needle}: {text}"))
            }
        }
        Err(other) => Err(format!("a backend failure of {what}, not {other}")),
        Ok(()) => Err(format!("a {what} wait that must time out returned")),
    }
}

/// Work held behind a host event: the bounded sync and the bounded event
/// wait give up by name, and the event is done only once the host signals it.
fn sync_bound_named(h: &MetalHarness) -> Clause {
    let device = &h.device;
    let stream = device.new_stream().map_err(why)?;
    let src = device.upload(&stream, &[1.0f32, 2.0]).map_err(why)?;
    let mut dst = device.zeroed::<f32>(&stream, Elems(2)).map_err(why)?;
    let gate = HostEvent::new(device).map_err(why)?;
    stream.wait(gate.event()).map_err(why)?;
    h.copy_f32(
        &stream,
        src.whole().map_err(why)?,
        dst.whole_mut().map_err(why)?,
    )
    .map_err(why)?;

    let held = expect_backend(
        stream.sync_for(Duration::from_millis(200)),
        "command buffer",
        "not done within 200 ms",
    )
    .and_then(|()| {
        if gate.event().done().map_err(why)? {
            return Err("a host event is done before it was signalled".to_string());
        }
        expect_backend(
            gate.event().sync_for(Duration::from_millis(50)),
            "event",
            "was not signalled within 50 ms",
        )
    });
    gate.signal();
    held?;
    if !gate.event().done().map_err(why)? {
        return Err("a signalled host event is not done".to_string());
    }
    stream.sync_for(Duration::from_secs(10)).map_err(why)?;
    Ok(Verdict::Pass)
}

/// Two threads wait on one held stream at once: neither's wait ends before
/// the host event opens, and both return `Ok` after it (the wait holds the
/// stream's lock, and a thread queued behind another thread's wait gives up
/// by name at its own bound).
fn sync_two_waiters(h: &MetalHarness) -> Clause {
    let device = &h.device;
    let stream = device.new_stream().map_err(why)?;
    let src = device.upload(&stream, &[1.0f32]).map_err(why)?;
    let mut dst = device.zeroed::<f32>(&stream, Elems(1)).map_err(why)?;
    let gate = HostEvent::new(device).map_err(why)?;
    stream.wait(gate.event()).map_err(why)?;
    h.copy_f32(
        &stream,
        src.whole().map_err(why)?,
        dst.whole_mut().map_err(why)?,
    )
    .map_err(why)?;

    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<()>();
    let (wait_tx, wait_rx) = std::sync::mpsc::channel::<(Result<(), DeviceError>, Instant)>();
    let waiter = &stream;
    let (opened, waits) = std::thread::scope(|scope| {
        for _ in 0..2 {
            let (ready_tx, wait_tx) = (ready_tx.clone(), wait_tx.clone());
            scope.spawn(move || {
                ready_tx.send(()).ok();
                let done = waiter.sync_for(Duration::from_secs(10));
                let _ = wait_tx.send((done, Instant::now()));
            });
        }
        drop((ready_tx, wait_tx));
        // Both waiters are in their wait; hold the work a moment longer.
        for _ in 0..2 {
            ready_rx.recv().ok();
        }
        std::thread::sleep(Duration::from_millis(300));
        let opened = Instant::now();
        gate.signal();
        let waits = (0..2)
            .map(|_| wait_rx.recv().ok())
            .collect::<Option<Vec<_>>>()
            .unwrap_or_default();
        (opened, waits)
    });
    if waits.len() != 2 {
        return Err("a waiter did not answer".to_string());
    }
    for (done, at) in waits {
        done.map_err(|e| format!("a waiter failed once the gate was open: {e}"))?;
        if at < opened {
            return Err("a wait ended Ok before the host event opened".to_string());
        }
    }
    stream.sync_for(Duration::from_secs(10)).map_err(why)?;
    Ok(Verdict::Pass)
}

/// The events one stream records rise in value, in the order recorded.
fn event_values_rise(device: &MetalDevice) -> Clause {
    let stream = device.stream();
    let values = [
        stream.record().map_err(why)?.value(),
        stream.record().map_err(why)?.value(),
        stream.record().map_err(why)?.value(),
    ];
    stream.sync().map_err(why)?;
    if values[0] >= values[1] || values[1] >= values[2] {
        return Err(format!("the events' values are {values:?}, not rising"));
    }
    Ok(Verdict::Pass)
}

/// A token recording of `stream`, its encoder ended, parked.
fn parked(stream: &MetalStream) -> Result<Parked, String> {
    let mut recording = stream.begin().map_err(why)?;
    recording.token_encoder().map_err(why)?.end();
    stream.park(recording).map_err(why)
}

fn parked_goes_stale(device: &MetalDevice) -> Clause {
    let stream = device.stream();
    let fresh = parked(stream)?;
    let back = stream
        .unpark(fresh)
        .ok_or("a parked recording came back stale with nothing committed since")?;
    let signalled = stream.submit_signalled(back).map_err(why)?;
    stream.settle(signalled).map_err(why)?;

    let stale = parked(stream)?;
    let mut between = stream.begin().map_err(why)?;
    between.token_encoder().map_err(why)?.end();
    let signalled = stream.submit_signalled(between).map_err(why)?;
    stream.settle(signalled).map_err(why)?;
    if stream.unpark(stale).is_some() {
        return Err(
            "a parked recording came back after a signalled buffer moved the boundary past its \
             wait"
                .to_string(),
        );
    }

    let foreign = parked(stream)?;
    let other = device.new_stream().map_err(why)?;
    if other.unpark(foreign).is_some() {
        return Err("another stream gave a parked recording back".to_string());
    }
    Ok(Verdict::Pass)
}

/// The inputs of the arithmetic kernel: products whose fused and rounded-apart
/// results differ.
const ARITH_A: [f32; 8] = [1.000_000_1, -3.5, 0.1, 16_777_217.0, 1e-20, 2.0, -0.3, 7.25];
const ARITH_B: [f32; 8] = [
    1.000_000_1,
    2.25,
    10.0,
    1.000_000_1,
    1e20,
    -0.5,
    0.3,
    1.0e-3,
];
const ARITH_C: [f32; 8] = [
    -1.000_000_2,
    8.0,
    -1.0,
    -16_777_218.0,
    -1.0,
    1.0,
    0.09,
    -0.007_25,
];

/// The whole of one of the arithmetic kernel's eight-element buffers.
fn whole_window(buffer: &MetalBuffer<f32>) -> Result<Window<'_, f32, MetalDevice>, String> {
    buffer.window(Elems(0), Elems(8)).map_err(why)
}

/// The arithmetic kernel's three outputs on `library`.
fn arith_outputs(device: &MetalDevice, library: &Library) -> Result<[[f32; 8]; 3], String> {
    let stream = device.stream();
    let pipeline = library.pipeline("arith_f32").map_err(why)?;
    let mut inputs = Vec::new();
    for values in [&ARITH_A, &ARITH_B, &ARITH_C] {
        inputs.push(device.upload(stream, values).map_err(why)?);
    }
    let mut outputs = Vec::new();
    for _ in 0..3 {
        outputs.push(device.zeroed::<f32>(stream, Elems(8)).map_err(why)?);
    }
    let windows = outputs
        .iter()
        .chain(inputs.iter())
        .map(whole_window)
        .collect::<Result<Vec<_>, _>>()?;
    let bindings = windows
        .iter()
        .map(|w| Binding::of(w).map_err(why))
        .collect::<Result<Vec<_>, _>>()?;
    let count = count_bytes(8).map_err(why)?;
    let mut all = bindings;
    all.push(Binding::bytes(&count).map_err(why)?);
    launch(stream, &pipeline, &all, 8, 8).map_err(why)?;
    let mut got = [[0.0f32; 8]; 3];
    for (out, slot) in outputs.iter().zip(got.iter_mut()) {
        device
            .read(stream, out.whole().map_err(why)?, slot)
            .map_err(why)?;
    }
    Ok(got)
}

fn arith_matches_host(device: &MetalDevice) -> Clause {
    let got = arith_outputs(device, device.library())?;
    for i in 0..8 {
        let (a, b, c) = (ARITH_A[i], ARITH_B[i], ARITH_C[i]);
        let fused = a.mul_add(b, c);
        let split = a * b + c;
        if got[0][i].to_bits() != fused.to_bits() {
            return Err(format!("fma({a}, {b}, {c}) is {}, not {fused}", got[0][i]));
        }
        if got[1][i].to_bits() != split.to_bits() {
            return Err(format!("{a} * {b} + {c} is {}, not {split}", got[1][i]));
        }
    }
    Ok(Verdict::Pass)
}

/// The metallib's output and the source's output are the same bits.
#[cfg(metal_lib = "metallib")]
fn metallib_equals_source(device: &MetalDevice) -> Clause {
    let metallib = Library::from_metallib(device).map_err(why)?;
    let source = Library::from_source(device).map_err(why)?;
    let from_metallib = arith_outputs(device, &metallib)?;
    let from_source = arith_outputs(device, &source)?;
    for (name, (a, b)) in ["fma", "split", "library"]
        .iter()
        .zip(from_metallib.iter().zip(from_source.iter()))
    {
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            if x.to_bits() != y.to_bits() {
                return Err(format!(
                    "{name} element {i}: the metallib read {x:?}, the source read {y:?}"
                ));
            }
        }
    }
    Ok(Verdict::Pass)
}

/// The metallib path does not exist on this machine: the clause cannot run.
#[cfg(not(metal_lib = "metallib"))]
fn metallib_equals_source(_device: &MetalDevice) -> Clause {
    Ok(Verdict::Skip(
        "the Metal Toolchain was not installed when this crate was built: no metallib exists; \
         install it and rebuild"
            .to_string(),
    ))
}

/// What the census says of this backend's devices beyond the contract: each
/// is integrated, and this one's total is the driver's working set and no
/// more than the machine's memory.
fn census_driver(device: &MetalDevice) -> Clause {
    let all = MetalCensus::census().map_err(why)?;
    if let Some(info) = all.iter().find(|info| !info.integrated) {
        return Err(format!(
            "{}: a Metal device that is not integrated",
            info.name
        ));
    }
    let own = device.info().map_err(why)?;
    let total = device.metal().recommendedMaxWorkingSetSize();
    if own.total_bytes != total {
        return Err(format!(
            "the census says {} bytes, the driver says {total}",
            own.total_bytes
        ));
    }
    let memory = NSProcessInfo::processInfo().physicalMemory();
    if own.total_bytes > memory {
        return Err(format!(
            "the device's {} bytes pass the machine's {memory}",
            own.total_bytes
        ));
    }
    Ok(Verdict::Pass)
}

/// The spin, then the flag, on one concurrent encoder. `barrier` encodes
/// `memory_barrier` between them. The spin's cap is [`SPIN`]: both dispatches
/// are already in this command buffer, so a GPU that overlaps them stores the
/// flag while the spin is still loading, and the kernel writes and returns
/// when the cap is reached either way.
fn spin_then_flag(device: &MetalDevice, barrier: bool) -> Result<u32, String> {
    let stream = device.stream();
    let spin = device.pipeline("spin_until_set").map_err(why)?;
    let set = device.pipeline("set_flag").map_err(why)?;
    let mut flag = device.shared::<u32>(Elems(1)).map_err(why)?;
    let mut out = device.shared::<u32>(Elems(1)).map_err(why)?;
    let limit = SPIN.to_ne_bytes();
    let mut recording = stream.begin().map_err(why)?;
    {
        let mut encoder = recording.compute_encoder_concurrent().map_err(why)?;
        encoder.set_pipeline(&spin);
        {
            let flag_w = flag.whole().map_err(why)?;
            let out_w = out.whole_mut().map_err(why)?;
            encoder
                .set_binding(0, Binding::of(&flag_w).map_err(why)?)
                .map_err(why)?;
            encoder
                .set_binding(1, Binding::of_mut(&out_w).map_err(why)?)
                .map_err(why)?;
            encoder
                .set_binding(2, Binding::bytes(&limit).map_err(why)?)
                .map_err(why)?;
            encoder.dispatch_groups(1, 1, 1).map_err(why)?;
        }
        if barrier {
            encoder.memory_barrier();
        }
        encoder.set_pipeline(&set);
        {
            let flag_w = flag.whole_mut().map_err(why)?;
            encoder
                .set_binding(0, Binding::of_mut(&flag_w).map_err(why)?)
                .map_err(why)?;
            encoder.dispatch_groups(1, 1, 1).map_err(why)?;
        }
        encoder.end();
    }
    stream.submit(recording).map_err(why)?;
    stream.sync_for(BARRIER_WAIT).map_err(why)?;
    match out.load(0, Ordering::Acquire).map_err(why)? {
        n @ (SPIN_RAN_OUT | SPIN_SAW_FLAG) => Ok(n),
        other => Err(format!("the spin wrote {other}")),
    }
}

/// The barrier-free arm first. No overlap, and the clause skips. Overlap, and
/// the arm with a memory barrier between the dispatches must write
/// [`SPIN_RAN_OUT`]. `drop_barrier` is `--drop-barrier`: that arm encodes no
/// barrier, so the overlap the free arm saw makes the clause fail.
fn concurrent_barrier(device: &MetalDevice, drop_barrier: bool) -> Clause {
    match spin_then_flag(device, false)? {
        SPIN_RAN_OUT => Ok(Verdict::Skip(
            "no overlap on this GPU: the spin ran out before the flag was set, so a barrier cannot be shown to matter"
                .to_string(),
        )),
        SPIN_SAW_FLAG => match spin_then_flag(device, !drop_barrier)? {
            SPIN_RAN_OUT => Ok(Verdict::Pass),
            SPIN_SAW_FLAG => Err(
                "the ordered arm wrote 2: the flag was set during the spin, so the dispatches overlapped"
                    .to_string(),
            ),
            other => Err(format!("the ordered arm wrote {other}")),
        },
        other => Err(format!("the barrier-free arm wrote {other}")),
    }
}

fn wide_heads_compile(device: &MetalDevice) -> Clause {
    let mut key = QWEN3MOE_A3B.key;
    key.attn.row = gqa_row(128, 8).ok_or("the shape table has no head-128 row of group 8")?;
    key.attn.n_head = 64;
    key.attn.n_kv = 8;
    let inst = open_key(key).map_err(|e| e.to_string())?;
    let (source, _) = sources_for(inst).map_err(why)?;
    let library = Library::from_source_text(device, &source).map_err(why)?;
    for entry in [
        "gqa256_flash_f16",
        "gqa256_flash_q8",
        "gqa256_flash_f16_rows",
    ] {
        library.pipeline(entry).map_err(why)?;
    }
    Ok(Verdict::Pass)
}

fn macos_floor() -> Clause {
    for (major, minor, patch) in [(26, 1, 9), (15, 0, 0), (0, 0, 0)] {
        match require_macos(major, minor, patch) {
            Err(DeviceError::Unsupported { what, detail }) => {
                if what != "macos" {
                    return Err(format!("the refusal named {what}, not macos"));
                }
                if !detail.contains("26.2.0") {
                    return Err(format!("the refusal did not name the floor: {detail}"));
                }
            }
            Err(other) => return Err(format!("an unsupported refusal, not {other}")),
            Ok(()) => return Err(format!("macOS {major}.{minor}.{patch} was accepted")),
        }
    }
    for (major, minor, patch) in [(26, 2, 0), (26, 3, 1), (27, 0, 0)] {
        require_macos(major, minor, patch).map_err(why)?;
    }
    Ok(Verdict::Pass)
}

fn help() {
    println!(
        "\
gate_metal_device: the Metal device gate.

The clauses run in order. The exit code is 0 when every clause passed, 1 when
one failed, 78 when this machine cannot run the backend.

  --help            print this and exit 0
  --drop-barrier    concurrent_barrier encodes no memory barrier on the ordered
                    arm. The gate uses this to show the clause goes red when
                    the barrier is gone. Nothing else reads it."
    );
}

pub fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        help();
        return ExitCode::SUCCESS;
    }
    let mut drop_barrier = false;
    for arg in &args {
        match arg.as_str() {
            "--drop-barrier" => drop_barrier = true,
            other => {
                println!("gate_metal_device: unknown argument {other}; --help lists the arguments");
                return ExitCode::from(64);
            }
        }
    }

    let device = match MetalDevice::open(&HOST) {
        Ok(device) => device,
        Err(e @ DeviceError::Unsupported { .. }) => {
            println!("clause open: REFUSED ({e})");
            return ExitCode::from(78);
        }
        Err(e) => {
            println!("clause open: FAIL ({e})");
            return ExitCode::FAILURE;
        }
    };
    let opened = device.info().and_then(|info| {
        let path = device.library().path().name();
        MetalHarness::new(device).map(|h| (h, info, path))
    });
    let (harness, info, path) = match opened {
        Ok(opened) => opened,
        Err(e) => {
            println!("clause open: FAIL ({e})");
            return ExitCode::FAILURE;
        }
    };
    println!(
        "metal-lib: path={path} device={} backend={}",
        info.name,
        <MetalDevice as Device>::BACKEND
    );

    let mut report = Report::new();
    report.clause("open", Ok(Verdict::Pass));
    conformance::run::<_, MetalCensus>(&harness, &mut report);
    let device = &harness.device;
    report.clause("encoder_refusals", encoder_refusals(device));
    report.clause("sync_bound_named", sync_bound_named(&harness));
    report.clause("sync_two_waiters", sync_two_waiters(&harness));
    report.clause("event_values_rise", event_values_rise(device));
    report.clause("parked_goes_stale", parked_goes_stale(device));
    report.clause("arith_matches_host", arith_matches_host(device));
    report.clause("metallib_equals_source", metallib_equals_source(device));
    report.clause("census_driver", census_driver(device));
    report.clause("macos_floor", macos_floor());
    report.clause("wide_heads_compile", wide_heads_compile(device));
    report.clause(
        "concurrent_barrier",
        concurrent_barrier(device, drop_barrier),
    );
    report.finish("gate_metal_device")
}
