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
//!    or too many, a dispatch of no threads, no pipeline or a too large group
//!    are refused by name.
//! 4. **sync_bound_named**: a sync and an event wait that outlast their bound
//!    are refused by name, and a host event is done only once signalled.
//! 5. **sync_two_waiters**: two threads waiting on one held stream at once,
//!    neither's wait ends before the host event opens, and both return `Ok`
//!    after it.
//! 6. **event_values_rise**: the events one stream records rise in value.
//! 7. **queue_overlap**: whether two command buffers of one queue overlap on
//!    this machine; it reports the answer and passes either way.
//! 8. **arith_matches_host**: a fused multiply-add and a multiply and add
//!    rounded apart are the host's bits.
//! 9. **metallib_equals_source**: the metallib's output and the source's
//!    output are the same bits, when both paths exist.
//! 10. **census_driver**: the census names an integrated device whose total is
//!     the driver's working set and no more than the machine's memory.
//! 11. **macos_floor**: a macOS below 26.2 is refused by name.
//! 12. **concurrent_independent**: one concurrent encoder runs two copies
//!     over four distinct buffers, with no barrier between them; each output
//!     equals the bytes uploaded to its source. No pipeline, no threads and a
//!     value of no bytes are refused on that encoder, and a refusal encodes
//!     nothing.
//! 13. **concurrent_barrier**: one concurrent encoder copies a pattern into a
//!     middle buffer, encodes a memory barrier, then copies the middle buffer
//!     to the output; the output equals the pattern.

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
use bloomery_metal::encode::{Binding, ComputeEncoder, MAX_BINDINGS, MAX_BYTES_BINDING};
use bloomery_metal::library::{Library, Pipeline};
use bloomery_metal::stream::{HostEvent, MetalStream};

/// How many times the long kernels read their flag: long enough that the host
/// is back from a call before the kernel is over, and short of the GPU
/// watchdog.
const SPIN: u32 = 1 << 22;

/// The largest group a gate dispatch asks for.
const GROUP: usize = 64;

/// How many f32s the concurrent clauses copy.
const CONCURRENT_LEN: usize = 1 << 20;

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
        encoder.dispatch(threads, group)?;
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
        expect_range(encoder.dispatch(4, 4), "dispatch", "no pipeline")?;
        encoder.set_pipeline(&pipeline);
        expect_range(
            encoder.set_binding(MAX_BINDINGS, binding),
            "binding",
            "index 31",
        )?;
        encoder
            .set_binding(MAX_BINDINGS - 1, binding)
            .map_err(why)?;
        expect_range(encoder.dispatch(0, 4), "dispatch", "0 threads")?;
        expect_range(encoder.dispatch(4, 0), "dispatch", "groups of 0")?;
        let too_many = pipeline.max_threads_per_group() + 1;
        expect_range(
            encoder.dispatch(too_many, too_many),
            "dispatch",
            "passes the pipeline's",
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

/// Two command buffers of one queue, the second ending the first's spin: it
/// ends early only when the two ran at once.
fn queue_overlap(h: &MetalHarness) -> Clause {
    let device = &h.device;
    let stream = device.stream();
    let mut flag = device.shared::<u32>(Elems(1)).map_err(why)?;
    let out = device.shared::<u32>(Elems(1)).map_err(why)?;
    {
        let (flag, out) = (
            flag.whole().map_err(why)?,
            out.window(Elems(0), Elems(1)).map_err(why)?,
        );
        let limit = SPIN.to_ne_bytes();
        let bindings = [
            Binding::of(&flag).map_err(why)?,
            Binding::of(&out).map_err(why)?,
            Binding::bytes(&limit).map_err(why)?,
        ];
        launch(stream, &h.spin, &bindings, 1, 1).map_err(why)?;
    }
    let set = device.pipeline("set_flag").map_err(why)?;
    let window = flag.whole_mut().map_err(why)?;
    launch(
        stream,
        &set,
        &[Binding::of_mut(&window).map_err(why)?],
        1,
        1,
    )
    .map_err(why)?;
    stream.sync().map_err(why)?;
    let overlap = match out.load(0, Ordering::Acquire).map_err(why)? {
        2 => "yes",
        1 => "no",
        other => return Err(format!("the spin wrote {other}")),
    };
    Ok(Verdict::Noted(format!("overlap={overlap}")))
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

/// One `copy_f32` dispatch: `src` into `dst`, `threads` of them, groups of
/// [`GROUP`].
fn encode_copy(
    encoder: &mut ComputeEncoder<'_>,
    dst: &WindowMut<'_, f32, MetalDevice>,
    src: &Window<'_, f32, MetalDevice>,
    count: &[u8],
    threads: usize,
) -> Result<(), DeviceError> {
    encoder.set_binding(0, Binding::of_mut(dst)?)?;
    encoder.set_binding(1, Binding::of(src)?)?;
    encoder.set_binding(2, Binding::bytes(count)?)?;
    encoder.dispatch(threads, GROUP)
}

/// `got` against `want`, bit for bit. The first difference names how many differ.
fn same_bits(got: &[f32], want: &[f32]) -> Result<(), String> {
    if got.len() != want.len() {
        return Err(format!(
            "{} values came back, not {}",
            got.len(),
            want.len()
        ));
    }
    let mut differ = 0usize;
    let mut first = None;
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        if g.to_bits() != w.to_bits() {
            if first.is_none() {
                first = Some((i, *g, *w));
            }
            differ += 1;
        }
    }
    match first {
        None => Ok(()),
        Some((i, g, w)) => Err(format!(
            "element {i} is {g}, not {w}; {differ} of {} differ",
            got.len()
        )),
    }
}

/// The values a concurrent clause uploads: index `i` as f32, or that value
/// negated.
fn ramp(n: usize, neg: bool) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = i as f32;
            if neg { -x - 1.0 } else { x }
        })
        .collect()
}

/// The buffer's f32s, once `stream`'s work is done.
fn read_f32(
    device: &MetalDevice,
    stream: &MetalStream,
    buffer: &MetalBuffer<f32>,
) -> Result<Vec<f32>, String> {
    let mut got = vec![0.0f32; buffer.len().get()];
    device
        .read(stream, buffer.whole().map_err(why)?, &mut got)
        .map_err(why)?;
    Ok(got)
}

/// Two independent copies on one concurrent encoder, and the refusals that
/// encode nothing.
fn concurrent_independent(device: &MetalDevice) -> Clause {
    let n = CONCURRENT_LEN;
    let stream = device.stream();
    let pipeline = device.pipeline("copy_f32").map_err(why)?;
    let count = count_bytes(n).map_err(why)?;
    let pat_a = ramp(n, false);
    let pat_b = ramp(n, true);
    let src_a = device.upload(stream, &pat_a).map_err(why)?;
    let src_b = device.upload(stream, &pat_b).map_err(why)?;
    let mut dst_a = device.zeroed::<f32>(stream, Elems(n)).map_err(why)?;
    let mut dst_b = device.zeroed::<f32>(stream, Elems(n)).map_err(why)?;

    let mut recording = stream.begin().map_err(why)?;
    {
        let mut encoder = recording.compute_encoder_concurrent().map_err(why)?;
        expect_range(Binding::bytes(&[]), "binding", "0 bytes")?;
        expect_range(encoder.dispatch(n, GROUP), "dispatch", "no pipeline")?;
        encoder.set_pipeline(&pipeline);
        expect_range(encoder.dispatch(0, GROUP), "dispatch", "0 threads")?;
        {
            let dst = dst_a.whole_mut().map_err(why)?;
            let src = src_a.whole().map_err(why)?;
            encode_copy(&mut encoder, &dst, &src, &count, n).map_err(why)?;
        }
        {
            let dst = dst_b.whole_mut().map_err(why)?;
            let src = src_b.whole().map_err(why)?;
            encode_copy(&mut encoder, &dst, &src, &count, n).map_err(why)?;
        }
        encoder.end();
    }
    stream.submit(recording).map_err(why)?;
    same_bits(&read_f32(device, stream, &dst_a)?, &pat_a)?;
    same_bits(&read_f32(device, stream, &dst_b)?, &pat_b)?;
    Ok(Verdict::Pass)
}

/// A dependent pair on one concurrent encoder: the pattern into a middle
/// buffer, a barrier, then the middle buffer to the output.
fn concurrent_barrier(device: &MetalDevice) -> Clause {
    let n = CONCURRENT_LEN;
    let stream = device.stream();
    let pipeline = device.pipeline("copy_f32").map_err(why)?;
    let count = count_bytes(n).map_err(why)?;
    let pattern = ramp(n, false);
    let src = device.upload(stream, &pattern).map_err(why)?;
    let mut middle = device.zeroed::<f32>(stream, Elems(n)).map_err(why)?;
    let mut out = device.zeroed::<f32>(stream, Elems(n)).map_err(why)?;

    let mut recording = stream.begin().map_err(why)?;
    {
        let mut encoder = recording.compute_encoder_concurrent().map_err(why)?;
        encoder.set_pipeline(&pipeline);
        {
            let dst = middle.whole_mut().map_err(why)?;
            let from = src.whole().map_err(why)?;
            encode_copy(&mut encoder, &dst, &from, &count, n).map_err(why)?;
        }
        encoder.memory_barrier();
        {
            let dst = out.whole_mut().map_err(why)?;
            let from = middle.whole().map_err(why)?;
            encode_copy(&mut encoder, &dst, &from, &count, n).map_err(why)?;
        }
        encoder.end();
    }
    stream.submit(recording).map_err(why)?;
    same_bits(&read_f32(device, stream, &out)?, &pattern)?;
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

pub fn main() -> ExitCode {
    let device = match MetalDevice::open() {
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
    report.clause("queue_overlap", queue_overlap(&harness));
    report.clause("arith_matches_host", arith_matches_host(device));
    report.clause("metallib_equals_source", metallib_equals_source(device));
    report.clause("census_driver", census_driver(device));
    report.clause("macos_floor", macos_floor());
    report.clause("concurrent_independent", concurrent_independent(device));
    report.clause("concurrent_barrier", concurrent_barrier(device));
    report.finish("gate_metal_device")
}
