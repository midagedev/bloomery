//! The Metal device gate's clauses (macOS only).
//!
//! One clause per line, `PASS`, `SKIP` or `FAIL`, and the run's exit code is
//! 1 when any clause failed, 78 when this machine cannot run the backend (a
//! macOS below the floor, no Metal device), 0 otherwise. The clauses:
//!
//! 1. **open**: the device opens, and the library path it came from is named.
//! 2. **zeroed_upload_roundtrip**: a zeroed buffer reads back all zero, an
//!    uploaded buffer reads back bit for bit, and a read of another length is
//!    refused by name.
//! 3. **window_range**: a window inside its buffer is taken, a window one
//!    element past the end, past the end by a wrapping sum, or a split or
//!    sub-window outside its window is refused by name.
//! 4. **window_binding**: a kernel bound to windows (`at` not zero) reads and
//!    writes exactly the window's elements.
//! 5. **encoder_refusals**: a binding slot past the last, a value of no bytes
//!    or too many, a dispatch of no threads, no pipeline or a too large group
//!    are refused by name.
//! 6. **shared_seen_by_kernel**: what the host stored in a shared buffer is
//!    what a kernel read out of it, and what a kernel wrote is what the host
//!    loads and borrows.
//! 7. **submit_registers**: a read waits for work that was only submitted.
//! 8. **event_wait_holds**: work enqueued after a `wait` on a host event does
//!    not run until the host signals it, and a wait that outlasts its bound is
//!    refused by name.
//! 9. **event_record_orders**: an event recorded after a long kernel fires
//!    only once the kernel's result is visible.
//! 10. **queue_overlap**: whether two command buffers of one queue overlap on
//!     this machine; it reports the answer and passes either way.
//! 11. **fault**: the device's word is, word for word, the contract's host
//!     model, for an ordered sequence of raises and for raises that race; a
//!     layer past the last writes no mask; a clear is clean.
//! 12. **fault_read_waits**: a fault raised by submitted work is seen by the
//!     read that follows it.
//! 13. **arith_matches_host**: a fused multiply-add and a multiply and add
//!     rounded apart are the host's bits.
//! 14. **metallib_equals_source**: the metallib's output and the source's
//!     output are the same bits, when both paths exist.
//! 15. **census**: the devices' ordinals and ids are distinct, this device is
//!     one of them, the free bytes are below the total and the total below the
//!     machine's memory, and an allocation lowers the free bytes.
//! 16. **macos_floor**: a macOS below 26.2 is refused by name.

use std::process::ExitCode;
use std::time::Duration;

use bloomery_device::fault::{self, FAULT_NONE, FAULT_WORDS, FaultSite, LAYER_HEAD, LAYER_NONE};
use bloomery_device::{
    Buffer, Bytes, Census, Device, DeviceError, Elems, Event, Shared, Stream, Window,
};
use objc2_foundation::NSProcessInfo;
use objc2_metal::MTLDevice;

use bloomery_metal::buffer::{MetalBuffer, MetalShared};
use bloomery_metal::census::MetalCensus;
use bloomery_metal::device::{MetalDevice, require_macos};
use bloomery_metal::encode::{Binding, MAX_BINDINGS, MAX_BYTES_BINDING};
use bloomery_metal::library::{Library, Pipeline};
use bloomery_metal::stream::{HostEvent, MetalStream};

/// How a clause that did not fail ended.
enum Verdict {
    Pass,
    /// Passed, with a fact worth printing.
    Noted(String),
    /// Could not run here, with the reason.
    Skip(String),
}

/// A clause's result: a verdict, or the reason it failed.
type Clause = Result<Verdict, String>;

/// The gate's own report: one line per clause.
struct Report {
    failures: Vec<&'static str>,
}

impl Report {
    fn clause(&mut self, name: &'static str, result: Clause) {
        match result {
            Ok(Verdict::Pass) => println!("clause {name}: PASS"),
            Ok(Verdict::Noted(note)) => println!("clause {name}: PASS ({note})"),
            Ok(Verdict::Skip(why)) => println!("clause {name}: SKIP ({why})"),
            Err(why) => {
                println!("clause {name}: FAIL ({why})");
                self.failures.push(name);
            }
        }
    }
}

/// The error of a clause, as its text.
fn why(e: DeviceError) -> String {
    e.to_string()
}

/// How many times the long kernels read their flag: long enough that the host
/// is back from a call before the kernel is over, and short of the GPU
/// watchdog.
const SPIN: u32 = 1 << 22;

/// Encode one dispatch of `pipeline` over `bindings` (slot = position) and
/// submit it; it does not wait.
fn launch(
    stream: &MetalStream,
    pipeline: &Pipeline,
    bindings: &[Binding<'_>],
    threads: usize,
    group: usize,
) -> Result<(), String> {
    let recording = stream.begin().map_err(why)?;
    {
        let mut encoder = recording.compute_encoder().map_err(why)?;
        encoder.set_pipeline(pipeline);
        for (slot, binding) in bindings.iter().enumerate() {
            encoder.set_binding(slot, *binding).map_err(why)?;
        }
        encoder.dispatch(threads, group).map_err(why)?;
        encoder.end();
    }
    stream.submit(recording).map_err(why)
}

/// A count as the kernels read it.
fn count_bytes(count: usize) -> [u8; 4] {
    u32::try_from(count)
        .expect("a gate's counts fit a u32")
        .to_ne_bytes()
}

/// Copy `count` f32s from `src` to `dst` with the `copy_f32` kernel of
/// `pipeline`; it does not wait.
fn copy_f32(
    stream: &MetalStream,
    pipeline: &Pipeline,
    dst: Binding<'_>,
    src: Binding<'_>,
    count: usize,
) -> Result<(), String> {
    let count = count_bytes(count);
    let bindings = [dst, src, Binding::bytes(&count).map_err(why)?];
    launch(stream, pipeline, &bindings, 4, 4)
}

/// Raise `site` in `layer` through the device's fault word; it does not wait.
fn raise_fault(
    device: &MetalDevice,
    stream: &MetalStream,
    layer: u32,
    site: FaultSite,
) -> Result<(), String> {
    let pipeline = device.pipeline("raise_fault").map_err(why)?;
    let layer = layer.to_ne_bytes();
    let site = (site as u32).to_ne_bytes();
    let bindings = [
        device.fault_word().binding(),
        Binding::bytes(&layer).map_err(why)?,
        Binding::bytes(&site).map_err(why)?,
    ];
    launch(stream, &pipeline, &bindings, 1, 1)
}

/// A refusal is a range refusal of `what` whose text holds `needle`.
fn expect_range<T>(result: Result<T, DeviceError>, what: &str, needle: &str) -> Result<(), String> {
    match result {
        Err(DeviceError::Range { what: got, detail }) => {
            if got != what {
                return Err(format!("a range refusal named {got}, not {what}"));
            }
            if !detail.contains(needle) {
                return Err(format!(
                    "the {what} refusal does not name {needle}: {detail}"
                ));
            }
            Ok(())
        }
        Err(other) => Err(format!("a range refusal of {what}, not {other}")),
        Ok(_) => Err(format!("a {what} that must be refused was accepted")),
    }
}

fn zeroed_upload_roundtrip(device: &MetalDevice, stream: &MetalStream) -> Clause {
    let zeroed = device.zeroed::<f32>(stream, Elems(16)).map_err(why)?;
    let mut got = [1.0f32; 16];
    zeroed.read(stream, &mut got).map_err(why)?;
    if got.iter().any(|v| v.to_bits() != 0.0f32.to_bits()) {
        return Err(format!("a zeroed buffer read back {got:?}"));
    }

    let src = [
        1.0f32,
        -2.5,
        0.0,
        3.25,
        -0.0,
        1e-30,
        1e30,
        f32::MIN_POSITIVE,
        f32::MAX,
        0.5,
        -0.5,
        2.0,
        -3.0,
        0.125,
        1024.0,
        -4096.0,
    ];
    let uploaded = device.upload(stream, &src).map_err(why)?;
    let mut back = [0.0f32; 16];
    uploaded.read(stream, &mut back).map_err(why)?;
    for (i, (got, want)) in back.iter().zip(src.iter()).enumerate() {
        if got.to_bits() != want.to_bits() {
            return Err(format!(
                "element {i} read back {got:?} ({}), not {want:?} ({})",
                got.to_bits(),
                want.to_bits()
            ));
        }
    }
    expect_range(
        uploaded.read(stream, &mut [0.0f32; 15]),
        "read",
        "16 elements",
    )?;
    expect_range(
        device.zeroed::<f32>(stream, Elems(0)),
        "buffer",
        "no elements",
    )?;
    Ok(Verdict::Pass)
}

fn window_range(device: &MetalDevice, stream: &MetalStream) -> Clause {
    let mut buffer = device.zeroed::<f32>(stream, Elems(8)).map_err(why)?;
    let mut shared = device.shared::<f32>(Elems(8)).map_err(why)?;
    for (at, len) in [(0, 8), (7, 1), (8, 0), (2, 3)] {
        buffer.window(Elems(at), Elems(len)).map_err(why)?;
        buffer.window_mut(Elems(at), Elems(len)).map_err(why)?;
        shared.window(Elems(at), Elems(len)).map_err(why)?;
    }
    for (at, len, needle) in [
        (7, 2, "7..9"),
        (0, 9, "0..9"),
        (9, 0, "9..9"),
        (usize::MAX, 2, "wrap"),
        (3, usize::MAX, "wrap"),
    ] {
        expect_range(buffer.window(Elems(at), Elems(len)), "window", needle)?;
        expect_range(buffer.window_mut(Elems(at), Elems(len)), "window", needle)?;
        expect_range(shared.window(Elems(at), Elems(len)), "window", needle)?;
        expect_range(shared.window_mut(Elems(at), Elems(len)), "window", needle)?;
    }

    let window = buffer.window(Elems(2), Elems(4)).map_err(why)?;
    let sub = window.sub(Elems(1), Elems(3)).map_err(why)?;
    let range = sub.byte_range().map_err(why)?;
    if range != (Bytes(12)..Bytes(24)) {
        return Err(format!("the sub-window's bytes are {range:?}, not 12..24"));
    }
    expect_range(
        window.sub(Elems(1), Elems(4)),
        "window",
        "window's 4 elements",
    )?;

    let window = buffer.window_mut(Elems(2), Elems(4)).map_err(why)?;
    let (left, right) = window.split_at(Elems(4)).map_err(why)?;
    if left.len() != Elems(4) || !right.is_empty() {
        return Err("a split at the end is not the whole and nothing".to_string());
    }
    let window = buffer.window_mut(Elems(2), Elems(4)).map_err(why)?;
    expect_range(window.split_at(Elems(5)), "window", "split at element 5")?;
    Ok(Verdict::Pass)
}

fn window_binding(device: &MetalDevice, stream: &MetalStream) -> Clause {
    let pipeline = device.pipeline("copy_f32").map_err(why)?;
    let mut src = device.shared::<f32>(Elems(8)).map_err(why)?;
    for i in 0..8u8 {
        src.store(usize::from(i), f32::from(i) * 10.0)
            .map_err(why)?;
    }
    let mut dst = device.zeroed::<f32>(stream, Elems(8)).map_err(why)?;
    {
        let from = src.window(Elems(2), Elems(2)).map_err(why)?;
        let to = dst.window_mut(Elems(5), Elems(2)).map_err(why)?;
        let (from, to) = (
            Binding::of(&from).map_err(why)?,
            Binding::of_mut(&to).map_err(why)?,
        );
        copy_f32(stream, &pipeline, to, from, 2)?;
    }
    let mut got = [9.0f32; 8];
    dst.read(stream, &mut got).map_err(why)?;
    let want = [0.0f32, 0.0, 0.0, 0.0, 0.0, 20.0, 30.0, 0.0];
    if got
        .iter()
        .zip(&want)
        .any(|(g, w)| g.to_bits() != w.to_bits())
    {
        return Err(format!("windows at 2 and 5 copied {got:?}, not {want:?}"));
    }
    Ok(Verdict::Pass)
}

fn encoder_refusals(device: &MetalDevice, stream: &MetalStream) -> Clause {
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

    let recording = stream.begin().map_err(why)?;
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

fn shared_seen_by_kernel(device: &MetalDevice, stream: &MetalStream) -> Clause {
    let pipeline = device.pipeline("copy_f32").map_err(why)?;
    let mut src = device.shared::<f32>(Elems(4)).map_err(why)?;
    let mut dst = device.shared::<f32>(Elems(4)).map_err(why)?;
    for (i, v) in [3.0f32, 42.5, -1.0, 8.0].iter().enumerate() {
        src.store(i, *v).map_err(why)?;
    }
    if src.load(1).map_err(why)?.to_bits() != 42.5f32.to_bits() {
        return Err("the shared buffer did not read back what it stored".to_string());
    }
    {
        let (from, to) = (
            src.window(Elems(0), Elems(4)).map_err(why)?,
            dst.window_mut(Elems(0), Elems(4)).map_err(why)?,
        );
        let (from, to) = (
            Binding::of(&from).map_err(why)?,
            Binding::of_mut(&to).map_err(why)?,
        );
        copy_f32(stream, &pipeline, to, from, 4)?;
    }
    stream.sync().map_err(why)?;
    // The kernel's writes, by the host: through a volatile load, and through
    // the borrowed pages.
    if dst.load(1).map_err(why)?.to_bits() != 42.5f32.to_bits() {
        return Err("the host's load did not see the kernel's write".to_string());
    }
    // SAFETY: the stream was synced, and nothing else is enqueued, so no
    // device work in flight writes `dst`.
    let seen = unsafe { dst.host() };
    if seen[3].to_bits() != 8.0f32.to_bits() || seen[2].to_bits() != (-1.0f32).to_bits() {
        return Err(format!("the host's pages hold {seen:?}"));
    }
    Ok(Verdict::Pass)
}

/// A long kernel is in flight; the read that follows it must wait for it.
fn submit_registers(device: &MetalDevice, stream: &MetalStream) -> Clause {
    let (flag, out) = flag_and_out(device)?;
    spin(device, stream, &flag, &out)?;
    let mut got = [0u32; 1];
    out.read(stream, &mut got).map_err(why)?;
    if got[0] != 1 {
        return Err(format!(
            "a read after a submitted kernel saw {}: it did not wait for it",
            got[0]
        ));
    }
    Ok(Verdict::Pass)
}

/// The two shared words the spin kernels use: the flag and the result.
fn flag_and_out(device: &MetalDevice) -> Result<(MetalShared<u32>, MetalShared<u32>), String> {
    Ok((
        device.shared::<u32>(Elems(1)).map_err(why)?,
        device.shared::<u32>(Elems(1)).map_err(why)?,
    ))
}

/// Submit the long kernel: it reads `flag` up to [`SPIN`] times and writes
/// `out`.
fn spin(
    device: &MetalDevice,
    stream: &MetalStream,
    flag: &MetalShared<u32>,
    out: &MetalShared<u32>,
) -> Result<(), String> {
    let pipeline = device.pipeline("spin_until_set").map_err(why)?;
    let (flag, out) = (
        flag.window(Elems(0), Elems(1)).map_err(why)?,
        out.window(Elems(0), Elems(1)).map_err(why)?,
    );
    let limit = SPIN.to_ne_bytes();
    let bindings = [
        Binding::of(&flag).map_err(why)?,
        Binding::of(&out).map_err(why)?,
        Binding::bytes(&limit).map_err(why)?,
    ];
    launch(stream, &pipeline, &bindings, 1, 1)
}

fn event_wait_holds(device: &MetalDevice, stream: &MetalStream) -> Clause {
    let pipeline = device.pipeline("copy_f32").map_err(why)?;
    let mut src = device.shared::<f32>(Elems(4)).map_err(why)?;
    let mut dst = device.shared::<f32>(Elems(4)).map_err(why)?;
    for i in 0..4u8 {
        src.store(usize::from(i), f32::from(i) + 1.0).map_err(why)?;
    }
    let gate = HostEvent::new(device).map_err(why)?;
    stream.wait(gate.event()).map_err(why)?;
    {
        let (from, to) = (
            src.window(Elems(0), Elems(4)).map_err(why)?,
            dst.window_mut(Elems(0), Elems(4)).map_err(why)?,
        );
        let (from, to) = (
            Binding::of(&from).map_err(why)?,
            Binding::of_mut(&to).map_err(why)?,
        );
        copy_f32(stream, &pipeline, to, from, 4)?;
    }

    // The host has not signalled: the copy must not have run, and a bounded
    // sync says so by name instead of waiting for ever.
    match stream.sync_for(Duration::from_millis(200)) {
        Err(DeviceError::Backend {
            what: "command buffer",
            detail,
        }) => {
            if !detail.contains("not done within 200 ms") {
                return Err(format!("the timeout does not name its bound: {detail}"));
            }
        }
        Err(other) => return Err(format!("a timeout by name, not {other}")),
        Ok(()) => return Err("work after a wait ran before the host signalled".to_string()),
    }
    if dst.load(0).map_err(why)?.to_bits() != 0.0f32.to_bits() {
        return Err("the copy wrote before the host signalled".to_string());
    }

    if gate.event().done().map_err(why)? {
        return Err("a host event is done before it was signalled".to_string());
    }
    expect_backend(gate.event().sync_for(Duration::from_millis(50)), "event")?;
    gate.signal();
    if !gate.event().done().map_err(why)? {
        return Err("a signalled host event is not done".to_string());
    }
    stream.sync_for(Duration::from_secs(10)).map_err(why)?;
    for i in 0..4u8 {
        let got = dst.load(usize::from(i)).map_err(why)?;
        if got.to_bits() != (f32::from(i) + 1.0).to_bits() {
            return Err(format!("element {i} is {got} after the signal"));
        }
    }
    Ok(Verdict::Pass)
}

/// A refusal is a backend failure of `what`.
fn expect_backend(result: Result<(), DeviceError>, what: &str) -> Result<(), String> {
    match result {
        Err(DeviceError::Backend { what: got, .. }) if got == what => Ok(()),
        Err(other) => Err(format!("a backend failure of {what}, not {other}")),
        Ok(()) => Err(format!("a {what} wait that must time out returned")),
    }
}

/// An event recorded after a long kernel fires only once the kernel is done:
/// the event alone is waited for, and the kernel's result is read without a
/// stream wait.
fn event_record_orders(device: &MetalDevice, stream: &MetalStream) -> Clause {
    let (flag, out) = flag_and_out(device)?;
    spin(device, stream, &flag, &out)?;
    let recorded = stream.record().map_err(why)?;
    recorded.sync().map_err(why)?;
    let seen = out.load(0).map_err(why)?;
    stream.sync().map_err(why)?;
    if seen != 1 {
        return Err(format!(
            "an event recorded after a kernel fired while its result was {seen}"
        ));
    }
    let first = stream.record().map_err(why)?;
    let second = stream.record().map_err(why)?;
    if recorded.value() >= first.value() || first.value() >= second.value() {
        return Err(format!(
            "the events' values are {}, {} and {}, not rising",
            recorded.value(),
            first.value(),
            second.value()
        ));
    }
    Ok(Verdict::Pass)
}

/// Two command buffers of one queue, the second ending the first's spin: it
/// ends early only when the two ran at once.
fn queue_overlap(device: &MetalDevice, stream: &MetalStream) -> Clause {
    let (flag, out) = flag_and_out(device)?;
    spin(device, stream, &flag, &out)?;
    let set = device.pipeline("set_flag").map_err(why)?;
    let window = flag.window(Elems(0), Elems(1)).map_err(why)?;
    launch(stream, &set, &[Binding::of(&window).map_err(why)?], 1, 1)?;
    stream.sync().map_err(why)?;
    let overlap = match out.load(0).map_err(why)? {
        2 => "yes",
        1 => "no",
        other => return Err(format!("the spin wrote {other}")),
    };
    Ok(Verdict::Noted(format!("overlap={overlap}")))
}

/// The host model of the word after `raises`, applied in order.
fn modelled(raises: &[(u32, FaultSite)]) -> Result<Vec<u32>, String> {
    let mut words = vec![0u32; FAULT_WORDS];
    words[0] = FAULT_NONE;
    for (layer, site) in raises {
        fault::raise(&mut words, *layer, *site).map_err(why)?;
    }
    Ok(words)
}

/// The device's word equals the host model's, word for word.
fn same_word(device: &MetalDevice, stream: &MetalStream, model: &[u32], when: &str) -> Clause {
    let words = device.fault_word().words(stream).map_err(why)?;
    if let Some(at) = words.iter().zip(model).position(|(d, m)| d != m) {
        return Err(format!(
            "{when}: word {at} is {:#x} on the device, {:#x} in the model",
            words[at], model[at]
        ));
    }
    let want = fault::Fault::decode(model).map_err(why)?;
    let got = device.fault().read(stream).map_err(why)?;
    if got != want {
        return Err(format!(
            "{when}: the device reads {got:?}, the model {want:?}"
        ));
    }
    Ok(Verdict::Pass)
}

fn fault_clause(device: &MetalDevice, stream: &MetalStream) -> Clause {
    let word = device.fault();
    word.clear(stream).map_err(why)?;
    same_word(device, stream, &modelled(&[])?, "a cleared word")?;

    // A later layer first, then an earlier layer's highest site, then its
    // lowest: the word keeps the earliest layer and the lowest site only
    // through the minimum, and every site through the mask.
    let sequence = [
        (7, FaultSite::Logit),
        (3, FaultSite::TokenId),
        (3, FaultSite::Router),
        (3, FaultSite::QuantColumn),
        (LAYER_NONE, FaultSite::Ple),
        (LAYER_HEAD, FaultSite::Logit),
    ];
    for (layer, site) in sequence {
        raise_fault(device, stream, layer, site)?;
    }
    same_word(device, stream, &modelled(&sequence)?, "an ordered sequence")?;
    let raised = word
        .read(stream)
        .map_err(why)?
        .ok_or("the word holds no fault")?;
    if raised.layer != 3 || raised.code != FaultSite::QuantColumn as u32 {
        return Err(format!(
            "the word names layer {} and code {}, not layer 3 and code {}",
            raised.layer,
            raised.code,
            FaultSite::QuantColumn as u32
        ));
    }

    // A layer past the last: the word names it, and no mask is written.
    word.clear(stream).map_err(why)?;
    let past = [(LAYER_NONE + 1, FaultSite::Router)];
    raise_fault(device, stream, past[0].0, past[0].1)?;
    same_word(device, stream, &modelled(&past)?, "a layer past the last")?;

    // Raises that race: every lane raises its own layer and site at once.
    for round in 0..8usize {
        word.clear(stream).map_err(why)?;
        raise_lanes(device, stream, round)?;
        let lanes = lane_raises(round);
        same_word(device, stream, &modelled(&lanes)?, "raises that race")?;
    }

    word.clear(stream).map_err(why)?;
    same_word(device, stream, &modelled(&[])?, "a word cleared again")
}

/// How many raises race in one dispatch.
const LANES: usize = 64;

/// The raise of every lane of `round`: layers 2..7, every site in turn.
fn lane_raises(round: usize) -> Vec<(u32, FaultSite)> {
    (0..LANES)
        .map(|lane| {
            let layer = u32::try_from((lane * 7 + round) % 5 + 2).expect("a small layer");
            let site = FaultSite::ALL[(lane * 3 + round) % FaultSite::ALL.len()];
            (layer, site)
        })
        .collect()
}

fn raise_lanes(device: &MetalDevice, stream: &MetalStream, round: usize) -> Result<(), String> {
    let pipeline = device.pipeline("raise_fault_lanes").map_err(why)?;
    let lanes = lane_raises(round);
    let layers: Vec<u8> = lanes.iter().flat_map(|(l, _)| l.to_ne_bytes()).collect();
    let sites: Vec<u8> = lanes
        .iter()
        .flat_map(|(_, s)| (*s as u32).to_ne_bytes())
        .collect();
    let count = count_bytes(lanes.len());
    let bindings = [
        device.fault_word().binding(),
        Binding::bytes(&layers).map_err(why)?,
        Binding::bytes(&sites).map_err(why)?,
        Binding::bytes(&count).map_err(why)?,
    ];
    launch(stream, &pipeline, &bindings, LANES, LANES)
}

/// A fault raised by submitted work behind a long kernel is seen by the read
/// that follows it, without any sync of the gate's own.
fn fault_read_waits(device: &MetalDevice, stream: &MetalStream) -> Clause {
    let word = device.fault();
    word.clear(stream).map_err(why)?;
    let (flag, out) = flag_and_out(device)?;
    spin(device, stream, &flag, &out)?;
    raise_fault(device, stream, 5, FaultSite::Logit)?;
    match word.read(stream).map_err(why)? {
        Some(fault) if fault.layer == 5 && fault.code == FaultSite::Logit as u32 => {}
        Some(other) => return Err(format!("the read saw {other:?}, not layer 5's logit")),
        None => return Err("a read after submitted work saw a clean word".to_string()),
    }
    word.clear(stream).map_err(why)?;
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
fn arith_outputs(
    device: &MetalDevice,
    stream: &MetalStream,
    library: &Library,
) -> Result<[[f32; 8]; 3], String> {
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
    let count = count_bytes(8);
    let mut all = bindings;
    all.push(Binding::bytes(&count).map_err(why)?);
    launch(stream, &pipeline, &all, 8, 8)?;
    let mut got = [[0.0f32; 8]; 3];
    for (out, slot) in outputs.iter().zip(got.iter_mut()) {
        out.read(stream, slot).map_err(why)?;
    }
    Ok(got)
}

fn arith_matches_host(device: &MetalDevice, stream: &MetalStream) -> Clause {
    let got = arith_outputs(device, stream, device.library())?;
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
fn metallib_equals_source(device: &MetalDevice, stream: &MetalStream) -> Clause {
    let metallib = Library::from_metallib(device).map_err(why)?;
    let source = Library::from_source(device).map_err(why)?;
    let from_metallib = arith_outputs(device, stream, &metallib)?;
    let from_source = arith_outputs(device, stream, &source)?;
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
fn metallib_equals_source(_device: &MetalDevice, _stream: &MetalStream) -> Clause {
    Ok(Verdict::Skip(
        "the Metal Toolchain was not installed when this crate was built: no metallib exists; \
         install it and rebuild"
            .to_string(),
    ))
}

fn census(device: &MetalDevice, stream: &MetalStream) -> Clause {
    let all = MetalCensus::census().map_err(why)?;
    for (at, info) in all.iter().enumerate() {
        if usize::try_from(info.ordinal) != Ok(at) {
            return Err(format!("device {at} is named ordinal {}", info.ordinal));
        }
        if !info.integrated || info.free_bytes > info.total_bytes {
            return Err(format!(
                "{}: an integrated device with free <= total",
                info.name
            ));
        }
        if all.iter().skip(at + 1).any(|other| other.uuid == info.uuid) {
            return Err(format!("{}: its id is another device's", info.name));
        }
    }
    let own = device.info();
    let listed = all
        .get(usize::try_from(own.ordinal).map_err(|e| e.to_string())?)
        .ok_or("the opened device's ordinal is not in the census")?;
    if (listed.name.as_str(), listed.uuid, listed.total_bytes)
        != (own.name.as_str(), own.uuid, own.total_bytes)
    {
        return Err("the opened device is not the census's device at its ordinal".to_string());
    }
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

    let before = free_bytes(own.ordinal)?;
    let big = device.zeroed::<u8>(stream, Elems(64 << 20)).map_err(why)?;
    let after = free_bytes(own.ordinal)?;
    drop(big);
    if after >= before {
        return Err(format!(
            "a 64 MiB buffer left the free bytes at {after}, from {before}"
        ));
    }
    Ok(Verdict::Pass)
}

/// The free bytes the census names for the device at `ordinal` now.
fn free_bytes(ordinal: u32) -> Result<u64, String> {
    let all = MetalCensus::census().map_err(why)?;
    all.iter()
        .find(|d| d.ordinal == ordinal)
        .map(|d| d.free_bytes)
        .ok_or_else(|| format!("no census row at ordinal {ordinal}"))
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
    let stream = match device.stream() {
        Ok(stream) => stream,
        Err(e) => {
            println!("clause open: FAIL ({e})");
            return ExitCode::FAILURE;
        }
    };
    println!(
        "metal-lib: path={} device={} backend={}",
        device.library().path().name(),
        device.info().name,
        <MetalDevice as Device>::BACKEND
    );

    let mut report = Report {
        failures: Vec::new(),
    };
    report.clause("open", Ok(Verdict::Pass));
    report.clause(
        "zeroed_upload_roundtrip",
        zeroed_upload_roundtrip(&device, &stream),
    );
    report.clause("window_range", window_range(&device, &stream));
    report.clause("window_binding", window_binding(&device, &stream));
    report.clause("encoder_refusals", encoder_refusals(&device, &stream));
    report.clause(
        "shared_seen_by_kernel",
        shared_seen_by_kernel(&device, &stream),
    );
    report.clause("submit_registers", submit_registers(&device, &stream));
    report.clause("event_wait_holds", event_wait_holds(&device, &stream));
    report.clause("event_record_orders", event_record_orders(&device, &stream));
    report.clause("queue_overlap", queue_overlap(&device, &stream));
    report.clause("fault", fault_clause(&device, &stream));
    report.clause("fault_read_waits", fault_read_waits(&device, &stream));
    report.clause("arith_matches_host", arith_matches_host(&device, &stream));
    report.clause(
        "metallib_equals_source",
        metallib_equals_source(&device, &stream),
    );
    report.clause("census", census(&device, &stream));
    report.clause("macos_floor", macos_floor());

    if report.failures.is_empty() {
        println!("gate_metal_device: PASS");
        return ExitCode::SUCCESS;
    }
    println!("gate_metal_device: FAIL ({})", report.failures.join(", "));
    ExitCode::FAILURE
}
