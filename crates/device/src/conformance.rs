//! The contract's clauses, run by every backend's gate over its own device:
//! one set, so a backend that passes them keeps what every other backend
//! keeps. A clause prints one line, `clause <name>: PASS`, `SKIP` or `FAIL`
//! with its reason; the gate's exit code is its own.
//!
//! What a clause needs that the contract does not give — a kernel that copies
//! between two windows, one that runs long, a raise of the fault word, a gate
//! the host holds on a stream, a second stream — the gate lends through a
//! [`Harness`] over its backend's own kernels. The clauses:
//!
//! 1. **zeroed_upload_roundtrip**: a zeroed buffer reads back all zero and an
//!    uploaded one bit for bit, for every element size; a read of another
//!    length and a buffer of no elements are refused by name.
//! 2. **window_range**: a window inside its buffer is taken, a window one
//!    element past the end, past the end by a wrapping sum, or a split or
//!    sub-window outside its window is refused by name, for plain and shared
//!    buffers; a cast and a span go through the backend's own windows.
//! 3. **window_binding**: a kernel bound to windows at an offset, and to a
//!    window cast from a byte arena, reads and writes exactly its elements.
//! 4. **copies**: a write into the middle of a buffer, a copy between two
//!    windows and a read of the middle land exactly there, in stream order
//!    with a kernel; a copy of another length is refused by name.
//! 5. **shared_seen_by_kernel**: what the host stored in a shared buffer is
//!    what a kernel read out of it, and what a kernel wrote is what the host
//!    loads and borrows; an index past the end is refused by name.
//! 6. **submit_registers**: a read waits for a long kernel enqueued before it.
//! 7. **event_wait_holds**: work on a second stream that waits for an event
//!    of a held stream does not run until the hold opens, and then does.
//! 8. **event_record_orders**: an event recorded after a long kernel fires
//!    only once the kernel's result is visible.
//! 9. **fault**: the device's word is, word for word, the contract's host
//!    model, for an ordered sequence of raises and for raises that race; a
//!    layer past the last writes no mask; a clear is clean.
//! 10. **fault_read_waits**: a fault raised behind a long kernel is seen by
//!     the read that follows it.
//! 11. **census**: the devices' ordinals and ids are distinct, this device is
//!     the census's device at its ordinal, the free bytes are below the total,
//!     and an allocation lowers the free bytes the device reports. It reads the
//!     free bytes twice, so it holds only on a device no other process frees
//!     memory on meanwhile.

use core::sync::atomic::Ordering;
use std::process::ExitCode;
use std::time::Duration;

use crate::fault::{self, FAULT_NONE, FAULT_WORDS, Fault, FaultSite, LAYER_HEAD, LAYER_NONE};
use crate::{
    Buffer, Bytes, Census, Device, DeviceError, DeviceStream, Elem, Elems, Event, Shared, Stream,
    Window, WindowMut,
};

/// How a clause that did not fail ended.
pub enum Verdict {
    /// Passed.
    Pass,
    /// Passed, with a fact worth printing.
    Noted(String),
    /// Could not run here, with the reason.
    Skip(String),
}

/// A clause's result: a verdict, or the reason it failed.
pub type Clause = Result<Verdict, String>;

/// A gate's report: one line per clause, and the names of those that failed.
#[derive(Default)]
pub struct Report {
    failures: Vec<&'static str>,
}

impl Report {
    /// A report of no clauses yet.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Print clause `name`'s line and keep its failure.
    pub fn clause(&mut self, name: &'static str, result: Clause) {
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

    /// The gate's last line, `<gate>: PASS` or `<gate>: FAIL (<clauses>)`, and
    /// its exit code: 0 when every clause passed, 1 otherwise.
    #[must_use]
    pub fn finish(self, gate: &str) -> ExitCode {
        if self.failures.is_empty() {
            println!("{gate}: PASS");
            return ExitCode::SUCCESS;
        }
        println!("{gate}: FAIL ({})", self.failures.join(", "));
        ExitCode::FAILURE
    }
}

/// The error of a clause, as its text.
#[must_use]
pub fn why(e: DeviceError) -> String {
    e.to_string()
}

/// A refusal is a range refusal of `what` whose text holds `needle`.
pub fn expect_range<T>(
    result: Result<T, DeviceError>,
    what: &str,
    needle: &str,
) -> Result<(), String> {
    match result {
        Err(DeviceError::Range { what: got, detail }) => {
            if got != what {
                return Err(format!("a range refusal named {got}, not {what}: {detail}"));
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

/// What a backend's gate lends the clauses: its device and the test kernels
/// the contract has no call for.
pub trait Harness {
    /// The device the clauses run on.
    type Device: Device;
    /// A gate the host holds on a stream ([`Harness::hold`]).
    type Hold: Hold;

    /// The device.
    fn device(&self) -> &Self::Device;

    /// Enqueue on `s` a kernel bound to the two windows that copies `from`'s
    /// f32s into `to` (of the same length).
    fn copy_f32(
        &self,
        s: &DeviceStream<Self::Device>,
        from: Window<'_, f32, Self::Device>,
        to: WindowMut<'_, f32, Self::Device>,
    ) -> Result<(), DeviceError>;

    /// Enqueue on `s` a kernel that runs long — far longer than the host
    /// takes to come back from a call — and then writes 1 to `out`'s one
    /// element.
    fn long(
        &self,
        s: &DeviceStream<Self::Device>,
        out: WindowMut<'_, u32, Self::Device>,
    ) -> Result<(), DeviceError>;

    /// Enqueue on `s` one launch in which every `(layer, site)` of `raises`
    /// is raised by its own thread, at once, through the device's fault word.
    /// A layer past [`LAYER_NONE`] names the word and writes no mask, as the
    /// host model ([`fault::raise`]) does.
    fn raise(
        &self,
        s: &DeviceStream<Self::Device>,
        raises: &[(u32, FaultSite)],
    ) -> Result<(), DeviceError>;

    /// Every word of the device's fault word, once `s`'s work so far is done.
    fn fault_words(&self, s: &DeviceStream<Self::Device>) -> Result<Vec<u32>, DeviceError>;

    /// A gate the host holds on `s`: the work enqueued on `s` after this call
    /// does not start until the hold opens.
    fn hold(&self, s: &DeviceStream<Self::Device>) -> Result<Self::Hold, DeviceError>;

    /// A stream of the device other than its own.
    fn other_stream(&self) -> Result<DeviceStream<Self::Device>, DeviceError>;
}

/// A gate the host holds on a stream. Dropping it opens it too, so a clause
/// that fails while it holds leaves no stream waiting for ever.
pub trait Hold {
    /// The held work may start.
    fn open(&self) -> Result<(), DeviceError>;
}

/// Run every clause on `h`'s device, `C` its census, into `report`.
pub fn run<H: Harness, C: Census>(h: &H, report: &mut Report) {
    report.clause("zeroed_upload_roundtrip", zeroed_upload_roundtrip(h));
    report.clause("window_range", window_range(h));
    report.clause("window_binding", window_binding(h));
    report.clause("copies", copies(h));
    report.clause("shared_seen_by_kernel", shared_seen_by_kernel(h));
    report.clause("submit_registers", submit_registers(h));
    report.clause("event_wait_holds", event_wait_holds(h));
    report.clause("event_record_orders", event_record_orders(h));
    report.clause("fault", fault_clause(h));
    report.clause("fault_read_waits", fault_read_waits(h));
    report.clause("census", census::<H, C>(h));
}

/// `got` against `want`, element by element as `bits` reads them.
fn same<T: Elem, B: PartialEq + core::fmt::Debug>(
    what: &str,
    got: &[T],
    want: &[T],
    bits: impl Fn(T) -> B,
) -> Result<(), String> {
    if got.len() != want.len() {
        return Err(format!(
            "{what}: {} elements, not {}",
            got.len(),
            want.len()
        ));
    }
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        if bits(*g) != bits(*w) {
            return Err(format!(
                "{what}: element {i} is {g:?} ({:?}), not {w:?} ({:?})",
                bits(*g),
                bits(*w)
            ));
        }
    }
    Ok(())
}

/// An upload of `values` reads back as them, and a zeroed buffer of their
/// length as zeros.
fn roundtrip<D: Device, T: Elem, B: PartialEq + core::fmt::Debug>(
    d: &D,
    values: &[T],
    zero: T,
    bits: impl Fn(T) -> B + Copy,
) -> Result<(), String> {
    let s = d.stream();
    let what = core::any::type_name::<T>();
    let uploaded = d.upload(s, values).map_err(why)?;
    let mut back = vec![zero; values.len()];
    d.read(s, uploaded.whole().map_err(why)?, &mut back)
        .map_err(why)?;
    same(what, &back, values, bits)?;
    let zeroed = d.zeroed::<T>(s, Elems(values.len())).map_err(why)?;
    // Dirty first, so a read that wrote nothing shows.
    let mut got = values.to_vec();
    d.read(s, zeroed.whole().map_err(why)?, &mut got)
        .map_err(why)?;
    same(what, &got, &vec![zero; values.len()], bits)
}

fn zeroed_upload_roundtrip<H: Harness>(h: &H) -> Clause {
    let d = h.device();
    let s = d.stream();
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
    roundtrip(d, &src, 0.0f32, f32::to_bits)?;
    roundtrip(d, &[0u8, 1, 0x7F, 0x80, 0xFF, 3, 5], 0, |v| v)?;
    roundtrip(d, &[0u16, 0xFFFF, 0x3C00, 1, 0x8000], 0, |v| v)?;
    roundtrip(d, &[i32::MIN, -1, 0, i32::MAX, 7], 0, |v| v)?;
    roundtrip(d, &[0u32, u32::MAX, 0xDEAD_BEEF], 0, |v| v)?;
    roundtrip(
        d,
        &[0u64, u64::MAX, 1 << 40, 0x0123_4567_89AB_CDEF],
        0,
        |v| v,
    )?;
    roundtrip(
        d,
        &[-0.0f64, 1e300, f64::MIN_POSITIVE, core::f64::consts::PI],
        0.0,
        f64::to_bits,
    )?;

    let uploaded = d.upload(s, &src).map_err(why)?;
    let whole = uploaded.whole().map_err(why)?;
    expect_range(d.read(s, whole, &mut [0.0f32; 15]), "read", "16 elements")?;
    expect_range(d.read(s, whole, &mut [0.0f32; 17]), "read", "16 elements")?;
    expect_range(d.zeroed::<f32>(s, Elems(0)), "buffer", "no elements")?;
    expect_range(d.upload::<u64>(s, &[]), "buffer", "no elements")?;
    expect_range(d.shared::<u8>(Elems(0)), "buffer", "no elements")?;
    Ok(Verdict::Pass)
}

fn window_range<H: Harness>(h: &H) -> Clause {
    let d = h.device();
    let s = d.stream();
    let mut buffer = d.zeroed::<f32>(s, Elems(8)).map_err(why)?;
    let mut shared = d.shared::<f32>(Elems(8)).map_err(why)?;
    for (at, len) in [(0, 8), (7, 1), (8, 0), (2, 3)] {
        buffer.window(Elems(at), Elems(len)).map_err(why)?;
        buffer.window_mut(Elems(at), Elems(len)).map_err(why)?;
        shared.window(Elems(at), Elems(len)).map_err(why)?;
        shared.window_mut(Elems(at), Elems(len)).map_err(why)?;
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
    expect_range(
        window.split_at(Elems(5)),
        "window",
        "elements 0..5 pass the window's 4 elements",
    )?;

    // A cast and a span through this backend's windows.
    let arena = d.zeroed::<u8>(s, Elems(32)).map_err(why)?;
    let cast = arena
        .window(Elems(8), Elems(16))
        .and_then(Window::cast::<f32>)
        .map_err(why)?;
    if (cast.len(), cast.byte_range().map_err(why)?) != (Elems(4), Bytes(8)..Bytes(24)) {
        return Err("bytes 8..24 as f32s are not four elements at byte 8".to_string());
    }
    expect_range(
        arena
            .window(Elems(2), Elems(8))
            .and_then(Window::cast::<f32>),
        "cast",
        "byte offset 2",
    )?;
    expect_range(
        arena
            .window(Elems(8), Elems(6))
            .and_then(Window::cast::<u64>),
        "cast",
        "byte length 6",
    )?;
    let span = buffer.window(Elems(5), Elems(3)).map_err(why)?.span();
    let again = span.window(&buffer).map_err(why)?;
    if again.byte_range().map_err(why)? != (Bytes(20)..Bytes(32)) {
        return Err("a span taken again is not the window it came from".to_string());
    }
    let short = d.zeroed::<f32>(s, Elems(6)).map_err(why)?;
    expect_range(span.window(&short), "window", "5..8")?;
    Ok(Verdict::Pass)
}

fn window_binding<H: Harness>(h: &H) -> Clause {
    let d = h.device();
    let s = d.stream();
    let values: Vec<f32> = (0..8u8).map(|i| f32::from(i) * 10.0).collect();
    let src = d.upload(s, &values).map_err(why)?;
    let mut dst = d.zeroed::<f32>(s, Elems(8)).map_err(why)?;
    h.copy_f32(
        s,
        src.window(Elems(2), Elems(2)).map_err(why)?,
        dst.window_mut(Elems(5), Elems(2)).map_err(why)?,
    )
    .map_err(why)?;
    let mut got = [9.0f32; 8];
    d.read(s, dst.whole().map_err(why)?, &mut got)
        .map_err(why)?;
    same(
        "windows at 2 and 5",
        &got,
        &[0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 30.0, 0.0],
        f32::to_bits,
    )?;

    // A window cast from a byte arena binds at its own bytes.
    let mut arena = d.zeroed::<u8>(s, Elems(64)).map_err(why)?;
    let to = arena
        .window_mut(Elems(16), Elems(8))
        .and_then(WindowMut::cast::<f32>)
        .map_err(why)?;
    h.copy_f32(s, src.window(Elems(6), Elems(2)).map_err(why)?, to)
        .map_err(why)?;
    let mut floats = [0.0f32; 2];
    let from = arena
        .window(Elems(16), Elems(8))
        .and_then(Window::cast::<f32>)
        .map_err(why)?;
    d.read(s, from, &mut floats).map_err(why)?;
    same(
        "an f32 window of a byte arena",
        &floats,
        &[60.0, 70.0],
        f32::to_bits,
    )?;
    let mut bytes = [1u8; 64];
    d.read(s, arena.whole().map_err(why)?, &mut bytes)
        .map_err(why)?;
    if bytes[..16].iter().chain(&bytes[24..]).any(|b| *b != 0) {
        return Err(format!(
            "the arena's bytes outside 16..24 are not zero: {bytes:?}"
        ));
    }
    Ok(Verdict::Pass)
}

fn copies<H: Harness>(h: &H) -> Clause {
    let d = h.device();
    let s = d.stream();
    let a = d
        .upload(s, &[0u32, 10, 20, 30, 40, 50, 60, 70])
        .map_err(why)?;
    let mut b = d.zeroed::<u32>(s, Elems(8)).map_err(why)?;
    d.write(
        s,
        b.window_mut(Elems(2), Elems(3)).map_err(why)?,
        &[7, 8, 9],
    )
    .map_err(why)?;
    d.copy(
        s,
        a.window(Elems(5), Elems(2)).map_err(why)?,
        b.window_mut(Elems(6), Elems(2)).map_err(why)?,
    )
    .map_err(why)?;
    let mut got = [1u32; 8];
    d.read(s, b.whole().map_err(why)?, &mut got).map_err(why)?;
    same(
        "a write at 2 and a copy at 6",
        &got,
        &[0, 0, 7, 8, 9, 0, 50, 60],
        |v| v,
    )?;
    let mut mid = [0u32; 2];
    d.read(s, a.window(Elems(3), Elems(2)).map_err(why)?, &mut mid)
        .map_err(why)?;
    same("a read of elements 3..5", &mid, &[30, 40], |v| v)?;

    let wide = d.upload(s, &[1.0f64, -0.5, 1e-300]).map_err(why)?;
    let mut wide_to = d.zeroed::<f64>(s, Elems(3)).map_err(why)?;
    d.copy(
        s,
        wide.whole().map_err(why)?,
        wide_to.whole_mut().map_err(why)?,
    )
    .map_err(why)?;
    let mut back = [0.0f64; 3];
    d.read(s, wide_to.whole().map_err(why)?, &mut back)
        .map_err(why)?;
    same("an f64 copy", &back, &[1.0, -0.5, 1e-300], f64::to_bits)?;

    // A write, then a kernel that reads it, then a read of the kernel's
    // output: one stream's order.
    let mut staged = d.zeroed::<f32>(s, Elems(4)).map_err(why)?;
    let mut out = d.zeroed::<f32>(s, Elems(4)).map_err(why)?;
    d.write(s, staged.whole_mut().map_err(why)?, &[1.5, -2.0, 4.25, 8.0])
        .map_err(why)?;
    h.copy_f32(
        s,
        staged.whole().map_err(why)?,
        out.whole_mut().map_err(why)?,
    )
    .map_err(why)?;
    let mut seen = [0.0f32; 4];
    d.read(s, out.whole().map_err(why)?, &mut seen)
        .map_err(why)?;
    same(
        "a kernel after a write",
        &seen,
        &[1.5, -2.0, 4.25, 8.0],
        f32::to_bits,
    )?;

    expect_range(
        d.write(s, b.window_mut(Elems(0), Elems(2)).map_err(why)?, &[1]),
        "write",
        "window's 2 elements",
    )?;
    expect_range(
        d.copy(
            s,
            a.window(Elems(0), Elems(3)).map_err(why)?,
            b.window_mut(Elems(0), Elems(2)).map_err(why)?,
        ),
        "copy",
        "window's 2 elements",
    )?;
    expect_range(
        d.read(s, a.window(Elems(0), Elems(2)).map_err(why)?, &mut [0; 3]),
        "read",
        "window's 2 elements",
    )?;
    Ok(Verdict::Pass)
}

fn shared_seen_by_kernel<H: Harness>(h: &H) -> Clause {
    let d = h.device();
    let s = d.stream();
    let src = d.shared::<f32>(Elems(4)).map_err(why)?;
    let mut dst = d.shared::<f32>(Elems(4)).map_err(why)?;
    for (i, v) in [3.0f32, 42.5, -1.0].iter().enumerate() {
        src.store(i, *v, Ordering::Relaxed).map_err(why)?;
    }
    src.store(3, 8.0, Ordering::Release).map_err(why)?;
    if src.load(1, Ordering::Acquire).map_err(why)?.to_bits() != 42.5f32.to_bits() {
        return Err("the shared buffer did not load back what it stored".to_string());
    }
    h.copy_f32(s, src.whole().map_err(why)?, dst.whole_mut().map_err(why)?)
        .map_err(why)?;
    s.sync().map_err(why)?;
    // The kernel's writes, by the host: through an atomic load, and through
    // the borrowed pages.
    if dst.load(1, Ordering::Acquire).map_err(why)?.to_bits() != 42.5f32.to_bits() {
        return Err("the host's load did not see the kernel's write".to_string());
    }
    // SAFETY: the stream was synced and nothing else is enqueued or stored,
    // so no device work in flight and no host store writes `dst`.
    let seen = unsafe { dst.host() };
    same(
        "the host's pages",
        seen,
        &[3.0, 42.5, -1.0, 8.0],
        f32::to_bits,
    )?;
    expect_range(dst.load(4, Ordering::Relaxed), "load", "element 4")?;
    expect_range(src.store(4, 1.0, Ordering::Relaxed), "store", "element 4")?;
    Ok(Verdict::Pass)
}

/// A long kernel is in flight; the read that follows it must wait for it.
fn submit_registers<H: Harness>(h: &H) -> Clause {
    let d = h.device();
    let s = d.stream();
    let mut out = d.zeroed::<u32>(s, Elems(1)).map_err(why)?;
    h.long(s, out.whole_mut().map_err(why)?).map_err(why)?;
    let mut got = [0u32; 1];
    d.read(s, out.whole().map_err(why)?, &mut got)
        .map_err(why)?;
    if got[0] != 1 {
        return Err(format!(
            "a read after a long kernel saw {}: it did not wait for it",
            got[0]
        ));
    }
    Ok(Verdict::Pass)
}

/// How long a clause gives held work to show it ran when it must not have: a
/// backend that runs held work later than this is not seen, so the wait bounds
/// the defect and does not prove its absence.
const HELD: Duration = Duration::from_millis(100);

fn event_wait_holds<H: Harness>(h: &H) -> Clause {
    let d = h.device();
    let s = d.stream();
    let src = d.upload(s, &[1.0f32, 2.0, 3.0, 4.0]).map_err(why)?;
    let mut dst = d.shared::<f32>(Elems(4)).map_err(why)?;
    s.sync().map_err(why)?;
    let other = h.other_stream().map_err(why)?;

    let hold = h.hold(s).map_err(why)?;
    let held = s.record().map_err(why)?;
    other.wait(&held).map_err(why)?;
    h.copy_f32(
        &other,
        src.whole().map_err(why)?,
        dst.whole_mut().map_err(why)?,
    )
    .map_err(why)?;
    let copied = other.record().map_err(why)?;

    std::thread::sleep(HELD);
    if held.done().map_err(why)? {
        return Err("an event behind a hold fired before the hold opened".to_string());
    }
    if copied.done().map_err(why)? {
        return Err("work after a wait ran before the event it waits for fired".to_string());
    }
    if dst.load(0, Ordering::Acquire).map_err(why)?.to_bits() != 0.0f32.to_bits() {
        return Err("the copy wrote before the event it waits for fired".to_string());
    }

    hold.open().map_err(why)?;
    copied.sync().map_err(why)?;
    if !held.done().map_err(why)? {
        return Err("the held event is not done after the work behind it".to_string());
    }
    // SAFETY: the copy's event has fired and nothing else is enqueued on
    // `dst`, so no device work writes it while the slice lives.
    let seen = unsafe { dst.host() };
    same(
        "the copy after the hold",
        seen,
        &[1.0, 2.0, 3.0, 4.0],
        f32::to_bits,
    )?;
    other.sync().map_err(why)?;
    s.sync().map_err(why)?;
    Ok(Verdict::Pass)
}

/// An event recorded after a long kernel fires only once the kernel is done:
/// the event alone is waited for, and the kernel's result is loaded without a
/// stream wait.
fn event_record_orders<H: Harness>(h: &H) -> Clause {
    let d = h.device();
    let s = d.stream();
    let mut out = d.shared::<u32>(Elems(1)).map_err(why)?;
    h.long(s, out.whole_mut().map_err(why)?).map_err(why)?;
    let recorded = s.record().map_err(why)?;
    recorded.sync().map_err(why)?;
    let seen = out.load(0, Ordering::Acquire).map_err(why)?;
    let done = recorded.done().map_err(why)?;
    s.sync().map_err(why)?;
    if seen != 1 {
        return Err(format!(
            "an event recorded after a kernel fired while its result was {seen}"
        ));
    }
    if !done {
        return Err("a synced event is not done".to_string());
    }
    Ok(Verdict::Pass)
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

/// The device's word equals the host model's, word for word, and reads as
/// the model's fault.
fn same_word<H: Harness>(h: &H, model: &[u32], when: &str) -> Result<(), String> {
    let d = h.device();
    let s = d.stream();
    let words = h.fault_words(s).map_err(why)?;
    if words.len() < model.len() {
        return Err(format!(
            "{when}: the device's word has {} u32s, the model {}",
            words.len(),
            model.len()
        ));
    }
    if let Some(at) = words.iter().zip(model).position(|(w, m)| w != m) {
        return Err(format!(
            "{when}: word {at} is {:#x} on the device, {:#x} in the model",
            words[at], model[at]
        ));
    }
    let want = Fault::decode(model).map_err(why)?;
    let got = d.fault().read(s).map_err(why)?;
    if got != want {
        return Err(format!(
            "{when}: the device reads {got:?}, the model {want:?}"
        ));
    }
    Ok(())
}

/// How many raises race in one launch.
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

fn fault_clause<H: Harness>(h: &H) -> Clause {
    let d = h.device();
    let s = d.stream();
    let word = d.fault();
    word.clear(s).map_err(why)?;
    same_word(h, &modelled(&[])?, "a cleared word")?;

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
    for raise in sequence {
        h.raise(s, &[raise]).map_err(why)?;
    }
    same_word(h, &modelled(&sequence)?, "an ordered sequence")?;
    let raised = word
        .read(s)
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
    word.clear(s).map_err(why)?;
    let past = [(LAYER_NONE + 1, FaultSite::Router)];
    h.raise(s, &past).map_err(why)?;
    same_word(h, &modelled(&past)?, "a layer past the last")?;

    // Raises that race: every lane raises its own layer and site at once.
    for round in 0..8usize {
        word.clear(s).map_err(why)?;
        let lanes = lane_raises(round);
        h.raise(s, &lanes).map_err(why)?;
        same_word(h, &modelled(&lanes)?, "raises that race")?;
    }

    word.clear(s).map_err(why)?;
    same_word(h, &modelled(&[])?, "a word cleared again")?;
    Ok(Verdict::Pass)
}

/// A fault raised by work behind a long kernel is seen by the read that
/// follows it, without any wait of the clause's own.
fn fault_read_waits<H: Harness>(h: &H) -> Clause {
    let d = h.device();
    let s = d.stream();
    let word = d.fault();
    word.clear(s).map_err(why)?;
    let mut out = d.zeroed::<u32>(s, Elems(1)).map_err(why)?;
    h.long(s, out.whole_mut().map_err(why)?).map_err(why)?;
    h.raise(s, &[(5, FaultSite::Logit)]).map_err(why)?;
    match word.read(s).map_err(why)? {
        Some(fault) if fault.layer == 5 && fault.code == FaultSite::Logit as u32 => {}
        Some(other) => return Err(format!("the read saw {other:?}, not layer 5's logit")),
        None => return Err("a read after a long kernel saw a clean word".to_string()),
    }
    word.clear(s).map_err(why)?;
    Ok(Verdict::Pass)
}

/// The bytes the census clause allocates to see the free bytes fall: the fall
/// is read from the device's own free count, so another process freeing more
/// than this meanwhile turns the clause red.
const CENSUS_PROBE: usize = 64 << 20;

fn census<H: Harness, C: Census>(h: &H) -> Clause {
    let d = h.device();
    let s = d.stream();
    let all = C::census().map_err(why)?;
    for (at, info) in all.iter().enumerate() {
        if usize::try_from(info.ordinal) != Ok(at) {
            return Err(format!("device {at} is named ordinal {}", info.ordinal));
        }
        if info.free_bytes > info.total_bytes {
            return Err(format!(
                "{}: {} free bytes pass its {} total",
                info.name, info.free_bytes, info.total_bytes
            ));
        }
        if all.iter().skip(at + 1).any(|other| other.uuid == info.uuid) {
            return Err(format!("{}: its id is another device's", info.name));
        }
    }
    let own = d.info().map_err(why)?;
    let listed = all
        .get(usize::try_from(own.ordinal).map_err(|e| e.to_string())?)
        .ok_or("the opened device's ordinal is not in the census")?;
    if (listed.name.as_str(), listed.uuid, listed.total_bytes)
        != (own.name.as_str(), own.uuid, own.total_bytes)
    {
        return Err(format!(
            "the opened device ({}, {} bytes) is not the census's device at ordinal {} ({}, {} \
             bytes)",
            own.name, own.total_bytes, own.ordinal, listed.name, listed.total_bytes
        ));
    }
    if own.free_bytes > own.total_bytes {
        return Err(format!(
            "the device's {} free bytes pass its {} total",
            own.free_bytes, own.total_bytes
        ));
    }

    let before = own.free_bytes;
    let big = d.zeroed::<u8>(s, Elems(CENSUS_PROBE)).map_err(why)?;
    s.sync().map_err(why)?;
    let after = d.info().map_err(why)?.free_bytes;
    drop(big);
    if after >= before {
        return Err(format!(
            "a {} MiB buffer left the free bytes at {after}, from {before}",
            CENSUS_PROBE >> 20
        ));
    }
    Ok(Verdict::Pass)
}
