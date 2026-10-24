//! `r_resident`: the session's kernel buffers are held resident.
//!
//! The driver wires a buffer when a command buffer first binds it and lets
//! the wiring lapse when none names it for a while, so a session that does
//! not hold its buffers resident pays the wiring again on the first token
//! after a pass or an idle gap. The session holds them in one residency set
//! ([`bloomery_metal::residency`]); this clause pins the three facts that
//! keep it so, none of them a timing:
//!
//! - the set holds the bytes the open allocated: the device's free bytes fall
//!   across `Session::open` by the set's bytes, to within one page a buffer
//!   (the two counts round a buffer differently), so a kernel buffer made
//!   outside the set shows as bytes the set lacks;
//! - the rows arena of the first pass joins the set the same way;
//! - a submit after a gap of [`KEEP`] renews the request once, and a submit
//!   straight after another does not.

use std::path::Path;
use std::thread::sleep;
use std::time::Duration;

use bloomery_device::Device;
use bloomery_device::conformance::{Clause, Verdict, why};
use bloomery_metal::arch::qwen3moe::session::{PassOut, Session, StateOut, StepMode};
use bloomery_metal::device::MetalDevice;
use bloomery_metal::library::Library;
use bloomery_metal::residency::KEEP;

use super::pass::prompt_ids;
use super::says;

/// The cache rows of the clause's session: a pass of a dozen ids and a few
/// steps.
const CTX: usize = 64;

/// The ids of the pass: more than a step's window.
const PASS_IDS: usize = 12;

/// The page both counts round a buffer to.
const PAGE: u64 = 16 * 1024;

/// Whether the device's `held` growth and the set's `bytes` agree to within a
/// page a buffer of `buffers`.
fn agree(held: u64, bytes: u64, buffers: usize) -> bool {
    held.abs_diff(bytes) <= buffers as u64 * PAGE
}

/// How far past [`KEEP`] the idle gap sleeps, so a clock that wakes late
/// cannot land the gap inside it.
const GAP_MARGIN: Duration = Duration::from_millis(50);

/// The bytes the device holds, as its census counts them.
fn held(device: &MetalDevice) -> Result<u64, String> {
    let info = device.info().map_err(why)?;
    Ok(info.total_bytes.saturating_sub(info.free_bytes))
}

pub fn r_resident(device: &MetalDevice, library: &Library, path: &Path) -> Clause {
    let before = held(device)?;
    let mut s = Session::open_with(device, library, path, CTX, 2048).map_err(says)?;
    let opened = held(device)? - before;
    let at_open = s.residency();
    if !agree(opened, at_open.bytes, at_open.buffers) || at_open.buffers == 0 {
        return Err(format!(
            "the open allocated {opened} bytes on the device; the residency set holds {} bytes in \
             {} buffers",
            at_open.bytes, at_open.buffers
        ));
    }
    if at_open.requests != 1 {
        return Err(format!(
            "the open requested residency {} times; the commit that wires the arena is one",
            at_open.requests
        ));
    }

    let ids = prompt_ids(PASS_IDS)?;
    let before_pass = held(device)?;
    s.pass(&ids, PassOut::Last, StateOut::default())
        .map_err(says)?;
    let rows = held(device)? - before_pass;
    let after_pass = s.residency();
    let joined = after_pass.buffers - at_open.buffers;
    if !agree(rows, after_pass.bytes - at_open.bytes, joined) || rows == 0 {
        return Err(format!(
            "the first pass allocated {rows} bytes on the device; the residency set grew by {} in \
             {joined} buffers",
            after_pass.bytes - at_open.bytes
        ));
    }

    let id = ids[PASS_IDS - 1];
    let renewed = s.residency().requests;
    s.step(id, StepMode::HostFed).map_err(says)?;
    s.step(id, StepMode::HostFed).map_err(says)?;
    let back_to_back = s.residency().requests;
    sleep(KEEP + GAP_MARGIN);
    s.step(id, StepMode::HostFed).map_err(says)?;
    let after_gap = s.residency().requests;
    if after_gap != back_to_back + 1 {
        return Err(format!(
            "a step after a gap of {:?} left {after_gap} requests; the {back_to_back} before it \
             and one renewal are expected",
            KEEP + GAP_MARGIN
        ));
    }
    if back_to_back != renewed {
        return Err(format!(
            "two steps straight after the pass renewed the request {} times; the gap rule is \
             {KEEP:?}",
            back_to_back - renewed
        ));
    }
    Ok(Verdict::Noted(format!(
        "the open allocated {opened} bytes; the residency set holds {} buffers of them to within a \
         page each; the first pass's rows arena added {rows} in {joined}; a step after {:?} idle renewed the request once, two \
         steps straight after the pass renewed none",
        at_open.buffers,
        KEEP + GAP_MARGIN
    )))
}
