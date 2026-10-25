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
//! - the plan of the same header (`generate_metal --plan`, read from the
//!   header alone) counts the buffers the set holds, and its bytes bound the
//!   set's: the bytes it asks of the driver at least, each buffer rounded up
//!   to a page at most;
//! - the open makes the rows arena too, so the first pass allocates nothing
//!   on the device and adds nothing to the set;
//! - a submit after a gap of [`KEEP`] renews the request once, and a submit
//!   straight after another does not.

use std::path::Path;
use std::thread::sleep;
use std::time::Duration;

use bloomery_device::Device;
use bloomery_device::conformance::{Clause, Verdict, why};
use bloomery_metal::arch::qwen3moe::check::{self, Tally};
use bloomery_metal::arch::qwen3moe::plan;
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

/// What the plan of `path`'s header says the open makes buffers of at [`CTX`]
/// rows: the same read `generate_metal --plan` makes, no weight byte touched.
fn planned(path: &Path) -> Result<Tally, String> {
    let header = plan::open_header(path)?;
    let outcome = check::check(&header.split, CTX);
    let items: Vec<String> = outcome.items.iter().map(|i| i.error.to_string()).collect();
    let checked = outcome
        .checked
        .ok_or_else(|| format!("the plan refuses the file the open opened: {items:?}"))?;
    Ok(checked.bytes().map_err(says)?.total())
}

pub fn r_resident(device: &MetalDevice, library: &Library, path: &Path) -> Clause {
    // Read before the open: the keep rule below counts the time from the
    // open's request, so nothing but the clause's own steps may fill it.
    let plan = planned(path)?;
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
    if plan.buffers != at_open.buffers as u64
        || !(plan.requested..=plan.paged).contains(&at_open.bytes)
    {
        return Err(format!(
            "the plan counts {} buffers of {}..={} bytes; the residency set holds {} buffers of \
             {} bytes",
            plan.buffers, plan.requested, plan.paged, at_open.buffers, at_open.bytes
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
    let after_pass = s.residency();
    let grew = held(device)?.abs_diff(before_pass);
    if grew != 0 || after_pass.buffers != at_open.buffers || after_pass.bytes != at_open.bytes {
        return Err(format!(
            "the first pass moved the device's bytes by {grew} and the residency set from {} \
             buffers of {} bytes to {} of {}; the open makes every buffer a pass binds",
            at_open.buffers, at_open.bytes, after_pass.buffers, after_pass.bytes
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
        "the open allocated {opened} bytes, the rows arena's {} among them; the residency set \
         holds {} buffers of them to within a page each, as many as the plan of the header counts \
         ({}..={} bytes asked, rounded to pages); the first pass allocated nothing and added \
         nothing to the set; a step after {:?} idle renewed the request once, two steps \
         straight after the pass renewed none",
        s.rows_arena_bytes(),
        at_open.buffers,
        plan.requested,
        plan.paged,
        KEEP + GAP_MARGIN
    )))
}
