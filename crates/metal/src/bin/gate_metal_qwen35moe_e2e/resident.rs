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
//! - every submit's decision agrees with the gap it measured: the residency
//!   keeps the longest gap a submit left the request at and the shortest it
//!   renewed it at, so the rule (renew when the last request is [`KEEP`] old,
//!   else leave it) is read on the clock the rule itself reads, not on the
//!   wall time the steps take; the step after a sleep of [`KEEP`] renews, and
//!   some submit comes within [`KEEP`] of the last request and leaves it.

use std::path::Path;
use std::thread::sleep;
use std::time::Duration;

use bloomery_device::conformance::{Clause, Verdict, why};
use bloomery_metal::arch::qwen3moe::check::{self, Tally};
use bloomery_metal::arch::qwen3moe::plan;
use bloomery_metal::arch::qwen3moe::session::{PassOut, Session, StateOut, StepMode};
use bloomery_metal::device::MetalDevice;
use bloomery_metal::library::Library;
use bloomery_metal::residency::{KEEP, ResidencyStats};

use super::pass::prompt_ids;
use super::says;

/// The cache rows of the clause's session: a pass of a dozen ids and a few
/// steps.
const CTX: usize = 64;

/// The ids of the pass: more than a step's window.
const PASS_IDS: usize = 12;

/// Whether the device's `held` growth and the set's `bytes` agree to within a
/// page a buffer of `buffers`.
fn agree(held: u64, bytes: u64, buffers: usize) -> bool {
    held.abs_diff(bytes) <= buffers as u64 * check::PAGE
}

/// How far past [`KEEP`] the idle gap sleeps, so a clock that wakes late
/// cannot land the gap inside it.
const GAP_MARGIN: Duration = Duration::from_millis(50);

/// The bytes this process holds on the device, as the driver counts them.
fn held(device: &MetalDevice) -> Result<u64, String> {
    device.allocated_bytes().map_err(why)
}

/// What the plan of `path`'s header says the open makes buffers of at [`CTX`]
/// rows: the same read `generate_metal --plan` makes, no weight byte touched.
fn planned(path: &Path) -> Result<Tally, String> {
    let header = plan::open_header(path)?;
    let outcome = check::check(header.split(), CTX);
    let items: Vec<String> = outcome.items.iter().map(|i| i.error.to_string()).collect();
    let checked = outcome
        .checked
        .ok_or_else(|| format!("the plan refuses the file the open opened: {items:?}"))?;
    Ok(checked.bytes().map_err(says)?.total())
}

/// The most steps run to find a submit within [`KEEP`] of the last request:
/// a step is that slow only on a machine too loaded to observe the rule.
const HOLD_STEPS: usize = 16;

/// The first decision of `stats` that disagrees with the gap it measured:
/// a renewal at a gap under [`KEEP`], or a submit that left the request at a
/// gap of [`KEEP`] or more.
fn decisions(stats: &ResidencyStats) -> Result<(), String> {
    if let Some(gap) = stats.renewed_gap_min
        && gap < KEEP
    {
        return Err(format!(
            "a submit renewed the request {gap:?} after the last, under the {KEEP:?} the rule waits"
        ));
    }
    if stats.held_gap_max >= KEEP {
        return Err(format!(
            "a submit left the request {:?} after the last, at or past the {KEEP:?} the rule \
             renews at",
            stats.held_gap_max
        ));
    }
    Ok(())
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

    // Steps straight after the pass, until a submit has left the request
    // (the pass itself may have): how many submits fall within KEEP of the
    // last request is the machine's, what each did about it is the rule's.
    let id = ids[PASS_IDS - 1];
    let mut steps = 0;
    while steps < 2 || (s.residency().held == 0 && steps < HOLD_STEPS) {
        s.step(id, StepMode::HostFed).map_err(says)?;
        steps += 1;
    }
    let straight = s.residency();
    decisions(&straight)?;
    if straight.held == 0 {
        return Err(format!(
            "no submit of the pass and {steps} steps came within {KEEP:?} of the last request; the \
             rule's leave-it half was not observed"
        ));
    }
    sleep(KEEP + GAP_MARGIN);
    s.step(id, StepMode::HostFed).map_err(says)?;
    let after_gap = s.residency();
    if after_gap.requests <= straight.requests {
        return Err(format!(
            "a step after a gap of {:?} left {} requests, the {} before it; the rule renews at \
             {KEEP:?}",
            KEEP + GAP_MARGIN,
            after_gap.requests,
            straight.requests
        ));
    }
    decisions(&after_gap)?;
    let renewed_at = after_gap.renewed_gap_min.ok_or_else(|| {
        "the requests grew across the gap step and no renewal recorded its gap".to_string()
    })?;
    Ok(Verdict::Noted(format!(
        "the open allocated {opened} bytes, the rows arena's {} among them; the residency set \
         holds {} buffers of them to within a page each, as many as the plan of the header counts \
         ({}..={} bytes asked, rounded to pages); the first pass allocated nothing and added \
         nothing to the set; a step after {:?} idle renewed the request; the {} submits that left \
         it did so at {:?} at most and the renewals came at {renewed_at:?} or more, against \
         the rule's {KEEP:?}",
        s.rows_arena_bytes(),
        at_open.buffers,
        plan.requested,
        plan.paged,
        KEEP + GAP_MARGIN,
        after_gap.held,
        after_gap.held_gap_max
    )))
}
