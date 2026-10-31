//! The engine threads' waits, summed over the process: on the card's work
//! (each finished wait inside a guarded engine call,
//! [`crate::watchdog::wait_done`]) and on the rows of a PLE table read from
//! NVMe for a call (`rows::RowRead`). A server reads the totals around
//! each decode round and books the difference by the round's width. Relaxed
//! adds only: no lock and no allocation on the engine path.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use crate::host::nanos;

/// ns the engine threads waited for the card's work.
static CARD_NS: AtomicU64 = AtomicU64::new(0);
/// ns the engine threads waited for PLE table rows.
static PLE_NS: AtomicU64 = AtomicU64::new(0);

/// A wait for the card's work of `d`.
pub(crate) fn card(d: Duration) {
    CARD_NS.fetch_add(nanos(d), Ordering::Relaxed);
}

/// A wait for PLE table rows of `d`.
pub(crate) fn ple(d: Duration) {
    PLE_NS.fetch_add(nanos(d), Ordering::Relaxed);
}

/// The time the process's engine threads have waited for the card's work.
#[must_use]
pub fn card_waited() -> Duration {
    Duration::from_nanos(CARD_NS.load(Ordering::Relaxed))
}

/// The time the process's engine threads have waited for PLE table rows.
#[must_use]
pub fn ple_waited() -> Duration {
    Duration::from_nanos(PLE_NS.load(Ordering::Relaxed))
}
