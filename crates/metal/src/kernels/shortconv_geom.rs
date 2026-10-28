//! The short conv's launch geometry: the threads a group runs along the
//! channels, the slots one step launch takes, and the sizes of the windows a
//! launch reads and writes. `kernels/shortconv.metal` takes every number as an
//! argument (the conv's width, the ring's rows, the channels), so the header
//! writes none of these.

use crate::encode::MAX_BYTES_BINDING;

/// Threads in the conv's threadgroup, one a channel.
pub const THREADS: usize = 128;

/// Slots one step launch takes: its positions travel as one bytes binding, a
/// word a slot.
pub const MAX_SLOTS: usize = MAX_BYTES_BINDING / 4;

/// Threadgroups along the `hidden` channels.
#[must_use]
pub const fn groups(hidden: usize) -> usize {
    hidden.div_ceil(THREADS)
}

/// Floats of the rings of `slots` slots: `ring_rows` rows of `hidden` each.
/// `None` when the product does not fit.
#[must_use]
pub fn ring_elems(slots: usize, ring_rows: usize, hidden: usize) -> Option<usize> {
    slots.checked_mul(ring_rows)?.checked_mul(hidden)
}

/// Rows a chunk's ring commit writes: its last `min(rows, ring_rows)`
/// positions, the only ones the ring can hold.
#[must_use]
pub fn kept_rows(rows: usize, ring_rows: usize) -> usize {
    rows.min(ring_rows)
}
