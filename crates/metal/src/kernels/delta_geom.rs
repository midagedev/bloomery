use crate::instance::HOST;

/// Taps of the causal conv: the host instance's.
pub const CONV_TAPS: usize = HOST.key.delta.conv_taps as usize;

/// Positions one step's position window may name.
pub const PASS_ROWS: usize = 8;

/// Slots of the conv ring of a conv of `taps` taps: the predecessors it
/// reads (the taps except the current token) plus the positions of one pass.
#[must_use]
pub const fn ring_rows(taps: usize) -> usize {
    taps - 1 + PASS_ROWS
}

/// Slots of the host instance's conv ring.
pub const RING_ROWS: usize = ring_rows(CONV_TAPS);

/// Copies of the state lane and the conv ring one rows launch may take after
/// rows inside it: a prompt call's first and last message start.
pub const SNAPS: usize = 2;
