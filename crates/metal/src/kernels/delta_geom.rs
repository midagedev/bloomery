use crate::geometry;
use crate::instance::{HOST, Instance, REGISTERED};

/// Taps of the causal conv. A shim: the build's host instance's delta axes.
/// Callers left: the arch, the e2e and kernel gates, their oracles and the
/// probes.
pub const CONV_TAPS: usize = HOST.delta_built().conv_taps as usize;

/// Positions one step's position window may name.
pub const PASS_ROWS: usize = 8;

/// Slots of the conv ring of a conv of `taps` taps: the predecessors it
/// reads (the taps except the current token) plus the positions of one pass.
#[must_use]
pub const fn ring_rows(taps: usize) -> usize {
    taps - 1 + PASS_ROWS
}

/// Slots of the conv ring. A shim: the build's host instance's. Callers left:
/// the arch, the e2e and kernel gates and their oracles.
pub const RING_ROWS: usize = ring_rows(CONV_TAPS);

/// Copies of the state lane and the conv ring one rows launch may take after
/// rows inside it: a prompt call's first and last message start.
pub const SNAPS: usize = 2;

/// The geometry the delta kernels are written for, held for `inst`'s delta
/// axes (the axes its kernels are compiled at, [`Instance::delta_built`]):
/// the kernels' own `static_assert`s, stated where the host sizes its
/// launches.
const fn holds(inst: &Instance) {
    let d = inst.delta_built();
    let head_cols = d.head_cols as usize;
    assert!(ring_rows(d.conv_taps as usize) == 11);
    assert!(d.value_heads as usize * geometry::PARTIALS_PER_HEAD == 256);
    assert!(head_cols == 128);
    assert!(head_cols / geometry::PARTIALS_PER_HEAD == 16);
}

const _: () = {
    let mut i = 0;
    while i < REGISTERED.len() {
        holds(&REGISTERED[i]);
        i += 1;
    }
};
