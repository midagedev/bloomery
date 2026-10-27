use crate::geom::{Geometry, KernelRefused};
use crate::geometry;
use crate::instance::HOST;

/// Taps of the causal conv. A shim: the build's host instance's delta axes.
/// Callers left: the arch, the e2e and kernel gates, their oracles and the
/// probes.
pub const CONV_TAPS: usize = HOST.delta_built().conv_taps as usize;

/// Positions one step's position window may name.
pub const PASS_ROWS: usize = 8;

/// Slots of the conv ring of a conv of `taps` taps: the predecessors it
/// reads (the taps except the current token, none for a conv of no taps)
/// plus the positions of one pass.
#[must_use]
pub const fn ring_rows(taps: usize) -> usize {
    taps.saturating_sub(1) + PASS_ROWS
}

/// Slots of the conv ring. A shim: the build's host instance's. Callers left:
/// the arch, the e2e and kernel gates and their oracles.
pub const RING_ROWS: usize = ring_rows(CONV_TAPS);

/// Copies of the state lane and the conv ring one rows launch may take after
/// rows inside it: a prompt call's first and last message start.
pub const SNAPS: usize = 2;

/// Rows of one chunk of the rows launch's chunked recurrence: the tile rows of
/// the intra-chunk products (a multiple of 8) and the width of a chunk's
/// triangular factors.
pub const CHUNK: usize = 32;

/// f32s a row keeps for each value head of the chunked recurrence's factors:
/// the chunk's triangular inverse row and the weighted score row (`CHUNK`
/// each), the cumulative decay and the decay to the chunk's last row, padded
/// to a multiple of four.
pub const AUX_PER_HEAD: usize = 2 * CHUNK + 4;

/// Rows of a tile of the chunked recurrence. The prepared rows a launch reads
/// are held to a multiple of this, so a tile's rows past the launch's last row
/// are the window's own.
pub const TILE_ROWS: usize = 8;

const _: () = assert!(CHUNK.is_multiple_of(TILE_ROWS));

/// The geometry the delta kernels are written for, held for `g`'s delta axes
/// (the axes its kernels are compiled at, [`crate::instance::Instance::delta_built`]):
/// the kernels' own `static_assert`s, stated where the host sizes its launches.
///
/// # Errors
///
/// [`KernelRefused`] naming the axis the delta step cannot take.
pub fn holds(g: &Geometry) -> Result<(), KernelRefused> {
    let d = &g.delta;
    if d.ring_rows != 11 {
        return Err(KernelRefused::new(
            "delta",
            "conv taps",
            d.conv_taps,
            "4 (the conv ring holds 11 slots)",
        ));
    }
    if d.groups != 256 {
        return Err(KernelRefused::new(
            "delta",
            "value heads",
            d.value_heads,
            "32 (256 partial groups)",
        ));
    }
    if d.head_cols != 128 || d.head_cols / geometry::PARTIALS_PER_HEAD != 16 {
        return Err(KernelRefused::new(
            "delta",
            "head columns",
            d.head_cols,
            "128 (16 columns a block)",
        ));
    }
    Ok(())
}
