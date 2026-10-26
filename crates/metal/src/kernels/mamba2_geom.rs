//! The Mamba-2 kernels' compile-time geometry: the numbers
//! `kernels/mamba2.metal` states as `MAMBA2_*` constants and its launchers
//! size their grids and buffers by. Heads, groups and the head width are
//! launch arguments, not constants: they only set trip counts and the grid,
//! so one pipeline serves every Mamba-2 layer of every file.

use super::delta_geom::{PASS_ROWS, RING_ROWS};

/// The scan's state length: the values of one channel's recurrent state.
pub const STATE: usize = 128;

/// Taps of the causal conv: the three predecessors and the current token.
pub const TAPS: usize = 4;

/// SIMD groups of the step's threadgroup.
const SIMD_GROUPS: usize = 4;

/// Threads of a SIMD group.
const SIMD_WIDTH: usize = 32;

/// Threads of a threadgroup, in the step and in the norm.
pub const THREADS: usize = SIMD_GROUPS * SIMD_WIDTH;

/// Channels one SIMD group walks.
const CH_PER_SIMD_GROUP: usize = 4;

/// Channels of one head a threadgroup takes.
pub const CH_PER_BLOCK: usize = SIMD_GROUPS * CH_PER_SIMD_GROUP;

/// Threadgroups of a head: the head's channels in chunks of
/// [`CH_PER_BLOCK`], the last one short when the width does not divide.
#[must_use]
pub const fn chunks(head_dim: usize) -> usize {
    head_dim.div_ceil(CH_PER_BLOCK)
}

// The kernel spells these as literals through `MAMBA2_*`: a lane holds the
// states `n = lane + 32 j`, and the staging loop's three rounds cover the
// group's `B` and `C` and the block's `x`. The conv ring is the delta step's
// (one slot per position modulo `RING_ROWS`): it holds the three
// predecessors and the pass's rows, so the slot a token writes is none of the
// slots it reads.
const _: () = assert!(THREADS == 128 && STATE == 128 && TAPS == 4 && CH_PER_BLOCK == 16);
const _: () = assert!(STATE == 4 * SIMD_WIDTH);
const _: () = assert!(RING_ROWS >= TAPS - 1 + PASS_ROWS);
