/// Taps of the causal conv.
pub const CONV_TAPS: usize = 4;

/// Predecessors the conv reads: the taps except the current token.
pub const CONV_ROWS: usize = CONV_TAPS - 1;

/// Positions one step's position window may name.
pub const PASS_ROWS: usize = 8;

/// Slots of the conv ring: the predecessors plus the positions of one pass.
pub const RING_ROWS: usize = CONV_ROWS + PASS_ROWS;
