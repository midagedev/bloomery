/// The width of a SIMD group on every Apple GPU.
pub const SIMD_WIDTH: usize = 32;

/// The output rows one SIMD group of a mat-vec owns. The kernels' row loop
/// and their tail are written for any of 2, 4 and 8, the counts the probe
/// sweeps; four is the landed count, the measured choice of the M1 Pro sweep
/// (the small-width families gain a third to a half: their input and gain
/// loads halve per row). A count outside that set is a kernel edit, not a
/// geometry.
pub const ROWS_PER_SIMD_GROUP: usize = 4;

/// The SIMD groups in a mat-vec's threadgroup: the default. The count is the
/// kernels' function constant, so each count is its own pipeline, and the
/// output head and the experts run their own counts of the same constant.
pub const SIMD_GROUPS_PER_GROUP: usize = 2;

/// The output rows one threadgroup of a mat-vec owns.
pub const ROWS_PER_GROUP: usize = ROWS_PER_SIMD_GROUP * SIMD_GROUPS_PER_GROUP;

/// The threads in a mat-vec's threadgroup.
pub const THREADS_PER_GROUP: usize = SIMD_WIDTH * SIMD_GROUPS_PER_GROUP;

/// The SIMD groups a threadgroup may hold: the device's 1024 threads, 32 of
/// them in a group.
pub const MAX_SIMD_GROUPS_PER_GROUP: usize = 1024 / SIMD_WIDTH;

/// The values one quantized block holds, in Q4_K and Q6_K alike.
pub const BLOCK_VALUES: usize = 256;

/// The row segments one segmented mat-vec runs.
pub const MAX_SEGMENTS: usize = 4;

/// The value heads a gated delta net's out projection reads: its head scale
/// and its partial sums are per head.
pub const VALUE_HEADS: usize = 32;

/// The columns one value head's gated output holds.
pub const HEAD_COLS: usize = 128;

/// The column blocks one head's partial sums are split into.
pub const PARTIALS_PER_HEAD: usize = 8;
