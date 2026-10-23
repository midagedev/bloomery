/// The routed rows, and the id of the shared expert.
pub const EXPERTS: usize = 256;

/// The rows the kernel reads: the routed experts and the shared expert.
pub const ROWS: usize = EXPERTS + 1;

/// The columns of every row.
pub const K: usize = 2048;

/// The splits of one row. One: a SIMD group owns its row's whole width, so
/// the input and its norm are read once per threadgroup, not once per row.
pub const SPLIT: usize = 1;

/// The partials of one launch: one per row per split.
pub const PARTIALS: usize = ROWS * SPLIT;

/// The SIMD groups in the router's threadgroup. Eight fill the threadgroup
/// the norm's cooperative load wants (256 lanes over the 512 float4s of the
/// input), and 257 rows make 33 threadgroups.
pub const SIMD_GROUPS: usize = 8;

/// The width of a SIMD group. The mat-vec geometry owns the same width.
pub const SIMD_WIDTH: usize = 32;

/// The threads in the router's threadgroup.
pub const THREADS: usize = SIMD_WIDTH * SIMD_GROUPS;

/// The threadgroups of one launch.
pub const GROUPS: usize = ROWS.div_ceil(SIMD_GROUPS);

/// The most routed experts one launch selects.
pub const TOP_K_MAX: usize = 8;

/// The float4s one lane loads from one split.
pub const FLOAT4S: usize = 16;

const _: () = assert!(K.is_multiple_of(4 * THREADS));
