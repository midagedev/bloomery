/// The width of a SIMD group on every Apple GPU.
pub const SIMD_WIDTH: usize = 32;

/// The output rows one SIMD group of a mat-vec owns; the kernels' tail step is
/// written for a pair.
pub const ROWS_PER_SIMD_GROUP: usize = 2;

/// The SIMD groups in a mat-vec's threadgroup.
pub const SIMD_GROUPS_PER_GROUP: usize = 2;

/// The output rows one threadgroup of a mat-vec owns.
pub const ROWS_PER_GROUP: usize = ROWS_PER_SIMD_GROUP * SIMD_GROUPS_PER_GROUP;

/// The threads in a mat-vec's threadgroup.
pub const THREADS_PER_GROUP: usize = SIMD_WIDTH * SIMD_GROUPS_PER_GROUP;

/// The values one quantized block holds, in Q4_K and Q6_K alike.
pub const BLOCK_VALUES: usize = 256;
