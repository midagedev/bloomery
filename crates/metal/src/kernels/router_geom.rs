use crate::instance::{HOST, Instance};

/// The routed rows, and the id of the shared expert: the host instance's
/// router row's.
pub const EXPERTS: usize = HOST.key.moe.router.experts as usize;

/// The rows `inst`'s router reads: the routed experts, and a gated shared
/// expert's gate after them.
#[must_use]
pub const fn rows(inst: &Instance) -> usize {
    inst.key.moe.router.experts as usize + inst.key.moe.router.rule.gated as usize
}

/// The rows the host instance's router reads.
pub const ROWS: usize = rows(&HOST);

/// The host instance's gated shared expert: one more router row, and one
/// more id and weight after the routed picks. Zero or one.
pub const SHARED: usize = HOST.key.moe.router.rule.gated as usize;

/// The columns of every row: the host instance's hidden width.
pub const K: usize = HOST.key.hidden as usize;

/// The splits of one row. One: a SIMD group owns its row's whole width, so
/// the input and its norm are read once per threadgroup, not once per row.
pub const SPLIT: usize = 1;

/// The partials of one launch: one per row per split.
pub const PARTIALS: usize = ROWS * SPLIT;

/// The SIMD groups in the router's threadgroup. Eight fill the threadgroup
/// the norm's cooperative load wants (256 lanes over the 512 float4s of the
/// input), and the rows make `ROWS.div_ceil(8)` threadgroups (33 for
/// Qwen3.6's 257, 16 for Qwen3-30B's 128). The kernel takes any count that
/// divides the input's float4s exactly (the norm's lanes cover the input, so
/// the probe can sweep the threadgroup count a card fills best); the norm
/// band's counts follow the shipped count.
pub const SIMD_GROUPS: usize = 8;

/// The width of a SIMD group. The mat-vec geometry owns the same width.
pub const SIMD_WIDTH: usize = 32;

/// The threads in the router's threadgroup.
pub const THREADS: usize = SIMD_WIDTH * SIMD_GROUPS;

/// The threadgroups of one launch.
pub const GROUPS: usize = ROWS.div_ceil(SIMD_GROUPS);

/// The most routed experts one launch selects: the host instance's picks.
pub const TOP_K_MAX: usize = HOST.key.moe.top_k as usize;

/// The float4s one lane loads from one split of a row of `k` columns.
#[must_use]
pub const fn float4s(k: usize) -> usize {
    k / (4 * SIMD_WIDTH * SPLIT)
}

/// The float4s one lane loads from one split of the host instance's rows.
pub const FLOAT4S: usize = float4s(K);

const _: () = assert!(K.is_multiple_of(4 * THREADS));
