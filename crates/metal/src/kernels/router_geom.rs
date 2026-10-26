use crate::geometry;
use crate::instance::{HOST, Instance, REGISTERED};

/// The routed rows, and the id of the shared expert. A shim: the build's host
/// instance's router row (`Geometry::router`'s `experts`). Callers left: the
/// arch, the e2e and kernel gates, their oracles and the probes.
pub const EXPERTS: usize = HOST.key.moe.router.experts as usize;

/// The rows `inst`'s router reads: the routed experts, and a gated shared
/// expert's gate after them.
#[must_use]
pub const fn rows(inst: &Instance) -> usize {
    inst.key.moe.router.experts as usize + inst.key.moe.router.rule.gated as usize
}

/// The rows the router reads. A shim: the build's host instance's. Callers
/// left: the arch, the kernel gate, its oracles and the probes.
pub const ROWS: usize = rows(&HOST);

/// A gated shared expert: one more router row, and one more id and weight
/// after the routed picks. Zero or one. A shim: the build's host instance's.
/// Callers left: the arch, the kernel gate, its oracles and the probes.
pub const SHARED: usize = HOST.key.moe.router.rule.gated as usize;

/// The columns of every row: the hidden width. A shim: the build's host
/// instance's. Callers left: the arch, the kernel gate, its oracles and the
/// probes.
pub const K: usize = HOST.key.hidden as usize;

/// The splits of one row. One: a SIMD group owns its row's whole width, so
/// the input and its norm are read once per threadgroup, not once per row.
pub const SPLIT: usize = 1;

/// The partials of one launch of `inst`'s router: one per row per split.
#[must_use]
pub const fn partials(inst: &Instance) -> usize {
    rows(inst) * SPLIT
}

/// The partials of one launch. A shim: the build's host instance's. Callers
/// left: the arch, the kernel gate and the probes.
pub const PARTIALS: usize = partials(&HOST);

/// The SIMD groups in the router's threadgroup. Eight fill the threadgroup
/// the norm's cooperative load wants (256 lanes over the 512 float4s of the
/// input), and the rows make `rows.div_ceil(8)` threadgroups (33 for
/// Qwen3.6's 257, 16 for Qwen3-30B's 128). The kernel takes any count that
/// divides the input's float4s exactly (the norm's lanes cover the input, so
/// the probe can sweep the threadgroup count a card fills best); the norm
/// band's counts follow the shipped count.
pub const SIMD_GROUPS: usize = 8;

/// The width of a SIMD group. The mat-vec geometry owns the same width.
pub const SIMD_WIDTH: usize = 32;

/// The threads in the router's threadgroup.
pub const THREADS: usize = SIMD_WIDTH * SIMD_GROUPS;

/// The threadgroups of one launch of `inst`'s router.
#[must_use]
pub const fn groups(inst: &Instance) -> usize {
    rows(inst).div_ceil(SIMD_GROUPS)
}

/// The threadgroups of one launch. A shim: the build's host instance's.
/// Callers left: the kernel gate.
pub const GROUPS: usize = groups(&HOST);

/// The most routed experts one launch selects. A shim: the build's host
/// instance's picks. Callers left: the kernel gate, its oracles and the
/// probes.
pub const TOP_K_MAX: usize = HOST.key.moe.top_k as usize;

/// The float4s one lane loads from one split of a row of `k` columns.
#[must_use]
pub const fn float4s(k: usize) -> usize {
    k / (4 * SIMD_WIDTH * SPLIT)
}

/// The logits row stride a pass keeps for a router of `rows` rows: the rows
/// rounded up to a multiple of four floats (the GEMM's output rows are
/// four-aligned).
#[must_use]
pub const fn logit_stride(rows: usize) -> usize {
    rows.div_ceil(4) * 4
}

/// The geometry the router kernel is written for, held for `inst`: the
/// kernel's own `static_assert`s, stated where the host sizes its launch.
const fn holds(inst: &Instance) {
    let n_rows = rows(inst);
    let k = inst.key.hidden as usize;
    let shared = inst.key.moe.router.rule.gated as usize;
    assert!(partials(inst) == n_rows * SPLIT);
    assert!(groups(inst) * SIMD_GROUPS >= n_rows && (groups(inst) - 1) * SIMD_GROUPS < n_rows);
    assert!(n_rows == inst.key.moe.router.experts as usize + shared && shared <= 1);
    assert!(SIMD_WIDTH * float4s(k) * 4 == k / SPLIT);
    assert!(SIMD_WIDTH * float4s(k) * SPLIT * 4 == k);
    assert!(k.is_multiple_of(4 * THREADS));
}

const _: () = {
    // The two that name no instance: the threadgroup the norm's cooperative
    // load wants, and the width the mat-vec geometry owns too.
    assert!(THREADS == 256);
    assert!(SIMD_WIDTH == geometry::SIMD_WIDTH);
    let mut i = 0;
    while i < REGISTERED.len() {
        holds(&REGISTERED[i]);
        i += 1;
    }
};
