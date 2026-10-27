use models::{RouterNorm, Score};

use crate::geom::{Geometry, KernelRefused, yes_no};
use crate::geometry;
use crate::instance::{HOST, Instance};

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

/// The geometry the router kernel is written for, held for `g`: the
/// kernel's own `static_assert`s, stated where the host sizes its launch.
///
/// # Errors
///
/// [`KernelRefused`] naming the axis the router cannot take.
pub fn holds(g: &Geometry) -> Result<(), KernelRefused> {
    // The rule the kernel computes: a softmax over the logits, no selection
    // bias, the kept scores divided by their sum.
    let rule = g.instance.key.moe.router.rule;
    if !matches!(rule.score, Score::Softmax)
        || rule.bias
        || !matches!(rule.norm, RouterNorm::SelectedSum)
    {
        return Err(KernelRefused::new(
            "router",
            "router rule",
            format_args!(
                "{:?} scores, bias {}, norm {:?}",
                rule.score,
                yes_no(rule.bias),
                rule.norm
            ),
            "Softmax scores, no bias, norm SelectedSum",
        ));
    }
    let k = g.router.k;
    // A lane loads float4s from one split of a row, and the norm's
    // cooperative load covers the input with the threadgroup's lanes.
    if !k.is_multiple_of(4 * SIMD_WIDTH * SPLIT) || !k.is_multiple_of(4 * THREADS) {
        return Err(KernelRefused::new(
            "router",
            "hidden",
            k,
            "a multiple of 1024 (a float4 a lane over 256 threads)",
        ));
    }
    Ok(())
}

const _: () = {
    // The two that name no instance: the threadgroup the norm's cooperative
    // load wants, and the width the mat-vec geometry owns too.
    assert!(THREADS == 256);
    assert!(SIMD_WIDTH == geometry::SIMD_WIDTH);
};
