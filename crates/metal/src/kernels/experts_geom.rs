use crate::geom::{Geometry, KernelRefused};
use crate::geometry;
use crate::instance::{HOST, Instance};

/// Slots in one launch of `inst`'s experts: its routed picks, and a gated
/// shared expert's slot after them.
#[must_use]
pub const fn slots(inst: &Instance) -> usize {
    inst.key.moe.top_k as usize + inst.key.moe.router.rule.gated as usize
}

/// Slots in one launch of the experts. A shim: the build's host instance's.
/// Callers left: the arch, the kernel gate, its oracles and the probes.
pub const EXPERT_SLOTS: usize = slots(&HOST);

/// Ids a launch reads: the picks. The shared slot has none. A shim: the
/// build's host instance's. Callers left: the arch, the e2e and kernel gates
/// and the probes.
pub const ROUTED_IDS: usize = HOST.key.moe.top_k as usize;

/// The slot of `inst`'s experts that reads the shared expert: the one after
/// the routed ids.
#[must_use]
pub const fn shared_slot(inst: &Instance) -> usize {
    inst.key.moe.top_k as usize
}

/// The slot of the experts that reads the shared expert. A shim: the build's
/// host instance's. Callers left: the e2e and kernel gates.
pub const SHARED_SLOT: usize = shared_slot(&HOST);

/// The expert intermediate width. The pair's rows and the fold's K are this.
/// A shim: the build's host instance's. Callers left: the arch, the e2e and
/// kernel gates, their oracles and the probes.
pub const EXPERT_K: usize = HOST.key.moe.expert_ff as usize;

/// The geometry the experts' kernels are written for, held for `g`: the
/// fold maps every slot's blocks as one K', in the 256-value quants' blocks
/// (A4's and Q8_0's folds walk their own counts).
///
/// # Errors
///
/// [`KernelRefused`] naming the axis the experts cannot take.
pub fn holds(g: &Geometry) -> Result<(), KernelRefused> {
    if !g.experts.k.is_multiple_of(geometry::BLOCK_VALUES) {
        return Err(KernelRefused::new(
            "experts",
            "expert ff",
            g.experts.k,
            "a multiple of 256 (the fold maps whole blocks)",
        ));
    }
    Ok(())
}

const _: () = assert!(geometry::BLOCK_VALUES == 256);
