use crate::geom::{Geometry, KernelRefused};
use crate::geometry;
use crate::instance::{FfnAxes, HOST, Instance};

/// Slots in one launch of `inst`'s experts: its routed picks, and a gated
/// shared expert's slot after them; a dense FFN's one.
#[must_use]
pub const fn slots(inst: &Instance) -> usize {
    match inst.key.ffn {
        FfnAxes::Routed(m) => m.top_k as usize + m.router.rule.gated as usize,
        FfnAxes::Dense { .. } => 1,
    }
}

/// Slots in one launch of the experts. A shim: the build's host instance's.
/// Callers left: the arch, the kernel gate, its oracles and the probes.
pub const EXPERT_SLOTS: usize = slots(&HOST);

/// Ids a launch of `inst`'s experts reads: the picks, or a dense FFN's one
/// (expert 0 of its one-expert stacks).
#[must_use]
pub const fn routed_ids(inst: &Instance) -> usize {
    match inst.key.ffn {
        FfnAxes::Routed(m) => m.top_k as usize,
        FfnAxes::Dense { .. } => 1,
    }
}

/// Ids a launch reads: the picks. The shared slot has none. A shim: the
/// build's host instance's. Callers left: the arch, the e2e and kernel gates
/// and the probes.
pub const ROUTED_IDS: usize = routed_ids(&HOST);

/// The slot of `inst`'s experts that reads the shared expert: the one after
/// the routed ids.
#[must_use]
pub const fn shared_slot(inst: &Instance) -> usize {
    routed_ids(inst)
}

/// The slot of the experts that reads the shared expert. A shim: the build's
/// host instance's. Callers left: the e2e and kernel gates.
pub const SHARED_SLOT: usize = shared_slot(&HOST);

/// The expert intermediate width of `inst`: a routed expert's, or a dense
/// FFN's own. The pair's rows and the fold's K are this.
#[must_use]
pub const fn expert_k(inst: &Instance) -> usize {
    match inst.key.ffn {
        FfnAxes::Routed(m) => m.expert_ff as usize,
        FfnAxes::Dense { ff } => ff as usize,
    }
}

/// The expert intermediate width. The pair's rows and the fold's K are this.
/// A shim: the build's host instance's. Callers left: the arch, the e2e and
/// kernel gates, their oracles and the probes.
pub const EXPERT_K: usize = expert_k(&HOST);

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
