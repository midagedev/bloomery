use crate::instance::{HOST, Instance};

/// Slots in one launch of `inst`'s experts: its routed picks, and a gated
/// shared expert's slot after them.
#[must_use]
pub const fn slots(inst: &Instance) -> usize {
    inst.key.moe.top_k as usize + inst.key.moe.router.rule.gated as usize
}

/// Slots in one launch of the host instance's experts.
pub const EXPERT_SLOTS: usize = slots(&HOST);

/// Ids a launch reads: the host instance's picks. The shared slot has none.
pub const ROUTED_IDS: usize = HOST.key.moe.top_k as usize;

/// The slot of `inst`'s experts that reads the shared expert: the one after
/// the routed ids.
#[must_use]
pub const fn shared_slot(inst: &Instance) -> usize {
    inst.key.moe.top_k as usize
}

/// The slot of the host instance's experts that reads the shared expert.
pub const SHARED_SLOT: usize = shared_slot(&HOST);

/// Whether the host instance's experts run a shared expert's slot after the
/// routed ones.
pub const HAS_SHARED: bool = SHARED_SLOT < EXPERT_SLOTS;

/// The expert intermediate width: the host instance's. The pair's rows and
/// the fold's K are this.
pub const EXPERT_K: usize = HOST.key.moe.expert_ff as usize;

/// Blocks of K': the slots, two blocks of [`EXPERT_K`] each. A block holds 256
/// values, the mat-vec geometry's block.
pub const FOLDED_BLOCKS: usize = EXPERT_SLOTS * (EXPERT_K / 256);
