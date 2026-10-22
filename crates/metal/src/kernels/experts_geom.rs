/// Slots in one launch: eight routed experts and the shared expert.
pub const EXPERT_SLOTS: usize = 9;

/// Ids a launch reads. The shared slot has none.
pub const ROUTED_IDS: usize = 8;

/// The slot that reads the shared expert.
pub const SHARED_SLOT: usize = 8;

/// The expert intermediate width. The pair's rows and the fold's K are this.
pub const EXPERT_K: usize = 512;

/// Blocks of K': the slots, two blocks of [`EXPERT_K`] each. A block holds 256
/// values, the mat-vec geometry's block.
pub const FOLDED_BLOCKS: usize = EXPERT_SLOTS * (EXPERT_K / 256);
