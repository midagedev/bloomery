//! The Qwen3-30B-A3B family of the fixture tier: ik's dumps on the fixture file
//! ([`crate::fixture::first_shard`]), with the same ik tree as the Qwen3.6 family and set names
//! that carry the `fx_` prefix. The identity is the fixture's: its first shard and its
//! `# fixture` line ([`crate::family::Identity::FixtureManifest`]). The file has no MTP layer,
//! so there is no draft family.

use super::{ARCH, IK_BUILD};
use crate::RefError;
use crate::family::{Build, Family, Identity};

/// The 5-token prefill, ik on the CPU.
pub const BATCH: &str = "fx_ref_qwen3moe";
/// Step 4 after a prefill run node by node under the dumped schedule.
pub const STEP4: &str = "fx_ref_qwen3moe_step4_every_node";
/// Step 1,024 of the prose after a fused prefill.
pub const D1K: &str = "fx_ref_qwen3moe_d1k";

/// The fixture file the tree runs, as a family's `runs`.
fn runs() -> Result<String, RefError> {
    crate::fixture::first_shard(ARCH)
}

/// ik's node dumps of the fixture: the batch set and the decode-step sets.
pub static IK: Family = Family {
    name: "fx-ik-qwen3moe",
    sets: &[BATCH, STEP4, D1K],
    resolve: None,
    recipe: "BLOOMERY_TIER=fixture just dump-ref-fixture qwen3moe [VARIANT]",
    identity: Identity::FixtureManifest,
    arch: Some(ARCH),
    build: Some(Build::Is(IK_BUILD)),
    runs: Some(runs),
    draft_runs: None,
    consumers: &[],
};
