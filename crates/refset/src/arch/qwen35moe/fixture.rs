//! The Qwen3.6-35B-A3B family of the fixture tier: ik's dumps on the fixture
//! file ([`crate::fixture::first_shard`]), mirroring the real family with the
//! same ik tree, the same consumers and a set name that carries the `fx_`
//! prefix. The identity is the fixture's: its first shard and its `# fixture`
//! line ([`crate::family::Identity::FixtureManifest`]). The file has no MTP
//! layer, so there is no draft family.

use super::{ARCH, IK_BUILD};
use crate::RefError;
use crate::family::{Build, Family, Identity};

/// [`super::BATCH`] on the fixture.
pub const BATCH: &str = "fx_ref_qwen35moe";
/// [`super::STEP4`] on the fixture.
pub const STEP4: &str = "fx_ref_qwen35moe_step4_every_node";
/// [`super::D1K`] on the fixture.
pub const D1K: &str = "fx_ref_qwen35moe_d1k";

/// The fixture file the tree runs, as a family's `runs`.
fn runs() -> Result<String, RefError> {
    crate::fixture::first_shard(ARCH)
}

/// [`super::IK`] on the fixture.
pub static IK: Family = Family {
    name: "fx-ik-qwen35moe",
    sets: &[BATCH, STEP4, D1K],
    resolve: None,
    recipe: "BLOOMERY_TIER=fixture just dump-ref-fixture qwen35moe [VARIANT]",
    identity: Identity::FixtureManifest,
    arch: Some(ARCH),
    build: Some(Build::Is(IK_BUILD)),
    runs: Some(runs),
    draft_runs: None,
    consumers: super::IK.consumers,
};
