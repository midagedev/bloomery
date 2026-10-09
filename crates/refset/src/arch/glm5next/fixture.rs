//! The GLM-5.3-Flash families of the fixture tier: ik's dumps on the fixture
//! file ([`crate::fixture::first_shard`]), each mirroring one family of the
//! real file with the same ik tree, the same consumers and a set name that
//! carries the `fx_` prefix. The identity is the fixture's: its first shard
//! and its `# fixture` line ([`crate::family::Identity::FixtureManifest`]).

use super::{ARCH, IK_BUILD, MTP_BUILD};
use crate::RefError;
use crate::family::{Build, Family, Identity};

/// [`super::BATCH`] on the fixture.
pub const BATCH: &str = "fx_ref_glm5next";
/// [`super::STEP4`] on the fixture.
pub const STEP4: &str = "fx_ref_glm5next_step4";
/// [`super::STEP4_EVERY_NODE`] on the fixture.
pub const STEP4_EVERY_NODE: &str = "fx_ref_glm5next_step4_every_node";
/// [`super::D1K`] on the fixture.
pub const D1K: &str = "fx_ref_glm5next_d1k";
/// [`super::D3K_DSA`] on the fixture.
pub const D3K_DSA: &str = "fx_ref_glm5next_d3kdsa";
/// [`super::D16K_DSA`] on the fixture.
pub const D16K_DSA: &str = "fx_ref_glm5next_d16kdsa";
/// [`super::MTP_SET`] on the fixture.
pub const MTP_SET: &str = "ref-mtp/fx_prose64_n64_k1";

/// The fixture file the tree runs, as a family's `runs`.
fn runs() -> Result<String, RefError> {
    crate::fixture::first_shard(ARCH)
}

/// [`super::IK`] on the fixture.
pub static IK: Family = Family {
    name: "fx-ik-glm5next",
    sets: &[BATCH, STEP4, STEP4_EVERY_NODE, D1K],
    resolve: None,
    recipe: "BLOOMERY_TIER=fixture just dump-ref-fixture glm5next [VARIANT]",
    identity: Identity::FixtureManifest,
    arch: Some(ARCH),
    build: Some(Build::Is(IK_BUILD)),
    runs: Some(runs),
    draft_runs: None,
    consumers: super::IK.consumers,
};

/// [`super::MTP`] on the fixture.
pub static MTP: Family = Family {
    name: "fx-mtp-glm5next",
    sets: &[MTP_SET],
    resolve: None,
    recipe: "BLOOMERY_TIER=fixture just dump-ref-mtp-fixture glm5next",
    identity: Identity::FixtureMtpManifest,
    arch: Some(ARCH),
    build: Some(Build::Is(MTP_BUILD)),
    runs: Some(runs),
    draft_runs: None,
    consumers: super::MTP.consumers,
};

/// [`super::IK_DSA`] on the fixture.
pub static IK_DSA: Family = Family {
    name: "fx-ik-glm5next-dsa",
    sets: &[D3K_DSA, D16K_DSA],
    resolve: None,
    recipe: "BLOOMERY_TIER=fixture just dump-ref-fixture glm5next [VARIANT]",
    identity: Identity::FixtureManifest,
    arch: Some(ARCH),
    build: Some(Build::Is(IK_BUILD)),
    runs: Some(runs),
    draft_runs: None,
    consumers: super::IK_DSA.consumers,
};
