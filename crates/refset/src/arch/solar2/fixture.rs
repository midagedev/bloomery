//! The Solar-Open2 family of the fixture tier: the PAI tree's dumps on the fixture file ([`crate::fixture::first_shard`]),
//! mirroring [`super::PAI`] with the same tree, the same consumers and set names that carry the `fx_` prefix. The identity is
//! the fixture's: its first shard and its `# fixture` line ([`crate::family::Identity::FixtureManifest`]).

use super::{ARCH, PAI_BUILD};
use crate::RefError;
use crate::family::{Build, Family, Identity};

/// [`super::A`] on the fixture.
pub const A: &str = "fx_ref_solar2_a";
/// [`super::B`] on the fixture.
pub const B: &str = "fx_ref_solar2_b";
/// [`super::C`] on the fixture.
pub const C: &str = "fx_ref_solar2_c";
/// [`super::D`] on the fixture.
pub const D: &str = "fx_ref_solar2_d";

/// The fixture file the tree runs, as a family's `runs`.
fn runs() -> Result<String, RefError> {
    crate::fixture::first_shard(ARCH)
}

/// [`super::PAI`] on the fixture.
pub static PAI: Family = Family {
    name: "fx-pai-solar2",
    sets: &[A, B, C, D],
    resolve: None,
    recipe: "BLOOMERY_TIER=fixture just dump-ref-solar2",
    identity: Identity::FixtureManifest,
    arch: Some(ARCH),
    build: Some(Build::Is(PAI_BUILD)),
    runs: Some(runs),
    draft_runs: None,
    consumers: super::PAI.consumers,
};
