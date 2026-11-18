//! The Hy4-flash family of the fixture tier: the PR tree's node dumps on the fixture file ([`crate::fixture::first_shard`]),
//! mirroring [`super::LCPP`] with the same tree and the same prompts. The set names carry the `fx_` prefix; the identity is the
//! fixture's: its first shard and its `# fixture` line ([`crate::family::Identity::FixtureManifest`]), so a set dumped from the
//! real file is stale for it and a set dumped from the fixture is stale for the real family, each by the `# model` line it states.

use super::{ARCH, LCPP_BUILD};
use crate::RefError;
use crate::family::{Build, Family, Identity};

/// [`super::A`] on the fixture.
pub const A: &str = "fx_ref_hy4_a";
/// [`super::B`] on the fixture.
pub const B: &str = "fx_ref_hy4_b";
/// [`super::C`] on the fixture.
pub const C: &str = "fx_ref_hy4_c";
/// [`super::D`] on the fixture.
pub const D: &str = "fx_ref_hy4_d";

/// [`super::E`] on the fixture.
pub const E: &str = "fx_ref_hy4_e";

/// The fixture file the tree runs, as a family's `runs`.
fn runs() -> Result<String, RefError> {
    crate::fixture::first_shard(ARCH)
}

/// [`super::LCPP`] on the fixture.
pub static LCPP: Family = Family {
    name: "fx-lcpp-hy4",
    sets: &[A, B, C, D, E],
    resolve: None,
    recipe: "BLOOMERY_TIER=fixture just dump-ref-hy4",
    identity: Identity::FixtureManifest,
    arch: Some(ARCH),
    build: Some(Build::Is(LCPP_BUILD)),
    runs: Some(runs),
    draft_runs: None,
    consumers: super::LCPP.consumers,
};
