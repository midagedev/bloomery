//! The Inkling 2 family of the fixture tier: the patched PR tree's node dumps on the fixture file
//! ([`crate::fixture::first_shard`]), mirroring [`super::LCPP`] with the same tree and the same
//! prompts. The set names carry the `fx_` prefix; the identity is the fixture's: its first shard
//! and its `# fixture` line ([`crate::family::Identity::FixtureManifest`]), so a set dumped from
//! the real file is stale for it and a set dumped from the fixture is stale for the real family,
//! each by the `# model` line it states.

use super::{ARCH, BUILD};
use crate::RefError;
use crate::family::{Build, Family, Identity};

/// [`super::A`] on the fixture.
pub const A: &str = "fx_ref_inkling2_a";
/// [`super::B`] on the fixture.
pub const B: &str = "fx_ref_inkling2_b";
/// [`super::C`] on the fixture.
pub const C: &str = "fx_ref_inkling2_c";
/// [`super::D`] on the fixture.
pub const D: &str = "fx_ref_inkling2_d";

/// [`super::E`] on the fixture.
pub const E: &str = "fx_ref_inkling2_e";

/// The fixture file the tree runs, as a family's `runs`.
fn runs() -> Result<String, RefError> {
    crate::fixture::first_shard(ARCH)
}

/// [`super::LCPP`] on the fixture.
pub static LCPP: Family = Family {
    name: "fx-lcpp-inkling2",
    sets: &[A, B, C, D, E],
    resolve: None,
    recipe: "BLOOMERY_TIER=fixture just dump-ref-inkling2",
    identity: Identity::FixtureManifest,
    arch: Some(ARCH),
    build: Some(Build::Is(BUILD)),
    runs: Some(runs),
    draft_runs: None,
    consumers: super::LCPP.consumers,
};
