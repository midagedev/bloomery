//! The Kimi K3 family of the fixture tier: mainline's node dumps on the fixture file ([`crate::fixture::first_shard`]), mirroring
//! [`super::LCPP`] with the same tree and the same prompts. The set names carry the `fx_` prefix; the identity is the fixture's: its
//! first shard and its `# fixture` line ([`crate::family::Identity::FixtureManifest`]), so a set dumped from the real file is stale
//! for it and a set dumped from the fixture is stale for the real family, each by the `# model` line it states.

use super::{ARCH, LCPP_BUILD};
use crate::RefError;
use crate::family::{Build, Family, Identity};

/// [`super::A`] on the fixture.
pub const A: &str = "fx_ref_kimik3_a";
/// [`super::B`] on the fixture.
pub const B: &str = "fx_ref_kimik3_b";
/// [`super::C`] on the fixture.
pub const C: &str = "fx_ref_kimik3_c";
/// [`super::D`] on the fixture.
pub const D: &str = "fx_ref_kimik3_d";

/// The fixture file the tree runs, as a family's `runs`.
fn runs() -> Result<String, RefError> {
    crate::fixture::first_shard(ARCH)
}

/// [`super::LCPP`] on the fixture.
pub static LCPP: Family = Family {
    name: "fx-lcpp-kimi-k3",
    sets: &[A, B, C, D],
    resolve: None,
    recipe: "BLOOMERY_TIER=fixture just dump-ref-kimi-k3",
    identity: Identity::FixtureManifest,
    arch: Some(ARCH),
    build: Some(Build::Is(LCPP_BUILD)),
    runs: Some(runs),
    draft_runs: None,
    consumers: super::LCPP.consumers,
};
