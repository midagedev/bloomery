//! The ERNIE 5.5 Flash Preview family of the fixture tier: the writer's sets of the cut folder (`fixture cut`, its first shard
//! [`crate::fixture::first_shard`]), with set names that carry the `fx_` prefix. The identity is the fixture's: its first shard and its
//! `# fixture` line ([`crate::family::Identity::FixtureManifest`]), so a set of the vendor's folder is stale for it by the `# model`
//! line it states, and a set of another generation of the cut by its `# fixture` line.

use super::{ARCH, BUILD};
use crate::RefError;
use crate::family::{Build, Family, Identity};

/// `short` on the fixture.
pub const SHORT: &str = "fx_ref_ernie55_short";
/// `long` on the fixture.
pub const LONG: &str = "fx_ref_ernie55_long";

/// The fixture cut the tree runs, as a family's `runs`.
fn runs() -> Result<String, RefError> {
    crate::fixture::first_shard(ARCH)
}

/// The fixture tier's family: the numpy writer's sets of the cut, read by the model's end-to-end gate.
pub static NP: Family = Family {
    name: "fx-np-ernie55",
    sets: &[SHORT, LONG],
    resolve: None,
    recipe: "BLOOMERY_TIER=fixture just dump-ref-ernie55",
    identity: Identity::FixtureManifest,
    arch: Some(ARCH),
    build: Some(Build::Is(BUILD)),
    runs: Some(runs),
    draft_runs: None,
    consumers: &["gate-gpu-ernie55-e2e"],
};
