//! The Nano family of the fixture tier: mainline's last-position logits on the fixture file
//! ([`crate::fixture::first_shard`]), mirroring [`super::LOGITS`] with the same mainline tree and a set name that
//! carries the `fx_` prefix. The identity is the fixture's: its first shard and its `# fixture` line
//! ([`crate::family::Identity::FixtureNemotronLogits`]). The Nano has no routed layers, so a set of this family
//! carries no routes column.

use super::{ARCH, BUILD};
use crate::RefError;
use crate::family::{Build, Family, Identity};

/// [`super::LOGITS_SET`] on the fixture.
pub const LOGITS_SET: &str = "fx_ref-nemotron-logits";

/// The fixture file the tree runs, as a family's `runs`.
fn runs() -> Result<String, RefError> {
    crate::fixture::first_shard(ARCH)
}

/// [`super::LOGITS`] on the fixture.
pub static LOGITS: Family = Family {
    name: "fx-ref-nemotron-logits",
    sets: &[LOGITS_SET],
    resolve: None,
    recipe: "BLOOMERY_TIER=fixture just dump-ref-nemotron",
    identity: Identity::FixtureNemotronLogits,
    arch: Some(ARCH),
    build: Some(Build::Is(BUILD)),
    runs: Some(runs),
    draft_runs: None,
    consumers: super::LOGITS.consumers,
};
