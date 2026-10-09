//! The DeepSeek-V4.1-Flash family of the fixture tier: ik's decode-step dumps
//! on the fixture file ([`crate::fixture::first_shard`]), mirroring [`super::IK`]
//! with the same ik tree and consumers and a set name that carries the `fx_`
//! prefix. There is one fixture file, so no set name takes the per-file suffix
//! of [`gguf::v41::set`]. The identity is the fixture's: its first shard and
//! its `# fixture` line ([`crate::family::Identity::FixtureManifest`]).

use super::{ARCH, IK_BUILD};
use crate::RefError;
use crate::family::{Build, Family, Identity};

/// [`super::STEP4`] on the fixture.
pub const STEP4: &str = "fx_ref_deepseek41_step4_every_node";
/// [`super::D1N`] on the fixture.
pub const D1N: &str = "fx_ref_deepseek41_d1n_every_node";
/// [`super::D1`] on the fixture.
pub const D1: &str = "fx_ref_deepseek41_d1_every_node";
/// [`super::D2`] on the fixture.
pub const D2: &str = "fx_ref_deepseek41_d2_every_node";

/// The fixture file the tree runs, as a family's `runs`.
fn runs() -> Result<String, RefError> {
    crate::fixture::first_shard(ARCH)
}

/// [`super::IK`] on the fixture.
pub static IK: Family = Family {
    name: "fx-ik-deepseek41",
    sets: &[STEP4, D1N, D1, D2],
    resolve: None,
    recipe: "BLOOMERY_TIER=fixture just dump-ref-fixture deepseek41 [VARIANT]",
    identity: Identity::FixtureManifest,
    arch: Some(ARCH),
    build: Some(Build::Is(IK_BUILD)),
    runs: Some(runs),
    draft_runs: None,
    consumers: super::IK.consumers,
};
