//! The DeepSeek-V4.1-Flash node-dump family: ik's decode-step dumps on the fixture file
//! ([`crate::fixture::first_shard`]), the only ik node dumps the architecture has. The sets carry
//! the `fx_` prefix; there is one fixture file, so no set name takes the per-file suffix of
//! [`gguf::v41::set`]. The identity is the fixture's: its first shard and its `# fixture` line
//! ([`crate::family::Identity::FixtureManifest`]).

use super::{ARCH, IK_BUILD};
use crate::RefError;
use crate::family::{Build, Family, Identity};

/// Step 4, at an even position, where no csa group completes.
pub const STEP4: &str = "fx_ref_deepseek41_step4_every_node";
/// Step 301 with the file's top-k, where neither stream builds an indexer.
pub const D1N: &str = "fx_ref_deepseek41_d1n_every_node";
/// Step 301, where a csa group completes; the window mask 512 cells wide.
pub const D1: &str = "fx_ref_deepseek41_d1_every_node";
/// Step 1,025, where a csa group completes; the window mask 1,280 cells wide.
pub const D2: &str = "fx_ref_deepseek41_d2_every_node";

/// The fixture file the tree runs, as a family's `runs`.
fn runs() -> Result<String, RefError> {
    crate::fixture::first_shard(ARCH)
}

/// ik's node dumps of the fixture file, one token after a prefill run node by node under the
/// dumped schedule (the model profile's `ref_step_variant`, `tools/ref/models/deepseek41.sh`).
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
    consumers: &["gate-ds41-oracle", "gate-ds41-plan", "gate-gpu-ds41-step"],
};
