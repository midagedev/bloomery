//! The Inkling 2 Small's family (`lcpp-inkling2s`): the same patched PR tree, build, prompts and taps as [`super::LCPP`] on the
//! Small's UD-Q2_K_XL file (six shards, the first shard names it), written by `tools/ref/inkling2-dump.sh --small`. The Small
//! is the family's base program (48 stored layers, no loop), so a node's slot is its layer and no `loop_*` tap exists. A set
//! dumped from the Mini's file is stale for this family and the Small's sets are stale for the Mini's, each by the `# model`
//! line it states.

use super::{ARCH, BUILD};
use crate::RefError;
use crate::family::{Build, Family, Identity};

/// The Small's first shard: the identity each set's `# model` line states.
pub const MODEL: &str =
    "/models/Inkling-2-Small-GGUF/UD-Q2_K_XL/Inkling-2-Small-UD-Q2_K_XL-00001-of-00006.gguf";

/// [`super::A`] on the Small.
pub const A: &str = "ref_inkling2s_a";
/// [`super::B`] on the Small.
pub const B: &str = "ref_inkling2s_b";
/// [`super::C`] on the Small.
pub const C: &str = "ref_inkling2s_c";
/// [`super::D`] on the Small.
pub const D: &str = "ref_inkling2s_d";
/// [`super::E`] on the Small: its last 14 positions attend exactly 1,024 keys.
pub const E: &str = "ref_inkling2s_e";

/// The prompt sets, in the order the dump writes them.
pub const SETS: &[&str] = &[A, B, C, D, E];

/// [`MODEL`], as a family's `runs`.
fn model() -> Result<String, RefError> {
    Ok(MODEL.to_string())
}

/// The patched PR tree's node dumps of the Small's file.
pub static LCPP: Family = Family {
    name: "lcpp-inkling2s",
    sets: SETS,
    resolve: None,
    recipe: "just dump-ref-inkling2s",
    identity: Identity::Manifest,
    arch: Some(ARCH),
    build: Some(Build::Is(BUILD)),
    runs: Some(model),
    draft_runs: None,
    consumers: &[],
};
