//! The MiMo-V2.6-Flash families. The node dumps are dumped from the MOPD
//! checkpoint's MXFP4 split set, [`MODEL`], by the ik tree [`IK_BUILD`]
//! (`tools/ref/models/mimo2.sh`).

use crate::family::{Build, Family, Identity};

/// The first shard of the split set every mimo2 set is dumped from and
/// the tree runs: the identity each set's `# model` line states. The family
/// owns the path until the model crate reads this architecture; a set of
/// the file moved elsewhere is stale by this line, not by its name.
pub const MODEL: &str =
    "/models/MiMo-V2.6-Flash-MOPD/MiMo-V2.6-Flash-MOPD-MXFP4-00001-of-00002.gguf";

/// The ik tree every mimo2 oracle family is dumped from, and the one
/// place a re-take of the oracle changes it.
// PIN(2026-10-07): /home/user/ik-mimo2 at its HEAD — ik 043ced9a, the first
// tree that loads this file's fused attn_qkv, with upstream's per-layer fix
// of that load and the one oracle commit upstream lacks
// (tools/ref/build-ik-mimo2.sh; the tree's PROVENANCE).
pub const IK_BUILD: &str = "fd0c6abd";

/// The architecture every mimo2 manifest names in its `# arch` line.
pub const ARCH: &str = "mimo2";

/// The 5-token prefill, ik on the CPU.
pub const BATCH: &str = "ref_mimo2";
/// Step 4 after a fused prefill of the batch set's first four ids.
pub const STEP4: &str = "ref_mimo2_step4";
/// [`STEP4`] after a prefill run node by node under the dumped schedule.
pub const STEP4_EVERY_NODE: &str = "ref_mimo2_step4_every_node";
/// Step 1,024 of the prose after a fused prefill: past the 128-position
/// sliding window, so every SWA layer's step reads a window of real cells.
pub const D1K: &str = "ref_mimo2_d1k";
/// Step 4,096 of the prose after a fused prefill: the full-attention layers
/// (9 of 48) read the whole context at depth.
pub const D4096: &str = "ref_mimo2_d4096";

/// The decode-step sets (the model profile's `ref_step_variant`), by
/// position.
pub const STEP_SETS: &[&str] = &[STEP4, STEP4_EVERY_NODE, D1K, D4096];

/// [`MODEL`], as a family's `runs`.
fn model() -> String {
    MODEL.to_string()
}

/// ik's node dumps: the batch set and the decode-step sets. No gate reads
/// them yet: the mimo2 gates do not exist, so `consumers` is empty.
pub static IK: Family = Family {
    name: "ik-mimo2",
    sets: &[BATCH, STEP4, STEP4_EVERY_NODE, D1K, D4096],
    resolve: None,
    recipe: "just dump-ref-mimo2 [VARIANT]",
    identity: Identity::Manifest,
    arch: Some(ARCH),
    build: Some(Build::Is(IK_BUILD)),
    runs: Some(model),
    draft_runs: None,
    consumers: &[],
};

/// The architecture's families, in the order `refset-check` lists them.
pub static FAMILIES: &[&Family] = &[&IK];

#[cfg(test)]
mod tests;
