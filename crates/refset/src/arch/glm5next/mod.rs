//! The GLM-5.3-Flash families. The node dumps are dumped from unsloth's
//! UD-Q4_K_XL split set, [`MODEL`], by the ik tree [`IK_BUILD`]
//! (`tools/ref/models/glm5next.sh`).

use crate::family::{Build, Family, Identity};

/// The first shard of the split set every glm5next set is dumped from and
/// the tree runs: the identity each set's `# model` line states. The family
/// owns the path until the model crate reads this architecture; a set of
/// the file moved elsewhere is stale by this line, not by its name.
pub const MODEL: &str =
    "/models/GLM-5.3-Flash-UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf";

/// The ik tree every glm5next oracle family is dumped from, and the one
/// place a re-take of the oracle changes it.
// PIN(2026-09-27): /home/user/ik-idxkey at its HEAD, whose libllama.so builds
// glm5next and which the installed dump_ref links against: the build the
// deepseek41 and qwen35moe families pin.
pub const IK_BUILD: &str = "db517b69";

/// The architecture every glm5next manifest names in its `# arch` line.
pub const ARCH: &str = "glm5next";

/// The 5-token prefill, ik on the CPU.
pub const BATCH: &str = "ref_glm5next";
/// Step 4 after a fused prefill of the batch set's first four ids.
pub const STEP4: &str = "ref_glm5next_step4";
/// [`STEP4`] after a prefill run node by node under the dumped schedule.
pub const STEP4_EVERY_NODE: &str = "ref_glm5next_step4_every_node";
/// Step 1,024 of the prose after a fused prefill.
pub const D1K: &str = "ref_glm5next_d1k";

/// The decode-step sets (the model profile's `ref_step_variant`), by
/// position.
pub const STEP_SETS: &[&str] = &[STEP4, STEP4_EVERY_NODE, D1K];

/// [`MODEL`], as a family's `runs`.
fn model() -> String {
    MODEL.to_string()
}

/// ik's node dumps: the batch set and the decode-step sets, which the
/// end-to-end gate reads.
pub static IK: Family = Family {
    name: "ik-glm5next",
    sets: &[BATCH, STEP4, STEP4_EVERY_NODE, D1K],
    resolve: None,
    recipe: "just dump-ref-glm5next [VARIANT]",
    identity: Identity::Manifest,
    arch: Some(ARCH),
    build: Some(Build::Is(IK_BUILD)),
    runs: Some(model),
    draft_runs: None,
    consumers: &["gate-gpu-glm5next-e2e"],
};

/// The architecture's families, in the order `refset-check` lists them.
pub static FAMILIES: &[&Family] = &[&IK];

#[cfg(test)]
mod tests;
