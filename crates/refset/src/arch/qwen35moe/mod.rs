//! The Qwen3.6-35B-A3B families. The node dumps are dumped from lmstudio's
//! Q4_K_M file, [`MODEL`], by the ik tree [`IK_BUILD`]
//! (`tools/ref/models/qwen35moe.sh`).

use crate::family::{Build, Family, Identity};

/// The file every qwen35moe set is dumped from and the tree runs. The
/// family owns the path until the model crate reads this architecture.
pub const MODEL: &str = "/models/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-Q4_K_M.gguf";

/// The ik tree every qwen35moe oracle family is dumped from, and the one
/// place a re-take of the oracle changes it.
// PIN(2026-09-27): /home/user/ik-idxkey at its HEAD, the tree the installed
// dump_ref links against.
pub const IK_BUILD: &str = "db517b69";

/// The architecture every qwen35moe manifest names in its `# arch` line.
pub const ARCH: &str = "qwen35moe";

/// The 5-token prefill, ik on the CPU.
pub const BATCH: &str = "ref_qwen35moe";
/// Step 4 after a prefill run node by node under the dumped schedule.
pub const STEP4: &str = "ref_qwen35moe_step4_every_node";
/// Step 1,024 of the prose after a fused prefill.
pub const D1K: &str = "ref_qwen35moe_d1k";

/// The decode-step sets (the model profile's `ref_step_variant`), by
/// position.
pub const STEP_SETS: &[&str] = &[STEP4, D1K];

/// [`MODEL`], as a family's `runs`.
fn model() -> String {
    MODEL.to_string()
}

/// ik's node dumps: the batch set and the decode-step sets.
pub static IK: Family = Family {
    name: "ik-qwen35moe",
    sets: &[BATCH, STEP4, D1K],
    resolve: None,
    recipe: "just dump-ref-qwen35moe [VARIANT]",
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
