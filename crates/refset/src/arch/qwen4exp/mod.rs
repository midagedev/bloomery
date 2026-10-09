//! The Qwen3.8-Flash-Next families. The node dumps are dumped from
//! unsloth's UD-Q4_K_XL split set, [`MODEL`], by the ik tree [`IK_BUILD`]
//! (`tools/ref/models/qwen4exp.sh`); the MTP draft set from the same file
//! and the shared draft file by the ik tree that carries the MTP graph
//! ([`mtp`]). The fixture tier's mirror of each is in [`fixture`].

use crate::RefError;
use crate::family::{Build, Family, Identity};

pub mod fixture;
pub mod mtp;

/// The first shard of the split set every qwen4exp set is dumped from and
/// the tree runs: the identity each set's `# model` line states. The family
/// owns the path until the model crate reads this architecture; a set of
/// the file moved elsewhere is stale by this line, not by its name.
pub const MODEL: &str =
    "/models/Qwen3.8-Flash-Next/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf";

/// The ik tree every qwen4exp oracle family is dumped from, and the one
/// place a re-take of the oracle changes it.
// PIN(2026-09-27): /home/user/ik-idxkey at its HEAD, whose libllama.so builds
// qwen4exp and which the installed dump_ref links against: the build the
// deepseek41, qwen35moe and glm5next families pin.
pub const IK_BUILD: &str = "db517b69";

/// The architecture every qwen4exp manifest names in its `# arch` line.
pub const ARCH: &str = "qwen4exp";

/// The 5-token prefill, ik on the CPU.
pub const BATCH: &str = "ref_qwen4exp";
/// Step 4 after a fused prefill of the batch set's first four ids.
pub const STEP4: &str = "ref_qwen4exp_step4";
/// [`STEP4`] after a prefill run node by node under the dumped schedule.
pub const STEP4_EVERY_NODE: &str = "ref_qwen4exp_step4_every_node";
/// Step 1,024 of the prose after a fused prefill.
pub const D1K: &str = "ref_qwen4exp_d1k";
/// Step 3,000 of the prose after a fused prefill: the QSA layers attend to
/// the cells their indexer selects, not to every cell.
pub const D3K: &str = "ref_qwen4exp_d3k";

/// The deepest context our numbers are held to ik's at: [`D3K`]'s decode
/// step at position 3,000, which reads 3,001 positions. A load prints it
/// beside the context it serves; it does not bound that context.
pub const VERIFIED_POSITIONS: u64 = 3001;

/// The decode-step sets (the model profile's `ref_step_variant`), by
/// position.
pub const STEP_SETS: &[&str] = &[STEP4, STEP4_EVERY_NODE, D1K, D3K];

/// [`MODEL`], as a family's `runs`.
fn model() -> Result<String, RefError> {
    Ok(MODEL.to_string())
}

/// ik's node dumps: the batch set and the decode-step sets.
pub static IK: Family = Family {
    name: "ik-qwen4exp",
    sets: &[BATCH, STEP4, STEP4_EVERY_NODE, D1K, D3K],
    resolve: None,
    recipe: "just dump-ref-qwen4exp [VARIANT]",
    identity: Identity::Manifest,
    arch: Some(ARCH),
    build: Some(Build::Is(IK_BUILD)),
    runs: Some(model),
    draft_runs: None,
    consumers: &["gate-gpu-qwen4exp-e2e"],
};

/// The architecture's families, in the order `refset-check` lists them.
pub static FAMILIES: &[&Family] = &[&IK, &mtp::MTP, &fixture::IK, &fixture::MTP];

#[cfg(test)]
mod tests;
