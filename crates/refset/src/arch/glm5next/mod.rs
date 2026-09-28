//! The GLM-5.3-Flash families. The node dumps are dumped from unsloth's
//! UD-Q4_K_XL split set, [`MODEL`], by the ik tree [`IK_BUILD`]
//! (`tools/ref/models/glm5next.sh`): the batch set and the decode-step sets
//! within the positions a latent layer attends whole ([`IK`]), and the DSA
//! sets past them, dumped with ik's k-pool indexer on ([`IK_DSA`]); the MTP
//! draft set from the same file by the ik tree that carries the MTP graph,
//! [`MTP_BUILD`].

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

/// Step 3,070 of the prose after a fused prefill, ik's k-pool indexer on
/// (`--dsa`): past the 2,051 positions a latent layer keeps whole.
pub const D3K_DSA: &str = "ref_glm5next_d3kdsa";
/// Step 16,382 of the prose after a fused prefill, `--dsa`: the deepest
/// context the program serves (`place::ORACLE_POSITIONS`).
pub const D16K_DSA: &str = "ref_glm5next_d16kdsa";

/// The `--dsa` decode-step sets, by position.
pub const DSA_SETS: &[&str] = &[D3K_DSA, D16K_DSA];

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

/// The ik tree the MTP draft set is dumped from: ik's glm5next MTP graph
/// merged onto an ancestor of [`IK_BUILD`], whose tree has no MTP graph for
/// this architecture. The profile's `GLM_MTP_SHA` names the same commit.
// PIN(2026-09-28): /home/user/ik-glm53-mtp at its HEAD, the tree whose
// llama-server measured the MTP accept rate on this box.
pub const MTP_BUILD: &str = "425a2c1d";

/// The MTP draft set: every node ik's MTP context computes while the target
/// decodes 64 positions after the first 64 ids of the prose, one draft token
/// a round (`tools/ref/dump-mtp.sh`).
pub const MTP_SET: &str = "ref-mtp/prose64_n64_k1";

/// ik's MTP draft sets, dumped from [`MODEL`], which carries the NextN block.
pub static MTP: Family = Family {
    name: "mtp-glm5next",
    sets: &[MTP_SET],
    resolve: None,
    recipe: "just dump-ref-mtp-glm5next",
    identity: Identity::MtpManifest,
    arch: Some(ARCH),
    build: Some(Build::Is(MTP_BUILD)),
    runs: Some(model),
    draft_runs: None,
    consumers: &[],
};

/// ik's node dumps with its k-pool indexer on (`--dsa`): the selector's
/// sets past the dense limit, which the selector gate reads. The same file
/// and build as [`IK`]; the gate refuses a set of this family whose
/// dumper's command line lacks `--dsa` or which holds no indexer node.
pub static IK_DSA: Family = Family {
    name: "ik-glm5next-dsa",
    sets: DSA_SETS,
    resolve: None,
    recipe: "just dump-ref-glm5next [VARIANT]",
    identity: Identity::Manifest,
    arch: Some(ARCH),
    build: Some(Build::Is(IK_BUILD)),
    runs: Some(model),
    draft_runs: None,
    consumers: &["gate-gpu-glm-sel", "gate-gpu-glm5next-e2e"],
};

/// The architecture's families, in the order `refset-check` lists them.
pub static FAMILIES: &[&Family] = &[&IK, &MTP, &IK_DSA];

#[cfg(test)]
mod tests;
