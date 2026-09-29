//! The Qwen3.8-Flash-Next MTP draft family: ik's MTP context over the
//! shared draft file [`DRAFT`] beside the target [`super::MODEL`], dumped by
//! the ik tree that carries the qwen4exp MTP graph, [`MTP_BUILD`]
//! (`tools/ref/dump-mtp.sh`, the qwen4exp profile). Unlike GLM's, whose
//! NextN block is the target file's, this draft is a file of its own that
//! borrows the target's `token_embd` and `output`: a set states both files,
//! `# model` and `# draft_model`.

use super::{ARCH, MODEL};
use crate::RefError;
use crate::family::{Build, Family, Identity};

/// The shared MTP draft file every set of the family is dumped with and
/// the tree runs: its one layer, no embedding and no output of its own.
pub const DRAFT: &str = "/models/Qwen3.8-Flash-Next/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf";

/// The ik tree the MTP draft set is dumped from: ik's glm5next MTP graph
/// merged onto the upstream commit whose qwen4exp graph builds the MTP
/// layer and loads a shared draft beside the target, so one dumper serves
/// both architectures.
// PIN(2026-09-30): /home/user/ik-glm53-mtp at its HEAD, glm5next's
// MTP_BUILD; its src/graphs/build_qwen4exp.cpp is upstream's with `is_mtp`.
pub const MTP_BUILD: &str = "425a2c1d";

/// The MTP draft set: every node ik's MTP context computes while the target
/// decodes 64 positions after the first 64 ids of the qwen4exp prose, one
/// draft token a round.
pub const MTP_SET: &str = "ref-mtp/qwen4exp_prose64_n64_k1";

/// [`MODEL`], as the family's `runs`.
fn model() -> String {
    MODEL.to_string()
}

/// [`DRAFT`], as the family's `draft_runs`.
fn draft() -> Result<String, RefError> {
    Ok(DRAFT.to_string())
}

/// ik's MTP draft sets of the target and the shared draft file.
pub static MTP: Family = Family {
    name: "mtp-qwen4exp",
    sets: &[MTP_SET],
    resolve: None,
    recipe: "just dump-ref-mtp-qwen4exp",
    identity: Identity::MtpManifest,
    arch: Some(ARCH),
    build: Some(Build::Is(MTP_BUILD)),
    runs: Some(model),
    draft_runs: Some(draft),
    consumers: &["gate-gpu-qwen4exp-mtp"],
};
