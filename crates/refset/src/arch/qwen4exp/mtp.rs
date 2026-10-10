//! The Qwen3.8-Flash-Next MTP draft family: ik's MTP context over the
//! shared draft file [`DRAFT`] beside the target [`super::MODEL`], dumped by
//! the ik tree that carries the qwen4exp MTP graph, [`MTP_BUILD`]
//! (`tools/ref/dump-mtp.sh`, the qwen4exp profile). Unlike GLM's, whose
//! NextN block is the target file's, this draft is a file of its own that
//! borrows the target's `token_embd` and `output`: a set states both files,
//! `# model` and `# draft_model`.

use std::path::{Path, PathBuf};

use super::{ARCH, MODEL};
use crate::RefError;
use crate::arch::BesideDraft;
pub use crate::arch::DraftFrom;
use crate::family::{Build, Family, Identity};

/// The shared MTP draft file every set of the family is dumped with and
/// the tree runs: its one layer, no embedding and no output of its own.
pub const DRAFT: &str = "/models/Qwen3.8-Flash-Next/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf";

/// The qwen4exp family's beside-file draft ([`DRAFT`], for the target files
/// that declare [`ARCH`]).
pub static BESIDE: BesideDraft = BesideDraft {
    arch: ARCH,
    path: DRAFT,
};

/// The MTP draft file a run of `target` opens: [`BESIDE`]'s pick
/// ([`BesideDraft::pick`]).
#[must_use]
pub fn draft_file(set: Option<&Path>, target: &Path) -> (PathBuf, DraftFrom) {
    BESIDE.pick(set, target)
}

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
fn model() -> Result<String, RefError> {
    Ok(MODEL.to_string())
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

#[cfg(test)]
mod tests {
    use super::{DRAFT, DraftFrom, draft_file};
    use std::path::{Path, PathBuf};

    /// The lever wins; unset, the file of the shared draft's name beside the
    /// target is the run's, and with none there the family's path is.
    #[test]
    fn draft_file_is_the_lever_then_beside_the_target_then_the_family_path() {
        let dir = std::env::temp_dir().join(format!("bloomery-mtp-draft-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
        let target = dir.join("target-00001-of-00004.gguf");
        let name = Path::new(DRAFT).file_name().expect("DRAFT names a file");
        let beside = dir.join(name);

        assert_eq!(
            draft_file(None, &target),
            (PathBuf::from(DRAFT), DraftFrom::Family)
        );
        std::fs::write(&beside, b"").unwrap_or_else(|e| panic!("{}: {e}", beside.display()));
        assert_eq!(
            draft_file(None, &target),
            (beside.clone(), DraftFrom::Beside)
        );
        let other = dir.join("elsewhere.gguf");
        assert_eq!(
            draft_file(Some(&other), &target),
            (other.clone(), DraftFrom::Lever)
        );
        std::fs::remove_dir_all(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
    }
}
