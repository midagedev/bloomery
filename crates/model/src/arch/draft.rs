//! What every MTP draft's host rule shares, and the table of each
//! architecture's rule for whether a set can run its draft
//! ([`draft_usable`]). A beside-file draft borrows the target's
//! `token_embd` and `output` ([`borrows`]) and holds its keys equal
//! ([`same`], [`same_token_ids`]). A qwen35moe draft is the target's own
//! next-token layer, not a sidecar. The readers of the draft files
//! themselves stay with their families (`qwen35moe::mtp`, `mimo2::mtp`).

use gguf::Split;
use models::Borrows;

use super::{metadata, mimo2, qwen35moe, registry};
use crate::placement::PlacementError;

/// The key that declares a file carries neither `token_embd` nor `output`.
const SHARED: &str = "nextn_shared_target_tensors";

/// The token embedding and the output matrix, by name.
pub(crate) const EMBEDDING: &str = "token_embd.weight";
pub(crate) const HEAD: &str = "output.weight";

/// Which of the target's matrices the draft file uses: those it carries none
/// of, or both when `nextn_shared_target_tensors` is true. A declaration the
/// tensors present contradict is refused.
///
/// This is the file format's rule, the same for every family: a file that
/// carries a matrix answers "carried", whatever its bytes are. Whether a
/// carried matrix is borrowed anyway because the target's bytes are equal is
/// a digest check at load, which this header-only rule cannot make.
pub(crate) fn borrows(draft: &Split) -> Result<Borrows, PlacementError> {
    let has_embedding = draft.find(EMBEDDING).is_some();
    let has_head = draft.find(HEAD).is_some();
    let declared = match draft.value(&draft.arch_key(SHARED)) {
        None => None,
        Some(v) => Some(
            v.as_bool()
                .ok_or_else(|| metadata(draft, SHARED, format!("is {v:?}, not a bool")))?,
        ),
    };
    let carried: Vec<&str> = [(EMBEDDING, has_embedding), (HEAD, has_head)]
        .iter()
        .filter(|(_, has)| *has)
        .map(|(n, _)| *n)
        .collect();
    match declared {
        Some(true) if !carried.is_empty() => Err(metadata(
            draft,
            SHARED,
            format!(
                "is true, and the file carries {}: a shared file uses the target's token_embd and output",
                carried.join(" and ")
            ),
        )),
        Some(false) if carried.len() != 2 => Err(metadata(
            draft,
            SHARED,
            format!(
                "is false, and the file carries only [{}] of token_embd and output",
                carried.join(", ")
            ),
        )),
        _ => Ok(Borrows {
            embedding: !has_embedding,
            head: !has_head,
        }),
    }
}

/// Err naming `<architecture>.<suffix>` when the draft's `value` is not the
/// target's.
pub(crate) fn same(
    draft: &Split,
    suffix: &str,
    value: usize,
    target: u32,
) -> Result<(), PlacementError> {
    if u64::try_from(value).ok() == Some(u64::from(target)) {
        return Ok(());
    }
    Err(metadata(
        draft,
        suffix,
        format!("is {value}; the target's is {target}"),
    ))
}

/// The eos and padding ids of the draft file are the target's; an id absent
/// in both is equal.
pub(crate) fn same_token_ids(draft: &Split, target: &Split) -> Result<(), PlacementError> {
    for key in [
        "tokenizer.ggml.eos_token_id",
        "tokenizer.ggml.padding_token_id",
    ] {
        let (d, t) = (token_id(draft, key)?, token_id(target, key)?);
        if d != t {
            return Err(PlacementError::Metadata {
                key: key.to_string(),
                detail: format!("is {d:?} in the draft file, {t:?} in the target's"),
            });
        }
    }
    Ok(())
}

/// `key`, an un-prefixed token id; `None` when absent.
fn token_id(split: &Split, key: &str) -> Result<Option<u64>, PlacementError> {
    match split.value(key) {
        None => Ok(None),
        Some(v) => v
            .as_unsigned()
            .map(Some)
            .ok_or_else(|| PlacementError::Metadata {
                key: key.to_string(),
                detail: format!("is {v:?}, not a token id"),
            }),
    }
}

/// Why a set cannot run its draft file ([`draft_usable`]).
#[derive(Debug, thiserror::Error)]
pub enum DraftUsableError {
    /// No rule says whether a set of this architecture can run a draft.
    #[error(
        "the file's architecture is {arch}; no rule says whether a {arch} set can run its draft"
    )]
    NoRule { arch: String },
    /// The set's plan cannot be described, or the draft file's checkpoint is
    /// not the set's.
    #[error("{0}")]
    Placement(#[from] PlacementError),
    /// A qwen4exp set whose `token_embd` or `output` the draft cannot borrow.
    #[error(transparent)]
    Qwen4Exp(#[from] qwen35moe::place::PlaceError),
}

/// One architecture's rule: the target's first shard, and the checkpoint the
/// candidate draft file is made from (`None`: the family's file serves every
/// checkpoint). The rule of each architecture that has a draft is its
/// registry row's ([`super::registry::ArchRow::draft`]); qwen35moe's draft is
/// the target file's own next-token layer. An architecture with no rule has
/// none in its row.
pub type Rule = fn(&Split, Option<&str>) -> Result<(), DraftUsableError>;

pub(super) fn qwen4exp_usable(
    target: &Split,
    checkpoint: Option<&str>,
) -> Result<(), DraftUsableError> {
    qwen35moe::mtp::target_usable(target, checkpoint).map_err(DraftUsableError::from)
}

pub(super) fn qwen35moe_usable(
    target: &Split,
    checkpoint: Option<&str>,
) -> Result<(), DraftUsableError> {
    if let Some(name) = checkpoint {
        return Err(PlacementError::Metadata {
            key: "general.name".to_string(),
            detail: format!(
                "is named {name:?} for a qwen35moe draft, which is the target file's own \
                 next-token layer"
            ),
        }
        .into());
    }
    match qwen35moe::mtp::in_file(target)? {
        qwen35moe::mtp::InFile::Draft(_) => Ok(()),
        qwen35moe::mtp::InFile::Absent => Err(PlacementError::Metadata {
            key: "qwen35moe.nextn_predict_layers".to_string(),
            detail: "is absent or names no next-token layer; the file has no in-file draft"
                .to_string(),
        }
        .into()),
        qwen35moe::mtp::InFile::Stripped { reason } => Err(PlacementError::Metadata {
            key: "qwen35moe.nextn_predict_layers".to_string(),
            detail: reason,
        }
        .into()),
        qwen35moe::mtp::InFile::Unshaped => Err(PlacementError::Metadata {
            key: "qwen35moe.nextn_predict_layers".to_string(),
            detail: "is 1 and the next-token tensors are not a draft's shapes".to_string(),
        }
        .into()),
    }
}

pub(super) fn mimo2_usable(
    target: &Split,
    checkpoint: Option<&str>,
) -> Result<(), DraftUsableError> {
    mimo2::mtp::target_usable(target, checkpoint).map_err(DraftUsableError::from)
}

/// Whether the set whose first shard is `target` can run a draft file made
/// from `checkpoint`: its architecture's rule, as its registry row holds it. An
/// architecture with no rule is refused by name.
pub fn draft_usable(target: &Split, checkpoint: Option<&str>) -> Result<(), DraftUsableError> {
    let arch = target.architecture().unwrap_or("<missing>");
    match registry::row_of(arch).and_then(|row| row.draft) {
        Some(rule) => rule(target, checkpoint),
        None => Err(DraftUsableError::NoRule {
            arch: arch.to_string(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use models::Borrows;

    use super::{EMBEDDING, HEAD, borrows, draft_usable};
    use crate::arch::synthetic::{Ty, V, header_typed};

    /// The two matrices of a draft that carries its own.
    fn own_matrices() -> Vec<(String, Vec<u64>, Ty)> {
        [EMBEDDING, HEAD]
            .iter()
            .map(|n| ((*n).to_string(), vec![64, 2], Ty::Q8_0))
            .collect()
    }

    /// The draft file's header, declared `arch`, with `kv` and `tensors`.
    fn read(
        tag: &str,
        arch: &str,
        kv: &[(&str, V)],
        tensors: &[(String, Vec<u64>, Ty)],
    ) -> Result<Borrows, String> {
        let path = header_typed(tag, arch, kv, &[], tensors);
        let draft = gguf::Split::open(&path).expect("the synthetic draft opens");
        let out = borrows(&draft).map_err(|e| e.to_string());
        let _ = std::fs::remove_file(&path);
        out
    }

    /// One layer tensor, so the file is not empty.
    fn layer() -> (String, Vec<u64>, Ty) {
        ("blk.4.attn_norm.weight".to_string(), vec![64], Ty::F32)
    }

    #[test]
    fn a_shared_file_borrows_both_matrices() {
        let kv = [("nextn_shared_target_tensors", V::Bool(true))];
        let got = read("draft-shared", "qwen4exp", &kv, &[layer()]).expect("the rule answers");
        assert_eq!(
            got,
            Borrows {
                embedding: true,
                head: true
            }
        );
    }

    #[test]
    fn an_unshared_file_borrows_neither() {
        let mut tensors = vec![layer()];
        tensors.extend(own_matrices());
        let got = read("draft-own", "qwen4exp", &[], &tensors).expect("the rule answers");
        assert_eq!(
            got,
            Borrows {
                embedding: false,
                head: false
            }
        );
    }

    /// With no declaration, each matrix the file lacks is the target's.
    #[test]
    fn absence_borrows_each_matrix() {
        let mut tensors = vec![layer()];
        tensors.extend(
            own_matrices()
                .into_iter()
                .filter(|(n, _, _)| n != EMBEDDING),
        );
        let got = read("draft-half", "qwen4exp", &[], &tensors).expect("the rule answers");
        assert_eq!(
            got,
            Borrows {
                embedding: true,
                head: false
            }
        );
    }

    #[test]
    fn a_shared_declaration_beside_the_matrices_is_refused() {
        let kv = [("nextn_shared_target_tensors", V::Bool(true))];
        let mut tensors = vec![layer()];
        tensors.extend(own_matrices().into_iter().take(1));
        let err = read("draft-contra-t", "qwen4exp", &kv, &tensors).expect_err("refused");
        let want =
            "qwen4exp.nextn_shared_target_tensors: is true, and the file carries token_embd.weight";
        assert!(err.contains(want), "want {want:?} in: {err}");
    }

    #[test]
    fn an_unshared_declaration_without_the_matrices_is_refused() {
        let kv = [("nextn_shared_target_tensors", V::Bool(false))];
        let err = read("draft-contra-f", "qwen4exp", &kv, &[layer()]).expect_err("refused");
        let want = "qwen4exp.nextn_shared_target_tensors: is false, and the file carries only []";
        assert!(err.contains(want), "want {want:?} in: {err}");
    }

    /// The rule is the file format's, not one family's: MiMo's sidecar, which
    /// carries both matrices, answers "carried" under its own key prefix.
    #[test]
    fn the_rule_reads_every_familys_prefix() {
        let mut tensors = vec![layer()];
        tensors.extend(own_matrices());
        let got = read("draft-mimo2", "mimo2", &[], &tensors).expect("the rule answers");
        assert_eq!(
            got,
            Borrows {
                embedding: false,
                head: false
            }
        );
        let kv = [("nextn_shared_target_tensors", V::Bool(true))];
        let err = read("draft-mimo2-contra", "mimo2", &kv, &tensors).expect_err("refused");
        assert!(err.contains("mimo2.nextn_shared_target_tensors"), "{err}");
    }

    /// An architecture with no rule is refused by name, not taken for a
    /// set that can run its draft.
    #[test]
    fn an_architecture_without_a_rule_is_refused_by_name() {
        let path = header_typed("draft-norule", "qwen35", &[], &[], &[layer()]);
        let target = gguf::Split::open(&path).expect("the synthetic target opens");
        let err = draft_usable(&target, None).expect_err("no rule for qwen35");
        let _ = std::fs::remove_file(&path);
        assert!(
            err.to_string()
                .contains("no rule says whether a qwen35 set can run its draft"),
            "{err}"
        );
    }
}
