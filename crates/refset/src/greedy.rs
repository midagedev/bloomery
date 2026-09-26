//! ik's greedy continuations (`tools/ref/argmax_ref.cpp --gen`, run by
//! `tools/ref/ik-greedy.sh`): which model file a greedy tsv was dumped from.
//! The tsv opens with `argmax_ref`'s own line,
//! `# argmax_ref\tik_llama.cpp\tmodel=<path>\tgen=<n>\tprefill=<how>`; its
//! rows are read by `bloomery_gpu_gates::prompts::read_greedy`.

use crate::RefError;
use crate::family::Family;
use std::path::Path;

/// The line a greedy tsv states its model file in.
pub const MODEL_LINE: &str = "# argmax_ref";

/// The model file greedy tsv `path` states in its `# argmax_ref` line
/// (`model=<path>`), `None` when the file has no such line or the line no
/// `model=` field.
pub fn dumped_from(path: &Path) -> Result<Option<String>, RefError> {
    let text = std::fs::read_to_string(path).map_err(|e| {
        RefError::missing(path, format!("greedy: cannot read {}: {e}", path.display()))
    })?;
    Ok(text
        .lines()
        .find(|l| l.split('\t').next() == Some(MODEL_LINE))
        .and_then(|l| l.split('\t').find_map(|f| f.strip_prefix("model=")))
        .map(str::to_string))
}

/// Greedy tsv `path` against its family's row: the model file its
/// `# argmax_ref` line states must be the one the tree runs
/// ([`RefError::Stale`] otherwise).
pub fn check(path: &Path, family: &Family) -> Result<(), RefError> {
    family.check_file(
        path,
        "`# argmax_ref … model=`",
        dumped_from(path)?.as_deref(),
    )
}

#[cfg(test)]
mod tests {
    use super::{check, dumped_from};
    use crate::RefError;
    use crate::family::{Family, Identity};

    fn runs() -> String {
        "/models/P/M-00001-of-00009.gguf".to_string()
    }

    static FAMILY: Family = Family {
        name: "test-greedy",
        sets: &[],
        resolve: None,
        recipe: "",
        identity: Identity::ArgmaxHeader,
        arch: None,
        build: None,
        runs: Some(runs),
        draft_runs: None,
        consumers: &[],
    };

    /// The model file is read from `argmax_ref`'s line, whatever follows
    /// it; a tsv of another file, or one that names none, is `Stale`.
    #[test]
    fn a_greedy_file_of_another_model_is_stale() {
        let path = std::env::temp_dir().join(format!("bloomery-greedy-{}.tsv", std::process::id()));
        let file = |model: &str| {
            let text = format!(
                "# argmax_ref\tik_llama.cpp\t{model}\tgen=64\tprefill=step\n\
                 #id\tn_tokens\targmax\ttop5_ids\ttop5_logits\tgen_ids\tgen_margins\n\
                 7\t4\t1031\t1031,1,2,3,4\t2.0,1.0,0.5,0.2,0.1\t1031,515\t0.9,2.0\n"
            );
            std::fs::write(&path, text).unwrap_or_else(|e| panic!("{e}"));
        };
        file("model=/models/P/M-00001-of-00009.gguf");
        assert_eq!(
            dumped_from(&path).ok().flatten().as_deref(),
            Some(runs().as_str())
        );
        check(&path, &FAMILY).unwrap_or_else(|e| panic!("{e}"));
        file("model=/models/Q/M-00001-of-00009.gguf");
        match check(&path, &FAMILY) {
            Err(RefError::Stale { dumped_from, .. }) => {
                assert_eq!(dumped_from, "/models/Q/M-00001-of-00009.gguf");
            }
            other => panic!("another file: {other:?}"),
        }
        file("M-00001-of-00009.gguf");
        assert!(matches!(check(&path, &FAMILY), Err(RefError::Stale { .. })));
        std::fs::remove_file(&path).unwrap_or_else(|e| panic!("{e}"));
    }
}
