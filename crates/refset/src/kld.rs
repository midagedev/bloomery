//! ik's KL-divergence base runs (`tools/ref/ik-ppl.sh --kld-base`): a base
//! is `<tag>.kld` (ik's binary format, read by `bloomery_gpu_gates::kld`) and
//! the log of the run that wrote it, `<tag>.log`, beside it. The log states
//! the model file the run read on its own line, `  model <path>`, before
//! anything ik prints, and ends with the run's result line, `ppl tag=<tag> …
//! head=<rev> …`, whose `head` is the ik tree's.

use crate::RefError;
use crate::family::Family;
use std::path::{Path, PathBuf};

/// The prefix of the log line that states the model file.
pub const MODEL_LINE: &str = "  model ";

/// What a base's run log states about the run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Run {
    /// The log: `<tag>.log` beside the base.
    pub log: PathBuf,
    /// The model file the run read (`  model <path>`).
    pub model: Option<String>,
    /// The ik tree's head, from the last result line of `tag`.
    pub head: Option<String>,
}

/// The log of the run that wrote base file `kld`, run `tag`: `<tag>.log` in
/// the base's directory.
#[must_use]
pub fn log_of(kld: &Path, tag: &str) -> PathBuf {
    kld.with_file_name(format!("{tag}.log"))
}

/// What the log of run `tag`, beside base file `kld`, states.
pub fn run_of(kld: &Path, tag: &str) -> Result<Run, RefError> {
    let log = log_of(kld, tag);
    let raw = std::fs::read(&log)
        .map_err(|e| RefError::missing(&log, format!("cannot read {}: {e}", log.display())))?;
    // ik's model-load lines carry bytes that are not UTF-8; the lines read
    // here are ASCII.
    let text = String::from_utf8_lossy(&raw);
    let model = text
        .lines()
        .find_map(|l| l.strip_prefix(MODEL_LINE))
        .map(str::to_string);
    let head_line = format!("ppl tag={tag} ");
    let head = text
        .lines()
        .rev()
        .find(|l| l.starts_with(&head_line))
        .and_then(|l| l.split_whitespace().find_map(|t| t.strip_prefix("head=")))
        .map(str::to_string);
    Ok(Run { log, model, head })
}

/// Base file `kld` of run `tag` against its family's row: the model file its
/// run read must be the one the tree runs ([`RefError::Stale`]), and the ik
/// tree its run's head ([`RefError::Foreign`]).
pub fn check(kld: &Path, tag: &str, family: &Family) -> Result<Run, RefError> {
    let run = run_of(kld, tag)?;
    family.check_file(kld, "`  model <path>` log", run.model.as_deref())?;
    family.check_build(kld, run.head.as_deref())?;
    Ok(run)
}

#[cfg(test)]
mod tests {
    use super::check;
    use crate::RefError;
    use crate::family::{Build, Family, Identity};

    fn runs() -> Result<String, RefError> {
        Ok("/models/P/M-00001-of-00009.gguf".to_string())
    }

    static FAMILY: Family = Family {
        name: "test-kld",
        sets: &[],
        resolve: None,
        recipe: "",
        identity: Identity::RunLog,
        arch: None,
        build: Some(Build::Is("b0")),
        runs: Some(runs),
        draft_runs: None,
        consumers: &[],
    };

    /// The model file is the log's `  model` line, not the witness blocks'
    /// `model:` or the result line's basename; the build is the result
    /// line's head. A run of another file is `Stale`, one of another tree
    /// `Foreign`, and a base whose log is gone is `Missing`.
    #[test]
    fn a_base_of_another_model_is_stale() {
        let dir = std::env::temp_dir().join(format!("bloomery-kld-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("{e}"));
        let kld = dir.join("base-a.kld");
        let log = |model: &str, head: &str| {
            let mut text = format!(
                "ik-ppl.sh: tag base-a, tree /t at {head}\n  llama-perplexity sha256=00\n  model {model}\n\
                 model: /models/Z/M-00001-of-00009.gguf\n"
            )
            .into_bytes();
            // A line of ik's that is not UTF-8.
            text.extend_from_slice(&[0xff, b' ', b'i', b'k', b'\n']);
            text.extend_from_slice(
                format!(
                    "ppl tag=base-a tree=/t head={head} dirty_files=0 model=M-00001-of-00009.gguf \
                     ctx=512 ppl=2.0\n"
                )
                .as_bytes(),
            );
            std::fs::write(dir.join("base-a.log"), text).unwrap_or_else(|e| panic!("{e}"));
        };
        log("/models/P/M-00001-of-00009.gguf", "b0");
        let run = check(&kld, "base-a", &FAMILY).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(run.head.as_deref(), Some("b0"));
        log("/models/Q/M-00001-of-00009.gguf", "b0");
        match check(&kld, "base-a", &FAMILY) {
            Err(RefError::Stale {
                set, dumped_from, ..
            }) => {
                assert!(set.ends_with("base-a.kld"), "{set}");
                assert_eq!(dumped_from, "/models/Q/M-00001-of-00009.gguf");
            }
            other => panic!("another file: {other:?}"),
        }
        log("/models/P/M-00001-of-00009.gguf", "b1");
        assert!(matches!(
            check(&kld, "base-a", &FAMILY),
            Err(RefError::Foreign { .. })
        ));
        assert!(matches!(
            check(&kld, "base-b", &FAMILY),
            Err(RefError::Missing { .. })
        ));
        std::fs::remove_dir_all(&dir).unwrap_or_else(|e| panic!("{e}"));
    }
}
