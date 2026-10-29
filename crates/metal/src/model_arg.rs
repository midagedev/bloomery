//! How a Metal binary is told its model: `-m PATH`, `--hf REPO`, or `$BLOOMERY_REF_MODEL` for the
//! binary that reads it. A path is a GGUF file (its first shard) or a checkpoint folder;
//! [`model_hf::ModelFile::open`] picks by what it is, so an MLX folder runs as downloaded. A repo
//! is fetched into the cache ([`hf::fetch`]): a repo with a GGUF set gives the set's first shard,
//! a safetensors checkpoint gives its cache folder. The model is named once: a path beside
//! `--hf`, a flag given twice, and a flag that differs from the variable are each refused by name
//! with both values ([`hf::source`]).

use std::path::PathBuf;

use hf::RepoRef;
use hf::fetch::{Client, Event, Sources};
use hf::source::{self, Source};

/// The model a binary opens: its path, and the repo it was fetched from.
#[derive(Debug, PartialEq, Eq)]
pub struct Model {
    /// A GGUF file's first shard, or a checkpoint folder.
    pub path: PathBuf,
    /// The `--hf` repo (`owner/name`, no quant); `None` for a path.
    pub repo: Option<String>,
}

/// The source that `given`, the value flags a binary parsed as (flag, value) in order, and `env`
/// (`$BLOOMERY_REF_MODEL`, `None` for a binary that reads none) name. `path_flags` spell the
/// path; the other flags of `given` are not the model's and are left alone. A source named
/// twice is refused by name, and none is refused with `how` (what to give).
pub fn named(
    given: &[(String, String)],
    path_flags: &[&str],
    env: Option<&str>,
    how: &str,
) -> Result<Source, String> {
    let args: Vec<String> = given
        .iter()
        .filter(|(flag, _)| flag == "--hf" || path_flags.contains(&flag.as_str()))
        .flat_map(|(flag, value)| [flag.clone(), value.clone()])
        .collect();
    let (flags, _) = source::take(&args, path_flags).map_err(|e| e.to_string())?;
    source::resolve(&flags, env)
        .map_err(|e| e.to_string())?
        .ok_or_else(|| format!("no model named: give {how}"))
}

/// The model `source` names: a path as given, a repo fetched into the cache (`$BLOOMERY_CACHE`,
/// else `~/.cache/bloomery/hf`; offline under `$HF_HUB_OFFLINE`), what the fetch does printed on
/// stderr a line an event ([`event_line`]).
pub fn resolve(source: &Source) -> Result<Model, String> {
    match source {
        Source::File(path) => Ok(Model {
            path: PathBuf::from(path),
            repo: None,
        }),
        Source::Hf(repo) => {
            let root = hf::cache_root(std::env::var_os("BLOOMERY_CACHE"), std::env::var_os("HOME"))
                .map_err(|e| format!("--hf {repo}: {e}"))?;
            let client = Client::new(root).map_err(|e| format!("--hf {repo}: {e}"))?;
            fetch(&client, repo, &mut |line| eprintln!("{line}"))
        }
        Source::Named(name) => Err(format!(
            "{name} names a catalog model, which this binary does not pick: name the model with \
             -m PATH or --hf REPO"
        )),
    }
}

/// `repo` through `client`: the GGUF set's first shard, or the checkpoint's folder. The draft
/// handling of the CUDA seats has no Metal reader, so none is fetched.
fn fetch(client: &Client, repo: &RepoRef, out: &mut dyn FnMut(&str)) -> Result<Model, String> {
    let files = client
        .resolve(repo, Sources::GgufOrCheckpoint, None, &mut |e| {
            out(&event_line(e));
        })
        .map_err(|e| format!("--hf {repo}: {e}"))?;
    let path = files
        .into_iter()
        .next()
        .ok_or_else(|| format!("--hf {repo}: the picked set has no file"))?;
    Ok(Model {
        path,
        repo: Some(repo.repo.clone()),
    })
}

/// A fetch's event as one stderr line, `hf <what> key=value …`.
fn event_line(e: &Event<'_>) -> String {
    match e {
        Event::Set { repo, set } => format!(
            "hf set repo={repo} set={} tag={} files={} bytes={}",
            set.name,
            if set.tag.is_empty() { "-" } else { &set.tag },
            set.files.len(),
            set.bytes()
        ),
        Event::Checkpoint { repo, files, bytes } => {
            format!("hf checkpoint repo={repo} files={files} bytes={bytes}")
        }
        Event::File {
            file,
            bytes,
            state,
            from,
        } => format!(
            "hf file file={file} bytes={bytes} state={} from={from}",
            state.name()
        ),
        Event::Offline { repo, why } => format!("hf offline repo={repo} reason={why}"),
        Event::DraftSkip { file, why } => format!("hf draft skipped file={file} reason={why}"),
        Event::Progress { file, have, of } => {
            format!("hf progress file={file} have={have} of={of}")
        }
        Event::Done {
            file,
            fetched,
            check,
            digest,
        } => format!(
            "hf done file={file} fetched_bytes={fetched} check={} digest={digest}",
            check.name()
        ),
    }
}

/// The name a server answers to when `--alias` and the file's `general.name` give none: the
/// folder's (or the file's own `general.name`, as [`model_hf::ModelFile::name`] gives it), else
/// the `--hf` repo's name, else the architecture's.
#[must_use]
pub fn default_alias(named: Option<String>, repo: Option<&str>, arch: &str) -> String {
    named
        .or_else(|| repo.and_then(|r| r.rsplit('/').next()).map(str::to_owned))
        .unwrap_or_else(|| arch.to_owned())
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    use super::*;

    fn given(flags: &[(&str, &str)]) -> Vec<(String, String)> {
        flags
            .iter()
            .map(|(f, v)| ((*f).to_owned(), (*v).to_owned()))
            .collect()
    }

    const PATHS: &[&str] = &["-m", "--model"];
    const HOW: &str = "-m PATH or --hf REPO";

    #[test]
    fn a_folder_path_is_the_model() {
        let source = named(
            &given(&[("--port", "0"), ("-m", "/x/folder")]),
            PATHS,
            None,
            HOW,
        );
        assert_eq!(source, Ok(Source::File("/x/folder".to_owned())));
        let long = named(&given(&[("--model", "/x/folder")]), PATHS, None, HOW);
        assert_eq!(long, Ok(Source::File("/x/folder".to_owned())));
    }

    #[test]
    fn a_repo_spec_is_the_model_and_a_bad_one_is_refused_by_name() {
        let source = named(
            &given(&[("--hf", "mlx-community/Qwen3-4bit")]),
            PATHS,
            None,
            HOW,
        );
        assert_eq!(
            source,
            Ok(Source::Hf(RepoRef {
                repo: "mlx-community/Qwen3-4bit".to_owned(),
                quant: None
            }))
        );
        let e = named(&given(&[("--hf", "mlx-community")]), PATHS, None, HOW).unwrap_err();
        assert!(e.contains("mlx-community"), "{e}");
    }

    #[test]
    fn a_path_and_a_repo_together_are_refused_with_both_values() {
        for flags in [
            &[("-m", "/x/folder"), ("--hf", "a/b")][..],
            &[("--hf", "a/b"), ("--model", "/x/folder")],
            &[("-m", "/x"), ("-m", "/x")],
            &[("--hf", "a/b"), ("--hf", "a/c")],
        ] {
            let e = named(&given(flags), PATHS, None, HOW).unwrap_err();
            assert!(
                e.contains("named twice") && e.contains(flags[0].1) && e.contains(flags[1].1),
                "{e}"
            );
        }
    }

    #[test]
    fn no_model_is_refused_with_what_to_give() {
        let e = named(&given(&[("--port", "0")]), PATHS, None, HOW).unwrap_err();
        assert_eq!(e, format!("no model named: give {HOW}"));
        let e = named(&[], PATHS, Some(""), HOW).unwrap_err();
        assert!(e.contains("no model named"), "{e}");
    }

    #[test]
    fn the_variable_names_a_model_a_flag_may_not_contradict() {
        let env = Some("/env/folder");
        assert_eq!(
            named(&[], PATHS, env, HOW),
            Ok(Source::File("/env/folder".to_owned()))
        );
        assert_eq!(
            named(&given(&[("-m", "/env/folder")]), PATHS, env, HOW),
            Ok(Source::File("/env/folder".to_owned()))
        );
        for flags in [&[("-m", "/other")][..], &[("--hf", "a/b")]] {
            let e = named(&given(flags), PATHS, env, HOW).unwrap_err();
            assert!(e.contains("BLOOMERY_REF_MODEL=/env/folder"), "{e}");
        }
    }

    #[test]
    fn the_alias_is_the_name_then_the_repo_then_the_architecture() {
        let name =
            |n: Option<&str>, r: Option<&str>| default_alias(n.map(str::to_owned), r, "qwen3moe");
        assert_eq!(name(Some("Qwen3-4bit"), Some("o/other")), "Qwen3-4bit");
        assert_eq!(name(None, Some("mlx-community/Qwen3-4bit")), "Qwen3-4bit");
        assert_eq!(name(None, None), "qwen3moe");
    }

    /// A cache root holding `repo`'s `files`, each beside a verified marker, so an offline client
    /// finds them as a fetch left them.
    fn cached(dir: &Path, repo: &str, files: &[&str]) -> PathBuf {
        let root = dir.join("cache");
        let at = root.join(repo);
        fs::create_dir_all(&at).unwrap();
        for f in files {
            fs::write(at.join(f), b"stored").unwrap();
            fs::write(at.join(format!("{f}.verified")), format!("{:064x}\n", 7)).unwrap();
        }
        root
    }

    /// A client that reads the cache alone: the hub is never reached.
    fn offline(root: PathBuf) -> Client {
        Client {
            base: "http://hub.invalid".to_owned(),
            token: None,
            root,
            curl: "curl-is-never-run".into(),
            offline: true,
            revision: "main".to_owned(),
        }
    }

    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("bloomery-model-arg-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn a_cached_checkpoint_repo_is_its_folder_with_nothing_fetched() {
        let dir = scratch("folder");
        let repo = "mlx-community/Tiny-4bit";
        let root = cached(
            &dir,
            repo,
            &[
                "config.json",
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
                "model.safetensors.index.json",
            ],
        );
        let r = RepoRef::parse(repo).unwrap();
        let mut lines = Vec::new();
        let model = fetch(&offline(root.clone()), &r, &mut |l| {
            lines.push(l.to_owned())
        })
        .unwrap();
        assert_eq!(model.path, root.join(repo));
        assert_eq!(model.repo.as_deref(), Some(repo));
        assert!(
            lines[0].starts_with("hf offline repo=mlx-community/Tiny-4bit reason="),
            "{lines:?}"
        );
        assert!(
            lines.contains(
                &"hf checkpoint repo=mlx-community/Tiny-4bit files=4 bytes=24".to_owned()
            ),
            "{lines:?}"
        );
        let files: Vec<&String> = lines.iter().filter(|l| l.starts_with("hf file ")).collect();
        assert_eq!(files.len(), 4, "{lines:?}");
        assert!(
            files.iter().all(|l| l.contains("state=offline-cached")),
            "{lines:?}"
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_cached_gguf_repo_is_its_first_shard_and_a_quant_picks_the_set() {
        let dir = scratch("gguf");
        let repo = "owner/Tiny-GGUF";
        let root = cached(
            &dir,
            repo,
            &[
                "Tiny-Q4_K_M.gguf",
                "Tiny-Q8_0-00001-of-00002.gguf",
                "Tiny-Q8_0-00002-of-00002.gguf",
                "README.md",
            ],
        );
        let pick = |spec: &str| {
            let r = RepoRef::parse(spec).unwrap();
            fetch(&offline(root.clone()), &r, &mut |_| {})
        };
        let model = pick("owner/Tiny-GGUF:Q8_0").unwrap();
        assert_eq!(
            model.path,
            root.join(repo).join("Tiny-Q8_0-00001-of-00002.gguf")
        );
        let model = pick("owner/Tiny-GGUF:Q4_K_M").unwrap();
        assert_eq!(model.path, root.join(repo).join("Tiny-Q4_K_M.gguf"));
        assert_eq!(model.repo.as_deref(), Some(repo));
        let e = pick(repo).unwrap_err();
        assert!(e.starts_with(&format!("--hf {repo}: ")), "{e}");
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_quant_on_a_checkpoint_repo_is_refused_by_name_and_a_missing_repo_too() {
        let dir = scratch("refused");
        let repo = "mlx-community/Tiny-4bit";
        let root = cached(&dir, repo, &["config.json", "model.safetensors.index.json"]);
        let with_quant = RepoRef::parse(&format!("{repo}:Q4_K_M")).unwrap();
        let e = fetch(&offline(root.clone()), &with_quant, &mut |_| {}).unwrap_err();
        assert!(
            e.starts_with(&format!("--hf {repo}:Q4_K_M: ")) && e.contains("Q4_K_M"),
            "{e}"
        );
        let absent = RepoRef::parse("mlx-community/Absent").unwrap();
        let e = fetch(&offline(root), &absent, &mut |_| {}).unwrap_err();
        assert!(e.starts_with("--hf mlx-community/Absent: "), "{e}");
        fs::remove_dir_all(dir).unwrap();
    }
}
