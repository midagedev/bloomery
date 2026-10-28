//! What fetches: `--hf <repo>[:<quant>]` into the cache ([`hf::fetch`]) and the files beside a set.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use gguf::Split;
use hf::HfError;
use hf::fetch::{Client, Draft, Event, Sources};
use hf::source::{self, Flags, Source};
use refset::arch::BesideDraft;
use refset::arch::draft::Candidate;

use super::{model_file, set_model_file};
use crate::GateError;
use crate::record::{self, Record};

/// The `--hf` repo the model file came from, set once beside it ([`name`]).
static MODEL_REPO: OnceLock<String> = OnceLock::new();

/// The `--hf` repo the process's model file came from; `None` for a file named by path.
#[must_use]
pub fn hf_repo() -> Option<&'static str> {
    MODEL_REPO.get().map(String::as_str)
}

/// The model flags taken out of `args` ([`source::take`]: `path_flags`
/// spell the path, `--hf` the repo), and the rest.
pub fn take(args: &[String], path_flags: &[&str]) -> Result<(Flags, Vec<String>), GateError> {
    Ok(source::take(args, path_flags)?)
}

/// The cache client: its root `$BLOOMERY_CACHE`, else
/// `~/.cache/bloomery/hf`; offline under `$HF_HUB_OFFLINE`.
fn client() -> Result<Client, GateError> {
    let root = hf::cache_root(std::env::var_os("BLOOMERY_CACHE"), std::env::var_os("HOME"))?;
    Ok(Client::new(root)?)
}

/// A fetch's event as its record line on stderr.
fn print(e: &Event<'_>) {
    match e {
        Event::Set { repo, set } => Record::new(&record::HF_SET)
            .w("repo", repo)
            .w("set", &set.name)
            .w("tag", if set.tag.is_empty() { "-" } else { &set.tag })
            .u("files", set.files.len())
            .u("bytes", set.bytes())
            .eprint(),
        Event::Checkpoint { repo, files, bytes } => Record::new(&record::HF_CHECKPOINT)
            .w("repo", repo)
            .u("files", files)
            .u("bytes", bytes)
            .eprint(),
        Event::File {
            file,
            bytes,
            state,
            from,
        } => Record::new(&record::HF_FILE)
            .w("file", file)
            .u("bytes", bytes)
            .w("state", state.name())
            .u("from", from)
            .eprint(),
        Event::Offline { repo, why } => Record::new(&record::HF_OFFLINE)
            .w("repo", repo)
            .w("reason", why)
            .eprint(),
        Event::DraftSkip { file, why } => Record::new(&record::HF_DRAFT_SKIP)
            .w("file", file)
            .w("reason", why)
            .eprint(),
        Event::Progress { file, have, of } => Record::new(&record::HF_PROGRESS)
            .w("file", file)
            .u("have", have)
            .u("of", of)
            .eprint(),
        Event::Done {
            file,
            fetched,
            check,
            digest,
        } => Record::new(&record::HF_DONE)
            .w("file", file)
            .u("fetched_bytes", fetched)
            .w("check", check.name())
            .w("digest", digest)
            .eprint(),
    }
}

/// A draft file's usability rule, [`draft_usable`] closed over its row and
/// file.
type Usable = Box<dyn Fn(&Path) -> Result<(), String>>;

/// Whether the set whose first shard is `first` can run the MTP draft file
/// `candidate` of `row`: the set declares the row's architecture, then that
/// architecture's rule (`model::arch::draft::draft_usable`: the plan's own
/// rule, and for a file a checkpoint, the set being made from that
/// checkpoint), why not as that rule says it; a shard the plan cannot
/// describe runs no draft either. A row of an architecture with no rule is
/// refused by name, and its draft is not fetched: a family adds its row to
/// `refset::arch::beside_drafts` and its rule to that table.
fn draft_usable(row: &BesideDraft, candidate: &Candidate, first: &Path) -> Result<(), String> {
    let split = Split::open(first).map_err(|e| format!("open {}: {e}", first.display()))?;
    let arch = split.architecture().unwrap_or("<missing>");
    if arch != row.arch {
        return Err(format!(
            "the set's architecture is {arch}, and the draft {} is {}'s",
            candidate.name(),
            row.arch
        ));
    }
    model::arch::draft::draft_usable(&split, candidate.checkpoint).map_err(|e| e.to_string())
}

/// The local path of what `flags` and `env` (the gates' variable, `None`
/// for a binary that reads none) name: a path as given, a repo's set
/// fetched and checked (its first shard), the family's MTP draft beside it
/// when the repo holds the file and the set can run it ([`draft_usable`]; an
/// `hf draft skipped` record says why not), each file of a row that has one a
/// checkpoint offered on its own. `sources` names what this caller can open; unsupported
/// checkpoint repos are refused before downloading files. `None` when nothing names one.
pub fn resolve(
    flags: &Flags,
    env: Option<&str>,
    sources: Sources,
) -> Result<Option<PathBuf>, GateError> {
    record::require(record::groups::fetch::FETCH, "model_file::resolve");
    record::require(record::groups::fetch::SET, "model_file::resolve");
    match source::resolve(flags, env)? {
        None => Ok(None),
        Some(Source::File(p)) => Ok(Some(PathBuf::from(p))),
        Some(Source::Hf(r)) => {
            let files: Vec<(&BesideDraft, Candidate)> = refset::arch::beside_drafts()
                .iter()
                .flat_map(|&row| row.candidates().into_iter().map(move |c| (row, c)))
                .collect();
            let usable: Vec<Usable> = files
                .iter()
                .map(|&(row, candidate)| -> Usable {
                    Box::new(move |first: &Path| draft_usable(row, &candidate, first))
                })
                .collect();
            let drafts: Vec<Draft<'_>> = files
                .iter()
                .zip(&usable)
                .map(|((_, candidate), usable)| Draft {
                    name: candidate.name(),
                    usable: usable.as_ref(),
                })
                .collect();
            let files = client()?.resolve_drafts(&r, sources, &drafts, &mut print)?;
            Ok(Some(files.into_iter().next().ok_or_else(|| {
                format!("--hf {r}: the picked set has no file")
            })?))
        }
    }
}

/// The files of `repo` at exactly `paths`, fetched and checked; their local
/// paths in that order.
pub fn fetch_exact(repo: &str, paths: &[&str]) -> Result<Vec<PathBuf>, GateError> {
    record::require(record::groups::fetch::FETCH, "model_file::fetch_exact");
    Ok(client()?.resolve_exact(repo, paths, &mut print)?)
}

/// Named embedding prefixes beside a local model, or in its HF repo or single base model.
pub fn sentence_prompts(
    flags: &Flags,
    model: &Path,
) -> Result<std::collections::BTreeMap<String, String>, GateError> {
    const NAME: &str = "config_sentence_transformers.json";
    let path = if flags.path.is_none()
        && let Some(repo) = &flags.hf
    {
        record::require(record::groups::fetch::FETCH, "model_file::sentence_prompts");
        let repo = hf::RepoRef::parse(repo)?.repo;
        let c = client()?;
        let paths = match c.resolve_exact(&repo, &[NAME], &mut print) {
            Ok(paths) => paths,
            Err(e) if no_file(&e) => {
                let Some(base) = model_card(&repo)?.as_deref().and_then(hf::card::base_model)
                else {
                    return Ok(std::collections::BTreeMap::default());
                };
                let base = hf::RepoRef::parse(&base)?.repo;
                match c.resolve_exact(&base, &[NAME], &mut print) {
                    Ok(paths) => paths,
                    Err(e) if no_file(&e) => return Ok(std::collections::BTreeMap::default()),
                    Err(e) => return Err(e.into()),
                }
            }
            Err(e) => return Err(e.into()),
        };
        paths
            .into_iter()
            .next()
            .ok_or("the prompt config fetch returned no file")?
    } else {
        model.with_file_name(NAME)
    };
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(e)
            if e.kind() == std::io::ErrorKind::NotFound
                && (flags.path.is_some() || flags.hf.is_none()) =>
        {
            return Ok(std::collections::BTreeMap::default());
        }
        Err(e) => return Err(format!("{}: {e}", path.display()).into()),
    };
    hf::sentence_transformers::prompts(&text).map_err(|e| format!("{}: {e}", path.display()).into())
}

/// The repo the model card of `repo` (its `README.md`, fetched and checked
/// as any file of it) says `repo` quantizes ([`hf::card::quantizes`]);
/// `None` when the repo has no card or its card says no such thing.
pub fn quantized_from(repo: &str) -> Result<Option<String>, GateError> {
    record::require(record::groups::fetch::FETCH, "model_file::quantized_from");
    Ok(model_card(repo)?.as_deref().and_then(hf::card::quantizes))
}

fn no_file(error: &HfError) -> bool {
    matches!(error, HfError::NoFile { .. })
        || matches!(error, HfError::Offline { cache, .. } if matches!(cache.as_ref(), HfError::NoFile { .. }))
}

fn model_card(repo: &str) -> Result<Option<String>, GateError> {
    let card = match client()?.resolve_exact(repo, &["README.md"], &mut print) {
        Ok(paths) => paths
            .into_iter()
            .next()
            .ok_or_else(|| format!("{repo}: the card's fetch returned no file"))?,
        Err(e) if no_file(&e) => return Ok(None),
        Err(e) => return Err(e.into()),
    };
    let text = std::fs::read_to_string(&card).map_err(|e| format!("{}: {e}", card.display()))?;
    Ok(Some(text))
}

/// The model file `flags` name, resolved beside `$BLOOMERY_REF_MODEL`
/// ([`resolve`]) and set as the process's ([`set_model_file`]) when they
/// name one. A checkpoint repo requires the `checkpoint` build feature before any file fetch.
pub fn name(flags: &Flags) -> Result<(), GateError> {
    record::require(record::groups::fetch::FETCH, "model_file::name");
    record::require(record::groups::fetch::SET, "model_file::name");
    let env = std::env::var("BLOOMERY_REF_MODEL").ok();
    let sources = if cfg!(feature = "checkpoint") {
        Sources::GgufOrCheckpoint
    } else {
        Sources::Gguf
    };
    if *flags != Flags::default()
        && let Some(path) = resolve(flags, env.as_deref(), sources)?
    {
        set_model_file(path)?;
        // A path flag wins over `--hf` ([`source::resolve`]), so the file is the repo's only
        // without one.
        if let (None, Some(h)) = (&flags.path, &flags.hf) {
            let repo = hf::RepoRef::parse(h)?.repo;
            MODEL_REPO.set(repo).map_err(|r| {
                format!("the model's --hf repo is set already, and a second set names {r}")
            })?;
        }
    }
    Ok(())
}

/// The image projector the `--hf` repo holds beside the process's model file, a set's first
/// shard in the cache (`<root>/<owner>/<name>/<path in the repo>`, which gives the cache root and
/// the model's path in the repo): the listing's `mmproj` sibling of the model
/// ([`hf::mmproj_sibling`], through [`Client::mmproj_path`]), fetched and checked. `None` when
/// the model file was named by path, or the repo holds no projector.
pub fn hf_projector() -> Result<Option<PathBuf>, GateError> {
    let (Some(repo), Some(model)) = (hf_repo(), model_file()) else {
        return Ok(None);
    };
    let in_repo = || {
        format!(
            "--hf {repo}: the model file {} is not under the repo's cache directory",
            model.display()
        )
    };
    let at = model
        .ancestors()
        .find(|a| a.ends_with(repo))
        .ok_or_else(in_repo)?;
    let rel = model
        .strip_prefix(at)?
        .to_str()
        .ok_or_else(|| format!("{} is not UTF-8", model.display()))?
        .replace(std::path::MAIN_SEPARATOR, "/");
    let root = at
        .ancestors()
        .nth(repo.split('/').count())
        .ok_or_else(in_repo)?;
    let Some(path) = Client::new(root.to_path_buf())?.mmproj_path(repo, &rel)? else {
        return Ok(None);
    };
    Ok(fetch_exact(repo, &[path.as_str()])?.into_iter().next())
}

#[cfg(test)]
mod tests {
    use super::{Flags, sentence_prompts};

    #[test]
    fn local_sentence_prompts_are_optional_but_malformed_files_are_refused() {
        let dir =
            std::env::temp_dir().join(format!("bloomery-sentence-prompts-{}", std::process::id()));
        std::fs::create_dir(&dir).unwrap();
        let model = dir.join("embedding.gguf");
        let flags = Flags {
            path: Some(("--embd-model".into(), model.display().to_string())),
            ..Default::default()
        };
        let config = dir.join("config_sentence_transformers.json");
        assert!(sentence_prompts(&flags, &model).unwrap().is_empty());
        std::fs::write(&config, r#"{"prompts":{"query":"query: "}}"#).unwrap();
        assert_eq!(
            sentence_prompts(&flags, &model).unwrap()["query"],
            "query: "
        );
        std::fs::write(&config, r#"{"prompts":{"query":7}}"#).unwrap();
        assert!(
            sentence_prompts(&flags, &model)
                .unwrap_err()
                .to_string()
                .contains("config_sentence_transformers.json")
        );
        std::fs::remove_file(config).unwrap();
        std::fs::remove_dir(dir).unwrap();
    }

    #[test]
    fn optional_files_only_ignore_missing_files() {
        use hf::HfError;
        for (error, missing) in [
            (
                HfError::NoFile {
                    repo: "fixture/quant".into(),
                    name: "config.json".into(),
                },
                true,
            ),
            (
                HfError::Curl {
                    what: "listing".into(),
                    why: "HTTP 403".into(),
                },
                false,
            ),
            (
                HfError::Listing {
                    repo: "fixture/quant".into(),
                    why: "malformed listing".into(),
                },
                false,
            ),
            (
                HfError::Io {
                    path: "cache".into(),
                    err: std::io::ErrorKind::PermissionDenied.into(),
                },
                false,
            ),
        ] {
            assert_eq!(super::no_file(&error), missing);
            let wrapped = HfError::Offline {
                repo: "fixture/quant".into(),
                why: "unreachable".into(),
                cache: Box::new(error),
            };
            assert_eq!(super::no_file(&wrapped), missing);
        }
    }

    #[cfg(unix)]
    #[test]
    fn checkpoint_support_is_checked_before_download() {
        use crate::model_file;
        use std::os::unix::fs::PermissionsExt;

        // The child half: the parent reruns this test in each case's folder, which holds `scenario`.
        if let Ok(case) = std::fs::read_to_string("scenario") {
            let flags = Flags {
                hf: Some(if case == "quant" {
                    "fixture/checkpoint:Q4_K_M".into()
                } else {
                    "fixture/checkpoint".into()
                }),
                ..Default::default()
            };
            let result = match case.as_str() {
                "name" => model_file::name(&flags),
                "resolve" => {
                    model_file::resolve(&flags, None, hf::fetch::Sources::Gguf).map(|_| ())
                }
                "quant" => {
                    let error =
                        model_file::resolve(&flags, None, hf::fetch::Sources::GgufOrCheckpoint)
                            .unwrap_err();
                    assert!(error.to_string().contains(":Q4_K_M picks a GGUF set"));
                    return;
                }
                "supported" => {
                    let path =
                        model_file::resolve(&flags, None, hf::fetch::Sources::GgufOrCheckpoint)
                            .unwrap()
                            .unwrap();
                    assert!(path.is_dir());
                    assert!(path.join("model-00002-of-00002.safetensors").is_file());
                    return;
                }
                #[cfg(feature = "deepseek41")]
                "side" => {
                    let (side, rest) = crate::side_seat::Flags::take(&[
                        "--embd-hf".into(),
                        "fixture/checkpoint".into(),
                    ])
                    .unwrap();
                    assert!(rest.is_empty());
                    side.read().map(|_| ())
                }
                _ => panic!("unknown scenario: {case}"),
            };
            if case == "name" && cfg!(feature = "checkpoint") {
                result.unwrap();
                return;
            }
            assert!(
                result
                    .expect_err("checkpoint support is absent")
                    .to_string()
                    .contains("does not support safetensors checkpoints")
            );
            return;
        }

        // A checkpoint-only repo on a `file://` hub, reached through a `curl` that logs each hub
        // path it is asked for in the case's folder.
        let dir =
            std::env::temp_dir().join(format!("bloomery-model-sources-{}", std::process::id()));
        let hub = dir.join("hub");
        let served = hub.join("fixture/checkpoint/resolve/main");
        std::fs::create_dir_all(&served).unwrap();
        let body = b"tiny fixture";
        let mut listing = Vec::new();
        for path in [
            "model.safetensors.index.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ] {
            let file = served.join(path);
            std::fs::write(&file, body).unwrap();
            let digest = hf::fetch::digest(&file, hf::Check::GitSha1, body.len() as u64).unwrap();
            listing.push(format!(
                r#"{{"type":"file","path":"{path}","size":{},"oid":"{digest}"}}"#,
                body.len()
            ));
        }
        let tree = hub.join("api/models/fixture/checkpoint/tree");
        std::fs::create_dir_all(&tree).unwrap();
        std::fs::write(tree.join("main"), format!("[{}]", listing.join(","))).unwrap();
        let bin = dir.join("bin");
        std::fs::create_dir(&bin).unwrap();
        let curl = bin.join("curl");
        std::fs::write(
            &curl,
            format!(
                r#"#!/bin/bash
args=()
for arg; do
    case "$arg" in
        https://huggingface.co/*)
            printf '%s\n' "${{arg#https://huggingface.co}}" >> requests || exit 1
            arg="file://{}/${{arg#https://huggingface.co/}}" ;;
    esac
    args+=("$arg")
done
exec /usr/bin/curl "${{args[@]}}"
"#,
                hub.display()
            ),
        )
        .unwrap();
        std::fs::set_permissions(&curl, std::fs::Permissions::from_mode(0o755)).unwrap();
        for case in [
            "name",
            "resolve",
            "quant",
            "supported",
            #[cfg(feature = "deepseek41")]
            "side",
        ] {
            let root = dir.join(case);
            std::fs::create_dir(&root).unwrap();
            std::fs::write(root.join("scenario"), case).unwrap();
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "model_file::fetch::tests::checkpoint_support_is_checked_before_download",
                    "--nocapture",
                ])
                .current_dir(&root)
                .env("PATH", &bin)
                .env("BLOOMERY_CACHE", root.join("cache"))
                .env("HOME", &root)
                .env("HF_HUB_OFFLINE", "0")
                .env_remove("HF_TOKEN")
                .env_remove("BLOOMERY_REF_MODEL")
                .output()
                .unwrap();
            assert!(
                output.status.success()
                    && String::from_utf8_lossy(&output.stdout).contains("1 passed; 0 failed"),
                "{case}: {}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            let requests = std::fs::read_to_string(root.join("requests")).unwrap();
            let requests: Vec<&str> = requests.lines().collect();
            let supported = case == "supported" || (case == "name" && cfg!(feature = "checkpoint"));
            let shards = requests
                .iter()
                .filter(|p| p.ends_with(".safetensors"))
                .count();
            println!(
                "{case}: listing requests: {}, shard requests: {shards}",
                requests
                    .iter()
                    .filter(|p| p.starts_with("/api/models/"))
                    .count()
            );
            assert_eq!(
                shards,
                if supported { 2 } else { 0 },
                "{case}: {requests:?}"
            );
            assert_eq!(
                requests.len(),
                if supported { 4 } else { 1 },
                "{case}: {requests:?}"
            );
            assert_eq!(root.join("cache/fixture/checkpoint").exists(), supported);
        }
        std::fs::remove_dir_all(dir).unwrap();
    }
}
