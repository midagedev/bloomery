//! The model file a server opens, named on its command line: `-m PATH` (or
//! another spelling the binary gives the path flag) or `--hf
//! <repo>[:<quant>]`, fetched into the cache ([`hf::fetch`]). A server takes
//! the flags out of its arguments ([`take`]) before anything else parses
//! them, and the path it resolves is the process's model file
//! ([`set_model_file`]), which [`crate::ref_model_path`] reads before the
//! gates' `$BLOOMERY_REF_MODEL`. The rules of a model named twice are
//! [`hf::source`]'s. A decision model's head comes from the repo a `--hf`
//! repo's model card says it quantizes ([`quantized_from`]), fetched by
//! [`fetch_exact`]. A family's MTP draft ([`refset::arch::beside_drafts`], one
//! row per family that keeps its draft in a file of its own) is fetched beside
//! a set only when the plan's own rule says the set can run it
//! ([`draft_usable`]).

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use gguf::Split;
use hf::HfError;
use hf::fetch::{Client, Draft, Event};
use hf::source::{self, Flags, Source};
use refset::arch::BesideDraft;

use crate::GateError;
use crate::record::{self, Record};

/// The model file a server named on its command line, set once.
static MODEL_FILE: OnceLock<PathBuf> = OnceLock::new();

/// The process's model file is `path` from now on; a second call is
/// refused by name.
pub fn set_model_file(path: PathBuf) -> Result<(), GateError> {
    let shown = path.display().to_string();
    MODEL_FILE.set(path).map_err(|_| {
        format!(
            "the model file is set already ({}), and a second set names {shown}",
            MODEL_FILE
                .get()
                .map_or_else(String::new, |p| p.display().to_string())
        )
        .into()
    })
}

/// The model file a server set, if one.
pub(crate) fn model_file() -> Option<&'static PathBuf> {
    MODEL_FILE.get()
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

/// A row's usability rule, [`draft_usable`] closed over the row.
type Usable = Box<dyn Fn(&Path) -> Result<(), String>>;

/// Whether the set whose first shard is `first` can run `row`'s MTP draft:
/// the set declares the row's architecture, then the plan's own rule
/// (`PlanInputs::mtp_borrows`, the target's `token_embd` and `output` in the
/// format the draft reads them), why not as that rule says it; a shard the
/// plan cannot describe runs no draft either. A row of an architecture this
/// has no rule for is refused by name, and its draft is not fetched: a family
/// adds its row to `refset::arch::beside_drafts` and its rule here.
fn draft_usable(row: &BesideDraft, first: &Path) -> Result<(), String> {
    let split = Split::open(first).map_err(|e| format!("open {}: {e}", first.display()))?;
    let arch = split.architecture().unwrap_or("<missing>");
    if arch != row.arch {
        return Err(format!(
            "the set's architecture is {arch}, and the draft {} is {}'s",
            row.name(),
            row.arch
        ));
    }
    match model::arch::qwen35moe_variant(&split) {
        Ok(model::arch::qwen35moe::hparams::Variant::Qwen4Exp) => {
            let inputs = model::arch::qwen35moe::place::PlanInputs::describe(&split)
                .map_err(|e| e.to_string())?;
            inputs.mtp_borrows().map_err(|e| e.to_string())
        }
        _ => Err(format!(
            "no rule says whether a {arch} set can run its draft"
        )),
    }
}

/// The local path of what `flags` and `env` (the gates' variable, `None`
/// for a binary that reads none) name: a path as given, a repo's set
/// fetched and checked (its first shard), the family's MTP draft beside it
/// when the repo holds the file and the set can run it ([`draft_usable`]; an
/// `hf draft skipped` record says why not). `None` when nothing names one.
pub fn resolve(flags: &Flags, env: Option<&str>) -> Result<Option<PathBuf>, GateError> {
    match source::resolve(flags, env)? {
        None => Ok(None),
        Some(Source::File(p)) => Ok(Some(PathBuf::from(p))),
        Some(Source::Hf(r)) => {
            let rows = refset::arch::beside_drafts();
            let usable: Vec<Usable> = rows
                .iter()
                .map(|&row| -> Usable { Box::new(move |first: &Path| draft_usable(row, first)) })
                .collect();
            let drafts: Vec<Draft<'_>> = rows
                .iter()
                .zip(&usable)
                .map(|(row, usable)| Draft {
                    name: row.name(),
                    usable: usable.as_ref(),
                })
                .collect();
            let files = client()?.resolve_drafts(&r, &drafts, &mut print)?;
            Ok(Some(files.into_iter().next().ok_or_else(|| {
                format!("--hf {r}: the picked set has no file")
            })?))
        }
    }
}

/// The files of `repo` at exactly `paths`, fetched and checked; their local
/// paths in that order.
pub fn fetch_exact(repo: &str, paths: &[&str]) -> Result<Vec<PathBuf>, GateError> {
    Ok(client()?.resolve_exact(repo, paths, &mut print)?)
}

/// The repo the model card of `repo` (its `README.md`, fetched and checked
/// as any file of it) says `repo` quantizes ([`hf::card::quantizes`]);
/// `None` when the repo has no card or its card says no such thing.
pub fn quantized_from(repo: &str) -> Result<Option<String>, GateError> {
    let card = match client()?.resolve_exact(repo, &["README.md"], &mut print) {
        Ok(paths) => paths
            .into_iter()
            .next()
            .ok_or_else(|| format!("{repo}: the card's fetch returned no file"))?,
        Err(HfError::NoFile { .. }) => return Ok(None),
        Err(e) => return Err(e.into()),
    };
    let text = std::fs::read_to_string(&card).map_err(|e| format!("{}: {e}", card.display()))?;
    Ok(hf::card::quantizes(&text))
}

/// The model file `flags` name, resolved beside `$BLOOMERY_REF_MODEL`
/// ([`resolve`]) and set as the process's ([`set_model_file`]) when they
/// name one.
pub fn name(flags: &Flags) -> Result<(), GateError> {
    let env = std::env::var("BLOOMERY_REF_MODEL").ok();
    if *flags != Flags::default()
        && let Some(path) = resolve(flags, env.as_deref())?
    {
        set_model_file(path)?;
    }
    Ok(())
}

/// `<bin> <version> (commit <c>)`: this crate's version and the commit the
/// build came from, `BLOOMERY_BUILD_COMMIT` at build time (the release
/// script sets it), `unknown` when it was unset.
pub fn version(bin: &str) -> String {
    format!(
        "{bin} {} (commit {})",
        env!("CARGO_PKG_VERSION"),
        option_env!("BLOOMERY_BUILD_COMMIT").unwrap_or("unknown")
    )
}
