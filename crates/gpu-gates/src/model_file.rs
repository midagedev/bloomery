//! The model file a server opens, named on its command line: `-m PATH` (or
//! another spelling the binary gives the path flag) or `--hf
//! <repo>[:<quant>]`, fetched into the cache ([`hf::fetch`]). A server takes
//! the flags out of its arguments ([`take`]) before anything else parses
//! them, and the path it resolves is the process's model file
//! ([`set_model_file`]), which [`crate::ref_model_path`] reads before the
//! gates' `$BLOOMERY_REF_MODEL`. The rules of a model named twice are
//! [`hf::source`]'s. A decision model's head comes from the repo a `--hf`
//! repo's model card says it quantizes ([`quantized_from`]), fetched by
//! [`fetch_exact`].

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use hf::HfError;
use hf::fetch::{Client, Event};
use hf::source::{self, Flags, Source};

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
/// `~/.cache/bloomery/hf`.
fn client() -> Result<Client, GateError> {
    let root = hf::cache_root(std::env::var_os("BLOOMERY_CACHE"), std::env::var_os("HOME"))?;
    Ok(Client::new(root))
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

/// The MTP draft file name a `--hf` resolve also fetches when the repo
/// holds a file of it, landing it beside the picked set where the engine's
/// draft rule opens it from: the qwen4exp family's shared draft
/// ([`refset::arch::qwen4exp::mtp::DRAFT`]'s file name), the one family
/// whose draft is a file of its own beside the target — GLM's NextN is
/// inside the target, so it needs none.
fn draft_name() -> &'static str {
    Path::new(refset::arch::qwen4exp::mtp::DRAFT)
        .file_name()
        .and_then(std::ffi::OsStr::to_str)
        .expect("the qwen4exp DRAFT path names no file")
}

/// The local path of what `flags` and `env` (the gates' variable, `None`
/// for a binary that reads none) name: a path as given, a repo's set
/// fetched and checked (its first shard), the family's MTP draft beside it
/// when the repo holds the file. `None` when nothing names one.
pub fn resolve(flags: &Flags, env: Option<&str>) -> Result<Option<PathBuf>, GateError> {
    match source::resolve(flags, env)? {
        None => Ok(None),
        Some(Source::File(p)) => Ok(Some(PathBuf::from(p))),
        Some(Source::Hf(r)) => {
            let files = client()?.resolve(&r, Some(draft_name()), &mut print)?;
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
