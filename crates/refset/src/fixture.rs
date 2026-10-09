//! The fixture tier's identity: where each architecture's fixture file lies,
//! and the `# fixture` line a reference set dumped from it carries.
//!
//! A fixture is regenerated in place at the same path, so the first shard's
//! path alone (`# model`) cannot tell a set dumped from one generation from a
//! set read against the next: a set of seed 1 read against a fixture of seed 2
//! would open without complaint, and its differences from the engine would
//! look like engine bugs. The `# fixture` line carries every
//! `bloomery.fixture.*` header key of the file the set was dumped from, and a
//! family of the fixture identities ([`crate::family::Identity::FixtureManifest`])
//! holds the set to the line of the file the tree runs ([`line`]).
//!
//! The path owner is [`first_shard`], the twin of `tools/ref/ref-paths.sh`'s
//! fixture table: the string it returns is the one the dumper is given, so it
//! is the one a set's `# model` states.

use crate::RefError;
use gguf::{Gguf, LoadError, Value};
use std::ffi::OsString;
use std::path::{Path, PathBuf};

/// The variable that moves the fixture root.
pub const ROOT_ENV: &str = "BLOOMERY_FIXTURE_ROOT";
/// The fixture root when [`ROOT_ENV`] is unset or empty.
pub const DEFAULT_ROOT: &str = "/models/fixtures";

/// What opens a set's `# fixture` line, tab included: the line is this and
/// [`keys`].
pub const LINE_PREFIX: &str = "# fixture\t";

/// The directory of each architecture's fixture under the root. An
/// architecture not listed has no fixture.
const DIRS: &[(&str, &str)] = &[
    ("qwen4exp", "qwen38"),
    ("deepseek41", "v41"),
    ("glm5next", "glm5next"),
];

/// The prefix of every key the fixture generator writes (`bloomery_model`'s
/// `fixture` module owns them).
const KEY_PREFIX: &str = "bloomery.fixture.";
/// u32: its presence marks a fixture.
const KEY_VERSION: &str = "bloomery.fixture.version";
/// String array: present only on a file that holds some of the planned
/// tensors.
const KEY_SUBSET: &str = "bloomery.fixture.subset";

#[cfg(test)]
thread_local! {
    static TEST_ROOT: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) };
}

/// Run `f` with [`root`] answering `root` on this thread only: a test's
/// fixture root without touching the process environment, which other tests
/// of the process read.
#[cfg(test)]
pub(crate) fn with_root<T>(root: &Path, f: impl FnOnce() -> T) -> T {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            TEST_ROOT.with(|r| *r.borrow_mut() = None);
        }
    }
    TEST_ROOT.with(|r| *r.borrow_mut() = Some(root.display().to_string()));
    let _reset = Reset;
    f()
}

/// The fixture root as the shell takes it: `$BLOOMERY_FIXTURE_ROOT`, with an
/// empty value counting as unset; [`DEFAULT_ROOT`] otherwise.
pub fn root() -> Result<String, RefError> {
    #[cfg(test)]
    if let Some(root) = TEST_ROOT.with(|r| r.borrow().clone()) {
        return Ok(root);
    }
    root_of(std::env::var_os(ROOT_ENV))
}

/// [`root`] over the variable's value.
fn root_of(var: Option<OsString>) -> Result<String, RefError> {
    match var {
        None => Ok(DEFAULT_ROOT.to_string()),
        Some(v) if v.is_empty() => Ok(DEFAULT_ROOT.to_string()),
        Some(v) => v.into_string().map_err(|v| {
            RefError::missing(
                PathBuf::new(),
                format!("{ROOT_ENV} is not UTF-8: {v:?}, which a set's # model line cannot state"),
            )
        }),
    }
}

/// The fixture file of architecture `arch`, as the dumper is given it: its
/// first shard, `<root>/<dir>/<file>` joined as the shell joins it. See
/// [`first_shard_in`].
pub fn first_shard(arch: &str) -> Result<String, RefError> {
    first_shard_in(&root()?, arch)
}

/// The fixture file of `arch` under `root`: the one file of the architecture's
/// directory whose name matches `*-00001-of-*.gguf`, joined as
/// `"{root}/{dir}/{file}"` (never normalized, so a trailing `/` on `root`
/// stays, as it does in the shell). An architecture with no fixture directory,
/// a directory that cannot be listed, and one holding no such file or more than
/// one are [`RefError::Missing`], naming the architecture or the directory and
/// what was found.
pub fn first_shard_in(root: &str, arch: &str) -> Result<String, RefError> {
    let Some(&(_, dir)) = DIRS.iter().find(|(a, _)| *a == arch) else {
        let known: Vec<&str> = DIRS.iter().map(|(a, _)| *a).collect();
        return Err(RefError::missing(
            PathBuf::new(),
            format!(
                "fixture: the {arch} architecture has no fixture directory, the table holds {known:?}"
            ),
        ));
    };
    let at = format!("{root}/{dir}");
    let entries = std::fs::read_dir(&at)
        .map_err(|e| RefError::missing(&at, format!("fixture: cannot list {at}: {e}")))?;
    let mut found: Vec<String> = Vec::new();
    for entry in entries {
        let name = entry
            .map_err(|e| RefError::missing(&at, format!("fixture: cannot list {at}: {e}")))?
            .file_name();
        if !is_first_shard(&name.to_string_lossy()) {
            continue;
        }
        let name = name.into_string().map_err(|name| {
            RefError::missing(
                &at,
                format!("fixture: {at} holds a first shard whose name is not UTF-8: {name:?}"),
            )
        })?;
        found.push(name);
    }
    found.sort();
    match found.as_slice() {
        [one] => Ok(format!("{at}/{one}")),
        _ => Err(RefError::missing(
            &at,
            format!(
                "fixture: {at} holds {} first shards (*-00001-of-*.gguf): {found:?}, want one \
                 (`fixture generate` writes it)",
                found.len()
            ),
        )),
    }
}

/// Whether the shell's glob `*-00001-of-*.gguf` matches `name`: not a hidden
/// file, ending `.gguf`, with `-00001-of-` somewhere before that.
fn is_first_shard(name: &str) -> bool {
    !name.starts_with('.')
        && name
            .strip_suffix(".gguf")
            .is_some_and(|stem| stem.contains("-00001-of-"))
}

/// The `# fixture` line of the fixture file whose first shard is
/// `first_shard`: [`LINE_PREFIX`] and [`keys`].
pub fn line(first_shard: &Path) -> Result<String, RefError> {
    Ok(format!("{LINE_PREFIX}{}", keys(first_shard)?))
}

/// What follows the tab of the `# fixture` line: every header key of the file
/// that starts with `bloomery.fixture.`, sorted by byte order, each as
/// `key=value`, joined by one space. A value is an integer in decimal, a bool
/// as `true` or `false`, a string verbatim, or an array of those joined by
/// `,` with no spaces. The header only is read: the file is mapped lazily, so
/// a shard of many GiB costs its header's pages.
///
/// A file that is not a fixture (no `bloomery.fixture.version`) is
/// [`RefError::Missing`]; a subset file (`bloomery.fixture.subset`: it holds
/// only some tensors, so no whole set was dumped from it), a float value, a
/// nested or empty array, a string holding whitespace, `=`, `,` or a control
/// character, and a key given twice are [`RefError::Malformed`], each naming
/// the file and the key.
pub fn keys(first_shard: &Path) -> Result<String, RefError> {
    let file = first_shard.display().to_string();
    let g = Gguf::open(first_shard).map_err(|e| match e {
        LoadError::Io(io) => {
            RefError::missing(first_shard, format!("fixture: cannot open {file}: {io}"))
        }
        e => RefError::malformed(
            format!("fixture: {file}"),
            format!("not a readable GGUF header: {e}"),
        ),
    })?;
    let mut held: Vec<(&str, &Value)> = g
        .iter_kv()
        .filter(|(k, _)| k.starts_with(KEY_PREFIX))
        .collect();
    held.sort_by(|a, b| a.0.cmp(b.0));
    if let Some(w) = held.windows(2).find(|w| w[0].0 == w[1].0) {
        return Err(RefError::malformed(
            format!("fixture: {file}"),
            format!(
                "{} is given twice, so the file states two values for it",
                w[0].0
            ),
        ));
    }
    if !held.iter().any(|(k, _)| *k == KEY_VERSION) {
        return Err(RefError::missing(
            first_shard,
            format!("fixture: {file} has no {KEY_VERSION} key: it is not a fixture file"),
        ));
    }
    if held.iter().any(|(k, _)| *k == KEY_SUBSET) {
        return Err(RefError::malformed(
            format!("fixture: {file}"),
            format!(
                "{KEY_SUBSET} is set: the file holds only some tensors, and no whole set is dumped from it"
            ),
        ));
    }
    let mut parts = Vec::with_capacity(held.len());
    for (key, v) in held {
        parts.push(format!("{key}={}", render(&file, key, v)?));
    }
    Ok(parts.join(" "))
}

/// The text of value `v` of `key` in the line: see [`keys`].
fn render(file: &str, key: &str, v: &Value) -> Result<String, RefError> {
    let bad =
        |why: String| RefError::malformed(format!("fixture: {file}"), format!("{key}: {why}"));
    match v {
        Value::U8(x) => Ok(x.to_string()),
        Value::I8(x) => Ok(x.to_string()),
        Value::U16(x) => Ok(x.to_string()),
        Value::I16(x) => Ok(x.to_string()),
        Value::U32(x) => Ok(x.to_string()),
        Value::I32(x) => Ok(x.to_string()),
        Value::U64(x) => Ok(x.to_string()),
        Value::I64(x) => Ok(x.to_string()),
        Value::Bool(x) => Ok(x.to_string()),
        Value::String(s) => match s
            .chars()
            .find(|c| c.is_whitespace() || *c == '=' || *c == ',' || c.is_control())
        {
            None => Ok(s.clone()),
            Some(c) => Err(bad(format!(
                "the string {s:?} holds {c:?}: whitespace, `=`, `,` and control characters \
                 cannot be told apart from the line's own separators"
            ))),
        },
        Value::F32(_) | Value::F64(_) => Err(bad(
            "a float value has no stable text: the line takes integers, bools, strings and arrays of them"
                .to_string(),
        )),
        Value::Array(items) if items.is_empty() => Err(bad("an empty array".to_string())),
        Value::Array(items) => items
            .iter()
            .map(|e| match e {
                Value::Array(_) => Err(bad("a nested array".to_string())),
                e => render(file, key, e),
            })
            .collect::<Result<Vec<_>, _>>()
            .map(|parts| parts.join(",")),
    }
}

#[cfg(test)]
mod tests;
