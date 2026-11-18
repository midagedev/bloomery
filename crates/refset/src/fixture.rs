//! The fixture tier's identity: where each architecture's fixture file lies,
//! and the `# fixture` line a reference set dumped from it carries.
//!
//! A fixture is regenerated in place at the same path, so the first shard's
//! path alone (`# model`) cannot tell a set dumped from one generation from a
//! set read against the next: a set of seed 1 read against a fixture of seed 2
//! would open without complaint, and its differences from the engine would
//! look like engine bugs. The `# fixture` line carries every
//! `bloomery.fixture.*` header key of the file the set was dumped from but the
//! card budget (a planning input that moves no byte of the file or of any
//! reference output, so a re-pinned budget needs no new dump), and a
//! family of the fixture identities ([`crate::family::Identity::FixtureManifest`])
//! holds the set to the line of the file the tree runs ([`line`]).
//!
//! The path owner is [`first_shard`], the twin of `tools/ref/ref-paths.sh`'s
//! fixture table: the string it returns is the one the dumper is given, so it
//! is the one a set's `# model` states.
//!
//! A fixture is a GGUF split (`fixture generate`) or a checkpoint folder cut in its
//! stored bits (`fixture cut`, `model_hf::cut`): a directory that holds
//! `model.safetensors.index.json` is the second kind, its first shard is the
//! `*-00001-of-*.safetensors` beside the index, and its `bloomery.fixture.*` keys are
//! the string entries of that shard's safetensors `__metadata__`, each already the
//! text the line shows.
//!
//! A fixture variant is the fixture requantized to a type map
//! (`tools/fixture-variants.tsv`): `$BLOOMERY_FIXTURE_VARIANT` moves the root to
//! `<parent of root>/fixture-variants/<tag>` ([`root`]), the file's `bloomery.fixture.variant`
//! key names the tag, and a variant's sets carry it in their names
//! ([`variant_set_name`]), so a base set and a variant's never share a path. The
//! key is part of the `# fixture` line, so a set dumped from one is stale for the
//! other twice over: by `# model` and by `# fixture`.

use crate::RefError;
use gguf::{LoadError, Split, Value};
use std::ffi::OsString;
use std::path::{Path, PathBuf};

/// The variable that moves the fixture root.
pub const ROOT_ENV: &str = "BLOOMERY_FIXTURE_ROOT";
/// The fixture root when [`ROOT_ENV`] is unset or empty.
pub const DEFAULT_ROOT: &str = "/models/fixtures";
/// The variable that selects a fixture variant: a tag of `tools/fixture-variants.tsv`, whose files
/// live under `<parent of root>/fixture-variants/<tag>`, beside the fixture root and not under it.
pub const VARIANT_ENV: &str = "BLOOMERY_FIXTURE_VARIANT";

/// What opens a set's `# fixture` line, tab included: the line is this and
/// [`keys`].
pub const LINE_PREFIX: &str = "# fixture\t";

/// The architecture-to-directory table, `tools/fixture-dirs.tsv`: a row `arch<TAB>dir` a line, `#` and blank lines aside. The
/// shell reads the same file (`tools/ref/ref-paths.sh`).
const DIRS_TABLE: &str = include_str!("../../../tools/fixture-dirs.tsv");

/// The index of a checkpoint folder, the file that makes a fixture directory a folder's.
const INDEX: &str = "model.safetensors.index.json";
/// The file name ends of a GGUF fixture's shards and of a cut folder's.
const GGUF: &str = ".gguf";
const SAFETENSORS: &str = ".safetensors";

/// The most a safetensors header may be: a length past it is a corrupt file, not a header to read.
const MAX_HEADER: u64 = 64 << 20;

/// A row's `dir` for a family with no fixture file of its own: a command runs on the profile's file, and there is no directory.
const SELF: &str = "self";

/// What the shell says of an architecture the table does not name; never a row's `dir`.
const NONE: &str = "none";

/// The rows of a table text, `(arch, dir)` in file order.
///
/// # Errors
/// A line that is not two columns, an architecture that is empty or in the table twice, a `dir` that is empty, `none` or holds a
/// byte other than a lower-case letter, a digit or `-`, and a directory two architectures share.
fn parse_dirs(text: &str) -> Result<Vec<(&str, &str)>, String> {
    let mut rows: Vec<(&str, &str)> = Vec::new();
    for (n, line) in text.lines().enumerate() {
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }
        let at = n + 1;
        let cols: Vec<&str> = line.split('\t').collect();
        let [arch, dir] = cols.as_slice() else {
            return Err(format!(
                "line {at} has {} columns, want arch and dir",
                cols.len()
            ));
        };
        if arch.is_empty() || rows.iter().any(|(a, _)| a == arch) {
            return Err(format!(
                "line {at}: the architecture {arch:?} is empty or in the table twice"
            ));
        }
        if dir.is_empty()
            || !dir
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
        {
            return Err(format!(
                "line {at}: the directory {dir:?} is lower-case letters, digits and '-'"
            ));
        }
        if *dir == NONE {
            return Err(format!(
                "line {at}: the directory {NONE:?} is what an architecture with no row has"
            ));
        }
        if *dir != SELF && rows.iter().any(|(_, d)| d == dir) {
            return Err(format!(
                "line {at}: the directory {dir:?} is another architecture's"
            ));
        }
        rows.push((arch, dir));
    }
    Ok(rows)
}

/// The table's rows with a directory: `(arch, dir)`, the `self` rows left out.
///
/// # Panics
/// On a table [`parse_dirs`] refuses; the embedded table is held by a test, so none ships.
fn dirs() -> Vec<(&'static str, &'static str)> {
    parse_dirs(DIRS_TABLE)
        .unwrap_or_else(|e| panic!("tools/fixture-dirs.tsv: {e}"))
        .into_iter()
        .filter(|&(_, dir)| dir != SELF)
        .collect()
}

/// The directory of `arch`'s fixture under the root, `None` for an architecture the table does not name and for a `self` one.
#[must_use]
pub fn dir_of(arch: &str) -> Option<&'static str> {
    dirs().into_iter().find(|&(a, _)| a == arch).map(|(_, d)| d)
}

/// The prefix of every key the fixture generator writes (`bloomery_model`'s
/// `fixture` module owns them).
const KEY_PREFIX: &str = "bloomery.fixture.";
/// u32: its presence marks a fixture.
const KEY_VERSION: &str = "bloomery.fixture.version";
/// u64: the card byte budget the fixture's gate plans under. A planning input: no byte of the file and no
/// reference output depends on it, so the `# fixture` line leaves it out ([`keys`]).
const KEY_CARD_BUDGET: &str = "bloomery.fixture.card_budget";
/// String array: present only on a file that holds some of the planned
/// tensors.
const KEY_SUBSET: &str = "bloomery.fixture.subset";
/// String: the tag of the variant a file is; present only on a requantized fixture.
const KEY_VARIANT: &str = "bloomery.fixture.variant";

#[cfg(test)]
thread_local! {
    static TEST_ROOT: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) };
    static TEST_VARIANT: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) };
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

/// Run `f` with [`selected`] answering `tag` on this thread only, as [`with_root`] does the root.
#[cfg(test)]
pub(crate) fn with_variant<T>(tag: &str, f: impl FnOnce() -> T) -> T {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            TEST_VARIANT.with(|r| *r.borrow_mut() = None);
        }
    }
    TEST_VARIANT.with(|r| *r.borrow_mut() = Some(tag.to_string()));
    let _reset = Reset;
    f()
}

/// The fixture root as the shell takes it: `$BLOOMERY_FIXTURE_ROOT`, with an
/// empty value counting as unset, [`DEFAULT_ROOT`] otherwise; for a selected
/// variant ([`selected`]) `fixture-variants/<tag>` beside that root.
pub fn root() -> Result<String, RefError> {
    #[cfg(test)]
    let base = match TEST_ROOT.with(|r| r.borrow().clone()) {
        Some(root) => Ok(root),
        None => root_of(std::env::var_os(ROOT_ENV)),
    };
    #[cfg(not(test))]
    let base = root_of(std::env::var_os(ROOT_ENV));
    Ok(variant_root(&base?, selected()?.as_deref()))
}

/// `base` for no variant, `<parent of base>/fixture-variants/<tag>` for one: the shell's
/// `__fixture_root=${__fixture_root%/*}/fixture-variants/$FIXTURE_VARIANT`. The variants sit beside the
/// fixture root because the box manifest keys every file under the root into every fixture-tier item.
fn variant_root(base: &str, variant: Option<&str>) -> String {
    match variant {
        None => base.to_string(),
        Some(tag) => {
            let parent = base.rsplit_once('/').map_or(base, |(parent, _)| parent);
            format!("{parent}/fixture-variants/{tag}")
        }
    }
}

/// The variant `$BLOOMERY_FIXTURE_VARIANT` selects, an empty value counting as unset.
///
/// # Errors
/// A value that is not UTF-8 or not lower-case letters and digits, which no tag of the table is.
pub fn selected() -> Result<Option<String>, RefError> {
    #[cfg(test)]
    if let Some(tag) = TEST_VARIANT.with(|r| r.borrow().clone()) {
        return Ok(Some(tag));
    }
    selected_of(std::env::var_os(VARIANT_ENV))
}

/// [`selected`] over the variable's value.
fn selected_of(var: Option<OsString>) -> Result<Option<String>, RefError> {
    let bad = |why: String| RefError::malformed(VARIANT_ENV.to_string(), why);
    match var {
        None => Ok(None),
        Some(v) if v.is_empty() => Ok(None),
        Some(v) => {
            let tag = v
                .into_string()
                .map_err(|v| bad(format!("{v:?} is not UTF-8")))?;
            if tag
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit())
            {
                Ok(Some(tag))
            } else {
                Err(bad(format!(
                    "{tag:?} is not a tag of tools/fixture-variants.tsv (lower-case letters and digits)"
                )))
            }
        }
    }
}

/// The name of a fixture set `base` (`fx_<set>`, or `ref-mtp/fx_<set>` for an MTP set) of the variant `tag`: `fx_<tag>_<set>`,
/// so no variant's set shares a path with the base's or another variant's. The shell's `SET_PREFIX` is `fx_<tag>_`.
#[must_use]
pub fn variant_set_name(tag: &str, base: &str) -> String {
    if let Some(set) = base.strip_prefix("fx_") {
        format!("fx_{tag}_{set}")
    } else if let Some(set) = base.strip_prefix("ref-mtp/fx_") {
        format!("ref-mtp/fx_{tag}_{set}")
    } else {
        base.to_string()
    }
}

/// The fixture file `first_shard` opened as a split, which is what tells a scale-less F8 table (ggml's id 43, whose bytes only the
/// extent of the tensor can tell from MXFP8's) from the other meaning of the id; a single shard's strict open refuses such a file.
///
/// # Errors
/// [`RefError::Missing`] for a file or a shard of its split that is not there, [`RefError::Malformed`] for one that is not a
/// readable GGUF header.
fn open_split(first_shard: &Path, file: &str) -> Result<Split, RefError> {
    Split::open(first_shard).map_err(|e| match e {
        LoadError::Io(io) => {
            RefError::missing(first_shard, format!("fixture: cannot open {file}: {io}"))
        }
        LoadError::MissingShard { path } => RefError::missing(
            first_shard,
            format!("fixture: cannot open {file}: split shard {path} is missing"),
        ),
        e => RefError::malformed(
            format!("fixture: {file}"),
            format!("not a readable GGUF header: {e}"),
        ),
    })
}

/// The variant the fixture file `first_shard` is: its `bloomery.fixture.variant` key, none on a base fixture
/// and on a cut folder's shard (a folder fixture has no variants).
///
/// # Errors
/// A file that is not a readable GGUF header, or a key that is not a string.
pub fn variant_of(first_shard: &Path) -> Result<Option<String>, RefError> {
    if is_safetensors(first_shard) {
        return Ok(None);
    }
    let file = first_shard.display().to_string();
    let g = open_split(first_shard, &file)?;
    match g.value(KEY_VARIANT) {
        None => Ok(None),
        Some(Value::String(tag)) => Ok(Some(tag.clone())),
        Some(other) => Err(RefError::malformed(
            format!("fixture: {file}"),
            format!("{KEY_VARIANT} is {other:?}, not a string"),
        )),
    }
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
    let first = first_shard_in(&root()?, arch)?;
    let (wants, has) = (selected()?, variant_of(Path::new(&first))?);
    if wants == has {
        return Ok(first);
    }
    let show = |v: &Option<String>| {
        v.as_deref()
            .map_or("none".to_string(), |t| format!("{t:?}"))
    };
    Err(RefError::malformed(
        format!("fixture: {first}"),
        format!(
            "{VARIANT_ENV} selects {}, and the file's {KEY_VARIANT} is {}",
            show(&wants),
            show(&has)
        ),
    ))
}

/// The fixture file of `arch` under `root`: for a directory that holds a checkpoint folder's index
/// (`model.safetensors.index.json`) the one `*-00001-of-*.safetensors` file beside it, for any other
/// the one file whose name matches `*-00001-of-*.gguf`, joined as
/// `"{root}/{dir}/{file}"` (never normalized, so a trailing `/` on `root`
/// stays, as it does in the shell). An architecture with no fixture directory,
/// a directory that cannot be listed, one holding no such file or more than
/// one, and one holding a folder's index and a GGUF first shard are [`RefError::Missing`], naming the
/// architecture or the directory and what was found.
pub fn first_shard_in(root: &str, arch: &str) -> Result<String, RefError> {
    let Some(dir) = dir_of(arch) else {
        let known: Vec<&str> = dirs().into_iter().map(|(a, _)| a).collect();
        return Err(RefError::missing(
            PathBuf::new(),
            format!(
                "fixture: the {arch} architecture has no fixture directory, the table holds {known:?}"
            ),
        ));
    };
    first_shard_at(&format!("{root}/{dir}"), "`fixture generate` writes it")
}

/// The first shard of directory `at`, joined as `"{at}/{file}"`: for a directory that holds a
/// checkpoint folder's index (`model.safetensors.index.json`) the one `*-00001-of-*.safetensors`
/// file beside it, for any other the one `*-00001-of-*.gguf`. A directory that cannot be listed,
/// an entry that cannot be read, a first shard whose name is not UTF-8, none or more than one, and
/// a folder's index beside a GGUF first shard are [`RefError::Missing`], naming the directory and
/// what was found; `maker` says what writes a GGUF split (a folder is cut by `fixture cut`).
pub fn first_shard_at(at: &str, maker: &str) -> Result<String, RefError> {
    let entries = std::fs::read_dir(at)
        .map_err(|e| RefError::missing(at, format!("fixture: cannot list {at}: {e}")))?;
    let (mut ggufs, mut shards): (Vec<String>, Vec<String>) = (Vec::new(), Vec::new());
    for entry in entries {
        let name = entry
            .map_err(|e| RefError::missing(at, format!("fixture: cannot list {at}: {e}")))?
            .file_name();
        let kind = [GGUF, SAFETENSORS]
            .into_iter()
            .find(|ext| is_first_shard(&name.to_string_lossy(), ext));
        let Some(kind) = kind else {
            continue;
        };
        let name = name.into_string().map_err(|name| {
            RefError::missing(
                at,
                format!("fixture: {at} holds a first shard whose name is not UTF-8: {name:?}"),
            )
        })?;
        if kind == GGUF {
            ggufs.push(name);
        } else {
            shards.push(name);
        }
    }
    let folder = Path::new(at).join(INDEX).is_file();
    if folder && !ggufs.is_empty() {
        ggufs.sort();
        return Err(RefError::missing(
            at,
            format!(
                "fixture: {at} holds both a checkpoint folder's index and GGUF first shards {ggufs:?}, \
                 which are two fixtures in one directory"
            ),
        ));
    }
    let (mut found, ext, made) = if folder {
        (shards, SAFETENSORS, "`fixture cut` writes it")
    } else {
        (ggufs, GGUF, maker)
    };
    found.sort();
    match found.as_slice() {
        [one] => Ok(format!("{at}/{one}")),
        _ => Err(RefError::missing(
            at,
            format!(
                "fixture: {at} holds {} first shards (*-00001-of-*{ext}): {found:?}, want one \
                 ({made})",
                found.len()
            ),
        )),
    }
}

/// Whether the shell's glob `*-00001-of-*<ext>` matches `name`: not a hidden
/// file, ending `ext`, with `-00001-of-` somewhere before that.
fn is_first_shard(name: &str, ext: &str) -> bool {
    !name.starts_with('.')
        && name
            .strip_suffix(ext)
            .is_some_and(|stem| stem.contains("-00001-of-"))
}

fn is_safetensors(path: &Path) -> bool {
    path.extension().is_some_and(|e| e == "safetensors")
}

/// The `# fixture` line of the fixture file whose first shard is
/// `first_shard`: [`LINE_PREFIX`] and [`keys`].
pub fn line(first_shard: &Path) -> Result<String, RefError> {
    Ok(format!("{LINE_PREFIX}{}", keys(first_shard)?))
}

/// `stated`, the text after the tab of the `# fixture` line set `set` carries, against the [`line`] of the fixture file
/// whose first shard is `first_shard`: [`RefError::Stale`] unless the two lines are the same string, naming both. A set
/// that states none is stale too — nothing says which generation of the fixture it came from. A line dumped while the
/// line still carried the card budget is read without that token ([`without_budget`]): the budget is not a generation.
pub fn check_stated(set: &Path, stated: Option<&str>, first_shard: &Path) -> Result<(), RefError> {
    let runs = line(first_shard)?;
    let dumped_from = stated.map(|s| format!("{LINE_PREFIX}{s}"));
    if stated.map(without_budget).as_deref() == runs.strip_prefix(LINE_PREFIX) {
        return Ok(());
    }
    Err(RefError::Stale {
        set: set.display().to_string(),
        dumped_from: dumped_from
            .unwrap_or_else(|| "an unstated fixture (the set has no # fixture line)".to_string()),
        runs,
    })
}

/// `text`, the body of a `# fixture` line, without its `bloomery.fixture.card_budget=` token: what sets dumped while
/// the line still carried the budget state, read as the line of today.
fn without_budget(text: &str) -> String {
    let token = format!("{KEY_CARD_BUDGET}=");
    text.split(' ')
        .filter(|part| !part.starts_with(&token))
        .collect::<Vec<_>>()
        .join(" ")
}

/// What follows the tab of the `# fixture` line: every header key of the file
/// that starts with `bloomery.fixture.` but the card budget ([`KEY_CARD_BUDGET`]),
/// sorted by byte order, each as
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
    if is_safetensors(first_shard) {
        return folder_keys(first_shard);
    }
    let file = first_shard.display().to_string();
    let g = open_split(first_shard, &file)?;
    let mut held: Vec<(&str, &Value)> = g
        .iter_kv()
        .filter(|(k, _)| k.starts_with(KEY_PREFIX))
        .collect();
    held.sort_by(|a, b| a.0.cmp(b.0));
    check_held(
        first_shard,
        &file,
        &held.iter().map(|(k, _)| *k).collect::<Vec<_>>(),
    )?;
    let mut parts = Vec::with_capacity(held.len());
    for (key, v) in held.into_iter().filter(|(k, _)| *k != KEY_CARD_BUDGET) {
        parts.push(format!("{key}={}", render(&file, key, v)?));
    }
    Ok(parts.join(" "))
}

/// `keys`, the fixture keys of a file (each once: both header readers refuse duplicate keys),
/// hold the version key and no subset key.
fn check_held(first_shard: &Path, file: &str, keys: &[&str]) -> Result<(), RefError> {
    if !keys.contains(&KEY_VERSION) {
        return Err(RefError::missing(
            first_shard,
            format!("fixture: {file} has no {KEY_VERSION} key: it is not a fixture file"),
        ));
    }
    if keys.contains(&KEY_SUBSET) {
        return Err(RefError::malformed(
            format!("fixture: {file}"),
            format!(
                "{KEY_SUBSET} is set: the file holds only some tensors, and no whole set is dumped from it"
            ),
        ));
    }
    Ok(())
}

/// [`keys`] of a checkpoint folder's first shard: the `bloomery.fixture.` entries of its
/// safetensors `__metadata__`, whose values are strings that already hold the text the line shows.
/// A value that is empty or holds whitespace, `=` or a control character, and a key stated twice in
/// the header, are [`RefError::Malformed`], as in a GGUF's; a shard with no metadata is no fixture.
fn folder_keys(first_shard: &Path) -> Result<String, RefError> {
    let file = first_shard.display().to_string();
    let bad = |why: String| RefError::malformed(format!("fixture: {file}"), why);
    let pairs = shard_metadata(first_shard)?;
    let mut held: Vec<(&str, &str)> = pairs
        .iter()
        .filter(|(k, _)| k.starts_with(KEY_PREFIX))
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();
    held.sort_by(|a, b| a.0.cmp(b.0));
    check_held(
        first_shard,
        &file,
        &held.iter().map(|(k, _)| *k).collect::<Vec<_>>(),
    )?;
    let mut parts = Vec::with_capacity(held.len());
    for (key, v) in held.into_iter().filter(|(k, _)| *k != KEY_CARD_BUDGET) {
        if let Some(c) = v
            .chars()
            .find(|c| c.is_whitespace() || *c == '=' || c.is_control())
        {
            return Err(bad(format!(
                "{key}: the string {v:?} holds {c:?}: whitespace, `=` and control characters \
                 cannot be told apart from the line's own separators"
            )));
        }
        if v.is_empty() {
            return Err(bad(format!("{key}: an empty value states nothing")));
        }
        parts.push(format!("{key}={v}"));
    }
    Ok(parts.join(" "))
}

/// A safetensors shard's `__metadata__` entries (every `bloomery.fixture.` one a string);
/// the tensors' table is not read.
fn shard_metadata(shard: &Path) -> Result<Vec<(String, String)>, RefError> {
    use std::io::Read;
    let file = shard.display().to_string();
    let bad = |why: String| RefError::malformed(format!("fixture: {file}"), why);
    let mut f = std::fs::File::open(shard)
        .map_err(|e| RefError::missing(shard, format!("fixture: cannot open {file}: {e}")))?;
    let mut len = [0u8; 8];
    f.read_exact(&mut len).map_err(|e| {
        bad(format!(
            "not a readable safetensors header: no length word: {e}"
        ))
    })?;
    let n = u64::from_le_bytes(len);
    if n > MAX_HEADER {
        return Err(bad(format!(
            "not a readable safetensors header: it claims {n} bytes, over {MAX_HEADER}"
        )));
    }
    let mut text =
        vec![0u8; usize::try_from(n).expect("a header of at most MAX_HEADER fits usize")];
    f.read_exact(&mut text)
        .map_err(|e| bad(format!("not a readable safetensors header: {e}")))?;
    let text = String::from_utf8(text).map_err(|_| bad("the header is not UTF-8".to_string()))?;
    let v = safetensors::parse_unique("the header", &text)
        .map_err(|e| bad(format!("not a readable safetensors header: {e}")))?;
    let Some(header) = v.as_object() else {
        return Err(bad(
            "not a readable safetensors header: not a JSON object".to_string()
        ));
    };
    let Some(meta) = header.get("__metadata__") else {
        return Ok(Vec::new());
    };
    let Some(meta) = meta.as_object() else {
        return Err(bad("__metadata__ is not an object".to_string()));
    };
    let mut pairs = Vec::with_capacity(meta.len());
    for (k, v) in meta {
        match v.as_str() {
            Some(s) => pairs.push((k.clone(), s.to_string())),
            None if k.starts_with(KEY_PREFIX) => {
                return Err(bad(format!(
                    "{k}: {v} is not a string, which a safetensors metadata value is"
                )));
            }
            None => {}
        }
    }
    Ok(pairs)
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
