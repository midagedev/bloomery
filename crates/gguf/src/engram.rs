//! A model's Engram in its header, and the file that may hold its tables.
//!
//! The header states the Engram's scalars, its hash constants (as KV arrays or
//! as integer tensors) and the file of its tables under each publisher's
//! spelling: [`EngramFacts`] is one architecture's table of them, read through
//! [`crate::facts`], and [`crate::arch::engram_facts`] looks it up by the
//! file's `general.architecture`. Both the Engram crate and the model open read
//! the facts from this crate, which is why the mechanism sits here: they depend
//! on it and it on neither.
//!
//! The Engram tables of some sets are not in the model split: a separate GGUF
//! (its architecture [`Companion::arch`]) holds them, and `<arch>.engram.file`
//! names it. [`Split::open`] attaches it ([`attach_companion`]) as a further
//! shard, so every reader that looks a tensor up in the split, or opens the
//! split's shard paths, sees the tables as part of the model.

use std::path::{Path, PathBuf};

use crate::facts::{Fact, HashFact, count, counts_opt, full_key, text_opt};
use crate::{GENERAL_ARCHITECTURE, Gguf, LoadError, Split, Value};

/// One architecture's Engram facts under every publisher's spelling.
#[derive(Clone, Copy, Debug)]
pub struct EngramFacts {
    /// The blocks that carry an Engram table.
    pub layer_ids: Fact,
    /// Hash heads per n-gram order.
    pub head_count: Fact,
    /// The longest n-gram.
    pub max_ngram: Fact,
    /// Values in one Engram row.
    pub key_length: Fact,
    /// The context value of a position before the sequence starts.
    pub pad_id: Fact,
    /// Rows of each site's table, when the file states them.
    pub rows: Fact,
    /// The companion file's name.
    pub file: Fact,
    /// The identity of the tables the model and the companion both state.
    pub source: Fact,
    /// Per site, one multiplier per n-gram position.
    pub multipliers: HashFact,
    /// Per site, one prime (bucket size) per hash head and order.
    pub primes: HashFact,
    /// Per site, each bucket's first row.
    pub offsets: HashFact,
    /// The compressed vocabulary, indexed by token id.
    pub token_map: HashFact,
    /// The tensor of Engram site `layer`'s table.
    pub table: fn(u64) -> String,
    /// The file that holds the tables when the model split lacks them.
    pub companion: Companion,
}

/// Where an architecture's Engram tables live when its model split lacks them.
#[derive(Clone, Copy, Debug)]
pub struct Companion {
    /// `general.architecture` of the companion file.
    pub arch: &'static str,
    /// The environment variable that names the companion file (the errors
    /// name it).
    pub env: &'static str,
    /// The path `env` holds, when set: used only when the model split lacks an
    /// Engram table.
    pub explicit: fn() -> Option<PathBuf>,
}

fn companion_err(path: &Path, detail: impl Into<String>) -> LoadError {
    LoadError::Companion {
        path: path.display().to_string(),
        detail: detail.into(),
    }
}

/// Where the companion named `name` sits: beside `first`, then in the
/// sibling directory named by `name`'s stem. Each is returned in lookup
/// order; `name` must be a bare file name.
fn companion_candidates(
    facts: &EngramFacts,
    first: &Path,
    name: &str,
) -> Result<[PathBuf; 2], LoadError> {
    let bare = Path::new(name);
    let stem = bare.file_stem().and_then(|s| s.to_str());
    if bare.components().count() != 1 || stem.is_none() || name.contains(['/', '\\']) {
        return Err(companion_err(
            first,
            format!("{} {name:?} is not a bare file name", facts.file.name),
        ));
    }
    let dir = first.parent().unwrap_or_else(|| Path::new("."));
    let sibling = dir.join("..").join(stem.unwrap_or_default());
    Ok([dir.join(name), sibling.join(name)])
}

/// The companion path for a model of architecture `arch` whose first shard is
/// `first`, naming `name`: `explicit` when given, else the first of
/// [`companion_candidates`] that exists; none present is an error naming
/// every path looked at.
fn find_companion(
    facts: &EngramFacts,
    arch: &str,
    first: &Path,
    name: Option<&str>,
    explicit: Option<&Path>,
) -> Result<PathBuf, LoadError> {
    let env = facts.companion.env;
    if let Some(p) = explicit {
        if p.is_file() {
            return Ok(p.to_path_buf());
        }
        return Err(companion_err(p, format!("the {env} path is not a file")));
    }
    let Some(name) = name else {
        return Err(companion_err(
            first,
            format!(
                "the model split lacks Engram tables, and neither {} nor {env} names the file that holds them",
                full_key(arch, facts.file.name)
            ),
        ));
    };
    let [beside, sibling] = companion_candidates(facts, first, name)?;
    for p in [&beside, &sibling] {
        if p.is_file() {
            return Ok(p.clone());
        }
    }
    Err(companion_err(
        first,
        format!(
            "the model split lacks Engram tables; {} names {name:?}, found neither at {} nor at {}",
            full_key(arch, facts.file.name),
            beside.display(),
            sibling.display()
        ),
    ))
}

/// The model's Engram facts the companion is checked against.
struct Sites {
    layers: Vec<u64>,
    key_length: u64,
    rows: Option<Vec<u64>>,
    source: Option<String>,
}

impl Sites {
    fn read(
        meta: &[(String, Value)],
        arch: &str,
        facts: &EngramFacts,
    ) -> Result<Option<Sites>, LoadError> {
        let Some(layers) = counts_opt(meta, arch, &facts.layer_ids).map_err(LoadError::from)?
        else {
            return Ok(None);
        };
        Ok(Some(Sites {
            key_length: count(meta, arch, &facts.key_length).map_err(LoadError::from)?,
            rows: counts_opt(meta, arch, &facts.rows).map_err(LoadError::from)?,
            source: text_opt(meta, arch, &facts.source).map_err(LoadError::from)?,
            layers,
        }))
    }
}

/// The companion `g` (at `path`) holds exactly the tables `missing` of the
/// model's `sites`, in the shape the model's header states.
fn check_companion(
    facts: &EngramFacts,
    arch: &str,
    path: &Path,
    g: &Gguf,
    sites: &Sites,
    missing: &[u64],
) -> Result<(), LoadError> {
    let bad = |detail: String| companion_err(path, detail);
    let file_arch = facts.companion.arch;
    match g.architecture() {
        Some(a) if a == file_arch => {}
        other => {
            return Err(bad(format!(
                "{GENERAL_ARCHITECTURE} is {:?}, want {file_arch:?}",
                other.unwrap_or("<missing>")
            )));
        }
    }
    if g.value("split.count")
        .and_then(Value::as_unsigned)
        .is_some_and(|c| c != 1)
    {
        return Err(bad(
            "is one shard of a split set; the companion is one file".to_string(),
        ));
    }
    let theirs = text_opt(g.keys(), arch, &facts.source).map_err(LoadError::from)?;
    if let (Some(ours), Some(theirs)) = (&sites.source, &theirs)
        && ours != theirs
    {
        return Err(bad(format!(
            "{} is {theirs:?}, the model's is {ours:?}: the tables are another export's",
            full_key(arch, facts.source.name)
        )));
    }
    for t in g.iter_tensors() {
        let site = sites
            .layers
            .iter()
            .position(|&l| (facts.table)(l) == t.name);
        let Some(site) = site else {
            return Err(bad(format!(
                "{} is not the table of an Engram layer of the model ({:?})",
                t.name, sites.layers
            )));
        };
        let rows = match t.dims[..] {
            [width, rows] if width == sites.key_length => rows,
            _ => {
                return Err(bad(format!(
                    "{} has dims {:?}, want [{}, rows]",
                    t.name, t.dims, sites.key_length
                )));
            }
        };
        if let Some(want) = sites.rows.as_ref().map(|r| r.get(site))
            && want != Some(&rows)
        {
            return Err(bad(format!(
                "{} has {rows} rows, {} says {want:?}",
                t.name,
                full_key(arch, facts.rows.name)
            )));
        }
    }
    for &l in missing {
        let name = (facts.table)(l);
        if g.find(&name).is_none() {
            return Err(bad(format!(
                "lacks {name}, which the model split lacks too"
            )));
        }
    }
    Ok(())
}

/// Attach the Engram companion to `split` when the model needs one: an
/// architecture with an Engram table ([`crate::arch::engram_facts`]) whose
/// header lists Engram sites, a model split that lacks a site's table, and the
/// file that holds it ([`find_companion`]; the architecture's
/// [`Companion::env`] names it outright). A split whose tables are all
/// present, or a header with no Engram sites, is left alone; the variable set
/// for such a model is refused, because the file it names would be ignored.
pub fn attach_companion(split: &mut Split) -> Result<(), LoadError> {
    let Some(arch) = split.architecture().map(str::to_string) else {
        return Ok(());
    };
    let Some(facts) = crate::arch::engram_facts(&arch) else {
        return Ok(());
    };
    let explicit = (facts.companion.explicit)();
    attach_companion_with(split, &arch, facts, explicit.as_deref())
}

pub(crate) fn attach_companion_with(
    split: &mut Split,
    arch: &str,
    facts: &EngramFacts,
    explicit: Option<&Path>,
) -> Result<(), LoadError> {
    let first = split
        .shard_path(0)
        .map(Path::to_path_buf)
        .unwrap_or_default();
    let sites = Sites::read(split.keys(), arch, facts)?;
    let missing: Vec<u64> = sites.as_ref().map_or_else(Vec::new, |s| {
        let lacking = |&l: &u64| split.find(&(facts.table)(l)).is_none();
        s.layers.iter().copied().filter(lacking).collect()
    });
    let (Some(sites), false) = (&sites, missing.is_empty()) else {
        return match explicit {
            Some(p) => Err(companion_err(
                p,
                format!(
                    "{} names a file, and the model split {} holds every Engram table it needs",
                    facts.companion.env,
                    first.display()
                ),
            )),
            None => Ok(()),
        };
    };
    let name = text_opt(split.keys(), arch, &facts.file).map_err(LoadError::from)?;
    if name.is_none() && explicit.is_none() {
        return Ok(());
    }
    let path = find_companion(facts, arch, &first, name.as_deref(), explicit)?;
    let g = crate::open_shard(&path)?;
    check_companion(facts, arch, &path, &g, sites, &missing)?;
    split.attach_shard(g, path)
}
