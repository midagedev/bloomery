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

use crate::facts::{Fact, FactError, HashFact, count, counts, counts_opt, full_key, text_opt};
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
    /// The tensor of Engram site `layer`'s table ([`EngramLayout::PerSite`]).
    pub table: fn(u64) -> String,
    /// The one tensor that holds every site's rows back to back, in site
    /// order ([`EngramLayout::Concatenated`]).
    pub concat_table: &'static str,
    /// Per site, each bucket's first row counted from the concatenated
    /// table's row 0: what places a site in it.
    pub concat_offsets: Fact,
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

/// How a file lays out its Engram tables.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EngramLayout {
    /// One tensor a site ([`EngramFacts::table`]), the site's rows from its
    /// row 0.
    PerSite,
    /// Every site's rows in one tensor ([`EngramFacts::concat_table`]), site
    /// after site in `layer_ids` order; a site starts at its first bucket's
    /// offset ([`EngramFacts::concat_offsets`]).
    Concatenated,
}

/// One Engram site's rows in the file: the tensor that holds them, the
/// tensor row the site's row 0 is, the site's row count, the rows' ggml type
/// id and their width (the tensor's `ne[0]`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SiteTable {
    /// The block the site is.
    pub layer: u64,
    pub tensor: String,
    pub base_row: u64,
    pub rows: u64,
    pub type_id: u32,
    pub width: u64,
}

/// A tensor's header as [`EngramTables::read`] looks it up: its dims
/// (`ne[0]` first) and its ggml type id.
pub type TableHeader = (Vec<u64>, u32);

/// Every Engram site's rows, by the file's layout: the one owner of where a
/// site's rows are, for the model's open and the Engram crate's header-only
/// one alike.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EngramTables {
    pub layout: EngramLayout,
    /// In `layer_ids` order.
    pub sites: Vec<SiteTable>,
}

fn table_err(name: &str, detail: impl Into<String>) -> LoadError {
    LoadError::EngramTable {
        name: name.to_string(),
        detail: detail.into(),
    }
}

impl EngramLayout {
    /// The layout of a file whose header is `meta` and which holds tensor
    /// `name` when `has(name)`; `None` when the header lists no Engram site.
    /// The concatenated table beside a per-site one is refused by name. A
    /// file with neither is [`EngramLayout::PerSite`]: its tables may sit in
    /// a companion ([`attach_companion`]), and [`EngramTables::read`] names
    /// what is missing.
    pub fn of(
        meta: &[(String, Value)],
        arch: &str,
        facts: &EngramFacts,
        has: impl Fn(&str) -> bool,
    ) -> Result<Option<EngramLayout>, LoadError> {
        let Some(layers) = counts_opt(meta, arch, &facts.layer_ids).map_err(LoadError::from)?
        else {
            return Ok(None);
        };
        let per_site: Vec<String> = layers
            .iter()
            .map(|&l| (facts.table)(l))
            .filter(|n| has(n))
            .collect();
        match (has(facts.concat_table), per_site.is_empty()) {
            (true, false) => Err(table_err(
                facts.concat_table,
                format!(
                    "is in the file beside the per-site tables {per_site:?}: two layouts of one Engram"
                ),
            )),
            (true, true) => Ok(Some(EngramLayout::Concatenated)),
            (false, _) => Ok(Some(EngramLayout::PerSite)),
        }
    }
}

impl EngramTables {
    /// Every site's rows in a file whose header is `meta` and whose tensor
    /// `name` has the header `find(name)`; `None` when the header lists no
    /// Engram site. Refused by name: a per-site table that is missing or not
    /// 2-D, and a concatenated table whose sites (each `Σ` its buckets'
    /// primes rows, at its first bucket's [`EngramFacts::concat_offsets`])
    /// do not follow one another from row 0 to its last row.
    pub fn read(
        meta: &[(String, Value)],
        arch: &str,
        facts: &EngramFacts,
        find: impl Fn(&str) -> Option<TableHeader>,
    ) -> Result<Option<EngramTables>, LoadError> {
        let Some(layout) = EngramLayout::of(meta, arch, facts, |n| find(n).is_some())? else {
            return Ok(None);
        };
        let layers = counts(meta, arch, &facts.layer_ids).map_err(LoadError::from)?;
        let two_d = |name: &str, (dims, ty): TableHeader| match dims[..] {
            [width, rows] => Ok((width, rows, ty)),
            _ => Err(table_err(
                name,
                format!("has dims {dims:?}, want [width, rows]"),
            )),
        };
        let sites = match layout {
            EngramLayout::PerSite => layers
                .iter()
                .map(|&layer| {
                    let tensor = (facts.table)(layer);
                    let header = find(&tensor).ok_or_else(|| {
                        table_err(
                            &tensor,
                            format!(
                                "is not in the file, nor is {}, and {} lists layer {layer}",
                                facts.concat_table,
                                full_key(arch, facts.layer_ids.name)
                            ),
                        )
                    })?;
                    let (width, rows, type_id) = two_d(&tensor, header)?;
                    Ok(SiteTable {
                        layer,
                        tensor,
                        base_row: 0,
                        rows,
                        type_id,
                        width,
                    })
                })
                .collect::<Result<Vec<_>, LoadError>>()?,
            EngramLayout::Concatenated => concatenated(meta, arch, facts, &layers, &find, two_d)?,
        };
        Ok(Some(EngramTables { layout, sites }))
    }

    /// The site of block `layer`, if the header lists it.
    #[must_use]
    pub fn site(&self, layer: u64) -> Option<&SiteTable> {
        self.sites.iter().find(|s| s.layer == layer)
    }
}

/// The sites of [`EngramLayout::Concatenated`]: site `e`'s rows are the sum
/// of its buckets' primes and start at its first bucket's absolute offset,
/// which must be where the sites before it end; the last ends at the
/// table's last row.
fn concatenated(
    meta: &[(String, Value)],
    arch: &str,
    facts: &EngramFacts,
    layers: &[u64],
    find: &impl Fn(&str) -> Option<TableHeader>,
    two_d: impl Fn(&str, TableHeader) -> Result<(u64, u64, u32), LoadError>,
) -> Result<Vec<SiteTable>, LoadError> {
    let name = facts.concat_table;
    let header = find(name).ok_or_else(|| table_err(name, "is not in the file"))?;
    let (width, total, type_id) = two_d(name, header)?;
    let fact = |e: FactError| LoadError::from(e);
    let primes = counts(meta, arch, &facts.primes.kv).map_err(fact)?;
    let offsets = counts(meta, arch, &facts.concat_offsets).map_err(fact)?;
    let (primes_key, offsets_key) = (
        full_key(arch, facts.primes.kv.name),
        full_key(arch, facts.concat_offsets.name),
    );
    let sites = layers.len();
    if primes.is_empty() || !primes.len().is_multiple_of(sites) || offsets.len() != primes.len() {
        return Err(table_err(
            name,
            format!(
                "{primes_key} holds {} buckets and {offsets_key} {}: not the same whole number for each of {sites} sites",
                primes.len(),
                offsets.len()
            ),
        ));
    }
    let n_cols = primes.len() / sites;
    let mut next = 0u64;
    let mut out = Vec::with_capacity(sites);
    for (e, &layer) in layers.iter().enumerate() {
        let chunk = &primes[e * n_cols..(e + 1) * n_cols];
        let rows = chunk
            .iter()
            .try_fold(0u64, |a, &p| a.checked_add(p))
            .ok_or_else(|| table_err(name, format!("site {e}'s buckets pass a u64 of rows")))?;
        let base_row = offsets[e * n_cols];
        if base_row != next {
            return Err(table_err(
                name,
                format!(
                    "{offsets_key} starts site {e} (layer {layer}) at row {base_row}; the sites before it end at row {next}"
                ),
            ));
        }
        next = base_row
            .checked_add(rows)
            .ok_or_else(|| table_err(name, format!("site {e} ends past a u64 row")))?;
        out.push(SiteTable {
            layer,
            tensor: name.to_string(),
            base_row,
            rows,
            type_id,
            width,
        });
    }
    if next != total {
        return Err(table_err(
            name,
            format!("has {total} rows; the sites' buckets ({primes_key}) cover {next}"),
        ));
    }
    Ok(out)
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
    let layout = EngramLayout::of(split.keys(), arch, facts, |n| split.find(n).is_some())?;
    let missing: Vec<u64> = match (&sites, layout) {
        (Some(s), Some(EngramLayout::PerSite)) => {
            let lacking = |&l: &u64| split.find(&(facts.table)(l)).is_none();
            s.layers.iter().copied().filter(lacking).collect()
        }
        _ => Vec::new(),
    };
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
