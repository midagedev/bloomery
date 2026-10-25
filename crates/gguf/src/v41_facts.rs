//! DeepSeek-V4.1's header facts under every publisher's spelling, and the
//! Engram companion file: the one owner of both.
//!
//! The engine reads each fact under one canonical key (`<arch>.<name>`, the
//! spelling of the `Q3_K_M` set). Other publishers write the same fact under
//! another key, as another type, or as a tensor:
//!
//! * ISTA-DASLab's sets: the router's scoring function as the string
//!   `expert_scoring_func` beside (or instead of) the number
//!   `expert_gating_func`; the Engram scalars as `engram_n_heads`,
//!   `engram_max_ngram_size`, `engram_head_dim`, `engram_pad_token_id`; the
//!   Engram hash constants as the four integer tensors of [`HASH_TENSORS`]
//!   (token map I32, primes, offsets and multipliers I64, one row a site)
//!   instead of KV arrays; the candidate mask under its own names.
//! * Unsloth's sets: the Engram as llama.cpp's `ple.*` keys
//!   (`ple.layers`, `ple.ngram_size`, `ple.heads_per_ngram`,
//!   `embedding_length_per_layer_input`, `ple.layer_multipliers`,
//!   `ple.head_vocab_sizes`, `ple.head_offsets`) and `engram.pad_token_id`.
//!
//! The spellings are Strata's packer tables (`ALIASES`, `ple_to_engram`,
//! `int_tensors_to_engram` of its v0.2.4 `pack_gguf.py`) checked against the
//! published headers. A fact is read as its [`Fact`] row says: the first
//! spelling present wins, and a file holding two spellings with different
//! values (compared after the typed conversion) is refused by name
//! ([`FactError::Conflict`]) rather than resolved by a pick.
//!
//! The facts are read from a header's metadata pairs ([`Split::keys`] for the
//! model open, [`Inventory`] for the Engram crate's header-only open, which
//! also reads the hash tensors' bytes). Both crates depend on this one and this
//! one on neither, which is why the owner sits here and not beside the model's
//! hyperparameters.
//!
//! The Engram tables of an ISTA set are not in its model split: a separate
//! GGUF (architecture [`ENGRAM_ARCH`]) holds them, and
//! `<arch>.engram.file` names it. [`Split::open`] attaches it
//! ([`attach_companion`]) as a further shard, so every reader that looks a
//! tensor up in the split, or opens the split's shard paths, sees the tables
//! as part of the model.

use std::fmt;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use crate::{GENERAL_ARCHITECTURE, GgmlType, Gguf, Inventory, LoadError, Split, Value};

/// ik's `LLM_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS`
/// (`llama-hparams.h:18`), which `llama-hparams.cpp:38` names
/// [`SQRT_SOFTPLUS`]. The DeepSeek reference's `score_func == "sqrtsoftplus"`
/// is `F.softplus(scores).sqrt()` (`inference/model.py:817`), the same
/// function as `sqrt(x > 20 ? x : ln(1 + e^x))`, so the string is this value.
pub const IK_SQRT_SOFTPLUS: u64 = 4;

/// The scoring function's name in the publishers' string spelling.
pub const SQRT_SOFTPLUS: &str = "sqrtsoftplus";

/// `general.architecture` of the Engram companion file.
pub const ENGRAM_ARCH: &str = "deepseek41-engram";

/// The architecture whose header holds the Engram facts.
pub const ARCH: &str = "deepseek41";

/// The environment variable that names the companion file: a path. Used only
/// when the model split lacks an Engram table.
pub const COMPANION_ENV: &str = "BLOOMERY_V41_ENGRAM";

/// The integer tensors an ISTA set stores the hash constants in, as
/// (token map, primes, offsets, multipliers).
pub const HASH_TENSORS: [&str; 4] = [
    "engram_token_map.weight",
    "engram_primes.weight",
    "engram_offsets.weight",
    "engram_multipliers.weight",
];

/// Whether `name` is one of the [`HASH_TENSORS`].
#[must_use]
pub fn is_hash_tensor(name: &str) -> bool {
    HASH_TENSORS.contains(&name)
}

/// The tensor of Engram site `layer`'s table.
#[must_use]
pub fn table_name(layer: u64) -> String {
    format!("blk.{layer}.engram_embd.weight")
}

/// What a spelling's value converts to, and so what two spellings are
/// compared by.
#[derive(Clone, Copy, Debug)]
enum Kind {
    /// An unsigned integer.
    Count,
    /// An array of unsigned integers.
    Counts,
    /// The router's scoring function: the number, or a name that is one.
    Scoring,
    /// A string.
    Text,
}

/// How a spelling's value becomes the canonical one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Conv {
    /// As stored.
    Plain,
    /// `ple.head_offsets` counts rows across the sites' tables, one after the
    /// other; the canonical offsets count from each site's own first row.
    PerSite,
}

/// One fact: the canonical key suffix and the other publishers' spellings.
#[derive(Clone, Copy, Debug)]
pub struct Fact {
    /// The canonical key, after `<arch>.`.
    pub name: &'static str,
    others: &'static [(&'static str, Conv)],
    kind: Kind,
}

const fn plain(name: &'static str, others: &'static [(&'static str, Conv)], kind: Kind) -> Fact {
    Fact { name, others, kind }
}

/// The router's scoring function, as the number the engine reads.
pub const GATING_FUNC: Fact = plain(
    "expert_gating_func",
    &[("expert_scoring_func", Conv::Plain)],
    Kind::Scoring,
);
/// The blocks that carry an Engram table.
pub const ENGRAM_LAYER_IDS: Fact = plain(
    "engram.layer_ids",
    &[("ple.layers", Conv::Plain)],
    Kind::Counts,
);
/// Hash heads per n-gram order.
pub const ENGRAM_HEAD_COUNT: Fact = plain(
    "engram.head_count",
    &[
        ("engram_n_heads", Conv::Plain),
        ("ple.heads_per_ngram", Conv::Plain),
    ],
    Kind::Count,
);
/// The longest n-gram.
pub const ENGRAM_MAX_NGRAM: Fact = plain(
    "engram.max_ngram_size",
    &[
        ("engram_max_ngram_size", Conv::Plain),
        ("ple.ngram_size", Conv::Plain),
    ],
    Kind::Count,
);
/// Values in one Engram row.
pub const ENGRAM_KEY_LENGTH: Fact = plain(
    "engram.key_length",
    &[
        ("engram_head_dim", Conv::Plain),
        ("embedding_length_per_layer_input", Conv::Plain),
    ],
    Kind::Count,
);
/// The context value of a position before the sequence starts.
pub const ENGRAM_PAD_ID: Fact = plain(
    "engram.pad_id",
    &[
        ("engram_pad_token_id", Conv::Plain),
        ("engram.pad_token_id", Conv::Plain),
    ],
    Kind::Count,
);
/// Rows of each site's table, when the file states them.
pub const ENGRAM_ROWS: Fact = plain("engram.num_embeddings", &[], Kind::Counts);
/// The companion file's name.
pub const ENGRAM_FILE: Fact = plain("engram.file", &[], Kind::Text);
/// The identity of the tables the model and the companion both state.
pub const ENGRAM_SOURCE: Fact = plain("engram.source", &[], Kind::Text);
/// The candidate mask's source layer.
pub const CANDIDATE_SOURCE: Fact = plain(
    "attention.candidate_source_layer_id",
    &[("attention.indexer.candidate_source_layer", Conv::Plain)],
    Kind::Count,
);
/// The candidate mask's rows per block.
pub const CANDIDATE_BLOCK: Fact = plain(
    "attention.candidate_block_size",
    &[("attention.indexer.candidate_block_size", Conv::Plain)],
    Kind::Count,
);
/// The candidate mask's kept blocks.
pub const CANDIDATE_TOPK: Fact = plain(
    "attention.candidate_topk_blocks",
    &[("attention.indexer.candidate_top_k_blocks", Conv::Plain)],
    Kind::Count,
);

/// One hash constant: a KV array under any of its spellings, or its tensor.
#[derive(Clone, Copy, Debug)]
pub struct HashFact {
    kv: Fact,
    tensor: &'static str,
}

impl HashFact {
    /// The canonical key of the array spelling, after `<arch>.`.
    #[must_use]
    pub const fn kv_name(&self) -> &'static str {
        self.kv.name
    }

    /// The name of the tensor spelling.
    #[must_use]
    pub const fn tensor_name(&self) -> &'static str {
        self.tensor
    }
}

/// Per site, one multiplier per n-gram position.
pub const MULTIPLIERS: HashFact = HashFact {
    kv: plain(
        "engram.multipliers",
        &[("ple.layer_multipliers", Conv::Plain)],
        Kind::Counts,
    ),
    tensor: HASH_TENSORS[3],
};
/// Per site, one prime (bucket size) per hash head and order.
pub const PRIMES: HashFact = HashFact {
    kv: plain(
        "engram.primes",
        &[("ple.head_vocab_sizes", Conv::Plain)],
        Kind::Counts,
    ),
    tensor: HASH_TENSORS[1],
};
/// Per site, each bucket's first row.
pub const OFFSETS: HashFact = HashFact {
    kv: plain(
        "engram.offsets",
        &[("ple.head_offsets", Conv::PerSite)],
        Kind::Counts,
    ),
    tensor: HASH_TENSORS[2],
};
/// The compressed vocabulary, indexed by token id.
pub const TOKEN_MAP: HashFact = HashFact {
    kv: plain("engram.token_map", &[], Kind::Counts),
    tensor: HASH_TENSORS[0],
};

/// A fact value after its typed conversion.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Norm {
    Count(u64),
    Counts(Vec<u64>),
    Text(String),
}

impl fmt::Display for Norm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Norm::Count(v) => write!(f, "{v}"),
            Norm::Text(s) => write!(f, "{s:?}"),
            Norm::Counts(v) if v.len() <= 6 => write!(f, "{v:?}"),
            Norm::Counts(v) => write!(f, "{} values from {}..{}", v.len(), v[0], v[v.len() - 1]),
        }
    }
}

/// Why a fact could not be read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FactError {
    /// No spelling of the fact is in the file.
    Absent { key: String, others: Vec<String> },
    /// A spelling holds a value the fact cannot take.
    Bad { key: String, detail: String },
    /// Two spellings hold different values.
    Conflict {
        first: String,
        first_value: String,
        second: String,
        second_value: String,
    },
}

impl FactError {
    /// The key the error names (the first of a conflict's two).
    #[must_use]
    pub fn key(&self) -> &str {
        match self {
            FactError::Absent { key, .. } | FactError::Bad { key, .. } => key,
            FactError::Conflict { first, .. } => first,
        }
    }

    /// What is wrong with [`FactError::key`], without the key.
    #[must_use]
    pub fn detail(&self) -> String {
        match self {
            FactError::Absent { others, .. } if others.is_empty() => "is absent".to_string(),
            FactError::Absent { others, .. } => {
                format!("is absent (also looked for {})", others.join(", "))
            }
            FactError::Bad { detail, .. } => detail.clone(),
            FactError::Conflict {
                first_value,
                second,
                second_value,
                ..
            } => format!(
                "is {first_value} but {second} is {second_value}: the file states one fact two ways"
            ),
        }
    }
}

impl fmt::Display for FactError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.key(), self.detail())
    }
}

impl std::error::Error for FactError {}

fn bad(key: &str, detail: impl Into<String>) -> FactError {
    FactError::Bad {
        key: key.to_string(),
        detail: detail.into(),
    }
}

/// An integer tensor's dims (ggml order, `ne[0]` first) and values.
#[derive(Clone, Debug, PartialEq, Eq)]
struct IntTensor {
    dims: Vec<u64>,
    values: Vec<u64>,
}

/// An I32 tensor's values must be non-negative ids; an I64 tensor's are taken
/// as their two's-complement bit pattern, which is what the hash's wrapping
/// `u64` arithmetic reads.
fn decode_ints(
    name: &str,
    ty: GgmlType,
    dims: &[u64],
    bytes: &[u8],
) -> Result<IntTensor, FactError> {
    let width = match ty {
        GgmlType::I32 => 4,
        GgmlType::I64 => 8,
        other => return Err(bad(name, format!("is {other}, not an I32 or I64 tensor"))),
    };
    let n = dims.iter().product::<u64>();
    if usize::try_from(n).ok().and_then(|n| n.checked_mul(width)) != Some(bytes.len()) {
        return Err(bad(
            name,
            format!(
                "holds {} bytes for dims {dims:?} of {width}-byte values",
                bytes.len()
            ),
        ));
    }
    let mut values = Vec::with_capacity(bytes.len() / width);
    for (i, c) in bytes.chunks_exact(width).enumerate() {
        values.push(match ty {
            GgmlType::I32 => {
                let v = i32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                u64::try_from(v).map_err(|_| bad(name, format!("entry {i} is {v}, negative")))?
            }
            _ => u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]),
        });
    }
    Ok(IntTensor {
        dims: dims.to_vec(),
        values,
    })
}

/// The I32 or I64 tensor `name` of the file `inv` describes: its dims and its
/// values widened to `u64`; `None` when the file has no such tensor.
fn int_tensor(inv: &Inventory, name: &str) -> Result<Option<IntTensor>, FactError> {
    let Some(t) = inv.tensors.iter().find(|t| t.name == name) else {
        return Ok(None);
    };
    let ty = GgmlType::from_u32(t.type_id);
    let (Some(path), Some(nbytes)) = (inv.path.as_deref(), t.nbytes) else {
        return Err(bad(name, "has no file to read its values from"));
    };
    let read = || -> std::io::Result<Vec<u8>> {
        let mut f = std::fs::File::open(path)?;
        f.seek(SeekFrom::Start(inv.data_base + t.offset))?;
        let mut bytes = vec![0; usize::try_from(nbytes).map_err(std::io::Error::other)?];
        f.read_exact(&mut bytes)?;
        Ok(bytes)
    };
    let bytes =
        read().map_err(|e| bad(name, format!("cannot be read from {}: {e}", path.display())))?;
    decode_ints(name, ty, &t.dims, &bytes).map(Some)
}

fn full_key(arch: &str, suffix: &str) -> String {
    format!("{arch}.{suffix}")
}

/// `value` converted as `kind` says.
fn norm(key: &str, value: &Value, kind: Kind) -> Result<Norm, FactError> {
    let unsigned = |v: &Value| v.as_unsigned();
    match kind {
        Kind::Count => unsigned(value)
            .map(Norm::Count)
            .ok_or_else(|| bad(key, format!("is {value:?}, not a non-negative integer"))),
        Kind::Counts => {
            let Value::Array(items) = value else {
                return Err(bad(key, format!("is {value:?}, not an array")));
            };
            items
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    unsigned(v).ok_or_else(|| {
                        bad(
                            key,
                            format!("entry {i} is {v:?}, not a non-negative integer"),
                        )
                    })
                })
                .collect::<Result<_, _>>()
                .map(Norm::Counts)
        }
        Kind::Scoring => match value {
            Value::String(s) if s == SQRT_SOFTPLUS => Ok(Norm::Count(IK_SQRT_SOFTPLUS)),
            Value::String(s) => Err(bad(
                key,
                format!(
                    "is {s:?}; the V4 and V4.1 routers score with {SQRT_SOFTPLUS:?} (ik's type {IK_SQRT_SOFTPLUS})"
                ),
            )),
            other => unsigned(other)
                .map(Norm::Count)
                .ok_or_else(|| bad(key, format!("is {other:?}, not a scoring function"))),
        },
        Kind::Text => match value {
            Value::String(s) => Ok(Norm::Text(s.clone())),
            other => Err(bad(key, format!("is {other:?}, not a string"))),
        },
    }
}

/// `ple.head_offsets` made per-site: each of `sites` equal chunks counted from
/// its own first row.
fn per_site(key: &str, v: Vec<u64>, sites: usize) -> Result<Vec<u64>, FactError> {
    if sites == 0 || !v.len().is_multiple_of(sites) {
        return Err(bad(
            key,
            format!(
                "has {} values, which {sites} sites do not share equally",
                v.len()
            ),
        ));
    }
    let mut out = Vec::with_capacity(v.len());
    for chunk in v.chunks(v.len() / sites) {
        let first = chunk[0];
        for &x in chunk {
            out.push(
                x.checked_sub(first).ok_or_else(|| {
                    bad(key, format!("has {x} below its site's first row {first}"))
                })?,
            );
        }
    }
    Ok(out)
}

/// The fact's value under the first spelling present, refusing two spellings
/// that disagree. `sites` is the Engram site count, which [`Conv::PerSite`]
/// divides by.
fn resolve(
    meta: &[(String, Value)],
    arch: &str,
    fact: &Fact,
    sites: usize,
) -> Result<Option<(String, Norm)>, FactError> {
    let mut found: Option<(String, Norm)> = None;
    let names = std::iter::once((fact.name, Conv::Plain)).chain(fact.others.iter().copied());
    for (suffix, conv) in names {
        let key = full_key(arch, suffix);
        let Some((_, value)) = meta.iter().find(|(k, _)| *k == key) else {
            continue;
        };
        let mut n = norm(&key, value, fact.kind)?;
        if conv == Conv::PerSite {
            let Norm::Counts(v) = n else {
                unreachable!("PerSite rows are Counts")
            };
            n = Norm::Counts(per_site(&key, v, sites)?);
        }
        match &found {
            Some((first, was)) if *was != n => {
                return Err(FactError::Conflict {
                    first: first.clone(),
                    first_value: was.to_string(),
                    second: key,
                    second_value: n.to_string(),
                });
            }
            Some(_) => {}
            None => found = Some((key, n)),
        }
    }
    Ok(found)
}

fn absent(arch: &str, fact: &Fact) -> FactError {
    FactError::Absent {
        key: full_key(arch, fact.name),
        others: fact.others.iter().map(|(s, _)| full_key(arch, s)).collect(),
    }
}

/// An unsigned count, `None` when no spelling is in the file.
pub fn count_opt(
    meta: &[(String, Value)],
    arch: &str,
    fact: &Fact,
) -> Result<Option<u64>, FactError> {
    match resolve(meta, arch, fact, 1)? {
        Some((_, Norm::Count(v))) => Ok(Some(v)),
        Some(_) => unreachable!("a Count fact resolves to a Count"),
        None => Ok(None),
    }
}

/// An unsigned count; absent is [`FactError::Absent`].
pub fn count(meta: &[(String, Value)], arch: &str, fact: &Fact) -> Result<u64, FactError> {
    count_opt(meta, arch, fact)?.ok_or_else(|| absent(arch, fact))
}

/// An array of unsigned counts, `None` when no spelling is in the file.
pub fn counts_opt(
    meta: &[(String, Value)],
    arch: &str,
    fact: &Fact,
) -> Result<Option<Vec<u64>>, FactError> {
    match resolve(meta, arch, fact, 1)? {
        Some((_, Norm::Counts(v))) => Ok(Some(v)),
        Some(_) => unreachable!("a Counts fact resolves to Counts"),
        None => Ok(None),
    }
}

/// An array of unsigned counts; absent is [`FactError::Absent`].
pub fn counts(meta: &[(String, Value)], arch: &str, fact: &Fact) -> Result<Vec<u64>, FactError> {
    counts_opt(meta, arch, fact)?.ok_or_else(|| absent(arch, fact))
}

/// A string, `None` when the file does not state it.
pub fn text_opt(
    meta: &[(String, Value)],
    arch: &str,
    fact: &Fact,
) -> Result<Option<String>, FactError> {
    match resolve(meta, arch, fact, 1)? {
        Some((_, Norm::Text(s))) => Ok(Some(s)),
        Some(_) => unreachable!("a Text fact resolves to Text"),
        None => Ok(None),
    }
}

/// The router's scoring function as the number the engine reads
/// ([`IK_SQRT_SOFTPLUS`]).
pub fn gating_func(meta: &[(String, Value)], arch: &str) -> Result<u64, FactError> {
    count(meta, arch, &GATING_FUNC)
}

/// Whether any spelling of `fact` is in the file.
#[must_use]
pub fn carries(meta: &[(String, Value)], arch: &str, fact: &Fact) -> bool {
    std::iter::once(fact.name)
        .chain(fact.others.iter().map(|(s, _)| *s))
        .any(|s| {
            let key = full_key(arch, s);
            meta.iter().any(|(k, _)| *k == key)
        })
}

/// One hash constant: its values, and the tensor's dims when it was stored
/// as a tensor (then the caller checks the orientation).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HashConstant {
    pub values: Vec<u64>,
    pub dims: Option<Vec<u64>>,
}

/// The hash constant `fact` of a `sites`-site Engram, from a KV array or from
/// its tensor; both present and different is [`FactError::Conflict`].
pub fn hash_constant(
    inv: &Inventory,
    arch: &str,
    fact: &HashFact,
    sites: usize,
) -> Result<HashConstant, FactError> {
    let kv = resolve(&inv.meta, arch, &fact.kv, sites)?;
    let tensor = int_tensor(inv, fact.tensor)?;
    match (kv, tensor) {
        (Some((_, Norm::Counts(values))), None) => Ok(HashConstant { values, dims: None }),
        (None, Some(t)) => Ok(HashConstant {
            values: t.values,
            dims: Some(t.dims),
        }),
        (Some((key, Norm::Counts(values))), Some(t)) => {
            if values != t.values {
                return Err(FactError::Conflict {
                    first: key,
                    first_value: Norm::Counts(values).to_string(),
                    second: fact.tensor.to_string(),
                    second_value: Norm::Counts(t.values).to_string(),
                });
            }
            Ok(HashConstant {
                values,
                dims: Some(t.dims),
            })
        }
        (None, None) => Err(FactError::Absent {
            key: full_key(arch, fact.kv.name),
            others: vec![fact.tensor.to_string()],
        }),
        (Some(_), _) => unreachable!("a Counts fact resolves to Counts"),
    }
}

// ------------------------------------------------------------- the companion

fn companion_err(path: &Path, detail: impl Into<String>) -> LoadError {
    LoadError::Companion {
        path: path.display().to_string(),
        detail: detail.into(),
    }
}

/// Where the companion named `name` sits: beside `first`, then in the
/// sibling directory named by `name`'s stem. Each is returned in lookup
/// order; `name` must be a bare file name.
pub fn companion_candidates(first: &Path, name: &str) -> Result<[PathBuf; 2], LoadError> {
    let bare = Path::new(name);
    let stem = bare.file_stem().and_then(|s| s.to_str());
    if bare.components().count() != 1 || stem.is_none() || name.contains(['/', '\\']) {
        return Err(companion_err(
            first,
            format!("{} {name:?} is not a bare file name", ENGRAM_FILE.name),
        ));
    }
    let dir = first.parent().unwrap_or_else(|| Path::new("."));
    let sibling = dir.join("..").join(stem.unwrap_or_default());
    Ok([dir.join(name), sibling.join(name)])
}

/// The companion path for a model whose first shard is `first`, naming
/// `name`: `explicit` when given, else the first of
/// [`companion_candidates`] that exists; none present is an error naming
/// every path looked at.
pub fn find_companion(
    first: &Path,
    name: Option<&str>,
    explicit: Option<&Path>,
) -> Result<PathBuf, LoadError> {
    if let Some(p) = explicit {
        if p.is_file() {
            return Ok(p.to_path_buf());
        }
        return Err(companion_err(
            p,
            format!("the {COMPANION_ENV} path is not a file"),
        ));
    }
    let Some(name) = name else {
        return Err(companion_err(
            first,
            format!(
                "the model split lacks Engram tables, and neither {} nor {COMPANION_ENV} names the file that holds them",
                full_key(ARCH, ENGRAM_FILE.name)
            ),
        ));
    };
    let [beside, sibling] = companion_candidates(first, name)?;
    for p in [&beside, &sibling] {
        if p.is_file() {
            return Ok(p.clone());
        }
    }
    Err(companion_err(
        first,
        format!(
            "the model split lacks Engram tables; {} names {name:?}, found neither at {} nor at {}",
            full_key(ARCH, ENGRAM_FILE.name),
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
    fn read(meta: &[(String, Value)]) -> Result<Option<Sites>, LoadError> {
        let fact = |e: FactError| LoadError::Fact(e);
        let Some(layers) = counts_opt(meta, ARCH, &ENGRAM_LAYER_IDS).map_err(fact)? else {
            return Ok(None);
        };
        Ok(Some(Sites {
            key_length: count(meta, ARCH, &ENGRAM_KEY_LENGTH).map_err(fact)?,
            rows: counts_opt(meta, ARCH, &ENGRAM_ROWS).map_err(fact)?,
            source: text_opt(meta, ARCH, &ENGRAM_SOURCE).map_err(fact)?,
            layers,
        }))
    }
}

/// The companion `g` (at `path`) holds exactly the tables `missing` of the
/// model's `sites`, in the shape the model's header states.
fn check_companion(path: &Path, g: &Gguf, sites: &Sites, missing: &[u64]) -> Result<(), LoadError> {
    let bad = |detail: String| companion_err(path, detail);
    match g.architecture() {
        Some(ENGRAM_ARCH) => {}
        other => {
            return Err(bad(format!(
                "{GENERAL_ARCHITECTURE} is {:?}, want {ENGRAM_ARCH:?}",
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
    let theirs = text_opt(g.keys(), ARCH, &ENGRAM_SOURCE).map_err(LoadError::Fact)?;
    if let (Some(ours), Some(theirs)) = (&sites.source, &theirs)
        && ours != theirs
    {
        return Err(bad(format!(
            "{} is {theirs:?}, the model's is {ours:?}: the tables are another export's",
            full_key(ARCH, ENGRAM_SOURCE.name)
        )));
    }
    for t in g.iter_tensors() {
        let site = sites.layers.iter().position(|&l| table_name(l) == t.name);
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
                full_key(ARCH, ENGRAM_ROWS.name)
            )));
        }
    }
    for &l in missing {
        let name = table_name(l);
        if g.find(&name).is_none() {
            return Err(bad(format!(
                "lacks {name}, which the model split lacks too"
            )));
        }
    }
    Ok(())
}

/// Attach the Engram companion to `split` when the model needs one: a
/// deepseek41 header with Engram sites, a model split that lacks a site's
/// table, and the file that holds it ([`find_companion`]; [`COMPANION_ENV`]
/// names it outright). A split whose tables are all present, or a header with
/// no Engram sites, is left alone; [`COMPANION_ENV`] set for such a model is
/// refused, because the file it names would be ignored.
pub fn attach_companion(split: &mut Split) -> Result<(), LoadError> {
    let explicit = std::env::var_os(COMPANION_ENV)
        .filter(|v| !v.is_empty())
        .map(PathBuf::from);
    attach_companion_with(split, explicit.as_deref())
}

fn attach_companion_with(split: &mut Split, explicit: Option<&Path>) -> Result<(), LoadError> {
    if split.architecture() != Some(ARCH) {
        return Ok(());
    }
    let first = split
        .shard_path(0)
        .map(Path::to_path_buf)
        .unwrap_or_default();
    let sites = Sites::read(split.keys())?;
    let missing: Vec<u64> = sites.as_ref().map_or_else(Vec::new, |s| {
        let lacking = |&l: &u64| split.find(&table_name(l)).is_none();
        s.layers.iter().copied().filter(lacking).collect()
    });
    let (Some(sites), false) = (&sites, missing.is_empty()) else {
        return match explicit {
            Some(p) => Err(companion_err(
                p,
                format!(
                    "{COMPANION_ENV} names a file, and the model split {} holds every Engram table it needs",
                    first.display()
                ),
            )),
            None => Ok(()),
        };
    };
    let name = text_opt(split.keys(), ARCH, &ENGRAM_FILE).map_err(LoadError::Fact)?;
    if name.is_none() && explicit.is_none() {
        return Ok(());
    }
    let path = find_companion(&first, name.as_deref(), explicit)?;
    let g = crate::open_shard(&path)?;
    check_companion(&path, &g, sites, &missing)?;
    split.attach_shard(g, path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inventory_of;
    use crate::write::{TensorDecl, write_file};

    fn dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("v41-facts-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn kv(key: &str, v: Value) -> (String, Value) {
        (key.to_string(), v)
    }

    fn text(v: &str) -> Value {
        Value::String(v.to_string())
    }

    fn u64s(v: &[u64]) -> Value {
        Value::Array(v.iter().copied().map(Value::U64).collect())
    }

    /// `<ARCH>.<suffix>` pairs on top of `general.architecture`.
    fn pairs(model: &[(&str, Value)]) -> Vec<(String, Value)> {
        let mut out = vec![kv(GENERAL_ARCHITECTURE, text(ARCH))];
        out.extend(
            model
                .iter()
                .map(|(k, v)| kv(&format!("{ARCH}.{k}"), v.clone())),
        );
        out
    }

    /// A tensor of `ty` holding `values` as little-endian integers of `width`.
    fn int_tensor_of(
        name: &str,
        ty: GgmlType,
        dims: &[u64],
        bytes: Vec<u8>,
    ) -> (TensorDecl, Vec<u8>) {
        let decl = TensorDecl {
            name: name.to_string(),
            dims: dims.to_vec(),
            type_id: ty.as_u32(),
            nbytes: bytes.len() as u64,
        };
        (decl, bytes)
    }

    fn i64_tensor(name: &str, dims: &[u64], values: &[i64]) -> (TensorDecl, Vec<u8>) {
        let bytes = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        int_tensor_of(name, GgmlType::I64, dims, bytes)
    }

    fn i32_tensor(name: &str, dims: &[u64], values: &[i32]) -> (TensorDecl, Vec<u8>) {
        let bytes = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        int_tensor_of(name, GgmlType::I32, dims, bytes)
    }

    /// The model file `<d>/<name>.gguf` of `kvs` and `tensors`.
    fn write(
        d: &Path,
        name: &str,
        kvs: &[(String, Value)],
        tensors: &[(TensorDecl, Vec<u8>)],
    ) -> PathBuf {
        let p = d.join(name);
        write_file(&p, kvs, tensors);
        p
    }

    /// The header facts of a model file carrying `model`'s keys, as
    /// [`Split::keys`] reads them.
    fn keys_of(name: &str, model: &[(&str, Value)]) -> Vec<(String, Value)> {
        let d = dir(name);
        let p = write(&d, "m.gguf", &pairs(model), &[]);
        let split = Split::open_model(&p).unwrap();
        let keys = split.keys().to_vec();
        std::fs::remove_dir_all(&d).unwrap();
        keys
    }

    #[test]
    fn a_fact_is_read_under_its_canonical_key_or_a_publishers_spelling() {
        let canonical = keys_of("canon", &[("expert_gating_func", Value::U64(4))]);
        assert_eq!(gating_func(&canonical, ARCH), Ok(IK_SQRT_SOFTPLUS));
        let ista = keys_of(
            "ista-score",
            &[("expert_scoring_func", text("sqrtsoftplus"))],
        );
        assert_eq!(gating_func(&ista, ARCH), Ok(IK_SQRT_SOFTPLUS));
        let ista = keys_of("ista-heads", &[("engram_n_heads", Value::U32(8))]);
        assert_eq!(count(&ista, ARCH, &ENGRAM_HEAD_COUNT), Ok(8));
        let unsloth = keys_of(
            "ple-heads",
            &[
                ("ple.heads_per_ngram", Value::U32(8)),
                ("engram.pad_token_id", Value::I32(2)),
            ],
        );
        assert_eq!(count(&unsloth, ARCH, &ENGRAM_HEAD_COUNT), Ok(8));
        assert_eq!(count(&unsloth, ARCH, &ENGRAM_PAD_ID), Ok(2));
        let none = keys_of("absent", &[]);
        assert_eq!(
            count(&none, ARCH, &ENGRAM_HEAD_COUNT)
                .unwrap_err()
                .to_string(),
            "deepseek41.engram.head_count is absent (also looked for deepseek41.engram_n_heads, deepseek41.ple.heads_per_ngram)"
        );
    }

    #[test]
    fn two_spellings_that_agree_read_once_and_that_differ_are_refused_by_name() {
        let both = keys_of(
            "agree",
            &[
                ("expert_gating_func", Value::U64(4)),
                ("expert_scoring_func", text("sqrtsoftplus")),
            ],
        );
        assert_eq!(gating_func(&both, ARCH), Ok(IK_SQRT_SOFTPLUS));
        let clash = keys_of(
            "clash",
            &[
                ("expert_gating_func", Value::U64(1)),
                ("expert_scoring_func", text("sqrtsoftplus")),
            ],
        );
        let e = gating_func(&clash, ARCH).unwrap_err();
        assert!(
            matches!(&e, FactError::Conflict { first, second, .. }
                if first == "deepseek41.expert_gating_func"
                    && second == "deepseek41.expert_scoring_func"),
            "{e}"
        );
        let heads = keys_of(
            "heads",
            &[
                ("engram.head_count", Value::U32(8)),
                ("engram_n_heads", Value::U32(4)),
            ],
        );
        let e = count(&heads, ARCH, &ENGRAM_HEAD_COUNT).unwrap_err();
        assert_eq!(
            e.to_string(),
            "deepseek41.engram.head_count is 8 but deepseek41.engram_n_heads is 4: the file states one fact two ways"
        );
    }

    #[test]
    fn a_scoring_function_that_is_not_sqrtsoftplus_is_refused_by_name() {
        for other in ["softmax", "sigmoid", ""] {
            let k = keys_of("score", &[("expert_scoring_func", text(other))]);
            let e = gating_func(&k, ARCH).unwrap_err();
            assert!(
                matches!(&e, FactError::Bad { key, detail }
                    if key == "deepseek41.expert_scoring_func" && detail.contains(&format!("{other:?}"))),
                "{other:?}: {e}"
            );
        }
        let k = keys_of("score-bool", &[("expert_scoring_func", Value::Bool(true))]);
        assert!(gating_func(&k, ARCH).is_err());
    }

    #[test]
    fn unsloths_head_offsets_count_from_each_sites_first_row() {
        let k = keys_of(
            "ple-offsets",
            &[("ple.head_offsets", u64s(&[0, 10, 20, 100, 110, 120]))],
        );
        let got = resolve(&k, ARCH, &OFFSETS.kv, 2).unwrap().unwrap().1;
        assert_eq!(got, Norm::Counts(vec![0, 10, 20, 0, 10, 20]));
        let e = resolve(&k, ARCH, &OFFSETS.kv, 4).unwrap_err();
        assert!(matches!(e, FactError::Bad { .. }), "{e}");
    }

    /// A file with the four hash tensors of a one-site toy Engram (2 columns,
    /// 3 n-gram slots, 5 tokens) and `extra` keys.
    fn hash_file(d: &Path, name: &str, extra: &[(&str, Value)]) -> PathBuf {
        write(
            d,
            name,
            &pairs(extra),
            &[
                i32_tensor(HASH_TENSORS[0], &[5], &[0, 1, 2, 1, 0]),
                i64_tensor(HASH_TENSORS[1], &[2, 1], &[7, 11]),
                i64_tensor(HASH_TENSORS[2], &[2, 1], &[0, 7]),
                i64_tensor(HASH_TENSORS[3], &[3, 1], &[3, 5, -1]),
            ],
        )
    }

    #[test]
    fn hash_constants_are_read_from_their_tensors_or_their_arrays() {
        let d = dir("hash");
        let inv = inventory_of(hash_file(&d, "t.gguf", &[])).unwrap();
        let mult = hash_constant(&inv, ARCH, &MULTIPLIERS, 1).unwrap();
        // An I64 entry is its two's-complement bit pattern, as the hash's wrapping u64 reads it.
        assert_eq!(mult.values, vec![3, 5, u64::MAX]);
        assert_eq!(mult.dims, Some(vec![3, 1]));
        let map = hash_constant(&inv, ARCH, &TOKEN_MAP, 1).unwrap();
        assert_eq!((map.values, map.dims), (vec![0, 1, 2, 1, 0], Some(vec![5])));

        let same = inventory_of(hash_file(
            &d,
            "same.gguf",
            &[("engram.primes", u64s(&[7, 11]))],
        ))
        .unwrap();
        assert_eq!(
            hash_constant(&same, ARCH, &PRIMES, 1).unwrap().values,
            vec![7, 11]
        );
        let apart = inventory_of(hash_file(
            &d,
            "apart.gguf",
            &[("engram.primes", u64s(&[7, 13]))],
        ))
        .unwrap();
        let e = hash_constant(&apart, ARCH, &PRIMES, 1).unwrap_err();
        assert!(
            matches!(&e, FactError::Conflict { first, second, .. }
                if first == "deepseek41.engram.primes" && second == HASH_TENSORS[1]),
            "{e}"
        );

        let kv_only = inventory_of(write(
            &d,
            "kv.gguf",
            &pairs(&[("engram.offsets", u64s(&[0, 7]))]),
            &[],
        ))
        .unwrap();
        let c = hash_constant(&kv_only, ARCH, &OFFSETS, 1).unwrap();
        assert_eq!((c.values, c.dims), (vec![0, 7], None));
        let e = hash_constant(&kv_only, ARCH, &PRIMES, 1).unwrap_err();
        assert!(matches!(e, FactError::Absent { .. }), "{e}");
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn a_hash_tensor_of_the_wrong_type_or_a_negative_id_is_refused_by_name() {
        let d = dir("hash-bad");
        let f32_primes = TensorDecl {
            name: HASH_TENSORS[1].to_string(),
            dims: vec![2, 1],
            type_id: GgmlType::F32.as_u32(),
            nbytes: 8,
        };
        let inv = inventory_of(write(
            &d,
            "f.gguf",
            &pairs(&[]),
            &[(f32_primes, vec![0; 8])],
        ))
        .unwrap();
        let e = hash_constant(&inv, ARCH, &PRIMES, 1).unwrap_err();
        assert!(
            matches!(&e, FactError::Bad { key, .. } if key == HASH_TENSORS[1]),
            "{e}"
        );

        let neg = write(
            &d,
            "n.gguf",
            &pairs(&[]),
            &[i32_tensor(HASH_TENSORS[0], &[2], &[1, -2])],
        );
        let e = hash_constant(&inventory_of(neg).unwrap(), ARCH, &TOKEN_MAP, 1).unwrap_err();
        assert!(
            matches!(&e, FactError::Bad { key, detail } if key == HASH_TENSORS[0] && detail.contains("-2")),
            "{e}"
        );
        std::fs::remove_dir_all(&d).unwrap();
    }

    /// The model file of a one-site Engram (block 1, rows 3 of width 4) that
    /// names `file` as its companion, plus the companion `engram.gguf`
    /// holding the table.
    fn pair_of_files(d: &Path, rows_in_header: u64, rows_in_table: u64) -> PathBuf {
        let model = write(
            d,
            "model.gguf",
            &pairs(&[
                ("engram.layer_ids", Value::Array(vec![Value::I32(1)])),
                ("engram.key_length", Value::U32(4)),
                ("engram.num_embeddings", u64s(&[rows_in_header])),
                ("engram.file", text("engram.gguf")),
            ]),
            &[],
        );
        let table = TensorDecl {
            name: table_name(1),
            dims: vec![4, rows_in_table],
            type_id: GgmlType::F32.as_u32(),
            nbytes: 16 * rows_in_table,
        };
        let comp_kvs = vec![kv(GENERAL_ARCHITECTURE, text(ENGRAM_ARCH))];
        write(
            d,
            "engram.gguf",
            &comp_kvs,
            &[(table, vec![9; 16 * rows_in_table as usize])],
        );
        model
    }

    fn attached(model: &Path, explicit: Option<&Path>) -> Result<Split, LoadError> {
        let mut split = Split::open_model(model)?;
        attach_companion_with(&mut split, explicit)?;
        Ok(split)
    }

    #[test]
    fn the_companion_joins_the_split_as_a_further_shard() {
        let d = dir("join");
        let model = pair_of_files(&d, 3, 3);
        let split = attached(&model, None).unwrap();
        assert_eq!(split.shard_count(), 2);
        assert_eq!(split.shard_path(1), Some(d.join("engram.gguf").as_path()));
        let (shard, info) = split.find(&table_name(1)).unwrap();
        assert_eq!((shard, info.dims.clone()), (1, vec![4, 3]));
        assert_eq!(
            split.data(&table_name(1)).unwrap(),
            vec![9u8; 48].as_slice()
        );
        // A table the header names and no file holds stays a lookup miss.
        assert!(split.find(&table_name(14)).is_none());
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn the_companion_is_looked_for_beside_the_model_then_in_the_sibling_directory() {
        let root = dir("lookup");
        let model_dir = root.join("model");
        let sibling = root.join("engram");
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::create_dir_all(&sibling).unwrap();
        // The model sits in model/, its companion in the sibling directory named by the file's stem.
        let model = pair_of_files(&model_dir, 3, 3);
        std::fs::rename(model_dir.join("engram.gguf"), sibling.join("engram.gguf")).unwrap();
        let split = attached(&model, None).unwrap();
        assert_eq!(
            split.shard_path(1).map(|p| p.canonicalize().unwrap()),
            Some(sibling.join("engram.gguf").canonicalize().unwrap())
        );

        // Beside the model comes first.
        std::fs::copy(sibling.join("engram.gguf"), model_dir.join("engram.gguf")).unwrap();
        let split = attached(&model, None).unwrap();
        assert_eq!(
            split.shard_path(1),
            Some(model_dir.join("engram.gguf").as_path())
        );

        // Neither present: the error names both paths.
        std::fs::remove_file(sibling.join("engram.gguf")).unwrap();
        std::fs::remove_file(model_dir.join("engram.gguf")).unwrap();
        let e = attached(&model, None).err().unwrap().to_string();
        let beside = model_dir.join("engram.gguf");
        let there = model_dir.join("..").join("engram").join("engram.gguf");
        assert!(
            e.contains(&beside.display().to_string()) && e.contains(&there.display().to_string()),
            "{e}"
        );
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn a_companion_that_does_not_match_the_header_is_refused_by_name() {
        let cases: [(&str, u64, u64, &str); 2] = [
            ("rows", 4, 3, "has 3 rows"),
            ("rows-more", 2, 3, "has 3 rows"),
        ];
        for (name, header_rows, table_rows, want) in cases {
            let d = dir(name);
            let model = pair_of_files(&d, header_rows, table_rows);
            let e = attached(&model, None).err().unwrap().to_string();
            assert!(
                e.contains(want) && e.contains("engram.num_embeddings"),
                "{name}: {e}"
            );
            std::fs::remove_dir_all(&d).unwrap();
        }

        // Another architecture, a table of no Engram layer, a table of another width.
        let d = dir("shape");
        let model = pair_of_files(&d, 3, 3);
        let comp = d.join("engram.gguf");
        let t = |name: String, dims: Vec<u64>| TensorDecl {
            name,
            dims,
            type_id: GgmlType::F32.as_u32(),
            nbytes: 4 * 12,
        };
        let keep = |arch: &str, tensor: TensorDecl| {
            write_file(
                &comp,
                &[kv(GENERAL_ARCHITECTURE, text(arch))],
                &[(tensor, vec![0; 48])],
            );
            attached(&model, None).err().unwrap().to_string()
        };
        assert!(
            keep("deepseek41", t(table_name(1), vec![4, 3])).contains("want \"deepseek41-engram\"")
        );
        assert!(
            keep(ENGRAM_ARCH, t(table_name(2), vec![4, 3]))
                .contains("not the table of an Engram layer")
        );
        assert!(keep(ENGRAM_ARCH, t(table_name(1), vec![6, 2])).contains("want [4, rows]"));
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn an_explicit_companion_is_taken_and_refused_for_a_model_that_needs_none() {
        let d = dir("explicit");
        let model = pair_of_files(&d, 3, 3);
        // The header names no file, so only the explicit path finds it.
        let mut split = Split::open_model(&model).unwrap();
        attach_companion_with(&mut split, Some(&d.join("engram.gguf"))).unwrap();
        assert_eq!(split.shard_count(), 2);
        let mut split = Split::open_model(&model).unwrap();
        let e = attach_companion_with(&mut split, Some(&d.join("none.gguf"))).unwrap_err();
        assert!(
            e.to_string().contains(COMPANION_ENV) && e.to_string().contains("none.gguf"),
            "{e}"
        );

        // A model whose split holds every table: the explicit file would be ignored.
        let full = write(
            &d,
            "full.gguf",
            &pairs(&[
                ("engram.layer_ids", Value::Array(vec![Value::I32(1)])),
                ("engram.key_length", Value::U32(4)),
            ]),
            &[(
                TensorDecl {
                    name: table_name(1),
                    dims: vec![4, 3],
                    type_id: GgmlType::F32.as_u32(),
                    nbytes: 48,
                },
                vec![0; 48],
            )],
        );
        let mut split = Split::open_model(&full).unwrap();
        assert!(attach_companion_with(&mut split, None).is_ok());
        assert_eq!(split.shard_count(), 1);
        let e = attach_companion_with(&mut split, Some(&d.join("engram.gguf"))).unwrap_err();
        assert!(e.to_string().contains("holds every Engram table"), "{e}");
        std::fs::remove_dir_all(&d).unwrap();
    }
}
