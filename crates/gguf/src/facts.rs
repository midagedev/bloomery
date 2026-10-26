//! A header fact read under every publisher's spelling.
//!
//! The engine reads each fact under one canonical key (`<arch>.<name>`, the
//! spelling of the reference converter). Other publishers write the same fact
//! under another key, as another type, or as a tensor. A [`Fact`] row lists the
//! canonical key's suffix and the other spellings, and a fact is read as its
//! row says: the first spelling present wins, and a file holding two spellings
//! with different values (compared after the typed conversion) is refused by
//! name ([`FactError::Conflict`]) rather than resolved by a pick. The rows are
//! each architecture's own (`crate::arch`); this module is the mechanism.
//!
//! The facts are read from a header's metadata pairs ([`crate::Split::keys`] for the
//! model open, [`Inventory`] for a header-only open, which also reads a hash
//! constant's tensor bytes).

use std::fmt;
use std::io::{Read, Seek, SeekFrom};

use crate::{GgmlType, Inventory, Value};

/// What a spelling's value converts to, and so what two spellings are
/// compared by.
#[derive(Clone, Copy, Debug)]
pub enum Kind {
    /// An unsigned integer.
    Count,
    /// An array of unsigned integers.
    Counts,
    /// A number the publishers also spell another way: the architecture's
    /// converter reads the stored value, and the refusal it returns is the
    /// detail of [`FactError::Bad`].
    Converted(fn(&Value) -> Result<u64, String>),
    /// A string.
    Text,
}

/// How a spelling's value becomes the canonical one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Conv {
    /// As stored.
    Plain,
    /// A publisher's offsets count rows across the sites' tables, one after
    /// the other; the canonical offsets count from each site's own first row.
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

impl Fact {
    #[must_use]
    pub const fn new(
        name: &'static str,
        others: &'static [(&'static str, Conv)],
        kind: Kind,
    ) -> Fact {
        Fact { name, others, kind }
    }
}

/// One hash constant: a KV array under any of its spellings, or its tensor.
#[derive(Clone, Copy, Debug)]
pub struct HashFact {
    pub(crate) kv: Fact,
    tensor: &'static str,
}

impl HashFact {
    #[must_use]
    pub const fn new(kv: Fact, tensor: &'static str) -> HashFact {
        HashFact { kv, tensor }
    }

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

/// A fact value after its typed conversion.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Norm {
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

pub(crate) fn full_key(arch: &str, suffix: &str) -> String {
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
        Kind::Converted(convert) => convert(value)
            .map(Norm::Count)
            .map_err(|detail| bad(key, detail)),
        Kind::Text => match value {
            Value::String(s) => Ok(Norm::Text(s.clone())),
            other => Err(bad(key, format!("is {other:?}, not a string"))),
        },
    }
}

/// A [`Conv::PerSite`] fact made per-site: each of `sites` equal chunks
/// counted from its own first row.
fn per_site(key: &str, v: Vec<u64>, sites: usize) -> Result<Vec<u64>, FactError> {
    if v.is_empty() || sites == 0 || !v.len().is_multiple_of(sites) {
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
pub(crate) fn resolve(
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
