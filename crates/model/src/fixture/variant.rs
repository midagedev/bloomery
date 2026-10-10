//! A fixture variant: a family's fixture requantized by ik's `llama-quantize` to one type map, so a
//! quant type's train gate runs on a small file. This module owns the table (`tools/fixture-variants.tsv`),
//! the map's expansion over the source's tensor names, the `--custom-q` argument that makes the
//! quantizer retype exactly those tensors, and the check that a written variant differs from its source
//! by the map alone.
//!
//! A table row is `tag<TAB>family<TAB>scope<TAB>map<TAB>recipes<TAB>why`. The map is `role=type`
//! words: `role` is a tensor name between `blk.<L>.` and `.weight`, `type` a ggml type name. The scope
//! `trunk` is every layer but the NextN one (the layer that holds `nextn.*` tensors), which keeps the
//! source's types.
//!
//! The quantizer picks a type for every tensor it sees, so the argument names every tensor of the
//! source by its exact name: the map's tensors at their new type, every other at its own, which the
//! quantizer copies byte for byte.

use std::collections::{BTreeSet, HashMap};

use gguf::{GgmlType, Split, Value};

use super::{FixtureError, KEY_VARIANT};

/// The table, relative to the repository root.
pub const TABLE: &str = "tools/fixture-variants.tsv";

/// The keys the quantizer writes itself: the file's type word and the quantization format version
/// carry its argument, not the file's content. The `split.*` keys are the shard layout, which the
/// check holds equal through the shard count and each tensor's shard.
const QUANTIZER_KEYS: [&str; 2] = ["general.file_type", "general.quantization_version"];

/// Which layers a row's map reaches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scope {
    /// Every layer but the NextN one.
    Trunk,
}

/// One row of the table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Row {
    pub tag: String,
    /// The architecture the source declares.
    pub family: String,
    pub scope: Scope,
    /// Tensor role and the type it takes, in the table's order.
    pub map: Vec<(String, GgmlType)>,
    /// The gate recipes that run on the variant.
    pub recipes: Vec<String>,
}

/// One tensor the map retypes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Retype {
    pub name: String,
    pub from: GgmlType,
    pub to: GgmlType,
}

/// What a passing check saw.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CheckStats {
    pub tensors: usize,
    pub retyped: usize,
    /// Bytes of the unlisted tensors, each compared whole.
    pub copied_bytes: u64,
    pub shards: usize,
}

fn err(tag: &str, detail: impl Into<String>) -> FixtureError {
    FixtureError::Variant {
        tag: tag.to_string(),
        detail: detail.into(),
    }
}

/// The ggml type named `name`, for the names [`GgmlType::name`] spells.
fn type_named(name: &str) -> Option<GgmlType> {
    (0..=64u32)
        .map(GgmlType::from_u32)
        .find(|t| t.name() == Some(name))
}

/// The table's rows.
///
/// # Errors
/// A line that is not six columns, a tag that repeats or is not lower-case letters and digits, a
/// family or scope that is empty or unknown, a map word that is not `role=type` with a type ggml names,
/// an empty map or recipe list.
pub fn parse_table(text: &str) -> Result<Vec<Row>, FixtureError> {
    let mut rows: Vec<Row> = Vec::new();
    for (n, line) in text.lines().enumerate() {
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }
        let at = n + 1;
        let cols: Vec<&str> = line.split('\t').collect();
        let [tag, family, scope, map, recipes, why] = cols.as_slice() else {
            return Err(err(
                "(table)",
                format!(
                    "line {at} has {} columns, want tag, family, scope, map, recipes, why",
                    cols.len()
                ),
            ));
        };
        if tag.is_empty()
            || !tag
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit())
        {
            return Err(err(
                tag,
                format!(
                    "line {at}: a tag is lower-case letters and digits (it joins set names and a \
                     directory name)"
                ),
            ));
        }
        if rows.iter().any(|r| r.tag == *tag) {
            return Err(err(
                tag,
                format!("line {at}: the tag is in the table twice"),
            ));
        }
        if family.is_empty() || why.trim().is_empty() {
            return Err(err(
                tag,
                format!("line {at}: the family and the reason are not empty"),
            ));
        }
        let scope = match *scope {
            "trunk" => Scope::Trunk,
            other => {
                return Err(err(
                    tag,
                    format!("line {at}: scope {other:?} is not `trunk`"),
                ));
            }
        };
        let mut pairs = Vec::new();
        for word in map.split_whitespace() {
            let Some((role, ty)) = word.split_once('=') else {
                return Err(err(
                    tag,
                    format!("line {at}: map word {word:?} is not role=type"),
                ));
            };
            let Some(ty) = type_named(ty) else {
                return Err(err(
                    tag,
                    format!("line {at}: map word {word:?}: ggml names no type {ty:?}"),
                ));
            };
            if role.is_empty()
                || !role
                    .bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_' || b == b'.')
            {
                return Err(err(
                    tag,
                    format!("line {at}: map word {word:?}: a bad role"),
                ));
            }
            if pairs.iter().any(|(r, _): &(String, GgmlType)| r == role) {
                return Err(err(tag, format!("line {at}: the map names {role} twice")));
            }
            pairs.push((role.to_string(), ty));
        }
        let recipes: Vec<String> = recipes.split_whitespace().map(str::to_string).collect();
        if pairs.is_empty() || recipes.is_empty() {
            return Err(err(
                tag,
                format!("line {at}: the map and the recipes are not empty"),
            ));
        }
        rows.push(Row {
            tag: (*tag).to_string(),
            family: (*family).to_string(),
            scope,
            map: pairs,
            recipes,
        });
    }
    Ok(rows)
}

/// The row of `tag`.
///
/// # Errors
/// A tag the table does not name; the error lists the tags it does.
pub fn row<'a>(rows: &'a [Row], tag: &str) -> Result<&'a Row, FixtureError> {
    rows.iter().find(|r| r.tag == tag).ok_or_else(|| {
        let known: Vec<&str> = rows.iter().map(|r| r.tag.as_str()).collect();
        err(
            tag,
            format!("the table {TABLE} names no such variant, it holds {known:?}"),
        )
    })
}

/// The layer and the rest of a `blk.<L>.<rest>` tensor name.
fn layer_of(name: &str) -> Option<(usize, &str)> {
    let rest = name.strip_prefix("blk.")?;
    let (layer, rest) = rest.split_once('.')?;
    Some((layer.parse().ok()?, rest))
}

/// The layers of `source` that carry `nextn.*` tensors.
fn nextn_layers(source: &Split) -> BTreeSet<usize> {
    source
        .iter_tensors()
        .filter_map(|(_, t)| layer_of(&t.name))
        .filter(|(_, rest)| rest.starts_with("nextn."))
        .map(|(l, _)| l)
        .collect()
}

/// Bytes of a tensor of `dims` at type `ty`.
fn row_bytes(ty: GgmlType, dims: &[u64]) -> Option<u64> {
    let (blck, size) = (ty.blck_size()?, ty.type_size()?);
    dims[0]
        .is_multiple_of(blck)
        .then(|| size * (dims[0] / blck) * dims[1..].iter().product::<u64>())
}

/// The tensors of `source` that `row`'s map retypes, in the source's tensor order.
///
/// # Errors
/// The source is of another architecture than the row's family; a role no tensor of the scope has; a
/// tensor already at the type the map gives it (a row that changes nothing is a mistake); a tensor the
/// type cannot hold (its rows are not whole blocks).
pub fn expand(row: &Row, source: &Split) -> Result<Vec<Retype>, FixtureError> {
    let tag = &row.tag;
    if source.architecture() != Some(row.family.as_str()) {
        return Err(err(
            tag,
            format!(
                "the row is a variant of {}, the source declares {:?}",
                row.family,
                source.architecture()
            ),
        ));
    }
    if source.value(KEY_VARIANT).is_some() {
        return Err(err(
            tag,
            format!(
                "the source carries {KEY_VARIANT}: a variant is made from the family's own fixture"
            ),
        ));
    }
    let nextn = nextn_layers(source);
    let mut out = Vec::new();
    for (role, to) in &row.map {
        let want = format!("{role}.weight");
        let mut found = 0usize;
        for (_, t) in source.iter_tensors() {
            let Some((layer, rest)) = layer_of(&t.name) else {
                continue;
            };
            if rest != want || nextn.contains(&layer) {
                continue;
            }
            found += 1;
            if t.ty == *to {
                return Err(err(
                    tag,
                    format!(
                        "tensor {}: the source is already {to}, the map changes nothing",
                        t.name
                    ),
                ));
            }
            if row_bytes(*to, &t.dims).is_none() {
                return Err(err(
                    tag,
                    format!(
                        "tensor {}: {to} takes rows of whole blocks, its rows are {} long",
                        t.name, t.dims[0]
                    ),
                ));
            }
            out.push(Retype {
                name: t.name.clone(),
                from: t.ty,
                to: *to,
            });
        }
        if found == 0 {
            return Err(err(
                tag,
                format!(
                    "the map names {role}, and the source has no blk.<L>.{want} outside the NextN \
                     layer {nextn:?}"
                ),
            ));
        }
    }
    let order: HashMap<&str, usize> = source
        .iter_tensors()
        .enumerate()
        .map(|(i, (_, t))| (t.name.as_str(), i))
        .collect();
    out.sort_by_key(|r| order[r.name.as_str()]);
    Ok(out)
}

/// ggml's name for `ty`, which `--custom-q` parses.
fn quantizer_name(tag: &str, name: &str, ty: GgmlType) -> Result<&'static str, FixtureError> {
    ty.name().ok_or_else(|| {
        err(
            tag,
            format!("tensor {name}: type {ty} has no name the quantizer's argument can carry"),
        )
    })
}

/// ik's `--custom-q` argument for `source` under `retypes`: one rule per tensor, in the source's order,
/// each `^<exact name>$=<type>`. A tensor the map lists takes its new type; every other its own, which
/// the quantizer sees as nothing to do and copies.
///
/// # Errors
/// A tensor name the argument cannot carry (a `,` or `=`), or a type with no ggml name.
pub fn custom_q(tag: &str, source: &Split, retypes: &[Retype]) -> Result<String, FixtureError> {
    let to: HashMap<&str, GgmlType> = retypes.iter().map(|r| (r.name.as_str(), r.to)).collect();
    let mut rules = Vec::new();
    for (_, t) in source.iter_tensors() {
        if t.name.contains([',', '=']) {
            return Err(err(
                tag,
                format!(
                    "tensor {}: a `,` or `=` in a name splits the quantizer's argument",
                    t.name
                ),
            ));
        }
        let ty = to.get(t.name.as_str()).copied().unwrap_or(t.ty);
        rules.push(format!(
            "^{}$={}",
            t.name.replace('.', "\\."),
            quantizer_name(tag, &t.name, ty)?
        ));
    }
    Ok(rules.join(","))
}

/// The key/value override that stamps the variant's tag into the written file's header.
#[must_use]
pub fn override_kv(tag: &str) -> String {
    format!("{KEY_VARIANT}=str:{tag}")
}

fn kv_diff(source: &Split, variant: &Split, tag: &str) -> Result<(), FixtureError> {
    let mismatch = |what: String| err(tag, format!("the header: {what}"));
    match variant.value(KEY_VARIANT) {
        Some(Value::String(s)) if s == tag => {}
        Some(other) => {
            return Err(mismatch(format!(
                "{KEY_VARIANT} is {other:?}, want {tag:?}"
            )));
        }
        None => return Err(mismatch(format!("{KEY_VARIANT} is missing, want {tag:?}"))),
    }
    let skipped =
        |k: &str| k == KEY_VARIANT || k.starts_with("split.") || QUANTIZER_KEYS.contains(&k);
    for (k, v) in source.iter_kv().filter(|(k, _)| !skipped(k)) {
        match variant.value(k) {
            Some(got) if got == v => {}
            Some(got) => {
                return Err(mismatch(format!("{k} is {got:?}, the source's is {v:?}")));
            }
            None => {
                return Err(mismatch(format!(
                    "{k} is missing (every {} key is kept)",
                    k.split('.').next().unwrap_or(k)
                )));
            }
        }
    }
    if let Some((k, _)) = variant
        .iter_kv()
        .find(|(k, _)| !skipped(k) && source.value(k).is_none())
    {
        return Err(mismatch(format!("{k} is not in the source")));
    }
    Ok(())
}

/// Check that `variant` is `source` requantized by `row`'s map and by nothing else: the same header but
/// for the tag, the same shard count, every tensor in the same shard with the same shape, the type the
/// map gives it (its own when unlisted), and every unlisted tensor's bytes equal.
///
/// # Errors
/// The first difference, naming the tensor or the key.
pub fn check(
    row: &Row,
    source: &Split,
    variant: &Split,
    line: &mut impl FnMut(&str),
) -> Result<CheckStats, FixtureError> {
    let tag = &row.tag;
    let retypes = expand(row, source)?;
    let to: HashMap<&str, GgmlType> = retypes.iter().map(|r| (r.name.as_str(), r.to)).collect();
    kv_diff(source, variant, tag)?;
    if variant.shard_count() != source.shard_count() {
        return Err(err(
            tag,
            format!(
                "{} shards, the source's {}",
                variant.shard_count(),
                source.shard_count()
            ),
        ));
    }
    if variant.tensor_count() != source.tensor_count() {
        let held: BTreeSet<&str> = source
            .iter_tensors()
            .map(|(_, t)| t.name.as_str())
            .collect();
        let extra: Vec<&str> = variant
            .iter_tensors()
            .map(|(_, t)| t.name.as_str())
            .filter(|n| !held.contains(n))
            .collect();
        return Err(err(
            tag,
            format!(
                "{} tensors, the source's {} (not in the source: {extra:?})",
                variant.tensor_count(),
                source.tensor_count()
            ),
        ));
    }
    let mut stats = CheckStats {
        tensors: 0,
        retyped: 0,
        copied_bytes: 0,
        shards: source.shard_count(),
    };
    for (shard, t) in source.iter_tensors() {
        let Some((vshard, v)) = variant.find(&t.name) else {
            return Err(err(tag, format!("tensor {} is missing", t.name)));
        };
        if vshard != shard {
            return Err(err(
                tag,
                format!(
                    "tensor {} is in shard {vshard}, the source's is {shard}",
                    t.name
                ),
            ));
        }
        if v.dims != t.dims {
            return Err(err(
                tag,
                format!(
                    "tensor {}: shape {:?}, the source's {:?}",
                    t.name, v.dims, t.dims
                ),
            ));
        }
        let want = to.get(t.name.as_str()).copied().unwrap_or(t.ty);
        if v.ty != want {
            return Err(err(
                tag,
                format!(
                    "tensor {}: type {}, want {want} ({})",
                    t.name,
                    v.ty,
                    if want == t.ty {
                        "the map does not name it".to_string()
                    } else {
                        format!("the map gives it {want}")
                    }
                ),
            ));
        }
        stats.tensors += 1;
        if want == t.ty {
            let a = source
                .shard(shard)
                .expect("a shard the tensor lists")
                .data(t)?;
            let b = variant
                .shard(vshard)
                .expect("a shard the tensor lists")
                .data(v)?;
            if a.len() != b.len() {
                return Err(err(
                    tag,
                    format!(
                        "tensor {}: {} bytes, the source's {}",
                        t.name,
                        b.len(),
                        a.len()
                    ),
                ));
            }
            if let Some(at) = a.iter().zip(b).position(|(x, y)| x != y) {
                return Err(err(
                    tag,
                    format!(
                        "tensor {} is not in the map and its byte {at} differs ({} against {})",
                        t.name, b[at], a[at]
                    ),
                ));
            }
            stats.copied_bytes += a.len() as u64;
            line(&format!(
                "fixture: variant-check {} type={} equal bytes={}",
                t.name,
                t.ty,
                a.len()
            ));
        } else {
            stats.retyped += 1;
            line(&format!(
                "fixture: variant-check {} type={} <- {} bytes={}",
                t.name, v.ty, t.ty, v.nbytes
            ));
        }
    }
    Ok(stats)
}

#[cfg(test)]
mod tests;
