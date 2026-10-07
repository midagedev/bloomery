//! The plan: every file's metadata, tensors and shards, from the source's
//! header and a [`FixtureSpec`].

use std::collections::HashMap;
use std::ops::Range;
use std::path::PathBuf;

use gguf::write::{Layout, TensorDecl, file_alignment};
use gguf::{GgmlType, Split, Value};
use sha2::{Digest, Sha256};

use super::fill::{Rule, Window, chunk_len, chunk_rng, fill_units, rule_for};
use super::spec::{CardBudget, DraftSpec, Family, FixtureSpec, KeyRule, Options, Tables};
use super::{
    FIXTURE_VERSION, FixtureError, KEY_CARD_BUDGET, KEY_SEED, KEY_SOURCE_LAYERS, KEY_SOURCE_SHA256,
    KEY_SUBSET, KEY_VERSION, meta,
};
use crate::fileio;

/// Every tensor of a layer is named `blk.<layer>.<leaf>`, in every family.
const BLK: &str = "blk.";
const SPLIT_NO: &str = "split.no";
const SPLIT_COUNT: &str = "split.count";
const SPLIT_TENSORS: &str = "split.tensors.count";

/// A file's metadata pairs, in file order.
pub type Kvs = Vec<(String, Value)>;

/// One tensor of a fixture file.
#[derive(Clone, Debug)]
pub struct PlannedTensor {
    /// Its name in the fixture.
    pub name: String,
    /// The source tensor it copies the shape of.
    pub source: String,
    /// The fixture layer, `None` for a global or a draft tensor.
    pub layer: Option<usize>,
    pub dims: Vec<u64>,
    pub ty: GgmlType,
    pub nbytes: u64,
    pub rule: Rule,
}

impl PlannedTensor {
    /// Chunks its stream is cut into.
    pub fn chunks(&self) -> usize {
        (self.nbytes as usize).div_ceil(chunk_len(self.ty))
    }

    /// The byte range of chunk `c`.
    pub fn chunk_range(&self, c: usize) -> Range<usize> {
        let len = chunk_len(self.ty);
        c * len..((c + 1) * len).min(self.nbytes as usize)
    }

    /// Chunk `c`'s bytes under `seed` into `out` (its length).
    pub fn fill_chunk(&self, seed: u64, c: usize, out: &mut [u8]) {
        let mut rng = chunk_rng(seed, &self.name, c);
        fill_units(&self.rule, &mut rng, out);
    }

    /// The whole tensor under `seed` into `out` (`nbytes` long), its chunks
    /// spread over `threads` threads.
    pub fn fill(&self, seed: u64, threads: usize, out: &mut [u8]) {
        let len = chunk_len(self.ty);
        let mut lanes: Vec<Vec<_>> = (0..threads.max(1)).map(|_| Vec::new()).collect();
        let n = lanes.len();
        for (c, part) in out.chunks_mut(len).enumerate() {
            lanes[c % n].push((c, part));
        }
        std::thread::scope(|s| {
            for lane in lanes {
                s.spawn(move || {
                    for (c, part) in lane {
                        self.fill_chunk(seed, c, part);
                    }
                });
            }
        });
    }

    /// The element standard deviation it is generated at, `None` for a
    /// constant.
    pub fn sigma(&self) -> Option<f64> {
        match self.rule {
            Rule::Const { .. } => None,
            _ => Some(1.0 / (self.dims[0] as f64).sqrt()),
        }
    }
}

/// One file set: the target's shards or the draft's single file.
#[derive(Clone, Debug)]
pub struct FilePlan {
    /// The first file's metadata; a split set's keys carry their final values.
    pub kvs: Vec<(String, Value)>,
    pub tensors: Vec<PlannedTensor>,
    /// Each file's tensors, as index ranges into `tensors`.
    pub shards: Vec<Range<usize>>,
    /// File names, one per shard.
    pub files: Vec<String>,
    /// The split keys every later shard carries, in the first shard's order.
    split_keys: Vec<(String, Value)>,
    /// The alignment `kvs` sets, by the writer's rule.
    align: u64,
}

impl FilePlan {
    /// Each file's name and layout.
    pub fn layouts(&self) -> Result<Vec<(String, Layout)>, FixtureError> {
        let mut out = Vec::with_capacity(self.shards.len());
        for (i, range) in self.shards.iter().enumerate() {
            let kvs = if i == 0 {
                self.kvs.clone()
            } else {
                self.split_keys
                    .iter()
                    .map(|(k, v)| {
                        let v = if k == SPLIT_NO {
                            int_like(k, v, i as u64)?
                        } else {
                            v.clone()
                        };
                        Ok((k.clone(), v))
                    })
                    .collect::<Result<Vec<_>, FixtureError>>()?
            };
            let decls = self.tensors[range.clone()]
                .iter()
                .map(|t| TensorDecl {
                    name: t.name.clone(),
                    dims: t.dims.clone(),
                    type_id: t.ty.as_u32(),
                    nbytes: t.nbytes,
                })
                .collect();
            let layout = Layout::new(&kvs, decls).map_err(|source| FixtureError::Write {
                path: PathBuf::from(&self.files[i]),
                source,
            })?;
            out.push((self.files[i].clone(), layout));
        }
        Ok(out)
    }

    /// The fixture layers whose tensors lie in more than one shard.
    pub fn spanning_layers(&self) -> Vec<usize> {
        let mut first: HashMap<usize, usize> = HashMap::new();
        let mut spans = Vec::new();
        for (s, range) in self.shards.iter().enumerate() {
            for t in &self.tensors[range.clone()] {
                if let Some(l) = t.layer {
                    let at = *first.entry(l).or_insert(s);
                    if at != s && !spans.contains(&l) {
                        spans.push(l);
                    }
                }
            }
        }
        spans
    }

    /// Bytes a tensor takes in its file, padding included.
    pub fn padded(&self, t: &PlannedTensor) -> u64 {
        t.nbytes.next_multiple_of(self.align)
    }
}

/// The target's plan and, when a draft source is given, the draft's.
#[derive(Clone, Debug)]
pub struct Plan {
    pub target: FilePlan,
    pub draft: Option<FilePlan>,
    /// The family's tables, one line each ([`Tables::lines`]), and the note
    /// of a planned card budget.
    pub notes: Vec<String>,
    /// The card budget the fixture records: the caller's, or the spec's
    /// ([`FixtureSpec::card_budget`]).
    pub card_budget: u64,
}

/// `template`'s integer variant holding `v`.
pub fn int_like(key: &str, template: &Value, v: u64) -> Result<Value, FixtureError> {
    let bad = || meta(key, format!("{v} does not fit the source's {template:?}"));
    Ok(match template {
        Value::U8(_) => Value::U8(u8::try_from(v).map_err(|_| bad())?),
        Value::U16(_) => Value::U16(u16::try_from(v).map_err(|_| bad())?),
        Value::U32(_) => Value::U32(u32::try_from(v).map_err(|_| bad())?),
        Value::U64(_) => Value::U64(v),
        Value::I8(_) => Value::I8(i8::try_from(v).map_err(|_| bad())?),
        Value::I16(_) => Value::I16(i16::try_from(v).map_err(|_| bad())?),
        Value::I32(_) => Value::I32(i32::try_from(v).map_err(|_| bad())?),
        Value::I64(_) => Value::I64(i64::try_from(v).map_err(|_| bad())?),
        _ => return Err(meta(key, format!("is {template:?}, not an integer"))),
    })
}

/// The items of array `v`, the value of `key`.
pub fn items<'a>(key: &str, v: &'a Value) -> Result<&'a [Value], FixtureError> {
    match v {
        Value::Array(a) => Ok(a),
        _ => Err(meta(key, "is not an array")),
    }
}

/// `v`, the value of `key`, as an unsigned integer.
pub fn unsigned(key: &str, v: &Value) -> Result<u64, FixtureError> {
    v.as_unsigned()
        .ok_or_else(|| meta(key, format!("{v:?} is not an unsigned integer")))
}

/// `blk.<n>.<rest>` → `(n, rest)`.
fn split_layer(name: &str) -> Option<(usize, &str)> {
    let rest = name.strip_prefix(BLK)?;
    let (n, rest) = rest.split_once('.')?;
    Some((n.parse().ok()?, rest))
}

/// lowercase hex sha256 of every shard's bytes before its data base, in
/// split order.
pub fn header_sha256(split: &Split) -> String {
    let mut h = Sha256::new();
    for i in 0..split.shard_count() {
        let g = split
            .shard(i)
            .expect("a split has a reader for every shard index");
        h.update(g.header_bytes());
    }
    fileio::hex(&h.finalize())
}

/// The fixture's own keys.
fn fixture_keys(spec: &FixtureSpec, opts: &Options, source_sha: String) -> Vec<(String, Value)> {
    vec![
        (KEY_VERSION.to_string(), Value::U32(FIXTURE_VERSION)),
        (KEY_SEED.to_string(), Value::U64(opts.seed)),
        (
            KEY_SOURCE_LAYERS.to_string(),
            Value::Array(spec.layers.iter().map(|&l| Value::U32(l as u32)).collect()),
        ),
        (KEY_SOURCE_SHA256.to_string(), Value::String(source_sha)),
        // The caller's budget, or a placeholder of the same size until the
        // spec's is known: the plan sets it (`set_budget`) without moving a
        // byte of the layout.
        (
            KEY_CARD_BUDGET.to_string(),
            Value::U64(opts.card_budget.unwrap_or(0)),
        ),
    ]
}

/// Keep only `subset` of `tensors` (every name must be planned) and record it.
fn apply_subset(
    tensors: Vec<PlannedTensor>,
    subset: Option<&Vec<String>>,
    kvs: &mut Vec<(String, Value)>,
) -> Result<Vec<PlannedTensor>, FixtureError> {
    let Some(names) = subset else {
        return Ok(tensors);
    };
    if let Some(n) = names
        .iter()
        .find(|n| !tensors.iter().any(|t| &t.name == *n))
    {
        return Err(FixtureError::UnknownTensor { name: n.clone() });
    }
    let kept: Vec<PlannedTensor> = tensors
        .into_iter()
        .filter(|t| names.contains(&t.name))
        .collect();
    kvs.push((
        KEY_SUBSET.to_string(),
        Value::Array(kept.iter().map(|t| Value::String(t.name.clone())).collect()),
    ));
    Ok(kept)
}

/// Rules are a function of (type, K) under the spec's window; planned once
/// per pair. A 1-D tensor takes the family's constant.
struct Rules<'a> {
    family: &'a dyn Family,
    window: Window,
    cache: HashMap<(u32, u64), Rule>,
}

impl Rules<'_> {
    fn get(&mut self, name: &str, ty: GgmlType, dims: &[u64]) -> Result<Rule, FixtureError> {
        if dims.len() < 2 {
            return self.constant(name, ty);
        }
        let key = (ty.as_u32(), dims[0]);
        if let Some(r) = self.cache.get(&key) {
            return Ok(r.clone());
        }
        let r = rule_for(name, ty, dims[0], self.window)?;
        self.cache.insert(key, r.clone());
        Ok(r)
    }

    /// The rule of the 1-D tensor `name` of type `ty`: the family's constant
    /// for its name past `blk.N.`, as F32.
    fn constant(&self, name: &str, ty: GgmlType) -> Result<Rule, FixtureError> {
        let leaf = split_layer(name).map_or(name, |(_, rest)| rest);
        match (ty, self.family.const_value(leaf)) {
            (GgmlType::F32, Some(value)) => Ok(Rule::Const { value }),
            _ => Err(FixtureError::NoFillRule {
                name: name.to_string(),
                ty,
            }),
        }
    }
}

fn planned(
    rules: &mut Rules,
    name: String,
    source: &str,
    layer: Option<usize>,
    dims: Vec<u64>,
    ty: GgmlType,
) -> Result<PlannedTensor, FixtureError> {
    let rule = rules.get(&name, ty, &dims)?;
    let (_, blck, tsz) =
        gguf::ggml_type_info(ty.as_u32()).ok_or_else(|| FixtureError::UnsupportedType {
            name: name.clone(),
            ty,
        })?;
    if !dims[0].is_multiple_of(blck) {
        return Err(FixtureError::Tensor {
            name,
            detail: format!("ne[0] {} is not whole blocks of {blck}", dims[0]),
        });
    }
    let nbytes = tsz * (dims[0] / blck) * dims[1..].iter().product::<u64>();
    Ok(PlannedTensor {
        name,
        source: source.to_string(),
        layer,
        dims,
        ty,
        nbytes,
        rule,
    })
}

/// The fixture `spec` names of `source` (the real first shard's split set)
/// and, with `draft`, of the family's real draft.
pub fn plan(
    spec: &FixtureSpec,
    source: &Split,
    draft: Option<&Split>,
    opts: &Options,
) -> Result<Plan, FixtureError> {
    if source.architecture() != Some(spec.arch) {
        return Err(FixtureError::Architecture {
            got: source.architecture().map(str::to_string),
            want: spec.arch,
        });
    }
    let n_layer = source
        .arch_get_u64("block_count")
        .ok_or_else(|| meta("block_count", "is absent"))? as usize;
    if let Some(&l) = spec.layers.iter().find(|&&l| l >= n_layer) {
        return Err(FixtureError::NoSourceLayer { layer: l, n_layer });
    }
    let tables = spec.family.tables(source, spec)?;
    let (mut kvs, split_keys, source_ff) = target_kvs(spec, source, n_layer, tables.as_ref())?;
    kvs.extend(fixture_keys(spec, opts, header_sha256(source)));
    let mut rules = Rules {
        family: spec.family,
        window: spec.window,
        cache: HashMap::new(),
    };
    let ff = source_ff.zip(spec.ff);
    let tensors = target_tensors(spec, source, tables.as_ref(), ff, &mut rules)?;
    let tensors = apply_subset(tensors, opts.tensors.as_ref(), &mut kvs)?;
    let mut target = shard(spec.stem, kvs, split_keys, tensors, opts.shard_bytes)?;
    let mut draft = match (draft, &spec.draft) {
        (None, _) => None,
        (Some(d), Some(ds)) => Some(plan_draft(spec, ds, d, n_layer, opts, &mut rules)?),
        (Some(_), None) => return Err(FixtureError::NoDraft { arch: spec.arch }),
    };
    let mut notes = tables.lines();
    let card_budget = match (opts.card_budget, spec.card_budget) {
        (Some(b), _) => b,
        (None, CardBudget::Fixed(b)) => b,
        (None, CardBudget::Planned(_)) if opts.tensors.is_some() => {
            return Err(FixtureError::Budget(
                "a subset file holds no whole plan to choose the budget from; give --card-budget"
                    .into(),
            ));
        }
        (None, CardBudget::Planned(planner)) => {
            let b = super::budget::choose(&planner, &target)?;
            notes.push(format!(
                "card budget {b} ({} MiB): half the experts on the card on every card-eligible \
                 layer, planned at {} positions",
                b / (1024 * 1024),
                planner.ctx
            ));
            b
        }
    };
    set_budget(&mut target.kvs, card_budget)?;
    if let Some(d) = draft.as_mut() {
        set_budget(&mut d.kvs, card_budget)?;
    }
    Ok(Plan {
        target,
        draft,
        notes,
        card_budget,
    })
}

/// Replace the recorded card budget in `kvs`, whose entry the placeholder
/// already holds.
fn set_budget(kvs: &mut Kvs, budget: u64) -> Result<(), FixtureError> {
    let (_, v) = kvs
        .iter_mut()
        .find(|(k, _)| k == KEY_CARD_BUDGET)
        .ok_or_else(|| meta(KEY_CARD_BUDGET, "the plan holds no budget entry to set"))?;
    *v = Value::U64(budget);
    Ok(())
}

/// The target's metadata in the source's order, each key by the family's
/// [`KeyRule`], the source's split keys, and the source's ff when the family
/// has a [`KeyRule::Ff`] key.
fn target_kvs(
    spec: &FixtureSpec,
    source: &Split,
    n_layer: usize,
    tables: &dyn Tables,
) -> Result<(Kvs, Kvs, Option<u64>), FixtureError> {
    let arch_prefix = format!("{}.", spec.arch);
    let mut kvs: Vec<(String, Value)> = Vec::new();
    let mut split_keys: Vec<(String, Value)> = Vec::new();
    let mut ratios_read = false;
    let mut source_ff = None;
    for (k, v) in source.iter_kv() {
        if k.starts_with("split.") {
            split_keys.push((k.to_string(), v.clone()));
            kvs.push((k.to_string(), v.clone()));
            continue;
        }
        if k.starts_with("general.") || k.starts_with("tokenizer.") || k.starts_with("quantize.") {
            kvs.push((k.to_string(), v.clone()));
            continue;
        }
        let (suffix, rule) = k
            .strip_prefix(&arch_prefix)
            .and_then(|s| spec.family.key_rule(s).map(|r| (s, r)))
            .ok_or_else(|| FixtureError::UncoveredKey { key: k.to_string() })?;
        let nv = match rule {
            KeyRule::Copy => v.clone(),
            KeyRule::BlockCount => int_like(k, v, spec.layers.len() as u64)?,
            KeyRule::Zero(why) => {
                if unsigned(k, v)? != 0 {
                    return Err(meta(k, format!("is not 0; {why}")));
                }
                v.clone()
            }
            KeyRule::Ratios => {
                ratios_read = true;
                mapped_ratios(spec, k, v, n_layer)?
            }
            KeyRule::PerLayer => match v {
                Value::Array(a) if a.len() == n_layer => {
                    Value::Array(spec.layers.iter().map(|&l| a[l].clone()).collect())
                }
                Value::Array(a) => {
                    return Err(meta(
                        k,
                        format!("has {} values for {n_layer} layers", a.len()),
                    ));
                }
                other => other.clone(),
            },
            KeyRule::LayerIds => {
                let a = items(k, v)?;
                let tag = a.first().ok_or_else(|| meta(k, "is empty"))?;
                let mut ids = Vec::with_capacity(a.len());
                for x in a {
                    let l = unsigned(k, x)? as usize;
                    let f = spec.layers.iter().position(|&m| m == l).ok_or_else(|| {
                        meta(k, format!("lists layer {l}, which the map does not hold"))
                    })?;
                    ids.push(int_like(k, tag, f as u64)?);
                }
                Value::Array(ids)
            }
            KeyRule::Ff => {
                source_ff = Some(unsigned(k, v)?);
                match spec.ff {
                    Some(ff) => int_like(k, v, ff)?,
                    None => v.clone(),
                }
            }
            KeyRule::Set(to) => int_like(k, v, to)?,
            KeyRule::Table => tables.key(k, suffix, v)?,
        };
        kvs.push((k.to_string(), nv));
    }
    if !spec.ratios.is_empty() && !ratios_read {
        return Err(meta(
            &format!("{arch_prefix}*"),
            format!(
                "the spec's ratios {:?} have no ratios key in the source",
                spec.ratios
            ),
        ));
    }
    if let Some(key) = spec
        .family
        .required_keys()
        .iter()
        .find(|k| source.value(&format!("{arch_prefix}{k}")).is_none())
    {
        return Err(meta(
            &format!("{arch_prefix}{key}"),
            "is absent from the source: the fixture's header is not right without it",
        ));
    }
    if let Some(ff) = spec.ff
        && source_ff.is_none()
    {
        return Err(meta(
            &format!("{arch_prefix}*"),
            format!("the spec's ff {ff} has no ff key in the source"),
        ));
    }
    Ok((kvs, split_keys, source_ff))
}

/// A ratios array at the map's layers, then the source's entries past its
/// `block_count`; the mapped entries must be the spec's ratios.
fn mapped_ratios(
    spec: &FixtureSpec,
    k: &str,
    v: &Value,
    n_layer: usize,
) -> Result<Value, FixtureError> {
    let a = items(k, v)?;
    if a.len() < n_layer {
        return Err(meta(
            k,
            format!("has {} values for {n_layer} layers", a.len()),
        ));
    }
    let mapped: Vec<Value> = spec.layers.iter().map(|&l| a[l].clone()).collect();
    let got = mapped
        .iter()
        .map(|x| unsigned(k, x))
        .collect::<Result<Vec<_>, _>>()?;
    if got != spec.ratios {
        return Err(meta(
            k,
            format!(
                "the map's layers read {got:?}, the fixture is built for {:?}",
                spec.ratios
            ),
        ));
    }
    Ok(Value::Array(
        mapped
            .into_iter()
            .chain(a[n_layer..].iter().cloned())
            .collect(),
    ))
}

/// The globals in the source's order, then each fixture layer's tensors in
/// the source's order, renamed; dims as the family's tables set them, and
/// with an ff override `(source, fixture)` its axis narrowed.
fn target_tensors(
    spec: &FixtureSpec,
    source: &Split,
    tables: &dyn Tables,
    ff: Option<(u64, u64)>,
    rules: &mut Rules,
) -> Result<Vec<PlannedTensor>, FixtureError> {
    let mut globals = Vec::new();
    let mut by_layer: Vec<Vec<PlannedTensor>> = vec![Vec::new(); spec.layers.len()];
    let mut narrowed = 0usize;
    for (_, t) in source.iter_tensors() {
        let Some((l, rest)) = split_layer(&t.name) else {
            let dims = tables
                .dims(&t.name, None, &t.name, &t.dims)?
                .unwrap_or_else(|| t.dims.clone());
            globals.push(planned(rules, t.name.clone(), &t.name, None, dims, t.ty)?);
            continue;
        };
        let Some(f) = spec.layers.iter().position(|&m| m == l) else {
            continue;
        };
        let mut dims = tables
            .dims(&t.name, Some(l), rest, &t.dims)?
            .unwrap_or_else(|| t.dims.clone());
        if let Some((from, to)) = ff
            && let Some(axis) = spec.family.ff_axis(rest)
        {
            if dims.get(axis) != Some(&from) {
                return Err(FixtureError::Tensor {
                    name: t.name.clone(),
                    detail: format!(
                        "has dims {dims:?}; its ff axis {axis} is not the source's ff {from}"
                    ),
                });
            }
            dims[axis] = to;
            narrowed += 1;
        }
        let name = format!("{BLK}{f}.{rest}");
        by_layer[f].push(planned(rules, name, &t.name, Some(f), dims, t.ty)?);
    }
    if let Some((from, to)) = ff
        && narrowed == 0
    {
        return Err(FixtureError::Tensor {
            name: format!("{BLK}*"),
            detail: format!(
                "an ff override {from} -> {to}, and the family names no ff axis on any mapped tensor"
            ),
        });
    }
    if let Some(f) = by_layer.iter().position(Vec::is_empty) {
        return Err(FixtureError::Tensor {
            name: format!("{BLK}{}.*", spec.layers[f]),
            detail: "the source holds no tensor of this mapped layer".into(),
        });
    }
    Ok(globals
        .into_iter()
        .chain(by_layer.into_iter().flatten())
        .collect())
}

/// Cut `tensors` into shards of at most `cap` data bytes each (a larger
/// tensor alone), named after `stem`, and set the split keys.
fn shard(
    stem: &str,
    mut kvs: Vec<(String, Value)>,
    split_keys: Vec<(String, Value)>,
    tensors: Vec<PlannedTensor>,
    cap: u64,
) -> Result<FilePlan, FixtureError> {
    let align = file_alignment(&kvs).map_err(|source| FixtureError::Alignment {
        set: "target",
        source,
    })?;
    let mut shards: Vec<Range<usize>> = Vec::new();
    let (mut start, mut bytes) = (0usize, 0u64);
    for (i, t) in tensors.iter().enumerate() {
        let b = t.nbytes.next_multiple_of(align);
        if i > start && bytes + b > cap {
            shards.push(start..i);
            (start, bytes) = (i, 0);
        }
        bytes += b;
    }
    shards.push(start..tensors.len());
    let n = shards.len();
    for key in [SPLIT_NO, SPLIT_COUNT, SPLIT_TENSORS] {
        if !split_keys.iter().any(|(k, _)| k == key) {
            return Err(meta(key, "is absent from the source's first shard"));
        }
    }
    for (k, v) in &mut kvs {
        match k.as_str() {
            SPLIT_NO => *v = int_like(k, v, 0)?,
            SPLIT_COUNT => *v = int_like(k, v, n as u64)?,
            SPLIT_TENSORS => *v = int_like(k, v, tensors.len() as u64)?,
            _ => {}
        }
    }
    let split_keys = split_keys
        .into_iter()
        .map(|(k, v)| {
            let v = match k.as_str() {
                SPLIT_COUNT => int_like(&k, &v, n as u64)?,
                SPLIT_TENSORS => int_like(&k, &v, tensors.len() as u64)?,
                _ => v,
            };
            Ok((k, v))
        })
        .collect::<Result<Vec<_>, FixtureError>>()?;
    let files = (0..n)
        .map(|i| format!("{stem}-{:05}-of-{n:05}.gguf", i + 1))
        .collect();
    let plan = FilePlan {
        kvs,
        tensors,
        shards,
        files,
        split_keys,
        align,
    };
    if n > 1 && plan.spanning_layers().is_empty() {
        return Err(FixtureError::NoSpanningLayer { shards: n });
    }
    Ok(plan)
}

/// The draft fixture: the real draft's metadata as the draft's rules rewrite
/// it, its tensors at their shapes, one file.
fn plan_draft(
    spec: &FixtureSpec,
    ds: &DraftSpec,
    draft: &Split,
    n_layer: usize,
    opts: &Options,
    rules: &mut Rules,
) -> Result<FilePlan, FixtureError> {
    if draft.architecture() != Some(ds.arch) {
        return Err(FixtureError::Architecture {
            got: draft.architecture().map(str::to_string),
            want: ds.arch,
        });
    }
    if let Some((k, _)) = draft.iter_kv().find(|(k, _)| k.starts_with("split.")) {
        return Err(meta(k, "the draft fixture is written as one file"));
    }
    let mut kvs = ds.rules.kvs(draft, n_layer, spec.layers.len())?;
    let align = file_alignment(&kvs).map_err(|source| FixtureError::Alignment {
        set: "draft",
        source,
    })?;
    kvs.extend(fixture_keys(spec, opts, header_sha256(draft)));
    let n_fixture = spec.layers.len();
    let tensors = draft
        .iter_tensors()
        .map(|(_, t)| {
            let name = ds.rules.tensor(&t.name, n_layer, n_fixture)?;
            planned(rules, name, &t.name, None, t.dims.clone(), t.ty)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let tensors = apply_subset(tensors, opts.draft_tensors.as_ref(), &mut kvs)?;
    let n = tensors.len();
    Ok(FilePlan {
        kvs,
        tensors,
        shards: vec![Range { start: 0, end: n }],
        files: vec![ds.file.to_string()],
        split_keys: Vec::new(),
        align,
    })
}
