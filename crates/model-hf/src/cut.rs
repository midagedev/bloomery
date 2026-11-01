//! A checkpoint folder cut to a smaller folder of the same bits. [`cut`] keeps the layers a [`Spec`]
//! names (renumbered from 0), slices the matrices a width cut touches on whole quantization blocks
//! (a matrix's codes and its scale planes together, so every plane keeps the same values), copies
//! every other kept tensor whole, and writes safetensors shards, an edited `config.json`, the
//! tokenizer and chat files, and [`MANIFEST`]: per kept tensor its source range and the SHA-256 of
//! its bytes. [`verify`] re-plans the cut from the source's headers, then hashes each tensor in the
//! cut and its source range and holds both to the record.
//!
//! The cut knows no model. Which tensors are kept, which are sliced along which axis and in what
//! block units, and which config fields move are the spec's data (`tools/fixture-cuts/<name>.json`
//! for a registry row's cut; `tools/mlx-fixture-cuts/<name>.json` for a Metal-only MLX folder, which
//! the Metal gates open by name and which has no row).
//! A layer's tensors keep the order they have in their source shard, so a layer stored with each
//! matrix's codes and scales adjacent stays so, and one stored with every scale plane before every
//! code plane stays so. A layer is never split across shards.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::time::Instant;

use gguf::digest::{Sha256, hex};
use safetensors::{Dtype, Folder};
use serde_json::{Map, Value};

/// The record of a cut, beside its shards: one row a tensor with its source range and hash.
pub const MANIFEST: &str = "bloomery-cut.tsv";
/// The tensor data a shard holds at most when a layer's group fits under it; a group larger than
/// the cap is a shard of its own.
pub const DEFAULT_SHARD_BYTES: u64 = 4 << 30;
/// The identity key that binds the shards to their record: the SHA-256 of [`MANIFEST`].
pub const KEY_MANIFEST_SHA256: &str = "bloomery.fixture.manifest_sha256";

const KEY_PREFIX: &str = "bloomery.fixture.";
const IDENTITY_VERSION: &str = "1";
const MANIFEST_VERSION: &str = "1";
const CONFIG: &str = "config.json";
const CHUNK_BYTES: u64 = 8 << 20;

/// Why a spec, a cut or a verify was refused; each names the tensor, width or file it is about.
#[derive(Debug, thiserror::Error)]
pub enum CutError {
    #[error("spec: {0}")]
    Spec(String),
    #[error("{path}: {op}: {source}")]
    Io {
        path: PathBuf,
        op: &'static str,
        source: io::Error,
    },
    #[error(transparent)]
    Tensors(#[from] safetensors::Error),
    #[error("{0} exists: a cut writes a directory of its own")]
    Exists(PathBuf),
    #[error("the source holds {have} {what}, the spec was written for {want}")]
    Identity {
        what: &'static str,
        have: String,
        want: String,
    },
    #[error("{name}: no keep or drop rule of the spec names this tensor")]
    Unclaimed { name: String },
    #[error("{0}: the source does not hold it")]
    Missing(String),
    #[error(
        "width {width} keeps {keep} values, not a multiple of its block {align}: {tensor} is cut \
         on whole blocks"
    )]
    Unaligned {
        width: String,
        keep: u64,
        align: u64,
        tensor: String,
    },
    #[error("{tensor}: width {width} keeps {keep} values, {why}")]
    Width {
        tensor: String,
        width: String,
        keep: u64,
        why: String,
    },
    #[error("{base}: the planes disagree: {why}")]
    Planes { base: String, why: String },
    #[error(
        "{name}: starts at {offset} in output shard {shard}, not a multiple of its {align}-byte \
         element"
    )]
    Misaligned {
        name: String,
        shard: usize,
        offset: u64,
        align: u64,
    },
    #[error("config.json: {0}")]
    Config(String),
    #[error("{MANIFEST}: {0}")]
    Manifest(String),
    #[error("{name}: the {side} bytes hash to {got}, the manifest records {want}")]
    Hash {
        name: String,
        side: &'static str,
        got: String,
        want: String,
    },
    #[error("{what}")]
    Differs { what: String },
}

fn io_err(path: &Path, op: &'static str) -> impl FnOnce(io::Error) -> CutError {
    let path = path.to_path_buf();
    move |source| CutError::Io { path, op, source }
}

fn spec_err(why: impl Into<String>) -> CutError {
    CutError::Spec(why.into())
}

fn differs(what: impl Into<String>) -> CutError {
    CutError::Differs { what: what.into() }
}

// ---------------------------------------------------------------------------------------------
// The spec

/// The axis of a plane set a width cuts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Axis {
    Rows,
    Cols,
}

/// A width of the cut: how many values it keeps and the block those must fill whole.
#[derive(Debug, Clone)]
struct Width {
    name: String,
    keep: u64,
    align: u64,
}

/// The tensors of one matrix, a plane each (its codes, its block scales), cut together on one
/// axis. `members` are the suffixes after the base and each plane's values per stored element
/// along the cut axis.
#[derive(Debug, Clone)]
struct PlaneSet {
    base: Vec<String>,
    axis: Axis,
    width: usize,
    members: Vec<(String, u64)>,
}

/// What the cut edits of `config.json`, as dotted paths.
#[derive(Debug, Clone)]
struct ConfigEdit {
    layer_count: String,
    widths: Vec<(String, usize)>,
    per_layer: Vec<String>,
    layer_names: Vec<String>,
}

/// A cut's rules, read from one JSON document.
#[derive(Debug, Clone)]
pub struct Spec {
    pub name: String,
    pub arch: String,
    source: String,
    rev: String,
    expect_tensors: usize,
    expect_data_bytes: u64,
    layers: Vec<usize>,
    layer_prefix: String,
    keep: Vec<String>,
    drop: Vec<String>,
    copy: Vec<String>,
    widths: Vec<Width>,
    planes: Vec<PlaneSet>,
    config: ConfigEdit,
    text: String,
}

/// An object whose keys are taken one by one; [`Fields::finish`] refuses the ones left.
struct Fields<'a> {
    what: String,
    map: &'a Map<String, Value>,
    seen: BTreeSet<&'a str>,
}

impl<'a> Fields<'a> {
    fn new(v: &'a Value, what: &str) -> Result<Fields<'a>, CutError> {
        let map = v
            .as_object()
            .ok_or_else(|| spec_err(format!("{what} is not an object")))?;
        Ok(Fields {
            what: what.to_string(),
            map,
            seen: BTreeSet::new(),
        })
    }

    fn get(&mut self, key: &str) -> Result<&'a Value, CutError> {
        let (k, v) = self
            .map
            .get_key_value(key)
            .ok_or_else(|| spec_err(format!("{} has no {key}", self.what)))?;
        self.seen.insert(k.as_str());
        Ok(v)
    }

    fn string(&mut self, key: &str) -> Result<String, CutError> {
        let v = self.get(key)?;
        v.as_str()
            .map(str::to_string)
            .ok_or_else(|| spec_err(format!("{}.{key} is not a string", self.what)))
    }

    fn uint(&mut self, key: &str) -> Result<u64, CutError> {
        let v = self.get(key)?;
        v.as_u64()
            .ok_or_else(|| spec_err(format!("{}.{key} is not a whole number", self.what)))
    }

    fn array(&mut self, key: &str) -> Result<&'a Vec<Value>, CutError> {
        let v = self.get(key)?;
        v.as_array()
            .ok_or_else(|| spec_err(format!("{}.{key} is not an array", self.what)))
    }

    fn strings(&mut self, key: &str) -> Result<Vec<String>, CutError> {
        let what = format!("{}.{key}", self.what);
        self.array(key)?
            .iter()
            .map(|v| {
                v.as_str()
                    .map(str::to_string)
                    .ok_or_else(|| spec_err(format!("{what} holds a value that is not a string")))
            })
            .collect()
    }

    fn object(&mut self, key: &str) -> Result<Fields<'a>, CutError> {
        let v = self.get(key)?;
        Fields::new(v, &format!("{}.{key}", self.what))
    }

    /// The entries of an object whose keys are the data (a width's name, a plane's suffix).
    fn entries(&mut self, key: &str) -> Result<Vec<(&'a str, &'a Value)>, CutError> {
        let o = self.object(key)?;
        Ok(o.map.iter().map(|(k, v)| (k.as_str(), v)).collect())
    }

    fn finish(self) -> Result<(), CutError> {
        match self.map.keys().find(|k| !self.seen.contains(k.as_str())) {
            Some(k) => Err(spec_err(format!("{} has the unknown key {k}", self.what))),
            None => Ok(()),
        }
    }
}

fn width_of(widths: &[Width], name: &str) -> Result<usize, CutError> {
    widths
        .iter()
        .position(|w| w.name == name)
        .ok_or_else(|| spec_err(format!("no width {name}")))
}

fn parse_layers(f: &mut Fields) -> Result<Vec<usize>, CutError> {
    let layers = f
        .array("layers")?
        .iter()
        .map(|l| {
            l.as_u64()
                .and_then(|l| usize::try_from(l).ok())
                .ok_or_else(|| spec_err("layers holds a value that is not a layer id"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if layers.is_empty() || layers.windows(2).any(|w| w[0] >= w[1]) {
        return Err(spec_err(
            "layers is not a non-empty strictly ascending list of ids",
        ));
    }
    Ok(layers)
}

/// The files the spec copies: plain names in the source's top level, never a file the cut writes.
fn parse_copy(f: &mut Fields) -> Result<Vec<String>, CutError> {
    let copy = f.strings("copy")?;
    match copy
        .iter()
        .find(|n| n.contains('/') || *n == CONFIG || *n == Folder::INDEX)
    {
        Some(n) => Err(spec_err(format!(
            "copy {n}: a plain file name of the source's top level that the cut does not write itself"
        ))),
        None => Ok(copy),
    }
}

fn parse_widths(f: &mut Fields) -> Result<Vec<Width>, CutError> {
    let mut widths = Vec::new();
    for (name, w) in f.entries("widths")? {
        let mut w = Fields::new(w, &format!("widths.{name}"))?;
        let width = Width {
            name: name.to_string(),
            keep: w.uint("keep")?,
            align: w.uint("align")?,
        };
        w.finish()?;
        if width.keep == 0 || width.align == 0 {
            return Err(spec_err(format!(
                "width {name}: keep and align are positive"
            )));
        }
        widths.push(width);
    }
    Ok(widths)
}

fn parse_planes(f: &mut Fields, widths: &[Width]) -> Result<Vec<PlaneSet>, CutError> {
    let mut planes = Vec::new();
    for (i, p) in f.array("planes")?.iter().enumerate() {
        let mut p = Fields::new(p, &format!("planes[{i}]"))?;
        let base = p.string("base")?.split('.').map(str::to_string).collect();
        let axis = match p.string("axis")?.as_str() {
            "rows" => Axis::Rows,
            "cols" => Axis::Cols,
            other => {
                return Err(spec_err(format!(
                    "planes[{i}].axis is {other}, rows or cols"
                )));
            }
        };
        let width = width_of(widths, &p.string("width")?)?;
        let mut members = Vec::new();
        for (suffix, unit) in p.entries("members")? {
            let unit = unit.as_u64().filter(|u| *u > 0).ok_or_else(|| {
                spec_err(format!(
                    "planes[{i}].members.{suffix}: a unit is a positive number"
                ))
            })?;
            members.push((suffix.to_string(), unit));
        }
        p.finish()?;
        if members.len() < 2 {
            return Err(spec_err(format!(
                "planes[{i}]: a plane set has at least two members (the codes and a scale plane)"
            )));
        }
        planes.push(PlaneSet {
            base,
            axis,
            width,
            members,
        });
    }
    Ok(planes)
}

fn parse_config(f: &mut Fields, widths: &[Width]) -> Result<ConfigEdit, CutError> {
    let mut c = f.object("config")?;
    let layer_count = c.string("layer_count")?;
    let mut by_width = Vec::new();
    for (path, w) in c.entries("widths")? {
        let w = w
            .as_str()
            .ok_or_else(|| spec_err(format!("config.widths.{path} is not a width's name")))?;
        by_width.push((path.to_string(), width_of(widths, w)?));
    }
    let edit = ConfigEdit {
        layer_count,
        widths: by_width,
        per_layer: c.strings("per_layer")?,
        layer_names: c.strings("layer_names")?,
    };
    c.finish()?;
    Ok(edit)
}

impl Spec {
    /// The spec the JSON `text` holds. Every key is required and a key it does not know is refused
    /// by name, so a rule that is misspelled never reads as an absent one.
    pub fn parse(text: &str) -> Result<Spec, CutError> {
        let v: Value =
            serde_json::from_str(text).map_err(|e| spec_err(format!("not JSON: {e}")))?;
        let mut f = Fields::new(&v, "the spec")?;
        let (name, arch) = (f.string("name")?, f.string("arch")?);
        let (source, rev) = (f.string("source")?, f.string("rev")?);
        let mut expect = f.object("expect")?;
        let expect_tensors = usize::try_from(expect.uint("tensors")?)
            .map_err(|_| spec_err("expect.tensors does not fit usize"))?;
        let expect_data_bytes = expect.uint("data_bytes")?;
        expect.finish()?;
        let layers = parse_layers(&mut f)?;
        let layer_prefix = f.string("layer_prefix")?;
        let (keep, drop) = (f.strings("keep")?, f.strings("drop")?);
        let copy = parse_copy(&mut f)?;
        let widths = parse_widths(&mut f)?;
        let planes = parse_planes(&mut f, &widths)?;
        let config = parse_config(&mut f, &widths)?;
        f.finish()?;
        Ok(Spec {
            name,
            arch,
            source,
            rev,
            expect_tensors,
            expect_data_bytes,
            layers,
            layer_prefix,
            keep,
            drop,
            copy,
            widths,
            planes,
            config,
            text: serde_json::to_string(&v).expect("a parsed JSON value serializes"),
        })
    }

    /// The identity keys a cut's first shard carries, `bloomery.fixture.*`, but the manifest's hash.
    fn identity(&self) -> BTreeMap<String, String> {
        let mut m = BTreeMap::new();
        let mut put = |k: &str, v: String| m.insert(format!("{KEY_PREFIX}{k}"), v);
        put("version", IDENTITY_VERSION.to_string());
        put("source", self.source.clone());
        put("rev", self.rev.clone());
        put(
            "layers",
            self.layers
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(","),
        );
        for w in &self.widths {
            put(&w.name, w.keep.to_string());
        }
        m
    }
}

// ---------------------------------------------------------------------------------------------
// The source: a folder's tensors, from headers alone

/// One tensor of a folder: where its bytes lie.
#[derive(Debug, Clone)]
struct Src {
    name: String,
    dtype: Dtype,
    shape: Vec<usize>,
    /// The shard's path under the folder.
    shard: String,
    /// The first byte of the tensor in the shard file.
    start: u64,
    len: u64,
}

/// A checkpoint folder's tensors and top-level files.
struct Source {
    dir: PathBuf,
    tensors: Vec<Src>,
}

impl Source {
    fn open(dir: &Path) -> Result<Source, CutError> {
        let folder = Folder::open(dir)?;
        let mut tensors = Vec::new();
        for name in folder.names() {
            let raw = folder.raw(name)?;
            let (path, range) = folder.file_range(name)?;
            let shard = path
                .strip_prefix(dir)
                .ok()
                .and_then(Path::to_str)
                .ok_or_else(|| {
                    differs(format!("{name}: its shard {} is not UTF-8", path.display()))
                })?
                .to_string();
            tensors.push(Src {
                name: name.to_string(),
                dtype: raw.dtype,
                shape: raw.shape.to_vec(),
                shard,
                start: range.start,
                len: range.end - range.start,
            });
        }
        Ok(Source {
            dir: dir.to_path_buf(),
            tensors,
        })
    }

    fn data_bytes(&self) -> u64 {
        self.tensors.iter().map(|t| t.len).sum()
    }

    /// The folder the spec was written for: its tensor count and data bytes.
    fn check_identity(&self, spec: &Spec) -> Result<(), CutError> {
        if self.tensors.len() != spec.expect_tensors {
            return Err(CutError::Identity {
                what: "tensors",
                have: self.tensors.len().to_string(),
                want: spec.expect_tensors.to_string(),
            });
        }
        if self.data_bytes() != spec.expect_data_bytes {
            return Err(CutError::Identity {
                what: "data bytes",
                have: self.data_bytes().to_string(),
                want: spec.expect_data_bytes.to_string(),
            });
        }
        Ok(())
    }

    fn read(&self, file: &str) -> Result<String, CutError> {
        let path = self.dir.join(file);
        fs::read_to_string(&path).map_err(io_err(&path, "read"))
    }
}

// ---------------------------------------------------------------------------------------------
// The plan: every kept tensor, its cut shape, its source view, its place in the output

/// The bytes of a tensor's source: `count` runs of `take` bytes, `stride` apart, from `start`. A
/// whole tensor and a row prefix are one run; a column prefix is one run a row.
#[derive(Debug, Clone, PartialEq, Eq)]
struct View {
    start: u64,
    count: u64,
    take: u64,
    stride: u64,
}

impl View {
    fn whole(start: u64, len: u64) -> View {
        View {
            start,
            count: 1,
            take: len,
            stride: len,
        }
    }

    fn len(&self) -> u64 {
        self.count * self.take
    }

    /// The bytes from the first run's start to the last run's end: what a read of it covers.
    fn span(&self) -> u64 {
        if self.count == 0 {
            0
        } else {
            (self.count - 1) * self.stride + self.take
        }
    }
}

/// One tensor of the cut.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Planned {
    name: String,
    dtype: Dtype,
    shape: Vec<usize>,
    src_name: String,
    src_shard: String,
    view: View,
    /// The output shard's number from 1 and the tensor's first byte in its file.
    shard: usize,
    offset: u64,
}

struct Plan {
    tensors: Vec<Planned>,
    shards: usize,
    dropped_tensors: usize,
    dropped_bytes: u64,
}

/// A matrix's planes found among the tensors: one per member of its plane set.
struct Instance {
    base: String,
    set: usize,
    layer: usize,
    /// The index of the member's tensor, in the set's member order.
    members: Vec<usize>,
}

fn base_matches(pattern: &[String], base: &str) -> bool {
    let parts: Vec<&str> = base.split('.').collect();
    parts.len() == pattern.len() && parts.iter().zip(pattern).all(|(p, q)| q == "*" || p == q)
}

/// The tensor `name` split under `prefix`: the layer id and the name inside the layer.
fn layered<'a>(prefix: &str, name: &'a str) -> Option<(usize, &'a str)> {
    let rest = name.strip_prefix(prefix)?;
    let (id, tail) = rest.split_once('.')?;
    Some((id.parse().ok()?, tail))
}

/// The matrices of `items` (a layer id and a name inside the layer each): every plane set's
/// instances with all their members, a member with no instance's other planes and a tensor two sets
/// claim refused by name.
fn find_instances(spec: &Spec, items: &[(usize, &str)]) -> Result<Vec<Instance>, CutError> {
    let mut found: BTreeMap<(usize, String, usize), Vec<Option<usize>>> = BTreeMap::new();
    for (i, &(layer, tail)) in items.iter().enumerate() {
        let Some((base, suffix)) = tail.rsplit_once('.') else {
            continue;
        };
        let mut hits = spec.planes.iter().enumerate().filter_map(|(s, set)| {
            let m = set.members.iter().position(|(sfx, _)| sfx == suffix)?;
            base_matches(&set.base, base).then_some((s, m))
        });
        let Some((s, m)) = hits.next() else {
            continue;
        };
        if hits.next().is_some() {
            return Err(CutError::Planes {
                base: base.to_string(),
                why: format!("{suffix} is claimed by two plane sets"),
            });
        }
        let slots = found
            .entry((layer, base.to_string(), s))
            .or_insert_with(|| vec![None; spec.planes[s].members.len()]);
        slots[m] = Some(i);
    }
    let mut out = Vec::new();
    for ((layer, base, set), slots) in found {
        if let Some(m) = slots.iter().position(Option::is_none) {
            return Err(CutError::Planes {
                base,
                why: format!(
                    "no {} plane beside the others",
                    spec.planes[set].members[m].0
                ),
            });
        }
        out.push(Instance {
            base,
            set,
            layer,
            members: slots.into_iter().flatten().collect(),
        });
    }
    Ok(out)
}

/// The values a matrix's planes span along the cut axis, each plane's stored extent times its
/// unit; the planes must be 2-D, share their rows and span the same values.
fn plane_extent(
    spec: &Spec,
    inst: &Instance,
    tensors: &[(&str, &[usize])],
) -> Result<u64, CutError> {
    let set = &spec.planes[inst.set];
    let axis = if set.axis == Axis::Rows { 0 } else { 1 };
    let mut common: Option<(u64, &str)> = None;
    let mut rows: Option<(usize, &str)> = None;
    for (&at, (suffix, unit)) in inst.members.iter().zip(&set.members) {
        let (name, shape) = tensors[at];
        let &[r, _] = shape else {
            return Err(CutError::Width {
                tensor: name.to_string(),
                width: spec.widths[set.width].name.clone(),
                keep: spec.widths[set.width].keep,
                why: format!("which cuts a 2-D plane, and this one is {}-D", shape.len()),
            });
        };
        let values = to_u64(shape[axis]) * unit;
        match (common, rows) {
            (Some((v, first)), _) if v != values => {
                return Err(CutError::Planes {
                    base: inst.base.clone(),
                    why: format!("{first} spans {v} values and {suffix} spans {values}"),
                });
            }
            (_, Some((n, first))) if n != r => {
                return Err(CutError::Planes {
                    base: inst.base.clone(),
                    why: format!("{first} has {n} rows and {suffix} has {r}"),
                });
            }
            _ => {}
        }
        common = Some((values, suffix.as_str()));
        rows = Some((r, suffix.as_str()));
    }
    Ok(common.expect("a plane set has members").0)
}

/// `tensor` cut to `keep` values along the set's axis, `unit` values to the element.
fn slice(spec: &Spec, set: &PlaneSet, src: &Src, unit: u64) -> (Vec<usize>, View) {
    let width = &spec.widths[set.width];
    let (rows, cols) = (src.shape[0], src.shape[1]);
    let kept = usize::try_from(width.keep / unit).expect("a kept count fits usize");
    let elem = to_u64(src.dtype.size());
    let (rows64, cols64) = (to_u64(rows), to_u64(cols));
    match set.axis {
        Axis::Rows => {
            let take = width.keep / unit * cols64 * elem;
            (
                vec![kept, cols],
                View {
                    start: src.start,
                    count: 1,
                    take,
                    stride: take,
                },
            )
        }
        Axis::Cols => (
            vec![rows, kept],
            View {
                start: src.start,
                count: rows64,
                take: width.keep / unit * elem,
                stride: cols64 * elem,
            },
        ),
    }
}

/// Which tensors the spec keeps: per source tensor its output group (0 for the unlayered ones,
/// 1 + the output layer id) and name, or `None` for a dropped one.
struct Fates {
    kept: Vec<Option<(usize, String)>>,
    dropped_tensors: usize,
    dropped_bytes: u64,
}

/// Every tensor of `src` kept, dropped or refused by name; every kept name and layer must exist.
fn classify(spec: &Spec, src: &[Src]) -> Result<Fates, CutError> {
    let renumber: BTreeMap<usize, usize> = spec.layers.iter().copied().zip(0..).collect();
    let mut fates = Fates {
        kept: Vec::with_capacity(src.len()),
        dropped_tensors: 0,
        dropped_bytes: 0,
    };
    for t in src {
        let fate = match layered(&spec.layer_prefix, &t.name) {
            Some((id, tail)) => renumber
                .get(&id)
                .map(|&new| (1 + new, format!("{}{new}.{tail}", spec.layer_prefix))),
            None if spec.keep.contains(&t.name) => Some((0, t.name.clone())),
            None if spec.drop.iter().any(|p| t.name.starts_with(p.as_str())) => None,
            None => {
                return Err(CutError::Unclaimed {
                    name: t.name.clone(),
                });
            }
        };
        if fate.is_none() {
            fates.dropped_tensors += 1;
            fates.dropped_bytes += t.len;
        }
        fates.kept.push(fate);
    }
    for name in &spec.keep {
        if !src.iter().any(|t| &t.name == name) {
            return Err(CutError::Missing(name.clone()));
        }
    }
    for (&old, &new) in &renumber {
        if !fates.kept.iter().flatten().any(|(g, _)| *g == 1 + new) {
            return Err(CutError::Missing(format!(
                "layer {old} (nothing under {}{old}.)",
                spec.layer_prefix
            )));
        }
    }
    Ok(fates)
}

/// The layer id and the name inside the layer of each tensor, `usize::MAX` and the whole name for
/// an unlayered one: what [`find_instances`] reads.
fn layer_items<'a>(spec: &Spec, names: impl Iterator<Item = &'a str>) -> Vec<(usize, &'a str)> {
    names
        .map(|n| layered(&spec.layer_prefix, n).unwrap_or((usize::MAX, n)))
        .collect()
}

/// The cut shape and source view of every plane of every matrix in `kept` (a subset of `src`):
/// the width checked against the block, the matrix and each plane's unit before any plane is cut.
fn slice_planes(
    spec: &Spec,
    kept: &[&Src],
) -> Result<HashMap<usize, (Vec<usize>, View)>, CutError> {
    let items = layer_items(spec, kept.iter().map(|t| t.name.as_str()));
    let shapes: Vec<(&str, &[usize])> = kept
        .iter()
        .map(|t| (t.name.as_str(), t.shape.as_slice()))
        .collect();
    let mut sliced = HashMap::new();
    for inst in find_instances(spec, &items)? {
        let set = &spec.planes[inst.set];
        let width = &spec.widths[set.width];
        let tensor = |m: usize| kept[inst.members[m]].name.clone();
        if !width.keep.is_multiple_of(width.align) {
            return Err(CutError::Unaligned {
                width: width.name.clone(),
                keep: width.keep,
                align: width.align,
                tensor: tensor(0),
            });
        }
        let values = plane_extent(spec, &inst, &shapes)?;
        for (m, (_, unit)) in set.members.iter().enumerate() {
            let why = if width.keep > values {
                Some(format!("and the matrix spans {values}"))
            } else if !width.keep.is_multiple_of(*unit) {
                Some(format!("which its {unit}-value elements do not fill whole"))
            } else {
                None
            };
            if let Some(why) = why {
                return Err(CutError::Width {
                    tensor: tensor(m),
                    width: width.name.clone(),
                    keep: width.keep,
                    why,
                });
            }
            sliced.insert(
                inst.members[m],
                slice(spec, set, kept[inst.members[m]], *unit),
            );
        }
    }
    Ok(sliced)
}

/// `tensors` (each with its group) in write order, each given its shard and offset: a group's
/// tensors keep their source order, groups follow one another whole, and a group opens a new shard
/// when it would take the current one past `shard_bytes`. Returns the tensors and the shard count.
fn place(
    mut tensors: Vec<(usize, Planned)>,
    shard_bytes: u64,
) -> Result<(Vec<Planned>, usize), CutError> {
    tensors.sort_by(|(ga, a), (gb, b)| {
        (ga, &a.src_shard, a.view.start).cmp(&(gb, &b.src_shard, b.view.start))
    });
    let mut out: Vec<Planned> = Vec::with_capacity(tensors.len());
    let (mut shard, mut used) = (1usize, 0u64);
    let mut i = 0;
    while i < tensors.len() {
        let group = tensors[i].0;
        let end = tensors[i..]
            .iter()
            .position(|(g, _)| *g != group)
            .map_or(tensors.len(), |n| i + n);
        let size: u64 = tensors[i..end].iter().map(|(_, t)| t.view.len()).sum();
        if used > 0 && used + size > shard_bytes {
            shard += 1;
            used = 0;
        }
        for (_, t) in &mut tensors[i..end] {
            let align = to_u64(t.dtype.size());
            if !used.is_multiple_of(align) {
                return Err(CutError::Misaligned {
                    name: t.name.clone(),
                    shard,
                    offset: used,
                    align,
                });
            }
            t.shard = shard;
            t.offset = used;
            used += t.view.len();
        }
        out.extend(tensors[i..end].iter().map(|(_, t)| t.clone()));
        i = end;
    }
    Ok((out, shard))
}

/// The layers and tensors the spec keeps, their cut shapes and source views, and their place in
/// the output shards.
fn plan(spec: &Spec, src: &[Src], shard_bytes: u64) -> Result<Plan, CutError> {
    let fates = classify(spec, src)?;
    let kept: Vec<&Src> = src
        .iter()
        .zip(&fates.kept)
        .filter_map(|(t, f)| f.as_ref().map(|_| t))
        .collect();
    let mut sliced = slice_planes(spec, &kept)?;
    let mut tensors = Vec::with_capacity(kept.len());
    for (k, (t, fate)) in kept.iter().zip(fates.kept.iter().flatten()).enumerate() {
        let (shape, view) = sliced
            .remove(&k)
            .unwrap_or_else(|| (t.shape.clone(), View::whole(t.start, t.len)));
        tensors.push((
            fate.0,
            Planned {
                name: fate.1.clone(),
                dtype: t.dtype,
                shape,
                src_name: t.name.clone(),
                src_shard: t.shard.clone(),
                view,
                shard: 0,
                offset: 0,
            },
        ));
    }
    let (tensors, shards) = place(tensors, shard_bytes)?;
    Ok(Plan {
        tensors,
        shards,
        dropped_tensors: fates.dropped_tensors,
        dropped_bytes: fates.dropped_bytes,
    })
}

/// The final name of shard `n` of `of`.
fn shard_name(n: usize, of: usize) -> String {
    format!("model-{n:05}-of-{of:05}.safetensors")
}

// ---------------------------------------------------------------------------------------------
// Reading a view

/// The shard files of a folder, opened when first read.
struct Files {
    dir: PathBuf,
    open: HashMap<String, File>,
}

impl Files {
    fn new(dir: &Path) -> Files {
        Files {
            dir: dir.to_path_buf(),
            open: HashMap::new(),
        }
    }

    fn get(&mut self, shard: &str) -> Result<(&File, PathBuf), CutError> {
        let path = self.dir.join(shard);
        if !self.open.contains_key(shard) {
            let f = File::open(&path).map_err(io_err(&path, "open"))?;
            self.open.insert(shard.to_string(), f);
        }
        Ok((&self.open[shard], path))
    }
}

fn to_u64(n: usize) -> u64 {
    u64::try_from(n).expect("a usize fits u64")
}

fn len_of(n: u64) -> usize {
    usize::try_from(n).expect("a piece of a chunk fits usize")
}

/// `f` over the bytes of `view` in order, in pieces of about `chunk` bytes. A column prefix is read
/// a block of rows at a time and compacted in place, so a read is one call, not one a row.
fn each_piece(
    file: &File,
    path: &Path,
    view: &View,
    chunk: u64,
    buf: &mut Vec<u8>,
    f: &mut dyn FnMut(&[u8]) -> Result<(), CutError>,
) -> Result<(), CutError> {
    let read = |buf: &mut [u8], at: u64| file.read_exact_at(buf, at).map_err(io_err(path, "read"));
    if view.count <= 1 || view.take == view.stride {
        let total = view.len();
        let mut done = 0;
        while done < total {
            let n = (total - done).min(chunk);
            buf.resize(len_of(n), 0);
            read(buf, view.start + done)?;
            f(buf)?;
            done += n;
        }
        return Ok(());
    }
    let rows_per = (chunk / view.stride).max(1);
    let mut row = 0;
    while row < view.count {
        let rows = rows_per.min(view.count - row);
        buf.resize(len_of((rows - 1) * view.stride + view.take), 0);
        read(buf, view.start + row * view.stride)?;
        let (stride, take) = (len_of(view.stride), len_of(view.take));
        for r in 0..len_of(rows) {
            buf.copy_within(r * stride..r * stride + take, r * take);
        }
        f(&buf[..len_of(rows) * take])?;
        row += rows;
    }
    Ok(())
}

/// The SHA-256 of `view`'s bytes in `shard` of `files`.
fn hash_view(
    files: &mut Files,
    shard: &str,
    view: &View,
    buf: &mut Vec<u8>,
) -> Result<String, CutError> {
    let (file, path) = files.get(shard)?;
    let mut sha = Sha256::new();
    each_piece(file, &path, view, CHUNK_BYTES, buf, &mut |piece| {
        sha.update(piece);
        Ok(())
    })?;
    Ok(hex(&sha.finish()))
}

// ---------------------------------------------------------------------------------------------
// config.json

fn at_path<'a>(v: &'a mut Value, path: &str) -> Result<&'a mut Value, CutError> {
    let mut cur = v;
    for part in path.split('.') {
        cur = cur
            .get_mut(part)
            .ok_or_else(|| CutError::Config(format!("no {path} (stopped at {part})")))?;
    }
    Ok(cur)
}

fn number(n: u64) -> Value {
    Value::from(n)
}

/// `text`, a source `config.json`, with the cut's edits: the layer count, the widths, each
/// per-layer array reduced to the kept layers, each list of layer-named strings renumbered.
fn edit_config(spec: &Spec, text: &str) -> Result<String, CutError> {
    let mut root: Value =
        serde_json::from_str(text).map_err(|e| CutError::Config(format!("not JSON: {e}")))?;
    let edit = &spec.config;
    let count = at_path(&mut root, &edit.layer_count)?
        .as_u64()
        .ok_or_else(|| CutError::Config(format!("{} is not a whole number", edit.layer_count)))?;
    if let Some(&last) = spec
        .layers
        .last()
        .filter(|&&l| u64::try_from(l).is_ok_and(|l| l >= count))
    {
        return Err(CutError::Config(format!(
            "{} says {count} layers and the spec keeps layer {last}",
            edit.layer_count
        )));
    }
    *at_path(&mut root, &edit.layer_count)? = number(to_u64(spec.layers.len()));
    for (path, w) in &edit.widths {
        *at_path(&mut root, path)? = number(spec.widths[*w].keep);
    }
    for path in &edit.per_layer {
        let arr = at_path(&mut root, path)?
            .as_array()
            .ok_or_else(|| CutError::Config(format!("{path} is not an array")))?
            .clone();
        if u64::try_from(arr.len()) != Ok(count) {
            return Err(CutError::Config(format!(
                "{path} holds {} entries, the config says {count} layers",
                arr.len()
            )));
        }
        let picked = spec.layers.iter().map(|&l| arr[l].clone()).collect();
        *at_path(&mut root, path)? = Value::Array(picked);
    }
    let renumber: BTreeMap<usize, usize> = spec.layers.iter().copied().zip(0..).collect();
    for path in &edit.layer_names {
        let names = at_path(&mut root, path)?
            .as_array()
            .ok_or_else(|| CutError::Config(format!("{path} is not an array")))?
            .clone();
        let mut out = Vec::new();
        for n in names {
            let s = n
                .as_str()
                .ok_or_else(|| CutError::Config(format!("{path} holds {n}, not a string")))?;
            match layered(&spec.layer_prefix, s) {
                None => out.push(n),
                Some((id, tail)) => {
                    if let Some(new) = renumber.get(&id) {
                        out.push(Value::from(format!("{}{new}.{tail}", spec.layer_prefix)));
                    }
                }
            }
        }
        *at_path(&mut root, path)? = Value::Array(out);
    }
    let mut out = serde_json::to_string_pretty(&root).expect("a JSON value serializes");
    out.push('\n');
    Ok(out)
}

// ---------------------------------------------------------------------------------------------
// The manifest

/// One manifest row: a tensor of the cut, its source range and the SHA-256 of its bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Row {
    planned: Planned,
    sha256: String,
}

struct Manifest {
    spec_text: String,
    source: String,
    shard_bytes: u64,
    rows: Vec<Row>,
}

fn shape_text(shape: &[usize]) -> String {
    if shape.is_empty() {
        "-".to_string()
    } else {
        shape
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",")
    }
}

impl Manifest {
    fn render(&self, shards: usize) -> String {
        let mut s = format!(
            "# bloomery-cut\t{MANIFEST_VERSION}\n# spec\t{}\n# source\t{}\n# shard_bytes\t{}\n",
            self.spec_text, self.source, self.shard_bytes
        );
        s.push_str("# columns\tname\tdtype\tshape\tshard\tsrc_name\tsrc_shard\tstart\tcount\ttake\tstride\tsha256\n");
        for r in &self.rows {
            let p = &r.planned;
            s.push_str(&format!(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
                p.name,
                p.dtype.name(),
                shape_text(&p.shape),
                shard_name(p.shard, shards),
                p.src_name,
                p.src_shard,
                p.view.start,
                p.view.count,
                p.view.take,
                p.view.stride,
                r.sha256
            ));
        }
        s.push_str(&format!("# complete\t{}\n", self.rows.len()));
        s
    }

    fn parse(text: &str) -> Result<Manifest, CutError> {
        let bad = |what: String| CutError::Manifest(what);
        let mut lines = text.lines();
        let mut header = |key: &str| -> Result<String, CutError> {
            let line = lines
                .next()
                .ok_or_else(|| bad(format!("ends before its # {key} line")))?;
            line.strip_prefix(&format!("# {key}\t"))
                .map(str::to_string)
                .ok_or_else(|| bad(format!("wants the # {key} line, found {line:?}")))
        };
        let version = header("bloomery-cut")?;
        if version != MANIFEST_VERSION {
            return Err(bad(format!(
                "is version {version}, this reader reads {MANIFEST_VERSION}"
            )));
        }
        let spec_text = header("spec")?;
        let source = header("source")?;
        let shard_bytes = header("shard_bytes")?
            .parse()
            .map_err(|_| bad("# shard_bytes is not a number".to_string()))?;
        header("columns")?;
        let mut rows = Vec::new();
        let mut complete = None;
        for line in lines {
            if let Some(n) = line.strip_prefix("# complete\t") {
                complete = Some(n.to_string());
                continue;
            }
            if complete.is_some() {
                return Err(bad("holds a line after its # complete line".to_string()));
            }
            rows.push(parse_row(line, &bad)?);
        }
        match complete {
            Some(n) if n == rows.len().to_string() => {}
            Some(n) => {
                return Err(bad(format!(
                    "# complete says {n} rows and {} follow",
                    rows.len()
                )));
            }
            None => {
                return Err(bad(
                    "has no # complete line: the cut did not finish".to_string()
                ));
            }
        }
        Ok(Manifest {
            spec_text,
            source,
            shard_bytes,
            rows,
        })
    }
}

fn parse_row(line: &str, bad: &dyn Fn(String) -> CutError) -> Result<Row, CutError> {
    let cols: Vec<&str> = line.split('\t').collect();
    let [
        name,
        dtype,
        shape,
        shard,
        src_name,
        src_shard,
        start,
        count,
        take,
        stride,
        sha256,
    ] = cols[..]
    else {
        return Err(bad(format!(
            "a row has {} columns, not 11: {line:.80}",
            cols.len()
        )));
    };
    let num = |what: &str, v: &str| {
        v.parse::<u64>()
            .map_err(|_| bad(format!("{name}: {what} {v:?} is not a number")))
    };
    let dtype =
        Dtype::of(dtype).ok_or_else(|| bad(format!("{name}: dtype {dtype:?} is unknown")))?;
    let shape = if shape == "-" {
        Vec::new()
    } else {
        shape
            .split(',')
            .map(|d| {
                d.parse::<usize>()
                    .map_err(|_| bad(format!("{name}: shape {shape:?}")))
            })
            .collect::<Result<_, _>>()?
    };
    let shard = shard
        .strip_prefix("model-")
        .and_then(|s| s.split_once("-of-"))
        .and_then(|(n, _)| n.parse::<usize>().ok())
        .ok_or_else(|| {
            bad(format!(
                "{name}: shard {shard:?} is not model-N-of-M.safetensors"
            ))
        })?;
    if sha256.len() != 64 || !sha256.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(bad(format!("{name}: {sha256:?} is not a SHA-256")));
    }
    Ok(Row {
        planned: Planned {
            name: name.to_string(),
            dtype,
            shape,
            src_name: src_name.to_string(),
            src_shard: src_shard.to_string(),
            view: View {
                start: num("start", start)?,
                count: num("count", count)?,
                take: num("take", take)?,
                stride: num("stride", stride)?,
            },
            shard,
            offset: 0,
        },
        sha256: sha256.to_string(),
    })
}

// ---------------------------------------------------------------------------------------------
// Writing the folder

/// What [`cut`] is asked beyond its spec.
pub struct Options {
    /// The tensor data a shard holds at most when the next layer's group fits under it.
    pub shard_bytes: u64,
    /// More `bloomery.fixture.*` keys for the first shard's metadata (the card budget), each key
    /// whole.
    pub extra: Vec<(String, String)>,
}

/// One tensor written, for the caller's per-tensor line.
pub struct Written<'a> {
    pub name: &'a str,
    pub src_name: &'a str,
    pub shard: &'a str,
    pub bytes: u64,
    pub sha256: &'a str,
}

/// What a cut wrote.
#[derive(Debug)]
pub struct Cut {
    pub out: PathBuf,
    pub tensors: usize,
    pub data_bytes: u64,
    /// The bytes of the source the cut covers by reads: a column prefix reads whole rows.
    pub read_bytes: u64,
    pub shards: usize,
    pub dropped_tensors: usize,
    pub dropped_bytes: u64,
    pub copied: Vec<String>,
    /// Top-level files and folders of the source the cut neither writes nor copies.
    pub left_out: Vec<String>,
    pub manifest_sha256: String,
    pub secs: f64,
}

/// The header of a shard holding `tensors` in order: its length word, the JSON padded to 8 bytes,
/// and where `placeholder` first stands in the file (to patch in place).
fn shard_header(
    tensors: &[&Planned],
    metadata: &BTreeMap<String, String>,
    placeholder: &str,
) -> (Vec<u8>, Option<u64>) {
    let json = |s: &str| Value::from(s).to_string();
    let mut text = String::from(r#"{"__metadata__":{"#);
    let meta: Vec<String> = metadata
        .iter()
        .map(|(k, v)| format!("{}:{}", json(k), json(v)))
        .collect();
    text.push_str(&meta.join(","));
    text.push('}');
    for t in tensors {
        let dims: Vec<String> = t.shape.iter().map(usize::to_string).collect();
        text.push_str(&format!(
            r#",{}:{{"dtype":"{}","shape":[{}],"data_offsets":[{},{}]}}"#,
            json(&t.name),
            t.dtype.name(),
            dims.join(","),
            t.offset,
            t.offset + t.view.len()
        ));
    }
    text.push('}');
    let patch = text.find(placeholder).map(|at| to_u64(at) + 8);
    while !text.len().is_multiple_of(8) {
        text.push(' ');
    }
    let mut out = to_u64(text.len()).to_le_bytes().to_vec();
    out.extend_from_slice(text.as_bytes());
    (out, patch)
}

fn write_all(w: &mut impl Write, path: &Path, bytes: &[u8]) -> Result<(), CutError> {
    w.write_all(bytes).map_err(io_err(path, "write"))
}

/// The top-level entries of `dir` the cut neither writes nor copies, sorted; a folder ends in `/`.
fn left_out(dir: &Path, spec: &Spec) -> Result<Vec<String>, CutError> {
    let mut out = Vec::new();
    for e in fs::read_dir(dir).map_err(io_err(dir, "list"))? {
        let e = e.map_err(io_err(dir, "list"))?;
        let name = e.file_name().to_string_lossy().into_owned();
        let is_dir = e.file_type().map_err(io_err(&e.path(), "stat"))?.is_dir();
        let own = name.ends_with(".safetensors")
            || name == Folder::INDEX
            || name == CONFIG
            || spec.copy.contains(&name);
        if !own {
            out.push(if is_dir { format!("{name}/") } else { name });
        }
    }
    out.sort();
    Ok(out)
}

/// The shard file `path` of `tensors` after `header`: each gathered from its source view, hashed on
/// the way, and the file synced. Returns each tensor's record.
fn write_shard(
    files: &mut Files,
    buf: &mut Vec<u8>,
    path: &Path,
    tensors: &[&Planned],
    header: &[u8],
    on_tensor: &mut dyn FnMut(&Written),
) -> Result<Vec<Row>, CutError> {
    let name = path
        .file_name()
        .map_or_else(String::new, |n| n.to_string_lossy().into_owned());
    let file = File::create(path).map_err(io_err(path, "create"))?;
    let mut w = BufWriter::with_capacity(1 << 20, file);
    write_all(&mut w, path, header)?;
    let mut rows = Vec::with_capacity(tensors.len());
    for &t in tensors {
        let (from, from_path) = files.get(&t.src_shard)?;
        let mut sha = Sha256::new();
        each_piece(from, &from_path, &t.view, CHUNK_BYTES, buf, &mut |piece| {
            sha.update(piece);
            write_all(&mut w, path, piece)
        })?;
        let sha256 = hex(&sha.finish());
        on_tensor(&Written {
            name: &t.name,
            src_name: &t.src_name,
            shard: &name,
            bytes: t.view.len(),
            sha256: &sha256,
        });
        rows.push(Row {
            planned: t.clone(),
            sha256,
        });
    }
    let file = w
        .into_inner()
        .map_err(|e| io_err(path, "write")(e.into_error()))?;
    file.sync_all().map_err(io_err(path, "sync"))?;
    Ok(rows)
}

/// The record into `tmp`, and its SHA-256 patched over the placeholder at file offset `patch_at`
/// of the first shard. Returns the hash.
fn write_record(
    tmp: &Path,
    shards: usize,
    manifest: &str,
    patch_at: u64,
) -> Result<String, CutError> {
    let record = tmp.join(MANIFEST);
    fs::write(&record, manifest).map_err(io_err(&record, "write"))?;
    let mut sha = Sha256::new();
    sha.update(manifest.as_bytes());
    let manifest_sha256 = hex(&sha.finish());
    let first = tmp.join(shard_name(1, shards));
    let f = fs::OpenOptions::new()
        .write(true)
        .open(&first)
        .map_err(io_err(&first, "open"))?;
    f.write_all_at(manifest_sha256.as_bytes(), patch_at)
        .map_err(io_err(&first, "write"))?;
    f.sync_all().map_err(io_err(&first, "sync"))?;
    Ok(manifest_sha256)
}

/// The index, the edited config and the copied files into `tmp`.
fn write_small_files(tmp: &Path, plan: &Plan, texts: &Texts) -> Result<(), CutError> {
    let weight_map: Map<String, Value> = plan
        .tensors
        .iter()
        .map(|t| {
            (
                t.name.clone(),
                Value::from(shard_name(t.shard, plan.shards)),
            )
        })
        .collect();
    let index = serde_json::json!({
        "metadata": {"total_size": plan.tensors.iter().map(|t| t.view.len()).sum::<u64>()},
        "weight_map": weight_map,
    });
    let mut index_text = serde_json::to_string_pretty(&index).expect("a JSON value serializes");
    index_text.push('\n');
    let writes = [
        (Folder::INDEX, index_text.as_bytes()),
        (CONFIG, texts.config.as_bytes()),
    ]
    .into_iter()
    .chain(texts.copies.iter().map(|(n, b)| (n.as_str(), b.as_slice())));
    for (name, bytes) in writes {
        let path = tmp.join(name);
        fs::write(&path, bytes).map_err(io_err(&path, "write"))?;
    }
    Ok(())
}

/// The shards, record, index, config and copies of a cut into the empty directory `tmp`. Returns
/// the record's hash.
fn write_folder(
    src: &Source,
    spec: &Spec,
    plan: &Plan,
    texts: &Texts,
    opts: &Options,
    tmp: &Path,
    on_tensor: &mut dyn FnMut(&Written),
) -> Result<String, CutError> {
    let mut files = Files::new(&src.dir);
    let mut buf = Vec::new();
    let placeholder = "0".repeat(64);
    let mut rows = Vec::with_capacity(plan.tensors.len());
    let mut patch_at = None;
    for n in 1..=plan.shards {
        let tensors: Vec<&Planned> = plan.tensors.iter().filter(|t| t.shard == n).collect();
        let mut meta = BTreeMap::from([("format".to_string(), "pt".to_string())]);
        if n == 1 {
            meta.extend(spec.identity());
            meta.extend(opts.extra.iter().cloned());
            meta.insert(KEY_MANIFEST_SHA256.to_string(), placeholder.clone());
        }
        let (header, patch) = shard_header(&tensors, &meta, &placeholder);
        if n == 1 {
            patch_at = patch;
        }
        let path = tmp.join(shard_name(n, plan.shards));
        rows.extend(write_shard(
            &mut files, &mut buf, &path, &tensors, &header, on_tensor,
        )?);
    }
    let manifest = Manifest {
        spec_text: spec.text.clone(),
        source: src.dir.display().to_string(),
        shard_bytes: opts.shard_bytes,
        rows,
    }
    .render(plan.shards);
    let patch_at = patch_at.expect("the first shard's header holds the placeholder");
    let manifest_sha256 = write_record(tmp, plan.shards, &manifest, patch_at)?;
    write_small_files(tmp, plan, texts)?;
    Ok(manifest_sha256)
}

/// The small files of a cut, read before any shard is written.
struct Texts {
    config: String,
    copies: Vec<(String, Vec<u8>)>,
}

impl Texts {
    fn of(src: &Source, spec: &Spec) -> Result<Texts, CutError> {
        let config = edit_config(spec, &src.read(CONFIG)?)?;
        let mut copies = Vec::new();
        for name in &spec.copy {
            let path = src.dir.join(name);
            match fs::read(&path) {
                Ok(bytes) => copies.push((name.clone(), bytes)),
                Err(e) if e.kind() == io::ErrorKind::NotFound => {
                    return Err(CutError::Missing(format!(
                        "{name} (a file the spec copies)"
                    )));
                }
                Err(e) => return Err(io_err(&path, "read")(e)),
            }
        }
        Ok(Texts { config, copies })
    }
}

/// Cut the folder `source` as `spec` says into the new directory `out`: the shards, the index, the
/// edited config, the copied files and [`MANIFEST`]. Everything is planned from headers before a
/// byte is read, and the folder is written beside `out` and renamed on success, so a refused or
/// failed cut leaves nothing.
pub fn cut(
    source: &Path,
    out: &Path,
    spec: &Spec,
    opts: &Options,
    on_tensor: &mut dyn FnMut(&Written),
) -> Result<Cut, CutError> {
    let t0 = Instant::now();
    if out.try_exists().map_err(io_err(out, "stat"))? {
        return Err(CutError::Exists(out.to_path_buf()));
    }
    if let Some((k, _)) = opts.extra.iter().find(|(k, _)| !k.starts_with(KEY_PREFIX)) {
        return Err(spec_err(format!(
            "the extra key {k} does not start with {KEY_PREFIX}"
        )));
    }
    let src = Source::open(source)?;
    src.check_identity(spec)?;
    let plan = plan(spec, &src.tensors, opts.shard_bytes)?;
    let texts = Texts::of(&src, spec)?;
    let left_out = left_out(&src.dir, spec)?;
    let mut tmp = out.as_os_str().to_owned();
    tmp.push(format!(".tmp.{}", std::process::id()));
    let tmp = PathBuf::from(tmp);
    fs::create_dir(&tmp).map_err(io_err(&tmp, "create"))?;
    let done = write_folder(&src, spec, &plan, &texts, opts, &tmp, on_tensor).and_then(|sha| {
        fs::rename(&tmp, out)
            .map_err(io_err(out, "rename"))
            .map(|()| sha)
    });
    let manifest_sha256 = match done {
        Ok(sha) => sha,
        Err(e) => {
            return Err(match fs::remove_dir_all(&tmp) {
                Ok(()) => e,
                Err(c) => differs(format!("{e}; and removing {} failed: {c}", tmp.display())),
            });
        }
    };
    Ok(Cut {
        out: out.to_path_buf(),
        tensors: plan.tensors.len(),
        data_bytes: plan.tensors.iter().map(|t| t.view.len()).sum(),
        read_bytes: plan.tensors.iter().map(|t| t.view.span()).sum(),
        shards: plan.shards,
        dropped_tensors: plan.dropped_tensors,
        dropped_bytes: plan.dropped_bytes,
        copied: texts.copies.into_iter().map(|(n, _)| n).collect(),
        left_out,
        manifest_sha256,
        secs: t0.elapsed().as_secs_f64(),
    })
}

// ---------------------------------------------------------------------------------------------
// Verifying a cut

/// One tensor checked, for the caller's per-tensor line.
pub struct Checked<'a> {
    pub name: &'a str,
    pub bytes: u64,
    pub sha256: &'a str,
}

/// What a verify held the cut to.
#[derive(Debug)]
pub struct Verified {
    pub tensors: usize,
    pub data_bytes: u64,
    pub read_bytes: u64,
    pub shards: usize,
    pub copied: usize,
    pub secs: f64,
}

/// A shard's `__metadata__`: every value a string.
pub(crate) fn read_metadata(path: &Path) -> Result<BTreeMap<String, String>, CutError> {
    let f = File::open(path).map_err(io_err(path, "open"))?;
    let mut len = [0u8; 8];
    f.read_exact_at(&mut len, 0).map_err(io_err(path, "read"))?;
    let mut text = vec![0u8; len_of(u64::from_le_bytes(len).min(CHUNK_BYTES))];
    f.read_exact_at(&mut text, 8)
        .map_err(io_err(path, "read"))?;
    let v: Value = serde_json::from_slice(&text)
        .map_err(|e| differs(format!("{}: the header is not JSON: {e}", path.display())))?;
    let Some(Value::Object(meta)) = v.get("__metadata__") else {
        return Err(differs(format!("{}: no __metadata__", path.display())));
    };
    meta.iter()
        .map(|(k, v)| match v.as_str() {
            Some(s) => Ok((k.clone(), s.to_string())),
            None => Err(differs(format!(
                "{}: __metadata__ {k} is not a string",
                path.display()
            ))),
        })
        .collect()
}

/// The first shard's identity: the spec's keys and the hash of the record.
fn check_identity_keys(
    dir: &Path,
    spec: &Spec,
    shards: usize,
    manifest: &str,
) -> Result<(), CutError> {
    let first = dir.join(shard_name(1, shards));
    let meta = read_metadata(&first)?;
    let mut want = spec.identity();
    want.insert(KEY_MANIFEST_SHA256.to_string(), {
        let mut sha = Sha256::new();
        sha.update(manifest.as_bytes());
        hex(&sha.finish())
    });
    for (k, v) in want {
        if meta.get(&k) != Some(&v) {
            return Err(differs(format!(
                "{}: {k} is {:?}, the record and spec make it {v}",
                first.display(),
                meta.get(&k)
            )));
        }
    }
    Ok(())
}

/// The cut folder `dir` held to its record, its spec and its source: every plane set consistent in
/// the cut's own headers, every tensor, shard, offset, the config and the copied files as the spec
/// makes them from the source's headers, and every tensor's SHA-256, of its bytes in the cut and of
/// its range in the source, equal to the record's. `source` overrides the folder the record names.
pub fn verify(
    dir: &Path,
    source: Option<&Path>,
    on_tensor: &mut dyn FnMut(&Checked),
) -> Result<Verified, CutError> {
    let t0 = Instant::now();
    let record = dir.join(MANIFEST);
    let manifest_text = fs::read_to_string(&record).map_err(io_err(&record, "read"))?;
    let manifest = Manifest::parse(&manifest_text)?;
    let spec = Spec::parse(&manifest.spec_text)?;
    let cut = Source::open(dir)?;
    check_cut_planes(&spec, &cut.tensors)?;
    let src =
        Source::open(&source.map_or_else(|| PathBuf::from(&manifest.source), Path::to_path_buf))?;
    src.check_identity(&spec)?;
    let plan = plan(&spec, &src.tensors, manifest.shard_bytes)?;
    check_rows(&plan, &manifest.rows)?;
    check_headers(dir, &plan, &cut)?;
    check_identity_keys(dir, &spec, plan.shards, &manifest_text)?;
    check_files(&src, dir, &spec)?;
    let (mut src_files, mut cut_files, mut buf) =
        (Files::new(&src.dir), Files::new(dir), Vec::new());
    for (p, r) in plan.tensors.iter().zip(&manifest.rows) {
        let held = |side: &'static str, got: String| {
            if got == r.sha256 {
                Ok(())
            } else {
                Err(CutError::Hash {
                    name: p.name.clone(),
                    side,
                    got,
                    want: r.sha256.clone(),
                })
            }
        };
        let at = cut
            .tensors
            .iter()
            .find(|t| t.name == p.name)
            .expect("check_headers found it");
        held(
            "cut",
            hash_view(
                &mut cut_files,
                &at.shard,
                &View::whole(at.start, at.len),
                &mut buf,
            )?,
        )?;
        held(
            "source",
            hash_view(&mut src_files, &p.src_shard, &p.view, &mut buf)?,
        )?;
        on_tensor(&Checked {
            name: &p.name,
            bytes: p.view.len(),
            sha256: &r.sha256,
        });
    }
    Ok(Verified {
        tensors: plan.tensors.len(),
        data_bytes: plan.tensors.iter().map(|t| t.view.len()).sum(),
        read_bytes: plan.tensors.iter().map(|t| t.view.span()).sum(),
        shards: plan.shards,
        copied: spec.copy.len(),
        secs: t0.elapsed().as_secs_f64(),
    })
}

/// Every plane set of the cut's own headers spans the width it was cut to, in every plane.
fn check_cut_planes(spec: &Spec, tensors: &[Src]) -> Result<(), CutError> {
    let items = layer_items(spec, tensors.iter().map(|t| t.name.as_str()));
    let shapes: Vec<(&str, &[usize])> = tensors
        .iter()
        .map(|t| (t.name.as_str(), t.shape.as_slice()))
        .collect();
    for inst in find_instances(spec, &items)? {
        let width = &spec.widths[spec.planes[inst.set].width];
        let values = plane_extent(spec, &inst, &shapes)?;
        if values != width.keep {
            return Err(CutError::Planes {
                base: format!("layers.{}.{}", inst.layer, inst.base),
                why: format!(
                    "the planes span {values} values, width {} keeps {}",
                    width.name, width.keep
                ),
            });
        }
    }
    Ok(())
}

/// The record's rows are the plan's tensors, one for one.
fn check_rows(plan: &Plan, rows: &[Row]) -> Result<(), CutError> {
    if plan.tensors.len() != rows.len() {
        return Err(differs(format!(
            "the spec and the source make {} tensors and the record has {} rows",
            plan.tensors.len(),
            rows.len()
        )));
    }
    for (p, r) in plan.tensors.iter().zip(rows) {
        let q = &r.planned;
        let same = (
            &p.name,
            p.dtype,
            &p.shape,
            &p.src_name,
            &p.src_shard,
            &p.view,
            p.shard,
        ) == (
            &q.name,
            q.dtype,
            &q.shape,
            &q.src_name,
            &q.src_shard,
            &q.view,
            q.shard,
        );
        if !same {
            return Err(differs(format!(
                "{}: the record's row ({q:?}) is not what the spec makes of the source ({p:?})",
                p.name
            )));
        }
    }
    Ok(())
}

/// The cut's headers hold the plan's tensors: dtype, shape, shard and offset, and nothing more.
fn check_headers(dir: &Path, plan: &Plan, cut: &Source) -> Result<(), CutError> {
    if cut.tensors.len() != plan.tensors.len() {
        return Err(differs(format!(
            "the cut holds {} tensors, the plan {}",
            cut.tensors.len(),
            plan.tensors.len()
        )));
    }
    let by_name: HashMap<&str, &Src> = cut.tensors.iter().map(|t| (t.name.as_str(), t)).collect();
    let mut data_base: HashMap<String, u64> = HashMap::new();
    for p in &plan.tensors {
        let t = by_name
            .get(p.name.as_str())
            .ok_or_else(|| differs(format!("{}: the cut does not hold it", p.name)))?;
        let file = shard_name(p.shard, plan.shards);
        let base = match data_base.get(&file) {
            Some(&b) => b,
            None => {
                let path = dir.join(&file);
                let f = File::open(&path).map_err(io_err(&path, "open"))?;
                let mut len = [0u8; 8];
                f.read_exact_at(&mut len, 0)
                    .map_err(io_err(&path, "read"))?;
                let b = 8 + u64::from_le_bytes(len);
                data_base.insert(file.clone(), b);
                b
            }
        };
        if (t.dtype, &t.shape, &t.shard, t.start, t.len)
            != (p.dtype, &p.shape, &file, base + p.offset, p.view.len())
        {
            return Err(differs(format!(
                "{}: the cut's header has {:?} {:?} in {} at {}..+{}, the plan {:?} {:?} in {file} at {}..+{}",
                p.name,
                t.dtype,
                t.shape,
                t.shard,
                t.start,
                t.len,
                p.dtype,
                p.shape,
                base + p.offset,
                p.view.len()
            )));
        }
    }
    Ok(())
}

/// The cut's config is the spec's edit of the source's, its index adds up, and its copies are the
/// source's bytes.
fn check_files(src: &Source, dir: &Path, spec: &Spec) -> Result<(), CutError> {
    let want = edit_config(spec, &src.read(CONFIG)?)?;
    let path = dir.join(CONFIG);
    let have = fs::read_to_string(&path).map_err(io_err(&path, "read"))?;
    if have != want {
        return Err(differs(format!(
            "{}: it is not the spec's edit of the source's {CONFIG}",
            path.display()
        )));
    }
    for name in &spec.copy {
        let (a, b) = (src.dir.join(name), dir.join(name));
        let (ab, bb) = (
            fs::read(&a).map_err(io_err(&a, "read"))?,
            fs::read(&b).map_err(io_err(&b, "read"))?,
        );
        if ab != bb {
            return Err(differs(format!(
                "{}: it is not the source's {name}",
                b.display()
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use serde_json::json;
    use std::cmp::Reverse;

    /// The toy checkpoint's hidden width, the routed width before the cut, and its expert ids (a
    /// string order that is not numeric: 0, 1, 10).
    const H: usize = 64;
    const F: usize = 128;
    const EXPERTS: [usize; 3] = [0, 1, 10];
    /// The width the toy spec keeps: two 16-value scale groups, one 32-value block.
    const KEEP: usize = 32;
    /// The source layer each output layer keeps (the toy spec's `layers`).
    const LAYERS: [usize; 3] = [0, 2, 3];

    fn temp_dir(tag: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("model-hf-cut-{tag}-{}-{nanos}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// `n` pseudo-random bytes from `seed`, so no two tensors share a byte pattern.
    fn fill(seed: &str, n: usize) -> Vec<u8> {
        let mut x = seed.bytes().fold(0x9E37_79B9_7F4A_7C15u64, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01B3)
        }) | 1;
        (0..n)
            .map(|_| {
                x ^= x >> 12;
                x ^= x << 25;
                x ^= x >> 27;
                (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 56) as u8
            })
            .collect()
    }

    /// One tensor of the toy source.
    pub(crate) struct T {
        name: String,
        dtype: &'static str,
        shape: Vec<usize>,
        data: Vec<u8>,
    }

    fn tensor(name: &str, dtype: &'static str, shape: &[usize]) -> T {
        let n = shape.iter().product::<usize>() * Dtype::of(dtype).unwrap().size();
        T {
            name: name.to_string(),
            dtype,
            shape: shape.to_vec(),
            data: fill(name, n),
        }
    }

    /// Layer `l`: attention, a norm, a dense FFN on layer 0, and on the others a router, a shared
    /// FFN and three experts whose six planes are the quantized layout the spec slices.
    fn layer(l: usize) -> Vec<T> {
        let p = format!("m.layers.{l}.");
        let mut v = vec![
            tensor(&format!("{p}input_layernorm.weight"), "BF16", &[H]),
            tensor(&format!("{p}self_attn.q_proj.weight"), "F8_E4M3", &[H, H]),
            tensor(
                &format!("{p}self_attn.q_proj.weight_scale_inv"),
                "U8",
                &[H, H / 32],
            ),
        ];
        if l == 0 {
            v.push(tensor(
                &format!("{p}mlp.gate_proj.weight"),
                "F8_E4M3",
                &[96, H],
            ));
            v.push(tensor(
                &format!("{p}mlp.gate_proj.weight_scale_inv"),
                "U8",
                &[96, H / 32],
            ));
            return v;
        }
        v.push(tensor(
            &format!("{p}moe.gate.weight"),
            "F32",
            &[EXPERTS.len(), H],
        ));
        v.push(tensor(&format!("{p}moe.bias"), "F32", &[EXPERTS.len()]));
        v.push(tensor(
            &format!("{p}moe.shared.up_proj.weight"),
            "F8_E4M3",
            &[32, H],
        ));
        v.push(tensor(
            &format!("{p}moe.shared.up_proj.weight_scale_inv"),
            "U8",
            &[32, H / 32],
        ));
        for e in EXPERTS {
            for (w, rows, cols) in [("w1", F, H), ("w2", H, F), ("w3", F, H)] {
                let q = format!("{p}moe.experts.{e}.{w}.");
                v.push(tensor(
                    &format!("{q}weight_packed"),
                    "U8",
                    &[rows, cols / 2],
                ));
                v.push(tensor(
                    &format!("{q}weight_scale"),
                    "F8_E4M3",
                    &[rows, cols / 16],
                ));
                v.push(tensor(&format!("{q}weight_global_scale"), "F32", &[]));
                v.push(tensor(&format!("{q}input_global_scale"), "F32", &[]));
            }
        }
        v
    }

    fn toy() -> Vec<T> {
        let mut v = vec![
            tensor("m.embed.weight", "BF16", &[16, H]),
            tensor("m.head.weight", "BF16", &[16, H]),
            tensor("m.norm.weight", "BF16", &[H]),
            tensor("vision.patch.weight", "BF16", &[8]),
            tensor("vision.bias", "F32", &[4]),
        ];
        for l in 0..4 {
            v.extend(layer(l));
        }
        v
    }

    /// The shard a source tensor lives in: the unlayered ones and layers 0 to 2 in shard 1, written
    /// in the order (element size, name) (each matrix's codes then its scales); layer 3 and the
    /// vision tensors in shard 2, written dtype by dtype with every F8 plane before every U8 one
    /// (a matrix's scales far from its codes).
    fn shard_of(t: &T) -> usize {
        if t.name.starts_with("m.layers.3.") || t.name.starts_with("vision.") {
            2
        } else {
            1
        }
    }

    fn write_source(dir: &Path, all: &[T], extra: &[(&str, &str)]) {
        let rank_b = |d: &str| match d {
            "F32" => 0,
            "BF16" => 1,
            "F8_E4M3" => 2,
            _ => 3,
        };
        for n in 1..=2 {
            let mut ts: Vec<&T> = all.iter().filter(|t| shard_of(t) == n).collect();
            if n == 1 {
                ts.sort_by_key(|t| (Reverse(Dtype::of(t.dtype).unwrap().size()), t.name.clone()));
            } else {
                ts.sort_by_key(|t| (rank_b(t.dtype), t.name.clone()));
            }
            let parts: Vec<(&str, &str, &[usize], &[u8])> = ts
                .iter()
                .map(|t| {
                    (
                        t.name.as_str(),
                        t.dtype,
                        t.shape.as_slice(),
                        t.data.as_slice(),
                    )
                })
                .collect();
            fs::write(
                dir.join(format!("model-0000{n}-of-00002.safetensors")),
                safetensors::write(&parts),
            )
            .unwrap();
        }
        let weight_map: Map<String, Value> = all
            .iter()
            .map(|t| {
                (
                    t.name.clone(),
                    json!(format!("model-0000{}-of-00002.safetensors", shard_of(t))),
                )
            })
            .collect();
        let index = json!({"metadata": {"total_size": all.iter().map(|t| t.data.len()).sum::<usize>()}, "weight_map": weight_map});
        fs::write(dir.join(Folder::INDEX), index.to_string()).unwrap();
        let ignored: Vec<String> = (1..4)
            .map(|l| format!("m.layers.{l}.moe.gate"))
            .chain(["head".to_string()])
            .collect();
        let config = json!({
            "model_type": "toy",
            "text_config": {
                "num_hidden_layers": 4,
                "intermediate_size": F,
                "dense_intermediate_size": 96,
                "moe_layer_freq": [0, 1, 1, 1],
                "sparse": {"freq": [1, 1, 1, 1]},
            },
            "quantization_config": {"ignored_layers": ignored},
        });
        fs::write(
            dir.join(CONFIG),
            serde_json::to_string_pretty(&config).unwrap(),
        )
        .unwrap();
        fs::write(dir.join("tokenizer.json"), b"{\"toy\": true}\n").unwrap();
        for (name, text) in extra {
            fs::write(dir.join(name), text).unwrap();
        }
    }

    /// The toy spec: three of the four layers, width `keep` on the block `align`, the w2 scale
    /// plane at `scale_unit` values to the element (16 is true).
    pub(crate) fn spec_json(all: &[T], keep: u64, align: u64, scale_unit: u64) -> String {
        let plane = |w: &str, axis: &str, codes: u64, scales: u64| {
            json!({"base": format!("moe.experts.*.{w}"), "axis": axis, "width": "ff",
                   "members": {"weight_packed": codes, "weight_scale": scales}})
        };
        json!({
            "name": "toy", "arch": "toy", "source": "toy/Toy", "rev": "r1",
            "expect": {"tensors": all.len(), "data_bytes": all.iter().map(|t| t.data.len()).sum::<usize>()},
            "layers": LAYERS,
            "layer_prefix": "m.layers.",
            "keep": ["m.embed.weight", "m.head.weight", "m.norm.weight"],
            "drop": ["vision."],
            "copy": ["tokenizer.json"],
            "widths": {"ff": {"keep": keep, "align": align}},
            "planes": [plane("w1", "rows", 1, 1), plane("w3", "rows", 1, 1), plane("w2", "cols", 2, scale_unit)],
            "config": {
                "layer_count": "text_config.num_hidden_layers",
                "widths": {"text_config.intermediate_size": "ff"},
                "per_layer": ["text_config.moe_layer_freq", "text_config.sparse.freq"],
                "layer_names": ["quantization_config.ignored_layers"],
            },
        })
        .to_string()
    }

    fn opts(shard_bytes: u64) -> Options {
        Options {
            shard_bytes,
            extra: Vec::new(),
        }
    }

    fn quiet(_: &Written) {}

    /// A written toy source and its tensors, in a directory of its own.
    pub(crate) struct World {
        pub(crate) root: PathBuf,
        pub(crate) src: PathBuf,
        pub(crate) all: Vec<T>,
    }

    pub(crate) fn world(tag: &str) -> World {
        let root = temp_dir(tag);
        let src = root.join("src");
        fs::create_dir(&src).unwrap();
        let all = toy();
        write_source(&src, &all, &[("README.md", "left out\n")]);
        World { root, src, all }
    }

    impl World {
        fn spec(&self) -> Spec {
            Spec::parse(&spec_json(&self.all, KEEP as u64, 32, 16)).unwrap()
        }

        fn cut(&self, out: &str, shard_bytes: u64) -> Result<Cut, CutError> {
            cut(
                &self.src,
                &self.root.join(out),
                &self.spec(),
                &opts(shard_bytes),
                &mut quiet,
            )
        }
    }

    /// The output name of source tensor `name`, `None` when the cut drops it.
    fn out_name(name: &str) -> Option<String> {
        match name.strip_prefix("m.layers.") {
            None => (!name.starts_with("vision.")).then(|| name.to_string()),
            Some(rest) => {
                let (id, tail) = rest.split_once('.').unwrap();
                let at = LAYERS.iter().position(|&l| l.to_string() == id)?;
                Some(format!("m.layers.{at}.{tail}"))
            }
        }
    }

    /// What the toy spec makes of `t`, worked out by hand: a w1 or w3 plane keeps its first KEEP
    /// rows, a w2 plane the first KEEP / unit columns of every row, every other tensor whole.
    fn expected(t: &T) -> (Vec<usize>, Vec<u8>) {
        let (base, suffix) = t.name.rsplit_once('.').unwrap();
        let is_plane =
            base.contains(".experts.") && matches!(suffix, "weight_packed" | "weight_scale");
        if !is_plane {
            return (t.shape.clone(), t.data.clone());
        }
        let (rows, row) = (t.shape[0], t.shape[1]);
        if base.ends_with(".w2") {
            let keep = if suffix == "weight_packed" {
                KEEP / 2
            } else {
                KEEP / 16
            };
            let bytes = (0..rows)
                .flat_map(|r| t.data[r * row..r * row + keep].to_vec())
                .collect();
            (vec![rows, keep], bytes)
        } else {
            (vec![KEEP, row], t.data[..KEEP * row].to_vec())
        }
    }

    fn open(dir: &Path) -> Folder {
        Folder::open(dir).unwrap()
    }

    /// Each kept tensor of the cut is the toy spec's slice of its source tensor, byte for byte: a w1
    /// and a w3 plane keep the first rows, a w2 plane the first columns of every row (the
    /// codes at 2 values a byte, the scales at 16), everything else whole, layers renumbered, the
    /// vision tensors and the dropped layer gone.
    #[test]
    fn a_cut_keeps_each_tensors_source_bytes() {
        let w = world("bytes");
        let c = w.cut("out", DEFAULT_SHARD_BYTES).unwrap();
        let folder = open(&w.root.join("out"));
        let mut want: BTreeMap<String, (Vec<usize>, Vec<u8>, &str)> = BTreeMap::new();
        for t in &w.all {
            if let Some(name) = out_name(&t.name) {
                let (shape, bytes) = expected(t);
                want.insert(name, (shape, bytes, t.dtype));
            }
        }
        let have: Vec<&str> = folder.names().collect();
        assert_eq!(have, want.keys().map(String::as_str).collect::<Vec<_>>());
        for (name, (shape, bytes, dtype)) in &want {
            let raw = folder.raw(name).unwrap();
            assert_eq!(
                (raw.dtype.name(), raw.shape),
                (*dtype, shape.as_slice()),
                "{name}"
            );
            assert!(
                raw.bytes == bytes.as_slice(),
                "{name}: the bytes differ from the hand-made slice"
            );
        }
        assert_eq!(c.tensors, want.len());
        assert_eq!(
            c.data_bytes,
            want.values().map(|(_, b, _)| b.len() as u64).sum::<u64>()
        );
        let dropped = w.all.iter().filter(|t| out_name(&t.name).is_none()).count();
        assert_eq!(c.dropped_tensors, dropped);
        assert_eq!(
            c.shards, 1,
            "everything fits one shard under the default cap"
        );
        assert_eq!(c.left_out, ["README.md"]);
        assert_eq!(c.copied, ["tokenizer.json"]);
        fs::remove_dir_all(&w.root).unwrap();
    }

    /// A layer is never split across shards, a group that does not fit the cap opens a shard, and
    /// a layer's tensors keep the order they have in their source shard: the layer of source
    /// shard 1 keeps each matrix's codes beside its scales, the layer of source shard 2 keeps every
    /// scale plane before every code plane.
    #[test]
    fn a_layer_keeps_its_source_order_and_stays_in_one_shard() {
        let w = world("order");
        let c = w.cut("out", 1).unwrap();
        assert_eq!(
            c.shards, 4,
            "the unlayered group and three layers, one shard each under a cap of 1"
        );
        let folder = open(&w.root.join("out"));
        let place = |name: &str| {
            let (path, range) = folder.file_range(name).unwrap();
            (
                path.file_name().unwrap().to_str().unwrap().to_string(),
                range,
            )
        };
        for (l, &src_layer) in LAYERS.iter().enumerate() {
            let pre = format!("m.layers.{l}.");
            let shards: BTreeSet<String> = folder
                .names()
                .filter(|n| n.starts_with(&pre))
                .map(|n| place(n).0)
                .collect();
            assert_eq!(shards.len(), 1, "layer {l} is in one shard: {shards:?}");
            let mut by_offset: Vec<(String, u64)> = folder
                .names()
                .filter(|n| n.starts_with(&pre))
                .map(|n| (n.to_string(), place(n).1.start))
                .collect();
            by_offset.sort_by_key(|(_, at)| *at);
            let mut source: Vec<(String, usize, String)> = Vec::new();
            for t in w
                .all
                .iter()
                .filter(|t| t.name.starts_with(&format!("m.layers.{src_layer}.")))
            {
                source.push((t.name.clone(), shard_of(t), t.dtype.to_string()));
            }
            let src_folder = open(&w.src);
            source.sort_by_key(|(n, _, _)| {
                let (path, range) = src_folder.file_range(n).unwrap();
                (path, range.start)
            });
            let want: Vec<String> = source
                .iter()
                .map(|(n, _, _)| out_name(n).unwrap())
                .collect();
            let have: Vec<String> = by_offset.into_iter().map(|(n, _)| n).collect();
            assert_eq!(have, want, "layer {l} keeps its source order");
        }
        let adjacent = |l: usize| {
            let pre = format!("m.layers.{l}.moe.experts.0.w1.weight_");
            let (_, codes) = place(&format!("{pre}packed"));
            let (_, scales) = place(&format!("{pre}scale"));
            codes.end == scales.start
        };
        assert!(
            adjacent(1),
            "the layer cut from source shard 1 keeps codes then scales"
        );
        assert!(
            !adjacent(2),
            "the layer cut from source shard 2 keeps its scale planes apart"
        );
        fs::remove_dir_all(&w.root).unwrap();
    }

    /// The edited config, the first shard's identity and the copied file: the layer count, the
    /// routed width and every per-layer array follow the kept layers, a layer-named entry is
    /// renumbered or dropped, the metadata names the spec's source, revision, layers, width, the
    /// extra keys the caller gave and the record's hash, a key outside `bloomery.fixture.` is
    /// refused, and a copied file is the source's bytes.
    #[test]
    fn the_config_and_the_first_shards_identity_follow_the_spec() {
        let w = world("config");
        let extra = |k: &str| Options {
            shard_bytes: DEFAULT_SHARD_BYTES,
            extra: vec![(k.to_string(), "8589934592".to_string())],
        };
        let e = cut(
            &w.src,
            &w.root.join("no"),
            &w.spec(),
            &extra("card_budget"),
            &mut quiet,
        )
        .unwrap_err();
        assert!(
            matches!(&e, CutError::Spec(m) if m.contains("card_budget does not start with")),
            "{e}"
        );
        let budget = "bloomery.fixture.card_budget";
        let c = cut(
            &w.src,
            &w.root.join("out"),
            &w.spec(),
            &extra(budget),
            &mut quiet,
        )
        .unwrap();
        let out = w.root.join("out");
        let have: Value =
            serde_json::from_str(&fs::read_to_string(out.join(CONFIG)).unwrap()).unwrap();
        let want = json!({
            "model_type": "toy",
            "text_config": {
                "num_hidden_layers": 3,
                "intermediate_size": KEEP,
                "dense_intermediate_size": 96,
                "moe_layer_freq": [0, 1, 1],
                "sparse": {"freq": [1, 1, 1]},
            },
            "quantization_config": {"ignored_layers": ["m.layers.1.moe.gate", "m.layers.2.moe.gate", "head"]},
        });
        assert_eq!(have, want);
        let meta = read_metadata(&out.join(shard_name(1, 1))).unwrap();
        let record = fs::read(out.join(MANIFEST)).unwrap();
        let mut sha = Sha256::new();
        sha.update(&record);
        let expect: BTreeMap<String, String> = [
            ("format", "pt".to_string()),
            ("bloomery.fixture.version", "1".to_string()),
            ("bloomery.fixture.source", "toy/Toy".to_string()),
            ("bloomery.fixture.rev", "r1".to_string()),
            ("bloomery.fixture.layers", "0,2,3".to_string()),
            ("bloomery.fixture.ff", KEEP.to_string()),
            (budget, "8589934592".to_string()),
            (KEY_MANIFEST_SHA256, hex(&sha.finish())),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
        assert_eq!(meta, expect);
        assert_eq!(c.manifest_sha256, expect[KEY_MANIFEST_SHA256]);
        assert_eq!(
            fs::read(out.join("tokenizer.json")).unwrap(),
            fs::read(w.src.join("tokenizer.json")).unwrap()
        );
        fs::remove_dir_all(&w.root).unwrap();
    }

    fn copy_dir(from: &Path, to: &Path) {
        fs::create_dir_all(to).unwrap();
        for e in fs::read_dir(from).unwrap() {
            let e = e.unwrap();
            fs::copy(e.path(), to.join(e.file_name())).unwrap();
        }
    }

    /// Flip the byte `at` into `name`'s bytes of the folder `dir`.
    fn flip(dir: &Path, name: &str, at: u64) {
        let (path, range) = {
            let folder = open(dir);
            folder.file_range(name).unwrap()
        };
        let f = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .unwrap();
        let mut b = [0u8];
        f.read_exact_at(&mut b, range.start + at).unwrap();
        f.write_all_at(&[b[0] ^ 0x5a], range.start + at).unwrap();
    }

    fn verified(dir: &Path, source: &Path) -> Result<Verified, CutError> {
        verify(dir, Some(source), &mut |_| {})
    }

    /// A verify holds the cut to its record and its source: a clean cut passes, with one line for
    /// every tensor; one flipped byte in a kept tensor of the cut, or in a byte of the source a
    /// kept range covers, names that tensor and the side it is on; a byte of the source outside
    /// every kept range moves nothing; a record that is not what the spec makes of the source, a
    /// record that never finished, and a config that is not the spec's edit are each refused.
    #[test]
    fn verify_holds_a_cut_to_its_record_and_its_source() {
        let w = world("verify");
        let c = w.cut("out", DEFAULT_SHARD_BYTES).unwrap();
        let out = w.root.join("out");
        let mut lines = 0;
        let v = verify(&out, Some(&w.src), &mut |_| lines += 1).unwrap();
        assert_eq!(
            (v.tensors, lines, v.shards),
            (c.tensors, c.tensors, c.shards)
        );
        assert_eq!(v.data_bytes, c.data_bytes);

        let w2 = "m.layers.1.moe.experts.0.w2.weight_packed";
        let w1 = "m.layers.1.moe.experts.0.w1.weight_scale";
        let mutant = |tag: &str| {
            let d = w.root.join(tag);
            copy_dir(&out, &d);
            d
        };
        let hash = |r: Result<Verified, CutError>| match r {
            Err(CutError::Hash { name, side, .. }) => (name, side),
            other => panic!("wanted a hash refusal, got {other:?}"),
        };

        let d = mutant("cut-flip");
        flip(&d, w2, 7);
        assert_eq!(hash(verified(&d, &w.src)), (w2.to_string(), "cut"));

        // The source byte a kept range covers: w1's scale rows 0..KEEP, row 0 byte 0 (source layer 2).
        let src_dir = w.root.join("src-flip");
        copy_dir(&w.src, &src_dir);
        flip(&src_dir, "m.layers.2.moe.experts.0.w1.weight_scale", 1);
        assert_eq!(hash(verified(&out, &src_dir)), (w1.to_string(), "source"));

        // A byte of w2's codes outside the kept columns (row 0, byte KEEP/2 and on) is in no range.
        let outside = w.root.join("src-outside");
        copy_dir(&w.src, &outside);
        flip(
            &outside,
            "m.layers.2.moe.experts.0.w2.weight_packed",
            (KEEP / 2) as u64 + 3,
        );
        verified(&out, &outside)
            .expect("a byte outside every kept range is not the cut's business");

        let record = fs::read_to_string(out.join(MANIFEST)).unwrap();
        let d = mutant("row");
        let first = record.lines().find(|l| !l.starts_with('#')).unwrap();
        let mut cols: Vec<&str> = first.split('\t').collect();
        cols[5] = "m.norm.weight";
        fs::write(
            d.join(MANIFEST),
            record.replacen(first, &cols.join("\t"), 1),
        )
        .unwrap();
        assert!(matches!(
            verified(&d, &w.src),
            Err(CutError::Differs { .. })
        ));

        let d = mutant("unfinished");
        let cut_short = &record[..record.rfind("# complete").unwrap()];
        fs::write(d.join(MANIFEST), cut_short).unwrap();
        let e = verified(&d, &w.src).unwrap_err().to_string();
        assert!(e.contains("no # complete line"), "{e}");

        let d = mutant("config");
        let config = fs::read_to_string(d.join(CONFIG)).unwrap();
        fs::write(
            d.join(CONFIG),
            config.replace("\"num_hidden_layers\": 3", "\"num_hidden_layers\": 4"),
        )
        .unwrap();
        assert!(matches!(
            verified(&d, &w.src),
            Err(CutError::Differs { .. })
        ));
        fs::remove_dir_all(&w.root).unwrap();
    }

    /// A width that is not a whole number of its blocks is refused by name before a byte is
    /// written, and leaves no directory behind: 48 values are three 16-value scale groups and
    /// no whole 32-value block (the card's down form); so are a width a plane's elements do not
    /// divide and one wider than the matrix.
    #[test]
    fn a_width_off_the_block_grid_is_refused_by_name_and_leaves_nothing() {
        let w = world("grid");
        let try_cut = |keep: u64, align: u64| {
            let spec = Spec::parse(&spec_json(&w.all, keep, align, 16)).unwrap();
            cut(
                &w.src,
                &w.root.join("out"),
                &spec,
                &opts(DEFAULT_SHARD_BYTES),
                &mut quiet,
            )
        };
        let e = try_cut(48, 32).unwrap_err();
        assert!(
            matches!(&e, CutError::Unaligned { width, keep: 48, align: 32, tensor } if width == "ff" && tensor.contains(".w1.")),
            "{e}"
        );
        assert!(
            e.to_string()
                .contains("width ff keeps 48 values, not a multiple of its block 32"),
            "{e}"
        );
        let e = try_cut(24, 8).unwrap_err();
        assert!(
            matches!(&e, CutError::Width { tensor, keep: 24, .. } if tensor.ends_with("weight_scale")),
            "{e}"
        );
        assert!(
            e.to_string()
                .contains("16-value elements do not fill whole"),
            "{e}"
        );
        let e = try_cut(256, 32).unwrap_err();
        assert!(
            matches!(&e, CutError::Width { keep: 256, .. })
                && e.to_string().contains("the matrix spans 128"),
            "{e}"
        );
        let leftovers: Vec<_> = fs::read_dir(&w.root)
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(
            leftovers,
            [std::ffi::OsString::from("src")],
            "a refused cut leaves nothing beside the source"
        );
        fs::remove_dir_all(&w.root).unwrap();
    }

    /// A scale plane that does not span the values its codes do is refused by name: in a spec whose
    /// scale unit disagrees with the source's shapes the cut stops, and a cut folder whose scale
    /// plane lost a column fails its verify before any hash is read.
    #[test]
    fn a_scale_plane_that_disagrees_with_its_codes_is_refused_by_name() {
        let w = world("planes");
        let wrong = Spec::parse(&spec_json(&w.all, KEEP as u64, 32, 32)).unwrap();
        let e = cut(
            &w.src,
            &w.root.join("wrong"),
            &wrong,
            &opts(DEFAULT_SHARD_BYTES),
            &mut quiet,
        )
        .unwrap_err();
        assert!(
            matches!(&e, CutError::Planes { base, why } if base.ends_with("experts.0.w2") && why.contains("weight_packed spans 128 values and weight_scale spans 256")),
            "{e}"
        );
        w.cut("out", DEFAULT_SHARD_BYTES).unwrap();
        let out = w.root.join("out");
        let shard = out.join(shard_name(1, 1));
        let st = safetensors::Safetensors::open(&shard).unwrap();
        let damaged = "m.layers.1.moe.experts.1.w2.weight_scale";
        let held: Vec<(String, &str, Vec<usize>, Vec<u8>)> = st
            .names()
            .map(|n| {
                let r = st.raw(n).unwrap();
                let (shape, bytes) = if n == damaged {
                    (vec![H, 1], r.bytes[..H].to_vec())
                } else {
                    (r.shape.to_vec(), r.bytes.to_vec())
                };
                (n.to_string(), r.dtype.name(), shape, bytes)
            })
            .collect();
        drop(st);
        let parts: Vec<(&str, &str, &[usize], &[u8])> = held
            .iter()
            .map(|(n, d, s, b)| (n.as_str(), *d, s.as_slice(), b.as_slice()))
            .collect();
        fs::write(&shard, safetensors::write(&parts)).unwrap();
        let e = verified(&out, &w.src).unwrap_err();
        assert!(
            matches!(&e, CutError::Planes { why, .. } if why.contains("weight_packed spans 32 values and weight_scale spans 16")),
            "{e}"
        );
        fs::remove_dir_all(&w.root).unwrap();
    }

    /// The spec and the source must agree by name: a tensor no keep or drop rule claims, a kept
    /// tensor or layer the source lacks, a source of another size than the spec was written for,
    /// and an existing output directory are each refused before a byte is written.
    #[test]
    fn the_spec_and_the_source_must_agree_by_name() {
        let w = world("agree");
        let run = |spec: String, out: &str| {
            cut(
                &w.src,
                &w.root.join(out),
                &Spec::parse(&spec).unwrap(),
                &opts(DEFAULT_SHARD_BYTES),
                &mut quiet,
            )
        };
        let base = spec_json(&w.all, KEEP as u64, 32, 16);
        let with = |from: &str, to: &str| {
            assert!(base.contains(from), "{from}");
            base.replace(from, to)
        };
        let e = run(with("\"drop\":[\"vision.\"]", "\"drop\":[]"), "a").unwrap_err();
        assert!(
            matches!(&e, CutError::Unclaimed { name } if name == "vision.bias" || name == "vision.patch.weight"),
            "{e}"
        );
        let e = run(
            with("\"m.norm.weight\"]", "\"m.norm.weight\",\"m.nothing\"]"),
            "b",
        )
        .unwrap_err();
        assert!(
            matches!(&e, CutError::Missing(n) if n == "m.nothing"),
            "{e}"
        );
        let e = run(with("\"layers\":[0,2,3]", "\"layers\":[0,2,7]"), "c").unwrap_err();
        assert!(
            matches!(&e, CutError::Missing(n) if n.starts_with("layer 7")),
            "{e}"
        );
        let tensors = format!("\"tensors\":{}", w.all.len());
        let e = run(
            with(&tensors, &format!("\"tensors\":{}", w.all.len() + 1)),
            "d",
        )
        .unwrap_err();
        assert!(
            matches!(
                &e,
                CutError::Identity {
                    what: "tensors",
                    ..
                }
            ),
            "{e}"
        );
        fs::create_dir(w.root.join("e")).unwrap();
        let e = run(base.clone(), "e").unwrap_err();
        assert!(matches!(&e, CutError::Exists(_)), "{e}");
        for bad in ["config.json", "sub/x.json"] {
            let e = Spec::parse(&with(
                "\"copy\":[\"tokenizer.json\"]",
                &format!("\"copy\":[\"{bad}\"]"),
            ))
            .unwrap_err()
            .to_string();
            assert!(e.contains(&format!("copy {bad}")), "{e}");
        }
        let unknown = base.replacen("\"name\"", "\"nmae\":1,\"name\"", 1);
        let e = Spec::parse(&unknown).unwrap_err();
        assert!(e.to_string().contains("unknown key nmae"), "{e}");
        fs::remove_dir_all(&w.root).unwrap();
    }

    /// A column prefix is read a block of rows at a time and compacted in place: over a file of
    /// rows of 7 bytes of which 3 are kept, a chunk of 1, 10, 20 and a whole file's worth hand `f`
    /// the same bytes as a row-by-row pick, and a chunk smaller than a row still reads row by row.
    #[test]
    fn a_column_prefix_read_is_the_same_across_chunk_sizes() {
        let dir = temp_dir("chunks");
        let path = dir.join("f");
        let data = fill("chunks", 7 * 50 + 11);
        fs::write(&path, &data).unwrap();
        let file = File::open(&path).unwrap();
        let view = View {
            start: 11,
            count: 50,
            take: 3,
            stride: 7,
        };
        let want: Vec<u8> = (0..50)
            .flat_map(|r| data[11 + r * 7..11 + r * 7 + 3].to_vec())
            .collect();
        for chunk in [1, 10, 20, 7 * 50] {
            let mut got = Vec::new();
            each_piece(&file, &path, &view, chunk, &mut Vec::new(), &mut |p| {
                got.extend_from_slice(p);
                Ok(())
            })
            .unwrap();
            assert_eq!(got, want, "chunk {chunk}");
        }
        let whole = View::whole(11, 100);
        let mut got = Vec::new();
        each_piece(&file, &path, &whole, 8, &mut Vec::new(), &mut |p| {
            got.extend_from_slice(p);
            Ok(())
        })
        .unwrap();
        assert_eq!(got, &data[11..111]);
        fs::remove_dir_all(&dir).unwrap();
    }

    /// The tensors of MiniMax-M3.1-Flash's layers, named and shaped as its shard headers have them:
    /// attention with its indexer and norms; a dense FFN on layers 0 to 2; on the others a router,
    /// a shared FFN and 128 experts of NVFP4 planes and two scalars a matrix.
    fn minimax_layer(l: usize, add: &mut dyn FnMut(String, Dtype, &[usize])) {
        let p = format!("language_model.model.layers.{l}.");
        let mxfp8 =
            |add: &mut dyn FnMut(String, Dtype, &[usize]), name: &str, n: usize, k: usize| {
                add(format!("{p}{name}.weight"), Dtype::F8E4M3, &[n, k]);
                add(
                    format!("{p}{name}.weight_scale_inv"),
                    Dtype::U8,
                    &[n, k / 32],
                );
            };
        for (name, n, k) in [
            ("self_attn.q_proj", 8192, 6144),
            ("self_attn.k_proj", 512, 6144),
            ("self_attn.v_proj", 512, 6144),
            ("self_attn.o_proj", 6144, 8192),
            ("self_attn.index_q_proj", 512, 6144),
            ("self_attn.index_k_proj", 128, 6144),
        ] {
            mxfp8(add, name, n, k);
        }
        for norm in ["input_layernorm", "post_attention_layernorm"] {
            add(format!("{p}{norm}.weight"), Dtype::Bf16, &[6144]);
        }
        for norm in ["q_norm", "k_norm", "index_q_norm", "index_k_norm"] {
            add(format!("{p}self_attn.{norm}.weight"), Dtype::Bf16, &[128]);
        }
        if l < 3 {
            for (name, n, k) in [
                ("gate_proj", 12288, 6144),
                ("up_proj", 12288, 6144),
                ("down_proj", 6144, 12288),
            ] {
                mxfp8(add, &format!("mlp.{name}"), n, k);
            }
            return;
        }
        add(
            format!("{p}block_sparse_moe.gate.weight"),
            Dtype::F32,
            &[128, 6144],
        );
        add(
            format!("{p}block_sparse_moe.e_score_correction_bias"),
            Dtype::F32,
            &[128],
        );
        for (name, n, k) in [
            ("gate_proj", 3072, 6144),
            ("up_proj", 3072, 6144),
            ("down_proj", 6144, 3072),
        ] {
            mxfp8(
                add,
                &format!("block_sparse_moe.shared_experts.{name}"),
                n,
                k,
            );
        }
        for e in 0..128 {
            for (w, n, k) in [("w1", 3072, 6144), ("w2", 6144, 3072), ("w3", 3072, 6144)] {
                let q = format!("{p}block_sparse_moe.experts.{e}.{w}.");
                add(format!("{q}weight_packed"), Dtype::U8, &[n, k / 2]);
                add(format!("{q}weight_scale"), Dtype::F8E4M3, &[n, k / 16]);
                add(format!("{q}weight_global_scale"), Dtype::F32, &[]);
                add(format!("{q}input_global_scale"), Dtype::F32, &[]);
            }
        }
    }

    /// The committed MiniMax spec makes the designed fixture of the checkpoint's catalog: layers 0,
    /// 3 and 59 as layers 0, 1, 2; routed width 512; 3,151 tensors and 6,975,526,912 bytes (the
    /// embed, head and norm 4,916,785,152, the dense layer 347,948,032, each MoE layer
    /// 855,396,864: derived from the real shard headers of shards 1, 2 and 48), in two shards
    /// that never split a layer; every other tensor of the catalog (layers 1, the vision tower)
    /// dropped by a named rule. The spec's arch and name are the row of `tools/fixture-dirs.tsv`.
    #[test]
    fn the_minimax_spec_makes_the_designed_fixture() {
        let spec =
            Spec::parse(include_str!("../../../tools/fixture-cuts/minimaxm31.json")).unwrap();
        let table = include_str!("../../../tools/fixture-dirs.tsv");
        assert!(
            table
                .lines()
                .any(|l| l == format!("{}\t{}", spec.arch, spec.name)),
            "tools/fixture-dirs.tsv has no row {}\t{}",
            spec.arch,
            spec.name
        );
        let mut src: Vec<Src> = Vec::new();
        let mut at = 0u64;
        let mut add = |name: String, dtype: Dtype, shape: &[usize]| {
            let len = (shape.iter().product::<usize>() * dtype.size()) as u64;
            src.push(Src {
                name,
                dtype,
                shape: shape.to_vec(),
                shard: "s".to_string(),
                start: at,
                len,
            });
            at += len;
        };
        for name in [
            "language_model.model.embed_tokens.weight",
            "language_model.lm_head.weight",
        ] {
            add(name.to_string(), Dtype::Bf16, &[200064, 6144]);
        }
        add(
            "language_model.model.norm.weight".to_string(),
            Dtype::Bf16,
            &[6144],
        );
        for l in [0, 1, 3, 59] {
            minimax_layer(l, &mut add);
        }
        for name in [
            "vision_tower.vision_model.pre_layrnorm.bias",
            "multi_modal_projector.linear_1.bias",
            "patch_merge_mlp.linear_1.bias",
        ] {
            add(name.to_string(), Dtype::Bf16, &[1280]);
        }
        let p = plan(&spec, &src, DEFAULT_SHARD_BYTES).unwrap();
        let bytes = |pred: &dyn Fn(&Planned) -> bool| {
            p.tensors
                .iter()
                .filter(|t| pred(t))
                .map(|t| t.view.len())
                .sum::<u64>()
        };
        let layer = |l: usize| {
            move |t: &Planned| {
                t.name
                    .starts_with(&format!("language_model.model.layers.{l}."))
            }
        };
        assert_eq!(p.tensors.len(), 3151);
        assert_eq!(bytes(&|_| true), 6_975_526_912);
        assert_eq!(bytes(&|t| !t.name.contains(".layers.")), 4_916_785_152);
        assert_eq!(bytes(&layer(0)), 347_948_032);
        assert_eq!(
            (bytes(&layer(1)), bytes(&layer(2))),
            (855_396_864, 855_396_864)
        );
        assert_eq!(p.dropped_tensors, 24 + 3);
        assert_eq!(p.shards, 2);
        for l in 0..3 {
            let shards: BTreeSet<usize> = p
                .tensors
                .iter()
                .filter(|t| layer(l)(t))
                .map(|t| t.shard)
                .collect();
            assert_eq!(shards.len(), 1, "layer {l} spans {shards:?}");
        }
        let w2 = p
            .tensors
            .iter()
            .find(|t| {
                t.name
                    .ends_with("layers.1.block_sparse_moe.experts.5.w2.weight_packed")
            })
            .unwrap();
        assert_eq!(
            (
                w2.shape.as_slice(),
                w2.view.count,
                w2.view.take,
                w2.view.stride
            ),
            ([6144, 256].as_slice(), 6144, 256, 1536)
        );
        let w1 = p
            .tensors
            .iter()
            .find(|t| {
                t.name
                    .ends_with("layers.2.block_sparse_moe.experts.5.w1.weight_scale")
            })
            .unwrap();
        assert_eq!(
            (w1.shape.as_slice(), w1.view.count, w1.view.take),
            ([512, 384].as_slice(), 1, 512 * 384)
        );
    }

    /// The committed fixture `name` of the hf crate (`tests/fixtures`), gunzipped.
    fn gunzipped(name: &str) -> Vec<u8> {
        use std::io::Read;
        let path = format!("{}/../hf/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
        let raw = fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let mut out = Vec::new();
        flate2::read::GzDecoder::new(raw.as_slice())
            .read_to_end(&mut out)
            .unwrap_or_else(|e| panic!("{path}: {e}"));
        out
    }

    /// The qwen38d spec: the Metal-only MLX fixture cut, which has no registry row.
    fn qwen38d_spec() -> Spec {
        Spec::parse(include_str!(
            "../../../tools/mlx-fixture-cuts/qwen38d-mlx.json"
        ))
        .unwrap()
    }

    /// The llama spec: the Metal-only MLX fixture cut of the 8B folder, which has no registry row.
    fn llama_spec() -> Spec {
        Spec::parse(include_str!(
            "../../../tools/mlx-fixture-cuts/llama-mlx.json"
        ))
        .unwrap()
    }

    /// The qwen3 spec: the Metal-only MLX fixture cut of the dense 8B folder, which has no
    /// registry row.
    fn qwen3_spec() -> Spec {
        Spec::parse(include_str!(
            "../../../tools/mlx-fixture-cuts/qwen3-mlx.json"
        ))
        .unwrap()
    }

    /// The tensors of an MLX folder from its one committed shard header
    /// (`model.safetensors`), each tensor's first byte where the file holds it.
    fn one_file_source(fixture: &str) -> Vec<Src> {
        let raw = gunzipped(fixture);
        let header: Value = serde_json::from_slice(&raw).unwrap();
        let base = 8 + to_u64(raw.len());
        let mut out = Vec::new();
        for (name, e) in header.as_object().unwrap() {
            if name == "__metadata__" {
                continue;
            }
            let (from, to) = (
                e["data_offsets"][0].as_u64().unwrap(),
                e["data_offsets"][1].as_u64().unwrap(),
            );
            out.push(Src {
                name: name.clone(),
                dtype: Dtype::of(e["dtype"].as_str().unwrap()).unwrap(),
                shape: e["shape"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|d| usize::try_from(d.as_u64().unwrap()).unwrap())
                    .collect(),
                shard: "model.safetensors".to_string(),
                start: base + from,
                len: to - from,
            });
        }
        out
    }

    /// The tensors of the 8B MLX folder from its one committed shard header
    /// (`model.safetensors`), each tensor's first byte where the file holds it.
    fn real_8b_source() -> Vec<Src> {
        one_file_source("checkpoint-headers/llama-3.1-8b-instruct-mlx-4bit-model.header.json.gz")
    }

    /// The tensors of the 27B MLX folder from its three committed shard headers, each tensor's first
    /// byte where its shard file holds it (the length word, the header, then its data offset): the
    /// source [`plan`] reads, with no tensor bytes behind it.
    fn real_27b_source() -> Vec<Src> {
        let mut out = Vec::new();
        for n in 1..=3 {
            let raw = gunzipped(&format!(
                "checkpoint-headers/qwen3.8-27b-mlx-4bit-model-0000{n}-of-00003.header.json.gz"
            ));
            let header: Value = serde_json::from_slice(&raw).unwrap();
            let base = 8 + to_u64(raw.len());
            for (name, e) in header.as_object().unwrap() {
                if name == "__metadata__" {
                    continue;
                }
                let (from, to) = (
                    e["data_offsets"][0].as_u64().unwrap(),
                    e["data_offsets"][1].as_u64().unwrap(),
                );
                out.push(Src {
                    name: name.clone(),
                    dtype: Dtype::of(e["dtype"].as_str().unwrap()).unwrap(),
                    shape: e["shape"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|d| usize::try_from(d.as_u64().unwrap()).unwrap())
                        .collect(),
                    shard: format!("model-0000{n}-of-00003.safetensors"),
                    start: base + from,
                    len: to - from,
                });
            }
        }
        out
    }

    /// The qwen38d spec over the real 27B folder's three shard headers (2,180 tensors,
    /// 16,054,262,240 bytes) plans the four-layer fixture: layers 0 to 3, which are three delta
    /// layers and the first full-attention layer (the interval is 4), with the embedding, the head
    /// and the final norm; every tensor whole in its stored bits, each quantized matrix's codes,
    /// scales and biases together; one shard, no layer split; the vision tower and layers 4 to 63
    /// dropped by name. From the headers: 122 tensors and 2,286,737,728 bytes (the seven unlayered
    /// tensors 1,430,333,440; layers 0 to 2 215,665,088 each; layer 3 209,409,024).
    #[test]
    fn the_qwen38d_spec_plans_the_four_layer_fixture_from_the_real_27b_headers() {
        let spec = qwen38d_spec();
        let src = real_27b_source();
        Source {
            dir: PathBuf::new(),
            tensors: src.clone(),
        }
        .check_identity(&spec)
        .unwrap();
        let p = plan(&spec, &src, DEFAULT_SHARD_BYTES).unwrap();
        assert_eq!(p.tensors.len(), 122);
        assert_eq!(
            p.tensors.iter().map(|t| t.view.len()).sum::<u64>(),
            2_286_737_728
        );
        assert_eq!(p.shards, 1);
        // The vision tower's 333, and layers 4 to 63: 45 delta layers of 30 and 15 attention layers of 25.
        assert_eq!(p.dropped_tensors, 333 + 45 * 30 + 15 * 25);
        assert_eq!(p.dropped_bytes, 13_767_524_512);

        let in_layer = |t: &Planned, l: usize| {
            t.name
                .starts_with(&format!("language_model.model.layers.{l}."))
        };
        let layer_bytes = |l: usize| {
            p.tensors
                .iter()
                .filter(|t| in_layer(t, l))
                .map(|t| t.view.len())
                .sum::<u64>()
        };
        assert_eq!(
            p.tensors
                .iter()
                .filter(|t| !t.name.contains(".layers."))
                .map(|t| t.view.len())
                .sum::<u64>(),
            1_430_333_440
        );
        assert_eq!(
            (0..4).map(layer_bytes).collect::<Vec<_>>(),
            [215_665_088, 215_665_088, 215_665_088, 209_409_024]
        );
        for l in 0..4 {
            let names: Vec<&str> = p
                .tensors
                .iter()
                .filter(|t| in_layer(t, l))
                .map(|t| t.name.as_str())
                .collect();
            let has = |part: &str| names.iter().any(|n| n.contains(part));
            // mlx-lm's rule: layer l is full attention when (l + 1) % 4 == 0.
            assert_eq!(has(".linear_attn."), (l + 1) % 4 != 0, "layer {l}");
            assert_eq!(has(".self_attn."), (l + 1) % 4 == 0, "layer {l}");
            for part in ["mlp.gate_proj.", "mlp.up_proj.", "mlp.down_proj."] {
                assert!(has(part), "layer {l}: no {part}");
            }
            assert_eq!(
                names.len(),
                if (l + 1) % 4 == 0 { 25 } else { 30 },
                "layer {l}"
            );
        }
        assert!(
            !p.tensors
                .iter()
                .any(|t| t.name.starts_with("vision_tower."))
        );
        assert!(
            !p.tensors
                .iter()
                .any(|t| layered(&spec.layer_prefix, &t.name).is_some_and(|(id, _)| id >= 4))
        );

        let by_name: HashMap<&str, &Planned> =
            p.tensors.iter().map(|t| (t.name.as_str(), t)).collect();
        let source_of: HashMap<&str, &Src> = src.iter().map(|t| (t.name.as_str(), t)).collect();
        for t in &p.tensors {
            let s = source_of[t.src_name.as_str()];
            assert_eq!(t.view, View::whole(s.start, s.len), "{}", t.name);
            assert_eq!((&t.shape, t.dtype), (&s.shape, s.dtype), "{}", t.name);
        }
        let mut matrices = 0;
        for t in p.tensors.iter().filter(|t| t.dtype == Dtype::U32) {
            let stem = t.name.strip_suffix(".weight").expect("a code plane");
            let groups = [t.shape[0], t.shape[1] / 8];
            for plane in ["scales", "biases"] {
                let c = by_name
                    .get(format!("{stem}.{plane}").as_str())
                    .unwrap_or_else(|| panic!("{stem}: no {plane} beside its codes"));
                assert_eq!(
                    (c.dtype, c.shape.as_slice()),
                    (Dtype::Bf16, groups.as_slice()),
                    "{stem}.{plane}"
                );
            }
            matrices += 1;
        }
        assert_eq!(matrices, 33);
    }

    /// The qwen38d spec over the real 27B `config.json` edits the two fields that name the layers
    /// and nothing else: `text_config.num_hidden_layers` 4 and `text_config.layer_types` its first
    /// four entries, three `linear_attention` then `full_attention`, which are the layer kinds
    /// mlx-lm builds from the kept `full_attention_interval` of 4. The config's `quantization` and
    /// `quantization_config` hold the one block for every module (no entry names a layer), so no
    /// per-module entry is left for a dropped layer.
    #[test]
    fn the_qwen38d_spec_edits_the_real_27b_config_to_four_layers() {
        let spec = qwen38d_spec();
        let real = String::from_utf8(gunzipped(
            "quant-configs/mlx-community__Qwen3.8-27B-4bit__config.json.gz",
        ))
        .unwrap();
        let before: Value = serde_json::from_str(&real).unwrap();
        for key in ["quantization", "quantization_config"] {
            let block: BTreeSet<&str> = before[key]
                .as_object()
                .unwrap()
                .keys()
                .map(String::as_str)
                .collect();
            assert_eq!(
                block,
                BTreeSet::from(["bits", "group_size", "mode"]),
                "{key}"
            );
        }
        let edited: Value = serde_json::from_str(&edit_config(&spec, &real).unwrap()).unwrap();
        let mut want = before;
        want["text_config"]["num_hidden_layers"] = json!(4);
        want["text_config"]["layer_types"] = json!([
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention"
        ]);
        assert_eq!(edited, want);
        let text = &edited["text_config"];
        let interval = text["full_attention_interval"].as_u64().unwrap();
        for (i, kind) in text["layer_types"].as_array().unwrap().iter().enumerate() {
            assert_eq!(
                (to_u64(i) + 1).is_multiple_of(interval),
                kind == "full_attention",
                "layer {i}: layer_types and the interval disagree"
            );
        }
    }

    /// The llama spec over the real 8B folder's one shard header (743 tensors,
    /// 4,517,404,672 bytes) plans the four-layer fixture: every layer of the
    /// trunk alike (23 tensors: four attention projections of three planes
    /// each, three of the FFN, two norm gains), with the embedding, the untied
    /// head and the final norm; every tensor whole in its stored bits, one
    /// shard, no layer split. From the header: 99 tensors and 1,081,810,944
    /// bytes (the seven unlayered tensors 591,011,840; each kept layer
    /// 122,699,776), and the 644 tensors of layers 4 to 31 dropped.
    #[test]
    fn the_llama_spec_plans_the_four_layer_fixture_from_the_real_8b_header() {
        let spec = llama_spec();
        let src = real_8b_source();
        Source {
            dir: PathBuf::new(),
            tensors: src.clone(),
        }
        .check_identity(&spec)
        .unwrap();
        let p = plan(&spec, &src, DEFAULT_SHARD_BYTES).unwrap();
        assert_eq!(p.tensors.len(), 99);
        assert_eq!(
            p.tensors.iter().map(|t| t.view.len()).sum::<u64>(),
            1_081_810_944
        );
        assert_eq!(p.shards, 1);
        assert_eq!(p.dropped_tensors, 28 * 23);
        assert_eq!(p.dropped_bytes, 4_517_404_672 - 1_081_810_944);
        let in_layer = |t: &Planned, l: usize| t.name.starts_with(&format!("model.layers.{l}."));
        let layer_bytes = |l: usize| {
            p.tensors
                .iter()
                .filter(|t| in_layer(t, l))
                .map(|t| t.view.len())
                .sum::<u64>()
        };
        assert_eq!(
            p.tensors
                .iter()
                .filter(|t| !t.name.contains(".layers."))
                .map(|t| t.view.len())
                .sum::<u64>(),
            591_011_840
        );
        assert_eq!(
            (0..4).map(layer_bytes).collect::<Vec<_>>(),
            [122_699_776; 4]
        );
        for l in 0..4 {
            let names: Vec<&str> = p
                .tensors
                .iter()
                .filter(|t| in_layer(t, l))
                .map(|t| t.name.as_str())
                .collect();
            assert_eq!(names.len(), 23, "layer {l}");
            for part in [
                "self_attn.q_proj.",
                "self_attn.k_proj.",
                "self_attn.v_proj.",
                "self_attn.o_proj.",
                "mlp.gate_proj.",
                "mlp.up_proj.",
                "mlp.down_proj.",
                "input_layernorm.",
                "post_attention_layernorm.",
            ] {
                assert!(
                    names.iter().any(|n| n.contains(part)),
                    "layer {l}: no {part}"
                );
            }
        }
        let by_name: HashMap<&str, &Planned> =
            p.tensors.iter().map(|t| (t.name.as_str(), t)).collect();
        let source_of: HashMap<&str, &Src> = src.iter().map(|t| (t.name.as_str(), t)).collect();
        for t in &p.tensors {
            let s = source_of[t.src_name.as_str()];
            assert_eq!(t.view, View::whole(s.start, s.len), "{}", t.name);
            assert_eq!((&t.shape, t.dtype), (&s.shape, s.dtype), "{}", t.name);
        }
        // Every quantized matrix keeps its two F16 planes beside its U32
        // codes: seven a layer and the two unlayered ones.
        let mut matrices = 0;
        for t in p.tensors.iter().filter(|t| t.dtype == Dtype::U32) {
            let stem = t.name.strip_suffix(".weight").expect("a code plane");
            let groups = [t.shape[0], t.shape[1] / 8];
            for plane in ["scales", "biases"] {
                let c = by_name
                    .get(format!("{stem}.{plane}").as_str())
                    .unwrap_or_else(|| panic!("{stem}: no {plane} beside its codes"));
                assert_eq!(
                    (c.dtype, c.shape.as_slice()),
                    (Dtype::F16, groups.as_slice()),
                    "{stem}.{plane}"
                );
            }
            matrices += 1;
        }
        assert_eq!(matrices, 4 * 7 + 2);
    }

    /// The llama spec over the real 8B `config.json` edits the one field that
    /// names the layers and nothing else: `num_hidden_layers` 4.
    #[test]
    fn the_llama_spec_edits_the_real_8b_config_to_four_layers() {
        let spec = llama_spec();
        let real = String::from_utf8(gunzipped(
            "quant-configs/mlx-community__Llama-3.1-8B-Instruct-4bit__config.json.gz",
        ))
        .unwrap();
        let before: Value = serde_json::from_str(&real).unwrap();
        let edited: Value = serde_json::from_str(&edit_config(&spec, &real).unwrap()).unwrap();
        let mut want = before;
        want["num_hidden_layers"] = json!(4);
        assert_eq!(edited, want);
    }

    /// The qwen3 spec over the real 8B folder's one shard header (907 tensors,
    /// 4,607,731,712 bytes) plans the four-layer fixture: every layer of the
    /// trunk alike (25 tensors: four attention projections of three planes
    /// each, the two per-head norm gains, two block norms, three FFN
    /// projections of three planes), with the embedding, the untied head and
    /// the final norm; every tensor whole in its stored bits, one shard, no
    /// layer split. From the header: 107 tensors and 1,134,307,328 bytes (the
    /// seven unlayered tensors 700,129,280; each kept layer 108,544,512), and
    /// the 800 tensors of layers 4 to 35 dropped.
    #[test]
    fn the_qwen3_spec_plans_the_four_layer_fixture_from_the_real_8b_header() {
        let spec = qwen3_spec();
        let src = one_file_source("checkpoint-headers/qwen3-8b-mlx-4bit-model.header.json.gz");
        Source {
            dir: PathBuf::new(),
            tensors: src.clone(),
        }
        .check_identity(&spec)
        .unwrap();
        let p = plan(&spec, &src, DEFAULT_SHARD_BYTES).unwrap();
        assert_eq!(p.tensors.len(), 107);
        assert_eq!(
            p.tensors.iter().map(|t| t.view.len()).sum::<u64>(),
            1_134_307_328
        );
        assert_eq!(p.shards, 1);
        assert_eq!(p.dropped_tensors, 32 * 25);
        assert_eq!(p.dropped_bytes, 4_607_731_712 - 1_134_307_328);
        let in_layer = |t: &Planned, l: usize| t.name.starts_with(&format!("model.layers.{l}."));
        let layer_bytes = |l: usize| {
            p.tensors
                .iter()
                .filter(|t| in_layer(t, l))
                .map(|t| t.view.len())
                .sum::<u64>()
        };
        assert_eq!(
            p.tensors
                .iter()
                .filter(|t| !t.name.contains(".layers."))
                .map(|t| t.view.len())
                .sum::<u64>(),
            700_129_280
        );
        assert_eq!(
            (0..4).map(layer_bytes).collect::<Vec<_>>(),
            [108_544_512; 4]
        );
        for l in 0..4 {
            let names: Vec<&str> = p
                .tensors
                .iter()
                .filter(|t| in_layer(t, l))
                .map(|t| t.name.as_str())
                .collect();
            assert_eq!(names.len(), 25, "layer {l}");
            for part in [
                "self_attn.q_proj.",
                "self_attn.q_norm.",
                "self_attn.k_proj.",
                "self_attn.k_norm.",
                "self_attn.v_proj.",
                "self_attn.o_proj.",
                "mlp.gate_proj.",
                "mlp.up_proj.",
                "mlp.down_proj.",
                "input_layernorm.",
                "post_attention_layernorm.",
            ] {
                assert!(
                    names.iter().any(|n| n.contains(part)),
                    "layer {l}: no {part}"
                );
            }
        }
        let by_name: HashMap<&str, &Planned> =
            p.tensors.iter().map(|t| (t.name.as_str(), t)).collect();
        let source_of: HashMap<&str, &Src> = src.iter().map(|t| (t.name.as_str(), t)).collect();
        for t in &p.tensors {
            let s = source_of[t.src_name.as_str()];
            assert_eq!(t.view, View::whole(s.start, s.len), "{}", t.name);
            assert_eq!((&t.shape, t.dtype), (&s.shape, s.dtype), "{}", t.name);
        }
        // Every quantized matrix keeps its two BF16 planes beside its U32 codes:
        // seven a layer and the two unlayered ones, the norm gains whole BF16.
        let mut matrices = 0;
        for t in p.tensors.iter().filter(|t| t.dtype == Dtype::U32) {
            let stem = t.name.strip_suffix(".weight").expect("a code plane");
            let groups = [t.shape[0], t.shape[1] / 8];
            for plane in ["scales", "biases"] {
                let c = by_name
                    .get(format!("{stem}.{plane}").as_str())
                    .unwrap_or_else(|| panic!("{stem}: no {plane} beside its codes"));
                assert_eq!(
                    (c.dtype, c.shape.as_slice()),
                    (Dtype::Bf16, groups.as_slice()),
                    "{stem}.{plane}"
                );
            }
            matrices += 1;
        }
        assert_eq!(matrices, 4 * 7 + 2);
        for (name, width) in [
            ("model.layers.0.self_attn.q_norm.weight", 128usize),
            ("model.layers.3.self_attn.k_norm.weight", 128),
            ("model.norm.weight", 4096),
        ] {
            let t = by_name[name];
            assert_eq!(
                (t.dtype, t.shape.as_slice()),
                (Dtype::Bf16, &[width][..]),
                "{name}"
            );
        }
    }

    /// The qwen3 spec over the real 8B `config.json` edits the one field that
    /// names the layers and nothing else: `num_hidden_layers` 4.
    #[test]
    fn the_qwen3_spec_edits_the_real_8b_config_to_four_layers() {
        let spec = qwen3_spec();
        let real = String::from_utf8(gunzipped(
            "quant-configs/mlx-community__Qwen3-8B-4bit__config.json.gz",
        ))
        .unwrap();
        let before: Value = serde_json::from_str(&real).unwrap();
        let edited: Value = serde_json::from_str(&edit_config(&spec, &real).unwrap()).unwrap();
        let mut want = before;
        want["num_hidden_layers"] = json!(4);
        assert_eq!(edited, want);
    }
}
