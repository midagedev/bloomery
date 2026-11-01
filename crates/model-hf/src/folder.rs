//! A checkpoint folder as the program reads it, whatever its architecture: the tensors of its
//! split, each declared in the GGUF type its stored format and role give ([`plan_tensors`]), the
//! header those tensors and the folder's keys make ([`read`]), and serving's tokenizer checked
//! against Hugging Face's ([`served_as_hf`]). A model supplies its checkpoint's facts as a
//! [`FolderMap`] (names, layouts, keys); everything else about a folder is here. `prepare` writes
//! that header and the tensors' bytes as a split; a load opens the folder as it is ([`open`]):
//! the same header held as a located split (`gguf::Split::located`), the tensors a card holds read
//! through the folder's part source ([`crate::part::Checkpoint`]), the tensors the host reads kept
//! where the folder stores them ([`Origin`]), and the row tables read there too
//! (`model::part::StoredRows`). Both read the folder through these functions alone, so a folder
//! and the split `prepare` writes of it describe one model.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};

use bloomery_levers::PREPARE_Q8;
use gguf::write::{Layout, TensorDecl};
use gguf::{GgmlType, Place, Split, StackLayout, Value};
use hf::modelopt::{self, Normal, Plain, Use};
use hf::quant_config::{self, Config, Format, ModuleClass, Placed, Plan};
use safetensors::{Dtype, Folder, Raw};
use serde_json::Value as Json;

use model::part::{StoredPlane, StoredRows};

use crate::part::{self, Made};

pub(crate) mod scales;

/// The vocabulary a map names: the config's vocabulary fields.
pub type VocabSpec = hf::vocab::Config;

/// Why a folder was refused.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A model's own refusal of its config, its names or a step.
    #[error(transparent)]
    Model(Box<dyn std::error::Error + Send + Sync>),
    #[error("{name}: {what}")]
    Tensor { name: String, what: String },
    #[error(transparent)]
    Part(#[from] part::Error),
    #[error(transparent)]
    Tensors(#[from] safetensors::Error),
    #[error(transparent)]
    Quant(#[from] quant_config::Error),
    #[error(transparent)]
    Normal(#[from] modelopt::Error),
    #[error(transparent)]
    Vocab(#[from] hf::vocab::Error),
    #[error(transparent)]
    Layout(#[from] gguf::write::WriteError),
    #[error(transparent)]
    Header(#[from] gguf::LoadError),
    #[error(transparent)]
    Tokenizer(#[from] tokenizer::Error),
    #[error("{0}")]
    Refused(String),
}

impl From<String> for Error {
    fn from(why: String) -> Error {
        Error::Refused(why)
    }
}

/// The tensor `name` refused for `what`.
pub fn refuse(name: &str, what: impl Into<String>) -> Error {
    Error::Tensor {
        name: name.to_string(),
        what: what.into(),
    }
}

/// A checkpoint folder's model, as [`read`] takes it: the facts of one architecture's checkpoint.
pub trait FolderMap: Sized {
    /// A move of a normalized tensor into the program's layout, one of the model's own.
    type Step: Clone + std::fmt::Debug + PartialEq;

    /// `config.json`'s `model_type` values (the top level's, else `text_config`'s) this map reads.
    const MODEL_TYPES: &'static [&'static str];

    /// The config read and checked: its geometry, and every field the map does not take refused
    /// by name.
    fn parse(config: &Json) -> Result<Self, Error>;

    /// The header's `general.architecture`: the config's alone, so a model is seated by it before
    /// its files are read ([`architecture`]).
    fn architecture(&self) -> &'static str;

    /// The class of the weight module `module`, for a quantization config that targets classes;
    /// `None` for another or unknown role.
    fn module_class(&self, module: &str) -> Option<ModuleClass>;

    /// The header's tensors of the folder tensors `tensors` (name and row-major logical shape: a
    /// packed weight's unpacked shape), each tensor of the folder a source of one of them or
    /// skipped with the reason (a vision tower, a draft), never dropped.
    fn map<'a>(&self, tensors: &[(&'a str, &'a [usize])]) -> Result<Mapping<Self::Step>, Error>;

    /// The move `steps` make of the normalized tensor `name`, for the part source; `None` when
    /// there are none.
    fn mover(&self, name: &str, steps: &[Self::Step]) -> Option<part::Step>;

    /// The header's model keys, from the config and the planned tensors (name and ggml dims).
    fn header(
        &self,
        planned: &[(&str, &[u64])],
        folder: &Folder,
    ) -> Result<Vec<(String, Value)>, Error>;

    /// The tokenizer's vocabulary fields of the config.
    fn vocab(&self) -> &VocabSpec;

    /// The tensors of the header that no folder tensor holds, by the program's name: values the
    /// map makes of the config. A name a mapped tensor has is refused.
    fn constants(&self) -> Result<BTreeMap<String, Constant>, Error> {
        Ok(BTreeMap::new())
    }

    /// The draft the folder holds beside the target the map reads (tensors the map skips as a
    /// draft's), as a load opens it: its header, named by `path` as the target's header is, and
    /// its planned tensors, each in its stored format, which join the part source; `None` for a
    /// folder that holds no draft, and for a map that reads none. The target's header keys are
    /// `kvs`. A refusal is the draft's alone ([`Opened::draft`]).
    fn draft(
        &self,
        path: &Path,
        folder: &Folder,
        plan: &Plan,
        kvs: &[(String, Value)],
    ) -> Result<Option<Draft<Self::Step>>, Error> {
        let _ = (path, folder, plan, kvs);
        Ok(None)
    }
}

/// A draft model a folder holds: its header and its planned tensors.
pub type Draft<S> = (Split, Vec<Planned<S>>);

/// What a map makes of a folder: its header tensors and the folder tensors it leaves out.
pub struct Mapping<S> {
    pub tensors: BTreeMap<String, Mapped<S>>,
    /// Each folder tensor the model does not read, with why.
    pub skipped: BTreeMap<String, String>,
}

/// Where a mapped tensor's bytes are served from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Origin {
    /// Bytes the load makes of the sources (a normalize, a cast, a step): no run of any file holds
    /// them.
    Made,
    /// The one source's bytes as the folder stores them, declared as the type: the host reads
    /// them in place.
    File(GgmlType),
    /// The sources, a matrix each in expert order, as the folder stores them, declared as the
    /// type in the layout: the host reads each matrix in place. Kept only where the stored
    /// format is that type's rows ([`modelopt::type_of`] of [`Use::Rows`]), or NVFP4 whose block
    /// scales are stored compressed, read back once into the scale store ([`scales`]); a stack
    /// stored in another (BF16) is made.
    Stack { ty: GgmlType, layout: StackLayout },
}

/// Where a plan reads a tensor its map keeps in place ([`Origin::File`], [`Origin::Stack`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Keep {
    /// Where the folder stores it: a load of the folder ([`open`]).
    InPlace,
    /// Nowhere: every tensor is made, as in the split `prepare` writes ([`read`]) and in a
    /// draft, whose tensors join the part source.
    Made,
}

/// One tensor of the header and how it is made of the folder's.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mapped<S> {
    /// The folder tensors it is made from: one, a stack's experts in expert order, or a row
    /// table's parts in part order.
    pub sources: Vec<String>,
    pub role: Use,
    pub steps: Vec<S>,
    /// ggml's dims, the fastest axis first.
    pub dims: Vec<u64>,
    pub origin: Origin,
}

/// The plan's place of the tensor `name`, or its refusal by name.
fn place<'p>(plan: &'p Plan, name: &str) -> Result<&'p Placed, Error> {
    plan.placed.get(name).ok_or_else(|| {
        let why = plan
            .refused
            .get(name)
            .map_or("not a weight of the plan", String::as_str);
        refuse(name, why)
    })
}

impl<S> Mapped<S> {
    /// Each source with its place in `plan`, in source order: what [`modelopt::normalize`]
    /// reads, and [`modelopt::format_of`] the one format of.
    pub fn placed<'p>(&'p self, plan: &'p Plan) -> Result<Vec<(&'p str, &'p Placed)>, Error> {
        self.sources
            .iter()
            .map(|s| Ok((s.as_str(), place(plan, s)?)))
            .collect()
    }

    /// A row table's parts, one at a time in order: each normalized by its place in `plan`, and
    /// refused when its type or tensor scale is not the first part's, as the table keeps one.
    pub fn parts<'s, 'a: 's>(
        &'s self,
        plan: &'s Plan,
        get: impl Fn(&str) -> Result<Raw<'a>, safetensors::Error> + Copy + 's,
    ) -> impl Iterator<Item = Result<Normal, Error>> + 's {
        let mut table: Option<(u32, Vec<f32>)> = None;
        self.sources.iter().map(move |src| {
            let p = place(plan, src)?;
            let n = modelopt::normalize(get, &[(src.as_str(), p)], self.role)
                .map_err(|e| refuse(src, e.to_string()))?;
            match &table {
                Some((ty, scale)) if (*ty, scale) != (n.type_id, &n.scale) => Err(refuse(
                    src,
                    format!(
                        "type {} scale {:?}, the table's type {ty} scale {scale:?}",
                        n.type_id, n.scale
                    ),
                )),
                _ => {
                    table.get_or_insert_with(|| (n.type_id, n.scale.clone()));
                    Ok(n)
                }
            }
        })
    }
}

/// A tensor of the header that no folder tensor holds: F32 values the map makes of the config,
/// such as the rope's frequency factors, which a GGUF file stores as a tensor.
#[derive(Clone, Debug, PartialEq)]
pub struct Constant {
    /// ggml's dims, the fastest axis first.
    pub dims: Vec<u64>,
    /// The values, as many as the dims hold, all finite.
    pub values: Vec<f32>,
}

impl Constant {
    /// The values as the little-endian bytes of the F32 tensor.
    pub fn bytes(&self) -> Vec<u8> {
        self.values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }
}

/// `x + 1` on an F32 norm gain: the Gemma norm scales by `1 + weight`, and the program's norm
/// scales by its gain. A map whose only move is this one takes it as its [`FolderMap::Step`]; a
/// map with moves of its own has a step that calls it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlusOne;

impl PlusOne {
    /// Add 1 to every value of the F32 gain `t`. A gain that is not F32 or holds a value that is
    /// not finite is refused with why (the caller names the tensor), and `t` is then as it was.
    pub fn apply(&self, t: &mut Normal) -> Result<(), String> {
        if t.type_id != GgmlType::F32.as_u32() {
            return Err(format!("+1 needs F32, not type {}", t.type_id));
        }
        let gains = t.bytes.as_chunks_mut::<4>().0;
        if let Some(x) = gains
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .find(|x| !x.is_finite())
        {
            return Err(format!("+1 of non-finite input {x}"));
        }
        for b in gains {
            *b = (f32::from_le_bytes(*b) + 1.0).to_le_bytes();
        }
        Ok(())
    }
}

/// The one unit dim of a 3-D tensor dropped, its bytes as they are: an MLX convolution stores its
/// taps `[channels, kernel, 1]` (ggml dims `[1, kernel, channels]`), rows of `kernel` values once
/// it is gone. A map whose only move is this one takes it as its [`FolderMap::Step`]; a map with
/// moves of its own has a step that calls it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Squeeze;

impl Squeeze {
    /// Drop the unit dim of the 3-D tensor `t`. A tensor of another rank, or with no unit dim or
    /// more than one, is refused with why (the caller names the tensor), and `t` is then as it
    /// was.
    pub fn apply(&self, t: &mut Normal) -> Result<(), String> {
        let unit = t.dims.iter().filter(|&&n| n == 1).count();
        if t.dims.len() != 3 || unit != 1 {
            return Err(format!(
                "{:?} is not a 3-D tensor with one unit dim",
                t.dims
            ));
        }
        t.dims.retain(|&n| n != 1);
        Ok(())
    }
}

/// One tensor of the split: its declaration and how its bytes are made.
pub struct Planned<S> {
    pub decl: TensorDecl,
    pub make: Make<S>,
}

/// How a planned tensor's bytes are made.
pub enum Make<S> {
    /// The mapped tensor, from its sources as the plan places them.
    Tensor(Mapped<S>),
    /// The `.scale` beside the tensor `of`: its tensor scales, one a source for a stack.
    Scale { of: String },
    /// The values the map gives ([`FolderMap::constants`]); no folder tensor is a source.
    Constant(Constant),
}

/// The folder as both readers take it: its config, its tensors, the quantization plan, the
/// planned tensors, the header's keys and serving's tokenizer.
pub struct Read<M: FolderMap> {
    pub checkpoint: M,
    pub folder: Folder,
    pub plan: Plan,
    pub planned: Vec<Planned<M::Step>>,
    /// Each folder tensor the map skipped, with why.
    pub skipped: BTreeMap<String, String>,
    pub kvs: Vec<(String, Value)>,
    pub tokenizer: tokenizer::Tokenizer,
}

/// A folder's config, read.
fn read_config(dir: &Path) -> Result<Json, Error> {
    let path = dir.join("config.json");
    std::fs::read(&path)
        .map_err(|e| e.to_string())
        .and_then(|b| serde_json::from_slice(&b).map_err(|e| e.to_string()))
        .map_err(|e| Error::Refused(format!("{}: {e}", path.display())))
}

/// The folder read up to its tokenizer: what both [`read`] and [`open`] build on.
struct Folded<M: FolderMap> {
    checkpoint: M,
    folder: Folder,
    plan: Plan,
    planned: Vec<Planned<M::Step>>,
    skipped: BTreeMap<String, String>,
    /// The map's keys, not yet the tokenizer's.
    kvs: Vec<(String, Value)>,
}

fn fold<M: FolderMap>(dir: &Path, config: &Json, q8: bool, keep: Keep) -> Result<Folded<M>, Error> {
    let checkpoint = M::parse(config)?;
    let folder = Folder::open(dir)?;
    let entries: Vec<(String, Dtype, Vec<usize>)> = folder
        .names()
        .map(|n| {
            folder
                .raw(n)
                .map(|r| (n.to_string(), r.dtype, r.shape.to_vec()))
        })
        .collect::<Result<_, _>>()?;
    let Planning {
        plan,
        planned,
        skipped,
    } = plan_tensors(&checkpoint, &Config::read(dir)?, &entries, q8, keep)?;
    let kvs = checkpoint.header(
        &planned
            .iter()
            .map(|p| (p.decl.name.as_str(), p.decl.dims.as_slice()))
            .collect::<Vec<_>>(),
        &folder,
    )?;
    Ok(Folded {
        checkpoint,
        folder,
        plan,
        planned,
        skipped,
        kvs,
    })
}

/// The map's keys `kvs` with the folder's tokenizer keys added, and serving's tokenizer built from
/// them and checked against Hugging Face's ([`served_as_hf`]).
fn tokenized<M: FolderMap>(
    dir: &Path,
    checkpoint: &M,
    mut kvs: Vec<(String, Value)>,
) -> Result<(Vec<(String, Value)>, tokenizer::Tokenizer), Error> {
    let vocab = hf::vocab::keys(dir, checkpoint.vocab())?;
    kvs.extend(vocab.keys.iter().cloned());
    let tokenizer = served_as_hf(&kvs, &vocab)?;
    Ok((kvs, tokenizer))
}

/// The folder `dir` read and checked as the map `M` reads it: `config.json`, its tensors placed
/// by its quantization config and planned for the split `prepare` writes ([`plan_tensors`],
/// [`Keep::Made`], `q8` as there), the header's keys
/// (the model's and the tokenizer's), and serving's tokenizer built from them and checked against
/// Hugging Face's ([`served_as_hf`]).
pub fn read<M: FolderMap>(dir: &Path, q8: bool) -> Result<Read<M>, Error> {
    let Folded {
        checkpoint,
        folder,
        plan,
        planned,
        skipped,
        kvs,
    } = fold::<M>(dir, &read_config(dir)?, q8, Keep::Made)?;
    let (kvs, tokenizer) = tokenized(dir, &checkpoint, kvs)?;
    Ok(Read {
        checkpoint,
        folder,
        plan,
        planned,
        skipped,
        kvs,
        tokenizer,
    })
}

/// A folder as a load opens it ([`open`]).
pub struct Opened {
    /// The header `prepare` writes as its split's first shard, with every tensor of the split and
    /// the place of each ([`gguf::Place`]): the tensors the host reads where the folder stores
    /// them are readable through it, every other tensor is read through the part source.
    pub header: Split,
    /// The draft's header, borrowing the target's embedding and head ([`FolderMap::draft`]):
    /// `None` for a folder that holds no draft, and the draft's refusal for one the map does not
    /// read, which refuses the draft alone.
    pub draft: Result<Option<Split>, Error>,
    /// Every tensor of the header but the row tables and the `.scale` tensors, whose scales each
    /// part carries, and the draft's tensors when the draft reads them.
    pub source: part::Checkpoint,
    /// The row tables read where the folder stores them, by header name.
    pub rows: BTreeMap<String, StoredRows>,
    /// Each folder tensor the map skipped, with why.
    pub skipped: BTreeMap<String, String>,
    /// Serving's tokenizer, checked against Hugging Face's.
    pub tokenizer: tokenizer::Tokenizer,
}

/// The folder `dir` opened for a load, every part in its stored format (no Q8_0 rewrite), by the
/// map its config's `model_type` names: a type no map reads is refused by name. Any refusal of
/// the map, of [`plan_tensors`] or of the places its tensors ask for ([`open_as`]) is the
/// folder's.
pub fn open(dir: &Path) -> Result<Opened, Error> {
    let config = read_config(dir)?;
    (map_of(&config, &dir.display())?.open)(dir, &config)
}

/// The `general.architecture` the header of a folder whose `config.json` holds `config` names,
/// read from the config by the map [`open`] would pick ([`FolderMap::architecture`]).
pub fn architecture(config: &[u8]) -> Result<&'static str, Error> {
    let config: Json =
        serde_json::from_slice(config).map_err(|e| Error::Refused(format!("config.json: {e}")))?;
    (map_of(&config, &"config.json")?.architecture)(&config)
}

/// The architecture the map `M` reads `config` as ([`architecture`]).
pub fn architecture_as<M: FolderMap>(config: &Json) -> Result<&'static str, Error> {
    Ok(M::parse(config)?.architecture())
}

/// The row of [`crate::arch::MAPS`] that reads a folder whose config is `config`, by its
/// `model_type`; `at` names the folder in a refusal.
fn map_of(
    config: &Json,
    at: &dyn std::fmt::Display,
) -> Result<&'static crate::arch::MapRow, Error> {
    let model_type =
        model_type_of(config).ok_or_else(|| format!("{at}: config.json names no model_type"))?;
    crate::arch::MAPS
        .iter()
        .find(|row| row.model_types.contains(&model_type))
        .ok_or_else(|| {
            let known: Vec<&str> = crate::arch::MAPS
                .iter()
                .flat_map(|row| row.model_types.iter().copied())
                .collect();
            format!("{at}: model_type {model_type:?} has no folder map; the maps read {known:?}")
                .into()
        })
}

/// The model type of the folder `dir`'s `config.json`, which [`open`] picks its map by.
pub fn model_type(dir: &Path) -> Result<String, Error> {
    let config = read_config(dir)?;
    model_type_of(&config)
        .map(str::to_string)
        .ok_or_else(|| format!("{}: config.json names no model_type", dir.display()).into())
}

/// `config.json`'s model type: the top level's, else `text_config`'s.
fn model_type_of(config: &Json) -> Option<&str> {
    config
        .get("model_type")
        .or_else(|| config.get("text_config")?.get("model_type"))?
        .as_str()
}

/// The folder `dir`, whose config is `config`, opened as the map `M` reads it ([`open`]), and the
/// draft it holds ([`FolderMap::draft`]). Any refusal of [`read`] or of the places the map's
/// tensors ask for is the folder's; one of the draft is the draft's, held in [`Opened::draft`].
pub fn open_as<M: FolderMap>(dir: &Path, config: &Json) -> Result<Opened, Error> {
    let Folded {
        checkpoint,
        folder,
        plan,
        planned,
        skipped,
        kvs,
    } = fold::<M>(dir, config, false, Keep::InPlace)?;
    let (kvs, tokenizer) = tokenized(dir, &checkpoint, kvs)?;
    let Assembled {
        header,
        draft,
        source,
        rows,
    } = assemble(
        &checkpoint,
        dir,
        (folder, plan, planned),
        &kvs,
        scales::store().as_deref(),
    )?;
    Ok(Opened {
        header,
        draft,
        source,
        rows,
        skipped,
        tokenizer,
    })
}

/// The plan of an opened folder, a line each: the header's tensors and bytes by where a load reads
/// them and by type, and the folder tensors the map skipped by reason. Every map prints its plan
/// through this one function, so a new map's folder needs no printing of its own.
pub fn plan_lines(header: &Split, skipped: &BTreeMap<String, String>) -> Vec<String> {
    let mut total = (0usize, 0u64);
    let mut by_place = [
        ("read in place from the folder's files", (0usize, 0u64)),
        ("stacks of matrices read in place", (0, 0)),
        ("made at load from the folder's tensors", (0, 0)),
        ("with no place", (0, 0)),
    ];
    let mut by_type: BTreeMap<u32, (usize, u64)> = BTreeMap::new();
    let tally = |(n, bytes): &mut (usize, u64), nbytes: u64| {
        *n += 1;
        *bytes += nbytes;
    };
    for (_, t) in header.iter_tensors() {
        tally(&mut total, t.nbytes);
        tally(by_type.entry(t.ty.as_u32()).or_default(), t.nbytes);
        let place = match header.place(&t.name) {
            Some(Place::File { .. }) => 0,
            Some(Place::Stack { .. }) => 1,
            Some(Place::Made) => 2,
            None => 3,
        };
        tally(&mut by_place[place].1, t.nbytes);
    }
    let mut lines = vec![format!(
        "{}: {} tensors, {} bytes; {} folder tensors skipped",
        header.architecture().unwrap_or("no architecture"),
        total.0,
        total.1,
        skipped.len()
    )];
    for (label, (n, bytes)) in by_place.into_iter().filter(|(_, (n, _))| *n > 0) {
        lines.push(format!("{n} tensors, {bytes} bytes {label}"));
    }
    for (&id, &(n, bytes)) in &by_type {
        let name = GgmlType::from_u32(id)
            .name()
            .map_or_else(|| format!("type {id}"), str::to_string);
        lines.push(format!("{n} tensors, {bytes} bytes of {name}"));
    }
    let mut reasons: BTreeMap<&str, usize> = BTreeMap::new();
    for why in skipped.values() {
        *reasons.entry(why).or_default() += 1;
    }
    for (why, n) in reasons {
        lines.push(format!("skipped {n} folder tensors: {why}"));
    }
    lines
}

/// What [`open_as`] builds of a folder's planned tensors, before the tokenizer.
struct Assembled {
    header: Split,
    draft: Result<Option<Split>, Error>,
    source: part::Checkpoint,
    rows: BTreeMap<String, StoredRows>,
}

/// Where a planned tensor is read before its file's index is known.
enum Spot {
    Made,
    File(PathBuf, std::ops::Range<u64>),
    Stack {
        codes_in: Vec<PathBuf>,
        codes_at: Vec<u64>,
        scales_in: Vec<PathBuf>,
        scales_at: Vec<u64>,
        codes_per: u64,
        scales_per: u64,
        layout: StackLayout,
        scales: Vec<f32>,
    },
}

/// The located header, the draft, the part source and the row tables of the folder `dir`'s
/// planned tensors, its compressed block scales read back into the scale store `store`
/// ([`scales`]).
fn assemble<M: FolderMap>(
    checkpoint: &M,
    dir: &Path,
    (folder, plan, planned): (Folder, Plan, Vec<Planned<M::Step>>),
    kvs: &[(String, Value)],
    store: Option<&Path>,
) -> Result<Assembled, Error> {
    let layout = Layout::new(kvs, planned.iter().map(|p| p.decl.clone()).collect())?;
    let mut wanted = scales::Wanted::new(dir, store)?;
    let mut spots = Vec::with_capacity(planned.len());
    let mut made = BTreeMap::new();
    let mut constants = BTreeMap::new();
    let mut rows = BTreeMap::new();
    for Planned { decl, make } in planned {
        let m = match make {
            Make::Tensor(m) => m,
            Make::Constant(c) => {
                constants.insert(decl.name.clone(), c);
                spots.push(Spot::Made);
                continue;
            }
            Make::Scale { .. } => {
                spots.push(Spot::Made);
                continue;
            }
        };
        if m.role == Use::Rows {
            rows.insert(decl.name.clone(), table_rows(&folder, &plan, &decl, &m)?);
            spots.push(Spot::Made);
            continue;
        }
        spots.push(match m.origin {
            Origin::Made => Spot::Made,
            Origin::File(_) => {
                let (file, range) = folder.file_range(&m.sources[0])?;
                Spot::File(file, range)
            }
            Origin::Stack { layout, .. } => {
                stack_in_place(&folder, &plan, &decl, &m, layout, &mut wanted)?
            }
        });
        let step = checkpoint.mover(&decl.name, &m.steps);
        made.insert(
            decl.name,
            Made {
                sources: m.sources,
                role: m.role,
                dims: m.dims,
                step,
            },
        );
    }
    let pending_path = wanted.path.clone();
    // Keep the lock until the header maps this cache generation.
    let (decoded, scale_lock) = wanted.open()?.unzip();
    let decoded_path = decoded.as_ref().map(|d| d.path().to_path_buf());
    if let (Some(pending), Some(decoded)) = (pending_path, decoded_path.as_ref()) {
        for spot in &mut spots {
            if let Spot::Stack { scales_in, .. } = spot {
                for file in scales_in {
                    if *file == pending {
                        file.clone_from(decoded);
                    }
                }
            }
        }
    }
    let used: BTreeSet<&PathBuf> = spots
        .iter()
        .flat_map(|s| match s {
            Spot::Made => Vec::new(),
            Spot::File(file, _) => vec![file],
            Spot::Stack {
                codes_in,
                scales_in,
                ..
            } => codes_in.iter().chain(scales_in).collect(),
        })
        .collect();
    // The scale store follows the folder shards.
    let mut files: Vec<PathBuf> = used
        .into_iter()
        .filter(|f| decoded_path.as_ref() != Some(*f))
        .cloned()
        .collect();
    if !files.is_empty() {
        let first = first_shard(&folder)?;
        if !files.contains(&first) {
            files.push(first);
        }
        files.sort();
    }
    files.extend(decoded_path);
    let index = |file: &PathBuf| {
        files
            .iter()
            .position(|f| f == file)
            .ok_or_else(|| format!("{} is no shard", file.display()))
    };
    let indices = |runs: Vec<PathBuf>| runs.iter().map(index).collect::<Result<Vec<_>, _>>();
    let places = spots
        .into_iter()
        .map(|spot| {
            Ok(match spot {
                Spot::Made => Place::Made,
                Spot::File(file, range) => Place::File {
                    shard: index(&file)?,
                    range,
                },
                Spot::Stack {
                    codes_in,
                    codes_at,
                    scales_in,
                    scales_at,
                    codes_per,
                    scales_per,
                    layout,
                    scales,
                } => Place::Stack {
                    codes_in: indices(codes_in)?,
                    codes_at,
                    scales_in: indices(scales_in)?,
                    scales_at,
                    codes_per,
                    scales_per,
                    layout,
                    scales,
                },
            })
        })
        .collect::<Result<Vec<_>, Error>>()?;
    let path = files.first().cloned().unwrap_or_else(|| dir.to_path_buf());
    let header = Split::located(path.clone(), &layout, &files, places)?;
    drop(scale_lock);
    let draft = checkpoint
        .draft(&path, &folder, &plan, kvs)
        .and_then(|drafted| {
            let Some((draft, tensors)) = drafted else {
                return Ok(None);
            };
            let mut parts = BTreeMap::new();
            for Planned { decl, make } in tensors {
                let m = match make {
                    Make::Tensor(m) => m,
                    Make::Scale { .. } => continue,
                    Make::Constant(_) => {
                        return Err(refuse(&decl.name, "a draft's constant is not read"));
                    }
                };
                if m.role == Use::Rows {
                    return Err(refuse(&decl.name, "a draft's row table is not read"));
                }
                let step = checkpoint.mover(&decl.name, &m.steps);
                parts.insert(
                    decl.name,
                    Made {
                        sources: m.sources,
                        role: m.role,
                        dims: m.dims,
                        step,
                    },
                );
            }
            made.extend(parts);
            Ok(Some(draft))
        });
    Ok(Assembled {
        header,
        draft,
        source: part::Checkpoint::new(folder, plan, made)?
            .with_constants(constants)
            .with_scales(decoded),
        rows,
    })
}

/// The folder's first shard file by name: the one that names a located split of it.
fn first_shard(folder: &Folder) -> Result<PathBuf, Error> {
    let mut first: Option<PathBuf> = None;
    for name in folder.names() {
        let (file, _) = folder.file_range(name)?;
        if first.as_ref().is_none_or(|f| file < *f) {
            first = Some(file);
        }
    }
    first.ok_or_else(|| {
        "a folder of no tensor has no first shard"
            .to_string()
            .into()
    })
}

/// The name of the `.scale` tensor beside the tensor `name`.
fn scale_name(name: &str) -> String {
    format!("{}.scale", name.strip_suffix(".weight").unwrap_or(name))
}

/// The stack `decl`, mapped as `m`, where `folder` stores it: each expert's code plane and its
/// block-scale plane, two runs wherever the folder keeps them, each in its own shard
/// ([`StackLayout::Planes`]), with the expert's tensor scale, the factor its results are
/// multiplied by; a block scale stored compressed is the plain one `wanted` places in the scale
/// store. Every code plane has the stack's size, and every block-scale plane.
fn stack_in_place<S>(
    folder: &Folder,
    plan: &Plan,
    decl: &TensorDecl,
    m: &Mapped<S>,
    layout: StackLayout,
    wanted: &mut scales::Wanted,
) -> Result<Spot, Error> {
    let name = &decl.name;
    if layout != StackLayout::Planes {
        return Err(
            format!("{name}: a folder keeps a stack's matrices as planes, not {layout:?}").into(),
        );
    }
    let get = |n: &str| folder.raw(n);
    let (mut codes_in, mut codes_at) = (Vec::new(), Vec::new());
    let (mut scales_in, mut scales_at) = (Vec::new(), Vec::new());
    let mut scales = Vec::with_capacity(m.sources.len());
    let mut per: Option<(u64, u64)> = None;
    for (source, placed) in m.placed(plan)? {
        let (scale, scale_file, planes) = if placed.format == placed.format.decoded() {
            let (scale, block_scale) = modelopt::nvfp4_rows_scale(&get, source, placed)?;
            let (file, planes) = folder.file_range(block_scale)?;
            (scale, file, planes)
        } else {
            wanted.push(folder, source, placed)?
        };
        let (codes_file, codes) = folder.file_range(source)?;
        let sizes = (codes.end - codes.start, planes.end - planes.start);
        if let Some(of) = per.filter(|&of| of != sizes) {
            return Err(format!(
                "{source}: codes of {} bytes and block scales of {} bytes, the stack's {} and {}",
                sizes.0, sizes.1, of.0, of.1
            )
            .into());
        }
        per = Some(sizes);
        codes_in.push(codes_file);
        codes_at.push(codes.start);
        scales_in.push(scale_file);
        scales_at.push(planes.start);
        scales.push(scale);
    }
    let Some((codes_per, scales_per)) = per else {
        return Err(format!("{name}: a stack of no matrix").into());
    };
    Ok(Spot::Stack {
        codes_in,
        codes_at,
        scales_in,
        scales_at,
        codes_per,
        scales_per,
        layout,
        scales,
    })
}

/// The row table `decl`, mapped as `m`, where `folder` stores it: each part's byte range in its
/// shard. F8 values or NVFP4 codes and block scales stay in their separate
/// stored planes. Each part must have the table's width and tensor scale,
/// and their rows must add up to the table's.
fn table_rows<S>(
    folder: &Folder,
    plan: &Plan,
    decl: &TensorDecl,
    m: &Mapped<S>,
) -> Result<StoredRows, Error> {
    let name = &decl.name;
    let ty = GgmlType::from_u32(decl.type_id);
    if !matches!(ty, GgmlType::F8E4M3_ROWS | GgmlType::NVFP4_ROWS) {
        return Err(format!("{name}: {ty} is not F8 or NVFP4 rows").into());
    }
    let (width, rows) = match decl.dims[..] {
        [width, rows]
            if width > 0 && rows > 0 && size_of(decl.type_id, &decl.dims) == Some(decl.nbytes) =>
        {
            (width, rows)
        }
        _ => return Err(format!("{name}: dims {:?} are not a table's", decl.dims).into()),
    };
    let get = |n: &str| folder.raw(n);
    let mut scale: Option<f32> = None;
    let mut runs = Vec::with_capacity(m.sources.len());
    let mut scale_runs = Vec::new();
    let divisor = if ty == GgmlType::NVFP4_ROWS { 2 } else { 1 };
    let mut held = 0u64;
    for (source, placed) in m.placed(plan)? {
        let s = if ty == GgmlType::NVFP4_ROWS {
            let (s, block_scale) = modelopt::nvfp4_rows_scale(&get, source, placed)?;
            scale_runs.push(folder.file_range(block_scale)?);
            s
        } else {
            modelopt::fp8_rows_scale(&get, source, placed)?.0
        };
        if let Some(t) = scale.filter(|&t| t != s) {
            return Err(format!("{source}: tensor scale {s}, the table's {t}").into());
        }
        scale = Some(s);
        let shape = folder.raw(source)?.shape;
        let &[part_rows, columns] = shape else {
            return Err(format!("{source}: shape {shape:?} is not a table part's").into());
        };
        let part_width = (columns as u64)
            .checked_mul(divisor)
            .ok_or_else(|| format!("{source}: row width overflow"))?;
        if part_width != width {
            return Err(
                format!("{source}: rows of {part_width} values, the table's {width}").into(),
            );
        }
        held = held
            .checked_add(part_rows as u64)
            .ok_or_else(|| format!("{name}: row count overflow"))?;
        runs.push(folder.file_range(source)?);
    }
    if held != rows {
        return Err(format!("{name}: its parts hold {held} rows, the table {rows}").into());
    }
    Ok(StoredRows {
        values: StoredPlane {
            row_bytes: width / divisor,
            runs,
        },
        block_scales: (ty == GgmlType::NVFP4_ROWS).then_some(StoredPlane {
            row_bytes: width / 16,
            runs: scale_runs,
        }),
        scale: scale.ok_or_else(|| format!("{name}: a table of no parts"))?,
    })
}

/// What [`plan_tensors`] makes of a folder.
pub struct Planning<S> {
    pub plan: Plan,
    pub planned: Vec<Planned<S>>,
    /// Each folder tensor the map skipped, with why.
    pub skipped: BTreeMap<String, String>,
}

/// Whether a tensor stored as `dtype` is the bytes of type `ty`.
fn stored_as(dtype: Dtype, ty: GgmlType) -> bool {
    matches!(
        (dtype, ty),
        (Dtype::Bf16, GgmlType::BF16) | (Dtype::F32, GgmlType::F32) | (Dtype::F16, GgmlType::F16)
    )
}

/// The folder's `entries` (name, dtype, shape) placed by its `config`, and each tensor the map
/// makes of them declared in the type its placed format and role give (or, for a tensor `keep`
/// keeps in place, the type its origin names), its `.scale` beside it where the type
/// reads one. An unquantized matrix the map reads as Q8_0 stays in its stored format unless `q8`
/// asks for the rewrite; F32 matrices need that rewrite. Unquantized expert stacks are
/// unsupported. A folder tensor the map neither reads nor skips is refused by name.
pub fn plan_tensors<M: FolderMap>(
    checkpoint: &M,
    config: &Config,
    entries: &[(String, Dtype, Vec<usize>)],
    q8: bool,
    keep: Keep,
) -> Result<Planning<M::Step>, Error> {
    let plan = quant_config::plan(
        config,
        entries
            .iter()
            .map(|(n, d, s)| (n.as_str(), *d, s.as_slice())),
        |module| checkpoint.module_class(module),
    );
    let logical: Vec<(&str, Vec<usize>)> = entries
        .iter()
        .filter_map(
            |(n, _, s)| match (plan.placed.get(n), plan.refused.contains_key(n)) {
                (Some(p), _) => Some((n.as_str(), p.format.logical(s))),
                (None, true) => Some((n.as_str(), s.clone())),
                (None, false) => None,
            },
        )
        .collect();
    let Mapping {
        tensors: mapped,
        skipped,
    } = checkpoint.map(
        &logical
            .iter()
            .map(|(n, s)| (*n, s.as_slice()))
            .collect::<Vec<_>>(),
    )?;
    let read: HashSet<&str> = mapped
        .values()
        .flat_map(|m| m.sources.iter().map(String::as_str))
        .chain(skipped.keys().map(String::as_str))
        .collect();
    if let Some((name, _)) = logical.iter().find(|(n, _)| !read.contains(n)) {
        return Err(refuse(name, "is neither read nor skipped by the map"));
    }
    let stored: HashMap<&str, (Dtype, &[usize])> =
        if mapped.values().any(|m| m.origin != Origin::Made) {
            entries
                .iter()
                .map(|(n, d, s)| (n.as_str(), (*d, s.as_slice())))
                .collect()
        } else {
            HashMap::new()
        };
    let mut planned = planned_tensors(&plan, mapped, q8, &stored, keep)?;
    for (name, constant) in checkpoint.constants()? {
        let one = planned_constant(&planned, name, constant)?;
        planned.push(one);
    }
    Ok(Planning {
        plan,
        planned,
        skipped,
    })
}

/// Each tensor of `mapped` declared in the type its place in `plan` and its role give (or, for a
/// tensor kept in place, the type its origin names, its source's stored dtype and shape in
/// `stored`), its `.scale` beside it where the type reads one ([`plan_tensors`], `q8` and
/// `keep` as there). A router stored F32 keeps F32 whatever role the map reads it in.
pub fn planned_tensors<S>(
    plan: &Plan,
    mapped: BTreeMap<String, Mapped<S>>,
    q8: bool,
    stored: &HashMap<&str, (Dtype, &[usize])>,
    keep: Keep,
) -> Result<Vec<Planned<S>>, Error> {
    let mut planned = Vec::new();
    for (name, mut m) in mapped {
        let source = &m.sources[0];
        let format = modelopt::format_of(&m.placed(plan)?)?;
        m.origin = match (keep, m.origin) {
            (Keep::Made, _) => Origin::Made,
            (Keep::InPlace, Origin::Stack { ty, .. })
                if modelopt::type_of(format.decoded(), Use::Rows) != Some(ty.as_u32()) =>
            {
                Origin::Made
            }
            (Keep::InPlace, origin) => origin,
        };
        let type_id = match m.origin {
            Origin::Made => {
                if m.role == Use::Matrix(Plain::Q8_0)
                    && matches!(format, Format::Bf16 | Format::F32)
                {
                    if m.dims.len() == 3 {
                        return Err(refuse(
                            source,
                            format!(
                                "{name}: {format} expert stacks are unsupported by the converter"
                            ),
                        ));
                    }
                    if !q8 {
                        if format == Format::F32 {
                            return Err(refuse(
                                source,
                                format!(
                                    "{name} is stored {format}, a form its card site does not read; {PREPARE_Q8}=1 writes Q8_0"
                                ),
                            ));
                        }
                        m.role = Use::Matrix(Plain::Bf16);
                    }
                }
                if format == Format::F32
                    && (name.ends_with(".ffn_gate_inp.weight")
                        || name.ends_with(".ffn_gate_inp_shexp.weight"))
                {
                    m.role = Use::Matrix(Plain::F32);
                }
                modelopt::type_of(format, m.role).ok_or_else(|| {
                    refuse(
                        source,
                        format!("{format} read as {:?} has no GGUF type here", m.role),
                    )
                })?
            }
            Origin::File(ty) => kept_file(&name, &m, ty, format, stored)?,
            Origin::Stack { ty, .. } => kept_stack(&name, &m, ty)?,
        };
        let nbytes = size_of(type_id, &m.dims).ok_or_else(|| {
            format!(
                "{name}: dims {:?} are not whole blocks of type {type_id}",
                m.dims
            )
        })?;
        // One tensor scale an expert of a stack; one for a row table, whatever its parts.
        let scales = GgmlType::from_u32(type_id)
            .reads_scale()
            .then_some(match m.role {
                Use::Rows => 1,
                Use::Matrix(_) | Use::Q8_0 => m.sources.len(),
            });
        let decl = TensorDecl {
            name: name.clone(),
            dims: m.dims.clone(),
            type_id,
            nbytes,
        };
        planned.push(Planned {
            decl,
            make: Make::Tensor(m),
        });
        if let Some(n) = scales {
            let n = u64::try_from(n).map_err(|_| format!("{name}: scale count exceeds u64"))?;
            let nbytes = size_of(0, &[n])
                .ok_or_else(|| format!("{name}: scale byte count overflows u64"))?;
            let decl = TensorDecl {
                name: scale_name(&name),
                dims: vec![n],
                type_id: 0,
                nbytes,
            };
            planned.push(Planned {
                decl,
                make: Make::Scale { of: name },
            });
        }
    }
    Ok(planned)
}

/// The constant `name` declared F32 at its dims. A name another tensor of the header has, dims
/// that are empty, hold a zero or do not take as many values as the constant has, and a value
/// that is not finite are refused by name.
fn planned_constant<S>(
    taken: &[Planned<S>],
    name: String,
    constant: Constant,
) -> Result<Planned<S>, Error> {
    if taken.iter().any(|p| p.decl.name == name) {
        return Err(refuse(
            &name,
            "is a constant of the map and a tensor of the folder's",
        ));
    }
    let count = constant
        .dims
        .iter()
        .try_fold(1u64, |n, &d| n.checked_mul(d))
        .filter(|&n| n > 0 && !constant.dims.is_empty());
    if count != Some(constant.values.len() as u64) {
        return Err(refuse(
            &name,
            format!(
                "dims {:?} do not take its {} values",
                constant.dims,
                constant.values.len()
            ),
        ));
    }
    if let Some(v) = constant.values.iter().find(|v| !v.is_finite()) {
        return Err(refuse(&name, format!("the constant holds {v}")));
    }
    let type_id = GgmlType::F32.as_u32();
    let nbytes = size_of(type_id, &constant.dims)
        .ok_or_else(|| format!("{name}: dims {:?} are not whole F32 rows", constant.dims))?;
    Ok(Planned {
        decl: TensorDecl {
            name,
            dims: constant.dims.clone(),
            type_id,
            nbytes,
        },
        make: Make::Constant(constant),
    })
}

/// The type id of the tensor `name`, mapped as `m` to stay in the one file tensor as stored: its
/// bytes read as `ty`, which its stored type, its dims and its role in the part source must all
/// give.
fn kept_file<S>(
    name: &str,
    m: &Mapped<S>,
    ty: GgmlType,
    format: Format,
    stored: &HashMap<&str, (Dtype, &[usize])>,
) -> Result<u32, Error> {
    let [source] = m.sources.as_slice() else {
        return Err(refuse(name, "kept in the file, it has exactly one source"));
    };
    if m.role == Use::Rows {
        return Err(refuse(
            source,
            "a row table is not kept as one run of bytes",
        ));
    }
    if !m.steps.is_empty() {
        return Err(refuse(source, "kept in the file, it has no step"));
    }
    let &(dtype, shape) = stored
        .get(source.as_str())
        .ok_or_else(|| refuse(source, "is not in the folder"))?;
    if !stored_as(dtype, ty) {
        return Err(refuse(
            source,
            format!(
                "is stored {}, not as {ty}: only its stored bytes can stay",
                dtype.name()
            ),
        ));
    }
    if m.dims
        .iter()
        .copied()
        .ne(shape.iter().rev().map(|&d| d as u64))
    {
        return Err(refuse(
            source,
            format!(
                "{shape:?} stored, the header's dims {:?}: the order of its bytes would change",
                m.dims
            ),
        ));
    }
    let type_id = ty.as_u32();
    if modelopt::type_of(format, m.role) != Some(type_id) {
        return Err(refuse(
            source,
            format!(
                "the part source reads {format} as {:?}, not as the {ty} that stays in the file",
                m.role
            ),
        ));
    }
    Ok(type_id)
}

/// The type id of the stack `name`, mapped as `m` to stay in the folder's tensors as stored: one
/// matrix a source, `ty`.
fn kept_stack<S>(name: &str, m: &Mapped<S>, ty: GgmlType) -> Result<u32, Error> {
    if m.role == Use::Rows {
        return Err(refuse(name, "a row table is not kept as a stack"));
    }
    if !m.steps.is_empty() {
        return Err(refuse(name, "kept in place, a stack has no step"));
    }
    if m.dims.len() != 3 || m.dims[2] != m.sources.len() as u64 {
        return Err(refuse(
            name,
            format!(
                "dims {:?} are not [k, rows, {}], one matrix a source",
                m.dims,
                m.sources.len()
            ),
        ));
    }
    Ok(ty.as_u32())
}

/// Serving's tokenizer, built from the header's keys `kvs`, which must preserve HF's split,
/// rendering, special ids, automatic BOS/EOS and generation stop sets.
pub fn served_as_hf(
    kvs: &[(String, Value)],
    vocab: &hf::vocab::Keys,
) -> Result<tokenizer::Tokenizer, Error> {
    let served = tokenizer::Tokenizer::from_keys(kvs)?;
    let mut split = served.special_tokens().to_vec();
    split.sort_unstable();
    if split != vocab.added {
        return Err(format!(
            "serving splits text on tokens {split:?}, HF on its added tokens {:?}",
            vocab.added
        )
        .into());
    }
    if let Some(id) = vocab.stop.iter().find(|id| !served.eog().contains(id)) {
        return Err(format!(
            "HF's generate stops at token {id}, where serving does not end generation"
        )
        .into());
    }
    if let Some(id) = served.eog().iter().find(|id| !vocab.stop.contains(id)) {
        return Err(format!("serving stops at token {id}, where HF's generate continues").into());
    }
    let expected = |key: &str| vocab.keys.iter().find(|(k, _)| k == key).map(|(_, v)| v);
    let Some(Value::Array(types)) = expected("tokenizer.ggml.token_type") else {
        return Err("HF tokenizer has no tokenizer.ggml.token_type"
            .to_string()
            .into());
    };
    if types.len() != served.n_vocab() {
        return Err("serving vocabulary size differs from HF's"
            .to_string()
            .into());
    }
    for (id, ty) in types.iter().enumerate() {
        let id = u32::try_from(id).map_err(|_| format!("token {id} past u32"))?;
        let control = served
            .attr(id)
            .is_some_and(|a| a & (tokenizer::attr::CONTROL | tokenizer::attr::UNKNOWN) != 0);
        if control != (*ty == Value::I32(3)) {
            return Err(
                format!("serving control/rendering at token {id} differs from HF's").into(),
            );
        }
    }
    for (key, actual) in [
        ("tokenizer.ggml.add_bos_token", served.add_bos()),
        ("tokenizer.ggml.add_eos_token", served.add_eos()),
    ] {
        // An absent key is the flag off; a value that is not a bool is refused, not read as off.
        let want = match expected(key) {
            None => false,
            Some(v) => v
                .as_bool()
                .ok_or_else(|| format!("HF {key} is {v:?}, not a bool"))?,
        };
        if actual != want {
            return Err(format!("serving {key} differs from HF's").into());
        }
    }
    let specials = served.specials();
    for (key, actual) in [
        ("tokenizer.ggml.bos_token_id", specials.bos),
        ("tokenizer.ggml.eos_token_id", specials.eos),
        ("tokenizer.ggml.padding_token_id", specials.pad),
        ("tokenizer.ggml.unknown_token_id", specials.unk),
        ("tokenizer.ggml.seperator_token_id", specials.sep),
    ] {
        // A BOS id is asked only where the vocabulary adds one, which the tokenizer refuses
        // without it; a folder that names none (HF's own: no `bos_token`) keeps none.
        if actual.is_none() && key == "tokenizer.ggml.eos_token_id" {
            return Err(format!("serving requires {key}").into());
        }
        if actual.map(u64::from) != expected(key).and_then(Value::as_u64) {
            return Err(format!("serving {key} differs from HF's").into());
        }
    }
    Ok(served)
}

/// The bytes of a tensor of `type_id` and `dims`, its blocks as the type's row sizes them.
pub fn size_of(type_id: u32, dims: &[u64]) -> Option<u64> {
    let ty = GgmlType::from_u32(type_id);
    let (blck, size) = ty.blck_size().zip(ty.type_size())?;
    if dims.is_empty() || !dims[0].is_multiple_of(blck) {
        return None;
    }
    gguf::tensor_bytes(blck, size, dims)
}

#[cfg(test)]
mod tests {
    use hf::quant_config::plan;
    use serde_json::json;

    use crate::test_support::temp_dir;

    use super::*;

    /// A table in F8 parts across two shards is read where the folder stores it: each part's
    /// byte range in its shard, in the parts' order, at the one tensor scale. Parts at unequal
    /// scales, a part of another width, parts short of the table's rows and a table stored in
    /// another layout (FP4-Local's NVFP4 rows) are refused by name.
    #[test]
    fn a_table_s_rows_are_read_where_the_folder_stores_them() {
        let dir = temp_dir("folder-ple");
        let make = |nvfp4: bool, second_scale: f32, width: usize| {
            let hq = json!({"quantization": {"quant_algo": "MIXED_PRECISION",
                "quantized_layers": {"e": {"quant_algo": if nvfp4 { "NVFP4" } else { "FP8" }}}}});
            let (one, two) = (1.5f32.to_le_bytes(), second_scale.to_le_bytes());
            let divisor = if nvfp4 { 2 } else { 1 };
            let (a, b) = (
                vec![0x38u8; 2 * 160 / divisor],
                vec![0x40u8; 3 * width / divisor],
            );
            let (a_shape, b_shape) = ([2, 160 / divisor], [3, width / divisor]);
            let (sa_shape, sb_shape) = ([2, 160 / 16], [3, width / 16]);
            let (sa, sb) = (vec![0x38; 2 * 160 / 16], vec![0x40; 3 * width / 16]);
            let dtype = if nvfp4 { "U8" } else { "F8_E4M3" };
            let mut first: Vec<(&str, &str, &[usize], &[u8])> = vec![
                ("e.shard_0.weight", dtype, &a_shape, &a),
                (
                    if nvfp4 {
                        "e.shard_0.weight_scale_2"
                    } else {
                        "e.shard_0.weight_scale"
                    },
                    "F32",
                    &[1],
                    &one,
                ),
            ];
            let mut second: Vec<(&str, &str, &[usize], &[u8])> = vec![
                ("e.shard_1.weight", dtype, &b_shape, &b),
                (
                    if nvfp4 {
                        "e.shard_1.weight_scale_2"
                    } else {
                        "e.shard_1.weight_scale"
                    },
                    "F32",
                    &[1],
                    &two,
                ),
            ];
            if nvfp4 {
                first.push(("e.shard_0.weight_scale", "F8_E4M3", &sa_shape, &sa));
                second.push(("e.shard_1.weight_scale", "F8_E4M3", &sb_shape, &sb));
            }
            std::fs::write(dir.join("a.safetensors"), safetensors::write(&first)).unwrap();
            std::fs::write(dir.join("b.safetensors"), safetensors::write(&second)).unwrap();
            std::fs::write(
                dir.join(Folder::INDEX),
                json!({"weight_map": first.iter().map(|t| (t.0, "a.safetensors"))
                    .chain(second.iter().map(|t| (t.0, "b.safetensors")))
                    .collect::<BTreeMap<_, _>>()})
                .to_string(),
            )
            .unwrap();
            let folder = Folder::open(&dir).unwrap();
            let entries: Vec<(String, Dtype, Vec<usize>)> = folder
                .names()
                .map(|n| {
                    let r = folder.raw(n).unwrap();
                    (n.to_string(), r.dtype, r.shape.to_vec())
                })
                .collect();
            let p = plan(
                &Config::parse(&json!({}), Some(&hq)).unwrap(),
                entries
                    .iter()
                    .map(|(n, d, s)| (n.as_str(), *d, s.as_slice())),
                crate::arch::module_class,
            );
            (folder, p)
        };
        let m = Mapped::<()> {
            sources: vec!["e.shard_0.weight".into(), "e.shard_1.weight".into()],
            role: Use::Rows,
            steps: vec![],
            dims: vec![160, 5],
            origin: Origin::Made,
        };
        let decl = |ty: GgmlType, rows: u64| TensorDecl {
            name: "per_layer_token_embd.weight".into(),
            dims: vec![160, rows],
            type_id: ty.as_u32(),
            nbytes: size_of(ty.as_u32(), &[160, rows]).unwrap(),
        };
        let (folder, p) = make(false, 1.5, 160);
        let got = table_rows(&folder, &p, &decl(GgmlType::F8E4M3_ROWS, 5), &m).unwrap();
        assert_eq!((got.values.row_bytes, got.scale), (160, 1.5));
        assert!(got.block_scales.is_none());
        let read = |(path, range): &(PathBuf, std::ops::Range<u64>)| {
            std::fs::read(path).unwrap()[range.start as usize..range.end as usize].to_vec()
        };
        assert_eq!(read(&got.values.runs[0]), vec![0x38u8; 320]);
        assert_eq!(read(&got.values.runs[1]), vec![0x40u8; 480]);
        let refusal = |folder: &Folder, p: &Plan, ty, rows| {
            table_rows(folder, p, &decl(ty, rows), &m)
                .expect_err("refused")
                .to_string()
        };
        for dims in [vec![0, 5], vec![160, 0], vec![160, u64::MAX]] {
            let mut bad = decl(GgmlType::F8E4M3_ROWS, 5);
            bad.dims = dims;
            bad.nbytes = 0;
            let e = table_rows(&folder, &p, &bad, &m).unwrap_err().to_string();
            assert!(e.contains("not a table"), "{e}");
        }
        let e = refusal(&folder, &p, GgmlType::F8E4M3_ROWS, 6);
        assert!(e.contains("its parts hold 5 rows, the table 6"), "{e}");
        let e = refusal(&folder, &p, GgmlType::NVFP4_ROWS, 5);
        assert!(e.contains("not plain NVFP4 code and scale rows"), "{e}");
        let (folder, p) = make(false, 2.0, 160);
        let e = refusal(&folder, &p, GgmlType::F8E4M3_ROWS, 5);
        assert!(e.contains("e.shard_1.weight: tensor scale 2"), "{e}");
        let (folder, p) = make(false, 1.5, 96);
        let e = refusal(&folder, &p, GgmlType::F8E4M3_ROWS, 5);
        assert!(e.contains("rows of 96 values, the table's 160"), "{e}");
        let (folder, p) = make(true, 1.5, 160);
        let got = table_rows(&folder, &p, &decl(GgmlType::NVFP4_ROWS, 5), &m).unwrap();
        assert_eq!((got.values.row_bytes, got.scale), (80, 1.5));
        assert_eq!(read(&got.values.runs[0]), vec![0x38; 160]);
        assert_eq!(read(&got.values.runs[1]), vec![0x40; 240]);
        let scales = got.block_scales.unwrap();
        assert_eq!(scales.row_bytes, 10);
        assert_eq!(read(&scales.runs[0]), vec![0x38; 20]);
        assert_eq!(read(&scales.runs[1]), vec![0x40; 30]);
        let (folder, p) = make(true, 2.0, 160);
        let e = refusal(&folder, &p, GgmlType::NVFP4_ROWS, 5);
        assert!(e.contains("e.shard_1.weight: tensor scale 2"), "{e}");
        let (folder, p) = make(true, 1.5, 96);
        let e = refusal(&folder, &p, GgmlType::NVFP4_ROWS, 5);
        assert!(e.contains("rows of 96 values, the table's 160"), "{e}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A model of a few tensors that asks for every way the owner keeps one: an embedding and a
    /// router kept as stored, a norm made with a step, three experts kept as a stack, and a
    /// vision tensor skipped. Anything else the folder holds the map reads nor skips.
    struct Toy(VocabSpec, bool, Option<String>);

    impl FolderMap for Toy {
        type Step = ();
        const MODEL_TYPES: &'static [&'static str] = &["toy"];

        fn parse(config: &Json) -> Result<Self, Error> {
            Ok(Toy(
                VocabSpec::parse(config)?,
                config["keep"].as_bool().unwrap_or(true),
                config["constant"].as_str().map(str::to_string),
            ))
        }

        fn module_class(&self, _: &str) -> Option<ModuleClass> {
            None
        }

        fn map<'a>(&self, tensors: &[(&'a str, &'a [usize])]) -> Result<Mapping<()>, Error> {
            let mut out = Mapping {
                tensors: BTreeMap::new(),
                skipped: BTreeMap::new(),
            };
            let mut experts = BTreeMap::new();
            // A model that keeps nothing in place makes every tensor.
            let kept = |origin| if self.1 { origin } else { Origin::Made };
            for &(name, shape) in tensors {
                let one = |role, origin, steps| Mapped {
                    sources: vec![name.to_string()],
                    role,
                    steps,
                    dims: shape.iter().rev().map(|&d| d as u64).collect(),
                    origin,
                };
                match name {
                    "emb.weight" => out.tensors.insert(
                        "token_embd.weight".into(),
                        one(
                            Use::Matrix(Plain::Bf16),
                            kept(Origin::File(GgmlType::BF16)),
                            vec![],
                        ),
                    ),
                    "gate.weight" => out.tensors.insert(
                        "ffn_gate_inp.weight".into(),
                        one(
                            Use::Matrix(Plain::F32),
                            kept(Origin::File(GgmlType::F32)),
                            vec![],
                        ),
                    ),
                    "norm.weight" => out.tensors.insert(
                        "output_norm.weight".into(),
                        one(Use::Matrix(Plain::F32), Origin::Made, vec![()]),
                    ),
                    "vision.weight" => out
                        .skipped
                        .insert(name.into(), "vision".into())
                        .map(|_| one(Use::Matrix(Plain::F32), Origin::Made, vec![])),
                    _ => {
                        if let Some(id) = name
                            .strip_prefix("ex.")
                            .and_then(|r| r.split('.').next()?.parse::<usize>().ok())
                        {
                            experts.insert(id, (name, shape));
                        }
                        None
                    }
                };
            }
            if !experts.is_empty() {
                let (sources, dims) = safetensors::expert_stack(&experts, 3)?;
                out.tensors.insert(
                    "exps.weight".into(),
                    Mapped {
                        sources,
                        role: Use::Matrix(Plain::Q8_0),
                        steps: vec![],
                        dims: dims.into_iter().map(|d| d as u64).collect(),
                        origin: kept(Origin::Stack {
                            ty: GgmlType::NVFP4_ROWS,
                            layout: StackLayout::Planes,
                        }),
                    },
                );
            }
            Ok(out)
        }

        fn mover(&self, _: &str, steps: &[()]) -> Option<part::Step> {
            (!steps.is_empty()).then(|| Box::new(|_: &mut Normal| Ok(())) as part::Step)
        }

        fn architecture(&self) -> &'static str {
            "toy"
        }

        fn header(&self, _: &[(&str, &[u64])], _: &Folder) -> Result<Vec<(String, Value)>, Error> {
            Ok(vec![(
                gguf::GENERAL_ARCHITECTURE.into(),
                Value::String(self.architecture().into()),
            )])
        }

        fn vocab(&self) -> &VocabSpec {
            &self.0
        }

        fn constants(&self) -> Result<BTreeMap<String, Constant>, Error> {
            Ok(self
                .2
                .iter()
                .map(|name| {
                    let values = vec![1.0, 1e30, -2.0, 0.5];
                    (
                        name.clone(),
                        Constant {
                            dims: vec![2, 2],
                            values,
                        },
                    )
                })
                .collect())
        }
    }

    /// How the toy folder stores its three NVFP4 experts.
    #[derive(Clone, Copy, PartialEq)]
    enum Experts {
        /// Each expert's codes then its block scales, then its tensor scale, the experts out of
        /// order, all in one file.
        Planes,
        /// Every expert's codes, then every block scale, then every tensor scale, in expert
        /// order, all in one file.
        Apart,
        /// As [`Experts::Planes`], the last expert in another file.
        TwoFiles,
        /// Every expert's codes in one file, every block scale and tensor scale in another.
        ScalesElsewhere,
        /// As [`Experts::Planes`], each expert 16 rows whose block scales are stored as `csf`
        /// ([`csf_scales`]).
        Csf,
        /// As [`Experts::Planes`], the last expert's codes one row short.
        Uneven,
    }

    fn codes(e: usize) -> Vec<u8> {
        (0..128)
            .map(|j| (0x10 * (e as u8 + 1)).wrapping_add(j))
            .collect()
    }

    fn block_scales(e: usize) -> Vec<u8> {
        (0..16).map(|j| 0x30 + 4 * e as u8 + j).collect()
    }

    /// Expert `e`'s plain block scales under [`Experts::Csf`]: 16 rows of 4 groups, each a byte a
    /// row's base and a 4-bit code give, but one the `csf` stream stores apart.
    fn csf_scales(e: usize) -> Vec<u8> {
        let mut s: Vec<u8> = (0..64)
            .map(|i| 0x30 + ((i / 4 + i % 4 + e) % 8) as u8)
            .collect();
        s[21] = 0x7e;
        s
    }

    /// [`csf_scales`] stored as `csf`: the fixed slab (16 bases, then each row's codes two a
    /// byte, low nibble first) and the exception word of the byte stored apart.
    fn csf_parts(e: usize) -> (Vec<u8>, Vec<u8>) {
        let s = csf_scales(e);
        let mut fixed = vec![0x30u8; 16];
        fixed.extend(
            s.chunks(2)
                .map(|p| (p[0] - 0x30) & 15 | ((p[1] - 0x30) & 15) << 4),
        );
        (fixed, (21u32 | 0x7e << 24).to_le_bytes().to_vec())
    }

    const TENSOR_SCALES: [f32; 3] = [1.5, 2.5, 3.5];

    /// The toy folder `dir` with `extra` tensors (name, dtype, shape, bytes) in file `a`: `a`
    /// holds the embedding, the norm and the extras, `b` the experts (the router with them
    /// under `Experts::Apart`).
    fn toy_folder(
        dir: &Path,
        experts: Experts,
        extra: &[(&str, &'static str, Vec<usize>, Vec<u8>)],
    ) {
        type Tensor = (String, &'static str, Vec<usize>, Vec<u8>);
        let emb: Vec<u8> = (0..64).collect();
        let gate: Vec<u8> = (0..64).map(|i| 0x80 + i).collect();
        let norm: Vec<u8> = (0..16).map(|i| 0x40 + i).collect();
        let expert = |e: usize, rows: usize| -> Vec<Tensor> {
            if experts == Experts::Csf {
                let (fixed, exceptions) = csf_parts(e);
                let codes: Vec<u8> = (0..4).flat_map(|_| codes(e)).collect();
                return vec![
                    (format!("ex.{e}.weight"), "U8", vec![16, 32], codes),
                    (
                        format!("ex.{e}.weight_scale.nvfp4_csf_fixed"),
                        "U8",
                        vec![1, 48],
                        fixed,
                    ),
                    (
                        format!("ex.{e}.weight_scale.nvfp4_csf_exceptions"),
                        "U32",
                        vec![1],
                        exceptions,
                    ),
                    (
                        format!("ex.{e}.weight_scale_2"),
                        "F32",
                        vec![1],
                        TENSOR_SCALES[e].to_le_bytes().to_vec(),
                    ),
                ];
            }
            vec![
                (
                    format!("ex.{e}.weight"),
                    "U8",
                    vec![rows, 32],
                    codes(e)[..rows * 32].to_vec(),
                ),
                (
                    format!("ex.{e}.weight_scale"),
                    "F8_E4M3",
                    vec![rows, 4],
                    block_scales(e)[..rows * 4].to_vec(),
                ),
                (
                    format!("ex.{e}.weight_scale_2"),
                    "F32",
                    vec![1],
                    TENSOR_SCALES[e].to_le_bytes().to_vec(),
                ),
            ]
        };
        let mut a: Vec<Tensor> = vec![
            ("emb.weight".into(), "BF16", vec![4, 8], emb),
            ("norm.weight".into(), "BF16", vec![8], norm),
        ];
        a.extend(
            extra
                .iter()
                .map(|(n, d, s, b)| (n.to_string(), *d, s.clone(), b.clone())),
        );
        let mut b: Vec<Tensor> = vec![("gate.weight".into(), "F32", vec![2, 8], gate)];
        let mut c: Vec<Tensor> = vec![];
        match experts {
            Experts::Planes | Experts::Uneven | Experts::Csf => {
                let last = if experts == Experts::Uneven { 3 } else { 4 };
                b.extend(expert(2, last));
                b.extend(expert(0, 4));
                b.extend(expert(1, 4));
            }
            Experts::TwoFiles => {
                b.extend(expert(0, 4));
                b.extend(expert(1, 4));
                c.extend(expert(2, 4));
            }
            Experts::ScalesElsewhere => {
                let all: Vec<Vec<Tensor>> = (0..3).map(|e| expert(e, 4)).collect();
                b.extend(all.iter().map(|t| t[0].clone()));
                c.extend(all.iter().flat_map(|t| t[1..].to_vec()));
            }
            Experts::Apart => {
                let all: Vec<Vec<Tensor>> = (0..3).map(|e| expert(e, 4)).collect();
                b.extend(all.iter().map(|t| t[0].clone()));
                b.extend(all.iter().map(|t| t[1].clone()));
                b.extend(all.iter().map(|t| t[2].clone()));
            }
        }
        let mut map = BTreeMap::new();
        for (file, tensors) in [("a", &a), ("b", &b), ("c", &c)] {
            if tensors.is_empty() {
                continue;
            }
            let view: Vec<(&str, &str, &[usize], &[u8])> = tensors
                .iter()
                .map(|(n, d, s, bytes)| (n.as_str(), *d, s.as_slice(), bytes.as_slice()))
                .collect();
            std::fs::write(
                dir.join(format!("{file}.safetensors")),
                safetensors::write(&view),
            )
            .unwrap();
            map.extend(
                tensors
                    .iter()
                    .map(|t| (t.0.clone(), format!("{file}.safetensors"))),
            );
        }
        std::fs::write(
            dir.join(Folder::INDEX),
            json!({"weight_map": map}).to_string(),
        )
        .unwrap();
        std::fs::write(
            dir.join("config.json"),
            json!({"model_type": "toy", "vocab_size": 16}).to_string(),
        )
        .unwrap();
        let mut ex = json!({"quant_algo": "NVFP4"});
        if experts == Experts::Csf {
            ex["weight_scale_encoding"] = "csf".into();
        }
        let hq = json!({"quantization": {"quant_algo": "MIXED_PRECISION",
            "quantized_layers": {"ex": ex}}});
        std::fs::write(dir.join("hf_quant_config.json"), hq.to_string()).unwrap();
    }

    fn toy_assembled(dir: &Path) -> Result<Assembled, Error> {
        let config = read_config(dir)?;
        let Folded {
            checkpoint,
            folder,
            plan,
            planned,
            kvs,
            ..
        } = fold::<Toy>(dir, &config, false, Keep::InPlace)?;
        let store = dir.join("scales");
        assemble(
            &checkpoint,
            dir,
            (folder, plan, planned),
            &kvs,
            Some(&store),
        )
    }

    /// The toy folder opens as a located split over its files: the embedding and the router are
    /// the files' bytes where they lie, the stack's matrices are read at their places with the
    /// experts' tensor scales beside them (each matrix's block scales directly after its codes
    /// when the folder stores them so, a plane apart when it keeps every code plane before every
    /// scale plane, in the file that holds each plane when a matrix or its block scales lie in
    /// another), the norm and every `.scale` are made, and the part source holds every tensor.
    /// FAIL-first: the stack's runs all named in its first matrix's file read another file's
    /// bytes, or refuse the stack.
    #[test]
    fn a_folder_opens_as_a_located_split() {
        for (experts, codes_shards, scales_shards) in [
            (Experts::Planes, [1, 1, 1], [1, 1, 1]),
            (Experts::Apart, [1, 1, 1], [1, 1, 1]),
            (Experts::TwoFiles, [1, 1, 2], [1, 1, 2]),
            (Experts::ScalesElsewhere, [1, 1, 1], [2, 2, 2]),
        ] {
            let dir = temp_dir("folder-located");
            toy_folder(&dir, experts, &[]);
            let a = toy_assembled(&dir).unwrap();
            let split = &a.header;
            let files = 1 + codes_shards.iter().chain(&scales_shards).max().unwrap();
            assert_eq!(split.shard_count(), files);
            for (i, f) in ["a", "b", "c"][..files].iter().enumerate() {
                assert_eq!(
                    split.shard_path(i).unwrap(),
                    dir.join(format!("{f}.safetensors"))
                );
            }
            assert_eq!(split.architecture(), Some("toy"));
            assert!(!split.is_header_only());
            let emb: Vec<u8> = (0..64).collect();
            let gate: Vec<u8> = (0..64).map(|i| 0x80 + i).collect();
            assert_eq!(split.data("token_embd.weight").unwrap(), emb);
            assert_eq!(split.data("ffn_gate_inp.weight").unwrap(), gate);
            let (_, info) = split.find("token_embd.weight").unwrap();
            assert_eq!((info.ty, info.dims.clone()), (GgmlType::BF16, vec![8, 4]));
            for e in 0..3 {
                let m = split.matrix("exps.weight", e).unwrap();
                assert_eq!(m.codes, codes(e), "expert {e}");
                assert_eq!(m.block_scales, block_scales(e), "expert {e}");
            }
            let Some(Place::Stack {
                codes_in,
                codes_at,
                scales_in,
                scales_at,
                codes_per,
                scales_per,
                layout,
                scales,
            }) = split.place("exps.weight")
            else {
                panic!("the stack is a stack");
            };
            assert_eq!(
                (*codes_per, *scales_per, *layout),
                (128, 16, StackLayout::Planes)
            );
            assert_eq!(scales, &TENSOR_SCALES);
            assert_eq!(
                (&codes_in[..], &scales_in[..]),
                (&codes_shards[..], &scales_shards[..]),
                "each run's shard"
            );
            let next_to_codes = |e: usize| scales_at[e] == codes_at[e] + 128;
            match experts {
                Experts::TwoFiles | Experts::ScalesElsewhere => {}
                Experts::Planes => {
                    assert!((0..3).all(next_to_codes), "{codes_at:?} {scales_at:?}");
                    assert!(
                        codes_at[2] < codes_at[0] && codes_at[0] < codes_at[1],
                        "the experts lie out of order: {codes_at:?}"
                    );
                }
                _ => {
                    assert!(!(0..3).any(next_to_codes), "{codes_at:?} {scales_at:?}");
                    assert!(
                        codes_at[0] < codes_at[1]
                            && codes_at[1] < codes_at[2]
                            && codes_at[2] < scales_at[0]
                            && scales_at[0] < scales_at[1]
                            && scales_at[1] < scales_at[2],
                        "every code plane lies before every scale plane: {codes_at:?} {scales_at:?}"
                    );
                }
            }
            let (_, info) = split.find("exps.weight").unwrap();
            assert_eq!(
                (info.ty, info.dims.clone(), info.nbytes),
                (GgmlType::NVFP4_ROWS, vec![64, 4, 3], 3 * 144)
            );
            for name in ["output_norm.weight", "exps.scale"] {
                assert_eq!(split.place(name), Some(&Place::Made), "{name}");
                let e = split.data(name).unwrap_err().to_string();
                assert!(e.contains(name) && e.contains("made"), "{e}");
            }
            let e = split.data("exps.weight").unwrap_err().to_string();
            assert!(e.contains("stack"), "{e}");
            let made: Vec<&str> = a.source.made().keys().map(String::as_str).collect();
            assert_eq!(
                made,
                [
                    "exps.weight",
                    "ffn_gate_inp.weight",
                    "output_norm.weight",
                    "token_embd.weight"
                ]
            );
            assert!(a.rows.is_empty());
            std::fs::remove_dir_all(&dir).unwrap();
        }
    }

    /// A stack whose block scales the folder stores as `csf` opens in place as a plain one: each
    /// expert's codes where the folder stores them, its plain block scale at its place in the scale
    /// store's file, which follows the folder's files. The open writes no scale: the file holds
    /// zeros until the part source makes them (`make_in_place`), then each expert's plain scale.
    /// A later open keeps a written file as it is (a byte changed in it stays), also through an
    /// alias of the folder's path; one after a folder file changed opens another generation,
    /// and an open before that one's scales are written keeps its file under the key's pending
    /// mark. Older headers reopen their own generation, whether already filled or still pending.
    /// FAIL-first: the stack was made, so the header held no matrix of it; replacing a generation
    /// at the same path makes an older header reopen the new one's unwritten scales.
    #[test]
    fn a_folder_reads_compressed_block_scales_back_once() {
        use model::part::Source;

        let dir = temp_dir("folder-csf");
        toy_folder(&dir, Experts::Csf, &[]);
        let a = toy_assembled(&dir).unwrap();
        let split = &a.header;
        assert_eq!(split.shard_count(), 3);
        let file = split.shard_path(2).unwrap().to_path_buf();
        assert_eq!(file.parent(), Some(dir.join("scales").as_path()));
        let Some(Place::Stack {
            codes_in,
            scales_in,
            scales_at,
            scales_per,
            scales,
            ..
        }) = split.place("exps.weight")
        else {
            panic!("the stack is kept in place");
        };
        assert_eq!((&codes_in[..], &scales_in[..]), (&[1; 3][..], &[2; 3][..]));
        assert_eq!((*scales_per, &scales[..]), (64, &TENSOR_SCALES[..]));
        let matrix = |split: &Split, e: usize| {
            let m = split.matrix("exps.weight", e).unwrap();
            (m.codes.to_vec(), m.block_scales.to_vec())
        };
        for e in 0..3 {
            let codes: Vec<u8> = (0..4).flat_map(|_| codes(e)).collect();
            assert_eq!(matrix(split, e), (codes, vec![0; 64]), "expert {e}");
        }
        a.source.make_in_place().unwrap();
        for e in 0..3 {
            assert_eq!(matrix(split, e).1, csf_scales(e), "expert {e}");
        }
        let alias = dir.with_extension("alias");
        std::os::unix::fs::symlink(&dir, &alias).unwrap();
        let through = toy_assembled(&alias).unwrap();
        assert_eq!(
            through
                .header
                .shard_path(2)
                .unwrap()
                .canonicalize()
                .unwrap(),
            file.canonicalize().unwrap()
        );
        assert_eq!(
            matrix(&through.header, 0).1,
            csf_scales(0),
            "an alias of the folder keeps its file"
        );
        std::fs::remove_file(&alias).unwrap();
        let poke = std::fs::OpenOptions::new().write(true).open(&file).unwrap();
        std::os::unix::fs::FileExt::write_all_at(&poke, &[0x55], scales_at[0]).unwrap();
        let b = toy_assembled(&dir).unwrap();
        assert_eq!(b.header.shard_path(2).unwrap(), file);
        b.source.make_in_place().unwrap();
        assert_eq!(matrix(&b.header, 0).1[0], 0x55, "the written file is kept");
        let later = std::time::SystemTime::now() + std::time::Duration::from_secs(60);
        std::fs::File::options()
            .write(true)
            .open(dir.join("b.safetensors"))
            .unwrap()
            .set_modified(later)
            .unwrap();
        let c = toy_assembled(&dir).unwrap();
        let old_bytes = std::fs::read(split.shard_path(2).unwrap()).unwrap();
        let at = usize::try_from(scales_at[0]).unwrap();
        assert_eq!(
            &old_bytes[at..at + 64],
            matrix(split, 0).1.as_slice(),
            "an older header reopens its mapped generation"
        );
        assert_ne!(c.header.shard_path(2).unwrap(), file);
        assert_eq!(
            matrix(&c.header, 0).1,
            vec![0; 64],
            "a new generation, not yet made"
        );
        let d = toy_assembled(&dir).unwrap();
        assert_eq!(c.header.shard_path(2), d.header.shard_path(2));
        std::fs::File::options()
            .write(true)
            .open(dir.join("b.safetensors"))
            .unwrap()
            .set_modified(later + std::time::Duration::from_secs(60))
            .unwrap();
        let e = toy_assembled(&dir).unwrap();
        c.source.make_in_place().unwrap();
        assert_eq!(matrix(&c.header, 0).1, csf_scales(0));
        assert_eq!(
            matrix(&d.header, 0).1,
            csf_scales(0),
            "an unwritten file under the key's pending mark is kept"
        );
        let pending_bytes = std::fs::read(c.header.shard_path(2).unwrap()).unwrap();
        assert_eq!(
            &pending_bytes[at..at + 64],
            csf_scales(0),
            "an older pending generation is filled at its own path"
        );
        assert_eq!(matrix(&e.header, 0).1, vec![0; 64]);
        e.source.make_in_place().unwrap();
        assert_eq!(matrix(&e.header, 0).1, csf_scales(0));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A constant of the map is a tensor of the header that no folder tensor holds: declared F32
    /// at its dims, made (no run of any file), held by the part source beside the folder's
    /// tensors, and read by it whole or cut along a dim as its values are. One that takes a name
    /// a folder tensor's program name has, and one whose dims do not take its values or that has a
    /// value that is not finite, are refused by name. Mutants (each red at its line): the
    /// constants not handed to the part source (`constants()` is empty), a cut read whole (the
    /// bytes of the part), the collision check dropped (`is a constant of the map`).
    #[test]
    fn a_constant_is_a_made_tensor_no_folder_tensor_holds() {
        use model::part::{Source, Take};
        let dir = temp_dir("folder-constant");
        toy_folder(&dir, Experts::Planes, &[]);
        let config = |name: &str| {
            std::fs::write(
                dir.join("config.json"),
                json!({"model_type": "toy", "vocab_size": 16, "constant": name}).to_string(),
            )
            .unwrap();
            read_config(&dir).unwrap()
        };
        let folded = |name: &str| fold::<Toy>(&dir, &config(name), false, Keep::InPlace);
        let Folded {
            checkpoint,
            folder,
            plan,
            planned,
            kvs,
            ..
        } = folded("rope.weight").unwrap();
        let constants: BTreeMap<String, Constant> = planned
            .iter()
            .filter_map(|p| match &p.make {
                Make::Constant(c) => Some((p.decl.name.clone(), c.clone())),
                _ => None,
            })
            .collect();
        assert_eq!(
            constants.keys().map(String::as_str).collect::<Vec<_>>(),
            ["rope.weight"]
        );
        let a = assemble(&checkpoint, &dir, (folder, plan, planned), &kvs, None).unwrap();
        let (_, info) = a.header.find("rope.weight").unwrap();
        assert_eq!(
            (info.ty, info.dims.clone(), info.nbytes),
            (GgmlType::F32, vec![2, 2], 16)
        );
        assert_eq!(a.header.place("rope.weight"), Some(&Place::Made));
        assert_eq!(
            a.source.made().len(),
            4,
            "a constant is no tensor of the folder"
        );
        assert_eq!(a.source.constants(), &constants);

        let Folded { folder, plan, .. } = folded("rope.weight").unwrap();
        let source = part::Checkpoint::new(folder, plan, BTreeMap::new())
            .unwrap()
            .with_constants(constants);
        let bytes: Vec<u8> = [1.0f32, 1e30, -2.0, 0.5]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let whole = source.plan(|_, _, _| Ok::<_, String>(Take::Whole)).unwrap();
        let [p] = &whole[..] else {
            panic!("one tensor: {}", whole.len());
        };
        assert_eq!(source.read(p).unwrap().spans().concat(), bytes);
        assert_eq!(source.read_bytes(p).unwrap(), 32);
        source.release(p).unwrap();
        // The second row of the two by two, cut along its second dim.
        let cut = source
            .plan(|_, _, _| Ok::<_, String>(Take::Part(1, 1..2)))
            .unwrap();
        let [p] = &cut[..] else {
            panic!("one part: {}", cut.len());
        };
        assert_eq!(p.info.dims, [2, 1]);
        assert_eq!(
            source.read(p).unwrap().spans().concat(),
            bytes[8..].to_vec()
        );

        let e = folded("output_norm.weight").err().unwrap().to_string();
        assert!(
            e.contains("output_norm.weight") && e.contains("is a constant of the map"),
            "{e}"
        );
        let planned_of = |dims: Vec<u64>, values: Vec<f32>| {
            planned_constant::<()>(&[], "c.weight".into(), Constant { dims, values })
        };
        for (what, dims, values) in [
            ("dims of no value", vec![], vec![1.0]),
            ("a zero dim", vec![0, 2], vec![]),
            ("too few values", vec![2, 2], vec![1.0; 3]),
            ("too many values", vec![2], vec![1.0; 3]),
            ("a NaN", vec![2], vec![1.0, f32::NAN]),
            ("an infinity", vec![2], vec![f32::INFINITY, 1.0]),
        ] {
            let e = planned_of(dims, values)
                .err()
                .unwrap_or_else(|| panic!("{what} is refused"))
                .to_string();
            assert!(e.contains("c.weight"), "{what}: {e}");
        }
        assert!(planned_of(vec![2, 2], vec![1.0; 4]).is_ok());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A map that keeps no tensor in place gets the header alone, named by the folder, as a header
    /// of no file is: one shard, no tensor readable, and the part source holds every tensor.
    #[test]
    fn a_map_that_keeps_nothing_gets_the_header_alone() {
        let dir = temp_dir("folder-made");
        toy_folder(&dir, Experts::Planes, &[]);
        std::fs::write(
            dir.join("config.json"),
            json!({"model_type": "toy", "vocab_size": 16, "keep": false}).to_string(),
        )
        .unwrap();
        let a = toy_assembled(&dir).unwrap();
        let split = &a.header;
        assert_eq!(split.shard_count(), 1);
        assert_eq!(split.shard_path(0).unwrap(), dir);
        assert!(split.is_header_only());
        for name in ["token_embd.weight", "exps.weight", "exps.scale"] {
            assert_eq!(split.place(name), Some(&Place::Made), "{name}");
            assert!(split.data(name).is_err(), "{name}");
        }
        assert_eq!(a.source.made().len(), 4);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A stack whose planes differ in size, a stack of rows, and a tensor the map neither reads
    /// nor skips are refused by name.
    // PIN(2026-10-31): a stack across files opens, each run in its own shard
    // (`a_folder_opens_as_a_located_split`).
    #[test]
    fn what_the_folder_does_not_store_as_the_map_says_is_refused_by_name() {
        let refusal = |experts, extra: &[(&str, &'static str, Vec<usize>, Vec<u8>)]| {
            let dir = temp_dir("folder-refuses");
            toy_folder(&dir, experts, extra);
            let e = match toy_assembled(&dir) {
                Ok(_) => panic!("the folder must be refused"),
                Err(e) => e.to_string(),
            };
            std::fs::remove_dir_all(&dir).unwrap();
            e
        };
        let dir = temp_dir("folder-uneven");
        toy_folder(&dir, Experts::Uneven, &[]);
        let folder = Folder::open(&dir).unwrap();
        let entries: Vec<(String, Dtype, Vec<usize>)> = folder
            .names()
            .map(|n| {
                let r = folder.raw(n).unwrap();
                (n.to_string(), r.dtype, r.shape.to_vec())
            })
            .collect();
        let p = plan(
            &Config::read(&dir).unwrap(),
            entries
                .iter()
                .map(|(n, d, s)| (n.as_str(), *d, s.as_slice())),
            |_| None,
        );
        let sources: Vec<String> = (0..3).map(|e| format!("ex.{e}.weight")).collect();
        let m = Mapped::<()> {
            sources,
            role: Use::Matrix(Plain::Q8_0),
            steps: vec![],
            dims: vec![64, 4, 3],
            origin: Origin::Made,
        };
        let decl = TensorDecl {
            name: "exps.weight".into(),
            dims: vec![64, 4, 3],
            type_id: GgmlType::NVFP4_ROWS.as_u32(),
            nbytes: 3 * 144,
        };
        let mut wanted = scales::Wanted::new(&dir, None).unwrap();
        let e = stack_in_place(&folder, &p, &decl, &m, StackLayout::Planes, &mut wanted)
            .err()
            .unwrap()
            .to_string();
        assert!(
            e.contains(
                "ex.2.weight: codes of 96 bytes and block scales of 12 bytes, the stack's 128 and 16"
            ),
            "{e}"
        );
        let e = stack_in_place(&folder, &p, &decl, &m, StackLayout::Rows, &mut wanted)
            .err()
            .unwrap()
            .to_string();
        assert!(e.contains("keeps a stack's matrices as planes"), "{e}");
        std::fs::remove_dir_all(&dir).unwrap();
        let e = refusal(
            Experts::Planes,
            &[("stray.weight", "F32", vec![2], vec![0; 8])],
        );
        assert!(
            e.contains("stray.weight: is neither read nor skipped by the map"),
            "{e}"
        );
        let vision = ("vision.weight", "F32", vec![2], vec![0; 8]);
        let dir = temp_dir("folder-skips");
        toy_folder(&dir, Experts::Planes, &[vision]);
        let planning = {
            let config = read_config(&dir).unwrap();
            let Folded { skipped, .. } = fold::<Toy>(&dir, &config, false, Keep::InPlace).unwrap();
            skipped
        };
        assert_eq!(
            planning.iter().collect::<Vec<_>>(),
            [(&"vision.weight".to_string(), &"vision".to_string())]
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A tensor kept as stored must be stored in the type it is declared as, in the order its
    /// dims give, with no step and no other source; a stack must be one matrix a source.
    #[test]
    fn a_kept_tensor_must_stay_as_stored() {
        let one = |sources: &[&str], dims: &[u64], steps: Vec<()>, role| Mapped {
            sources: sources.iter().map(ToString::to_string).collect(),
            role,
            steps,
            dims: dims.to_vec(),
            origin: Origin::Made,
        };
        let shape: &[usize] = &[4, 8];
        let stored: HashMap<&str, (Dtype, &[usize])> =
            HashMap::from([("w", (Dtype::F32, shape)), ("v", (Dtype::Bf16, shape))]);
        let f32s = Use::Matrix(Plain::F32);
        let kept = |m: &Mapped<()>, ty, format| kept_file("x", m, ty, format, &stored);
        assert_eq!(
            kept(
                &one(&["w"], &[8, 4], vec![], f32s),
                GgmlType::F32,
                Format::F32
            )
            .unwrap(),
            GgmlType::F32.as_u32()
        );
        for (m, ty, format, want) in [
            (
                one(&["w", "v"], &[8, 4], vec![], f32s),
                GgmlType::F32,
                Format::F32,
                "x: kept in the file, it has exactly one source",
            ),
            (
                one(&["w"], &[8, 4], vec![()], f32s),
                GgmlType::F32,
                Format::F32,
                "w: kept in the file, it has no step",
            ),
            (
                one(&["w"], &[8, 4], vec![], Use::Rows),
                GgmlType::F32,
                Format::F32,
                "w: a row table is not kept as one run of bytes",
            ),
            (
                one(&["w"], &[8, 4], vec![], f32s),
                GgmlType::BF16,
                Format::F32,
                "w: is stored F32, not as bf16",
            ),
            (
                one(&["w"], &[4, 8], vec![], f32s),
                GgmlType::F32,
                Format::F32,
                "the order of its bytes would change",
            ),
            (
                one(&["v"], &[8, 4], vec![], f32s),
                GgmlType::BF16,
                Format::Bf16,
                "the part source reads BF16 as",
            ),
            (
                one(&["absent"], &[8, 4], vec![], f32s),
                GgmlType::F32,
                Format::F32,
                "absent: is not in the folder",
            ),
        ] {
            let e = kept(&m, ty, format).unwrap_err().to_string();
            assert!(e.contains(want), "{want}: {e}");
        }
        let stack = |sources: &[&str], dims: &[u64], steps| {
            kept_stack(
                "s",
                &one(sources, dims, steps, Use::Matrix(Plain::Q8_0)),
                GgmlType::NVFP4_ROWS,
            )
        };
        assert!(stack(&["a", "b"], &[64, 4, 2], vec![]).is_ok());
        for (r, want) in [
            (
                stack(&["a", "b"], &[64, 4, 3], vec![]),
                "[k, rows, 2], one matrix a source",
            ),
            (
                stack(&["a", "b"], &[64, 4], vec![]),
                "[k, rows, 2], one matrix a source",
            ),
            (
                stack(&["a", "b"], &[64, 4, 2], vec![()]),
                "a stack has no step",
            ),
        ] {
            let e = r.unwrap_err().to_string();
            assert!(e.contains(want), "{want}: {e}");
        }
    }

    /// A folder's config names its map by `model_type`, the top level's or `text_config`'s; one
    /// no map reads, or none, is refused by name.
    #[test]
    fn a_model_type_no_map_reads_is_refused_by_name() {
        let dir = temp_dir("folder-types");
        let refusal = |config: Json| {
            std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
            open(&dir).err().expect("refused").to_string()
        };
        let e = refusal(json!({"model_type": "nope"}));
        let read: Vec<&str> = crate::arch::MAPS
            .iter()
            .flat_map(|row| row.model_types.iter().copied())
            .collect();
        assert!(
            e.contains(&format!(
                "model_type \"nope\" has no folder map; the maps read {read:?}"
            )),
            "{e}"
        );
        let e = refusal(json!({"text_config": {"model_type": "nope2"}}));
        assert!(e.contains("\"nope2\""), "{e}");
        let e = refusal(json!({"architectures": []}));
        assert!(e.contains("config.json names no model_type"), "{e}");
        assert_eq!(
            model_type_of(&json!({"model_type": "a", "text_config": {"model_type": "b"}})),
            Some("a")
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// The plan of a folder lists its header's tensors and bytes by where a load reads them and
    /// by type, and the folder tensors the map skipped by reason: the toy folder's embedding and
    /// router are read in place, its experts one stack, the norm and the experts' scales made.
    #[test]
    fn the_plan_of_a_folder_lists_where_each_tensor_is_read_and_what_is_skipped() {
        let dir = temp_dir("folder-plan");
        toy_folder(
            &dir,
            Experts::Planes,
            &[("vision.weight", "F32", vec![2], vec![0; 8])],
        );
        let config = read_config(&dir).unwrap();
        let Folded {
            checkpoint,
            folder,
            plan,
            planned,
            skipped,
            kvs,
        } = fold::<Toy>(&dir, &config, false, Keep::InPlace).unwrap();
        let a = assemble(&checkpoint, &dir, (folder, plan, planned), &kvs, None).unwrap();
        let name = |ty: GgmlType| ty.name().unwrap();
        // Header: token_embd BF16 64 B and ffn_gate_inp F32 64 B in place; exps NVFP4 3 x 144 B a
        // stack; output_norm F32 32 B and exps.scale F32 12 B made.
        assert_eq!(
            plan_lines(&a.header, &skipped),
            [
                "toy: 5 tensors, 604 bytes; 1 folder tensors skipped".to_string(),
                "2 tensors, 128 bytes read in place from the folder's files".to_string(),
                "1 tensors, 432 bytes stacks of matrices read in place".to_string(),
                "2 tensors, 44 bytes made at load from the folder's tensors".to_string(),
                format!("3 tensors, 108 bytes of {}", name(GgmlType::F32)),
                format!("1 tensors, 64 bytes of {}", name(GgmlType::BF16)),
                format!("1 tensors, 432 bytes of {}", name(GgmlType::NVFP4_ROWS)),
                "skipped 1 folder tensors: vision".to_string(),
            ]
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// `+1` adds 1 to every value of an F32 gain; a gain that is not F32 or holds a value that is
    /// not finite is refused with why and left as it was (the earlier values too), and the largest
    /// finite gain stays finite, so no result is out of range.
    #[test]
    fn plus_one_adds_one_and_refuses_what_is_not_a_finite_f32_gain() {
        let gain = |type_id: u32, v: &[f32]| Normal {
            type_id,
            dims: vec![v.len() as u64],
            bytes: v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            scale: vec![],
        };
        let f32_id = GgmlType::F32.as_u32();
        let mut n = gain(f32_id, &[0.5, -1.0, 0.0, f32::MAX]);
        PlusOne.apply(&mut n).unwrap();
        assert_eq!(n.bytes, gain(f32_id, &[1.5, 0.0, 1.0, f32::MAX]).bytes);
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut n = gain(f32_id, &[1.0, bad]);
            let before = n.bytes.clone();
            let e = PlusOne.apply(&mut n).unwrap_err();
            assert!(e.contains("+1 of non-finite input"), "{bad}: {e}");
            assert_eq!(n.bytes, before, "{bad}");
        }
        let mut n = gain(GgmlType::BF16.as_u32(), &[1.0]);
        let e = PlusOne.apply(&mut n).unwrap_err();
        assert!(e.contains("+1 needs F32, not type"), "{e}");
    }
}
