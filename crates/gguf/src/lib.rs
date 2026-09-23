//! gguf — GGUF v3 reader for bloomery stage 1 (round 1-1).
//!
//! Turns a GGUF file into named, shaped, typed tensors backed by an mmap,
//! plus the scalar reference dequantizer (`quant`) the oracle gate compares
//! against ggml itself. No attention, no MoE, no GPU — later rounds build on
//! `TensorInfo::dims` staying in **ggml's own `ne[]` order** (ne[0] = row
//! length, the contiguous axis), which every downstream comparison assumes.
//!
//! Two entry points share one header parser: `Gguf::open*` (strict — every
//! tensor must be sized by this engine's `GgmlType` and sit inside the file)
//! and [`inventory_of`] (header-only — unknown type ids are carried as
//! numbers with ggml's size table, so files with types the engine cannot
//! dequantize still inventory). A model split across shards opens as a
//! [`Split`]: one strict reader per shard, validated as one model.
//!
//! Format facts mirrored from the vendored ik_llama.cpp reader
//! (`gguf_init_from_file`, ggml.c:31219-31475):
//!   * header: magic `GGUF`, u32 version (3), u64 tensor count, u64 KV count;
//!   * KVs: `key` string, u32 value type tag, value (GGUF value tags 0-12,
//!     see [`Value`]);
//!   * tensor infos: name string, u32 n_dims, n_dims × u64 `ne` (ne[0] first),
//!     u32 ggml type tag, u64 offset **relative to the data base**;
//!   * alignment: `general.alignment` KV (u32) or 32; the data base is the
//!     end of the tensor-info section rounded up to that alignment
//!     (ggml.c:31436-31447).
//!
//! Errors, not panics: every file-derived value is bounds-checked — a
//! truncated or hostile file yields [`LoadError`].

pub mod quant;

pub use quant::{
    ActivationFormat, GgmlType, QuantError, activation_format, dequant_row, quantize_activations,
    quantize_row_q8_2_x4_roundtrip, quantize_row_q8_k_roundtrip,
};

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::{Mmap, MmapMut};

/// One GGUF metadata value. Tags are the on-disk value-type ids:
/// 0 u8, 1 i8, 2 u16, 3 i16, 4 u32, 5 i32, 6 f32, 7 bool, 8 string,
/// 9 array, 10 u64, 11 i64, 12 f64.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
    U64(u64),
    I64(i64),
    F64(f64),
}

/// One tensor, as the header describes it. `offset` is relative to the data
/// base (see [`Gguf::data`]); `nbytes` is the contiguous row-major size
/// `type_size * (ne[0]/blck) * ne[1] * …` — the same product ggml's
/// `ggml_row_size` computes for the whole tensor.
#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    /// ggml `ne[]` order, NOT transposed: dims[0] is the contiguous axis.
    pub dims: Vec<u64>,
    pub ty: GgmlType,
    pub offset: u64,
    pub nbytes: u64,
}

/// One tensor as the header states it, before any engine requirement is
/// imposed: the ggml type id is carried as a number ([`RawTensorInfo::nbytes`]
/// resolved from ggml's type table, `None` when no table knows the type).
/// This is what [`inventory_of`] reports; the strict path re-derives its own
/// sizes from `GgmlType` and refuses what it cannot size.
#[derive(Clone, Debug)]
pub struct RawTensorInfo {
    pub name: String,
    /// ggml `ne[]` order, NOT transposed: dims[0] is the contiguous axis.
    pub dims: Vec<u64>,
    pub type_id: u32,
    pub offset: u64,
    /// `type_size * (ne[0]/blck) * ne[1] * …` from ggml's table — the same
    /// product `ggml_row_size` computes — or `None` for a type id outside
    /// both [`GgmlType`] and [`ggml_type_info`].
    pub nbytes: Option<u64>,
}

#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("not a GGUF file: magic {0:#010x}")]
    BadMagic(u32),
    #[error("unsupported GGUF version {0} (this reader handles 3)")]
    Version(u32),
    #[error("truncated file: need {need} bytes at offset {at}, file is {len}")]
    Truncated { at: u64, need: u64, len: u64 },
    #[error("metadata value type tag {0} is not a GGUF value type (0-12)")]
    BadValueType(u32),
    #[error("tensor {name:?}: n_dims {n} outside 1..=4")]
    BadDims { name: String, n: u32 },
    #[error("tensor {name:?}: dims must all be >= 1, got {dims:?}")]
    BadDimValue { name: String, dims: Vec<u64> },
    #[error("tensor {name:?}: ggml type {ty} has no size in this build's table")]
    UnsupportedType { name: String, ty: GgmlType },
    #[error("tensor {name:?}: ne[0] {ne0} is not a multiple of the block size {blck}")]
    UnalignedRow { name: String, ne0: u64, blck: u64 },
    #[error("tensor {name:?}: data at {base}+{off} size {nbytes} exceeds the file ({len})")]
    OutOfBounds {
        name: String,
        base: u64,
        off: u64,
        nbytes: u64,
        len: u64,
    },
    /// A shard of a split that its own open refused; the error names the shard.
    #[error("{path}: {source}")]
    Shard {
        path: String,
        source: Box<LoadError>,
    },
    #[error("split shard {path} is missing")]
    MissingShard { path: String },
    #[error("{path}: {key} {detail}")]
    SplitKey {
        path: String,
        key: &'static str,
        detail: String,
    },
    #[error("{path}: split.count says a split, but the name does not end in {suffix:?}")]
    SplitName { path: String, suffix: String },
    #[error("tensor {name:?} appears twice: in {first} and in {second}")]
    DuplicateTensor {
        name: String,
        first: String,
        second: String,
    },
}

/// A parsed, mmap-backed GGUF file.
pub struct Gguf {
    // Kept alive for the mapping's lifetime; the mapping borrows the fd.
    _file: File,
    map: Backing,
    meta: Vec<(String, Value)>,
    tensors: Vec<TensorInfo>,
    /// File offset where the (aligned) tensor data section starts.
    data_base: u64,
    alignment: u64,
}

/// Where the file's bytes live while the model runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Weights {
    /// The file mapping itself: page cache, 4 KiB pages.
    Mapped { populate: bool },
    /// A private anonymous copy; `huge` asks for transparent 2 MiB pages before
    /// the first touch. Same bytes at the same offsets, so nothing downstream
    /// can tell — only the TLB reach and the prefetcher's page boundaries move.
    Resident { huge: bool },
}

enum Backing {
    File(Mmap),
    Anon(MmapMut),
}

impl std::ops::Deref for Backing {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            Backing::File(m) => m,
            Backing::Anon(m) => m,
        }
    }
}

impl Gguf {
    /// Parse `path` and validate every tensor's placement. The mmap is
    /// created read-only; nothing is copied.
    pub fn open(path: impl AsRef<Path>) -> Result<Gguf, LoadError> {
        Self::open_with(path, false)
    }

    /// `open`, optionally prefaulting the whole mapping once the header has
    /// passed (`MADV_POPULATE_READ`). A lazy mapping takes a page fault on the
    /// first touch of every weight page, and a MoE keeps meeting untouched
    /// experts for hundreds of steps; a process that will decode pays that once
    /// here instead. Tests open lazily.
    pub fn open_with(path: impl AsRef<Path>, populate: bool) -> Result<Gguf, LoadError> {
        Self::open_backed(path, Weights::Mapped { populate })
    }

    /// `open`, with the caller choosing where the bytes live. The header is
    /// parsed and every tensor validated on a lazy mapping first, so a file
    /// that is refused costs its header's pages; only then is the mapping
    /// populated or copied.
    pub fn open_backed(path: impl AsRef<Path>, weights: Weights) -> Result<Gguf, LoadError> {
        let file = File::open(path)?;
        let len = file.metadata()?.len();
        // SAFETY: the file is opened read-only and nothing maps it writable;
        // a concurrent truncation would surface as SIGBUS, the same contract
        // ik_llama.cpp's own mmap loader accepts.
        let file_map = unsafe { memmap2::MmapOptions::new().map(&file)? };
        let h = parse_header(&file_map, len)?;
        let tensors = strict_tensors(h.tensors, h.data_base, len)?;
        let map = match weights {
            Weights::Mapped { populate } => {
                #[cfg(target_os = "linux")]
                if populate {
                    file_map.advise(memmap2::Advice::PopulateRead)?;
                }
                #[cfg(not(target_os = "linux"))]
                let _ = populate;
                Backing::File(file_map)
            }
            Weights::Resident { huge } => {
                let mut anon = memmap2::MmapOptions::new().len(len as usize).map_anon()?;
                // Before the first touch: a page faulted in small stays small
                // until khugepaged gets to it.
                #[cfg(target_os = "linux")]
                if huge {
                    anon.advise(memmap2::Advice::HugePage)?;
                }
                #[cfg(not(target_os = "linux"))]
                let _ = huge;
                anon.copy_from_slice(&file_map);
                Backing::Anon(anon)
            }
        };
        Ok(Gguf {
            _file: file,
            map,
            meta: h.meta,
            tensors,
            data_base: h.data_base,
            alignment: h.alignment,
        })
    }

    /// Number of metadata KV pairs.
    pub fn kv_count(&self) -> usize {
        self.meta.len()
    }

    /// The i-th metadata pair, in file order.
    pub fn kv(&self, i: usize) -> Option<(&str, &Value)> {
        self.meta.get(i).map(|(k, v)| (k.as_str(), v))
    }

    /// Iterate all metadata pairs.
    pub fn iter_kv(&self) -> impl Iterator<Item = (&str, &Value)> {
        self.meta.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Raw metadata lookup by exact key.
    pub fn value(&self, key: &str) -> Option<&Value> {
        self.meta.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    /// `general.architecture` (e.g. "deepseek2").
    pub fn architecture(&self) -> Option<&str> {
        self.value("general.architecture").and_then(Value::as_str)
    }

    /// `<architecture>.<suffix>` as an unsigned integer — the deepseek2
    /// keys stage 1 reads are all u32: `deepseek2.block_count`,
    /// `.expert_count`, `.expert_used_count`, `.embedding_length`,
    /// `.attention.head_count`. Anything else the file carries under
    /// `deepseek2.*` is reachable through [`Gguf::iter_kv`] filtered on the
    /// prefix.
    pub fn arch_get_u64(&self, suffix: &str) -> Option<u64> {
        let arch = self.architecture()?;
        let key = format!("{arch}.{suffix}");
        self.value(&key).and_then(Value::as_u64)
    }

    /// The full metadata key `<architecture>.<suffix>`, for the callers that
    /// need to name a key they could not read. The bare suffix when the file
    /// declares no architecture — every reader of such a file has already failed.
    pub fn arch_key(&self, suffix: &str) -> String {
        match self.architecture() {
            Some(arch) => format!("{arch}.{suffix}"),
            None => suffix.to_string(),
        }
    }

    /// `<architecture>.<suffix>` as f32 — the same prefixing as
    /// [`Gguf::arch_get_u64`], for the float hyperparameters (rms epsilon, the
    /// rope scaling factor and log multiplier, the rope base frequency).
    pub fn arch_get_f32(&self, suffix: &str) -> Option<f32> {
        let arch = self.architecture()?;
        let key = format!("{arch}.{suffix}");
        self.value(&key).and_then(Value::as_f32)
    }

    /// `<architecture>.<suffix>` as a string — same prefixing again, for the
    /// keys whose value is an enumeration name (`rope.scaling.type`).
    pub fn arch_get_str(&self, suffix: &str) -> Option<&str> {
        let arch = self.architecture()?;
        let key = format!("{arch}.{suffix}");
        self.value(&key).and_then(Value::as_str)
    }

    /// `<architecture>.<suffix>` as an array's items — same prefixing, for the
    /// per-layer tables (`attention.compress_ratios`, `swiglu_clamp_exp`).
    pub fn arch_get_arr(&self, suffix: &str) -> Option<&[Value]> {
        let arch = self.architecture()?;
        let key = format!("{arch}.{suffix}");
        match self.value(&key) {
            Some(Value::Array(items)) => Some(items),
            _ => None,
        }
    }

    /// Typed getters for the stage-1 hyperparameters (see [`Gguf::arch_get_u64`]).
    pub fn block_count(&self) -> Option<u64> {
        self.arch_get_u64("block_count")
    }

    pub fn expert_count(&self) -> Option<u64> {
        self.arch_get_u64("expert_count")
    }

    pub fn expert_used_count(&self) -> Option<u64> {
        self.arch_get_u64("expert_used_count")
    }

    pub fn embedding_length(&self) -> Option<u64> {
        self.arch_get_u64("embedding_length")
    }

    pub fn attention_head_count(&self) -> Option<u64> {
        self.arch_get_u64("attention.head_count")
    }

    /// Number of tensors.
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// The i-th tensor, in file order.
    pub fn tensor(&self, i: usize) -> Option<&TensorInfo> {
        self.tensors.get(i)
    }

    /// Iterate all tensors in file order.
    pub fn iter_tensors(&self) -> impl Iterator<Item = &TensorInfo> {
        self.tensors.iter()
    }

    /// Lookup by exact tensor name (e.g. "blk.0.attn_q.weight").
    pub fn find(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// File offset of the aligned data section.
    pub fn data_base(&self) -> u64 {
        self.data_base
    }

    /// The alignment in use (`general.alignment`, default 32).
    pub fn alignment(&self) -> u64 {
        self.alignment
    }

    /// The tensor's bytes in the mapping. Bounds were validated at open; the
    /// slice arithmetic is checked again here, so a stale `TensorInfo` from
    /// another file errors instead of panicking.
    pub fn data<'a>(&'a self, t: &TensorInfo) -> Result<&'a [u8], LoadError> {
        let start = self.data_base as usize + t.offset as usize;
        let end = start
            .checked_add(t.nbytes as usize)
            // `ok_or_else`: the eager form cloned the name on every call, hit or not.
            .ok_or_else(|| LoadError::OutOfBounds {
                name: t.name.clone(),
                base: self.data_base,
                off: t.offset,
                nbytes: t.nbytes,
                len: self.map.len() as u64,
            })?;
        if end > self.map.len() {
            return Err(LoadError::OutOfBounds {
                name: t.name.clone(),
                base: self.data_base,
                off: t.offset,
                nbytes: t.nbytes,
                len: self.map.len() as u64,
            });
        }
        Ok(&self.map[start..end])
    }

    /// The whole mapping, file offset `o` at index `o`. Index 0 sits on a page
    /// boundary (the kernel places a mapping on one), so a sub-slice of whole
    /// pages is what a caller locks in RAM (`mlock`) or asks the residency of
    /// (`mincore`); [`Gguf::data`] is one tensor's slice of it.
    pub fn mapping(&self) -> &[u8] {
        &self.map
    }
}

/// The split keys llama.cpp's splitter writes into every shard
/// (ik examples/gguf-split/gguf-split.cpp:246-248): `split.no` and
/// `split.count` as u16, `split.tensors.count` — the whole model's count —
/// as i32.
const SPLIT_NO: &str = "split.no";
const SPLIT_COUNT: &str = "split.count";
const SPLIT_TENSORS: &str = "split.tensors.count";

/// A model as one GGUF file or as llama.cpp's split set. The first shard
/// names the set: its siblings are `<prefix>-%05d-of-%05d.gguf`
/// (`llama_split_path`, ik src/llama.cpp:13829; the prefix is what
/// `llama_split_prefix`, :13904, strips). The set is one model or it is
/// refused: every shard's `split.no` is its place, every `split.count`
/// agrees, the tensor counts sum to the first shard's `split.tensors.count`,
/// and no tensor name appears twice — the ik loader's checks
/// (src/llama-model-loader.cpp:361-431) plus the per-shard keys it does not
/// read.
///
/// Every shard opens with the strict reader and lazily, so opening touches
/// the headers only. Metadata is the first shard's: the splitter writes the
/// model's keys there alone (gguf-split.cpp:242).
pub struct Split {
    shards: Vec<Gguf>,
    paths: Vec<PathBuf>,
    /// Tensor name to (shard, index in that shard's table).
    by_name: HashMap<String, (usize, usize)>,
}

impl Split {
    /// Open the model whose first shard is `first`. A file without
    /// `split.count` is a one-shard model.
    pub fn open(first: impl AsRef<Path>) -> Result<Split, LoadError> {
        let first = first.as_ref().to_path_buf();
        let head = open_shard(&first)?;
        let Some(count) = head.value(SPLIT_COUNT) else {
            return Split::assemble(vec![head], vec![first]);
        };
        let count = split_int(&first, SPLIT_COUNT, Some(count))?;
        if count == 0 {
            return Err(split_key(&first, SPLIT_COUNT, "is 0".to_string()));
        }
        check_place(&head, &first, 0)?;
        let mut shards = vec![head];
        let mut paths = vec![first];
        if count > 1 {
            let suffix = shard_suffix(0, count);
            let name = paths[0].to_string_lossy().into_owned();
            let prefix = match name.strip_suffix(&suffix) {
                Some(p) if !p.is_empty() => p.to_string(),
                _ => return Err(LoadError::SplitName { path: name, suffix }),
            };
            for i in 1..count {
                let path = PathBuf::from(format!("{prefix}{}", shard_suffix(i, count)));
                let g = open_shard(&path)?;
                check_place(&g, &path, i)?;
                let theirs = split_int(&path, SPLIT_COUNT, g.value(SPLIT_COUNT))?;
                if theirs != count {
                    let detail = format!("is {theirs}, the first shard's is {count}");
                    return Err(split_key(&path, SPLIT_COUNT, detail));
                }
                shards.push(g);
                paths.push(path);
            }
        }
        let want = split_int(&paths[0], SPLIT_TENSORS, shards[0].value(SPLIT_TENSORS))?;
        let got: u64 = shards.iter().map(|g| g.tensor_count() as u64).sum();
        if got != want {
            let detail = format!("is {want}, but the {count} shards hold {got} tensors");
            return Err(split_key(&paths[0], SPLIT_TENSORS, detail));
        }
        Split::assemble(shards, paths)
    }

    /// Index every tensor by name, refusing a name that appears twice.
    fn assemble(shards: Vec<Gguf>, paths: Vec<PathBuf>) -> Result<Split, LoadError> {
        let mut by_name: HashMap<String, (usize, usize)> = HashMap::new();
        for (s, g) in shards.iter().enumerate() {
            for (i, t) in g.iter_tensors().enumerate() {
                if let Some(&(first, _)) = by_name.get(&t.name) {
                    return Err(LoadError::DuplicateTensor {
                        name: t.name.clone(),
                        first: paths[first].display().to_string(),
                        second: paths[s].display().to_string(),
                    });
                }
                by_name.insert(t.name.clone(), (s, i));
            }
        }
        Ok(Split {
            shards,
            paths,
            by_name,
        })
    }

    /// Number of shards (1 for a single file).
    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    /// Shard `i`'s reader — the one whose [`Gguf::data`] holds its tensors.
    pub fn shard(&self, i: usize) -> Option<&Gguf> {
        self.shards.get(i)
    }

    /// Shard `i`'s path.
    pub fn shard_path(&self, i: usize) -> Option<&Path> {
        self.paths.get(i).map(PathBuf::as_path)
    }

    /// Tensors across all shards.
    pub fn tensor_count(&self) -> usize {
        self.by_name.len()
    }

    /// Lookup by exact tensor name: the shard that holds it and its info.
    pub fn find(&self, name: &str) -> Option<(usize, &TensorInfo)> {
        let &(s, i) = self.by_name.get(name)?;
        Some((s, self.shards.get(s)?.tensor(i)?))
    }

    /// Every tensor with its shard, in shard order then file order.
    pub fn iter_tensors(&self) -> impl Iterator<Item = (usize, &TensorInfo)> {
        self.shards
            .iter()
            .enumerate()
            .flat_map(|(s, g)| g.iter_tensors().map(move |t| (s, t)))
    }

    /// Raw metadata lookup by exact key, in the first shard.
    pub fn value(&self, key: &str) -> Option<&Value> {
        self.shards[0].value(key)
    }

    /// The first shard's metadata pairs, in file order.
    pub fn iter_kv(&self) -> impl Iterator<Item = (&str, &Value)> {
        self.shards[0].iter_kv()
    }

    /// [`Gguf::architecture`] of the first shard.
    pub fn architecture(&self) -> Option<&str> {
        self.shards[0].architecture()
    }

    /// [`Gguf::arch_key`] of the first shard.
    pub fn arch_key(&self, suffix: &str) -> String {
        self.shards[0].arch_key(suffix)
    }

    /// [`Gguf::arch_get_u64`] of the first shard.
    pub fn arch_get_u64(&self, suffix: &str) -> Option<u64> {
        self.shards[0].arch_get_u64(suffix)
    }

    /// [`Gguf::arch_get_f32`] of the first shard.
    pub fn arch_get_f32(&self, suffix: &str) -> Option<f32> {
        self.shards[0].arch_get_f32(suffix)
    }

    /// [`Gguf::arch_get_str`] of the first shard.
    pub fn arch_get_str(&self, suffix: &str) -> Option<&str> {
        self.shards[0].arch_get_str(suffix)
    }

    /// [`Gguf::arch_get_arr`] of the first shard.
    pub fn arch_get_arr(&self, suffix: &str) -> Option<&[Value]> {
        self.shards[0].arch_get_arr(suffix)
    }
}

/// `-%05d-of-%05d.gguf` for shard `i` of `count`, the part `llama_split_path`
/// appends to the prefix (numbered from 1).
fn shard_suffix(i: u64, count: u64) -> String {
    format!("-{:05}-of-{count:05}.gguf", i + 1)
}

/// The strict open of one shard, its error naming the file.
fn open_shard(path: &Path) -> Result<Gguf, LoadError> {
    Gguf::open(path).map_err(|e| match e {
        LoadError::Io(io) if io.kind() == std::io::ErrorKind::NotFound => LoadError::MissingShard {
            path: path.display().to_string(),
        },
        other => LoadError::Shard {
            path: path.display().to_string(),
            source: Box::new(other),
        },
    })
}

fn split_key(path: &Path, key: &'static str, detail: String) -> LoadError {
    LoadError::SplitKey {
        path: path.display().to_string(),
        key,
        detail,
    }
}

/// A split key's value as a count, or the error that names it.
fn split_int(path: &Path, key: &'static str, v: Option<&Value>) -> Result<u64, LoadError> {
    match v {
        None => Err(split_key(path, key, "is absent".to_string())),
        Some(v) => v
            .as_unsigned()
            .ok_or_else(|| split_key(path, key, format!("is {v:?}, not a count"))),
    }
}

/// Shard `place` of its set must say so in `split.no`.
fn check_place(g: &Gguf, path: &Path, place: u64) -> Result<(), LoadError> {
    let no = split_int(path, SPLIT_NO, g.value(SPLIT_NO))?;
    if no != place {
        let detail = format!("is {no}, but the file is shard {place} of its set (from 0)");
        return Err(split_key(path, SPLIT_NO, detail));
    }
    Ok(())
}

/// A header-only parse: everything the header states, nothing the engine
/// requires. Placement, block alignment and dequantizability are NOT
/// validated, and type ids the engine has no `GgmlType` for are carried as
/// numbers with their size from ggml's table (or `None`). This is the
/// inventory for files the strict reader refuses — e.g. i-quant or q4_0
/// files — where the point is to know what is in the file, not to run it.
pub struct Inventory {
    pub version: u32,
    pub meta: Vec<(String, Value)>,
    pub tensors: Vec<RawTensorInfo>,
    /// End of the tensor-info section; [`Inventory::data_base`] is this
    /// rounded up to [`Inventory::alignment`].
    pub header_end: u64,
    pub data_base: u64,
    pub alignment: u64,
    pub file_len: u64,
}

impl Inventory {
    /// Raw metadata lookup by exact key.
    pub fn value(&self, key: &str) -> Option<&Value> {
        self.meta.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }
}

/// Parse only the header of `path`: magic, version, KVs, tensor infos.
/// The mapping is created lazily and never populated — the file's pages are
/// touched only as far as the tensor-info section reaches, so a multi-
/// hundred-GiB split shards in seconds without reading tensor bytes.
pub fn inventory_of(path: impl AsRef<Path>) -> Result<Inventory, LoadError> {
    let file = File::open(path)?;
    let len = file.metadata()?.len();
    // SAFETY: read-only mapping, nothing maps it writable — the same
    // contract `Gguf::open_backed` accepts.
    let map = unsafe { memmap2::MmapOptions::new().map(&file)? };
    let h = parse_header(&map, len)?;
    Ok(Inventory {
        version: h.version,
        meta: h.meta,
        tensors: h.tensors,
        header_end: h.header_end,
        data_base: h.data_base,
        alignment: h.alignment,
        file_len: len,
    })
}

/// ggml's `(type_name, blck_size, type_size)` for type ids this engine's
/// [`GgmlType`] does not model. Ids are `enum ggml_type` values (ik_llama.cpp
/// ggml.h:391-427); block sizes and byte sizes are the block structs'
/// static_asserts in ggml-common.h, cited per entry:
///   * 2 `q4_0` 32/18 (ggml-common.h:166-171), 3 `q4_1` 32/20 (:173-178,
///     `GGML_SCALE_TYPE1` = two halves, :16),
///   * 9 `q8_1` 32/36 (:240-244),
///   * 10 `q2_K` 256/84 (:307-313), 15 `q8_K` 256/296 — ik layout, `float d;
///     float sum; int8_t qs[256]; int16_t bsums[16]` (:404-410); mainline
///     ggml's block_q8_K is 264 B (no sum/bsums), a divergence to remember if
///     a q8_K file's byte sums ever disagree with its size,
///   * 16 `iq2_xxs` 256/66 (:436-442), 17 `iq2_xs` 256/74 (:452-457),
///     18 `iq3_xxs` 256/98 (:485-491), 19 `iq1_s` 256/50 (:521-526),
///     20 `iq4_nl` 32/18 (:585-589), 21 `iq3_s` 256/110 (:498-510),
///     22 `iq2_s` 256/82 (:467-473), 23 `iq4_xs` 256/136 (:602-607),
///     29 `iq1_m` 256/56 (:534-539),
///   * 24 `i8` 1/1, 25 `i16` 1/2, 26 `i32` 1/4, 27 `i64` 1/8, 28 `f64` 1/8
///     (type_traits table, ggml.c:621-652).
///
/// The engine's own numbers stay owned by `GgmlType::blck_size`/`type_size`
/// for the ids it models; ids absent from both return `None` and inventory
/// with unknown size.
pub fn ggml_type_info(id: u32) -> Option<(&'static str, u64, u64)> {
    let ty = GgmlType::from_u32(id);
    if let Some(name) = ty.name() {
        // An id `GgmlType` names is one it sizes; the engine's own numbers
        // stay the owner for these.
        return ty
            .blck_size()
            .zip(ty.type_size())
            .map(|(b, t)| (name, b, t));
    }
    let (name, blck, tsz) = match id {
        2 => ("q4_0", 32, 18),
        3 => ("q4_1", 32, 20),
        9 => ("q8_1", 32, 36),
        10 => ("q2_K", 256, 84),
        15 => ("q8_K", 256, 296),
        16 => ("iq2_xxs", 256, 66),
        17 => ("iq2_xs", 256, 74),
        18 => ("iq3_xxs", 256, 98),
        19 => ("iq1_s", 256, 50),
        20 => ("iq4_nl", 32, 18),
        21 => ("iq3_s", 256, 110),
        22 => ("iq2_s", 256, 82),
        23 => ("iq4_xs", 256, 136),
        24 => ("i8", 1, 1),
        25 => ("i16", 1, 2),
        26 => ("i32", 1, 4),
        27 => ("i64", 1, 8),
        28 => ("f64", 1, 8),
        29 => ("iq1_m", 256, 56),
        _ => return None,
    };
    Some((name, blck, tsz))
}

/// `type_size * (ne[0]/blck) * ne[1] * …` for a type id ggml's table knows,
/// else `None`. The product is `ggml_row_size`'s: integer division, missing
/// dims implicit 1s.
fn row_bytes(type_id: u32, dims: &[u64]) -> Option<u64> {
    let (_, blck, tsz) = ggml_type_info(type_id)?;
    Some(tsz * (dims[0] / blck) * dims[1..].iter().product::<u64>())
}

impl Value {
    /// String view if this value is a string.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    /// Unsigned view for any unsigned integer flavor (u8/u16/u32/u64) —
    /// the deepseek2 hyperparameter keys are u32 on disk.
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::U8(v) => Some(*v as u64),
            Value::U16(v) => Some(*v as u64),
            Value::U32(v) => Some(*v as u64),
            Value::U64(v) => Some(*v),
            _ => None,
        }
    }

    /// Any integer flavor, signed or not, when the value is not negative —
    /// llama.cpp writes some counts and per-layer tables as i32
    /// (`split.tensors.count`, `<arch>.attention.compress_ratios`).
    pub fn as_unsigned(&self) -> Option<u64> {
        match *self {
            Value::I8(v) => u64::try_from(v).ok(),
            Value::I16(v) => u64::try_from(v).ok(),
            Value::I32(v) => u64::try_from(v).ok(),
            Value::I64(v) => u64::try_from(v).ok(),
            _ => self.as_u64(),
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Value::F32(v) => Some(*v),
            Value::F64(v) => Some(*v as f32),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(v) => Some(*v),
            _ => None,
        }
    }
}

fn meta_u32(meta: &[(String, Value)], key: &str) -> Option<u32> {
    match meta.iter().find(|(k, _)| k == key).map(|(_, v)| v) {
        Some(Value::U32(v)) => Some(*v),
        Some(Value::U16(v)) => Some(*v as u32),
        Some(Value::U8(v)) => Some(*v as u32),
        _ => None,
    }
}

/// The strict pass over a parsed header's tensors: every tensor must be sized
/// by this engine's `GgmlType` and its first dim must be block-aligned; then
/// every tensor must sit inside the `len`-byte file — the placement invariant
/// the coverage gate re-checks per tensor.
fn strict_tensors(
    raw: Vec<RawTensorInfo>,
    data_base: u64,
    len: u64,
) -> Result<Vec<TensorInfo>, LoadError> {
    let mut strict = Vec::with_capacity(raw.len());
    for t in raw {
        let ty = GgmlType::from_u32(t.type_id);
        let blck = ty.blck_size().ok_or_else(|| LoadError::UnsupportedType {
            name: t.name.clone(),
            ty,
        })?;
        let tsz = ty.type_size().ok_or_else(|| LoadError::UnsupportedType {
            name: t.name.clone(),
            ty,
        })?;
        if t.dims[0] % blck != 0 {
            return Err(LoadError::UnalignedRow {
                name: t.name,
                ne0: t.dims[0],
                blck,
            });
        }
        // Missing dims are implicit 1s (ggml tensors are 4-D); the
        // product below folds them in.
        let nbytes = tsz * (t.dims[0] / blck) * t.dims[1..].iter().product::<u64>();
        strict.push(TensorInfo {
            name: t.name,
            dims: t.dims,
            ty,
            offset: t.offset,
            nbytes,
        });
    }
    for t in &strict {
        let out_of_bounds = || LoadError::OutOfBounds {
            name: t.name.clone(),
            base: data_base,
            off: t.offset,
            nbytes: t.nbytes,
            len,
        };
        let end = data_base
            .checked_add(t.offset)
            .and_then(|o| o.checked_add(t.nbytes))
            .ok_or_else(out_of_bounds)?;
        if end > len {
            return Err(out_of_bounds());
        }
    }
    Ok(strict)
}

/// What [`parse_header`] returns: the file's own statement of itself.
struct Header {
    version: u32,
    meta: Vec<(String, Value)>,
    tensors: Vec<RawTensorInfo>,
    header_end: u64,
    alignment: u64,
    data_base: u64,
}

/// The one parser both entry points share: magic, version, KVs, tensor
/// infos, alignment and the data base — everything the header states, with
/// no engine requirement imposed. The strict path layers `GgmlType` sizing
/// and placement checks on the result; [`inventory_of`] returns it as-is.
fn parse_header(b: &[u8], len: u64) -> Result<Header, LoadError> {
    let mut rd = Reader { b, pos: 0, len };
    let magic = rd.u32()?;
    if magic != 0x4655_4747 {
        // "GGUF" read little-endian: file bytes 47 47 55 46.
        return Err(LoadError::BadMagic(magic));
    }
    let version = rd.u32()?;
    if version != 3 {
        return Err(LoadError::Version(version));
    }
    let n_tensors = rd.u64()?;
    let n_kv = rd.u64()?;

    let mut meta = Vec::new();
    for _ in 0..n_kv {
        let key = rd.string()?;
        let tag = rd.u32()?;
        let val = read_value(&mut rd, tag)?;
        meta.push((key, val));
    }

    let mut tensors = Vec::new();
    for _ in 0..n_tensors {
        let name = rd.string()?;
        let n_dims = rd.u32()?;
        if !(1..=4).contains(&n_dims) {
            return Err(LoadError::BadDims { name, n: n_dims });
        }
        let mut dims = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            let d = rd.u64()?;
            if d == 0 {
                return Err(LoadError::BadDimValue { name, dims });
            }
            dims.push(d);
        }
        let type_id = rd.u32()?;
        let offset = rd.u64()?;
        tensors.push(RawTensorInfo {
            nbytes: row_bytes(type_id, &dims),
            name,
            dims,
            type_id,
            offset,
        });
    }

    // Alignment and the data base, mirroring ggml.c:31432-31447:
    // `general.alignment` (u32 KV) or 32, then round the end of the
    // tensor-info section up to it.
    let alignment = meta_u32(&meta, "general.alignment").unwrap_or(32) as u64;
    let header_end = rd.pos as u64;
    let data_base = header_end.div_ceil(alignment) * alignment;
    Ok(Header {
        version,
        meta,
        tensors,
        header_end,
        alignment,
        data_base,
    })
}

/// Bounds-checked little-endian reader over the mapping.
struct Reader<'a> {
    b: &'a [u8],
    pos: usize,
    len: u64,
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8], LoadError> {
        let end = self.pos.checked_add(n).ok_or(LoadError::Truncated {
            at: self.pos as u64,
            need: n as u64,
            len: self.len,
        })?;
        if end > self.b.len() {
            return Err(LoadError::Truncated {
                at: self.pos as u64,
                need: n as u64,
                len: self.len,
            });
        }
        let s = &self.b[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn u8(&mut self) -> Result<u8, LoadError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, LoadError> {
        let s = self.take(2)?;
        Ok(u16::from_le_bytes([s[0], s[1]]))
    }

    fn u32(&mut self) -> Result<u32, LoadError> {
        let s = self.take(4)?;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }

    fn u64(&mut self) -> Result<u64, LoadError> {
        let s = self.take(8)?;
        Ok(u64::from_le_bytes([
            s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7],
        ]))
    }

    /// GGUF string: u64 byte length + bytes. The length is checked against
    /// the remaining file before anything is allocated.
    fn string(&mut self) -> Result<String, LoadError> {
        let n = self.u64()?;
        if n > (self.b.len() - self.pos) as u64 {
            return Err(LoadError::Truncated {
                at: self.pos as u64,
                need: n,
                len: self.len,
            });
        }
        let s = self.take(n as usize)?;
        // GGUF strings are UTF-8; lossy so a stray byte cannot abort a load
        // of an otherwise-readable file.
        Ok(String::from_utf8_lossy(s).into_owned())
    }
}

fn read_value(rd: &mut Reader<'_>, tag: u32) -> Result<Value, LoadError> {
    Ok(match tag {
        0 => Value::U8(rd.u8()?),
        1 => Value::I8(rd.u8()? as i8),
        2 => Value::U16(rd.u16()?),
        3 => Value::I16(rd.u16()? as i16),
        4 => Value::U32(rd.u32()?),
        5 => Value::I32(rd.u32()? as i32),
        6 => Value::F32(f32::from_bits(rd.u32()?)),
        7 => Value::Bool(rd.u8()? != 0),
        8 => Value::String(rd.string()?),
        9 => {
            // Array: u32 element tag, u64 count, then the elements. The
            // capacity is capped so a lying count cannot pre-allocate
            // gigabytes — a count that outruns the file dies at Truncated
            // when the elements actually run out.
            let elem_tag = rd.u32()?;
            let n = rd.u64()?;
            let mut items = Vec::with_capacity(n.min(4096) as usize);
            for _ in 0..n {
                items.push(read_value(rd, elem_tag)?);
            }
            Value::Array(items)
        }
        10 => Value::U64(rd.u64()?),
        11 => Value::I64(rd.u64()? as i64),
        12 => Value::F64(f64::from_bits(rd.u64()?)),
        other => return Err(LoadError::BadValueType(other)),
    })
}

#[cfg(test)]
mod split_tests {
    use super::{Gguf, LoadError, Split};
    use std::path::{Path, PathBuf};

    /// A metadata value as the synthetic shards carry it.
    enum Kv {
        U16(u16),
        I32(i32),
        Str(&'static str),
    }

    fn put_str(b: &mut Vec<u8>, s: &str) {
        b.extend_from_slice(&(s.len() as u64).to_le_bytes());
        b.extend_from_slice(s.as_bytes());
    }

    /// A GGUF v3 file with `kvs` and one 4-value F32 tensor per name, 32 bytes
    /// apart over zero data — a few hundred bytes the strict reader opens.
    fn write_gguf(path: &Path, kvs: &[(&str, Kv)], tensors: &[&str]) {
        write_gguf_as(path, kvs, tensors, 0, 0);
    }

    /// [`write_gguf`] with every tensor tagged ggml type id `ty` and `extra`
    /// more zero bytes after the tensors' data.
    fn write_gguf_as(path: &Path, kvs: &[(&str, Kv)], tensors: &[&str], ty: u32, extra: usize) {
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        b.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
        for (k, v) in kvs {
            put_str(&mut b, k);
            match v {
                Kv::U16(x) => {
                    b.extend_from_slice(&2u32.to_le_bytes());
                    b.extend_from_slice(&x.to_le_bytes());
                }
                Kv::I32(x) => {
                    b.extend_from_slice(&5u32.to_le_bytes());
                    b.extend_from_slice(&x.to_le_bytes());
                }
                Kv::Str(s) => {
                    b.extend_from_slice(&8u32.to_le_bytes());
                    put_str(&mut b, s);
                }
            }
        }
        for (i, name) in tensors.iter().enumerate() {
            put_str(&mut b, name);
            b.extend_from_slice(&1u32.to_le_bytes());
            b.extend_from_slice(&4u64.to_le_bytes());
            b.extend_from_slice(&ty.to_le_bytes());
            b.extend_from_slice(&(32 * i as u64).to_le_bytes());
        }
        b.resize(b.len().div_ceil(32) * 32 + 32 * tensors.len() + extra, 0);
        std::fs::write(path, b).unwrap();
    }

    fn set_dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("gguf-split-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn shard_path(dir: &Path, i: u16) -> PathBuf {
        dir.join(format!("m-{:05}-of-00003.gguf", i + 1))
    }

    /// Shard `i` of a 3-shard set whose keys say `split.no = no`, `split.count
    /// = count`, `split.tensors.count = total`; the first also names the model.
    fn write_shard(dir: &Path, i: u16, no: u16, count: u16, total: i32, tensors: &[&str]) {
        let mut kvs = Vec::new();
        if i == 0 {
            kvs.push(("general.architecture", Kv::Str("test")));
        }
        kvs.push(("split.no", Kv::U16(no)));
        kvs.push(("split.tensors.count", Kv::I32(total)));
        kvs.push(("split.count", Kv::U16(count)));
        write_gguf(&shard_path(dir, i), &kvs, tensors);
    }

    /// A consistent set: tensors [a, b], [c], [d]. Returns the directory.
    fn good_set(name: &str) -> PathBuf {
        let d = set_dir(name);
        write_shard(&d, 0, 0, 3, 4, &["a", "b"]);
        write_shard(&d, 1, 1, 3, 4, &["c"]);
        write_shard(&d, 2, 2, 3, 4, &["d"]);
        d
    }

    fn refusal(dir: &Path) -> LoadError {
        let e = match Split::open(shard_path(dir, 0)) {
            Ok(_) => panic!("the split in {} must be refused", dir.display()),
            Err(e) => e,
        };
        std::fs::remove_dir_all(dir).unwrap();
        e
    }

    #[test]
    fn a_consistent_split_opens_as_one_model() {
        let d = good_set("happy");
        let s = Split::open(shard_path(&d, 0)).unwrap();
        assert_eq!(s.shard_count(), 3);
        assert_eq!(s.tensor_count(), 4);
        let order: Vec<(usize, &str)> = s
            .iter_tensors()
            .map(|(i, t)| (i, t.name.as_str()))
            .collect();
        assert_eq!(order, [(0, "a"), (0, "b"), (1, "c"), (2, "d")]);
        let (i, t) = s.find("c").unwrap();
        assert_eq!(i, 1);
        assert_eq!(s.shard(i).unwrap().data(t).unwrap().len(), 16);
        assert_eq!(s.shard_path(2).unwrap(), shard_path(&d, 2));
        assert!(s.find("e").is_none());
        assert_eq!(s.architecture(), Some("test"));
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn a_file_without_split_count_is_one_shard() {
        let d = set_dir("single");
        let p = d.join("plain.gguf");
        write_gguf(
            &p,
            &[("general.architecture", Kv::Str("test"))],
            &["a", "b"],
        );
        let s = Split::open(&p).unwrap();
        assert_eq!((s.shard_count(), s.tensor_count()), (1, 2));
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn a_missing_shard_is_named() {
        let d = good_set("missing");
        let gone = shard_path(&d, 1);
        std::fs::remove_file(&gone).unwrap();
        match refusal(&d) {
            LoadError::MissingShard { path } => assert_eq!(path, gone.display().to_string()),
            other => panic!("wrong refusal: {other}"),
        }
    }

    #[test]
    fn a_name_in_two_shards_is_refused() {
        let d = good_set("duplicate");
        write_shard(&d, 2, 2, 3, 4, &["a"]);
        match refusal(&d) {
            LoadError::DuplicateTensor {
                name,
                first,
                second,
            } => {
                assert_eq!(name, "a");
                assert_eq!(first, shard_path(&d, 0).display().to_string());
                assert_eq!(second, shard_path(&d, 2).display().to_string());
            }
            other => panic!("wrong refusal: {other}"),
        }
    }

    #[test]
    fn a_shard_with_another_split_count_is_refused() {
        let d = good_set("count");
        write_shard(&d, 1, 1, 4, 4, &["c"]);
        match refusal(&d) {
            LoadError::SplitKey { path, key, .. } => {
                assert_eq!(key, "split.count");
                assert_eq!(path, shard_path(&d, 1).display().to_string());
            }
            other => panic!("wrong refusal: {other}"),
        }
    }

    #[test]
    fn the_tensor_counts_must_sum_to_the_declared_total() {
        let d = good_set("total");
        write_shard(&d, 0, 0, 3, 5, &["a", "b"]);
        match refusal(&d) {
            LoadError::SplitKey { path, key, .. } => {
                assert_eq!(key, "split.tensors.count");
                assert_eq!(path, shard_path(&d, 0).display().to_string());
            }
            other => panic!("wrong refusal: {other}"),
        }
    }

    #[test]
    fn a_shard_out_of_place_is_refused() {
        let d = good_set("place");
        write_shard(&d, 2, 1, 3, 4, &["d"]);
        match refusal(&d) {
            LoadError::SplitKey { path, key, .. } => {
                assert_eq!(key, "split.no");
                assert_eq!(path, shard_path(&d, 2).display().to_string());
            }
            other => panic!("wrong refusal: {other}"),
        }
    }

    /// Page faults the calling thread has taken, minor and major.
    fn thread_faults() -> u64 {
        // SAFETY: `rusage` is a C struct of integers; all-zero bytes are a value of it.
        let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
        // SAFETY: `usage` is a live, writable `rusage` for the duration of the call.
        let rc = unsafe { libc::getrusage(libc::RUSAGE_THREAD, &mut usage) };
        assert_eq!(rc, 0, "getrusage(RUSAGE_THREAD)");
        u64::try_from(usage.ru_minflt + usage.ru_majflt).unwrap()
    }

    /// A file refused by the strict pass costs a populating open its header's
    /// pages, not its data's: the populate waits for the validation.
    #[test]
    fn a_refused_file_is_not_populated() {
        const DATA: usize = 64 << 20;
        let d = set_dir("populate");
        let p = d.join("refused.gguf");
        // Type id 99 is no `GgmlType`, so the strict pass refuses the file.
        let kvs = [("general.architecture", Kv::Str("test"))];
        write_gguf_as(&p, &kvs, &["a"], 99, DATA);
        let before = thread_faults();
        let r = Gguf::open_with(&p, true);
        let faults = thread_faults() - before;
        std::fs::remove_dir_all(&d).unwrap();
        assert!(
            matches!(r, Err(LoadError::UnsupportedType { .. })),
            "the file must be refused by the strict pass"
        );
        // SAFETY: `sysconf` reads a static system value and touches no memory of ours.
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        let pages = u64::try_from(DATA).unwrap() / u64::try_from(page).unwrap();
        println!(
            "populating open of a refused file: {faults} page faults, data section {pages} pages"
        );
        // Fault-around maps at most 16 pages per fault, so populating the data
        // costs at least pages/16 faults; the header alone costs a handful.
        assert!(
            faults < pages / 256,
            "{faults} faults: the data section ({pages} pages) was populated before the header was refused"
        );
    }
}
