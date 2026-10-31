//! gguf — GGUF v3 reader for bloomery stage 1 (round 1-1).
//!
//! Turns a GGUF file into named, shaped, typed tensors backed by an mmap,
//! plus the scalar reference dequantizer (`quant`) the oracle gate compares
//! against ggml itself. No attention, no MoE, no GPU — later rounds build on
//! `TensorInfo::dims` staying in **ggml's own `ne[]` order** (ne[0] = row
//! length, the contiguous axis), which every downstream comparison assumes.
//!
//! Two entry points share one header parser: `Gguf::open*` (strict — every
//! tensor must be sized by this engine's `GgmlType`, or by the caller's
//! [`PrivateType`] table in [`Gguf::open_private`], and sit inside the file
//! without overlapping another tensor;
//! bloomery's own value types open through a [`Split`] alone)
//! and [`inventory_of`] (header-only — unknown type ids are carried as
//! numbers with ggml's size table, so files with types the engine cannot
//! dequantize still inventory). A model split across shards opens as a
//! [`Split`]: one strict reader per shard, validated as one model. The
//! [`write`] module writes the files this reader opens. The [`qtype`] module
//! holds one row of facts per type, ggml's and bloomery's own (sizes, name,
//! decode, activation), which `quant`, [`ggml_type_info`] and the other
//! crates read.
//!
//! The [`digest`] module owns shared SHA-256 file, bounded-reader and structured feeds used by
//! checkpoint, reference, fixture and vocabulary consumers. Each consumer keeps its format framing.
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
//! Metadata keys and tensor names must be unique, including in header-only inventories.
//!
//! Errors, not panics: every file-derived value is bounds-checked — a
//! truncated or hostile file yields [`LoadError`].

pub mod arch;
pub mod card;
pub mod digest;
pub mod engram;
pub mod facts;
pub mod iq_tables;
pub mod qtype;
pub mod quant;
pub mod v41;
pub mod write;

use qtype::OWN_ROWS;
pub use qtype::{HostAct, QROWS, QRow};
pub use quant::{
    ActivationFormat, GgmlType, QuantError, activation_format, dequant_row, quantize_activations,
    quantize_row_q8_2_x4_roundtrip, quantize_row_q8_k_roundtrip,
};

use std::collections::HashMap;
use std::fs::File;
use std::ops::Range;
use std::path::{Path, PathBuf};

use memmap2::{Mmap, MmapMut};

/// The metadata key that names a file's architecture ([`Gguf::architecture`]).
pub const GENERAL_ARCHITECTURE: &str = "general.architecture";

/// The metadata key that sets a file's data alignment: a u32, 32 when absent.
pub const GENERAL_ALIGNMENT: &str = "general.alignment";

/// One GGUF metadata value. Tags are the on-disk value-type ids:
/// 0 u8, 1 i8, 2 u16, 3 i16, 4 u32, 5 i32, 6 f32, 7 bool, 8 string,
/// 9 array, 10 u64, 11 i64, 12 f64. An array's elements share one tag, and
/// it is not 9: ggml reads no array of arrays.
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
    /// both [`GgmlType`] and [`ggml_type_info`]. A product past u64 is
    /// [`LoadError::TooLarge`], never a value here.
    pub nbytes: Option<u64>,
}

/// A tensor type id this build does not know, sized by the caller for
/// [`Gguf::open_private`]: `blck` values in `size` bytes, as ggml's
/// `blck_size` and `type_size` size its own types. It is for layouts of our
/// own whose id no ggml table holds, so every other reader refuses the file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrivateType {
    pub id: u32,
    pub blck: u64,
    pub size: u64,
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
    #[error("duplicate metadata key {key:?} at offset {second}; first declared at offset {first}")]
    DuplicateMetadata {
        key: String,
        first: u64,
        second: u64,
    },
    #[error(
        "tensor {first:?} range {first_range:?} overlaps tensor {second:?} range {second_range:?} (relative to data base)"
    )]
    OverlappingTensors {
        first: String,
        first_range: Range<u64>,
        second: String,
        second_range: Range<u64>,
    },
    /// An array whose element tag is 9: ggml reads arrays of scalars and of
    /// strings only, and aborts on an array of arrays.
    #[error("metadata {key:?} is an array of arrays; GGUF arrays hold scalars or strings")]
    NestedArray { key: String },
    #[error("{} is {value}: the alignment is a nonzero u32", GENERAL_ALIGNMENT)]
    Alignment { value: String },
    #[error("tensor {name:?}: n_dims {n} outside 1..=4")]
    BadDims { name: String, n: u32 },
    #[error("tensor {name:?}: dims must all be >= 1, got {dims:?}")]
    BadDimValue { name: String, dims: Vec<u64> },
    #[error("tensor {name:?}: ggml type {ty} has no size in this build's table")]
    UnsupportedType { name: String, ty: GgmlType },
    #[error("tensor {name:?}: ne[0] {ne0} is not a multiple of the block size {blck}")]
    UnalignedRow { name: String, ne0: u64, blck: u64 },
    /// A tensor whose byte count passes u64: no file holds it.
    #[error("tensor {name:?}: {ty} dims {dims:?} hold more bytes than a u64 counts")]
    TooLarge {
        name: String,
        dims: Vec<u64>,
        ty: GgmlType,
    },
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
    /// A tensor name repeated at two shard paths or two descriptor offsets in one file.
    #[error("tensor {name:?} appears twice: in {first} and in {second}")]
    DuplicateTensor {
        name: String,
        first: String,
        second: String,
    },
    /// A [`PrivateType`] entry whose id this build already sizes: a file
    /// tagged with it would open everywhere as that ggml type.
    #[error("private type id {id} is {name}: a private id must be unknown to this build")]
    KnownPrivateType { id: u32, name: &'static str },
    /// A [`PrivateType`] entry that cannot size a tensor, or one listed twice.
    #[error("private type id {id}: {detail}")]
    BadPrivateType { id: u32, detail: &'static str },
    /// The tensor scales beside a tensor of a type that reads them ([`tensor_scales`]).
    #[error("tensor {name}: {detail}")]
    Scale { name: String, detail: String },
    /// A read of bytes no run of the mapping holds: a tensor of a located split that is a stack
    /// or is made, or a name the split does not have ([`Split::data`]).
    #[error("tensor {name:?}: {what}")]
    Unread { name: String, what: &'static str },
    /// A [`Split::located`] argument that does not describe a split.
    #[error("located split: {detail}")]
    Located { detail: String },
    /// The Engram companion file of a set ([`engram::attach_companion`]).
    #[error("engram companion {path}: {detail}")]
    Companion { path: String, detail: String },
    /// A header fact the file states wrongly or twice ([`facts`]).
    #[error("{0}")]
    Fact(Box<facts::FactError>),
}

/// Boxed: a fact's error names its keys and values, and a large error variant
/// would widen every `Result` that carries a `LoadError`.
impl From<facts::FactError> for LoadError {
    fn from(e: facts::FactError) -> Self {
        LoadError::Fact(Box::new(e))
    }
}

/// A parsed, mmap-backed GGUF file.
pub struct Gguf {
    // Kept alive for the mapping's lifetime; the mapping borrows the fd. A
    // header alone ([`Split::header_only`]) has no file.
    _file: Option<File>,
    map: Backing,
    meta: Vec<(String, Value)>,
    /// The file offset of each `meta` pair's value first byte, one per pair
    /// in the same order.
    offsets: Vec<u64>,
    tensors: Vec<TensorInfo>,
    /// File offset where the (aligned) tensor data section starts.
    data_base: u64,
    alignment: u64,
    /// The tensors of a located shard ([`Split::located`]) that no run of its mapping holds,
    /// each with why; empty for a file.
    unread: HashMap<String, &'static str>,
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
    /// The header's bytes and no tensor's ([`Split::header_only`]).
    Header(Vec<u8>),
}

impl std::ops::Deref for Backing {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            Backing::File(m) => m,
            Backing::Anon(m) => m,
            Backing::Header(h) => h,
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
        Self::open_sized(path.as_ref(), weights, &[], false)
    }

    /// `open_backed`, with the tensors whose type id `private` lists sized by
    /// that entry; they keep `ty = GgmlType::Unknown(id)`. Any other type
    /// this build cannot size is still [`LoadError::UnsupportedType`]. An
    /// entry for an id this build sizes is [`LoadError::KnownPrivateType`],
    /// one that sizes nothing or repeats an id [`LoadError::BadPrivateType`],
    /// both before the file is opened.
    pub fn open_private(
        path: impl AsRef<Path>,
        weights: Weights,
        private: &[PrivateType],
    ) -> Result<Gguf, LoadError> {
        check_private(private)?;
        Self::open_sized(path.as_ref(), weights, private, false)
    }

    /// The open every entry point shares: `private` is empty except in
    /// [`Gguf::open_private`], and `own` (bloomery's own value types,
    /// [`OWN_ROWS`]) holds only for a [`Split`]'s shards.
    fn open_sized(
        path: &Path,
        weights: Weights,
        private: &[PrivateType],
        own: bool,
    ) -> Result<Gguf, LoadError> {
        let file = File::open(path)?;
        let len = file.metadata()?.len();
        let file_map = map_read_only(&file)?;
        let h = parse_header(&file_map, len)?;
        let tensors = strict_tensors(h.tensors, h.data_base, len, private, own)?;
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
            _file: Some(file),
            map,
            meta: h.meta,
            offsets: h.offsets,
            tensors,
            data_base: h.data_base,
            alignment: h.alignment,
            unread: HashMap::new(),
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

    /// Every metadata pair, in file order.
    pub fn keys(&self) -> &[(String, Value)] {
        &self.meta
    }

    /// Raw metadata lookup by exact key.
    pub fn value(&self, key: &str) -> Option<&Value> {
        self.meta.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    /// The file offset of `key`'s value first byte: for a scalar the value
    /// itself, for a string its length prefix, for an array its element tag.
    /// Recorded while the header parsed, so no bytes are walked again; `None`
    /// when the header does not name `key`.
    pub fn value_offset(&self, key: &str) -> Option<u64> {
        let i = self.meta.iter().position(|(k, _)| k == key)?;
        Some(self.offsets[i])
    }

    /// `general.architecture` (e.g. "deepseek2").
    pub fn architecture(&self) -> Option<&str> {
        self.value(GENERAL_ARCHITECTURE).and_then(Value::as_str)
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
        if let Some(&what) = self.unread.get(&t.name) {
            return Err(LoadError::Unread {
                name: t.name.clone(),
                what,
            });
        }
        // A closure, so the name is cloned only for a refusal.
        let out_of_bounds = || LoadError::OutOfBounds {
            name: t.name.clone(),
            base: self.data_base,
            off: t.offset,
            nbytes: t.nbytes,
            len: self.map.len() as u64,
        };
        let start = (self.data_base as usize)
            .checked_add(t.offset as usize)
            .ok_or_else(out_of_bounds)?;
        let end = start
            .checked_add(t.nbytes as usize)
            .filter(|&end| end <= self.map.len())
            .ok_or_else(out_of_bounds)?;
        Ok(&self.map[start..end])
    }

    /// The whole mapping, file offset `o` at index `o`. Index 0 sits on a page
    /// boundary (the kernel places a mapping on one), so a sub-slice of whole
    /// pages is what a caller locks in RAM (`mlock`) or asks the residency of
    /// (`mincore`); [`Gguf::data`] is one tensor's slice of it.
    pub fn mapping(&self) -> &[u8] {
        &self.map
    }

    /// Whether [`Gguf::mapping`] is the file's own read-only mapping
    /// ([`Weights::Mapped`]) rather than an anonymous copy
    /// ([`Weights::Resident`]): only on the file's mapping does
    /// `MADV_DONTNEED` drop page-table entries the next touch re-reads from
    /// the file — on the copy it zeroes the bytes.
    #[must_use]
    pub fn file_backed(&self) -> bool {
        matches!(self.map, Backing::File(_))
    }

    /// The file's bytes before its data base: the header and its padding. A
    /// file that ends inside that padding (it holds no tensors) gives every
    /// byte it has.
    pub fn header_bytes(&self) -> &[u8] {
        let end = usize::try_from(self.data_base).map_or(self.map.len(), |b| b.min(self.map.len()));
        &self.map[..end]
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
/// the headers only, sizing bloomery's own value types too (`qtype::OWN_ROWS`).
/// Metadata is the first shard's: the splitter writes the
/// model's keys there alone (gguf-split.cpp:242).
pub struct Split {
    shards: Vec<Gguf>,
    paths: Vec<PathBuf>,
    /// Tensor name to (shard, index in that shard's table).
    by_name: HashMap<String, (usize, usize)>,
    /// Every tensor's place, for a split made by [`Split::located`]; empty for any other.
    places: HashMap<String, Place>,
}

/// How a stack's matrices lie in their bytes. A tensor's type says how a row encodes its values;
/// the layout says which bytes a row is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StackLayout {
    /// Row after row, each in the declared type's own blocks: one run a matrix.
    Rows,
    /// A matrix is two runs, its codes for every row and its block scales, wherever the file
    /// keeps them: side by side, or a layer's worth of other tensors apart.
    Planes,
}

/// Where a tensor of a [`Split`] keeps its bytes.
#[derive(Clone, Debug, PartialEq)]
pub enum Place {
    /// The declared type's bytes, at `range` (file offsets) of shard `shard`: [`Split::data`] is
    /// those bytes.
    File { shard: usize, range: Range<u64> },
    /// A stack of equal matrices, each read where the files keep it ([`Split::matrix`]);
    /// [`Split::data`] refuses the stack as one run. Matrix `e`'s codes are the `codes_per`
    /// bytes at file offset `codes_at[e]` of shard `codes_in[e]`. Under [`StackLayout::Planes`]
    /// its block scales are the `scales_per` bytes at `scales_at[e]` of shard `scales_in[e]`,
    /// and `scales[e]` is the factor its results are multiplied by (what the `.scale` tensor of
    /// the stack holds for it, one finite positive value an expert); under [`StackLayout::Rows`]
    /// those four are empty or 0. A run lies in any shard, apart from its matrix's other run and
    /// from the stack's other matrices; no two runs of a shard overlap, and no order of codes and
    /// scales is assumed. The first matrix's codes shard lists the stack in its table.
    Stack {
        codes_in: Vec<usize>,
        codes_at: Vec<u64>,
        scales_in: Vec<usize>,
        scales_at: Vec<u64>,
        codes_per: u64,
        scales_per: u64,
        layout: StackLayout,
        scales: Vec<f32>,
    },
    /// Bytes a step makes of the file's tensors (a cast, a `+1`, a normalize): no file run holds
    /// them, so [`Split::data`] refuses them and the part source makes them.
    Made,
}

/// One matrix of a [`Place::Stack`], as its file holds it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StackMatrix<'a> {
    pub codes: &'a [u8],
    /// The block-scale plane; empty for [`StackLayout::Rows`].
    pub block_scales: &'a [u8],
}

const UNREAD_STACK: &str =
    "is a stack of matrices read one at a time (Split::matrix), not one run of bytes";
const UNREAD_MADE: &str = "is made by a step from the file's tensors: no run of any file holds it";

impl Place {
    /// The shard whose table lists the tensor: a file run's, a stack's first matrix's codes
    /// shard; `None` for a made tensor, which no shard holds.
    fn listed_in(&self) -> Option<usize> {
        match self {
            Place::File { shard, .. } => Some(*shard),
            Place::Stack { codes_in, .. } => Some(codes_in.first().copied().unwrap_or(0)),
            Place::Made => None,
        }
    }

    /// Whether this place describes `nbytes` declared bytes inside the files of `lens` bytes
    /// (by shard; unread for [`Place::Made`]): the refusal names what is wrong.
    fn check(&self, name: &str, lens: &[u64], nbytes: u64) -> Result<(), String> {
        let len = |shard: usize| {
            lens.get(shard)
                .copied()
                .ok_or_else(|| format!("{name}: shard {shard}, of {} files", lens.len()))
        };
        match self {
            Place::Made => Ok(()),
            Place::File { shard, range } => {
                let file_len = len(*shard)?;
                if range.start > range.end || range.end > file_len {
                    return Err(format!(
                        "{name}: bytes {range:?} lie outside shard {shard}'s {file_len}"
                    ));
                }
                if range.end - range.start != nbytes {
                    return Err(format!(
                        "{name}: {} bytes at {range:?}, the declared {nbytes}",
                        range.end - range.start
                    ));
                }
                Ok(())
            }
            Place::Stack {
                codes_in,
                codes_at,
                scales_in,
                scales_at,
                codes_per,
                scales_per,
                layout,
                scales,
            } => {
                let n = codes_at.len() as u64;
                if n == 0 || *codes_per == 0 {
                    return Err(format!("{name}: a stack of no matrix, or of no bytes"));
                }
                if codes_in.len() != codes_at.len() || scales_in.len() != scales_at.len() {
                    return Err(format!(
                        "{name}: {} codes runs in {} shards, {} scale runs in {}",
                        codes_at.len(),
                        codes_in.len(),
                        scales_at.len(),
                        scales_in.len()
                    ));
                }
                match layout {
                    StackLayout::Rows => {
                        if !scales_at.is_empty() || *scales_per != 0 || !scales.is_empty() {
                            return Err(format!(
                                "{name}: a stack of rows has no scale plane and no tensor scales"
                            ));
                        }
                    }
                    StackLayout::Planes => {
                        if scales_at.len() as u64 != n || *scales_per == 0 {
                            return Err(format!(
                                "{name}: {n} matrices, {} scale planes of {scales_per} bytes",
                                scales_at.len()
                            ));
                        }
                        if scales.len() as u64 != n {
                            return Err(format!("{name}: {n} matrices, {} scales", scales.len()));
                        }
                    }
                }
                if let Some((e, s)) = scales
                    .iter()
                    .enumerate()
                    .find(|(_, s)| !(s.is_finite() && **s > 0.0))
                {
                    return Err(format!(
                        "{name}: the scale of matrix {e} is {s}, not finite and positive"
                    ));
                }
                let per = codes_per.checked_add(*scales_per);
                if per.and_then(|per| per.checked_mul(n)) != Some(nbytes) {
                    return Err(format!(
                        "{name}: {n} matrices of {codes_per} + {scales_per} bytes, the declared {nbytes}"
                    ));
                }
                let mut runs: Vec<(usize, u64, u64)> = codes_in
                    .iter()
                    .zip(codes_at)
                    .map(|(&shard, &at)| (shard, at, *codes_per))
                    .chain(
                        scales_in
                            .iter()
                            .zip(scales_at)
                            .map(|(&shard, &at)| (shard, at, *scales_per)),
                    )
                    .collect();
                runs.sort_unstable();
                for (i, &(shard, start, run)) in runs.iter().enumerate() {
                    let file_len = len(shard)?;
                    let end = start
                        .checked_add(run)
                        .filter(|&e| e <= file_len)
                        .ok_or_else(|| {
                            format!(
                                "{name}: the run at {start} runs outside shard {shard}'s {file_len}"
                            )
                        })?;
                    if runs
                        .get(i + 1)
                        .is_some_and(|&(s, next, _)| s == shard && next < end)
                    {
                        return Err(format!("{name}: the run at {start} overlaps the next"));
                    }
                }
                Ok(())
            }
        }
    }
}

impl Split {
    /// Open the model whose first shard is `first`. A file without
    /// `split.count` is a one-shard model. A set whose Engram tables are
    /// in a file of their own gets that companion as a further shard
    /// ([`engram::attach_companion`]): its tensors are the split's, and
    /// `split.tensors.count` still counts the model shards' alone.
    pub fn open(first: impl AsRef<Path>) -> Result<Split, LoadError> {
        let mut split = Split::open_model(first.as_ref())?;
        engram::attach_companion(&mut split)?;
        Ok(split)
    }

    /// The model shards `split.count` names, checked against each other.
    fn open_model(first: &Path) -> Result<Split, LoadError> {
        let first = first.to_path_buf();
        let head = open_shard(&first)?;
        let Some(count) = head.value(SPLIT_COUNT) else {
            return Split::assemble(vec![head], vec![first], None);
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
        Split::assemble(shards, paths, None)
    }

    /// The header `layout` lays out, as one shard named `path`, holding no tensor's bytes: the
    /// model of a checkpoint folder whose tensors a load reads through its part source
    /// (`model::part::Source`) while its plan, tokenizer and chat surface read this header as
    /// they read a split's. Every tensor is checked as a shard's is (its type, block alignment
    /// and size, against the file `layout` would write), and every read of a tensor's bytes is
    /// refused by name ([`Gguf::data`]: past the data the shard holds).
    pub fn header_only(path: PathBuf, layout: &write::Layout) -> Result<Split, LoadError> {
        let bytes = layout.header_bytes().to_vec();
        let h = parse_header(&bytes, layout.file_len())?;
        let tensors = strict_tensors(h.tensors, h.data_base, layout.file_len(), &[], true)?;
        let shard = Gguf {
            _file: None,
            map: Backing::Header(bytes),
            meta: h.meta,
            offsets: h.offsets,
            tensors,
            data_base: h.data_base,
            alignment: h.alignment,
            unread: HashMap::new(),
        };
        Split::assemble(vec![shard], vec![path], None)
    }

    /// The header `layout` lays out, over the files that hold the tensors' bytes where they
    /// are: shard `i` is `files[i]`, mapped read-only as a GGUF shard is, and `places[i]` is
    /// where the `i`-th tensor of `layout` keeps its bytes ([`Place`]). The keys are shard 0's,
    /// so `path`, which names the split ([`Split::shard_path`] of shard 0), is `files[0]`. Every
    /// tensor is checked as a shard's is (its type, block alignment and size, against the file
    /// `layout` would write), then against its place: a file run must lie inside its file and
    /// be as long as the declared bytes, a stack's matrices must lie inside their file, not
    /// overlap, and add up to the declared bytes. A [`Place::File`] tensor's `info.offset` is its
    /// file offset (`data_base` is 0), a stack's or a made tensor's is `u64::MAX`: no file offset
    /// names it. With no file the split is [`Split::header_only`], every tensor of it
    /// [`Place::Made`].
    pub fn located(
        path: PathBuf,
        layout: &write::Layout,
        files: &[PathBuf],
        places: Vec<Place>,
    ) -> Result<Split, LoadError> {
        let bad = |detail: String| LoadError::Located { detail };
        let bytes = layout.header_bytes().to_vec();
        let h = parse_header(&bytes, layout.file_len())?;
        let tensors = strict_tensors(h.tensors, h.data_base, layout.file_len(), &[], true)?;
        if places.len() != tensors.len() {
            return Err(bad(format!(
                "{} places for the header's {} tensors",
                places.len(),
                tensors.len()
            )));
        }
        if files.is_empty() {
            let kept = tensors
                .iter()
                .zip(&places)
                .find_map(|(t, p)| Some((&t.name, p.listed_in()?)));
            if let Some((name, shard)) = kept {
                return Err(bad(format!("{name}: shard {shard}, of 0 files")));
            }
            return Split::header_only(path, layout);
        }
        if files.first().is_some_and(|first| *first != path) {
            return Err(bad(format!(
                "the split is named {}, its first file is {}",
                path.display(),
                files[0].display()
            )));
        }
        let mut handles = Vec::with_capacity(files.len());
        for f in files {
            let open = || -> Result<(File, Mmap), LoadError> {
                let file = File::open(f)?;
                let map = map_read_only(&file)?;
                Ok((file, map))
            };
            handles.push(open().map_err(|e| match e {
                LoadError::Io(io) if io.kind() == std::io::ErrorKind::NotFound => {
                    LoadError::MissingShard {
                        path: f.display().to_string(),
                    }
                }
                other => LoadError::Shard {
                    path: f.display().to_string(),
                    source: Box::new(other),
                },
            })?);
        }
        let lens: Vec<u64> = handles.iter().map(|(_, m)| m.len() as u64).collect();
        let n = files.len();
        let mut tables: Vec<Vec<TensorInfo>> = vec![vec![]; n];
        let mut unread: Vec<HashMap<String, &'static str>> = vec![HashMap::new(); n];
        let mut placed = HashMap::with_capacity(places.len());
        for (mut info, place) in tensors.into_iter().zip(places) {
            let name = info.name.clone();
            place.check(&name, &lens, info.nbytes).map_err(bad)?;
            let shard = place.listed_in().unwrap_or(0);
            match &place {
                Place::File { range, .. } => info.offset = range.start,
                Place::Stack { .. } => {
                    info.offset = u64::MAX;
                    unread[shard].insert(name.clone(), UNREAD_STACK);
                }
                Place::Made => {
                    info.offset = u64::MAX;
                    unread[0].insert(name.clone(), UNREAD_MADE);
                }
            }
            tables[shard].push(info);
            placed.insert(name, place);
        }
        let mut meta = Some((h.meta, h.offsets));
        let shards = tables
            .into_iter()
            .zip(unread)
            .zip(handles)
            .map(|((table, unread), (file, map))| {
                let (meta, offsets) = meta.take().unwrap_or_default();
                Gguf {
                    _file: Some(file),
                    map: Backing::File(map),
                    meta,
                    offsets,
                    tensors: table,
                    data_base: 0,
                    alignment: h.alignment,
                    unread,
                }
            })
            .collect();
        Split::assemble(shards, files.to_vec(), Some(placed))
    }

    /// Index every tensor by name, refusing a name that appears twice. `placed` is the places a
    /// located split was given; any other split keeps each tensor's run of its file, or
    /// [`Place::Made`] where the shard is a header alone.
    fn assemble(
        shards: Vec<Gguf>,
        paths: Vec<PathBuf>,
        placed: Option<HashMap<String, Place>>,
    ) -> Result<Split, LoadError> {
        let mut by_name: HashMap<String, (usize, usize)> = HashMap::new();
        let mut places = placed.unwrap_or_default();
        let given = !places.is_empty();
        for (s, g) in shards.iter().enumerate() {
            index_shard(&mut by_name, &mut places, &paths, s, g, !given)?;
        }
        Ok(Split {
            shards,
            paths,
            by_name,
            places,
        })
    }

    /// Add `shard`, the file at `path`, after the shards the split has: its tensors are the
    /// split's, and a name the split already holds is refused. The shard is no part of the
    /// model's `split.count` set.
    pub(crate) fn attach_shard(&mut self, shard: Gguf, path: PathBuf) -> Result<(), LoadError> {
        let s = self.shards.len();
        self.paths.push(path);
        index_shard(
            &mut self.by_name,
            &mut self.places,
            &self.paths,
            s,
            &shard,
            true,
        )?;
        self.shards.push(shard);
        Ok(())
    }

    /// Number of shards (1 for a single file).
    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    /// Whether tensors are supplied by a part source rather than this split's mappings: a header
    /// alone ([`Split::header_only`], or [`Split::located`] over no file).
    pub fn is_header_only(&self) -> bool {
        matches!(&self.shards[0].map, Backing::Header(_))
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

    /// Where tensor `name` keeps its bytes: the place a [`Split::located`] split was given it; in
    /// a file, the run of its bytes; in a header alone ([`Split::header_only`]), nowhere
    /// ([`Place::Made`]). `None` for a name the split does not have.
    pub fn place(&self, name: &str) -> Option<&Place> {
        self.places.get(name)
    }

    /// The bytes of tensor `name`, in its shard's mapping: [`Gguf::data`] of the tensor's shard. A
    /// stack or a made tensor of a located split is refused by name.
    pub fn data(&self, name: &str) -> Result<&[u8], LoadError> {
        let (shard, info) = self.find(name).ok_or_else(|| LoadError::Unread {
            name: name.to_string(),
            what: "is not a tensor of the split",
        })?;
        self.shards[shard].data(info)
    }

    /// The run of file bytes tensor `name` keeps: its shard and file offsets ([`Place::File`]). A
    /// stack or a made tensor of a located split is refused by name, as [`Split::data`] refuses
    /// it.
    pub fn run(&self, name: &str) -> Result<(usize, Range<u64>), LoadError> {
        let unread = |what| LoadError::Unread {
            name: name.to_string(),
            what,
        };
        match self.places.get(name) {
            Some(Place::File { shard, range }) => Ok((*shard, range.clone())),
            Some(Place::Stack { .. }) => Err(unread(UNREAD_STACK)),
            Some(Place::Made) => Err(unread(UNREAD_MADE)),
            None => Err(unread("is not a tensor of the split")),
        }
    }

    /// Matrix `e` of the stack `name` of a located split ([`Place::Stack`]): its runs where the
    /// file keeps them.
    pub fn matrix(&self, name: &str, e: usize) -> Result<StackMatrix<'_>, LoadError> {
        let unread = |what| LoadError::Unread {
            name: name.to_string(),
            what,
        };
        let Some(Place::Stack {
            codes_in,
            codes_at,
            scales_in,
            scales_at,
            codes_per,
            scales_per,
            ..
        }) = self.places.get(name)
        else {
            return Err(unread("is not a stack of a located split"));
        };
        if e >= codes_at.len() {
            return Err(unread("has no matrix of that index"));
        }
        // `located` checked every run inside its file.
        let run = |shard: usize, at: u64, len: u64| {
            let end = at.checked_add(len)?;
            self.shards[shard]
                .mapping()
                .get(usize::try_from(at).ok()?..usize::try_from(end).ok()?)
        };
        let codes = run(codes_in[e], codes_at[e], *codes_per);
        let block_scales = match scales_at.get(e) {
            Some(&at) => run(scales_in[e], at, *scales_per),
            None => Some(&[][..]),
        };
        codes
            .zip(block_scales)
            .map(|(codes, block_scales)| StackMatrix {
                codes,
                block_scales,
            })
            .ok_or_else(|| unread("has a matrix outside its file"))
    }

    /// Raw metadata lookup by exact key, in the first shard.
    pub fn value(&self, key: &str) -> Option<&Value> {
        self.shards[0].value(key)
    }

    /// The first shard's metadata pairs, in file order.
    pub fn iter_kv(&self) -> impl Iterator<Item = (&str, &Value)> {
        self.shards[0].iter_kv()
    }

    /// [`Gguf::keys`] of the first shard.
    pub fn keys(&self) -> &[(String, Value)] {
        self.shards[0].keys()
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

/// Index shard `s` (`g`, whose path is `paths[s]`) into `by_name`, refusing a tensor name the
/// index already holds; with `keep_runs`, record each tensor's run of its file in `places`
/// ([`Place::File`], or [`Place::Made`] for a header alone).
fn index_shard(
    by_name: &mut HashMap<String, (usize, usize)>,
    places: &mut HashMap<String, Place>,
    paths: &[PathBuf],
    s: usize,
    g: &Gguf,
    keep_runs: bool,
) -> Result<(), LoadError> {
    let in_file = !matches!(g.map, Backing::Header(_));
    for (i, t) in g.iter_tensors().enumerate() {
        if let Some(&(first, _)) = by_name.get(&t.name) {
            return Err(LoadError::DuplicateTensor {
                name: t.name.clone(),
                first: paths[first].display().to_string(),
                second: paths[s].display().to_string(),
            });
        }
        by_name.insert(t.name.clone(), (s, i));
        if keep_runs {
            // `open` checked the run inside its file.
            let start = g.data_base() + t.offset;
            places.insert(
                t.name.clone(),
                if in_file {
                    Place::File {
                        shard: s,
                        range: start..start + t.nbytes,
                    }
                } else {
                    Place::Made
                },
            );
        }
    }
    Ok(())
}

/// `-%05d-of-%05d.gguf` for shard `i` of `count`, the part `llama_split_path`
/// appends to the prefix (numbered from 1).
fn shard_suffix(i: u64, count: u64) -> String {
    format!("-{:05}-of-{count:05}.gguf", i + 1)
}

/// `file`'s read-only mapping, whole.
fn map_read_only(file: &File) -> std::io::Result<Mmap> {
    // SAFETY: the file is opened read-only and nothing maps it writable;
    // a concurrent truncation would surface as SIGBUS, the same contract
    // ik_llama.cpp's own mmap loader accepts.
    unsafe { memmap2::MmapOptions::new().map(file) }
}

/// The strict open of one shard, its error naming the file.
fn open_shard(path: &Path) -> Result<Gguf, LoadError> {
    let lazy = Weights::Mapped { populate: false };
    Gguf::open_sized(path, lazy, &[], true).map_err(|e| match e {
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
/// inventory for files the strict reader refuses — e.g. a file holding a q8_K
/// tensor — where the point is to know what is in the file, not to run it.
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
    /// The file the header was read from, where a tensor's bytes are read
    /// ([`facts::hash_constant`]); `None` for a header built in memory.
    pub path: Option<PathBuf>,
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
    let path = path.as_ref().to_path_buf();
    let file = File::open(&path)?;
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
        path: Some(path),
    })
}

/// ggml's `(type_name, blck_size, type_size)` for a type id: the row of every id [`GgmlType`]
/// names ([`QROWS`], which cite their block structs, and bloomery's own `OWN_ROWS`). Every other id returns `None` and
/// inventories with unknown size.
pub fn ggml_type_info(id: u32) -> Option<(&'static str, u64, u64)> {
    GgmlType::from_u32(id)
        .row()
        .map(|row| (row.name, row.blck, row.bytes))
}

/// `type_size * (ne[0]/blck) * ne[1] * …`, the bytes of a tensor of `tsz`-byte
/// blocks of `blck` values, or `None` when the product passes u64. The
/// product is `ggml_row_size`'s: integer division (a caller that needs whole
/// blocks checks ne[0] first), missing dims implicit 1s. The reader, the
/// writer and the model crate's fixture planner size every tensor here.
///
/// # Panics
/// On empty `dims` or a `blck` of 0.
pub fn tensor_bytes(blck: u64, tsz: u64, dims: &[u64]) -> Option<u64> {
    let row = tsz.checked_mul(dims[0] / blck)?;
    dims[1..].iter().try_fold(row, |acc, &d| acc.checked_mul(d))
}

/// The byte runs, in order, that hold the values `at` along dim `dim` of a tensor of `dims` held as
/// `tsz`-byte blocks of `blck` values along dim 0 ([`tensor_bytes`]' layout): along dim 0 one run a
/// row, whole blocks; along a dim past it one run of whole rows for each index of the dims past
/// `dim`. `None` for no dim `dim`, an empty `at` or one past the dim, a dim 0 or an `at` along it
/// off the blocks, and a byte count past u64.
#[must_use]
pub fn tensor_runs(
    blck: u64,
    tsz: u64,
    dims: &[u64],
    dim: usize,
    at: std::ops::Range<u64>,
) -> Option<Vec<std::ops::Range<u64>>> {
    let extent = *dims.get(dim)?;
    if blck == 0
        || tsz == 0
        || dims.contains(&0)
        || at.start >= at.end
        || at.end > extent
        || !dims[0].is_multiple_of(blck)
    {
        return None;
    }
    // Every run lies inside the tensor's bytes, so a count that fits u64 bounds every offset.
    tensor_bytes(blck, tsz, dims)?;
    let row = tsz * (dims[0] / blck);
    if dim == 0 {
        if !at.start.is_multiple_of(blck) || !at.end.is_multiple_of(blck) {
            return None;
        }
        let (lo, hi) = (at.start / blck * tsz, at.end / blck * tsz);
        let rows: u64 = dims[1..].iter().product();
        return Some((0..rows).map(|i| i * row + lo..i * row + hi).collect());
    }
    let inner = row * dims[1..dim].iter().product::<u64>();
    let outer: u64 = dims[dim + 1..].iter().product();
    let stride = inner * extent;
    Some(
        (0..outer)
            .map(|o| o * stride + at.start * inner..o * stride + at.end * inner)
            .collect(),
    )
}

/// The tensor scales `t`'s values are multiplied by: the f32 values of `<stem>.scale` beside
/// `<stem>.weight`, found through `find` (the file's tensors by name: its shard and entry), one
/// for a tensor and one an expert, in expert order, for a routed stack (three dims, mainline's
/// `.scale` `[n_expert]`). A type that reads them ([`GgmlType::reads_scale`]: NVFP4 and
/// bloomery's F8 and NVFP4 rows) carries them; `None` for any other type.
///
/// # Errors
///
/// A name without the `.weight` suffix, a missing scale, one of another type or count, and a
/// value that is not finite, as [`LoadError::Scale`]; a scale's bytes [`Gguf::data`] refuses.
pub fn tensor_scales<'g>(
    t: &TensorInfo,
    find: impl Fn(&str) -> Option<(&'g Gguf, &'g TensorInfo)>,
) -> Result<Option<Vec<f32>>, LoadError> {
    if !t.ty.reads_scale() {
        return Ok(None);
    }
    let refuse = |detail: String| LoadError::Scale {
        name: t.name.clone(),
        detail,
    };
    let stem = t.name.strip_suffix(".weight").ok_or_else(|| {
        refuse("a scaled tensor's name ends in .weight, its scale's in .scale".into())
    })?;
    let name = format!("{stem}.scale");
    let (g, s) =
        find(&name).ok_or_else(|| refuse(format!("its scale {name} is not in the file")))?;
    let (want, of) = match t.dims.as_slice() {
        [_, _, n] => (*n, "one f32 value an expert is a stack's scales"),
        _ => (1, "one f32 value is the tensor's scale"),
    };
    if s.ty != GgmlType::F32 || s.dims.iter().try_fold(1u64, |a, &d| a.checked_mul(d)) != Some(want)
    {
        return Err(refuse(format!(
            "its scale {name} is {} {:?}; {of}",
            s.ty, s.dims
        )));
    }
    let scales: Vec<f32> = g
        .data(s)?
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    if u64::try_from(scales.len()).ok() != Some(want) {
        return Err(refuse(format!("its scale {name} is not {want} f32 values")));
    }
    if let Some((i, v)) = scales.iter().enumerate().find(|(_, v)| !v.is_finite()) {
        return Err(refuse(format!(
            "its scale {name} [{i}] is {v}, not a finite value"
        )));
    }
    Ok(Some(scales))
}

/// The tensor `name` is the tensor scale of ([`tensor_scales`]): `<stem>.weight` of `split`
/// beside `<stem>.scale`, of a type that reads one. `None` for any other name.
#[must_use]
pub fn scaled_weight(split: &Split, name: &str) -> Option<String> {
    let weight = format!("{}.weight", name.strip_suffix(".scale")?);
    let (_, t) = split.find(&weight)?;
    t.ty.reads_scale().then_some(weight)
}

/// [`tensor_bytes`], or [`LoadError::TooLarge`] naming the tensor.
fn sized(name: &str, ty: GgmlType, blck: u64, tsz: u64, dims: &[u64]) -> Result<u64, LoadError> {
    tensor_bytes(blck, tsz, dims).ok_or_else(|| LoadError::TooLarge {
        name: name.to_string(),
        dims: dims.to_vec(),
        ty,
    })
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

/// The alignment `general.alignment` sets: a u32, as ggml reads
/// it, and not 0, which no offset can be rounded to; 32 when absent.
fn alignment_of(meta: &[(String, Value)]) -> Result<u64, LoadError> {
    match meta
        .iter()
        .find(|(k, _)| k == GENERAL_ALIGNMENT)
        .map(|(_, v)| v)
    {
        None => Ok(write::DEFAULT_ALIGNMENT),
        Some(Value::U32(a)) if *a != 0 => Ok(u64::from(*a)),
        Some(other) => Err(LoadError::Alignment {
            value: format!("{other:?}"),
        }),
    }
}

/// A [`PrivateType`] table's own refusals: an id this build sizes (by a row,
/// [`ggml_type_info`]), a block of no values
/// or no bytes, an id listed twice.
fn check_private(private: &[PrivateType]) -> Result<(), LoadError> {
    for (i, p) in private.iter().enumerate() {
        if let Some((name, _, _)) = ggml_type_info(p.id) {
            return Err(LoadError::KnownPrivateType { id: p.id, name });
        }
        let detail = if p.blck == 0 || p.size == 0 {
            "a block of 0 values or 0 bytes sizes nothing"
        } else if private[..i].iter().any(|q| q.id == p.id) {
            "is listed twice"
        } else {
            continue;
        };
        return Err(LoadError::BadPrivateType { id: p.id, detail });
    }
    Ok(())
}

/// The strict pass over a parsed header's tensors: every tensor must be sized
/// by this engine's `GgmlType` (bloomery's own value types only when `own`)
/// or by its entry in `private`, its first dim must be
/// block-aligned and its bytes must fit a u64; then every tensor must sit
/// inside the `len`-byte file without sharing bytes — the placement invariant
/// the coverage gate re-checks per tensor. A q8_1 or q8_K tensor is refused: they are ggml's
/// activation forms, which no quantizer writes to a file, though their rows size them.
fn strict_tensors(
    raw: Vec<RawTensorInfo>,
    data_base: u64,
    len: u64,
    private: &[PrivateType],
    own: bool,
) -> Result<Vec<TensorInfo>, LoadError> {
    let mut strict = Vec::with_capacity(raw.len());
    for t in raw {
        let ty = GgmlType::from_u32(t.type_id);
        let own_type = OWN_ROWS.iter().any(|r| r.ty == ty);
        if matches!(ty, GgmlType::Q8_1 | GgmlType::Q8_K) || (own_type && !own) {
            return Err(LoadError::UnsupportedType { name: t.name, ty });
        }
        let own = private.iter().find(|p| p.id == t.type_id);
        let blck =
            ty.blck_size()
                .or(own.map(|p| p.blck))
                .ok_or_else(|| LoadError::UnsupportedType {
                    name: t.name.clone(),
                    ty,
                })?;
        let tsz =
            ty.type_size()
                .or(own.map(|p| p.size))
                .ok_or_else(|| LoadError::UnsupportedType {
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
        let nbytes = sized(&t.name, ty, blck, tsz, &t.dims)?;
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
    let mut ranges: Vec<_> = strict.iter().collect();
    ranges.sort_unstable_by_key(|t| t.offset);
    for (first, second) in ranges.iter().zip(ranges.iter().skip(1)) {
        if second.offset < first.offset + first.nbytes {
            return Err(LoadError::OverlappingTensors {
                first: first.name.clone(),
                first_range: first.offset..first.offset + first.nbytes,
                second: second.name.clone(),
                second_range: second.offset..second.offset + second.nbytes,
            });
        }
    }
    Ok(strict)
}

/// What [`parse_header`] returns: the file's own statement of itself.
struct Header {
    version: u32,
    meta: Vec<(String, Value)>,
    /// The file offset of each `meta` pair's value first byte, aligned with
    /// `meta` by construction: one is pushed per pair.
    offsets: Vec<u64>,
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
    let mut offsets = Vec::new();
    let mut keys = HashMap::new();
    for _ in 0..n_kv {
        let key_at = rd.pos as u64;
        let key = rd.string()?;
        if let Some(first) = keys.insert(key.clone(), key_at) {
            return Err(LoadError::DuplicateMetadata {
                key,
                first,
                second: key_at,
            });
        }
        let tag = rd.u32()?;
        // The value starts at the reader's position; `read_value` moves past it.
        let at = rd.pos as u64;
        let val = read_value(&mut rd, &key, tag)?;
        meta.push((key, val));
        offsets.push(at);
    }

    let mut tensors = Vec::new();
    let mut names = HashMap::new();
    for _ in 0..n_tensors {
        let name_at = rd.pos;
        let name = rd.string()?;
        if let Some(first) = names.insert(name.clone(), name_at) {
            return Err(LoadError::DuplicateTensor {
                name,
                first: format!("offset {first}"),
                second: format!("offset {name_at}"),
            });
        }
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
        let nbytes = ggml_type_info(type_id)
            .map(|(_, blck, tsz)| sized(&name, GgmlType::from_u32(type_id), blck, tsz, &dims))
            .transpose()?;
        tensors.push(RawTensorInfo {
            nbytes,
            name,
            dims,
            type_id,
            offset,
        });
    }

    // Alignment and the data base, mirroring ggml.c:31432-31447:
    // `general.alignment` (u32 KV) or 32, then round the end of the
    // tensor-info section up to it.
    let alignment = alignment_of(&meta)?;
    let header_end = rd.pos as u64;
    let data_base = header_end.div_ceil(alignment) * alignment;
    Ok(Header {
        version,
        meta,
        offsets,
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

/// The value of metadata pair `key`, of type `tag`. An array's elements are
/// never arrays, so the recursion is one level deep.
fn read_value(rd: &mut Reader<'_>, key: &str, tag: u32) -> Result<Value, LoadError> {
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
            // element tag is a scalar or string type, checked before the
            // count so an empty array is held to it too. The capacity is
            // capped so a lying count cannot pre-allocate gigabytes — a count
            // that outruns the file dies at Truncated when the elements
            // actually run out.
            let elem_tag = rd.u32()?;
            match elem_tag {
                0..=8 | 10..=12 => {}
                9 => {
                    return Err(LoadError::NestedArray {
                        key: key.to_string(),
                    });
                }
                other => return Err(LoadError::BadValueType(other)),
            }
            let n = rd.u64()?;
            let mut items = Vec::with_capacity(n.min(4096) as usize);
            for _ in 0..n {
                items.push(read_value(rd, key, elem_tag)?);
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
    use super::{GENERAL_ARCHITECTURE, Gguf, LoadError, Place, Split, Value};
    use crate::write::{TensorDecl, write_file};
    use std::path::{Path, PathBuf};

    #[test]
    fn duplicate_header_keys_are_refused() {
        for values in [[1, 2], [2, 1]] {
            let layout = crate::write::Layout::new(
                &[
                    kv("test.block_count", Value::U32(values[0])),
                    kv("test.block_counX", Value::U32(values[1])),
                ],
                vec![],
            )
            .unwrap();
            let mut bytes = layout.header_bytes().to_vec();
            let key = b"test.block_counX";
            let second = bytes.windows(key.len()).position(|w| w == key).unwrap();
            bytes[second + key.len() - 1] = b't';
            let Err(e) = super::parse_header(&bytes, layout.file_len()) else {
                panic!("metadata key uniqueness must not depend on declaration order")
            };
            let text = e.to_string();
            assert!(
                text.contains("test.block_count")
                    && text.contains("duplicate")
                    && text.contains("offset"),
                "{text}"
            );
        }
        let layout = gate_and_up();
        let mut bytes = layout.header_bytes().to_vec();
        let second = bytes.windows(4).position(|w| w == b"up__").unwrap();
        bytes[second..second + 4].copy_from_slice(b"gate");
        let Err(e) = super::parse_header(&bytes, layout.file_len()) else {
            panic!("a repeated tensor name must be refused before building the name table")
        };
        assert!(
            e.to_string().contains("gate") && e.to_string().contains("twice"),
            "{e}"
        );
    }

    #[test]
    fn overlapping_tensor_ranges_are_refused() {
        let layout = gate_and_up();
        let strict_at = |offsets: [u64; 2]| {
            let mut h = super::parse_header(layout.header_bytes(), layout.file_len()).unwrap();
            for (t, off) in h.tensors.iter_mut().zip(offsets) {
                t.offset = off;
            }
            super::strict_tensors(h.tensors, h.data_base, layout.file_len(), &[], false)
        };
        for offsets in [[0, 0], [0, 8], [8, 0]] {
            let Err(e) = strict_at(offsets) else {
                panic!("overlapping tensor data must be refused in either declaration order")
            };
            let text = e.to_string();
            assert!(
                text.contains("gate")
                    && text.contains("up__")
                    && text.contains("overlap")
                    && text.contains(".."),
                "{text}"
            );
        }
        for offsets in [[0, 16], [32, 0]] {
            assert!(
                strict_at(offsets).is_ok(),
                "adjacent and disjoint tensors must still open"
            );
        }
    }

    /// A header of two 16-byte f32 tensors, `gate` and `up__`.
    fn gate_and_up() -> crate::write::Layout {
        crate::write::Layout::new(
            &[],
            ["gate", "up__"]
                .into_iter()
                .map(|name| TensorDecl {
                    name: name.into(),
                    dims: vec![4],
                    type_id: 0,
                    nbytes: 16,
                })
                .collect(),
        )
        .unwrap()
    }

    /// A metadata pair as the synthetic shards carry it.
    fn kv(key: &str, v: Value) -> (String, Value) {
        (key.to_string(), v)
    }

    /// A GGUF v3 file with `kvs` and one 4-value F32 tensor of zeros per name,
    /// 32 bytes apart — a few hundred bytes the strict reader opens.
    fn write_gguf(path: &Path, kvs: &[(String, Value)], tensors: &[&str]) {
        let tensors: Vec<(TensorDecl, Vec<u8>)> = tensors
            .iter()
            .map(|name| {
                let t = TensorDecl {
                    name: name.to_string(),
                    dims: vec![4],
                    type_id: 0,
                    nbytes: 16,
                };
                (t, vec![0; 16])
            })
            .collect();
        write_file(path, kvs, &tensors);
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
            kvs.push(kv(GENERAL_ARCHITECTURE, Value::String("test".into())));
        }
        kvs.push(kv("split.no", Value::U16(no)));
        kvs.push(kv("split.tensors.count", Value::I32(total)));
        kvs.push(kv("split.count", Value::U16(count)));
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
        let Err(e) = Split::open(shard_path(dir, 0)) else {
            panic!("the split in {} must be refused", dir.display())
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
        let base = s.shard(i).unwrap().data_base() + t.offset;
        let run = Place::File {
            shard: 1,
            range: base..base + 16,
        };
        assert_eq!(s.place("c"), Some(&run));
        assert_eq!(s.place("e"), None);
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
            &[kv(GENERAL_ARCHITECTURE, Value::String("test".into()))],
            &["a", "b"],
        );
        let s = Split::open(&p).unwrap();
        assert_eq!((s.shard_count(), s.tensor_count()), (1, 2));
        std::fs::remove_dir_all(&d).unwrap();
    }

    /// A layout's header alone reads as the file the layout writes reads, and holds no tensor
    /// bytes: every read of one is refused by name.
    #[test]
    fn a_header_alone_reads_as_its_file_and_refuses_every_byte() {
        let d = set_dir("header");
        let p = d.join("whole.gguf");
        let kvs = [kv(GENERAL_ARCHITECTURE, Value::String("test".into()))];
        write_gguf(&p, &kvs, &["a", "b"]);
        let decl = |name: &str| TensorDecl {
            name: name.into(),
            dims: vec![4],
            type_id: 0,
            nbytes: 16,
        };
        let layout = crate::write::Layout::new(&kvs, vec![decl("a"), decl("b")]).unwrap();
        let file = Split::open(&p).unwrap();
        let head = Split::header_only(d.clone(), &layout).unwrap();
        assert_eq!(head.keys(), file.keys());
        let infos = |s: &Split| {
            s.iter_tensors()
                .map(|(i, t)| (i, t.name.clone(), t.dims.clone(), t.ty, t.offset, t.nbytes))
                .collect::<Vec<_>>()
        };
        assert_eq!(infos(&head), infos(&file));
        assert_eq!(head.place("a"), Some(&Place::Made));
        assert_eq!(head.shard_path(0), Some(d.as_path()));
        let shard = head.shard(0).unwrap();
        assert!(!shard.file_backed());
        for (_, t) in head.iter_tensors() {
            let e = shard.data(t).unwrap_err().to_string();
            assert!(e.contains(&format!("{:?}", t.name)), "{e}");
        }
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
        os::thread_faults().expect("getrusage(RUSAGE_THREAD)")
    }

    /// A file refused by the strict pass costs a populating open its header's
    /// pages, not its data's: the populate waits for the validation.
    #[cfg_attr(
        not(target_os = "linux"),
        ignore = "the Linux fact it reads: a per-thread page-fault count"
    )]
    #[test]
    fn a_refused_file_is_not_populated() {
        const DATA: usize = 64 << 20;
        let d = set_dir("populate");
        let p = d.join("refused.gguf");
        // Type id 99 is no `GgmlType`, so the strict pass refuses the file; no
        // table sizes it either, so the writer takes the byte count as stated.
        let kvs = [kv(GENERAL_ARCHITECTURE, Value::String("test".into()))];
        let big = TensorDecl {
            name: "a".to_string(),
            dims: vec![4],
            type_id: 99,
            nbytes: u64::try_from(DATA).unwrap(),
        };
        write_file(&p, &kvs, &[(big, vec![0; DATA])]);
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

#[cfg(test)]
mod private_tests {
    use super::{GgmlType, Gguf, LoadError, PrivateType, Split, Weights};
    use crate::write::{TensorDecl, write_file};
    use std::path::{Path, PathBuf};

    /// The private id the clauses tag with: bloomery's 1000 + Q3_K's 11.
    const TAG: u32 = 1011;
    const TABLE: [PrivateType; 1] = [PrivateType {
        id: TAG,
        blck: 256,
        size: 110,
    }];
    const LAZY: Weights = Weights::Mapped { populate: false };

    fn set_dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("gguf-private-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// A file of an F32 tensor, then `q` — 8 rows of 512 values, 1,760 bytes
    /// at Q3_K's 110 per 256 — tagged type id `id`. Returns its bytes of `q`.
    fn tagged(path: &Path, id: u32) -> Vec<u8> {
        let q: Vec<u8> = (0..1760u32).map(|i| (i * 7 + 3) as u8).collect();
        let f = TensorDecl {
            name: "f".to_string(),
            dims: vec![4],
            type_id: 0,
            nbytes: 16,
        };
        let t = TensorDecl {
            name: "q".to_string(),
            dims: vec![512, 8],
            type_id: id,
            nbytes: 1760,
        };
        write_file(path, &[], &[(f, vec![1; 16]), (t, q.clone())]);
        q
    }

    /// A q4_0, q4_1 or q1_0 tensor opens, sized by its row and typed `GgmlType::Q4_0`,
    /// `GgmlType::Q4_1` or `GgmlType::Q1_0`; a q8_1 or q8_K tensor (an activation form) is
    /// refused by name.
    #[test]
    fn sized_types_open_and_activation_forms_are_refused() {
        let d = set_dir("table");
        for (id, ty, ne0, bytes) in [
            (2u32, GgmlType::Q4_0, 64u64, 72u64),
            (3, GgmlType::Q4_1, 64, 80),
            (41, GgmlType::Q1_0, 128, 36),
        ] {
            let p = d.join(format!("t{id}.gguf"));
            let t = TensorDecl {
                name: "q".to_string(),
                dims: vec![ne0, 2],
                type_id: id,
                nbytes: bytes,
            };
            write_file(&p, &[], &[(t, vec![5; usize::try_from(bytes).unwrap()])]);
            let g = Gguf::open(&p).unwrap_or_else(|e| panic!("type {id} must open: {e}"));
            let q = g.tensor(0).unwrap();
            assert_eq!((q.ty, q.nbytes), (ty, bytes), "type {id}");
        }
        for (id, ne0, bytes) in [(9u32, 32u64, 36u64), (15, 256, 292)] {
            let p = d.join(format!("t{id}.gguf"));
            let t = TensorDecl {
                name: "k".to_string(),
                dims: vec![ne0],
                type_id: id,
                nbytes: bytes,
            };
            write_file(&p, &[], &[(t, vec![0; usize::try_from(bytes).unwrap()])]);
            let r = Gguf::open(&p);
            assert!(
                matches!(r, Err(LoadError::UnsupportedType { ref name, .. }) if name == "k"),
                "type {id} must be refused by the tensor's name"
            );
        }
        std::fs::remove_dir_all(&d).unwrap();
    }

    /// A file tagged with a private id opens only through a table that lists
    /// it: the strict opens and the split refuse it by the tensor's name,
    /// `open_private` sizes it from the entry and keeps the id, an id outside
    /// the table is still refused, an entry whose size makes the tensor's
    /// bytes pass u64 is refused by the tensor's name, and a table naming an
    /// id this build sizes, sizing nothing or repeating an id is refused
    /// before any file is read.
    #[test]
    fn a_private_type_opens_only_through_its_table() {
        let d = set_dir("tag");
        let p = d.join("tagged.gguf");
        let q = tagged(&p, TAG);
        match Gguf::open(&p) {
            Err(LoadError::UnsupportedType { name, ty }) => {
                assert_eq!((name.as_str(), ty), ("q", GgmlType::Unknown(TAG)));
            }
            other => panic!(
                "open must refuse the tag by name, got {:?}",
                other.map(|_| ())
            ),
        }
        match Split::open(&p) {
            Err(LoadError::Shard { path, source }) => {
                assert_eq!(path, p.display().to_string());
                assert!(
                    matches!(*source, LoadError::UnsupportedType { ref name, .. } if name == "q"),
                    "{source}"
                );
            }
            other => panic!(
                "Split::open must refuse the tag, got {:?}",
                other.map(|_| ())
            ),
        }
        let g = Gguf::open_private(&p, LAZY, &TABLE).unwrap();
        let t = g.find("q").unwrap();
        assert_eq!(
            (t.ty, t.nbytes, t.dims.as_slice()),
            (GgmlType::Unknown(TAG), 1760, &[512u64, 8][..])
        );
        assert_eq!(g.data(t).unwrap(), q.as_slice());
        let other = d.join("other.gguf");
        tagged(&other, TAG + 1);
        match Gguf::open_private(&other, LAZY, &TABLE) {
            Err(LoadError::UnsupportedType { name, ty }) => {
                assert_eq!((name.as_str(), ty), ("q", GgmlType::Unknown(TAG + 1)));
            }
            other => panic!(
                "an id outside the table must be refused, got {:?}",
                other.map(|_| ())
            ),
        }
        let missing = d.join("absent.gguf");
        for (id, name) in [(11, "q3_K"), (2, "q4_0")] {
            let table = [PrivateType {
                id,
                blck: 256,
                size: 110,
            }];
            match Gguf::open_private(&missing, LAZY, &table) {
                Err(LoadError::KnownPrivateType { id: got, name: n }) => {
                    assert_eq!((got, n), (id, name))
                }
                other => panic!(
                    "id {id} must be refused as ggml's {name}, got {:?}",
                    other.map(|_| ())
                ),
            }
        }
        let huge = [PrivateType {
            id: TAG,
            blck: 256,
            size: u64::MAX / 2,
        }];
        match Gguf::open_private(&p, LAZY, &huge) {
            Err(LoadError::TooLarge { name, ty, .. }) => {
                assert_eq!((name.as_str(), ty), ("q", GgmlType::Unknown(TAG)));
            }
            other => panic!(
                "a size past u64 must be refused, got {:?}",
                other.map(|_| ())
            ),
        }
        let empty = [PrivateType {
            id: TAG,
            blck: 0,
            size: 110,
        }];
        let twice = [TABLE[0], TABLE[0]];
        for table in [&empty[..], &twice[..]] {
            match Gguf::open_private(&missing, LAZY, table) {
                Err(LoadError::BadPrivateType { id, .. }) => assert_eq!(id, TAG),
                other => panic!("{table:?} must be refused, got {:?}", other.map(|_| ())),
            }
        }
        std::fs::remove_dir_all(&d).unwrap();
    }

    /// bloomery's own value types open as they did before their rows named
    /// them: through a split and nowhere else. The strict open refuses each
    /// by the tensor's name; `Split::open` sizes F8 rows one byte a
    /// value, MXFP8 33 bytes a 32 values, NVFP4 rows 9 bytes a 16 values,
    /// FP8_B128 132 bytes a 128 values, MLX_A4 and MLX_A4H 36 bytes a 64 values and
    /// MLX_A8 and MLX_A8H 68 bytes a 64 values,
    /// keeps each id and hands back
    /// the bytes as written; an id past them is refused by both. A private
    /// table cannot claim their ids.
    #[test]
    fn bloomerys_own_types_open_through_a_split() {
        let d = set_dir("own");
        let p = d.join("own.gguf");
        let f8: Vec<u8> = (0..320u32).map(|i| (i * 7) as u8).collect();
        let mx: Vec<u8> = (0..132u32).map(|i| (i * 5 + 1) as u8).collect();
        let fp: Vec<u8> = (0..180u32).map(|i| (i * 3 + 2) as u8).collect();
        let b8: Vec<u8> = (0..264u32).map(|i| (i * 11 + 4) as u8).collect();
        let a4: Vec<u8> = (0..72u32).map(|i| (i * 13 + 6) as u8).collect();
        let a8: Vec<u8> = (0..136u32).map(|i| (i * 17 + 8) as u8).collect();
        let a4h: Vec<u8> = (0..72u32).map(|i| (i * 19 + 10) as u8).collect();
        let a8h: Vec<u8> = (0..136u32).map(|i| (i * 23 + 12) as u8).collect();
        let decl = |name: &str, dims: Vec<u64>, ty: GgmlType, nbytes| TensorDecl {
            name: name.to_string(),
            dims,
            type_id: ty.as_u32(),
            nbytes,
        };
        let (f8_ty, mx_ty, fp_ty) = (GgmlType::F8E4M3_ROWS, GgmlType::MXFP8, GgmlType::NVFP4_ROWS);
        let b8_ty = GgmlType::FP8_B128;
        let (a4_ty, a8_ty) = (GgmlType::MLX_A4, GgmlType::MLX_A8);
        let (a4h_ty, a8h_ty) = (GgmlType::MLX_A4H, GgmlType::MLX_A8H);
        write_file(
            &p,
            &[],
            &[
                (decl("f8", vec![160, 2], f8_ty, 320), f8.clone()),
                (decl("mx", vec![64, 2], mx_ty, 132), mx.clone()),
                (decl("fp", vec![160, 2], fp_ty, 180), fp.clone()),
                (decl("b8", vec![128, 2], b8_ty, 264), b8.clone()),
                (decl("a4", vec![128, 1], a4_ty, 72), a4.clone()),
                (decl("a8", vec![128, 1], a8_ty, 136), a8.clone()),
                (decl("a4h", vec![128, 1], a4h_ty, 72), a4h.clone()),
                (decl("a8h", vec![128, 1], a8h_ty, 136), a8h.clone()),
            ],
        );
        assert!(
            matches!(Gguf::open(&p), Err(LoadError::UnsupportedType { ref name, ty }) if name == "f8" && ty == f8_ty)
        );
        for (name, ty, dims, bytes) in [
            ("mx", mx_ty, vec![64, 2], &mx),
            ("fp", fp_ty, vec![160, 2], &fp),
            ("b8", b8_ty, vec![128, 2], &b8),
            ("a4", a4_ty, vec![128, 1], &a4),
            ("a8", a8_ty, vec![128, 1], &a8),
            ("a4h", a4h_ty, vec![128, 1], &a4h),
            ("a8h", a8h_ty, vec![128, 1], &a8h),
        ] {
            let only = d.join(format!("{name}-only.gguf"));
            let t = decl(name, dims, ty, bytes.len() as u64);
            write_file(&only, &[], &[(t, bytes.clone())]);
            assert!(
                matches!(Gguf::open(&only), Err(LoadError::UnsupportedType { name: ref got, ty: t }) if got == name && t == ty)
            );
        }
        let split = Split::open(&p).unwrap();
        for (name, ty, id, dims, bytes) in [
            ("f8", f8_ty, 2000, [160u64, 2], &f8),
            ("mx", mx_ty, 2001, [64, 2], &mx),
            ("fp", fp_ty, 2002, [160, 2], &fp),
            ("b8", b8_ty, 2003, [128, 2], &b8),
            ("a4", a4_ty, 2004, [128, 1], &a4),
            ("a8", a8_ty, 2005, [128, 1], &a8),
            ("a4h", a4h_ty, 2006, [128, 1], &a4h),
            ("a8h", a8h_ty, 2007, [128, 1], &a8h),
        ] {
            let (s, t) = split.find(name).unwrap();
            assert_eq!(
                (t.ty, t.ty.as_u32(), t.nbytes, t.dims.as_slice()),
                (ty, id, bytes.len() as u64, &dims[..])
            );
            assert_eq!(split.shard(s).unwrap().data(t).unwrap(), bytes.as_slice());
        }
        assert_eq!((f8_ty.blck_size(), f8_ty.type_size()), (Some(1), Some(1)));
        assert_eq!((mx_ty.blck_size(), mx_ty.type_size()), (Some(32), Some(33)));
        assert_eq!((fp_ty.blck_size(), fp_ty.type_size()), (Some(16), Some(9)));
        assert_eq!(
            (b8_ty.blck_size(), b8_ty.type_size()),
            (Some(128), Some(132))
        );
        assert_eq!((a4_ty.blck_size(), a4_ty.type_size()), (Some(64), Some(36)));
        assert_eq!((a8_ty.blck_size(), a8_ty.type_size()), (Some(64), Some(68)));
        assert_eq!(
            (a4h_ty.blck_size(), a4h_ty.type_size()),
            (Some(64), Some(36))
        );
        assert_eq!(
            (a8h_ty.blck_size(), a8h_ty.type_size()),
            (Some(64), Some(68))
        );
        let other = d.join("other.gguf");
        write_file(
            &other,
            &[],
            &[(
                decl("o", vec![32, 2], GgmlType::Unknown(2008), 66),
                vec![0; 66],
            )],
        );
        assert!(matches!(
            Split::open(&other),
            Err(LoadError::Shard { ref source, .. })
                if matches!(**source, LoadError::UnsupportedType { ty: GgmlType::Unknown(2008), .. })
        ));
        for id in [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007] {
            let table = [PrivateType {
                id,
                blck: 1,
                size: 1,
            }];
            assert!(matches!(
                Gguf::open_private(&p, LAZY, &table),
                Err(LoadError::KnownPrivateType { id: got, .. }) if got == id
            ));
        }
        std::fs::remove_dir_all(&d).unwrap();
    }
}

#[cfg(test)]
mod value_offset_tests {
    use super::{GENERAL_ARCHITECTURE, Gguf, Split, Value};
    use crate::write::{TensorDecl, write_file};

    /// A written header's recorded value offsets read back each value's own bytes: a scalar's
    /// bytes at the offset, a string's length prefix at it, an array's element tag and count, a
    /// key no header names has no offset, and a split's shared keys sit at each shard's own
    /// offset in its own file.
    #[test]
    fn a_written_headers_value_offset_reads_back_the_value() {
        let d = std::env::temp_dir().join(format!("gguf-offset-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let kv = |k: &str, v: Value| (k.to_string(), v);
        let decl = |name: &str| TensorDecl {
            name: name.to_string(),
            dims: vec![4],
            type_id: 0,
            nbytes: 16,
        };
        let p = d.join("one.gguf");
        write_file(
            &p,
            &[
                kv("scalar.u64", Value::U64(0x0102_0304_0506_0708)),
                kv("scalar.f32", Value::F32(1.5)),
                kv("scalar.bool", Value::Bool(true)),
                kv("a.string", Value::String("an id".into())),
                kv(
                    "an.array",
                    Value::Array(vec![Value::String("a".into()), Value::String("bcd".into())]),
                ),
            ],
            &[(decl("t"), vec![0; 16])],
        );
        let g = Gguf::open(&p).unwrap();
        let m = g.mapping();
        let le64 = |b: &[u8]| u64::from_le_bytes(b.try_into().unwrap());
        let le32 = |b: &[u8]| u32::from_le_bytes(b.try_into().unwrap());
        let at = |key: &str| usize::try_from(g.value_offset(key).unwrap()).unwrap();
        assert_eq!(le64(&m[at("scalar.u64")..][..8]), 0x0102_0304_0506_0708);
        assert_eq!(f32::from_bits(le32(&m[at("scalar.f32")..][..4])), 1.5);
        assert_eq!(m[at("scalar.bool")], 1);
        let s = at("a.string");
        assert_eq!(le64(&m[s..][..8]), 5, "the string's length prefix");
        assert_eq!(&m[s + 8..][..5], b"an id");
        let a = at("an.array");
        assert_eq!(le32(&m[a..][..4]), 8, "the array's string element tag");
        assert_eq!(le64(&m[a + 4..][..8]), 2);
        assert_eq!(g.value_offset("absent.key"), None);
        // A split's shared keys sit at each shard's own offset: shard 0's header
        // carries one more pair, so the same key lands elsewhere in each file.
        for i in 0..2u16 {
            let mut kvs = vec![
                kv("split.no", Value::U16(i)),
                kv("split.count", Value::U16(2)),
                kv("split.tensors.count", Value::I32(2)),
            ];
            if i == 0 {
                kvs.insert(0, kv(GENERAL_ARCHITECTURE, Value::String("test".into())));
            }
            write_file(
                &d.join(format!("s-{:05}-of-00002.gguf", i + 1)),
                &kvs,
                &[(decl(if i == 0 { "a" } else { "b" }), vec![0; 16])],
            );
        }
        let s = Split::open(d.join("s-00001-of-00002.gguf")).unwrap();
        for i in 0..2u16 {
            let shard = s.shard(i as usize).unwrap();
            let at = usize::try_from(shard.value_offset("split.no").unwrap()).unwrap();
            let m = shard.mapping();
            assert_eq!(u16::from_le_bytes(m[at..][..2].try_into().unwrap()), i);
        }
        std::fs::remove_dir_all(&d).unwrap();
    }
}

#[cfg(test)]
mod located_tests {
    use super::{GENERAL_ARCHITECTURE, LoadError, Place, Split, StackLayout, Value, write};
    use std::fmt::Debug;
    use std::path::{Path, PathBuf};

    const A_LEN: usize = 64;
    const B_LEN: usize = 96;

    /// The two files' bytes: file A holds `i`, file B `100 + i`, at offset `i`.
    fn file_bytes(shard: usize) -> Vec<u8> {
        let (len, base) = if shard == 0 { (A_LEN, 0) } else { (B_LEN, 100) };
        (0..len).map(|i| base + i as u8).collect()
    }

    fn dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("gguf-located-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn files(d: &Path) -> Vec<PathBuf> {
        (0..2)
            .map(|i| {
                let p = d.join(format!("s-{i}.bin"));
                std::fs::write(&p, file_bytes(i)).unwrap();
                p
            })
            .collect()
    }

    fn f32s(name: &str, dims: &[u64]) -> write::TensorDecl {
        write::TensorDecl {
            name: name.to_string(),
            dims: dims.to_vec(),
            type_id: 0,
            nbytes: 4 * dims.iter().product::<u64>(),
        }
    }

    /// The header of the five tensors the places below keep: `f.weight` and `g.weight` in a file
    /// each, the stacks `s.weight` (three matrices of 12 + 4 bytes) and `t.weight` (two of
    /// 8 + 4), and `m.weight`, which no file holds.
    fn layout() -> write::Layout {
        let kvs = [(
            GENERAL_ARCHITECTURE.to_string(),
            Value::String("test".into()),
        )];
        write::Layout::new(
            &kvs,
            vec![
                f32s("f.weight", &[4]),
                f32s("g.weight", &[2]),
                f32s("s.weight", &[4, 3]),
                f32s("t.weight", &[6]),
                f32s("m.weight", &[2]),
            ],
        )
        .unwrap()
    }

    /// A stack of planes in file B: matrices `codes_at`, `scales_at` apart as given.
    fn planes(
        codes_at: &[u64],
        scales_at: &[u64],
        (codes_per, scales_per): (u64, u64),
        scales: &[f32],
    ) -> Place {
        Place::Stack {
            codes_in: vec![1; codes_at.len()],
            codes_at: codes_at.to_vec(),
            scales_in: vec![1; scales_at.len()],
            scales_at: scales_at.to_vec(),
            codes_per,
            scales_per,
            layout: StackLayout::Planes,
            scales: scales.to_vec(),
        }
    }

    /// `t.weight`'s two matrices of 8 + 4 bytes, their runs in shards `codes_in` and
    /// `scales_in`: both codes at offset 32, the block scales at 28 and 80.
    fn across(codes_in: [usize; 2], scales_in: [usize; 2]) -> Place {
        Place::Stack {
            codes_in: codes_in.to_vec(),
            codes_at: vec![32, 32],
            scales_in: scales_in.to_vec(),
            scales_at: vec![28, 80],
            codes_per: 8,
            scales_per: 4,
            layout: StackLayout::Planes,
            scales: vec![4.0, 0.125],
        }
    }

    /// The places of the five: `s.weight`'s block scales each directly after its codes, the
    /// matrices out of order; `t.weight`'s runs over both files ([`across`]).
    fn places() -> Vec<Place> {
        vec![
            Place::File {
                shard: 0,
                range: 8..24,
            },
            Place::File {
                shard: 1,
                range: 56..64,
            },
            planes(&[64, 0, 16], &[76, 12, 28], (12, 4), &[0.5, 2.0, 0.25]),
            across([1, 0], [0, 1]),
            Place::Made,
        ]
    }

    fn located(d: &Path, places: Vec<Place>) -> Result<Split, LoadError> {
        let files = files(d);
        Split::located(files[0].clone(), &layout(), &files, places)
    }

    fn refused(d: &Path, places: Vec<Place>) -> String {
        match located(d, places) {
            Ok(_) => panic!("the places must be refused"),
            Err(e) => e.to_string(),
        }
    }

    fn named<T: Debug>(r: Result<T, LoadError>, parts: &[&str]) {
        let e = r.expect_err("refused").to_string();
        assert!(parts.iter().all(|p| e.contains(p)), "{e}");
    }

    /// A two-file folder's split holds each file tensor's bytes where the file keeps them, reads
    /// a stack's matrices out of order with their block scales next to their codes, and a
    /// stack's runs each in its own file, and refuses a made tensor and a stack as one run by
    /// name, as bytes and as a file run. A header alone, for the same header, ends every read in
    /// `OutOfBounds`. FAIL-first: runs read from the stack's first file, or checked for overlap
    /// across files, read or refuse `t.weight`, whose two files hold a run at the same offset; a
    /// file run taken from the header's offset (`u64::MAX` for a stack) is not refused.
    #[test]
    fn a_located_split_reads_the_files_bytes_in_place() {
        let d = dir("reads");
        let s = located(&d, places()).unwrap();
        let (a, b) = (file_bytes(0), file_bytes(1));
        assert_eq!(s.shard_count(), 2);
        assert_eq!(s.shard_path(0).unwrap(), d.join("s-0.bin"));
        assert_eq!(s.shard_path(1).unwrap(), d.join("s-1.bin"));
        assert_eq!(s.architecture(), Some("test"));
        assert_eq!(s.tensor_count(), 5);
        assert!(!s.is_header_only());
        assert_eq!(s.data("f.weight").unwrap(), &a[8..24]);
        assert_eq!(s.data("g.weight").unwrap(), &b[56..64]);
        let (shard, info) = s.find("g.weight").unwrap();
        assert_eq!((shard, info.offset), (1, 56));
        assert_eq!(s.shard(shard).unwrap().data_base() + info.offset, 56);
        assert_eq!(s.shard(shard).unwrap().data(info).unwrap(), &b[56..64]);
        assert_eq!(s.run("g.weight").unwrap(), (1, 56..64));
        // Adjacent: each matrix's scales begin where its codes end.
        for (e, at) in [64usize, 0, 16].into_iter().enumerate() {
            let m = s.matrix("s.weight", e).unwrap();
            assert_eq!(m.codes, &b[at..at + 12]);
            assert_eq!(m.block_scales, &b[at + 12..at + 16]);
        }
        // Across: the planes of one matrix lie in different files, scales before codes.
        let t = [(&b[32..40], &a[28..32]), (&a[32..40], &b[80..84])];
        for (e, (codes, scales)) in t.into_iter().enumerate() {
            let m = s.matrix("t.weight", e).unwrap();
            assert_eq!((m.codes, m.block_scales), (codes, scales), "matrix {e}");
        }
        let (shard, _) = s.find("t.weight").unwrap();
        assert_eq!(shard, 1, "the first matrix's codes shard lists the stack");
        named(s.data("m.weight"), &["m.weight", "made"]);
        named(s.data("s.weight"), &["s.weight", "stack"]);
        named(s.data("t.weight"), &["t.weight", "stack"]);
        named(s.data("absent"), &["absent", "not a tensor"]);
        named(s.run("m.weight"), &["m.weight", "made"]);
        named(s.run("s.weight"), &["s.weight", "stack"]);
        named(s.run("t.weight"), &["t.weight", "stack"]);
        named(s.run("absent"), &["absent", "not a tensor"]);
        let (shard, info) = s.find("m.weight").unwrap();
        named(s.shard(shard).unwrap().data(info), &["m.weight", "made"]);
        named(s.matrix("s.weight", 3), &["s.weight", "no matrix"]);
        named(s.matrix("f.weight", 0), &["f.weight", "not a stack"]);
        let ps = places();
        assert_eq!(s.place("f.weight"), Some(&ps[0]));
        assert_eq!(s.place("s.weight"), Some(&ps[2]));
        assert_eq!(s.place("t.weight"), Some(&ps[3]));
        assert_eq!(s.place("m.weight"), Some(&Place::Made));
        assert_eq!(s.place("absent"), None);

        let alone = Split::header_only(d.join("s-0.bin"), &layout()).unwrap();
        let (shard, info) = alone.find("f.weight").unwrap();
        let e = alone.shard(shard).unwrap().data(info).unwrap_err();
        assert!(matches!(e, LoadError::OutOfBounds { .. }), "{e}");
        assert!(alone.is_header_only());
        std::fs::remove_dir_all(&d).unwrap();
    }

    /// A located split over no file is the header alone, its tensors all made; a place a file
    /// does not hold, a run of the wrong length, a stack with a run off its file, overlapping
    /// another, short of the declared bytes or with a scale that is no finite positive factor,
    /// and a split not named by its first file are refused by name. A stack of rows has no scale
    /// plane and no scales.
    #[test]
    fn places_that_do_not_describe_the_files_are_refused_by_name() {
        let d = dir("refuses");
        let l = layout();
        let path = d.join("s-0.bin");
        let alone = Split::located(path.clone(), &l, &[], vec![Place::Made; 5]).unwrap();
        let plain = Split::header_only(path.clone(), &l).unwrap();
        assert!(alone.is_header_only());
        assert_eq!(alone.shard_path(0).unwrap(), path);
        assert_eq!(alone.keys(), plain.keys());
        let tensors = |s: &Split| -> Vec<(usize, String, Vec<u64>, u64, u64)> {
            s.iter_tensors()
                .map(|(i, t)| (i, t.name.clone(), t.dims.clone(), t.offset, t.nbytes))
                .collect()
        };
        assert_eq!(tensors(&alone), tensors(&plain));
        assert_eq!(alone.place("f.weight"), Some(&Place::Made));
        let e = alone.data("f.weight").unwrap_err();
        assert!(matches!(e, LoadError::OutOfBounds { .. }), "{e}");
        let e = Split::located(path, &l, &[], places())
            .err()
            .unwrap()
            .to_string();
        assert!(e.contains("f.weight: shard 0, of 0 files"), "{e}");

        let with = |i: usize, p: Place| {
            let mut ps = places();
            ps[i] = p;
            ps
        };
        let file = |shard, range| Place::File { shard, range };
        let (codes, scales) = ([64, 0, 16], [76, 12, 28]);
        let three = |codes: &[u64], scales: &[u64], s: &[f32]| planes(codes, scales, (12, 4), s);
        let ok = [0.5, 2.0, 0.25];
        let stack_in_file_a = Place::Stack {
            codes_in: vec![0; 3],
            codes_at: vec![0, 16, 52],
            scales_in: vec![0; 3],
            scales_at: vec![12, 28, 64],
            codes_per: 12,
            scales_per: 4,
            layout: StackLayout::Planes,
            scales: ok.to_vec(),
        };
        let mut one_codes_shard = across([1, 0], [0, 1]);
        if let Place::Stack { codes_in, .. } = &mut one_codes_shard {
            codes_in.pop();
        }
        let rows_with_planes = Place::Stack {
            codes_in: vec![1; 2],
            codes_at: vec![32, 48],
            scales_in: vec![1; 2],
            scales_at: vec![88, 92],
            codes_per: 12,
            scales_per: 0,
            layout: StackLayout::Rows,
            scales: vec![],
        };
        let rows_with_scales = Place::Stack {
            codes_in: vec![1; 2],
            codes_at: vec![32, 48],
            scales_in: vec![],
            scales_at: vec![],
            codes_per: 12,
            scales_per: 0,
            layout: StackLayout::Rows,
            scales: vec![1.0, 1.0],
        };
        for (i, p, want) in [
            (0, file(2, 8..24), "f.weight: shard 2, of 2 files"),
            (
                0,
                file(0, 8..20),
                "f.weight: 12 bytes at 8..20, the declared 16",
            ),
            (
                0,
                file(0, 56..72),
                "f.weight: bytes 56..72 lie outside shard 0's 64",
            ),
            (
                2,
                planes(&[64, 0], &[76, 12], (12, 4), &[0.5, 2.0]),
                "s.weight: 2 matrices of 12 + 4 bytes, the declared 48",
            ),
            (
                2,
                three(&[64, 0, 88], &scales, &ok),
                "s.weight: the run at 88 runs outside shard 1's 96",
            ),
            (
                2,
                three(&[64, 0, u64::MAX], &scales, &ok),
                "s.weight: the run at 18446744073709551615 runs outside shard 1's 96",
            ),
            (
                2,
                three(&[0, 8, 32], &scales, &ok),
                "s.weight: the run at 0 overlaps the next",
            ),
            (
                2,
                three(&[0, 16, 64], &[14, 28, 76], &ok),
                "s.weight: the run at 14 overlaps the next",
            ),
            (
                2,
                planes(&[], &[], (12, 4), &[]),
                "s.weight: a stack of no matrix",
            ),
            (
                2,
                planes(&codes, &scales, (0, 4), &ok),
                "s.weight: a stack of no matrix, or of no bytes",
            ),
            (
                2,
                stack_in_file_a,
                "s.weight: the run at 64 runs outside shard 0's 64",
            ),
            (
                2,
                three(&codes, &scales[..2], &ok),
                "s.weight: 3 matrices, 2 scale planes of 4 bytes",
            ),
            (
                2,
                three(&codes, &scales, &ok[..2]),
                "s.weight: 3 matrices, 2 scales",
            ),
            (
                2,
                three(&codes, &scales, &[0.5, 0.0, 0.25]),
                "s.weight: the scale of matrix 1 is 0, not finite and positive",
            ),
            (
                2,
                three(&codes, &scales, &[f32::NAN, 2.0, 0.25]),
                "s.weight: the scale of matrix 0 is NaN, not finite and positive",
            ),
            (
                2,
                three(&codes, &scales, &[0.5, 2.0, -0.25]),
                "s.weight: the scale of matrix 2 is -0.25, not finite and positive",
            ),
            (
                2,
                three(&codes, &scales, &[0.5, f32::INFINITY, 0.25]),
                "s.weight: the scale of matrix 1 is inf, not finite and positive",
            ),
            (3, across([1, 2], [0, 1]), "t.weight: shard 2, of 2 files"),
            (
                3,
                across([1, 1], [0, 1]),
                "t.weight: the run at 32 overlaps the next",
            ),
            (
                3,
                one_codes_shard,
                "t.weight: 2 codes runs in 1 shards, 2 scale runs in 2",
            ),
            (
                3,
                rows_with_planes,
                "t.weight: a stack of rows has no scale plane and no tensor scales",
            ),
            (
                3,
                rows_with_scales,
                "t.weight: a stack of rows has no scale plane and no tensor scales",
            ),
        ] {
            let e = refused(&d, with(i, p));
            assert!(e.contains(want), "{want}: {e}");
        }
        let rows = Place::Stack {
            codes_in: vec![1; 2],
            codes_at: vec![32, 48],
            scales_in: vec![],
            scales_at: vec![],
            codes_per: 12,
            scales_per: 0,
            layout: StackLayout::Rows,
            scales: vec![],
        };
        let s = located(&d, with(3, rows)).unwrap();
        let m = s.matrix("t.weight", 1).unwrap();
        assert_eq!(m.codes, &file_bytes(1)[48..60]);
        assert!(m.block_scales.is_empty());
        let e = refused(&d, places()[..3].to_vec());
        assert!(e.contains("3 places for the header's 5 tensors"), "{e}");
        let fs = files(&d);
        let e = Split::located(fs[1].clone(), &l, &fs, places())
            .err()
            .unwrap()
            .to_string();
        assert!(e.contains("its first file is"), "{e}");
        let missing = [d.join("absent.bin")];
        let e = Split::located(missing[0].clone(), &l, &missing, places())
            .err()
            .unwrap();
        assert!(matches!(e, LoadError::MissingShard { .. }), "{e}");
        std::fs::remove_dir_all(&d).unwrap();
    }
}
