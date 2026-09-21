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
//! dequantize still inventory).
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

use std::fs::File;
use std::path::Path;

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

    /// `open`, optionally prefaulting the whole mapping (`MAP_POPULATE`). A lazy
    /// mapping takes a page fault on the first touch of every weight page, and
    /// a MoE keeps meeting untouched experts for hundreds of steps; a process
    /// that will decode pays that once here instead. Tests open lazily.
    pub fn open_with(path: impl AsRef<Path>, populate: bool) -> Result<Gguf, LoadError> {
        Self::open_backed(path, Weights::Mapped { populate })
    }

    /// `open`, with the caller choosing where the bytes live.
    pub fn open_backed(path: impl AsRef<Path>, weights: Weights) -> Result<Gguf, LoadError> {
        let file = File::open(path)?;
        let len = file.metadata()?.len();
        let mut opts = memmap2::MmapOptions::new();
        if weights == (Weights::Mapped { populate: true }) {
            opts.populate();
        }
        // SAFETY: the file is opened read-only and nothing maps it writable;
        // a concurrent truncation would surface as SIGBUS, the same contract
        // ik_llama.cpp's own mmap loader accepts.
        let file_map = unsafe { opts.map(&file)? };
        let map = match weights {
            Weights::Mapped { .. } => Backing::File(file_map),
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

        let h = parse_header(&map, len)?;

        // The strict pass: every tensor must be sized by this engine's
        // `GgmlType` and its first dim must be block-aligned.
        let mut strict = Vec::with_capacity(h.tensors.len());
        for t in h.tensors {
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

        let gguf = Gguf {
            _file: file,
            map,
            meta: h.meta,
            tensors: strict,
            data_base: h.data_base,
            alignment: h.alignment,
        };
        // Every tensor must sit inside the file — this is the placement
        // invariant the coverage gate re-checks per tensor.
        for t in &gguf.tensors {
            let end = gguf
                .data_base
                .checked_add(t.offset)
                .and_then(|o| o.checked_add(t.nbytes))
                .ok_or(LoadError::OutOfBounds {
                    name: t.name.clone(),
                    base: gguf.data_base,
                    off: t.offset,
                    nbytes: t.nbytes,
                    len,
                })?;
            if end > len {
                return Err(LoadError::OutOfBounds {
                    name: t.name.clone(),
                    base: gguf.data_base,
                    off: t.offset,
                    nbytes: t.nbytes,
                    len,
                });
            }
        }
        Ok(gguf)
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
}

/// A header-only parse: everything the header states, nothing the engine
/// requires. Placement, block alignment and dequantizability are NOT
/// validated, and type ids the engine has no `GgmlType` for are carried as
/// numbers with their size from ggml's table (or `None`). This is the
/// inventory for files the strict reader refuses — e.g. BF16/Q8_0/MXFP4
/// splits — where the point is to know what is in the file, not to run it.
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
///   * 8 `q8_0` 32/34 (:233-238), 9 `q8_1` 32/36 (:240-244),
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
///     (type_traits table, ggml.c:621-652), 30 `bf16` 1/2 (ggml.c:1485),
///   * 39 `mxfp4` 32/17 — one E8M0 byte + 16 nibble bytes per 32 values
///     (ggml-common.h:182-187).
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
        8 => ("q8_0", 32, 34),
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
        30 => ("bf16", 1, 2),
        39 => ("mxfp4", 32, 17),
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
