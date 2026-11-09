//! A safetensors file: its header and its tensors, from a mapping of the file or from bytes.
//!
//! The layout: an 8-byte little-endian header length, the JSON header (name → `dtype`, `shape`,
//! `data_offsets` relative to the end of the header, plus an optional `__metadata__`), then the
//! data. The header is validated whole on open: every entry's dtype is known, its byte range
//! matches its shape and lies in the data section, and no two ranges overlap. JSON object keys
//! are unique at every level, including the folder index. A tensor is read
//! three ways: widened to f32 ([`Safetensors::tensor`], BF16, F16 and F32), as its stored bits
//! ([`Safetensors::raw`], [`Safetensors::bf16`]), or as integers ([`Safetensors::i64s`]). The
//! packed and 8-bit dtypes quantized checkpoints store (U8, I8, U16, U32, I32, F8_E4M3, F8_E5M2,
//! F8_E8M0) are read as stored bytes only.
//!
//! A model folder ([`Folder`]) is `model.safetensors.index.json` and the shards it names, opened as
//! one table of tensors, each read whole as stored.

use std::collections::BTreeMap;
use std::ops::Deref;
use std::path::Path;

use serde_json::Value;

#[cfg(not(target_endian = "little"))]
compile_error!("safetensors stores little-endian values and `bf16` hands them out in place");

/// Why a file, a header entry or a read was refused.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}: {1}")]
    Io(String, std::io::Error),
    #[error("safetensors: {0}")]
    Format(String),
    #[error("safetensors: {name} is {dtype}, a type this reader does not hold")]
    Dtype { name: String, dtype: String },
    #[error("safetensors: {name} is {have}; this read wants {want}")]
    Kind {
        name: String,
        have: &'static str,
        want: &'static str,
    },
    #[error("weight {0} is missing")]
    Missing(String),
}

fn bad(what: String) -> Error {
    Error::Format(what)
}

/// Parses `text`, the JSON `what` names, and refuses a repeated object key at any level.
pub fn parse_unique(what: &str, text: &str) -> Result<Value, Error> {
    struct NoRepeat;

    impl<'de> serde::Deserialize<'de> for NoRepeat {
        fn deserialize<D: serde::Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
            struct Visitor;
            impl<'de> serde::de::Visitor<'de> for Visitor {
                type Value = NoRepeat;

                fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    f.write_str("JSON with unique object keys")
                }

                fn visit_bool<E>(self, _: bool) -> Result<NoRepeat, E> {
                    Ok(NoRepeat)
                }

                fn visit_i64<E>(self, _: i64) -> Result<NoRepeat, E> {
                    Ok(NoRepeat)
                }

                fn visit_u64<E>(self, _: u64) -> Result<NoRepeat, E> {
                    Ok(NoRepeat)
                }

                fn visit_f64<E>(self, _: f64) -> Result<NoRepeat, E> {
                    Ok(NoRepeat)
                }

                fn visit_str<E>(self, _: &str) -> Result<NoRepeat, E> {
                    Ok(NoRepeat)
                }

                fn visit_unit<E>(self) -> Result<NoRepeat, E> {
                    Ok(NoRepeat)
                }

                fn visit_seq<A: serde::de::SeqAccess<'de>>(
                    self,
                    mut seq: A,
                ) -> Result<NoRepeat, A::Error> {
                    while seq.next_element::<NoRepeat>()?.is_some() {}
                    Ok(NoRepeat)
                }

                fn visit_map<A: serde::de::MapAccess<'de>>(
                    self,
                    mut map: A,
                ) -> Result<NoRepeat, A::Error> {
                    let mut seen = std::collections::HashSet::new();
                    while let Some(key) = map.next_key::<String>()? {
                        if !seen.insert(key.clone()) {
                            return Err(serde::de::Error::custom(format!("duplicate key {key:?}")));
                        }
                        map.next_value::<NoRepeat>()?;
                    }
                    Ok(NoRepeat)
                }
            }
            de.deserialize_any(Visitor)
        }
    }

    let not_json = |e: serde_json::Error| bad(format!("{what} is not JSON: {e}"));
    serde_json::from_str::<NoRepeat>(text).map_err(not_json)?;
    serde_json::from_str(text).map_err(not_json)
}

/// A stored element type this reader holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype {
    Bf16,
    F16,
    F32,
    I64,
    Bool,
    U8,
    I8,
    U16,
    U32,
    I32,
    F8E4M3,
    F8E5M2,
    F8E8M0,
}

impl Dtype {
    /// The dtype a safetensors header spells `name`, or `None` for one this reader does not know.
    #[must_use]
    pub fn of(name: &str) -> Option<Dtype> {
        match name {
            "BF16" => Some(Dtype::Bf16),
            "F16" => Some(Dtype::F16),
            "F32" => Some(Dtype::F32),
            "I64" => Some(Dtype::I64),
            "BOOL" => Some(Dtype::Bool),
            "U8" => Some(Dtype::U8),
            "I8" => Some(Dtype::I8),
            "U16" => Some(Dtype::U16),
            "U32" => Some(Dtype::U32),
            "I32" => Some(Dtype::I32),
            "F8_E4M3" => Some(Dtype::F8E4M3),
            "F8_E5M2" => Some(Dtype::F8E5M2),
            "F8_E8M0" => Some(Dtype::F8E8M0),
            _ => None,
        }
    }

    /// The name the header spells it with.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Dtype::Bf16 => "BF16",
            Dtype::F16 => "F16",
            Dtype::F32 => "F32",
            Dtype::I64 => "I64",
            Dtype::Bool => "BOOL",
            Dtype::U8 => "U8",
            Dtype::I8 => "I8",
            Dtype::U16 => "U16",
            Dtype::U32 => "U32",
            Dtype::I32 => "I32",
            Dtype::F8E4M3 => "F8_E4M3",
            Dtype::F8E5M2 => "F8_E5M2",
            Dtype::F8E8M0 => "F8_E8M0",
        }
    }

    /// Bytes per element.
    #[must_use]
    pub fn size(self) -> usize {
        match self {
            Dtype::Bool | Dtype::U8 | Dtype::I8 | Dtype::F8E4M3 | Dtype::F8E5M2 | Dtype::F8E8M0 => {
                1
            }
            Dtype::Bf16 | Dtype::F16 | Dtype::U16 => 2,
            Dtype::F32 | Dtype::U32 | Dtype::I32 => 4,
            Dtype::I64 => 8,
        }
    }
}

/// One header entry.
#[derive(Clone, Debug)]
pub struct Entry {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    /// Byte range in the data section.
    start: usize,
    end: usize,
}

/// A tensor converted to f32, row-major.
#[derive(Clone, Debug)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// A tensor as stored: its dtype, its shape and its bytes in the file, not converted.
#[derive(Clone, Copy, Debug)]
pub struct Raw<'a> {
    pub dtype: Dtype,
    pub shape: &'a [usize],
    pub bytes: &'a [u8],
}

/// Where the file's bytes live.
enum Backing {
    Map(memmap2::Mmap),
    Heap(Vec<u8>),
}

impl Deref for Backing {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            Backing::Map(m) => m,
            Backing::Heap(v) => v,
        }
    }
}

/// A whole file with its parsed header.
pub struct Safetensors {
    bytes: Backing,
    data: usize,
    entries: BTreeMap<String, Entry>,
}

impl Safetensors {
    /// Map and parse the file at `path`. The mapping is read-only; no tensor is copied until it is
    /// asked for.
    pub fn open(path: &Path) -> Result<Safetensors, Error> {
        let io = |e| Error::Io(path.display().to_string(), e);
        let file = std::fs::File::open(path).map_err(io)?;
        if file.metadata().map_err(io)?.len() == 0 {
            return Err(bad("0 bytes hold no header length".into()));
        }
        // SAFETY: the file is opened read-only and nothing maps it writable; a concurrent
        // truncation would surface as SIGBUS, the contract gguf's own mapping accepts.
        let map = unsafe { memmap2::MmapOptions::new().map(&file) }.map_err(io)?;
        Safetensors::from_backing(Backing::Map(map))
    }

    /// Parse a whole file's bytes.
    pub fn parse(bytes: Vec<u8>) -> Result<Safetensors, Error> {
        Safetensors::from_backing(Backing::Heap(bytes))
    }

    fn from_backing(bytes: Backing) -> Result<Safetensors, Error> {
        let len_bytes: [u8; 8] = bytes
            .get(..8)
            .and_then(|b| b.try_into().ok())
            .ok_or_else(|| bad(format!("{} bytes hold no header length", bytes.len())))?;
        let n = usize::try_from(u64::from_le_bytes(len_bytes))
            .map_err(|_| bad("the header length does not fit usize".into()))?;
        let data = n
            .checked_add(8)
            .filter(|&d| d <= bytes.len())
            .ok_or_else(|| {
                bad(format!(
                    "a header of {n} bytes runs past the file's {} bytes",
                    bytes.len()
                ))
            })?;
        let text = std::str::from_utf8(&bytes[8..data])
            .map_err(|_| bad("the header is not UTF-8".into()))?;
        let Value::Object(pairs) = parse_unique("the header", text)? else {
            return Err(bad("the header is not a JSON object".into()));
        };
        let mut entries = BTreeMap::new();
        for (name, v) in pairs {
            if name == "__metadata__" {
                continue;
            }
            let e = entry(&name, &v, bytes.len() - data)?;
            entries.insert(name, e);
        }
        let mut ranges: Vec<(usize, usize, &str)> = entries
            .iter()
            .map(|(k, e)| (e.start, e.end, k.as_str()))
            .collect();
        ranges.sort_unstable();
        for w in ranges.windows(2) {
            if w[1].0 < w[0].1 {
                return Err(bad(format!("{} and {} share bytes", w[0].2, w[1].2)));
            }
        }
        Ok(Safetensors {
            bytes,
            data,
            entries,
        })
    }

    /// Every tensor name, sorted.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(String::as_str)
    }

    /// The header entry of `name`.
    #[must_use]
    pub fn entry(&self, name: &str) -> Option<&Entry> {
        self.entries.get(name)
    }

    /// Where `name`'s bytes lie in the file: the byte range from the file's start.
    pub fn file_range(&self, name: &str) -> Result<std::ops::Range<u64>, Error> {
        let e = self
            .entries
            .get(name)
            .ok_or_else(|| Error::Missing(name.to_string()))?;
        Ok((self.data + e.start) as u64..(self.data + e.end) as u64)
    }

    /// The whole of the file's own read-only mapping, file offset `o` at index `o`; `None` for
    /// bytes held on the heap ([`Safetensors::parse`]).
    pub fn file_mapping(&self) -> Option<&[u8]> {
        match &self.bytes {
            Backing::Map(m) => Some(m),
            Backing::Heap(_) => None,
        }
    }

    /// `name` as stored: dtype, shape and the bytes in the file.
    pub fn raw(&self, name: &str) -> Result<Raw<'_>, Error> {
        let e = self
            .entries
            .get(name)
            .ok_or_else(|| Error::Missing(name.to_string()))?;
        Ok(Raw {
            dtype: e.dtype,
            shape: &e.shape,
            bytes: &self.bytes[self.data + e.start..self.data + e.end],
        })
    }

    /// A BF16 tensor's bits in place, with its shape: no copy and no widening. Refuses any other
    /// dtype, and bytes that do not sit on a 2-byte boundary.
    pub fn bf16(&self, name: &str) -> Result<(&[u16], &[usize]), Error> {
        let raw = self.raw(name)?;
        if raw.dtype != Dtype::Bf16 {
            return Err(Error::Kind {
                name: name.to_string(),
                have: raw.dtype.name(),
                want: "BF16",
            });
        }
        // SAFETY: every bit pattern is a `u16`; `align_to` returns the aligned middle only, and
        // the prefix and suffix are refused below, so the whole tensor is that middle.
        let (head, mid, tail) = unsafe { raw.bytes.align_to::<u16>() };
        if !head.is_empty() || !tail.is_empty() {
            return Err(bad(format!(
                "{name}: its bytes are not 2-byte aligned in the file"
            )));
        }
        Ok((mid, raw.shape))
    }

    /// An I64 tensor's values, row-major.
    pub fn i64s(&self, name: &str) -> Result<Vec<i64>, Error> {
        let raw = self.raw(name)?;
        if raw.dtype != Dtype::I64 {
            return Err(Error::Kind {
                name: name.to_string(),
                have: raw.dtype.name(),
                want: "I64",
            });
        }
        Ok(raw
            .bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| i64::from_le_bytes(*b))
            .collect())
    }

    /// `name` converted to f32. BF16, F16 and F32 convert exactly; any other dtype is refused by
    /// name.
    pub fn tensor(&self, name: &str) -> Result<Tensor, Error> {
        let raw = self.raw(name)?;
        Ok(Tensor {
            shape: raw.shape.to_vec(),
            data: f32_values(name, &raw)?,
        })
    }
}

/// The values of the tensor `name` stored as `raw`, converted to f32. BF16, F16 and F32 convert
/// exactly; any other dtype is refused by name. The shape must describe the complete byte span.
pub fn f32_values(name: &str, raw: &Raw<'_>) -> Result<Vec<f32>, Error> {
    let expected = raw
        .shape
        .iter()
        .try_fold(raw.dtype.size(), |bytes, &dim| bytes.checked_mul(dim));
    if expected != Some(raw.bytes.len()) {
        return Err(bad(format!(
            "{name}: shape {:?} does not match {} {} bytes",
            raw.shape,
            raw.bytes.len(),
            raw.dtype.name()
        )));
    }
    Ok(match raw.dtype {
        Dtype::Bf16 => raw
            .bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| f32::from_bits(u32::from(u16::from_le_bytes(*b)) << 16))
            .collect(),
        Dtype::F16 => raw
            .bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| gguf::quant::half_to_f32(u16::from_le_bytes(*b)))
            .collect(),
        Dtype::F32 => raw
            .bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect(),
        Dtype::I64
        | Dtype::Bool
        | Dtype::U8
        | Dtype::I8
        | Dtype::U16
        | Dtype::U32
        | Dtype::I32
        | Dtype::F8E4M3
        | Dtype::F8E5M2
        | Dtype::F8E8M0 => {
            return Err(Error::Dtype {
                name: name.to_string(),
                dtype: raw.dtype.name().to_string(),
            });
        }
    })
}

/// A Hugging Face model folder: `model.safetensors.index.json`'s `weight_map` (a tensor's name to
/// its shard's file name) and every shard it names, each mapped and its header validated as
/// [`Safetensors::open`] does. Opening reads the index and the headers, no tensor's bytes. Every
/// name of the index is in the shard it names, and every tensor of a shard is in the index under
/// that shard; a shard named by anything but a relative path inside the folder (`/`-separated file
/// and folder names with no `\` or `:`, as `tensors/native-QB.safetensors`), and either mismatch,
/// are refused by name.
pub struct Folder {
    dir: std::path::PathBuf,
    /// Each shard's file name and file, in the order the index first names them.
    shards: Vec<(String, Safetensors)>,
    /// A tensor's name to its shard.
    by_name: BTreeMap<String, usize>,
}

impl Folder {
    /// The index file's name in the folder.
    pub const INDEX: &str = "model.safetensors.index.json";

    /// The index at `path`: its nonempty `weight_map`, a tensor's name to its shard, every shard
    /// a relative path inside the folder (`/`-separated parts, none empty, `.` or `..`, and no
    /// `\` or `:`), refused by name otherwise.
    pub fn weight_map(path: &Path) -> Result<BTreeMap<String, String>, Error> {
        let text =
            std::fs::read_to_string(path).map_err(|e| Error::Io(path.display().to_string(), e))?;
        let mut index = parse_unique(&path.display().to_string(), &text)?;
        let Some(Value::Object(map)) = index.as_object_mut().and_then(|m| m.remove("weight_map"))
        else {
            return Err(bad(format!("{}: no weight_map object", path.display())));
        };
        if map.is_empty() {
            return Err(bad(format!(
                "{}: weight_map holds no tensors",
                path.display()
            )));
        }
        map.into_iter()
            .map(|(name, file)| {
                let shard = file
                    .as_str()
                    .filter(|f| {
                        !f.contains(['\\', ':'])
                            && f.split('/')
                                .all(|part| !part.is_empty() && part != "." && part != "..")
                    })
                    .ok_or_else(|| {
                        bad(format!(
                            "{}: {name}'s shard {file} is not a path inside the folder",
                            Folder::INDEX
                        ))
                    })?;
                Ok((name, shard.to_owned()))
            })
            .collect()
    }

    /// The model folder `dir`.
    pub fn open(dir: &Path) -> Result<Folder, Error> {
        let mut files: Vec<String> = Vec::new();
        let mut by_name = BTreeMap::new();
        for (name, file) in Folder::weight_map(&dir.join(Folder::INDEX))? {
            let at = files.iter().position(|f| *f == file).unwrap_or_else(|| {
                files.push(file);
                files.len() - 1
            });
            by_name.insert(name, at);
        }
        let shards = files
            .into_iter()
            .map(|f| match Safetensors::open(&dir.join(&f)) {
                Ok(st) => Ok((f, st)),
                Err(Error::Format(what)) => Err(bad(format!("{f}: {what}"))),
                Err(e) => Err(e),
            })
            .collect::<Result<Vec<_>, Error>>()?;
        for (name, &at) in &by_name {
            if shards[at].1.entry(name).is_none() {
                return Err(bad(format!(
                    "{name}: the index puts it in {}, which does not hold it",
                    shards[at].0
                )));
            }
        }
        for (at, (file, st)) in shards.iter().enumerate() {
            if let Some(name) = st.names().find(|n| by_name.get(*n) != Some(&at)) {
                return Err(bad(format!(
                    "{name}: in {file}, which the index does not name for it"
                )));
            }
        }
        Ok(Folder {
            dir: dir.to_path_buf(),
            shards,
            by_name,
        })
    }

    /// Where `name`'s bytes lie: its shard's path and the byte range in that file
    /// ([`Safetensors::file_range`]), for a reader that maps or reads the file itself.
    pub fn file_range(
        &self,
        name: &str,
    ) -> Result<(std::path::PathBuf, std::ops::Range<u64>), Error> {
        let (file, st) = self.shard(name)?;
        Ok((self.dir.join(file), st.file_range(name)?))
    }

    /// Tensor `name`'s shard file, that file's whole read-only mapping
    /// ([`Safetensors::file_mapping`]) and the tensor's byte range in it.
    pub fn mapped(
        &self,
        name: &str,
    ) -> Result<(std::path::PathBuf, &[u8], std::ops::Range<u64>), Error> {
        let (file, st) = self.shard(name)?;
        let map = st
            .file_mapping()
            .ok_or_else(|| bad(format!("{file} is no file's mapping")))?;
        Ok((self.dir.join(file), map, st.file_range(name)?))
    }

    /// The folder's directory.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Each shard's path with the bytes of its header (the length and the JSON), which every
    /// open of the folder reads, in the order the index first names the shards.
    pub fn headers(&self) -> Vec<(std::path::PathBuf, std::ops::Range<u64>)> {
        self.shards
            .iter()
            .map(|(f, st)| (self.dir.join(f), 0..st.data as u64))
            .collect()
    }

    /// Every tensor name, sorted.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.by_name.keys().map(String::as_str)
    }

    /// `name` as stored, from its shard.
    pub fn raw(&self, name: &str) -> Result<Raw<'_>, Error> {
        self.shard(name)?.1.raw(name)
    }

    /// The file name and file of the shard that holds `name`.
    fn shard(&self, name: &str) -> Result<&(String, Safetensors), Error> {
        let &at = self
            .by_name
            .get(name)
            .ok_or_else(|| Error::Missing(name.to_string()))?;
        Ok(&self.shards[at])
    }
}

/// A routed expert stack from one tensor (name, row-major shape) an expert id: the tensor names
/// in expert order, so the stack's expert `e` is HF's expert `e`, and its ggml dims
/// `[cols, rows, count]`. Ids other than exactly `0..count`, a tensor that is not 2-D and a shape
/// that differs from expert 0's are refused by name.
pub fn expert_stack(
    by_id: &BTreeMap<usize, (&str, &[usize])>,
    count: usize,
) -> Result<(Vec<String>, Vec<usize>), Error> {
    let Some((_, &(first, shape))) = by_id.first_key_value() else {
        return Err(bad("an expert stack with no expert".into()));
    };
    if by_id.keys().copied().ne(0..count) {
        return Err(bad(format!(
            "{first}: the stack's {} expert ids are not 0..{count}",
            by_id.len()
        )));
    }
    let &[rows, cols] = shape else {
        return Err(bad(format!(
            "{first}: expert projection {shape:?} is not 2-D"
        )));
    };
    if let Some((name, _)) = by_id.values().find(|(_, s)| *s != shape) {
        return Err(bad(format!(
            "{name}: shape differs from {first}'s {shape:?}"
        )));
    }
    let sources = by_id.values().map(|(n, _)| n.to_string()).collect();
    Ok((sources, vec![cols, rows, count]))
}

fn entry(name: &str, v: &Value, data_len: usize) -> Result<Entry, Error> {
    let field = |key: &str| v.get(key).ok_or_else(|| bad(format!("{name}: no {key}")));
    let dtype_name = field("dtype")?
        .as_str()
        .ok_or_else(|| bad(format!("{name}: dtype is not a string")))?;
    let dtype = Dtype::of(dtype_name).ok_or_else(|| Error::Dtype {
        name: name.to_string(),
        dtype: dtype_name.to_string(),
    })?;
    let usizes = |key: &str| -> Result<Vec<usize>, Error> {
        let Value::Array(items) = field(key)? else {
            return Err(bad(format!("{name}: {key} is not an array")));
        };
        items
            .iter()
            .map(|d| {
                d.as_u64()
                    .and_then(|d| usize::try_from(d).ok())
                    .ok_or_else(|| bad(format!("{name}: {key} holds {d}")))
            })
            .collect()
    };
    let shape = usizes("shape")?;
    let offsets = usizes("data_offsets")?;
    let [start, end] = offsets[..] else {
        return Err(bad(format!(
            "{name}: data_offsets holds {} values, not 2",
            offsets.len()
        )));
    };
    let want = shape
        .iter()
        .try_fold(dtype.size(), |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| bad(format!("{name}: shape {shape:?} overflows")))?;
    if start > end || end > data_len || end - start != want {
        return Err(bad(format!(
            "{name}: bytes {start}..{end} for shape {shape:?} {dtype_name} ({want} bytes) in a data section of {data_len}"
        )));
    }
    Ok(Entry {
        dtype,
        shape,
        start,
        end,
    })
}

/// A safetensors file's bytes from `(name, dtype, shape, raw data)`, in the given order: the
/// writer the tests and synthetic heads use.
#[must_use]
pub fn write(tensors: &[(&str, &str, &[usize], &[u8])]) -> Vec<u8> {
    let entries: Vec<(&str, &str, &[usize], usize)> = tensors
        .iter()
        .map(|&(name, dtype, shape, raw)| (name, dtype, shape, raw.len()))
        .collect();
    let mut out = header(&entries);
    for (_, _, _, raw) in tensors {
        out.extend_from_slice(raw);
    }
    out
}

/// The bytes before a safetensors file's data, its header length and header, for tensors
/// `(name, dtype, shape, data bytes)` laid out in the given order: a writer that streams the data
/// after it ([`write`]).
#[must_use]
pub fn header(tensors: &[(&str, &str, &[usize], usize)]) -> Vec<u8> {
    // The header text is built by hand: the entries keep the given order, which a `Value` map
    // without `preserve_order` would sort.
    let json = |v: &Value| v.to_string();
    let ints = |v: &[usize]| {
        json(&Value::Array(
            v.iter().map(|&d| Value::from(d as u64)).collect(),
        ))
    };
    let mut text = String::from(r#"{"__metadata__":{"format":"pt"}"#);
    let mut at = 0usize;
    for &(name, dtype, shape, len) in tensors {
        text.push(',');
        text.push_str(&json(&Value::from(name)));
        text.push_str(&format!(
            r#":{{"dtype":{},"shape":{},"data_offsets":{}}}"#,
            json(&Value::from(dtype)),
            ints(shape),
            ints(&[at, at + len]),
        ));
        at += len;
    }
    text.push('}');
    while !text.len().is_multiple_of(8) {
        text.push(' ');
    }
    let mut out = (text.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(text.as_bytes());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bf16(v: &[f32]) -> Vec<u8> {
        v.iter()
            .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }

    #[test]
    fn f32_conversion_validates_constructed_raw_spans() {
        let malformed: &[(Dtype, &[usize], &[u8])] = &[
            (Dtype::F32, &[1], &[0, 0, 0]),
            (Dtype::Bf16, &[1], &[0]),
            (Dtype::F16, &[1], &[0]),
            (Dtype::F32, &[2], &[0; 4]),
            (Dtype::F32, &[1], &[0; 8]),
            (Dtype::F32, &[usize::MAX], &[]),
        ];
        for &(dtype, shape, bytes) in malformed {
            let raw = Raw {
                dtype,
                shape,
                bytes,
            };
            assert!(matches!(f32_values("raw", &raw), Err(Error::Format(_))));
        }
        let scalar = 1.25f32.to_le_bytes();
        let raw = Raw {
            dtype: Dtype::F32,
            shape: &[],
            bytes: &scalar,
        };
        assert_eq!(f32_values("scalar", &raw).unwrap(), [1.25]);
        let empty = Raw {
            dtype: Dtype::F32,
            shape: &[0],
            bytes: &[],
        };
        assert!(f32_values("empty", &empty).unwrap().is_empty());
    }

    #[test]
    fn reads_bf16_f16_f32_and_scalars() {
        let f32s: Vec<u8> = [1.5f32, -2.25]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        let f16s: Vec<u8> = [0x3c00u16, 0xc000]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        let file = write(&[
            (
                "a.weight",
                "BF16",
                &[2, 2],
                &bf16(&[1.0, -0.5, 3.0, 0.15625]),
            ),
            ("b", "F32", &[2], &f32s),
            ("c", "F16", &[1, 2], &f16s),
            ("scale", "BF16", &[], &bf16(&[0.75])),
        ]);
        let st = Safetensors::parse(file).unwrap();
        assert_eq!(
            st.names().collect::<Vec<_>>(),
            ["a.weight", "b", "c", "scale"]
        );
        let a = st.tensor("a.weight").unwrap();
        assert_eq!(
            (a.shape.as_slice(), a.data.as_slice()),
            ([2, 2].as_slice(), [1.0, -0.5, 3.0, 0.15625].as_slice())
        );
        assert_eq!(st.tensor("b").unwrap().data, [1.5, -2.25]);
        assert_eq!(st.tensor("c").unwrap().data, [1.0, -2.0]);
        let s = st.tensor("scale").unwrap();
        assert_eq!((s.shape.len(), s.data.as_slice()), (0, [0.75].as_slice()));
        assert!(matches!(st.tensor("d"), Err(Error::Missing(n)) if n == "d"));
    }

    #[test]
    fn the_raw_accessors_keep_the_stored_bits_of_bf16_i64_and_bool() {
        let ids: Vec<u8> = [-3i64, 0, 7].iter().flat_map(|x| x.to_le_bytes()).collect();
        let bits = bf16(&[1.0, -2.0]);
        let file = write(&[
            ("ids", "I64", &[3], &ids),
            ("flags", "BOOL", &[2, 2], &[1, 0, 0, 1]),
            ("w", "BF16", &[1, 2], &bits),
        ]);
        let st = Safetensors::parse(file).unwrap();
        let ids = st.raw("ids").unwrap();
        assert_eq!(
            (ids.dtype, ids.shape, ids.bytes.len()),
            (Dtype::I64, [3].as_slice(), 24)
        );
        assert_eq!(st.i64s("ids").unwrap(), [-3, 0, 7]);
        let flags = st.raw("flags").unwrap();
        assert_eq!(
            (flags.dtype, flags.shape, flags.bytes),
            (Dtype::Bool, [2, 2].as_slice(), [1, 0, 0, 1].as_slice())
        );
        // The in-place read is the file's bits, not a widened copy; the file's offsets here
        // (the header is padded to 8) leave the tensor on a 2-byte boundary.
        let (w, shape) = st.bf16("w").unwrap();
        assert_eq!(
            (w, shape),
            ([0x3f80u16, 0xc000].as_slice(), [1, 2].as_slice())
        );
        // Each read refuses a dtype it does not hold, by name.
        assert!(matches!(
            st.bf16("ids"),
            Err(Error::Kind {
                have: "I64",
                want: "BF16",
                ..
            })
        ));
        assert!(matches!(
            st.i64s("w"),
            Err(Error::Kind {
                have: "BF16",
                want: "I64",
                ..
            })
        ));
        assert!(matches!(st.tensor("ids"), Err(Error::Dtype { ref dtype, .. }) if dtype == "I64"));
        assert!(matches!(st.raw("none"), Err(Error::Missing(n)) if n == "none"));
    }

    /// Rows of a ModelOpt NVFP4 shard under their own header entries (the fixture's
    /// `modelopt_nvfp4.py slice`): a U8 weight and an F8_E4M3 scale open and read as their
    /// stored bytes, and the f32 read refuses both by name.
    #[test]
    fn a_modelopt_shard_slice_opens_with_u8_and_f8_e4m3_held_raw() {
        let st = Safetensors::parse(
            include_bytes!("../../hf/tests/fixtures/modelopt-nvfp4.safetensors").to_vec(),
        )
        .unwrap();
        let stem = "model.language_model.layers.0.mlp.experts.0.gate_proj";
        let w = st.raw(&format!("{stem}.weight")).unwrap();
        let s = st.raw(&format!("{stem}.weight_scale")).unwrap();
        assert_eq!(
            (w.dtype, w.shape, w.bytes.len()),
            (Dtype::U8, [4, 1280].as_slice(), 4 * 1280)
        );
        assert_eq!(
            (s.dtype, s.shape, s.bytes.len()),
            (Dtype::F8E4M3, [4, 160].as_slice(), 4 * 160)
        );
        assert_eq!((Dtype::U8.name(), Dtype::F8E4M3.name()), ("U8", "F8_E4M3"));
        for (name, want) in [("weight", "U8"), ("weight_scale", "F8_E4M3")] {
            let r = st.tensor(&format!("{stem}.{name}"));
            assert!(
                matches!(r, Err(Error::Dtype { ref dtype, .. }) if dtype == want),
                "{r:?}"
            );
        }
    }

    #[test]
    fn a_file_on_disk_is_mapped_and_reads_the_same() {
        let bits = bf16(&[0.5, 4.0]);
        let file = write(&[("w", "BF16", &[2], &bits)]);
        let path =
            std::env::temp_dir().join(format!("hf-safetensors-{}.safetensors", std::process::id()));
        std::fs::write(&path, &file).unwrap();
        let st = Safetensors::open(&path);
        std::fs::remove_file(&path).unwrap();
        let st = st.unwrap();
        assert_eq!(st.tensor("w").unwrap().data, [0.5, 4.0]);
        assert!(matches!(
            Safetensors::open(&path.with_extension("none")),
            Err(Error::Io(..))
        ));
    }

    #[test]
    fn refuses_unknown_dtypes_and_bad_ranges() {
        let four = [0u8; 4];
        let r = Safetensors::parse(write(&[("x", "C64", &[4], &four)]));
        assert!(
            matches!(r, Err(Error::Dtype { ref name, ref dtype }) if name == "x" && dtype == "C64"),
            "{:?}",
            r.err()
        );
        // F32 of 2 elements is 8 bytes, not 4.
        assert!(matches!(
            Safetensors::parse(write(&[("x", "F32", &[2], &four)])),
            Err(Error::Format(_))
        ));
        let mut short = write(&[("x", "F32", &[1], &four)]);
        short.pop();
        assert!(matches!(Safetensors::parse(short), Err(Error::Format(_))));
        assert!(matches!(
            Safetensors::parse(vec![1, 0, 0]),
            Err(Error::Format(_))
        ));
        let mut huge = write(&[("x", "F32", &[1], &four)]);
        huge[..8].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(matches!(Safetensors::parse(huge), Err(Error::Format(_))));
    }

    #[test]
    fn duplicate_header_keys_are_refused() {
        let unique = r#"{"a":[true,false,null,-1,0,4,1.5,18446744073709551616,1.234567890123456789,1e100,"s",{"b":{}}]}"#;
        assert_eq!(
            parse_unique("text", unique).unwrap(),
            serde_json::from_str::<Value>(unique).unwrap(),
            "JSON with unique keys must parse to the value serde_json gives it"
        );
        for (text, why) in [
            (r#"{"a":1,"a":2}"#, "text is not JSON: duplicate key \"a\""),
            ("{", "text is not JSON: EOF"),
        ] {
            assert!(
                matches!(parse_unique("text", text), Err(Error::Format(e)) if e.starts_with(why)),
                "the parser's refusal is the crate's Format error naming its input: {text}"
            );
        }
        let a = r#"{"dtype":"F32","shape":[1],"data_offsets":[0,4]}"#;
        let b = r#"{"dtype":"F32","shape":[1],"data_offsets":[4,8]}"#;
        let tensor_twice = |first: &str, second: &str| {
            format!(r#"{{"model.layers.0.weight":{first},"model.layers.0.weight":{second}}}"#)
        };
        for (text, key) in [
            (tensor_twice(a, b), "model.layers.0.weight"),
            (tensor_twice(b, a), "model.layers.0.weight"),
            (
                r#"{"x":{"dtype":"F32","dtype":"U32","shape":[1],"data_offsets":[0,4]}}"#.into(),
                "dtype",
            ),
            (
                r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4],"data_offsets":[4,8]}}"#
                    .into(),
                "data_offsets",
            ),
            (
                r#"{"__metadata__":{"source":"a","source":"b"}}"#.into(),
                "source",
            ),
            (
                r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4],"extra":[{"k":1,"k":2}]}}"#
                    .into(),
                "k",
            ),
        ] {
            let mut file = (text.len() as u64).to_le_bytes().to_vec();
            file.extend_from_slice(text.as_bytes());
            file.extend_from_slice(&[0; 8]);
            assert!(
                matches!(Safetensors::parse(file), Err(Error::Format(e))
                if e.contains(&format!("duplicate key {key:?}"))),
                "a repeated key must be refused in either order and at every level: {key}"
            );
        }
    }

    #[test]
    fn duplicate_index_keys_are_refused() {
        let d = Dir::new("duplicate-index");
        for shard in ["a.safetensors", "b.safetensors"] {
            d.file(
                shard,
                &write(&[("model.layers.0.weight", "F32", &[1], &[0; 4])]),
            );
        }
        for (text, key) in [
            (
                r#"{"weight_map":{"model.layers.0.weight":"a.safetensors","model.layers.0.weight":"b.safetensors"}}"#,
                "model.layers.0.weight",
            ),
            (
                r#"{"weight_map":{"model.layers.0.weight":"b.safetensors","model.layers.0.weight":"a.safetensors"}}"#,
                "model.layers.0.weight",
            ),
            (
                r#"{"weight_map":{"model.layers.0.weight":"a.safetensors"},"weight_map":{"model.layers.0.weight":"b.safetensors"}}"#,
                "weight_map",
            ),
        ] {
            d.file(Folder::INDEX, text.as_bytes());
            let Err(e) = Folder::open(&d.0) else {
                panic!("a duplicate index key must be refused: {key}")
            };
            assert!(
                e.to_string().contains(&format!("duplicate key {key:?}")),
                "{e}"
            );
        }
    }

    #[test]
    fn overlapping_ranges_are_refused() {
        let text = r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let mut file = (text.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(text.as_bytes());
        file.extend_from_slice(&[0; 4]);
        assert!(
            matches!(Safetensors::parse(file), Err(Error::Format(e)) if e.contains("share bytes"))
        );
    }

    /// A scratch folder under the system temp directory, removed on drop.
    struct Dir(std::path::PathBuf);

    impl Dir {
        fn new(tag: &str) -> Dir {
            let d = std::env::temp_dir().join(format!("hf-folder-{tag}-{}", std::process::id()));
            std::fs::create_dir_all(&d).unwrap();
            Dir(d)
        }

        /// `name` in the folder holding `bytes`.
        fn file(&self, name: &str, bytes: &[u8]) {
            std::fs::write(self.0.join(name), bytes).unwrap();
        }

        /// An index whose `weight_map` is `pairs`.
        fn index(&self, pairs: &[(&str, &str)]) {
            let map: serde_json::Map<String, Value> = pairs
                .iter()
                .map(|(n, f)| (n.to_string(), Value::from(*f)))
                .collect();
            let text = serde_json::json!({ "metadata": {}, "weight_map": map }).to_string();
            self.file(Folder::INDEX, text.as_bytes());
        }
    }

    impl Drop for Dir {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).unwrap();
        }
    }

    /// Every dtype a quantized checkpoint stores opens and is read as its stored bytes with its
    /// element size; the f32 read refuses each one that is not BF16, F16 or F32 by name.
    #[test]
    fn every_stored_dtype_is_held_raw() {
        let bytes: Vec<u8> = (0..16u8).collect();
        for (name, size) in [
            ("U8", 1),
            ("I8", 1),
            ("U16", 2),
            ("U32", 4),
            ("I32", 4),
            ("I64", 8),
            ("BOOL", 1),
            ("F8_E4M3", 1),
            ("F8_E5M2", 1),
            ("F8_E8M0", 1),
            ("BF16", 2),
            ("F16", 2),
            ("F32", 4),
        ] {
            let st = Safetensors::parse(write(&[("t", name, &[2, 8 / size], &bytes)]))
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            let raw = st.raw("t").unwrap();
            assert_eq!((raw.dtype.name(), raw.dtype.size()), (name, size));
            assert_eq!(raw.bytes, &bytes[..16]);
            let widened = matches!(name, "BF16" | "F16" | "F32");
            assert_eq!(st.tensor("t").is_ok(), widened, "{name}");
        }
    }

    /// A folder of two shards opens as one table: each name reads from its own shard, a range of
    /// rows is that range of the stored bytes, and a range of columns is each row's piece in order.
    #[test]
    fn a_folder_reads_tensors_rows_and_columns_from_its_shards() {
        let d = Dir::new("read");
        let a: Vec<u8> = (0..24u8).collect();
        let b: Vec<u8> = (100..112u8).collect();
        d.file("one.safetensors", &write(&[("a", "U16", &[3, 4], &a)]));
        d.file(
            "two.safetensors",
            &write(&[("b", "U8", &[12], &b), ("c", "F32", &[], &[0; 4])]),
        );
        d.index(&[
            ("a", "one.safetensors"),
            ("b", "two.safetensors"),
            ("c", "two.safetensors"),
        ]);
        let f = Folder::open(&d.0).unwrap();
        // A shard in a folder of the folder, as FP4-Local's `tensors/native-QB.safetensors`.
        std::fs::create_dir(d.0.join("sub")).unwrap();
        d.file("sub/one.safetensors", &write(&[("a", "U16", &[3, 4], &a)]));
        d.index(&[
            ("a", "sub/one.safetensors"),
            ("b", "two.safetensors"),
            ("c", "two.safetensors"),
        ]);
        assert_eq!(
            Folder::open(&d.0).unwrap().raw("a").unwrap().bytes,
            a.as_slice()
        );
        d.index(&[
            ("a", "one.safetensors"),
            ("b", "two.safetensors"),
            ("c", "two.safetensors"),
        ]);
        assert_eq!(f.names().collect::<Vec<_>>(), ["a", "b", "c"]);
        assert_eq!(f.raw("b").unwrap().bytes, b.as_slice());
        assert!(matches!(f.raw("d"), Err(Error::Missing(n)) if n == "d"));
    }

    /// An index that disagrees with its shards, a shard named by a path, a shard whose ranges
    /// overlap, a dtype outside the list and an index with no tensors are each refused by name.
    #[test]
    fn a_folder_refuses_what_its_index_and_shards_disagree_on() {
        let refused = |d: &Dir, want: &str| match Folder::open(&d.0) {
            Err(e) => assert!(e.to_string().contains(want), "want {want:?}: {e}"),
            Ok(_) => panic!("want {want:?}: opened"),
        };
        let d = Dir::new("refuse");
        d.file(
            "s.safetensors",
            &write(&[("a", "F32", &[1], &[0; 4]), ("b", "F32", &[1], &[0; 4])]),
        );
        d.index(&[
            ("a", "s.safetensors"),
            ("b", "s.safetensors"),
            ("z", "s.safetensors"),
        ]);
        refused(
            &d,
            "z: the index puts it in s.safetensors, which does not hold it",
        );
        d.index(&[("a", "s.safetensors")]);
        refused(
            &d,
            "b: in s.safetensors, which the index does not name for it",
        );
        for outside in [
            "../s.safetensors",
            "/s.safetensors",
            "sub//s.safetensors",
            "sub/./s.safetensors",
            "C:/s.safetensors",
        ] {
            d.index(&[("a", outside)]);
            refused(
                &d,
                &format!("a's shard \"{}\" is not a path inside the folder", outside),
            );
        }
        let text = r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let mut overlap = (text.len() as u64).to_le_bytes().to_vec();
        overlap.extend_from_slice(text.as_bytes());
        overlap.extend_from_slice(&[0; 4]);
        d.file("s.safetensors", &overlap);
        d.index(&[("a", "s.safetensors"), ("b", "s.safetensors")]);
        refused(&d, "s.safetensors: a and b share bytes");
        d.file("s.safetensors", &write(&[("a", "F4", &[2], &[0])]));
        d.index(&[("a", "s.safetensors")]);
        refused(&d, "a is F4, a type this reader does not hold");
        d.index(&[]);
        refused(&d, "weight_map holds no tensors");
        std::fs::remove_file(d.0.join(Folder::INDEX)).unwrap();
        refused(&d, Folder::INDEX);
    }

    /// `name`, a real checkpoint's shard header kept gzipped in the fixtures, written as that
    /// shard in `d`: its length, the header, then its data section as a hole of the size its
    /// entries end at, so the file has the real layout and no tensor bytes.
    fn header_shard(d: &Dir, file: &str, gz: &[u8]) {
        use std::io::Read;
        let mut header = Vec::new();
        flate2::read::GzDecoder::new(gz)
            .read_to_end(&mut header)
            .unwrap();
        let entries: serde_json::Map<String, Value> = serde_json::from_slice(&header).unwrap();
        let data = entries
            .values()
            .filter_map(|e| e.get("data_offsets")?.get(1)?.as_u64())
            .max()
            .unwrap();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&header);
        d.file(file, &bytes);
        let f = std::fs::OpenOptions::new()
            .write(true)
            .open(d.0.join(file))
            .unwrap();
        f.set_len(8 + header.len() as u64 + data).unwrap();
    }

    /// Two shards of nvidia/GLM-5.3-Flash-NVFP4 (their real headers, data absent) and the
    /// index's entries for them open as a folder, the ModelOpt triple and the BF16 tensors held
    /// with the index's names; a shard of deepseek-ai/DeepSeek-V4.1-Flash holds I8 and F8_E8M0.
    #[test]
    fn real_checkpoint_headers_open_without_their_data() {
        use std::io::Read;
        let d = Dir::new("real");
        let fixture = |name: &str| {
            std::fs::read(format!(
                "{}/../hf/tests/fixtures/checkpoint-headers/{name}",
                env!("CARGO_MANIFEST_DIR")
            ))
            .unwrap()
        };
        for n in [1, 2] {
            header_shard(
                &d,
                &format!("model-0000{n}-of-00033.safetensors"),
                &fixture(&format!(
                    "glm-5.3-flash-nvfp4-model-0000{n}-of-00033.header.json.gz"
                )),
            );
        }
        let mut index = Vec::new();
        flate2::read::GzDecoder::new(
            fixture("glm-5.3-flash-nvfp4-index-00001-00002.json.gz").as_slice(),
        )
        .read_to_end(&mut index)
        .unwrap();
        d.file(Folder::INDEX, &index);
        let f = Folder::open(&d.0).unwrap();
        assert_eq!(f.names().count(), 3855 + 1826);
        let mut count = BTreeMap::new();
        for name in f.names() {
            *count.entry(f.raw(name).unwrap().dtype.name()).or_insert(0) += 1;
        }
        assert_eq!(
            count.into_iter().collect::<Vec<_>>(),
            [
                ("BF16", 565),
                ("F32", 2092),
                ("F8_E4M3", 1513),
                ("U8", 1511)
            ]
        );
        let stem = "model.language_model.layers.3.mlp.experts.0.gate_proj";
        let w = f.raw(&format!("{stem}.weight")).unwrap();
        assert_eq!((w.dtype, w.shape), (Dtype::U8, [2048, 2048].as_slice()));

        header_shard(
            &d,
            "v41.safetensors",
            &fixture("deepseek-v4.1-flash-model-00024-of-00048.header.json.gz"),
        );
        let st = Safetensors::open(&d.0.join("v41.safetensors")).unwrap();
        let mut count = BTreeMap::new();
        for name in st.names() {
            *count
                .entry(st.entry(name).unwrap().dtype.name())
                .or_insert(0) += 1;
        }
        assert_eq!(
            count.into_iter().collect::<Vec<_>>(),
            [
                ("BF16", 5),
                ("F32", 9),
                ("F8_E4M3", 8),
                ("F8_E8M0", 1160),
                ("I8", 1152)
            ]
        );
    }

    /// A stack holds HF's expert `e` at `e` whatever order the tensors arrive in, with ggml dims;
    /// ids other than `0..count`, a tensor that is not 2-D and a shape unlike expert 0's are
    /// refused by name.
    #[test]
    fn an_expert_stack_holds_expert_e_at_e() {
        type ById<'a> = BTreeMap<usize, (&'a str, &'a [usize])>;
        let m = [4usize, 8];
        let by_id = ById::from([(2, ("e2", &m[..])), (0, ("e0", &m)), (1, ("e1", &m))]);
        assert_eq!(
            expert_stack(&by_id, 3).unwrap(),
            (vec!["e0".into(), "e1".into(), "e2".into()], vec![8, 4, 3])
        );
        for (by_id, count, want) in [
            (
                by_id.clone(),
                4,
                "e0: the stack's 3 expert ids are not 0..4",
            ),
            (by_id, 2, "e0: the stack's 3 expert ids are not 0..2"),
            (
                ById::from([(0, ("e0", &m[..])), (2, ("e2", &m))]),
                2,
                "e0: the stack's 2 expert ids are not 0..2",
            ),
            (
                ById::from([(0, ("e0", &[4][..]))]),
                1,
                "e0: expert projection [4] is not 2-D",
            ),
            (
                ById::from([(0, ("e0", &m[..])), (1, ("e1", &[8, 4]))]),
                2,
                "e1: shape differs from e0's [4, 8]",
            ),
            (ById::new(), 0, "an expert stack with no expert"),
        ] {
            let e = expert_stack(&by_id, count).unwrap_err().to_string();
            assert!(e.contains(want), "{e}");
        }
    }
}
