//! A safetensors file: its header and its tensors, from a mapping of the file or from bytes.
//!
//! The layout: an 8-byte little-endian header length, the JSON header (name → `dtype`, `shape`,
//! `data_offsets` relative to the end of the header, plus an optional `__metadata__`), then the
//! data. The header is validated whole on open: every entry's dtype is known, its byte range
//! matches its shape and lies in the data section, and no two ranges overlap. A tensor is read
//! three ways: widened to f32 ([`Safetensors::tensor`], BF16, F16 and F32), as its stored bits
//! ([`Safetensors::raw`], [`Safetensors::bf16`]), or as integers ([`Safetensors::i64s`]).

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

/// A stored element type this reader holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype {
    Bf16,
    F16,
    F32,
    I64,
    Bool,
}

impl Dtype {
    fn of(name: &str) -> Option<Dtype> {
        match name {
            "BF16" => Some(Dtype::Bf16),
            "F16" => Some(Dtype::F16),
            "F32" => Some(Dtype::F32),
            "I64" => Some(Dtype::I64),
            "BOOL" => Some(Dtype::Bool),
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
        }
    }

    /// Bytes per element.
    #[must_use]
    pub fn size(self) -> usize {
        match self {
            Dtype::Bool => 1,
            Dtype::Bf16 | Dtype::F16 => 2,
            Dtype::F32 => 4,
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
        let Value::Object(pairs) = serde_json::from_str::<Value>(text)
            .map_err(|e| bad(format!("the header is not JSON: {e}")))?
        else {
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
        let data = match raw.dtype {
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
            Dtype::I64 | Dtype::Bool => {
                return Err(Error::Dtype {
                    name: name.to_string(),
                    dtype: raw.dtype.name().to_string(),
                });
            }
        };
        Ok(Tensor {
            shape: raw.shape.to_vec(),
            data,
        })
    }
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
    for (name, dtype, shape, raw) in tensors {
        text.push(',');
        text.push_str(&json(&Value::from(*name)));
        text.push_str(&format!(
            r#":{{"dtype":{},"shape":{},"data_offsets":{}}}"#,
            json(&Value::from(*dtype)),
            ints(shape),
            ints(&[at, at + raw.len()]),
        ));
        at += raw.len();
    }
    text.push('}');
    while !text.len().is_multiple_of(8) {
        text.push(' ');
    }
    let mut out = (text.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(text.as_bytes());
    for (_, _, _, raw) in tensors {
        out.extend_from_slice(raw);
    }
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
        let r = Safetensors::parse(write(&[("x", "I8", &[4], &four)]));
        assert!(
            matches!(r, Err(Error::Dtype { ref name, ref dtype }) if name == "x" && dtype == "I8"),
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
    fn overlapping_ranges_are_refused() {
        let text = r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let mut file = (text.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(text.as_bytes());
        file.extend_from_slice(&[0; 4]);
        assert!(
            matches!(Safetensors::parse(file), Err(Error::Format(e)) if e.contains("share bytes"))
        );
    }
}
