//! A safetensors file: its header and its tensors as f32.
//!
//! The layout: an 8-byte little-endian header length, the JSON header (name → `dtype`, `shape`,
//! `data_offsets` relative to the end of the header, plus an optional `__metadata__`), then the
//! data. BF16, F16 and F32 convert to f32 exactly; any other dtype is refused by name.

use std::collections::BTreeMap;
use std::path::Path;

use crate::Error;
use crate::json::{self, Json};

/// A stored element type this reader converts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype {
    Bf16,
    F16,
    F32,
}

impl Dtype {
    fn of(name: &str) -> Option<Dtype> {
        match name {
            "BF16" => Some(Dtype::Bf16),
            "F16" => Some(Dtype::F16),
            "F32" => Some(Dtype::F32),
            _ => None,
        }
    }

    fn size(self) -> usize {
        match self {
            Dtype::Bf16 | Dtype::F16 => 2,
            Dtype::F32 => 4,
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

/// A whole file in memory with its parsed header.
pub struct Safetensors {
    bytes: Vec<u8>,
    data: usize,
    entries: BTreeMap<String, Entry>,
}

/// A tensor converted to f32, row-major.
#[derive(Clone, Debug)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

fn bad(what: String) -> Error {
    Error::Safetensors(what)
}

impl Safetensors {
    /// Read and parse the file at `path`.
    pub fn open(path: &Path) -> Result<Safetensors, Error> {
        let bytes = std::fs::read(path).map_err(|e| Error::Io(path.display().to_string(), e))?;
        Safetensors::parse(bytes)
    }

    /// Parse a whole file's bytes: every entry's dtype is known, its byte range matches its shape,
    /// lies in the data section, and no two ranges overlap.
    pub fn parse(bytes: Vec<u8>) -> Result<Safetensors, Error> {
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
        let Json::Object(pairs) = json::parse(text)? else {
            return Err(bad("the header is not a JSON object".into()));
        };
        let mut entries = BTreeMap::new();
        for (name, v) in pairs {
            if name == "__metadata__" {
                continue;
            }
            let entry = entry(&name, &v, bytes.len() - data)?;
            entries.insert(name, entry);
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

    /// `name` converted to f32.
    pub fn tensor(&self, name: &str) -> Result<Tensor, Error> {
        let e = self
            .entries
            .get(name)
            .ok_or_else(|| Error::MissingWeight(name.to_string()))?;
        let raw = &self.bytes[self.data + e.start..self.data + e.end];
        let data = match e.dtype {
            Dtype::Bf16 => raw
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| f32::from_bits(u32::from(u16::from_le_bytes(*b)) << 16))
                .collect(),
            Dtype::F16 => raw
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| gguf::quant::half_to_f32(u16::from_le_bytes(*b)))
                .collect(),
            Dtype::F32 => raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect(),
        };
        Ok(Tensor {
            shape: e.shape.clone(),
            data,
        })
    }
}

fn entry(name: &str, v: &Json, data_len: usize) -> Result<Entry, Error> {
    let field = |key: &str| v.get(key).ok_or_else(|| bad(format!("{name}: no {key}")));
    let dtype_name = field("dtype")?
        .as_str()
        .ok_or_else(|| bad(format!("{name}: dtype is not a string")))?;
    let dtype = Dtype::of(dtype_name).ok_or_else(|| Error::Dtype {
        name: name.to_string(),
        dtype: dtype_name.to_string(),
    })?;
    let usizes = |key: &str| -> Result<Vec<usize>, Error> {
        let Json::Array(items) = field(key)? else {
            return Err(bad(format!("{name}: {key} is not an array")));
        };
        items
            .iter()
            .map(|d| {
                d.as_u64()
                    .and_then(|d| usize::try_from(d).ok())
                    .ok_or_else(|| bad(format!("{name}: {key} holds {}", d.kind())))
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
    let mut header = vec![(
        "__metadata__".to_string(),
        Json::Object(vec![("format".to_string(), Json::Str("pt".into()))]),
    )];
    let mut at = 0usize;
    for (name, dtype, shape, raw) in tensors {
        let ints = |v: &[usize]| Json::Array(v.iter().map(|d| Json::Int(d.to_string())).collect());
        header.push((
            (*name).to_string(),
            Json::Object(vec![
                ("dtype".into(), Json::Str((*dtype).to_string())),
                ("shape".into(), ints(shape)),
                ("data_offsets".into(), ints(&[at, at + raw.len()])),
            ]),
        ));
        at += raw.len();
    }
    let mut text = crate::render::dumps(&Json::Object(header), false);
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
        assert!(matches!(st.tensor("d"), Err(Error::MissingWeight(n)) if n == "d"));
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
            Err(Error::Safetensors(_))
        ));
        let mut short = write(&[("x", "F32", &[1], &four)]);
        short.pop();
        assert!(matches!(
            Safetensors::parse(short),
            Err(Error::Safetensors(_))
        ));
        assert!(matches!(
            Safetensors::parse(vec![1, 0, 0]),
            Err(Error::Safetensors(_))
        ));
        let mut huge = write(&[("x", "F32", &[1], &four)]);
        huge[..8].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(matches!(
            Safetensors::parse(huge),
            Err(Error::Safetensors(_))
        ));
    }

    #[test]
    fn overlapping_ranges_are_refused() {
        let text = r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let mut file = (text.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(text.as_bytes());
        file.extend_from_slice(&[0; 4]);
        assert!(
            matches!(Safetensors::parse(file), Err(Error::Safetensors(e)) if e.contains("share bytes"))
        );
    }
}
