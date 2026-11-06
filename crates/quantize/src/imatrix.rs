//! The importance matrix as llama-quantize reads it (`common_imatrix_load`, `tools/quantize/quantize.cpp`): a GGUF
//! file whose entries are `<tensor>.in_sum2` and `<tensor>.counts` pairs, with `imatrix.datasets`,
//! `imatrix.chunk_count` and `imatrix.chunk_size` in its metadata. What the quantizer needs of it: which tensors it
//! has an entry for, the importance values of one, and the numbers `quantize.imatrix.*` records in the output.

use crate::{Error, Result};
use gguf::Value;
use std::collections::HashMap;
use std::path::Path;

/// A loaded importance matrix.
pub struct Imatrix {
    /// The path as the command line gave it: `quantize.imatrix.file` holds its first 127 bytes.
    pub path: String,
    gguf: gguf::Gguf,
    /// Entry name to the number of `in_sum2` values.
    entries: HashMap<String, u64>,
    /// `imatrix.chunk_count`.
    pub chunk_count: u32,
    /// `imatrix.datasets`.
    pub datasets: Vec<String>,
}

impl Imatrix {
    /// Open `path`; a file llama-quantize refuses (no metadata, an entry without both of its tensors) is refused by
    /// name.
    pub fn open(path: &str) -> Result<Imatrix> {
        let g = gguf::Gguf::open(Path::new(path))
            .map_err(|e| Error::refused(format!("{path}: {e}")))?;
        let (Some(ds), Some(Value::U32(chunk_count)), Some(_)) = (
            g.value("imatrix.datasets"),
            g.value("imatrix.chunk_count"),
            g.value("imatrix.chunk_size"),
        ) else {
            return Err(Error::refused(format!(
                "{path}: missing imatrix metadata (imatrix.datasets, a u32 imatrix.chunk_count and imatrix.chunk_size)"
            )));
        };
        let chunk_count = *chunk_count;
        let datasets = match ds {
            Value::Array(a) => a
                .iter()
                .filter_map(|v| match v {
                    Value::String(s) => Some(s.clone()),
                    _ => None,
                })
                .collect(),
            _ => Vec::new(),
        };
        let mut stems: HashMap<String, (Option<u64>, bool, bool)> = HashMap::new();
        for t in g.iter_tensors() {
            let n = t.dims.iter().product::<u64>();
            if let Some(s) = t.name.strip_suffix(".in_sum2") {
                stems.entry(s.to_string()).or_default().0 = Some(n);
            } else if let Some(s) = t.name.strip_suffix(".counts") {
                stems.entry(s.to_string()).or_default().1 = true;
            } else if let Some(s) = t.name.strip_suffix(".in_sum") {
                stems.entry(s.to_string()).or_default().2 = true;
            }
        }
        let mut entries = HashMap::with_capacity(stems.len());
        for (name, (sum2, counts, _)) in stems {
            match (sum2, counts) {
                (Some(n), true) => {
                    entries.insert(name, n);
                }
                _ => {
                    return Err(Error::refused(format!(
                        "{path}: mismatched sums and counts for {name}"
                    )));
                }
            }
        }
        Ok(Imatrix {
            path: path.to_string(),
            gguf: g,
            entries,
            chunk_count,
            datasets,
        })
    }

    /// The number of entries (`quantize.imatrix.entries_count`).
    #[must_use]
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// The number of importance values the entry for `name` holds, when there is one.
    #[must_use]
    pub fn len_of(&self, name: &str) -> Option<u64> {
        self.entries.get(name).copied()
    }

    /// The importance values for `name`: `ne0` per expert, as llama-quantize hands them to its quantizer.
    pub fn weights(&self, name: &str) -> Result<Vec<f32>> {
        gguf::iq_quant::imatrix_weights(&self.gguf, name).map_err(|e| Error::refused(e.to_string()))
    }
}
