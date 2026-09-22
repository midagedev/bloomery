//! Resident weights (package P10): every tensor of the model file uploaded
//! once, in the device format its kernel consumes (docs/gpu-design.md
//! decision 4 — format conversion is load-time work; nothing in this file
//! runs per step). A [`Weights`] owns one contiguous block range plus,
//! optionally, the non-block tensors, so a `Stage` loads exactly its own
//! layers. The gate (`gate_p10`) pins the uploads against the per-gate
//! reference packings bit for bit.

use crate::GpuError;
use crate::q5::{pack_q5_0, pack_q5_1};
use crate::tensor::DeviceTensor;
use cuda_core::CudaStream;
use gguf::quant::{GgmlType, half_to_f32};
use gguf::{Gguf, TensorInfo};
// The Q8_0 block `q8_0_planes` packs; the gate binaries name it through here.
pub use model::attn::Q8Block;
use std::collections::BTreeMap;
use std::ops::Range;

/// One resident weight in the format its kernel loads. A new meaning gets a
/// new variant: derived weights have their own (`Q8_0Derived`) and never
/// travel as a file tensor.
pub enum DevWeight {
    /// Q3_K/Q4_K/Q6_K file tensor: the raw row stream as little-endian u32
    /// words, the final word zero-padded when the stream is not 4-aligned —
    /// the kernels address rows by BYTE offset (`row * row_bytes`), so the
    /// words are the flat stream, never per-row padded. A 3-D expert stack
    /// stays one flat tensor of `dims[1]·dims[2]` rows: the `_sel` kernels
    /// address expert e as rows `e·R .. (e+1)·R`, which is the same flat
    /// layout. `k` is a multiple of 256; row bytes are `type_size · k/256`.
    KQuant {
        ty: GgmlType,
        w: DeviceTensor<u32>,
        k: usize,
    },
    /// Q5_0 file tensor in the `pack_q5_0` layout: `q_stride + k/32` words
    /// per row, `q_stride = 256·ceil(k/1024)`. `k` is a multiple of 32.
    Q5_0 { w: DeviceTensor<u32>, k: usize },
    /// Q5_1 file tensor in the `pack_q5_1` layout: `q_stride + 2·k/32` words
    /// per row. `k` is a multiple of 32.
    Q5_1 { w: DeviceTensor<u32>, k: usize },
    /// Q8_0 in the q8f32 two-plane layout: `qs` rows × k/4 u32 words (code j
    /// of a 32-value block in word j/4, byte j%4) and `d` rows × k/32 f32
    /// block scales converted from the f16 storage at load. No file tensor
    /// fills this variant today — this reader has no Q8_0 arm and refuses
    /// such a file at open — but the format is the contract the derived
    /// variant shares.
    Q8_0 {
        qs: DeviceTensor<u32>,
        d: DeviceTensor<f32>,
        k: usize,
    },
    /// A derived weight — computed at load by an architecture's plan, never
    /// read from the file — in the same two planes as `Q8_0` (`qs` rows ×
    /// k/4 words, `d` rows × k/32 scales); its row geometry is the plan's
    /// ([`ChainBody::derive`](crate::model::ChainBody::derive)). A distinct
    /// variant so nothing can read derived bytes as a file tensor or vice
    /// versa.
    Q8_0Derived {
        qs: DeviceTensor<u32>,
        d: DeviceTensor<f32>,
        k: usize,
    },
    /// F32 file tensor: rows × k f32, row-major with `k = dims[0]` (the
    /// contiguous axis the gemv kernels dot along). 1-D norm vectors are
    /// rows = 1.
    F32 { w: DeviceTensor<f32>, k: usize },
}

impl DevWeight {
    /// Rows of the resident tensor. An expert stack counts every expert's
    /// rows; a derived weight, every row its plan uploaded.
    pub fn rows(&self) -> usize {
        match self {
            DevWeight::KQuant { w, .. } | DevWeight::Q5_0 { w, .. } | DevWeight::Q5_1 { w, .. } => {
                w.rows()
            }
            DevWeight::Q8_0 { d, .. } | DevWeight::Q8_0Derived { d, .. } => d.rows(),
            DevWeight::F32 { w, .. } => w.rows(),
        }
    }

    /// Values per row: `dims[0]` of a file tensor, the plan's row width for a
    /// derived weight.
    pub(crate) fn k(&self) -> usize {
        match self {
            DevWeight::KQuant { k, .. }
            | DevWeight::Q5_0 { k, .. }
            | DevWeight::Q5_1 { k, .. }
            | DevWeight::Q8_0 { k, .. }
            | DevWeight::Q8_0Derived { k, .. }
            | DevWeight::F32 { k, .. } => *k,
        }
    }

    /// Device bytes held across all planes.
    #[must_use]
    pub(crate) fn resident_bytes(&self) -> usize {
        match self {
            DevWeight::KQuant { w, .. } | DevWeight::Q5_0 { w, .. } | DevWeight::Q5_1 { w, .. } => {
                w.buf().len() * 4
            }
            DevWeight::Q8_0 { qs, d, .. } | DevWeight::Q8_0Derived { qs, d, .. } => {
                qs.buf().len() * 4 + d.buf().len() * 4
            }
            DevWeight::F32 { w, .. } => w.buf().len() * 4,
        }
    }
}

/// The resident weights of a layer range: every `blk.L.*` tensor with L in
/// `layers`, what the architecture derives for those blocks, and — when
/// loaded with globals — the non-block tensors. Load-time only; per-step
/// code reads through `get` and never allocates.
pub struct Weights {
    by_name: BTreeMap<String, DevWeight>,
}

impl Weights {
    /// Upload the tensors of `layers` (and the non-block tensors when
    /// `globals`) in the device format each kernel consumes. Packs and
    /// uploads tensor by tensor, so at most one host packing of one tensor
    /// is alive at a time. A tensor whose type has no device format is an
    /// error naming the tensor and type — never a silent skip. File tensors
    /// only: what an architecture derives from them is filed afterwards,
    /// through `Weights::insert_derived`.
    pub fn load(
        stream: &CudaStream,
        gguf: &Gguf,
        layers: Range<usize>,
        globals: bool,
    ) -> Result<Weights, GpuError> {
        let n_layers =
            gguf.block_count()
                .ok_or(GpuError::metadata("Weights::load", "block_count"))? as usize;
        if layers.start > layers.end || layers.end > n_layers {
            return Err(GpuError::shape(
                "Weights::load",
                format!("layer range {layers:?} outside 0..{n_layers}"),
            ));
        }
        let mut by_name = BTreeMap::new();
        for t in gguf.iter_tensors() {
            let take = match block_index(&t.name)? {
                Some(l) => layers.contains(&l),
                None => globals,
            };
            if take {
                let dw = upload_file_tensor(stream, gguf, t)?;
                by_name.insert(t.name.clone(), dw);
            }
        }
        Ok(Weights { by_name })
    }

    /// File `w` as a derived weight under `name`. The `derived.` prefix keeps
    /// it out of the file-tensor namespace: a name without it is refused, and
    /// so is a name already resident, so a derived weight never shadows a
    /// file tensor or another derived one.
    pub(crate) fn insert_derived(&mut self, name: String, w: DevWeight) -> Result<(), GpuError> {
        derived_slot_free(&self.by_name, &name)?;
        self.by_name.insert(name, w);
        Ok(())
    }

    /// The resident weight for `name` — a file tensor name, or the `derived.`
    /// name an architecture filed a derived weight under.
    pub fn get(&self, name: &str) -> Option<&DevWeight> {
        self.by_name.get(name)
    }

    /// Every resident name, sorted.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.by_name.keys().map(String::as_str)
    }

    /// Total device bytes held (file tensors plus derived).
    pub fn resident_bytes(&self) -> usize {
        self.by_name.values().map(DevWeight::resident_bytes).sum()
    }
}

/// The prefix every derived weight's name carries.
const DERIVED_PREFIX: &str = "derived.";

/// Err unless `name` is free for a derived weight in `by_name`: under
/// [`DERIVED_PREFIX`] and not yet resident. Generic over the value so the
/// rule is checkable without a device.
fn derived_slot_free<V>(by_name: &BTreeMap<String, V>, name: &str) -> Result<(), GpuError> {
    if !name.starts_with(DERIVED_PREFIX) {
        return Err(GpuError::shape(
            "Weights::insert_derived",
            format!("{name:?} is not a {DERIVED_PREFIX:?} name"),
        ));
    }
    if by_name.contains_key(name) {
        return Err(GpuError::shape(
            "Weights::insert_derived",
            format!("{name:?} is already resident"),
        ));
    }
    Ok(())
}

/// `Some(L)` for a `blk.L.*` tensor name, `None` for a non-block name; a
/// `blk.` name whose layer does not parse is an error naming the tensor.
fn block_index(name: &str) -> Result<Option<usize>, GpuError> {
    let Some(rest) = name.strip_prefix("blk.") else {
        return Ok(None);
    };
    let l = rest.split('.').next().unwrap_or_default();
    match l.parse::<usize>() {
        Ok(l) => Ok(Some(l)),
        Err(_) => Err(GpuError::shape(
            "Weights::load",
            format!("tensor {name:?} is not blk.<layer>.*"),
        )),
    }
}

/// Rows of a tensor as the kernels count them: the product of dims[1..]
/// (1 for a 1-D tensor) — a 3-D expert stack is its experts' rows stacked.
fn tensor_rows(t: &TensorInfo) -> usize {
    t.dims[1..].iter().product::<u64>() as usize
}

/// Resident device bytes a file tensor of this type and shape takes in the
/// formats above (`rows` as [`tensor_rows`] counts them); `None` when the
/// type has no device format. The uploads follow the same arithmetic, so a
/// census totals check against a loaded `Weights::resident_bytes` pins the
/// two together.
#[must_use]
pub fn resident_size(ty: GgmlType, k: usize, rows: usize) -> Option<usize> {
    let q_stride_words = |k: usize| 256 * (k / 32).div_ceil(32);
    match ty {
        GgmlType::F32 => Some(rows * k * 4),
        GgmlType::Q5_0 => Some(rows * (q_stride_words(k) + k / 32) * 4),
        GgmlType::Q5_1 => Some(rows * (q_stride_words(k) + 2 * k / 32) * 4),
        GgmlType::Q3_K | GgmlType::Q4_K | GgmlType::Q6_K => {
            let blck = ty.blck_size()? as usize;
            if !k.is_multiple_of(blck) {
                return None;
            }
            let row_bytes = ty.type_size()? as usize * (k / blck);
            Some(row_bytes * rows.div_ceil(4) * 4)
        }
        GgmlType::F16 | GgmlType::Q5_K | GgmlType::Unknown(_) => None,
    }
}

/// Pack and upload one file tensor in its kernel's device format.
fn upload_file_tensor(
    stream: &CudaStream,
    gguf: &Gguf,
    t: &TensorInfo,
) -> Result<DevWeight, GpuError> {
    let name = t.name.as_str();
    let k = t.dims[0] as usize;
    let rows = tensor_rows(t);
    if k == 0 || rows == 0 {
        return Err(GpuError::shape(
            "Weights::load",
            format!("tensor {name} has a zero dimension"),
        ));
    }
    let bytes = gguf.data(t)?;
    match t.ty {
        GgmlType::Q3_K | GgmlType::Q4_K | GgmlType::Q6_K => {
            let blck = t.ty.blck_size().ok_or_else(|| {
                GpuError::shape(
                    "Weights::load",
                    format!("ggml type of {name} has no block size"),
                )
            })? as usize;
            let rb = t.ty.type_size().ok_or_else(|| {
                GpuError::shape(
                    "Weights::load",
                    format!("ggml type of {name} has no type size"),
                )
            })? as usize
                * (k / blck);
            if bytes.len() < rb * rows {
                return Err(GpuError::shape(
                    "Weights::load",
                    format!(
                        "tensor {name} holds {} bytes, rows need {}",
                        bytes.len(),
                        rb * rows
                    ),
                ));
            }
            // The gates' packing verbatim (gate_p1/gate_p9): the whole row
            // span as one word stream, so the kernels' byte-offset row
            // addressing sees contiguous rows.
            let words = words_of(&bytes[..rb * rows]);
            if !words.len().is_multiple_of(rows) {
                return Err(GpuError::shape(
                    "Weights::load",
                    format!(
                        "tensor {name} packs {} words not divisible by \
                     {rows} rows — the row stream cannot be represented per row",
                        words.len()
                    ),
                ));
            }
            Ok(DevWeight::KQuant {
                ty: t.ty,
                w: DeviceTensor::upload(stream, &words, rows, words.len() / rows)?,
                k,
            })
        }
        GgmlType::Q5_0 => {
            let packed = pack_q5_0(bytes, k, rows)?;
            let cols = 256 * (k / 32).div_ceil(32) + k / 32;
            Ok(DevWeight::Q5_0 {
                w: DeviceTensor::upload(stream, &packed, rows, cols)?,
                k,
            })
        }
        GgmlType::Q5_1 => {
            let packed = pack_q5_1(bytes, k, rows)?;
            let cols = 256 * (k / 32).div_ceil(32) + 2 * k / 32;
            Ok(DevWeight::Q5_1 {
                w: DeviceTensor::upload(stream, &packed, rows, cols)?,
                k,
            })
        }
        GgmlType::F32 => {
            if bytes.len() < rows * k * 4 {
                return Err(GpuError::shape(
                    "Weights::load",
                    format!(
                        "tensor {name} holds {} bytes, rows×k×4 = {}",
                        bytes.len(),
                        rows * k * 4
                    ),
                ));
            }
            let vals: Vec<f32> = bytes[..rows * k * 4]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect();
            Ok(DevWeight::F32 {
                w: DeviceTensor::upload(stream, &vals, rows, k)?,
                k,
            })
        }
        GgmlType::F16 | GgmlType::Q5_K | GgmlType::Unknown(_) => Err(GpuError::shape(
            "Weights::load",
            format!("tensor {name} has type {} with no device format", t.ty),
        )),
    }
}

/// Raw little-endian bytes as u32 words; a length not divisible by 4
/// zero-pads the last word. Same semantics as the gates' `bytes_to_words`
/// (that helper lives in bloomery-gpu-gates, which depends on this crate —
/// the dependency direction forbids sharing it the other way).
fn words_of(b: &[u8]) -> Vec<u32> {
    b.chunks(4)
        .map(|c| {
            let mut w = [0u8; 4];
            w[..c.len()].copy_from_slice(c);
            u32::from_le_bytes(w)
        })
        .collect()
}

/// The q8f32 two-plane layout of Q8_0 blocks: per block 8 code words (code j
/// in word j/4, byte j%4) and one f32 scale converted from the stored f16
/// bits — the exact value the reference dequantizes with, so the device side
/// never does f16 arithmetic.
pub(crate) fn q8_0_planes(blocks: &[Q8Block]) -> (Vec<u32>, Vec<f32>) {
    let mut qs = Vec::with_capacity(blocks.len() * 8);
    let mut d = Vec::with_capacity(blocks.len());
    for b in blocks {
        let mut w = [0u32; 8];
        for (j, &q) in b.q.iter().enumerate() {
            w[j / 4] |= u32::from(q as u8) << (8 * (j % 4));
        }
        qs.extend_from_slice(&w);
        d.push(half_to_f32(b.d));
    }
    (qs, d)
}

#[cfg(test)]
mod tests {
    use super::derived_slot_free;
    use std::collections::BTreeMap;

    /// `insert_derived`'s guard, which runs before anything is filed: a name
    /// outside the `derived.` prefix is refused even when nothing holds it,
    /// and a `derived.` name already resident is refused.
    #[test]
    fn derived_slot_refuses_a_file_name_and_a_resident_name() {
        let mut by_name = BTreeMap::new();
        by_name.insert("output.weight".to_string(), ());
        by_name.insert("derived.a".to_string(), ());
        assert!(derived_slot_free(&by_name, "derived.b").is_ok());
        assert!(derived_slot_free(&by_name, "output_norm.weight").is_err());
        assert!(derived_slot_free(&by_name, "derived.a").is_err());
    }
}
