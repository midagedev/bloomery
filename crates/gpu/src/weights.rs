//! Resident weights (package P10): every tensor of the model file uploaded
//! once, in the device format its kernel consumes (docs/gpu-design.md
//! decision 4 — format conversion is load-time work; nothing in this file
//! runs per step). A [`Weights`] owns one contiguous block range plus,
//! optionally, the non-block tensors, so a `Stage` loads exactly its own
//! layers — or, for a placed model, every segment its plan puts on one card
//! ([`Weights::load_placed`]). The formats and their byte arithmetic are
//! `model::placement::CardFormat`'s. The gates (`gate_p10`, `gate_load_v41`)
//! pin the uploads against independent host packings bit for bit.

use crate::GpuError;
use crate::q5::{pack_q5_0, pack_q5_1};
use crate::tensor::DeviceTensor;
use ::model::placement::{CardFormat, Device, Format, ModelTensor, Plan, Segment};
use cuda_core::CudaStream;
use gguf::quant::{GgmlType, dequant_row, half_to_f32};
use gguf::{Gguf, Split, TensorInfo};
// The Q8_0 block `q8_0_planes` packs; the gate binaries name it through here.
pub use gguf::quant::Q8Block;
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
    /// Q8_0 file tensor in the q8f32 two-plane layout: `qs` rows × k/4 u32
    /// words (code j of a 32-value block in word j/4, byte j%4) and `d` rows
    /// × k/32 f32 block scales converted from the f16 storage at load. The
    /// derived variant shares the format.
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
    /// F32 file tensor, or a BF16 one decoded to f32 at load: rows × k f32,
    /// row-major with `k = dims[0]` (the contiguous axis the gemv kernels dot
    /// along). 1-D norm vectors are rows = 1.
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
    pub fn resident_bytes(&self) -> usize {
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

    /// Upload every segment `plan` puts on card `card`, from `split`, in the
    /// segment's card format: a whole tensor, or an expert stack's experts
    /// `[0, n_l)` — the stack's leading rows, which the `_sel` kernels
    /// address as experts `0..n_l`. Segments on other devices are skipped.
    /// A segment whose upload is not the plan's `resident_bytes` for it is an
    /// error naming the tensor: plan and loader share one arithmetic, so a
    /// difference means the plan is not a plan of this file. `stream`'s
    /// context is made current first — allocations land in the calling
    /// thread's current context, and a process with one context per card
    /// must be on this card's.
    pub fn load_placed(
        stream: &CudaStream,
        split: &Split,
        plan: &Plan<'_>,
        card: usize,
    ) -> Result<Weights, GpuError> {
        stream.context().bind_to_thread()?;
        let mut by_name = BTreeMap::new();
        for row in &plan.rows {
            let t = plan.model.tensors.get(row.tensor).ok_or_else(|| {
                GpuError::shape(PLACED, format!("plan row names tensor {}", row.tensor))
            })?;
            for seg in row
                .segments
                .iter()
                .filter(|s| s.device == Device::Card(card))
            {
                let dw = upload_segment(stream, split, plan, t, seg)?;
                if by_name.insert(t.name.clone(), dw).is_some() {
                    return Err(placed_refusal(
                        t,
                        format!("has two segments on card {card}"),
                    ));
                }
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
/// type has no device format or the shape has no layout in it. The uploads
/// refuse and check by the same arithmetic ([`CardFormat::resident_bytes`]),
/// so a census totals check against a loaded `Weights::resident_bytes` pins
/// the two together.
#[must_use]
pub fn resident_size(ty: GgmlType, k: usize, rows: usize) -> Option<usize> {
    let (k, rows) = (u64::try_from(k).ok()?, u64::try_from(rows).ok()?);
    let bytes = CardFormat::of(ty)?.resident_bytes(ty, k, rows)?;
    usize::try_from(bytes).ok()
}

/// The `what` of every refusal `Weights::load_placed` makes.
const PLACED: &str = "Weights::load_placed";

/// `load_placed`'s refusal of tensor `t`.
fn placed_refusal(t: &ModelTensor, detail: String) -> GpuError {
    GpuError::shape(PLACED, format!("tensor {}: {detail}", t.name))
}

/// Upload plan segment `seg` of tensor `t`: find the tensor in `split`,
/// confirm it is the tensor the plan was made from, and upload the rows the
/// segment holds, which must lead the tensor.
fn upload_segment(
    stream: &CudaStream,
    split: &Split,
    plan: &Plan<'_>,
    t: &ModelTensor,
    seg: &Segment,
) -> Result<DevWeight, GpuError> {
    let Format::Card(format) = seg.format else {
        return Err(placed_refusal(t, format!("is on a card as {}", seg.format)));
    };
    let span = seg
        .span(t, plan.model.experts)
        .map_err(|e| GpuError::shape(PLACED, e.to_string()))?;
    if span.rows.start != 0 {
        let detail = format!("rows {:?} do not lead the tensor", span.rows);
        return Err(placed_refusal(t, detail));
    }
    let (s, info) = split
        .find(&t.name)
        .ok_or_else(|| placed_refusal(t, "is not in the split".to_string()))?;
    if s != t.shard || info.ty != t.ty || info.dims != t.dims || info.nbytes != t.file_bytes {
        let detail = format!(
            "the split's shard {s} {} {:?} ({} bytes) is not the plan's shard {} {} {:?} ({} bytes)",
            info.ty, info.dims, info.nbytes, t.shard, t.ty, t.dims, t.file_bytes
        );
        return Err(placed_refusal(t, detail));
    }
    let g = split
        .shard(s)
        .ok_or_else(|| placed_refusal(t, format!("shard {s} is not in the split")))?;
    let rows = usize::try_from(span.rows.end)
        .map_err(|_| placed_refusal(t, format!("{} rows pass usize", span.rows.end)))?;
    let dw = upload_rows(stream, PLACED, format, info, rows, g.data(info)?)?;
    if u64::try_from(dw.resident_bytes()).ok() != Some(seg.resident_bytes) {
        let detail = format!(
            "uploaded {} device bytes, the plan has {}",
            dw.resident_bytes(),
            seg.resident_bytes
        );
        return Err(placed_refusal(t, detail));
    }
    Ok(dw)
}

/// Pack and upload one file tensor in its kernel's device format.
fn upload_file_tensor(
    stream: &CudaStream,
    gguf: &Gguf,
    t: &TensorInfo,
) -> Result<DevWeight, GpuError> {
    let Some(format) = CardFormat::of(t.ty) else {
        return Err(GpuError::shape(
            "Weights::load",
            format!("tensor {} has type {} with no device format", t.name, t.ty),
        ));
    };
    upload_rows(
        stream,
        "Weights::load",
        format,
        t,
        tensor_rows(t),
        gguf.data(t)?,
    )
}

/// Pack the first `rows` rows of file tensor `t` — all of them, or an
/// expert stack's leading experts — from `bytes` (the tensor's file bytes,
/// or at least those rows') in card format `format`, and upload them.
/// Refuses exactly where [`CardFormat::resident_bytes`] has no size for the
/// rows, and checks the upload's device bytes against that size; `what`
/// names the caller in both errors.
fn upload_rows(
    stream: &CudaStream,
    what: &'static str,
    format: CardFormat,
    t: &TensorInfo,
    rows: usize,
    bytes: &[u8],
) -> Result<DevWeight, GpuError> {
    let name = t.name.as_str();
    let all_rows: u64 = t.dims[1..].iter().product();
    let layout = (rows as u64 <= all_rows)
        .then(|| format.resident_bytes(t.ty, t.dims[0], rows as u64))
        .flatten()
        .zip(usize::try_from(t.dims[0]).ok());
    let Some((size, k)) = layout else {
        let detail = format!(
            "tensor {name}: {rows} of its {all_rows} rows of {} {} values have no {format:?} layout",
            t.dims[0], t.ty
        );
        return Err(GpuError::shape(what, detail));
    };
    // File bytes per row: exact, because the reader sized the tensor from its type.
    let need = t.nbytes / all_rows * rows as u64;
    let Some(bytes) = usize::try_from(need).ok().and_then(|n| bytes.get(..n)) else {
        let detail = format!(
            "tensor {name} holds {} bytes, {rows} rows need {need}",
            bytes.len()
        );
        return Err(GpuError::shape(what, detail));
    };
    let dw = match format {
        // The gates' packing verbatim (gate_p1/gate_p9): the whole row span as
        // one word stream, so the kernels' byte-offset row addressing sees
        // contiguous rows.
        CardFormat::KQuant => {
            let words = words_of(bytes);
            DevWeight::KQuant {
                ty: t.ty,
                w: DeviceTensor::upload(stream, &words, rows, words.len() / rows)?,
                k,
            }
        }
        CardFormat::Q5_0 => {
            let packed = pack_q5_0(bytes, k, rows)?;
            let cols = packed.len() / rows;
            DevWeight::Q5_0 {
                w: DeviceTensor::upload(stream, &packed, rows, cols)?,
                k,
            }
        }
        CardFormat::Q5_1 => {
            let packed = pack_q5_1(bytes, k, rows)?;
            let cols = packed.len() / rows;
            DevWeight::Q5_1 {
                w: DeviceTensor::upload(stream, &packed, rows, cols)?,
                k,
            }
        }
        CardFormat::Q8_0Planes => {
            let blocks: Vec<Q8Block> = bytes
                .as_chunks::<34>()
                .0
                .iter()
                .map(Q8Block::from_bytes)
                .collect();
            let (qs, d) = q8_0_planes(&blocks);
            DevWeight::Q8_0 {
                qs: DeviceTensor::upload(stream, &qs, rows, k / 4)?,
                d: DeviceTensor::upload(stream, &d, rows, k / 32)?,
                k,
            }
        }
        CardFormat::F32 => {
            let vals: Vec<f32> = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect();
            DevWeight::F32 {
                w: DeviceTensor::upload(stream, &vals, rows, k)?,
                k,
            }
        }
        CardFormat::Bf16AsF32 => {
            let mut vals = vec![0.0f32; rows * k];
            dequant_row(t.ty, bytes, &mut vals).map_err(::model::ModelError::from)?;
            DevWeight::F32 {
                w: DeviceTensor::upload(stream, &vals, rows, k)?,
                k,
            }
        }
    };
    if u64::try_from(dw.resident_bytes()).ok() != Some(size) {
        return Err(GpuError::shape(
            what,
            format!(
                "tensor {name}: {format:?} upload holds {} device bytes, its layout {size}",
                dw.resident_bytes()
            ),
        ));
    }
    Ok(dw)
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
