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
use crate::tensor::{DeviceTensor, window};
use crate::upload::{Stage, UploadRing, bytes_of};
use ::model::placement::host_lock::PageDrop;
use ::model::placement::{
    CardFormat, Device, Format, ModelTensor, ModelTensors, Plan, Role, Row, Segment,
};
use cuda_core::{CudaStream, DeviceCopy};
use gguf::quant::{GgmlType, dequant_row};
use gguf::{Gguf, Split, TensorInfo};
use runtime::words::stream_words;
// The Q8_0 block `q8_0_planes` packs; the gate binaries name it through here.
pub use gguf::quant::Q8Block;
use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::mem::ManuallyDrop;
use std::ops::Range;

/// The lm_head tensor's name, and the token embedding that serves as it when the file ties them.
pub const HEAD_TENSOR: &str = "output.weight";
const EMBED_TENSOR: &str = "token_embd.weight";

/// The file tensor that is `split`'s lm_head: `output.weight`, or `token_embd.weight` when the
/// file carries no `output.weight` (tied embeddings: llama.cpp's loader falls back to the token
/// embedding for the output: `src/models/qwen35.cpp:52`, the `output == NULL` arm). A file with
/// neither is answered `output.weight`, which every reader then refuses by name as absent.
#[must_use]
pub fn head_tensor(split: &Split) -> &'static str {
    if split.find(HEAD_TENSOR).is_none() && split.find(EMBED_TENSOR).is_some() {
        EMBED_TENSOR
    } else {
        HEAD_TENSOR
    }
}

/// One resident weight in the format its kernel loads. A new meaning gets a
/// new variant: derived weights have their own (`Q8_0Derived`) and never
/// travel as a file tensor.
pub enum DevWeight {
    /// A file tensor as the file stores its rows (`CardFormat::KQuant`): a
    /// Q3_K/Q4_K/Q5_K/Q6_K tensor, or a routed Q5_1, Q8_0, IQ3_XXS, IQ4_XS
    /// or IQ4_NL stack a program's plan
    /// puts on the card in the file's `block_q5_1`s, `block_q8_0`s,
    /// `block_iq3_xxs`es, `block_iq4_xs`es or `block_iq4_nl`s, which its
    /// `_sel` and GEMM entries read (`ty` says which). The raw row stream as
    /// little-endian u32
    /// words, zero-padded at its end to a whole number of words per row —
    /// the kernels address rows by BYTE offset (`row * row_bytes`), so the
    /// words are the flat stream, never per-row padded. A 3-D expert stack
    /// stays one flat tensor of `dims[1]·dims[2]` rows: the `_sel` kernels
    /// address expert e as rows `e·R .. (e+1)·R`, which is the same flat
    /// layout. `k` is a multiple of the type's block; row bytes are
    /// `type_size · k / blck_size` (a Q5_1 row of 640 values is 480 bytes,
    /// 120 words).
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
    /// × k/32 block scales, each the block's f16 bits as the file stores
    /// them. The derived variant shares the format.
    Q8_0 {
        qs: DeviceTensor<u32>,
        d: DeviceTensor<u16>,
        k: usize,
    },
    /// A derived weight — computed at load by an architecture's plan, never
    /// read from the file — in the same two planes as `Q8_0` (`qs` rows ×
    /// k/4 words, `d` rows × k/32 f16 scales); its row geometry is the
    /// plan's (the constructor's derive step, see `GpuModel::load_placed`). A
    /// distinct variant so nothing can read derived bytes as a file tensor
    /// or vice versa.
    Q8_0Derived {
        qs: DeviceTensor<u32>,
        d: DeviceTensor<u16>,
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

    /// The device buffers this weight holds, in bytes, in allocation order:
    /// each buffer's element count times its element size, read from the
    /// buffers themselves — what was allocated, never recomputed from the
    /// format, so it is the independent side of every check against
    /// [`CardFormat::buffer_bytes`].
    #[must_use]
    pub fn buffer_bytes(&self) -> Vec<usize> {
        match self {
            DevWeight::KQuant { w, .. } | DevWeight::Q5_0 { w, .. } | DevWeight::Q5_1 { w, .. } => {
                vec![held(w)]
            }
            DevWeight::Q8_0 { qs, d, .. } | DevWeight::Q8_0Derived { qs, d, .. } => {
                vec![held(qs), held(d)]
            }
            DevWeight::F32 { w, .. } => vec![held(w)],
        }
    }

    /// Device bytes held across all planes: the sum of [`DevWeight::buffer_bytes`].
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        self.buffer_bytes().into_iter().sum()
    }
}

/// Bytes `t`'s buffer holds.
fn held<T: cuda_core::DeviceCopy>(t: &DeviceTensor<T>) -> usize {
    t.buf().len() * std::mem::size_of::<T>()
}

/// The resident weights of a layer range: every `blk.L.*` tensor with L in
/// `layers`, what the architecture derives for those blocks, and — when
/// loaded with globals — the non-block tensors. Load-time only; per-step
/// code reads through `get` and never allocates.
pub struct Weights {
    by_name: BTreeMap<String, DevWeight>,
    /// The file ties its head to the token embedding ([`head_tensor`]): `output.weight` is the
    /// resident `token_embd.weight`.
    tied: bool,
}

impl Weights {
    /// Upload the tensors of `layers` (and the non-block tensors when
    /// `globals`) of `split` in the device format each kernel consumes: the
    /// names are chosen first — a `blk.` name without a layer number is an
    /// error before anything is uploaded — and [`Weights::load_where`]
    /// uploads them, each from its own shard. A tensor whose type has no
    /// device format is an error naming the tensor and type — never a silent
    /// skip. File tensors only: what an architecture derives from them is
    /// filed afterwards, through `Weights::insert_derived`. A file with the
    /// non-block tensors and no `output.weight` ties its head to the token
    /// embedding ([`head_tensor`]): `get("output.weight")` answers the one
    /// resident embedding, so every reader of the head finds it by its one
    /// name and no second copy is uploaded.
    pub fn load(
        stream: &CudaStream,
        split: &Split,
        layers: Range<usize>,
        globals: bool,
    ) -> Result<Weights, GpuError> {
        let n_layers = split
            .arch_get_u64("block_count")
            .ok_or(GpuError::metadata("Weights::load", "block_count"))?
            as usize;
        if layers.start > layers.end || layers.end > n_layers {
            return Err(GpuError::shape(
                "Weights::load",
                format!("layer range {layers:?} outside 0..{n_layers}"),
            ));
        }
        let mut keep = BTreeSet::new();
        for (_, t) in split.iter_tensors() {
            let take = match block_index(&t.name)? {
                Some(l) => layers.contains(&l),
                None => globals,
            };
            if take {
                keep.insert(t.name.as_str());
            }
        }
        let tied = globals && head_tensor(split) == EMBED_TENSOR;
        Weights::load_named(stream, split, |name| keep.contains(name), tied)
    }

    /// Upload every segment `plan` puts on card `card` ([`Weights::load_rows`]
    /// of the plan's rows), and — when `card_dontneed` — release each
    /// uploaded segment's file pages from the page cache as soon as its
    /// bytes are staged ([`PageDrop`]; the device copies read the ring's
    /// pinned slots, not the file, so no copy reads a released page): the plan names every later reader of the file, and none of them reads
    /// a card segment's bytes. Whole pages inside the segment's bytes only,
    /// so a page shared with a host segment or a neighbouring tensor stays;
    /// the token embedding and the engram tables, read from the file by
    /// every step, are never released.
    pub fn load_placed(
        stream: &CudaStream,
        split: &Split,
        plan: &Plan<'_>,
        card: usize,
        card_dontneed: bool,
    ) -> Result<Weights, GpuError> {
        let mut release = card_dontneed.then(|| PageDrop::new(split));
        load_segments(
            stream,
            split,
            plan.model,
            &plan.rows,
            card,
            release.as_mut(),
        )
    }

    /// Upload every segment of `rows` (placement rows of `model`'s tensors)
    /// on card `card`, from `split`, in the segment's card format: a whole
    /// tensor, or an expert stack's listed experts, their rows gathered into
    /// one buffer in the list's order, so the `_sel` kernels address expert
    /// `list[s]` as slot `s`. Segments on other devices are skipped; a tensor
    /// with two segments on one card is refused. A segment whose upload is
    /// not its `resident_bytes` is an error naming the tensor: placement and
    /// loader share one arithmetic, so a difference means the rows are not a
    /// placement of this file. `stream`'s context is made current first —
    /// allocations land in the calling thread's current context, and a
    /// process with one context per card must be on this card's.
    pub fn load_rows(
        stream: &CudaStream,
        split: &Split,
        model: &ModelTensors,
        rows: &[Row],
        card: usize,
    ) -> Result<Weights, GpuError> {
        load_segments(stream, split, model, rows, card, None)
    }

    /// Upload every tensor of `split` whose name `keep` accepts, each from
    /// its own shard, in the device format its kernel consumes — the one
    /// uploader of a file without a placement plan: [`Weights::load`]'s layer
    /// range, or the one layer or one piece of a step a gate makes resident.
    /// A kept tensor whose type has no device format is an error naming it.
    /// One staging ring carries the whole load's copies, and one synchronize
    /// ends them.
    pub fn load_where(
        stream: &CudaStream,
        split: &Split,
        keep: impl FnMut(&str) -> bool,
    ) -> Result<Weights, GpuError> {
        Weights::load_named(stream, split, keep, false)
    }

    /// [`Weights::load_where`], and with `tied` the token embedding is also the head
    /// ([`Weights::get`]).
    fn load_named(
        stream: &CudaStream,
        split: &Split,
        mut keep: impl FnMut(&str) -> bool,
        tied: bool,
    ) -> Result<Weights, GpuError> {
        stream.context().bind_to_thread()?;
        let mut by_name = BTreeMap::new();
        let mut kept: Vec<(usize, &TensorInfo)> = Vec::new();
        for (shard, t) in split.iter_tensors() {
            if keep(&t.name) {
                kept.push((shard, t));
            }
        }
        // The ring is made after `by_name` and drops before it: its drop
        // drains the copies, so no buffer of a refused load is freed while a
        // copy still writes it.
        let budget = kept.iter().map(|&(_, t)| upload_budget(t)).sum();
        let mut ring = UploadRing::new(stream, budget)?;
        for (shard, t) in kept {
            let gguf = split.shard(shard).ok_or_else(|| {
                GpuError::shape(
                    "Weights::load_where",
                    format!("{} names shard {shard}, which the split lacks", t.name),
                )
            })?;
            by_name.insert(
                t.name.clone(),
                upload_file_tensor_with(&mut ring, stream, gguf, t)?,
            );
        }
        ring.finish(stream)?;
        Ok(Weights { by_name, tied })
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

    /// Move the K-quant weights `parts` — one type, one row width — into one
    /// row stream filed as the derived weight `joint`: part `i`'s rows follow
    /// part `i − 1`'s, so one gemv over `joint` computes every part's rows,
    /// each row the bits its own tensor gives (the kernels address a row by
    /// its byte offset). The parts leave the map and their buffers are freed,
    /// so nothing is resident twice and the resident bytes are unchanged.
    /// Refused unless every part is resident as a K-quant of the first
    /// part's type and width, and every part but the last ends on a word
    /// boundary (its rows' bytes a multiple of 4). F32 parts join the same
    /// way ([`Weights::join_f32_rows`]) when the first part is F32, and Q8_0
    /// parts plane by plane ([`Weights::join_q8_0_rows`]) when it is Q8_0.
    /// Load-time only: the copies are synchronized before the parts are
    /// freed.
    pub fn join_rows(
        &mut self,
        stream: &CudaStream,
        parts: &[&str],
        joint: String,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "Weights::join_rows";
        derived_slot_free(&self.by_name, &joint)?;
        match parts.first().and_then(|&n| self.by_name.get(n)) {
            Some(DevWeight::F32 { .. }) => return self.join_f32_rows(stream, parts, joint),
            Some(DevWeight::Q8_0 { .. }) => return self.join_q8_0_rows(stream, parts, joint),
            _ => {}
        }
        let mut shape: Option<(GgmlType, usize, usize)> = None;
        let mut rows = 0usize;
        let mut spans = Vec::with_capacity(parts.len());
        for (i, &name) in parts.iter().enumerate() {
            let Some(DevWeight::KQuant { ty, w, k }) = self.by_name.get(name) else {
                return Err(GpuError::shape(
                    WHAT,
                    format!("{name} is not resident as a K-quant"),
                ));
            };
            let row_bytes = ty
                .type_size()
                .and_then(|b| usize::try_from(b).ok())
                .map(|b| b * (k / 256))
                .ok_or_else(|| GpuError::shape(WHAT, format!("{name} is {ty}")))?;
            let want = *shape.get_or_insert((*ty, *k, w.cols()));
            if want != (*ty, *k, w.cols()) {
                return Err(GpuError::shape(
                    WHAT,
                    format!(
                        "{name} is {ty} K={k} in {} words a row; {} is {} K={} in {}",
                        w.cols(),
                        parts[0],
                        want.0,
                        want.1,
                        want.2
                    ),
                ));
            }
            let bytes = w.rows() * row_bytes;
            if i + 1 < parts.len() && !bytes.is_multiple_of(4) {
                return Err(GpuError::shape(
                    WHAT,
                    format!("{name}'s {bytes} row bytes end inside a word"),
                ));
            }
            spans.push(bytes.div_ceil(4));
            rows += w.rows();
        }
        let Some((ty, k, cols)) = shape else {
            return Err(GpuError::shape(WHAT, "no parts to join"));
        };
        let w = DeviceTensor::<u32>::zeroed(stream, rows, cols)?;
        let base = w.buf().cu_deviceptr();
        let mut at = 0usize;
        for (&name, &words) in parts.iter().zip(&spans) {
            let Some(DevWeight::KQuant { w: part, .. }) = self.by_name.get(name) else {
                return Err(GpuError::shape(WHAT, format!("{name} left the map")));
            };
            let off = u64::try_from(at * 4)
                .map_err(|_| GpuError::shape(WHAT, format!("word {at} passes u64")))?;
            // SAFETY: the words `at .. at + words` lie inside the joint's
            // `rows · cols` (each part's stream is at most its `rows · cols`),
            // and the part's first `words` inside its own buffer; both are
            // `cuMemAlloc` allocations that outlive the windows, which are
            // released below and free nothing.
            let (mut dst, src) = unsafe {
                (
                    window::<u32>(base + off, words, stream.context()),
                    window::<u32>(part.buf().cu_deviceptr(), words, stream.context()),
                )
            };
            let copied = dst.copy_from_device_async(&src, stream);
            drop(ManuallyDrop::into_inner(dst).into_raw_parts());
            drop(ManuallyDrop::into_inner(src).into_raw_parts());
            copied?;
            at += words;
        }
        stream.synchronize()?;
        for name in parts {
            self.by_name.remove(*name);
        }
        self.by_name.insert(joint, DevWeight::KQuant { ty, w, k });
        Ok(())
    }

    /// [`Weights::join_rows`] over F32 parts: every part resident as an F32
    /// weight of the first part's width `k`, their rows one after another in
    /// one `rows × k` tensor filed as the derived F32 weight `joint` (a
    /// router's expert rows with a shared expert's gate row below them). The
    /// parts leave the map and their buffers are freed. Refused unless every
    /// part is F32 of that width.
    fn join_f32_rows(
        &mut self,
        stream: &CudaStream,
        parts: &[&str],
        joint: String,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "Weights::join_rows";
        let mut width: Option<usize> = None;
        let mut rows = 0usize;
        for &name in parts {
            let Some(DevWeight::F32 { w, k }) = self.by_name.get(name) else {
                return Err(GpuError::shape(
                    WHAT,
                    format!("{name} is not resident as F32 beside F32 {}", parts[0]),
                ));
            };
            let want = *width.get_or_insert(*k);
            if *k != want || w.cols() != want {
                return Err(GpuError::shape(
                    WHAT,
                    format!("{name} is F32 K={k}; {} is K={want}", parts[0]),
                ));
            }
            rows += w.rows();
        }
        let Some(k) = width else {
            return Err(GpuError::shape(WHAT, "no parts to join"));
        };
        let w = DeviceTensor::<f32>::zeroed(stream, rows, k)?;
        let base = w.buf().cu_deviceptr();
        let mut at = 0usize;
        for &name in parts {
            let Some(DevWeight::F32 { w: part, .. }) = self.by_name.get(name) else {
                return Err(GpuError::shape(WHAT, format!("{name} left the map")));
            };
            let len = part.rows() * k;
            let off = u64::try_from(at * size_of::<f32>())
                .map_err(|_| GpuError::shape(WHAT, format!("value {at} passes u64")))?;
            // SAFETY: the values `at .. at + len` lie inside the joint's
            // `rows · k` (the parts' rows sum to `rows`), and the part's
            // `len` values are its whole buffer; both are `cuMemAlloc`
            // allocations that outlive the windows, which are released below
            // and free nothing.
            let (mut dst, src) = unsafe {
                (
                    window::<f32>(base + off, len, stream.context()),
                    window::<f32>(part.buf().cu_deviceptr(), len, stream.context()),
                )
            };
            let copied = dst.copy_from_device_async(&src, stream);
            drop(ManuallyDrop::into_inner(dst).into_raw_parts());
            drop(ManuallyDrop::into_inner(src).into_raw_parts());
            copied?;
            at += len;
        }
        stream.synchronize()?;
        for name in parts {
            self.by_name.remove(*name);
        }
        self.by_name.insert(joint, DevWeight::F32 { w, k });
        Ok(())
    }

    /// [`Weights::join_rows`] over Q8_0 parts: every part resident as a Q8_0
    /// file tensor of the first part's width `k`, their rows one after another
    /// in each of the two planes (`qs` words, `d` scales), filed as the Q8_0
    /// weight `joint` — every row the bits its own tensor holds. The parts
    /// leave the map and their buffers are freed. Refused unless every part is
    /// Q8_0 of that width.
    fn join_q8_0_rows(
        &mut self,
        stream: &CudaStream,
        parts: &[&str],
        joint: String,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "Weights::join_rows";
        let mut width: Option<usize> = None;
        let mut rows = 0usize;
        for &name in parts {
            let Some(DevWeight::Q8_0 { qs, d, k }) = self.by_name.get(name) else {
                return Err(GpuError::shape(
                    WHAT,
                    format!("{name} is not resident as Q8_0 beside Q8_0 {}", parts[0]),
                ));
            };
            let want = *width.get_or_insert(*k);
            if *k != want || qs.cols() != want / 4 || d.cols() != want / 32 || qs.rows() != d.rows()
            {
                return Err(GpuError::shape(
                    WHAT,
                    format!(
                        "{name} is Q8_0 K={k} in {}x{} words and {}x{} scales; {} is K={want}",
                        qs.rows(),
                        qs.cols(),
                        d.rows(),
                        d.cols(),
                        parts[0]
                    ),
                ));
            }
            rows += d.rows();
        }
        let Some(k) = width else {
            return Err(GpuError::shape(WHAT, "no parts to join"));
        };
        let qs = DeviceTensor::<u32>::zeroed(stream, rows, k / 4)?;
        let d = DeviceTensor::<u16>::zeroed(stream, rows, k / 32)?;
        let (qs_base, d_base) = (qs.buf().cu_deviceptr(), d.buf().cu_deviceptr());
        let mut at = 0usize;
        for &name in parts {
            let Some(DevWeight::Q8_0 {
                qs: part_qs,
                d: part_d,
                ..
            }) = self.by_name.get(name)
            else {
                return Err(GpuError::shape(WHAT, format!("{name} left the map")));
            };
            let n = part_d.rows();
            let (words, scales) = (n * (k / 4), n * (k / 32));
            let qs_off = u64::try_from(at * (k / 4) * size_of::<u32>())
                .map_err(|_| GpuError::shape(WHAT, format!("row {at} passes u64")))?;
            let d_off = u64::try_from(at * (k / 32) * size_of::<u16>())
                .map_err(|_| GpuError::shape(WHAT, format!("row {at} passes u64")))?;
            // SAFETY: rows `at .. at + n` of each plane lie inside the joint's
            // `rows` (the parts' rows sum to `rows`), and each window over the
            // part is that plane's whole buffer; all are `cuMemAlloc`
            // allocations that outlive the windows, which are released below
            // and free nothing.
            let (mut dst_qs, src_qs, mut dst_d, src_d) = unsafe {
                (
                    window::<u32>(qs_base + qs_off, words, stream.context()),
                    window::<u32>(part_qs.buf().cu_deviceptr(), words, stream.context()),
                    window::<u16>(d_base + d_off, scales, stream.context()),
                    window::<u16>(part_d.buf().cu_deviceptr(), scales, stream.context()),
                )
            };
            let copied = dst_qs
                .copy_from_device_async(&src_qs, stream)
                .and_then(|()| dst_d.copy_from_device_async(&src_d, stream));
            drop(ManuallyDrop::into_inner(dst_qs).into_raw_parts());
            drop(ManuallyDrop::into_inner(src_qs).into_raw_parts());
            drop(ManuallyDrop::into_inner(dst_d).into_raw_parts());
            drop(ManuallyDrop::into_inner(src_d).into_raw_parts());
            copied?;
            at += n;
        }
        stream.synchronize()?;
        for name in parts {
            self.by_name.remove(*name);
        }
        self.by_name.insert(joint, DevWeight::Q8_0 { qs, d, k });
        Ok(())
    }

    /// The resident weight for `name` — a file tensor name, or the `derived.`
    /// name an architecture filed a derived weight under.
    pub fn get(&self, name: &str) -> Option<&DevWeight> {
        let name = if self.tied && name == HEAD_TENSOR {
            EMBED_TENSOR
        } else {
            name
        };
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

/// The file bytes a segment's upload may release once its bytes are staged:
/// each entry the shard and the bytes within its mapping.
type PendingRelease = Vec<(usize, Range<u64>)>;

/// `load_placed`'s refusal of tensor `t`.
fn placed_refusal(t: &ModelTensor, detail: String) -> GpuError {
    GpuError::shape(PLACED, format!("tensor {}: {detail}", t.name))
}

/// [`Weights::load_rows`], releasing each uploaded segment's file pages
/// through `release` when there is one, each as soon as its bytes are
/// staged. One staging ring carries the whole load's copies and one
/// synchronize ends them.
fn load_segments(
    stream: &CudaStream,
    split: &Split,
    model: &ModelTensors,
    rows: &[Row],
    card: usize,
    mut release: Option<&mut PageDrop<'_>>,
) -> Result<Weights, GpuError> {
    stream.context().bind_to_thread()?;
    let mut by_name = BTreeMap::new();
    // The ring is made after `by_name` and drops before it: its drop drains
    // the copies, so no buffer of a refused load is freed while a copy still
    // writes it.
    let budget = rows
        .iter()
        .flat_map(|row| row.segments.iter())
        .filter(|seg| seg.device == Device::Card(card))
        .map(|seg| seg.resident_bytes)
        .sum::<u64>();
    let mut ring = UploadRing::new(stream, usize::try_from(budget).unwrap_or(usize::MAX))?;
    for row in rows {
        let t = model.tensors.get(row.tensor).ok_or_else(|| {
            GpuError::shape(PLACED, format!("placement row names tensor {}", row.tensor))
        })?;
        for seg in row
            .segments
            .iter()
            .filter(|s| s.device == Device::Card(card))
        {
            let (dw, release_ranges) =
                upload_segment(&mut ring, stream, split, model.experts, t, seg)?;
            if by_name.insert(t.name.clone(), dw).is_some() {
                return Err(placed_refusal(
                    t,
                    format!("has two segments on card {card}"),
                ));
            }
            // The ring's copies read its pinned slots, never the file: once a
            // segment's pieces are staged, its pages can go at once, which
            // keeps the load's page-cache footprint to one segment rather
            // than letting a card's whole upload evict the host set.
            if let Some(release) = release.as_mut() {
                for (shard, at) in release_ranges {
                    release
                        .release(shard, at)
                        .map_err(|e| GpuError::plan(PLACED, e))?;
                }
            }
        }
    }
    ring.finish(stream)?;
    Ok(Weights {
        by_name,
        tied: false,
    })
}

/// Upload placement segment `seg` of tensor `t`, a stack of `experts` when
/// it is one: find the tensor in `split`, confirm it is the tensor the
/// placement was made from, upload the rows the segment holds (their file
/// bytes as one borrowed slice per span, so a whole-tensor segment uploads
/// from the mapping itself and an expert stack streams its spans without a
/// gathered copy), and hand back the file bytes the caller may release now
/// that they are staged — empty for a tensor a step reads from the file.
fn upload_segment(
    ring: &mut UploadRing,
    stream: &CudaStream,
    split: &Split,
    experts: u64,
    t: &ModelTensor,
    seg: &Segment,
) -> Result<(DevWeight, PendingRelease), GpuError> {
    let Format::Card(format) = seg.format else {
        return Err(placed_refusal(t, format!("is on a card as {}", seg.format)));
    };
    let spans = seg
        .spans(t, experts)
        .map_err(|e| GpuError::plan(PLACED, e))?;
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
    let held: u64 = spans.iter().map(|s| s.rows.end - s.rows.start).sum();
    let rows =
        usize::try_from(held).map_err(|_| placed_refusal(t, format!("{held} rows pass usize")))?;
    let data = g.data(info)?;
    let mut src = Vec::with_capacity(spans.len());
    for span in &spans {
        let range = (|| {
            Some(usize::try_from(span.bytes.start).ok()?..usize::try_from(span.bytes.end).ok()?)
        })()
        .and_then(|r| data.get(r));
        let slice = range.ok_or_else(|| {
            placed_refusal(
                t,
                format!("spans {spans:?} run past its {} file bytes", info.nbytes),
            )
        })?;
        src.push(slice);
    }
    let dw = upload_rows(stream, ring, PLACED, format, info, rows, &src)?;
    if u64::try_from(dw.resident_bytes()).ok() != Some(seg.resident_bytes) {
        // The upload's copies may still be running into the buffers this
        // error is about to drop; drain before they do.
        let _ = stream.synchronize();
        let detail = format!(
            "uploaded {} device bytes, the plan has {}",
            dw.resident_bytes(),
            seg.resident_bytes
        );
        return Err(placed_refusal(t, detail));
    }
    let read_by_steps = matches!(t.role, Role::TokenEmbedding | Role::EngramTable);
    let release = if read_by_steps {
        PendingRelease::new()
    } else {
        let base = g.data_base() + info.offset;
        spans
            .iter()
            .map(|span| (s, base + span.bytes.start..base + span.bytes.end))
            .collect()
    };
    Ok((dw, release))
}

/// Pack and upload one file tensor in its kernel's device format, on a
/// staging ring of its own (a caller uploading many tensors makes one ring
/// for them all through [`Weights::load_where`]).
pub fn upload_file_tensor(
    stream: &CudaStream,
    gguf: &Gguf,
    t: &TensorInfo,
) -> Result<DevWeight, GpuError> {
    let mut ring = UploadRing::new(stream, upload_budget(t))?;
    let dw = upload_file_tensor_with(&mut ring, stream, gguf, t)?;
    ring.finish(stream)?;
    Ok(dw)
}

/// [`upload_file_tensor`] on a caller's ring, which the caller synchronizes.
fn upload_file_tensor_with(
    ring: &mut UploadRing,
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
        ring,
        "Weights::load",
        format,
        t,
        tensor_rows(t),
        &[gguf.data(t)?],
    )
}

/// The bytes `t`'s upload moves, for sizing a load's staging ring: its
/// resident size where its type has a device layout, its file bytes
/// otherwise — a sizing hint, never a contract (the uploads check their own
/// lengths).
fn upload_budget(t: &TensorInfo) -> usize {
    usize::try_from(t.dims[0])
        .ok()
        .and_then(|k| resident_size(t.ty, k, tensor_rows(t)))
        .or_else(|| usize::try_from(t.nbytes).ok())
        .unwrap_or(usize::MAX)
}

/// Pack `rows` rows of file tensor `t` — all of them, or an expert stack's
/// listed experts — from `src` (those rows' file bytes as one borrowed slice
/// per span, in span order) in card format `format`, and upload them through
/// `ring`. The formats whose device bytes are the file bytes verbatim
/// ([`CardFormat::KQuant`], [`CardFormat::F32`] — the host is little-endian,
/// the module refuses to build otherwise) stage the slices themselves into a
/// zeroed buffer, so no word is packed on the host; every other format packs
/// its run on the host and stages the packed words. Refuses exactly where
/// [`CardFormat::resident_bytes`] has no size for the rows, and checks the
/// upload's device bytes against that size; `what` names the caller in both
/// errors.
fn upload_rows(
    stream: &CudaStream,
    ring: &mut UploadRing,
    what: &'static str,
    format: CardFormat,
    t: &TensorInfo,
    rows: usize,
    src: &[&[u8]],
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
    let held: usize = src.iter().map(|s| s.len()).sum();
    let Some(take) = usize::try_from(need).ok().filter(|&n| held >= n) else {
        let detail = format!("tensor {name} holds {held} bytes, {rows} rows need {need}");
        return Err(GpuError::shape(what, detail));
    };
    let dw = match format {
        // The gates' packing verbatim (gate_p1/gate_p9): the whole row span as
        // one word stream, so the kernels' byte-offset row addressing sees
        // contiguous rows. A routed Q5_1 stack's span is its experts' 24-byte
        // blocks in slot order, the layout `q5_1_gemv_sel` reads. The words
        // are the file's bytes themselves on this host, so the rows stream
        // into the zeroed buffer and only the layout's zero tail is packed —
        // by the buffer's memset.
        CardFormat::KQuant => {
            let words = stream_words(need, rows as u64)
                .and_then(|w| usize::try_from(w).ok())
                .ok_or_else(|| {
                    GpuError::shape(what, format!("tensor {name}: its word stream passes usize"))
                })?;
            let w = DeviceTensor::<u32>::zeroed(stream, rows, words / rows)?;
            let stage = Stage {
                what,
                name,
                dst: w.buf().cu_deviceptr(),
                dst_bytes: w.buf().num_bytes(),
            };
            ring.copy_bytes(stream, &stage, src, take)?;
            DevWeight::KQuant { ty: t.ty, w, k }
        }
        CardFormat::F32 => {
            // The file's f32 bytes are the device's f32 bytes on this host;
            // a run that is not the rows' whole values is a broken file, the
            // layout's size and the run's need disagreeing.
            if take != rows * k * 4 {
                let detail = format!(
                    "tensor {name}: {take} bytes are not its {rows} rows of {k} f32 values"
                );
                return Err(GpuError::shape(what, detail));
            }
            let w = DeviceTensor::<f32>::zeroed(stream, rows, k)?;
            let stage = Stage {
                what,
                name,
                dst: w.buf().cu_deviceptr(),
                dst_bytes: w.buf().num_bytes(),
            };
            ring.copy_bytes(stream, &stage, src, take)?;
            DevWeight::F32 { w, k }
        }
        CardFormat::Q5_0 => {
            let packed = pack_q5_0(&flat_of(src, take)?, k, rows)?;
            let cols = packed.len() / rows;
            DevWeight::Q5_0 {
                w: upload_packed(stream, ring, what, name, &packed, rows, cols)?,
                k,
            }
        }
        CardFormat::Q5_1 => {
            let packed = pack_q5_1(&flat_of(src, take)?, k, rows)?;
            let cols = packed.len() / rows;
            DevWeight::Q5_1 {
                w: upload_packed(stream, ring, what, name, &packed, rows, cols)?,
                k,
            }
        }
        CardFormat::Q8_0Planes => {
            let blocks: Vec<Q8Block> = flat_of(src, take)?
                .as_chunks::<34>()
                .0
                .iter()
                .map(Q8Block::from_bytes)
                .collect();
            let (qs, d) = q8_0_planes(&blocks);
            DevWeight::Q8_0 {
                qs: upload_packed(stream, ring, what, name, &qs, rows, k / 4)?,
                d: upload_packed(stream, ring, what, name, &d, rows, k / 32)?,
                k,
            }
        }
        CardFormat::Bf16Raw => {
            let detail = format!(
                "tensor {name}: {format:?} has no resident weight here; the reader that picks it \
                 uploads its words itself"
            );
            return Err(GpuError::shape(what, detail));
        }
        CardFormat::Bf16AsF32 => {
            let mut vals = vec![0.0f32; rows * k];
            dequant_row(t.ty, &flat_of(src, take)?, &mut vals)
                .map_err(::model::ModelError::from)?;
            DevWeight::F32 {
                w: upload_packed(stream, ring, what, name, &vals, rows, k)?,
                k,
            }
        }
    };
    if u64::try_from(dw.resident_bytes()).ok() != Some(size) {
        // The upload's copies may still be running into the buffers this
        // error is about to drop; drain before they do.
        let _ = stream.synchronize();
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

/// The first `take` bytes of `src`'s slices as one run, for a format that
/// packs on the host: the slice itself when there is one, a copy of the
/// spans' bytes in order when there are several.
fn flat_of<'a>(src: &[&'a [u8]], take: usize) -> Result<Cow<'a, [u8]>, GpuError> {
    match src {
        [one] => Ok(Cow::Borrowed(&one[..take])),
        _ => {
            let mut out = Vec::with_capacity(take);
            for slice in src {
                let len = slice.len().min(take - out.len());
                out.extend_from_slice(&slice[..len]);
                if out.len() == take {
                    break;
                }
            }
            Ok(Cow::Owned(out))
        }
    }
}

/// `data` — exactly `rows * cols` elements, a format's finished packing — as
/// a device tensor staged through `ring`, the buffer zeroed first so a short
/// packing would leave a zero tail rather than stale bytes (the packed
/// formats always fill theirs).
fn upload_packed<T: DeviceCopy>(
    stream: &CudaStream,
    ring: &mut UploadRing,
    what: &'static str,
    name: &str,
    data: &[T],
    rows: usize,
    cols: usize,
) -> Result<DeviceTensor<T>, GpuError> {
    if data.len() != rows * cols {
        return Err(GpuError::shape(
            what,
            format!(
                "tensor {name}: {} elements for {rows} rows of {cols}",
                data.len()
            ),
        ));
    }
    let t = DeviceTensor::zeroed(stream, rows, cols)?;
    let bytes = bytes_of(data);
    let stage = Stage {
        what,
        name,
        dst: t.buf().cu_deviceptr(),
        dst_bytes: t.buf().num_bytes(),
    };
    ring.copy_bytes(stream, &stage, &[bytes], bytes.len())?;
    Ok(t)
}

/// The q8f32 two-plane layout of Q8_0 blocks: per block 8 code words (code j
/// in word j/4, byte j%4) and the block's f16 scale bits unchanged. The
/// kernels widen a scale with the hardware convert, and widening f16 to f32
/// is exact, so they multiply by the same f32 the reference dequantizes with.
pub fn q8_0_planes(blocks: &[Q8Block]) -> (Vec<u32>, Vec<u16>) {
    let mut qs = Vec::with_capacity(blocks.len() * 8);
    let mut d = Vec::with_capacity(blocks.len());
    for b in blocks {
        let mut w = [0u32; 8];
        for (j, &q) in b.q.iter().enumerate() {
            w[j / 4] |= u32::from(q as u8) << (8 * (j % 4));
        }
        qs.extend_from_slice(&w);
        d.push(b.d);
    }
    (qs, d)
}

#[cfg(test)]
mod tests {
    use super::{HEAD_TENSOR, derived_slot_free, head_tensor};
    use gguf::Split;
    use gguf::write::{Layout, TensorDecl, Writer};
    use std::collections::BTreeMap;

    /// A GGUF of F32 tensors of two rows of four named `names`, at `path`.
    fn write_file(path: &std::path::Path, names: &[&str]) {
        let decls: Vec<TensorDecl> = names
            .iter()
            .map(|n| TensorDecl {
                name: (*n).to_string(),
                dims: vec![4, 2],
                type_id: 0,
                nbytes: 32,
            })
            .collect();
        let layout = Layout::new(&[], decls).unwrap_or_else(|e| panic!("{e}"));
        let file =
            std::io::BufWriter::new(std::fs::File::create(path).unwrap_or_else(|e| panic!("{e}")));
        let mut w = Writer::new(file, layout).unwrap_or_else(|e| panic!("{e}"));
        for n in names {
            w.tensor(n, &[0u8; 32]).unwrap_or_else(|e| panic!("{e}"));
        }
        w.finish().unwrap_or_else(|e| panic!("{e}"));
    }

    /// The head is `output.weight` when the file has it, the token embedding
    /// when it has no output (tied), and `output.weight` again when it has
    /// neither, which every reader refuses by name as absent.
    #[test]
    fn the_head_is_the_output_else_the_tied_embedding() {
        let dir = std::env::temp_dir().join(format!("bloomery-head-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("{e}"));
        let cases = [
            (&["token_embd.weight", "output.weight"][..], HEAD_TENSOR),
            (&["token_embd.weight"][..], "token_embd.weight"),
            (&["output.weight"][..], HEAD_TENSOR),
            (&["output_norm.weight"][..], HEAD_TENSOR),
        ];
        for (i, (names, want)) in cases.iter().enumerate() {
            let path = dir.join(format!("{i}.gguf"));
            write_file(&path, names);
            let split = Split::open(&path).unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(head_tensor(&split), *want, "{names:?}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

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
