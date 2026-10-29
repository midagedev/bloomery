//! Resident weights (package P10): every tensor of the model file uploaded
//! once, in the device format its kernel consumes (docs/gpu-design.md
//! decision 4 — format conversion is load-time work; per step a site finds
//! what was uploaded, by name, and launches the kernel its type reads:
//! `crate::site`).
//! A [`Weights`] owns one contiguous block range plus,
//! optionally, the non-block tensors, so a `Stage` loads exactly its own
//! layers — or, for a placed model, every segment its plan puts on one card
//! ([`Weights::load_placed`]). The formats and their byte arithmetic are
//! `model::placement::CardFormat`'s. The gates (`gate_p10`, `gate_load_v41`)
//! pin the uploads against independent host packings bit for bit.

use crate::GpuError;
use crate::mxfp4::{Nvfp4, Nvfp4Weight, Planes, repack};
use crate::mxfp4_sel::Nvfp4Stack;
use crate::q5::{pack_q5_0, pack_q5_1};
use crate::q8f32::{E4m3Scales, E4m3Weight, e4m3_planes};
use crate::tensor::{DeviceTensor, window};
use crate::upload::{Stage, UploadRing, bytes_of};
use ::model::placement::host_lock::PageDrop;
use ::model::placement::{
    CardFormat, Device, Format, ModelTensor, ModelTensors, Plan, Role, Row, Segment, Span, tp,
};
use cuda_core::{CudaStream, DeviceBuffer, DeviceCopy};
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
pub use ::model::placement::HEAD_TENSOR;
const EMBED_TENSOR: &str = "token_embd.weight";

fn file_format(t: &TensorInfo) -> Option<CardFormat> {
    ::model::placement::file_card_format(&t.name, t.ty, *t.dims.first()?)
}

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
    /// Q3_K/Q4_K/Q5_K/Q6_K tensor, or a routed Q5_1, Q8_0, IQ2_XS, IQ2_S, IQ1_S,
    /// IQ1_M, IQ3_XXS, IQ4_XS, IQ4_NL or MXFP4 stack a program's plan
    /// puts on the card in the file's `block_q5_1`s, `block_q8_0`s,
    /// `block_iq2_xs`es, `block_iq2_s`es, `block_iq1_s`es, `block_iq1_m`s,
    /// `block_iq3_xxs`es, `block_iq4_xs`es, `block_iq4_nl`s or
    /// `block_mxfp4`s,
    /// which its `_sel` and GEMM entries read (`ty` says which). The raw row
    /// stream as little-endian u32 words, zero-padded at its end to a whole
    /// number of words per row —
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
    /// Q5_0 file tensor in the `pack_q5_0` layout: `q5_0_row_words(k/32) =
    /// 5·k/32 + ⌈k/64⌉` words per row, the file's 22 bytes a block regrouped
    /// into planes. `k` is a multiple of 32.
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
    /// NVFP4 file tensor in the two planes of `mxfp4::repack::<Nvfp4>`
    /// (`CardFormat::Nvfp4Planes`): `qs` rows × k/8 code words, `e` rows ×
    /// k/64 scale words (a block's four UE4M3 bytes), and the tensor's scale,
    /// the f32 of its `<stem>.scale` tensor, read from the file at load: on
    /// the host for the gemv's argument and in `scales` (one value) for the
    /// GEMM's store. `k` is a multiple of 64.
    Nvfp4 {
        qs: DeviceTensor<u32>,
        e: DeviceTensor<u32>,
        k: usize,
        scale: f32,
        scales: DeviceTensor<f32>,
    },
    /// bloomery's MXFP8 file tensor (`GgmlType::MXFP8`) in the two planes of
    /// `q8f32::e4m3_planes` (`CardFormat::Mxfp8Planes`): `qs` rows × k/4
    /// value words, `e` rows × k/128 scale words (four E8M0 bytes a word).
    /// `k` is a multiple of 128.
    Mxfp8 {
        qs: DeviceTensor<u32>,
        e: DeviceTensor<u32>,
        k: usize,
    },
    /// bloomery's FP8_B128 file tensor (`GgmlType::FP8_B128`) in the same
    /// two planes (`CardFormat::Fp8B128Planes`), `e` one f32 scale a word
    /// (a 128-value block's). `k` is a multiple of 128.
    Fp8B128 {
        qs: DeviceTensor<u32>,
        e: DeviceTensor<u32>,
        k: usize,
    },
    /// A BF16 file tensor as the file stores it (`CardFormat::Bf16Raw`):
    /// rows × k/2 u32 words, value `2j` of a row in the low half of its word
    /// `j`, the high half value `2j + 1`; the kernels widen a value to f32 as
    /// they read it (`crate::q8f32::Bf16Words`). `k` is even.
    Bf16 { w: DeviceTensor<u32>, k: usize },
    /// An NVFP4 routed stack (three dims) in the same two planes over the
    /// experts it holds, and their tensor scales: one f32 a row of `scales`,
    /// in the order the planes hold the experts, from the stack's
    /// `<stem>.scale` of one value an expert.
    Nvfp4Stack {
        qs: DeviceTensor<u32>,
        e: DeviceTensor<u32>,
        scales: DeviceTensor<f32>,
        k: usize,
    },
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
            DevWeight::Bf16 { w, .. } => w.rows(),
            DevWeight::Nvfp4 { qs, .. }
            | DevWeight::Nvfp4Stack { qs, .. }
            | DevWeight::Mxfp8 { qs, .. }
            | DevWeight::Fp8B128 { qs, .. } => qs.rows(),
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
            | DevWeight::F32 { k, .. }
            | DevWeight::Bf16 { k, .. }
            | DevWeight::Nvfp4 { k, .. }
            | DevWeight::Nvfp4Stack { k, .. }
            | DevWeight::Mxfp8 { k, .. }
            | DevWeight::Fp8B128 { k, .. } => *k,
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
            DevWeight::Bf16 { w, .. } => vec![held(w)],
            DevWeight::Nvfp4 { qs, e, scales, .. }
            | DevWeight::Nvfp4Stack { qs, e, scales, .. } => {
                vec![held(qs), held(e), held(scales)]
            }
            DevWeight::Mxfp8 { qs, e, .. } | DevWeight::Fp8B128 { qs, e, .. } => {
                vec![held(qs), held(e)]
            }
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

/// One tensor, or one part of a tensor, to upload ([`Weights::add_parts`]): its header entry as
/// the part's (name, type, dims and bytes; the offset is not read), its rows' bytes, one slice a
/// run of rows, in row order, and, for a type that reads them ([`GgmlType::reads_scale`]), the
/// tensor scales its values are multiplied by: one, or one an expert of a stack, in expert order.
pub struct TensorPart<'a> {
    pub info: TensorInfo,
    pub spans: Vec<&'a [u8]>,
    pub scale: Option<Vec<f32>>,
}

/// What [`Weights::add_parts`]' errors name.
const PARTS: &str = "Weights::add_parts";

/// The card format and staging bytes of a part with header entry `t`, from the entry alone: a
/// name no slot takes, a type with no card format, and a size its type and dims do not give are
/// refused by name. NVFP4 is the one type with a card format whose values a tensor scale
/// multiplies; its part carries the scale ([`TensorPart::scale`]). A load checks every part it plans here before it reads any
/// ([`Weights::add_parts`] checks each again with its bytes).
pub fn part_format(t: &TensorInfo) -> Result<(CardFormat, usize), GpuError> {
    file_slot(&t.name)?;
    let format = file_format(t).ok_or_else(|| {
        GpuError::shape(
            PARTS,
            format!(
                "tensor {} has type {}, which no part upload takes",
                t.name, t.ty
            ),
        )
    })?;
    tp::check_bytes(t).map_err(|e| GpuError::shape(PARTS, e.to_string()))?;
    let bytes = usize::try_from(t.dims[0])
        .ok()
        .and_then(|k| format.resident_bytes(t.ty, k as u64, tensor_rows(t) as u64))
        .and_then(|bytes| usize::try_from(bytes).ok())
        .ok_or_else(|| {
            GpuError::shape(
                PARTS,
                format!("tensor {}: resident bytes pass usize", t.name),
            )
        })?;
    Ok((format, bytes))
}

/// Each part's format, shape, scales, spans and staging bytes, checked before anything moves; a
/// name `held` already holds is refused.
fn part_formats(
    parts: &[TensorPart<'_>],
    held: impl Fn(&str) -> bool,
) -> Result<(Vec<CardFormat>, usize), GpuError> {
    let mut seen = BTreeSet::new();
    let mut budget = 0usize;
    let mut formats = Vec::with_capacity(parts.len());
    for p in parts {
        let t = &p.info;
        if !seen.insert(t.name.as_str()) || held(&t.name) {
            return Err(GpuError::shape(PARTS, format!("{} given twice", t.name)));
        }
        let (format, bytes) = part_format(t)?;
        formats.push(format);
        let want = match t.dims.as_slice() {
            [_, _, experts] => *experts,
            _ => 1,
        };
        match (t.ty.reads_scale(), &p.scale) {
            (false, None) => {}
            (true, Some(s))
                if u64::try_from(s.len()).ok() == Some(want) && s.iter().all(|v| v.is_finite()) => {
            }
            (reads, scale) => {
                let carries = scale.as_ref().map_or("none".to_string(), |s| {
                    let finite = s.iter().filter(|v| v.is_finite()).count();
                    format!("{} values, {finite} finite", s.len())
                });
                return Err(GpuError::shape(
                    PARTS,
                    format!(
                        "tensor {}: {} {:?} reads {} finite tensor scales; the part carries {carries}",
                        t.name,
                        t.ty,
                        t.dims,
                        if reads { want } else { 0 }
                    ),
                ));
            }
        }
        let held = p
            .spans
            .iter()
            .try_fold(0usize, |sum, s| sum.checked_add(s.len()));
        if held.and_then(|v| u64::try_from(v).ok()) != Some(t.nbytes) {
            return Err(GpuError::shape(
                PARTS,
                format!(
                    "tensor {}: spans hold {held:?} bytes, want {}",
                    t.name, t.nbytes
                ),
            ));
        }
        budget = budget
            .checked_add(bytes)
            .ok_or_else(|| GpuError::shape(PARTS, "the parts' staging bytes pass usize"))?;
    }
    Ok((formats, budget))
}

/// Upload `parts` through `ring`, whose copies the caller ends with one synchronize, each part
/// in the card format `card` names for it, else its type's ([`part_format`]): every part checked
/// first as [`Weights::add_parts`] checks them, a name `held` holds refused. The uploaded weights
/// by name, for the caller to file once its copies end.
pub(crate) fn stage_parts(
    stream: &CudaStream,
    ring: &mut UploadRing,
    parts: &[TensorPart<'_>],
    held: impl Fn(&str) -> bool,
    card: impl Fn(&TensorInfo) -> Option<CardFormat>,
) -> Result<Vec<(String, DevWeight)>, GpuError> {
    let (formats, _) = part_formats(parts, held)?;
    let mut added = Vec::with_capacity(parts.len());
    for (p, format) in parts.iter().zip(formats) {
        let t = &p.info;
        let dw = upload_rows(
            stream,
            ring,
            PARTS,
            card(t).unwrap_or(format),
            t,
            tensor_rows(t),
            &p.spans,
            p.scale.clone(),
        )?;
        added.push((t.name.clone(), dw));
    }
    Ok(added)
}

/// The resident weights of a layer range: every `blk.L.*` tensor with L in
/// `layers`, what the architecture derives for those blocks, and — when
/// loaded with globals — the non-block tensors. Load-time only; per-step
/// code reads through `get` and never allocates. The default holds none, for a load that adds
/// its parts a group at a time ([`Weights::add_parts`]).
#[derive(Default)]
pub struct Weights {
    by_name: BTreeMap<String, DevWeight>,
    /// The file ties its head to the token embedding ([`head_tensor`]): `output.weight` is the
    /// resident `token_embd.weight`.
    tied: bool,
}

/// A weight's one plane and width in a format [`Weights::join_rows`] joins
/// by rows, `None` for another format.
type PlaneOf<T> = for<'a> fn(&'a DevWeight) -> Option<(&'a DeviceTensor<T>, usize)>;

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
        Weights::load_as(stream, split, layers, globals, &|_| None)
    }

    /// [`Weights::load`] with each tensor in the card format `card` gives it, and in its type's
    /// own where `card` gives none: a model's description
    /// ([`ModelTensors::card_format`]) picks a format no file type loads in by itself, as BF16
    /// words a kernel widens as it reads them.
    pub(crate) fn load_as(
        stream: &CudaStream,
        split: &Split,
        layers: Range<usize>,
        globals: bool,
        card: &dyn Fn(&TensorInfo) -> Option<CardFormat>,
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
        Weights::load_named(stream, split, |name| keep.contains(name), tied, card)
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
        Weights::load_named(stream, split, keep, false, &|_| None)
    }

    /// Upload `parts` beside the weights held, each a tensor or a part of one that a caller cut
    /// from its source, in the device format its kernel consumes, as [`Weights::load_where`]
    /// uploads a whole tensor: [`TensorPart::info`] states the part's name, type and dims,
    /// [`TensorPart::spans`] holds its rows' bytes in row order and [`TensorPart::scale`] its
    /// tensor scales. A name given twice or held already, one a derived weight would take, a type
    /// with no device format, scales a type reads missing, of another count or not finite, scales
    /// a type does not read, a shape unlike its byte count and spans unlike the part's bytes are
    /// refused by name before any of `parts` moves. One staging ring carries their copies, and one
    /// synchronize ends them.
    pub fn add_parts(
        &mut self,
        stream: &CudaStream,
        parts: &[TensorPart<'_>],
    ) -> Result<(), GpuError> {
        let held = |name: &str| self.by_name.contains_key(name);
        let (_, budget) = part_formats(parts, held)?;
        stream.context().bind_to_thread()?;
        let mut ring = UploadRing::new(stream, budget)?;
        let added = stage_parts(stream, &mut ring, parts, held, |_| None)?;
        ring.finish(stream)?;
        self.by_name.extend(added);
        Ok(())
    }

    /// The weights a load staged ([`stage_parts`]), once its ring's copies have ended.
    pub(crate) fn of_staged(by_name: BTreeMap<String, DevWeight>) -> Weights {
        Weights {
            by_name,
            tied: false,
        }
    }

    /// [`Weights::load_where`], and with `tied` the token embedding is also the head
    /// ([`Weights::get`]); each tensor in the format `card` gives it ([`Weights::load_as`]).
    fn load_named(
        stream: &CudaStream,
        split: &Split,
        mut keep: impl FnMut(&str) -> bool,
        tied: bool,
        card: &dyn Fn(&TensorInfo) -> Option<CardFormat>,
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
            file_slot(&t.name)?;
            let gguf = split.shard(shard).ok_or_else(|| {
                GpuError::shape(
                    "Weights::load_where",
                    format!("{} names shard {shard}, which the split lacks", t.name),
                )
            })?;
            by_name.insert(
                t.name.clone(),
                upload_file_tensor_with(&mut ring, stream, gguf, t, card(t), |n| {
                    let (s, info) = split.find(n)?;
                    Some((split.shard(s)?, info))
                })?,
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

    /// Whether every part is resident in one format at one width `k` — the
    /// parts [`Weights::join_rows`] joins; a missing part is not joinable.
    #[must_use]
    pub fn joinable(&self, parts: &[&str]) -> bool {
        let key = |name: &str| {
            self.by_name.get(name).map(|w| {
                let ty = match w {
                    DevWeight::KQuant { ty, .. } => Some(*ty),
                    _ => None,
                };
                (std::mem::discriminant(w), ty, w.k())
            })
        };
        let first = parts.first().and_then(|n| key(n));
        first.is_some() && parts.iter().all(|n| key(n) == first)
    }

    /// Move the K-quant weights `parts` — one type, one row width — into one
    /// row stream filed as the derived weight `joint`: part `i`'s rows follow
    /// part `i − 1`'s, so one gemv over `joint` computes every part's rows,
    /// each row the bits its own tensor gives (the kernels address a row by
    /// its byte offset). The parts leave the map and their buffers are freed,
    /// so nothing is resident twice and the resident bytes are unchanged.
    /// Refused unless every part is resident as a K-quant of the first
    /// part's type and width, and every part but the last ends on a word
    /// boundary (its rows' bytes a multiple of 4). F32 and BF16 parts join
    /// the same way ([`Weights::join_plane_rows`]) when the first part is one
    /// of those, and Q8_0, MXFP8 or FP8_B128 parts plane by plane
    /// ([`Weights::join_byte_rows`]) when it is one of those.
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
            Some(DevWeight::F32 { .. }) => {
                return self.join_plane_rows(
                    stream,
                    (parts, joint),
                    1,
                    |w| match w {
                        DevWeight::F32 { w, k } => Some((w, *k)),
                        _ => None,
                    },
                    |w, k| DevWeight::F32 { w, k },
                );
            }
            Some(DevWeight::Bf16 { .. }) => {
                return self.join_plane_rows(
                    stream,
                    (parts, joint),
                    2,
                    |w| match w {
                        DevWeight::Bf16 { w, k } => Some((w, *k)),
                        _ => None,
                    },
                    |w, k| DevWeight::Bf16 { w, k },
                );
            }
            Some(DevWeight::Q8_0 { .. }) => {
                return self.join_byte_rows(
                    stream,
                    (parts, joint),
                    32,
                    |w| match w {
                        DevWeight::Q8_0 { qs, d, k } => Some((qs, d, *k)),
                        _ => None,
                    },
                    |qs, d, k| DevWeight::Q8_0 { qs, d, k },
                );
            }
            Some(DevWeight::Mxfp8 { .. }) => {
                return self.join_byte_rows(
                    stream,
                    (parts, joint),
                    128,
                    |w| match w {
                        DevWeight::Mxfp8 { qs, e, k } => Some((qs, e, *k)),
                        _ => None,
                    },
                    |qs, e, k| DevWeight::Mxfp8 { qs, e, k },
                );
            }
            Some(DevWeight::Fp8B128 { .. }) => {
                return self.join_byte_rows(
                    stream,
                    (parts, joint),
                    128,
                    |w| match w {
                        DevWeight::Fp8B128 { qs, e, k } => Some((qs, e, *k)),
                        _ => None,
                    },
                    |qs, e, k| DevWeight::Fp8B128 { qs, e, k },
                );
            }
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

    /// [`Weights::join_rows`] over parts in one plane of `per` values an
    /// element (F32 one, BF16's words two): every part resident in the first
    /// part's format at its width `k`, their rows one after another in one
    /// `rows × k / per` tensor filed as `joint` in that format (a router's
    /// expert rows with a shared expert's gate row below them). `plane` reads
    /// a part's plane and width, `None` for another format; `wrap` files the
    /// joined plane. The parts leave the map and their buffers are freed.
    /// Refused unless every part is of that format and width.
    fn join_plane_rows<T: DeviceCopy>(
        &mut self,
        stream: &CudaStream,
        (parts, joint): (&[&str], String),
        per: usize,
        plane: PlaneOf<T>,
        wrap: fn(DeviceTensor<T>, usize) -> DevWeight,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "Weights::join_rows";
        let mut width: Option<usize> = None;
        let mut rows = 0usize;
        let mut found = Vec::with_capacity(parts.len());
        for &name in parts {
            let Some((w, k)) = self.by_name.get(name).and_then(plane) else {
                return Err(GpuError::shape(
                    WHAT,
                    format!("{name} is not resident in the format of {}", parts[0]),
                ));
            };
            let want = *width.get_or_insert(k);
            if k != want || w.cols() * per != want {
                return Err(GpuError::shape(
                    WHAT,
                    format!("{name} is K={k}; {} is K={want}", parts[0]),
                ));
            }
            rows += w.rows();
            found.push(w);
        }
        let Some(k) = width else {
            return Err(GpuError::shape(WHAT, "no parts to join"));
        };
        let cols = k / per;
        let w = DeviceTensor::<T>::zeroed(stream, rows, cols)?;
        let base = w.buf().cu_deviceptr();
        let mut at = 0usize;
        for part in found {
            let len = part.rows() * cols;
            let off = u64::try_from(at * size_of::<T>())
                .map_err(|_| GpuError::shape(WHAT, format!("element {at} passes u64")))?;
            // SAFETY: the elements `at .. at + len` lie inside the joint's
            // `rows · cols` (the parts' rows sum to `rows`), and the part's
            // `len` elements are its whole buffer; both are `cuMemAlloc`
            // allocations that outlive the windows, which are released below
            // and free nothing.
            let (mut dst, src) = unsafe {
                (
                    window::<T>(base + off, len, stream.context()),
                    window::<T>(part.buf().cu_deviceptr(), len, stream.context()),
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
        self.by_name.insert(joint, wrap(w, k));
        Ok(())
    }

    /// [`Weights::join_rows`] over parts in two planes, one byte a value
    /// (Q8_0, MXFP8 or FP8_B128): every part resident in the first part's format at its
    /// width `k`, a `scale_values`-value scale an entry of the scale plane,
    /// their rows one after another in each of the two planes (`qs` words,
    /// the scales), filed as `joint` in that format — every row the bits its
    /// own tensor holds. `planes` reads a part's planes and width, `None` for
    /// another format; `wrap` files the joined planes. The parts leave the map
    /// and their buffers are freed. Refused unless every part is of that
    /// format and width.
    fn join_byte_rows<S: DeviceCopy>(
        &mut self,
        stream: &CudaStream,
        (parts, joint): (&[&str], String),
        scale_values: usize,
        planes: PartPlanes<S>,
        wrap: fn(DeviceTensor<u32>, DeviceTensor<S>, usize) -> DevWeight,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "Weights::join_rows";
        let mut width: Option<usize> = None;
        let mut rows = 0usize;
        let mut found = Vec::with_capacity(parts.len());
        for &name in parts {
            let Some((qs, d, k)) = self.by_name.get(name).and_then(planes) else {
                return Err(GpuError::shape(
                    WHAT,
                    format!("{name} is not resident in the format of {}", parts[0]),
                ));
            };
            let want = *width.get_or_insert(k);
            if k != want
                || qs.cols() != want / 4
                || d.cols() != want / scale_values
                || qs.rows() != d.rows()
            {
                return Err(GpuError::shape(
                    WHAT,
                    format!(
                        "{name} is K={k} in {}x{} words and {}x{} scales; {} is K={want}",
                        qs.rows(),
                        qs.cols(),
                        d.rows(),
                        d.cols(),
                        parts[0]
                    ),
                ));
            }
            rows += d.rows();
            found.push((qs, d));
        }
        let Some(k) = width else {
            return Err(GpuError::shape(WHAT, "no parts to join"));
        };
        let qs = DeviceTensor::<u32>::zeroed(stream, rows, k / 4)?;
        let d = DeviceTensor::<S>::zeroed(stream, rows, k / scale_values)?;
        let (qs_base, d_base) = (qs.buf().cu_deviceptr(), d.buf().cu_deviceptr());
        let mut at = 0usize;
        for (part_qs, part_d) in found {
            let n = part_d.rows();
            let (words, scales) = (n * (k / 4), n * (k / scale_values));
            let qs_off = u64::try_from(at * (k / 4) * size_of::<u32>())
                .map_err(|_| GpuError::shape(WHAT, format!("row {at} passes u64")))?;
            let d_off = u64::try_from(at * (k / scale_values) * size_of::<S>())
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
                    window::<S>(d_base + d_off, scales, stream.context()),
                    window::<S>(part_d.buf().cu_deviceptr(), scales, stream.context()),
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
        self.by_name.insert(joint, wrap(qs, d, k));
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

    /// The resident set's own copy of the name `name` holds a weight under,
    /// which a view of the weight (a [`crate::card_leg::LegStacks`]) keeps
    /// for its refusals; `None` where [`Weights::get`] finds nothing by
    /// exactly that name.
    pub fn name_of(&self, name: &str) -> Option<&str> {
        self.by_name.get_key_value(name).map(|(k, _)| k.as_str())
    }

    /// Move the weight `name` out of the resident set: the caller owns its
    /// buffers and [`Weights::get`] no longer finds it. `None` when no entry
    /// has exactly that name (a tied head is the embedding's entry, not its
    /// own); the caller names the error.
    pub fn take(&mut self, name: &str) -> Option<DevWeight> {
        self.by_name.remove(name)
    }

    /// Re-upload the resident file tensor `name` of `split` in `format`, a
    /// layout its type has ([`CardFormat::holds`]): for a body loaded by
    /// [`Weights::load`] whose routed card entry reads a stack in another
    /// layout than [`CardFormat::of`] names (a Q8_0 down's file blocks as
    /// words). The resident form is freed before the upload. A name not
    /// resident or not in the file, a format the type has not and a type whose
    /// values a tensor scale multiplies are refused by name. Load-time only.
    pub fn relayout(
        &mut self,
        stream: &CudaStream,
        split: &Split,
        name: &str,
        format: CardFormat,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "Weights::relayout";
        let (shard, t) = split
            .find(name)
            .ok_or_else(|| GpuError::tensor(WHAT, name, "a tensor of the file"))?;
        if !format.holds(t.ty) || t.ty.reads_scale() {
            let detail = format!("tensor {name} of {} has no {format:?} layout here", t.ty);
            return Err(GpuError::shape(WHAT, detail));
        }
        let gguf = split
            .shard(shard)
            .ok_or_else(|| GpuError::tensor(WHAT, name, "a shard the split holds"))?;
        let src = gguf.data(t)?;
        if self.by_name.remove(name).is_none() {
            return Err(GpuError::tensor(WHAT, name, "a resident weight"));
        }
        let mut ring = UploadRing::new(stream, upload_budget(t))?;
        let dw = upload_rows(
            stream,
            &mut ring,
            WHAT,
            format,
            t,
            tensor_rows(t),
            &[src],
            None,
        )?;
        ring.finish(stream)?;
        self.by_name.insert(name.to_owned(), dw);
        Ok(())
    }

    /// The resident weight `name`, refused by name as `what`'s when absent.
    pub fn resident(&self, what: &'static str, name: &str) -> Result<&DevWeight, GpuError> {
        self.get(name)
            .ok_or_else(|| GpuError::tensor(what, name, "a resident weight"))
    }

    /// The two planes (`qs`, `d`) of the resident Q8_0 weight `name`; any
    /// other variant is refused by name as `what`'s.
    pub fn q8_planes(
        &self,
        what: &'static str,
        name: &str,
    ) -> Result<(&DeviceTensor<u32>, &DeviceTensor<u16>), GpuError> {
        match self.resident(what, name)? {
            DevWeight::Q8_0 { qs, d, .. } => Ok((qs, d)),
            _ => Err(GpuError::tensor(what, name, "a q8_0 weight")),
        }
    }

    /// The planes and the scale of the resident NVFP4 weight `name`; any
    /// other variant is refused by name as `what`'s.
    pub fn nvfp4(&self, what: &'static str, name: &str) -> Result<Nvfp4Weight<'_>, GpuError> {
        match self.resident(what, name)? {
            DevWeight::Nvfp4 { qs, e, scale, .. } => Ok(Nvfp4Weight {
                qs,
                e,
                scale: *scale,
            }),
            _ => Err(GpuError::tensor(what, name, "an nvfp4 weight")),
        }
    }

    /// The planes of the resident E4M3 weight `name` (MXFP8 or FP8_B128) and
    /// the scales they hold; any other variant is refused by name as
    /// `what`'s.
    pub fn e4m3(&self, what: &'static str, name: &str) -> Result<E4m3Weight<'_>, GpuError> {
        let (qs, e, scales) = match self.resident(what, name)? {
            DevWeight::Mxfp8 { qs, e, .. } => (qs, e, E4m3Scales::Mxfp8),
            DevWeight::Fp8B128 { qs, e, .. } => (qs, e, E4m3Scales::Fp8B128),
            _ => return Err(GpuError::tensor(what, name, "an mxfp8 or fp8_b128 weight")),
        };
        Ok(E4m3Weight { qs, e, scales })
    }

    /// The words of the resident BF16 weight `name` (rows × k/2); any other
    /// variant is refused by name as `what`'s.
    pub fn bf16(&self, what: &'static str, name: &str) -> Result<&DeviceTensor<u32>, GpuError> {
        match self.resident(what, name)? {
            DevWeight::Bf16 { w, .. } => Ok(w),
            _ => Err(GpuError::tensor(what, name, "a bf16 weight")),
        }
    }

    /// The planes and the experts' scales of the resident NVFP4 stack
    /// `name`; any other variant is refused by name as `what`'s.
    pub fn nvfp4_stack(&self, what: &'static str, name: &str) -> Result<Nvfp4Stack<'_>, GpuError> {
        match self.resident(what, name)? {
            DevWeight::Nvfp4Stack { qs, e, scales, .. } => Ok(Nvfp4Stack { qs, e, scales }),
            _ => Err(GpuError::tensor(what, name, "an nvfp4 stack")),
        }
    }

    /// The resident F32 tensor `name` (an f32 file tensor, or a bf16 one
    /// decoded at load); any other variant is refused by name as `what`'s.
    pub fn f32_tensor(
        &self,
        what: &'static str,
        name: &str,
    ) -> Result<&DeviceTensor<f32>, GpuError> {
        match self.resident(what, name)? {
            DevWeight::F32 { w, .. } => Ok(w),
            _ => Err(GpuError::tensor(what, name, "an f32 tensor")),
        }
    }

    /// The resident F32 tensor `name` as the buffer a kernel reads (a norm
    /// gain, a bias); refused as [`Weights::f32_tensor`] is.
    pub fn f32_buf(&self, what: &'static str, name: &str) -> Result<&DeviceBuffer<f32>, GpuError> {
        self.f32_tensor(what, name).map(DeviceTensor::buf)
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

/// A file tensor's name, refused in the [`DERIVED_PREFIX`] namespace: a file
/// tensor never stands where a derived weight is filed, so a lookup of a
/// derived name finds only what `insert_derived` filed.
fn file_slot(name: &str) -> Result<(), GpuError> {
    if name.starts_with(DERIVED_PREFIX) {
        return Err(GpuError::shape(
            "Weights::load",
            format!("file tensor {name:?} has a {DERIVED_PREFIX:?} name, which the engine derives"),
        ));
    }
    Ok(())
}

/// `Some(L)` for a `blk.L.*` tensor name, `None` for a non-block name; a
/// `blk.` name whose layer does not parse is an error naming the tensor.
pub fn block_index(name: &str) -> Result<Option<usize>, GpuError> {
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

/// Resident bytes in the type's default card format, with `rows` rows of
/// `k` values; `None` when the type or shape has no layout. A file tensor's
/// name can select another format: its census uses [`part_format`].
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

/// Reads a part's `qs` words, its scales and its width, `None` for another format.
type PartPlanes<S> =
    for<'a> fn(&'a DevWeight) -> Option<(&'a DeviceTensor<u32>, &'a DeviceTensor<S>, usize)>;

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
        file_slot(&t.name)?;
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

/// Where placement segment `seg` of tensor `t`, a stack of `experts` when it
/// is one, lies in a split ([`SegmentFile::of`]): the tensor's shard and
/// header, that shard's reader, the segment's spans, and their file bytes as
/// one borrowed slice a span. The upload ([`upload_segment`]) and a tier
/// node's digest of the rows it holds (`host::node`) read these bytes.
pub(crate) struct SegmentFile<'f> {
    pub shard: usize,
    pub info: &'f TensorInfo,
    pub gguf: &'f Gguf,
    pub spans: Vec<Span>,
    pub bytes: Vec<&'f [u8]>,
}

impl<'f> SegmentFile<'f> {
    /// Find `t` in `split`, confirm it is the tensor the placement was made
    /// from, and borrow the file bytes of `seg`'s spans; a tensor the split
    /// lacks or holds otherwise, and spans past its bytes, are refused by
    /// name.
    pub(crate) fn of(
        split: &'f Split,
        experts: u64,
        t: &ModelTensor,
        seg: &Segment,
    ) -> Result<SegmentFile<'f>, GpuError> {
        let spans = seg
            .spans(t, experts)
            .map_err(|e| GpuError::plan(PLACED, e))?;
        let (shard, info) = split
            .find(&t.name)
            .ok_or_else(|| placed_refusal(t, "is not in the split".to_string()))?;
        if shard != t.shard || info.ty != t.ty || info.dims != t.dims || info.nbytes != t.file_bytes
        {
            let detail = format!(
                "the split's shard {shard} {} {:?} ({} bytes) is not the plan's shard {} {} {:?} \
                 ({} bytes)",
                info.ty, info.dims, info.nbytes, t.shard, t.ty, t.dims, t.file_bytes
            );
            return Err(placed_refusal(t, detail));
        }
        let gguf = split
            .shard(shard)
            .ok_or_else(|| placed_refusal(t, format!("shard {shard} is not in the split")))?;
        let data = gguf.data(info)?;
        let mut bytes = Vec::with_capacity(spans.len());
        for span in &spans {
            let range = (|| {
                Some(
                    usize::try_from(span.bytes.start).ok()?
                        ..usize::try_from(span.bytes.end).ok()?,
                )
            })()
            .and_then(|r| data.get(r));
            let slice = range.ok_or_else(|| {
                placed_refusal(
                    t,
                    format!("spans {spans:?} run past its {} file bytes", info.nbytes),
                )
            })?;
            bytes.push(slice);
        }
        Ok(SegmentFile {
            shard,
            info,
            gguf,
            spans,
            bytes,
        })
    }
}

/// Upload placement segment `seg` of tensor `t`, a stack of `experts` when
/// it is one: find its bytes in `split` ([`SegmentFile::of`]), upload the
/// rows the segment holds (their file bytes as one borrowed slice per span,
/// so a whole-tensor segment uploads from the mapping itself and an expert
/// stack streams its spans without a gathered copy), and hand back the file
/// bytes the caller may release now that they are staged — empty for a
/// tensor a step reads from the file.
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
    let SegmentFile {
        shard: s,
        info,
        gguf: g,
        spans,
        bytes: src,
    } = SegmentFile::of(split, experts, t, seg)?;
    let held: u64 = spans.iter().map(|s| s.rows.end - s.rows.start).sum();
    let rows =
        usize::try_from(held).map_err(|_| placed_refusal(t, format!("{held} rows pass usize")))?;
    let scale = tensor_scales(PLACED, info, |n| {
        let (s, t) = split.find(n)?;
        Some((split.shard(s)?, t))
    })?
    .map(|all| match &seg.experts {
        // The listed experts' scales, in the order the planes hold them.
        Some(list) if info.dims.len() == 3 => list.ids().iter().map(|&e| all[e as usize]).collect(),
        _ => all,
    });
    let dw = upload_rows(stream, ring, PLACED, format, info, rows, &src, scale)?;
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
    let dw = upload_file_tensor_with(&mut ring, stream, gguf, t, None, |n| {
        Some((gguf, gguf.find(n)?))
    })?;
    ring.finish(stream)?;
    Ok(dw)
}

/// [`upload_file_tensor`] on a caller's ring, which the caller synchronizes, in format `card`, or
/// its type's own when `card` is `None`. `find` resolves a tensor of the file by name (its shard
/// and entry), so an NVFP4 weight's `.scale` may sit in another shard of the split.
fn upload_file_tensor_with<'g>(
    ring: &mut UploadRing,
    stream: &CudaStream,
    gguf: &Gguf,
    t: &TensorInfo,
    card: Option<CardFormat>,
    find: impl Fn(&str) -> Option<(&'g Gguf, &'g TensorInfo)>,
) -> Result<DevWeight, GpuError> {
    let Some(format) = card.or_else(|| file_format(t)) else {
        return Err(GpuError::shape(
            "Weights::load",
            format!("tensor {} has type {} with no device format", t.name, t.ty),
        ));
    };
    let scale = tensor_scales("Weights::load", t, find)?;
    upload_rows(
        stream,
        ring,
        "Weights::load",
        format,
        t,
        tensor_rows(t),
        &[gguf.data(t)?],
        scale,
    )
}

/// The tensor scales of `t` ([`gguf::tensor_scales`]), its refusals named as `what`'s.
pub(crate) fn tensor_scales<'g>(
    what: &'static str,
    t: &TensorInfo,
    find: impl Fn(&str) -> Option<(&'g Gguf, &'g TensorInfo)>,
) -> Result<Option<Vec<f32>>, GpuError> {
    gguf::tensor_scales(t, find).map_err(|e| match e {
        gguf::LoadError::Scale { .. } => GpuError::shape(what, e.to_string()),
        e => e.into(),
    })
}

/// The bytes `t`'s upload moves, for sizing a load's staging ring: its
/// resident size in its named file format, its file bytes otherwise. This
/// is a sizing hint, never a contract: the uploads check their own lengths.
fn upload_budget(t: &TensorInfo) -> usize {
    part_format(t).map_or_else(
        |_| usize::try_from(t.nbytes).unwrap_or(usize::MAX),
        |(_, bytes)| bytes,
    )
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
/// errors. `scale` is an NVFP4 tensor's scales ([`tensor_scales`]; a stack's
/// those of the experts `rows` hold, in their order), which its planes
/// carry.
#[allow(
    clippy::too_many_arguments,
    reason = "one upload's stream, ring, caller, format, tensor, rows, bytes and scale (rust-quality R8)"
)]
fn upload_rows(
    stream: &CudaStream,
    ring: &mut UploadRing,
    what: &'static str,
    format: CardFormat,
    t: &TensorInfo,
    rows: usize,
    src: &[&[u8]],
    scale: Option<Vec<f32>>,
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
        CardFormat::Nvfp4Planes => {
            let scale = scale.ok_or_else(|| {
                GpuError::shape(
                    what,
                    format!("tensor {name}: an nvfp4 upload without its scale"),
                )
            })?;
            // A stack holds a scale for each of its experts the rows hold;
            // any other tensor holds one.
            let stack = match (t.dims.as_slice(), scale.as_slice()) {
                (&[_, per, _], held) if held.len() as u64 * per == rows as u64 => true,
                (dims, &[_]) if dims.len() != 3 => false,
                _ => {
                    let detail = format!(
                        "tensor {name}: {} scales for {rows} rows of {:?}",
                        scale.len(),
                        t.dims
                    );
                    return Err(GpuError::shape(what, detail));
                }
            };
            let Planes { qs, e } = repack::<Nvfp4>(&flat_of(src, take)?, rows, k)?;
            let qs = upload_packed(stream, ring, what, name, &qs, rows, k / 8)?;
            let e = upload_packed(stream, ring, what, name, &e, rows, k / 64)?;
            let scales = upload_packed(stream, ring, what, name, &scale, scale.len(), 1)?;
            if stack {
                DevWeight::Nvfp4Stack { qs, e, scales, k }
            } else {
                DevWeight::Nvfp4 {
                    qs,
                    e,
                    k,
                    scale: scale[0],
                    scales,
                }
            }
        }
        CardFormat::Mxfp8Planes | CardFormat::Fp8B128Planes => {
            let scales = if format == CardFormat::Mxfp8Planes {
                E4m3Scales::Mxfp8
            } else {
                E4m3Scales::Fp8B128
            };
            let bytes = flat_of(src, take)?;
            let (qs, e) = if t.ty == GgmlType::BF16 {
                crate::q8f32::bf16_mxfp8_planes(&bytes, rows, k)
            } else {
                e4m3_planes(scales, &bytes, rows, k)
            }
            .map_err(|err| GpuError::shape(what, format!("tensor {name}: {err}")))?;
            let qs = upload_packed(stream, ring, what, name, &qs, rows, k / 4)?;
            let e = upload_packed(stream, ring, what, name, &e, rows, k / 128)?;
            match scales {
                E4m3Scales::Mxfp8 => DevWeight::Mxfp8 { qs, e, k },
                E4m3Scales::Fp8B128 => DevWeight::Fp8B128 { qs, e, k },
            }
        }
        CardFormat::Bf16Raw => {
            // The file's bf16 bytes are the words' bytes on this host, two
            // values a word, so the rows stream as they are; a run that is not
            // the rows' whole values, or a row that ends inside a word, is
            // refused.
            if !k.is_multiple_of(2) || take != rows * k * 2 {
                let detail = format!(
                    "tensor {name}: {take} bytes are not its {rows} rows of {k} bf16 values in whole words"
                );
                return Err(GpuError::shape(what, detail));
            }
            let w = DeviceTensor::<u32>::zeroed(stream, rows, k / 2)?;
            let stage = Stage {
                what,
                name,
                dst: w.buf().cu_deviceptr(),
                dst_bytes: w.buf().num_bytes(),
            };
            ring.copy_bytes(stream, &stage, src, take)?;
            DevWeight::Bf16 { w, k }
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
    // An nvfp4 tensor's scales sit beside its planes, one f32 an expert (a
    // dense tensor's one): the planner's third buffer of it.
    let size = match &dw {
        DevWeight::Nvfp4 { scales, .. } | DevWeight::Nvfp4Stack { scales, .. } => {
            size + 4 * scales.rows() as u64
        }
        _ => size,
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
    use super::{
        HEAD_TENSOR, TensorPart, derived_slot_free, file_slot, head_tensor, part_format,
        part_formats, upload_budget,
    };
    use gguf::Split;
    use gguf::write::{Layout, TensorDecl, Writer};
    use gguf::{GgmlType, TensorInfo};
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

    #[test]
    fn file_upload_budget_counts_the_named_format() {
        for (name, ty, k, bytes) in [
            (HEAD_TENSOR, GgmlType::BF16, 256, 16_896),
            (HEAD_TENSOR, GgmlType::BF16, 320, 40_960),
            ("token_embd.weight", GgmlType::BF16, 256, 65_536),
            ("plain.weight", GgmlType::BF16, 256, 65_536),
            (HEAD_TENSOR, GgmlType::F32, 256, 65_536),
            (HEAD_TENSOR, GgmlType::Q8_0, 256, 17_408),
        ] {
            let (block, size) = (ty.blck_size().unwrap(), ty.type_size().unwrap());
            let t = TensorInfo {
                name: name.into(),
                ty,
                dims: vec![k, 64],
                offset: 0,
                nbytes: k / block * size * 64,
            };
            assert_eq!(part_format(&t).unwrap().1, bytes, "{name} {ty} k={k}");
            assert_eq!(upload_budget(&t), bytes, "{name} {ty} k={k}");
        }
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

    /// A file tensor in the derived namespace is refused by name, so it never
    /// stands in for a weight the engine derives (a nemotron_h zero conv
    /// bias); every other file name is taken.
    #[test]
    fn a_file_tensor_never_takes_a_derived_name() {
        let err = file_slot("derived.nemotron_h.zero_conv_bias")
            .expect_err("a derived name in the file")
            .to_string();
        assert!(err.contains("derived.nemotron_h.zero_conv_bias"), "{err}");
        assert!(file_slot("output.weight").is_ok());
    }

    #[test]
    fn malformed_part_sizes_are_refused_before_upload() {
        let mut accepted = Vec::new();
        for (dims, nbytes, spans) in [
            (vec![256, 1], 143, vec![&[0u8; 143][..]]),
            (vec![256, 1], 145, vec![&[0u8; 145][..]]),
            (vec![0, 1], 0, vec![]),
            (vec![256, 0], 0, vec![]),
            (vec![], 0, vec![]),
            (vec![257, 1], 144, vec![&[0u8; 144][..]]),
            (vec![256, u64::MAX], 144, vec![&[0u8; 144][..]]),
            (vec![256, 1], 144, vec![&[0u8; 143][..]]),
            (vec![256, 1], 144, vec![&[0u8; 145][..]]),
        ] {
            let part = TensorPart {
                info: TensorInfo {
                    name: "bad".into(),
                    dims: dims.clone(),
                    ty: GgmlType::Q4_K,
                    offset: 0,
                    nbytes,
                },
                spans,
                scale: None,
            };
            if !std::panic::catch_unwind(|| part_formats(&[part], |_| false))
                .is_ok_and(|result| result.is_err())
            {
                accepted.push((dims, nbytes));
            }
        }
        assert!(accepted.is_empty(), "not refused safely: {accepted:?}");
        let part = TensorPart {
            info: TensorInfo {
                name: "good".into(),
                dims: vec![256, 1],
                ty: GgmlType::Q4_K,
                offset: 0,
                nbytes: 144,
            },
            spans: vec![&[0u8; 72], &[0u8; 72]],
            scale: None,
        };
        assert!(part_formats(&[part], |_| false).is_ok());
    }

    /// Every type with a card format takes a part, and of the types a tensor scale multiplies only
    /// NVFP4, whose part carries its scale: a scaled type that gains a card format fails here
    /// until its part upload reads the scale.
    #[test]
    fn part_formats_follows_the_scale_rule() {
        for row in gguf::QROWS {
            let nbytes = gguf::tensor_bytes(row.blck, row.bytes, &[256, 2]).unwrap();
            let bytes = vec![0u8; usize::try_from(nbytes).unwrap()];
            let part = TensorPart {
                info: TensorInfo {
                    name: "part".into(),
                    dims: vec![256, 2],
                    ty: row.ty,
                    offset: 0,
                    nbytes,
                },
                spans: vec![&bytes],
                scale: row.ty.reads_scale().then(|| vec![1.0]),
            };
            let result = part_formats(&[part], |_| false);
            if row.card.layout.is_some() && (!row.ty.reads_scale() || row.ty == GgmlType::NVFP4) {
                assert!(result.is_ok(), "{}: {result:?}", row.ty);
            } else {
                let error = result.unwrap_err().to_string();
                assert!(
                    error.contains("no part upload takes"),
                    "{}: {error}",
                    row.ty
                );
            }
        }
    }

    /// A part list is checked whole before any upload: a name given twice or held already, a
    /// derived name, a type with no part upload, and tensor scales a type reads missing, of
    /// another count or not finite, or given to a type that reads none, are refused by name;
    /// distinct parts of types with a card format pass, an NVFP4 part with its scale and a stack
    /// with one an expert among them, their staging bytes summed.
    #[test]
    fn a_part_list_is_refused_before_any_upload() {
        let part = |name: &str, ty: GgmlType, dims: &[u64], scale: Option<Vec<f32>>| {
            let nbytes =
                gguf::tensor_bytes(ty.blck_size().unwrap(), ty.type_size().unwrap(), dims).unwrap();
            TensorPart {
                info: TensorInfo {
                    name: name.to_string(),
                    dims: dims.to_vec(),
                    ty,
                    offset: 0,
                    nbytes,
                },
                spans: vec![&[0u8; 512][..nbytes as usize]],
                scale,
            }
        };
        let (q8, wide) = (GgmlType::Q8_0, GgmlType::F32);
        let row = [32, 2];
        let none = |_: &str| false;
        let ok = [part("a", q8, &row, None), part("b", wide, &row, None)];
        let (formats, budget) = part_formats(&ok, none).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(formats.len(), 2);
        // Q8_0 [32, 2] stages 68 plane bytes, F32 [32, 2] 256.
        assert_eq!(budget, 68 + 256);
        let nv = |name, dims: &[u64], scale| part(name, GgmlType::NVFP4, dims, scale);
        let scaled = [
            nv("n", &[64, 2], Some(vec![0.5])),
            nv("s", &[64, 1, 3], Some(vec![0.5; 3])),
        ];
        assert!(part_formats(&scaled, none).is_ok());
        for (bad, says) in [
            (
                vec![part("a", q8, &row, None), part("a", wide, &row, None)],
                "given twice",
            ),
            (vec![part("derived.x", wide, &row, None)], "derived"),
            (
                vec![part("r", GgmlType::F8E4M3_ROWS, &row, Some(vec![1.0]))],
                "no part upload takes",
            ),
            (
                vec![nv("n", &[64, 2], None)],
                "reads 1 finite tensor scales; the part carries none",
            ),
            (
                vec![nv("s", &[64, 1, 3], Some(vec![0.5]))],
                "reads 3 finite tensor scales; the part carries 1 values",
            ),
            (
                vec![nv("n", &[64, 2], Some(vec![f32::NAN]))],
                "the part carries 1 values, 0 finite",
            ),
            (
                vec![part("q", q8, &row, Some(vec![1.0]))],
                "reads 0 finite tensor scales",
            ),
        ] {
            let e = part_formats(&bad, none)
                .map(|_| ())
                .unwrap_err()
                .to_string();
            assert!(e.contains(says), "{e}");
        }
        let e = part_formats(&ok[..1], |name| name == "a")
            .map(|_| ())
            .unwrap_err();
        assert!(e.to_string().contains("a given twice"), "{e}");
    }
}
