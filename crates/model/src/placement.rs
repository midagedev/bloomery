//! Where every tensor of a model lives on this machine: one row per tensor with
//! its device, device format, resident bytes and per-token read, the per-device
//! totals, and a check of the invariants the table must keep.
//!
//! The inputs are header facts and device figures only: the model as its
//! architecture classifies it ([`ModelTensors`], one [`Role`] per tensor), the
//! GPU cards in stage order with the layers each runs and the host with its
//! reserves ([`Machine`]), `ctx_max`, and the architecture's KV size
//! ([`KvBytes`]). This module spells no tensor name and knows no architecture;
//! the classifier that feeds it does.
//!
//! Routed experts follow one rule ([`plan`]): the eligible layers of a card
//! keep the same number of experts on it, give or take one, as the id prefix
//! `[0, n_l)`, and the rest stay on the host — a prefix, so the `_sel` kernels
//! need no remapping table (a slot whose id is past their expert count is left
//! untouched).

use std::fmt;
use std::num::NonZeroU64;
use std::ops::Range;

use gguf::GgmlType;

pub mod host_lock;
pub mod workstation;

/// What a tensor does in a decode step; the role decides its device.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Role {
    /// Attention, its compressor and its indexer: on the layer's card.
    Attention,
    /// Hyper-connection mixing: on the layer's card.
    HyperConnection,
    /// The router weight and its selection bias: on the layer's card.
    Router,
    /// The pre-FFN norm: on the layer's card.
    FfnNorm,
    /// The shared expert: on the layer's card.
    SharedExpert,
    /// Engram weights every token reads in full: on the layer's card.
    EngramDense,
    /// An engram table: on NVMe, a few rows gathered per token.
    EngramTable,
    /// A routed expert stack: split by the expert rule between the layer's card and the host.
    RoutedExperts,
    /// The token embedding: on the host, one row gathered per token.
    TokenEmbedding,
    /// The output head and its norm: on the head card.
    Head,
    /// In the file, never read by text decode.
    Unread,
}

impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Role::Attention => "attn",
            Role::HyperConnection => "hc",
            Role::Router => "router",
            Role::FfnNorm => "ffn_norm",
            Role::SharedExpert => "shexp",
            Role::EngramDense => "engram_dense",
            Role::EngramTable => "engram_table",
            Role::RoutedExperts => "routed",
            Role::TokenEmbedding => "token_embd",
            Role::Head => "head",
            Role::Unread => "unread",
        })
    }
}

/// One tensor as its architecture classifies it: the header's facts and its role.
#[derive(Clone, Debug)]
pub struct ModelTensor {
    pub name: String,
    /// The shard that holds it (0 for a single file).
    pub shard: usize,
    /// Its layer; `None` for the model-level tensors.
    pub layer: Option<usize>,
    pub role: Role,
    pub ty: GgmlType,
    /// ggml `ne[]` order: `dims[0]` is the row width, a routed stack's last dim its experts.
    pub dims: Vec<u64>,
    pub file_bytes: u64,
    /// Rows one token reads from a row-gathered table ([`Role::TokenEmbedding`],
    /// [`Role::EngramTable`]); `None` for every other role.
    pub gathered_rows: Option<u64>,
}

/// A model as placement sees it.
#[derive(Clone, Debug)]
pub struct ModelTensors {
    pub tensors: Vec<ModelTensor>,
    pub layers: usize,
    /// Experts in each routed stack, and experts one token uses.
    pub experts: u64,
    pub experts_used: u64,
}

/// A GPU card and the stage it runs.
#[derive(Clone, Debug)]
pub struct Card {
    pub name: String,
    /// What the driver leaves for us.
    pub usable_bytes: u64,
    pub context_bytes: u64,
    pub scratch_bytes: u64,
    /// Kept free: the expert rule never plans into it.
    pub margin_bytes: u64,
    /// The device allocator's granule: an allocation of this size or more
    /// takes whole granules of its own, smaller ones share granules.
    pub granule_bytes: NonZeroU64,
    /// The layers this card's stage runs.
    pub layers: Range<usize>,
    /// Whether this stage ends with the head.
    pub head: bool,
}

/// The host tier: its usable bytes and what is set aside before any tensor.
#[derive(Clone, Debug)]
pub struct Host {
    pub usable_bytes: u64,
    /// Fixed reserves, by what they are for.
    pub reserves: Vec<(String, u64)>,
}

/// The devices: the cards in stage order and the host. The NVMe tier has no
/// figure here — it holds its tensors in place and never loads them.
#[derive(Clone, Debug)]
pub struct Machine {
    pub cards: Vec<Card>,
    pub host: Host,
}

/// An architecture's KV cache size: the bytes `layer` holds at `ctx_max` tokens.
pub trait KvBytes {
    fn layer_bytes(&self, layer: usize, ctx_max: u64) -> u64;
}

/// Where a segment lives.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Device {
    /// A card, by its index in [`Machine::cards`].
    Card(usize),
    Host,
    Nvme,
    /// Nowhere: never loaded.
    Unused,
}

/// A card's layout for a file tensor: the GPU loader's `DevWeight` formats
/// (`crates/gpu/src/weights.rs`). [`CardFormat::buffer_bytes`] is the one
/// owner of their file → device arithmetic: the plan sums it and counts its
/// buffers through the allocator, and the loader refuses and checks every
/// upload by it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CardFormat {
    /// q3_K/q4_K/q6_K: the rows' byte stream as u32 words, the last word
    /// zero-padded, one row per `words / rows` words.
    KQuant,
    /// q5_0 in the gemv's layout: per row one byte per 5-bit code in 1024-value
    /// windows, then one f32 scale per 32-value block.
    Q5_0,
    /// q5_1: the q5_0 layout plus one f32 min per block.
    Q5_1,
    /// q8_0 in two planes: per 32-value block 8 code words and the block's
    /// f16 scale bits as the file stores them.
    Q8_0Planes,
    /// f32: the file's values.
    F32,
    /// bf16 decoded to f32 at load.
    Bf16AsF32,
}

impl CardFormat {
    /// The format a file tensor of type `ty` loads in on a card; `None` when
    /// the loader has none.
    #[must_use]
    pub fn of(ty: GgmlType) -> Option<CardFormat> {
        match ty {
            GgmlType::Q3_K | GgmlType::Q4_K | GgmlType::Q6_K => Some(CardFormat::KQuant),
            GgmlType::Q5_0 => Some(CardFormat::Q5_0),
            GgmlType::Q5_1 => Some(CardFormat::Q5_1),
            GgmlType::Q8_0 => Some(CardFormat::Q8_0Planes),
            GgmlType::F32 => Some(CardFormat::F32),
            GgmlType::BF16 => Some(CardFormat::Bf16AsF32),
            GgmlType::F16 | GgmlType::Q5_K | GgmlType::Unknown(_) => None,
        }
    }

    /// The device buffers of `rows` rows of `k` values of file type `ty` in
    /// this format, in bytes, in the order the upload allocates them: q8_0's
    /// code plane, then its scale plane; one buffer in every other format —
    /// the upload's own arithmetic. `None` exactly where the upload refuses: a
    /// zero dimension, `k` off the type's block, a KQuant stream whose words
    /// do not split evenly into `rows`, or a `ty` this format is not the
    /// loader's format for.
    #[must_use]
    pub fn buffer_bytes(self, ty: GgmlType, k: u64, rows: u64) -> Option<Vec<u64>> {
        let blck = ty.blck_size()?;
        if CardFormat::of(ty) != Some(self) || k == 0 || rows == 0 || !k.is_multiple_of(blck) {
            return None;
        }
        let blocks = k / blck;
        // The q5 code section: one byte per code, in whole 1024-value windows.
        let window = || k.div_ceil(1024).checked_mul(1024);
        Some(match self {
            CardFormat::KQuant => {
                let words = ty
                    .type_size()?
                    .checked_mul(blocks)?
                    .checked_mul(rows)?
                    .div_ceil(4);
                let bytes = words
                    .is_multiple_of(rows)
                    .then_some(words)?
                    .checked_mul(4)?;
                vec![bytes]
            }
            CardFormat::Q5_0 => {
                vec![rows.checked_mul(window()?.checked_add(blocks.checked_mul(4)?)?)?]
            }
            CardFormat::Q5_1 => {
                vec![rows.checked_mul(window()?.checked_add(blocks.checked_mul(8)?)?)?]
            }
            CardFormat::Q8_0Planes => {
                vec![
                    rows.checked_mul(k)?,
                    rows.checked_mul(blocks.checked_mul(2)?)?,
                ]
            }
            CardFormat::F32 | CardFormat::Bf16AsF32 => vec![rows.checked_mul(k.checked_mul(4)?)?],
        })
    }

    /// Device bytes of `rows` rows of `k` values of file type `ty` in this
    /// format: the sum of its [`CardFormat::buffer_bytes`], `None` where they
    /// are.
    #[must_use]
    pub fn resident_bytes(self, ty: GgmlType, k: u64, rows: u64) -> Option<u64> {
        self.buffer_bytes(ty, k, rows)?
            .into_iter()
            .try_fold(0u64, u64::checked_add)
    }
}

/// Where a shared granule's next allocation may start: a multiple of this
/// [assumed — gate ② checks only the totals it implies].
const SMALL_ALIGN: u64 = 512;

/// A card's device allocator as the plan counts it: an allocation of a
/// granule or more takes whole granules of its own; a smaller one takes its
/// size, rounded up to [`SMALL_ALIGN`], from the first shared granule, in
/// allocation order, that has that much left, and opens a new shared granule
/// when none has.
struct Heap {
    granule: u64,
    /// Bytes of the granules taken so far.
    taken: u64,
    /// Bytes left in each shared granule, in the order they were opened.
    left: Vec<u64>,
}

impl Heap {
    fn new(granule: NonZeroU64) -> Heap {
        Heap {
            granule: granule.get(),
            taken: 0,
            left: Vec::new(),
        }
    }

    fn alloc(&mut self, bytes: u64) {
        if bytes >= self.granule {
            self.taken += bytes.div_ceil(self.granule) * self.granule;
            return;
        }
        let room = bytes.next_multiple_of(SMALL_ALIGN).min(self.granule);
        match self.left.iter_mut().find(|left| **left >= room) {
            Some(left) => *left -= room,
            None => {
                self.left.push(self.granule - room);
                self.taken += self.granule;
            }
        }
    }
}

/// How a segment's bytes are laid out where it lives.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Format {
    Card(CardFormat),
    /// On the host: the shard's mapped bytes, file bytes.
    HostFile,
    /// On NVMe: the file's bytes, never loaded.
    NvmeFile,
    /// Not loaded: 0 bytes.
    Unused,
}

impl fmt::Display for Format {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Format::Card(CardFormat::KQuant) => "kquant",
            Format::Card(CardFormat::Q5_0) => "q5_0",
            Format::Card(CardFormat::Q5_1) => "q5_1",
            Format::Card(CardFormat::Q8_0Planes) => "q8_0_planes",
            Format::Card(CardFormat::F32) => "f32",
            Format::Card(CardFormat::Bf16AsF32) => "bf16_as_f32",
            Format::HostFile => "host_file",
            Format::NvmeFile => "nvme_file",
            Format::Unused => "unused",
        })
    }
}

/// One piece of a tensor on one device.
#[derive(Clone, Debug)]
pub struct Segment {
    pub device: Device,
    pub format: Format,
    /// The experts `[start, end)` this piece holds; `None` for a tensor that is
    /// not an expert stack.
    pub experts: Option<Range<u64>>,
    pub resident_bytes: u64,
}

/// Where a segment's piece lies in its tensor, counted from the tensor's
/// first row and first byte in the file.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Span {
    pub rows: Range<u64>,
    pub bytes: Range<u64>,
}

impl Segment {
    /// The rows of `t` this segment holds — all of them, or its experts' rows
    /// of a stack of `experts` — and the file bytes those rows take.
    pub fn span(&self, t: &ModelTensor, experts: u64) -> Result<Span, PlacementError> {
        let rb = row_bytes(t)?;
        let rows = match &self.experts {
            None => 0..rows_of(t),
            Some(e) => {
                let (per, _) = per_expert(t, experts)?;
                if e.start > e.end || e.end > experts {
                    return Err(PlacementError::tensor(
                        t,
                        format!("expert range {e:?} is not inside 0..{experts}"),
                    ));
                }
                e.start * per..e.end * per
            }
        };
        Ok(Span {
            bytes: rows.start * rb..rows.end * rb,
            rows,
        })
    }

    /// The device buffers this card segment of `t` uploads, in bytes, in
    /// upload order — [`CardFormat::buffer_bytes`] of the rows it holds — or
    /// why they cannot be derived: a segment in no card format, a span
    /// outside the tensor, rows the upload refuses.
    pub fn buffer_bytes(&self, t: &ModelTensor, experts: u64) -> Result<Vec<u64>, PlacementError> {
        let Format::Card(format) = self.format else {
            return Err(PlacementError::tensor(
                t,
                format!("a segment as {} has no card buffers", self.format),
            ));
        };
        let span = self.span(t, experts)?;
        card_buffers(t, format, span.rows.end - span.rows.start)
    }
}

/// One tensor's placement.
#[derive(Clone, Debug)]
pub struct Row {
    /// Index into [`ModelTensors::tensors`].
    pub tensor: usize,
    pub segments: Vec<Segment>,
    /// Bytes one decode token reads from it, in the file's format.
    pub read_bytes: u64,
    /// The card whose stage uses it.
    pub stage: usize,
}

/// A card's side of the plan.
#[derive(Clone, Debug)]
pub struct CardTotals {
    /// Resident bytes of everything on the card but the routed experts.
    pub dense_bytes: u64,
    pub expert_bytes: u64,
    /// What the allocator takes for the card's uploads beyond their resident
    /// bytes: their granules, counted in upload order ([`Heap`]), minus dense
    /// and experts.
    pub rounding_bytes: u64,
    /// Experts held, summed over the card's layers.
    pub experts: u64,
    pub kv_bytes: u64,
    pub scratch_bytes: u64,
    pub context_bytes: u64,
    /// usable − dense − experts − rounding − KV − scratch − context; the
    /// margin is inside it.
    pub headroom_bytes: i128,
}

/// The host's side of the plan.
#[derive(Clone, Debug)]
pub struct HostTotals {
    pub expert_bytes: u64,
    pub experts: u64,
    /// Row-gathered tables held on the host.
    pub table_bytes: u64,
    pub reserve_bytes: u64,
    /// usable − experts − tables − reserves.
    pub headroom_bytes: i128,
}

/// The placement of one model on one machine.
#[derive(Clone, Debug)]
pub struct Plan<'a> {
    pub model: &'a ModelTensors,
    pub machine: &'a Machine,
    pub ctx_max: u64,
    /// One row per tensor, in the model's tensor order.
    pub rows: Vec<Row>,
    /// Per card, in stage order.
    pub cards: Vec<CardTotals>,
    pub host: HostTotals,
    pub nvme_bytes: u64,
    /// Per layer: its card holds experts `[0, n_l)`, the host the rest.
    pub n_l: Vec<u64>,
}

/// Why a placement could not be built; each variant names what it refused.
#[derive(Debug, thiserror::Error)]
pub enum PlacementError {
    /// Every name the classifier has no rule for, not only the first.
    #[error("{} tensors have no role: {}", names.len(), names.join(", "))]
    Unclassified { names: Vec<String> },
    #[error("metadata {key}: {detail}")]
    Metadata { key: String, detail: String },
    #[error("tensor {name}: {detail}")]
    Tensor { name: String, detail: String },
    /// The cards' layer ranges do not cover the model once, in order, with one head.
    #[error("stage map: {0}")]
    Stages(String),
    /// The host tier's lock, or its residency query, failed: the span and the
    /// kernel's answer.
    #[error("host lock: {0}")]
    Host(String),
}

impl PlacementError {
    fn tensor(t: &ModelTensor, detail: impl Into<String>) -> PlacementError {
        PlacementError::Tensor {
            name: t.name.clone(),
            detail: detail.into(),
        }
    }
}

/// An invariant a plan breaks. [`Plan::violations`] returns every one.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Violation {
    /// A tensor with no row, or with more than one.
    PlacedTimes { tensor: String, times: usize },
    /// Segments its role does not allow: an expert stack that does not cover
    /// every expert exactly once, a whole tensor in pieces, or a card segment
    /// whose buffers cannot be derived or do not sum to its resident bytes.
    Segments { tensor: String, detail: String },
    /// A segment on a card whose stage does not run the tensor's layer or head.
    WrongCard { tensor: String, card: String },
    /// A card whose uploads' granules, KV, scratch and context pass usable − margin.
    CardOver {
        card: String,
        total: u64,
        limit: u64,
    },
    /// The host's tensors and reserves pass its usable bytes.
    HostOver { total: u64, usable: u64 },
}

impl fmt::Display for Violation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Violation::PlacedTimes { tensor, times } => {
                write!(f, "tensor {tensor} is placed {times} times, not once")
            }
            Violation::Segments { tensor, detail } => write!(f, "tensor {tensor}: {detail}"),
            Violation::WrongCard { tensor, card } => {
                write!(
                    f,
                    "tensor {tensor} is on card {card}, whose stage does not use it"
                )
            }
            Violation::CardOver { card, total, limit } => write!(
                f,
                "card {card}: resident + rounding + KV + scratch + context = {total} B passes usable − margin = {limit} B by {} B",
                total - limit
            ),
            Violation::HostOver { total, usable } => write!(
                f,
                "host: tensors + reserves = {total} B pass usable {usable} B by {} B",
                total - usable
            ),
        }
    }
}

/// Which card runs each layer, and which ends with the head.
struct Stages {
    of_layer: Vec<usize>,
    head: usize,
}

impl Stages {
    fn new(layers: usize, cards: &[Card]) -> Result<Stages, PlacementError> {
        let mut of_layer = Vec::with_capacity(layers);
        for (c, card) in cards.iter().enumerate() {
            if card.layers.start != of_layer.len() || card.layers.is_empty() {
                return Err(PlacementError::Stages(format!(
                    "card {} runs layers {:?}, but the next layer to place is {}",
                    card.name,
                    card.layers,
                    of_layer.len()
                )));
            }
            of_layer.extend(card.layers.clone().map(|_| c));
        }
        if of_layer.len() != layers {
            return Err(PlacementError::Stages(format!(
                "the cards run {} layers, the model has {layers}",
                of_layer.len()
            )));
        }
        let heads: Vec<usize> = (0..cards.len()).filter(|&c| cards[c].head).collect();
        let &[head] = heads.as_slice() else {
            return Err(PlacementError::Stages(format!(
                "{} cards carry the head, not one",
                heads.len()
            )));
        };
        Ok(Stages { of_layer, head })
    }

    /// The card whose stage uses `t`: its layer's, the head's for the head,
    /// the first stage's for a model-level tensor.
    fn stage_of(&self, t: &ModelTensor, layers: usize) -> Result<usize, PlacementError> {
        match (t.role, t.layer) {
            (Role::Head, _) => Ok(self.head),
            (_, None) => Ok(self.of_layer[0]),
            (_, Some(_)) => Ok(self.of_layer[layer_of(t, layers)?]),
        }
    }
}

/// Rows as the loader counts them: the product of every dim past the row width.
fn rows_of(t: &ModelTensor) -> u64 {
    t.dims.iter().skip(1).product()
}

/// Bytes of one row in the file's format.
fn row_bytes(t: &ModelTensor) -> Result<u64, PlacementError> {
    let rows = rows_of(t);
    if rows == 0 || !t.file_bytes.is_multiple_of(rows) {
        return Err(PlacementError::tensor(
            t,
            format!("{} file bytes do not split into {rows} rows", t.file_bytes),
        ));
    }
    Ok(t.file_bytes / rows)
}

/// `t`'s layer, which its role requires.
fn layer_of(t: &ModelTensor, layers: usize) -> Result<usize, PlacementError> {
    match t.layer {
        Some(l) if l < layers => Ok(l),
        Some(l) => Err(PlacementError::tensor(
            t,
            format!("layer {l} is past the model's {layers}"),
        )),
        None => Err(PlacementError::tensor(
            t,
            format!("role {} needs a layer", t.role),
        )),
    }
}

/// Device bytes of `rows` rows of `t` in `format`, or the refusal that names
/// it: rows the upload refuses are refused, not given another formula.
fn card_bytes(t: &ModelTensor, format: CardFormat, rows: u64) -> Result<u64, PlacementError> {
    let k = t.dims.first().copied().unwrap_or(0);
    format
        .resident_bytes(t.ty, k, rows)
        .ok_or_else(|| refused(t, format, k, rows))
}

/// The device buffers of `rows` rows of `t` in `format`, in upload order, or
/// the refusal [`card_bytes`] gives.
fn card_buffers(
    t: &ModelTensor,
    format: CardFormat,
    rows: u64,
) -> Result<Vec<u64>, PlacementError> {
    let k = t.dims.first().copied().unwrap_or(0);
    format
        .buffer_bytes(t.ty, k, rows)
        .ok_or_else(|| refused(t, format, k, rows))
}

fn refused(t: &ModelTensor, format: CardFormat, k: u64, rows: u64) -> PlacementError {
    PlacementError::tensor(
        t,
        format!("{rows} rows of {k} values: the {format:?} upload refuses them"),
    )
}

/// Each card's allocator after its segments' uploads, fed in the loader's
/// order — the rows' order, which is the model's tensor order and the order
/// `Weights::load_placed` walks — and each card segment whose buffers cannot
/// be derived or do not sum to its resident bytes. Such a segment still
/// counts, as one allocation of its resident bytes, so a heap never holds
/// less than its card's resident bytes.
fn card_heaps(model: &ModelTensors, cards: &[Card], rows: &[Row]) -> (Vec<Heap>, Vec<Violation>) {
    let mut heaps: Vec<Heap> = cards.iter().map(|c| Heap::new(c.granule_bytes)).collect();
    let mut bad = Vec::new();
    for r in rows {
        let Some(t) = model.tensors.get(r.tensor) else {
            continue;
        };
        for s in &r.segments {
            let Device::Card(c) = s.device else {
                continue;
            };
            let (bufs, detail) = match s.buffer_bytes(t, model.experts) {
                Ok(bufs) if bufs.iter().sum::<u64>() == s.resident_bytes => (bufs, None),
                Ok(bufs) => (
                    vec![s.resident_bytes],
                    Some(format!(
                        "card buffers {bufs:?} do not sum to the segment's {} resident bytes",
                        s.resident_bytes
                    )),
                ),
                Err(PlacementError::Tensor { detail, .. }) => {
                    (vec![s.resident_bytes], Some(detail))
                }
                Err(e) => (vec![s.resident_bytes], Some(e.to_string())),
            };
            if let Some(heap) = heaps.get_mut(c) {
                for b in bufs {
                    heap.alloc(b);
                }
            }
            if let Some(detail) = detail {
                bad.push(Violation::Segments {
                    tensor: t.name.clone(),
                    detail,
                });
            }
        }
    }
    (heaps, bad)
}

/// A card segment of `rows` rows of `t` — all of it, or an expert slice.
fn card_segment(
    t: &ModelTensor,
    card: usize,
    rows: u64,
    experts: Option<Range<u64>>,
) -> Result<Segment, PlacementError> {
    let Some(format) = CardFormat::of(t.ty) else {
        return Err(PlacementError::tensor(
            t,
            format!(
                "type {} has no device format, but its role puts it on a card",
                t.ty
            ),
        ));
    };
    Ok(Segment {
        device: Device::Card(card),
        format: Format::Card(format),
        experts,
        resident_bytes: card_bytes(t, format, rows)?,
    })
}

/// A segment that holds a tensor's file bytes where they are.
fn in_place(device: Device, format: Format, experts: Option<Range<u64>>, bytes: u64) -> Segment {
    Segment {
        device,
        format,
        experts,
        resident_bytes: bytes,
    }
}

/// Every tensor but a routed stack: one segment, where its role says.
fn place_whole(
    i: usize,
    t: &ModelTensor,
    stages: &Stages,
    layers: usize,
) -> Result<Row, PlacementError> {
    let stage = stages.stage_of(t, layers)?;
    let (segment, read_bytes) = match t.role {
        Role::Attention
        | Role::HyperConnection
        | Role::Router
        | Role::FfnNorm
        | Role::SharedExpert
        | Role::EngramDense => {
            layer_of(t, layers)?;
            (card_segment(t, stage, rows_of(t), None)?, t.file_bytes)
        }
        Role::Head => (card_segment(t, stage, rows_of(t), None)?, t.file_bytes),
        Role::TokenEmbedding => (
            in_place(Device::Host, Format::HostFile, None, t.file_bytes),
            gathered(t)?,
        ),
        Role::EngramTable => {
            layer_of(t, layers)?;
            (
                in_place(Device::Nvme, Format::NvmeFile, None, t.file_bytes),
                gathered(t)?,
            )
        }
        Role::Unread => (in_place(Device::Unused, Format::Unused, None, 0), 0),
        Role::RoutedExperts => {
            return Err(PlacementError::tensor(
                t,
                "an expert stack is placed by the expert rule, not whole",
            ));
        }
    };
    Ok(Row {
        tensor: i,
        segments: vec![segment],
        read_bytes,
        stage,
    })
}

/// Bytes a token reads from a row-gathered table: its gathered rows.
fn gathered(t: &ModelTensor) -> Result<u64, PlacementError> {
    let Some(rows) = t.gathered_rows else {
        return Err(PlacementError::tensor(
            t,
            "a row-gathered table without a per-token row count",
        ));
    };
    Ok(rows * row_bytes(t)?)
}

/// A routed stack's rows and file bytes per expert; its last dim must be the
/// expert count and both totals must split evenly.
fn per_expert(t: &ModelTensor, experts: u64) -> Result<(u64, u64), PlacementError> {
    let rows = rows_of(t);
    if experts == 0
        || t.dims.last() != Some(&experts)
        || !rows.is_multiple_of(experts)
        || !t.file_bytes.is_multiple_of(experts)
    {
        return Err(PlacementError::tensor(
            t,
            format!(
                "dims {:?} and {} file bytes are not a stack of {experts} experts",
                t.dims, t.file_bytes
            ),
        ));
    }
    Ok((rows / experts, t.file_bytes / experts))
}

/// A routed stack of the layer whose card is `card` and keeps `n` experts:
/// `[0, n)` on the card, `[n, experts)` on the host, an empty side omitted.
fn place_routed(
    i: usize,
    t: &ModelTensor,
    card: usize,
    n: u64,
    model: &ModelTensors,
) -> Result<Row, PlacementError> {
    let (rows_per_expert, file_per_expert) = per_expert(t, model.experts)?;
    let mut segments = Vec::with_capacity(2);
    if n > 0 {
        segments.push(card_segment(t, card, n * rows_per_expert, Some(0..n))?);
    }
    if n < model.experts {
        let bytes = (model.experts - n) * file_per_expert;
        segments.push(in_place(
            Device::Host,
            Format::HostFile,
            Some(n..model.experts),
            bytes,
        ));
    }
    Ok(Row {
        tensor: i,
        segments,
        read_bytes: file_per_expert * model.experts_used,
        stage: card,
    })
}

/// A card's eligible layers: the layers of `card` that route, every stack of
/// which has a card format.
fn eligible(card: &Card, routed: &[Vec<usize>], model: &ModelTensors) -> Vec<usize> {
    card.layers
        .clone()
        .filter(|&l| {
            !routed[l].is_empty()
                && routed[l]
                    .iter()
                    .all(|&i| CardFormat::of(model.tensors[i].ty).is_some())
        })
        .collect()
}

/// The expert rule on one card, in the order of
/// `docs/research/v41-placement/spread.py`'s `plan_spread`: one more expert at
/// a time on the eligible layers in ascending order, cycling, while the card
/// still fits `budget` with it and its layer is below the expert count —
/// stopping at the first that does not. What must fit is `footprint` of the
/// card's uploads at those counts, the allocator's rounding included, so an
/// expert costs what its rows add to the card's granules, not its bytes. A
/// card whose uploads pass `budget` with no experts plans none.
fn spread(
    n_l: &mut [u64],
    eligible: &[usize],
    experts: u64,
    budget: i128,
    footprint: impl Fn(&[u64]) -> Result<u64, PlacementError>,
) -> Result<(), PlacementError> {
    if eligible.is_empty() || i128::from(footprint(n_l)?) > budget {
        return Ok(());
    }
    for &l in eligible.iter().cycle() {
        if n_l[l] >= experts {
            break;
        }
        n_l[l] += 1;
        if i128::from(footprint(n_l)?) > budget {
            n_l[l] -= 1;
            break;
        }
    }
    Ok(())
}

/// The rows one card upload holds: a whole tensor's, or a routed stack's
/// leading experts, as many as its layer keeps.
#[derive(Clone, Copy)]
enum Held {
    Rows(u64),
    Experts { layer: usize, rows_per_expert: u64 },
}

/// Card `c`'s uploads in the loader's order — the model's tensor order, which
/// `Weights::load_placed` walks: each whole tensor `rows` puts on the card,
/// and each routed stack of the card's `eligible` layers.
fn card_uploads<'m>(
    model: &'m ModelTensors,
    c: usize,
    rows: &[Option<Row>],
    eligible: &[usize],
) -> Result<Vec<(&'m ModelTensor, CardFormat, Held)>, PlacementError> {
    let mut out = Vec::new();
    for (t, row) in model.tensors.iter().zip(rows) {
        let (format, held) = match row {
            Some(r) => match r.segments.as_slice() {
                [s] if s.device == Device::Card(c) => match s.format {
                    Format::Card(f) => (f, Held::Rows(rows_of(t))),
                    _ => continue,
                },
                _ => continue,
            },
            None => {
                let layer = layer_of(t, model.layers)?;
                let Some(f) = CardFormat::of(t.ty).filter(|_| eligible.contains(&layer)) else {
                    continue;
                };
                let (rows_per_expert, _) = per_expert(t, model.experts)?;
                (
                    f,
                    Held::Experts {
                        layer,
                        rows_per_expert,
                    },
                )
            }
        };
        out.push((t, format, held));
    }
    Ok(out)
}

/// The bytes of the granules `uploads` take with `n_l` experts on each layer:
/// every upload's buffers, in order, through the card's allocator ([`Heap`]).
fn footprint(
    granule: NonZeroU64,
    uploads: &[(&ModelTensor, CardFormat, Held)],
    n_l: &[u64],
) -> Result<u64, PlacementError> {
    let mut heap = Heap::new(granule);
    for &(t, format, held) in uploads {
        let rows = match held {
            Held::Rows(rows) => rows,
            Held::Experts {
                layer,
                rows_per_expert,
            } => n_l[layer] * rows_per_expert,
        };
        if rows > 0 {
            for b in card_buffers(t, format, rows)? {
                heap.alloc(b);
            }
        }
    }
    Ok(heap.taken)
}

/// Place every tensor of `model` on `machine` for a context of `ctx_max`
/// tokens. Dense tensors go where their role says. Routed stacks follow the
/// expert rule per card ([`spread`]): the granules the card's uploads take —
/// dense tensors and expert prefixes, in upload order, through the card's
/// allocator — within usable − KV − context − scratch − margin. A card whose
/// layers cannot keep one expert keeps none; one that cannot hold even its
/// dense tensors shows up in [`Plan::violations`], not as an error here.
pub fn plan<'a>(
    model: &'a ModelTensors,
    machine: &'a Machine,
    ctx_max: u64,
    kv: &dyn KvBytes,
) -> Result<Plan<'a>, PlacementError> {
    let stages = Stages::new(model.layers, &machine.cards)?;
    let mut rows: Vec<Option<Row>> = vec![None; model.tensors.len()];
    let mut routed: Vec<Vec<usize>> = vec![Vec::new(); model.layers];
    for (i, t) in model.tensors.iter().enumerate() {
        if let Some(format) = CardFormat::of(t.ty) {
            card_bytes(t, format, rows_of(t))?;
        }
        if t.role == Role::RoutedExperts {
            routed[layer_of(t, model.layers)?].push(i);
        } else {
            rows[i] = Some(place_whole(i, t, &stages, model.layers)?);
        }
    }
    let mut n_l = vec![0u64; model.layers];
    let mut kv_bytes = Vec::with_capacity(machine.cards.len());
    for (c, card) in machine.cards.iter().enumerate() {
        let kv_card: u64 = card
            .layers
            .clone()
            .map(|l| kv.layer_bytes(l, ctx_max))
            .sum();
        let budget = i128::from(card.usable_bytes)
            - i128::from(kv_card)
            - i128::from(card.context_bytes)
            - i128::from(card.scratch_bytes)
            - i128::from(card.margin_bytes);
        let eligible = eligible(card, &routed, model);
        let uploads = card_uploads(model, c, &rows, &eligible)?;
        spread(&mut n_l, &eligible, model.experts, budget, |n| {
            footprint(card.granule_bytes, &uploads, n)
        })?;
        kv_bytes.push(kv_card);
    }
    for (l, stacks) in routed.iter().enumerate() {
        for &i in stacks {
            let row = place_routed(i, &model.tensors[i], stages.of_layer[l], n_l[l], model)?;
            rows[i] = Some(row);
        }
    }
    let rows: Vec<Row> = rows.into_iter().flatten().collect();
    let routing: Vec<bool> = routed.iter().map(|s| !s.is_empty()).collect();
    Ok(totals(
        model, machine, ctx_max, rows, &kv_bytes, n_l, &routing,
    ))
}

/// The per-device sums of a finished set of rows.
fn totals<'a>(
    model: &'a ModelTensors,
    machine: &'a Machine,
    ctx_max: u64,
    rows: Vec<Row>,
    kv_bytes: &[u64],
    n_l: Vec<u64>,
    routing: &[bool],
) -> Plan<'a> {
    let mut card_dense = vec![0u64; machine.cards.len()];
    let mut card_experts = vec![0u64; machine.cards.len()];
    let (mut host_experts, mut host_tables, mut nvme_bytes) = (0u64, 0u64, 0u64);
    for r in &rows {
        let is_routed = model.tensors[r.tensor].role == Role::RoutedExperts;
        for s in &r.segments {
            match (s.device, is_routed) {
                (Device::Card(c), false) => card_dense[c] += s.resident_bytes,
                (Device::Card(c), true) => card_experts[c] += s.resident_bytes,
                (Device::Host, true) => host_experts += s.resident_bytes,
                (Device::Host, false) => host_tables += s.resident_bytes,
                (Device::Nvme, _) => nvme_bytes += s.resident_bytes,
                (Device::Unused, _) => {}
            }
        }
    }
    // Rows from `card_segment` carry their buffers' sum, so no card segment
    // here is one `violations` would report.
    let (heaps, _) = card_heaps(model, &machine.cards, &rows);
    let cards = machine
        .cards
        .iter()
        .zip(&heaps)
        .enumerate()
        .map(|(c, (card, heap))| {
            let rounding = heap.taken - (card_dense[c] + card_experts[c]);
            let used = heap.taken + kv_bytes[c] + card.scratch_bytes + card.context_bytes;
            CardTotals {
                dense_bytes: card_dense[c],
                expert_bytes: card_experts[c],
                rounding_bytes: rounding,
                experts: card.layers.clone().map(|l| n_l[l]).sum(),
                kv_bytes: kv_bytes[c],
                scratch_bytes: card.scratch_bytes,
                context_bytes: card.context_bytes,
                headroom_bytes: i128::from(card.usable_bytes) - i128::from(used),
            }
        })
        .collect();
    let reserve_bytes: u64 = machine.host.reserves.iter().map(|(_, b)| b).sum();
    let host = HostTotals {
        expert_bytes: host_experts,
        experts: (0..model.layers)
            .filter(|&l| routing[l])
            .map(|l| model.experts - n_l[l])
            .sum(),
        table_bytes: host_tables,
        reserve_bytes,
        headroom_bytes: i128::from(machine.host.usable_bytes)
            - i128::from(host_experts + host_tables + reserve_bytes),
    };
    Plan {
        model,
        machine,
        ctx_max,
        rows,
        cards,
        host,
        nvme_bytes,
        n_l,
    }
}

impl Plan<'_> {
    /// Every invariant the rows break, re-derived from the rows themselves:
    /// each tensor placed once; an expert stack's segments cover every expert
    /// once and a whole tensor is one segment; a card segment's buffers sum to
    /// its resident bytes; a card holds only what its stage uses; the granules
    /// of its uploads + KV + scratch + context ≤ usable − margin on each card;
    /// the host's tensors and reserves ≤ its usable bytes.
    pub fn violations(&self) -> Vec<Violation> {
        let (model, cards) = (self.model, &self.machine.cards);
        let mut out = Vec::new();
        let mut times = vec![0usize; model.tensors.len()];
        let mut host_resident = 0u64;
        for r in &self.rows {
            let Some(t) = model.tensors.get(r.tensor) else {
                continue;
            };
            times[r.tensor] += 1;
            if let Some(detail) = segment_shape(t, &r.segments, model.experts) {
                out.push(Violation::Segments {
                    tensor: t.name.clone(),
                    detail,
                });
            }
            for s in &r.segments {
                match s.device {
                    Device::Card(c) => {
                        let uses = cards.get(c).is_some_and(|card| match (t.role, t.layer) {
                            (Role::Head, _) => card.head,
                            (_, Some(l)) => card.layers.contains(&l),
                            (_, None) => false,
                        });
                        if !uses {
                            out.push(Violation::WrongCard {
                                tensor: t.name.clone(),
                                card: cards.get(c).map_or(format!("#{c}"), |k| k.name.clone()),
                            });
                        }
                    }
                    Device::Host => host_resident += s.resident_bytes,
                    Device::Nvme | Device::Unused => {}
                }
            }
        }
        for (t, &n) in model.tensors.iter().zip(&times) {
            if n != 1 {
                out.push(Violation::PlacedTimes {
                    tensor: t.name.clone(),
                    times: n,
                });
            }
        }
        let (heaps, unsized_segments) = card_heaps(model, cards, &self.rows);
        out.extend(unsized_segments);
        for ((card, totals), heap) in cards.iter().zip(&self.cards).zip(&heaps) {
            let total = heap.taken + totals.kv_bytes + card.scratch_bytes + card.context_bytes;
            let limit = card.usable_bytes.saturating_sub(card.margin_bytes);
            if total > limit {
                out.push(Violation::CardOver {
                    card: card.name.clone(),
                    total,
                    limit,
                });
            }
        }
        let host = &self.machine.host;
        let total = host_resident + host.reserves.iter().map(|(_, b)| b).sum::<u64>();
        if total > host.usable_bytes {
            out.push(Violation::HostOver {
                total,
                usable: host.usable_bytes,
            });
        }
        out
    }
}

/// What is wrong with a tensor's segments, if anything: an expert stack must
/// cover `0..experts` with non-empty, non-overlapping ranges; anything else is
/// one segment with no range.
fn segment_shape(t: &ModelTensor, segments: &[Segment], experts: u64) -> Option<String> {
    if t.role != Role::RoutedExperts {
        return match segments {
            [s] if s.experts.is_none() => None,
            _ => Some(format!("a whole tensor in {} segments", segments.len())),
        };
    }
    let mut ranges = Vec::with_capacity(segments.len());
    for s in segments {
        match &s.experts {
            Some(r) if !r.is_empty() => ranges.push(r.clone()),
            other => return Some(format!("an expert segment with range {other:?}")),
        }
    }
    ranges.sort_by_key(|r| r.start);
    let mut next = 0;
    for r in &ranges {
        if r.start > next {
            return Some(format!(
                "segments {ranges:?} put experts {next}..{} nowhere",
                r.start
            ));
        }
        if r.start < next {
            return Some(format!(
                "segments {ranges:?} put experts {}..{next} twice",
                r.start
            ));
        }
        next = r.end;
    }
    (next != experts).then(|| format!("segments {ranges:?} end at {next}, not at {experts}"))
}
