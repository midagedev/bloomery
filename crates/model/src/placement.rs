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
use std::ops::Range;

use gguf::GgmlType;

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
/// (`crates/gpu/src/weights.rs`), with the loader's resident arithmetic.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CardFormat {
    /// q3_K/q4_K/q6_K: the file's bytes as u32 words (weights.rs:270-313), so
    /// resident = file bytes; the upload needs `rows % 4 == 0` and row bytes `% 4 == 0`.
    KQuant,
    /// q8_0 in two planes: per 32-value block 8 code words and one f32 scale
    /// (weights.rs:42-51, `q8_0_planes` :379), `rows × (k + 4k/32)`.
    Q8_0Planes,
    /// f32: the file's values (weights.rs:330-351), `rows × k × 4`.
    F32,
    /// bf16 decoded to f32 at load, `rows × k × 4`.
    Bf16AsF32,
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
    /// Experts held, summed over the card's layers.
    pub experts: u64,
    pub kv_bytes: u64,
    pub scratch_bytes: u64,
    pub context_bytes: u64,
    /// usable − dense − experts − KV − scratch − context; the margin is inside it.
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
    /// every expert exactly once, or a whole tensor in pieces.
    Segments { tensor: String, detail: String },
    /// A segment on a card whose stage does not run the tensor's layer or head.
    WrongCard { tensor: String, card: String },
    /// A card whose resident bytes, KV, scratch and context pass usable − margin.
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
                "card {card}: resident + KV + scratch + context = {total} B passes usable − margin = {limit} B by {} B",
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

/// The card format a tensor of this type loads in, `Ok(None)` when the loader
/// has none (weights.rs:247-249, `resident_size`'s refusal arm).
fn card_format(t: &ModelTensor) -> Result<Option<CardFormat>, PlacementError> {
    match t.ty {
        GgmlType::Q3_K | GgmlType::Q4_K | GgmlType::Q6_K => Ok(Some(CardFormat::KQuant)),
        GgmlType::Q8_0 => Ok(Some(CardFormat::Q8_0Planes)),
        GgmlType::F32 => Ok(Some(CardFormat::F32)),
        GgmlType::BF16 => Ok(Some(CardFormat::Bf16AsF32)),
        GgmlType::Q5_K | GgmlType::F16 | GgmlType::Unknown(_) => Ok(None),
        GgmlType::Q5_0 | GgmlType::Q5_1 => Err(PlacementError::tensor(
            t,
            format!(
                "type {} has a loader format this table does not model",
                t.ty
            ),
        )),
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

/// A card segment of `rows` rows of `t` — all of it, or an expert slice.
fn card_segment(
    t: &ModelTensor,
    card: usize,
    rows: u64,
    experts: Option<Range<u64>>,
) -> Result<Segment, PlacementError> {
    let Some(format) = card_format(t)? else {
        return Err(PlacementError::tensor(
            t,
            format!(
                "type {} has no device format, but its role puts it on a card",
                t.ty
            ),
        ));
    };
    let k = t.dims.first().copied().unwrap_or(0);
    let rb = row_bytes(t)?;
    let resident_bytes = match format {
        CardFormat::KQuant => {
            kquant_words(t, rows)?;
            rb * rows
        }
        CardFormat::Q8_0Planes => rows * (k + 4 * (k / 32)),
        CardFormat::F32 | CardFormat::Bf16AsF32 => rows * k * 4,
    };
    Ok(Segment {
        device: Device::Card(card),
        format: Format::Card(format),
        experts,
        resident_bytes,
    })
}

/// The KQuant upload's condition on `rows` rows of `t`: rows and row bytes
/// both multiples of 4, so the words split evenly per row and resident = file
/// bytes. A tensor that breaks it is refused, not given another formula.
fn kquant_words(t: &ModelTensor, rows: u64) -> Result<(), PlacementError> {
    let rb = row_bytes(t)?;
    if !rows.is_multiple_of(4) || !rb.is_multiple_of(4) {
        return Err(PlacementError::tensor(
            t,
            format!("{rows} rows of {rb} B: the word upload needs both to be multiples of 4"),
        ));
    }
    Ok(())
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

/// A card's eligible layers with the file bytes of one expert: the layers of
/// `card` that route, every stack of which has a card format.
fn eligible(
    card: &Card,
    routed: &[Vec<usize>],
    model: &ModelTensors,
) -> Result<Vec<(usize, u64)>, PlacementError> {
    let mut out = Vec::new();
    for l in card.layers.clone() {
        let mut on_card = !routed[l].is_empty();
        let mut expert_bytes = 0;
        for &i in &routed[l] {
            let t = &model.tensors[i];
            on_card &= card_format(t)?.is_some();
            expert_bytes += per_expert(t, model.experts)?.1;
        }
        if on_card {
            out.push((l, expert_bytes));
        }
    }
    Ok(out)
}

/// The expert rule on one card (`docs/research/v41-placement/spread.py`,
/// `plan_spread`): the same count on every eligible layer, then one more at a
/// time in ascending layer order, cycling, while the next expert fits and its
/// layer is below the expert count — stopping at the first that does not.
/// `eligible` is (layer, bytes of one expert); a budget at or below zero plans
/// no experts.
fn spread(n_l: &mut [u64], eligible: &[(usize, u64)], budget: i128, experts: u64) {
    let round: i128 = eligible.iter().map(|&(_, b)| i128::from(b)).sum();
    if round == 0 || budget <= 0 {
        return;
    }
    let per = u64::try_from((budget / round).min(i128::from(experts)))
        .expect("a positive budget over a positive round, capped at the expert count, fits u64");
    for &(l, _) in eligible {
        n_l[l] = per;
    }
    let mut rem = budget - i128::from(per) * round;
    for &(l, b) in eligible.iter().cycle() {
        if n_l[l] >= experts || rem < i128::from(b) {
            break;
        }
        n_l[l] += 1;
        rem -= i128::from(b);
    }
}

/// Place every tensor of `model` on `machine` for a context of `ctx_max`
/// tokens. Dense tensors go where their role says. Routed stacks follow the
/// expert rule per card, over the budget usable − dense − KV − context −
/// scratch − margin, with one expert costing its layer's routed file bytes
/// over the expert count. A card whose layers cannot keep one expert keeps
/// none; one that cannot hold even its dense tensors shows up in
/// [`Plan::violations`], not as an error here.
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
        if card_format(t)? == Some(CardFormat::KQuant) {
            kquant_words(t, rows_of(t))?;
        }
        if t.role == Role::RoutedExperts {
            routed[layer_of(t, model.layers)?].push(i);
        } else {
            rows[i] = Some(place_whole(i, t, &stages, model.layers)?);
        }
    }
    let mut dense = vec![0u64; machine.cards.len()];
    for s in rows.iter().flatten().flat_map(|r| &r.segments) {
        if let Device::Card(c) = s.device {
            dense[c] += s.resident_bytes;
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
            - i128::from(dense[c])
            - i128::from(kv_card)
            - i128::from(card.context_bytes)
            - i128::from(card.scratch_bytes)
            - i128::from(card.margin_bytes);
        spread(
            &mut n_l,
            &eligible(card, &routed, model)?,
            budget,
            model.experts,
        );
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
    let cards = machine
        .cards
        .iter()
        .enumerate()
        .map(|(c, card)| {
            let used = card_dense[c]
                + card_experts[c]
                + kv_bytes[c]
                + card.scratch_bytes
                + card.context_bytes;
            CardTotals {
                dense_bytes: card_dense[c],
                expert_bytes: card_experts[c],
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
    /// once and a whole tensor is one segment; a card holds only what its
    /// stage uses; resident + KV + scratch + context ≤ usable − margin on each
    /// card; the host's tensors and reserves ≤ its usable bytes.
    pub fn violations(&self) -> Vec<Violation> {
        let (model, cards) = (self.model, &self.machine.cards);
        let mut out = Vec::new();
        let mut times = vec![0usize; model.tensors.len()];
        let mut card_resident = vec![0u64; cards.len()];
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
                        if let Some(sum) = card_resident.get_mut(c) {
                            *sum += s.resident_bytes;
                        }
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
        for ((card, totals), &resident) in cards.iter().zip(&self.cards).zip(&card_resident) {
            let total = resident + totals.kv_bytes + card.scratch_bytes + card.context_bytes;
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
