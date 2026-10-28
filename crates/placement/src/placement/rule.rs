use std::cell::RefCell;
use std::num::NonZeroU64;
use std::ops::Range;

use gguf::GgmlType;

use super::machine::{
    Card, Device, ExpertList, Format, Heap, Machine, ModelTensor, ModelTensors, allocator_bytes,
};
use super::refusal::{PlacementError, Violation};
use super::table::{Row, Segment};
use super::{CardFormat, Role};

/// Which card runs each layer, which ends with the head, and which holds
/// the token embedding when one does.
pub(super) struct Stages {
    pub(super) of_layer: Vec<usize>,
    head: usize,
    pub(super) embedding: Option<usize>,
}

impl Stages {
    pub(super) fn new(layers: usize, cards: &[Card]) -> Result<Stages, PlacementError> {
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
        let embeds: Vec<usize> = (0..cards.len())
            .filter(|&c| cards[c].token_embedding)
            .collect();
        let embedding = match embeds.as_slice() {
            [] => None,
            &[c] if of_layer.first() == Some(&c) => Some(c),
            &[c] => {
                return Err(PlacementError::Stages(format!(
                    "card {} holds the token embedding but does not run layer 0",
                    cards[c].name
                )));
            }
            many => {
                return Err(PlacementError::Stages(format!(
                    "{} cards hold the token embedding, not one",
                    many.len()
                )));
            }
        };
        Ok(Stages {
            of_layer,
            head,
            embedding,
        })
    }

    /// The card whose stage uses `t`: its layer's, the head's for the head,
    /// the first stage's for a model-level tensor and for one never loaded
    /// ([`Role::Unused`], whose layer need not be one the model runs).
    fn stage_of(&self, t: &ModelTensor, layers: usize) -> Result<usize, PlacementError> {
        match (t.role, t.layer) {
            (Role::Head, _) => Ok(self.head),
            (Role::Unused, _) | (_, None) => Ok(self.of_layer[0]),
            (_, Some(_)) => Ok(self.of_layer[layer_of(t, layers)?]),
        }
    }
}

/// Rows as the loader counts them: the product of every dim past the row width.
pub(super) fn rows_of(t: &ModelTensor) -> u64 {
    t.dims.iter().skip(1).product()
}

/// Bytes of one row in the file's format.
pub(super) fn row_bytes(t: &ModelTensor) -> Result<u64, PlacementError> {
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
pub(super) fn layer_of(t: &ModelTensor, layers: usize) -> Result<usize, PlacementError> {
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
pub(super) fn card_bytes(
    t: &ModelTensor,
    format: CardFormat,
    rows: u64,
) -> Result<u64, PlacementError> {
    card_buffers(t, format, rows)?
        .into_iter()
        .try_fold(0u64, u64::checked_add)
        .ok_or_else(|| PlacementError::tensor(t, format!("{rows} rows pass u64 bytes")))
}

/// The device buffers of `rows` rows of `t` in `format`, in upload order —
/// [`CardFormat::buffer_bytes`], and for an nvfp4 tensor one f32 scale an
/// expert the rows hold (a stack, three dims) or its one (any other), the
/// scales its upload gathers from its `.scale` — or the refusal
/// [`card_bytes`] gives.
pub(super) fn card_buffers(
    t: &ModelTensor,
    format: CardFormat,
    rows: u64,
) -> Result<Vec<u64>, PlacementError> {
    let k = t.dims.first().copied().unwrap_or(0);
    let mut bufs = format
        .buffer_bytes(t.ty, k, rows)
        .ok_or_else(|| refused(t, format, k, rows))?;
    match (format, t.dims.as_slice()) {
        (CardFormat::Nvfp4Planes, &[_, per, _]) => {
            if per == 0 || !rows.is_multiple_of(per) {
                return Err(refused(t, format, k, rows));
            }
            bufs.push(rows / per * 4);
        }
        (CardFormat::Nvfp4Planes, _) => bufs.push(4),
        _ => {}
    }
    Ok(bufs)
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
pub(super) fn card_heaps(
    model: &ModelTensors,
    machine: &Machine,
    rows: &[Row],
) -> (Vec<Heap>, Vec<Violation>) {
    let mut heaps: Vec<Heap> = machine
        .all_cards()
        .map(|c| Heap::new(c.granule_bytes))
        .collect();
    let mut bad = Vec::new();
    for r in rows {
        let Some(t) = model.tensors.get(r.tensor) else {
            continue;
        };
        for s in &r.segments {
            let Device::Card(c) = s.device else {
                continue;
            };
            if c >= machine.cards.len()
                && c < heaps.len()
                && t.role == Role::RoutedExperts
                && s.format == Format::HostFile
                && s.resident_bytes == 0
            {
                continue;
            }
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

/// A card segment of `rows` rows of `t` — all of it, or an expert slice —
/// in the format `model` gives it ([`ModelTensors::card_format`]).
fn card_segment(
    model: &ModelTensors,
    t: &ModelTensor,
    card: usize,
    rows: u64,
    experts: Option<ExpertList>,
) -> Result<Segment, PlacementError> {
    card_segment_as(t, card, rows, experts, model.card_format(t))
}

/// [`card_segment`] in `format`; `None` is the refusal of a tensor no card
/// format loads.
pub(super) fn card_segment_as(
    t: &ModelTensor,
    card: usize,
    rows: u64,
    experts: Option<ExpertList>,
    format: Option<CardFormat>,
) -> Result<Segment, PlacementError> {
    let Some(format) = format else {
        return Err(PlacementError::NoCardFormat {
            name: t.name.clone(),
            ty: t.ty,
            role: t.role,
        });
    };
    Ok(Segment {
        device: Device::Card(card),
        format: Format::Card(format),
        experts,
        resident_bytes: card_bytes(t, format, rows)?,
    })
}

/// A segment that holds a tensor's file bytes where they are.
pub(super) fn in_place(
    device: Device,
    format: Format,
    experts: Option<ExpertList>,
    bytes: u64,
) -> Segment {
    Segment {
        device,
        format,
        experts,
        resident_bytes: bytes,
    }
}

/// Every tensor but a routed stack: one segment, where its role says.
pub(super) fn place_whole(
    i: usize,
    t: &ModelTensor,
    stages: &Stages,
    model: &ModelTensors,
) -> Result<Row, PlacementError> {
    let layers = model.layers;
    let stage = stages.stage_of(t, layers)?;
    let (segment, read_bytes) = match t.role {
        Role::Attention
        | Role::HyperConnection
        | Role::Router
        | Role::FfnNorm
        | Role::SharedExpert
        | Role::DenseFfn
        | Role::EngramDense
        | Role::EngramGain => {
            layer_of(t, layers)?;
            (
                card_segment(model, t, stage, rows_of(t), None)?,
                t.file_bytes,
            )
        }
        Role::Head => (
            card_segment(model, t, stage, rows_of(t), None)?,
            t.file_bytes,
        ),
        Role::TokenEmbedding => match stages.embedding {
            Some(card) => (
                card_segment(model, t, card, rows_of(t), None)?,
                gathered(t)?,
            ),
            None => (
                in_place(Device::Host, Format::HostFile, None, t.file_bytes),
                gathered(t)?,
            ),
        },
        Role::EngramTable => {
            layer_of(t, layers)?;
            (
                in_place(Device::Nvme, Format::NvmeFile, None, t.file_bytes),
                gathered(t)?,
            )
        }
        Role::HashTable => {
            layer_of(t, layers)?;
            (
                in_place(Device::Host, Format::HostFile, None, t.file_bytes),
                gathered(t)?,
            )
        }
        Role::Unread | Role::Unused | Role::EngramHash => {
            (in_place(Device::Unused, Format::Unused, None, 0), 0)
        }
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
pub fn per_expert(t: &ModelTensor, experts: u64) -> Result<(u64, u64), PlacementError> {
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

/// Row `i`, the routed stack `t` of a layer whose card is `card`: the experts
/// `on_card` on the card, the rest on the host in the file, an empty side
/// omitted ([`routed_row_on`] with the one card).
pub fn routed_row(
    i: usize,
    t: &ModelTensor,
    card: usize,
    on_card: ExpertList,
    model: &ModelTensors,
) -> Result<Row, PlacementError> {
    routed_row_on(i, t, card, vec![(card, on_card)], model)
}

/// Row `i`, the routed stack `t` of a layer whose stage card is `stage`: each
/// list of `on_cards` on its card (by its index in [`Machine::all_cards`]),
/// in that order, the rest on the host in the file, an empty list or host
/// side omitted. An expert on two of the lists is refused by name
/// ([`PlacementError::ExpertOnTwoCards`]). The one owner of a stack's split
/// between the cards and the host: the expert rule places by it, and so does
/// a load that fixes its own split.
pub fn routed_row_on(
    i: usize,
    t: &ModelTensor,
    stage: usize,
    on_cards: Vec<(usize, ExpertList)>,
    model: &ModelTensors,
) -> Result<Row, PlacementError> {
    routed_row_on_as(i, t, stage, on_cards, model, model.card_format(t))
}

/// [`routed_row_on`] with the card segments in `format`: the expert rule's
/// rows, in the format the plan's [`RoutedRule`] counted them in. `None`
/// is refused only when a list of `on_cards` holds an expert.
pub(super) fn routed_row_on_as(
    i: usize,
    t: &ModelTensor,
    stage: usize,
    on_cards: Vec<(usize, ExpertList)>,
    model: &ModelTensors,
    format: Option<CardFormat>,
) -> Result<Row, PlacementError> {
    let experts = model.experts_of(t);
    let (rows_per_expert, file_per_expert) = per_expert(t, experts)?;
    let mut owner: Vec<(u32, usize)> = on_cards
        .iter()
        .flat_map(|(c, list)| list.ids().iter().map(move |&e| (e, *c)))
        .collect();
    owner.sort_unstable();
    if let Some(w) = owner.windows(2).find(|w| w[0].0 == w[1].0) {
        return Err(PlacementError::ExpertOnTwoCards {
            tensor: t.name.clone(),
            expert: w[0].0,
            first: w[0].1.min(w[1].1),
            second: w[0].1.max(w[1].1),
        });
    }
    let held = ExpertList::new(owner.iter().map(|&(e, _)| e).collect(), experts)?;
    let host = held.complement(experts)?;
    let mut segments = Vec::with_capacity(on_cards.len() + 1);
    for (card, list) in on_cards {
        if !list.is_empty() {
            let rows = list.len() * rows_per_expert;
            segments.push(card_segment_as(t, card, rows, Some(list), format)?);
        }
    }
    if !host.is_empty() {
        // The host dots an nvfp4 stack's file blocks with its scales; a card
        // holds them as planes, and a residency swap copies file bytes into a
        // card slot, so one stack is not split between the two.
        if t.ty == GgmlType::NVFP4 && !segments.is_empty() {
            return Err(PlacementError::tensor(
                t,
                format!(
                    "an nvfp4 stack with experts {host} on the host and the rest on a card, \
                     which holds them as planes where a residency swap copies the file's \
                     blocks: every expert of it must be on a card, or every one on the host"
                ),
            ));
        }
        let bytes = host.len() * file_per_expert;
        segments.push(in_place(Device::Host, Format::HostFile, Some(host), bytes));
    }
    Ok(Row {
        tensor: i,
        segments,
        read_bytes: file_per_expert * model.experts_used,
        stage,
    })
}

/// Row `i`, the tensor `t` of `model` whole on card `card` in its card
/// format, whatever its role — for a load that fixes its own split
/// ([`routed_row`] places the routed stacks).
pub fn whole_on_card(
    i: usize,
    t: &ModelTensor,
    card: usize,
    model: &ModelTensors,
) -> Result<Row, PlacementError> {
    Ok(Row {
        tensor: i,
        segments: vec![card_segment(model, t, card, rows_of(t), None)?],
        read_bytes: t.file_bytes,
        stage: card,
    })
}

/// The card format a routed stack of type `ty` loads in under one card
/// expert kernel, `None` for a type it does not read.
pub type RoutedFormat = fn(GgmlType) -> Option<CardFormat>;

/// The part of a layer's routed experts a stack holds, as its name gives it
/// (`….ffn_{gate,up,down}_exps.weight`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpertPart {
    Gate,
    Up,
    Down,
}

impl ExpertPart {
    /// The part `name` names; `None` for any other name, a fused gate·up
    /// stack's among them.
    #[must_use]
    pub fn of_name(name: &str) -> Option<Self> {
        match name.strip_suffix("_exps.weight")?.rsplit_once("ffn_")?.1 {
            "gate" => Some(Self::Gate),
            "up" => Some(Self::Up),
            "down" => Some(Self::Down),
            _ => None,
        }
    }
}

/// A routed tensor that keeps its layer on the host under a card rule.
#[derive(Clone, Copy, Debug)]
pub struct HostOnly<'a> {
    pub layer: usize,
    pub tensor: &'a ModelTensor,
    pub why: &'static str,
}

/// Which routed stacks a program's card expert kernels run, and the card
/// format each loads in. A layer with a stack the rule gives no format keeps
/// its experts on the host.
#[derive(Clone, Copy, Debug)]
pub enum RoutedRule {
    /// Every stack by its type alone, whatever part it holds.
    PerType(RoutedFormat),
    /// A card leg of a fused gate·up kernel and a down kernel: a gate and an
    /// up of one type in `gate_up`'s format, a down in `down`'s. A layer whose
    /// gate and up differ in type has no card format; a routed stack whose
    /// name names no part is refused by name. With a `pair`, a layer also
    /// runs only where `pair(gate and up type, down type, on a tier)` holds: a
    /// leg whose launches read the down's type by the gate·up's, or whose
    /// expert tier runs fewer layers than its stage card.
    Leg {
        gate_up: RoutedFormat,
        down: RoutedFormat,
        pair: Option<LegPair>,
    },
}

/// A card leg's layer rule: whether a layer whose gate and up are of the
/// first type and whose down is of the second runs on a stage card, or, for
/// `true`, on an expert tier.
pub type LegPair = fn(GgmlType, GgmlType, bool) -> bool;

impl RoutedRule {
    /// The card format stack `t` loads in, `None` where the rule reads none.
    pub(super) fn format(self, t: &ModelTensor) -> Result<Option<CardFormat>, PlacementError> {
        match self {
            Self::PerType(f) => Ok(f(t.ty)),
            Self::Leg { gate_up, down, .. } => match ExpertPart::of_name(&t.name) {
                Some(ExpertPart::Gate | ExpertPart::Up) => Ok(gate_up(t.ty)),
                Some(ExpertPart::Down) => Ok(down(t.ty)),
                None => Err(PlacementError::tensor(
                    t,
                    "a routed stack whose name names no gate, up or down part, which a card \
                     leg's rule places by part",
                )),
            },
        }
    }

    fn refusal(
        self,
        tensor: &ModelTensor,
        gate_up_ty: &mut Option<GgmlType>,
    ) -> Result<Option<&'static str>, PlacementError> {
        let unsupported = self.format(tensor)?.is_none();
        let mismatched = matches!(self, Self::Leg { .. })
            && matches!(
                ExpertPart::of_name(&tensor.name),
                Some(ExpertPart::Gate | ExpertPart::Up)
            )
            && *gate_up_ty.get_or_insert(tensor.ty) != tensor.ty;
        Ok(if unsupported {
            Some("no card expert kernel for this routed part")
        } else if mismatched {
            Some("the routed gate and up have different types")
        } else {
            None
        })
    }

    /// The tensors that disqualify their layers under this rule, in file
    /// order, independent of the card's byte budget. Uses the same refusal
    /// as the planner's layer eligibility.
    pub fn host_only(self, model: &ModelTensors) -> Result<Vec<HostOnly<'_>>, PlacementError> {
        let mut gate_up_types = vec![None; model.layers];
        let mut by_layer: Vec<Vec<&ModelTensor>> = vec![Vec::new(); model.layers];
        let mut held = vec![false; model.layers];
        let mut out = Vec::new();
        for tensor in model
            .tensors
            .iter()
            .filter(|t| t.role == Role::RoutedExperts)
        {
            let layer = layer_of(tensor, model.layers)?;
            by_layer[layer].push(tensor);
            if let Some(why) = self.refusal(tensor, &mut gate_up_types[layer])? {
                held[layer] = true;
                out.push(HostOnly { layer, tensor, why });
            }
        }
        // The pair rule reads whole layers: the down of a layer the parts
        // above leave on the card, whose type does not pair with its gate
        // and up's, keeps the layer on the host.
        for (layer, stacks) in by_layer.iter().enumerate() {
            if held[layer] {
                continue;
            }
            if let Some(down) = self.unpaired_down(stacks, false) {
                out.push(HostOnly {
                    layer,
                    tensor: down,
                    why: "the routed down's type does not pair with the gate and up's on this \
                          card leg",
                });
            }
        }
        Ok(out)
    }

    /// The down stack among the layer's `stacks` when the rule's pair refuses
    /// the layer's (gate and up type, down type), `None` where there is no
    /// pair rule, a part is missing or the pair holds.
    fn unpaired_down<'m>(self, stacks: &[&'m ModelTensor], tier: bool) -> Option<&'m ModelTensor> {
        let Self::Leg {
            pair: Some(pair), ..
        } = self
        else {
            return None;
        };
        let part = |p: ExpertPart| {
            stacks
                .iter()
                .copied()
                .find(|t| ExpertPart::of_name(&t.name) == Some(p))
        };
        let (gate, down) = (part(ExpertPart::Gate)?, part(ExpertPart::Down)?);
        (!pair(gate.ty, down.ty, tier)).then_some(down)
    }

    /// Whether the card runs a layer of the routed stacks `stacks`: each
    /// loads in a card format, on a leg its gate and up are one type and the
    /// leg's pair rule, where it has one, holds for the layer.
    pub(crate) fn layer_runs(
        self,
        stacks: &[usize],
        model: &ModelTensors,
        tier: bool,
    ) -> Result<bool, PlacementError> {
        let mut runs = true;
        let mut gate_up_ty = None;
        for &i in stacks {
            runs &= self.refusal(&model.tensors[i], &mut gate_up_ty)?.is_none();
        }
        if runs {
            let layer: Vec<&ModelTensor> = stacks.iter().map(|&i| &model.tensors[i]).collect();
            runs = self.unpaired_down(&layer, tier).is_none();
        }
        Ok(runs)
    }
}

/// The eligible layers of `layers`: those that route, whose stacks the card
/// runs under `routed` ([`RoutedRule::layer_runs`]) — a stage card's own
/// layers, or every layer for a tier.
pub(super) fn eligible(
    layers: Range<usize>,
    stacks: &[Vec<usize>],
    model: &ModelTensors,
    routed: RoutedRule,
    tier: bool,
) -> Result<Vec<usize>, PlacementError> {
    let mut out = Vec::new();
    for l in layers {
        if !stacks[l].is_empty() && routed.layer_runs(&stacks[l], model, tier)? {
            out.push(l);
        }
    }
    Ok(out)
}

/// The expert rule on one card: one more expert at a time on the eligible
/// layer whose total — `held` on other devices plus `n_l` here — is the
/// lowest, the first in ascending order on a tie, while the card still fits
/// `budget` with it and that total is below the expert count — stopping at
/// the first that does not. With nothing held elsewhere this is the order of
/// `docs/research/v41-placement/spread.py`'s `plan_spread`: the eligible
/// layers in ascending order, cycling. What must fit is `footprint` of the
/// card's uploads at those counts, the allocator's rounding included, so an
/// expert costs what its rows add to the card's granules, not its bytes. A
/// card whose uploads pass `budget` with no experts plans none.
pub(super) fn spread(
    n_l: &mut [u64],
    held: &[u64],
    eligible: &[usize],
    caps: &[u64],
    budget: i128,
    footprint: impl Fn(&[u64]) -> Result<u64, PlacementError>,
) -> Result<(), PlacementError> {
    if eligible.is_empty() || i128::from(footprint(n_l)?) > budget {
        return Ok(());
    }
    let total = |n_l: &[u64], l: usize| held[l] + n_l[l];
    while let Some(&l) = eligible
        .iter()
        .filter(|&&l| total(n_l, l) < caps[l])
        .min_by_key(|&&l| total(n_l, l))
    {
        n_l[l] += 1;
        if i128::from(footprint(n_l)?) > budget {
            n_l[l] -= 1;
            break;
        }
    }
    Ok(())
}

/// The rows one card upload holds: a whole tensor's, or a routed stack's
/// experts, as many as its layer keeps.
#[derive(Clone, Copy)]
pub(super) enum Held {
    Rows(u64),
    Experts { layer: usize, rows_per_expert: u64 },
}

/// Card `c`'s uploads in the loader's order — the model's tensor order, which
/// `Weights::load_placed` walks: each whole tensor `rows` puts on the card,
/// and each routed stack of the card's `eligible` layers in its `routed`
/// format.
pub(super) fn card_uploads<'m>(
    model: &'m ModelTensors,
    c: usize,
    rows: &[Option<Row>],
    eligible: &[usize],
    routed: RoutedRule,
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
                if !eligible.contains(&layer) {
                    continue;
                }
                let Some(f) = routed.format(t)? else {
                    continue;
                };
                let (rows_per_expert, _) = per_expert(t, model.experts_of(t))?;
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

/// The bytes of the granules a card's uploads take with `n_l` experts on each layer: every
/// upload's buffers, in order, through the card's allocator ([`Heap`]). The expert rule asks it
/// once an expert ([`spread`]), and each ask changes one layer's count, so each step of the
/// uploads keeps what it worked out last: its buffers at its last rows, and what its small
/// buffers did to the shared granules from the granules it found. A buffer of a granule or more
/// takes whole granules of its own wherever it falls in the order, so only the small buffers
/// depend on the granules a step finds; a step that finds them as it last found them, with the
/// same small buffers, leaves them as it last left them.
pub(super) struct Footprint<'m> {
    granule: NonZeroU64,
    steps: Vec<Step<'m>>,
}

struct Step<'m> {
    what: What<'m>,
    last: RefCell<Last>,
}

/// One step of [`Footprint`]'s uploads, in order.
enum What<'m> {
    /// Fixed uploads' buffers.
    Fixed,
    /// An upload whose buffers follow its layer's count: a routed stack, or one whose buffers
    /// are refused, which each count refuses in its place.
    Upload(&'m ModelTensor, CardFormat, Held),
}

/// What a step worked out last.
#[derive(Default)]
struct Last {
    /// The rows its buffers are of; `None` before the first.
    rows: Option<u64>,
    /// The bytes of its buffers of a granule or more, rounded to whole granules.
    large: u64,
    /// Its smaller buffers, in order.
    small: Vec<u64>,
    /// The shared granules' bytes left as its small buffers found them, and what the buffers
    /// did there; `None` until they are allocated.
    effect: Option<(Vec<u64>, Effect)>,
}

/// What a run of small buffers did to the shared granules: the bytes left it set by index, the
/// granules it opened (their bytes left), and the bytes of those granules.
struct Effect {
    set: Vec<(usize, u64)>,
    opened: Vec<u64>,
    took: u64,
}

impl Last {
    fn of(granule: NonZeroU64, rows: u64, buffers: Vec<u64>) -> Last {
        let (large, small): (Vec<u64>, Vec<u64>) =
            buffers.into_iter().partition(|&b| b >= granule.get());
        Last {
            rows: Some(rows),
            large: allocator_bytes(granule, large),
            small,
            effect: None,
        }
    }
}

impl<'m> Footprint<'m> {
    pub(super) fn new(
        granule: NonZeroU64,
        uploads: &[(&'m ModelTensor, CardFormat, Held)],
    ) -> Footprint<'m> {
        let mut steps: Vec<Step<'m>> = Vec::new();
        for &(t, format, held) in uploads {
            let fixed = match held {
                Held::Rows(0) => continue,
                Held::Rows(rows) => card_buffers(t, format, rows).ok(),
                Held::Experts { .. } => None,
            };
            let Some(buffers) = fixed else {
                steps.push(Step {
                    what: What::Upload(t, format, held),
                    last: RefCell::default(),
                });
                continue;
            };
            let add = Last::of(granule, 0, buffers);
            if let Some(Step {
                what: What::Fixed,
                last,
            }) = steps.last_mut()
            {
                let last = last.get_mut();
                last.large += add.large;
                last.small.extend(add.small);
            } else {
                steps.push(Step {
                    what: What::Fixed,
                    last: RefCell::new(add),
                });
            }
        }
        Footprint { granule, steps }
    }

    pub(super) fn at(&self, n_l: &[u64]) -> Result<u64, PlacementError> {
        let mut heap = Heap::new(self.granule);
        // Whether the shared granules are as the last ask found them at this step.
        let mut same = true;
        for step in &self.steps {
            let mut last = step.last.borrow_mut();
            if let What::Upload(t, format, held) = step.what {
                let rows = match held {
                    Held::Rows(rows) => rows,
                    Held::Experts {
                        layer,
                        rows_per_expert,
                    } => n_l[layer] * rows_per_expert,
                };
                if last.rows != Some(rows) {
                    let buffers = if rows > 0 {
                        card_buffers(t, format, rows)?
                    } else {
                        Vec::new()
                    };
                    let now = Last::of(self.granule, rows, buffers);
                    let effect = last.effect.take().filter(|_| now.small == last.small);
                    *last = Last { effect, ..now };
                }
            }
            let last = &mut *last;
            heap.taken += last.large;
            match &last.effect {
                Some((found, e)) if same || *found == heap.left => {
                    for &(i, left) in &e.set {
                        heap.left[i] = left;
                    }
                    heap.left.extend_from_slice(&e.opened);
                    heap.taken += e.took;
                    same = true;
                }
                _ => {
                    let (found, open, taken) = (heap.left.clone(), heap.left.len(), heap.taken);
                    for &b in &last.small {
                        heap.alloc(b);
                    }
                    let set = (0..open)
                        .filter(|&i| heap.left[i] != found[i])
                        .map(|i| (i, heap.left[i]))
                        .collect();
                    let effect = Effect {
                        set,
                        opened: heap.left[open..].to_vec(),
                        took: heap.taken - taken,
                    };
                    last.effect = Some((found, effect));
                    same = false;
                }
            }
        }
        Ok(heap.taken)
    }
}
