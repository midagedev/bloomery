use std::fmt;
use std::num::NonZeroU64;
use std::ops::Range;

use gguf::GgmlType;

use super::devices;
use super::refusal::PlacementError;
use super::rule::ExpertPart;
use super::workstation;
use super::{CardFormat, Role, Unimplemented};

/// `features` grouped by feature in first-seen order, each with its layers:
/// `hash routing (layers 0, 1, 2); …`.
pub(super) fn unimplemented_list(features: &[Unimplemented]) -> String {
    let mut groups: Vec<(&str, Vec<usize>)> = Vec::new();
    for f in features {
        let at = match groups.iter().position(|(g, _)| *g == f.feature) {
            Some(i) => i,
            None => {
                groups.push((&f.feature, Vec::new()));
                groups.len() - 1
            }
        };
        groups[at].1.extend(f.layer);
    }
    let parts: Vec<String> = groups
        .iter()
        .map(|(feature, layers)| match layers.as_slice() {
            [] => (*feature).to_string(),
            [l] => format!("{feature} (layer {l})"),
            ls => {
                let list: Vec<String> = ls.iter().map(ToString::to_string).collect();
                format!("{feature} (layers {})", list.join(", "))
            }
        })
        .collect();
    parts.join("; ")
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
    /// Experts in each routed stack — the largest layer's, on a model whose
    /// MoE layers carry their own counts ([`stack_experts`]) — and experts
    /// one token uses.
    pub experts: u64,
    pub experts_used: u64,
    /// Whether its program reads bf16 tensor `t` of a dense role or the
    /// router in its words ([`CardFormat::Bf16Raw`],
    /// [`ModelTensors::card_format`]): the architecture's classifier names
    /// each tensor a bf16 site, head, embedding lookup or router reads
    /// (qwen4exp's projections and router); a tensor of the same role its
    /// program reads as f32 values (a conv, an inject, the parts of a joined
    /// stack) is not one. `|_| false` loads every bf16 tensor as f32
    /// values ([`CardFormat::Bf16AsF32`]).
    pub bf16_words: fn(&ModelTensor) -> bool,
}

/// The experts of routed stack `t` of a model of `experts`: its last dim
/// where that is a count of at most `experts` (a model whose MoE layers carry
/// their own counts, `experts` the largest — Nemotron 3.5 Bolt's), else
/// `experts`, which [`per_expert`] then holds the stack to.
#[must_use]
pub fn stack_experts(t: &ModelTensor, experts: u64) -> u64 {
    match t.dims.as_slice() {
        [_, _, n] if t.role == Role::RoutedExperts && *n > 0 && *n <= experts => *n,
        _ => experts,
    }
}

impl ModelTensors {
    /// The format file tensor `t` of this model loads in on a card:
    /// [`CardFormat::of`] of its type, except
    /// - an engram gain, which the card reads as f32 values whatever its
    ///   type: a K-quant one is decoded at load;
    /// - a bf16 tensor [`ModelTensors::bf16_words`] names, of a role whose
    ///   tensors the bf16 dense site, head, embedding lookup and router read
    ///   (attention, a hyper-connection, a shared expert with its gate, a
    ///   dense FFN, an engram's dense projections, the head, the token
    ///   embedding, the router): [`CardFormat::Bf16Raw`], which those kernels
    ///   widen as they read it (`crate::kernels`' `SiteEntry::Bf16`,
    ///   `HeadEntry::Bf16`, `EmbedEntry::Bf16`, the router's bf16 entries).
    ///   Every other bf16 tensor (a norm, a tensor it does not name) stays
    ///   [`CardFormat::Bf16AsF32`]: the kernels that read those read f32
    ///   values.
    #[must_use]
    pub fn card_format(&self, t: &ModelTensor) -> Option<CardFormat> {
        if t.role == Role::Head && t.name == HEAD_TENSOR {
            return file_card_format(HEAD_TENSOR, t.ty, *t.dims.first()?);
        }
        match (t.role, card_format_of_role(t.ty, t.role)) {
            (
                Role::Attention
                | Role::HyperConnection
                | Role::SharedExpert
                | Role::DenseFfn
                | Role::EngramDense
                | Role::Head
                | Role::TokenEmbedding
                | Role::Router,
                Some(CardFormat::Bf16AsF32),
            ) if (self.bf16_words)(t) && t.ty == GgmlType::BF16 => Some(CardFormat::Bf16Raw),
            (_, f) => f,
        }
    }

    /// Each file tensor whose card format ([`ModelTensors::card_format`]) is a
    /// quantization other than its file's, in tensor order: the loader's
    /// choice ([`file_card_format`]: a BF16 head in MXFP8 planes), which the
    /// plan record states so the change of values is never silent. A re-layout
    /// of the file's own values is not listed: its own type's planes, bf16
    /// widened or kept as words, a K-quant decoded to f32.
    #[must_use]
    pub fn requantized(&self) -> Vec<Requantized> {
        self.tensors
            .iter()
            .filter_map(|t| {
                let card = self.card_format(t)?;
                requantizes(t.ty, card).then(|| Requantized {
                    name: t.name.clone(),
                    file: t.ty,
                    card,
                })
            })
            .collect()
    }

    /// The experts of routed stack `t` ([`stack_experts`]).
    #[must_use]
    pub fn experts_of(&self, t: &ModelTensor) -> u64 {
        stack_experts(t, self.experts)
    }

    /// The experts of layer `l`'s routed stacks: its first stack's
    /// ([`ModelTensors::experts_of`]), `experts` on a layer with none.
    #[must_use]
    pub fn layer_experts(&self, l: usize) -> u64 {
        self.tensors
            .iter()
            .find(|t| t.role == Role::RoutedExperts && t.layer == Some(l))
            .map_or(self.experts, |t| self.experts_of(t))
    }
}

/// A GPU card and the stage it runs. The same type serves an expert tier
/// card ([`Machine::tiers`]), which runs no stage: its `layers` are empty and
/// `head` and `token_embedding` are false, and the plan refuses a tier that
/// says otherwise.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Card {
    /// The name the plan and its records give the card.
    pub name: String,
    /// The device the card was resolved to ([`devices::resolve`]); `None`
    /// on a census-free plan, whose card the open finds by `name`.
    pub device: Option<devices::DeviceId>,
    /// What the driver leaves for us.
    pub usable_bytes: u64,
    pub context_bytes: u64,
    pub scratch_bytes: u64,
    /// Kept free: the expert rule never plans into it.
    pub margin_bytes: u64,
    /// The device allocator's granule: an allocation of this size or more
    /// takes whole granules of its own, smaller ones share granules.
    pub granule_bytes: NonZeroU64,
    /// The device's free bytes at plan time, when the card came from a
    /// census ([`devices::spec_of_device`]); `None` on a census-free card.
    /// The card's budget term is its usable bytes capped by this
    /// ([`capped`]): while another process holds the device, fewer experts
    /// fit, and a trunk that alone passes it is refused by name
    /// ([`PlacementError::CardFreeFloor`]).
    pub free_bytes: Option<u64>,
    /// The other processes holding the device at plan time, one named list,
    /// when the census named them; `None` names none.
    pub held_by: Option<&'static str>,
    /// The layers this card's stage runs.
    pub layers: Range<usize>,
    /// Whether this stage ends with the head.
    pub head: bool,
    /// Whether this card holds the token embedding table whole — a plan of
    /// the whole model on its cards. Only the card that runs layer 0 may;
    /// on every other plan the table stays on the host, one row gathered per
    /// token.
    pub token_embedding: bool,
    /// Fixed reserves beside the plan's tensors, by what they are for (a
    /// draft model's resident bytes, say): taken from the budget with the
    /// context and the scratch ([`Card::set_aside_bytes`]).
    pub reserves: Vec<(String, u64)>,
    /// A tier card holding only reserves, not experts.
    pub reserve_only: bool,
    /// A tier leg's scratch, jointly selected with its resident experts.
    pub tier_leg: Option<workstation::TierLegScratch>,
}

impl Card {
    /// Whether `self` and `other` are one device: by device when both were
    /// resolved, else by name.
    #[must_use]
    pub fn same_device(&self, other: &Card) -> bool {
        match (self.device, other.device) {
            (Some(a), Some(b)) => a.uuid == b.uuid,
            _ => self.name == other.name,
        }
    }

    /// The reserves' bytes; `None` past u64 bytes.
    #[must_use]
    pub fn reserve_bytes(&self) -> Option<u64> {
        self.reserves
            .iter()
            .try_fold(0u64, |sum, (_, bytes)| sum.checked_add(*bytes))
    }

    /// The tier leg's bound and selected scratch; absent on fixed-reserve paths.
    pub(super) fn selected_scratch(
        &self,
        model: &ModelTensors,
        counts: &[u64],
    ) -> Result<Option<(u64, u64)>, PlacementError> {
        let Some(leg) = self.tier_leg else {
            return Ok(None);
        };
        Ok(Some((
            leg.bytes(leg.downs)?,
            selected_leg_scratch(
                leg,
                counts,
                routed_down_acts(model).map(|(l, act)| (l, crate::kernels::DownForms::of(act))),
            )?,
        )))
    }

    pub(super) fn selected_reserve_bytes(
        &self,
        model: &ModelTensors,
        counts: &[u64],
    ) -> Result<u64, PlacementError> {
        let fixed = self.reserve_bytes().ok_or_else(|| self.bytes_past_u64())?;
        match self.selected_scratch(model, counts)? {
            Some((bound, selected)) => fixed
                .checked_sub(bound)
                .and_then(|n| n.checked_add(selected))
                .ok_or_else(|| self.bytes_past_u64()),
            None => Ok(fixed),
        }
    }

    pub(super) fn floor_with_reserves(
        &self,
        dense: u64,
        kv: u64,
        past: u64,
        reserves: u64,
    ) -> Option<u64> {
        [kv, self.context_bytes, self.scratch_bytes, reserves, past]
            .into_iter()
            .try_fold(dense, u64::checked_add)
    }

    /// The machine's context, scratch and reserve bound before uploads.
    /// Placement replaces a tier leg's bound with its candidate's actual
    /// scratch. The margin is not in it. `None` past u64 bytes.
    #[must_use]
    pub fn set_aside_bytes(&self) -> Option<u64> {
        self.context_bytes
            .checked_add(self.scratch_bytes)?
            .checked_add(self.reserve_bytes()?)
    }

    /// The fixed/bounded floor under `dense` upload bytes, `kv` cache bytes
    /// and `past` bytes held beside them ([`WholeLoad::past_bytes`]).
    /// Selected plans use the same sum with their actual reserves.
    /// `None` past u64 bytes.
    #[must_use]
    pub fn floor_bytes(&self, dense: u64, kv: u64, past: u64) -> Option<u64> {
        self.floor_with_reserves(dense, kv, past, self.reserve_bytes()?)
    }

    pub(super) fn bytes_past_u64(&self) -> PlacementError {
        PlacementError::CardBytesPastU64 {
            card: self.name.clone(),
        }
    }

    /// The part of the card's context term a census free reading already
    /// carries: the census reads on the device's primary context, so its
    /// reading is net of the context's own creation cost
    /// ([`workstation::CONTEXT_SELF`]), never of more than the card's whole
    /// context term; 0 on a census-free card.
    pub(super) fn context_in_free(&self) -> u64 {
        self.free_bytes
            .map_or(0, |_| self.context_bytes.min(workstation::CONTEXT_SELF))
    }

    /// The card's usable bytes capped by the device's free reading as the
    /// census took it — the context's own creation cost already out of it,
    /// which the plan's own cap ([`capped`]) adds back — and by `budget`, for
    /// a caller that sizes a load beside a plan against what the device read.
    #[must_use]
    pub fn read_capped_bytes(&self, budget: Option<u64>) -> u64 {
        let usable = self
            .free_bytes
            .map_or(self.usable_bytes, |free| self.usable_bytes.min(free));
        budget.map_or(usable, |b| usable.min(b))
    }
}

/// Routed down forms with their layers, before a placement selects any experts.
pub(super) fn routed_down_acts(
    model: &ModelTensors,
) -> impl Iterator<Item = (usize, crate::kernels::DownAct)> + '_ {
    model.tensors.iter().filter_map(|t| {
        if t.role != Role::RoutedExperts || ExpertPart::of_name(&t.name) != Some(ExpertPart::Down) {
            return None;
        }
        Some((t.layer?, crate::kernels::down(t.ty)?.1))
    })
}

/// Price the actual selected union, from tensors or cached per-layer forms.
/// Empty counts and layers with no down keep the descriptor's empty-form scratch.
pub(super) fn selected_leg_scratch(
    leg: workstation::TierLegScratch,
    counts: &[u64],
    downs: impl IntoIterator<Item = (usize, crate::kernels::DownForms)>,
) -> Result<u64, PlacementError> {
    use crate::kernels::{DownAct, DownForms};

    let forms = downs
        .into_iter()
        .filter(|(l, _)| counts.get(*l).is_some_and(|&n| n > 0))
        .fold(DownForms::NONE, |mut forms, (_, down)| {
            for act in [DownAct::Q8Act, DownAct::Q8Blocks32] {
                if down.holds(act) {
                    forms = forms.with(act);
                }
            }
            forms
        });
    leg.bytes(forms)
}

/// The host tier: its usable bytes and what is set aside before any tensor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Host {
    pub usable_bytes: u64,
    /// Fixed reserves, by what they are for.
    pub reserves: Vec<(String, u64)>,
}

impl Host {
    /// The reserves' bytes; `None` past u64 bytes. The one owner of that
    /// sum: a plan's host totals and [`Plan::violations`] both take it from
    /// here.
    #[must_use]
    pub fn reserve_bytes(&self) -> Option<u64> {
        self.reserves
            .iter()
            .try_fold(0u64, |sum, (_, bytes)| sum.checked_add(*bytes))
    }
}

/// One memory behind the cards and the host (GB10): each side's figure
/// reads the same bytes, so a plan holds their sum to it as well
/// ([`Violation::PoolOver`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnifiedPool {
    /// The pool's room at plan time, as the census read it for the card
    /// ([`Card::free_bytes`]): net of the primary context's own creation
    /// cost, like any census reading.
    pub bytes: u64,
}

/// One node of a machine: its host and, where its cards allocate from host
/// memory, the pool they share.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Node {
    pub host: Host,
    /// The pool the node's cards and its host share, on a node whose card
    /// allocates from host memory ([`workstation::unified_of`]); `None` where
    /// each card has memory of its own.
    pub unified: Option<UnifiedPool>,
}

/// Bytes a load puts on a device beside the plan's tensors, by what they are
/// for (a draft model's resident bytes, say): the one shape a named reserve
/// takes, entered by [`Machine::reserve`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Reserve {
    /// The device that holds them: `Card(i)` over [`Machine::all_cards`], the
    /// stage cards first (0 is the first stage card), or the host.
    pub on: Device,
    pub name: &'static str,
    pub bytes: u64,
}

/// The devices: the cards in stage order, the expert tier cards and the
/// nodes. The NVMe tier has no figure here — it holds its tensors in place
/// and never loads them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Machine {
    pub cards: Vec<Card>,
    /// Expert tier cards: no stage, only routed experts, planned after every
    /// stage card in this order.
    pub tiers: Vec<Card>,
    /// The nodes, this process's first: the plan reads [`Machine::host`] and
    /// [`Machine::unified`], both of `nodes[0]`. A card plan holds one node,
    /// or one more a tier card, tier `t` on node `t + 1`, whose card the pool
    /// of `nodes[0]` does not hold.
    pub nodes: Vec<Node>,
}

impl Machine {
    /// This process's node, `nodes[0]`. A machine with no node is refused by
    /// name: there is no host to plan beside.
    fn this_node(&self) -> &Node {
        self.nodes
            .first()
            .expect("a machine lists this process's node first")
    }

    /// This process's node, to change.
    fn this_node_mut(&mut self) -> &mut Node {
        self.nodes
            .first_mut()
            .expect("a machine lists this process's node first")
    }

    /// The host of this process's node.
    #[must_use]
    pub fn host(&self) -> &Host {
        &self.this_node().host
    }

    /// The host of this process's node, to change.
    pub fn host_mut(&mut self) -> &mut Host {
        &mut self.this_node_mut().host
    }

    /// The pool of this process's node ([`Node::unified`]).
    #[must_use]
    pub fn unified(&self) -> Option<UnifiedPool> {
        self.this_node().unified
    }

    /// The pool of this process's node, to change.
    pub fn unified_mut(&mut self) -> &mut Option<UnifiedPool> {
        &mut self.this_node_mut().unified
    }

    /// The pool [`PoolNeed::pool`] reads: this process's node's unified
    /// pool with each card's context self cost added back, as [`capped`]
    /// adds it to a card; `None` where each card has memory of its own. A
    /// sum past u64 is refused by name, so every card plan is made on a pool
    /// that passed here ([`plan_rule`]).
    pub(super) fn pool_bytes(&self) -> Result<Option<u64>, PlacementError> {
        let Some(unified) = self.unified() else {
            return Ok(None);
        };
        let context = self
            .all_cards()
            .enumerate()
            .filter(|&(i, _)| self.on_this_node(i))
            .map(|(_, c)| c.context_in_free())
            .sum();
        unified
            .bytes
            .checked_add(context)
            .map(Some)
            .ok_or(PlacementError::PoolPastU64 {
                bytes: unified.bytes,
                context,
            })
    }

    /// Every card, stage cards then tiers: the index space of
    /// [`Device::Card`] and of [`Plan::cards`].
    pub fn all_cards(&self) -> impl Iterator<Item = &Card> {
        self.cards.iter().chain(&self.tiers)
    }

    /// The tier cards that hold experts, in tier order: [`Machine::tiers`]
    /// less the reserve-only members ([`Card::reserve_only`]), which the plan
    /// lists after them. What a body hangs under its host tier, and what the
    /// tier's prompt batch is sized for.
    pub fn expert_tiers(&self) -> impl Iterator<Item = &Card> {
        self.tiers.iter().filter(|t| !t.reserve_only)
    }

    /// Whether card `i` of [`Machine::all_cards`] takes its bytes from this
    /// process's node, whose pool and host the plan holds: every stage card,
    /// and a tier card unless the machine lists a node of its own for each
    /// tier, tier `t` on node `t + 1`, whose bound is that card's
    /// ([`workstation::node_of`]).
    #[must_use]
    pub(crate) fn on_this_node(&self, i: usize) -> bool {
        i < self.cards.len() || self.nodes.len() == 1
    }

    /// The card at index `i` of [`Machine::all_cards`].
    #[must_use]
    pub fn card(&self, i: usize) -> Option<&Card> {
        match i.checked_sub(self.cards.len()) {
            None => self.cards.get(i),
            Some(t) => self.tiers.get(t),
        }
    }

    /// Set `reserve.bytes` aside on `reserve.on`, by what they are for: the one
    /// way a named reserve enters a machine. On a card the plan takes them from
    /// the card's budget with its context and scratch, before any expert
    /// ([`Card::set_aside_bytes`]): a reserve larger than the room leaves the
    /// card no expert, and one past its dense tensors is the budget floor's
    /// refusal. On the host they join its reserves. A card past the device list
    /// and the NVMe tier or nowhere are refused by name
    /// ([`PlacementError::ReserveOffPlan`]) and nothing is set aside: the bytes
    /// would sit on a device the plan does not count.
    pub fn reserve(&mut self, reserve: Reserve) -> Result<(), PlacementError> {
        let Reserve { on, name, bytes } = reserve;
        let stages = self.cards.len();
        let cards = stages + self.tiers.len();
        let held = match on {
            Device::Card(i) => match i.checked_sub(stages) {
                None => self.cards.get_mut(i),
                Some(t) => self.tiers.get_mut(t),
            }
            .map(|card| &mut card.reserves),
            Device::Host => Some(&mut self.host_mut().reserves),
            Device::Nvme | Device::Unused => None,
        };
        let Some(held) = held else {
            return Err(PlacementError::ReserveOffPlan {
                reserve: name.to_string(),
                on,
                cards,
            });
        };
        held.push((name.to_string(), bytes));
        Ok(())
    }

    /// The stage card a whole load of this machine is sized against
    /// ([`whole_need`]): the first, and on a unified machine with the host's
    /// reserves among its own, since a whole load has no host plan and the
    /// pool its card takes from holds them too. `None` with no stage card.
    #[must_use]
    pub fn whole_card(&self) -> Option<Card> {
        let mut card = self.cards.first()?.clone();
        if self.unified().is_some() {
            card.reserves.extend(self.host().reserves.iter().cloned());
        }
        Some(card)
    }
}

/// An architecture's context-sized bytes: the cache `layer` holds at
/// `ctx_max` tokens, and what a stage keeps beside it.
pub trait KvBytes {
    fn layer_bytes(&self, layer: usize, ctx_max: u64) -> u64;
    /// Bytes the stage that runs layer 0 holds at `ctx_max` positions beside
    /// its cache, one copy for every resident sequence: the rope rows its
    /// body copies to the card, one row a position (`gpu::rope_table::RopeRows`).
    /// The plan counts them on that card ([`CardTotals::stage_rows_bytes`]).
    /// The default is none.
    fn stage_position_bytes(&self, ctx_max: u64) -> u64 {
        let _ = ctx_max;
        0
    }
    /// Bytes beside the cache that `layer` keeps so the cache can be cut back:
    /// a copy of a ring that overwrites its rows, in page-locked host memory
    /// the card writes. The plan counts them on the host
    /// ([`HostTotals::shadow_bytes`]), not in the card's KV term. The default
    /// is none.
    fn shadow_bytes(&self, layer: usize, ctx_max: u64) -> u64 {
        let _ = (layer, ctx_max);
        0
    }
}

pub(super) struct NoKv;

impl KvBytes for NoKv {
    fn layer_bytes(&self, _layer: usize, _ctx_max: u64) -> u64 {
        0
    }
}

/// Where a segment lives.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Device {
    /// A card, by its index in [`Machine::all_cards`]: `cards ++ tiers`, so
    /// `Card(i)` for `i < cards.len()` is a stage card and the rest are the
    /// tiers in order.
    Card(usize),
    Host,
    Nvme,
    /// Nowhere: never loaded.
    Unused,
}

/// The independent output projection, separate from its norm and hyper-connection.
pub const HEAD_TENSOR: &str = "output.weight";

/// Resident format of file tensor `name` of type `ty`, its rows `k` values wide, as the loader
/// makes it: a BF16 [`HEAD_TENSOR`] occupies MXFP8 planes when `k` is a multiple of 128 and keeps
/// its words otherwise; every other tensor, a tied embedding included, takes its type's format.
#[must_use]
pub fn file_card_format(name: &str, ty: GgmlType, k: u64) -> Option<CardFormat> {
    if name != HEAD_TENSOR || ty != GgmlType::BF16 {
        CardFormat::of(ty)
    } else if k.is_multiple_of(128) {
        Some(CardFormat::Mxfp8Planes)
    } else {
        Some(CardFormat::Bf16Raw)
    }
}

/// Whether `card` is a quantized layout other than file type `file`'s own
/// ([`CardFormat::of`]): the card then holds values the file does not.
fn requantizes(file: GgmlType, card: CardFormat) -> bool {
    CardFormat::of(file) != Some(card)
        && matches!(
            card,
            CardFormat::Q5_0
                | CardFormat::Q5_1
                | CardFormat::Q8_0Planes
                | CardFormat::Nvfp4Planes
                | CardFormat::Mxfp8Planes
                | CardFormat::Fp8B128Planes
        )
}

/// A file tensor the card holds in another quantization than its file's
/// ([`ModelTensors::requantized`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Requantized {
    pub name: String,
    /// The tensor's type in the file.
    pub file: GgmlType,
    /// Its format on the card.
    pub card: CardFormat,
}

/// `name:file type:card format`, as the plan record lists it.
impl fmt::Display for Requantized {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}:{}", self.name, self.file, self.card.label())
    }
}

/// The card format by role. Engram gains are decoded to f32 at load.
#[must_use]
pub fn card_format_of_role(ty: GgmlType, role: Role) -> Option<CardFormat> {
    match (role, CardFormat::of(ty)) {
        (Role::EngramGain, Some(CardFormat::KQuant)) => Some(CardFormat::Bf16AsF32),
        (_, f) => f,
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
pub(super) struct Heap {
    granule: u64,
    /// Bytes of the granules taken so far.
    pub(super) taken: u64,
    /// Bytes left in each shared granule, in the order they were opened.
    pub(super) left: Vec<u64>,
}

impl Heap {
    pub(super) fn new(granule: NonZeroU64) -> Heap {
        Heap {
            granule: granule.get(),
            taken: 0,
            left: Vec::new(),
        }
    }

    pub(super) fn alloc(&mut self, bytes: u64) {
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

/// The bytes a card's allocator takes for `buffers`, allocated in this order
/// on a card of allocation granule `granule`: the plan's allocator model
/// ([`Heap`]), for a load the plan's rows do not describe.
#[must_use]
pub fn allocator_bytes(granule: NonZeroU64, buffers: impl IntoIterator<Item = u64>) -> u64 {
    let mut heap = Heap::new(granule);
    for b in buffers {
        heap.alloc(b);
    }
    heap.taken
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

impl Format {
    /// Every format, at its rank ([`Format::rank`]).
    const ALL: [Format; 13] = [
        Format::Card(CardFormat::KQuant),
        Format::Card(CardFormat::Q5_0),
        Format::Card(CardFormat::Q5_1),
        Format::Card(CardFormat::Q8_0Planes),
        Format::Card(CardFormat::F32),
        Format::Card(CardFormat::Bf16AsF32),
        Format::Card(CardFormat::Bf16Raw),
        Format::Card(CardFormat::Nvfp4Planes),
        Format::Card(CardFormat::Mxfp8Planes),
        Format::Card(CardFormat::Fp8B128Planes),
        Format::HostFile,
        Format::NvmeFile,
        Format::Unused,
    ];

    /// The format's place in [`Format::ALL`]: an exhaustive match, so a new
    /// format does not compile until it is given one, and the assertion
    /// below holds the list in that order.
    const fn rank(self) -> usize {
        match self {
            Format::Card(CardFormat::KQuant) => 0,
            Format::Card(CardFormat::Q5_0) => 1,
            Format::Card(CardFormat::Q5_1) => 2,
            Format::Card(CardFormat::Q8_0Planes) => 3,
            Format::Card(CardFormat::F32) => 4,
            Format::Card(CardFormat::Bf16AsF32) => 5,
            Format::Card(CardFormat::Bf16Raw) => 6,
            Format::Card(CardFormat::Nvfp4Planes) => 7,
            Format::Card(CardFormat::Mxfp8Planes) => 8,
            Format::Card(CardFormat::Fp8B128Planes) => 9,
            Format::HostFile => 10,
            Format::NvmeFile => 11,
            Format::Unused => 12,
        }
    }

    /// The format `word` names (its `Display`); `None` for a word no format
    /// has.
    #[must_use]
    pub fn of_word(word: &str) -> Option<Format> {
        Format::ALL.into_iter().find(|f| f.to_string() == word)
    }
}

const _: () = {
    let mut i = 0;
    while i < Format::ALL.len() {
        assert!(Format::ALL[i].rank() == i);
        i += 1;
    }
};

impl fmt::Display for Format {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Format::Card(c) => c.label(),
            Format::HostFile => "host_file",
            Format::NvmeFile => "nvme_file",
            Format::Unused => "unused",
        })
    }
}

/// The experts of one routed stack that one segment holds: ascending ids,
/// each once. The order is the slot order — a card segment uploads its
/// experts' rows contiguously in this order, so expert `ids()[s]` sits in
/// slot `s` — and every reader of a card stack (the loader's gather, the
/// slot map) walks it the same way.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ExpertList(Box<[u32]>);

impl ExpertList {
    /// `ids` sorted, as a list of a stack of `experts`; refused when an id
    /// repeats or is not below `experts`.
    pub fn new(mut ids: Vec<u32>, experts: u64) -> Result<ExpertList, PlacementError> {
        ids.sort_unstable();
        if let Some(w) = ids.windows(2).find(|w| w[0] == w[1]) {
            return Err(PlacementError::Experts(format!(
                "expert {} is listed twice",
                w[0]
            )));
        }
        if let Some(&last) = ids.last().filter(|&&e| u64::from(e) >= experts) {
            return Err(PlacementError::Experts(format!(
                "expert {last} is not below the stack's {experts}"
            )));
        }
        Ok(ExpertList(ids.into_boxed_slice()))
    }

    /// The experts `range` — the id prefix `0..n` is `range(0..n)`; refused
    /// when an id does not fit `u32`.
    pub fn range(range: Range<u64>) -> Result<ExpertList, PlacementError> {
        let ids: Option<Vec<u32>> = range.clone().map(|e| u32::try_from(e).ok()).collect();
        ids.map(|ids| ExpertList(ids.into_boxed_slice()))
            .ok_or_else(|| PlacementError::Experts(format!("experts {range:?} pass u32")))
    }

    /// The id prefix `0..n`.
    pub fn prefix(n: u64) -> Result<ExpertList, PlacementError> {
        ExpertList::range(0..n)
    }

    /// The experts of `0..experts` not in this list.
    pub fn complement(&self, experts: u64) -> Result<ExpertList, PlacementError> {
        let mut out = Vec::new();
        let mut held = self.0.iter().peekable();
        for e in 0..experts {
            let id = u32::try_from(e)
                .map_err(|_| PlacementError::Experts(format!("expert {e} passes u32")))?;
            if held.next_if_eq(&&id).is_none() {
                out.push(id);
            }
        }
        if let Some(e) = held.next() {
            return Err(PlacementError::Experts(format!(
                "expert {e} is not below the stack's {experts}"
            )));
        }
        Ok(ExpertList(out.into_boxed_slice()))
    }

    /// The ids, ascending.
    #[must_use]
    pub fn ids(&self) -> &[u32] {
        &self.0
    }

    /// How many experts the list holds.
    #[must_use]
    pub fn len(&self) -> u64 {
        self.0.len() as u64
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// `Some(n)` when the list is the id prefix `0..n`.
    #[must_use]
    pub fn as_prefix(&self) -> Option<u64> {
        self.0
            .iter()
            .zip(0u32..)
            .all(|(&e, i)| e == i)
            .then(|| self.len())
    }

    /// The slot that holds `id` on a card segment of this list.
    #[must_use]
    pub fn slot_of(&self, id: u32) -> Option<usize> {
        self.0.binary_search(&id).ok()
    }

    /// The list as ascending runs of consecutive ids, `[start, end)` each.
    #[must_use]
    pub fn runs(&self) -> Vec<Range<u64>> {
        let mut runs: Vec<Range<u64>> = Vec::new();
        for &e in self.0.iter() {
            let e = u64::from(e);
            match runs.last_mut() {
                Some(r) if r.end == e => r.end = e + 1,
                _ => runs.push(e..e + 1),
            }
        }
        runs
    }
}

impl IntoIterator for ExpertList {
    type Item = u32;
    type IntoIter = std::vec::IntoIter<u32>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_vec().into_iter()
    }
}

/// The runs, `start..end` each, comma-separated: a prefix prints as `0..n`.
impl fmt::Display for ExpertList {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, r) in self.runs().iter().enumerate() {
            let sep = if i == 0 { "" } else { "," };
            write!(f, "{sep}{}..{}", r.start, r.end)?;
        }
        Ok(())
    }
}
