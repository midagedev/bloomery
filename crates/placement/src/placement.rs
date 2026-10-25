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
//! keep the same number of experts on it, give or take one — the count
//! `n_l` — and the rest stay on the host. Which experts a layer keeps is an
//! [`ExpertList`]: the id prefix `[0, n_l)`. A card byte budget
//! (`BLOOMERY_CARD_BUDGET`, [`card_budget`]: one for every card, or one per
//! card) caps a card's usable bytes before the rule runs — the lever comes in
//! as [`PlanLevers`], which a binary parses once — and the census's free
//! reading caps them below that:
//! while another process holds the device, fewer experts fit
//! ([`Card::free_bytes`], [`capped`]).
//!
//! A machine may also carry expert tier cards ([`Machine::tiers`]): cards
//! that run no stage and hold only routed experts beside the host. The stage
//! cards are planned first, exactly as without tiers; then each tier, in
//! order, takes per eligible layer the ids after the stage card's `n_l` and
//! the earlier tiers', in id order, one
//! expert at a time on the layer that holds the fewest so far, so the
//! per-layer totals stay even ([`Plan::tier_n_l`]).

use std::cell::RefCell;
use std::fmt;
use std::num::NonZeroU64;
use std::ops::Range;

use gguf::GgmlType;

pub mod card_budget;
pub mod churn;
pub mod ctx;
pub mod ctx_ask;
pub mod devices;
pub mod need_hook;
pub mod nodes;
pub mod paged_drop;
pub mod tp;
pub mod workstation;

#[cfg(test)]
mod nodes_scale;
#[cfg(test)]
mod qwen38_cards;

pub use gguf::card::CardFormat;
pub use models::{Role, Unimplemented};

/// `features` grouped by feature in first-seen order, each with its layers:
/// `hash routing (layers 0, 1, 2); …`.
fn unimplemented_list(features: &[Unimplemented]) -> String {
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
    /// A tier card with an expert cap of 0: it holds its reserves and no
    /// expert, and serves no tier. The plan refuses one with no reserve
    /// ([`PlacementError::IdleTier`]); a body hangs a tier only under the
    /// tier cards that are not reserve-only ([`Machine::expert_tiers`]).
    pub reserve_only: bool,
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

    /// What the card sets aside before any tensor or cache: context,
    /// scratch and reserves. The one owner of that sum: the expert rule's
    /// budget, the budget floor, the headroom and [`Plan::violations`] all
    /// take it from here. The margin is not in it. `None` past u64 bytes.
    #[must_use]
    pub fn set_aside_bytes(&self) -> Option<u64> {
        self.context_bytes
            .checked_add(self.scratch_bytes)?
            .checked_add(self.reserve_bytes()?)
    }

    /// The card's floor under `dense` bytes of uploads and `kv` bytes of
    /// cache: those, its set-aside, and `past` — what the load keeps beside
    /// them: a plan's margin and what its caller holds beside the plan
    /// (`plan_floor` sums those), or a whole load's own arena and reserve
    /// ([`WholeLoad::past_bytes`]). The one owner of that sum: the budget
    /// floor, the free floor and the whole fit ([`whole_need`]) all take it
    /// from here. `None` past u64 bytes.
    #[must_use]
    pub fn floor_bytes(&self, dense: u64, kv: u64, past: u64) -> Option<u64> {
        [kv, self.set_aside_bytes()?, past]
            .into_iter()
            .try_fold(dense, u64::checked_add)
    }

    fn bytes_past_u64(&self) -> PlacementError {
        PlacementError::CardBytesPastU64 {
            card: self.name.clone(),
        }
    }

    /// The part of the card's context term a census free reading already
    /// carries: the census reads on the device's primary context, so its
    /// reading is net of the context's own creation cost
    /// ([`workstation::CONTEXT_SELF`]), never of more than the card's whole
    /// context term; 0 on a census-free card.
    fn context_in_free(&self) -> u64 {
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
    fn pool_bytes(&self) -> Result<Option<u64>, PlacementError> {
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

struct NoKv;

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

/// One piece of a tensor on one device.
#[derive(Clone, Debug)]
pub struct Segment {
    pub device: Device,
    pub format: Format,
    /// The experts this piece holds; `None` for a tensor that is not an
    /// expert stack.
    pub experts: Option<ExpertList>,
    pub resident_bytes: u64,
}

/// Where a run of a segment's piece lies in its tensor, counted from the
/// tensor's first row and first byte in the file.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Span {
    pub rows: Range<u64>,
    pub bytes: Range<u64>,
}

impl Segment {
    /// The rows of `t` this segment holds, as the tensor's contiguous runs
    /// in slot order — all its rows in one run, or its experts' rows of a
    /// stack of `experts`, one run per run of consecutive ids — and the file
    /// bytes each run takes.
    pub fn spans(&self, t: &ModelTensor, experts: u64) -> Result<Vec<Span>, PlacementError> {
        let rb = row_bytes(t)?;
        let span = |rows: Range<u64>| Span {
            bytes: rows.start * rb..rows.end * rb,
            rows,
        };
        match &self.experts {
            None => Ok(vec![span(0..rows_of(t))]),
            Some(list) => {
                let experts = stack_experts(t, experts);
                let (per, _) = per_expert(t, experts)?;
                if list.ids().last().is_some_and(|&e| u64::from(e) >= experts) {
                    return Err(PlacementError::tensor(
                        t,
                        format!("experts {list} are not inside 0..{experts}"),
                    ));
                }
                Ok(list
                    .runs()
                    .into_iter()
                    .map(|r| span(r.start * per..r.end * per))
                    .collect())
            }
        }
    }

    /// The rows of `t` this segment holds: the sum of its [`Segment::spans`].
    pub fn rows(&self, t: &ModelTensor, experts: u64) -> Result<u64, PlacementError> {
        Ok(self
            .spans(t, experts)?
            .iter()
            .map(|s| s.rows.end - s.rows.start)
            .sum())
    }

    /// The device buffers this card segment of `t` uploads, in bytes, in
    /// upload order — [`CardFormat::buffer_bytes`] of the rows it holds — or
    /// why they cannot be derived: a segment in no card format, experts
    /// outside the stack, rows the upload refuses.
    pub fn buffer_bytes(&self, t: &ModelTensor, experts: u64) -> Result<Vec<u64>, PlacementError> {
        let Format::Card(format) = self.format else {
            return Err(PlacementError::tensor(
                t,
                format!("a segment as {} has no card buffers", self.format),
            ));
        };
        card_buffers(t, format, self.rows(t, experts)?)
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
    /// The cache ([`KvBytes::layer_bytes`]).
    pub kv_bytes: u64,
    /// The rows the stage keeps a position beside its cache, on the card that
    /// runs layer 0 ([`KvBytes::stage_position_bytes`]); 0 on every other.
    pub stage_rows_bytes: u64,
    pub scratch_bytes: u64,
    pub context_bytes: u64,
    /// The card's reserves ([`Card::reserves`]).
    pub reserve_bytes: u64,
    /// usable − dense − experts − rounding − KV − stage rows − scratch −
    /// context − reserves, usable capped by the plan's card budget; the
    /// margin is inside it.
    pub headroom_bytes: i128,
}

impl CardTotals {
    /// The bytes the plan puts on the card: every term it counts against the
    /// card's usable bytes but the headroom, whose margin stays free. A sum
    /// past u64 bytes reads `u64::MAX`, as [`workstation::HostNeed::bytes`]
    /// does.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        let sum: u128 = [
            self.dense_bytes,
            self.expert_bytes,
            self.rounding_bytes,
            self.kv_bytes,
            self.stage_rows_bytes,
            self.scratch_bytes,
            self.context_bytes,
            self.reserve_bytes,
        ]
        .into_iter()
        .map(u128::from)
        .sum();
        u64::try_from(sum).unwrap_or(u64::MAX)
    }

    /// `bytes` more in the card's KV class, out of its headroom: the one way
    /// a model's plan adds a term to a card after the totals (the bytes its
    /// resident sequences hold beside their stores, reserved out of the
    /// expert budget), so the headroom stays usable less every term.
    pub fn grow_kv(&mut self, bytes: u64) {
        self.kv_bytes += bytes;
        self.headroom_bytes -= i128::from(bytes);
    }
}

/// The host's side of the plan.
#[derive(Clone, Debug)]
pub struct HostTotals {
    /// Routed-expert bytes resident on the host ([`Device::Host`]).
    pub expert_bytes: u64,
    /// Routed experts the host leg serves, summed over layers: the ids no
    /// card holds, whether their bytes are resident on the host or on the
    /// NVMe tier ([`HostTotals::nvme_expert_bytes`]).
    pub experts: u64,
    /// Routed-expert bytes the host leg reads from the NVMe tier
    /// ([`expert_nvme_tier`]); 0 when the host holds them all. Inside
    /// [`Plan::nvme_bytes`], outside [`HostTotals::expert_bytes`].
    pub nvme_expert_bytes: u64,
    /// The RAM arena the NVMe expert tier fills for the experts
    /// [`HostTotals::nvme_expert_bytes`] counts — the split dial's choice
    /// for this plan's room ([`nvme_arena_of`]): 0, or at least the
    /// arena's floor ([`ArenaFloor`]). Beside the plan's own bytes: the
    /// arena is the tier's memory, not a segment.
    pub nvme_arena_bytes: u64,
    /// The columns of the widest call the arena's floor counts
    /// ([`ArenaFloor::of`]): the resident slots a load steps as one pass,
    /// capped at the step port's width, set by the plan's maker before the
    /// split runs; 1 for a plan that steps one column.
    pub arena_columns: u64,
    /// Row-gathered tables held on the host.
    pub table_bytes: u64,
    /// The cards' ring shadows, page-locked ([`KvBytes::shadow_bytes`]).
    pub shadow_bytes: u64,
    /// The machine's host reserves and [`HostTotals::row_reserve_bytes`],
    /// and on a plan the NVMe expert tier split, the prompt run-ahead's
    /// page-cache window ([`expert_nvme_tier`]).
    pub reserve_bytes: u64,
    /// What the host sets aside for the row-gathered tables the program
    /// reads from the NVMe tier ([`row_table_tier`]); 0 when none lies
    /// there. Inside [`HostTotals::reserve_bytes`].
    pub row_reserve_bytes: u64,
    /// On a unified machine, what the cards take from the pool the host's
    /// room is read from: each card's terms with its margin, less the
    /// context self cost its census reading carries
    /// ([`Card::context_in_free`]), and the reserve its maker kept beside the
    /// plan ([`plan_routed_reserving`], [`plan_host_routed_reserving`]); 0
    /// where each card has memory of its own. The host's need counts it
    /// ([`workstation::HostNeed::cards`]), so the headroom below, the load's
    /// host check and the NVMe tier's split all leave the cards their share.
    pub pool_card_bytes: u64,
    /// What the plan's room leaves past its host need
    /// ([`workstation::HostNeed::headroom`], the one definition): the room
    /// the machine's figure leaves ([`workstation::HostNeed::machine_room`])
    /// less usable − experts − tables − shadows − reserves − the cards' share
    /// of a unified pool, the tighter of it and a room the caller stated
    /// ([`Plan::plan_at`]); on a plan the NVMe expert tier split
    /// ([`expert_nvme_tier`]), the room the split read less the arena and the
    /// plan's host need, since the room binds there.
    pub headroom_bytes: i128,
    /// The host room the plan was made at: the reading it took and the
    /// room it was held to ([`Plan::plan_at`], including the machine caps);
    /// `None` for a plan made without one.
    pub room: Option<(u64, u64)>,
}

impl HostTotals {
    /// Counts `slots` resident sequences stepped as one pass of one row each
    /// into the arena's floor: their columns, to the step port's `port`
    /// width. Refused by name: no slot, or a port of no column.
    pub fn step_slots(&mut self, slots: usize, port: usize) -> Result<(), PlacementError> {
        if slots == 0 || port == 0 {
            return Err(PlacementError::Metadata {
                key: "the NVMe tier arena's columns".to_string(),
                detail: format!("{slots} slots over a step port of {port} columns"),
            });
        }
        self.arena_columns = slots.min(port) as u64;
        Ok(())
    }

    /// The load's refusal names the room its plan was made at, so a report
    /// shows how far the reading fell between the plan and the load:
    /// `"; the plan was made at a reading of R B (plan room P B)"`, empty
    /// for a plan made without a host room.
    #[must_use]
    pub fn room_clause(&self) -> String {
        self.room.map_or_else(String::new, |(read, planned)| {
            format!("; the plan was made at a reading of {read} B (plan room {planned} B)")
        })
    }
}

/// The placement of one model on one machine.
#[derive(Clone, Debug)]
pub struct Plan<'a> {
    pub model: &'a ModelTensors,
    pub machine: &'a Machine,
    pub ctx_max: u64,
    /// One row per tensor, in the model's tensor order.
    pub rows: Vec<Row>,
    /// Per card of [`Machine::all_cards`]: the stage cards in stage order,
    /// then the tiers.
    pub cards: Vec<CardTotals>,
    pub host: HostTotals,
    pub nvme_bytes: u64,
    /// Per layer: how many experts its stage card holds (the ids are its
    /// routed stacks' segments on that card, [`Segment::experts`]).
    pub n_l: Vec<u64>,
    /// Per tier of [`Machine::tiers`], per layer: how many experts the tier
    /// holds — the ids after the stage card's `n_l` and the earlier
    /// tiers'. The host holds the rest.
    pub tier_n_l: Vec<Vec<u64>>,
    /// The card byte budget the plan was made under: a card planned with
    /// `min(usable, budget)` usable bytes, its budget [`card_budget::of`]
    /// ([`Plan::usable_bytes`]).
    pub card_budget: card_budget::CardBudget,
    /// The card budget is the one [`plan_in_pool`] lowered to the NVMe
    /// expert tier's floor, not one the caller gave.
    pub budget_from_pool: bool,
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
    /// A tensor its role puts on a card, of a type no card format loads.
    #[error(
        "tensor {name}: type {ty} has no device format, but its role ({role}) puts it on a card"
    )]
    NoCardFormat {
        name: String,
        ty: GgmlType,
        role: Role,
    },
    /// The cards' layer ranges do not cover the model once, in order, with one head.
    #[error("stage map: {0}")]
    Stages(String),
    /// The host tier's lock, or its residency query, failed: the span and the
    /// kernel's answer.
    #[error("host lock: {0}")]
    Host(String),
    /// The host's available bytes could not be read
    /// ([`workstation::host_room`]'s reading): the file or value that failed.
    #[error("the host's room: {0}")]
    HostRoom(String),
    /// A host set, a walk or a page release met an r8 sidecar it cannot take
    /// beside its pair: the sidecar's own named refusal.
    #[error(transparent)]
    R8(#[from] crate::r8::R8Error),
    /// A tensor-parallel cut or rank of a named tensor refused.
    #[error(transparent)]
    Tp(#[from] tp::TpError),
    /// An expert list that is not one: a repeated id, an id past the stack.
    #[error("expert list: {0}")]
    Experts(String),
    /// A host room under the floor of the NVMe expert tier: the plan's host
    /// terms outside the routed experts, and the transient reads of two
    /// layers' NVMe-tier experts and one layer's host experts beside them —
    /// and, where the split dial would give the room an arena, the arena's
    /// least ([`ArenaFloor`]): a room between the two is the one the dial
    /// refuses rather than page through the file mapping.
    #[error(
        "the host room {room} B is under the NVMe expert tier's floor {floor} B (the host terms \
         beside the routed experts {base} B + 3 × the heaviest layer's host-served experts \
         {layer} B: two layers' transient reads and one layer's resident experts{least}), the \
         plan's need being {need} B: free host memory, or load by a placement whose cards hold \
         more of the model; a room read from the host is the reading less {drift} B, the host \
         bytes the load's own start may take",
        least = least_term(.arena.as_ref()),
        drift = workstation::HOST_DRIFT
    )]
    HostRoomFloor {
        room: u64,
        floor: u64,
        base: u64,
        layer: u64,
        need: u64,
        /// The arena's least, counted in `floor`; `None` where the floor
        /// holds none (a set `BLOOMERY_NVTIER_BYTES`, or a plan the dial
        /// gives no arena).
        arena: Option<ArenaFloor>,
    },
    /// An arena the lever `BLOOMERY_NVTIER_BYTES` names that the room cannot
    /// take: the host segments it leaves fall under the NVMe expert tier's
    /// floor.
    #[error(
        "the NVMe tier's arena {arena} B leaves the host room {room} B under its floor {floor} B \
         (the arena takes at most room − floor): lower the arena, or free host memory"
    )]
    NvTierArena { arena: u64, room: u64, floor: u64 },
    /// A unified machine's plan the NVMe expert tier's floor refused at the
    /// cards' full share, planned again under lowered card budgets
    /// ([`plan_in_pool`]) and refused at the last: the rounds, the last
    /// budget and the last refusal.
    #[error(
        "on a unified pool the cards' share was lowered {rounds} times, to a card budget of \
         {budget} B, and the plan is still refused: {last}"
    )]
    PoolBudget {
        rounds: u32,
        budget: u64,
        last: Box<PlacementError>,
    },
    /// An arena the lever `BLOOMERY_NVTIER_BYTES` names under the arena's
    /// floor ([`ArenaFloor`]): a call would name more distinct ids of a
    /// paged layer than it has slots.
    #[error(
        "the NVMe tier's arena {arena} B (BLOOMERY_NVTIER_BYTES) is under its floor, {least}: one \
         call's ids would evict each other; set at least {bytes} B, or 0 to page through the \
         file mapping",
        bytes = .least.bytes
    )]
    NvTierArenaUnder { arena: u64, least: ArenaFloor },
    /// A routed stack's expert on two cards at once, by their index in
    /// [`Machine::all_cards`]: an expert lives on one device.
    #[error("tensor {tensor}: expert {expert} is on card #{first} and on card #{second}")]
    ExpertOnTwoCards {
        tensor: String,
        expert: u32,
        first: usize,
        second: usize,
    },
    /// An expert tier card that says it runs a stage: layers, the head or
    /// the token embedding.
    #[error("tier card {card}: {detail}; a tier card runs no stage")]
    Tier { card: String, detail: String },
    /// Cards of one device ([`Card::same_device`]) whose budgets (each card's usable bytes, under a card budget the
    /// capped ones) sum past that device's usable bytes: one device's
    /// budget counted twice.
    #[error(
        "card {card}: {cards} plan cards are this one device, and their budgets sum to {budgets} \
         B, past its usable {usable} B: one device's budget counted twice (a card budget that \
         splits the device between them is planned)"
    )]
    DeviceTwice {
        card: String,
        cards: usize,
        budgets: u64,
        usable: u64,
    },
    /// An expert tier card left with no expert, because the cards before it
    /// already hold every routed expert a card can: a tier the plan does
    /// not need.
    #[error(
        "tier card {card} (tier {tier}) holds no expert: the cards before it hold every routed \
         expert of the layers whose stacks a card can hold; plan without it"
    )]
    IdleTier { card: String, tier: usize },
    /// A machine of several nodes other than one a tier card: this plan
    /// holds one node's pool and host and each tier's card on its own node,
    /// and [`nodes::plan_nodes`] plans the others.
    #[error(
        "a machine of {nodes} nodes for {tiers} tier cards: the card plan plans one node, or one \
         more a tier card; plan the others with placement::nodes"
    )]
    Nodes { nodes: usize, tiers: usize },
    /// A unified pool whose bytes and its cards' context self cost
    /// ([`PoolNeed::pool`]) pass u64.
    #[error("a unified pool of {bytes} B and its cards' context of {context} B pass u64 bytes")]
    PoolPastU64 { bytes: u64, context: u64 },
    /// A tier card on a plan that keeps every routed stack on the host
    /// ([`plan_host_routed`]): no card holds a routed expert, so the tier
    /// has nothing to hold.
    #[error(
        "tier card {card} (tier {tier}) holds no expert: this plan puts every \
         routed expert on the host; plan without it"
    )]
    HostRoutedTier { card: String, tier: usize },
    /// A named reserve ([`Machine::reserve`]) on a device the plan does not
    /// count: a card past the device list (`cards` counts the stage cards and
    /// the tier cards), the NVMe tier, or nowhere.
    #[error(
        "the {reserve} reserve is on {on:?}, outside the plan: a reserve sits on one of the \
         plan's {cards} cards, stage first, or on the host"
    )]
    ReserveOffPlan {
        reserve: String,
        on: Device,
        cards: usize,
    },
    /// A layer the plan's per-layer expert counts ([`Plan::n_l`],
    /// [`Plan::tier_n_l`]) do not reach: `layers` is how many they hold.
    #[error("layer {layer} is past the plan's per-layer expert counts ({layers} layers)")]
    LayerPastCounts { layer: usize, layers: usize },
    /// A layer whose stage card and tier cards hold more experts than the
    /// model has on it: the counts the plan holds, and the model's.
    #[error(
        "layer {layer}: the stage card holds {stage} experts and the tier cards {tiers}, past \
         the model's {experts}"
    )]
    LayerOverHeld {
        layer: usize,
        stage: u64,
        tiers: u64,
        experts: u64,
    },
    /// A tier card's prompt-batch staging and scratch summed past u64.
    #[error(
        "the tier card's prompt-batch staging of {staging} B and scratch of {scratch} B pass u64 bytes"
    )]
    TierBatchCard { staging: u64, scratch: u64 },
    /// The tiers' prompt-batch host rows, one set a tier, summed past u64.
    #[error("{tiers} tier cards' prompt-batch host rows of {bytes} B each pass u64 bytes")]
    TierBatchHost { tiers: usize, bytes: u64 },
    /// Features of the file the engine does not run yet — every one, each with
    /// the layer it is on, or none for a model-wide one.
    #[error("{} feature(s) of this file are not implemented: {}", .0.len(), unimplemented_list(.0))]
    Unimplemented(Vec<Unimplemented>),
    /// Adaptive residency's churn pool does not fit the host: the stage card
    /// experts past the pinned ones, which the host set holds too
    /// ([`churn::ChurnPool`]), take more than the plan's host headroom.
    #[error("{0}")]
    ResidencyOverHost(Box<churn::OverHost>),
    /// A card's reserves, set-aside, load floor or shared-pool total summed past u64 bytes.
    #[error("card {card}: required card or shared-pool bytes pass u64 bytes")]
    CardBytesPastU64 { card: String },
    /// The host's reserves, with the row-gathered tables' page-cache reserve,
    /// summed past u64 bytes: no host figure holds them.
    #[error("the host's reserves pass u64 bytes")]
    HostBytesPastU64,
    /// A card budget below what the card needs with no expert on it: the
    /// floor counts what the caller holds beside the plan
    /// ([`plan_routed_reserving`]) as the expert rule's budget does.
    #[error(
        "card {card}: budget {budget} B is below its floor {floor} B = dense {dense} B (the \
         allocator's granules) + KV {kv} B + context {context} B + scratch {scratch} B + \
         reserves {reserves} B + margin {margin} B + beside {beside} B (what the caller holds \
         beside the plan)"
    )]
    CardBudgetFloor {
        card: String,
        budget: u64,
        floor: u64,
        dense: u64,
        kv: u64,
        context: u64,
        scratch: u64,
        reserves: u64,
        margin: u64,
        beside: u64,
    },
    /// A per-card budget on a plan with a card no CUDA ordinal of this
    /// process names ([`card_budget::check`]).
    #[error("card {card}: the per-card budget {budget} names cards by CUDA ordinal, and {why}")]
    CardBudgetCard {
        card: String,
        budget: String,
        why: &'static str,
    },
    /// A per-card budget or margin (`what`) naming a device the census does
    /// not list ([`card_budget::check_visible`]).
    #[error(
        "the per-card {what} {budget} names CUDA{ordinal}, which this process's census does not \
         list (visible: {visible})"
    )]
    CardBudgetUnseen {
        what: &'static str,
        ordinal: u32,
        budget: String,
        visible: String,
    },
    /// A margin that leaves a card no usable byte ([`card_budget::keep_free`]).
    #[error(
        "card {card}: a margin of {margin} B kept free leaves none of its {usable} B a plan may \
         use"
    )]
    CardMargin {
        card: String,
        margin: u64,
        usable: u64,
    },
    /// A budget by shares the plan cannot turn into a budget per card
    /// ([`card_budget::split`]), with the terms that say why.
    #[error("card budget {shares}: {why}")]
    CardShares { shares: String, why: String },
    /// A card whose device another process holds: what it had free at plan
    /// time is below the card's floor — the dense tensors, KV, set-aside,
    /// margin and what the caller holds beside the plan, before any expert
    /// moved to the host. [`CardFreeFloor`]'s terms, boxed to keep the error
    /// small.
    #[error("{}", .0.as_ref())]
    CardFreeFloor(Box<CardFreeFloor>),
}

/// The terms of [`PlacementError::CardFreeFloor`]: a card whose device
/// another process holds, what it had free, and the trunk that alone passes
/// it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CardFreeFloor {
    pub card: String,
    /// The device's free bytes at plan time.
    pub free: u64,
    /// The card's usable bytes.
    pub usable: u64,
    /// The dense trunk's floor: dense, KV, set-aside, margin and what the
    /// caller holds beside the plan summed.
    pub floor: u64,
    /// How far the floor passes the free bytes.
    pub over: u64,
    pub dense: u64,
    pub kv: u64,
    /// The card's context term less the part the free reading already
    /// carries.
    pub context: u64,
    pub scratch: u64,
    pub reserves: u64,
    pub margin: u64,
    /// What the caller holds beside the plan on the card
    /// ([`plan_routed_reserving`]).
    pub beside: u64,
    /// "" or " (held by …)", the census's holder list.
    pub held: String,
}

impl fmt::Display for CardFreeFloor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let CardFreeFloor {
            card,
            free,
            usable,
            floor,
            over,
            dense,
            kv,
            context,
            scratch,
            reserves,
            margin,
            beside,
            held,
        } = self;
        write!(
            f,
            "card {card}: the device had {free} B free of its usable {usable} B at plan \
             time{held}, and the plan's dense trunk alone needs {floor} B = dense {dense} B (the \
             allocator's granules) + KV {kv} B + context {context} B + scratch {scratch} B + \
             reserves {reserves} B + margin {margin} B + beside {beside} B (what the caller holds \
             beside the plan), past the {free} B by {over} B: free the card, or plan a placement \
             on a card with room"
        )
    }
}

impl std::error::Error for CardFreeFloor {}

impl PlacementError {
    /// The refusal that names a tensor: `host_lock` (in `bloomery-model`)
    /// builds these through the re-export.
    pub fn tensor(t: &ModelTensor, detail: impl Into<String>) -> PlacementError {
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
    /// A segment on a card whose stage does not run the tensor's layer or
    /// head, or on a tier card of a tensor that is not a routed stack.
    WrongCard { tensor: String, card: String },
    /// A card whose uploads' granules, KV, scratch, context and reserves pass
    /// usable − margin.
    CardOver {
        card: String,
        total: u64,
        limit: u64,
    },
    /// The plan's own host terms (the host's tensors, the cards' ring
    /// shadows and the machine's reserves, as a [`workstation::HostNeed`])
    /// pass the room the machine's figure leaves
    /// ([`workstation::HostNeed::machine_room`]): `total` is them with the
    /// OS reserve in it, beside the host's usable bytes.
    HostOver { total: u64, usable: u64 },
    /// On a unified machine, the `cards`' [`Violation::CardOver`] totals
    /// with their margins, the `host`'s need and the `beside` bytes the load
    /// holds past the plan's own terms ([`Plan::violations_beside`])
    /// together pass the pool: each side may fit its own figure while both
    /// read the same bytes. `pool` is the pool's bytes with each card's
    /// context self cost added back, as [`capped`] adds it to a card.
    PoolOver {
        cards: u64,
        host: u64,
        beside: u64,
        pool: u64,
    },
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
                "card {card}: resident + rounding + KV + scratch + context + reserves = {total} B passes usable − margin = {limit} B by {} B",
                total - limit
            ),
            Violation::HostOver { total, usable } => write!(
                f,
                "host: tensors + ring shadows + reserves = {total} B pass usable {usable} B by {} B",
                total - usable
            ),
            Violation::PoolOver {
                cards,
                host,
                beside,
                pool,
            } => {
                let total = cards + host + beside;
                write!(
                    f,
                    "unified pool: the cards with their margins {cards} B + the host's need \
                     {host} B + {beside} B the load holds beside the plan = {total} B pass the \
                     pool {pool} B by {} B",
                    total - pool
                )
            }
        }
    }
}

/// Which card runs each layer, which ends with the head, and which holds
/// the token embedding when one does.
struct Stages {
    of_layer: Vec<usize>,
    head: usize,
    embedding: Option<usize>,
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
fn card_buffers(
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
fn card_heaps(
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
fn card_segment_as(
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
fn in_place(device: Device, format: Format, experts: Option<ExpertList>, bytes: u64) -> Segment {
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
fn routed_row_on_as(
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
    /// name names no part is refused by name.
    Leg {
        gate_up: RoutedFormat,
        down: RoutedFormat,
    },
}

impl RoutedRule {
    /// The card format stack `t` loads in, `None` where the rule reads none.
    fn format(self, t: &ModelTensor) -> Result<Option<CardFormat>, PlacementError> {
        match self {
            Self::PerType(f) => Ok(f(t.ty)),
            Self::Leg { gate_up, down } => match ExpertPart::of_name(&t.name) {
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
        let mut out = Vec::new();
        for tensor in model
            .tensors
            .iter()
            .filter(|t| t.role == Role::RoutedExperts)
        {
            let layer = layer_of(tensor, model.layers)?;
            if let Some(why) = self.refusal(tensor, &mut gate_up_types[layer])? {
                out.push(HostOnly { layer, tensor, why });
            }
        }
        Ok(out)
    }

    /// Whether the card runs a layer of the routed stacks `stacks`: each
    /// loads in a card format, and on a leg its gate and up are one type.
    pub(crate) fn layer_runs(
        self,
        stacks: &[usize],
        model: &ModelTensors,
    ) -> Result<bool, PlacementError> {
        let mut runs = true;
        let mut gate_up_ty = None;
        for &i in stacks {
            runs &= self.refusal(&model.tensors[i], &mut gate_up_ty)?.is_none();
        }
        Ok(runs)
    }
}

/// The eligible layers of `layers`: those that route, whose stacks the card
/// runs under `routed` ([`RoutedRule::layer_runs`]) — a stage card's own
/// layers, or every layer for a tier.
fn eligible(
    layers: Range<usize>,
    stacks: &[Vec<usize>],
    model: &ModelTensors,
    routed: RoutedRule,
) -> Result<Vec<usize>, PlacementError> {
    let mut out = Vec::new();
    for l in layers {
        if !stacks[l].is_empty() && routed.layer_runs(&stacks[l], model)? {
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
fn spread(
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
enum Held {
    Rows(u64),
    Experts { layer: usize, rows_per_expert: u64 },
}

/// Card `c`'s uploads in the loader's order — the model's tensor order, which
/// `Weights::load_placed` walks: each whole tensor `rows` puts on the card,
/// and each routed stack of the card's `eligible` layers in its `routed`
/// format.
fn card_uploads<'m>(
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
struct Footprint<'m> {
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
    fn new(granule: NonZeroU64, uploads: &[(&'m ModelTensor, CardFormat, Held)]) -> Footprint<'m> {
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

    fn at(&self, n_l: &[u64]) -> Result<u64, PlacementError> {
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

/// The placement's levers, which a binary parses once
/// ([`PlanLevers::from_levers`]): the card budget `BLOOMERY_CARD_BUDGET`
/// sets. The default is none: each card's own usable bytes, or on a unified
/// pool the budget [`plan_in_pool`] lowers to the NVMe expert tier's floor.
#[derive(Clone, Debug, Default)]
pub struct PlanLevers {
    /// The card budget `BLOOMERY_CARD_BUDGET` sets, in bytes: a card of the
    /// plan plans with `min(usable, budget)`, its budget one for every card
    /// or its own entry ([`card_budget::of`]); unset, each card's own usable
    /// bytes, or on a unified pool the budget [`plan_in_pool`] lowers to the
    /// NVMe expert tier's floor. One value converts from `Option<u64>`.
    pub card_budget_bytes: card_budget::CardBudget,
}

impl PlanLevers {
    /// The placement's levers of a binary's one parse.
    pub fn from_levers(levers: &bloomery_levers::Levers) -> Result<PlanLevers, PlacementError> {
        Ok(PlanLevers {
            card_budget_bytes: levers.card_budget(),
        })
    }
}

/// [`plan_with`] under `levers`: the card budget, or none.
pub fn plan<'a>(
    model: &'a ModelTensors,
    machine: &'a Machine,
    ctx_max: u64,
    kv: &dyn KvBytes,
    levers: &PlanLevers,
) -> Result<Plan<'a>, PlacementError> {
    plan_rule(
        model,
        machine,
        ctx_max,
        kv,
        &levers.card_budget_bytes,
        Some(RoutedRule::PerType(CardFormat::of_routed)),
        0,
    )
}

/// The bytes the plan may take from `card`: its usable bytes, capped by the
/// free bytes the census read at plan time ([`Card::free_bytes`]) and by the
/// card budget, whichever holds less — the one place the free cap is taken.
/// The reading is already net of the context's own creation cost, which the
/// card's context term ([`Card::set_aside_bytes`]) counts again, so the cap
/// adds it back ([`Card::context_in_free`]): the context is counted once, and
/// an idle card's reading caps as its usable bytes do. The expert rule's
/// budget, the free floor, the headroom and [`Plan::violations`] all take it
/// from here.
fn capped(card: &Card, budget: Option<u64>) -> u64 {
    let usable = card.free_bytes.map_or(card.usable_bytes, |free| {
        card.usable_bytes
            .min(free.saturating_add(card.context_in_free()))
    });
    budget.map_or(usable, |b| usable.min(b))
}

/// Place every tensor of `model` on `machine` for a context of `ctx_max`
/// tokens. Dense tensors go where their role says. Routed stacks follow the
/// expert rule per card ([`spread`]): the granules the card's uploads take —
/// dense tensors and each layer's `n_l` experts, in upload order, through
/// the card's allocator — within usable − KV − context − scratch − margin. A
/// card whose layers cannot keep one expert keeps none; one that cannot hold
/// even its dense tensors shows up in [`Plan::violations`], not as an error
/// here. The `n_l` experts of a layer are its id prefix `[0, n_l)`. With
/// `card_budget`, one value for every card, every card plans
/// with `min(usable, budget)` usable bytes, and a card whose dense tensors,
/// KV, context, scratch and margin pass that is refused
/// ([`PlacementError::CardBudgetFloor`]). A card whose census free reading
/// is below its usable bytes plans within the free bytes instead
/// ([`capped`]), and one whose dense trunk alone passes them is refused
/// ([`PlacementError::CardFreeFloor`]). Cards of one name whose budgets sum
/// past that device's usable bytes ([`PlacementError::DeviceTwice`]) and a
/// tier left with nothing to hold ([`PlacementError::IdleTier`]) are refused
/// too.
pub fn plan_with<'a>(
    model: &'a ModelTensors,
    machine: &'a Machine,
    ctx_max: u64,
    kv: &dyn KvBytes,
    card_budget: Option<u64>,
) -> Result<Plan<'a>, PlacementError> {
    plan_rule(
        model,
        machine,
        ctx_max,
        kv,
        &card_budget.into(),
        Some(RoutedRule::PerType(CardFormat::of_routed)),
        0,
    )
}

/// [`plan`] with the expert rule run on the layers whose routed stacks the
/// card runs under `routed`, for a program whose card expert kernels read
/// other types than [`CardFormat::of_routed`]'s: the card budget, or none.
pub fn plan_routed<'a>(
    model: &'a ModelTensors,
    machine: &'a Machine,
    ctx_max: u64,
    kv: &dyn KvBytes,
    levers: &PlanLevers,
    routed: RoutedRule,
) -> Result<Plan<'a>, PlacementError> {
    plan_rule(
        model,
        machine,
        ctx_max,
        kv,
        &levers.card_budget_bytes,
        Some(routed),
        0,
    )
}

/// [`plan_routed`] with `reserve` bytes of every card kept for what the
/// caller plans beside this plan on the same card — an MTP draft's granules
/// and store: the expert rule spreads within the card's budget less
/// `reserve`, and so does the card's floor, so the floor a refusal names
/// ([`PlacementError::CardBudgetFloor`], [`PlacementError::CardFreeFloor`])
/// is the least budget that plans. Nothing else on the card sees it: the
/// plan's usable bytes, card budget, headroom and violations are its own; the
/// caller checks the card's bound on the sum. On a unified machine the host's need counts it
/// in the cards' share of the pool ([`HostTotals::pool_card_bytes`]).
pub fn plan_routed_reserving<'a>(
    model: &'a ModelTensors,
    machine: &'a Machine,
    ctx_max: u64,
    kv: &dyn KvBytes,
    levers: &PlanLevers,
    routed: RoutedRule,
    reserve: u64,
) -> Result<Plan<'a>, PlacementError> {
    plan_rule(
        model,
        machine,
        ctx_max,
        kv,
        &levers.card_budget_bytes,
        Some(routed),
        reserve,
    )
}

/// [`plan`] with every routed stack on the host: no card is eligible for the
/// expert rule, for a program with no card kernel for the model's routed
/// experts. A machine with an expert tier card is refused by name
/// ([`PlacementError::HostRoutedTier`]): the tier would hold nothing. The
/// card budget applies as in [`plan`].
pub fn plan_host_routed<'a>(
    model: &'a ModelTensors,
    machine: &'a Machine,
    ctx_max: u64,
    kv: &dyn KvBytes,
    levers: &PlanLevers,
) -> Result<Plan<'a>, PlacementError> {
    plan_host_routed_reserving(model, machine, ctx_max, kv, levers, 0)
}

/// [`plan_host_routed`] with `reserve` bytes of every card kept for what the
/// caller plans beside this plan on the same card, as
/// [`plan_routed_reserving`] keeps them: no expert fill sees them here, the
/// card's floor counts them, and on a unified machine the host's need counts
/// them in the cards' share of the pool ([`HostTotals::pool_card_bytes`]).
pub fn plan_host_routed_reserving<'a>(
    model: &'a ModelTensors,
    machine: &'a Machine,
    ctx_max: u64,
    kv: &dyn KvBytes,
    levers: &PlanLevers,
    reserve: u64,
) -> Result<Plan<'a>, PlacementError> {
    if let Some((t, tier)) = machine
        .tiers
        .iter()
        .enumerate()
        .find(|(_, tier)| !tier.reserve_only)
    {
        return Err(PlacementError::HostRoutedTier {
            card: tier.name.clone(),
            tier: t,
        });
    }
    plan_rule(
        model,
        machine,
        ctx_max,
        kv,
        &levers.card_budget_bytes,
        None,
        reserve,
    )
}

/// [`plan_with`], the expert rule run on each card's layers eligible under
/// `routed`, and on none without it, within each card's budget less
/// `reserve` ([`plan_routed_reserving`]), the budget per card under a
/// per-card `card_budget` ([`card_budget::of`], refused on a machine whose
/// card no CUDA ordinal of this process names: [`card_budget::check`]); the
/// routed stacks' card segments in the format `routed` gives them.
#[allow(
    clippy::too_many_arguments,
    reason = "the planners' one body: the model, machine, context and cache, and the rule's three inputs"
)]
fn plan_rule<'a>(
    model: &'a ModelTensors,
    machine: &'a Machine,
    ctx_max: u64,
    kv: &dyn KvBytes,
    card_budget: &card_budget::CardBudget,
    routed: Option<RoutedRule>,
    reserve: u64,
) -> Result<Plan<'a>, PlacementError> {
    let tiers = machine.tiers.len();
    let nodes = machine.nodes.len();
    if nodes != 1 && (tiers == 0 || nodes != 1 + tiers) {
        return Err(PlacementError::Nodes { nodes, tiers });
    }
    machine.pool_bytes()?;
    let stages = Stages::new(model.layers, &machine.cards)?;
    let mut rows: Vec<Option<Row>> = vec![None; model.tensors.len()];
    let mut stacks: Vec<Vec<usize>> = vec![Vec::new(); model.layers];
    for (i, t) in model.tensors.iter().enumerate() {
        if t.role == Role::RoutedExperts {
            stacks[layer_of(t, model.layers)?].push(i);
        } else {
            rows[i] = Some(place_whole(i, t, &stages, model)?);
        }
    }
    check_tiers(&machine.tiers)?;
    card_budget::check(card_budget, machine)?;
    let by_shares;
    let card_budget = match card_budget {
        card_budget::CardBudget::Shares(shares) => {
            by_shares = resolve_shares(
                shares, model, machine, ctx_max, kv, &rows, &stacks, routed, reserve,
            )?;
            &by_shares
        }
        other => other,
    };
    check_devices(machine, card_budget)?;
    let mut n_l = vec![0u64; model.layers];
    let none_held = vec![0u64; model.layers];
    let mut kv_bytes = Vec::with_capacity(machine.cards.len() + machine.tiers.len());
    for (c, card) in machine.cards.iter().enumerate() {
        let (kv_card, shadow, stage_rows) = card_kv(card, kv, ctx_max);
        let eligible = match routed {
            Some(r) => eligible(card.layers.clone(), &stacks, model, r)?,
            None => Vec::new(),
        };
        let fill = Fill {
            model,
            card,
            c,
            rows: &rows,
            eligible: &eligible,
            routed,
            card_budget: card_budget::of(card_budget, card),
            kv: kv_card + stage_rows,
            reserve,
        };
        fill.run(&mut n_l, &none_held)?;
        kv_bytes.push((kv_card, shadow, stage_rows));
    }
    let mut tier_n_l: Vec<Vec<u64>> = Vec::with_capacity(machine.tiers.len());
    for (t, tier) in machine.tiers.iter().enumerate() {
        let (kv_tier, shadow, _) = card_kv(tier, kv, ctx_max);
        let mut held = n_l.clone();
        for prev in &tier_n_l {
            for (h, p) in held.iter_mut().zip(prev) {
                *h += p;
            }
        }
        // A reserve-only member has an expert cap of 0: no layer is eligible,
        // so its fill plans no expert and still holds its reserves to the
        // card's budget and free floors.
        let eligible = match routed {
            Some(r) if !tier.reserve_only => eligible(0..model.layers, &stacks, model, r)?,
            _ => Vec::new(),
        };
        let fill = Fill {
            model,
            card: tier,
            c: machine.cards.len() + t,
            rows: &rows,
            eligible: &eligible,
            routed,
            card_budget: card_budget::of(card_budget, tier),
            kv: kv_tier,
            reserve,
        };
        let mut own = vec![0u64; model.layers];
        fill.run(&mut own, &held)?;
        // Only a layer whose stacks a card can hold leaves the tier work. A
        // planner with no card format has no tier to reach here
        // (`plan_host_routed` refuses one), and its empty `eligible` would
        // refuse the tier as idle.
        let left = eligible.iter().any(|&l| held[l] < model.layer_experts(l));
        let idle = if tier.reserve_only {
            tier.reserves.is_empty()
        } else {
            !left && own.iter().all(|&n| n == 0)
        };
        if idle {
            return Err(PlacementError::IdleTier {
                card: tier.name.clone(),
                tier: t,
            });
        }
        tier_n_l.push(own);
        kv_bytes.push((kv_tier, shadow, 0));
    }
    for (l, layer_stacks) in stacks.iter().enumerate() {
        if layer_stacks.is_empty() {
            continue;
        }
        let stage = stages.of_layer[l];
        let mut lists = vec![(stage, ExpertList::range(0..n_l[l])?)];
        let mut next = n_l[l];
        for (t, own) in tier_n_l.iter().enumerate() {
            let ids = next..next + own[l];
            next = ids.end;
            let list = ExpertList::range(ids)?;
            lists.push((machine.cards.len() + t, list));
        }
        for &i in layer_stacks {
            let t = &model.tensors[i];
            rows[i] = Some(routed_row_on_as(
                i,
                t,
                stage,
                lists.clone(),
                model,
                routed.map(|r| r.format(t)).transpose()?.flatten(),
            )?);
        }
    }
    let rows: Vec<Row> = rows.into_iter().flatten().collect();
    let mut plan = totals(
        model,
        machine,
        ctx_max,
        rows,
        &kv_bytes,
        (n_l, tier_n_l),
        card_budget.clone(),
    )?;
    plan.reserve_beside(reserve)?;
    Ok(plan)
}

/// A budget by shares ([`card_budget::CardBudget::Shares`]) as the budget per
/// card the expert rule plans with. Each card the shares name that takes
/// experts plans with its floor — the granules of its dense uploads, its KV,
/// context, scratch, reserves, margin and `reserve` — and its part of the
/// routed experts' bytes ([`card_budget::split`]): every routed stack whole,
/// as a tier card is offered them, in the proportion of the shares and as many
/// in all as the card whose room is tightest for its share holds, the host
/// tier taking the rest. A card the shares do not name plans with its own
/// bytes. Refused by name ([`PlacementError::CardShares`]): a plan that puts no
/// routed expert on a card (`routed` none), shares none of whose cards take
/// experts or has a share above 0, and a named card whose floor passes the
/// bytes it may plan with.
#[allow(
    clippy::too_many_arguments,
    reason = "the planner's own inputs: shares, model, machine, context, cache, rows, stacks, rule"
)]
fn resolve_shares(
    shares: &[(u32, u64)],
    model: &ModelTensors,
    machine: &Machine,
    ctx_max: u64,
    kv: &dyn KvBytes,
    rows: &[Option<Row>],
    stacks: &[Vec<usize>],
    routed: Option<RoutedRule>,
    reserve: u64,
) -> Result<card_budget::CardBudget, PlacementError> {
    let text = card_budget::CardBudget::Shares(shares.to_vec()).to_string();
    let refuse = |why: String| PlacementError::CardShares {
        shares: text.clone(),
        why,
    };
    let Some(routed) = routed else {
        return Err(refuse(
            "this plan places no routed expert on a card, so there is nothing to share out"
                .to_owned(),
        ));
    };
    let cards: Vec<&Card> = machine.all_cards().collect();
    let zeros = vec![0u64; model.layers];
    // `(ordinal, floor)` and the split's input, one entry per named card that takes experts.
    let mut named: Vec<(u32, u64)> = Vec::new();
    let mut parts: Vec<card_budget::ShareCard> = Vec::new();
    for (c, card) in cards.iter().enumerate() {
        let Some(ordinal) = card.device.map(|d| d.ordinal) else {
            continue;
        };
        let Some(&(_, weight)) = shares.iter().find(|&&(o, _)| o == ordinal) else {
            continue;
        };
        if card.reserve_only {
            continue;
        }
        let tier = c >= machine.cards.len();
        let (kv_card, _, stage_rows) = card_kv(card, kv, ctx_max);
        let layers = if tier {
            0..model.layers
        } else {
            card.layers.clone()
        };
        let own = eligible(layers, stacks, model, routed)?;
        let uploads = card_uploads(model, c, rows, &own, routed)?;
        let dense = Footprint::new(card.granule_bytes, &uploads).at(&zeros)?;
        let kv = kv_card.saturating_add(stage_rows);
        let floor = plan_floor(card, dense, kv, reserve);
        let cap = capped(card, None);
        let Some((floor, room)) = floor.and_then(|f| Some((f, cap.checked_sub(f)?))) else {
            return Err(refuse(format!(
                "card {}: its floor {} B = dense uploads {dense} B + KV {} B + context {} B + \
                 scratch {} B + reserves {} B + margin {} B + beside {reserve} B passes the {cap} B \
                 it may plan with, so it holds no expert",
                card.name,
                floor.map_or_else(|| "past u64".to_owned(), |f| f.to_string()),
                kv,
                card.context_bytes,
                card.scratch_bytes,
                card.reserve_bytes().unwrap_or(u64::MAX),
                card.margin_bytes,
            )));
        };
        named.push((ordinal, floor));
        parts.push(card_budget::ShareCard { weight, room });
    }
    if named.is_empty() {
        return Err(refuse(format!(
            "it names no card of this plan that takes experts (the plan's cards: {})",
            cards
                .iter()
                .map(|c| c.name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        )));
    }
    if parts.iter().all(|p| p.weight == 0) {
        return Err(refuse(
            "none of the plan's cards it names has a share above 0".to_owned(),
        ));
    }
    let routable = eligible(0..model.layers, stacks, model, routed)?;
    let caps: Vec<u64> = (0..model.layers).map(|l| model.layer_experts(l)).collect();
    let experts = card_uploads(model, cards.len(), rows, &routable, routed)?;
    let total = Footprint::new(cards[0].granule_bytes, &experts).at(&caps)?;
    let per = named
        .iter()
        .zip(card_budget::split(&parts, total))
        .map(|(&(ordinal, floor), x)| (ordinal, floor + x))
        .collect();
    Ok(card_budget::CardBudget::Per(per))
}

/// A card's KV cache and its ring shadows over the layers its stage runs,
/// and the stage's rows beside the cache on the card that runs layer 0;
/// none on a tier.
fn card_kv(card: &Card, kv: &dyn KvBytes, ctx_max: u64) -> (u64, u64, u64) {
    let kv_card = card
        .layers
        .clone()
        .map(|l| kv.layer_bytes(l, ctx_max))
        .sum();
    let shadow = card
        .layers
        .clone()
        .map(|l| kv.shadow_bytes(l, ctx_max))
        .sum();
    let rows = if card.layers.contains(&0) {
        kv.stage_position_bytes(ctx_max)
    } else {
        0
    };
    (kv_card, shadow, rows)
}

/// Refuse a tier card that says it runs a stage.
fn check_tiers(tiers: &[Card]) -> Result<(), PlacementError> {
    for t in tiers {
        let detail = if !t.layers.is_empty() {
            format!("it runs layers {:?}", t.layers)
        } else if t.head {
            "it carries the head".to_string()
        } else if t.token_embedding {
            "it holds the token embedding".to_string()
        } else {
            continue;
        };
        return Err(PlacementError::Tier {
            card: t.name.clone(),
            detail,
        });
    }
    Ok(())
}

/// Refuse cards of one device ([`Card::same_device`]) whose budgets under
/// `card_budget` sum past the device's usable bytes
/// ([`PlacementError::DeviceTwice`]).
fn check_devices(
    machine: &Machine,
    card_budget: &card_budget::CardBudget,
) -> Result<(), PlacementError> {
    let cards: Vec<&Card> = machine.all_cards().collect();
    for (i, card) in cards.iter().enumerate() {
        if cards[..i].iter().any(|c| c.same_device(card)) {
            continue;
        }
        let same: Vec<&&Card> = cards.iter().filter(|c| c.same_device(card)).collect();
        if same.len() < 2 {
            continue;
        }
        let usable = same.iter().map(|c| c.usable_bytes).max().unwrap_or(0);
        let budgets = same.iter().try_fold(0u64, |sum, c| {
            sum.checked_add(capped(c, card_budget::of(card_budget, c)))
        });
        if budgets.is_none_or(|b| b > usable) {
            return Err(PlacementError::DeviceTwice {
                card: card.name.clone(),
                cards: same.len(),
                budgets: budgets.unwrap_or(u64::MAX),
                usable,
            });
        }
    }
    Ok(())
}

/// One card's expert rule: card `c` of [`Machine::all_cards`], its budget
/// usable − KV − set-aside − margin − `reserve`, its uploads over `eligible`;
/// `card_budget` is the card's own ([`card_budget::of`]).
struct Fill<'m, 'r> {
    model: &'m ModelTensors,
    card: &'m Card,
    c: usize,
    rows: &'r [Option<Row>],
    eligible: &'r [usize],
    routed: Option<RoutedRule>,
    card_budget: Option<u64>,
    kv: u64,
    /// What the caller plans beside this plan on the card
    /// ([`plan_routed_reserving`]): out of the expert budget, and in the
    /// floor.
    reserve: u64,
}

impl Fill<'_, '_> {
    /// Fill `n_l` ([`spread`]) beside the counts `held` on other devices;
    /// under a card budget, refuse a card whose floor — `reserve` among its
    /// terms — passes it first, and under a free reading below the card's
    /// usable bytes, refuse a card whose floor passes what the device had
    /// free.
    fn run(&self, n_l: &mut [u64], held: &[u64]) -> Result<(), PlacementError> {
        let card = self.card;
        let format = self
            .routed
            .unwrap_or(RoutedRule::PerType(CardFormat::of_routed));
        let uploads = card_uploads(self.model, self.c, self.rows, self.eligible, format)?;
        let footprint = Footprint::new(card.granule_bytes, &uploads);
        let budget = i128::from(capped(card, self.card_budget))
            - i128::from(self.kv)
            - i128::from(
                card.set_aside_bytes()
                    .ok_or_else(|| card.bytes_past_u64())?,
            )
            - i128::from(card.margin_bytes)
            - i128::from(self.reserve);
        if let Some(b) = self.card_budget {
            let dense = footprint.at(n_l)?;
            check_floor(card, b, dense, self.kv, self.reserve)?;
        }
        if let Some(free) = card.free_bytes.filter(|&f| f < card.usable_bytes) {
            let dense = footprint.at(n_l)?;
            check_free_floor(card, free, dense, self.kv, self.reserve)?;
        }
        let caps: Vec<u64> = (0..n_l.len())
            .map(|l| self.model.layer_experts(l))
            .collect();
        spread(n_l, held, self.eligible, &caps, budget, |n| footprint.at(n))
    }
}

/// `card`'s floor in a plan whose caller holds `beside` bytes of it beside
/// the plan ([`Fill::reserve`]): [`Card::floor_bytes`] with the margin and
/// those as its `past`. Both floor checks ([`check_floor`],
/// [`check_free_floor`]) and the share split's floors ([`resolve_shares`])
/// take it from here, so the floor they name is the least budget at which
/// the expert rule's own budget (usable less KV, set-aside, margin and
/// `beside`, in [`Fill::run`]) reaches the dense uploads. `None` past u64
/// bytes.
fn plan_floor(card: &Card, dense: u64, kv: u64, beside: u64) -> Option<u64> {
    card.margin_bytes
        .checked_add(beside)
        .and_then(|past| card.floor_bytes(dense, kv, past))
}

/// Refuse `card` under the card budget `budget` when its floor — the
/// granules of its dense uploads `dense`, its KV `kv`, context, scratch,
/// reserves, margin and `beside`, what the caller holds beside the plan —
/// passes the budget. A floor within a budget above the card's own
/// usable bytes but past those shows up in [`Plan::violations`], as without
/// a budget: the card, not the budget, is too small.
fn check_floor(
    card: &Card,
    budget: u64,
    dense: u64,
    kv: u64,
    beside: u64,
) -> Result<(), PlacementError> {
    let floor = plan_floor(card, dense, kv, beside);
    match floor {
        Some(floor) if floor <= budget => Ok(()),
        _ => Err(PlacementError::CardBudgetFloor {
            card: card.name.clone(),
            budget,
            floor: floor.unwrap_or(u64::MAX),
            dense,
            kv,
            context: card.context_bytes,
            scratch: card.scratch_bytes,
            reserves: card.reserve_bytes().ok_or_else(|| card.bytes_past_u64())?,
            margin: card.margin_bytes,
            beside,
        }),
    }
}

/// Refuse `card` when its floor — the same terms [`check_floor`] sums, the
/// context less the part the reading already carries
/// ([`Card::context_in_free`]) — passes `free`, what its device had free at
/// plan time: another process holds enough of it that no expert rule can save
/// the plan, because the dense tensors alone do not fit. A floor within the
/// free bytes of a card whose census read none plans as before.
fn check_free_floor(
    card: &Card,
    free: u64,
    dense: u64,
    kv: u64,
    beside: u64,
) -> Result<(), PlacementError> {
    let context = card.context_bytes - card.context_in_free();
    let floor = plan_floor(card, dense, kv, beside).map(|f| f - card.context_in_free());
    match floor {
        Some(floor) if floor <= free => Ok(()),
        floor => Err(PlacementError::CardFreeFloor(Box::new(CardFreeFloor {
            card: card.name.clone(),
            free,
            usable: card.usable_bytes,
            floor: floor.unwrap_or(u64::MAX),
            over: floor.unwrap_or(u64::MAX).saturating_sub(free),
            dense,
            kv,
            context,
            scratch: card.scratch_bytes,
            reserves: card.reserve_bytes().ok_or_else(|| card.bytes_past_u64())?,
            margin: card.margin_bytes,
            beside,
            held: card
                .held_by
                .map_or(String::new(), |h| format!(" (held by {h})")),
        }))),
    }
}

/// What a program's whole load holds on its card past its weights, its
/// cache and its card's set-aside ([`Card::set_aside_bytes`]: the context
/// and the step arenas' scratch), as its own load counts it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WholeLoad {
    /// Its ubatch arena, which the load checks against the card's free
    /// bytes before allocating, and what that check keeps free past it.
    Checked { arena: u64, reserve: u64 },
    /// A program whose load allocates its arena with no such check and whose
    /// arena's bytes have no formula outside the allocation: the card's
    /// margin is what covers it.
    Unchecked,
}

impl WholeLoad {
    /// The bytes it holds past the set-aside on `card`: the arena and the
    /// reserve, or the card's margin. `None` past u64 bytes.
    #[must_use]
    pub fn past_bytes(self, card: &Card) -> Option<u64> {
        match self {
            WholeLoad::Checked { arena, reserve } => arena.checked_add(reserve),
            WholeLoad::Unchecked => Some(card.margin_bytes),
        }
    }
}

/// What a load of the whole model asks of one card ([`whole_need`]) and
/// what the card had; its `Display` is the verdict in one line, term by
/// term.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WholeNeed {
    /// The card, its context, scratch, reserves, margin and free reading.
    pub card: Card,
    /// The granules the uploads take through the card's allocator.
    pub weights: u64,
    /// The same uploads' buffer bytes, before the allocator's rounding.
    pub weights_unrounded: u64,
    /// The cache the caller counted: the load's context, every slot.
    pub kv: u64,
    /// What the program holds past them.
    pub load: WholeLoad,
    /// The card's floor over all of it ([`Card::floor_bytes`]).
    pub need: u64,
    /// What the card had: [`Card::read_capped_bytes`].
    pub budget: u64,
}

impl WholeNeed {
    /// Whether the load fits: its need within what the card had.
    #[must_use]
    pub fn fits(&self) -> bool {
        self.need <= self.budget
    }
}

impl fmt::Display for WholeNeed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let c = &self.card;
        write!(
            f,
            "the whole load on {}: {}: need {} B = weights {} B (the allocator's granules; {} B \
             of buffers) + cache {} B + context {} B + scratch {} B + reserves ",
            c.name,
            if self.fits() { "fits" } else { "does not fit" },
            self.need,
            self.weights,
            self.weights_unrounded,
            self.kv,
            c.context_bytes,
            c.scratch_bytes,
        )?;
        match c.reserve_bytes() {
            Some(bytes) => write!(f, "{bytes} B + ")?,
            None => write!(f, "past u64 bytes + ")?,
        }
        match self.load {
            WholeLoad::Checked { arena, reserve } => {
                write!(f, "arena {arena} B + reserve past it {reserve} B")?;
            }
            WholeLoad::Unchecked => write!(
                f,
                "margin {} B (the program's arena under it)",
                c.margin_bytes
            )?,
        }
        write!(f, ", of {} B the card had (", self.budget)?;
        match c.free_bytes {
            Some(free) => write!(
                f,
                "{free} B free of its usable {} B at plan time",
                c.usable_bytes
            )?,
            None => write!(f, "no free reading, its usable {} B", c.usable_bytes)?,
        }
        if let Some(held) = c.held_by {
            write!(f, ", held by {held}")?;
        }
        f.write_str(")")
    }
}

/// What a load of the whole of `model` on `card` with `kv` bytes of cache
/// asks of the card, against what the card had: the one owner of the
/// whole-fit verdict. The load uploads every tensor its loader takes — each
/// of the model's layers and every model-level tensor, whatever its role —
/// in the model's order, each in the card format its role gives its type
/// ([`ModelTensors::card_format`]) and each one's buffers through the card's
/// allocator ([`Heap`]), so the granules are counted; then the cache, the
/// card's set-aside and what the program holds past them (`load`), summed
/// by the card's floor ([`Card::floor_bytes`]). The budget is the card's
/// usable bytes capped by the census's free reading as taken
/// ([`Card::read_capped_bytes`]), not the plan's cap ([`capped`]), which
/// adds the context's own creation back: between the reading and its arena
/// a whole load takes more than these terms — its joined stacks' and its
/// caches' rounding, its rope rows and prompt image, its step arenas — and
/// the context term's whole size is what covers that. A tensor no card
/// format loads is refused by name ([`PlacementError::NoCardFormat`]): the
/// whole load cannot upload it.
pub fn whole_need(
    model: &ModelTensors,
    card: &Card,
    kv: u64,
    load: WholeLoad,
) -> Result<WholeNeed, PlacementError> {
    let mut heap = Heap::new(card.granule_bytes);
    let mut unrounded = 0u64;
    let loaded = model
        .tensors
        .iter()
        .filter(|t| t.layer.is_none_or(|l| l < model.layers));
    for t in loaded {
        let format = model
            .card_format(t)
            .ok_or_else(|| PlacementError::NoCardFormat {
                name: t.name.clone(),
                ty: t.ty,
                role: t.role,
            })?;
        for b in card_buffers(t, format, rows_of(t))? {
            unrounded += b;
            heap.alloc(b);
        }
    }
    let need = load
        .past_bytes(card)
        .and_then(|past| card.floor_bytes(heap.taken, kv, past))
        .ok_or_else(|| card.bytes_past_u64())?;
    Ok(WholeNeed {
        card: card.clone(),
        weights: heap.taken,
        weights_unrounded: unrounded,
        kv,
        load,
        need,
        budget: card.read_capped_bytes(None),
    })
}

/// Whether `r` places a row-gathered table the program reads by row id
/// ([`Role::EngramTable`]): what [`row_table_tier`] places and the row
/// reserve is for.
fn row_read(model: &ModelTensors, r: &Row) -> bool {
    model
        .tensors
        .get(r.tensor)
        .is_some_and(|t| t.role == Role::EngramTable)
}

/// The row reserve [`totals`] sets aside: [`workstation::ROW_CACHE`], the
/// engram row cache — the hot rows a reader keeps beside the drive — when
/// any row-gathered table lies on the NVMe tier, 0 when none does. A plan
/// whose maker sizes its tables' reader ([`row_table_tier`]) replaces it.
fn row_cache_reserve(model: &ModelTensors, rows: &[Row]) -> u64 {
    let on_nvme = rows
        .iter()
        .filter(|r| row_read(model, r))
        .flat_map(|r| &r.segments)
        .any(|s| s.device == Device::Nvme);
    if on_nvme { workstation::ROW_CACHE } else { 0 }
}

/// The page-cache bytes a reader of `model`'s row-gathered tables
/// ([`Role::EngramTable`]) holds while it reads one call of up to
/// `positions` positions from the NVMe tier: per table, its rows a position
/// ([`ModelTensor::gathered_rows`]) for every position, each physical plane of
/// a row on at most `1 + ceil((plane bytes - 1) / page)` host base pages.
/// Split rows are packed; NVFP4 part-source rows have separate code and scale
/// planes. The ids are hashed, so a call's rows are counted on pages of their
/// own, and each plane may straddle a page boundary. The reader
/// advises and copies one call at a time, so one call's pages are what must
/// stay resident until its copy. A table with no per-token row count, or a
/// room past u64, is refused by name.
pub fn row_room(
    model: &ModelTensors,
    positions: u64,
    page: u64,
    source: RowTableSource,
) -> Result<u64, PlacementError> {
    model
        .tensors
        .iter()
        .filter(|t| t.role == Role::EngramTable)
        .try_fold(0u64, |sum, t| {
            let rows = t.gathered_rows.ok_or_else(|| {
                PlacementError::tensor(t, "a row-gathered table without a per-token row count")
            })?;
            let bytes = row_bytes(t)?;
            let span = |bytes: u64| 1 + bytes.saturating_sub(1).div_ceil(page);
            let pages = match (source, t.ty) {
                (RowTableSource::Parts, GgmlType::NVFP4_ROWS) => {
                    if !bytes.is_multiple_of(9) {
                        return Err(PlacementError::tensor(
                            t,
                            "NVFP4 row bytes are not whole groups",
                        ));
                    }
                    let groups = bytes / 9;
                    span(groups * 8) + span(groups)
                }
                _ => span(bytes),
            };
            positions
                .checked_mul(rows)
                .and_then(|n| n.checked_mul(pages))
                .and_then(|n| n.checked_mul(page))
                .and_then(|n| sum.checked_add(n))
                .ok_or_else(|| {
                    PlacementError::tensor(
                        t,
                        format!("the row room of a call of {positions} positions passes u64 bytes"),
                    )
                })
        })
}

/// Where a row reader obtains its table. Part sources have no split host set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowTableSource {
    /// A split's mappings, held on the host when its room permits.
    Split,
    /// A part source's file ranges, gathered from the NVMe tier.
    Parts,
}

/// Where `plan`'s row-gathered tables ([`Role::EngramTable`]) are read
/// from, the one rule for a plan whose maker sizes their reader: on the
/// host in their file bytes when `room` — the host's available bytes
/// ([`workstation::host_room`]'s reading, or a room the caller gives) —
/// holds the plan's host need with them there ([`workstation::HostNeed`],
/// no churn pool beside it) and, on a unified machine, `floor` bytes more;
/// else on the NVMe tier, where the role rule placed them ([`place_whole`]),
/// with their reader's room for a call of up to `positions` positions
/// ([`row_room`], on `page` pages) as the plan's row reserve, in place of
/// the row cache [`totals`] set aside. [`RowTableSource::Parts`] stays on
/// NVMe, with its row window sized for the stored planes.
///
/// `floor` is the prompt cache's: the caller's two saved sequence states at
/// the plan's context. The cache's default budget is half of what the room
/// leaves past the need and it holds nothing under one state, so a table
/// that takes the room whole leaves a load with no cache to serve from. A
/// machine whose cards have memory of their own takes the room alone, as
/// every such plan did.
///
/// `page` is the host's base page ([`row_room`]): the row room counts the
/// pages a row sits on, and a host whose base page is not the one this
/// workstation has counts its own.
///
/// A plan with no such table is left as it is; [`Plan::row_tier`] reads the
/// outcome. A table of other than one NVMe file segment, or a sum past u64,
/// is refused by name.
pub fn row_table_tier(
    plan: &mut Plan<'_>,
    room: u64,
    floor: u64,
    positions: u64,
    page: u64,
    source: RowTableSource,
) -> Result<(), PlacementError> {
    let model = plan.model;
    let tables: Vec<usize> = (0..plan.rows.len())
        .filter(|&i| row_read(model, &plan.rows[i]))
        .collect();
    let mut moved = 0u64;
    for &i in &tables {
        let r = &plan.rows[i];
        let t = &model.tensors[r.tensor];
        let [s] = r.segments.as_slice() else {
            return Err(PlacementError::tensor(
                t,
                format!(
                    "the row-gathered table in {} segments, not one",
                    r.segments.len()
                ),
            ));
        };
        if (s.device, s.format) != (Device::Nvme, Format::NvmeFile) {
            return Err(PlacementError::tensor(
                t,
                format!(
                    "the row-gathered table on {:?} as {}, not the NVMe tier's file bytes",
                    s.device, s.format
                ),
            ));
        }
        moved = moved
            .checked_add(s.resident_bytes)
            .ok_or_else(|| PlacementError::tensor(t, "the row-gathered tables' bytes pass u64"))?;
    }
    let Some(&first) = tables.first() else {
        return Ok(());
    };
    let named = &model.tensors[plan.rows[first].tensor];
    let past = |what: &str| PlacementError::tensor(named, format!("{what} passes u64 bytes"));
    let old = plan.host.row_reserve_bytes;
    let mut host_arm = workstation::HostNeed::of(plan, 0);
    host_arm.tables = host_arm
        .tables
        .checked_add(moved)
        .ok_or_else(|| past("the host's tables with the row-gathered ones"))?;
    host_arm.reserves = host_arm.reserves.checked_sub(old).ok_or_else(|| {
        PlacementError::tensor(
            named,
            format!(
                "the plan's row reserve {old} B is past its host reserves {} B",
                host_arm.reserves
            ),
        )
    })?;
    let floor = if plan.machine.unified().is_some() {
        floor
    } else {
        0
    };
    let held = host_arm
        .bytes()
        .checked_add(floor)
        .ok_or_else(|| past("the host's need with the prompt cache's floor"))?;
    let new = if source == RowTableSource::Split && held <= room {
        plan.nvme_bytes = plan.nvme_bytes.checked_sub(moved).ok_or_else(|| {
            PlacementError::tensor(
                named,
                format!(
                    "the row-gathered tables' {moved} B pass the NVMe tier's {} B",
                    plan.nvme_bytes
                ),
            )
        })?;
        plan.host.table_bytes = host_arm.tables;
        plan.host.headroom_bytes -= i128::from(moved);
        for &i in &tables {
            let s = &mut plan.rows[i].segments[0];
            (s.device, s.format) = (Device::Host, Format::HostFile);
        }
        0
    } else {
        row_room(model, positions, page, source)?
    };
    plan.host.reserve_bytes = host_arm
        .reserves
        .checked_add(new)
        .ok_or_else(|| past("the host's reserves with the row room"))?;
    plan.host.row_reserve_bytes = new;
    plan.host.headroom_bytes += i128::from(old) - i128::from(new);
    Ok(())
}

/// The room a caller gave through the lever `name` (bytes, `M` and `G`
/// binary units), read as [`workstation::HostRead::Given`]: `Ok(None)` for
/// an unset one — the machine's reading stands. A value that is not bytes,
/// is past u64 or is not UTF-8 is refused by name.
pub fn host_room_given(
    name: &str,
    value: Option<&std::ffi::OsStr>,
) -> Result<Option<(u64, workstation::HostRead)>, String> {
    let Some(value) = value else {
        return Ok(None);
    };
    let text = value
        .to_str()
        .ok_or_else(|| format!("{name}={value:?} is not UTF-8; it takes bytes"))?;
    match bloomery_levers::parse_bytes(text) {
        Ok(bytes) => Ok(Some((bytes, workstation::HostRead::Given))),
        Err(e) => Err(format!(
            "{name}={e}; it takes bytes: digits, `_` separators allowed, then nothing, `M` or `G`"
        )),
    }
}

/// One layer's routed stacks as the NVMe expert tier splits them: each
/// stack's row and the index of its host segment, the ids that segment holds
/// (the same for every stack of the layer), and the bytes of one expert
/// across the stacks.
struct HostLayer {
    stacks: Vec<(usize, usize)>,
    ids: Vec<u32>,
    unit: u64,
}

impl HostLayer {
    fn bytes(&self) -> u64 {
        self.unit * self.ids.len() as u64
    }
}

/// Puts the routed experts the host cannot hold on the NVMe tier, for a plan
/// whose host need ([`workstation::HostNeed`], what `HostNeed::check` holds
/// the host's available bytes to) passes `room`: each layer's host-served
/// ids split into the first `r_l`, which stay on the host, and the rest,
/// which the host leg reads from the NVMe tier ([`Device::Nvme`],
/// [`Format::NvmeFile`]). The `r_l` are spread over the layers by their host
/// bytes, so the host's resident bytes fall by the overflow, to within one
/// layer's expert; every stack of a layer splits at the same ids. A plan
/// whose need `room` holds is left as it is.
///
/// Runs after [`row_table_tier`], which puts the row-gathered tables on the
/// NVMe tier first: a plan that still needs a split with such a table on the
/// host is refused by name. The need is the plan's, with no residency churn
/// pool beside it and no transient window: a reader of the NVMe tier takes
/// its two layers of transient reads from the room beside the plan's need
/// ([`PlacementError::HostRoomFloor`] keeps the room for them).
///
/// The page every window of an NVMe-tier arena slot is aligned to: the
/// `O_DIRECT` alignment the slot's fills read at (engram's `DIRECT_ALIGN`,
/// which the arena's crate holds equal to it at compile time).
pub const ARENA_ALIGN: u64 = 4096;

/// The step port's width: the columns past it do not enter the arena's floor
/// ([`HostTotals::step_slots`]). The model crate's `ops::DEFER_MAX_COLS`
/// holds it equal, by a compile-time assertion in the gpu crate's tier.
pub const STEP_PORT_COLUMNS: usize = 8;

/// The slots of a paged layer one residency lane holds at once beside the
/// step's call: `4 + 1`. Four are open reads: each lane thread pins the slot
/// of the id it stages from `open_read` until its `end_read`, and a flush's
/// or a landing's jobs stage on all `LANE_THREADS` (4) threads at once
/// (`host/lane.rs`, `Shared::serve`). One is a victim claim: the machine's
/// pick check (`SwapMachine::check_victims`) prepares a victim on its own
/// thread while the lane's threads hold their pins (`NvTier::ensure_victim`,
/// published unpinned). There is one lane per swap machine, so a paged layer
/// sees one lane's count. Placement cannot depend on the gpu crate: a test
/// there holds `LANE_THREADS + 1` equal to this value. A claim that finds
/// every candidate pinned is refused by name.
pub const ARENA_LANE_SLOTS: u64 = 4 + 1;

/// The experts of the NVMe tier's arena one batch of a prompt union holds:
/// the unit a claim, a fill and a pin move in. A union over more of a
/// layer's arena ids than its slots less two batches runs in batches, each
/// pinned while it computes and the next claimed and filled beside it, so
/// two batches are pinned at once ([`ArenaFloor::union`]). The model side's
/// batched union and the arena's floor read this one value.
pub const UNION_BATCH_IDS: u64 = 16;

/// The bytes one part of `len` bytes takes in an arena slot: the part and
/// the two partial pages an aligned read of it can span, in whole
/// [`ARENA_ALIGN`] pages. The one shape of a slot's window, which the
/// arena carves and its floor counts.
#[must_use]
pub fn arena_window_bytes(len: u64) -> u64 {
    (len + 2 * ARENA_ALIGN).div_ceil(ARENA_ALIGN) * ARENA_ALIGN
}

/// The NVMe tier arena's floor: the slots a paged layer needs so that one
/// call reads every distinct id it names from its own slot while the lane's
/// pins and claim beside it take others — the call's `k × columns` ids and
/// the lane's slots — or, for a prompt union run in batches, the two batches
/// it pins at once and the lane's slots ([`ArenaFloor::union`]), whichever
/// is larger, and their bytes over one slot of every layer the arena may page.
/// The one owner of the floor: the split dial gives no arena under it
/// ([`nvme_arena_of`]), and the arena's carve at load refuses a budget
/// under its slots.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArenaFloor {
    /// The routed width: the ids one column names.
    pub k: u64,
    /// The widest call's columns.
    pub columns: u64,
    /// The slots the lane holds beside the call: [`ARENA_LANE_SLOTS`]
    /// where a stage card holds a routed expert (the residency machine can
    /// run), else none.
    pub lane_slots: u64,
    /// One slot of every layer the host leg serves routed experts of, each
    /// part in its window ([`arena_window_bytes`]).
    pub slot_set: u64,
    /// The slots a prompt union run in batches holds at once: two batches
    /// ([`UNION_BATCH_IDS`]) and the lane's slots ([`ARENA_LANE_SLOTS`],
    /// whether or not a lane runs: the union's own refusal counts it so).
    pub union: u64,
    /// The larger of `k × columns + lane_slots` and `union`.
    pub slots: u64,
    /// `slots × slot_set`.
    pub bytes: u64,
}

impl ArenaFloor {
    /// The floor of calls of `columns` columns of `k` ids beside `lane_slots`
    /// lane slots, over a slot set of `slot_set` bytes. Refused by name: a
    /// count or a byte total past u64.
    pub fn new(
        k: u64,
        columns: u64,
        lane_slots: u64,
        slot_set: u64,
    ) -> Result<ArenaFloor, PlacementError> {
        let past = || PlacementError::Metadata {
            key: "the NVMe tier arena's floor".to_string(),
            detail: format!(
                "{k} × {columns} ids + {lane_slots} lane slots over a {slot_set} B slot set pass u64"
            ),
        };
        let call = k
            .checked_mul(columns)
            .and_then(|ids| ids.checked_add(lane_slots))
            .ok_or_else(past)?;
        let union = 2 * UNION_BATCH_IDS + ARENA_LANE_SLOTS;
        let slots = call.max(union);
        let bytes = slots.checked_mul(slot_set).ok_or_else(past)?;
        Ok(ArenaFloor {
            k,
            columns,
            lane_slots,
            slot_set,
            union,
            slots,
            bytes,
        })
    }

    /// `plan`'s floor for calls of [`HostTotals::arena_columns`] columns: `k`
    /// the model's routed width, the lane slots where a stage card holds a routed
    /// expert, and the slot set over every layer whose routed stacks the
    /// host leg serves (a host or an NVMe segment) — on a plan the split
    /// has not run on yet, the layers it can page, a bound over the ones it
    /// does, so the arena's carve at load never falls under the slots the
    /// dial sized it for.
    pub fn of(plan: &Plan<'_>) -> Result<ArenaFloor, PlacementError> {
        let model = plan.model;
        let mut slot_set = 0u64;
        for r in &plan.rows {
            let t = &model.tensors[r.tensor];
            if t.role != Role::RoutedExperts
                || t.layer.is_none()
                || !r
                    .segments
                    .iter()
                    .any(|s| matches!(s.device, Device::Host | Device::Nvme))
            {
                continue;
            }
            let (_, per) = per_expert(t, model.experts_of(t))?;
            slot_set = slot_set
                .checked_add(arena_window_bytes(per))
                .ok_or_else(|| PlacementError::tensor(t, "the arena's slot set passes u64"))?;
        }
        let lane_slots = if plan.n_l.iter().any(|&n| n > 0) {
            ARENA_LANE_SLOTS
        } else {
            0
        };
        ArenaFloor::new(
            model.experts_used,
            plan.host.arena_columns,
            lane_slots,
            slot_set,
        )
    }
}

/// The slots `plan`'s arena holds: its bytes over one slot of every layer the
/// host leg serves routed experts of ([`ArenaFloor::slot_set`]); 0 on a plan
/// with no arena. Refused by name: an arena on a plan with no such layer.
pub fn arena_slots(plan: &Plan<'_>) -> Result<u64, PlacementError> {
    let arena = plan.host.nvme_arena_bytes;
    if arena == 0 {
        return Ok(0);
    }
    let set = ArenaFloor::of(plan)?.slot_set;
    arena
        .checked_div(set)
        .ok_or_else(|| PlacementError::Metadata {
            key: "the NVMe tier arena's slots".to_string(),
            detail: format!("an arena of {arena} B over a slot set of 0 B"),
        })
}

impl fmt::Display for ArenaFloor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} slots a paged layer (the larger of {} × {} ids a call + {} lane slots and {} for \
             a union in batches) × {} B a slot of every paged layer = {} B",
            self.slots,
            self.k,
            self.columns,
            self.lane_slots,
            self.union,
            self.slot_set,
            self.bytes
        )
    }
}

/// [`PlacementError::HostRoomFloor`]'s term for the arena's least, when
/// the floor counts it.
fn least_term(least: Option<&ArenaFloor>) -> String {
    least.map_or_else(String::new, |a| {
        format!(
            " + the RAM arena's least, {a}, which BLOOMERY_NVTIER_BYTES=0 leaves out to page \
             through the file mapping"
        )
    })
}

/// What the split dial reads of a plan the NVMe expert tier splits
/// ([`expert_nvme_tier`]).
#[derive(Clone, Copy, Debug)]
pub struct ArenaDial {
    /// The host's room the split reads.
    pub room: u64,
    /// The plan's host terms beside the routed experts.
    pub base: u64,
    /// `W`: the host-served expert bytes of the layer with the most.
    pub heavy: u64,
    /// `H`: the host leg's routed-expert bytes before the split.
    pub held: u64,
    /// The plan's host need before the split.
    pub need: u64,
    /// The arena's floor.
    pub least: ArenaFloor,
}

impl ArenaDial {
    /// The tier's room floor, `base + 3 W` ([`expert_nvme_tier`]'s terms).
    /// Refused by name past u64.
    pub fn floor(&self) -> Result<u64, PlacementError> {
        3u64.checked_mul(self.heavy)
            .and_then(|w| w.checked_add(self.base))
            .ok_or_else(|| PlacementError::Metadata {
                key: "the NVMe expert tier's floor".to_string(),
                detail: "passes u64 bytes".to_string(),
            })
    }

    /// What the room leaves above the floor where it reaches half the host
    /// leg, `H/2 − floor` (the deep side's arenas lie under it), when the
    /// arena's least fits under it; `None` on a plan the dial gives no
    /// arena.
    fn peak(&self, floor: u64) -> Option<u64> {
        (self.held / 2)
            .checked_sub(floor)
            .filter(|&peak| peak >= self.least.bytes)
    }

    /// The least room the dial plans at under the lever's `given` arena:
    /// the floor, and with the lever unset on a plan with a deep side, the
    /// arena's least above it.
    pub fn room_floor(&self, given: Option<u64>) -> Result<u64, PlacementError> {
        let floor = self.floor()?;
        Ok(match (given, self.peak(floor)) {
            (None, Some(_)) => floor + self.least.bytes,
            _ => floor,
        })
    }

    /// The last room the dial gives an arena, `H/2 − 1` — the deep side's
    /// largest, `H/2 − 1 − F`, its one step to none at `H/2` — or `None` on
    /// a plan with no deep side.
    pub fn step_room(&self) -> Result<Option<u64>, PlacementError> {
        let floor = self.floor()?;
        Ok(self.peak(floor).map(|_| self.held / 2 - 1))
    }
}

/// The arena bytes the split dial gives `dial`'s room: the lever
/// `BLOOMERY_NVTIER_BYTES` (`given`) when set, else the default. The one
/// owner of the arena's bytes; the plan states what it chose in
/// [`HostTotals::nvme_arena_bytes`].
///
/// The default, with `F` the floor, `H` the host leg and `M` the arena's
/// least ([`ArenaFloor`]): the room above the floor, `room − F`, while the
/// room is under `H/2` (a deep split: the prompt is NVMe-bound whatever the
/// split, so the arena's warm set costs prefill little); from `H/2` on,
/// none (the split keeps every host expert it can and pages through the
/// file mapping, which serves several columns and the draft). No room gets
/// an arena under `M`: a room under `F + M` on a plan with a deep side is
/// refused. The one step is at `H/2`, from `H/2 − 1 − F` to none; a byte of
/// room moves the arena by one byte everywhere else. A plan whose deep side
/// cannot hold `M` gets none at any room.
///
/// Refused by name: a room under [`ArenaDial::room_floor`]
/// ([`PlacementError::HostRoomFloor`], naming the arena's least when it
/// counts it); a given arena under `M` ([`PlacementError::NvTierArenaUnder`])
/// or past `room − F` ([`PlacementError::NvTierArena`]). A given 0 pages
/// through the file mapping at any room the floor holds.
pub fn nvme_arena_of(dial: &ArenaDial, given: Option<u64>) -> Result<u64, PlacementError> {
    let floor = dial.floor()?;
    let room_floor = dial.room_floor(given)?;
    let room = dial.room;
    if room < room_floor {
        return Err(PlacementError::HostRoomFloor {
            room,
            floor: room_floor,
            base: dial.base,
            layer: dial.heavy,
            need: dial.need,
            arena: (room_floor > floor).then_some(dial.least),
        });
    }
    let least = dial.least;
    match given {
        Some(0) => return Ok(0),
        Some(arena) if arena < least.bytes => {
            return Err(PlacementError::NvTierArenaUnder { arena, least });
        }
        Some(arena) if arena > room - floor => {
            return Err(PlacementError::NvTierArena { arena, room, floor });
        }
        Some(arena) => return Ok(arena),
        None => {}
    }
    if dial.peak(floor).is_none() || room >= dial.held / 2 {
        return Ok(0);
    }
    Ok(room - floor)
}

/// The floor is `base + 3 W`: `base` the need with no routed expert on the
/// host, `W` the host bytes of the layer with the most — the room's host
/// terms beside the routed experts hold `1 W` of host-served expert
/// segments, `2 W` of the prompt run-ahead's page-cache window (a host
/// reserve of the split plan) and the RAM arena [`nvme_arena_of`] picks:
/// `1 W + 2 W + arena = room − base` at the largest arena,
/// `room − base − 3 W`. A room under it, or under it and the arena's least
/// where the dial would give the room an arena, is refused by name
/// ([`ArenaDial::room_floor`]); the plan's `host.experts` counts the
/// experts the host leg serves, on either tier.
/// The split plan's headroom is what the room leaves past the arena and the
/// host need, the budget a churn pool beside them must fit.
///
/// `given` is the lever `BLOOMERY_NVTIER_BYTES`'s value ([`nvme_arena_of`]),
/// read once by the family's `describe` beside the room: this function reads
/// no environment, so a gate plans each arm by its own arena.
pub fn expert_nvme_tier(
    plan: &mut Plan<'_>,
    room: u64,
    given: Option<u64>,
) -> Result<(), PlacementError> {
    let model = plan.model;
    let need = workstation::HostNeed::of(plan, 0);
    // The exact sum: a need past u64 bytes reads `u64::MAX` in `bytes()`,
    // which a room of `u64::MAX` would hold.
    if need.exact(true) <= u128::from(room) {
        return Ok(());
    }
    if let Some(r) = plan
        .rows
        .iter()
        .find(|r| row_read(model, r) && r.segments.iter().any(|s| s.device == Device::Host))
    {
        return Err(PlacementError::tensor(
            &model.tensors[r.tensor],
            format!(
                "the host need {} B passes the room {room} B with a row-gathered table still on \
                 the host: row_table_tier places it on the NVMe tier first",
                need.bytes()
            ),
        ));
    }
    let mut layers: Vec<HostLayer> = Vec::new();
    let mut layer_of: Vec<Option<usize>> = vec![None; model.layers];
    for (i, r) in plan.rows.iter().enumerate() {
        let t = &model.tensors[r.tensor];
        let Some(l) = t.layer.filter(|_| t.role == Role::RoutedExperts) else {
            continue;
        };
        let Some((at, s)) = r
            .segments
            .iter()
            .enumerate()
            .find(|(_, s)| s.device == Device::Host)
        else {
            continue;
        };
        if s.format != Format::HostFile {
            return Err(PlacementError::tensor(
                t,
                format!("a host segment as {}, not the file's bytes", s.format),
            ));
        }
        let Some(list) = s.experts.as_ref() else {
            return Err(PlacementError::tensor(
                t,
                "a host segment of a routed stack with no list",
            ));
        };
        let (_, per) = per_expert(t, model.experts_of(t))?;
        let Some(slot) = layer_of.get_mut(l) else {
            return Err(PlacementError::tensor(
                t,
                format!("layer {l} is past the model's"),
            ));
        };
        match *slot {
            None => {
                *slot = Some(layers.len());
                layers.push(HostLayer {
                    stacks: vec![(i, at)],
                    ids: list.ids().to_vec(),
                    unit: per,
                });
            }
            Some(k) => {
                let h = &mut layers[k];
                if h.ids != list.ids() {
                    return Err(PlacementError::tensor(
                        t,
                        format!("host experts {list} are not the other stacks' of layer {l}"),
                    ));
                }
                h.stacks.push((i, at));
                h.unit += per;
            }
        }
    }
    let base = workstation::HostNeed { experts: 0, ..need }.bytes();
    let heavy = layers.iter().map(HostLayer::bytes).max().unwrap_or(0);
    // The split dial: the arena's bytes come out of the host segments the
    // room holds, the split running on the room that remains.
    let held = plan.host.expert_bytes;
    let dial = ArenaDial {
        room,
        base,
        heavy,
        held,
        need: need.bytes(),
        least: ArenaFloor::of(plan)?,
    };
    let arena = nvme_arena_of(&dial, given)?;
    let room = room - arena;
    plan.host.nvme_arena_bytes = arena;
    // The prompt run-ahead's page-cache window, `2 W`, is a host reserve of
    // the split plan: the host keeps `keep` of its `held` expert bytes in
    // what the room leaves past the arena and the window, which the floor
    // keeps at least one layer's worth.
    // The split's sums are exact (the need may pass u64 bytes with the floor
    // under it): the reserve is checked, `keep` is computed in u128 from the
    // exact need, which the room does not hold (the early return above), so
    // `over` has no negative part. `3 W` fit u64 (the dial's floor).
    let past = |what: &str| PlacementError::Metadata {
        key: "the NVMe expert tier's split".to_string(),
        detail: format!("{what} passes u64 bytes"),
    };
    let window = 2 * heavy;
    plan.host.reserve_bytes = plan
        .host
        .reserve_bytes
        .checked_add(window)
        .ok_or_else(|| past("the host reserve beside the window"))?;
    let over = need.exact(true) + u128::from(window) - u128::from(room);
    let keep = u128::from(held)
        .checked_sub(over)
        .and_then(|keep| u64::try_from(keep).ok())
        .ok_or_else(|| past("the bytes the host keeps"))?;
    let share = |l: &HostLayer| u128::from(keep) * u128::from(l.bytes()) / u128::from(held);
    let mut r: Vec<u64> = layers
        .iter()
        .map(|l| u64::try_from(share(l) / u128::from(l.unit)).unwrap_or(u64::MAX))
        .collect();
    let mut kept: u64 = r.iter().zip(&layers).map(|(&r, l)| r * l.unit).sum();
    // The rest of the bytes to keep go one expert at a time to the layers
    // whose share was cut the most.
    let mut order: Vec<usize> = (0..layers.len()).collect();
    let cut = |k: usize| share(&layers[k]) - u128::from(r[k] * layers[k].unit);
    order.sort_by_key(|&k| std::cmp::Reverse(cut(k) * 1_000_000 / u128::from(layers[k].unit)));
    for k in order {
        let l = &layers[k];
        if r[k] < l.ids.len() as u64 && kept + l.unit <= keep {
            r[k] += 1;
            kept += l.unit;
        }
    }
    let mut moved = 0u64;
    for (l, &keep_l) in layers.iter().zip(&r) {
        let n = usize::try_from(keep_l)
            .unwrap_or(usize::MAX)
            .min(l.ids.len());
        for &(i, at) in &l.stacks {
            let t = &model.tensors[plan.rows[i].tensor];
            let (_, per) = per_expert(t, model.experts_of(t))?;
            let on_host = &l.ids[..n];
            let off_host = &l.ids[n..];
            let mut pieces = Vec::with_capacity(2);
            if !on_host.is_empty() {
                pieces.push(Segment {
                    device: Device::Host,
                    format: Format::HostFile,
                    experts: Some(ExpertList::new(on_host.to_vec(), model.experts)?),
                    resident_bytes: on_host.len() as u64 * per,
                });
            }
            if !off_host.is_empty() {
                let bytes = off_host.len() as u64 * per;
                moved += bytes;
                pieces.push(Segment {
                    device: Device::Nvme,
                    format: Format::NvmeFile,
                    experts: Some(ExpertList::new(off_host.to_vec(), model.experts)?),
                    resident_bytes: bytes,
                });
            }
            plan.rows[i].segments.splice(at..=at, pieces);
        }
    }
    plan.host.expert_bytes -= moved;
    plan.host.nvme_expert_bytes += moved;
    plan.nvme_bytes += moved;
    // The room binds a split plan: its headroom is what the room leaves
    // past the arena (`room` here) and the plan's host need.
    plan.host.headroom_bytes = workstation::HostNeed::of(plan, 0).headroom(room);
    Ok(())
}

/// The rounds [`plan_in_pool`] lowers the card budget before it refuses by
/// name. A round lowers the budget by at least a [`workstation::GRANULE`],
/// and a shortfall under one expert's card bytes takes a round a granule
/// until an expert leaves the card, so the rounds cover an expert of 128 MiB;
/// a plan still short past them meets a floor that falls too slowly for a
/// load to wait on.
const POOL_ROUNDS: u32 = 64;

/// The plan `make` builds under `levers` (the placement, then the NVMe
/// expert tier's split, [`expert_nvme_tier`]); on a unified machine whose
/// cards' share leaves the host's room under the tier's floor
/// ([`PlacementError::HostRoomFloor`]), the same plan under a card budget
/// lowered until the floor holds, marked [`Plan::budget_from_pool`]. The
/// cards and the host read one pool, and the cards' share is a host term of
/// the floor ([`HostTotals::pool_card_bytes`]); the experts the cards give
/// up are the host's, so the floor falls by less than the budget does. Each
/// round lowers the budget by the shortfall its refusal names, at least a
/// [`workstation::GRANULE`]. The card keeps each layer's id prefix, as under
/// any budget.
///
/// A first plan that is not refused for the floor, a plan under a given
/// budget (`BLOOMERY_CARD_BUDGET`), a plan of a machine whose cards have
/// memory of their own, and a shortfall past the card's whole budget are the
/// first plan's result. A lowered budget the plan refuses for another
/// reason, a budget that runs out, and a floor still unmet after
/// [`POOL_ROUNDS`] rounds are refused by name ([`PlacementError::PoolBudget`]).
pub fn plan_in_pool<'a>(
    machine: &Machine,
    levers: &PlanLevers,
    mut make: impl FnMut(&PlanLevers) -> Result<Plan<'a>, PlacementError>,
) -> Result<Plan<'a>, PlacementError> {
    let first = make(levers);
    let mut short = match &first {
        Err(PlacementError::HostRoomFloor { room, floor, .. }) => floor.saturating_sub(*room),
        _ => return first,
    };
    let (Some(_), card_budget::CardBudget::Unset, Some(card)) = (
        machine.unified(),
        &levers.card_budget_bytes,
        machine.cards.first(),
    ) else {
        return first;
    };
    let mut budget = capped(card, None);
    let mut last = None;
    for round in 1..=POOL_ROUNDS {
        let Some(lower) = budget.checked_sub(short.max(workstation::GRANULE.get())) else {
            break;
        };
        budget = lower;
        let refused = match make(&PlanLevers {
            card_budget_bytes: card_budget::CardBudget::Every(budget),
        }) {
            Ok(mut plan) => {
                plan.budget_from_pool = true;
                return Ok(plan);
            }
            Err(e) => e,
        };
        let PlacementError::HostRoomFloor { room, floor, .. } = refused else {
            return Err(PlacementError::PoolBudget {
                rounds: round,
                budget,
                last: Box::new(refused),
            });
        };
        short = floor.saturating_sub(room);
        last = Some((round, refused));
    }
    match last {
        None => first,
        Some((rounds, refused)) => Err(PlacementError::PoolBudget {
            rounds,
            budget,
            last: Box::new(refused),
        }),
    }
}

fn host_tier(machine: &Machine, rows: &[Row], card: usize) -> bool {
    card >= machine.cards.len()
        && rows.iter().any(|r| {
            r.segments
                .iter()
                .any(|s| s.device == Device::Card(card) && s.format == Format::HostFile)
        })
}

/// The per-device sums of a finished set of rows; `kv_bytes` is, per card of
/// [`Machine::all_cards`], its layers' cache, their ring shadows, which the
/// host holds, and the stage's rows beside the cache; `counts` the stage
/// cards' `n_l` and the tiers'.
fn totals<'a>(
    model: &'a ModelTensors,
    machine: &'a Machine,
    ctx_max: u64,
    rows: Vec<Row>,
    kv_bytes: &[(u64, u64, u64)],
    counts: (Vec<u64>, Vec<Vec<u64>>),
    card_budget: card_budget::CardBudget,
) -> Result<Plan<'a>, PlacementError> {
    let (n_l, tier_n_l) = counts;
    let mut routing = vec![false; model.layers];
    for t in model
        .tensors
        .iter()
        .filter(|t| t.role == Role::RoutedExperts)
    {
        if let Some(r) = t.layer.and_then(|l| routing.get_mut(l)) {
            *r = true;
        }
    }
    let cards: Vec<&Card> = machine.all_cards().collect();
    let mut card_dense = vec![0u64; cards.len()];
    let mut card_experts = vec![0u64; cards.len()];
    let (mut host_experts, mut host_tables, mut nvme_bytes) = (0u64, 0u64, 0u64);
    let mut nvme_experts = 0u64;
    for r in &rows {
        let is_routed = model.tensors[r.tensor].role == Role::RoutedExperts;
        for s in &r.segments {
            match (s.device, is_routed) {
                (Device::Card(c), false) => card_dense[c] += s.resident_bytes,
                (Device::Card(c), true) => card_experts[c] += s.resident_bytes,
                (Device::Host, true) => host_experts += s.resident_bytes,
                (Device::Host, false) => host_tables += s.resident_bytes,
                (Device::Nvme, _) => {
                    nvme_bytes += s.resident_bytes;
                    if is_routed {
                        nvme_experts += s.resident_bytes;
                    }
                }
                (Device::Unused, _) => {}
            }
        }
    }
    // Rows from `card_segment` carry their buffers' sum, so no card segment
    // here is one `violations` would report.
    let (heaps, _) = card_heaps(model, machine, &rows);
    let held = |c: usize, card: &Card| -> u64 {
        match c.checked_sub(machine.cards.len()) {
            None => card.layers.clone().map(|l| n_l[l]).sum(),
            Some(t) => tier_n_l[t].iter().sum(),
        }
    };
    let mut pool_cards = 0u64;
    let card_totals = cards
        .iter()
        .zip(&heaps)
        .enumerate()
        .map(|(c, (card, heap))| {
            let rounding = heap.taken - (card_dense[c] + card_experts[c]);
            let (kv, _, stage_rows) = kv_bytes[c];
            let used = card
                .floor_bytes(heap.taken, kv, stage_rows)
                .ok_or_else(|| card.bytes_past_u64())?;
            if machine.unified().is_some()
                && machine.on_this_node(c)
                && !host_tier(machine, &rows, c)
            {
                pool_cards = pool_cards
                    .checked_add(used - card.context_in_free())
                    .and_then(|bytes| bytes.checked_add(card.margin_bytes))
                    .ok_or_else(|| card.bytes_past_u64())?;
            }
            Ok(CardTotals {
                dense_bytes: card_dense[c],
                expert_bytes: card_experts[c],
                rounding_bytes: rounding,
                experts: held(c, card),
                kv_bytes: kv,
                stage_rows_bytes: stage_rows,
                scratch_bytes: card.scratch_bytes,
                context_bytes: card.context_bytes,
                reserve_bytes: card
                    .reserve_bytes()
                    .expect("the card floor checked its reserves"),
                headroom_bytes: i128::from(capped(card, card_budget::of(&card_budget, card)))
                    - i128::from(used),
            })
        })
        .collect::<Result<_, PlacementError>>()?;
    let row_reserve = row_cache_reserve(model, &rows);
    let reserve_bytes = machine
        .host()
        .reserve_bytes()
        .and_then(|bytes| bytes.checked_add(row_reserve))
        .ok_or(PlacementError::HostBytesPastU64)?;
    let shadow_bytes: u64 = kv_bytes.iter().map(|&(_, shadow, _)| shadow).sum();
    let mut host = HostTotals {
        expert_bytes: host_experts,
        experts: (0..model.layers)
            .filter(|&l| routing[l])
            .map(|l| model.layer_experts(l) - n_l[l] - tier_n_l.iter().map(|t| t[l]).sum::<u64>())
            .sum(),
        nvme_expert_bytes: nvme_experts,
        nvme_arena_bytes: 0,
        arena_columns: 1,
        table_bytes: host_tables,
        shadow_bytes,
        reserve_bytes,
        row_reserve_bytes: row_reserve,
        pool_card_bytes: if machine.unified().is_some() {
            pool_cards
        } else {
            0
        },
        headroom_bytes: 0,
        room: None,
    };
    // The room the machine's figure leaves past the plan's host need, the
    // need's own arithmetic (`HostNeed`).
    host.headroom_bytes =
        workstation::HostNeed::of_totals(machine.host(), &host, card_budget.is_set(), 0)
            .headroom(workstation::HostNeed::machine_room(machine.host()));
    Ok(Plan {
        model,
        machine,
        ctx_max,
        rows,
        cards: card_totals,
        host,
        nvme_bytes,
        n_l,
        tier_n_l,
        card_budget,
        budget_from_pool: false,
    })
}

/// What a unified plan holds in its one pool, and the pool
/// ([`Plan::pool_need`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoolNeed {
    /// The cards' [`Violation::CardOver`] totals with their margins.
    pub cards: u64,
    /// The host's need.
    pub host: u64,
    /// The pool's bytes with each card's context self cost added back, as
    /// [`capped`] adds it to a card.
    pub pool: u64,
}

/// `card`'s total against usable − margin: its uploads' granules, its KV,
/// its stage rows and its set-aside.
fn card_total(card: &Card, totals: &CardTotals, heap: &Heap) -> u64 {
    card.floor_bytes(heap.taken, totals.kv_bytes, totals.stage_rows_bytes)
        .expect("the plan's card totals fit u64")
}

impl Plan<'_> {
    fn reserve_beside(&mut self, reserve: u64) -> Result<(), PlacementError> {
        if self.machine.unified().is_some() {
            let here = (0..self.machine.all_cards().count())
                .filter(|&c| self.machine.on_this_node(c))
                .count() as u64;
            let past = || PlacementError::Metadata {
                key: "unified pool reserve".into(),
                detail: "the card share and bytes beside the plan pass u64".into(),
            };
            let beside = reserve.checked_mul(here).ok_or_else(past)?;
            self.host.pool_card_bytes = self
                .host
                .pool_card_bytes
                .checked_add(beside)
                .ok_or_else(past)?;
            self.host.headroom_bytes -= i128::from(beside);
        }
        Ok(())
    }

    /// The room the plan's tiers use: [`workstation::plan_room`], capped by
    /// the machine's host room and its unified pool. The pool's reading already
    /// leaves out the cards' context self cost; [`workstation::HostNeed`] counts
    /// the cards and the bytes reserved beside the plan, so none is subtracted
    /// from the room again. The reading and effective room are recorded in
    /// [`HostTotals::room`].
    ///
    /// A [`workstation::HostRead::Given`] room also sets the host headroom from
    /// the plan's terms at that room. A read room keeps the headroom at the
    /// machine's figure until a tier splits the plan: the reading can move
    /// between the plan and its load, where [`workstation::HostNeed::check`]
    /// holds it. A split plan's headroom is what its effective room leaves.
    pub fn plan_at(&mut self, room: &(u64, workstation::HostRead)) -> u64 {
        let figure = workstation::HostNeed::machine_room(self.machine.host());
        let pool = self.machine.unified().map_or(u64::MAX, |pool| pool.bytes);
        let planned = workstation::plan_room(room).min(figure).min(pool);
        self.host.room = Some((room.0, planned));
        if room.1 == workstation::HostRead::Given {
            self.host.headroom_bytes = workstation::HostNeed::of(self, 0).headroom(planned);
        }
        planned
    }

    /// The usable bytes `card` planned with: its own, capped by the device's
    /// free bytes at plan time and by its budget under the plan's card budget
    /// ([`capped`]'s one cap, [`card_budget::of`]).
    #[must_use]
    pub fn usable_bytes(&self, card: &Card) -> u64 {
        capped(card, card_budget::of(&self.card_budget, card))
    }

    /// The experts the host serves on layer `l`, resident or paged from the
    /// NVMe tier: the model's per layer, less the stage card's
    /// ([`Plan::n_l`]) and each tier card's ([`Plan::tier_n_l`]). A layer the
    /// host serves none of is 0. A layer without routed stacks reads as the
    /// model's expert count; ask it only of a routed layer. Refused by name:
    /// a layer past the plan's counts, and cards that hold more of the layer
    /// than the model has.
    pub fn host_experts(&self, l: usize) -> Result<u64, PlacementError> {
        let past = || PlacementError::LayerPastCounts {
            layer: l,
            layers: self.n_l.len(),
        };
        let stage = *self.n_l.get(l).ok_or_else(past)?;
        let mut tiers = 0u64;
        for tier in &self.tier_n_l {
            tiers = tiers.saturating_add(*tier.get(l).ok_or_else(past)?);
        }
        let experts = self.model.layer_experts(l);
        experts
            .checked_sub(stage.saturating_add(tiers))
            .ok_or(PlacementError::LayerOverHeld {
                layer: l,
                stage,
                tiers,
                experts,
            })
    }

    /// Every invariant the rows break, re-derived from the rows themselves:
    /// each tensor placed once; an expert stack's segments cover every expert
    /// once and a whole tensor is one segment; a card segment's buffers sum to
    /// its resident bytes; a card holds only what its stage uses, a tier card
    /// only routed experts; the granules of its uploads + KV + scratch +
    /// context + reserves ≤ usable − margin on each card, usable capped by the
    /// card budget;
    /// the plan's own host terms within the room its machine's figure leaves
    /// ([`workstation::HostNeed::check`] at
    /// [`workstation::HostNeed::machine_room`]), the NVMe tier's arena and
    /// window held to the room its split read instead; on a unified machine,
    /// the cards and the host together within the pool.
    pub fn violations(&self) -> Vec<Violation> {
        self.violations_beside(0)
    }

    /// [`Plan::violations`] of a plan whose load holds `beside` bytes past
    /// the plan's own terms (a draft's card bytes, a prompt front), which
    /// its maker holds to the card's bound itself: on a unified machine the
    /// pool holds them with the rest ([`Violation::PoolOver`]).
    pub fn violations_beside(&self, beside: u64) -> Vec<Violation> {
        let model = self.model;
        let cards: Vec<&Card> = self.machine.all_cards().collect();
        let stage_cards = self.machine.cards.len();
        let mut out = Vec::new();
        let mut times = vec![0usize; model.tensors.len()];
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
                        let tier = c >= stage_cards;
                        let uses = cards.get(c).is_some_and(|card| match (t.role, t.layer) {
                            (Role::RoutedExperts, Some(_)) if tier => true,
                            _ if tier => false,
                            (Role::Head, _) => card.head,
                            (Role::TokenEmbedding, _) => card.token_embedding,
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
                    Device::Host | Device::Nvme | Device::Unused => {}
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
        let (heaps, unsized_segments) = card_heaps(model, self.machine, &self.rows);
        out.extend(unsized_segments);
        for (c, ((card, totals), heap)) in cards.iter().zip(&self.cards).zip(&heaps).enumerate() {
            if host_tier(self.machine, &self.rows, c) {
                continue;
            }
            let total = card_total(card, totals, heap);
            let limit = self.usable_bytes(card).saturating_sub(card.margin_bytes);
            if total > limit {
                out.push(Violation::CardOver {
                    card: card.name.clone(),
                    total,
                    limit,
                });
            }
        }
        // The host's bound is the load's own check (`HostNeed::check`) at
        // the room the machine's figure leaves, over the plan's own terms:
        // the rows' resident bytes, the ring shadows and the machine's
        // reserves. The NVMe tier's arena and run-ahead window are the
        // split's, sized to the room it read and held to that room
        // ([`expert_nvme_tier`]), not to the machine's figure; a unified
        // pool's share of the cards is the pool's bound, below.
        let host = self.machine.host();
        let need = workstation::HostNeed {
            experts: self.host_resident(),
            tables: 0,
            reserves: host
                .reserve_bytes()
                .and_then(|bytes| bytes.checked_add(self.host.row_reserve_bytes))
                .expect("the plan's host reserves fit u64"),
            arena: 0,
            cards: 0,
            ..workstation::HostNeed::of(self, 0)
        };
        if let Err(short) = need.check(workstation::HostNeed::machine_room(host)) {
            out.push(Violation::HostOver {
                total: short.need.bytes().saturating_add(short.need.os),
                usable: host.usable_bytes,
            });
        }
        if let Some(PoolNeed { cards, host, pool }) = self.pool_need()
            && cards + host + beside > pool
        {
            out.push(Violation::PoolOver {
                cards,
                host,
                beside,
                pool,
            });
        }
        out
    }

    /// The bytes the rows hold on the host.
    fn host_resident(&self) -> u64 {
        self.rows
            .iter()
            .filter(|r| r.tensor < self.model.tensors.len())
            .flat_map(|r| &r.segments)
            .filter(|s| s.device == Device::Host)
            .map(|s| s.resident_bytes)
            .sum()
    }

    /// On a unified machine, what the plan holds in the one pool and the
    /// pool ([`Violation::PoolOver`]'s terms): the cards, re-derived from
    /// the rows, and the host's whole need with the split's arena and
    /// window. `None` where each card has memory of its own.
    #[must_use]
    pub fn pool_need(&self) -> Option<PoolNeed> {
        let pool = self
            .machine
            .pool_bytes()
            .expect("the plan was made on a pool that fits u64")?;
        let cards: Vec<&Card> = self.machine.all_cards().collect();
        let (heaps, _) = card_heaps(self.model, self.machine, &self.rows);
        let cards_bytes = cards
            .iter()
            .zip(&self.cards)
            .zip(&heaps)
            .enumerate()
            .filter(|&(i, _)| {
                self.machine.on_this_node(i) && !host_tier(self.machine, &self.rows, i)
            })
            .map(|(_, ((card, totals), heap))| card_total(card, totals, heap) + card.margin_bytes)
            .sum();
        let host = workstation::HostNeed {
            experts: self.host_resident(),
            tables: 0,
            cards: 0,
            ..workstation::HostNeed::of(self, 0)
        };
        Some(PoolNeed {
            cards: cards_bytes,
            host: host.bytes(),
            pool,
        })
    }

    /// The tier the plan's row-gathered tables ([`Role::EngramTable`]) are
    /// read from: `Some(Device::Host)` in their file bytes on the host,
    /// `Some(Device::Nvme)` from the NVMe tier, `None` for a model with
    /// none. A table on a card, unused or in other than one file segment,
    /// and tables split between the two tiers, are refused by name: a reader
    /// would read them from a tier the plan does not name.
    pub fn row_tier(&self) -> Result<Option<Device>, PlacementError> {
        let mut tier = None;
        for r in self.rows.iter().filter(|r| row_read(self.model, r)) {
            let t = &self.model.tensors[r.tensor];
            let here = match r.segments.as_slice() {
                [s] if (s.device, s.format) == (Device::Host, Format::HostFile) => Device::Host,
                [s] if (s.device, s.format) == (Device::Nvme, Format::NvmeFile) => Device::Nvme,
                segments => {
                    return Err(PlacementError::tensor(
                        t,
                        format!(
                            "the row-gathered table as {segments:?}, neither the host's nor the \
                             NVMe tier's file bytes"
                        ),
                    ));
                }
            };
            match tier {
                Some(d) if d != here => {
                    return Err(PlacementError::tensor(
                        t,
                        format!("the row-gathered table on {here:?}, another on {d:?}"),
                    ));
                }
                _ => tier = Some(here),
            }
        }
        Ok(tier)
    }
}

/// What is wrong with a tensor's segments, if anything: an expert stack must
/// hold every expert of `0..experts` exactly once, in non-empty lists;
/// anything else is one segment with no list.
fn segment_shape(t: &ModelTensor, segments: &[Segment], experts: u64) -> Option<String> {
    if t.role != Role::RoutedExperts {
        return match segments {
            [s] if s.experts.is_none() => None,
            _ => Some(format!("a whole tensor in {} segments", segments.len())),
        };
    }
    let experts = stack_experts(t, experts);
    let Ok(n) = usize::try_from(experts) else {
        return Some(format!("{experts} experts pass usize"));
    };
    let mut times = vec![0u32; n];
    for s in segments {
        let Some(list) = s.experts.as_ref().filter(|l| !l.is_empty()) else {
            return Some(format!("an expert segment with list {:?}", s.experts));
        };
        for &e in list.ids() {
            match usize::try_from(e).ok().and_then(|e| times.get_mut(e)) {
                Some(c) => *c += 1,
                None => return Some(format!("segment experts {list} pass 0..{experts}")),
            }
        }
    }
    let lists: Vec<String> = segments
        .iter()
        .filter_map(|s| s.experts.as_ref().map(ToString::to_string))
        .collect();
    if let Some(e) = times.iter().position(|&c| c == 0) {
        return Some(format!(
            "segments [{}] put expert {e} nowhere",
            lists.join("; ")
        ));
    }
    let twice = times.iter().position(|&c| c > 1)?;
    Some(format!(
        "segments [{}] put expert {twice} twice",
        lists.join("; ")
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn machine_accessors_use_the_first_node() {
        let nodes = [11, 22].map(|bytes| Node {
            host: Host {
                usable_bytes: bytes,
                ..host()
            },
            unified: Some(UnifiedPool { bytes: bytes + 1 }),
        });
        let mut machine = Machine {
            cards: vec![],
            tiers: vec![],
            nodes: nodes.to_vec(),
        };
        assert_eq!(machine.host(), &nodes[0].host);
        assert_eq!(machine.unified(), nodes[0].unified);
        machine.host_mut().usable_bytes = 33;
        *machine.unified_mut() = Some(UnifiedPool { bytes: 44 });
        assert_eq!(machine.nodes[0].host.usable_bytes, 33);
        assert_eq!(machine.nodes[0].unified, Some(UnifiedPool { bytes: 44 }));
        assert_eq!(machine.nodes[1], nodes[1]);
        machine.nodes.clear();
        for mutable in [false, true] {
            let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                if mutable {
                    machine.host_mut();
                } else {
                    let _ = machine.host();
                }
            }))
            .unwrap_err();
            let message = panic
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| panic.downcast_ref::<&str>().copied());
            assert_eq!(message, Some("a machine lists this process's node first"));
        }
    }

    /// Each format's word names it back, every word apart; another word
    /// names none.
    #[test]
    fn a_formats_word_names_it() {
        for (word, format) in [
            ("kquant", Format::Card(CardFormat::KQuant)),
            ("q5_0", Format::Card(CardFormat::Q5_0)),
            ("q5_1", Format::Card(CardFormat::Q5_1)),
            ("q8_0_planes", Format::Card(CardFormat::Q8_0Planes)),
            ("f32", Format::Card(CardFormat::F32)),
            ("bf16_as_f32", Format::Card(CardFormat::Bf16AsF32)),
            ("bf16_raw", Format::Card(CardFormat::Bf16Raw)),
            ("mxfp8_planes", Format::Card(CardFormat::Mxfp8Planes)),
            ("fp8_b128_planes", Format::Card(CardFormat::Fp8B128Planes)),
            ("host_file", Format::HostFile),
            ("nvme_file", Format::NvmeFile),
            ("unused", Format::Unused),
        ] {
            assert_eq!(Format::of_word(word), Some(format), "{word}");
        }
        assert_eq!(Format::of_word("q4_k"), None);
        assert_eq!(Format::of_word(""), None);
    }

    /// The floor is the larger of a call's `k × columns + lane_slots` and the
    /// batched union's two batches and the lane's slots, and the union's term
    /// counts the lane's slots whether or not a lane runs: the model side's
    /// refusal of a union under it (`TierError::BatchFloor`) never reads the
    /// plan's lane slots. A lane's slots are its four open reads and one
    /// victim claim.
    #[test]
    fn the_floor_carries_the_batched_unions_pins() {
        assert_eq!(ARENA_LANE_SLOTS, 4 + 1);
        let union = 2 * UNION_BATCH_IDS + ARENA_LANE_SLOTS;
        let slot_set = 1 << 20;
        // The routed width a call names, one column, a lane: under the union.
        let narrow = ArenaFloor::new(10, 1, ARENA_LANE_SLOTS, slot_set).expect("a floor");
        assert_eq!((narrow.union, narrow.slots), (union, union));
        assert_eq!(narrow.bytes, union * slot_set);
        // No lane runs: the union's term still counts the lane's slots.
        let no_lane = ArenaFloor::new(10, 1, 0, slot_set).expect("a floor");
        assert_eq!(no_lane.slots, union);
        // A call wider than the union's pins: its own term.
        let wide = ArenaFloor::new(16, 3, ARENA_LANE_SLOTS, slot_set).expect("a floor");
        assert_eq!((wide.union, wide.slots), (union, 16 * 3 + ARENA_LANE_SLOTS));
        assert_eq!(wide.bytes, wide.slots * slot_set);
        let text = narrow.to_string();
        assert!(
            text.contains("37 slots a paged layer") && text.contains("37 for a union in batches"),
            "{text}"
        );
        // Past u64 is refused by name, not wrapped.
        assert!(ArenaFloor::new(u64::MAX, 2, 0, 1).is_err());
    }

    /// PIN(2026-10-12): a lane holds its open reads and its victim claim at
    /// once, so the union's floor is `2 × UNION_BATCH_IDS + LANE_THREADS + 1`
    /// slots (two batches pinned, four lane threads each pinning the id they
    /// stage, one victim claim): 37. The earlier one-slot lane term, 33,
    /// refused a claim beside four pins and two pinned batches.
    #[test]
    fn a_lane_holds_its_open_reads_and_its_victim_claim() {
        let lane_threads = 4;
        assert_eq!(ARENA_LANE_SLOTS, lane_threads + 1);
        let floor = ArenaFloor::new(10, 1, ARENA_LANE_SLOTS, 1 << 20).expect("a floor");
        assert_eq!(floor.slots, 2 * UNION_BATCH_IDS + lane_threads + 1);
        assert_eq!(floor.slots, 37);
        // The call's term moves with the lane's slots too: k ids and the lane.
        let call = ArenaFloor::new(40, 1, ARENA_LANE_SLOTS, 1).expect("a floor");
        assert_eq!(call.slots, 40 + lane_threads + 1);
    }

    /// A plan of no rows under `budget`, for a planner [`plan_in_pool`]
    /// calls: only its budget is read.
    fn budget_plan<'a>(model: &'a ModelTensors, machine: &'a Machine, budget: u64) -> Plan<'a> {
        Plan {
            model,
            machine,
            ctx_max: 1,
            rows: Vec::new(),
            cards: Vec::new(),
            host: HostTotals {
                expert_bytes: 0,
                experts: 0,
                nvme_expert_bytes: 0,
                nvme_arena_bytes: 0,
                arena_columns: 1,
                table_bytes: 0,
                shadow_bytes: 0,
                reserve_bytes: 0,
                row_reserve_bytes: 0,
                pool_card_bytes: 0,
                headroom_bytes: 0,
                room: None,
            },
            nvme_bytes: 0,
            n_l: Vec::new(),
            tier_n_l: Vec::new(),
            card_budget: card_budget::CardBudget::Every(budget),
            budget_from_pool: false,
        }
    }

    /// [`plan_in_pool`] on a planner whose floor passes the room by
    /// `short(budget)` B: on a unified machine with no given budget each
    /// round lowers the budget by the shortfall the last refusal named, at
    /// least a granule, and the plan that fits is marked as the pool's, so
    /// the host's refusal does not name a lever no one set. A refusal other
    /// than the floor (on a machine with no node too), a discrete card, a
    /// given budget and a shortfall past the card's whole budget are the
    /// first plan's result; a lowered budget refused for another reason, a
    /// budget that runs out and a floor unmet after [`POOL_ROUNDS`] rounds
    /// are refused by name, with the rounds, the budget and the last refusal.
    #[test]
    fn a_unified_pool_lowers_the_card_budget_to_the_tiers_floor() {
        const ROOM: u64 = 100 << 30;
        const POOL: u64 = 64 << 30;
        let granule = workstation::GRANULE.get();
        let model = ModelTensors {
            tensors: Vec::new(),
            layers: 1,
            experts: 1,
            experts_used: 1,
            bf16_words: |_| false,
        };
        let pool = |unified: bool| {
            let mut c = card("GB10", 0..1, true, true);
            c.usable_bytes = 120 << 30;
            c.free_bytes = Some(POOL);
            Machine {
                cards: vec![c],
                tiers: Vec::new(),
                nodes: vec![Node {
                    host: Host {
                        usable_bytes: POOL,
                        reserves: Vec::new(),
                    },
                    unified: unified.then_some(UnifiedPool { bytes: POOL }),
                }],
            }
        };
        let floor = |short: u64| PlacementError::HostRoomFloor {
            room: ROOM,
            floor: ROOM + short,
            base: 0,
            layer: 0,
            need: 0,
            arena: None,
        };
        // Every budget the planner is called with, and its result: refused
        // by `short(budget)` B over the room, or for another reason at and
        // under `other_at`.
        let run = |m: &Machine, levers: &PlanLevers, short: &dyn Fn(u64) -> u64, other_at: u64| {
            let mut calls = Vec::new();
            let got = plan_in_pool(m, levers, |l| {
                let budget = capped(&m.cards[0], l.card_budget_bytes.every());
                calls.push(l.card_budget_bytes.every());
                if budget <= other_at {
                    return Err(PlacementError::HostRoom("other".to_string()));
                }
                match short(budget) {
                    0 => Ok(budget_plan(&model, m, budget)),
                    s => Err(floor(s)),
                }
            });
            (
                got.map(|p| (p.card_budget.every(), p.budget_from_pool)),
                calls,
            )
        };
        let (on, off) = (pool(true), pool(false));
        let unset = PlanLevers::default();
        let full = capped(&on.cards[0], None);
        assert_eq!(full, POOL + on.cards[0].context_in_free());

        // A shortfall that falls by half the bytes the budget gives up: the
        // first round lowers by it, the next by the half left, and a
        // shortfall under a granule by a granule.
        let fit = full - (8 << 30);
        let halves = |b: u64| b.saturating_sub(fit).div_ceil(2);
        let (got, calls) = run(&on, &unset, &halves, 0);
        let mut want = vec![None];
        let mut b = full;
        while halves(b) > 0 {
            b -= halves(b).max(granule);
            want.push(Some(b));
        }
        assert_eq!(calls, want);
        assert!(calls.len() > 3, "{calls:?}");
        assert_eq!(got.expect("the lowered plan"), (Some(b), true));
        let need = |from_pool| {
            let mut p = budget_plan(&model, &on, b);
            p.budget_from_pool = from_pool;
            workstation::HostNeed::of(&p, 0).card_budget
        };
        assert!(!need(true) && need(false));

        // The first plan's result: a discrete card, a given budget, a
        // shortfall past the card's whole budget.
        let given = PlanLevers {
            card_budget_bytes: Some(96 << 30).into(),
        };
        for (m, levers, short) in [(&off, &unset, 1), (&on, &given, 1), (&on, &unset, full + 1)] {
            let (got, calls) = run(m, levers, &|_| short, 0);
            assert!(matches!(got, Err(PlacementError::HostRoomFloor { .. })));
            assert_eq!(calls.len(), 1);
        }
        // A refusal before the floor is the first plan's result, even on a
        // machine with no node to read.
        let none = Machine {
            nodes: Vec::new(),
            ..pool(true)
        };
        let got = plan_in_pool(&none, &unset, |_| {
            Err(PlacementError::Nodes { nodes: 0, tiers: 0 })
        });
        assert!(matches!(got, Err(PlacementError::Nodes { nodes: 0, .. })));

        // Refused by name: another refusal of a lowered budget, a budget
        // that runs out, a floor still unmet after the rounds.
        let pooled = |got: Result<_, PlacementError>| match got {
            Err(PlacementError::PoolBudget {
                rounds,
                budget,
                last,
            }) => (rounds, budget, *last),
            other => panic!("not refused by name: {other:?}"),
        };
        let (rounds, budget, last) = pooled(run(&on, &unset, &|_| 1 << 30, full - (1 << 30)).0);
        assert_eq!((rounds, budget), (1, full - (1 << 30)));
        assert!(matches!(last, PlacementError::HostRoom(_)));
        let (rounds, budget, last) = pooled(run(&on, &unset, &|b| b / 2 + 1, 0).0);
        assert!(rounds > 1 && budget < full, "{rounds} {budget}");
        assert!(matches!(last, PlacementError::HostRoomFloor { .. }));
        let (got, calls) = run(&on, &unset, &|_| 1, 0);
        let (rounds, budget, last) = pooled(got);
        assert_eq!(rounds, POOL_ROUNDS);
        assert_eq!(budget, full - u64::from(POOL_ROUNDS) * granule);
        assert_eq!(calls.len(), 1 + POOL_ROUNDS as usize);
        assert!(matches!(last, PlacementError::HostRoomFloor { .. }));
        let text = PlacementError::PoolBudget {
            rounds,
            budget,
            last: Box::new(last),
        }
        .to_string();
        assert!(text.contains("lowered 64 times"), "{text}");
    }

    /// A card running `layers`, with the head and the token embedding as
    /// given; its byte figures play no part in the stage map.
    fn card(name: &str, layers: Range<usize>, head: bool, token_embedding: bool) -> Card {
        Card {
            name: name.to_string(),
            device: None,
            usable_bytes: 0,
            context_bytes: 0,
            scratch_bytes: 0,
            margin_bytes: 0,
            granule_bytes: NonZeroU64::MIN,
            free_bytes: None,
            held_by: None,
            layers,
            head,
            token_embedding,
            reserve_only: false,
            reserves: Vec::new(),
        }
    }

    /// The stage map's token-embedding rule: none, or the one card that runs
    /// layer 0; a card that does not run it, or a second card, is refused.
    #[test]
    fn token_embedding_is_the_layer_0_card_alone() {
        let map = |embed: [bool; 2]| {
            Stages::new(
                8,
                &[
                    card("card0", 0..5, false, embed[0]),
                    card("card1", 5..8, true, embed[1]),
                ],
            )
        };
        let refused = |embed: [bool; 2]| match map(embed) {
            Err(PlacementError::Stages(msg)) => msg,
            Err(e) => panic!("{embed:?}: {e}, want a stage-map refusal"),
            Ok(_) => panic!("{embed:?}: the stage map was accepted"),
        };
        let held = |embed: [bool; 2]| {
            map(embed)
                .unwrap_or_else(|e| panic!("{embed:?}: {e}"))
                .embedding
        };
        assert_eq!(held([false, false]), None);
        assert_eq!(held([true, false]), Some(0));
        assert_eq!(
            refused([false, true]),
            "card card1 holds the token embedding but does not run layer 0"
        );
        assert_eq!(
            refused([true, true]),
            "2 cards hold the token embedding, not one"
        );
    }

    /// A model of `layers` layers, each one routed q4_K stack of 8 experts of
    /// four 256-value rows (576 B an expert), with a q8_0 token embedding and
    /// head of 16 rows (4,352 B).
    fn layered(layers: usize) -> ModelTensors {
        let tensor = |name: String, layer, role, ty: GgmlType, rows: &[u64]| {
            let mut dims = vec![256];
            dims.extend_from_slice(rows);
            let blocks = 256 / ty.blck_size().expect("a sized type");
            ModelTensor {
                name,
                shard: 0,
                layer,
                role,
                ty,
                dims,
                file_bytes: ty.type_size().expect("a sized type")
                    * blocks
                    * rows.iter().product::<u64>(),
                gathered_rows: (role == Role::TokenEmbedding).then_some(1),
            }
        };
        let mut tensors = vec![tensor(
            "embedding".into(),
            None,
            Role::TokenEmbedding,
            GgmlType::Q8_0,
            &[16],
        )];
        for l in 0..layers {
            tensors.push(tensor(
                format!("experts {l}"),
                Some(l),
                Role::RoutedExperts,
                GgmlType::Q4_K,
                &[4, 8],
            ));
        }
        tensors.push(tensor(
            "head".into(),
            None,
            Role::Head,
            GgmlType::Q8_0,
            &[16],
        ));
        ModelTensors {
            tensors,
            layers,
            experts: 8,
            experts_used: 2,
            bf16_words: |_| false,
        }
    }

    const HEAD: u64 = 4_352;
    const EXPERT: u64 = 576;

    /// A BF16 head the loader places in MXFP8 planes is the one tensor the plan
    /// names as requantized: a head off the planes' 128-value block keeps its
    /// words, a Q8_0 head its own planes, a BF16 embedding is widened, and none
    /// of those changes the file's values. Mutant: `requantizes` true for every
    /// card format other than the file type's own, which lists the words-kept
    /// head as well.
    #[test]
    fn a_bf16_head_in_mxfp8_planes_is_named_as_requantized() {
        let tensor = |name: &str, role, ty, k: u64| ModelTensor {
            name: name.to_string(),
            shard: 0,
            layer: None,
            role,
            ty,
            dims: vec![k, 16],
            file_bytes: 0,
            gathered_rows: (role == Role::TokenEmbedding).then_some(1),
        };
        let model = |tensors: Vec<ModelTensor>| ModelTensors {
            tensors,
            layers: 0,
            experts: 8,
            experts_used: 2,
            bf16_words: |_| false,
        };
        let embed = tensor(
            "token_embd.weight",
            Role::TokenEmbedding,
            GgmlType::BF16,
            256,
        );
        let planes = model(vec![
            embed.clone(),
            tensor(HEAD_TENSOR, Role::Head, GgmlType::BF16, 256),
        ]);
        let listed = planes.requantized();
        assert_eq!(
            listed,
            vec![Requantized {
                name: HEAD_TENSOR.to_string(),
                file: GgmlType::BF16,
                card: CardFormat::Mxfp8Planes,
            }]
        );
        assert_eq!(
            listed[0].to_string(),
            format!("output.weight:{}:mxfp8_planes", GgmlType::BF16)
        );
        for (ty, k) in [(GgmlType::BF16, 320), (GgmlType::Q8_0, 256)] {
            let kept = model(vec![embed.clone(), tensor(HEAD_TENSOR, Role::Head, ty, k)]);
            assert!(kept.requantized().is_empty(), "a {ty} head of {k} values");
        }
    }

    /// The footprint the expert rule asks once an expert is the allocator's replay of every
    /// buffer in order, whatever counts it is asked at and in whatever order: fixed tensors
    /// below and above a granule between stacks whose buffers cross it as their counts grow.
    #[test]
    fn a_footprint_is_the_replay_of_every_buffer_at_any_counts() {
        let tensor = |name: &str, layer, role, ty: GgmlType, dims: Vec<u64>| ModelTensor {
            name: name.into(),
            shard: 0,
            layer,
            role,
            ty,
            dims,
            file_bytes: 0,
            gathered_rows: None,
        };
        let (layers, granule) = (3, NonZeroU64::new(4096).expect("nonzero"));
        let mut tensors = Vec::new();
        for l in 0..layers {
            tensors.push(tensor(
                "norm",
                Some(l),
                Role::Attention,
                GgmlType::F32,
                vec![256],
            ));
            tensors.push(tensor(
                "big",
                Some(l),
                Role::Attention,
                GgmlType::Q8_0,
                vec![256, 20],
            ));
            tensors.push(tensor(
                "small",
                Some(l),
                Role::Attention,
                GgmlType::Q8_0,
                vec![256, 3],
            ));
            tensors.push(tensor(
                "stack",
                Some(l),
                Role::RoutedExperts,
                GgmlType::Q4_K,
                vec![256, 8, 9],
            ));
        }
        let uploads: Vec<_> = tensors
            .iter()
            .map(|t| {
                let format = CardFormat::of(t.ty).expect("a card format");
                let held = match t.role {
                    Role::RoutedExperts => Held::Experts {
                        layer: t.layer.expect("a layer"),
                        rows_per_expert: 8,
                    },
                    _ => Held::Rows(rows_of(t)),
                };
                (t, format, held)
            })
            .collect();
        let replay = |n_l: &[u64]| {
            let buffers = uploads.iter().flat_map(|&(t, format, held)| {
                let rows = match held {
                    Held::Rows(rows) => rows,
                    Held::Experts {
                        layer,
                        rows_per_expert,
                    } => n_l[layer] * rows_per_expert,
                };
                if rows == 0 {
                    Vec::new()
                } else {
                    card_buffers(t, format, rows).expect("buffers")
                }
            });
            allocator_bytes(granule, buffers)
        };
        let footprint = Footprint::new(granule, &uploads);
        let mut n_l = vec![0u64; layers];
        let mut seed = 7u64;
        for ask in 0..400 {
            if ask < 27 {
                n_l[ask % layers] += 1;
            } else {
                seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                n_l[(seed >> 33) as usize % layers] = (seed >> 40) % 10;
            }
            assert_eq!(
                footprint.at(&n_l).expect("a footprint"),
                replay(&n_l),
                "{n_l:?}"
            );
        }
    }

    #[test]
    fn only_card_uploads_require_the_card_layout() {
        for width in [64, 160] {
            for (role, embedding, device) in [
                (Role::EngramTable, false, Device::Nvme),
                (Role::HashTable, false, Device::Host),
                (Role::TokenEmbedding, false, Device::Host),
                (Role::Unread, false, Device::Unused),
                (Role::Unused, false, Device::Unused),
                (Role::Attention, false, Device::Card(0)),
                (Role::TokenEmbedding, true, Device::Card(0)),
            ] {
                let mut model = layered(1);
                model.tensors[0] = ModelTensor {
                    name: "rows.weight".into(),
                    shard: 0,
                    layer: (role != Role::TokenEmbedding).then_some(0),
                    role,
                    ty: GgmlType::NVFP4_ROWS,
                    dims: vec![width, 5],
                    file_bytes: width / 16 * 9 * 5,
                    gathered_rows: Some(1),
                };
                let machine = Machine {
                    cards: vec![Card {
                        usable_bytes: 1 << 30,
                        ..card("card", 0..1, true, embedding)
                    }],
                    tiers: Vec::new(),
                    nodes: vec![Node {
                        host: host(),
                        unified: None,
                    }],
                };
                let planned = plan_with(&model, &machine, 4096, &NoKv, None);
                if width == 160 && device == Device::Card(0) {
                    let error = planned
                        .expect_err("the upload needs whole card blocks")
                        .to_string();
                    assert!(error.contains("rows.weight"), "{error}");
                    assert!(error.contains("Nvfp4Planes upload refuses"), "{error}");
                } else {
                    let plan = planned.unwrap_or_else(|e| panic!("{role:?} {width}: {e}"));
                    assert_eq!(plan.rows[0].segments[0].device, device);
                }
            }
        }
        // A routed stack the host planner keeps whole on the host takes no
        // card layout, while the expert rule's card segments of it are refused.
        let mut model = layered(1);
        model.tensors[1] = ModelTensor {
            ty: GgmlType::NVFP4_ROWS,
            dims: vec![160, 4, 8],
            file_bytes: 160 / 16 * 9 * 4 * 8,
            ..model.tensors[1].clone()
        };
        let machine = Machine {
            cards: vec![Card {
                usable_bytes: 1 << 30,
                ..card("card", 0..1, true, true)
            }],
            tiers: Vec::new(),
            nodes: vec![Node {
                host: host(),
                unified: None,
            }],
        };
        let plan = plan_host_routed(&model, &machine, 4096, &NoKv, &PlanLevers::default())
            .unwrap_or_else(|e| panic!("host-routed 160: {e}"));
        let devices: Vec<Device> = plan.rows[1].segments.iter().map(|s| s.device).collect();
        assert_eq!(devices, vec![Device::Host]);
        let error = plan_with(&model, &machine, 4096, &NoKv, None)
            .expect_err("card experts need whole card blocks")
            .to_string();
        assert!(error.contains("experts 0"), "{error}");
        assert!(error.contains("Nvfp4Planes upload refuses"), "{error}");
    }

    /// A model of 4 layers with every role's bf16 tensor (an attention
    /// matrix, a hyper-connection, a shared expert and its one-row gate, a
    /// router, an FFN norm, the head and the token embedding, 256-value rows)
    /// beside a routed q4_K stack, its readers' choice `bf16_words`, planned
    /// on `n` cards of `4 / n` layers each: every bf16 tensor's card segment
    /// is on the card that runs it, in the format `ModelTensors::card_format`
    /// names, at its format's bytes. With `bf16_words` the tensors of the
    /// dense roles and the router keep their words (the file's bytes); the
    /// norm, and every tensor without it, are f32 values (twice the file's
    /// bytes): the plan of a model that does not set it is the one it was.
    #[test]
    fn a_bf16_matrix_keeps_its_words_where_its_readers_read_them() {
        let bf16 = |name: String, layer, role, rows: &[u64]| {
            let mut dims = vec![256];
            dims.extend_from_slice(rows);
            ModelTensor {
                name,
                shard: 0,
                layer,
                role,
                ty: GgmlType::BF16,
                file_bytes: 2 * dims.iter().product::<u64>(),
                dims,
                gathered_rows: (role == Role::TokenEmbedding).then_some(1),
            }
        };
        assert_eq!(
            file_card_format(HEAD_TENSOR, GgmlType::BF16, 320),
            Some(CardFormat::Bf16Raw)
        );
        assert_eq!(
            file_card_format(HEAD_TENSOR, GgmlType::Q8_0, 256),
            Some(CardFormat::Q8_0Planes)
        );
        assert_eq!(
            file_card_format("token_embd.weight", GgmlType::BF16, 256),
            CardFormat::of(GgmlType::BF16)
        );
        let raw = [
            Role::Attention,
            Role::HyperConnection,
            Role::SharedExpert,
            Role::Head,
            Role::TokenEmbedding,
            Role::Router,
        ];
        for bf16_words in [false, true] {
            for n in [1usize, 2, 4] {
                let mut model = layered(4);
                model.bf16_words = if bf16_words { |_| true } else { |_| false };
                for l in 0..4 {
                    for (what, role, rows) in [
                        ("attn", Role::Attention, &[16][..]),
                        ("hc", Role::HyperConnection, &[4]),
                        ("shexp", Role::SharedExpert, &[8]),
                        ("shexp gate", Role::SharedExpert, &[]),
                        ("router", Role::Router, &[9]),
                        ("norm", Role::FfnNorm, &[]),
                    ] {
                        model
                            .tensors
                            .push(bf16(format!("{what} {l}"), Some(l), role, rows));
                    }
                }
                for (name, role) in [
                    (HEAD_TENSOR, Role::Head),
                    ("output_hc_up.weight", Role::Head),
                    ("bf16 embd", Role::TokenEmbedding),
                ] {
                    model.tensors.push(bf16(name.into(), None, role, &[16]));
                }
                let per = 4 / n;
                let cards = (0..n)
                    .map(|c| Card {
                        usable_bytes: 1 << 30,
                        ..card(
                            &format!("card{c}"),
                            c * per..(c + 1) * per,
                            c == n - 1,
                            c == 0,
                        )
                    })
                    .collect();
                let machine = Machine {
                    cards,
                    tiers: Vec::new(),
                    nodes: vec![Node {
                        host: host(),
                        unified: None,
                    }],
                };
                let plan = plan_with(&model, &machine, 4096, &NoKv, None)
                    .unwrap_or_else(|e| panic!("n={n} words={bf16_words}: {e}"));
                let mut seen = 0;
                for r in &plan.rows {
                    let t = &model.tensors[r.tensor];
                    if t.ty != GgmlType::BF16 {
                        continue;
                    }
                    seen += 1;
                    let (format, bytes) = if t.name == HEAD_TENSOR {
                        (
                            CardFormat::Mxfp8Planes,
                            t.file_bytes / 2 + t.file_bytes / 64,
                        )
                    } else if bf16_words && raw.contains(&t.role) {
                        (CardFormat::Bf16Raw, t.file_bytes)
                    } else {
                        (CardFormat::Bf16AsF32, 2 * t.file_bytes)
                    };
                    let card = match (t.layer, t.role) {
                        (Some(l), _) => l / per,
                        (None, Role::Head) => n - 1,
                        (None, _) => 0,
                    };
                    assert_eq!(model.card_format(t), Some(format), "{}", t.name);
                    let got: Vec<_> = r
                        .segments
                        .iter()
                        .map(|s| (s.device, s.format, s.experts.is_none(), s.resident_bytes))
                        .collect();
                    assert_eq!(
                        got,
                        [(Device::Card(card), Format::Card(format), true, bytes)],
                        "n={n} words={bf16_words} {}",
                        t.name
                    );
                }
                assert_eq!(seen, 4 * 6 + 3, "n={n}");
            }
        }
    }

    /// A card of `usable` bytes and nothing else set aside.
    fn bytes_card(name: &str, usable: u64, layers: Range<usize>) -> Card {
        Card {
            usable_bytes: usable,
            ..card(name, layers.clone(), !layers.is_empty(), false)
        }
    }

    fn host() -> Host {
        Host {
            usable_bytes: 1 << 40,
            reserves: Vec::new(),
        }
    }

    #[test]
    fn card_sums_preserve_the_u64_boundary() {
        let boundary = Card {
            context_bytes: 2,
            scratch_bytes: 3,
            reserves: vec![("beside".into(), u64::MAX - 5)],
            ..bytes_card("boundary", u64::MAX, 0..1)
        };
        assert_eq!(boundary.reserve_bytes(), Some(u64::MAX - 5));
        assert_eq!(boundary.set_aside_bytes(), Some(u64::MAX));
        assert_eq!(boundary.floor_bytes(0, 0, 0), Some(u64::MAX));
        for (dense, kv, past) in [(1, 0, 0), (0, 1, 0), (0, 0, 1)] {
            assert_eq!(boundary.floor_bytes(dense, kv, past), None);
        }
        for (context, scratch, reserves, reserve_sum) in [
            (0, 0, vec![u64::MAX, 1], None),
            (1, 0, vec![u64::MAX], Some(u64::MAX)),
            (0, 1, vec![u64::MAX], Some(u64::MAX)),
            (u64::MAX, 1, vec![], Some(0)),
        ] {
            let card = Card {
                context_bytes: context,
                scratch_bytes: scratch,
                reserves: reserves.into_iter().map(|b| ("beside".into(), b)).collect(),
                ..boundary.clone()
            };
            assert_eq!(card.reserve_bytes(), reserve_sum);
            assert_eq!(card.set_aside_bytes(), None);
            assert_eq!(card.floor_bytes(0, 0, 0), None);
        }
    }

    #[test]
    fn host_reserve_sum_preserves_the_u64_boundary() {
        let host = |reserves: &[u64]| Host {
            usable_bytes: u64::MAX,
            reserves: reserves.iter().map(|&b| ("beside".into(), b)).collect(),
        };
        assert_eq!(host(&[]).reserve_bytes(), Some(0));
        assert_eq!(host(&[u64::MAX]).reserve_bytes(), Some(u64::MAX));
        assert_eq!(host(&[u64::MAX - 5, 5]).reserve_bytes(), Some(u64::MAX));
        assert_eq!(host(&[u64::MAX, 1]).reserve_bytes(), None);
        assert_eq!(host(&[1 << 63, 1 << 63]).reserve_bytes(), None);
    }

    /// A host whose reserves sum past u64 bytes wraps to a small number
    /// under unchecked addition (release) or panics (debug): the plan then
    /// passes the host bound or never returns. It is a named refusal.
    #[test]
    fn overflowing_host_reserves_are_a_named_placement_refusal() {
        let model = layered(3);
        for reserves in [
            vec![u64::MAX, 1],
            vec![1 << 63, 1 << 63],
            vec![u64::MAX - 5, 3, 3],
        ] {
            let machine = Machine {
                cards: vec![bytes_card("stage", HEAD + 12 * EXPERT, 0..3)],
                tiers: Vec::new(),
                nodes: vec![Node {
                    host: Host {
                        usable_bytes: 1 << 40,
                        reserves: reserves.iter().map(|&b| ("beside".into(), b)).collect(),
                    },
                    unified: None,
                }],
            };
            assert!(
                matches!(
                    plan_with(&model, &machine, 4096, &NoKv, None),
                    Err(PlacementError::HostBytesPastU64)
                ),
                "{reserves:?}"
            );
        }
    }

    /// Reserves that fit u64 alone but not beside the plan's host experts:
    /// the host's need passes every room, so the plan is a `HostOver`, not a
    /// wrapped small need that fits.
    #[test]
    fn a_host_need_past_u64_is_a_host_over() {
        let model = layered(3);
        let machine = Machine {
            cards: vec![bytes_card("stage", HEAD + 12 * EXPERT, 0..3)],
            tiers: Vec::new(),
            nodes: vec![Node {
                host: Host {
                    usable_bytes: 1 << 40,
                    reserves: vec![("beside".into(), u64::MAX - 100)],
                },
                unified: None,
            }],
        };
        let plan = plan_with(&model, &machine, 4096, &NoKv, None).expect("the plan");
        assert!(plan.host.expert_bytes > 100, "the host holds experts");
        let over: Vec<_> = plan
            .violations()
            .into_iter()
            .filter(|v| matches!(v, Violation::HostOver { .. }))
            .collect();
        assert_eq!(
            over,
            vec![Violation::HostOver {
                total: u64::MAX,
                usable: 1 << 40
            }]
        );
        assert!(workstation::HostNeed::of(&plan, 0).check(u64::MAX).is_err());
    }

    /// A host that holds eight layers of eight 1 TiB experts each, with a
    /// reserve of `u64::MAX − 5 W` beside them (`W`, a layer's bytes: 8
    /// experts): the host's need is `u64::MAX + 3 W` and a bit, past u64
    /// bytes, while the split's floor `base + 3 W` is `u64::MAX − 2 W`, under
    /// it.
    fn host_need_past_u64() -> (ModelTensors, Machine, u64) {
        const BIG: u64 = 1 << 40;
        let w = 8 * BIG;
        let mut model = layered(8);
        for t in model
            .tensors
            .iter_mut()
            .filter(|t| t.role == Role::RoutedExperts)
        {
            t.file_bytes = 8 * BIG;
        }
        let machine = Machine {
            cards: vec![bytes_card("stage", HEAD, 0..8)],
            tiers: Vec::new(),
            nodes: vec![Node {
                host: Host {
                    usable_bytes: u64::MAX,
                    reserves: vec![("beside".into(), u64::MAX - 5 * w)],
                },
                unified: None,
            }],
        };
        (model, machine, BIG)
    }

    /// The split's room arithmetic is exact where the host's need passes u64
    /// bytes: the host keeps what the room leaves past the exact need and the
    /// window, within one expert, and the rest moves to the NVMe tier. An
    /// unchecked `need + window − room` wraps in release (a small, wrong
    /// `keep`) and panics in debug.
    #[test]
    fn an_nvme_split_of_a_need_past_u64_is_exact() {
        let (model, machine, expert) = host_need_past_u64();
        let w = 8 * expert;
        let mut plan = plan_with(&model, &machine, 4096, &NoKv, None).expect("the plan");
        let held = plan.host.expert_bytes;
        assert_eq!(held, 8 * w, "every expert is on the host");
        let need = workstation::HostNeed::of(&plan, 0);
        assert!(
            need.exact(true) > u128::from(u64::MAX),
            "the need passes u64"
        );
        let room = u64::MAX - w;
        expert_nvme_tier(&mut plan, room, None).expect("the split");
        let keep = u128::from(held) - (need.exact(true) + u128::from(2 * w) - u128::from(room));
        let kept = u128::from(plan.host.expert_bytes);
        assert!(
            kept <= keep && kept + u128::from(expert) > keep,
            "the host keeps {kept} B of {keep}"
        );
        assert_eq!(plan.host.nvme_expert_bytes, held - plan.host.expert_bytes);
        assert_eq!(plan.nvme_bytes, plan.host.nvme_expert_bytes);
    }

    /// A room of `u64::MAX` does not hold a need past u64 bytes: the split
    /// runs. `HostNeed::bytes` reads `u64::MAX` for that need, which compared
    /// to the room would leave the plan whole.
    #[test]
    fn a_room_of_u64_max_does_not_hold_a_need_past_u64() {
        let (model, machine, _) = host_need_past_u64();
        let mut plan = plan_with(&model, &machine, 4096, &NoKv, None).expect("the plan");
        let held = plan.host.expert_bytes;
        assert_eq!(workstation::HostNeed::of(&plan, 0).bytes(), u64::MAX);
        expert_nvme_tier(&mut plan, u64::MAX, None).expect("the split");
        assert!(
            plan.host.expert_bytes < held && plan.host.nvme_expert_bytes > 0,
            "the split moved experts to the NVMe tier"
        );
    }

    #[test]
    fn overflowing_card_sums_are_named_placement_refusals() {
        let model = layered(1);
        for (context, scratch, reserves) in [
            (0, 0, vec![u64::MAX, 1]),
            (1, 0, vec![u64::MAX]),
            (0, 1, vec![u64::MAX]),
            (u64::MAX, 1, vec![]),
        ] {
            for free_bytes in [None, Some(u64::MAX - 1)] {
                let machine = Machine {
                    cards: vec![Card {
                        context_bytes: context,
                        scratch_bytes: scratch,
                        reserves: reserves.iter().map(|&b| ("beside".into(), b)).collect(),
                        free_bytes,
                        ..bytes_card("overflow", u64::MAX, 0..1)
                    }],
                    tiers: Vec::new(),
                    nodes: vec![Node {
                        host: host(),
                        unified: None,
                    }],
                };
                for budget in [None, Some(u64::MAX)] {
                    assert!(matches!(
                        plan_with(&model, &machine, 4096, &NoKv, budget),
                        Err(PlacementError::CardBytesPastU64 { card }) if card == "overflow"
                    ));
                }
                assert!(matches!(
                    whole_need(&model, &machine.cards[0], 0, WholeLoad::Unchecked),
                    Err(PlacementError::CardBytesPastU64 { card }) if card == "overflow"
                ));
            }
        }
        // The set-aside itself fits; adding the actual uploads does not.
        let machine = Machine {
            cards: vec![Card {
                reserves: vec![("beside".into(), u64::MAX)],
                ..bytes_card("load-overflow", u64::MAX, 0..1)
            }],
            tiers: Vec::new(),
            nodes: vec![Node {
                host: host(),
                unified: None,
            }],
        };
        assert!(matches!(
            plan_with(&model, &machine, 4096, &NoKv, None),
            Err(PlacementError::CardBytesPastU64 { card }) if card == "load-overflow"
        ));
        assert!(matches!(
            whole_need(&model, &machine.cards[0], 0, WholeLoad::Unchecked),
            Err(PlacementError::CardBytesPastU64 { card }) if card == "load-overflow"
        ));
    }

    #[test]
    fn pooled_card_sums_are_checked_only_for_shared_memory() {
        let model = layered(2);
        for unified in [None, Some(UnifiedPool { bytes: u64::MAX })] {
            let machine = Machine {
                cards: (0..2)
                    .map(|i| Card {
                        head: i == 1,
                        reserves: vec![("resident".into(), 1 << 63)],
                        ..bytes_card(&format!("stage{i}"), u64::MAX, i..i + 1)
                    })
                    .collect(),
                tiers: Vec::new(),
                nodes: vec![Node {
                    host: host(),
                    unified,
                }],
            };
            let result = plan_with(&model, &machine, 4096, &NoKv, None);
            if unified.is_some() {
                assert!(matches!(
                    result,
                    Err(PlacementError::CardBytesPastU64 { card }) if card == "stage1"
                ));
            } else {
                let plan = result.expect("separate card memories need no combined card total");
                assert_eq!(plan.n_l, vec![8, 8]);
                assert!(plan.violations().is_empty());
            }
        }
    }

    /// A tier after a stage card that holds every expert is refused by name;
    /// a tier with experts left to take, one whose budget leaves it none
    /// while the host keeps the rest, and the host-routed planner, which
    /// refuses a tier card by name.
    #[test]
    fn a_tier_with_nothing_left_is_refused() {
        let model = layered(3);
        let machine = |stage: u64, tier: u64| Machine {
            cards: vec![bytes_card("stage", HEAD + stage * EXPERT, 0..3)],
            tiers: vec![bytes_card("tier", tier * EXPERT, 0..0)],
            nodes: vec![Node {
                host: host(),
                unified: None,
            }],
        };
        let whole = machine(24, 4);
        match plan_with(&model, &whole, 4096, &NoKv, None) {
            Err(e @ PlacementError::IdleTier { .. }) => assert_eq!(
                e.to_string(),
                "tier card tier (tier 0) holds no expert: the cards before it hold every routed \
                 expert of the layers whose stacks a card can hold; plan without it"
            ),
            Err(e) => panic!("{e}, not IdleTier"),
            Ok(_) => panic!("the idle tier was planned"),
        }
        let part = machine(5, 4);
        let plan = plan_with(&model, &part, 4096, &NoKv, None).expect("a tier with work");
        assert_eq!(plan.tier_n_l, vec![vec![1, 1, 2]]);
        let small = machine(5, 0);
        let plan = plan_with(&model, &small, 4096, &NoKv, None).expect("a tier too small");
        assert_eq!(plan.tier_n_l, vec![vec![0, 0, 0]]);
        assert!(plan.host.experts > 0);
        match plan_host_routed(&model, &whole, 4096, &NoKv, &PlanLevers::default()) {
            Err(e @ PlacementError::HostRoutedTier { .. }) => assert_eq!(
                e.to_string(),
                "tier card tier (tier 0) holds no expert: this plan puts every routed expert \
                 on the host; plan without it"
            ),
            Err(e) => panic!("{e}, not HostRoutedTier"),
            Ok(_) => panic!("the host-routed planner accepted a tier card"),
        }
    }

    /// A tier card on a node of its own ([`workstation::node_of`])
    /// takes nothing from this node's pool: the same plan's pool need is the
    /// stage card's and the host's alone, so a pool of exactly that plans clean
    /// while the tier on this node's pool passes it by the tier card's bytes.
    /// A machine of other than one node, or one more a tier card, is refused
    /// by name.
    #[test]
    fn a_tier_on_its_own_node_takes_nothing_of_this_nodes_pool() {
        let model = layered(3);
        let tier = 4 * EXPERT;
        let node = |bytes: u64| Node {
            host: Host {
                usable_bytes: bytes,
                reserves: Vec::new(),
            },
            unified: Some(UnifiedPool { bytes }),
        };
        let machine = |pool: u64, own: bool| Machine {
            cards: vec![bytes_card("stage", HEAD + 5 * EXPERT, 0..3)],
            tiers: vec![bytes_card("tier", tier, 0..0)],
            nodes: if own {
                vec![node(pool), node(tier)]
            } else {
                vec![node(pool)]
            },
        };
        let planned = |pool: u64, own: bool| {
            let m = machine(pool, own);
            let plan = plan_with(&model, &m, 4096, &NoKv, None).expect("plans");
            assert_eq!(plan.tier_n_l, vec![vec![1, 1, 2]], "own node {own}");
            let need = plan.pool_need().expect("a unified pool");
            (plan.host.pool_card_bytes, need, plan.violations())
        };
        let (shared_cards, shared, _) = planned(1 << 40, false);
        let (own_cards, own, _) = planned(1 << 40, true);
        assert_eq!(shared_cards - own_cards, tier);
        assert_eq!(shared.cards - own.cards, tier);
        assert_eq!(shared.host, own.host);
        let (_, _, v) = planned(own.cards + own.host, true);
        assert!(v.is_empty(), "{v:?}");
        match planned(own.cards + own.host, false).2.as_slice() {
            [Violation::PoolOver { cards, .. }] => assert_eq!(*cards, shared.cards),
            v => panic!("{v:?}, not one PoolOver"),
        }
        // A census tier card's context comes out of its own node's pool.
        for (own, pool) in [(true, 1 << 40), (false, (1 << 40) + 1)] {
            let mut m = machine(1 << 40, own);
            m.tiers[0].context_bytes = 1;
            m.tiers[0].free_bytes = Some(tier);
            assert_eq!(
                m.pool_bytes().expect("a pool"),
                Some(pool),
                "own node {own}"
            );
        }
        // What the caller plans beside each card comes out of the pool of
        // that card's node.
        for (own, cards) in [(true, 1), (false, 2)] {
            let m = machine(1 << 40, own);
            let plan = plan_routed_reserving(
                &model,
                &m,
                4096,
                &NoKv,
                &PlanLevers::default(),
                RoutedRule::PerType(CardFormat::of_routed),
                1,
            )
            .expect("plans");
            let need = plan.pool_need().expect("a unified pool");
            assert_eq!(
                plan.host.pool_card_bytes - need.cards,
                cards,
                "own node {own}"
            );
        }
        let two = Machine {
            tiers: vec![bytes_card("tier", tier, 0..0); 2],
            ..machine(1 << 40, true)
        };
        match plan_with(&model, &two, 4096, &NoKv, None) {
            Err(e @ PlacementError::Nodes { nodes: 2, tiers: 2 }) => assert_eq!(
                e.to_string(),
                "a machine of 2 nodes for 2 tier cards: the card plan plans one node, or one more \
                 a tier card; plan the others with placement::nodes"
            ),
            Err(e) => panic!("{e}, not Nodes"),
            Ok(_) => panic!("two tiers on one remote node planned"),
        }
    }

    /// A layer's host experts are the model's less the stage card's and the
    /// tier's, they sum to the plan's host experts, a layer the two cards
    /// hold whole hosts none, and a layer past the plan's counts or cards
    /// that hold more than the model has are refused by name.
    #[test]
    fn a_layers_host_experts_are_the_models_less_the_cards() {
        let model = layered(3);
        let machine = |stage: u64, tier: u64| Machine {
            cards: vec![bytes_card("stage", HEAD + stage * EXPERT, 0..3)],
            tiers: vec![bytes_card("tier", tier * EXPERT, 0..0)],
            nodes: vec![Node {
                host: host(),
                unified: None,
            }],
        };
        let hosted = |plan: &Plan<'_>| -> Vec<u64> {
            (0..model.layers)
                .map(|l| plan.host_experts(l).expect("a layer of the plan"))
                .collect()
        };
        let (small, big) = (machine(5, 4), machine(20, 4));
        let part = plan_with(&model, &small, 4096, &NoKv, None).expect("a tier with work");
        let want: Vec<u64> = (0..model.layers)
            .map(|l| model.experts - part.n_l[l] - part.tier_n_l[0][l])
            .collect();
        assert_eq!(hosted(&part), want);
        assert_eq!(hosted(&part).iter().sum::<u64>(), part.host.experts);
        assert!(part.host.experts > 0);
        let whole = plan_with(&model, &big, 4096, &NoKv, None).expect("two cards");
        assert_eq!(hosted(&whole), vec![0; model.layers]);
        assert_eq!(whole.host.experts, 0);
        match part.host_experts(model.layers) {
            Err(e @ PlacementError::LayerPastCounts { .. }) => assert_eq!(
                e.to_string(),
                "layer 3 is past the plan's per-layer expert counts (3 layers)"
            ),
            other => panic!("{other:?}, not LayerPastCounts"),
        }
        let mut over = part.clone();
        over.n_l[1] = 5;
        over.tier_n_l[0][1] = 4;
        match over.host_experts(1) {
            Err(e @ PlacementError::LayerOverHeld { .. }) => assert_eq!(
                e.to_string(),
                "layer 1: the stage card holds 5 experts and the tier cards 4, past the \
                 model's 8"
            ),
            other => panic!("{other:?}, not LayerOverHeld"),
        }
    }

    #[test]
    fn a_node_host_projection_keeps_non_routed_rows_on_node_zero() {
        let model = layered(3);
        let machine = Machine {
            cards: vec![bytes_card("stage", HEAD + 5 * EXPERT, 0..3)],
            tiers: vec![bytes_card("tier", 4 * EXPERT, 0..0)],
            nodes: vec![Node {
                host: host(),
                unified: None,
            }],
        };
        let plan = plan_with(&model, &machine, 4096, &NoKv, None).expect("a tier with work");
        let remote = Machine {
            cards: Vec::new(),
            tiers: Vec::new(),
            nodes: machine.nodes.clone(),
        };
        let projected = nodes::routed_host(&model, &plan.rows, 1, &remote, 4)
            .expect("the routed tier's host projection");
        assert!(std::ptr::eq(projected.machine, &remote));
        assert_eq!(projected.host.arena_columns, 4);
        assert_eq!(projected.host.experts, plan.tier_n_l[0].iter().sum::<u64>());
        assert!(projected.host.experts > 0);
        assert_eq!(projected.host.table_bytes, 0);
        assert_eq!(projected.host.pool_card_bytes, 0);
        let mut bytes = 0;
        assert_eq!(projected.rows.len(), model.tensors.len());
        for (index, row) in projected.rows.iter().enumerate() {
            assert_eq!(row.tensor, index);
            let t = &model.tensors[row.tensor];
            if row.segments[0].device == Device::Unused {
                continue;
            }
            assert_eq!(t.role, Role::RoutedExperts);
            let source = plan.rows.iter().find(|r| r.tensor == row.tensor).unwrap();
            let assigned = source
                .segments
                .iter()
                .find(|s| s.device == Device::Card(1))
                .unwrap();
            assert_eq!(row.segments.len(), 1);
            let held = &row.segments[0];
            assert_eq!((held.device, held.format), (Device::Host, Format::HostFile));
            assert_eq!(held.experts, assigned.experts);
            bytes += held.resident_bytes;
        }
        assert_eq!(projected.host.expert_bytes, bytes);
        let stage = nodes::routed_host(&model, &plan.rows, 0, &remote, 1).unwrap();
        assert!(
            stage
                .rows
                .iter()
                .filter(|r| r.segments[0].device != Device::Unused)
                .all(|r| model.tensors[r.tensor].role == Role::RoutedExperts)
        );
        assert_eq!(stage.host.experts, plan.n_l.iter().sum::<u64>());
        let mut changed = plan.rows.clone();
        let mut extra = changed
            .iter()
            .find(|r| r.segments.iter().any(|s| s.device == Device::Card(1)))
            .unwrap()
            .clone();
        let segment = extra
            .segments
            .iter_mut()
            .find(|s| s.device == Device::Card(1))
            .unwrap();
        let ids = segment.experts.as_ref().unwrap();
        let others = (0..model.experts as u32)
            .filter(|id| !ids.ids().contains(id))
            .take(ids.len() as usize)
            .collect();
        segment.experts = Some(ExpertList::new(others, model.experts).unwrap());
        changed.push(extra);
        assert!(
            nodes::routed_host(&model, &changed, 1, &remote, 1)
                .unwrap_err()
                .to_string()
                .contains("different expert lists")
        );
        let empty = nodes::routed_host(&model, &plan.rows, 2, &remote, 1).unwrap();
        assert!(
            empty
                .rows
                .iter()
                .all(|r| r.segments[0].device == Device::Unused)
        );
        assert_eq!(empty.host.expert_bytes, 0);

        let mut pool_machine = machine.clone();
        pool_machine.nodes[0].unified = Some(UnifiedPool { bytes: 1 << 30 });
        pool_machine.tiers[0].scratch_bytes = EXPERT;
        let mut lowered = plan;
        lowered.machine = &pool_machine;
        let half = model.experts / 2;
        nodes::lower_routed(&mut lowered, &[half, model.experts - half]).unwrap();
        assert_eq!(
            lowered.tier_n_l,
            vec![vec![model.experts - half; model.layers]]
        );
        assert!(lowered.n_l.iter().all(|&n| n <= half));
        assert_eq!(lowered.cards[1].expert_bytes, 0);
        assert!(
            lowered.violations().is_empty(),
            "{:?}",
            lowered.violations()
        );
        let stage = &lowered.cards[0];
        assert_eq!(
            lowered.pool_need().unwrap().cards,
            stage.dense_bytes + stage.expert_bytes + stage.rounding_bytes
        );
        assert_eq!(
            lowered.host.pool_card_bytes,
            stage.dense_bytes + stage.expert_bytes + stage.rounding_bytes
        );
        let projected = nodes::routed_host(&model, &lowered.rows, 1, &remote, 4).unwrap();
        assert!(
            projected.host.experts
                > plan_with(&model, &machine, 4096, &NoKv, None)
                    .unwrap()
                    .cards[1]
                    .experts
        );
        for row in &lowered.rows {
            let t = &model.tensors[row.tensor];
            if t.role == Role::RoutedExperts {
                let assigned = row
                    .segments
                    .iter()
                    .find(|s| s.device == Device::Card(1))
                    .unwrap();
                assert_eq!(
                    assigned.experts.as_ref().unwrap().ids(),
                    (half as u32..model.experts as u32).collect::<Vec<_>>()
                );
                assert_eq!(
                    (assigned.format, assigned.resident_bytes),
                    (Format::HostFile, 0)
                );
            }
        }
        assert!(nodes::lower_routed(&mut lowered, &[model.experts, 0]).is_err());
        assert!(nodes::lower_routed(&mut lowered, &[half, half + 1]).is_err());
        // An explicit ownership seeds the stage alone on node 0, whether
        // the tier shares its pool or sits on a node of its own.
        let mut apart = pool_machine.clone();
        apart.nodes.push(Node {
            host: Host {
                usable_bytes: 1 << 30,
                reserves: Vec::new(),
            },
            unified: Some(UnifiedPool { bytes: 1 << 30 }),
        });
        for m in [&pool_machine, &apart] {
            let owned = nodes::loaded_routed(
                &model,
                m,
                4096,
                &NoKv,
                &PlanLevers::default(),
                RoutedRule::PerType(CardFormat::of_routed),
                0,
                Some(&[half, model.experts - half]),
            )
            .unwrap_or_else(|e| panic!("{} nodes: {e}", m.nodes.len()));
            assert_eq!(owned.tier_n_l, lowered.tier_n_l, "{} nodes", m.nodes.len());
        }

        for ty in [
            GgmlType::Q5_1,
            GgmlType::IQ3_XXS,
            GgmlType::IQ4_XS,
            GgmlType::IQ4_NL,
        ] {
            let mut blocks = model.clone();
            for t in blocks
                .tensors
                .iter_mut()
                .filter(|t| t.role == Role::RoutedExperts)
            {
                t.ty = ty;
                t.file_bytes = t.dims.iter().product::<u64>() / ty.blck_size().unwrap()
                    * ty.type_size().unwrap();
            }
            let mut seed = machine.clone();
            seed.tiers.clear();
            for budget in [None, Some(HEAD)] {
                let mut plan = plan_routed(
                    &blocks,
                    &seed,
                    4096,
                    &NoKv,
                    &PlanLevers {
                        card_budget_bytes: budget.into(),
                    },
                    RoutedRule::PerType(|_| Some(CardFormat::KQuant)),
                )
                .unwrap();
                if budget.is_some() {
                    assert_eq!(plan.n_l, vec![0; blocks.layers]);
                }
                plan.machine = &machine;
                for owned in [[0, model.experts], [half, half]] {
                    let mut lowered = plan.clone();
                    nodes::lower_routed(&mut lowered, &owned)
                        .unwrap_or_else(|e| panic!("{ty}, {budget:?}, {owned:?}: {e}"));
                    assert!(lowered.violations().is_empty());
                    for row in &lowered.rows {
                        let t = &blocks.tensors[row.tensor];
                        if t.role != Role::RoutedExperts {
                            let original = &plan.rows[row.tensor];
                            assert_eq!(
                                (row.tensor, row.stage, row.read_bytes),
                                (original.tensor, original.stage, original.read_bytes)
                            );
                            assert_eq!(row.segments.len(), original.segments.len());
                            for (a, b) in row.segments.iter().zip(&original.segments) {
                                assert_eq!(
                                    (a.device, a.format, &a.experts, a.resident_bytes),
                                    (b.device, b.format, &b.experts, b.resident_bytes)
                                );
                            }
                            continue;
                        }
                        assert_eq!(row.read_bytes, plan.rows[row.tensor].read_bytes);
                        assert_eq!(row.stage, plan.rows[row.tensor].stage);
                        let kept = lowered.n_l[t.layer.unwrap()];
                        let remote = row
                            .segments
                            .iter()
                            .find(|s| s.device == Device::Card(1))
                            .unwrap();
                        assert_eq!(
                            (remote.format, remote.resident_bytes),
                            (Format::HostFile, 0)
                        );
                        assert_eq!(
                            remote.experts,
                            Some(ExpertList::range(owned[0]..model.experts).unwrap())
                        );
                        let card = row.segments.iter().find(|s| s.device == Device::Card(0));
                        assert_eq!(card.is_some(), kept != 0);
                        if let Some(card) = card {
                            assert_eq!(card.format, Format::Card(CardFormat::KQuant));
                        }
                        let host = row.segments.iter().find(|s| s.device == Device::Host);
                        assert_eq!(host.is_some(), kept < owned[0]);
                        if let Some(host) = host {
                            assert_eq!(
                                host.experts,
                                Some(ExpertList::range(kept..owned[0]).unwrap())
                            );
                            assert_eq!(host.format, Format::HostFile);
                            assert_eq!(
                                host.resident_bytes,
                                (owned[0] - kept) * t.file_bytes / model.experts
                            );
                        }
                    }
                    assert_eq!(lowered.cards[1].expert_bytes, 0);
                }
            }
        }
    }

    /// [`layered`]'s embedding and head over one layer of routed stacks a
    /// row of `types` (gate, up, down), each named by the tail a file gives it
    /// (the part is read from that tail, the layer from `layer`).
    fn legged(types: &[[GgmlType; 3]]) -> ModelTensors {
        let mut model = layered(types.len());
        model.tensors.retain(|t| t.role != Role::RoutedExperts);
        for (l, parts) in types.iter().enumerate() {
            for (part, &ty) in ["gate", "up", "down"].into_iter().zip(parts) {
                let blocks = 256 / ty.blck_size().expect("a sized type");
                model.tensors.push(ModelTensor {
                    name: format!("ffn_{part}_exps.weight"),
                    shard: 0,
                    layer: Some(l),
                    role: Role::RoutedExperts,
                    ty,
                    dims: vec![256, 4, 8],
                    file_bytes: ty.type_size().expect("a sized type") * blocks * 32,
                    gathered_rows: None,
                });
            }
        }
        model
    }

    /// An nvfp4 stack loads in its planes under the default expert rule, and
    /// its card segment counts one f32 scale an expert it holds after the
    /// planes, where a dense nvfp4 tensor's segment holds its one scale. A
    /// split that leaves some experts of it on the host and the rest on a card
    /// is refused by name (the card's planes are not the file's blocks a swap
    /// copies); the whole stack on the host is its file bytes there, which the
    /// host's nvfp4 dot reads.
    #[test]
    fn an_nvfp4_stack_holds_a_scale_an_expert_and_is_not_split() {
        use GgmlType::NVFP4;
        assert_eq!(CardFormat::of_routed(NVFP4), Some(CardFormat::Nvfp4Planes));
        assert_eq!(CardFormat::of_routed(GgmlType::Q5_K), None);
        let model = legged(&[[NVFP4; 3]]);
        let i = model
            .tensors
            .iter()
            .position(|t| t.role == Role::RoutedExperts)
            .expect("a stack");
        let t = &model.tensors[i];
        let list = |ids: &[u32]| ExpertList::new(ids.to_vec(), 8).expect("a list");
        let row = routed_row_on(
            i,
            t,
            0,
            vec![(0, list(&[6, 1, 4])), (1, list(&[0, 2, 3, 5, 7]))],
            &model,
        )
        .expect("every expert on a card");
        // Three experts of four rows of 256 values: 32 code words and 4 scale
        // words a row, then three scales.
        let held = [12 * 128, 12 * 16, 3 * 4];
        let seg = &row.segments[0];
        assert_eq!(seg.buffer_bytes(t, model.experts).expect("buffers"), held);
        assert_eq!(seg.resident_bytes, held.iter().sum::<u64>());
        let dense = ModelTensor {
            dims: vec![256, 4],
            role: Role::Attention,
            ..t.clone()
        };
        // PIN(2026-10-20): a dense nvfp4 tensor now holds its one scale for the GEMM's store; it was the planes alone.
        assert_eq!(
            card_buffers(&dense, CardFormat::Nvfp4Planes, 4).expect("buffers"),
            [4 * 128, 4 * 16, 4]
        );
        let err = routed_row(i, t, 0, list(&[0, 1]), &model)
            .expect_err("a split between a card and the host")
            .to_string();
        assert!(
            err.contains("ffn_gate_exps.weight") && err.contains("every one on the host"),
            "{err}"
        );
        let host = routed_row_on(i, t, 0, Vec::new(), &model).expect("every expert on the host");
        let [seg] = host.segments.as_slice() else {
            panic!("one host segment, got {:?}", host.segments);
        };
        assert_eq!(
            (seg.device, seg.format, seg.resident_bytes),
            (Device::Host, Format::HostFile, t.file_bytes)
        );
        assert_eq!(seg.experts.as_ref().map(ExpertList::len), Some(8));
    }

    /// A card leg judges a layer by its parts: Qwen3-30B's leg (a Q4_K or
    /// Q5_K gate·up, a Q4_K, Q5_K or Q6_K down) cards a Q4_K gate and up over
    /// a Q6_K down, and keeps on the host a Q6_K gate and up (a type only its
    /// down reads) and a gate and up of two types — the layers its load
    /// refuses — where the per-type rule over the union of its types cards
    /// all four. A routed stack whose name names no part is refused by name
    /// under the leg and placed under the per-type rule.
    #[test]
    fn a_leg_cards_a_layer_by_its_parts() {
        use GgmlType::{Q4_K, Q5_K, Q6_K};
        let gate_up: RoutedFormat = |ty| matches!(ty, Q4_K | Q5_K).then_some(CardFormat::KQuant);
        let down: RoutedFormat =
            |ty| matches!(ty, Q4_K | Q5_K | Q6_K).then_some(CardFormat::KQuant);
        let leg = RoutedRule::Leg { gate_up, down };
        let union = RoutedRule::PerType(down);
        let machine = |layers: usize| Machine {
            cards: vec![bytes_card("stage", 1 << 30, 0..layers)],
            tiers: Vec::new(),
            nodes: vec![Node {
                host: host(),
                unified: None,
            }],
        };
        let (m1, m4) = (machine(1), machine(4));
        let levers = PlanLevers::default();
        let model = legged(&[
            [Q4_K, Q4_K, Q6_K],
            [Q6_K, Q6_K, Q6_K],
            [Q4_K, Q5_K, Q4_K],
            [Q5_K, Q5_K, Q5_K],
        ]);
        let plan = plan_routed(&model, &m4, 4096, &NoKv, &levers, leg).expect("the leg");
        assert_eq!(plan.n_l, vec![8, 0, 0, 8]);
        let plan = plan_routed(&model, &m4, 4096, &NoKv, &levers, union).expect("per type");
        assert_eq!(plan.n_l, vec![8, 8, 8, 8]);
        let unnamed = layered(1);
        match plan_routed(&unnamed, &m1, 4096, &NoKv, &levers, leg) {
            Err(e) => assert_eq!(
                e.to_string(),
                "tensor experts 0: a routed stack whose name names no gate, up or down part, \
                 which a card leg's rule places by part"
            ),
            Ok(_) => panic!("the leg placed a stack of no part"),
        }
        let plan = plan_routed(&unnamed, &m1, 4096, &NoKv, &levers, union)
            .expect("the per-type rule places a stack of no part");
        assert_eq!(plan.n_l, vec![8]);
    }

    // PIN(2026-10-22): the Qwen3.6 leg runs a Q8_0 down and (once the leg has the
    // arm) a Q6_K gate and up through the common leg, so the layers that held those
    // types hold types it still has no GEMM for (an MXFP4 down, an IQ2_XS gate and
    // up); its rows are the ones the test held, and Qwen3.8's gain the MXFP4 down
    // its Q8_0 down did not have.
    #[test]
    fn host_only_reasons_follow_the_program_and_part_in_file_order() {
        use crate::accepts::{Program, routed_rule};
        use GgmlType::{IQ2_XS, MXFP4, Q4_K, Q5_K, Q6_K};
        let model = legged(&[
            [Q4_K, Q4_K, Q6_K],
            [Q4_K, Q4_K, MXFP4],
            [IQ2_XS, IQ2_XS, Q6_K],
            [Q4_K, Q5_K, Q4_K],
        ]);
        let rows = |program| {
            routed_rule(program)
                .host_only(&model)
                .unwrap()
                .into_iter()
                .map(|h| (h.layer, h.tensor.name.as_str()))
                .collect::<Vec<_>>()
        };
        assert_eq!(
            rows(Program::Qwen35Body),
            [
                (1, "ffn_down_exps.weight"),
                (2, "ffn_gate_exps.weight"),
                (2, "ffn_up_exps.weight"),
                (3, "ffn_up_exps.weight"),
            ]
        );
        assert_eq!(
            rows(Program::Qwen38Body),
            [
                (0, "ffn_down_exps.weight"),
                (1, "ffn_down_exps.weight"),
                (2, "ffn_gate_exps.weight"),
                (2, "ffn_up_exps.weight"),
                (2, "ffn_down_exps.weight"),
                (3, "ffn_up_exps.weight"),
                (3, "ffn_down_exps.weight"),
            ]
        );
        // The reason names the failing kernel contract: an unsupported part
        // is one cause, a gate/up type mismatch another, and neither is empty
        // or collapsed into the other. The wording itself is not pinned.
        let why = |program: Program, layer: usize, tensor: &str| {
            routed_rule(program)
                .host_only(&model)
                .unwrap()
                .into_iter()
                .find(|h| h.layer == layer && h.tensor.name == tensor)
                .map(|h| h.why)
                .expect("the cause is named")
        };
        let unsupported = why(Program::Qwen35Body, 1, "ffn_down_exps.weight");
        let mismatch = why(Program::Qwen35Body, 3, "ffn_up_exps.weight");
        assert!(!unsupported.is_empty(), "{unsupported}");
        assert!(!mismatch.is_empty(), "{mismatch}");
        assert_ne!(
            unsupported, mismatch,
            "an unsupported part and a gate/up mismatch are distinct causes"
        );
        let empty = ModelTensors {
            tensors: Vec::new(),
            ..model
        };
        assert!(
            routed_rule(Program::Qwen35Body)
                .host_only(&empty)
                .unwrap()
                .is_empty()
        );
    }

    /// The part a routed stack's name names, a draft layer's among them; a
    /// fused gate·up stack and any other name name none.
    #[test]
    fn a_stack_name_names_its_part() {
        for (name, part) in [
            ("ffn_gate_exps.weight", Some(ExpertPart::Gate)),
            ("ffn_up_exps.weight", Some(ExpertPart::Up)),
            ("ffn_down_exps.weight", Some(ExpertPart::Down)),
            ("nextn.ffn_down_exps.weight", Some(ExpertPart::Down)),
            ("ffn_gate_up_exps.weight", None),
            ("ffn_down_shexp.weight", None),
            ("experts 0", None),
        ] {
            assert_eq!(ExpertPart::of_name(name), part, "{name}");
        }
    }

    /// A card another process holds: the census's free reading caps the
    /// expert rule — fewer experts fit than the card's usable bytes hold,
    /// the headroom and the violations against the free bytes — and a card
    /// budget below the free reading still caps below it.
    #[test]
    fn a_held_card_plans_within_its_free_bytes() {
        let model = layered(3);
        let usable = HEAD + 24 * EXPERT;
        let machine = |free: Option<u64>| Machine {
            cards: vec![Card {
                free_bytes: free,
                held_by: free.map(|_| "pid 7 (977 MiB)"),
                ..bytes_card("held", usable, 0..3)
            }],
            tiers: Vec::new(),
            nodes: vec![Node {
                host: host(),
                unified: None,
            }],
        };
        let held_machine = machine(Some(HEAD + 12 * EXPERT));
        let plan = plan_with(&model, &held_machine, 4096, &NoKv, None).expect("a held card plans");
        let held: u64 = plan.n_l.iter().sum();
        assert!(held > 0 && held < 24, "a held card keeps fewer than all 24");
        assert_eq!(
            plan.usable_bytes(&plan.machine.cards[0]),
            HEAD + 12 * EXPERT
        );
        assert!(plan.violations().is_empty());
        // A card budget above the free reading leaves the free cap binding;
        // one below it caps below.
        let above = plan_with(&model, &held_machine, 4096, &NoKv, Some(usable))
            .expect("plans under a budget above free");
        assert_eq!(
            above.n_l.iter().sum::<u64>(),
            held,
            "a budget past free changes nothing"
        );
        let below = plan_with(&model, &held_machine, 4096, &NoKv, Some(HEAD + 6 * EXPERT))
            .expect("plans under a budget below free");
        assert_eq!(below.n_l.iter().sum::<u64>(), 6);
        // A full free reading (a quiet card) plans every expert, as the
        // census-free card always did.
        for free in [usable, u64::MAX] {
            let full = machine(Some(free));
            let plan = plan_with(&model, &full, 4096, &NoKv, None).expect("plans");
            assert_eq!(plan.n_l.iter().sum::<u64>(), 24, "free {free}");
            assert!(plan.violations().is_empty());
        }
        let census_free = machine(None);
        let plan = plan_with(&model, &census_free, 4096, &NoKv, None).expect("plans");
        assert_eq!(plan.n_l.iter().sum::<u64>(), 24);
    }

    /// The host's need by page class: every addend of the need in the class
    /// the one table gives it, the OS reserve out, so the classes sum to
    /// [`workstation::HostNeed::bytes`] under each way of holding the set. The
    /// lock moves the host set, the tables and the churn pool to the locked
    /// class and leaves the row cache and the run-ahead window in the page
    /// cache; a pinned arena moves from the anonymous class to the pinned one.
    /// A reserve with no class, a need whose OS reserve is not the host's and
    /// one whose reserves are under the host's own are refused by name.
    #[test]
    fn a_host_need_is_held_by_page_class() {
        use workstation::{
            CHECKPOINTS_RESERVE, HostNeed, Hosting, NeedClassError, NeedClasses, NeedTerm,
            OS_RESERVE, POINT_ROWS_RESERVE, PageClass, TIER_BATCH_HOST_RESERVE,
        };
        let model = layered(3);
        let machine_with = |reserves: Vec<(String, u64)>| Machine {
            cards: vec![bytes_card("c", HEAD + 6 * EXPERT, 0..3)],
            tiers: Vec::new(),
            nodes: vec![Node {
                host: Host {
                    usable_bytes: 1 << 40,
                    reserves,
                },
                unified: None,
            }],
        };
        let machine = machine_with(vec![
            (OS_RESERVE.to_string(), 100),
            (TIER_BATCH_HOST_RESERVE.to_string(), 40),
            (CHECKPOINTS_RESERVE.to_string(), 30),
            (POINT_ROWS_RESERVE.to_string(), 20),
        ]);
        let mut plan = plan_with(&model, &machine, 4096, &NoKv, None).expect("plans");
        assert!(plan.host.experts > 0, "the host holds experts");
        // The NVMe split's arena and run-ahead window, the row cache, the
        // tables and a unified pool's share, none of which the plan above has.
        let (rows, window, arena, cards, tables, shadows, extra) = (8, 5, 7, 11, 13, 17, 6);
        plan.host.row_reserve_bytes = rows;
        plan.host.reserve_bytes += rows + window;
        plan.host.nvme_arena_bytes = arena;
        plan.host.pool_card_bytes = cards;
        plan.host.table_bytes = tables;
        plan.host.shadow_bytes = shadows;
        let need = HostNeed::of(&plan, extra);
        let experts = need.experts;
        assert_eq!(need.reserves, 100 + 40 + 30 + 20 + rows + window);
        let files = experts + tables + extra;
        let cases = [
            (
                Hosting {
                    lock: false,
                    arena_pinned: false,
                },
                NeedClasses {
                    pinned: shadows + 40 + 30 + 20,
                    anon: arena,
                    cache: files + rows + window,
                    locked: 0,
                    unified: cards,
                },
            ),
            (
                Hosting {
                    lock: true,
                    arena_pinned: false,
                },
                NeedClasses {
                    pinned: shadows + 40 + 30 + 20,
                    anon: arena,
                    cache: rows + window,
                    locked: files,
                    unified: cards,
                },
            ),
            (
                Hosting {
                    lock: false,
                    arena_pinned: true,
                },
                NeedClasses {
                    pinned: shadows + 40 + 30 + 20 + arena,
                    anon: 0,
                    cache: files + rows + window,
                    locked: 0,
                    unified: cards,
                },
            ),
        ];
        for (how, want) in cases {
            let got = need.classes(&plan, how).expect("a need of this plan");
            assert_eq!(got, want, "{how:?}");
            assert_eq!(
                got.bytes(),
                need.bytes(),
                "{how:?}: the classes sum to the need"
            );
            let reading = HostNeed {
                read_ahead: 17,
                ..need
            };
            let with_reads = reading.classes(&plan, how).expect("a load's read reserve");
            assert_eq!(with_reads.anon, got.anon + 17);
            assert_eq!(with_reads.bytes(), reading.bytes());
        }
        // The table: each term's class under both holdings.
        let locked = Hosting {
            lock: true,
            arena_pinned: true,
        };
        let loose = Hosting {
            lock: false,
            arena_pinned: false,
        };
        let class = |t: NeedTerm| (t.class(loose), t.class(locked));
        assert_eq!(
            NeedTerm::ALL.map(class),
            [
                (PageClass::Cache, PageClass::Locked),
                (PageClass::Cache, PageClass::Locked),
                (PageClass::Pinned, PageClass::Pinned),
                (PageClass::Pinned, PageClass::Pinned),
                (PageClass::Pinned, PageClass::Pinned),
                (PageClass::Pinned, PageClass::Pinned),
                (PageClass::Cache, PageClass::Cache),
                (PageClass::Cache, PageClass::Cache),
                (PageClass::Cache, PageClass::Locked),
                (PageClass::Anon, PageClass::Anon),
                (PageClass::Anon, PageClass::Pinned),
                (PageClass::Unified, PageClass::Unified),
            ]
        );
        // A reserve nobody classed is refused by name, not counted nowhere.
        let odd = machine_with(vec![
            (OS_RESERVE.to_string(), 100),
            ("a new reserve".to_string(), 3),
        ]);
        let odd_plan = plan_with(&model, &odd, 4096, &NoKv, None).expect("plans");
        assert_eq!(
            HostNeed::of(&odd_plan, 0).classes(&odd_plan, loose),
            Err(NeedClassError::Reserve("a new reserve".to_string()))
        );
        // A need that is not the plan's.
        let other_os = HostNeed { os: 1, ..need };
        assert!(matches!(
            other_os.classes(&plan, loose),
            Err(NeedClassError::NotThePlans { .. })
        ));
        let thin = HostNeed {
            reserves: 100 + 40 + 30 + 20 + rows - 1,
            ..need
        };
        assert!(matches!(
            thin.classes(&plan, loose),
            Err(NeedClassError::NotThePlans { .. })
        ));
    }

    /// A card's bytes are every term the plan counts on it but the headroom.
    #[test]
    fn a_cards_bytes_sum_its_terms() {
        let model = layered(3);
        let machine = Machine {
            cards: vec![bytes_card("c", HEAD + 24 * EXPERT + 100_000, 0..3)],
            tiers: Vec::new(),
            nodes: vec![Node {
                host: host(),
                unified: None,
            }],
        };
        let plan = plan_with(&model, &machine, 4096, &NoKv, None).expect("plans");
        let c = &plan.cards[0];
        assert_eq!(
            c.bytes(),
            c.dense_bytes
                + c.expert_bytes
                + c.rounding_bytes
                + c.kv_bytes
                + c.stage_rows_bytes
                + c.scratch_bytes
                + c.context_bytes
                + c.reserve_bytes
        );
        assert!(c.bytes() > 0);
        let mut huge = c.clone();
        huge.dense_bytes = u64::MAX;
        huge.expert_bytes = 1;
        assert_eq!(huge.bytes(), u64::MAX, "a sum past u64 reads u64::MAX");
    }

    /// A unified machine's pool holds the card and the host together: the
    /// model's bytes, wherever the fill put an expert, the card's cache,
    /// set-aside and margin, the host's ring shadows, reserves and NVMe
    /// arena and window, and the bytes the load holds beside the plan, each
    /// counted once. A plan whose card and host each fit their own figure
    /// plans clean at a pool of exactly that sum, as with no pool at all,
    /// and is refused by name one byte under it, term by term; the host's
    /// headroom is what the pool leaves past the card and the host's need.
    /// The census reads the pool net of the context's own creation cost
    /// ([`workstation::CONTEXT_SELF`]), which the card's context term counts
    /// again: whole on a context term below that cost, and that cost alone
    /// on the workstation's [`workstation::CONTEXT`], which is above it.
    #[test]
    fn a_unified_pool_holds_card_and_host_together() {
        struct Kv;
        impl KvBytes for Kv {
            fn layer_bytes(&self, _layer: usize, _ctx_max: u64) -> u64 {
                64
            }
            fn shadow_bytes(&self, _layer: usize, _ctx_max: u64) -> u64 {
                32
            }
        }
        let model = layered(3);
        let margin = 200;
        let (kv, shadows, pinned) = (3 * 64, 3 * 32, 50);
        // An NVMe split's arena and run-ahead window, set after the plan.
        let (arena, window) = (7, 5);
        let plan_at = |context: u64, experts: u64, pool: Option<u64>, beside: u64| {
            let usable = HEAD + experts * EXPERT + context + 300 + 100 + margin + kv;
            let machine = Machine {
                cards: vec![Card {
                    context_bytes: context,
                    scratch_bytes: 300,
                    margin_bytes: margin,
                    free_bytes: Some(usable - context.min(workstation::CONTEXT_SELF)),
                    reserves: vec![("draft".to_string(), 100)],
                    ..bytes_card("pool", usable, 0..3)
                }],
                tiers: Vec::new(),
                // On a unified machine the host is the pool
                // (`workstation::host_of`).
                nodes: vec![Node {
                    host: Host {
                        usable_bytes: pool.unwrap_or(1 << 40),
                        reserves: vec![("pinned".to_string(), pinned)],
                    },
                    unified: pool.map(|bytes| UnifiedPool { bytes }),
                }],
            };
            let mut plan = plan_with(&model, &machine, 4096, &Kv, None).expect("plans");
            assert!(plan.host.experts > 0, "the host holds experts");
            let headroom = plan.host.headroom_bytes;
            plan.host.nvme_arena_bytes = arena;
            plan.host.reserve_bytes += window;
            (headroom, plan.violations_beside(beside))
        };
        let model_bytes: u64 = model.tensors.iter().map(|t| t.file_bytes).sum();
        let cases = [1_000, workstation::CONTEXT]
            .into_iter()
            .flat_map(|c| [(c, 6, 0), (c, 12, 9)]);
        for (context, experts, beside) in cases {
            let on_card = HEAD + experts * EXPERT;
            let cards = on_card + kv + context + 300 + 100 + margin;
            let host = model_bytes - on_card + shadows + pinned + arena + window;
            let total = cards + host + beside;
            let own = context.min(workstation::CONTEXT_SELF);
            let (_, v) = plan_at(context, experts, None, beside);
            assert!(
                v.is_empty(),
                "context {context}, {experts} experts on the card, no pool: {v:?}"
            );
            let (headroom, v) = plan_at(context, experts, Some(total - own), beside);
            assert!(
                v.is_empty(),
                "context {context}, {experts} experts on the card, a pool of the sum: {v:?}"
            );
            assert_eq!(
                headroom,
                i128::from(arena + window + beside),
                "context {context}, {experts} experts on the card: the pool past the plan's own \
                 terms"
            );
            match plan_at(context, experts, Some(total - own - 1), beside)
                .1
                .as_slice()
            {
                [v @ Violation::PoolOver { .. }] => assert_eq!(
                    v.to_string(),
                    format!(
                        "unified pool: the cards with their margins {cards} B + the host's need \
                         {host} B + {beside} B the load holds beside the plan = {total} B pass \
                         the pool {} B by 1 B",
                        total - 1
                    )
                ),
                v => panic!(
                    "context {context}, {experts} experts on the card: {v:?}, not one PoolOver"
                ),
            }
        }
    }

    #[test]
    fn plan_room_uses_the_host_and_unified_limits_once() {
        use workstation::{HOST_DRIFT, HostNeed, HostRead, OS_RESERVE};
        let model = layered(3);
        let pool = 1 << 30;
        for context in [1_000, workstation::CONTEXT] {
            for host_room in [pool - 100, pool + 100] {
                let machine = Machine {
                    cards: vec![Card {
                        context_bytes: context,
                        free_bytes: Some(pool),
                        ..bytes_card("pool", 2 * pool, 0..3)
                    }],
                    tiers: Vec::new(),
                    nodes: vec![Node {
                        host: Host {
                            usable_bytes: host_room + 64,
                            reserves: vec![(OS_RESERVE.into(), 64)],
                        },
                        unified: Some(UnifiedPool { bytes: pool }),
                    }],
                };
                let mut plan = plan_with(&model, &machine, 4096, &NoKv, None).unwrap();
                for read in [HostRead::Given, HostRead::MemAvailable] {
                    for room in [pool - 200, pool + HOST_DRIFT + 200] {
                        let reading = (room, read.clone());
                        let want = workstation::plan_room(&reading).min(host_room).min(pool);
                        assert_eq!(plan.plan_at(&reading), want);
                        assert_eq!(plan.host.room, Some((room, want)));
                        if read == HostRead::Given {
                            assert_eq!(
                                plan.host.headroom_bytes,
                                HostNeed::of(&plan, 0).headroom(want)
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn a_pool_reserve_counts_local_cards_without_overflow() {
        let model = layered(4);
        for ranks in [1, 2, 4] {
            let machine = Machine {
                cards: (0..ranks)
                    .map(|rank| Card {
                        head: rank + 1 == ranks,
                        token_embedding: rank == 0,
                        ..bytes_card(
                            &format!("pool-{rank}"),
                            1 << 30,
                            rank * 4 / ranks..(rank + 1) * 4 / ranks,
                        )
                    })
                    .collect(),
                tiers: Vec::new(),
                nodes: vec![Node {
                    host: host(),
                    unified: Some(UnifiedPool { bytes: 1 << 40 }),
                }],
            };
            let mut plan = plan_with(&model, &machine, 4096, &NoKv, None).unwrap();
            let before = plan.host.pool_card_bytes;
            let headroom = plan.host.headroom_bytes;
            let n = ranks as u64;
            let reserve = (u64::MAX - before) / n;
            plan.reserve_beside(reserve)
                .expect("the largest reserve fits u64");
            assert_eq!(plan.host.pool_card_bytes, before + reserve * n);
            assert_eq!(plan.host.headroom_bytes, headroom - i128::from(reserve * n));
            let before = (plan.host.pool_card_bytes, plan.host.headroom_bytes);
            for reserve in [1, u64::MAX, (u64::MAX / n).saturating_add(1)] {
                assert!(plan.reserve_beside(reserve).is_err());
                assert_eq!(
                    (plan.host.pool_card_bytes, plan.host.headroom_bytes),
                    before
                );
            }
        }
    }

    /// The census reads a card on its primary context, so an idle card's
    /// reading is its usable bytes less the context's own creation cost,
    /// which the card's context term counts again: the plan counts the
    /// context once. An idle reading plans what the census-free card plans;
    /// a reading another process holds 12 experts' bytes of plans what a
    /// card budget that much below usable plans; and the free floor takes the
    /// context less that cost — a trunk at the reading plans, one byte past it
    /// is refused.
    #[test]
    fn a_census_reading_counts_the_context_once() {
        let model = layered(3);
        let context = workstation::CONTEXT;
        let own = workstation::CONTEXT_SELF;
        let usable = HEAD + 24 * EXPERT + context;
        let machine = |free: Option<u64>| Machine {
            cards: vec![Card {
                context_bytes: context,
                free_bytes: free,
                ..bytes_card("card", usable, 0..3)
            }],
            tiers: Vec::new(),
            nodes: vec![Node {
                host: host(),
                unified: None,
            }],
        };
        let (none, idle, busy, budgeted) = (
            machine(None),
            machine(Some(usable - own)),
            machine(Some(usable - own - 12 * EXPERT)),
            machine(None),
        );
        let census_free = plan_with(&model, &none, 4096, &NoKv, None).expect("plans");
        assert_eq!(census_free.n_l.iter().sum::<u64>(), 24);
        let idle = plan_with(&model, &idle, 4096, &NoKv, None).expect("an idle card plans");
        assert_eq!(
            idle.n_l, census_free.n_l,
            "an idle reading plans as no reading"
        );
        assert_eq!(idle.usable_bytes(&idle.machine.cards[0]), usable);
        assert!(idle.violations().is_empty());
        let busy = plan_with(&model, &busy, 4096, &NoKv, None).expect("a held card plans");
        let budget = plan_with(&model, &budgeted, 4096, &NoKv, Some(usable - 12 * EXPERT))
            .expect("plans under a budget");
        assert_eq!(busy.n_l.iter().sum::<u64>(), 12);
        assert_eq!(busy.n_l, budget.n_l, "a held reading plans as its budget");
        assert!(busy.violations().is_empty());
        let at = HEAD + context - own;
        plan_with(&model, &machine(Some(at)), 4096, &NoKv, None).expect("a trunk at the reading");
        match plan_with(&model, &machine(Some(at - 1)), 4096, &NoKv, None) {
            Err(PlacementError::CardFreeFloor(terms)) => {
                assert_eq!(
                    (terms.free, terms.floor, terms.context),
                    (at - 1, at, context - own)
                );
            }
            Err(e) => panic!("{e}, not CardFreeFloor"),
            Ok(_) => panic!("a trunk past the reading was planned"),
        }
    }

    /// A card whose dense trunk alone passes what its device had free is
    /// refused by name, every term in the message — the card, the free and
    /// the usable bytes, the floor's terms and the holder the census named.
    #[test]
    fn a_trunk_past_the_free_bytes_is_refused_by_name() {
        let model = layered(3);
        let usable = HEAD + 24 * EXPERT;
        let machine = |free: u64| Machine {
            cards: vec![Card {
                free_bytes: Some(free),
                held_by: Some("pid 7 (977 MiB)"),
                ..bytes_card("held", usable, 0..3)
            }],
            tiers: Vec::new(),
            nodes: vec![Node {
                host: host(),
                unified: None,
            }],
        };
        let held = machine(HEAD - 1);
        match plan_with(&model, &held, 4096, &NoKv, None) {
            Err(PlacementError::CardFreeFloor(terms)) => {
                assert_eq!((terms.free, terms.floor), (HEAD - 1, HEAD));
                let text = terms.to_string();
                for part in [
                    "card held: the device had 4351 B free of its usable",
                    "(held by pid 7 (977 MiB))",
                    "needs 4352 B = dense 4352 B (the allocator's granules)",
                    "+ KV 0 B + context 0 B + scratch 0 B + reserves 0 B + margin 0 B",
                    "past the 4351 B by 1 B",
                    "free the card, or plan a placement on a card with room",
                ] {
                    assert!(text.contains(part), "{part:?} in {text}");
                }
            }
            Err(e) => panic!("{e}, not CardFreeFloor"),
            Ok(_) => panic!("a trunk past the free bytes was planned"),
        }
        // A floor within the free bytes never refuses; the holder list is
        // absent when the census named none.
        let mut quiet = machine(HEAD);
        quiet.cards[0].held_by = None;
        plan_with(&model, &quiet, 4096, &NoKv, None).expect("a trunk at the free bytes");
    }

    /// A reserve entered by [`Machine::reserve`] leaves the card before any
    /// expert: the stage card of three layers holds all 24 experts with none,
    /// 19 under five experts' bytes, none under the whole room. Under a card
    /// budget, the reserve and the dense tensors are the floor: one byte under
    /// it is the budget floor's refusal, at it the card holds no expert, and
    /// one expert's bytes more hold one.
    #[test]
    fn a_reserve_leaves_the_card_fewer_experts() {
        let model = layered(3);
        let machine = |reserve: Option<u64>| {
            let mut machine = Machine {
                cards: vec![bytes_card("stage", HEAD + 24 * EXPERT, 0..3)],
                tiers: Vec::new(),
                nodes: vec![Node {
                    host: host(),
                    unified: None,
                }],
            };
            if let Some(bytes) = reserve {
                machine
                    .reserve(Reserve {
                        on: Device::Card(0),
                        name: "draft",
                        bytes,
                    })
                    .expect("the stage card is card 0");
            }
            machine
        };
        let experts = |reserve: Option<u64>, budget: Option<u64>| {
            plan_with(&model, &machine(reserve), 4096, &NoKv, budget)
                .map(|plan| plan.cards[0].experts)
        };
        assert_eq!(experts(None, None).expect("plans"), 24);
        assert_eq!(experts(Some(0), None).expect("plans"), 24);
        assert_eq!(experts(Some(5 * EXPERT), None).expect("plans"), 19);
        assert_eq!(experts(Some(24 * EXPERT), None).expect("plans"), 0);
        let reserve = 10 * EXPERT;
        let floor = HEAD + reserve;
        match experts(Some(reserve), Some(floor - 1)) {
            Err(PlacementError::CardBudgetFloor {
                floor: named,
                reserves,
                ..
            }) => assert_eq!((named, reserves), (floor, reserve)),
            other => panic!("{other:?}, not the budget floor's refusal"),
        }
        assert_eq!(experts(Some(reserve), Some(floor)).expect("plans"), 0);
        assert_eq!(
            experts(Some(reserve), Some(floor + EXPERT - 1)).expect("plans"),
            0
        );
        assert_eq!(
            experts(Some(reserve), Some(floor + EXPERT)).expect("plans"),
            1
        );
        assert_eq!(
            experts(None, Some(floor + EXPERT)).expect("plans"),
            11,
            "the same budget without the reserve holds what the reserve took"
        );
    }

    /// [`Machine::reserve`] takes a card by its place in
    /// [`Machine::all_cards`]: the stage cards first, then the tiers; the host
    /// by its own device. A place past the list, the NVMe tier and nowhere are
    /// refused by name and set nothing aside.
    #[test]
    fn a_reserve_is_addressed_by_the_whole_device_list() {
        let mut machine = Machine {
            cards: vec![bytes_card("stage", 1_000, 0..3)],
            tiers: vec![card("tier", 0..0, false, false)],
            nodes: vec![Node {
                host: host(),
                unified: None,
            }],
        };
        let reserve = |on, name, bytes| Reserve { on, name, bytes };
        for r in [
            reserve(Device::Card(0), "on the stage", 1),
            reserve(Device::Card(1), "on the tier", 2),
            reserve(Device::Host, "on the host", 3),
        ] {
            machine.reserve(r).expect("a device of the plan");
        }
        assert_eq!(machine.cards[0].reserves, [("on the stage".to_string(), 1)]);
        assert_eq!(machine.tiers[0].reserves, [("on the tier".to_string(), 2)]);
        assert!(
            machine
                .host()
                .reserves
                .contains(&("on the host".to_string(), 3))
        );
        let before = machine.clone();
        for (on, shown) in [
            (Device::Card(2), "Card(2)"),
            (Device::Nvme, "Nvme"),
            (Device::Unused, "Unused"),
        ] {
            match machine.reserve(reserve(on, "off the list", 4)) {
                Err(e @ PlacementError::ReserveOffPlan { cards: 2, .. }) => assert_eq!(
                    e.to_string(),
                    format!(
                        "the off the list reserve is on {shown}, outside the plan: a reserve \
                         sits on one of the plan's 2 cards, stage first, or on the host"
                    )
                ),
                other => panic!("{other:?}, not ReserveOffPlan"),
            }
        }
        assert_eq!(machine, before, "a refused reserve sets nothing aside");
    }

    /// A reserve-only tier member ([`Card::reserve_only`]) holds its reserve and
    /// no expert: the stage card's experts are what the plan without the member
    /// holds, the member's counts are 0 on every layer, and it is the expert
    /// tiers' count that excludes it. One with no reserve is the idle tier's
    /// refusal; one whose reserve passes its room is the budget floor's, under
    /// a card budget, as for any card.
    #[test]
    fn a_reserve_only_member_holds_a_reserve_and_no_expert() {
        let model = layered(3);
        let member = |reserve: Option<u64>, tiers: usize| {
            let mut machine = Machine {
                cards: vec![bytes_card("stage", HEAD + 5 * EXPERT, 0..3)],
                tiers: (0..tiers)
                    .map(|i| Card {
                        reserve_only: true,
                        ..bytes_card(&format!("holder{i}"), 10 * EXPERT, 0..0)
                    })
                    .collect(),
                nodes: vec![Node {
                    host: host(),
                    unified: None,
                }],
            };
            if let Some(bytes) = reserve {
                machine
                    .reserve(Reserve {
                        on: Device::Card(1),
                        name: "draft",
                        bytes,
                    })
                    .expect("the member is card 1");
            }
            machine
        };
        let stage = member(None, 0);
        let alone = plan_with(&model, &stage, 4096, &NoKv, None).expect("plans");
        let held = member(Some(4 * EXPERT), 1);
        let plan = plan_with(&model, &held, 4096, &NoKv, None).expect("plans");
        assert_eq!(plan.n_l, alone.n_l, "the stage card's prefix does not move");
        assert_eq!(plan.tier_n_l, vec![vec![0, 0, 0]]);
        assert_eq!(plan.cards[1].experts, 0);
        assert_eq!(plan.host.experts, alone.host.experts);
        assert_eq!(held.expert_tiers().count(), 0);
        assert_eq!(held.tiers.len(), 1);
        match plan_with(&model, &member(None, 1), 4096, &NoKv, None) {
            Err(PlacementError::IdleTier { card, tier: 0 }) => assert_eq!(card, "holder0"),
            other => panic!("{other:?}, not IdleTier"),
        }
        let over = member(Some(11 * EXPERT), 1);
        assert!(
            !plan_with(&model, &over, 4096, &NoKv, None)
                .expect("plans")
                .violations()
                .is_empty(),
            "a reserve past the member's room is a violation"
        );
        match plan_with(&model, &over, 4096, &NoKv, Some(10 * EXPERT)) {
            Err(PlacementError::CardBudgetFloor { card, .. }) => assert_eq!(card, "holder0"),
            other => panic!("{other:?}, not the budget floor's refusal"),
        }
        let host_routed = plan_host_routed(&model, &held, 4096, &NoKv, &PlanLevers::default());
        assert!(host_routed.is_ok(), "{host_routed:?}: no expert tier card");
        assert_eq!(
            host_routed.expect("plans").host.experts,
            24,
            "every routed expert on the host"
        );
    }

    /// A per-card budget caps a card by its own entry, found by the card's
    /// device ordinal: the expert rule and [`Plan::usable_bytes`] read it, a
    /// card the list does not name plans with its own bytes, an entry under
    /// the card's floor is refused naming the card, and a census-free card
    /// under a list is refused by name ([`card_budget::check`]).
    #[test]
    fn a_per_card_budget_caps_each_card_by_its_own_entry() {
        let model = layered(3);
        let full = HEAD + 24 * EXPERT + 100_000;
        let half = HEAD + 12 * EXPERT;
        let on = |device: Option<u32>| Machine {
            cards: vec![Card {
                device: device.map(|ordinal| devices::DeviceId {
                    ordinal,
                    uuid: [7; 16],
                }),
                ..bytes_card("stage", full, 0..3)
            }],
            tiers: Vec::new(),
            nodes: vec![Node {
                host: host(),
                unified: None,
            }],
        };
        let list = |entries: &[(u32, u64)]| PlanLevers {
            card_budget_bytes: card_budget::CardBudget::Per(entries.to_vec()),
        };
        let named = on(Some(1));
        let p = plan(&model, &named, 4096, &NoKv, &list(&[(0, 1), (1, half)])).expect("plans");
        assert_eq!(
            (p.cards[0].experts, p.usable_bytes(&named.cards[0])),
            (12, half)
        );
        let other = on(Some(0));
        let p = plan(&model, &other, 4096, &NoKv, &list(&[(1, half)])).expect("plans");
        assert_eq!(
            (p.cards[0].experts, p.usable_bytes(&other.cards[0])),
            (24, full)
        );
        match plan(&model, &named, 4096, &NoKv, &list(&[(1, HEAD - 1)])) {
            Err(PlacementError::CardBudgetFloor { card, budget, .. }) => {
                assert_eq!((card.as_str(), budget), ("stage", HEAD - 1));
            }
            other => panic!("{:?}, not the stage's budget floor", other.map(|_| ())),
        }
        let census_free = on(None);
        match plan(&model, &census_free, 4096, &NoKv, &list(&[(0, half)])) {
            Err(PlacementError::CardBudgetCard { card, .. }) => assert_eq!(card, "stage"),
            other => panic!("{:?}, not a census-free card's refusal", other.map(|_| ())),
        }
    }

    /// A budget by shares becomes a budget per card from the plan's own terms: each named
    /// card plans with its floor and its part of the routed experts, in the proportion of the
    /// shares and as many in all as the tightest card's room for its share lets, the host tier
    /// taking the rest, and the plan's record states the bytes. 3:1 over a stage and a tier of
    /// 24 experts' room each holds all 24 as 18 and 6 (the default fill leaves the tier idle);
    /// 1:1 with a tier of 2 experts' room holds 4 as 2 and 2 and leaves 20 to the host. A floor
    /// past the card, a plan that places no routed expert on a card, shares that name no card
    /// of the plan and shares of 0 are refused naming the cause.
    #[test]
    fn shares_become_a_budget_per_card_from_the_plans_terms() {
        let model = layered(3);
        let device = |ordinal: u32| {
            Some(devices::DeviceId {
                ordinal,
                uuid: [u8::try_from(ordinal).expect("small") + 7; 16],
            })
        };
        let on = |stage: u64, tier: u64| Machine {
            cards: vec![Card {
                device: device(0),
                ..bytes_card("stage", stage, 0..3)
            }],
            tiers: vec![Card {
                device: device(1),
                ..bytes_card("tier", tier, 0..0)
            }],
            nodes: vec![Node {
                host: host(),
                unified: None,
            }],
        };
        let shares = |entries: &[(u32, u64)]| PlanLevers {
            card_budget_bytes: card_budget::CardBudget::Shares(entries.to_vec()),
        };
        let roomy = on(HEAD + 24 * EXPERT, 24 * EXPERT);
        let p = plan(&model, &roomy, 4096, &NoKv, &shares(&[(0, 3), (1, 1)])).expect("plans");
        assert_eq!(p.n_l, vec![6, 6, 6]);
        assert_eq!(p.tier_n_l, vec![vec![2, 2, 2]]);
        assert_eq!(p.host.experts, 0);
        assert_eq!(
            p.card_budget,
            card_budget::CardBudget::Per(vec![(0, HEAD + 18 * EXPERT), (1, 6 * EXPERT)])
        );
        assert!(matches!(
            plan_with(&model, &roomy, 4096, &NoKv, None),
            Err(PlacementError::IdleTier { .. })
        ));
        let tight = on(HEAD + 24 * EXPERT, 2 * EXPERT);
        let p = plan(&model, &tight, 4096, &NoKv, &shares(&[(0, 1), (1, 1)])).expect("plans");
        assert_eq!(p.n_l, vec![1, 1, 0]);
        assert_eq!(p.tier_n_l, vec![vec![1, 0, 1]]);
        assert_eq!(p.host.experts, 20);
        assert_eq!(
            p.card_budget,
            card_budget::CardBudget::Per(vec![(0, HEAD + 2 * EXPERT), (1, 2 * EXPERT)])
        );
        // Every term of a card's floor is in its part: context, scratch, reserves, margin and
        // what the caller holds beside the plan.
        let held = |c: &Card| Card {
            context_bytes: 1_000,
            scratch_bytes: 2_000,
            margin_bytes: 3_000,
            reserves: vec![("draft".to_string(), 4_000)],
            ..c.clone()
        };
        let slack = on(HEAD + 24 * EXPERT + 100_000, 24 * EXPERT + 100_000);
        let held_machine = Machine {
            cards: vec![held(&slack.cards[0])],
            tiers: vec![held(&slack.tiers[0])],
            ..slack.clone()
        };
        let beside = 5_000;
        let p = plan_routed_reserving(
            &model,
            &held_machine,
            4096,
            &NoKv,
            &shares(&[(0, 3), (1, 1)]),
            RoutedRule::PerType(CardFormat::of_routed),
            beside,
        )
        .expect("plans");
        assert_eq!(p.n_l, vec![6, 6, 6]);
        let past = 1_000 + 2_000 + 3_000 + 4_000 + beside;
        assert_eq!(
            p.card_budget,
            card_budget::CardBudget::Per(vec![
                (0, HEAD + past + 18 * EXPERT),
                (1, past + 6 * EXPERT)
            ])
        );
        let refused = |machine: &Machine, levers: &PlanLevers| match plan(
            &model, machine, 4096, &NoKv, levers,
        ) {
            Err(PlacementError::CardShares { shares, why }) => (shares, why),
            other => panic!("{:?}, not a shares refusal", other.map(|_| ())),
        };
        let (named, why) = refused(&on(HEAD - 1, 24 * EXPERT), &shares(&[(0, 1), (1, 1)]));
        assert_eq!(named, "CUDA0=1x,CUDA1=1x");
        assert!(
            why.starts_with("card stage: its floor 4352 B = dense uploads 4352 B")
                && why.contains("passes the 4351 B it may plan with"),
            "{why}"
        );
        let (_, why) = refused(&roomy, &shares(&[(5, 1)]));
        assert!(
            why.contains("names no card of this plan that takes experts")
                && why.contains("stage, tier"),
            "{why}"
        );
        let (_, why) = refused(&roomy, &shares(&[(0, 0), (5, 1)]));
        assert!(why.contains("has a share above 0"), "{why}");
        let stage_only = Machine {
            tiers: Vec::new(),
            ..roomy.clone()
        };
        match plan_host_routed(&model, &stage_only, 4096, &NoKv, &shares(&[(0, 1)])) {
            Err(PlacementError::CardShares { why, .. }) => {
                assert!(why.contains("places no routed expert on a card"), "{why}");
            }
            other => panic!("{:?}, not a shares refusal", other.map(|_| ())),
        }
    }

    /// A plan whose caller keeps `BESIDE` bytes of the card beside it
    /// ([`plan_routed_reserving`]) names a budget floor that counts them, as
    /// the expert rule's budget does: the floor is the least budget that
    /// plans, so the plan at the floor leaves the margin and the reserve both
    /// unspent (the caller's bound on the sum), and one byte under it is
    /// refused naming that floor, whatever the card's own reserves, context,
    /// scratch and margin are.
    #[test]
    fn the_budget_floor_counts_what_the_caller_holds_beside_the_plan() {
        const CONTEXT: u64 = 1_000;
        const SCRATCH: u64 = 2_000;
        const MARGIN: u64 = 3_000;
        const CARD_RESERVE: u64 = 4_000;
        const BESIDE: u64 = 5_000;
        let model = layered(3);
        let machine = Machine {
            cards: vec![Card {
                context_bytes: CONTEXT,
                scratch_bytes: SCRATCH,
                margin_bytes: MARGIN,
                reserves: vec![("draft".to_string(), CARD_RESERVE)],
                ..bytes_card("stage", HEAD + 24 * EXPERT + 100_000, 0..3)
            }],
            tiers: Vec::new(),
            nodes: vec![Node {
                host: host(),
                unified: None,
            }],
        };
        let plan_at = |budget: u64, beside: u64| {
            let levers = PlanLevers {
                card_budget_bytes: Some(budget).into(),
            };
            plan_routed_reserving(
                &model,
                &machine,
                4096,
                &NoKv,
                &levers,
                RoutedRule::PerType(CardFormat::of_routed),
                beside,
            )
        };
        let floor_of = |beside: u64| match plan_at(1, beside) {
            Err(PlacementError::CardBudgetFloor { floor, .. }) => floor,
            Err(e) => panic!("{e}, not CardBudgetFloor"),
            Ok(_) => panic!("a budget of 1 B planned"),
        };
        let trunk = HEAD + CONTEXT + SCRATCH + CARD_RESERVE + MARGIN;
        assert_eq!(floor_of(0), trunk, "a plan with nothing beside it");
        let floor = floor_of(BESIDE);
        let mut bad = Vec::new();
        if floor != trunk + BESIDE {
            bad.push(format!(
                "the floor beside {BESIDE} B is {floor} B, not the trunk's {trunk} B plus them"
            ));
        }
        match plan_at(floor, BESIDE) {
            Ok(plan) => {
                let card = &plan.cards[0];
                let left = card.headroom_bytes - i128::from(MARGIN);
                if left < i128::from(BESIDE) {
                    bad.push(format!(
                        "at the floor {floor} B the plan leaves {left} B past the margin, under \
                         the {BESIDE} B beside it"
                    ));
                }
                if card.experts != 0 || !plan.violations().is_empty() {
                    bad.push(format!(
                        "at the floor {floor} B: {} experts, violations {:?}",
                        card.experts,
                        plan.violations()
                    ));
                }
            }
            Err(e) => bad.push(format!("at the floor {floor} B: {e}")),
        }
        match plan_at(floor - 1, BESIDE) {
            Err(e @ PlacementError::CardBudgetFloor { .. }) => {
                let want = format!(
                    "budget {} B is below its floor {floor} B = dense {HEAD} B (the allocator's \
                     granules) + KV 0 B + context {CONTEXT} B + scratch {SCRATCH} B + reserves \
                     {CARD_RESERVE} B + margin {MARGIN} B + beside {BESIDE} B (what the caller \
                     holds beside the plan)",
                    floor - 1
                );
                if !e.to_string().contains(&want) {
                    bad.push(format!("one byte under the floor: {e}, not {want:?}"));
                }
            }
            Err(e) => bad.push(format!("one byte under the floor {floor} B: {e}")),
            Ok(_) => bad.push(format!("one byte under the floor {floor} B planned")),
        }
        // A reserve that takes the sum past u64 bytes is refused with the floor
        // printed as u64::MAX, never wrapped.
        match plan_at(u64::MAX, u64::MAX) {
            Err(PlacementError::CardBudgetFloor { floor, beside, .. })
                if (floor, beside) == (u64::MAX, u64::MAX) => {}
            other => bad.push(format!("a reserve of u64::MAX: {:?}", other.map(|p| p.n_l))),
        }
        assert!(bad.is_empty(), "{bad:#?}");
    }

    /// The free floor counts what the caller keeps beside the plan the same
    /// way ([`the_budget_floor_counts_what_the_caller_holds_beside_the_plan`]):
    /// a card whose device had its floor free plans with the margin and the
    /// reserve both left, and one byte under it is refused naming that floor.
    #[test]
    fn the_free_floor_counts_what_the_caller_holds_beside_the_plan() {
        const MARGIN: u64 = 3_000;
        const BESIDE: u64 = 5_000;
        let model = layered(3);
        let context = workstation::CONTEXT;
        let own = workstation::CONTEXT_SELF;
        let usable = HEAD + 24 * EXPERT + context + 100_000;
        let machine = |free: u64| Machine {
            cards: vec![Card {
                context_bytes: context,
                margin_bytes: MARGIN,
                free_bytes: Some(free),
                ..bytes_card("held", usable, 0..3)
            }],
            tiers: Vec::new(),
            nodes: vec![Node {
                host: host(),
                unified: None,
            }],
        };
        let plan_at = |free: u64, beside: u64| {
            plan_routed_reserving(
                &model,
                &machine(free),
                4096,
                &NoKv,
                &PlanLevers::default(),
                RoutedRule::PerType(CardFormat::of_routed),
                beside,
            )
            .map(|plan| (plan.cards[0].headroom_bytes, plan.cards[0].experts))
        };
        let floor_of = |beside: u64| match plan_at(1, beside) {
            Err(PlacementError::CardFreeFloor(terms)) => terms.floor,
            Err(e) => panic!("{e}, not CardFreeFloor"),
            Ok(_) => panic!("a reading of 1 B planned"),
        };
        let trunk = HEAD + context - own + MARGIN;
        assert_eq!(floor_of(0), trunk, "a plan with nothing beside it");
        let floor = floor_of(BESIDE);
        let mut bad = Vec::new();
        if floor != trunk + BESIDE {
            bad.push(format!(
                "the floor beside {BESIDE} B is {floor} B, not the trunk's {trunk} B plus them"
            ));
        }
        match plan_at(floor, BESIDE) {
            Ok((headroom, experts)) => {
                let left = headroom - i128::from(MARGIN);
                if left < i128::from(BESIDE) || experts != 0 {
                    bad.push(format!(
                        "at the floor {floor} B the plan leaves {left} B past the margin, under \
                         the {BESIDE} B beside it, and keeps {experts} experts"
                    ));
                }
            }
            Err(e) => bad.push(format!("at the floor {floor} B: {e}")),
        }
        match plan_at(floor - 1, BESIDE) {
            Err(PlacementError::CardFreeFloor(terms)) => {
                if (terms.free, terms.floor, terms.beside) != (floor - 1, floor, BESIDE) {
                    bad.push(format!("one byte under the floor {floor} B: {terms:?}"));
                }
                let want = format!(
                    "+ margin {MARGIN} B + beside {BESIDE} B (what the caller holds beside the plan)"
                );
                if !terms.to_string().contains(&want) {
                    bad.push(format!("one byte under the floor: {terms}, not {want:?}"));
                }
            }
            Err(e) => bad.push(format!("one byte under the floor {floor} B: {e}")),
            Ok(_) => bad.push(format!("one byte under the floor {floor} B planned")),
        }
        assert!(bad.is_empty(), "{bad:#?}");
    }

    /// The whole fit counts what the whole load asks for: every tensor's
    /// buffers through the card's allocator (the embedding's two q8_0
    /// planes, each layer's stack, the head's planes: 22,528 B of buffers,
    /// 36,864 B of 4 KiB granules — the planes' small halves share one), the
    /// cache, the context, the scratch, and the program's arena with the
    /// reserve its load keeps free past it. A card whose free reading holds
    /// all of that but one byte of the arena does not fit; at the need it
    /// fits. The budget is the reading as taken — the context's own creation
    /// not added back, as the plan's cap adds it; a program with no checked
    /// arena keeps the card's margin in its place; and a tensor no card
    /// format loads is refused by name.
    #[test]
    fn the_whole_fit_counts_the_arena_and_the_granules() {
        const GRANULE: u64 = 4096;
        const KV: u64 = 3_000;
        const ARENA: u64 = 50_000;
        const RESERVE: u64 = 7_000;
        const MARGIN: u64 = 90_000;
        let model = layered(3);
        let context = workstation::CONTEXT;
        let scratch = workstation::SCRATCH;
        let load = WholeLoad::Checked {
            arena: ARENA,
            reserve: RESERVE,
        };
        let at = |free: u64| Card {
            context_bytes: context,
            scratch_bytes: scratch,
            margin_bytes: MARGIN,
            granule_bytes: NonZeroU64::new(GRANULE).expect("not zero"),
            free_bytes: Some(free),
            ..bytes_card("card", u64::MAX, 0..3)
        };
        let weights = 9 * GRANULE;
        let need = weights + KV + context + scratch + ARENA + RESERVE;
        let fit = whole_need(&model, &at(need), KV, load).expect("every tensor has a format");
        assert_eq!((fit.weights, fit.weights_unrounded), (weights, 22_528));
        assert_eq!((fit.kv, fit.need, fit.budget), (KV, need, need));
        assert!(fit.fits(), "a reading at the need fits");
        let short = whole_need(&model, &at(need - 1), KV, load).expect("sized");
        assert!(
            !short.fits(),
            "the weights, the cache and all but one byte of the arena are not a whole load"
        );
        assert_eq!(short.budget, need - 1, "the reading as taken");
        let text = short.to_string();
        for part in [
            "the whole load on card: does not fit: need",
            "weights 36864 B (the allocator's granules; 22528 B of buffers) + cache 3000 B",
            "arena 50000 B + reserve past it 7000 B",
        ] {
            assert!(text.contains(part), "{part:?} in {text}");
        }
        let unchecked = whole_need(&model, &at(need), KV, WholeLoad::Unchecked).expect("sized");
        assert_eq!(
            unchecked.need,
            weights + KV + context + scratch + MARGIN,
            "the margin in the arena's place"
        );
        // A routed stack of a type with no device format (iq1_s has none) is refused by name.
        let mut odd = layered(3);
        odd.tensors[1].ty = GgmlType::IQ1_S;
        match whole_need(&odd, &at(need), KV, load) {
            Err(e @ PlacementError::NoCardFormat { .. }) => assert_eq!(
                e.to_string(),
                "tensor experts 0: type iq1_s has no device format, but its role (routed) puts it \
                 on a card"
            ),
            Err(e) => panic!("{e}, not NoCardFormat"),
            Ok(n) => panic!("an iq1_s stack was sized as {n:?}"),
        }
    }

    /// Two plan cards of one name are one device: their budgets past its
    /// usable bytes are refused by name, and a card budget that splits the
    /// device between them plans.
    #[test]
    fn one_device_counted_twice_is_refused() {
        let model = layered(3);
        let usable = HEAD + 24 * EXPERT;
        let machine = Machine {
            cards: vec![bytes_card("dev", usable, 0..3)],
            tiers: vec![bytes_card("dev", usable, 0..0)],
            nodes: vec![Node {
                host: host(),
                unified: None,
            }],
        };
        match plan_with(&model, &machine, 4096, &NoKv, None) {
            Err(e @ PlacementError::DeviceTwice { .. }) => assert_eq!(
                e.to_string(),
                format!(
                    "card dev: 2 plan cards are this one device, and their budgets sum to {} B, \
                     past its usable {usable} B: one device's budget counted twice (a card \
                     budget that splits the device between them is planned)",
                    2 * usable
                )
            ),
            Err(e) => panic!("{e}, not DeviceTwice"),
            Ok(_) => panic!("one device was planned twice"),
        }
        let half = usable / 2;
        let plan = plan_with(&model, &machine, 4096, &NoKv, Some(half)).expect("split budgets");
        assert!(plan.tier_n_l[0].iter().sum::<u64>() > 0);
        assert!(matches!(
            plan_with(&model, &machine, 4096, &NoKv, Some(half + 1)),
            Err(PlacementError::DeviceTwice { cards: 2, .. })
        ));
    }

    /// Two plan cards of one name resolved to two devices (two 3090s) are
    /// two devices: each keeps its whole budget. The same device under both
    /// is still one device counted twice.
    #[test]
    fn two_devices_of_one_name_plan() {
        let model = layered(3);
        let usable = HEAD + 5 * EXPERT;
        let dev = |ordinal: u32| {
            Some(devices::DeviceId {
                ordinal,
                uuid: [u8::try_from(ordinal).expect("small") + 1; 16],
            })
        };
        let on = |c: Card, d| Card { device: d, ..c };
        let machine = |a, b| Machine {
            cards: vec![on(bytes_card("3090", usable, 0..3), a)],
            tiers: vec![on(bytes_card("3090", usable, 0..0), b)],
            nodes: vec![Node {
                host: host(),
                unified: None,
            }],
        };
        let two = machine(dev(0), dev(1));
        let plan = plan_with(&model, &two, 4096, &NoKv, None).expect("two 3090s");
        assert!(plan.tier_n_l[0].iter().sum::<u64>() > 0);
        assert!(matches!(
            plan_with(&model, &machine(dev(0), dev(0)), 4096, &NoKv, None),
            Err(PlacementError::DeviceTwice { cards: 2, .. })
        ));
    }

    /// The list type itself: sorted on construction, duplicates and ids past
    /// the stack refused, the complement and runs of a scattered list, and a
    /// prefix recognised as one.
    #[test]
    fn expert_list_shapes() {
        let l = ExpertList::new(vec![6, 1, 2], 8).expect("a list");
        assert_eq!(l.ids(), &[1, 2, 6]);
        assert_eq!(l.runs(), vec![1..3, 6..7]);
        assert_eq!(l.to_string(), "1..3,6..7");
        assert_eq!(l.complement(8).expect("fits").ids(), &[0, 3, 4, 5, 7]);
        assert_eq!(l.slot_of(6), Some(2));
        assert_eq!(l.as_prefix(), None);
        assert!(ExpertList::new(vec![1, 1], 8).is_err());
        assert!(ExpertList::new(vec![8], 8).is_err());
        let p = ExpertList::prefix(3).expect("a prefix");
        assert_eq!(
            (p.as_prefix(), p.to_string()),
            (Some(3), "0..3".to_string())
        );
    }

    /// The whole-fit verdict a load of N resident slots opens by is the
    /// total's, never one slot's share: the cache is linear in the context,
    /// so on one card a total one position past the whole fit is short while
    /// its half fits — a verdict asked at the share would open the whole load
    /// and the slots added after it would hold a cache the verdict never
    /// counted (the qwen3 seat makes its verdict once, at the total).
    #[test]
    fn a_total_past_the_whole_fit_fits_at_half() {
        const ROW: u64 = 2_048;
        const TOTAL: u64 = 4_096;
        let model = layered(3);
        let load = WholeLoad::Unchecked;
        let kv = |ctx: u64| ctx * ROW;
        let at = |free: u64| Card {
            granule_bytes: NonZeroU64::new(4096).expect("not zero"),
            free_bytes: Some(free),
            ..bytes_card("card", u64::MAX, 0..3)
        };
        let edge = whole_need(&model, &at(u64::MAX), kv(TOTAL - 1), load)
            .expect("every tensor has a format")
            .need;
        let card = at(edge);
        let total = whole_need(&model, &card, kv(TOTAL), load).expect("sized");
        let share = whole_need(&model, &card, kv(TOTAL / 2), load).expect("sized");
        assert!(
            whole_need(&model, &card, kv(TOTAL - 1), load)
                .expect("sized")
                .fits(),
            "the reading holds the whole load one position short of the total"
        );
        assert!(!total.fits(), "the total does not fit: {total}");
        assert!(share.fits(), "a slot's share of it does: {share}");
    }

    /// A card's KV class grown after the totals (`CardTotals::grow_kv`: the
    /// bytes a model's resident sequences hold beside their stores, reserved
    /// out of the expert budget) leaves its headroom the usable bytes less
    /// every term, the KV it gained among them.
    #[test]
    fn grown_kv_leaves_the_headroom_usable_less_every_term() {
        let model = layered(3);
        let machine = Machine {
            cards: vec![Card {
                context_bytes: 1_000,
                scratch_bytes: 2_000,
                reserves: vec![("draft".to_string(), 3_000)],
                ..bytes_card("stage", HEAD + 12 * EXPERT + 6_000 + 5_000, 0..3)
            }],
            tiers: Vec::new(),
            nodes: vec![Node {
                host: host(),
                unified: None,
            }],
        };
        let mut plan = plan_routed_reserving(
            &model,
            &machine,
            4096,
            &NoKv,
            &PlanLevers::default(),
            RoutedRule::PerType(CardFormat::of_routed),
            5_000,
        )
        .expect("the plan");
        let rest = |p: &Plan<'_>| {
            let t = &p.cards[0];
            i128::from(p.usable_bytes(&machine.cards[0]))
                - i128::from(
                    t.dense_bytes
                        + t.expert_bytes
                        + t.rounding_bytes
                        + t.kv_bytes
                        + t.scratch_bytes
                        + t.context_bytes
                        + t.reserve_bytes,
                )
        };
        assert_eq!(plan.cards[0].headroom_bytes, rest(&plan));
        plan.cards[0].grow_kv(5_000);
        assert_eq!(plan.cards[0].kv_bytes, 5_000);
        assert_eq!(plan.cards[0].headroom_bytes, rest(&plan));
    }

    /// The plan's host bound is the load's own check at the room the
    /// machine's figure leaves (`HostNeed::check` at
    /// `HostNeed::machine_room`): a host whose usable bytes hold the plan's
    /// host tensors and reserves to the byte passes and one byte under is a
    /// `HostOver` naming both, the headroom the usable bytes less those terms
    /// — whether the OS reserve carries the name a reading leaves out or
    /// another — and the NVMe tier's arena and run-ahead window, sized to
    /// the room the split read and held to it, stay out of the bound.
    #[test]
    fn the_host_bound_is_the_needs_check_at_the_machine_room() {
        let model = layered(3);
        for os in [workstation::OS_RESERVE, "other"] {
            let machine = |usable: u64| Machine {
                cards: vec![bytes_card("stage", HEAD + 12 * EXPERT, 0..3)],
                tiers: Vec::new(),
                nodes: vec![Node {
                    host: Host {
                        usable_bytes: usable,
                        reserves: vec![(os.to_string(), 7_000), ("pool".to_string(), 1_000)],
                    },
                    unified: None,
                }],
            };
            let probe = machine(1 << 40);
            let plan = plan_with(&model, &probe, 4096, &NoKv, None).expect("the plan");
            assert!(plan.host.expert_bytes > 0, "{os}: the host holds experts");
            let terms =
                plan.host.expert_bytes + plan.host.table_bytes + plan.host.shadow_bytes + 8_000;
            let over = |p: &Plan<'_>| {
                p.violations()
                    .iter()
                    .filter(|v| matches!(v, Violation::HostOver { .. }))
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
            };
            for usable in [terms, terms - 1] {
                let m = machine(usable);
                let plan = plan_with(&model, &m, 4096, &NoKv, None).expect("the plan");
                let want = if usable < terms {
                    vec![format!(
                        "host: tensors + ring shadows + reserves = {terms} B pass usable {usable} \
                         B by 1 B"
                    )]
                } else {
                    Vec::new()
                };
                assert_eq!(over(&plan), want, "{os} at usable {usable}");
                assert_eq!(
                    plan.host.headroom_bytes,
                    i128::from(usable) - i128::from(terms),
                    "{os} at usable {usable}"
                );
            }
            let m = machine(terms);
            let mut plan = plan_with(&model, &m, 4096, &NoKv, None).expect("the plan");
            plan.host.nvme_arena_bytes = 1;
            plan.host.reserve_bytes += 1;
            assert_eq!(
                over(&plan),
                Vec::<String>::new(),
                "{os}: the split's arena and window are held to its room"
            );
        }
    }

    /// The row room counts the pages a row sits on, in the host's base page:
    /// a 90 B row can straddle a page boundary at 4, 16 or 64 KiB, so the
    /// room of a call of 4,096 positions reading 16 rows a position is
    /// 4,096 · 16 · 2 · page —
    /// 536,870,912 B on this workstation's page and 2,147,483,648 B on the
    /// 16 KiB one, and 8,589,934,592 B on the 64 KiB one.
    #[test]
    fn the_row_room_counts_the_hosts_base_page() {
        let model = ModelTensors {
            tensors: vec![ModelTensor {
                name: "ple table".into(),
                shard: 0,
                layer: Some(0),
                role: Role::EngramTable,
                ty: GgmlType::Q4_K,
                dims: vec![90, 16],
                file_bytes: 16 * 90,
                gathered_rows: Some(16),
            }],
            layers: 1,
            experts: 0,
            experts_used: 0,
            bf16_words: |_| false,
        };
        for source in [RowTableSource::Split, RowTableSource::Parts] {
            let room = |page| row_room(&model, 4096, page, source).expect("the row room");
            assert_eq!(room(4096), 536_870_912);
            assert_eq!(room(16384), 2_147_483_648);
            assert_eq!(room(65536), 8_589_934_592);
        }

        let mut model = model;
        let table = &mut model.tensors[0];
        table.ty = GgmlType::NVFP4_ROWS;
        table.dims = vec![160, 100];
        table.file_bytes = 90 * 100;
        table.gathered_rows = Some(1);
        let touched: u64 = [80, 10]
            .into_iter()
            .map(|width| {
                let start = 256 + 99 * width;
                (start + width - 1) / 4096 - start / 4096 + 1
            })
            .sum();
        assert_eq!(touched, 3);
        let reserve =
            row_room(&model, 1, 4096, RowTableSource::Parts).expect("the separate-plane row room");
        assert!(
            reserve >= touched * 4096,
            "{reserve} B reserved for {touched} pages"
        );
        model.tensors[0].gathered_rows = Some(16);
        for (source, want) in [
            (RowTableSource::Split, 536_870_912),
            (RowTableSource::Parts, 1_073_741_824),
        ] {
            for page in [4096, 16384, 65536] {
                assert_eq!(
                    row_room(&model, 4096, page, source).unwrap(),
                    want * (page / 4096)
                );
            }
        }
        model.tensors[0].file_bytes = 89 * 100;
        let e = row_room(&model, 4096, 4096, RowTableSource::Parts).unwrap_err();
        assert!(
            e.to_string()
                .contains("NVFP4 row bytes are not whole groups")
        );
    }
}
