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
//! (`BLOOMERY_CARD_BUDGET`, [`card_budget`]) caps every card's usable bytes
//! before the rule runs — the lever comes in as [`PlanLevers`], which a
//! binary parses once — and the census's free reading caps them below that:
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

use std::fmt;
use std::num::NonZeroU64;
use std::ops::Range;

use gguf::GgmlType;

pub mod card_budget;
pub mod churn;
pub mod ctx;
pub mod devices;
pub mod paged_drop;
pub mod workstation;

#[cfg(test)]
mod qwen38_cards;

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
    /// Experts in each routed stack, and experts one token uses.
    pub experts: u64,
    pub experts_used: u64,
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

    /// The reserves' bytes.
    #[must_use]
    pub fn reserve_bytes(&self) -> u64 {
        self.reserves.iter().map(|(_, b)| b).sum()
    }

    /// What the card sets aside before any tensor or cache: context,
    /// scratch and reserves. The one owner of that sum: the expert rule's
    /// budget, the budget floor, the headroom and [`Plan::violations`] all
    /// take it from here. The margin is not in it.
    #[must_use]
    pub fn set_aside_bytes(&self) -> u64 {
        self.context_bytes + self.scratch_bytes + self.reserve_bytes()
    }

    /// The card's floor under `dense` bytes of uploads and `kv` bytes of
    /// cache: those, its set-aside, and `past` — what the load keeps beside
    /// them: a plan's margin, or a whole load's own arena and reserve
    /// ([`WholeLoad::past_bytes`]). The one owner of that sum: the budget
    /// floor, the free floor and the whole fit ([`whole_need`]) all take it
    /// from here. `None` past u64 bytes.
    #[must_use]
    pub fn floor_bytes(&self, dense: u64, kv: u64, past: u64) -> Option<u64> {
        [kv, self.set_aside_bytes(), past]
            .into_iter()
            .try_fold(dense, u64::checked_add)
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

/// The devices: the cards in stage order, the expert tier cards and the
/// host. The NVMe tier has no figure here — it holds its tensors in place and
/// never loads them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Machine {
    pub cards: Vec<Card>,
    /// Expert tier cards: no stage, only routed experts, planned after every
    /// stage card in this order.
    pub tiers: Vec<Card>,
    pub host: Host,
}

impl Machine {
    /// Every card, stage cards then tiers: the index space of
    /// [`Device::Card`] and of [`Plan::cards`].
    pub fn all_cards(&self) -> impl Iterator<Item = &Card> {
        self.cards.iter().chain(&self.tiers)
    }

    /// The card at index `i` of [`Machine::all_cards`].
    #[must_use]
    pub fn card(&self, i: usize) -> Option<&Card> {
        match i.checked_sub(self.cards.len()) {
            None => self.cards.get(i),
            Some(t) => self.tiers.get(t),
        }
    }
}

/// An architecture's KV cache size: the bytes `layer` holds at `ctx_max` tokens.
pub trait KvBytes {
    fn layer_bytes(&self, layer: usize, ctx_max: u64) -> u64;
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

/// A card's layout for a file tensor: the GPU loader's `DevWeight` formats
/// (`crates/gpu/src/weights.rs`). [`CardFormat::buffer_bytes`] is the one
/// owner of their file → device arithmetic: the plan sums it and counts its
/// buffers through the allocator, and the loader refuses and checks every
/// upload by it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CardFormat {
    /// The file's rows as they are stored, their byte stream as u32 words,
    /// zero-padded at its end to `rows · ceil(words / rows)` words: every
    /// q3_K/q4_K/q5_K/q6_K tensor, and a routed q5_1, q8_0, iq3_xxs, iq4_xs
    /// or iq4_nl stack a
    /// program's card experts read in the file's `block_q5_1`, `block_q8_0`,
    /// `block_iq3_xxs`, `block_iq4_xs` or `block_iq4_nl`
    /// bytes (a program's [`RoutedFormat`] names it; [`CardFormat::of`] keeps
    /// q5_1 in [`CardFormat::Q5_1`] and q8_0 in [`CardFormat::Q8_0Planes`]).
    /// The kernels address a row by its byte offset in the stream, so a row
    /// need not start on a word (q6_K at k = 768 is 630 bytes); the padding
    /// only makes the buffer a whole number of words per row.
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
    /// Decoded to f32 at load (`dequant_row`): bf16 by its type, and a
    /// K-quant that [`CardFormat::of_role`] gives it by its role.
    Bf16AsF32,
    /// bf16 as the file stores it, two values to a little-endian u32 word
    /// (value `2j` in the low half of word `j`), for a kernel that widens on
    /// the card. No file type loads in it by [`CardFormat::of`]: a reader
    /// picks it for its own tensors (the DSpark draft's Markov weights and
    /// the target embedding rows it reads), and uploads it itself; the GPU
    /// loader's `Weights` refuses it.
    Bf16Raw,
}

impl CardFormat {
    /// The format a file tensor of type `ty` loads in on a card; `None` when
    /// the loader has none. `None` is not a refusal by itself: a routed stack
    /// of such a type stays on the host, and a tensor its role puts on a card
    /// is refused by the plan with [`PlacementError::NoCardFormat`]. Every
    /// type is listed by name, so a new [`GgmlType`] must be decided here.
    #[must_use]
    pub fn of(ty: GgmlType) -> Option<CardFormat> {
        match ty {
            GgmlType::Q3_K | GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q6_K => {
                Some(CardFormat::KQuant)
            }
            GgmlType::Q5_0 => Some(CardFormat::Q5_0),
            GgmlType::Q5_1 => Some(CardFormat::Q5_1),
            GgmlType::Q8_0 => Some(CardFormat::Q8_0Planes),
            GgmlType::F32 => Some(CardFormat::F32),
            GgmlType::BF16 => Some(CardFormat::Bf16AsF32),
            GgmlType::F16
            | GgmlType::MXFP4
            | GgmlType::Q2_K
            | GgmlType::IQ2_XXS
            | GgmlType::IQ2_XS
            | GgmlType::IQ3_XXS
            | GgmlType::IQ1_S
            | GgmlType::IQ4_NL
            | GgmlType::IQ3_S
            | GgmlType::IQ2_S
            | GgmlType::IQ4_XS
            | GgmlType::I8
            | GgmlType::I16
            | GgmlType::I32
            | GgmlType::I64
            | GgmlType::F64
            | GgmlType::IQ1_M
            | GgmlType::TQ1_0
            | GgmlType::TQ2_0
            | GgmlType::Unknown(_) => None,
        }
    }

    /// The device buffers of `rows` rows of `k` values of file type `ty` in
    /// this format, in bytes, in the order the upload allocates them: q8_0's
    /// code plane, then its scale plane; one buffer in every other format —
    /// the upload's own arithmetic. `None` exactly where the upload refuses: a
    /// zero dimension, `k` off the type's block, or a `ty` this format is not
    /// the loader's format for.
    #[must_use]
    pub fn buffer_bytes(self, ty: GgmlType, k: u64, rows: u64) -> Option<Vec<u64>> {
        let blck = ty.blck_size()?;
        if !self.holds(ty) || k == 0 || rows == 0 || !k.is_multiple_of(blck) {
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
                vec![words.div_ceil(rows).checked_mul(rows)?.checked_mul(4)?]
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
            CardFormat::Bf16Raw => {
                // Whole words per row: a row never shares a word with the next.
                if !k.is_multiple_of(2) {
                    return None;
                }
                vec![rows.checked_mul(k.checked_mul(2)?)?]
            }
        })
    }

    /// The format a file tensor of type `ty` in role `role` loads in on a
    /// card: [`CardFormat::of`], except an engram gain, which the card reads
    /// as f32 values whatever its type — a K-quant one is decoded at load.
    #[must_use]
    pub fn of_role(ty: GgmlType, role: Role) -> Option<CardFormat> {
        match (role, CardFormat::of(ty)) {
            (Role::EngramGain, Some(CardFormat::KQuant)) => Some(CardFormat::Bf16AsF32),
            (_, f) => f,
        }
    }

    /// The format an expert stack of type `ty` loads in on a card under the
    /// default expert rule ([`plan_with`]): [`CardFormat::of`] less q5_K,
    /// which this rule does not admit. A card expert kernel does read q5_K
    /// (`kq_gate_up_act_q5k`, `q5k_gemv_sel`), and the program that runs
    /// them names q5_K in its own [`RoutedFormat`] (glm5next's
    /// `card_routed`). A stack of a type with none stays on the host.
    #[must_use]
    pub fn of_routed(ty: GgmlType) -> Option<CardFormat> {
        CardFormat::of(ty).filter(|_| ty != GgmlType::Q5_K)
    }

    /// A file tensor of type `ty` can be laid out in this format: the
    /// format [`CardFormat::of`] names for `ty`, [`CardFormat::Bf16AsF32`] for
    /// a K-quant ([`CardFormat::of_role`]'s engram gain),
    /// [`CardFormat::Bf16Raw`] for bf16, which only a reader that picks it
    /// uses, or [`CardFormat::KQuant`] for q5_1, q8_0, iq3_xxs, iq4_xs and
    /// iq4_nl, the file's blocks,
    /// which only a program's [`RoutedFormat`] picks.
    #[must_use]
    pub fn holds(self, ty: GgmlType) -> bool {
        CardFormat::of(ty) == Some(self)
            || (self == CardFormat::Bf16AsF32 && CardFormat::of(ty) == Some(CardFormat::KQuant))
            || (self == CardFormat::Bf16Raw && ty == GgmlType::BF16)
            || (self == CardFormat::KQuant
                && matches!(
                    ty,
                    GgmlType::Q5_1
                        | GgmlType::Q8_0
                        | GgmlType::IQ3_XXS
                        | GgmlType::IQ4_XS
                        | GgmlType::IQ4_NL
                ))
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

impl fmt::Display for Format {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Format::Card(CardFormat::KQuant) => "kquant",
            Format::Card(CardFormat::Q5_0) => "q5_0",
            Format::Card(CardFormat::Q5_1) => "q5_1",
            Format::Card(CardFormat::Q8_0Planes) => "q8_0_planes",
            Format::Card(CardFormat::F32) => "f32",
            Format::Card(CardFormat::Bf16AsF32) => "bf16_as_f32",
            Format::Card(CardFormat::Bf16Raw) => "bf16_raw",
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
    pub scratch_bytes: u64,
    pub context_bytes: u64,
    /// The card's reserves ([`Card::reserves`]).
    pub reserve_bytes: u64,
    /// usable − dense − experts − rounding − KV − scratch − context −
    /// reserves, usable capped by the plan's card budget; the margin is
    /// inside it.
    pub headroom_bytes: i128,
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
    /// for this plan's room ([`nvme_arena_of`]): the largest arena the room
    /// leaves above the tier's floor on a room that must move over half the
    /// host leg's bytes, else 0. Beside the plan's own bytes: the arena is
    /// the tier's memory, not a segment.
    pub nvme_arena_bytes: u64,
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
    /// usable − experts − tables − shadows − reserves; on a plan the NVMe
    /// expert tier split ([`expert_nvme_tier`]), the room the split read
    /// less the arena and the plan's host need ([`workstation::HostNeed`]),
    /// since the room binds there.
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
    /// The card byte budget the plan was made under: every card planned
    /// with `min(usable, budget)` usable bytes ([`Plan::usable_bytes`]).
    pub card_budget: Option<u64>,
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
    /// An expert list that is not one: a repeated id, an id past the stack.
    #[error("expert list: {0}")]
    Experts(String),
    /// A host room under the floor of the NVMe expert tier: the plan's host
    /// terms outside the routed experts, and the transient reads of two
    /// layers' NVMe-tier experts and one layer's host experts beside them.
    #[error(
        "the host room {room} B is under the NVMe expert tier's floor {floor} B (the host terms \
         beside the routed experts {base} B + 3 × the heaviest layer's host-served experts \
         {layer} B: two layers' transient reads and one layer's resident experts), the plan's \
         need being {need} B: free host memory, or load by a placement whose cards hold more of \
         the model"
    )]
    HostRoomFloor {
        room: u64,
        floor: u64,
        base: u64,
        layer: u64,
        need: u64,
    },
    /// An arena the lever `BLOOMERY_NVTIER_BYTES` names that the room cannot
    /// take: the host segments it leaves fall under the NVMe expert tier's
    /// floor.
    #[error(
        "the NVMe tier's arena {arena} B leaves the host room {room} B under its floor {floor} B \
         (the arena takes at most room − floor): lower the arena, or free host memory"
    )]
    NvTierArena { arena: u64, room: u64, floor: u64 },
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
    /// A tier card on a plan that keeps every routed stack on the host
    /// ([`plan_host_routed`]): no card holds a routed expert, so the tier
    /// has nothing to hold.
    #[error(
        "tier card {card} (tier {tier}) holds no expert: this plan puts every \
         routed expert on the host; plan without it"
    )]
    HostRoutedTier { card: String, tier: usize },
    /// A draft reserve on a tier the plan does not have.
    #[error(
        "the draft's reserve is on tier {on}, and the plan has {tiers} tier cards: the draft's \
         card is outside the plan"
    )]
    DraftOffPlan { on: usize, tiers: usize },
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
    /// A card budget below what the card needs with no expert on it.
    #[error(
        "card {card}: budget {budget} B is below its floor {floor} B = dense {dense} B (the \
         allocator's granules) + KV {kv} B + context {context} B + scratch {scratch} B + \
         reserves {reserves} B + margin {margin} B"
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
    },
    /// A card whose device another process holds: what it had free at plan
    /// time is below the card's floor — the dense tensors, KV, set-aside and
    /// margin, before any expert moved to the host. [`CardFreeFloor`]'s
    /// terms, boxed to keep the error small.
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
    /// The dense trunk's floor: dense, KV, set-aside and margin summed.
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
            held,
        } = self;
        write!(
            f,
            "card {card}: the device had {free} B free of its usable {usable} B at plan \
             time{held}, and the plan's dense trunk alone needs {floor} B = dense {dense} B (the \
             allocator's granules) + KV {kv} B + context {context} B + scratch {scratch} B + \
             reserves {reserves} B + margin {margin} B, past the {free} B by {over} B: free the \
             card, or plan a placement on a card with room"
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
    /// The host's tensors, the cards' ring shadows and the reserves pass its
    /// usable bytes.
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
                "card {card}: resident + rounding + KV + scratch + context + reserves = {total} B passes usable − margin = {limit} B by {} B",
                total - limit
            ),
            Violation::HostOver { total, usable } => write!(
                f,
                "host: tensors + ring shadows + reserves = {total} B pass usable {usable} B by {} B",
                total - usable
            ),
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
fn card_heaps(model: &ModelTensors, cards: &[&Card], rows: &[Row]) -> (Vec<Heap>, Vec<Violation>) {
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

/// A card segment of `rows` rows of `t` — all of it, or an expert slice —
/// in the format its role gives its type ([`CardFormat::of_role`]).
fn card_segment(
    t: &ModelTensor,
    card: usize,
    rows: u64,
    experts: Option<ExpertList>,
) -> Result<Segment, PlacementError> {
    card_segment_as(t, card, rows, experts, CardFormat::of_role(t.ty, t.role))
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
    layers: usize,
) -> Result<Row, PlacementError> {
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
            (card_segment(t, stage, rows_of(t), None)?, t.file_bytes)
        }
        Role::Head => (card_segment(t, stage, rows_of(t), None)?, t.file_bytes),
        Role::TokenEmbedding => match stages.embedding {
            Some(card) => (card_segment(t, card, rows_of(t), None)?, gathered(t)?),
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
        Role::Unread | Role::Unused => (in_place(Device::Unused, Format::Unused, None, 0), 0),
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
    routed_row_on_as(
        i,
        t,
        stage,
        on_cards,
        model,
        CardFormat::of_role(t.ty, t.role),
    )
}

/// [`routed_row_on`] with the card segments in `format`: the expert rule's
/// rows, in the format the plan's [`RoutedFormat`] counted them in. `None`
/// is refused only when a list of `on_cards` holds an expert.
fn routed_row_on_as(
    i: usize,
    t: &ModelTensor,
    stage: usize,
    on_cards: Vec<(usize, ExpertList)>,
    model: &ModelTensors,
    format: Option<CardFormat>,
) -> Result<Row, PlacementError> {
    let (rows_per_expert, file_per_expert) = per_expert(t, model.experts)?;
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
    let held = ExpertList::new(owner.iter().map(|&(e, _)| e).collect(), model.experts)?;
    let host = held.complement(model.experts)?;
    let mut segments = Vec::with_capacity(on_cards.len() + 1);
    for (card, list) in on_cards {
        if !list.is_empty() {
            let rows = list.len() * rows_per_expert;
            segments.push(card_segment_as(t, card, rows, Some(list), format)?);
        }
    }
    if !host.is_empty() {
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

/// Row `i`, the tensor `t` whole on card `card` in its card format, whatever
/// its role — for a load that fixes its own split ([`routed_row`] places the
/// routed stacks).
pub fn whole_on_card(i: usize, t: &ModelTensor, card: usize) -> Result<Row, PlacementError> {
    Ok(Row {
        tensor: i,
        segments: vec![card_segment(t, card, rows_of(t), None)?],
        read_bytes: t.file_bytes,
        stage: card,
    })
}

/// Which routed stacks a program's card expert kernels run: the card format
/// a stack of type `ty` loads in, `None` for a type none of them reads. A
/// layer with such a stack keeps its experts on the host.
pub type RoutedFormat = fn(GgmlType) -> Option<CardFormat>;

/// The eligible layers of `layers`: those that route, every stack of which
/// `routed` gives a card format — a stage card's own layers, or every layer
/// for a tier.
fn eligible(
    layers: Range<usize>,
    stacks: &[Vec<usize>],
    model: &ModelTensors,
    routed: RoutedFormat,
) -> Vec<usize> {
    layers
        .filter(|&l| {
            !stacks[l].is_empty()
                && stacks[l]
                    .iter()
                    .all(|&i| routed(model.tensors[i].ty).is_some())
        })
        .collect()
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
    experts: u64,
    budget: i128,
    footprint: impl Fn(&[u64]) -> Result<u64, PlacementError>,
) -> Result<(), PlacementError> {
    if eligible.is_empty() || i128::from(footprint(n_l)?) > budget {
        return Ok(());
    }
    let total = |n_l: &[u64], l: usize| held[l] + n_l[l];
    while let Some(&l) = eligible.iter().min_by_key(|&&l| total(n_l, l)) {
        if total(n_l, l) >= experts {
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
    routed: RoutedFormat,
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
                let Some(f) = routed(t.ty).filter(|_| eligible.contains(&layer)) else {
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

/// The placement's levers, which a binary parses once
/// ([`PlanLevers::from_levers`]): the card budget `BLOOMERY_CARD_BUDGET`
/// sets. The default is none: each card's own usable bytes.
#[derive(Clone, Debug, Default)]
pub struct PlanLevers {
    /// The card budget `BLOOMERY_CARD_BUDGET` sets, in bytes: every card of
    /// the plan plans with `min(usable, budget)`; `None` unset, each card's
    /// own usable bytes.
    pub card_budget_bytes: Option<u64>,
}

impl PlanLevers {
    /// The placement's levers of a binary's one parse.
    pub fn from_levers(levers: &bloomery_levers::Levers) -> Result<PlanLevers, PlacementError> {
        Ok(PlanLevers {
            card_budget_bytes: levers.card_budget_bytes(),
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
    plan_with(model, machine, ctx_max, kv, levers.card_budget_bytes)
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
/// `card_budget`, every card plans
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
        card_budget,
        Some(CardFormat::of_routed),
        0,
    )
}

/// [`plan`] with the expert rule run on the layers every routed stack of
/// which `routed` gives a card format, for a program whose card expert
/// kernels read other types than [`CardFormat::of_routed`]'s: the card
/// budget, or none.
pub fn plan_routed<'a>(
    model: &'a ModelTensors,
    machine: &'a Machine,
    ctx_max: u64,
    kv: &dyn KvBytes,
    levers: &PlanLevers,
    routed: RoutedFormat,
) -> Result<Plan<'a>, PlacementError> {
    plan_rule(
        model,
        machine,
        ctx_max,
        kv,
        levers.card_budget_bytes,
        Some(routed),
        0,
    )
}

/// [`plan_routed`] with `reserve` bytes of every card kept for what the
/// caller plans beside this plan on the same card — an MTP draft's granules
/// and store: the expert rule spreads within the card's budget less
/// `reserve`, and nothing else sees it. The plan's usable bytes, card budget,
/// headroom and violations are its own; the caller checks the card's bound
/// on the sum.
pub fn plan_routed_reserving<'a>(
    model: &'a ModelTensors,
    machine: &'a Machine,
    ctx_max: u64,
    kv: &dyn KvBytes,
    levers: &PlanLevers,
    routed: RoutedFormat,
    reserve: u64,
) -> Result<Plan<'a>, PlacementError> {
    plan_rule(
        model,
        machine,
        ctx_max,
        kv,
        levers.card_budget_bytes,
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
    if let Some((t, tier)) = machine.tiers.iter().enumerate().next() {
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
        levers.card_budget_bytes,
        None,
        0,
    )
}

/// [`plan_with`], the expert rule run on each card's layers eligible under
/// `routed`, and on none without it, within each card's budget less
/// `reserve` ([`plan_routed_reserving`]); the routed stacks' card segments
/// in the format `routed` gives them.
#[allow(
    clippy::too_many_arguments,
    reason = "the planners' one body: the model, machine, context and cache, and the rule's three inputs"
)]
fn plan_rule<'a>(
    model: &'a ModelTensors,
    machine: &'a Machine,
    ctx_max: u64,
    kv: &dyn KvBytes,
    card_budget: Option<u64>,
    routed: Option<RoutedFormat>,
    reserve: u64,
) -> Result<Plan<'a>, PlacementError> {
    let stages = Stages::new(model.layers, &machine.cards)?;
    let mut rows: Vec<Option<Row>> = vec![None; model.tensors.len()];
    let mut stacks: Vec<Vec<usize>> = vec![Vec::new(); model.layers];
    for (i, t) in model.tensors.iter().enumerate() {
        // A never-loaded tensor's shape is nothing a card must take.
        if let Some(format) = CardFormat::of_role(t.ty, t.role).filter(|_| t.role != Role::Unused) {
            card_bytes(t, format, rows_of(t))?;
        }
        if t.role == Role::RoutedExperts {
            stacks[layer_of(t, model.layers)?].push(i);
        } else {
            rows[i] = Some(place_whole(i, t, &stages, model.layers)?);
        }
    }
    check_tiers(&machine.tiers)?;
    check_devices(machine, card_budget)?;
    let mut n_l = vec![0u64; model.layers];
    let none_held = vec![0u64; model.layers];
    let mut kv_bytes = Vec::with_capacity(machine.cards.len() + machine.tiers.len());
    for (c, card) in machine.cards.iter().enumerate() {
        let (kv_card, shadow) = card_kv(card, kv, ctx_max);
        let eligible = routed.map_or_else(Vec::new, |f| {
            eligible(card.layers.clone(), &stacks, model, f)
        });
        let fill = Fill {
            model,
            card,
            c,
            rows: &rows,
            eligible: &eligible,
            routed,
            card_budget,
            kv: kv_card,
            reserve,
        };
        fill.run(&mut n_l, &none_held)?;
        kv_bytes.push((kv_card, shadow));
    }
    let mut tier_n_l: Vec<Vec<u64>> = Vec::with_capacity(machine.tiers.len());
    for (t, tier) in machine.tiers.iter().enumerate() {
        let (kv_tier, shadow) = card_kv(tier, kv, ctx_max);
        let mut held = n_l.clone();
        for prev in &tier_n_l {
            for (h, p) in held.iter_mut().zip(prev) {
                *h += p;
            }
        }
        let eligible =
            routed.map_or_else(Vec::new, |f| eligible(0..model.layers, &stacks, model, f));
        let fill = Fill {
            model,
            card: tier,
            c: machine.cards.len() + t,
            rows: &rows,
            eligible: &eligible,
            routed,
            card_budget,
            kv: kv_tier,
            reserve,
        };
        let mut own = vec![0u64; model.layers];
        fill.run(&mut own, &held)?;
        // Only a layer whose stacks a card can hold leaves the tier work. A
        // planner with no card format has no tier to reach here
        // (`plan_host_routed` refuses one), and its empty `eligible` would
        // refuse the tier as idle.
        let left = eligible.iter().any(|&l| held[l] < model.experts);
        if !left && own.iter().all(|&n| n == 0) {
            return Err(PlacementError::IdleTier {
                card: tier.name.clone(),
                tier: t,
            });
        }
        tier_n_l.push(own);
        kv_bytes.push((kv_tier, shadow));
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
                routed.and_then(|f| f(t.ty)),
            )?);
        }
    }
    let rows: Vec<Row> = rows.into_iter().flatten().collect();
    Ok(totals(
        model,
        machine,
        ctx_max,
        rows,
        &kv_bytes,
        (n_l, tier_n_l),
        card_budget,
    ))
}

/// A card's KV cache and its ring shadows over the layers its stage runs;
/// none on a tier.
fn card_kv(card: &Card, kv: &dyn KvBytes, ctx_max: u64) -> (u64, u64) {
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
    (kv_card, shadow)
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
fn check_devices(machine: &Machine, card_budget: Option<u64>) -> Result<(), PlacementError> {
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
        let budgets = same
            .iter()
            .try_fold(0u64, |sum, c| sum.checked_add(capped(c, card_budget)));
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
/// usable − KV − set-aside − margin − `reserve`, its uploads over `eligible`.
struct Fill<'m, 'r> {
    model: &'m ModelTensors,
    card: &'m Card,
    c: usize,
    rows: &'r [Option<Row>],
    eligible: &'r [usize],
    routed: Option<RoutedFormat>,
    card_budget: Option<u64>,
    kv: u64,
    /// What the caller plans beside this plan on the card
    /// ([`plan_routed_reserving`]).
    reserve: u64,
}

impl Fill<'_, '_> {
    /// Fill `n_l` ([`spread`]) beside the counts `held` on other devices;
    /// under a card budget, refuse a card whose floor passes it first, and
    /// under a free reading below the card's usable bytes, refuse a card
    /// whose floor passes what the device had free.
    fn run(&self, n_l: &mut [u64], held: &[u64]) -> Result<(), PlacementError> {
        let card = self.card;
        let format = self.routed.unwrap_or(CardFormat::of_routed);
        let uploads = card_uploads(self.model, self.c, self.rows, self.eligible, format)?;
        let budget = i128::from(capped(card, self.card_budget))
            - i128::from(self.kv)
            - i128::from(card.set_aside_bytes())
            - i128::from(card.margin_bytes)
            - i128::from(self.reserve);
        if let Some(b) = self.card_budget {
            let dense = footprint(card.granule_bytes, &uploads, n_l)?;
            check_floor(card, b, dense, self.kv)?;
        }
        if let Some(free) = card.free_bytes.filter(|&f| f < card.usable_bytes) {
            let dense = footprint(card.granule_bytes, &uploads, n_l)?;
            check_free_floor(card, free, dense, self.kv)?;
        }
        spread(n_l, held, self.eligible, self.model.experts, budget, |n| {
            footprint(card.granule_bytes, &uploads, n)
        })
    }
}

/// Refuse `card` under the card budget `budget` when its floor — the
/// granules of its dense uploads `dense`, its KV `kv`, context, scratch,
/// reserves and margin — passes the budget. A floor within a budget above the card's own
/// usable bytes but past those shows up in [`Plan::violations`], as without
/// a budget: the card, not the budget, is too small.
fn check_floor(card: &Card, budget: u64, dense: u64, kv: u64) -> Result<(), PlacementError> {
    let floor = card.floor_bytes(dense, kv, card.margin_bytes);
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
            reserves: card.reserve_bytes(),
            margin: card.margin_bytes,
        }),
    }
}

/// Refuse `card` when its floor — the same terms [`check_floor`] sums, the
/// context less the part the reading already carries
/// ([`Card::context_in_free`]) — passes `free`, what its device had free at
/// plan time: another process holds enough of it that no expert rule can save
/// the plan, because the dense tensors alone do not fit. A floor within the
/// free bytes of a card whose census read none plans as before.
fn check_free_floor(card: &Card, free: u64, dense: u64, kv: u64) -> Result<(), PlacementError> {
    let context = card.context_bytes - card.context_in_free();
    let floor = card
        .floor_bytes(dense, kv, card.margin_bytes)
        .map(|f| f - card.context_in_free());
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
            reserves: card.reserve_bytes(),
            margin: card.margin_bytes,
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
    /// The card's floor over all of it ([`Card::floor_bytes`]); `u64::MAX`
    /// when the sum passes u64 bytes.
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
             of buffers) + cache {} B + context {} B + scratch {} B + reserves {} B + ",
            c.name,
            if self.fits() { "fits" } else { "does not fit" },
            self.need,
            self.weights,
            self.weights_unrounded,
            self.kv,
            c.context_bytes,
            c.scratch_bytes,
            c.reserve_bytes(),
        )?;
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
/// ([`CardFormat::of_role`]) and each one's buffers through the card's
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
        let format =
            CardFormat::of_role(t.ty, t.role).ok_or_else(|| PlacementError::NoCardFormat {
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
        .unwrap_or(u64::MAX);
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
/// ([`ModelTensor::gathered_rows`]) for every position, each row on at most
/// `1 + ⌈(row bytes − 1) / page⌉` pages of [`workstation::ROW_PAGE`] — the
/// ids are hashed, so a call's rows are counted on pages of their own, and
/// a row may straddle a page boundary. The reader advises and copies one
/// call at a time, so one call's pages are what must stay resident until
/// its copy. A table with no per-token row count, or a room past u64, is
/// refused by name.
pub fn row_room(model: &ModelTensors, positions: u64) -> Result<u64, PlacementError> {
    let page = workstation::ROW_PAGE;
    model
        .tensors
        .iter()
        .filter(|t| t.role == Role::EngramTable)
        .try_fold(0u64, |sum, t| {
            let rows = t.gathered_rows.ok_or_else(|| {
                PlacementError::tensor(t, "a row-gathered table without a per-token row count")
            })?;
            let pages = 1 + row_bytes(t)?.saturating_sub(1).div_ceil(page);
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

/// Where `plan`'s row-gathered tables ([`Role::EngramTable`]) are read
/// from, the one rule for a plan whose maker sizes their reader: on the
/// host in their file bytes when `room` — the host's available bytes
/// ([`workstation::host_room`]'s reading, or a room the caller gives) —
/// holds the plan's host need with them there ([`workstation::HostNeed`],
/// no churn pool beside it); else on the NVMe tier, where the role rule
/// placed them ([`place_whole`]), with their reader's room for a call of up
/// to `positions` positions ([`row_room`]) as the plan's row reserve, in
/// place of the row cache [`totals`] set aside. A plan with no such table
/// is left as it is; [`Plan::row_tier`] reads the outcome. A table of other
/// than one NVMe file segment, or a sum past u64, is refused by name.
pub fn row_table_tier(
    plan: &mut Plan<'_>,
    room: u64,
    positions: u64,
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
    let new = if host_arm.bytes() <= room {
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
        row_room(model, positions)?
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
/// The arena bytes the split dial gives this plan's room: the lever
/// `BLOOMERY_NVTIER_BYTES` (`given`) when set — refused by name where it
/// pushes the host segments the room leaves under the tier's `floor` — else
/// the largest arena the room leaves above the floor (`room − floor`) when
/// the room cannot hold half the host leg's `held` routed-expert bytes (a
/// deep split: the prompt is NVMe-bound either way, so prefill loses little
/// while decode's warm set doubles), else 0 (a room that covers the host
/// leg: the split keeps every host expert it can and the arena is the
/// lever's alone). The one owner of the arena's bytes; the plan states what
/// it chose in [`HostTotals::nvme_arena_bytes`].
pub fn nvme_arena_of(
    room: u64,
    held: u64,
    floor: u64,
    given: Option<u64>,
) -> Result<u64, PlacementError> {
    let max = room - floor;
    match given {
        Some(arena) if arena > max => Err(PlacementError::NvTierArena { arena, room, floor }),
        Some(arena) => Ok(arena),
        None => Ok(u64::from(room < held / 2) * max),
    }
}

/// The floor is `base + 3 W`: `base` the need with no routed expert on the
/// host, `W` the host bytes of the layer with the most — the room's host
/// terms beside the routed experts hold `1 W` of host-served expert
/// segments, `2 W` of the prompt run-ahead's page-cache window (a host
/// reserve of the split plan) and the RAM arena [`nvme_arena_of`] picks:
/// `1 W + 2 W + arena = room − base` at the largest arena,
/// `room − base − 3 W`. A room under it is refused by name; the plan's
/// `host.experts` counts the experts the host leg serves, on either tier.
/// The split plan's headroom is what the room leaves past the arena and the
/// host need, the budget a churn pool beside them must fit.
pub fn expert_nvme_tier(plan: &mut Plan<'_>, room: u64) -> Result<(), PlacementError> {
    let model = plan.model;
    let need = workstation::HostNeed::of(plan, 0);
    if need.bytes() <= room {
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
        let (_, per) = per_expert(t, model.experts)?;
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
    let floor = 3u64
        .checked_mul(heavy)
        .and_then(|w| w.checked_add(base))
        .ok_or_else(|| PlacementError::Metadata {
            key: "the NVMe expert tier's floor".to_string(),
            detail: "passes u64 bytes".to_string(),
        })?;
    if room < floor {
        return Err(PlacementError::HostRoomFloor {
            room,
            floor,
            base,
            layer: heavy,
            need: need.bytes(),
        });
    }
    // The split dial: the arena's bytes come out of the host segments the
    // room holds, the split running on the room that remains.
    let held = plan.host.expert_bytes;
    let given = bloomery_levers::nvtier_levers()
        .map_err(|e| PlacementError::HostRoom(format!("BLOOMERY_NVTIER_BYTES: {e}")))?
        .bytes;
    let arena = nvme_arena_of(room, held, floor, given)?;
    let room = room - arena;
    plan.host.nvme_arena_bytes = arena;
    // The prompt run-ahead's page-cache window, `2 W`, is a host reserve of
    // the split plan: the host keeps `keep` of its `held` expert bytes in
    // what the room leaves past the arena and the window, which the floor
    // keeps at least one layer's worth.
    let window = 2 * heavy;
    plan.host.reserve_bytes += window;
    let keep = held - (need.bytes() + window - room);
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
            let (_, per) = per_expert(t, model.experts)?;
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
    plan.host.headroom_bytes =
        i128::from(room) - i128::from(workstation::HostNeed::of(plan, 0).plan_bytes());
    Ok(())
}

/// The per-device sums of a finished set of rows; `kv_bytes` is, per card of
/// [`Machine::all_cards`], its layers' cache and their ring shadows, which
/// the host holds; `counts` the stage cards' `n_l` and the tiers'.
fn totals<'a>(
    model: &'a ModelTensors,
    machine: &'a Machine,
    ctx_max: u64,
    rows: Vec<Row>,
    kv_bytes: &[(u64, u64)],
    counts: (Vec<u64>, Vec<Vec<u64>>),
    card_budget: Option<u64>,
) -> Plan<'a> {
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
    let (heaps, _) = card_heaps(model, &cards, &rows);
    let held = |c: usize, card: &Card| -> u64 {
        match c.checked_sub(machine.cards.len()) {
            None => card.layers.clone().map(|l| n_l[l]).sum(),
            Some(t) => tier_n_l[t].iter().sum(),
        }
    };
    let card_totals = cards
        .iter()
        .zip(&heaps)
        .enumerate()
        .map(|(c, (card, heap))| {
            let rounding = heap.taken - (card_dense[c] + card_experts[c]);
            let (kv, _) = kv_bytes[c];
            let used = heap.taken + kv + card.set_aside_bytes();
            CardTotals {
                dense_bytes: card_dense[c],
                expert_bytes: card_experts[c],
                rounding_bytes: rounding,
                experts: held(c, card),
                kv_bytes: kv,
                scratch_bytes: card.scratch_bytes,
                context_bytes: card.context_bytes,
                reserve_bytes: card.reserve_bytes(),
                headroom_bytes: i128::from(capped(card, card_budget)) - i128::from(used),
            }
        })
        .collect();
    let row_reserve = row_cache_reserve(model, &rows);
    let reserve_bytes: u64 =
        machine.host.reserves.iter().map(|(_, b)| b).sum::<u64>() + row_reserve;
    let shadow_bytes: u64 = kv_bytes.iter().map(|&(_, shadow)| shadow).sum();
    let host = HostTotals {
        expert_bytes: host_experts,
        experts: (0..model.layers)
            .filter(|&l| routing[l])
            .map(|l| model.experts - n_l[l] - tier_n_l.iter().map(|t| t[l]).sum::<u64>())
            .sum(),
        nvme_expert_bytes: nvme_experts,
        nvme_arena_bytes: 0,
        table_bytes: host_tables,
        shadow_bytes,
        reserve_bytes,
        row_reserve_bytes: row_reserve,
        headroom_bytes: i128::from(machine.host.usable_bytes)
            - i128::from(host_experts + host_tables + shadow_bytes + reserve_bytes),
    };
    Plan {
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
    }
}

impl Plan<'_> {
    /// The usable bytes `card` planned with: its own, capped by the device's
    /// free bytes at plan time and by the plan's card budget
    /// ([`capped`]'s one cap).
    #[must_use]
    pub fn usable_bytes(&self, card: &Card) -> u64 {
        capped(card, self.card_budget)
    }

    /// Every invariant the rows break, re-derived from the rows themselves:
    /// each tensor placed once; an expert stack's segments cover every expert
    /// once and a whole tensor is one segment; a card segment's buffers sum to
    /// its resident bytes; a card holds only what its stage uses, a tier card
    /// only routed experts; the granules of its uploads + KV + scratch +
    /// context + reserves ≤ usable − margin on each card, usable capped by the
    /// card budget;
    /// the host's tensors, the cards' ring shadows and the reserves ≤ its
    /// usable bytes.
    pub fn violations(&self) -> Vec<Violation> {
        let model = self.model;
        let cards: Vec<&Card> = self.machine.all_cards().collect();
        let stage_cards = self.machine.cards.len();
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
        let (heaps, unsized_segments) = card_heaps(model, &cards, &self.rows);
        out.extend(unsized_segments);
        for ((card, totals), heap) in cards.iter().zip(&self.cards).zip(&heaps) {
            let total = heap.taken + totals.kv_bytes + card.set_aside_bytes();
            let limit = self.usable_bytes(card).saturating_sub(card.margin_bytes);
            if total > limit {
                out.push(Violation::CardOver {
                    card: card.name.clone(),
                    total,
                    limit,
                });
            }
        }
        let host = &self.machine.host;
        let total = host_resident
            + self.host.shadow_bytes
            + host.reserves.iter().map(|(_, b)| b).sum::<u64>()
            + self.host.row_reserve_bytes;
        if total > host.usable_bytes {
            out.push(Violation::HostOver {
                total,
                usable: host.usable_bytes,
            });
        }
        out
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
        }
    }

    struct NoKv;

    impl KvBytes for NoKv {
        fn layer_bytes(&self, _layer: usize, _ctx_max: u64) -> u64 {
            0
        }
    }

    const HEAD: u64 = 4_352;
    const EXPERT: u64 = 576;

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
            host: host(),
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
            host: host(),
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
            host: host(),
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
            host: host(),
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
        let mut odd = layered(3);
        odd.tensors[1].ty = GgmlType::F16;
        match whole_need(&odd, &at(need), KV, load) {
            Err(e @ PlacementError::NoCardFormat { .. }) => assert_eq!(
                e.to_string(),
                "tensor experts 0: type f16 has no device format, but its role (routed) puts it \
                 on a card"
            ),
            Err(e) => panic!("{e}, not NoCardFormat"),
            Ok(n) => panic!("an f16 stack was sized as {n:?}"),
        }
    }

    /// The whole-fit search over contexts stops where the program's arena
    /// does: the verdict's `fits` falls as the context grows, so the grid
    /// search ([`ctx::searched`]) over it lands on the last context the
    /// arena and its reserve leave room for. A card whose reading is under
    /// the arena alone — `1_151_139_840` B free against the `1_642_419_556` B
    /// arena a Qwen3.6 whole load refused on a 3090 (the gate census's
    /// reading and the arena check's message) — holds no context: the
    /// search says so, and the caller takes the placed plan. The same
    /// search with the arena left out of the need finds a context past the
    /// floor, whose load the arena check then refuses. The cache is
    /// 16,384 B a position and the model the layered fixture's: the rows'
    /// own terms, not a file's.
    #[test]
    fn the_whole_search_stops_where_the_arena_does() {
        const KV_ROW: u64 = 16_384;
        const RESERVE: u64 = 7_000;
        const FLOOR: usize = 4096;
        const GRAN: usize = 1024;
        const TRAINED: usize = 262_144;
        let model = layered(3);
        let found = |free: u64, arena: u64| {
            let at = |kv: u64| {
                let card = Card {
                    context_bytes: workstation::CONTEXT,
                    scratch_bytes: workstation::SCRATCH,
                    margin_bytes: 90_000,
                    granule_bytes: NonZeroU64::new(4096).expect("not zero"),
                    free_bytes: Some(free),
                    ..bytes_card("card", u64::MAX, 0..3)
                };
                let load = WholeLoad::Checked {
                    arena,
                    reserve: RESERVE,
                };
                whole_need(&model, &card, kv, load).expect("every tensor has a format")
            };
            let fits = |c: usize| Ok::<_, std::convert::Infallible>(at(c as u64 * KV_ROW).fits());
            ctx::searched(Some(TRAINED), FLOOR, GRAN, &fits).expect("infallible")
        };
        let weights = 9 * 4096;
        let rest = weights + workstation::CONTEXT + workstation::SCRATCH + RESERVE;
        let (free, arena) = (1_151_139_840, 1_642_419_556);
        assert_eq!(
            found(free, arena),
            None,
            "the arena alone passes the reading"
        );
        // The arena left out: the cache's room is the reading less the rest.
        let room = (free - rest) / KV_ROW;
        let want = FLOOR + (usize::try_from(room).expect("small") - FLOOR) / GRAN * GRAN;
        assert_eq!(
            found(free, 0),
            Some(want),
            "{room} positions of room without the arena"
        );
        assert!(want > FLOOR);
        // A reading with room for the arena and 20,000 positions of cache.
        let free = rest + arena + 20_000 * KV_ROW;
        assert_eq!(found(free, arena), Some(FLOOR + 15 * GRAN), "19,456");
        // Past the floor by the arena's own positions: 120,245 of room.
        assert_eq!(
            found(free, 0),
            Some(FLOOR + (20_000 + 100_245 - FLOOR) / GRAN * GRAN)
        );
        // One position short of the floor's cache: no context.
        assert_eq!(found(rest + arena + FLOOR as u64 * KV_ROW - 1, arena), None);
        assert_eq!(
            found(rest + arena + FLOOR as u64 * KV_ROW, arena),
            Some(FLOOR),
            "the floor's cache and nothing more"
        );
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
            host: host(),
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
            host: host(),
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
}
