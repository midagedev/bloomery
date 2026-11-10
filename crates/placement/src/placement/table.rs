use std::ops::Range;
use std::sync::Arc;

use bloomery_levers::KvHost;

use super::Role;
use super::card_budget;
use super::machine::{
    Device, ExpertList, Format, Machine, ModelTensor, ModelTensors, stack_experts,
};
use super::planner::capped;
use super::rank::ExpertRank;
use super::refusal::{PlacementError, Violation};
use super::rule::{card_buffers, card_heaps, per_expert, routed_row_on_as, row_bytes, rows_of};
use super::workstation;

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
    /// The selected reserves; a tier leg's descriptor bound is replaced by
    /// its actual resident layers' scratch.
    pub reserve_bytes: u64,
    /// The card bytes the model's first prompt pass makes beyond the load's
    /// uploads (its batch buffers at the largest call), reserved out of the
    /// expert budget by the model's plan ([`CardTotals::grow_prompt`]); 0 for
    /// a plan that counts none.
    pub prompt_bytes: u64,
    /// The card bytes the expert stream's ring takes when a prompt call streams
    /// its host experts (`BLOOMERY_XSTREAM=split`): the ring's two halves of
    /// [`CardTotals::ring_half`] slots and the buffers of the ring's pass: a floor
    /// the model's plan reserved out of the expert budget, grown last from what
    /// every other term leaves ([`CardTotals::grow_ring`]); 0 for a plan that
    /// counts none.
    pub ring_bytes: u64,
    /// The slots of each half of the ring [`CardTotals::ring_bytes`] counts; 0
    /// with it.
    pub ring_half: u64,
    /// The device bytes the driver keeps for the graphs the load captures past
    /// its load reading (its drafts' verify passes and walks, its rounds'
    /// passes: [`super::graphs::LoadCaptures`]), reserved out of the expert
    /// budget by the model's plan ([`CardTotals::grow_graphs`]); 0 for a plan
    /// whose load captures nothing past its reading.
    pub graph_bytes: u64,
    /// usable − dense − experts − rounding − KV − stage rows − scratch −
    /// context − reserves − prompt − ring − graphs, usable capped by the plan's card budget; the
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
        let sum: u128 = self.terms().into_iter().map(|(_, b)| u128::from(b)).sum();
        u64::try_from(sum).unwrap_or(u64::MAX)
    }

    /// [`CardTotals::bytes`] without the prompt and ring terms: the bytes a prompt
    /// call's reserve (the batch buffers and the ring's floor, which the plan counts
    /// as those two terms) is checked to fit beside. The one named variant of the sum;
    /// every other term stays in it.
    #[must_use]
    pub fn bytes_beside_prompt(&self) -> u64 {
        let sum: u128 = self
            .terms()
            .into_iter()
            .filter(|(name, _)| !matches!(*name, "prompt" | "ring"))
            .map(|(_, b)| u128::from(b))
            .sum();
        u64::try_from(sum).unwrap_or(u64::MAX)
    }

    /// Every term [`CardTotals::bytes`] sums, by the name the records and
    /// refusals give it: the one list both the sum and a message naming the
    /// terms read ([`super::card_made`]).
    #[must_use]
    pub fn terms(&self) -> [(&'static str, u64); 11] {
        [
            ("dense", self.dense_bytes),
            ("experts", self.expert_bytes),
            ("rounding", self.rounding_bytes),
            ("kv", self.kv_bytes),
            ("stage_rows", self.stage_rows_bytes),
            ("scratch", self.scratch_bytes),
            ("context", self.context_bytes),
            ("reserves", self.reserve_bytes),
            ("prompt", self.prompt_bytes),
            ("ring", self.ring_bytes),
            ("graphs", self.graph_bytes),
        ]
    }

    /// `bytes` more in the card's KV class, out of its headroom: the one way
    /// a model's plan adds a term to a card after the totals (the bytes its
    /// resident sequences hold beside their stores, reserved out of the
    /// expert budget), so the headroom stays usable less every term.
    pub fn grow_kv(&mut self, bytes: u64) {
        self.kv_bytes += bytes;
        self.headroom_bytes -= i128::from(bytes);
    }

    /// `bytes` more in the card's reserves, out of its headroom: a draft's
    /// card bytes planned beside the target's totals (the MTP plan's draft term
    /// is the reserve the expert rule spread within, never a row of the
    /// target's totals), so the card-byte invariant's count holds both.
    pub fn grow_reserve(&mut self, bytes: u64) {
        self.reserve_bytes += bytes;
        self.headroom_bytes -= i128::from(bytes);
    }

    /// `bytes` more in the card's prompt term, out of its headroom: the card
    /// bytes the first prompt pass makes (the model's one formula of them,
    /// reserved out of the expert budget before the experts fill), so the
    /// card-byte invariant's count at the first prompt holds them.
    pub fn grow_prompt(&mut self, bytes: u64) {
        self.prompt_bytes += bytes;
        self.headroom_bytes -= i128::from(bytes);
    }

    /// `bytes` more in the card's graphs term, out of its headroom: the device
    /// bytes the graphs the load captures past its load reading keep
    /// ([`super::graphs::LoadCaptures::card_bytes`], reserved out of the expert
    /// budget before the experts fill), so the card-byte invariant's count
    /// after the load's captures holds them.
    pub fn grow_graphs(&mut self, bytes: u64) {
        self.graph_bytes += bytes;
        self.headroom_bytes -= i128::from(bytes);
    }

    /// The expert stream's ring of `half` slots a half, `bytes` of card in all, as a named
    /// term out of the headroom: counted last, after every other term, so the experts the
    /// plan places are those of a plan with the ring's floor reserved beside its own terms
    /// (the floor's bytes still in the headroom) and the ring takes the rest from what they
    /// left. Counted once: a second ring is refused by name.
    ///
    /// # Panics
    ///
    /// When the card already counts a ring.
    pub fn grow_ring(&mut self, bytes: u64, half: u64) {
        assert!(
            self.ring_bytes == 0 && self.ring_half == 0,
            "CardTotals::grow_ring: a card counts one ring"
        );
        self.ring_bytes = bytes;
        self.ring_half = half;
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
    /// Dense tensors held on the host: the row-gathered tables, and with no
    /// stage card every tensor the host stage runs, the file's bytes in
    /// place.
    pub table_bytes: u64,
    /// The cards' ring shadows, page-locked ([`KvBytes::shadow_bytes`]).
    pub shadow_bytes: u64,
    /// The K/V planes a host-backed store holds on the host
    /// (`BLOOMERY_KV_HOST=all`), page-rounded at the page a host-capable
    /// room takes: 0 under `off` and `spill`. Set by [`Plan::kv_host_planes`]
    /// past the headroom below, and counted with the shadows in the plan's
    /// host need ([`workstation::HostNeed`]).
    pub kv_host_bytes: u64,
    /// The host stage's cache ([`KvBytes::layer_bytes`] over its layers and
    /// the position rows beside it); 0 with a stage card.
    pub stage_kv_bytes: u64,
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
    /// The host room the plan was made at: the reading's bytes, and the room
    /// it was held to ([`Plan::plan_at`], including the machine caps) with
    /// the reading that decided it, [`workstation::HostRead::Machine`] where
    /// a cap did; `None` for a plan made without one.
    pub room: Option<(u64, (u64, workstation::HostRead))>,
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
        self.room
            .as_ref()
            .map_or_else(String::new, |(read, (planned, _))| {
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
    /// routed stacks' segments on that card, [`Segment::experts`]: the id
    /// prefix, or under a rank the layer's hottest).
    pub n_l: Vec<u64>,
    /// Per tier of [`Machine::tiers`], per layer: how many experts the tier
    /// holds — the ids after the stage card's `n_l` and the earlier
    /// tiers', or under a rank the layer's coldest, tier 0 the coldest block.
    /// The host holds the rest.
    pub tier_n_l: Vec<Vec<u64>>,
    /// The card byte budget the plan was made under: a card planned with
    /// `min(usable, budget)` usable bytes, its budget [`card_budget::of`]
    /// ([`Plan::usable_bytes`]).
    pub card_budget: card_budget::CardBudget,
    /// The card budget is the one [`plan_in_pool`] lowered to the NVMe
    /// expert tier's floor, not one the caller gave.
    pub budget_from_pool: bool,
    /// The router rank the plan took each routed layer's ids in
    /// (`PlanLevers::expert_rank`); none, the id prefix and the ids after it.
    pub expert_rank: Option<Arc<ExpertRank>>,
    /// The KV floor the plan's expert spread ran at
    /// (`PlanLevers::kv_floor`, `BLOOMERY_KV_FLOOR`): the plan's cache term
    /// holds the floor's bytes while its `ctx_max` keeps the range and rows
    /// the rooms are planned at. None, every term at `ctx_max`.
    pub kv_floor: Option<u64>,
    /// The K/V planes' backing the plan was made under
    /// (`PlanLevers::kv_host`, `BLOOMERY_KV_HOST`): under [`KvHost::All`] the
    /// cards' cache terms drop the host-able rooms ([`KvBytes::rooms`]) and the
    /// host holds them ([`HostTotals::kv_host_bytes`]); `spill` and `off`
    /// leave the card terms as they were.
    pub kv_host: KvHost,
    /// The cap on the cards' routed experts the plan was made under
    /// (`PlanLevers::card_experts`, `BLOOMERY_CARD_EXPERTS_CAP`): the plan's cards
    /// hold exactly that many. None, the budget alone decided.
    pub card_experts_cap: Option<u64>,
}

/// What the load of one stage card does about the output head ([`Plan::head_site`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadSite {
    /// The card ends its stage with the head: its weights hold the head tensors.
    Card,
    /// No card carries the head and this card runs the last layer: the head tensors stay in the
    /// file, read in place, and the load's body hands the model the head's host side.
    Host,
    /// Another stage ends the pipeline: this load carries no head.
    Elsewhere,
}

impl Plan<'_> {
    /// What the load of stage card `card` does about the output head. With a card carrying the
    /// head, that card is [`HeadSite::Card`] and the others [`HeadSite::Elsewhere`]. With none the
    /// head is read on the host: [`HeadSite::Host`] for the card that runs the last layer,
    /// [`HeadSite::Elsewhere`] for the others.
    ///
    /// # Errors
    ///
    /// [`PlacementError::Stages`] naming the tensor and where it sits, when no card carries the
    /// head and a head row is not one host-file segment: the host head reads nothing else.
    pub fn head_site(&self, card: usize) -> Result<HeadSite, PlacementError> {
        let cards = &self.machine.cards;
        if cards.iter().any(|c| c.head) {
            return Ok(if cards.get(card).is_some_and(|c| c.head) {
                HeadSite::Card
            } else {
                HeadSite::Elsewhere
            });
        }
        let elsewhere: Vec<String> = self
            .rows
            .iter()
            .filter(|r| self.model.tensors[r.tensor].role == Role::Head)
            .filter(|r| {
                !matches!(
                    r.segments.as_slice(),
                    [s] if s.device == Device::Host && s.format == Format::HostFile
                )
            })
            .map(|r| {
                format!(
                    "{} {:?}",
                    self.model.tensors[r.tensor].name,
                    r.segments
                        .iter()
                        .map(|s| (s.device, s.format))
                        .collect::<Vec<_>>()
                )
            })
            .collect();
        if !elsewhere.is_empty() {
            return Err(PlacementError::Stages(format!(
                "each stage card carries no head and the plan does not leave its tensors on the \
                 host, read in place from the file, which the host head stage reads (device, \
                 format): {}",
                elsewhere.join("; ")
            )));
        }
        let last = cards
            .get(card)
            .is_some_and(|c| c.layers.end == self.model.layers);
        Ok(if last {
            HeadSite::Host
        } else {
            HeadSite::Elsewhere
        })
    }
}

/// The one-stage reference of a one-expert-tier plan: the stage holds the union
/// of its routed ids and the tier's, and the host segments are unchanged.
/// `flat` is the same stage machine with the tier removed. The reference uses
/// the stage's actual capacity, not the artificial budget that split the sets.
pub fn union_plan<'a>(plan: &Plan<'a>, flat: &'a Machine) -> Result<Plan<'a>, PlacementError> {
    if plan.machine.cards.len() != 1
        || plan.machine.expert_tiers().count() != 1
        || flat.cards.len() != 1
        || !flat.tiers.is_empty()
        || !plan.machine.cards[0].same_device(&flat.cards[0])
    {
        return Err(PlacementError::Stages(
            "union reference needs one stage, one expert tier and the same stage without tiers"
                .into(),
        ));
    }
    let model = plan.model;
    let tier = Device::Card(plan.machine.cards.len());
    let mut out = plan.clone();
    for row in &mut out.rows {
        let t = &model.tensors[row.tensor];
        if t.role != Role::RoutedExperts {
            continue;
        }
        let ids = |d: Device| -> Vec<u32> {
            row.segments
                .iter()
                .filter(|s| s.device == d)
                .filter_map(|s| s.experts.as_ref())
                .flat_map(|e| e.ids().iter().copied())
                .collect()
        };
        let on_tier = ids(tier);
        if on_tier.is_empty() {
            continue;
        }
        let mut union = ids(Device::Card(0));
        union.extend(on_tier);
        let n = union.len() as u64;
        let mut format = None;
        for segment in row
            .segments
            .iter()
            .filter(|s| s.device == Device::Card(0) || s.device == tier)
        {
            let Format::Card(selected) = segment.format else {
                return Err(PlacementError::tensor(
                    t,
                    "union reference needs resident card expert segments",
                ));
            };
            if format.is_some_and(|f| f != selected) {
                return Err(PlacementError::tensor(
                    t,
                    "union reference cannot join different selected card formats",
                ));
            }
            format = Some(selected);
        }
        let card = routed_row_on_as(
            row.tensor,
            t,
            0,
            vec![(0, ExpertList::new(union, model.experts_of(t))?)],
            model,
            format,
        )?;
        let segments = card
            .segments
            .into_iter()
            .filter(|s| s.device == Device::Card(0))
            .chain(
                row.segments
                    .iter()
                    .filter(|s| !matches!(s.device, Device::Card(_)))
                    .cloned(),
            )
            .collect();
        *row = Row {
            segments,
            ..row.clone()
        };
        if let Some(l) = t.layer.and_then(|l| out.n_l.get_mut(l)) {
            *l = n;
        }
    }
    out.machine = flat;
    out.cards.truncate(1);
    out.tier_n_l.clear();
    out.card_budget = card_budget::CardBudget::default();
    out.budget_from_pool = false;
    out.expert_rank = None;
    let (heaps, _) = card_heaps(model, flat, &out.rows);
    let card = &flat.cards[0];
    let total = &mut out.cards[0];
    total.expert_bytes = out
        .rows
        .iter()
        .flat_map(|r| {
            r.segments.iter().filter_map(|s| {
                (model.tensors[r.tensor].role == Role::RoutedExperts && s.device == Device::Card(0))
                    .then_some(s.resident_bytes)
            })
        })
        .sum();
    total.experts = out.n_l.iter().sum();
    total.rounding_bytes = heaps[0].taken - total.dense_bytes - total.expert_bytes;
    total.reserve_bytes = card.selected_reserve_bytes(model, &out.n_l)?;
    let used = card
        .floor_with_reserves(
            heaps[0].taken,
            total.kv_bytes,
            total.stage_rows_bytes,
            total.reserve_bytes,
        )
        .ok_or_else(|| card.bytes_past_u64())?;
    total.headroom_bytes = i128::from(capped(card, None)) - i128::from(used);
    let limit = capped(card, None).saturating_sub(card.margin_bytes);
    if used > limit {
        return Err(PlacementError::Metadata {
            key: "union reference".into(),
            detail: Violation::CardOver {
                card: card.name.clone(),
                total: used,
                limit,
            }
            .to_string(),
        });
    }
    out.host.headroom_bytes = workstation::HostNeed::of_totals(flat.host(), &out.host, false, 0)
        .headroom(workstation::HostNeed::machine_room(flat.host()));
    Ok(out)
}
