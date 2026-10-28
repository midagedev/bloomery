use std::ops::Range;

use super::Role;
use super::card_budget;
use super::machine::{
    Device, ExpertList, Format, Machine, ModelTensor, ModelTensors, stack_experts,
};
use super::planner::capped;
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
