use super::Role;
use super::card_budget;
use super::machine::{
    Card, Device, Format, Heap, Machine, ModelTensor, ModelTensors, stack_experts,
};
use super::planner::{PlanLevers, capped};
use super::refusal::{PlacementError, Violation};
use super::room::{row_cache_reserve, row_read};
use super::rule::card_heaps;
use super::table::{CardTotals, HostTotals, Plan, Row, Segment};
use super::workstation;

/// The rounds [`plan_in_pool`] lowers the card budget before it refuses by
/// name. A round lowers the budget by at least a [`workstation::GRANULE`],
/// and a shortfall under one expert's card bytes takes a round a granule
/// until an expert leaves the card, so the rounds cover an expert of 128 MiB;
/// a plan still short past them meets a floor that falls too slowly for a
/// load to wait on.
pub(super) const POOL_ROUNDS: u32 = 64;

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
pub(super) fn totals<'a>(
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
            let counts = c
                .checked_sub(machine.cards.len())
                .map_or(n_l.as_slice(), |t| tier_n_l[t].as_slice());
            let reserve_bytes = card.selected_reserve_bytes(model, counts)?;
            let used = card
                .floor_with_reserves(heap.taken, kv, stage_rows, reserve_bytes)
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
                reserve_bytes,
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
    card.floor_with_reserves(
        heap.taken,
        totals.kv_bytes,
        totals.stage_rows_bytes,
        totals.reserve_bytes,
    )
    .expect("the plan's card totals fit u64")
}

impl Plan<'_> {
    /// One named reserve at its selected amount, not the machine's scratch
    /// bound. Missing or duplicate rows and an invalid card return `None`;
    /// invalid scratch descriptors propagate their named error.
    pub fn card_reserve(&self, c: usize, name: &str) -> Result<Option<u64>, PlacementError> {
        let Some(card) = self.machine.card(c) else {
            return Ok(None);
        };
        let mut rows = card.reserves.iter().filter(|(n, _)| n == name);
        let Some((_, bytes)) = rows.next() else {
            return Ok(None);
        };
        if rows.next().is_some() {
            return Ok(None);
        }
        if name != workstation::TIER_BATCH_RESERVE || card.tier_leg.is_none() {
            return Ok(Some(*bytes));
        }
        let counts = match c.checked_sub(self.machine.cards.len()) {
            None => self.n_l.as_slice(),
            Some(t) => match self.tier_n_l.get(t) {
                Some(counts) => counts.as_slice(),
                None => return Ok(None),
            },
        };
        match card.selected_scratch(self.model, counts)? {
            Some((bound, selected)) => bytes
                .checked_sub(bound)
                .and_then(|bytes| bytes.checked_add(selected))
                .map(Some)
                .ok_or_else(|| card.bytes_past_u64()),
            None => Ok(Some(*bytes)),
        }
    }

    pub(super) fn reserve_beside(&mut self, reserve: u64) -> Result<(), PlacementError> {
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
        if let Err(short) = need.check(&(
            workstation::HostNeed::machine_room(host),
            workstation::HostRead::Given,
        )) {
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
