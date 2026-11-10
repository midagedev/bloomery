use super::card_budget;
use super::machine::{
    Card, ExpertList, KvBytes, Machine, ModelTensors, routed_down_acts, selected_leg_scratch,
};
use super::pool::totals;
use super::rank::ExpertRank;
use super::refusal::{CardFreeFloor, PlacementError};
use super::rule::{
    Footprint, RoutedRule, Stages, card_uploads, eligible, layer_of, place_whole, routed_row_on_as,
    spread,
};
use super::table::{Plan, Row};
use super::{CardFormat, Role};
use bloomery_levers::{KvFloor, KvHost};
use std::sync::Arc;

/// The placement's levers, which a binary parses once
/// ([`PlanLevers::from_levers`]): the card budget `BLOOMERY_CARD_BUDGET`
/// sets and the router rank `BLOOMERY_EXPERT_RANK` names. The default is
/// none of either: each card's own usable bytes, or on a unified pool the
/// budget [`plan_in_pool`] lowers to the NVMe expert tier's floor; the id
/// order.
#[derive(Clone, Debug, Default)]
pub struct PlanLevers {
    /// The card budget `BLOOMERY_CARD_BUDGET` sets, in bytes: a card of the
    /// plan plans with `min(usable, budget)`, its budget one for every card
    /// or its own entry ([`card_budget::of`]); unset, each card's own usable
    /// bytes, or on a unified pool the budget [`plan_in_pool`] lowers to the
    /// NVMe expert tier's floor. One value converts from `Option<u64>`.
    pub card_budget_bytes: card_budget::CardBudget,
    /// The router rank `BLOOMERY_EXPERT_RANK` names, read: the order the plan
    /// takes each routed layer's experts in. The stage card keeps the layer's
    /// hottest `n_l` ids and each expert tier the coldest ids, tier 0 the
    /// coldest block (`ExpertRank::split`); none, the id prefix and the ids
    /// after it. The plan refuses by name a rank that disagrees with the
    /// model, and one on a plan that places no routed expert on a card.
    pub expert_rank: Option<Arc<ExpertRank>>,
    /// The KV floor `BLOOMERY_KV_FLOOR` sets, in positions: the context the
    /// expert spread is planned at, the plan's every other term at its own
    /// `ctx_max`. `None` with no [`PlanLevers::kv_floor_default`], the spread
    /// at `ctx_max` itself — the plan made before the lever. A floor above the plan's `ctx_max` is refused
    /// by name ([`PlacementError::KvFloor`]). A binary whose loads hold
    /// their stores in rooms alone parses the lever; a plan handed one here
    /// is every plan that binary makes.
    pub kv_floor: Option<u64>,
    /// The floor a plan takes when `kv_floor` is `None` and its context is
    /// above it: [`PlanLevers::resolve_kv`] sets it from a load's default. A context no more than it plans no floor,
    /// where a `kv_floor` above the context is refused by name.
    pub kv_floor_default: Option<u64>,
    /// The K/V planes' backing `BLOOMERY_KV_HOST` names: `all` plans the
    /// host-able rooms ([`KvBytes::rooms`]'s `Select` buffers) at no card position, so the expert
    /// spread gets their bytes, and counts them against the plan's host room
    /// instead; `spill` and `off` leave the plan's card terms as they are.
    /// The type's default is [`KvHost::Off`]. `all` beside a `kv_floor` is refused by name
    /// ([`PlacementError::KvHostFloor`]): it already plans the spread at a
    /// zero card position.
    pub kv_host: KvHost,
    /// The most routed experts the plan's cards hold, all cards together,
    /// `BLOOMERY_CARD_EXPERTS_CAP` sets: the expert rule ([`spread`]) stops there
    /// even where the budget holds more, and a plan whose cards hold fewer is
    /// refused by name ([`PlacementError::CardExpertsCap`]). Since the rule's
    /// per-layer counts are a function of its total, the cap names each
    /// layer's prefix too. `None`, the budget alone decides.
    pub card_experts_cap: Option<u64>,
}

impl PlanLevers {
    /// The placement's levers of a binary's one parse: the rank file the
    /// lever names is read here, refused by name when it is not a rank.
    pub fn from_levers(levers: &bloomery_levers::Levers) -> Result<PlanLevers, PlacementError> {
        Ok(PlanLevers {
            card_budget_bytes: levers.card_budget(),
            expert_rank: levers
                .expert_rank()
                .map(|path| ExpertRank::read(path).map(Arc::new))
                .transpose()?,
            kv_floor: levers.kv_floor(),
            kv_floor_default: None,
            kv_host: levers.kv_host(),
            card_experts_cap: levers.card_experts_cap(),
        })
    }

    /// [`PlanLevers::from_levers`] for a load with its own defaults for the
    /// K/V levers it leaves unset ([`PlanLevers::resolve_kv`]).
    pub fn from_levers_with(
        levers: &bloomery_levers::Levers,
        default: KvDefault,
    ) -> Result<PlanLevers, PlacementError> {
        let host = levers.kv_host_set().and_then(KvHost::parse);
        Ok(PlanLevers::from_levers(levers)?.resolve_kv(levers.kv_floor_set(), host, default))
    }

    /// `self` with the K/V levers resolved against a load's `default`: a set
    /// `host` is its own, else `default.host`; a set floor is its own and
    /// `off` the whole context, else `default.floor` unless the host is `all`
    /// (a floor beside it is refused, and it plans at a zero floor itself).
    #[must_use]
    pub fn resolve_kv(
        self,
        floor: KvFloor,
        host: Option<KvHost>,
        default: KvDefault,
    ) -> PlanLevers {
        let host = host.unwrap_or(default.host);
        let (kv_floor, kv_floor_default) = match floor {
            KvFloor::At(n) => (Some(n), None),
            KvFloor::Off => (None, None),
            KvFloor::Unset if host == KvHost::All => (None, None),
            KvFloor::Unset => (None, Some(default.floor)),
        };
        PlanLevers {
            kv_floor,
            kv_floor_default,
            kv_host: host,
            ..self
        }
    }
}

/// What a load takes for the K/V levers it leaves unset: the floor a plan
/// spreads its experts at (a context no more than it has none) and the
/// planes' backing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvDefault {
    /// The positions of an unset `BLOOMERY_KV_FLOOR`.
    pub floor: u64,
    /// The backing of an unset `BLOOMERY_KV_HOST`.
    pub host: KvHost,
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
        levers,
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
pub(super) fn capped(card: &Card, budget: Option<u64>) -> u64 {
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
        &PlanLevers {
            card_budget_bytes: card_budget.into(),
            ..PlanLevers::default()
        },
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
    plan_rule(model, machine, ctx_max, kv, levers, Some(routed), 0)
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
    plan_rule(model, machine, ctx_max, kv, levers, Some(routed), reserve)
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
    plan_rule(model, machine, ctx_max, kv, levers, None, reserve)
}

/// [`plan_with`], the expert rule run on each card's layers eligible under
/// `routed`, and on none without it, within each card's budget less
/// `reserve` ([`plan_routed_reserving`]), the budget per card under a
/// per-card `levers.card_budget_bytes` ([`card_budget::of`], refused on a
/// machine whose card no CUDA ordinal of this process names:
/// [`card_budget::check`]); the routed stacks' card segments in the format
/// `routed` gives them, their ids by `levers.expert_rank` when it names one
/// ([`ExpertRank::split`]), refused by name when the rank disagrees with the
/// model's routed layers or the plan places no routed expert on a card.
/// A load whose cache declares no rooms refuses a K/V lever set to something it would have to
/// read: a floor count, or `BLOOMERY_KV_HOST` `spill` or `all`. Unset and `off` pass.
fn refuse_unroomed(kv: &dyn KvBytes, levers: &PlanLevers) -> Result<(), PlacementError> {
    if kv.rooms().is_some() {
        return Ok(());
    }
    if let Some(floor) = levers.kv_floor {
        return Err(PlacementError::KvUnroomed {
            lever: bloomery_levers::KV_FLOOR,
            value: floor.to_string(),
        });
    }
    let host = match levers.kv_host {
        KvHost::Off => return Ok(()),
        KvHost::Spill => "spill",
        KvHost::All => "all",
    };
    Err(PlacementError::KvUnroomed {
        lever: bloomery_levers::KV_HOST,
        value: host.to_owned(),
    })
}

#[allow(
    clippy::too_many_arguments,
    reason = "the planners' one body: the model, machine, context and cache, and the rule's three inputs"
)]
fn plan_rule<'a>(
    model: &'a ModelTensors,
    machine: &'a Machine,
    ctx_max: u64,
    kv: &dyn KvBytes,
    levers: &PlanLevers,
    routed: Option<RoutedRule>,
    reserve: u64,
) -> Result<Plan<'a>, PlacementError> {
    // The positions the expert spread's KV term is read at: the floor the
    // levers name, else the default floor where the context is above it,
    // else the cap itself, so with neither every term of the plan is read at
    // `ctx_max` exactly as it was before the lever. A named floor above the
    // cap is refused by name: the spread never runs past the context the
    // rooms' range and rows are reserved at.
    let floor_in_force = match levers.kv_floor {
        Some(floor) if floor > ctx_max => {
            return Err(PlacementError::KvFloor {
                floor,
                cap: ctx_max,
            });
        }
        Some(floor) => Some(floor),
        None => levers.kv_floor_default.filter(|&floor| floor < ctx_max),
    };
    let kv_at = floor_in_force.unwrap_or(ctx_max);
    refuse_unroomed(kv, levers)?;
    // `all` is a floor-0 plan for the planes and nothing may say another
    // floor: the two levers name where the spread runs.
    if levers.kv_host == KvHost::All
        && let Some(floor) = levers.kv_floor
    {
        return Err(PlacementError::KvHostFloor { floor });
    }
    // Under `all` the host-able rooms hold no card bytes: the plan counts every card's cache
    // through the view that drops them ([`KvView`]), the rest of the layer at the cap.
    let kv_counted = KvView {
        kv,
        host: levers.kv_host,
    };
    let kv_counted = &kv_counted;
    let card_budget = &levers.card_budget_bytes;
    let rank = levers.expert_rank.as_deref();
    let tiers = machine.tiers.len();
    let nodes = machine.nodes.len();
    if nodes != 1 && (tiers == 0 || nodes != 1 + tiers) {
        return Err(PlacementError::Nodes { nodes, tiers });
    }
    machine.pool_bytes()?;
    let stages = Stages::new(model.layers, &machine.cards)?;
    // The host stage runs its routed stacks on the host: no card holds an expert.
    let routed = if stages.host { None } else { routed };
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
                shares,
                model,
                machine,
                (kv_at, ctx_max),
                kv_counted,
                &rows,
                &stacks,
                routed,
                reserve,
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
        let (kv_card, shadow, stage_rows) = card_kv(card, kv_counted, (kv_at, ctx_max));
        let eligible = match routed {
            Some(r) => eligible(card.layers.clone(), &stacks, model, r, false)?,
            None => Vec::new(),
        };
        let fill = Fill {
            model,
            card,
            c,
            unified: machine.unified().is_some(),
            rows: &rows,
            eligible: &eligible,
            routed,
            card_budget: card_budget::of(card_budget, card),
            kv: kv_card + stage_rows,
            reserve,
            most: levers.card_experts_cap.unwrap_or(u64::MAX),
        };
        fill.run(&mut n_l, &none_held)?;
        kv_bytes.push((kv_card, shadow, stage_rows));
    }
    let mut tier_n_l: Vec<Vec<u64>> = Vec::with_capacity(machine.tiers.len());
    for (t, tier) in machine.tiers.iter().enumerate() {
        let (kv_tier, shadow, _) = card_kv(tier, kv_counted, (kv_at, ctx_max));
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
            Some(r) if !tier.reserve_only => eligible(0..model.layers, &stacks, model, r, true)?,
            _ => Vec::new(),
        };
        let fill = Fill {
            model,
            card: tier,
            c: machine.cards.len() + t,
            unified: machine.nodes[if nodes == 1 { 0 } else { t + 1 }]
                .unified
                .is_some(),
            rows: &rows,
            eligible: &eligible,
            routed,
            card_budget: card_budget::of(card_budget, tier),
            kv: kv_tier,
            reserve,
            most: levers.card_experts_cap.unwrap_or(u64::MAX),
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
    if let Some(cap) = levers.card_experts_cap {
        let held: u64 = n_l.iter().chain(tier_n_l.iter().flatten()).sum();
        if held < cap {
            return Err(PlacementError::CardExpertsCap { cap, held });
        }
    }
    if let Some(rank) = rank {
        let on_cards = n_l.iter().chain(tier_n_l.iter().flatten()).any(|&n| n != 0);
        check_rank(rank, model, &stacks, on_cards)?;
    }
    for (l, layer_stacks) in stacks.iter().enumerate() {
        if layer_stacks.is_empty() {
            continue;
        }
        let stage = stages.of_layer[l];
        let lists = match rank {
            Some(rank) => {
                let own: Vec<u64> = tier_n_l.iter().map(|own| own[l]).collect();
                let (on_stage, on_tiers) = rank.split(l, model.layer_experts(l), n_l[l], &own)?;
                std::iter::once((stage, on_stage))
                    .chain(
                        on_tiers
                            .into_iter()
                            .enumerate()
                            .map(|(t, list)| (machine.cards.len() + t, list)),
                    )
                    .collect()
            }
            None => {
                let mut lists = vec![(stage, ExpertList::range(0..n_l[l])?)];
                let mut next = n_l[l];
                for (t, own) in tier_n_l.iter().enumerate() {
                    let ids = next..next + own[l];
                    next = ids.end;
                    let list = ExpertList::range(ids)?;
                    lists.push((machine.cards.len() + t, list));
                }
                lists
            }
        };
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
    if stages.host {
        plan.host_stage_kv(host_stage_kv(model, ctx_max, kv)?);
    }
    plan.reserve_beside(reserve)?;
    plan.expert_rank = levers.expert_rank.clone();
    plan.kv_floor = floor_in_force;
    plan.kv_host = levers.kv_host;
    plan.card_experts_cap = levers.card_experts_cap;
    Ok(plan)
}

/// Refuse by name unless `rank` orders exactly the layers of `model` that
/// hold routed stacks (`stacks`, each layer's stack rows), over their experts,
/// and the filled plan places routed experts on a card (`on_cards`): a rank on
/// a plan that keeps every expert on the host would order nothing.
fn check_rank(
    rank: &ExpertRank,
    model: &ModelTensors,
    stacks: &[Vec<usize>],
    on_cards: bool,
) -> Result<(), PlacementError> {
    if !on_cards {
        return Err(PlacementError::ExpertRank {
            file: rank.path().to_string(),
            detail: "this plan places no routed expert on a card, so it has nothing to order"
                .to_string(),
        });
    }
    let routed: Vec<(usize, u64)> = stacks
        .iter()
        .enumerate()
        .filter(|(_, layer_stacks)| !layer_stacks.is_empty())
        .map(|(l, _)| (l, model.layer_experts(l)))
        .collect();
    rank.check(&routed)
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
    reason = "the planner's own inputs: shares, model, machine, contexts, cache, rows, stacks, rule"
)]
fn resolve_shares(
    shares: &[(u32, u64)],
    model: &ModelTensors,
    machine: &Machine,
    spread: (u64, u64),
    kv: &KvView<'_>,
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
        let (kv_card, _, stage_rows) = card_kv(card, kv, spread);
        let layers = if tier {
            0..model.layers
        } else {
            card.layers.clone()
        };
        let own = eligible(layers, stacks, model, routed, tier)?;
        let uploads = card_uploads(model, c, rows, &own, routed)?;
        let dense = Footprint::new(card.granule_bytes, &uploads).at(&zeros)?;
        let kv = kv_card.saturating_add(stage_rows);
        let reserves = card.selected_reserve_bytes(model, &zeros)?;
        let floor = plan_floor(card, dense, kv, reserve, reserves);
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
                reserves,
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
    let routable = eligible(0..model.layers, stacks, model, routed, false)?;
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

/// The host stage's cache: its layers' [`KvBytes::layer_bytes`] and the
/// position rows beside it, held in host memory.
fn host_stage_kv(
    model: &ModelTensors,
    ctx_max: u64,
    kv: &dyn KvBytes,
) -> Result<u64, PlacementError> {
    let past = || PlacementError::Metadata {
        key: "host stage cache".to_owned(),
        detail: format!("the cache of {} layers passes u64 bytes", model.layers),
    };
    (0..model.layers)
        .map(|l| kv.layer_bytes(l, ctx_max))
        .chain([kv.stage_position_bytes(ctx_max)])
        .try_fold(0u64, u64::checked_add)
        .ok_or_else(past)
}

/// `kv` as the plan counts a card's cache: where the expert spread's floor reads it, the rooms'
/// by-position buffers at the floor and every other byte of a layer at the cap (a load whose
/// stores declare no rooms plans every byte at the cap, whatever the floor), the host-able
/// rooms' bytes dropped under `BLOOMERY_KV_HOST=all`.
struct KvView<'a> {
    kv: &'a dyn KvBytes,
    host: KvHost,
}

impl KvView<'_> {
    /// `layer`'s card bytes with the spread read at `spread` positions and the context at `cap`.
    fn layer_at(&self, layer: usize, spread: u64, cap: u64) -> u64 {
        let whole = self.kv.layer_bytes(layer, cap);
        let Some(rooms) = self.kv.rooms() else {
            return whole;
        };
        let fixed = whole.saturating_sub(rooms.bytes(layer, cap));
        let dropped = if self.host == KvHost::All {
            rooms.host_able(layer, spread)
        } else {
            0
        };
        fixed
            .saturating_add(rooms.bytes(layer, spread))
            .saturating_sub(dropped)
    }
}

/// A card's KV cache and its ring shadows over the layers its stage runs,
/// and the stage's rows beside the cache on the card that runs layer 0;
/// none on a tier. The rooms' cache is read at the expert spread's floor
/// (`spread`, the cap itself unset), every other cache byte, the shadows and
/// the rows at the cap `ctx_max`: a room-backed cache holds its floor's pages
/// while its range and the rows beside it exist at the cap.
fn card_kv(card: &Card, kv: &KvView<'_>, (spread, ctx_max): (u64, u64)) -> (u64, u64, u64) {
    let kv_card = card
        .layers
        .clone()
        .map(|l| kv.layer_at(l, spread, ctx_max))
        .sum();
    let shadow = card
        .layers
        .clone()
        .map(|l| kv.kv.shadow_bytes(l, ctx_max))
        .sum();
    let rows = if card.layers.contains(&0) {
        kv.kv.stage_position_bytes(ctx_max)
    } else {
        0
    };
    (kv_card, shadow, rows)
}

/// Refuse a tier card that says it runs a stage.
pub(super) fn check_tiers(tiers: &[Card]) -> Result<(), PlacementError> {
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
pub(super) fn check_devices(
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
    unified: bool,
    rows: &'r [Option<Row>],
    eligible: &'r [usize],
    routed: Option<RoutedRule>,
    card_budget: Option<u64>,
    kv: u64,
    /// What the caller plans beside this plan on the card
    /// ([`plan_routed_reserving`]): out of the expert budget, and in the
    /// floor.
    reserve: u64,
    /// The most experts the plan's cards hold together
    /// ([`PlanLevers::card_experts_cap`]); `u64::MAX` is no cap.
    most: u64,
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
        // Only tier legs prepare the per-layer forms. Candidates reuse these
        // rather than scanning tensors, including when a layer has several downs.
        let mut down_forms = if card.tier_leg.is_some() {
            vec![crate::kernels::DownForms::NONE; self.model.layers]
        } else {
            Vec::new()
        };
        if card.tier_leg.is_some() {
            for (l, act) in routed_down_acts(self.model) {
                if let Some(forms) = down_forms.get_mut(l) {
                    *forms = forms.with(act);
                }
            }
        }
        let scratch = |counts: &[u64]| {
            card.tier_leg.map_or(Ok(0), |leg| {
                selected_leg_scratch(leg, counts, down_forms.iter().copied().enumerate())
            })
        };
        let bound = card
            .tier_leg
            .map(|leg| leg.bytes(leg.downs))
            .transpose()?
            .unwrap_or(0);
        let selected = scratch(n_l)?;
        let reserves = card
            .reserve_bytes()
            .and_then(|fixed| fixed.checked_sub(bound))
            .and_then(|fixed| fixed.checked_add(selected))
            .ok_or_else(|| card.bytes_past_u64())?;
        let footprint = Footprint::new(card.granule_bytes, &uploads);
        let budget = i128::from(capped(card, self.card_budget))
            - i128::from(self.kv)
            - i128::from(
                card.set_aside_bytes()
                    .ok_or_else(|| card.bytes_past_u64())?,
            )
            - i128::from(card.margin_bytes)
            - i128::from(self.reserve)
            + i128::from(bound);
        if let Some(b) = self.card_budget {
            let dense = footprint.at(n_l)?;
            check_floor(card, b, dense, self.kv, self.reserve, reserves)?;
        }
        if let Some(free) = card.free_bytes.filter(|&f| f < card.usable_bytes) {
            let dense = footprint.at(n_l)?;
            check_free_floor(
                card,
                free,
                dense,
                self.kv,
                self.reserve,
                reserves,
                self.unified,
            )?;
        }
        let caps: Vec<u64> = (0..n_l.len())
            .map(|l| self.model.layer_experts(l))
            .collect();
        spread(n_l, held, self.eligible, &caps, (budget, self.most), |n| {
            footprint
                .at(n)?
                .checked_add(scratch(n)?)
                .ok_or_else(|| card.bytes_past_u64())
        })
    }
}

/// `card`'s floor in a plan whose caller holds `beside` bytes of it beside
/// the plan ([`Fill::reserve`]): [`Card::floor_bytes`] with the margin and
/// those as its `past`. Both floor checks ([`check_floor`],
/// [`check_free_floor`]) and the share split's floors ([`resolve_shares`])
/// take it from here, using each candidate's selected reserves, so the floor
/// is the least budget at which its uploads and scratch fit beside the
/// fixed terms. `None` past u64 bytes.
fn plan_floor(card: &Card, dense: u64, kv: u64, beside: u64, reserves: u64) -> Option<u64> {
    card.margin_bytes
        .checked_add(beside)
        .and_then(|past| card.floor_with_reserves(dense, kv, past, reserves))
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
    reserves: u64,
) -> Result<(), PlacementError> {
    let floor = plan_floor(card, dense, kv, beside, reserves);
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
            reserves,
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
    reserves: u64,
    unified: bool,
) -> Result<(), PlacementError> {
    let context = card.context_bytes - card.context_in_free();
    let floor = plan_floor(card, dense, kv, beside, reserves).map(|f| f - card.context_in_free());
    match floor {
        Some(floor) if floor <= free => Ok(()),
        floor => Err(PlacementError::CardFreeFloor(Box::new(CardFreeFloor {
            card: card.name.clone(),
            unified,
            free,
            usable: card.usable_bytes,
            floor: floor.unwrap_or(u64::MAX),
            over: floor.unwrap_or(u64::MAX).saturating_sub(free),
            dense,
            kv,
            context,
            scratch: card.scratch_bytes,
            reserves,
            margin: card.margin_bytes,
            beside,
            held: card
                .held_by
                .map_or(String::new(), |h| format!(" (held by {h})")),
        }))),
    }
}
