//! The card-budget window of Kimi K3's fixture on the `bp` machine, what the gates that place the fixture
//! share (`gate_kimi_k3_place`, `gate_kimi_k3_imatrix`). The fixture's experts all fit the stage card, so a
//! plan under no budget leaves the tier card idle and is refused by name (`PlacementError::IdleTier`); the
//! window is the card budgets between the first that plans and the first that leaves the tier idle, each
//! plan in it moving experts from the tier to the stage. A gate includes it by `#[path]`, with
//! `#[allow(dead_code)]` for the helpers it does not read; the context is each gate's, passed in.

use std::num::NonZeroU64;

use bloomery_gpu_gates::GateError;
use bloomery_gpu_gates::generate::Place;
use bloomery_levers::CardBudget;
use model::arch::kimi_k3::place::{PlaceError, PlanInputs, tier_batch, tier_batch_at};
use model::placement::{Device, ExpertList, Machine, PlacementError, Plan, PlanLevers, Role};

/// The allocator granule the gates' plans count each card's uploads at. The fixture's
/// experts (about 38 KB each) fit the shared granules the dense trunk leaves open at the
/// card's 2 MiB, so no budget at or above the plan's floor splits them between the cards; at
/// a page the budget has a window of the experts' own bytes. A plan input only: the driver
/// allocates as it does, and every child a gate runs under a window's budget gets the same
/// `--granule`.
pub const GRANULE: u64 = 4 << 10;

/// The gates' `bp` machine: the A6000 the stage, the 3090 the tier, each at [`GRANULE`], the
/// tier's batch reserved.
pub fn bp_machine(inputs: &PlanInputs) -> Result<Machine, GateError> {
    bp_machine_at(inputs, None)
}

/// [`bp_machine`] with the tier's batch at `rows` rows (`None`: the step's).
pub fn bp_machine_at(inputs: &PlanInputs, rows: Option<u64>) -> Result<Machine, GateError> {
    let batch = match rows {
        Some(rows) => tier_batch_at(&inputs.hp, rows)?,
        None => tier_batch(&inputs.hp)?,
    };
    let bp = Place::Bp.on_host()?.machine(None, Some(batch))?;
    let mut machine = bp(inputs.model.layers);
    let granule = NonZeroU64::new(GRANULE).ok_or("the gate's granule is at least one byte")?;
    for card in machine.cards.iter_mut().chain(machine.tiers.iter_mut()) {
        card.granule_bytes = granule;
    }
    Ok(machine)
}

/// The plan levers of the card budget `budget`.
pub fn levers_of(budget: CardBudget) -> PlanLevers {
    PlanLevers {
        card_budget_bytes: budget,
        ..PlanLevers::default()
    }
}

/// The plan levers of a card budget of `bytes` on every card.
pub fn under(bytes: u64) -> PlanLevers {
    levers_of(CardBudget::Every(bytes))
}

/// Per routed layer, the experts its stage card, its tier card and the host hold (the first
/// routed stack's segments: every stack of a layer is split alike).
pub fn layer_splits(plan: &Plan<'_>) -> Vec<(usize, u64, u64, u64)> {
    let tier = Device::Card(plan.machine.cards.len());
    let mut out: Vec<(usize, u64, u64, u64)> = Vec::new();
    for row in &plan.rows {
        let t = &plan.model.tensors[row.tensor];
        let Some(l) = t.layer.filter(|_| t.role == Role::RoutedExperts) else {
            continue;
        };
        if out.iter().any(|o| o.0 == l) {
            continue;
        }
        let on = |d: Device| -> u64 {
            row.segments
                .iter()
                .filter(|s| s.device == d)
                .filter_map(|s| s.experts.as_ref())
                .map(ExpertList::len)
                .sum()
        };
        out.push((l, on(Device::Card(0)), on(tier), on(Device::Host)));
    }
    out
}

/// The experts the stage, the tier and the host hold across the layers.
pub fn totals(splits: &[(usize, u64, u64, u64)]) -> (u64, u64, u64) {
    splits
        .iter()
        .fold((0, 0, 0), |a, s| (a.0 + s.1, a.1 + s.2, a.2 + s.3))
}

/// One split of the experts and the first card budget that gives it.
pub struct Budget {
    pub from: u64,
    pub stage: u64,
    pub tier: u64,
    pub host: u64,
}

/// How the plan under a card budget reads.
pub enum Cut {
    /// Refused for room (the budget does not hold the stage's dense trunk), with the reason.
    Short(String),
    /// Every expert on the stage: the tier card would hold none.
    Idle,
    /// A plan: the experts the stage, the tier and the host hold.
    Split(u64, u64, u64),
}

/// The plan of `inputs` on `machine` at a context of `ctx` positions under `levers`, read as a
/// [`Cut`].
pub fn cut(inputs: &PlanInputs, machine: &Machine, ctx: u64, levers: &PlanLevers) -> Cut {
    match inputs.plan(machine, ctx, levers) {
        Ok(plan) => {
            let (stage, tier, host) = totals(&layer_splits(&plan));
            Cut::Split(stage, tier, host)
        }
        Err(PlaceError::Placement(PlacementError::IdleTier { .. })) => Cut::Idle,
        Err(e) => Cut::Short(e.to_string()),
    }
}

/// The plans the card budgets give `inputs` on `machine` at a context of `ctx` positions, one
/// [`Budget`] per distinct split: the budgets between the first that plans and the first that
/// leaves the tier idle (a plan is refused for room below the one and every expert sits on the
/// stage above the other, the window being the experts' bytes and the allocator's shared
/// granules), found by bisection from the header's budget and then scanned in 512 steps.
pub fn split_table(
    inputs: &PlanInputs,
    machine: &Machine,
    ctx: u64,
    header: u64,
) -> Result<Vec<Budget>, GateError> {
    let top = header
        .checked_mul(2)
        .ok_or("the header's budget doubles past u64")?;
    match cut(inputs, machine, ctx, &under(top)) {
        Cut::Idle => {}
        Cut::Short(why) => {
            return Err(
                format!("twice the header's budget ({top} B) is short of room: {why}").into(),
            );
        }
        Cut::Split(s, t, h) => {
            return Err(format!(
                "twice the header's budget ({top} B) still splits the experts ({s} stage, {t} tier, {h} host)"
            )
            .into());
        }
    }
    // The first budget that plans: bisect (short, not short).
    let (mut short, mut plans) = (0u64, top);
    while plans - short > 1 {
        let mid = short + (plans - short) / 2;
        if matches!(cut(inputs, machine, ctx, &under(mid)), Cut::Short(_)) {
            short = mid;
        } else {
            plans = mid;
        }
    }
    let first = plans;
    // The first budget that leaves the tier idle: bisect (not idle, idle).
    let (mut busy, mut idle) = (first, top);
    if matches!(cut(inputs, machine, ctx, &under(first)), Cut::Idle) {
        return Err(format!(
            "the first budget that plans ({first} B) already leaves the tier idle: no split"
        )
        .into());
    }
    while idle - busy > 1 {
        let mid = busy + (idle - busy) / 2;
        if matches!(cut(inputs, machine, ctx, &under(mid)), Cut::Idle) {
            idle = mid;
        } else {
            busy = mid;
        }
    }
    let step = ((idle - first) / 512).max(1);
    let mut out: Vec<Budget> = Vec::new();
    let mut b = first;
    while b < idle {
        if let Cut::Split(stage, tier, host) = cut(inputs, machine, ctx, &under(b))
            && out
                .last()
                .is_none_or(|l| (l.stage, l.tier, l.host) != (stage, tier, host))
        {
            out.push(Budget {
                from: b,
                stage,
                tier,
                host,
            });
        }
        b += step;
    }
    println!(
        "budgets: plans from {first} B, leaves the tier idle from {idle} B (header {header} B); \
         {} distinct splits (stage, tier, host experts)",
        out.len()
    );
    for o in &out {
        println!("  from {} B: {}, {}, {}", o.from, o.stage, o.tier, o.host);
    }
    Ok(out)
}

/// The budget of the most even split with experts on both cards and none on the host: the
/// first such budget whose stage and tier counts differ least, so the routed slots reach both
/// cards' experts and the card sum and the tier sum are both read.
pub fn pick_budget(table: &[Budget]) -> Result<u64, GateError> {
    table
        .iter()
        .filter(|o| o.stage > 0 && o.tier > 0 && o.host == 0)
        .min_by_key(|o| o.stage.abs_diff(o.tier))
        .map(|o| o.from)
        .ok_or_else(|| {
            "no budget puts experts on both cards and none on the host (the splits above)".into()
        })
}

/// The first budget of a split with every expert on the tier card and none on the stage or
/// the host: the real file's shape, where the dense trunk leaves the stage no room for an
/// expert.
pub fn pick_tier_only(table: &[Budget]) -> Option<u64> {
    table
        .iter()
        .find(|o| o.stage == 0 && o.tier > 0 && o.host == 0)
        .map(|o| o.from)
}
