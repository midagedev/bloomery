//! The MiMo-V2.6-Flash two-card arm (`gate_mimo2_e2e --two-card`, `just gate-gpu-mimo2-twocard` under
//! `BLOOMERY_CARD=both`): plan (b′) (`workstation::plan_bp`: the A6000 the stage card, the 3090 an expert tier holding
//! each routed layer's next ids after the stage's prefix, the host the rest), planned under one card budget no
//! environment moves ([`TWO_BUDGET`]), against the reference — the same plan with every tier segment on the stage card
//! (`model::placement::union_plan`), the A6000 alone, opened first and released. Its clauses:
//! - `two_card_load`: the plan puts experts on the tier, and the load's tier opened on the 3090 beside the stage card
//!   with the plan's experts on each layer;
//! - `two_card_first_tokens_are_union`: [`TWO_WALK`] decode steps of the walk read the reference's ids and logits rows
//!   bit for bit — the routed block joins the tier's slots in the slot order the union's card sum runs — and the walk
//!   sent every tier layer a routed slot (or the run proves nothing there and fails by name);
//! - `two_card_eager_is_replay`: the walk eager and as the captured graph's replays reads the same ids and logits bits;
//! - `two_card_batch_is_union`: [`TWO_PROMPT`] ids fed in batches at a prompt group of two (two batches in one group, so
//!   a tier layer's join runs after the next unit's front has routed over the shared buffers), then one step, from a
//!   reset: the prefill's argmax and logits row and the step's id and logits row bit for bit the reference's same feed
//!   — a batch writes the steps' bits on one card, and the tier's leg computes a slot's rows as the stage card's does;
//! - `two_card_batch_sent`: that feed sent every tier layer a routed slot and the tier served its batches.
//!
//! The arm runs on the real file only: it plans the A6000's and the 3090's bytes.

use std::path::Path;
use std::time::Instant;

use bloomery_gpu::model::StepMode;
use bloomery_gpu_gates::{GateError, verdict};
use bloomery_gpu_mimo2::{Body, Mimo2Model, T_MAX, prefill, set_prefill_group};
use gguf::Split;
use model::arch::mimo2::place::PlanInputs;
use model::placement::workstation;
use model::placement::{Machine, PlanLevers, union_plan};

use crate::tier_twocard::{Leg, Sent, same, sent_between, sent_every, tier_counts, tier_on_3090};

/// What the arm's errors name.
const WHAT: &str = "gate_mimo2_e2e two-card";

/// The card budget both loads are planned under: the stage card's dense terms and experts, and at most this much of
/// the tier's experts, so the reference's union holds at most twice it of the A6000's bytes [derived from the
/// placement's rule].
pub const TWO_BUDGET: u64 = 20 << 30;

/// The walk's decode steps.
pub const TWO_WALK: usize = 24;

/// The batch feed's ids: two batches, the second short, one group of two.
pub const TWO_PROMPT: usize = T_MAX + 100;

/// The prompt group the batch feed runs at: its two batches are one group.
const TWO_GROUP: usize = 2;

/// The reference's or the two-card load's reads: the walk's, and the batch feed's.
struct Reads {
    walk: Leg,
    batch: Leg,
}

/// The tier's counters on `m`'s load since it opened.
fn counts(m: &Mimo2Model) -> Result<Option<Sent>, GateError> {
    Ok(tier_counts(m.body(WHAT)?.hybrid()))
}

fn split(path: &Path) -> Result<Split, GateError> {
    Ok(Split::open(path).map_err(|e| format!("open {}: {e}", path.display()))?)
}

/// `ids` one decode step each from a reset, in `m`'s current mode.
fn walk_leg(m: &mut Mimo2Model, ids: &[u32]) -> Result<Leg, GateError> {
    m.reset()?;
    let mut leg = Leg::default();
    for &id in ids {
        leg.tokens.push(m.step(&[id])?);
        leg.logits.push(m.logits()?);
    }
    Ok(leg)
}

/// `prompt` fed in batches at [`TWO_GROUP`] from a reset, then a step of `next`: the prefill's argmax and logits row,
/// the step's id and logits row.
fn batch_leg(m: &mut Mimo2Model, prompt: &[u32], next: u32) -> Result<Leg, GateError> {
    m.reset()?;
    set_prefill_group(m, TWO_GROUP)?;
    let mut leg = Leg::default();
    leg.tokens.push(prefill(m, prompt)?);
    leg.logits.push(m.logits()?);
    leg.tokens.push(m.step(&[next])?);
    leg.logits.push(m.logits()?);
    Ok(leg)
}

/// The walk and the batch feed on `m`, graph mode.
fn reads(m: &mut Mimo2Model, (walk, prompt): (&[u32], &[u32])) -> Result<Reads, GateError> {
    m.set_mode(StepMode::Graph);
    let next = walk.first().copied().ok_or("an empty walk")?;
    Ok(Reads {
        walk: walk_leg(m, walk)?,
        batch: batch_leg(m, prompt, next)?,
    })
}

/// The arm on the file at `path` under the gate's `levers` (module doc): `Ok(pass)`. A budget the environment sets, a
/// union past the A6000's usable bytes, a plan with no tier expert and a reference opened elsewhere or holding a tier
/// are refused by name.
pub fn run(path: &Path, levers: &bloomery_levers::Levers) -> Result<bool, GateError> {
    if PlanLevers::from_levers(levers)?.card_budget_bytes.is_set() {
        return Err(
            "the two-card arm plans under its own card budget: unset BLOOMERY_CARD_BUDGET".into(),
        );
    }
    let inputs = PlanInputs::read(&split(path)?)?;
    let layers = inputs.model.layers;
    let walk = crate::e2e::walk(TWO_WALK, inputs.hp.n_vocab);
    let prompt = crate::e2e::walk(TWO_PROMPT, inputs.hp.n_vocab);
    let two = workstation::plan_bp(layers, None, bloomery_gpu_mimo2::tier::tier_batch(&inputs)?);
    let flat = Machine {
        tiers: Vec::new(),
        ..two.clone()
    };
    let ctx = u64::try_from(TWO_WALK.max(TWO_PROMPT + 1))?;
    let budget = PlanLevers {
        card_budget_bytes: Some(TWO_BUDGET).into(),
        ..PlanLevers::default()
    };
    let plan = inputs.plan(&two, ctx, &budget)?;
    let on_tier = plan.tier_n_l.first().cloned().unwrap_or_default();
    let planned: Vec<u64> = (0..layers)
        .map(|l| on_tier.get(l).copied().unwrap_or(0))
        .collect();
    let tier_layers: Vec<usize> = (0..layers).filter(|&l| planned[l] > 0).collect();
    let stage = &plan.cards[0];
    let tier_bytes = plan.cards.get(1).map_or(0, |c| c.expert_bytes);
    let union_bytes = stage.dense_bytes + stage.expert_bytes + stage.kv_bytes + tier_bytes;
    let usable = workstation::A6000.usable_bytes();
    println!(
        "two-card plan (b′) under a card budget of {} at ctx {ctx}: stage {} experts ({} B), tier {} experts \
         ({tier_bytes} B) on {} layers, host {} experts ({} B); the reference's union {union_bytes} B of the A6000's \
         {usable} (plan values)",
        model::placement::card_budget::in_bytes(&budget.card_budget_bytes),
        stage.experts,
        stage.expert_bytes,
        plan.cards.get(1).map_or(0, |c| c.experts),
        tier_layers.len(),
        plan.host.experts,
        plan.host.expert_bytes,
    );
    if tier_layers.is_empty() {
        return Err("the two-card plan puts no expert on the tier: the arm proves nothing".into());
    }
    if union_bytes > usable {
        return Err(format!(
            "the reference's union of {union_bytes} B passes the A6000's {usable} B: lower the two-card budget"
        )
        .into());
    }
    let host = levers.host();
    let want = {
        let uplan = union_plan(&plan, &flat)?;
        let mut r = Body::open_placed(split(path)?, &uplan, &inputs, 0, host)?;
        if !r.body(WHAT)?.hybrid().tiers().is_empty() {
            return Err("the reference load holds a tier card".into());
        }
        let device = r.gpu().device_name()?;
        if !device.contains(workstation::A6000.name) {
            return Err(format!(
                "the reference opened on {device}, not the {}: the union is planned for its bytes",
                workstation::A6000.name
            )
            .into());
        }
        reads(&mut r, (&walk, &prompt))?
    };
    let opened = Instant::now();
    let mut m = Body::open_placed(split(path)?, &plan, &inputs, 0, host)?;
    let mut pass = true;

    tier_on_3090(
        "two_card_load",
        opened,
        (&m.gpu().device_name()?, m.resident_bytes()),
        m.body(WHAT)?.hybrid(),
    )?;
    let loaded: Vec<u64> = {
        let hybrid = m.body(WHAT)?.hybrid();
        let set = hybrid.tiers().first().map(|t| t.set());
        (0..layers)
            .map(|l| set.and_then(|s| s.ids(l)).map_or(0, |ids| ids.len() as u64))
            .collect()
    };
    let load_ok = loaded == planned;
    println!(
        "two_card_load: the tier holds experts on {} layers; the load's tier experts by layer {} the plan's {}",
        tier_layers.len(),
        if load_ok { "are" } else { "are not" },
        verdict(load_ok)
    );
    if !load_ok {
        println!("  loaded {loaded:?}\n  planned {planned:?}");
    }
    pass &= load_ok;

    m.set_mode(StepMode::Graph);
    let before = counts(&m)?;
    let replay = walk_leg(&mut m, &walk)?;
    let sent = sent_between(before, counts(&m)?);
    pass &= same("two_card_first_tokens_are_union", &want.walk, &replay);
    pass &= sent_every(
        "two_card_first_tokens_are_union walk",
        sent.as_ref(),
        &tier_layers,
        false,
    );

    m.set_mode(StepMode::Eager);
    let eager = walk_leg(&mut m, &walk)?;
    pass &= same("two_card_eager_is_replay", &replay, &eager);
    m.set_mode(StepMode::Graph);
    m.capture_step()?;

    let before = counts(&m)?;
    let next = walk.first().copied().ok_or("an empty walk")?;
    let batch = batch_leg(&mut m, &prompt, next)?;
    let sent = sent_between(before, counts(&m)?);
    pass &= same("two_card_batch_is_union", &want.batch, &batch);
    pass &= sent_every("two_card_batch_sent", sent.as_ref(), &tier_layers, true);
    m.reset()?;
    Ok(pass)
}
